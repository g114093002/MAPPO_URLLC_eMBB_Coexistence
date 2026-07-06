from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .run_fixed_user_blocklength_compare import (
    _apply_channel_setting_env,
    _build_policy_config,
    _build_scene_id,
)
from .unified_policy_runner import _clone_cfg, _configure_eval_env, _load_mappo_model


def _count_rate_stats(env, rates: np.ndarray) -> Dict[str, float]:
    safe_rates = np.asarray(rates, dtype=float).reshape(-1)
    min_rate = float(
        getattr(env.embb_cfg, "min_rate_per_user_bps", getattr(env.embb_cfg, "min_rate", 0.0)) or 0.0
    )
    if min_rate > 0.0:
        minrate = int(np.count_nonzero(safe_rates >= (min_rate - 1.0e-9)))
        blocked = int(np.count_nonzero(safe_rates < (min_rate - 1.0e-9)))
        served = int(np.count_nonzero(safe_rates >= (min_rate - 1.0e-9)))
        partial = int(np.count_nonzero((safe_rates > 1.0e-9) & (safe_rates < (min_rate - 1.0e-9))))
    else:
        blocked = int(np.count_nonzero(safe_rates <= 1.0e-9))
        served = int(np.count_nonzero(safe_rates > 1.0e-9))
        minrate = served
        partial = 0
    return {
        "blocked_users": float(blocked),
        "served_users": float(served),
        "minrate_users": float(minrate),
        "partial_minrate_users": float(partial),
        "feasible_minrate": float(partial == 0),
    }


def _summarize_projection(env, projection: Dict[str, np.ndarray | float]) -> Dict[str, float]:
    rates = np.asarray(projection.get("rates", np.zeros(0, dtype=float)), dtype=float)
    summary = _count_rate_stats(env, rates)
    summary.update(
        {
            "total_rate_mbps": float(projection.get("total_rate", 0.0) or 0.0) / 1.0e6,
            "total_power_w": float(projection.get("total_power", 0.0) or 0.0),
        }
    )
    return summary


def _state_rank_key(summary: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(summary["total_rate_mbps"]),
        -float(summary["blocked_users"]),
        -float(summary["total_power_w"]),
    )


def _owner_candidates(env, uav_idx: int, rb_idx: int, current_owner: int) -> List[int]:
    raw = []
    if getattr(env, "embb_owner_candidates_by_uav_rb", None):
        raw = list(env.embb_owner_candidates_by_uav_rb[uav_idx][rb_idx])
    out: List[int] = []
    seen = set()
    for owner in raw:
        owner_i = int(owner)
        if owner_i == int(current_owner):
            continue
        if owner_i < 0 or owner_i >= int(env.sys_cfg.num_embb_users):
            continue
        if owner_i in seen:
            continue
        seen.add(owner_i)
        out.append(owner_i)
    return out


@dataclass
class SearchState:
    owner_map: np.ndarray
    scale_map: np.ndarray
    summary: Dict[str, float]
    path: List[Dict[str, object]]


def _projection_for_state(env, owner_map: np.ndarray, scale_map: np.ndarray) -> Dict[str, np.ndarray | float]:
    return env._project_embb_baseline_from_owner_map(np.asarray(owner_map, dtype=int), np.asarray(scale_map, dtype=float))


def _state_key(owner_map: np.ndarray, scale_map: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    owner_key = tuple(int(x) for x in np.asarray(owner_map, dtype=int).ravel().tolist())
    scale_key = tuple(round(float(x), 6) for x in np.asarray(scale_map, dtype=float).ravel().tolist())
    return owner_key, scale_key


def _generate_neighbors(
    env,
    state: SearchState,
    *,
    include_power: bool,
    power_scales: Iterable[float],
) -> List[SearchState]:
    owner_map = np.asarray(state.owner_map, dtype=int)
    scale_map = np.asarray(state.scale_map, dtype=float)
    neighbors: List[SearchState] = []
    for uav_idx in range(int(owner_map.shape[0])):
        for rb_idx in range(int(owner_map.shape[1])):
            current_owner = int(owner_map[uav_idx, rb_idx])
            current_scale = float(scale_map[uav_idx, rb_idx])

            if include_power:
                for new_scale in power_scales:
                    new_scale_f = float(new_scale)
                    if abs(new_scale_f - current_scale) <= 1.0e-9:
                        continue
                    trial_scale = scale_map.copy()
                    trial_scale[uav_idx, rb_idx] = new_scale_f
                    trial_summary = _summarize_projection(env, _projection_for_state(env, owner_map, trial_scale))
                    neighbors.append(
                        SearchState(
                            owner_map=owner_map.copy(),
                            scale_map=trial_scale,
                            summary=trial_summary,
                            path=state.path
                            + [
                                {
                                    "move_type": "power_only",
                                    "uav": int(uav_idx),
                                    "rb": int(rb_idx),
                                    "from_owner": int(current_owner),
                                    "to_owner": int(current_owner),
                                    "from_scale": float(current_scale),
                                    "to_scale": float(new_scale_f),
                                    "delta_rate_mbps": float(trial_summary["total_rate_mbps"] - state.summary["total_rate_mbps"]),
                                }
                            ],
                        )
                    )

            for new_owner in _owner_candidates(env, uav_idx, rb_idx, current_owner):
                trial_owner = owner_map.copy()
                trial_owner[uav_idx, rb_idx] = int(new_owner)

                trial_summary = _summarize_projection(env, _projection_for_state(env, trial_owner, scale_map))
                neighbors.append(
                    SearchState(
                        owner_map=trial_owner,
                        scale_map=scale_map.copy(),
                        summary=trial_summary,
                        path=state.path
                        + [
                            {
                                "move_type": "owner_only",
                                "uav": int(uav_idx),
                                "rb": int(rb_idx),
                                "from_owner": int(current_owner),
                                "to_owner": int(new_owner),
                                "from_scale": float(current_scale),
                                "to_scale": float(current_scale),
                                "delta_rate_mbps": float(trial_summary["total_rate_mbps"] - state.summary["total_rate_mbps"]),
                            }
                        ],
                    )
                )

                if include_power:
                    for new_scale in power_scales:
                        new_scale_f = float(new_scale)
                        if abs(new_scale_f - current_scale) <= 1.0e-9:
                            continue
                        trial_scale = scale_map.copy()
                        trial_scale[uav_idx, rb_idx] = new_scale_f
                        trial_summary2 = _summarize_projection(env, _projection_for_state(env, trial_owner, trial_scale))
                        neighbors.append(
                            SearchState(
                                owner_map=trial_owner.copy(),
                                scale_map=trial_scale,
                                summary=trial_summary2,
                                path=state.path
                                + [
                                    {
                                        "move_type": "owner_plus_power",
                                        "uav": int(uav_idx),
                                        "rb": int(rb_idx),
                                        "from_owner": int(current_owner),
                                        "to_owner": int(new_owner),
                                        "from_scale": float(current_scale),
                                        "to_scale": float(new_scale_f),
                                        "delta_rate_mbps": float(
                                            trial_summary2["total_rate_mbps"] - state.summary["total_rate_mbps"]
                                        ),
                                    }
                                ],
                            )
                        )
    return neighbors


def _serialize_state(state: SearchState, baseline_summary: Dict[str, float]) -> Dict[str, object]:
    return {
        "summary": dict(state.summary),
        "gain_vs_snapshot_mbps": float(state.summary["total_rate_mbps"] - baseline_summary["total_rate_mbps"]),
        "path": list(state.path),
    }


def audit_search_scenario(
    *,
    checkpoint_path: str,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    seed: int,
    depth: int,
    beam_width: int,
    include_power: bool,
    power_scales: List[float],
) -> Dict[str, object]:
    scene_id = _build_scene_id(
        embb_users=int(embb_users),
        urllc_users=int(urllc_users),
        packet_bits=int(packet_bits),
        channel_setting_index=int(seed),
        share_scene_across_packet_bits=True,
        share_scene_across_urllc_users=False,
        urllc_scene_anchor=int(urllc_users),
    )
    cfg_dict = _build_policy_config(
        policy="mappo",
        embb_users=int(embb_users),
        urllc_users=int(urllc_users),
        packet_bits=int(packet_bits),
        channel_uses=None,
        lambda_per_user=None,
        target_error_probability=None,
        mappo_checkpoint_path=checkpoint_path,
        geometry_profile=None,
        min_overlay_retention=None,
        nested_urllc_subset_from_max=False,
        nested_max_urllc_users=None,
    )
    cfg = _clone_cfg(cfg_dict)
    component_overrides = {
        name: dict(cfg_dict.get(name, {}) or {})
        for name in ("system", "simulation", "algorithm", "urllc", "embb")
    }
    env_backup = _apply_channel_setting_env(scene_id)
    try:
        env, _model = _load_mappo_model(cfg, checkpoint_path, component_overrides=component_overrides)
        _configure_eval_env(env, total_load=None, mix_ratio=None, explicit_mix_weights=None)
        env.reset(seed=int(seed))

        if include_power:
            env.rl_cfg.env.embb_power_scale_min = min(float(min(power_scales)), float(env.rl_cfg.env.embb_power_scale_min))
            env.rl_cfg.env.embb_power_scale_max = max(float(max(power_scales)), float(env.rl_cfg.env.embb_power_scale_max))

        snapshot_owner = np.asarray(env.phase0_snapshot_owner_per_uav_rb, dtype=int).copy()
        snapshot_scale = np.asarray(env.embb_power_scale_per_uav_rb, dtype=float).copy()
        snapshot_summary = _summarize_projection(env, _projection_for_state(env, snapshot_owner, snapshot_scale))

        root = SearchState(
            owner_map=snapshot_owner,
            scale_map=snapshot_scale,
            summary=snapshot_summary,
            path=[],
        )
        beam = [root]
        best_states_by_depth: Dict[int, List[SearchState]] = {0: [root]}
        expanded_counts: Dict[int, int] = {}

        for step in range(1, max(int(depth), 1) + 1):
            generated: List[SearchState] = []
            for state in beam:
                generated.extend(
                    _generate_neighbors(
                        env,
                        state,
                        include_power=bool(include_power),
                        power_scales=power_scales,
                    )
                )
            expanded_counts[step] = int(len(generated))

            dedup: Dict[Tuple[Tuple[int, ...], Tuple[float, ...]], SearchState] = {}
            for state in generated:
                key = _state_key(state.owner_map, state.scale_map)
                incumbent = dedup.get(key)
                if incumbent is None or _state_rank_key(state.summary) > _state_rank_key(incumbent.summary):
                    dedup[key] = state

            ranked = sorted(dedup.values(), key=lambda item: _state_rank_key(item.summary), reverse=True)
            beam = ranked[: max(int(beam_width), 1)]
            best_states_by_depth[step] = beam

        result = {
            "scenario": {
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "scene_id": str(scene_id),
                "embb_users": int(embb_users),
                "urllc_users": int(urllc_users),
                "packet_bits": int(packet_bits),
                "seed": int(seed),
                "depth": int(depth),
                "beam_width": int(beam_width),
                "include_power": bool(include_power),
                "power_scales": [float(x) for x in power_scales],
            },
            "snapshot": snapshot_summary,
            "expanded_counts": {str(k): int(v) for k, v in expanded_counts.items()},
            "best_by_depth": {
                str(depth_k): [_serialize_state(state, snapshot_summary) for state in states[:10]]
                for depth_k, states in best_states_by_depth.items()
            },
        }
        return result
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Depth-limited Phase-0 search audit around snapshot.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--urllc-users", type=str, default="10,20,30")
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--include-power", action="store_true")
    parser.add_argument("--power-scales", type=str, default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    urllc_values = [int(token.strip()) for token in str(args.urllc_users).split(",") if token.strip()]
    power_scales = [float(token.strip()) for token in str(args.power_scales).split(",") if token.strip()]
    result = {
        "audit_type": "phase0_search_depth_limited",
        "scenarios": [
            audit_search_scenario(
                checkpoint_path=str(args.checkpoint),
                embb_users=int(args.embb_users),
                urllc_users=int(urllc_users),
                packet_bits=int(args.packet_bits),
                seed=int(args.seed),
                depth=int(args.depth),
                beam_width=int(args.beam_width),
                include_power=bool(args.include_power),
                power_scales=power_scales,
            )
            for urllc_users in urllc_values
        ],
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if str(args.out or "").strip():
        out_path = Path(str(args.out)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
