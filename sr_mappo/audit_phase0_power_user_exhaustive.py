from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
            "user_tx_powers_w": [float(x) for x in np.asarray(projection.get("user_tx_powers", []), dtype=float).tolist()],
        }
    )
    return summary


def _summary_rank_key(summary: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(summary["total_rate_mbps"]),
        -float(summary["blocked_users"]),
        -float(summary["total_power_w"]),
    )


def _build_per_user_scale_map(snapshot_owner: np.ndarray, user_scales: np.ndarray) -> np.ndarray:
    owner_map = np.asarray(snapshot_owner, dtype=int)
    scales = np.asarray(user_scales, dtype=float).reshape(-1)
    scale_map = np.ones_like(owner_map, dtype=float)
    valid = owner_map >= 0
    if np.any(valid):
        scale_map[valid] = scales[owner_map[valid]]
    return scale_map


def _serialize_candidate(
    *,
    summary: Dict[str, float],
    snapshot_summary: Dict[str, float],
    user_scales: np.ndarray,
) -> Dict[str, object]:
    return {
        "summary": dict(summary),
        "gain_vs_snapshot_mbps": float(summary["total_rate_mbps"] - snapshot_summary["total_rate_mbps"]),
        "delta_blocked_users": float(summary["blocked_users"] - snapshot_summary["blocked_users"]),
        "delta_power_w": float(summary["total_power_w"] - snapshot_summary["total_power_w"]),
        "user_scales": [float(x) for x in np.asarray(user_scales, dtype=float).tolist()],
    }


def exhaustive_power_user_scenario(
    *,
    checkpoint_path: str,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    seed: int,
    power_scales: List[float],
    progress_every: int,
    top_k: int,
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

        snapshot_owner = np.asarray(env.phase0_snapshot_owner_per_uav_rb, dtype=int).copy()
        snapshot_scale = np.asarray(env.embb_power_scale_per_uav_rb, dtype=float).copy()
        snapshot_projection = env._project_embb_baseline_from_owner_map(snapshot_owner, snapshot_scale)
        snapshot_summary = _summarize_projection(env, snapshot_projection)

        num_embb = int(env.sys_cfg.num_embb_users)
        scale_values = [float(x) for x in power_scales]
        total_combinations = int(len(scale_values) ** num_embb)

        best_summary = dict(snapshot_summary)
        best_scales = np.ones(num_embb, dtype=float)
        top_candidates: List[Dict[str, object]] = [
            _serialize_candidate(summary=snapshot_summary, snapshot_summary=snapshot_summary, user_scales=best_scales)
        ]

        positive_count = 0
        non_worse_count = 1
        progress_every = max(int(progress_every), 1)

        for combo_idx, combo in enumerate(itertools.product(scale_values, repeat=num_embb), start=1):
            user_scales = np.asarray(combo, dtype=float)
            if np.allclose(user_scales, 1.0):
                continue

            trial_scale_map = _build_per_user_scale_map(snapshot_owner, user_scales)
            trial_projection = env._project_embb_baseline_from_owner_map(snapshot_owner, trial_scale_map)
            trial_summary = _summarize_projection(env, trial_projection)
            gain_mbps = float(trial_summary["total_rate_mbps"] - snapshot_summary["total_rate_mbps"])

            if gain_mbps > 1.0e-9:
                positive_count += 1
            if _summary_rank_key(trial_summary) >= _summary_rank_key(snapshot_summary):
                non_worse_count += 1

            if _summary_rank_key(trial_summary) > _summary_rank_key(best_summary):
                best_summary = dict(trial_summary)
                best_scales = user_scales.copy()

            candidate_row = _serialize_candidate(
                summary=trial_summary,
                snapshot_summary=snapshot_summary,
                user_scales=user_scales,
            )
            top_candidates.append(candidate_row)
            top_candidates = sorted(
                top_candidates,
                key=lambda item: (
                    float(item["summary"]["total_rate_mbps"]),
                    -float(item["summary"]["blocked_users"]),
                    -float(item["summary"]["total_power_w"]),
                ),
                reverse=True,
            )[: max(int(top_k), 1)]

            if combo_idx % progress_every == 0:
                print(
                    f"[POWER-EXHAUSTIVE] {combo_idx}/{total_combinations} "
                    f"best_rate={best_summary['total_rate_mbps']:.6f} Mbps "
                    f"best_blocked={best_summary['blocked_users']:.0f} "
                    f"best_power={best_summary['total_power_w']:.6f} W",
                    flush=True,
                )

        return {
            "scenario": {
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "scene_id": str(scene_id),
                "embb_users": int(embb_users),
                "urllc_users": int(urllc_users),
                "packet_bits": int(packet_bits),
                "seed": int(seed),
                "power_scales": list(scale_values),
                "total_combinations": int(total_combinations),
            },
            "snapshot": snapshot_summary,
            "search_stats": {
                "positive_count": int(positive_count),
                "positive_ratio": float(positive_count / max(total_combinations - 1, 1)),
                "non_worse_count": int(non_worse_count),
                "non_worse_ratio": float(non_worse_count / max(total_combinations, 1)),
            },
            "best": _serialize_candidate(
                summary=best_summary,
                snapshot_summary=snapshot_summary,
                user_scales=best_scales,
            ),
            "top_k": list(top_candidates),
        }
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustive per-user Phase-0 power search with snapshot owner fixed."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--urllc-users", type=int, default=30)
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-scales", type=str, default="1.0,0.9,0.8,0.7,0.6")
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    power_scales = [float(token.strip()) for token in str(args.power_scales).split(",") if token.strip()]
    result = {
        "audit_type": "phase0_power_user_exhaustive",
        "scenario": exhaustive_power_user_scenario(
            checkpoint_path=str(args.checkpoint),
            embb_users=int(args.embb_users),
            urllc_users=int(args.urllc_users),
            packet_bits=int(args.packet_bits),
            seed=int(args.seed),
            power_scales=power_scales,
            progress_every=int(args.progress_every),
            top_k=int(args.top_k),
        ),
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
