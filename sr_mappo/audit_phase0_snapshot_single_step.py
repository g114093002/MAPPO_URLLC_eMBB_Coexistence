from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .run_fixed_user_blocklength_compare import (
    _apply_channel_setting_env,
    _build_policy_config,
    _build_scene_id,
)
from .unified_policy_runner import _clone_cfg, _configure_eval_env, _load_mappo_model


def _count_minrate_and_blocked(env, rates: np.ndarray) -> Tuple[int, int, int]:
    safe_rates = np.asarray(rates, dtype=float).reshape(-1)
    min_rate = float(
        getattr(env.embb_cfg, "min_rate_per_user_bps", getattr(env.embb_cfg, "min_rate", 0.0)) or 0.0
    )
    if min_rate > 0.0:
        minrate = int(np.count_nonzero(safe_rates >= (min_rate - 1.0e-9)))
        blocked = int(np.count_nonzero(safe_rates < (min_rate - 1.0e-9)))
        served = int(np.count_nonzero(safe_rates >= (min_rate - 1.0e-9)))
    else:
        served = int(np.count_nonzero(safe_rates > 1.0e-9))
        blocked = int(np.count_nonzero(safe_rates <= 1.0e-9))
        minrate = served
    return minrate, blocked, served


def _projection_summary(env, projection: Dict[str, np.ndarray | float]) -> Dict[str, float]:
    rates = np.asarray(projection.get("rates", np.zeros(0, dtype=float)), dtype=float)
    minrate, blocked, served = _count_minrate_and_blocked(env, rates)
    return {
        "total_rate_mbps": float(projection.get("total_rate", 0.0) or 0.0) / 1.0e6,
        "total_power_w": float(projection.get("total_power", 0.0) or 0.0),
        "minrate_users": float(minrate),
        "blocked_users": float(blocked),
        "served_users": float(served),
    }


def _move_record(
    *,
    move_type: str,
    uav_idx: int,
    rb_idx: int,
    snapshot_owner: int,
    new_owner: int,
    snapshot_scale: float,
    new_scale: float,
    base_summary: Dict[str, float],
    trial_summary: Dict[str, float],
) -> Dict[str, float | int | str]:
    return {
        "move_type": move_type,
        "uav": int(uav_idx),
        "rb": int(rb_idx),
        "snapshot_owner": int(snapshot_owner),
        "new_owner": int(new_owner),
        "snapshot_scale": float(snapshot_scale),
        "new_scale": float(new_scale),
        "delta_rate_mbps": float(trial_summary["total_rate_mbps"] - base_summary["total_rate_mbps"]),
        "delta_power_w": float(trial_summary["total_power_w"] - base_summary["total_power_w"]),
        "delta_minrate_users": float(trial_summary["minrate_users"] - base_summary["minrate_users"]),
        "delta_blocked_users": float(trial_summary["blocked_users"] - base_summary["blocked_users"]),
        "trial_rate_mbps": float(trial_summary["total_rate_mbps"]),
        "trial_minrate_users": float(trial_summary["minrate_users"]),
        "trial_blocked_users": float(trial_summary["blocked_users"]),
    }


def _top_and_stats(records: List[Dict[str, float | int | str]], top_k: int) -> Dict[str, object]:
    positive = [r for r in records if float(r["delta_rate_mbps"]) > 1.0e-9]
    nonnegative = [r for r in records if float(r["delta_rate_mbps"]) >= -1.0e-9]
    sorted_desc = sorted(records, key=lambda item: float(item["delta_rate_mbps"]), reverse=True)
    sorted_asc = sorted(records, key=lambda item: float(item["delta_rate_mbps"]))
    return {
        "count": int(len(records)),
        "positive_count": int(len(positive)),
        "nonnegative_count": int(len(nonnegative)),
        "positive_ratio": float(len(positive) / max(len(records), 1)),
        "best_delta_mbps": float(sorted_desc[0]["delta_rate_mbps"]) if sorted_desc else 0.0,
        "worst_delta_mbps": float(sorted_asc[0]["delta_rate_mbps"]) if sorted_asc else 0.0,
        "mean_delta_mbps": float(
            np.mean([float(item["delta_rate_mbps"]) for item in records], dtype=float) if records else 0.0
        ),
        "top_positive": sorted_desc[: max(int(top_k), 1)],
        "top_negative": sorted_asc[: max(int(top_k), 1)],
    }


def _candidate_owner_list(env, uav_idx: int, rb_idx: int, snapshot_owner: int) -> List[int]:
    raw_candidates = []
    if getattr(env, "embb_owner_candidates_by_uav_rb", None):
        raw_candidates = list(env.embb_owner_candidates_by_uav_rb[uav_idx][rb_idx])
    out: List[int] = []
    seen = set()
    for owner in raw_candidates:
        owner_i = int(owner)
        if owner_i == int(snapshot_owner):
            continue
        if owner_i < 0 or owner_i >= int(env.sys_cfg.num_embb_users):
            continue
        if owner_i in seen:
            continue
        seen.add(owner_i)
        out.append(owner_i)
    return out


def _power_delta_grid(env) -> List[float]:
    current = 1.0
    delta_limit = float(getattr(env.rl_cfg.action, "embb_power_delta_limit", 0.0) or 0.0)
    scale_min = float(getattr(env.rl_cfg.env, "embb_power_scale_min", 0.0) or 0.0)
    scale_max = float(getattr(env.rl_cfg.env, "embb_power_scale_max", 1.0) or 1.0)
    candidate_deltas = [-1.0, -0.5, 0.5, 1.0]
    accepted: List[float] = []
    seen_scales = set()
    for delta in candidate_deltas:
        scale = float(np.clip(current + delta_limit * float(delta), scale_min, scale_max))
        if abs(scale - current) <= 1.0e-9:
            continue
        key = round(scale, 8)
        if key in seen_scales:
            continue
        seen_scales.add(key)
        accepted.append(float(scale))
    return accepted


def audit_single_scenario(
    *,
    checkpoint_path: str,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    seed: int,
    share_scene_across_packet_bits: bool = True,
) -> Dict[str, object]:
    scene_id = _build_scene_id(
        embb_users=int(embb_users),
        urllc_users=int(urllc_users),
        packet_bits=int(packet_bits),
        channel_setting_index=int(seed),
        share_scene_across_packet_bits=bool(share_scene_across_packet_bits),
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
        base_projection = env._project_embb_baseline_from_owner_map(snapshot_owner, snapshot_scale)
        base_summary = _projection_summary(env, base_projection)

        owner_records: List[Dict[str, float | int | str]] = []
        power_records: List[Dict[str, float | int | str]] = []
        joint_records: List[Dict[str, float | int | str]] = []
        power_scale_grid = _power_delta_grid(env)

        for uav_idx in range(int(snapshot_owner.shape[0])):
            for rb_idx in range(int(snapshot_owner.shape[1])):
                current_owner = int(snapshot_owner[uav_idx, rb_idx])
                current_scale = float(snapshot_scale[uav_idx, rb_idx])

                for new_owner in _candidate_owner_list(env, uav_idx, rb_idx, current_owner):
                    trial_owner = snapshot_owner.copy()
                    trial_owner[uav_idx, rb_idx] = int(new_owner)
                    trial_proj = env._project_embb_baseline_from_owner_map(trial_owner, snapshot_scale)
                    trial_summary = _projection_summary(env, trial_proj)
                    owner_records.append(
                        _move_record(
                            move_type="owner_only",
                            uav_idx=uav_idx,
                            rb_idx=rb_idx,
                            snapshot_owner=current_owner,
                            new_owner=new_owner,
                            snapshot_scale=current_scale,
                            new_scale=current_scale,
                            base_summary=base_summary,
                            trial_summary=trial_summary,
                        )
                    )

                    for new_scale in power_scale_grid:
                        trial_scale = snapshot_scale.copy()
                        trial_scale[uav_idx, rb_idx] = float(new_scale)
                        trial_proj2 = env._project_embb_baseline_from_owner_map(trial_owner, trial_scale)
                        trial_summary2 = _projection_summary(env, trial_proj2)
                        joint_records.append(
                            _move_record(
                                move_type="owner_plus_power",
                                uav_idx=uav_idx,
                                rb_idx=rb_idx,
                                snapshot_owner=current_owner,
                                new_owner=new_owner,
                                snapshot_scale=current_scale,
                                new_scale=float(new_scale),
                                base_summary=base_summary,
                                trial_summary=trial_summary2,
                            )
                        )

                for new_scale in power_scale_grid:
                    trial_scale = snapshot_scale.copy()
                    trial_scale[uav_idx, rb_idx] = float(new_scale)
                    trial_proj = env._project_embb_baseline_from_owner_map(snapshot_owner, trial_scale)
                    trial_summary = _projection_summary(env, trial_proj)
                    power_records.append(
                        _move_record(
                            move_type="power_only",
                            uav_idx=uav_idx,
                            rb_idx=rb_idx,
                            snapshot_owner=current_owner,
                            new_owner=current_owner,
                            snapshot_scale=current_scale,
                            new_scale=float(new_scale),
                            base_summary=base_summary,
                            trial_summary=trial_summary,
                        )
                    )

        return {
            "scenario": {
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "scene_id": str(scene_id),
                "embb_users": int(embb_users),
                "urllc_users": int(urllc_users),
                "packet_bits": int(packet_bits),
                "seed": int(seed),
                "power_delta_limit": float(getattr(env.rl_cfg.action, "embb_power_delta_limit", 0.0) or 0.0),
                "power_scale_min": float(getattr(env.rl_cfg.env, "embb_power_scale_min", 0.0) or 0.0),
                "power_scale_max": float(getattr(env.rl_cfg.env, "embb_power_scale_max", 1.0) or 1.0),
                "power_scale_grid": [float(x) for x in power_scale_grid],
            },
            "snapshot": base_summary,
            "owner_only": _top_and_stats(owner_records, top_k=10),
            "power_only": _top_and_stats(power_records, top_k=10),
            "owner_plus_power": _top_and_stats(joint_records, top_k=10),
        }
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-step Phase-0 snapshot neighborhood audit.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--urllc-users", type=str, default="10,20,30")
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    urllc_values = [int(token.strip()) for token in str(args.urllc_users).split(",") if token.strip()]
    result = {
        "audit_type": "phase0_snapshot_single_step",
        "scenarios": [
            audit_single_scenario(
                checkpoint_path=str(args.checkpoint),
                embb_users=int(args.embb_users),
                urllc_users=int(urllc_users),
                packet_bits=int(args.packet_bits),
                seed=int(args.seed),
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
