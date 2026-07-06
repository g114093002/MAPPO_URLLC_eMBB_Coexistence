import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .run_fixed_user_blocklength_compare import (
    _apply_channel_setting_env,
    _build_policy_config,
    _build_scene_id,
    _restore_env,
)
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, CandidatePacket, HybridAction
from .unified_policy_runner import (
    _build_env,
    _clone_cfg,
    _configure_eval_env,
    _enable_phase_a_joint_minrate_protection,
    _mode_rank_tuple,
    _planning_baseline_action,
)


def _component_overrides(config_dict: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        name: dict(config_dict.get(name, {}) or {})
        for name in ("system", "simulation", "algorithm", "urllc", "embb")
    }


def _candidate_pool_upper_bound(max_urllc_users: int) -> int:
    return max(int(max_urllc_users) * 7, 1)


def _find_packet_option(obs, packet_id: int, mode: int) -> int:
    for idx, candidate in enumerate(obs.candidates, start=1):
        if int(candidate.packet_id) == int(packet_id) and candidate.is_mode_feasible(int(mode)):
            return int(idx)
    return 0


def _find_locked_packet_option(obs, locked_entry: Dict[str, int]) -> int:
    packet_id = int(locked_entry.get("packet_id", -1))
    mode = int(locked_entry.get("mode", MODE_KEEP))
    packet_option = _find_packet_option(obs, packet_id, mode)
    if packet_option > 0:
        return int(packet_option)

    source_user = int(locked_entry.get("source_user", -1))
    release_minislot = int(locked_entry.get("release_minislot", -1))
    for idx, candidate in enumerate(obs.candidates, start=1):
        if not candidate.is_mode_feasible(int(mode)):
            continue
        if int(candidate.source_user) != int(source_user):
            continue
        cand_release = int(getattr(candidate, "packet_release_minislot", -1) or -1)
        if cand_release == int(release_minislot):
            return int(idx)
    return 0


def _eligible_candidates(
    obs,
    *,
    allowed_modes: Iterable[int],
    max_user_idx_exclusive: int,
) -> List[Tuple[int, int, CandidatePacket]]:
    packet_mask = np.asarray(obs.masks.packet_mask, dtype=float)
    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
    out: List[Tuple[int, int, CandidatePacket]] = []
    for packet_option, candidate in enumerate(obs.candidates, start=1):
        if int(candidate.source_user) >= int(max_user_idx_exclusive):
            continue
        for mode in allowed_modes:
            mode_int = int(mode)
            if not candidate.is_mode_feasible(mode_int):
                continue
            if (
                mode_int < mode_mask.size
                and mode_mask[mode_int] > 0.5
                and packet_mask.ndim == 2
                and mode_int < packet_mask.shape[0]
                and packet_option < packet_mask.shape[1]
                and packet_mask[mode_int, packet_option] > 0.5
            ):
                out.append((int(packet_option), mode_int, candidate))
    return out


def _best_hard_admit_action(
    env,
    obs,
    *,
    uav_idx: int,
    rb_idx: int,
    minislot_idx: int,
    allowed_modes: Iterable[int],
    max_user_idx_exclusive: int,
) -> HybridAction:
    feasible = _eligible_candidates(
        obs,
        allowed_modes=allowed_modes,
        max_user_idx_exclusive=int(max_user_idx_exclusive),
    )
    if not feasible:
        return HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)

    best_action: Optional[HybridAction] = None
    best_rank: Optional[Tuple[float, ...]] = None
    for packet_option, mode, candidate in feasible:
        global_embb_throughput = float(
            env._global_embb_throughput_if_apply_candidate_cell(
                int(uav_idx),
                int(rb_idx),
                int(minislot_idx),
                candidate,
                int(mode),
                0.0,
            )
        )
        rank = _mode_rank_tuple(
            env,
            global_embb_throughput,
            candidate,
            int(mode),
            puncturing_selection_rule="max_embb_sum_rate",
            superposition_selection_rule="max_embb_sum_rate",
        )
        if best_rank is None or tuple(rank) > tuple(best_rank):
            best_rank = tuple(rank)
            best_action = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
    return best_action if best_action is not None else HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)


def _random_incremental_action(
    obs,
    rng: np.random.Generator,
    *,
    max_user_idx_exclusive: int,
) -> HybridAction:
    eligible = [
        (idx, candidate)
        for idx, candidate in enumerate(obs.candidates, start=1)
        if int(candidate.source_user) < int(max_user_idx_exclusive)
    ]
    if not eligible:
        return HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
    packet_option, _candidate = eligible[int(rng.integers(0, len(eligible)))]
    mode = int(MODE_OVERLAY if float(rng.random()) < 0.5 else MODE_PUNCTURE)
    return HybridAction(mode=mode, packet_option=int(packet_option), power_delta=0.0)


def _build_stage_env(
    *,
    policy: str,
    embb_users: int,
    max_urllc_users: int,
    packet_bits: int,
    disable_joint_rewrites: bool = True,
) -> Tuple[Dict[str, object], object]:
    cfg_dict = _build_policy_config(
        policy=str(policy),
        embb_users=int(embb_users),
        urllc_users=int(max_urllc_users),
        packet_bits=int(packet_bits),
        channel_uses=None,
        lambda_per_user=None,
        target_error_probability=None,
        mappo_checkpoint_path=None,
        geometry_profile=None,
        min_overlay_retention=None,
        nested_urllc_subset_from_max=False,
        nested_max_urllc_users=int(max_urllc_users),
    )
    cfg = _clone_cfg(cfg_dict)
    _enable_phase_a_joint_minrate_protection(cfg)
    if bool(disable_joint_rewrites):
        cfg.shield.apply_joint_reliability_rewrite = False
        cfg.shield.apply_joint_minrate_rewrite = False
    cfg.action.max_candidate_packets = max(
        int(getattr(cfg.action, "max_candidate_packets", 0) or 0),
        int(_candidate_pool_upper_bound(int(max_urllc_users))),
    )
    component_overrides = _component_overrides(cfg_dict)
    env = _build_env(cfg, component_overrides=component_overrides)
    env._eval_method_name = str(policy)
    _configure_eval_env(env, total_load=None, mix_ratio=None, explicit_mix_weights=None)
    return cfg_dict, env


def _extract_selected_cells(env) -> Dict[Tuple[int, int, int], Dict[str, int]]:
    locked: Dict[Tuple[int, int, int], Dict[str, int]] = {}
    for packet_id in range(int(env.num_packets)):
        if int(env.scheduled_uavs[packet_id]) < 0:
            continue
        positions = np.argwhere(env.packet_grid == int(packet_id))
        if positions.size <= 0:
            continue
        uav_idx, rb_idx, minislot_idx = [int(v) for v in positions[0].tolist()]
        locked[(int(minislot_idx), int(rb_idx), int(uav_idx))] = {
            "packet_id": int(packet_id),
            "mode": int(env.mode_grid[int(uav_idx), int(rb_idx), int(minislot_idx)]),
            "source_user": int(env.packet_sources[packet_id]) if packet_id < env.packet_sources.size else -1,
            "release_minislot": int(env.packet_release_minislots[packet_id]) if packet_id < env.packet_release_minislots.size else -1,
        }
    return locked


def _run_incremental_stage(
    *,
    policy: str,
    embb_users: int,
    max_urllc_users: int,
    active_urllc_users: int,
    packet_bits: int,
    seed: int,
    locked_cells: Optional[Dict[Tuple[int, int, int], Dict[str, int]]] = None,
    disable_joint_rewrites: bool = True,
) -> Dict[str, object]:
    _cfg_dict, env = _build_stage_env(
        policy=str(policy),
        embb_users=int(embb_users),
        max_urllc_users=int(max_urllc_users),
        packet_bits=int(packet_bits),
        disable_joint_rewrites=bool(disable_joint_rewrites),
    )
    observations, _ = env.reset(seed=int(seed))
    rng = np.random.default_rng(int(seed))
    lock_misses = 0

    while True:
        if bool(getattr(env, "episode_done", False)):
            break
        planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
        if planning_phase:
            joint_actions = {
                aid: _planning_baseline_action(env, observations[aid])
                for aid in env.agent_ids
            }
            resolved = {
                aid: env._raw_action_to_shielded_action(joint_actions[aid], observations[aid])
                for aid in env.agent_ids
            }
            observations, _rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            if all(dones.values()):
                break
            continue

        minislot_idx, rb_idx = env._current_cell()
        joint_actions: Dict[str, HybridAction] = {}
        forced_keep_agents = set()
        for uav_idx, agent_id in enumerate(env.agent_ids):
            obs = observations[agent_id]
            locked_entry = None
            if isinstance(locked_cells, dict):
                locked_entry = locked_cells.get((int(minislot_idx), int(rb_idx), int(uav_idx)))
            if locked_entry is not None:
                packet_option = _find_locked_packet_option(obs, locked_entry)
                if packet_option > 0:
                    joint_actions[agent_id] = HybridAction(
                        mode=int(locked_entry["mode"]),
                        packet_option=int(packet_option),
                        power_delta=0.0,
                    )
                    continue
                lock_misses += 1
                joint_actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
                forced_keep_agents.add(agent_id)
                continue
            if str(policy) == "pure_superposition":
                joint_actions[agent_id] = _best_hard_admit_action(
                    env,
                    obs,
                    uav_idx=int(uav_idx),
                    rb_idx=int(rb_idx),
                    minislot_idx=int(minislot_idx),
                    allowed_modes=(MODE_OVERLAY,),
                    max_user_idx_exclusive=int(active_urllc_users),
                )
            elif str(policy) == "random_scheduler":
                joint_actions[agent_id] = _random_incremental_action(
                    obs,
                    rng,
                    max_user_idx_exclusive=int(active_urllc_users),
                )
            else:
                raise ValueError(f"Unsupported policy for incremental analysis: {policy}")

        resolved = env._resolve_executed_actions(
            joint_actions,
            observations,
            minislot=int(minislot_idx),
            rb=int(rb_idx),
        )
        for agent_id in forced_keep_agents:
            resolved[agent_id] = env._raw_action_to_shielded_action(
                HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
                observations[agent_id],
            )
        observations, _rewards, dones, _infos = env.step(
            joint_actions,
            prebuilt_observations=observations,
            pre_resolved_actions=resolved,
        )
        if all(dones.values()):
            break

    summary = env.summarize_episode()
    return {
        "policy": str(policy),
        "active_urllc_users": int(active_urllc_users),
        "throughput_mbps": float(summary.get("embb_total_rate_after_puncture_deduction", 0.0) / 1.0e6),
        "admitted_urllc_count": int(np.count_nonzero(env.scheduled_uavs >= 0)),
        "urllc_admission_ratio": float(
            np.count_nonzero(env.scheduled_uavs >= 0) / max(int(env.num_packets), 1)
        ),
        "locked_cells": _extract_selected_cells(env),
        "lock_misses": int(lock_misses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental locked-load analysis on a fixed mother scene.")
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--max-urllc-users", type=int, default=30)
    parser.add_argument("--active-urllc-users", default="10,20,30")
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--policies", default="pure_superposition,random_scheduler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-joint-rewrites", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    active_users = [int(token.strip()) for token in str(args.active_urllc_users).split(",") if token.strip()]
    policies = [str(token).strip() for token in str(args.policies).split(",") if token.strip()]

    scene_id = _build_scene_id(
        embb_users=int(args.embb_users),
        urllc_users=int(args.max_urllc_users),
        packet_bits=int(args.packet_bits),
        channel_setting_index=1,
        share_scene_across_packet_bits=True,
        share_scene_across_urllc_users=True,
        urllc_scene_anchor=int(args.max_urllc_users),
    )
    env_backup = _apply_channel_setting_env(scene_id)
    try:
        payload: Dict[str, object] = {
            "scene_id": str(scene_id),
            "embb_users": int(args.embb_users),
            "max_urllc_users": int(args.max_urllc_users),
            "packet_bits": int(args.packet_bits),
            "seed": int(args.seed),
            "results": {},
        }
        for policy in policies:
            locked_cells: Optional[Dict[Tuple[int, int, int], Dict[str, int]]] = None
            stages: List[Dict[str, object]] = []
            for active_u in active_users:
                stage = _run_incremental_stage(
                    policy=str(policy),
                    embb_users=int(args.embb_users),
                    max_urllc_users=int(args.max_urllc_users),
                    active_urllc_users=int(active_u),
                    packet_bits=int(args.packet_bits),
                    seed=int(args.seed),
                    locked_cells=deepcopy(locked_cells),
                    disable_joint_rewrites=bool(getattr(args, "disable_joint_rewrites", False)),
                )
                locked_cells = dict(stage["locked_cells"])
                stage["locked_cells"] = int(len(locked_cells))
                stages.append(stage)
            payload["results"][str(policy)] = stages
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        _restore_env(env_backup)


if __name__ == "__main__":
    main()
