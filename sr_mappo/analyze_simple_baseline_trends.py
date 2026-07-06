import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .run_fixed_user_blocklength_compare import (
    _apply_channel_setting_env,
    _build_policy_config,
    _build_scene_id,
    _restore_env,
)
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, HybridAction
from .unified_policy_runner import (
    _build_env,
    _clone_cfg,
    _configure_eval_env,
    _enable_phase_a_joint_minrate_protection,
    _extract_final_embb_user_tx_powers,
    _greedy_actions,
    _naive_random_action_and_resolution,
    _pure_puncturing_actions,
    _pure_superposition_actions,
    _random_policy_actions,
)


MODE_NAME = {
    MODE_KEEP: "keep",
    MODE_OVERLAY: "overlay",
    MODE_PUNCTURE: "puncture",
}


def _selector_for_policy(policy: str) -> Callable:
    normalized = str(policy).strip().lower()
    if normalized == "greedy":
        return lambda env, obs: _greedy_actions(env, obs)
    if normalized == "pure_puncturing":
        return lambda env, obs: _pure_puncturing_actions(env, obs, "max_embb_sum_rate")
    if normalized == "pure_superposition":
        return lambda env, obs: _pure_superposition_actions(env, obs, "max_embb_sum_rate")
    if normalized == "random_scheduler":
        return None
    raise ValueError(f"Unsupported policy for analysis: {policy}")


def _component_overrides(config_dict: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        name: dict(config_dict.get(name, {}) or {})
        for name in ("system", "simulation", "algorithm", "urllc", "embb")
    }


def _candidate_pool_upper_bound(component_overrides: Dict[str, Dict[str, object]]) -> int:
    system_cfg = dict(component_overrides.get("system", {}) or {})
    num_urllc = int(system_cfg.get("num_urllc_users", 0) or 0)
    return max(num_urllc * 7, 1)


def _reason_flags(candidate) -> List[str]:
    mapping = (
        ("overlay_reliability", "cause_overlay_reliability_failed"),
        ("overlay_gain_ratio", "cause_gain_ratio_unqualified"),
        ("overlay_sic", "cause_overlay_sic_failed"),
        ("embb_min_rate", "cause_embb_retention_below_threshold"),
        ("intercell", "cause_cross_uav_interference_too_high"),
        ("power", "cause_required_power_exceeds_budget"),
        ("overlay_margin", "cause_overlay_margin_blocked"),
        ("overlay_retention_gate", "cause_overlay_retention_gate_blocked"),
        ("overlay_positive_gate", "cause_overlay_positive_gate_blocked"),
        ("overlay_owner_missing", "cause_no_overlay_owner_available"),
        ("structural", "cause_other_structural_reason"),
    )
    out: List[str] = []
    for label, attr in mapping:
        if bool(getattr(candidate, attr, False)):
            out.append(label)
    return out


def _classify_blocked_packet(stats: Dict[str, object]) -> str:
    reasons = set(stats.get("reasons", set()) or set())
    if bool(stats.get("selected_feasible_any", False)):
        return "feasible_but_not_selected"
    if bool(stats.get("raw_feasible_any", False)) and not bool(stats.get("selected_seen", False)):
        return "pruned_by_candidate_subset"
    for key in (
        "overlay_reliability",
        "overlay_gain_ratio",
        "overlay_sic",
        "embb_min_rate",
        "intercell",
        "power",
        "overlay_margin",
        "overlay_retention_gate",
        "overlay_positive_gate",
        "overlay_owner_missing",
        "structural",
    ):
        if key in reasons:
            return key
    if not bool(stats.get("raw_seen", False)):
        return "no_candidate_seen"
    return "unknown"


def _extract_selected_packet_ids(env) -> Dict[int, Dict[str, int]]:
    packet_meta: Dict[int, Dict[str, int]] = {}
    for pid in range(int(env.num_packets)):
        uav = int(env.scheduled_uavs[pid]) if pid < env.scheduled_uavs.size else -1
        if uav < 0:
            continue
        positions = np.argwhere(env.packet_grid == int(pid))
        if positions.size <= 0:
            packet_meta[int(pid)] = {"uav": uav, "rb": -1, "minislot": -1, "mode": MODE_KEEP}
            continue
        cell = positions[0]
        packet_meta[int(pid)] = {
            "uav": int(cell[0]),
            "rb": int(cell[1]),
            "minislot": int(cell[2]),
            "mode": int(env.mode_grid[int(cell[0]), int(cell[1]), int(cell[2])]),
        }
    return packet_meta


def _run_single_scenario(policy: str, embb_users: int, urllc_users: int, packet_bits: int, seed: int) -> Dict[str, object]:
    cfg_dict = _build_policy_config(
        policy=policy,
        embb_users=embb_users,
        urllc_users=urllc_users,
        packet_bits=packet_bits,
        channel_uses=None,
        lambda_per_user=None,
        target_error_probability=None,
        mappo_checkpoint_path=None,
        geometry_profile=None,
        min_overlay_retention=None,
    )
    cfg = _clone_cfg(cfg_dict)
    _enable_phase_a_joint_minrate_protection(cfg)
    component_overrides = _component_overrides(cfg_dict)
    if policy in {"greedy", "pure_puncturing", "pure_superposition", "random_scheduler"}:
        cfg.action.max_candidate_packets = max(
            int(getattr(cfg.action, "max_candidate_packets", 0) or 0),
            int(_candidate_pool_upper_bound(component_overrides)),
        )

    env = _build_env(cfg, component_overrides=component_overrides)
    env._eval_method_name = policy
    _configure_eval_env(env, total_load=None, mix_ratio=None, explicit_mix_weights=None)
    observations, _ = env.reset(seed=seed)

    rng = np.random.default_rng(int(seed))
    selector = _selector_for_policy(policy)
    decision_logs: List[Dict[str, object]] = []
    packet_stats = defaultdict(
        lambda: {
            "raw_seen": False,
            "selected_seen": False,
            "raw_feasible_any": False,
            "selected_feasible_any": False,
            "reasons": set(),
            "last_seen_minislot": -1,
            "last_feasible_minislot": -1,
        }
    )

    while True:
        if bool(getattr(env, "episode_done", False)):
            break
        planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
        if planning_phase:
            if policy == "random_scheduler":
                joint_actions = {aid: HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0) for aid in env.agent_ids}
            else:
                joint_actions = selector(env, observations)
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

        minislot, rb = env._current_cell()
        raw_candidates_by_uav: List[List[object]] = []
        env.agent_ids
        for uav_idx in range(env.sys_cfg.num_uavs):
            raw_candidates_by_uav.append(env._enumerate_candidates_for_cell(uav_idx, rb, minislot))
        env._annotate_candidate_contention(raw_candidates_by_uav)
        selected_by_uav: List[List[object]] = []
        for uav_idx in range(env.sys_cfg.num_uavs):
            selected_by_uav.append(env._select_candidate_subset(raw_candidates_by_uav[uav_idx], minislot, uav_idx))

        for uav_idx in range(env.sys_cfg.num_uavs):
            for candidate in raw_candidates_by_uav[uav_idx]:
                pid = int(candidate.packet_id)
                stats = packet_stats[pid]
                stats["raw_seen"] = True
                stats["last_seen_minislot"] = int(minislot)
                raw_feasible = bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible)
                stats["raw_feasible_any"] = bool(stats["raw_feasible_any"]) or raw_feasible
                if raw_feasible:
                    stats["last_feasible_minislot"] = int(minislot)
                stats["reasons"].update(_reason_flags(candidate))
            for candidate in selected_by_uav[uav_idx]:
                pid = int(candidate.packet_id)
                stats = packet_stats[pid]
                stats["selected_seen"] = True
                selected_feasible = bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible)
                stats["selected_feasible_any"] = bool(stats["selected_feasible_any"]) or selected_feasible

        if policy == "random_scheduler":
            joint_actions, resolved = _random_policy_actions(env, observations, rng)
        else:
            joint_actions = selector(env, observations)
            resolved = env._resolve_executed_actions(
                joint_actions,
                observations,
                minislot=int(minislot),
                rb=int(rb),
            )

        action_log = []
        for uav_idx, agent_id in enumerate(env.agent_ids):
            obs = observations[agent_id]
            raw = raw_candidates_by_uav[uav_idx]
            selected = selected_by_uav[uav_idx]
            shielded = resolved[agent_id]
            candidate = shielded.candidate
            action_log.append(
                {
                    "agent_id": str(agent_id),
                    "uav": int(uav_idx),
                    "raw_candidate_count": int(len(raw)),
                    "raw_overlay_feasible": int(sum(int(bool(c.overlay_feasible)) for c in raw)),
                    "raw_puncture_feasible": int(sum(int(bool(c.puncture_feasible)) for c in raw)),
                    "selected_candidate_count": int(len(selected)),
                    "selected_overlay_feasible": int(sum(int(bool(c.overlay_feasible)) for c in selected)),
                    "selected_puncture_feasible": int(sum(int(bool(c.puncture_feasible)) for c in selected)),
                    "chosen_mode": str(MODE_NAME.get(int(shielded.action.mode), int(shielded.action.mode))),
                    "chosen_packet_id": int(candidate.packet_id) if candidate is not None else -1,
                }
            )
        decision_logs.append(
            {
                "minislot": int(minislot),
                "rb": int(rb),
                "actions": action_log,
            }
        )

        observations, _rewards, dones, _infos = env.step(
            joint_actions,
            prebuilt_observations=observations,
            pre_resolved_actions=resolved,
        )
        if all(dones.values()):
            break

    packet_selection_meta = _extract_selected_packet_ids(env)
    packet_records: List[Dict[str, object]] = []
    primary_reason_counts = Counter()
    multi_reason_counts = Counter()
    admitted_packet_ids: List[int] = []
    blocked_packet_ids: List[int] = []
    for pid in range(int(env.num_packets)):
        stats = packet_stats[int(pid)]
        scheduled = int(env.scheduled_uavs[pid]) >= 0
        reasons = sorted(set(stats["reasons"]))
        record = {
            "packet_id": int(pid),
            "release_minislot": int(env.packet_release_minislots[pid]) if pid < env.packet_release_minislots.size else -1,
            "source_user": int(env.packet_sources[pid]) if pid < env.packet_sources.size else -1,
            "scheduled": bool(scheduled),
            "raw_seen": bool(stats["raw_seen"]),
            "selected_seen": bool(stats["selected_seen"]),
            "raw_feasible_any": bool(stats["raw_feasible_any"]),
            "selected_feasible_any": bool(stats["selected_feasible_any"]),
            "last_seen_minislot": int(stats["last_seen_minislot"]),
            "last_feasible_minislot": int(stats["last_feasible_minislot"]),
            "reasons": reasons,
        }
        if scheduled:
            admitted_packet_ids.append(int(pid))
            record.update(packet_selection_meta.get(int(pid), {}))
        else:
            blocked_packet_ids.append(int(pid))
            primary = _classify_blocked_packet(stats)
            record["primary_reason"] = str(primary)
            primary_reason_counts[str(primary)] += 1
            for reason in reasons:
                multi_reason_counts[str(reason)] += 1
            if bool(stats["selected_feasible_any"]):
                multi_reason_counts["feasible_but_not_selected"] += 1
            if bool(stats["raw_feasible_any"]) and not bool(stats["selected_seen"]):
                multi_reason_counts["pruned_by_candidate_subset"] += 1
        packet_records.append(record)

    return {
        "policy": str(policy),
        "embb_users": int(embb_users),
        "urllc_users": int(urllc_users),
        "packet_bits": int(packet_bits),
        "seed": int(seed),
        "arrivals": int(env.num_packets),
        "arrivals_by_minislot": [int(x) for x in np.asarray(env.packet_arrivals_by_minislot, dtype=int).tolist()],
        "packet_sources": [int(x) for x in np.asarray(env.packet_sources, dtype=int).tolist()],
        "packet_release_minislots": [int(x) for x in np.asarray(env.packet_release_minislots, dtype=int).tolist()],
        "admitted_count": int(len(admitted_packet_ids)),
        "blocked_count": int(len(blocked_packet_ids)),
        "admission_ratio": float(len(admitted_packet_ids) / max(int(env.num_packets), 1)) if int(env.num_packets) > 0 else 1.0,
        "throughput_mbps": float(env.summarize_episode().get("embb_total_rate_after_puncture_deduction", 0.0) / 1e6),
        "total_power": float(env.summarize_episode().get("total_power", 0.0)),
        "selected_packets": admitted_packet_ids,
        "blocked_packets": blocked_packet_ids,
        "blocked_primary_reason_counts": dict(primary_reason_counts),
        "blocked_multi_reason_counts": dict(multi_reason_counts),
        "decision_logs": decision_logs,
        "packet_records": packet_records,
        "final_embb_user_tx_powers": list(_extract_final_embb_user_tx_powers(env)),
    }


def _compare_k_pair(base: Dict[str, object], other: Dict[str, object]) -> Dict[str, object]:
    base_admitted = set(int(x) for x in list(base.get("selected_packets", []) or []))
    other_admitted = set(int(x) for x in list(other.get("selected_packets", []) or []))
    base_packets = {int(row["packet_id"]): row for row in list(base.get("packet_records", []) or [])}
    other_packets = {int(row["packet_id"]): row for row in list(other.get("packet_records", []) or [])}
    lost = sorted(base_admitted - other_admitted)
    gained = sorted(other_admitted - base_admitted)
    lost_reason_counts = Counter()
    gained_reason_counts = Counter()
    for pid in lost:
        for reason in list(other_packets.get(int(pid), {}).get("reasons", []) or []):
            lost_reason_counts[str(reason)] += 1
    for pid in gained:
        for reason in list(base_packets.get(int(pid), {}).get("reasons", []) or []):
            gained_reason_counts[str(reason)] += 1
    return {
        "from_k": int(base.get("packet_bits", 0) or 0),
        "to_k": int(other.get("packet_bits", 0) or 0),
        "lost_admitted_packet_count": int(len(lost)),
        "gained_admitted_packet_count": int(len(gained)),
        "lost_packets": lost,
        "gained_packets": gained,
        "lost_packet_reason_counts_at_to_k": dict(lost_reason_counts),
        "gained_packet_reason_counts_at_from_k": dict(gained_reason_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze simple-throughput baseline trends across U and k.")
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--urllc-users", default="10,20,30")
    parser.add_argument("--packet-bits", default="24")
    parser.add_argument("--policies", default="greedy,pure_puncturing,pure_superposition,random_scheduler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    embb_users = int(args.embb_users)
    urllc_users_list = [int(token.strip()) for token in str(args.urllc_users).split(",") if token.strip()]
    packet_bits_list = [int(token.strip()) for token in str(args.packet_bits).split(",") if token.strip()]
    policies = [str(token).strip() for token in str(args.policies).split(",") if token.strip()]
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenarios: Dict[str, Dict[str, object]] = {}
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)

    for urllc_users in urllc_users_list:
        for packet_bits in packet_bits_list:
            scene_id = _build_scene_id(
                embb_users=int(embb_users),
                urllc_users=int(urllc_users),
                packet_bits=int(packet_bits),
                channel_setting_index=1,
                share_scene_across_packet_bits=True,
            )
            env_backup = _apply_channel_setting_env(scene_id)
            try:
                for policy in policies:
                    result = _run_single_scenario(
                        policy=str(policy),
                        embb_users=int(embb_users),
                        urllc_users=int(urllc_users),
                        packet_bits=int(packet_bits),
                        seed=int(args.seed),
                    )
                    key = f"{policy}__u{urllc_users}__k{packet_bits}"
                    scenarios[key] = result
                    grouped[(str(policy), int(urllc_users))].append(result)
            finally:
                _restore_env(env_backup)

    comparisons: Dict[str, object] = {}
    for (policy, urllc_users), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda item: int(item["packet_bits"]))
        arrival_signatures = [
            (
                tuple(int(x) for x in row.get("packet_sources", []) or []),
                tuple(int(x) for x in row.get("packet_release_minislots", []) or []),
            )
            for row in rows_sorted
        ]
        same_arrivals = all(sig == arrival_signatures[0] for sig in arrival_signatures[1:]) if arrival_signatures else True
        pairwise = []
        for idx in range(len(rows_sorted) - 1):
            pairwise.append(_compare_k_pair(rows_sorted[idx], rows_sorted[idx + 1]))
        comparisons[f"{policy}__u{urllc_users}"] = {
            "same_arrivals_across_k": bool(same_arrivals),
            "k_summaries": [
                {
                    "packet_bits": int(row["packet_bits"]),
                    "throughput_mbps": float(row["throughput_mbps"]),
                    "admission_ratio": float(row["admission_ratio"]),
                    "blocked_primary_reason_counts": dict(row["blocked_primary_reason_counts"]),
                }
                for row in rows_sorted
            ],
            "pairwise": pairwise,
        }

    payload = {
        "embb_users": int(embb_users),
        "seed": int(args.seed),
        "scenarios": scenarios,
        "comparisons": comparisons,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
