"""Unified baseline runner for greedy, random, MAPPO, and IPPO comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import _bootstrap  # noqa: F401
from .compare import _build_main_like_configs
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .evaluate import (
    _hard_feasible_throughput_actions,
    _myopic_throughput_actions,
    _policy_actions,
)
from .ippo_baseline import IPPOBaselineTrainer
from .networks import SRMAPPOActorCritic
from .trainer import (
    configure_env_for_users_per_uav,
    disable_eval_fallback,
    restore_eval_fallback,
    run_training_loop,
)
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, AgentObservation, CandidatePacket, HybridAction


SUPPORTED_POLICIES = {
    "random_scheduler",
    "pure_puncturing",
    "pure_superposition",
    "greedy",
    "global_frontier_greedy",
    "mappo",
    "mappo_overlay_forced",
    "mappo_puncture_forced",
    "ippo",
}

MIX_PRESETS = {
    "7:3": 0.3,
    "5:5": 0.5,
    "3:7": 0.7,
}

NONPREFIXED_TERMINAL_REWARD_KEYS = {
    "urllc_admission_over_service_tradeoff_penalty",
    "phaseA_power_reduction_l2_penalty",
    "phaseA_power_saturation_penalty",
    "embb_service_floor_hinge_penalty",
}


def _clone_cfg(config: SRMAPPOConfig | Dict[str, object] | None) -> SRMAPPOConfig:
    if isinstance(config, SRMAPPOConfig):
        return deepcopy(config)
    cfg = SRMAPPOConfig()
    if not isinstance(config, dict):
        return cfg

    training_cfg = dict(config.get("training", {}) or {})
    env_cfg = dict(config.get("env", {}) or {})
    reward_cfg = dict(config.get("reward", {}) or {})
    shield_cfg = dict(config.get("shield", {}) or {})
    action_cfg = dict(config.get("action", {}) or {})
    network_cfg = dict(config.get("network", {}) or {})

    for key, value in training_cfg.items():
        if hasattr(cfg.training, key):
            setattr(cfg.training, key, value)
    for key, value in env_cfg.items():
        if hasattr(cfg.env, key):
            setattr(cfg.env, key, value)
    for key, value in reward_cfg.items():
        if hasattr(cfg.reward, key):
            setattr(cfg.reward, key, value)
    for key, value in shield_cfg.items():
        if hasattr(cfg.shield, key):
            setattr(cfg.shield, key, value)
    for key, value in action_cfg.items():
        if hasattr(cfg.action, key):
            setattr(cfg.action, key, value)
    for key, value in network_cfg.items():
        if hasattr(cfg.network, key):
            setattr(cfg.network, key, value)
    return cfg


def _build_env(cfg: SRMAPPOConfig, component_overrides: Optional[Dict[str, Dict[str, object]]] = None) -> SRMAPPOPhaseAEnv:
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()
    overrides = dict(component_overrides or {})
    system_overrides = dict(overrides.get("system", {}) or {})
    channel_uses_override = system_overrides.pop("channel_uses_per_minislot", None)
    for name, target in (
        ("system", sys_cfg),
        ("urllc", urllc_cfg),
        ("embb", embb_cfg),
        ("algorithm", algo_cfg),
        ("simulation", sim_cfg),
    ):
        source_overrides = system_overrides if name == "system" else dict(overrides.get(name, {}) or {})
        for key, value in source_overrides.items():
            if hasattr(target, key):
                setattr(target, key, value)
    if hasattr(sys_cfg, "refresh_derived_params"):
        sys_cfg.refresh_derived_params()
    if channel_uses_override is not None:
        sys_cfg.channel_uses_per_minislot = int(max(channel_uses_override, 1))
    sim_cfg.verbose = False
    sim_cfg.plot_results = False
    return SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)


def _configure_eval_env(
    env: SRMAPPOPhaseAEnv,
    *,
    total_load: Optional[float],
    mix_ratio: Optional[float],
) -> float:
    if mix_ratio is not None:
        env.sim_cfg.urllc_user_ratio = float(np.clip(mix_ratio, 0.0, 0.95))
    if total_load is None:
        return float((env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / env.sys_cfg.num_uavs)
    return float(configure_env_for_users_per_uav(env, float(total_load)))


def _reference_rate(env: SRMAPPOPhaseAEnv) -> float:
    return float(env._reference_embb_total_rate_for_local_shaping())


def _planning_baseline_action(env: SRMAPPOPhaseAEnv, obs: AgentObservation) -> HybridAction:
    baseline_policy = str(
        getattr(env.rl_cfg.env, "fixed_embb_baseline_policy", "minrate_then_throughput") or "minrate_then_throughput"
    )
    return env._planning_owner_action_for_baseline(obs, baseline_policy)


def _feasible_mode_packet_pairs(obs: AgentObservation, allowed_modes: Iterable[int]) -> List[Tuple[int, int, CandidatePacket]]:
    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
    packet_mask = np.asarray(obs.masks.packet_mask, dtype=float)
    feasible: List[Tuple[int, int, CandidatePacket]] = []
    for packet_option, candidate in enumerate(obs.candidates, start=1):
        for mode in allowed_modes:
            if (
                mode >= mode_mask.size
                or mode_mask[mode] <= 0.5
                or packet_mask.ndim != 2
                or mode >= packet_mask.shape[0]
                or packet_option >= packet_mask.shape[1]
                or packet_mask[mode, packet_option] <= 0.5
                or not candidate.is_mode_feasible(mode)
            ):
                continue
            feasible.append((packet_option, mode, candidate))
    return feasible


def _random_policy_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    rng: np.random.Generator,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    for agent_id, obs in observations.items():
        if planning_phase:
            actions[agent_id] = _planning_baseline_action(env, obs)
            continue
        feasible = _feasible_mode_packet_pairs(obs, (MODE_OVERLAY, MODE_PUNCTURE))
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue
        packet_option, mode, _candidate = feasible[int(rng.integers(0, len(feasible)))]
        actions[agent_id] = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
    return actions


def _puncture_rank_key(
    env: SRMAPPOPhaseAEnv,
    candidate: CandidatePacket,
    rule: str,
) -> Tuple[float, ...]:
    rule = str(rule or "max_urllc_sinr_margin").strip().lower()
    if rule in {"min_embb_rate_loss", "min_rate_loss", "min_loss"}:
        return (-float(candidate.puncture_loss), float(candidate.puncture_reliability), -float(candidate.puncture_power))
    target_linear = float(getattr(env.algo_cfg, "noma_snir_threshold", 0.0) or 0.0)
    margin = float(candidate.puncture_urllc_snir - target_linear)
    return (margin, -float(candidate.puncture_loss), float(candidate.puncture_reliability))


def _pure_puncturing_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    selection_rule: str,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    for agent_id, obs in observations.items():
        if planning_phase:
            actions[agent_id] = _planning_baseline_action(env, obs)
            continue
        feasible = _feasible_mode_packet_pairs(obs, (MODE_PUNCTURE,))
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue
        packet_option, mode, _candidate = max(
            feasible,
            key=lambda item: _puncture_rank_key(env, item[2], selection_rule),
        )
        actions[agent_id] = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
    return actions


def _superposition_rank_key(
    env: SRMAPPOPhaseAEnv,
    candidate: CandidatePacket,
    rule: str,
) -> Tuple[float, ...]:
    rule = str(rule or "max_estimated_global_sum_rate").strip().lower()
    reference_rate = _reference_rate(env)
    packet_bits = float(env._packet_bits_for_user(int(candidate.source_user)))
    minislot_s = max(float(env.sys_cfg.minislot_duration) * 1.0e-3, 1.0e-9)
    urllc_rate = packet_bits * float(candidate.overlay_reliability) / minislot_s
    est_sum_rate = float(reference_rate - candidate.overlay_loss + urllc_rate)
    if rule in {"min_embb_degradation", "minimum_embb_degradation", "min_degradation"}:
        return (-float(candidate.overlay_loss), float(candidate.overlay_retention), float(candidate.overlay_reliability))
    return (est_sum_rate, -float(candidate.overlay_loss), float(candidate.overlay_retention))


def _pure_superposition_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    selection_rule: str,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    for agent_id, obs in observations.items():
        if planning_phase:
            actions[agent_id] = _planning_baseline_action(env, obs)
            continue
        feasible = _feasible_mode_packet_pairs(obs, (MODE_OVERLAY,))
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue
        packet_option, mode, _candidate = max(
            feasible,
            key=lambda item: _superposition_rank_key(env, item[2], selection_rule),
        )
        actions[agent_id] = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
    return actions


def _greedy_actions(env: SRMAPPOPhaseAEnv, observations: Dict[str, AgentObservation]) -> Dict[str, HybridAction]:
    actions, _debug = _myopic_throughput_actions(env, observations)
    return actions


def _global_frontier_greedy_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    for agent_id, obs in observations.items():
        if planning_phase:
            actions[agent_id] = _planning_baseline_action(env, obs)
            continue
        action, _debug = env.global_frontier_greedy_action(obs)
        actions[agent_id] = action
    return actions


def _collect_candidate_violation_counts(observations: Dict[str, AgentObservation]) -> Dict[str, int]:
    counts = {
        "urllc_reliability": 0,
        "embb_min_rate": 0,
        "power": 0,
        "collision": 0,
        "already_scheduled": 0,
        "intercell": 0,
        "deadline": 0,
        "structural": 0,
    }
    for obs in observations.values():
        for candidate in obs.candidates:
            counts["urllc_reliability"] += int(bool(candidate.cause_urllc_sinr_unachievable))
            counts["embb_min_rate"] += int(bool(candidate.cause_embb_retention_below_threshold))
            counts["power"] += int(bool(candidate.cause_required_power_exceeds_budget))
            counts["collision"] += int(bool(candidate.cause_rb_minislot_collision))
            counts["already_scheduled"] += int(bool(candidate.cause_packet_already_scheduled_elsewhere))
            counts["intercell"] += int(bool(candidate.cause_cross_uav_interference_too_high))
            counts["deadline"] += int(bool(candidate.cause_deadline_or_release_violation))
            counts["structural"] += int(bool(candidate.cause_other_structural_reason))
    return counts


def _merge_counts(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0) + int(value))


def _extract_shared_reward_terms(
    infos: Dict[str, Dict[str, object]],
    agent_ids: List[str],
) -> Dict[str, float]:
    for agent_id in agent_ids:
        info = infos.get(agent_id, {})
        if not isinstance(info, dict):
            continue
        reward_terms = info.get("reward_terms", {})
        if isinstance(reward_terms, dict) and reward_terms:
            return {
                str(key): float(value or 0.0)
                for key, value in reward_terms.items()
            }
    return {}


def _update_reward_term_stats(
    stats: Dict[str, Dict[str, float]],
    reward_terms: Dict[str, float],
) -> None:
    for raw_key, raw_value in reward_terms.items():
        key = str(raw_key)
        value = float(raw_value or 0.0)
        entry = stats.setdefault(
            key,
            {
                "sum": 0.0,
                "sum_abs": 0.0,
                "min": math.inf,
                "max": -math.inf,
                "max_abs": 0.0,
                "count": 0.0,
                "finite_count": 0.0,
                "num_nan": 0.0,
                "num_inf": 0.0,
            },
        )
        entry["count"] += 1.0
        if math.isnan(value):
            entry["num_nan"] += 1.0
            continue
        if math.isinf(value):
            entry["num_inf"] += 1.0
            continue
        entry["finite_count"] += 1.0
        entry["sum"] += value
        abs_value = abs(value)
        entry["sum_abs"] += abs_value
        entry["min"] = min(float(entry["min"]), value)
        entry["max"] = max(float(entry["max"]), value)
        entry["max_abs"] = max(float(entry["max_abs"]), abs_value)


def _is_terminal_reward_key(key: str) -> bool:
    normalized = str(key or "")
    return normalized.startswith("terminal_") or normalized in NONPREFIXED_TERMINAL_REWARD_KEYS


def _split_reward_terms(
    reward_terms: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    step_terms: Dict[str, float] = {}
    terminal_terms: Dict[str, float] = {}
    for raw_key, raw_value in dict(reward_terms or {}).items():
        key = str(raw_key)
        value = float(raw_value or 0.0)
        if _is_terminal_reward_key(key):
            terminal_terms[key] = value
        else:
            step_terms[key] = value
    return step_terms, terminal_terms


def _finalize_reward_term_stats(
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    finalized: Dict[str, Dict[str, float]] = {}
    for key, entry in stats.items():
        finite_count = max(float(entry.get("finite_count", 0.0)), 0.0)
        total_sum = float(entry.get("sum", 0.0) or 0.0)
        sum_abs = float(entry.get("sum_abs", 0.0) or 0.0)
        finalized[key] = {
            "sum": total_sum,
            "mean": total_sum / finite_count if finite_count > 0.0 else 0.0,
            "min": float(entry["min"]) if finite_count > 0.0 else 0.0,
            "max": float(entry["max"]) if finite_count > 0.0 else 0.0,
            "mean_abs": sum_abs / finite_count if finite_count > 0.0 else 0.0,
            "num_nan": float(entry.get("num_nan", 0.0) or 0.0),
            "num_inf": float(entry.get("num_inf", 0.0) or 0.0),
            "max_abs": float(entry.get("max_abs", 0.0) or 0.0),
            "count": float(entry.get("count", 0.0) or 0.0),
            "finite_count": finite_count,
        }
    return finalized


def _rank_reward_terms(
    stats: Dict[str, Dict[str, float]],
    *,
    field: str,
) -> List[Dict[str, float]]:
    ranked: List[Dict[str, float]] = []
    for key, payload in stats.items():
        item = {"key": str(key)}
        item.update({name: float(value or 0.0) for name, value in payload.items()})
        ranked.append(item)
    ranked.sort(key=lambda item: abs(float(item.get(field, 0.0) or 0.0)), reverse=True)
    return ranked


def _standardize_metrics(
    summary: Dict[str, object],
    *,
    runtime_sec: float,
    avg_urllc_sinr_linear: float,
    episode_count: float = 1.0,
    episode_step_count: float = 0.0,
    episode_reward_mean: float = 0.0,
    training_curve: Optional[List[Dict[str, float]]] = None,
    constraint_violations: Optional[Dict[str, int]] = None,
    reward_term_stats: Optional[Dict[str, Dict[str, float]]] = None,
    reward_term_rankings: Optional[Dict[str, List[Dict[str, float]]]] = None,
    step_reward_term_stats: Optional[Dict[str, Dict[str, float]]] = None,
    step_reward_term_rankings: Optional[Dict[str, List[Dict[str, float]]]] = None,
    terminal_reward_term_stats: Optional[Dict[str, Dict[str, float]]] = None,
    terminal_reward_term_rankings: Optional[Dict[str, List[Dict[str, float]]]] = None,
    episode_reward_total_sum: float = 0.0,
    logged_step_reward_sum: float = 0.0,
    logged_terminal_reward_sum: float = 0.0,
    reward_trace_steps: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    embb_rate = float(
        summary.get(
            "embb_total_rate_after_puncture_deduction",
            summary.get("embb_total_rate", 0.0),
        )
        or 0.0
    )
    avg_embb_rate = float(
        summary.get(
            "embb_user_rate_mean_after_puncture_deduction",
            summary.get("embb_user_rate_mean", 0.0),
        )
        or 0.0
    )
    scheduled_packets = float(summary.get("scheduled_packets", 0.0) or 0.0)
    active_packets = float(summary.get("active_packets", 0.0) or 0.0)
    embb_user_count = float(summary.get("embb_user_count", 0.0) or 0.0)
    minrate_ok_ratio = float(
        summary.get(
            "embb_min_rate_satisfaction_after_puncture_deduction",
            summary.get("embb_min_rate_satisfaction_ratio", 0.0),
        )
        or 0.0
    )
    minrate_violation_count = max(int(round(embb_user_count * (1.0 - minrate_ok_ratio))), 0)
    avg_sinr_db = 10.0 * np.log10(max(avg_urllc_sinr_linear, 1.0e-15)) if avg_urllc_sinr_linear > 0.0 else 0.0
    logged_total_reward_sum = float(logged_step_reward_sum) + float(logged_terminal_reward_sum)

    return {
        "total_embb_throughput": embb_rate,
        "scheduled_urllc_packets": scheduled_packets,
        "urllc_admission_ratio": float(summary.get("urllc_admission_rate", 0.0) or 0.0),
        "dropped_urllc_packets": max(active_packets - scheduled_packets, 0.0),
        "average_urllc_sinr": float(avg_sinr_db),
        "average_urllc_sinr_db": float(avg_sinr_db),
        "average_embb_rate": avg_embb_rate,
        "embb_minimum_rate_violation_count": float(minrate_violation_count),
        "overlay_action_count": float(summary.get("overlay_count", 0.0) or 0.0),
        "puncturing_action_count": float(summary.get("puncture_count", 0.0) or 0.0),
        "episode_count": float(episode_count),
        "episode_step_count": float(episode_step_count),
        "episode_reward_mean": float(episode_reward_mean),
        "episode_reward_total_sum": float(episode_reward_total_sum),
        "logged_step_reward_sum": float(logged_step_reward_sum),
        "logged_terminal_reward_sum": float(logged_terminal_reward_sum),
        "logged_total_reward_sum": float(logged_total_reward_sum),
        "reward_residual_unlogged": float(episode_reward_total_sum - logged_total_reward_sum),
        "runtime": float(runtime_sec),
        "training_reward_curve": list(training_curve or []),
        "constraint_violation_log": dict(constraint_violations or {}),
        "reward_term_stats": dict(reward_term_stats or {}),
        "reward_term_rankings": dict(reward_term_rankings or {}),
        "step_reward_term_stats": dict(step_reward_term_stats or {}),
        "step_reward_term_rankings": dict(step_reward_term_rankings or {}),
        "terminal_reward_term_stats": dict(terminal_reward_term_stats or {}),
        "terminal_reward_term_rankings": dict(terminal_reward_term_rankings or {}),
        "reward_trace_steps": list(reward_trace_steps or []),
        "raw_summary": dict(summary),
    }


def _rollout_with_selector(
    env: SRMAPPOPhaseAEnv,
    *,
    seed: int,
    selector: Callable[[SRMAPPOPhaseAEnv, Dict[str, AgentObservation]], Dict[str, HybridAction]],
    debug_reward_terms: bool = False,
    debug_reward_steps: bool = False,
    reward_trace_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    observations, _info = env.reset(seed=seed)
    done = False
    avg_sinr_samples: List[float] = []
    constraint_violations: Dict[str, int] = {}
    correction_count = 0
    episode_reward_sum = 0.0
    episode_step_count = 0
    reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    step_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    terminal_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    reward_trace_steps: List[Dict[str, object]] = []
    start = perf_counter()
    fallback_state = disable_eval_fallback(env)
    try:
        while not done:
            _merge_counts(constraint_violations, _collect_candidate_violation_counts(observations))
            planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
            joint_actions = selector(env, observations)
            if planning_phase:
                resolved = {
                    aid: env._raw_action_to_shielded_action(joint_actions[aid], observations[aid])
                    for aid in env.agent_ids
                }
            else:
                minislot, rb = env._current_cell()
                resolved = env._resolve_executed_actions(joint_actions, observations, minislot=minislot, rb=rb)
                for shielded in resolved.values():
                    correction_count += int(
                        bool(shielded.used_greedy_fallback)
                        or bool(shielded.collision_rewritten)
                        or bool(shielded.mode_corrected)
                        or bool(shielded.packet_invalid_fallback)
                        or bool(shielded.mask_invalid_fallback)
                        or bool(shielded.joint_reliability_rewritten)
                    )
                    candidate = shielded.candidate
                    if candidate is None:
                        continue
                    if int(shielded.action.mode) == MODE_OVERLAY:
                        avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                    elif int(shielded.action.mode) == MODE_PUNCTURE:
                        avg_sinr_samples.append(float(candidate.puncture_urllc_snir))

            observations, rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            team_reward = float(np.mean([float(rewards[aid]) for aid in env.agent_ids]))
            if debug_reward_terms:
                all_reward_terms = _extract_shared_reward_terms(_infos, env.agent_ids)
                step_reward_terms, terminal_reward_terms = _split_reward_terms(all_reward_terms)
                _update_reward_term_stats(reward_term_stats_accum, all_reward_terms)
                _update_reward_term_stats(step_reward_term_stats_accum, step_reward_terms)
                _update_reward_term_stats(terminal_reward_term_stats_accum, terminal_reward_terms)
                if debug_reward_steps:
                    trace_entry: Dict[str, object] = {
                        "step_index": int(episode_step_count),
                        "team_reward": float(team_reward),
                        "reward_terms": dict(all_reward_terms),
                        "step_reward_terms": dict(step_reward_terms),
                        "terminal_reward_terms": dict(terminal_reward_terms),
                        "step_reward_sum": float(sum(step_reward_terms.values())),
                        "terminal_reward_sum": float(sum(terminal_reward_terms.values())),
                    }
                    if isinstance(reward_trace_context, dict) and reward_trace_context:
                        trace_entry.update({str(k): v for k, v in reward_trace_context.items()})
                    reward_trace_steps.append(trace_entry)
            episode_reward_sum += team_reward
            episode_step_count += 1
            done = all(dones.values())
    finally:
        restore_eval_fallback(env, fallback_state)

    summary = env.summarize_episode()
    if bool(getattr(env, "enable_mode_downstream_logging", False)):
        try:
            summary.update(env._summarize_mode_downstream_stats())
        except Exception:
            pass
    constraint_violations["action_corrections"] = int(correction_count)
    finalized_reward_term_stats = _finalize_reward_term_stats(reward_term_stats_accum) if debug_reward_terms else {}
    finalized_step_reward_term_stats = (
        _finalize_reward_term_stats(step_reward_term_stats_accum) if debug_reward_terms else {}
    )
    finalized_terminal_reward_term_stats = (
        _finalize_reward_term_stats(terminal_reward_term_stats_accum) if debug_reward_terms else {}
    )
    reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    step_reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_step_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_step_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_step_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    terminal_reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_terminal_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_terminal_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_terminal_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    logged_step_reward_sum = float(sum(item.get("sum", 0.0) or 0.0 for item in finalized_step_reward_term_stats.values()))
    logged_terminal_reward_sum = float(
        sum(item.get("sum", 0.0) or 0.0 for item in finalized_terminal_reward_term_stats.values())
    )
    return _standardize_metrics(
        summary,
        runtime_sec=float(perf_counter() - start),
        avg_urllc_sinr_linear=float(np.mean(avg_sinr_samples)) if avg_sinr_samples else 0.0,
        episode_count=1.0,
        episode_step_count=float(episode_step_count),
        episode_reward_mean=float(episode_reward_sum / max(episode_step_count, 1)),
        episode_reward_total_sum=float(episode_reward_sum),
        training_curve=None,
        constraint_violations=constraint_violations,
        reward_term_stats=finalized_reward_term_stats,
        reward_term_rankings=reward_term_rankings,
        step_reward_term_stats=finalized_step_reward_term_stats,
        step_reward_term_rankings=step_reward_term_rankings,
        terminal_reward_term_stats=finalized_terminal_reward_term_stats,
        terminal_reward_term_rankings=terminal_reward_term_rankings,
        logged_step_reward_sum=logged_step_reward_sum,
        logged_terminal_reward_sum=logged_terminal_reward_sum,
        reward_trace_steps=reward_trace_steps if debug_reward_steps else None,
    )


def _load_mappo_model(
    cfg: SRMAPPOConfig,
    checkpoint_path: str | Path,
    component_overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[SRMAPPOPhaseAEnv, SRMAPPOActorCritic]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_cfg = payload.get("cfg")
    if isinstance(checkpoint_cfg, SRMAPPOConfig):
        model_cfg = replace(checkpoint_cfg)
    elif isinstance(checkpoint_cfg, dict):
        model_cfg = _clone_cfg(checkpoint_cfg)
    else:
        model_cfg = replace(cfg)

    # Keep the checkpoint's policy/reward structure, but honor caller-side
    # arrival semantics so MAPPO and non-learning baselines are evaluated on
    # the same lambda definition in unified sweeps.
    model_cfg.env.urllc_poisson_rate_is_per_user = bool(cfg.env.urllc_poisson_rate_is_per_user)
    model_cfg.env.urllc_poisson_rate_is_slot_level = bool(cfg.env.urllc_poisson_rate_is_slot_level)
    env = _build_env(model_cfg, component_overrides=component_overrides)
    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return env, model


def _extract_training_curve_from_history(history: List[Dict[str, object]]) -> List[Dict[str, float]]:
    curve: List[Dict[str, float]] = []
    for record in history:
        rollout = dict(record.get("rollout", {}) or {})
        update = dict(record.get("update", {}) or {})
        curve.append(
            {
                "iteration": float(record.get("iteration", len(curve) + 1)),
                "mean_reward": float(rollout.get("mean_reward", 0.0) or 0.0),
                "rollout_steps": float(rollout.get("num_steps", 0.0) or 0.0),
                "policy_loss": float(update.get("policy_loss", 0.0) or 0.0),
                "value_loss": float(update.get("value_loss", 0.0) or 0.0),
                "entropy": float(update.get("entropy", 0.0) or 0.0),
            }
        )
    return curve


def _rollout_with_mappo_model(
    env: SRMAPPOPhaseAEnv,
    model: SRMAPPOActorCritic,
    *,
    seed: int,
    forced_mode: Optional[int] = None,
    debug_reward_terms: bool = False,
    debug_reward_steps: bool = False,
    reward_trace_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    observations, _ = env.reset(seed=seed)
    done = False
    avg_sinr_samples: List[float] = []
    start = perf_counter()
    episode_reward_sum = 0.0
    episode_step_count = 0
    reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    step_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    terminal_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    reward_trace_steps: List[Dict[str, object]] = []
    actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids), device=model.power_log_std.device)
    fallback_state = disable_eval_fallback(env)
    constraint_violations: Dict[str, int] = {}
    try:
        while not done:
            _merge_counts(constraint_violations, _collect_candidate_violation_counts(observations))
            planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
            joint_actions, actor_hidden, critic_hidden = _policy_actions(
                env,
                model,
                observations,
                actor_hidden,
                critic_hidden,
                deterministic=True,
            )
            if (not planning_phase) and forced_mode in {MODE_OVERLAY, MODE_PUNCTURE}:
                joint_actions = {
                    aid: _force_phase_a_mode_action(
                        observations[aid],
                        joint_actions.get(aid, HybridAction()),
                        int(forced_mode),
                    )
                    for aid in env.agent_ids
                }
            if planning_phase:
                resolved = {
                    aid: env._raw_action_to_shielded_action(joint_actions[aid], observations[aid])
                    for aid in env.agent_ids
                }
            else:
                minislot, rb = env._current_cell()
                resolved = env._resolve_executed_actions(joint_actions, observations, minislot=minislot, rb=rb)
                for shielded in resolved.values():
                    candidate = shielded.candidate
                    if candidate is None:
                        continue
                    if int(shielded.action.mode) == MODE_OVERLAY:
                        avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                    elif int(shielded.action.mode) == MODE_PUNCTURE:
                        avg_sinr_samples.append(float(candidate.puncture_urllc_snir))
            observations, rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            team_reward = float(np.mean([float(rewards[aid]) for aid in env.agent_ids]))
            if debug_reward_terms:
                all_reward_terms = _extract_shared_reward_terms(_infos, env.agent_ids)
                step_reward_terms, terminal_reward_terms = _split_reward_terms(all_reward_terms)
                _update_reward_term_stats(reward_term_stats_accum, all_reward_terms)
                _update_reward_term_stats(step_reward_term_stats_accum, step_reward_terms)
                _update_reward_term_stats(terminal_reward_term_stats_accum, terminal_reward_terms)
                if debug_reward_steps:
                    trace_entry: Dict[str, object] = {
                        "step_index": int(episode_step_count),
                        "team_reward": float(team_reward),
                        "reward_terms": dict(all_reward_terms),
                        "step_reward_terms": dict(step_reward_terms),
                        "terminal_reward_terms": dict(terminal_reward_terms),
                        "step_reward_sum": float(sum(step_reward_terms.values())),
                        "terminal_reward_sum": float(sum(terminal_reward_terms.values())),
                    }
                    if isinstance(reward_trace_context, dict) and reward_trace_context:
                        trace_entry.update({str(k): v for k, v in reward_trace_context.items()})
                    reward_trace_steps.append(trace_entry)
            episode_reward_sum += team_reward
            episode_step_count += 1
            done = all(dones.values())
    finally:
        restore_eval_fallback(env, fallback_state)

    finalized_reward_term_stats = _finalize_reward_term_stats(reward_term_stats_accum) if debug_reward_terms else {}
    finalized_step_reward_term_stats = (
        _finalize_reward_term_stats(step_reward_term_stats_accum) if debug_reward_terms else {}
    )
    finalized_terminal_reward_term_stats = (
        _finalize_reward_term_stats(terminal_reward_term_stats_accum) if debug_reward_terms else {}
    )
    reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    step_reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_step_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_step_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_step_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    terminal_reward_term_rankings = (
        {
            "by_max_abs": _rank_reward_terms(finalized_terminal_reward_term_stats, field="max_abs"),
            "by_mean_abs": _rank_reward_terms(finalized_terminal_reward_term_stats, field="mean_abs"),
            "by_abs_sum": _rank_reward_terms(finalized_terminal_reward_term_stats, field="sum"),
        }
        if debug_reward_terms
        else {}
    )
    logged_step_reward_sum = float(sum(item.get("sum", 0.0) or 0.0 for item in finalized_step_reward_term_stats.values()))
    logged_terminal_reward_sum = float(
        sum(item.get("sum", 0.0) or 0.0 for item in finalized_terminal_reward_term_stats.values())
    )
    return _standardize_metrics(
        ({**env.summarize_episode(), **(env._summarize_mode_downstream_stats() if bool(getattr(env, 'enable_mode_downstream_logging', False)) else {})}),
        runtime_sec=float(perf_counter() - start),
        avg_urllc_sinr_linear=float(np.mean(avg_sinr_samples)) if avg_sinr_samples else 0.0,
        episode_count=1.0,
        episode_step_count=float(episode_step_count),
        episode_reward_mean=float(episode_reward_sum / max(episode_step_count, 1)),
        episode_reward_total_sum=float(episode_reward_sum),
        training_curve=None,
        constraint_violations=constraint_violations,
        reward_term_stats=finalized_reward_term_stats,
        reward_term_rankings=reward_term_rankings,
        step_reward_term_stats=finalized_step_reward_term_stats,
        step_reward_term_rankings=step_reward_term_rankings,
        terminal_reward_term_stats=finalized_terminal_reward_term_stats,
        terminal_reward_term_rankings=terminal_reward_term_rankings,
        logged_step_reward_sum=logged_step_reward_sum,
        logged_terminal_reward_sum=logged_terminal_reward_sum,
        reward_trace_steps=reward_trace_steps if debug_reward_steps else None,
    )


def _candidate_feasible_for_mode(candidate: CandidatePacket, mode: int) -> bool:
    if int(mode) == MODE_OVERLAY:
        return bool(candidate.overlay_feasible)
    if int(mode) == MODE_PUNCTURE:
        return bool(candidate.puncture_feasible)
    return False


def _force_phase_a_mode_action(
    observation: AgentObservation,
    action: HybridAction,
    forced_mode: int,
) -> HybridAction:
    mode_int = int(forced_mode)
    if bool(observation.metadata.get("planning_phase", 0.0)):
        return action
    if mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
        return action
    if int(action.mode) == MODE_KEEP and int(action.packet_option) <= 0:
        return action

    selected_option = int(action.packet_option)
    if 1 <= selected_option <= len(observation.candidates):
        selected_candidate = observation.candidates[selected_option - 1]
        if _candidate_feasible_for_mode(selected_candidate, mode_int):
            return HybridAction(
                mode=mode_int,
                packet_option=selected_option,
                power_delta=float(action.power_delta),
                embb_owner_option=int(action.embb_owner_option),
                embb_power_delta=float(action.embb_power_delta),
            )

    best_option = 0
    best_score = float("-inf")
    for idx, candidate in enumerate(observation.candidates, start=1):
        if not _candidate_feasible_for_mode(candidate, mode_int):
            continue
        score = float(candidate.utility_for_mode(mode_int))
        if score > best_score:
            best_score = score
            best_option = idx

    if best_option <= 0:
        return action

    return HybridAction(
        mode=mode_int,
        packet_option=int(best_option),
        power_delta=float(action.power_delta),
        embb_owner_option=int(action.embb_owner_option),
        embb_power_delta=float(action.embb_power_delta),
    )


def run_policy(
    policy_name: str,
    config: SRMAPPOConfig | Dict[str, object] | None,
    seed: int,
    *,
    debug_reward_terms: bool = False,
    debug_reward_steps: bool = False,
    reward_trace_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run one policy with a unified result schema."""

    normalized = str(policy_name or "").strip().lower()
    if normalized not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy_name={policy_name!r}. Supported={sorted(SUPPORTED_POLICIES)}")

    cfg = _clone_cfg(config)
    config_dict = config if isinstance(config, dict) else {}
    component_overrides = {
        name: dict(config_dict.get(name, {}) or {})
        for name in ("system", "simulation", "algorithm", "urllc", "embb")
    }
    total_load = config_dict.get("total_load") if isinstance(config_dict, dict) else None
    mix_ratio = config_dict.get("mix_ratio") if isinstance(config_dict, dict) else None

    if normalized in {"mappo", "mappo_overlay_forced", "mappo_puncture_forced"}:
        forced_mode = None
        if normalized == "mappo_overlay_forced":
            forced_mode = MODE_OVERLAY
        elif normalized == "mappo_puncture_forced":
            forced_mode = MODE_PUNCTURE
        checkpoint_path = config_dict.get("checkpoint_path")
        if not checkpoint_path and bool(config_dict.get("train", False)):
            result = run_training_loop(cfg, evaluation_fn=None)
            training_curve = _extract_training_curve_from_history(list(result.get("history", []) or []))
            checkpoint_dir = Path(result["checkpoint_dir"])
            final_path = checkpoint_dir / f"{cfg.training.run_name}_final.pt"
            env, model = _load_mappo_model(cfg, final_path, component_overrides=component_overrides)
            env.enable_mode_downstream_logging = True
            env.mode_downstream_horizons = (5, 10)
            _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio)
            metrics = _rollout_with_mappo_model(
                env,
                model,
                seed=seed,
                forced_mode=forced_mode,
                debug_reward_terms=debug_reward_terms,
                debug_reward_steps=debug_reward_steps,
                reward_trace_context=reward_trace_context,
            )
            metrics["training_reward_curve"] = training_curve
            return metrics
        if not checkpoint_path:
            raise ValueError("MAPPO requires config['checkpoint_path'] or config['train']=True.")
        env, model = _load_mappo_model(cfg, checkpoint_path, component_overrides=component_overrides)
        env.enable_mode_downstream_logging = True
        env.mode_downstream_horizons = (5, 10)
        _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio)
        metrics = _rollout_with_mappo_model(
            env,
            model,
            seed=seed,
            forced_mode=forced_mode,
            debug_reward_terms=debug_reward_terms,
            debug_reward_steps=debug_reward_steps,
            reward_trace_context=reward_trace_context,
        )
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        history = list(((payload.get("extra") or {}).get("history") or []))
        metrics["training_reward_curve"] = _extract_training_curve_from_history(history)
        return metrics

    if normalized == "ippo":
        trainer = IPPOBaselineTrainer(cfg, component_overrides=component_overrides)
        trainer.configure_load(total_load, mix_ratio)
        if bool(config_dict.get("checkpoint_path")):
            trainer.load(config_dict["checkpoint_path"])
        elif bool(config_dict.get("train", True)):
            trainer.train(
                seed=seed,
                total_load=total_load,
                mix_ratio=mix_ratio,
                iterations=int(config_dict.get("train_iterations", getattr(cfg.training, "total_iterations", 50))),
                reward_scope=str(config_dict.get("reward_scope", "global")),
            )
            if bool(config_dict.get("save_path")):
                trainer.save(config_dict["save_path"])
        result = trainer.evaluate(seed=seed, total_load=total_load, mix_ratio=mix_ratio, deterministic=True)
        metrics = _standardize_metrics(
            result["summary"],
            runtime_sec=float(result["runtime_sec"]),
            avg_urllc_sinr_linear=float(result.get("avg_urllc_sinr_linear", 0.0) or 0.0),
            training_curve=list(result.get("training_curve", []) or []),
            constraint_violations={},
        )
        return metrics

    env = _build_env(cfg, component_overrides=component_overrides)
    env.enable_mode_downstream_logging = True
    env.mode_downstream_horizons = (5, 10)
    _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio)
    rng = np.random.default_rng(int(seed))

    if normalized == "random_scheduler":
        selector = lambda e, o: _random_policy_actions(e, o, rng)
    elif normalized == "pure_puncturing":
        selection_rule = str(config_dict.get("puncturing_selection_rule", "max_urllc_sinr_margin"))
        selector = lambda e, o: _pure_puncturing_actions(e, o, selection_rule)
    elif normalized == "pure_superposition":
        selection_rule = str(config_dict.get("superposition_selection_rule", "max_estimated_global_sum_rate"))
        selector = lambda e, o: _pure_superposition_actions(e, o, selection_rule)
    elif normalized == "greedy":
        selector = _greedy_actions
    elif normalized == "global_frontier_greedy":
        selector = _global_frontier_greedy_actions
    else:
        raise AssertionError(f"Unexpected non-learning policy: {normalized}")
    return _rollout_with_selector(
        env,
        seed=seed,
        selector=selector,
        debug_reward_terms=debug_reward_terms,
        debug_reward_steps=debug_reward_steps,
        reward_trace_context=reward_trace_context,
    )


def _mean_metric(records: List[Dict[str, object]], key: str) -> float:
    if not records:
        return 0.0
    values = [float(record.get(key, 0.0) or 0.0) for record in records]
    return float(np.mean(np.asarray(values, dtype=float)))


def _plot_series(
    output_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: List[float],
    series_by_label: Dict[str, List[float]],
) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for label, y_values in series_by_label.items():
        plt.plot(x_values, y_values, marker="o", linewidth=2.0, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def run_comparison_experiments(config: SRMAPPOConfig | Dict[str, object] | None) -> Dict[str, object]:
    """Run all configured policies across mixes, loads, and seeds, then save plots."""

    cfg = _clone_cfg(config)
    config_dict = config if isinstance(config, dict) else {}
    policies = list(config_dict.get("policies", ["random_scheduler", "pure_puncturing", "pure_superposition", "greedy", "mappo", "ippo"]))
    seeds = list(config_dict.get("seeds", [42]))
    total_loads = [float(value) for value in config_dict.get("total_system_loads", getattr(cfg.training, "eval_loads", [10.0, 15.0, 20.0, 25.0]))]
    mix_names = list(config_dict.get("mixes", ["7:3", "5:5", "3:7"]))
    output_dir = Path(config_dict.get("output_dir", Path("sr_mappo") / "results" / "unified_baselines"))
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated: Dict[str, Dict[str, Dict[float, Dict[str, object]]]] = {}
    for mix_name in mix_names:
        mix_ratio = MIX_PRESETS.get(mix_name, float(config_dict.get("mix_ratio_override", 0.5)))
        aggregated[mix_name] = {}
        for policy in policies:
            aggregated[mix_name][policy] = {}
            for total_load in total_loads:
                runs: List[Dict[str, object]] = []
                for seed in seeds:
                    policy_cfg = deepcopy(config_dict)
                    policy_cfg["total_load"] = float(total_load)
                    policy_cfg["mix_ratio"] = float(mix_ratio)
                    runs.append(run_policy(policy, policy_cfg if isinstance(config_dict, dict) else cfg, int(seed)))
                aggregated[mix_name][policy][float(total_load)] = {
                    "runs": runs,
                    "mean_metrics": {
                        "total_embb_throughput": _mean_metric(runs, "total_embb_throughput"),
                        "scheduled_urllc_packets": _mean_metric(runs, "scheduled_urllc_packets"),
                        "urllc_admission_ratio": _mean_metric(runs, "urllc_admission_ratio"),
                        "overlay_action_count": _mean_metric(runs, "overlay_action_count"),
                        "puncturing_action_count": _mean_metric(runs, "puncturing_action_count"),
                        "average_embb_rate": _mean_metric(runs, "average_embb_rate"),
                    },
                }

    for mix_name in mix_names:
        mix_dir = output_dir / mix_name.replace(":", "")
        mix_dir.mkdir(parents=True, exist_ok=True)
        _plot_series(
            mix_dir,
            "embb_throughput_vs_load.png",
            f"eMBB Throughput vs Total Load ({mix_name})",
            "Total System Load",
            "eMBB Throughput (bps)",
            total_loads,
            {
                policy: [aggregated[mix_name][policy][load]["mean_metrics"]["total_embb_throughput"] for load in total_loads]
                for policy in policies
            },
        )
        _plot_series(
            mix_dir,
            "scheduled_urllc_vs_load.png",
            f"Scheduled URLLC Packets vs Total Load ({mix_name})",
            "Total System Load",
            "Scheduled URLLC Packets",
            total_loads,
            {
                policy: [aggregated[mix_name][policy][load]["mean_metrics"]["scheduled_urllc_packets"] for load in total_loads]
                for policy in policies
            },
        )
        _plot_series(
            mix_dir,
            "urllc_admission_vs_load.png",
            f"URLLC Admission Ratio vs Total Load ({mix_name})",
            "Total System Load",
            "Admission Ratio",
            total_loads,
            {
                policy: [aggregated[mix_name][policy][load]["mean_metrics"]["urllc_admission_ratio"] for load in total_loads]
                for policy in policies
            },
        )

        plt.figure(figsize=(6.0, 4.5))
        for policy in policies:
            x_values = [aggregated[mix_name][policy][load]["mean_metrics"]["urllc_admission_ratio"] for load in total_loads]
            y_values = [aggregated[mix_name][policy][load]["mean_metrics"]["total_embb_throughput"] for load in total_loads]
            plt.plot(x_values, y_values, marker="o", linewidth=2.0, label=policy)
        plt.title(f"eMBB Throughput vs URLLC Admission ({mix_name})")
        plt.xlabel("URLLC Admission Ratio")
        plt.ylabel("eMBB Throughput (bps)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(mix_dir / "throughput_admission_tradeoff.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7.0, 4.5))
        x_axis = np.arange(len(policies))
        width = 0.38
        overlay_counts = [
            np.mean([aggregated[mix_name][policy][load]["mean_metrics"]["overlay_action_count"] for load in total_loads])
            for policy in policies
        ]
        puncture_counts = [
            np.mean([aggregated[mix_name][policy][load]["mean_metrics"]["puncturing_action_count"] for load in total_loads])
            for policy in policies
        ]
        plt.bar(x_axis - width / 2, puncture_counts, width=width, label="puncturing")
        plt.bar(x_axis + width / 2, overlay_counts, width=width, label="superposition")
        plt.xticks(x_axis, policies, rotation=20, ha="right")
        plt.ylabel("Average Action Count")
        plt.title(f"Action Distribution ({mix_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(mix_dir / "action_distribution.png", dpi=180)
        plt.close()

    reward_curve_dir = output_dir / "training_curves"
    reward_curve_dir.mkdir(parents=True, exist_ok=True)
    for learning_policy in ("mappo", "ippo"):
        if learning_policy not in policies:
            continue
        plt.figure(figsize=(7.0, 4.5))
        for mix_name in mix_names:
            sample_runs = aggregated[mix_name][learning_policy][total_loads[0]]["runs"]
            if not sample_runs:
                continue
            curve = list(sample_runs[0].get("training_reward_curve", []) or [])
            if not curve:
                continue
            x_values = [float(point.get("iteration", idx + 1)) for idx, point in enumerate(curve)]
            y_values = [float(point.get("mean_reward", 0.0) or 0.0) for point in curve]
            plt.plot(x_values, y_values, linewidth=2.0, label=f"{learning_policy}-{mix_name}")
        plt.title(f"Training Reward Curves ({learning_policy.upper()})")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_curve_dir / f"{learning_policy}_training_reward.png", dpi=180)
        plt.close()

    return {
        "results": aggregated,
        "output_dir": str(output_dir.resolve()),
    }
