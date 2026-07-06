from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .compare import _build_main_like_configs
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label, normalize_experiment_line
from .networks import SRMAPPOActorCritic
from .trainer import configure_env_for_users_per_uav
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, AgentObservation, HybridAction


@dataclass
class CleanStepRecord:
    env_id: int
    agent_id: str
    timestep: int
    planning_phase: float
    local_obs: np.ndarray
    global_obs: np.ndarray
    mode_mask: np.ndarray
    packet_mask: np.ndarray
    embb_owner_mask: np.ndarray
    mode_action: int
    packet_action: int
    embb_owner_action: int
    old_log_prob: float
    value: float
    next_value: float
    reward: float
    done: float
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class RolloutWorkerResult:
    records: List[CleanStepRecord]
    episode_summaries: List[Dict[str, float]]
    sampled_loads: List[float]
    sampled_rates: List[float]
    env_steps_per_env: List[int]
    worker_wall_time: float


def _resolve_device(device: Optional[str]) -> torch.device:
    raw = str(device or "auto").strip().lower()
    if raw in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _build_env(cfg: SRMAPPOConfig) -> SRMAPPOPhaseAEnv:
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()
    sim_cfg.verbose = False
    sim_cfg.plot_results = False
    return SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)


def _configure_load(env: SRMAPPOPhaseAEnv, target_load: Optional[float]) -> float:
    if target_load is None:
        return float((env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / env.sys_cfg.num_uavs)
    return float(configure_env_for_users_per_uav(env, float(target_load)))


def _set_episode_conditions(
    env: SRMAPPOPhaseAEnv,
    *,
    target_load: Optional[float],
    urllc_poisson_rate: Optional[float],
) -> float:
    actual_load = _configure_load(env, target_load)
    if urllc_poisson_rate is not None:
        env.sim_cfg.urllc_poisson_rate = float(max(urllc_poisson_rate, 0.0))
        env.sim_cfg.fixed_urllc_poisson_rate = True
    return float(actual_load)


def _sample_episode_conditions(
    env: SRMAPPOPhaseAEnv,
    *,
    rng: np.random.Generator,
    base_target_load: Optional[float],
    random_load_choices: Optional[List[float]],
    urllc_poisson_rate_range: Optional[List[float]],
    urllc_poisson_rate_sampling: str = "uniform",
) -> Dict[str, float]:
    sampled_load = float(base_target_load) if base_target_load is not None else float(
        (env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / env.sys_cfg.num_uavs
    )
    if random_load_choices:
        sampled_load = float(rng.choice(np.asarray(random_load_choices, dtype=float)))
    sampled_rate = None
    if urllc_poisson_rate_range:
        lo = float(min(urllc_poisson_rate_range))
        hi = float(max(urllc_poisson_rate_range))
        sampling_mode = str(urllc_poisson_rate_sampling or "uniform").strip().lower()
        if sampling_mode == "high_bias":
            span = max(hi - lo, 0.0)
            low_hi = lo + 0.22 * span
            mid_hi = lo + 0.56 * span
            draw = float(rng.random())
            if draw < 0.5:
                band_lo, band_hi = mid_hi, hi
            elif draw < 0.8:
                band_lo, band_hi = low_hi, mid_hi
            else:
                band_lo, band_hi = lo, low_hi
            band_hi = max(band_hi, band_lo)
            sampled_rate = float(rng.uniform(band_lo, band_hi)) if band_hi > band_lo else float(band_lo)
        else:
            sampled_rate = float(rng.uniform(lo, hi))
    actual_load = _set_episode_conditions(
        env,
        target_load=sampled_load,
        urllc_poisson_rate=sampled_rate,
    )
    return {
        "target_load": float(sampled_load),
        "actual_load": float(actual_load),
        "urllc_poisson_rate": float(sampled_rate if sampled_rate is not None else env.sim_cfg.urllc_poisson_rate),
    }


def _to_action_dict(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    output,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    for idx, agent_id in enumerate(env.agent_ids):
        obs = observations[agent_id]
        planning_phase = bool(obs.metadata.get("planning_phase", 0.0))
        if planning_phase:
            if bool(getattr(env.rl_cfg.env, "learn_embb_baseline", False)):
                actions[agent_id] = HybridAction(
                    mode=MODE_KEEP,
                    packet_option=0,
                    power_delta=0.0,
                    embb_owner_option=int(output.embb_owner_option[idx].item()),
                    embb_power_delta=float(output.embb_power_delta[idx].item()) if hasattr(output, "embb_power_delta") else 0.0,
                )
            else:
                baseline_policy = str(
                    getattr(env.rl_cfg.env, "fixed_embb_baseline_policy", "minrate_then_throughput")
                    or "minrate_then_throughput"
                )
                actions[agent_id] = env._planning_owner_action_for_baseline(obs, baseline_policy)
            continue
        actions[agent_id] = HybridAction(
            mode=int(output.mode[idx].item()),
            packet_option=int(output.packet_option[idx].item()),
            power_delta=0.0,
            embb_owner_option=int(output.embb_owner_option[idx].item()) if bool(getattr(env.rl_cfg.env, "learn_embb_baseline", False)) else 0,
            embb_power_delta=float(output.embb_power_delta[idx].item()) if hasattr(output, "embb_power_delta") else 0.0,
        )
    return actions


def _mean_summary(records: List[Dict[str, float]], key: str) -> float:
    if not records:
        return 0.0
    values = [float(item.get(key, 0.0) or 0.0) for item in records]
    return float(np.mean(np.asarray(values, dtype=float)))


def _mean_nested_summary(
    records: List[Dict[str, object]],
    key: str,
) -> Dict[str, float]:
    if not records:
        return {}
    accum: Dict[str, List[float]] = {}
    for item in records:
        payload = item.get(key, {})
        if not isinstance(payload, dict):
            continue
        for raw_name, raw_value in payload.items():
            name = str(raw_name)
            accum.setdefault(name, []).append(float(raw_value or 0.0))
    return {
        name: float(np.mean(np.asarray(values, dtype=float)))
        for name, values in accum.items()
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    denom = float(denominator)
    if abs(denom) <= 1.0e-9:
        return 0.0
    return float(numerator) / denom


def _init_episode_packet_tracking() -> Dict[str, object]:
    return {
        "candidate_packet_ids": set(),
        "feasible_packet_ids": set(),
    }


def _update_episode_packet_tracking(
    tracking: Dict[str, object],
    observations: Dict[str, AgentObservation],
) -> None:
    candidate_packet_ids = tracking.setdefault("candidate_packet_ids", set())
    feasible_packet_ids = tracking.setdefault("feasible_packet_ids", set())
    if not isinstance(candidate_packet_ids, set) or not isinstance(feasible_packet_ids, set):
        return
    for obs in observations.values():
        planning_phase = bool(obs.metadata.get("planning_phase", 0.0))
        if planning_phase:
            continue
        for candidate in list(getattr(obs, "candidates", []) or []):
            packet_id = int(getattr(candidate, "packet_id", -1))
            if packet_id < 0:
                continue
            candidate_packet_ids.add(packet_id)
            if bool(getattr(candidate, "overlay_feasible", False)) or bool(getattr(candidate, "puncture_feasible", False)):
                feasible_packet_ids.add(packet_id)


def _episode_packet_breakdown(env: SRMAPPOPhaseAEnv, tracking: Optional[Dict[str, object]]) -> Dict[str, float]:
    candidate_packet_ids = set()
    feasible_packet_ids = set()
    if isinstance(tracking, dict):
        candidate_packet_ids = set(tracking.get("candidate_packet_ids", set()) or set())
        feasible_packet_ids = set(tracking.get("feasible_packet_ids", set()) or set())
    total_arrivals = int(max(getattr(env, "num_packets", 0) or 0, 0))
    admitted_packet_ids = {
        int(packet_id)
        for packet_id, uav in enumerate(np.asarray(getattr(env, "scheduled_uavs", [])))
        if int(uav) >= 0
    }
    candidate_packets = int(len(candidate_packet_ids))
    feasible_packets = int(len(feasible_packet_ids))
    admitted_packets = int(len(admitted_packet_ids))
    blocked_no_candidate = int(max(total_arrivals - candidate_packets, 0))
    blocked_infeasible = int(max(candidate_packets - feasible_packets, 0))
    blocked_resource = int(max(feasible_packets - admitted_packets, 0))
    return {
        "total_arrivals": float(total_arrivals),
        "candidate_packets": float(candidate_packets),
        "feasible_packets": float(feasible_packets),
        "admitted_packets_breakdown": float(admitted_packets),
        "candidate_ratio": _safe_ratio(candidate_packets, total_arrivals),
        "admitted_given_candidate": _safe_ratio(admitted_packets, candidate_packets),
        "feasible_given_candidate": _safe_ratio(feasible_packets, candidate_packets),
        "admitted_given_feasible": _safe_ratio(admitted_packets, feasible_packets),
        "blocked_no_candidate": float(blocked_no_candidate),
        "blocked_infeasible": float(blocked_infeasible),
        "blocked_resource": float(blocked_resource),
    }


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


def _policy_output_slice(output, start: int, end: int):
    return type(
        "EnvPolicyOutput",
        (),
        {
            "mode": output.mode[start:end],
            "packet_option": output.packet_option[start:end],
            "embb_owner_option": output.embb_owner_option[start:end],
            "embb_power_delta": output.embb_power_delta[start:end],
        },
    )()


def _summarize_episode(env: SRMAPPOPhaseAEnv) -> Dict[str, float]:
    summary = env.summarize_episode()
    final_rates = np.asarray(summary.get("embb_user_rates_after_puncture_deduction", []), dtype=float)
    final_blocked_users = float(np.count_nonzero(final_rates <= 1.0e-9)) if final_rates.size > 0 else 0.0
    phase0_blocked_users = float(summary.get("phase0_minrate_blocked_user_count", 0.0) or 0.0)
    embb_jain_after = float(summary.get("jain_fairness", 0.0) or 0.0)
    embb_5th_percentile_after = float(np.percentile(final_rates, 5.0) / 1.0e6) if final_rates.size > 0 else 0.0
    return {
        "embb_rate_mbps": float(summary.get("embb_total_rate_after_puncture_deduction", summary.get("embb_total_rate", 0.0)) or 0.0) / 1.0e6,
        "avg_embb_rate_mbps": float(
            summary.get("embb_user_rate_mean_after_puncture_deduction", summary.get("embb_user_rate_mean", 0.0)) or 0.0
        ) / 1.0e6,
        "embb_min_rate_satisfaction": float(
            summary.get(
                "embb_min_rate_satisfaction_after_puncture_deduction",
                summary.get("embb_min_rate_satisfaction_ratio", 0.0),
            )
            or 0.0
        ),
        "embb_min_rate_shortfall": float(summary.get("embb_min_rate_shortfall", 0.0) or 0.0),
        "admission": float(summary.get("urllc_admission_rate", 0.0) or 0.0),
        "admitted_packets": float(summary.get("scheduled_packets", 0.0) or 0.0),
        "active_packets": float(summary.get("active_packets", 0.0) or 0.0),
        "embb_blocked_users": phase0_blocked_users,
        "phase0_blocked_users": phase0_blocked_users,
        "phase0_partial_minrate_users": float(summary.get("phase0_actual_partial_minrate_user_count", 0.0) or 0.0),
        "phase0_refill_rb_count": float(summary.get("phase0_actual_refill_rb_count", 0.0) or 0.0),
        "phase0_refill_gain_mbps": float(summary.get("phase0_actual_refill_gain_mbps", 0.0) or 0.0),
        "phase0_refill_intercell_delta_over_noise": float(
            summary.get("phase0_actual_refill_intercell_delta_over_noise", 0.0) or 0.0
        ),
        "final_blocked_users": final_blocked_users,
        "phaseA_newly_blocked_users": float(max(final_blocked_users - phase0_blocked_users, 0.0)),
        "urllc_blocked_users": float(
            max(
                float(summary.get("active_packets", 0.0) or 0.0)
                - float(summary.get("scheduled_packets", 0.0) or 0.0),
                0.0,
            )
        ),
        "total_power": float(summary.get("total_power", 0.0) or 0.0),
        "overlay_ratio": float(summary.get("overlay_ratio", 0.0) or 0.0),
        "puncture_ratio": float(summary.get("puncture_ratio", 0.0) or 0.0),
        "aggregate_embb_rate": float(summary.get("aggregate_embb_rate", 0.0) or 0.0),
        "aggregate_embb_reference_rate": float(summary.get("aggregate_embb_reference_rate", 0.0) or 0.0),
        "embb_rate_ratio": float(summary.get("embb_rate_ratio", 0.0) or 0.0),
        "embb_jain_after": embb_jain_after,
        "embb_5th_percentile_after": embb_5th_percentile_after,
        "terminal_embb_rate_ratio": float(summary.get("terminal_embb_rate_ratio", 0.0) or 0.0),
        "embb_rate_target_ratio": float(summary.get("embb_rate_target_ratio", 0.0) or 0.0),
        "terminal_embb_rate_target_ratio": float(summary.get("terminal_embb_rate_target_ratio", 0.0) or 0.0),
        "terminal_embb_rate_guardrail_penalty": float(summary.get("terminal_embb_rate_guardrail_penalty", 0.0) or 0.0),
        "embb_guardrail_violation_amount": float(summary.get("embb_guardrail_violation_amount", 0.0) or 0.0),
        "embb_guardrail_active_count": float(summary.get("embb_guardrail_active_count", 0.0) or 0.0),
        "step_embb_rate_deficit_penalty": float(summary.get("step_embb_rate_deficit_penalty", 0.0) or 0.0),
        "step_embb_rate_deficit_penalty_mean": float(summary.get("step_embb_rate_deficit_penalty_mean", 0.0) or 0.0),
        "step_embb_deficit_target_ratio": float(summary.get("step_embb_deficit_target_ratio", 0.0) or 0.0),
        "step_embb_rate_ratio": float(summary.get("step_embb_rate_ratio", 0.0) or 0.0),
        "step_embb_rate_deficit_amount": float(summary.get("step_embb_rate_deficit_amount", 0.0) or 0.0),
        "step_embb_rate_deficit_active_count": float(summary.get("step_embb_rate_deficit_active_count", 0.0) or 0.0),
        "step_embb_rate_deficit_active_ratio": float(summary.get("step_embb_rate_deficit_active_ratio", 0.0) or 0.0),
        "avg_embb_rate": float(summary.get("avg_embb_rate", 0.0) or 0.0),
        "embb_minrate_satisfied_users": float(summary.get("embb_minrate_satisfied_users", 0.0) or 0.0),
        "embb_minrate_satisfaction_ratio": float(summary.get("embb_minrate_satisfaction_ratio", 0.0) or 0.0),
        "embb_power": float(summary.get("embb_power", 0.0) or 0.0),
        "embb_power_share": float(summary.get("embb_power_share", 0.0) or 0.0),
        "urllc_power": float(summary.get("urllc_power", 0.0) or 0.0),
        "urllc_power_share": float(summary.get("urllc_power_share", 0.0) or 0.0),
        "power_penalty_active_ratio": float(summary.get("power_penalty_active_ratio", 0.0) or 0.0),
        "terminal_embb_min_rate_satisfaction_bonus": float(
            summary.get("terminal_embb_min_rate_satisfaction_bonus", 0.0) or 0.0
        ),
        "terminal_embb_min_rate_satisfaction_bonus_weight": float(
            summary.get("terminal_embb_min_rate_satisfaction_bonus_weight", 0.0) or 0.0
        ),
        "terminal_urllc_admission_tail_reward": float(
            summary.get("terminal_urllc_admission_tail_reward", 0.0) or 0.0
        ),
        "terminal_urllc_admission_tail_weight_ratio": float(
            summary.get("terminal_urllc_admission_tail_weight_ratio", 0.0) or 0.0
        ),
        "urgency_bonus": float(summary.get("urgency_bonus", 0.0) or 0.0),
        "urgency_reward_weight": float(summary.get("urgency_reward_weight", 0.0) or 0.0),
        "episode_sum_schedule_success": float(summary.get("episode_sum_schedule_success", 0.0) or 0.0),
        "episode_sum_urgency_bonus": float(summary.get("episode_sum_urgency_bonus", 0.0) or 0.0),
        "episode_sum_embb_damage": float(summary.get("episode_sum_embb_damage", 0.0) or 0.0),
        "episode_sum_power_penalty": float(summary.get("episode_sum_power_penalty", 0.0) or 0.0),
        "episode_sum_planning_embb_rate_delta_reward": float(
            summary.get("episode_sum_planning_embb_rate_delta_reward", 0.0) or 0.0
        ),
        "episode_sum_step_embb_rate_deficit_penalty": float(
            summary.get("episode_sum_step_embb_rate_deficit_penalty", 0.0) or 0.0
        ),
        "episode_sum_embb_related_reward": float(summary.get("episode_sum_embb_related_reward", 0.0) or 0.0),
        "episode_sum_urllc_related_reward": float(summary.get("episode_sum_urllc_related_reward", 0.0) or 0.0),
        "episode_sum_power_related_reward": float(summary.get("episode_sum_power_related_reward", 0.0) or 0.0),
        "per_step_mean_schedule_success": float(summary.get("per_step_mean_schedule_success", 0.0) or 0.0),
        "per_step_mean_urgency_bonus": float(summary.get("per_step_mean_urgency_bonus", 0.0) or 0.0),
        "per_step_mean_embb_damage": float(summary.get("per_step_mean_embb_damage", 0.0) or 0.0),
        "per_step_mean_power_penalty": float(summary.get("per_step_mean_power_penalty", 0.0) or 0.0),
        "per_step_mean_planning_embb_rate_delta_reward": float(
            summary.get("per_step_mean_planning_embb_rate_delta_reward", 0.0) or 0.0
        ),
        "per_step_mean_step_embb_rate_deficit_penalty": float(
            summary.get("per_step_mean_step_embb_rate_deficit_penalty", 0.0) or 0.0
        ),
        "per_step_mean_embb_related_reward": float(summary.get("per_step_mean_embb_related_reward", 0.0) or 0.0),
        "per_step_mean_urllc_related_reward": float(summary.get("per_step_mean_urllc_related_reward", 0.0) or 0.0),
        "per_step_mean_power_related_reward": float(summary.get("per_step_mean_power_related_reward", 0.0) or 0.0),
        "total_embb_related_reward": float(summary.get("total_embb_related_reward", 0.0) or 0.0),
        "total_urllc_related_reward": float(summary.get("total_urllc_related_reward", 0.0) or 0.0),
        "total_power_related_reward": float(summary.get("total_power_related_reward", 0.0) or 0.0),
        "power_excess_ratio": float(summary.get("power_excess_ratio", 0.0) or 0.0),
        "power_excess_mean": float(summary.get("power_excess_mean", 0.0) or 0.0),
        "admission_ratio": float(summary.get("admission_ratio", 0.0) or 0.0),
        "admission_bonus_pre_target": float(summary.get("admission_bonus_pre_target", 0.0) or 0.0),
        "admission_bonus_tail": float(summary.get("admission_bonus_tail", 0.0) or 0.0),
    }


def _system_power_budget(env: SRMAPPOPhaseAEnv) -> float:
    embb_limits_dbm = list(getattr(env.embb_cfg, "power_limits", []) or [])
    urllc_limits_dbm = list(getattr(env.urllc_cfg, "power_limits", []) or [])
    embb_budget = float(
        sum(
            min(env.allocator._dbm_to_watts(float(value)), env.algo_cfg.power_upper_bound)
            for value in embb_limits_dbm
        )
    )
    urllc_budget = float(
        sum(
            min(env.allocator._dbm_to_watts(float(value)), env.algo_cfg.power_upper_bound)
            for value in urllc_limits_dbm
        )
    )
    return float(max(embb_budget + urllc_budget, 1.0e-9))


def _evaluate_episode_summary(
    env: SRMAPPOPhaseAEnv,
    reward_term_totals: Dict[str, float],
    packet_tracking: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    base = _summarize_episode(env)
    full = env.summarize_episode()
    step_reward_sum = 0.0
    terminal_reward_sum = 0.0
    for key, value in reward_term_totals.items():
        if str(key).startswith("terminal_"):
            terminal_reward_sum += float(value)
        else:
            step_reward_sum += float(value)
    total_power = float(full.get("total_power", 0.0) or 0.0)
    power_budget = _system_power_budget(env)
    power_violation = float(max(total_power / max(power_budget, 1.0e-9) - 1.0, 0.0))
    base.update(
        {
            "step_reward_sum": float(step_reward_sum),
            "terminal_reward_sum": float(terminal_reward_sum),
            "total_reward": float(step_reward_sum + terminal_reward_sum),
            "planning_embb_rate_delta_reward": float(reward_term_totals.get("planning_embb_rate_delta_reward", 0.0) or 0.0),
            "urgency_bonus": float(reward_term_totals.get("urgency_bonus", 0.0) or 0.0),
            "embb_damage": float(reward_term_totals.get("embb_damage", 0.0) or 0.0),
            "power_penalty": float(reward_term_totals.get("power_penalty", 0.0) or 0.0),
            "step_embb_rate_deficit_penalty": float(
                reward_term_totals.get("step_embb_rate_deficit_penalty", 0.0) or 0.0
            ),
            "terminal_urllc_admission": float(reward_term_totals.get("terminal_urllc_admission", 0.0) or 0.0),
            "terminal_embb_minrate_bonus": float(
                reward_term_totals.get("terminal_embb_min_rate_satisfaction_bonus", 0.0) or 0.0
            ),
            "terminal_embb_rate_guardrail_penalty": float(
                reward_term_totals.get("terminal_embb_rate_guardrail_penalty", 0.0) or 0.0
            ),
            "terminal_power_budget_penalty": float(
                reward_term_totals.get("terminal_total_power_budget_penalty", 0.0) or 0.0
            ),
            "mean_intercell_interference_mw": float(full.get("mean_intercell_interference_mw", 0.0) or 0.0),
            "mean_intercell_interference_over_noise": float(
                full.get("mean_intercell_interference_over_noise", 0.0) or 0.0
            ),
            "selected_action_intercell_cost_after_source_mask_mean": float(
                full.get("selected_action_intercell_cost_after_source_mask_mean", 0.0) or 0.0
            ),
            "selected_action_intercell_cost_after_source_mask_over_noise_mean": float(
                full.get("selected_action_intercell_cost_after_source_mask_over_noise_mean", 0.0) or 0.0
            ),
            "intercell_per_admitted_packet": float(full.get("intercell_per_admitted_packet", 0.0) or 0.0),
            "embb_rate_loss_due_to_intercell_ratio": float(
                full.get("embb_rate_loss_due_to_intercell_ratio", 0.0) or 0.0
            ),
            "overlay_count": float(full.get("overlay_count", 0.0) or 0.0),
            "puncturing_count": float(full.get("puncture_count", 0.0) or 0.0),
            "power_violation": float(power_violation),
            "aggregate_embb_rate": float(full.get("aggregate_embb_rate", 0.0) or 0.0),
            "aggregate_embb_reference_rate": float(full.get("aggregate_embb_reference_rate", 0.0) or 0.0),
            "embb_rate_ratio": float(full.get("embb_rate_ratio", 0.0) or 0.0),
            "terminal_embb_rate_ratio": float(full.get("terminal_embb_rate_ratio", 0.0) or 0.0),
            "embb_rate_target_ratio": float(full.get("embb_rate_target_ratio", 0.0) or 0.0),
            "terminal_embb_rate_target_ratio": float(full.get("terminal_embb_rate_target_ratio", 0.0) or 0.0),
            "embb_guardrail_violation_amount": float(full.get("embb_guardrail_violation_amount", 0.0) or 0.0),
            "embb_guardrail_active_count": float(full.get("embb_guardrail_active_count", 0.0) or 0.0),
            "step_embb_rate_deficit_penalty": float(full.get("step_embb_rate_deficit_penalty", 0.0) or 0.0),
            "step_embb_rate_deficit_penalty_mean": float(full.get("step_embb_rate_deficit_penalty_mean", 0.0) or 0.0),
            "step_embb_deficit_target_ratio": float(full.get("step_embb_deficit_target_ratio", 0.0) or 0.0),
            "step_embb_rate_ratio": float(full.get("step_embb_rate_ratio", 0.0) or 0.0),
            "step_embb_rate_deficit_amount": float(full.get("step_embb_rate_deficit_amount", 0.0) or 0.0),
            "step_embb_rate_deficit_active_count": float(full.get("step_embb_rate_deficit_active_count", 0.0) or 0.0),
            "step_embb_rate_deficit_active_ratio": float(full.get("step_embb_rate_deficit_active_ratio", 0.0) or 0.0),
            "avg_embb_rate": float(full.get("avg_embb_rate", 0.0) or 0.0),
            "embb_minrate_satisfied_users": float(full.get("embb_minrate_satisfied_users", 0.0) or 0.0),
            "embb_minrate_satisfaction_ratio": float(full.get("embb_minrate_satisfaction_ratio", 0.0) or 0.0),
            "embb_power": float(full.get("embb_power", 0.0) or 0.0),
            "embb_power_share": float(full.get("embb_power_share", 0.0) or 0.0),
            "urllc_power": float(full.get("urllc_power", 0.0) or 0.0),
            "urllc_power_share": float(full.get("urllc_power_share", 0.0) or 0.0),
            "power_penalty_active_ratio": float(full.get("power_penalty_active_ratio", 0.0) or 0.0),
            "terminal_embb_min_rate_satisfaction_bonus": float(
                full.get("terminal_embb_min_rate_satisfaction_bonus", 0.0) or 0.0
            ),
            "terminal_embb_min_rate_satisfaction_bonus_weight": float(
                full.get("terminal_embb_min_rate_satisfaction_bonus_weight", 0.0) or 0.0
            ),
            "terminal_urllc_admission_tail_reward": float(
                full.get("terminal_urllc_admission_tail_reward", 0.0) or 0.0
            ),
            "terminal_urllc_admission_tail_weight_ratio": float(
                full.get("terminal_urllc_admission_tail_weight_ratio", 0.0) or 0.0
            ),
            "urgency_reward_weight": float(full.get("urgency_reward_weight", 0.0) or 0.0),
            "episode_sum_schedule_success": float(full.get("episode_sum_schedule_success", 0.0) or 0.0),
            "episode_sum_urgency_bonus": float(full.get("episode_sum_urgency_bonus", 0.0) or 0.0),
            "episode_sum_embb_damage": float(full.get("episode_sum_embb_damage", 0.0) or 0.0),
            "episode_sum_power_penalty": float(full.get("episode_sum_power_penalty", 0.0) or 0.0),
            "episode_sum_planning_embb_rate_delta_reward": float(
                full.get("episode_sum_planning_embb_rate_delta_reward", 0.0) or 0.0
            ),
            "episode_sum_step_embb_rate_deficit_penalty": float(
                full.get("episode_sum_step_embb_rate_deficit_penalty", 0.0) or 0.0
            ),
            "episode_sum_embb_related_reward": float(full.get("episode_sum_embb_related_reward", 0.0) or 0.0),
            "episode_sum_urllc_related_reward": float(full.get("episode_sum_urllc_related_reward", 0.0) or 0.0),
            "episode_sum_power_related_reward": float(full.get("episode_sum_power_related_reward", 0.0) or 0.0),
            "per_step_mean_schedule_success": float(full.get("per_step_mean_schedule_success", 0.0) or 0.0),
            "per_step_mean_urgency_bonus": float(full.get("per_step_mean_urgency_bonus", 0.0) or 0.0),
            "per_step_mean_embb_damage": float(full.get("per_step_mean_embb_damage", 0.0) or 0.0),
            "per_step_mean_power_penalty": float(full.get("per_step_mean_power_penalty", 0.0) or 0.0),
            "per_step_mean_planning_embb_rate_delta_reward": float(
                full.get("per_step_mean_planning_embb_rate_delta_reward", 0.0) or 0.0
            ),
            "per_step_mean_step_embb_rate_deficit_penalty": float(
                full.get("per_step_mean_step_embb_rate_deficit_penalty", 0.0) or 0.0
            ),
            "per_step_mean_embb_related_reward": float(full.get("per_step_mean_embb_related_reward", 0.0) or 0.0),
            "per_step_mean_urllc_related_reward": float(full.get("per_step_mean_urllc_related_reward", 0.0) or 0.0),
            "per_step_mean_power_related_reward": float(full.get("per_step_mean_power_related_reward", 0.0) or 0.0),
            "total_embb_related_reward": float(full.get("total_embb_related_reward", 0.0) or 0.0),
            "total_urllc_related_reward": float(full.get("total_urllc_related_reward", 0.0) or 0.0),
            "total_power_related_reward": float(full.get("total_power_related_reward", 0.0) or 0.0),
            "power_excess_ratio": float(full.get("power_excess_ratio", 0.0) or 0.0),
            "power_excess_mean": float(full.get("power_excess_mean", 0.0) or 0.0),
            "admission_ratio": float(full.get("admission_ratio", 0.0) or 0.0),
            "admission_bonus_pre_target": float(full.get("admission_bonus_pre_target", 0.0) or 0.0),
            "admission_bonus_tail": float(full.get("admission_bonus_tail", 0.0) or 0.0),
        }
    )
    base.update(_episode_packet_breakdown(env, packet_tracking))
    return base


def _reward_weight_snapshot(env: SRMAPPOPhaseAEnv) -> Dict[str, float]:
    reward = env.rl_cfg.reward
    return {
        "planning_embb_rate_weight": float(getattr(reward, "planning_embb_rate_weight", 0.0) or 0.0),
        "planning_phase0_total_power_penalty_weight": float(
            getattr(reward, "planning_phase0_total_power_penalty_weight", 0.0) or 0.0
        ),
        "planning_phase0_positive_power_delta_penalty_weight": float(
            getattr(reward, "planning_phase0_positive_power_delta_penalty_weight", 0.0) or 0.0
        ),
        "planning_embb_service_weight": float(getattr(reward, "planning_embb_service_weight", 0.0) or 0.0),
        "planning_embb_min_rate_weight": float(getattr(reward, "planning_embb_min_rate_weight", 0.0) or 0.0),
        "planning_embb_fairness_weight": float(getattr(reward, "planning_embb_fairness_weight", 0.0) or 0.0),
        "planning_cell_edge_weight": float(getattr(reward, "planning_cell_edge_weight", 0.0) or 0.0),
        "planning_phase0_intercell_penalty_weight": float(
            getattr(reward, "planning_phase0_intercell_penalty_weight", 0.0) or 0.0
        ),
        "planning_phase0_blocked_user_penalty_weight": float(
            getattr(reward, "planning_phase0_blocked_user_penalty_weight", 0.0) or 0.0
        ),
        "planning_phase0_near_zero_user_penalty_weight": float(
            getattr(reward, "planning_phase0_near_zero_user_penalty_weight", 0.0) or 0.0
        ),
        "planning_phase0_near_zero_rate_ratio": float(
            getattr(reward, "planning_phase0_near_zero_rate_ratio", 0.35) or 0.35
        ),
        "schedule_success_weight": float(getattr(reward, "schedule_success_weight", 0.0) or 0.0),
        "urgency_reward_weight": float(getattr(reward, "urgency_reward_weight", 0.0) or 0.0),
        "embb_damage_weight": float(getattr(reward, "embb_damage_weight", 0.0) or 0.0),
        "power_penalty_scale": float(getattr(reward, "power_penalty_scale", 0.0) or 0.0),
        "power_penalty_soft_budget": float(getattr(reward, "power_penalty_soft_budget", 0.0) or 0.0),
        "overlay_power_surcharge_weight": float(getattr(reward, "overlay_power_surcharge_weight", 0.0) or 0.0),
        "terminal_embb_rate_weight": float(getattr(reward, "terminal_embb_rate_weight", 0.0) or 0.0),
        "terminal_urllc_admission_weight": float(getattr(reward, "terminal_urllc_admission_weight", 0.0) or 0.0),
        "terminal_urllc_admission_tail_weight_ratio": float(
            getattr(reward, "terminal_urllc_admission_tail_weight_ratio", 0.0) or 0.0
        ),
        "terminal_embb_rate_guardrail_penalty_weight": float(
            getattr(reward, "terminal_embb_rate_guardrail_penalty_weight", 0.0) or 0.0
        ),
        "terminal_embb_rate_target_ratio": float(getattr(reward, "terminal_embb_rate_target_ratio", 0.0) or 0.0),
        "step_embb_deficit_weight": float(getattr(reward, "step_embb_deficit_weight", 0.0) or 0.0),
        "step_embb_deficit_target_ratio": float(getattr(reward, "step_embb_deficit_target_ratio", 0.0) or 0.0),
        "step_intercell_outgoing_delta_penalty_weight": float(
            getattr(reward, "step_intercell_outgoing_delta_penalty_weight", 0.0) or 0.0
        ),
        "step_action_intercell_penalty_weight": float(
            getattr(reward, "step_action_intercell_penalty_weight", 0.0) or 0.0
        ),
        "terminal_embb_min_rate_satisfaction_bonus_weight": float(
            getattr(reward, "terminal_embb_min_rate_satisfaction_bonus_weight", 0.0) or 0.0
        ),
        "terminal_intercell_rate_loss_ratio_penalty_weight": float(
            getattr(reward, "terminal_intercell_rate_loss_ratio_penalty_weight", 0.0) or 0.0
        ),
        "terminal_intercell_power_penalty_weight": float(
            getattr(reward, "terminal_intercell_power_penalty_weight", 0.0) or 0.0
        ),
        "terminal_puncture_intercell_penalty_weight": float(
            getattr(reward, "terminal_puncture_intercell_penalty_weight", 0.0) or 0.0
        ),
        "terminal_overlay_intercell_penalty_weight": float(
            getattr(reward, "terminal_overlay_intercell_penalty_weight", 0.0) or 0.0
        ),
        "terminal_total_power_budget_penalty_weight": float(
            getattr(reward, "terminal_total_power_budget_penalty_weight", 0.0) or 0.0
        ),
        "terminal_power_saturation_penalty_weight": float(
            getattr(reward, "terminal_power_saturation_penalty_weight", 0.0) or 0.0
        ),
        "terminal_embb_rate_normalizer": float(getattr(reward, "terminal_embb_rate_normalizer", 0.0) or 0.0),
    }


def _episode_env_steps_expected(env: SRMAPPOPhaseAEnv) -> int:
    coexistence_steps = int(len(getattr(env, "_cell_schedule", []) or []))
    planning_steps = int(len(getattr(env, "_embb_plan_schedule", []) or [])) if bool(env.rl_cfg.env.learn_embb_baseline) else 0
    return int(coexistence_steps + planning_steps)


def _rollout_env_seed(base_seed: int, env_idx: int, episode_serial: int) -> int:
    return int(base_seed + env_idx * 1_000_003 + episode_serial * 9_176)


def _rollout_horizon_agent_transitions(
    *,
    rollout_horizon: int,
    rollout_horizon_env_steps: int,
    num_rollout_envs: int,
    agent_count: int,
) -> int:
    if int(rollout_horizon_env_steps or 0) > 0:
        return int(max(rollout_horizon_env_steps, 1) * max(num_rollout_envs, 1) * max(agent_count, 1))
    return int(max(rollout_horizon, 1))


def _rollout_horizon_env_steps_per_env(
    *,
    rollout_horizon_agent_transitions: int,
    num_rollout_envs: int,
    agent_count: int,
) -> float:
    return float(rollout_horizon_agent_transitions) / float(max(num_rollout_envs, 1) * max(agent_count, 1))


def _consistency_snapshot(
    env: SRMAPPOPhaseAEnv,
    *,
    target_load: Optional[float],
    urllc_poisson_rate: Optional[float],
    deterministic_action: bool,
    rollout_horizon: Optional[int] = None,
    rollout_horizon_env_steps: Optional[int] = None,
    num_rollout_envs: int = 1,
    eval_episodes: Optional[int] = None,
) -> Dict[str, object]:
    agent_count = int(len(env.agent_ids))
    episode_env_steps_expected = int(_episode_env_steps_expected(env))
    snapshot: Dict[str, object] = {
        "partial_reuse_enabled": bool(int(os.getenv("SR_MAPPO_ENABLE_PARTIAL_REUSE", "0") or "0")),
        "partial_reuse_pattern": str(os.getenv("SR_MAPPO_PARTIAL_REUSE_PATTERN", "") or ""),
        "target_load": None if target_load is None else float(target_load),
        "urllc_poisson_rate": None if urllc_poisson_rate is None else float(urllc_poisson_rate),
        "num_uavs": int(env.sys_cfg.num_uavs),
        "num_subcarriers": int(env.sys_cfg.num_subcarriers),
        "num_minislots": int(env.sys_cfg.num_minislots),
        "episode_env_steps_expected": episode_env_steps_expected,
        "agent_count": agent_count,
        "num_rollout_envs": int(max(num_rollout_envs, 1)),
        "learn_embb_baseline": bool(env.rl_cfg.env.learn_embb_baseline),
        "multi_rb_agents": bool(env.rl_cfg.env.multi_rb_agents),
        "power_upper_bound": float(getattr(env.algo_cfg, "power_upper_bound", 0.0) or 0.0),
        "allowed_step_reward_terms": list(getattr(env.rl_cfg.reward, "allowed_step_reward_terms", []) or []),
        "allowed_terminal_reward_terms": list(getattr(env.rl_cfg.reward, "allowed_terminal_reward_terms", []) or []),
        "reward_weights": _reward_weight_snapshot(env),
        "deterministic_action": bool(deterministic_action),
    }
    if rollout_horizon is not None:
        resolved_rollout_horizon = int(max(rollout_horizon, 1))
        snapshot["rollout_horizon_agent_transitions"] = resolved_rollout_horizon
        snapshot["rollout_horizon_env_step_equivalent_per_env"] = _rollout_horizon_env_steps_per_env(
            rollout_horizon_agent_transitions=resolved_rollout_horizon,
            num_rollout_envs=int(max(num_rollout_envs, 1)),
            agent_count=agent_count,
        )
        snapshot["effective_episodes_per_iteration"] = float(resolved_rollout_horizon) / float(
            max(agent_count * max(episode_env_steps_expected, 1), 1)
        )
    if rollout_horizon_env_steps is not None and int(rollout_horizon_env_steps) > 0:
        snapshot["rollout_horizon_env_steps"] = int(max(rollout_horizon_env_steps, 1))
    if eval_episodes is not None:
        snapshot["eval_episodes"] = int(max(eval_episodes, 1))
    return snapshot


def _print_consistency_log(tag: str, payload: Dict[str, object]) -> None:
    if False:
        print(f"[{tag}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _evaluate_policy(
    env: SRMAPPOPhaseAEnv,
    model: SRMAPPOActorCritic,
    *,
    episodes: int,
    seed: int,
    target_load: Optional[float],
    nominal_urllc_poisson_rate: Optional[float],
    deterministic_action: bool = True,
    metric_prefix: str = "eval",
) -> Dict[str, float]:
    results: List[Dict[str, float]] = []
    _print_consistency_log(
        "EVAL CONFIG",
        _consistency_snapshot(
            env,
            target_load=target_load,
            urllc_poisson_rate=nominal_urllc_poisson_rate,
            deterministic_action=bool(deterministic_action),
            eval_episodes=episodes,
        ),
    )
    model.eval()
    with torch.no_grad():
        for ep in range(max(int(episodes), 1)):
            _set_episode_conditions(
                env,
                target_load=target_load,
                urllc_poisson_rate=nominal_urllc_poisson_rate,
            )
            observations, _info = env.reset(seed=seed + ep)
            packet_tracking = _init_episode_packet_tracking()
            _update_episode_packet_tracking(packet_tracking, observations)
            done = False
            reward_term_totals: Dict[str, float] = {}
            actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids), device=next(model.parameters()).device)
            while not done:
                local_obs = torch.from_numpy(np.stack([observations[aid].local_obs for aid in env.agent_ids]).astype(np.float32)).to(next(model.parameters()).device)
                global_obs = torch.from_numpy(np.stack([observations[aid].global_obs for aid in env.agent_ids]).astype(np.float32)).to(next(model.parameters()).device)
                mode_mask = torch.from_numpy(np.stack([observations[aid].masks.mode_mask for aid in env.agent_ids]).astype(np.float32)).to(next(model.parameters()).device)
                packet_mask = torch.from_numpy(np.stack([observations[aid].masks.packet_mask for aid in env.agent_ids]).astype(np.float32)).to(next(model.parameters()).device)
                owner_mask = torch.from_numpy(np.stack([observations[aid].masks.embb_owner_mask for aid in env.agent_ids]).astype(np.float32)).to(next(model.parameters()).device)
                output = model.act(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    mode_mask=mode_mask,
                    packet_mask=packet_mask,
                    embb_owner_mask=owner_mask,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                    deterministic=bool(deterministic_action),
                )
                actions = _to_action_dict(env, observations, output)
                observations, _rewards, dones, _infos = env.step(actions, prebuilt_observations=observations)
                _update_episode_packet_tracking(packet_tracking, observations)
                step_reward_terms = _extract_shared_reward_terms(_infos, env.agent_ids)
                for key, value in step_reward_terms.items():
                    reward_term_totals[str(key)] = reward_term_totals.get(str(key), 0.0) + float(value)
                actor_hidden = output.actor_hidden
                critic_hidden = output.critic_hidden
                done = all(dones.values())
            results.append(_evaluate_episode_summary(env, reward_term_totals, packet_tracking))
    prefix = str(metric_prefix or "eval")
    return {
        f"{prefix}_embb_rate_mbps": _mean_summary(results, "embb_rate_mbps"),
        f"{prefix}_avg_embb_rate_mbps": _mean_summary(results, "avg_embb_rate_mbps"),
        f"{prefix}_embb_min_rate_satisfaction": _mean_summary(results, "embb_min_rate_satisfaction"),
        f"{prefix}_embb_min_rate_shortfall": _mean_summary(results, "embb_min_rate_shortfall"),
        f"{prefix}_admission": _mean_summary(results, "admission"),
        f"{prefix}_admitted_packets": _mean_summary(results, "admitted_packets"),
        f"{prefix}_active_packets": _mean_summary(results, "active_packets"),
        f"{prefix}_embb_blocked_users": _mean_summary(results, "embb_blocked_users"),
        f"{prefix}_phase0_blocked_users": _mean_summary(results, "phase0_blocked_users"),
        f"{prefix}_phase0_partial_minrate_users": _mean_summary(results, "phase0_partial_minrate_users"),
        f"{prefix}_phase0_refill_rb_count": _mean_summary(results, "phase0_refill_rb_count"),
        f"{prefix}_phase0_refill_gain_mbps": _mean_summary(results, "phase0_refill_gain_mbps"),
        f"{prefix}_phase0_refill_intercell_delta_over_noise": _mean_summary(
            results, "phase0_refill_intercell_delta_over_noise"
        ),
        f"{prefix}_final_blocked_users": _mean_summary(results, "final_blocked_users"),
        f"{prefix}_phaseA_newly_blocked_users": _mean_summary(results, "phaseA_newly_blocked_users"),
        f"{prefix}_urllc_blocked_users": _mean_summary(results, "urllc_blocked_users"),
        f"{prefix}_total_power": _mean_summary(results, "total_power"),
        f"{prefix}_overlay_ratio": _mean_summary(results, "overlay_ratio"),
        f"{prefix}_puncture_ratio": _mean_summary(results, "puncture_ratio"),
        f"{prefix}_step_reward_sum": _mean_summary(results, "step_reward_sum"),
        f"{prefix}_terminal_reward_sum": _mean_summary(results, "terminal_reward_sum"),
        f"{prefix}_total_reward": _mean_summary(results, "total_reward"),
        f"{prefix}_planning_embb_rate_delta_reward": _mean_summary(results, "planning_embb_rate_delta_reward"),
        f"{prefix}_urgency_bonus": _mean_summary(results, "urgency_bonus"),
        f"{prefix}_embb_damage": _mean_summary(results, "embb_damage"),
        f"{prefix}_power_penalty": _mean_summary(results, "power_penalty"),
        f"{prefix}_step_embb_rate_deficit_penalty": _mean_summary(results, "step_embb_rate_deficit_penalty"),
        f"{prefix}_terminal_urllc_admission": _mean_summary(results, "terminal_urllc_admission"),
        f"{prefix}_terminal_embb_minrate_bonus": _mean_summary(results, "terminal_embb_minrate_bonus"),
        f"{prefix}_terminal_embb_rate_guardrail_penalty": _mean_summary(results, "terminal_embb_rate_guardrail_penalty"),
        f"{prefix}_terminal_power_budget_penalty": _mean_summary(results, "terminal_power_budget_penalty"),
        f"{prefix}_overlay_count": _mean_summary(results, "overlay_count"),
        f"{prefix}_puncturing_count": _mean_summary(results, "puncturing_count"),
        f"{prefix}_power_violation": _mean_summary(results, "power_violation"),
        f"{prefix}_aggregate_embb_rate": _mean_summary(results, "aggregate_embb_rate"),
        f"{prefix}_aggregate_embb_reference_rate": _mean_summary(results, "aggregate_embb_reference_rate"),
        f"{prefix}_embb_rate_ratio": _mean_summary(results, "embb_rate_ratio"),
        f"{prefix}_embb_jain_after": _mean_summary(results, "embb_jain_after"),
        f"{prefix}_embb_5th_percentile_after": _mean_summary(results, "embb_5th_percentile_after"),
        f"{prefix}_embb_rate_target_ratio": _mean_summary(results, "embb_rate_target_ratio"),
        f"{prefix}_embb_guardrail_violation_amount": _mean_summary(results, "embb_guardrail_violation_amount"),
        f"{prefix}_embb_guardrail_active_count": _mean_summary(results, "embb_guardrail_active_count"),
        f"{prefix}_step_embb_rate_deficit_penalty_mean": _mean_summary(results, "step_embb_rate_deficit_penalty_mean"),
        f"{prefix}_step_embb_deficit_target_ratio": _mean_summary(results, "step_embb_deficit_target_ratio"),
        f"{prefix}_step_embb_rate_ratio": _mean_summary(results, "step_embb_rate_ratio"),
        f"{prefix}_step_embb_rate_deficit_amount": _mean_summary(results, "step_embb_rate_deficit_amount"),
        f"{prefix}_step_embb_rate_deficit_active_count": _mean_summary(results, "step_embb_rate_deficit_active_count"),
        f"{prefix}_step_embb_rate_deficit_active_ratio": _mean_summary(results, "step_embb_rate_deficit_active_ratio"),
        f"{prefix}_total_embb_related_reward": _mean_summary(results, "total_embb_related_reward"),
        f"{prefix}_total_urllc_related_reward": _mean_summary(results, "total_urllc_related_reward"),
        f"{prefix}_total_power_related_reward": _mean_summary(results, "total_power_related_reward"),
        f"{prefix}_power_excess_ratio": _mean_summary(results, "power_excess_ratio"),
        f"{prefix}_power_excess_mean": _mean_summary(results, "power_excess_mean"),
        f"{prefix}_admission_ratio": _mean_summary(results, "admission_ratio"),
        f"{prefix}_admission_bonus_pre_target": _mean_summary(results, "admission_bonus_pre_target"),
        f"{prefix}_admission_bonus_tail": _mean_summary(results, "admission_bonus_tail"),
        f"{prefix}_total_arrivals": _mean_summary(results, "total_arrivals"),
        f"{prefix}_candidate_packets": _mean_summary(results, "candidate_packets"),
        f"{prefix}_feasible_packets": _mean_summary(results, "feasible_packets"),
        f"{prefix}_admitted_packets_breakdown": _mean_summary(results, "admitted_packets_breakdown"),
        f"{prefix}_candidate_ratio": _mean_summary(results, "candidate_ratio"),
        f"{prefix}_feasible_given_candidate": _mean_summary(results, "feasible_given_candidate"),
        f"{prefix}_admitted_given_candidate": _mean_summary(results, "admitted_given_candidate"),
        f"{prefix}_admitted_given_feasible": _mean_summary(results, "admitted_given_feasible"),
        f"{prefix}_blocked_no_candidate": _mean_summary(results, "blocked_no_candidate"),
        f"{prefix}_blocked_infeasible": _mean_summary(results, "blocked_infeasible"),
        f"{prefix}_blocked_resource": _mean_summary(results, "blocked_resource"),
        f"{prefix}_phase0_owner_change_ratio_vs_snapshot_raw": _mean_summary(
            results, "phase0_owner_change_ratio_vs_snapshot_raw"
        ),
        f"{prefix}_phase0_owner_change_ratio_vs_snapshot_executed": _mean_summary(
            results, "phase0_owner_change_ratio_vs_snapshot_executed"
        ),
        f"{prefix}_phase0_owner_fallback_to_candidate0_ratio": _mean_summary(
            results, "phase0_owner_fallback_to_candidate0_ratio"
        ),
        f"{prefix}_phase0_owner_invalid_option_ratio": _mean_summary(
            results, "phase0_owner_invalid_option_ratio"
        ),
        f"{prefix}_phase0_owner_null_selected_ratio": _mean_summary(
            results, "phase0_owner_null_selected_ratio"
        ),
        f"{prefix}_phase0_owner_invalid_to_snapshot_ratio": _mean_summary(
            results, "phase0_owner_invalid_to_snapshot_ratio"
        ),
        f"{prefix}_phase0_owner_invalid_to_non_snapshot_ratio": _mean_summary(
            results, "phase0_owner_invalid_to_non_snapshot_ratio"
        ),
        f"{prefix}_phase0_owner_restored_to_snapshot_ratio": _mean_summary(
            results, "phase0_owner_restored_to_snapshot_ratio"
        ),
        f"{prefix}_phase0_owner_replaced_with_non_snapshot_ratio": _mean_summary(
            results, "phase0_owner_replaced_with_non_snapshot_ratio"
        ),
        f"{prefix}_phase0_owner_non_null_ratio_raw": _mean_summary(
            results, "phase0_owner_non_null_ratio_raw"
        ),
        f"{prefix}_phase0_owner_non_null_ratio_executed": _mean_summary(
            results, "phase0_owner_non_null_ratio_executed"
        ),
        f"{prefix}_phase0_owner_changed_and_effective_ratio": _mean_summary(
            results, "phase0_owner_changed_and_effective_ratio"
        ),
        f"{prefix}_phase0_owner_effective_rate_gain_vs_snapshot_mean": _mean_summary(
            results, "phase0_owner_effective_rate_gain_vs_snapshot_mean"
        ),
        f"{prefix}_phase0_snapshot_embb_total_power": _mean_summary(
            results, "phase0_snapshot_embb_total_power"
        ),
        f"{prefix}_phase0_executed_embb_total_power": _mean_summary(
            results, "phase0_executed_embb_total_power"
        ),
        f"{prefix}_phase0_embb_power_delta_mean": _mean_summary(
            results, "phase0_embb_power_delta_mean"
        ),
        f"{prefix}_phase0_executed_vs_snapshot_power_ratio": _mean_summary(
            results, "phase0_executed_vs_snapshot_power_ratio"
        ),
        f"{prefix}_phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps": _mean_summary(
            results, "phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps"
        ),
        f"{prefix}_phase0_owner_change_harmful_ratio": _mean_summary(
            results, "phase0_owner_change_harmful_ratio"
        ),
        f"{prefix}_owner_snapshot_fallback_taken": _mean_summary(
            results, "owner_snapshot_fallback_taken"
        ),
    }


def _evaluate_stress_suite(
    env: SRMAPPOPhaseAEnv,
    model: SRMAPPOActorCritic,
    *,
    episodes_per_lambda: int,
    seed: int,
    target_load: float,
    lambdas: List[float],
) -> Dict[str, float]:
    lambda_rows: List[Dict[str, float]] = []
    for idx, lam in enumerate(list(lambdas or [])):
        result = _evaluate_policy(
            env,
            model,
            episodes=int(max(episodes_per_lambda, 1)),
            seed=int(seed + 10_000 * idx),
            target_load=float(target_load),
            nominal_urllc_poisson_rate=float(lam),
        )
        result["lambda_per_user"] = float(lam)
        lambda_rows.append(result)

    if not lambda_rows:
        return {}

    return {
        "stress_eval_embb_rate_mbps": float(np.mean([row["eval_embb_rate_mbps"] for row in lambda_rows])),
        "stress_eval_embb_min_rate_satisfaction": float(np.mean([row["eval_embb_min_rate_satisfaction"] for row in lambda_rows])),
        "stress_eval_embb_min_rate_shortfall": float(np.mean([row["eval_embb_min_rate_shortfall"] for row in lambda_rows])),
        "stress_eval_admission": float(np.mean([row["eval_admission"] for row in lambda_rows])),
        "stress_eval_admitted_packets": float(np.mean([row["eval_admitted_packets"] for row in lambda_rows])),
        "stress_eval_active_packets": float(np.mean([row["eval_active_packets"] for row in lambda_rows])),
        "stress_eval_total_power": float(np.mean([row["eval_total_power"] for row in lambda_rows])),
        "stress_eval_overlay_ratio": float(np.mean([row["eval_overlay_ratio"] for row in lambda_rows])),
        "stress_eval_puncture_ratio": float(np.mean([row["eval_puncture_ratio"] for row in lambda_rows])),
        "stress_eval_lambda_min": float(min(float(row["lambda_per_user"]) for row in lambda_rows)),
        "stress_eval_lambda_max": float(max(float(row["lambda_per_user"]) for row in lambda_rows)),
    }


def _stack_batch(records: List[CleanStepRecord], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "local_obs": torch.from_numpy(np.stack([r.local_obs for r in records]).astype(np.float32)).to(device),
        "global_obs": torch.from_numpy(np.stack([r.global_obs for r in records]).astype(np.float32)).to(device),
        "mode_mask": torch.from_numpy(np.stack([r.mode_mask for r in records]).astype(np.float32)).to(device),
        "packet_mask": torch.from_numpy(np.stack([r.packet_mask for r in records]).astype(np.float32)).to(device),
        "embb_owner_mask": torch.from_numpy(np.stack([r.embb_owner_mask for r in records]).astype(np.float32)).to(device),
        "mode_actions": torch.tensor([r.mode_action for r in records], dtype=torch.long, device=device),
        "packet_actions": torch.tensor([r.packet_action for r in records], dtype=torch.long, device=device),
        "embb_owner_actions": torch.tensor([r.embb_owner_action for r in records], dtype=torch.long, device=device),
        "old_log_prob": torch.tensor([r.old_log_prob for r in records], dtype=torch.float32, device=device),
        "values": torch.tensor([r.value for r in records], dtype=torch.float32, device=device),
        "rewards": torch.tensor([r.reward for r in records], dtype=torch.float32, device=device),
        "dones": torch.tensor([r.done for r in records], dtype=torch.float32, device=device),
        "advantages": torch.tensor([r.advantage for r in records], dtype=torch.float32, device=device),
        "returns": torch.tensor([r.return_value for r in records], dtype=torch.float32, device=device),
    }


def _finalize_stream_advantages(
    stream_records: List[CleanStepRecord],
    *,
    gamma: float,
    gae_lambda: float,
) -> None:
    last_gae = 0.0
    for record in reversed(stream_records):
        next_non_terminal = 1.0 - float(record.done)
        delta = float(record.reward) + float(gamma) * float(record.next_value) * next_non_terminal - float(record.value)
        last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        record.advantage = float(last_gae)
        record.return_value = float(record.advantage + float(record.value))


def _normalize_record_advantages(records: List[CleanStepRecord]) -> None:
    if not records:
        return
    advantages = np.asarray([float(record.advantage) for record in records], dtype=np.float32)
    mean = float(np.mean(advantages))
    std = float(np.std(advantages))
    scale = max(std, 1.0e-8)
    for record in records:
        record.advantage = float((float(record.advantage) - mean) / scale)


def _rollout_sequential_env_batch(
    envs: List[SRMAPPOPhaseAEnv],
    model: SRMAPPOActorCritic,
    *,
    horizon: int,
    seed: int,
    device: torch.device,
    base_target_load: Optional[float],
    random_load_choices: Optional[List[float]],
    urllc_poisson_rate_range: Optional[List[float]],
    urllc_poisson_rate_sampling: str,
    rollout_horizon_env_steps: int = 0,
    env_id_offset: int = 0,
) -> Dict[str, object]:
    if not envs:
        raise ValueError("envs must contain at least one rollout environment")
    records: List[CleanStepRecord] = []
    episode_summaries: List[Dict[str, float]] = []
    completed_episode_summaries: List[Dict[str, float]] = []
    partial_episode_summaries: List[Dict[str, float]] = []
    num_envs = int(len(envs))
    agent_ids = list(envs[0].agent_ids)
    agent_count = int(len(agent_ids))
    stream_records: Dict[Tuple[int, str], List[CleanStepRecord]] = {
        (env_idx, agent_id): []
        for env_idx in range(num_envs)
        for agent_id in agent_ids
    }
    steps_collected = 0
    episode_serial = 0
    sampled_loads: List[float] = []
    sampled_rates: List[float] = []
    env_rngs = [
        np.random.default_rng(int(seed) + 17 + env_idx * 1009)
        for env_idx in range(num_envs)
    ]
    observations_by_env: List[Optional[Dict[str, AgentObservation]]] = [None] * num_envs
    episode_reward_sum = [0.0 for _ in range(num_envs)]
    episode_step_count = [0 for _ in range(num_envs)]
    episode_reward_terms_sum: List[Dict[str, float]] = [{} for _ in range(num_envs)]
    episode_packet_tracking: List[Dict[str, object]] = [_init_episode_packet_tracking() for _ in range(num_envs)]
    env_step_cursor = [0 for _ in range(num_envs)]

    def _reset_rollout_env(env_idx: int) -> None:
        nonlocal episode_serial
        env = envs[env_idx]
        sampled = _sample_episode_conditions(
            env,
            rng=env_rngs[env_idx],
            base_target_load=base_target_load,
            random_load_choices=random_load_choices,
            urllc_poisson_rate_range=urllc_poisson_rate_range,
            urllc_poisson_rate_sampling=urllc_poisson_rate_sampling,
        )
        sampled_loads.append(float(sampled["actual_load"]))
        sampled_rates.append(float(sampled["urllc_poisson_rate"]))
        env_seed = _rollout_env_seed(int(seed), env_idx, episode_serial)
        episode_serial += 1
        observations, _info = env.reset(seed=env_seed)
        observations_by_env[env_idx] = observations
        episode_reward_sum[env_idx] = 0.0
        episode_step_count[env_idx] = 0
        episode_reward_terms_sum[env_idx] = {}
        episode_packet_tracking[env_idx] = _init_episode_packet_tracking()
        _update_episode_packet_tracking(episode_packet_tracking[env_idx], observations)
        env_step_cursor[env_idx] = 0

    def _finalize_env_episode(env_idx: int, *, completed: bool) -> None:
        env = envs[env_idx]
        summary = _summarize_episode(env)
        summary.update(_episode_packet_breakdown(env, episode_packet_tracking[env_idx]))
        summary["episode_reward_mean"] = float(episode_reward_sum[env_idx] / max(episode_step_count[env_idx], 1))
        summary["episode_reward_components_mean"] = {
            key: float(value / max(episode_step_count[env_idx], 1))
            for key, value in episode_reward_terms_sum[env_idx].items()
        }
        summary["episode_completed"] = 1.0 if completed else 0.0
        episode_summaries.append(summary)
        if completed:
            completed_episode_summaries.append(summary)
        else:
            partial_episode_summaries.append(summary)

    model.eval()
    _print_consistency_log(
        "TRAIN CONFIG",
        _consistency_snapshot(
            envs[0],
            target_load=base_target_load,
            urllc_poisson_rate=None if urllc_poisson_rate_range else envs[0].sim_cfg.urllc_poisson_rate,
            deterministic_action=False,
            rollout_horizon=horizon,
            rollout_horizon_env_steps=rollout_horizon_env_steps,
            num_rollout_envs=num_envs,
        ),
    )
    for env_idx in range(num_envs):
        _reset_rollout_env(env_idx)

    actor_hidden, critic_hidden = model.initial_state(batch_size=num_envs * agent_count, device=device)
    while steps_collected < int(horizon):
        flat_local_obs: List[np.ndarray] = []
        flat_global_obs: List[np.ndarray] = []
        flat_mode_masks: List[np.ndarray] = []
        flat_packet_masks: List[np.ndarray] = []
        flat_owner_masks: List[np.ndarray] = []
        env_agent_pairs: List[Tuple[int, str]] = []
        for env_idx in range(num_envs):
            observations = observations_by_env[env_idx]
            assert observations is not None
            for agent_id in agent_ids:
                obs = observations[agent_id]
                flat_local_obs.append(np.asarray(obs.local_obs, dtype=np.float32))
                flat_global_obs.append(np.asarray(obs.global_obs, dtype=np.float32))
                flat_mode_masks.append(np.asarray(obs.masks.mode_mask, dtype=np.float32))
                flat_packet_masks.append(np.asarray(obs.masks.packet_mask, dtype=np.float32))
                flat_owner_masks.append(np.asarray(obs.masks.embb_owner_mask, dtype=np.float32))
                env_agent_pairs.append((env_idx, agent_id))

        local_obs = torch.from_numpy(np.stack(flat_local_obs).astype(np.float32)).to(device)
        global_obs = torch.from_numpy(np.stack(flat_global_obs).astype(np.float32)).to(device)
        mode_mask = torch.from_numpy(np.stack(flat_mode_masks).astype(np.float32)).to(device)
        packet_mask = torch.from_numpy(np.stack(flat_packet_masks).astype(np.float32)).to(device)
        owner_mask = torch.from_numpy(np.stack(flat_owner_masks).astype(np.float32)).to(device)

        with torch.no_grad():
            output = model.act(
                local_obs=local_obs,
                global_obs=global_obs,
                mode_mask=mode_mask,
                packet_mask=packet_mask,
                embb_owner_mask=owner_mask,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                deterministic=False,
            )

        actions_by_env: List[Dict[str, HybridAction]] = [{} for _ in range(num_envs)]
        next_observations_by_env: List[Dict[str, AgentObservation]] = [dict() for _ in range(num_envs)]
        rewards_by_env: List[Dict[str, float]] = [{} for _ in range(num_envs)]
        dones_by_env: List[Dict[str, bool]] = [{} for _ in range(num_envs)]
        infos_by_env: List[Dict[str, Dict[str, object]]] = [{} for _ in range(num_envs)]
        for env_idx in range(num_envs):
            start = env_idx * agent_count
            end = start + agent_count
            env = envs[env_idx]
            observations = observations_by_env[env_idx]
            assert observations is not None
            actions = _to_action_dict(
                env,
                observations,
                type(
                    "EnvPolicyOutput",
                    (),
                    {
                        "mode": output.mode[start:end],
                        "packet_option": output.packet_option[start:end],
                        "embb_owner_option": output.embb_owner_option[start:end],
                        "embb_power_delta": output.embb_power_delta[start:end],
                    },
                )(),
            )
            actions_by_env[env_idx] = actions
            next_observations, rewards, dones, infos = env.step(actions, prebuilt_observations=observations)
            next_observations_by_env[env_idx] = next_observations
            rewards_by_env[env_idx] = rewards
            dones_by_env[env_idx] = dones
            infos_by_env[env_idx] = infos
            _update_episode_packet_tracking(episode_packet_tracking[env_idx], next_observations)

        bootstrap_next_values = torch.zeros((num_envs * agent_count,), dtype=torch.float32, device=device)
        continuing_pairs: List[Tuple[int, str]] = []
        continuing_next_obs_local: List[np.ndarray] = []
        continuing_next_obs_global: List[np.ndarray] = []
        continuing_next_mode_masks: List[np.ndarray] = []
        continuing_next_packet_masks: List[np.ndarray] = []
        continuing_next_owner_masks: List[np.ndarray] = []
        continuing_hidden_indices: List[int] = []
        for env_idx in range(num_envs):
            if all(bool(dones_by_env[env_idx][agent_id]) for agent_id in agent_ids):
                continue
            for agent_offset, agent_id in enumerate(agent_ids):
                next_obs = next_observations_by_env[env_idx][agent_id]
                continuing_pairs.append((env_idx, agent_id))
                continuing_next_obs_local.append(np.asarray(next_obs.local_obs, dtype=np.float32))
                continuing_next_obs_global.append(np.asarray(next_obs.global_obs, dtype=np.float32))
                continuing_next_mode_masks.append(np.asarray(next_obs.masks.mode_mask, dtype=np.float32))
                continuing_next_packet_masks.append(np.asarray(next_obs.masks.packet_mask, dtype=np.float32))
                continuing_next_owner_masks.append(np.asarray(next_obs.masks.embb_owner_mask, dtype=np.float32))
                continuing_hidden_indices.append(env_idx * agent_count + agent_offset)

        if continuing_pairs:
            continuing_idx_tensor = torch.tensor(continuing_hidden_indices, dtype=torch.long, device=device)
            with torch.no_grad():
                bootstrap_output = model.act(
                    local_obs=torch.from_numpy(np.stack(continuing_next_obs_local).astype(np.float32)).to(device),
                    global_obs=torch.from_numpy(np.stack(continuing_next_obs_global).astype(np.float32)).to(device),
                    mode_mask=torch.from_numpy(np.stack(continuing_next_mode_masks).astype(np.float32)).to(device),
                    packet_mask=torch.from_numpy(np.stack(continuing_next_packet_masks).astype(np.float32)).to(device),
                    embb_owner_mask=torch.from_numpy(np.stack(continuing_next_owner_masks).astype(np.float32)).to(device),
                    actor_hidden=output.actor_hidden.index_select(1, continuing_idx_tensor),
                    critic_hidden=output.critic_hidden.index_select(1, continuing_idx_tensor),
                    deterministic=True,
                )
            for idx, hidden_index in enumerate(continuing_hidden_indices):
                bootstrap_next_values[hidden_index] = bootstrap_output.value[idx]

        for env_idx in range(num_envs):
            rewards = rewards_by_env[env_idx]
            dones = dones_by_env[env_idx]
            infos = infos_by_env[env_idx]
            observations = observations_by_env[env_idx]
            assert observations is not None
            actions = actions_by_env[env_idx]
            for agent_offset, agent_id in enumerate(agent_ids):
                hidden_index = env_idx * agent_count + agent_offset
                record = CleanStepRecord(
                    env_id=int(env_id_offset + env_idx),
                    agent_id=str(agent_id),
                    timestep=int(env_step_cursor[env_idx]),
                    planning_phase=float(observations[agent_id].metadata.get("planning_phase", 0.0) or 0.0),
                    local_obs=np.asarray(observations[agent_id].local_obs, dtype=np.float32),
                    global_obs=np.asarray(observations[agent_id].global_obs, dtype=np.float32),
                    mode_mask=np.asarray(observations[agent_id].masks.mode_mask, dtype=np.float32),
                    packet_mask=np.asarray(observations[agent_id].masks.packet_mask, dtype=np.float32),
                    embb_owner_mask=np.asarray(observations[agent_id].masks.embb_owner_mask, dtype=np.float32),
                    mode_action=int(actions[agent_id].mode),
                    packet_action=int(actions[agent_id].packet_option),
                    embb_owner_action=int(actions[agent_id].embb_owner_option),
                    old_log_prob=float(output.log_prob[hidden_index].item()),
                    value=float(output.value[hidden_index].item()),
                    next_value=float(bootstrap_next_values[hidden_index].item()) if not bool(dones[agent_id]) else 0.0,
                    reward=float(rewards[agent_id]),
                    done=float(dones[agent_id]),
                )
                records.append(record)
                stream_records[(env_idx, agent_id)].append(record)

            observations_by_env[env_idx] = next_observations_by_env[env_idx]
            episode_reward_sum[env_idx] += float(np.mean([float(rewards[aid]) for aid in agent_ids]))
            episode_step_count[env_idx] += 1
            env_step_cursor[env_idx] += 1
            step_reward_terms = _extract_shared_reward_terms(infos, agent_ids)
            for key, value in step_reward_terms.items():
                episode_reward_terms_sum[env_idx][key] = episode_reward_terms_sum[env_idx].get(key, 0.0) + float(value)

            if all(bool(dones[agent_id]) for agent_id in agent_ids):
                _finalize_env_episode(env_idx, completed=True)
                _reset_rollout_env(env_idx)
                start = env_idx * agent_count
                end = start + agent_count
                actor_hidden[:, start:end, :] = 0.0
                critic_hidden[:, start:end, :] = 0.0

        next_actor_hidden = output.actor_hidden.detach().clone()
        next_critic_hidden = output.critic_hidden.detach().clone()
        for env_idx in range(num_envs):
            if all(bool(dones_by_env[env_idx][agent_id]) for agent_id in agent_ids):
                start = env_idx * agent_count
                end = start + agent_count
                next_actor_hidden[:, start:end, :] = 0.0
                next_critic_hidden[:, start:end, :] = 0.0
        actor_hidden = next_actor_hidden
        critic_hidden = next_critic_hidden
        steps_collected += num_envs * agent_count

    for env_idx in range(num_envs):
        if episode_step_count[env_idx] > 0:
            _finalize_env_episode(env_idx, completed=False)

    metrics_episodes = completed_episode_summaries if completed_episode_summaries else episode_summaries

    histogram = _phase_a_action_histogram(records)
    return {
        "records": records,
        "next_seed": int(seed + episode_serial),
        "episodes": episode_summaries,
        "rollout_episode_count": float(len(completed_episode_summaries)),
        "rollout_partial_episode_count": float(len(partial_episode_summaries)),
        "rollout_reward_mean": _mean_summary(metrics_episodes, "episode_reward_mean"),
        "rollout_reward_components_mean": _mean_nested_summary(metrics_episodes, "episode_reward_components_mean"),
        "rollout_embb_rate_mbps": _mean_summary(metrics_episodes, "embb_rate_mbps"),
        "rollout_admission": _mean_summary(metrics_episodes, "admission"),
        "rollout_admitted_packets": _mean_summary(metrics_episodes, "admitted_packets"),
        "rollout_active_packets": _mean_summary(metrics_episodes, "active_packets"),
        "rollout_phase0_blocked_users": _mean_summary(metrics_episodes, "phase0_blocked_users"),
        "rollout_phase0_partial_minrate_users": _mean_summary(metrics_episodes, "phase0_partial_minrate_users"),
        "rollout_phase0_refill_rb_count": _mean_summary(metrics_episodes, "phase0_refill_rb_count"),
        "rollout_phase0_refill_gain_mbps": _mean_summary(metrics_episodes, "phase0_refill_gain_mbps"),
        "rollout_phase0_refill_intercell_delta_over_noise": _mean_summary(
            metrics_episodes, "phase0_refill_intercell_delta_over_noise"
        ),
        "rollout_final_blocked_users": _mean_summary(metrics_episodes, "final_blocked_users"),
        "rollout_phaseA_newly_blocked_users": _mean_summary(metrics_episodes, "phaseA_newly_blocked_users"),
        "rollout_embb_rate_ratio": _mean_summary(metrics_episodes, "embb_rate_ratio"),
        "rollout_embb_jain_after": _mean_summary(metrics_episodes, "embb_jain_after"),
        "rollout_embb_5th_percentile_after": _mean_summary(metrics_episodes, "embb_5th_percentile_after"),
        "rollout_total_embb_related_reward": _mean_summary(metrics_episodes, "total_embb_related_reward"),
        "rollout_total_urllc_related_reward": _mean_summary(metrics_episodes, "total_urllc_related_reward"),
        "rollout_total_power_related_reward": _mean_summary(metrics_episodes, "total_power_related_reward"),
        "rollout_episode_sum_embb_related_reward": _mean_summary(metrics_episodes, "episode_sum_embb_related_reward"),
        "rollout_episode_sum_urllc_related_reward": _mean_summary(metrics_episodes, "episode_sum_urllc_related_reward"),
        "rollout_episode_sum_power_related_reward": _mean_summary(metrics_episodes, "episode_sum_power_related_reward"),
        "rollout_per_step_mean_embb_related_reward": _mean_summary(metrics_episodes, "per_step_mean_embb_related_reward"),
        "rollout_per_step_mean_urllc_related_reward": _mean_summary(metrics_episodes, "per_step_mean_urllc_related_reward"),
        "rollout_per_step_mean_power_related_reward": _mean_summary(metrics_episodes, "per_step_mean_power_related_reward"),
        "rollout_terminal_embb_rate_guardrail_penalty": _mean_summary(metrics_episodes, "terminal_embb_rate_guardrail_penalty"),
        "rollout_step_embb_rate_deficit_penalty": _mean_summary(metrics_episodes, "step_embb_rate_deficit_penalty"),
        "rollout_step_embb_rate_deficit_penalty_mean": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_penalty_mean"
        ),
        "rollout_step_embb_rate_ratio": _mean_summary(metrics_episodes, "step_embb_rate_ratio"),
        "rollout_step_embb_rate_deficit_amount": _mean_summary(metrics_episodes, "step_embb_rate_deficit_amount"),
        "rollout_step_embb_rate_deficit_active_count": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_active_count"
        ),
        "rollout_step_embb_rate_deficit_active_ratio": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_active_ratio"
        ),
        "rollout_embb_guardrail_violation_amount": _mean_summary(metrics_episodes, "embb_guardrail_violation_amount"),
        "rollout_embb_guardrail_active_count": _mean_summary(metrics_episodes, "embb_guardrail_active_count"),
        "rollout_power_excess_ratio": _mean_summary(metrics_episodes, "power_excess_ratio"),
        "rollout_power_excess_mean": _mean_summary(metrics_episodes, "power_excess_mean"),
        "rollout_total_arrivals": _mean_summary(metrics_episodes, "total_arrivals"),
        "rollout_candidate_packets": _mean_summary(metrics_episodes, "candidate_packets"),
        "rollout_feasible_packets": _mean_summary(metrics_episodes, "feasible_packets"),
        "rollout_candidate_ratio": _mean_summary(metrics_episodes, "candidate_ratio"),
        "rollout_feasible_given_candidate": _mean_summary(metrics_episodes, "feasible_given_candidate"),
        "rollout_admitted_given_candidate_packet": _mean_summary(metrics_episodes, "admitted_given_candidate"),
        "rollout_admitted_given_feasible": _mean_summary(metrics_episodes, "admitted_given_feasible"),
        "rollout_blocked_no_candidate": _mean_summary(metrics_episodes, "blocked_no_candidate"),
        "rollout_blocked_infeasible": _mean_summary(metrics_episodes, "blocked_infeasible"),
        "rollout_blocked_resource": _mean_summary(metrics_episodes, "blocked_resource"),
        "rollout_mean_intercell_interference_mw": _mean_summary(metrics_episodes, "mean_intercell_interference_mw"),
        "rollout_mean_intercell_interference_over_noise": _mean_summary(
            metrics_episodes, "mean_intercell_interference_over_noise"
        ),
        "rollout_selected_action_intercell_cost_after_source_mask_mean": _mean_summary(
            metrics_episodes, "selected_action_intercell_cost_after_source_mask_mean"
        ),
        "rollout_selected_action_intercell_cost_after_source_mask_over_noise_mean": _mean_summary(
            metrics_episodes, "selected_action_intercell_cost_after_source_mask_over_noise_mean"
        ),
        "rollout_intercell_per_admitted_packet": _mean_summary(metrics_episodes, "intercell_per_admitted_packet"),
        "rollout_embb_rate_loss_due_to_intercell_ratio": _mean_summary(
            metrics_episodes, "embb_rate_loss_due_to_intercell_ratio"
        ),
        "rollout_admission_bonus_pre_target": _mean_summary(metrics_episodes, "admission_bonus_pre_target"),
        "rollout_admission_bonus_tail": _mean_summary(metrics_episodes, "admission_bonus_tail"),
        "sampled_load_mean": float(np.mean(np.asarray(sampled_loads, dtype=float))) if sampled_loads else 0.0,
        "sampled_urllc_poisson_rate_mean": float(np.mean(np.asarray(sampled_rates, dtype=float))) if sampled_rates else 0.0,
        "rollout_total_agent_transitions": float(steps_collected),
        "rollout_env_steps_per_env": _rollout_horizon_env_steps_per_env(
            rollout_horizon_agent_transitions=steps_collected,
            num_rollout_envs=num_envs,
            agent_count=agent_count,
        ),
        **histogram,
    }


def _finalize_rollout_records(
    records: List[CleanStepRecord],
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    grouped: Dict[Tuple[int, str], List[CleanStepRecord]] = {}
    for record in records:
        grouped.setdefault((int(record.env_id), str(record.agent_id)), []).append(record)
    for stream in grouped.values():
        stream.sort(key=lambda item: int(item.timestep))
        _finalize_stream_advantages(stream, gamma=gamma, gae_lambda=gae_lambda)
    _normalize_record_advantages(records)
    return _stack_batch(records, device)


def _phase_a_action_histogram(records: List[CleanStepRecord]) -> Dict[str, float]:
    phase_a_records = [record for record in records if float(record.planning_phase) < 0.5]
    phase_a_count = float(len(phase_a_records))
    if phase_a_count <= 0.0:
        return {
            "phaseA_count": 0.0,
            "phaseA_has_candidate_ratio": 0.0,
            "phaseA_mode_KEEP_ratio": 0.0,
            "phaseA_mode_OVERLAY_ratio": 0.0,
            "phaseA_mode_PUNCTURE_ratio": 0.0,
            "phaseA_packet_0_ratio": 0.0,
            "phaseA_valid_packet_ratio": 0.0,
            "phaseA_keep_given_candidate": 0.0,
            "phaseA_pkt0_given_candidate": 0.0,
            "phaseA_nonkeep_given_candidate": 0.0,
            "phaseA_feasible_candidate_count_mean": 0.0,
            "rollout_total_feasible_candidates": 0.0,
            "rollout_admitted_given_candidate": 0.0,
        }
    candidate_available_count = 0.0
    feasible_candidate_total = 0.0
    keep_given_candidate_count = 0.0
    pkt0_given_candidate_count = 0.0
    nonkeep_given_candidate_count = 0.0
    for record in phase_a_records:
        packet_mask = np.asarray(record.packet_mask, dtype=np.float32)
        if packet_mask.ndim != 2 or packet_mask.shape[1] <= 1:
            feasible_candidate_count = 0.0
        else:
            feasible_candidate_count = float(
                np.sum(np.any(packet_mask[[MODE_OVERLAY, MODE_PUNCTURE], 1:] > 0.5, axis=0))
            )
        feasible_candidate_total += feasible_candidate_count
        has_candidate = feasible_candidate_count > 0.0
        if has_candidate:
            candidate_available_count += 1.0
            if int(record.mode_action) == MODE_KEEP:
                keep_given_candidate_count += 1.0
            if int(record.packet_action) == 0:
                pkt0_given_candidate_count += 1.0
            if int(record.mode_action) != MODE_KEEP:
                nonkeep_given_candidate_count += 1.0
    keep_count = float(sum(1 for record in phase_a_records if int(record.mode_action) == MODE_KEEP))
    overlay_count = float(sum(1 for record in phase_a_records if int(record.mode_action) == MODE_OVERLAY))
    puncture_count = float(sum(1 for record in phase_a_records if int(record.mode_action) == MODE_PUNCTURE))
    packet_zero_count = float(sum(1 for record in phase_a_records if int(record.packet_action) == 0))
    valid_packet_count = float(sum(1 for record in phase_a_records if int(record.packet_action) > 0))
    return {
        "phaseA_count": phase_a_count,
        "phaseA_has_candidate_ratio": _safe_ratio(candidate_available_count, phase_a_count),
        "phaseA_mode_KEEP_ratio": _safe_ratio(keep_count, phase_a_count),
        "phaseA_mode_OVERLAY_ratio": _safe_ratio(overlay_count, phase_a_count),
        "phaseA_mode_PUNCTURE_ratio": _safe_ratio(puncture_count, phase_a_count),
        "phaseA_packet_0_ratio": _safe_ratio(packet_zero_count, phase_a_count),
        "phaseA_valid_packet_ratio": _safe_ratio(valid_packet_count, phase_a_count),
        "phaseA_keep_given_candidate": _safe_ratio(keep_given_candidate_count, candidate_available_count),
        "phaseA_pkt0_given_candidate": _safe_ratio(pkt0_given_candidate_count, candidate_available_count),
        "phaseA_nonkeep_given_candidate": _safe_ratio(nonkeep_given_candidate_count, candidate_available_count),
        "phaseA_feasible_candidate_count_mean": _safe_ratio(feasible_candidate_total, phase_a_count),
        "rollout_total_feasible_candidates": feasible_candidate_total,
        "rollout_admitted_given_candidate": _safe_ratio(nonkeep_given_candidate_count, candidate_available_count),
    }


def _rollout_worker_collect(
    *,
    cfg: SRMAPPOConfig,
    model_state_dict: Dict[str, torch.Tensor],
    seed: int,
    env_offset: int,
    env_count: int,
    env_steps_per_env: int,
    base_target_load: Optional[float],
    random_load_choices: Optional[List[float]],
    urllc_poisson_rate_range: Optional[List[float]],
    urllc_poisson_rate_sampling: str,
    fixed_urllc_poisson_rate: Optional[float],
    worker_device: str,
) -> RolloutWorkerResult:
    worker_start = perf_counter()
    worker_cfg = deepcopy(cfg)
    worker_envs = [_build_env(worker_cfg) for _ in range(int(max(env_count, 1)))]
    for worker_env in worker_envs:
        _configure_load(worker_env, base_target_load)
        if fixed_urllc_poisson_rate is not None:
            worker_env.sim_cfg.urllc_poisson_rate = float(max(fixed_urllc_poisson_rate, 0.0))
            worker_env.sim_cfg.fixed_urllc_poisson_rate = True
    worker_device_resolved = _resolve_device(worker_device)
    worker_model = SRMAPPOActorCritic(worker_envs[0].local_obs_dim, worker_envs[0].global_obs_dim, worker_cfg).to(worker_device_resolved)
    worker_model.load_state_dict(model_state_dict)
    worker_model.eval()
    agent_count = int(len(worker_envs[0].agent_ids))
    worker_horizon = int(max(env_steps_per_env, 1) * max(env_count, 1) * max(agent_count, 1))
    rollout = _rollout_sequential_env_batch(
        worker_envs,
        worker_model,
        horizon=worker_horizon,
        seed=int(seed + env_offset * 10_000_019),
        device=worker_device_resolved,
        base_target_load=base_target_load,
        random_load_choices=random_load_choices,
        urllc_poisson_rate_range=urllc_poisson_rate_range,
        urllc_poisson_rate_sampling=urllc_poisson_rate_sampling,
        rollout_horizon_env_steps=int(max(env_steps_per_env, 0)),
        env_id_offset=int(env_offset),
    )
    records = list(rollout.get("records", []))
    return RolloutWorkerResult(
        records=records,
        episode_summaries=list(rollout.get("episodes", [])),
        sampled_loads=[float(rollout.get("sampled_load_mean", 0.0) or 0.0)] * int(max(env_count, 1)),
        sampled_rates=[float(rollout.get("sampled_urllc_poisson_rate_mean", 0.0) or 0.0)] * int(max(env_count, 1)),
        env_steps_per_env=[int(max(env_steps_per_env, 0)) for _ in range(int(max(env_count, 1)))],
        worker_wall_time=float(perf_counter() - worker_start),
    )


def _rollout(
    envs: List[SRMAPPOPhaseAEnv],
    model: SRMAPPOActorCritic,
    *,
    horizon: int,
    seed: int,
    device: torch.device,
    base_target_load: Optional[float],
    random_load_choices: Optional[List[float]],
    urllc_poisson_rate_range: Optional[List[float]],
    urllc_poisson_rate_sampling: str,
    rollout_horizon_env_steps: int = 0,
    parallel_rollout_workers: int = 1,
    rollout_worker_device: str = "cpu",
    disable_parallel_rollout: bool = False,
) -> Dict[str, object]:
    agent_count = int(len(envs[0].agent_ids))
    num_envs = int(len(envs))
    if int(rollout_horizon_env_steps or 0) > 0:
        env_steps_per_env = int(max(rollout_horizon_env_steps, 1))
    else:
        env_steps_per_env = int(np.ceil(float(max(horizon, 1)) / float(max(num_envs * agent_count, 1))))
    effective_horizon = int(max(env_steps_per_env, 1) * max(num_envs, 1) * max(agent_count, 1))
    use_parallel = (
        not bool(disable_parallel_rollout)
        and int(max(parallel_rollout_workers, 1)) > 1
        and num_envs > 1
    )
    rollout_start = perf_counter()
    if not use_parallel:
        result = _rollout_sequential_env_batch(
            envs,
            model,
            horizon=effective_horizon,
            seed=seed,
            device=device,
            base_target_load=base_target_load,
            random_load_choices=random_load_choices,
            urllc_poisson_rate_range=urllc_poisson_rate_range,
            urllc_poisson_rate_sampling=urllc_poisson_rate_sampling,
            rollout_horizon_env_steps=env_steps_per_env,
            env_id_offset=0,
        )
        records = list(result.get("records", []))
        batch = _finalize_rollout_records(
            records,
            gamma=float(getattr(model.cfg.training, "gamma", 0.99) or 0.99),
            gae_lambda=float(getattr(model.cfg.training, "gae_lambda", 0.95) or 0.95),
            device=device,
        )
        result["batch"] = batch
        result["rollout_parallel_wall_time"] = float(perf_counter() - rollout_start)
        result["rollout_worker_mean_time"] = float(result["rollout_parallel_wall_time"])
        result["rollout_worker_max_time"] = float(result["rollout_parallel_wall_time"])
        return result

    worker_count = int(min(max(parallel_rollout_workers, 1), num_envs))
    env_counts = [num_envs // worker_count for _ in range(worker_count)]
    for idx in range(num_envs % worker_count):
        env_counts[idx] += 1
    worker_specs: List[Tuple[int, int]] = []
    cursor = 0
    for env_count in env_counts:
        worker_specs.append((cursor, env_count))
        cursor += env_count

    state_dict_cpu = {
        key: value.detach().to("cpu")
        for key, value in model.state_dict().items()
    }
    worker_results: List[RolloutWorkerResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _rollout_worker_collect,
                cfg=deepcopy(model.cfg),
                model_state_dict=state_dict_cpu,
                seed=int(seed),
                env_offset=int(env_offset),
                env_count=int(env_count),
                env_steps_per_env=int(env_steps_per_env),
                base_target_load=base_target_load,
                random_load_choices=random_load_choices,
                urllc_poisson_rate_range=urllc_poisson_rate_range,
                urllc_poisson_rate_sampling=str(urllc_poisson_rate_sampling or "uniform"),
                fixed_urllc_poisson_rate=None if urllc_poisson_rate_range else float(envs[0].sim_cfg.urllc_poisson_rate),
                worker_device=str(rollout_worker_device or "cpu"),
            )
            for env_offset, env_count in worker_specs
        ]
        for future in futures:
            worker_results.append(future.result())

    merged_records: List[CleanStepRecord] = []
    merged_episodes: List[Dict[str, float]] = []
    sampled_loads: List[float] = []
    sampled_rates: List[float] = []
    env_steps_list: List[int] = []
    worker_times: List[float] = []
    for worker_result in worker_results:
        merged_records.extend(worker_result.records)
        merged_episodes.extend(worker_result.episode_summaries)
        sampled_loads.extend(worker_result.sampled_loads)
        sampled_rates.extend(worker_result.sampled_rates)
        env_steps_list.extend(worker_result.env_steps_per_env)
        worker_times.append(float(worker_result.worker_wall_time))

    batch = _finalize_rollout_records(
        merged_records,
        gamma=float(getattr(model.cfg.training, "gamma", 0.99) or 0.99),
        gae_lambda=float(getattr(model.cfg.training, "gae_lambda", 0.95) or 0.95),
        device=device,
    )
    total_transitions = int(len(merged_records))
    completed_episodes = [
        episode for episode in merged_episodes
        if float(episode.get("episode_completed", 0.0) or 0.0) > 0.5
    ]
    partial_episodes = [
        episode for episode in merged_episodes
        if float(episode.get("episode_completed", 0.0) or 0.0) <= 0.5
    ]
    metrics_episodes = completed_episodes if completed_episodes else merged_episodes
    histogram = _phase_a_action_histogram(merged_records)
    return {
        "batch": batch,
        "records": merged_records,
        "next_seed": int(seed + num_envs),
        "episodes": merged_episodes,
        "rollout_episode_count": float(len(completed_episodes)),
        "rollout_partial_episode_count": float(len(partial_episodes)),
        "rollout_reward_mean": _mean_summary(metrics_episodes, "episode_reward_mean"),
        "rollout_reward_components_mean": _mean_nested_summary(metrics_episodes, "episode_reward_components_mean"),
        "rollout_embb_rate_mbps": _mean_summary(metrics_episodes, "embb_rate_mbps"),
        "rollout_admission": _mean_summary(metrics_episodes, "admission"),
        "rollout_admitted_packets": _mean_summary(metrics_episodes, "admitted_packets"),
        "rollout_active_packets": _mean_summary(metrics_episodes, "active_packets"),
        "rollout_embb_rate_ratio": _mean_summary(metrics_episodes, "embb_rate_ratio"),
        "rollout_total_embb_related_reward": _mean_summary(metrics_episodes, "total_embb_related_reward"),
        "rollout_total_urllc_related_reward": _mean_summary(metrics_episodes, "total_urllc_related_reward"),
        "rollout_total_power_related_reward": _mean_summary(metrics_episodes, "total_power_related_reward"),
        "rollout_episode_sum_embb_related_reward": _mean_summary(metrics_episodes, "episode_sum_embb_related_reward"),
        "rollout_episode_sum_urllc_related_reward": _mean_summary(metrics_episodes, "episode_sum_urllc_related_reward"),
        "rollout_episode_sum_power_related_reward": _mean_summary(metrics_episodes, "episode_sum_power_related_reward"),
        "rollout_per_step_mean_embb_related_reward": _mean_summary(metrics_episodes, "per_step_mean_embb_related_reward"),
        "rollout_per_step_mean_urllc_related_reward": _mean_summary(metrics_episodes, "per_step_mean_urllc_related_reward"),
        "rollout_per_step_mean_power_related_reward": _mean_summary(metrics_episodes, "per_step_mean_power_related_reward"),
        "rollout_terminal_embb_rate_guardrail_penalty": _mean_summary(metrics_episodes, "terminal_embb_rate_guardrail_penalty"),
        "rollout_step_embb_rate_deficit_penalty": _mean_summary(metrics_episodes, "step_embb_rate_deficit_penalty"),
        "rollout_step_embb_rate_deficit_penalty_mean": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_penalty_mean"
        ),
        "rollout_step_embb_rate_ratio": _mean_summary(metrics_episodes, "step_embb_rate_ratio"),
        "rollout_step_embb_rate_deficit_amount": _mean_summary(metrics_episodes, "step_embb_rate_deficit_amount"),
        "rollout_step_embb_rate_deficit_active_count": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_active_count"
        ),
        "rollout_step_embb_rate_deficit_active_ratio": _mean_summary(
            metrics_episodes, "step_embb_rate_deficit_active_ratio"
        ),
        "rollout_embb_guardrail_violation_amount": _mean_summary(metrics_episodes, "embb_guardrail_violation_amount"),
        "rollout_embb_guardrail_active_count": _mean_summary(metrics_episodes, "embb_guardrail_active_count"),
        "rollout_power_excess_ratio": _mean_summary(metrics_episodes, "power_excess_ratio"),
        "rollout_power_excess_mean": _mean_summary(metrics_episodes, "power_excess_mean"),
        "rollout_total_arrivals": _mean_summary(metrics_episodes, "total_arrivals"),
        "rollout_candidate_packets": _mean_summary(metrics_episodes, "candidate_packets"),
        "rollout_feasible_packets": _mean_summary(metrics_episodes, "feasible_packets"),
        "rollout_candidate_ratio": _mean_summary(metrics_episodes, "candidate_ratio"),
        "rollout_feasible_given_candidate": _mean_summary(metrics_episodes, "feasible_given_candidate"),
        "rollout_admitted_given_candidate_packet": _mean_summary(metrics_episodes, "admitted_given_candidate"),
        "rollout_admitted_given_feasible": _mean_summary(metrics_episodes, "admitted_given_feasible"),
        "rollout_blocked_no_candidate": _mean_summary(metrics_episodes, "blocked_no_candidate"),
        "rollout_blocked_infeasible": _mean_summary(metrics_episodes, "blocked_infeasible"),
        "rollout_blocked_resource": _mean_summary(metrics_episodes, "blocked_resource"),
        "rollout_admission_bonus_pre_target": _mean_summary(metrics_episodes, "admission_bonus_pre_target"),
        "rollout_admission_bonus_tail": _mean_summary(metrics_episodes, "admission_bonus_tail"),
        "sampled_load_mean": float(np.mean(np.asarray(sampled_loads, dtype=float))) if sampled_loads else 0.0,
        "sampled_urllc_poisson_rate_mean": float(np.mean(np.asarray(sampled_rates, dtype=float))) if sampled_rates else 0.0,
        "rollout_total_agent_transitions": float(total_transitions),
        "rollout_env_steps_per_env": float(np.mean(np.asarray(env_steps_list, dtype=float))) if env_steps_list else 0.0,
        "rollout_parallel_wall_time": float(perf_counter() - rollout_start),
        "rollout_worker_mean_time": float(np.mean(np.asarray(worker_times, dtype=float))) if worker_times else 0.0,
        "rollout_worker_max_time": float(np.max(np.asarray(worker_times, dtype=float))) if worker_times else 0.0,
        **histogram,
    }


def _update_policy(
    model: SRMAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    *,
    ppo_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    total_updates = 0
    batch_size = int(batch["local_obs"].shape[0])
    zeros_column = torch.zeros((batch_size, 1), dtype=torch.float32, device=batch["local_obs"].device)

    for _ in range(int(max(ppo_epochs, 1))):
        indices = torch.randperm(batch_size, device=batch["local_obs"].device)
        for start in range(0, batch_size, int(max(minibatch_size, 1))):
            mb_idx = indices[start : start + int(max(minibatch_size, 1))]
            outputs = model.evaluate_actions(
                local_obs=batch["local_obs"][mb_idx],
                global_obs=batch["global_obs"][mb_idx],
                mode_actions=batch["mode_actions"][mb_idx],
                packet_actions=batch["packet_actions"][mb_idx],
                power_pre_tanh=zeros_column[mb_idx],
                embb_owner_actions=batch["embb_owner_actions"][mb_idx],
                embb_power_pre_tanh=zeros_column[mb_idx],
                mode_mask=batch["mode_mask"][mb_idx],
                packet_mask=batch["packet_mask"][mb_idx],
                embb_owner_mask=batch["embb_owner_mask"][mb_idx],
                actor_hidden=None,
                critic_hidden=None,
            )
            log_ratio = outputs["log_prob"] - batch["old_log_prob"][mb_idx]
            ratio = torch.exp(log_ratio)
            surr1 = ratio * batch["advantages"][mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch["advantages"][mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(outputs["value"], batch["returns"][mb_idx])
            entropy = outputs["entropy"].mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_approx_kl += float(approx_kl.item())
            total_clip_fraction += float(clip_fraction.item())
            total_updates += 1

    denom = max(total_updates, 1)
    model.eval()
    with torch.no_grad():
        full_outputs = model.evaluate_actions(
            local_obs=batch["local_obs"],
            global_obs=batch["global_obs"],
            mode_actions=batch["mode_actions"],
            packet_actions=batch["packet_actions"],
            power_pre_tanh=zeros_column,
            embb_owner_actions=batch["embb_owner_actions"],
            embb_power_pre_tanh=zeros_column,
            mode_mask=batch["mode_mask"],
            packet_mask=batch["packet_mask"],
            embb_owner_mask=batch["embb_owner_mask"],
            actor_hidden=None,
            critic_hidden=None,
        )
        returns = batch["returns"]
        pred_values = full_outputs["value"]
        residual = returns - pred_values
        returns_var = torch.var(returns, unbiased=False)
        if float(returns_var.item()) > 1.0e-12:
            explained_variance = float(1.0 - (torch.var(residual, unbiased=False) / returns_var).item())
        else:
            explained_variance = 0.0
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "approx_kl": total_approx_kl / denom,
        "clip_fraction": total_clip_fraction / denom,
        "explained_variance": explained_variance,
    }


def _checkpoint_dir(cfg: SRMAPPOConfig) -> Path:
    out_dir = Path(cfg.training.checkpoint_dir) / "clean_mappo"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


_CLEAN_CHECKPOINT_HISTORY_TAIL = 200


def _save_clean_checkpoint(
    path: Path,
    *,
    cfg: SRMAPPOConfig,
    model: SRMAPPOActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    history: List[Dict[str, object]],
    target_load: float,
    metadata: Optional[Dict[str, float]] = None,
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    history_tail = list(history[-_CLEAN_CHECKPOINT_HISTORY_TAIL:]) if history else []
    payload = {
        "cfg": cfg,
        "model_state_dict": model.state_dict(),
        "history": history_tail,
        "history_tail": history_tail,
        "history_length": int(len(history)),
        "clean_trainer": True,
        "target_load": float(target_load),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_state:
        payload.update(dict(extra_state))
    torch.save(payload, path)


def _write_clean_history(path: Path, history: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_clean_history_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def _load_clean_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in clean history file: {path}")
    return payload


_TAIPEI_TZ = timezone(timedelta(hours=8))


def _taipei_timestamp() -> str:
    return datetime.now(_TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _clean_eval_metadata(record: Dict[str, float], iteration: int, eval_score: Optional[float] = None) -> Dict[str, float]:
    metadata: Dict[str, float] = {
        "iteration": float(iteration),
        "eval_embb_rate_mbps": float(record["eval_embb_rate_mbps"]),
        "eval_embb_min_rate_satisfaction": float(record["eval_embb_min_rate_satisfaction"]),
        "eval_embb_min_rate_shortfall": float(record["eval_embb_min_rate_shortfall"]),
        "eval_admission": float(record["eval_admission"]),
        "eval_deterministic_admission": float(record.get("eval_deterministic_admission", record["eval_admission"])),
        "eval_stochastic_admission": float(record.get("eval_stochastic_admission", 0.0) or 0.0),
        "eval_admission_gap": float(record.get("eval_admission_gap", 0.0) or 0.0),
        "eval_admitted_packets": float(record["eval_admitted_packets"]),
        "eval_active_packets": float(record["eval_active_packets"]),
        "eval_embb_blocked_users": float(record.get("eval_embb_blocked_users", 0.0) or 0.0),
        "eval_phase0_blocked_users": float(record.get("eval_phase0_blocked_users", 0.0) or 0.0),
        "eval_phase0_partial_minrate_users": float(record.get("eval_phase0_partial_minrate_users", 0.0) or 0.0),
        "eval_phase0_refill_rb_count": float(record.get("eval_phase0_refill_rb_count", 0.0) or 0.0),
        "eval_phase0_refill_gain_mbps": float(record.get("eval_phase0_refill_gain_mbps", 0.0) or 0.0),
        "eval_phase0_refill_intercell_delta_over_noise": float(
            record.get("eval_phase0_refill_intercell_delta_over_noise", 0.0) or 0.0
        ),
        "eval_final_blocked_users": float(record.get("eval_final_blocked_users", 0.0) or 0.0),
        "eval_phaseA_newly_blocked_users": float(record.get("eval_phaseA_newly_blocked_users", 0.0) or 0.0),
        "eval_urllc_blocked_users": float(record.get("eval_urllc_blocked_users", 0.0) or 0.0),
        "eval_embb_jain_after": float(record.get("eval_embb_jain_after", 0.0) or 0.0),
        "eval_embb_5th_percentile_after": float(record.get("eval_embb_5th_percentile_after", 0.0) or 0.0),
        "phaseA_has_candidate_ratio": float(record.get("phaseA_has_candidate_ratio", 0.0) or 0.0),
        "phaseA_mode_KEEP_ratio": float(record.get("phaseA_mode_KEEP_ratio", 0.0) or 0.0),
        "phaseA_mode_OVERLAY_ratio": float(record.get("phaseA_mode_OVERLAY_ratio", 0.0) or 0.0),
        "phaseA_mode_PUNCTURE_ratio": float(record.get("phaseA_mode_PUNCTURE_ratio", 0.0) or 0.0),
        "phaseA_packet_0_ratio": float(record.get("phaseA_packet_0_ratio", 0.0) or 0.0),
        "phaseA_valid_packet_ratio": float(record.get("phaseA_valid_packet_ratio", 0.0) or 0.0),
        "phaseA_keep_given_candidate": float(record.get("phaseA_keep_given_candidate", 0.0) or 0.0),
        "phaseA_pkt0_given_candidate": float(record.get("phaseA_pkt0_given_candidate", 0.0) or 0.0),
        "phaseA_nonkeep_given_candidate": float(record.get("phaseA_nonkeep_given_candidate", 0.0) or 0.0),
        "phaseA_feasible_candidate_count_mean": float(record.get("phaseA_feasible_candidate_count_mean", 0.0) or 0.0),
        "rollout_total_feasible_candidates": float(record.get("rollout_total_feasible_candidates", 0.0) or 0.0),
        "rollout_admitted_given_candidate": float(record.get("rollout_admitted_given_candidate", 0.0) or 0.0),
        "rollout_total_arrivals": float(record.get("rollout_total_arrivals", record.get("rollout_active_packets", 0.0)) or 0.0),
        "rollout_candidate_packets": float(record.get("rollout_candidate_packets", 0.0) or 0.0),
        "rollout_feasible_packets": float(record.get("rollout_feasible_packets", 0.0) or 0.0),
        "rollout_candidate_ratio": float(record.get("rollout_candidate_ratio", 0.0) or 0.0),
        "rollout_feasible_given_candidate": float(record.get("rollout_feasible_given_candidate", 0.0) or 0.0),
        "rollout_admitted_given_candidate_packet": float(record.get("rollout_admitted_given_candidate_packet", 0.0) or 0.0),
        "rollout_admitted_given_feasible": float(record.get("rollout_admitted_given_feasible", 0.0) or 0.0),
        "rollout_blocked_no_candidate": float(record.get("rollout_blocked_no_candidate", 0.0) or 0.0),
        "rollout_blocked_infeasible": float(record.get("rollout_blocked_infeasible", 0.0) or 0.0),
        "rollout_blocked_resource": float(record.get("rollout_blocked_resource", 0.0) or 0.0),
        "eval_phase0_owner_change_ratio_vs_snapshot_raw": float(
            record.get("eval_phase0_owner_change_ratio_vs_snapshot_raw", 0.0) or 0.0
        ),
        "eval_phase0_owner_change_ratio_vs_snapshot_executed": float(
            record.get("eval_phase0_owner_change_ratio_vs_snapshot_executed", 0.0) or 0.0
        ),
        "eval_phase0_owner_fallback_to_candidate0_ratio": float(
            record.get("eval_phase0_owner_fallback_to_candidate0_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_invalid_option_ratio": float(
            record.get("eval_phase0_owner_invalid_option_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_null_selected_ratio": float(
            record.get("eval_phase0_owner_null_selected_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_invalid_to_snapshot_ratio": float(
            record.get("eval_phase0_owner_invalid_to_snapshot_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_invalid_to_non_snapshot_ratio": float(
            record.get("eval_phase0_owner_invalid_to_non_snapshot_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_restored_to_snapshot_ratio": float(
            record.get("eval_phase0_owner_restored_to_snapshot_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_replaced_with_non_snapshot_ratio": float(
            record.get("eval_phase0_owner_replaced_with_non_snapshot_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_non_null_ratio_raw": float(
            record.get("eval_phase0_owner_non_null_ratio_raw", 0.0) or 0.0
        ),
        "eval_phase0_owner_non_null_ratio_executed": float(
            record.get("eval_phase0_owner_non_null_ratio_executed", 0.0) or 0.0
        ),
        "eval_phase0_owner_changed_and_effective_ratio": float(
            record.get("eval_phase0_owner_changed_and_effective_ratio", 0.0) or 0.0
        ),
        "eval_phase0_owner_effective_rate_gain_vs_snapshot_mean": float(
            record.get("eval_phase0_owner_effective_rate_gain_vs_snapshot_mean", 0.0) or 0.0
        ),
        "eval_phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps": float(
            record.get("eval_phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps", 0.0) or 0.0
        ),
        "eval_phase0_owner_change_harmful_ratio": float(
            record.get("eval_phase0_owner_change_harmful_ratio", 0.0) or 0.0
        ),
        "eval_owner_snapshot_fallback_taken": float(
            record.get("eval_owner_snapshot_fallback_taken", 0.0) or 0.0
        ),
        "stress_eval_embb_rate_mbps": float(record.get("stress_eval_embb_rate_mbps", 0.0) or 0.0),
        "stress_eval_embb_min_rate_satisfaction": float(record.get("stress_eval_embb_min_rate_satisfaction", 0.0) or 0.0),
        "stress_eval_embb_min_rate_shortfall": float(record.get("stress_eval_embb_min_rate_shortfall", 0.0) or 0.0),
        "stress_eval_admission": float(record.get("stress_eval_admission", 0.0) or 0.0),
        "stress_eval_admitted_packets": float(record.get("stress_eval_admitted_packets", 0.0) or 0.0),
        "stress_eval_total_power": float(record.get("stress_eval_total_power", 0.0) or 0.0),
        "stress_eval_overlay_ratio": float(record.get("stress_eval_overlay_ratio", 0.0) or 0.0),
        "stress_eval_puncture_ratio": float(record.get("stress_eval_puncture_ratio", 0.0) or 0.0),
    }
    if eval_score is not None:
        metadata["eval_score"] = float(eval_score)
    return metadata


def run_clean_mappo(
    *,
    experiment: Optional[str],
    iterations: int,
    rollout_horizon: int,
    rollout_horizon_env_steps: Optional[int],
    num_rollout_envs: Optional[int],
    parallel_rollout_workers: Optional[int],
    rollout_worker_device: Optional[str],
    disable_parallel_rollout: bool,
    ppo_epochs: int,
    minibatch_size: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
    target_load: Optional[float],
    device: Optional[str],
    random_loads: Optional[List[float]] = None,
    urllc_poisson_rate_range: Optional[List[float]] = None,
    urllc_poisson_rate_sampling: str = "uniform",
    urllc_poisson_rate: Optional[float] = None,
    progress_every: Optional[int] = None,
    resume_from: Optional[str] = None,
    embb_power_scale_max: Optional[float] = None,
) -> Dict[str, object]:
    os.environ["SR_MAPPO_LOG_EFFECTIVE_LAMBDA"] = "0"
    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment)
    cfg = deepcopy(cfg)
    cfg.network.use_recurrent = False
    # Chase the older short-run v17s behavior more closely: do not expand the
    # clean-trainer observation with the newer RB-summary block unless a future
    # experiment opts in explicitly.
    cfg.env.include_rb_summary_observation = False
    cfg.env.simplified_rb_summary_observation = False
    cfg.training.rollout_horizon = int(max(rollout_horizon, 1))
    if rollout_horizon_env_steps is not None:
        cfg.training.rollout_horizon_env_steps = int(max(rollout_horizon_env_steps, 0))
    if num_rollout_envs is not None:
        cfg.training.num_rollout_envs = int(max(num_rollout_envs, 1))
    if parallel_rollout_workers is not None:
        cfg.training.parallel_rollout_workers = int(max(parallel_rollout_workers, 1))
    if rollout_worker_device is not None:
        cfg.training.rollout_worker_device = str(rollout_worker_device)
    if disable_parallel_rollout:
        cfg.training.disable_parallel_rollout = True
    if progress_every is not None:
        cfg.training.progress_every = int(max(progress_every, 1))
    if embb_power_scale_max is not None:
        cfg.env.embb_power_scale_max = float(embb_power_scale_max)
    resolved_device = _resolve_device(device or cfg.training.device)
    cfg.training.device = str(resolved_device)

    rollout_envs = [_build_env(cfg) for _ in range(int(max(cfg.training.num_rollout_envs, 1)))]
    env = rollout_envs[0]
    actual_load = _configure_load(env, target_load)
    for extra_env in rollout_envs[1:]:
        _configure_load(extra_env, target_load)
    if urllc_poisson_rate is not None:
        for rollout_env in rollout_envs:
            rollout_env.sim_cfg.urllc_poisson_rate = float(max(urllc_poisson_rate, 0.0))
            rollout_env.sim_cfg.fixed_urllc_poisson_rate = True
    nominal_urllc_poisson_rate = float(getattr(env.sim_cfg, "urllc_poisson_rate", 0.0) or 0.0)
    if urllc_poisson_rate is None and urllc_poisson_rate_range:
        lo = float(min(urllc_poisson_rate_range))
        hi = float(max(urllc_poisson_rate_range))
        sampling_mode = str(urllc_poisson_rate_sampling or "uniform").strip().lower()
        if sampling_mode == "high_bias":
            span = max(hi - lo, 0.0)
            low_mean = lo + 0.11 * span
            mid_mean = lo + 0.39 * span
            high_mean = lo + 0.78 * span
            nominal_urllc_poisson_rate = float(0.2 * low_mean + 0.3 * mid_mean + 0.5 * high_mean)
        else:
            nominal_urllc_poisson_rate = float(0.5 * (lo + hi))
    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, cfg).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            getattr(
                cfg.training,
                "actor_learning_rate",
                getattr(cfg.training, "learning_rate", 2.0e-4),
            )
            or 2.0e-4
        ),
    )

    history: List[Dict[str, float]] = []
    cumulative_rollout_episode_count = 0
    run_seed = int(seed)
    checkpoint_dir = _checkpoint_dir(cfg)
    history_path = checkpoint_dir / f"{cfg.training.run_name}_clean_history.json"
    history_jsonl_path = checkpoint_dir / f"{cfg.training.run_name}_clean_history.jsonl"
    resume_path = Path(str(resume_from)).expanduser() if resume_from else None
    start_iteration = 1
    progress_every = int(max(getattr(cfg.training, "progress_every", 20) or 20, 1))
    history_flush_every = int(
        max(
            1,
            getattr(
                cfg.training,
                "clean_history_flush_every",
                int(max(eval_every, 1)),
            )
            or int(max(eval_every, 1)),
        )
    )
    agent_count = int(len(env.agent_ids))
    episode_env_steps_expected = int(_episode_env_steps_expected(env))
    effective_rollout_horizon = _rollout_horizon_agent_transitions(
        rollout_horizon=int(max(rollout_horizon, 1)),
        rollout_horizon_env_steps=int(max(getattr(cfg.training, "rollout_horizon_env_steps", 0), 0)),
        num_rollout_envs=int(max(cfg.training.num_rollout_envs, 1)),
        agent_count=agent_count,
    )
    cumulative_rollout_agent_transitions = 0.0
    best_eval_score = float("-inf")
    best_eval_iter = -1
    best_stress_eval_score = float("-inf")
    best_stress_eval_iter = -1
    milestone_eval_iters = {
        int(value)
        for value in list(getattr(cfg.training, "clean_eval_checkpoint_iterations", []) or [])
        if int(value) > 0
    }
    fixed_eval_ckpt_interval = int(getattr(cfg.training, "clean_eval_checkpoint_interval", 0) or 0)
    fixed_eval_ckpt_start = int(getattr(cfg.training, "clean_eval_checkpoint_start_iteration", 0) or 0)
    if fixed_eval_ckpt_interval > 0:
        start_iter = max(fixed_eval_ckpt_start, fixed_eval_ckpt_interval)
        milestone_eval_iters.update(
            range(
                start_iter,
                int(max(iterations, 1)) + 1,
                fixed_eval_ckpt_interval,
            )
        )
    interrupted = False
    last_completed_iteration = 0
    if resume_path is not None:
        payload = torch.load(str(resume_path), map_location=resolved_device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported clean resume checkpoint payload in {resume_path}")
        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Missing model_state_dict in resume checkpoint: {resume_path}")
        model.load_state_dict(state_dict)
        optimizer_state = payload.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        if history_path.exists():
            history = [dict(item) for item in _load_clean_history(history_path)]
        else:
            history = [dict(item) for item in list(payload.get("history", []) or payload.get("history_tail", []) or [])]
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        history_iteration_values = [
            int(float(item.get("iteration", 0) or 0))
            for item in history
            if isinstance(item, dict) and item.get("iteration", None) is not None
        ]
        history_max_iteration = max(history_iteration_values) if history_iteration_values else 0
        payload_last_iteration = int(payload.get("last_iteration", 0) or 0)
        metadata_iteration = int(metadata.get("iteration", 0) or 0)
        last_completed_iteration = max(
            metadata_iteration,
            payload_last_iteration,
            history_max_iteration,
        )
        start_iteration = max(last_completed_iteration + 1, 1)
        cumulative_rollout_episode_count = int(
            payload.get("cumulative_rollout_episode_count", 0)
            or (history[-1].get("cumulative_rollout_episode_count", 0) if history else 0)
            or 0
        )
        cumulative_rollout_agent_transitions = float(
            payload.get("cumulative_rollout_agent_transitions", 0.0)
            or (history[-1].get("train_effective_episodes", 0.0) if history else 0.0)
            * float(max(agent_count * max(episode_env_steps_expected, 1), 1))
        )
        run_seed = int(payload.get("run_seed", run_seed) or run_seed)
        best_eval_score = float(payload.get("best_eval_score", best_eval_score) or best_eval_score)
        best_eval_iter = int(payload.get("best_eval_iter", best_eval_iter) or best_eval_iter)
        best_stress_eval_score = float(payload.get("best_stress_eval_score", best_stress_eval_score) or best_stress_eval_score)
        best_stress_eval_iter = int(payload.get("best_stress_eval_iter", best_stress_eval_iter) or best_stress_eval_iter)
        if not history_jsonl_path.exists():
            history_jsonl_path.write_text("", encoding="utf-8")
    else:
        history_jsonl_path.write_text("", encoding="utf-8")
    try:
        for iteration in range(int(start_iteration), int(max(iterations, 1)) + 1):
            iter_start = perf_counter()
            eval_wall_time = 0.0
            rollout = _rollout(
                rollout_envs,
                model,
                horizon=int(max(effective_rollout_horizon, 1)),
                seed=run_seed,
                device=resolved_device,
                base_target_load=target_load,
                random_load_choices=random_loads,
                urllc_poisson_rate_range=urllc_poisson_rate_range,
                urllc_poisson_rate_sampling=str(urllc_poisson_rate_sampling or "uniform"),
                rollout_horizon_env_steps=int(max(getattr(cfg.training, "rollout_horizon_env_steps", 0), 0)),
                parallel_rollout_workers=int(max(getattr(cfg.training, "parallel_rollout_workers", 1), 1)),
                rollout_worker_device=str(getattr(cfg.training, "rollout_worker_device", "cpu") or "cpu"),
                disable_parallel_rollout=bool(getattr(cfg.training, "disable_parallel_rollout", False)),
            )
            run_seed = int(rollout["next_seed"])
            rollout_wall_time = float(rollout.get("rollout_parallel_wall_time", 0.0) or 0.0)
            update_start = perf_counter()
            update = _update_policy(
                model,
                optimizer,
                rollout["batch"],
                ppo_epochs=int(max(ppo_epochs, 1)),
                minibatch_size=int(max(minibatch_size, 1)),
                clip_ratio=float(getattr(cfg.training, "clip_ratio", 0.2) or 0.2),
                value_coef=float(getattr(cfg.training, "value_coef", 0.5) or 0.5),
                entropy_coef=float(getattr(cfg.training, "entropy_coef", 0.01) or 0.01),
                max_grad_norm=float(getattr(cfg.training, "max_grad_norm", 0.5) or 0.5),
            )
            update_wall_time = float(perf_counter() - update_start)
            rollout_episode_count = int(float(rollout.get("rollout_episode_count", 0.0) or 0.0))
            cumulative_rollout_episode_count += rollout_episode_count
            rollout_total_agent_transitions = float(rollout.get("rollout_total_agent_transitions", effective_rollout_horizon) or 0.0)
            cumulative_rollout_agent_transitions += rollout_total_agent_transitions
            effective_episodes_per_iter = float(rollout_total_agent_transitions) / float(
                max(agent_count * max(episode_env_steps_expected, 1), 1)
            )

            record: Dict[str, float] = {
                "iteration": float(iteration),
                "target_load": float(target_load if target_load is not None else actual_load),
                "actual_load": float(actual_load),
                "rollout_sampled_load_mean": float(rollout["sampled_load_mean"]),
                "rollout_sampled_urllc_poisson_rate_mean": float(rollout["sampled_urllc_poisson_rate_mean"]),
                "rollout_episode_count": float(rollout_episode_count),
                "cumulative_rollout_episode_count": float(cumulative_rollout_episode_count),
                "rollout_total_agent_transitions": float(rollout_total_agent_transitions),
                "num_rollout_envs": float(int(max(cfg.training.num_rollout_envs, 1))),
                "rollout_horizon_agent_transitions": float(effective_rollout_horizon),
                "rollout_horizon_env_steps": float(int(max(getattr(cfg.training, "rollout_horizon_env_steps", 0), 0))),
                "rollout_env_steps_per_env": float(rollout.get("rollout_env_steps_per_env", 0.0) or 0.0),
                "effective_episodes_per_iter": float(effective_episodes_per_iter),
                "train_effective_episodes": float(
                    cumulative_rollout_agent_transitions / float(max(agent_count * max(episode_env_steps_expected, 1), 1))
                ),
                "rollout_reward": float(rollout["rollout_reward_mean"]),
                "rollout_embb_rate_mbps": float(rollout["rollout_embb_rate_mbps"]),
                "rollout_admission": float(rollout["rollout_admission"]),
                "rollout_admitted_packets": float(rollout["rollout_admitted_packets"]),
                "rollout_active_packets": float(rollout["rollout_active_packets"]),
                "rollout_phase0_blocked_users": float(
                    rollout.get("rollout_phase0_blocked_users", rollout.get("rollout_embb_blocked_users", 0.0)) or 0.0
                ),
                "rollout_phase0_partial_minrate_users": float(
                    rollout.get("rollout_phase0_partial_minrate_users", 0.0) or 0.0
                ),
                "rollout_phase0_refill_rb_count": float(rollout.get("rollout_phase0_refill_rb_count", 0.0) or 0.0),
                "rollout_phase0_refill_gain_mbps": float(
                    rollout.get("rollout_phase0_refill_gain_mbps", 0.0) or 0.0
                ),
                "rollout_phase0_refill_intercell_delta_over_noise": float(
                    rollout.get("rollout_phase0_refill_intercell_delta_over_noise", 0.0) or 0.0
                ),
                "rollout_final_blocked_users": float(rollout.get("rollout_final_blocked_users", 0.0) or 0.0),
                "rollout_phaseA_newly_blocked_users": float(rollout.get("rollout_phaseA_newly_blocked_users", 0.0) or 0.0),
                "rollout_embb_rate_ratio": float(rollout.get("rollout_embb_rate_ratio", 0.0) or 0.0),
                "rollout_embb_jain_after": float(rollout.get("rollout_embb_jain_after", 0.0) or 0.0),
                "rollout_embb_5th_percentile_after": float(rollout.get("rollout_embb_5th_percentile_after", 0.0) or 0.0),
                "rollout_total_embb_related_reward": float(rollout.get("rollout_total_embb_related_reward", 0.0) or 0.0),
                "rollout_total_urllc_related_reward": float(rollout.get("rollout_total_urllc_related_reward", 0.0) or 0.0),
                "rollout_total_power_related_reward": float(rollout.get("rollout_total_power_related_reward", 0.0) or 0.0),
                "rollout_episode_sum_embb_related_reward": float(
                    rollout.get("rollout_episode_sum_embb_related_reward", 0.0) or 0.0
                ),
                "rollout_episode_sum_urllc_related_reward": float(
                    rollout.get("rollout_episode_sum_urllc_related_reward", 0.0) or 0.0
                ),
                "rollout_episode_sum_power_related_reward": float(
                    rollout.get("rollout_episode_sum_power_related_reward", 0.0) or 0.0
                ),
                "rollout_per_step_mean_embb_related_reward": float(
                    rollout.get("rollout_per_step_mean_embb_related_reward", 0.0) or 0.0
                ),
                "rollout_per_step_mean_urllc_related_reward": float(
                    rollout.get("rollout_per_step_mean_urllc_related_reward", 0.0) or 0.0
                ),
                "rollout_per_step_mean_power_related_reward": float(
                    rollout.get("rollout_per_step_mean_power_related_reward", 0.0) or 0.0
                ),
                "rollout_terminal_embb_rate_guardrail_penalty": float(
                    rollout.get("rollout_terminal_embb_rate_guardrail_penalty", 0.0) or 0.0
                ),
                "rollout_step_embb_rate_deficit_penalty": float(
                    rollout.get("rollout_step_embb_rate_deficit_penalty", 0.0) or 0.0
                ),
                "rollout_step_embb_rate_deficit_penalty_mean": float(
                    rollout.get("rollout_step_embb_rate_deficit_penalty_mean", 0.0) or 0.0
                ),
                "rollout_step_embb_rate_ratio": float(rollout.get("rollout_step_embb_rate_ratio", 0.0) or 0.0),
                "rollout_step_embb_rate_deficit_amount": float(
                    rollout.get("rollout_step_embb_rate_deficit_amount", 0.0) or 0.0
                ),
                "rollout_step_embb_rate_deficit_active_count": float(
                    rollout.get("rollout_step_embb_rate_deficit_active_count", 0.0) or 0.0
                ),
                "rollout_step_embb_rate_deficit_active_ratio": float(
                    rollout.get("rollout_step_embb_rate_deficit_active_ratio", 0.0) or 0.0
                ),
                "rollout_embb_guardrail_violation_amount": float(
                    rollout.get("rollout_embb_guardrail_violation_amount", 0.0) or 0.0
                ),
                "rollout_embb_guardrail_active_count": float(
                    rollout.get("rollout_embb_guardrail_active_count", 0.0) or 0.0
                ),
                "rollout_power_excess_ratio": float(rollout.get("rollout_power_excess_ratio", 0.0) or 0.0),
                "rollout_power_excess_mean": float(rollout.get("rollout_power_excess_mean", 0.0) or 0.0),
                "rollout_total_arrivals": float(rollout.get("rollout_total_arrivals", rollout.get("rollout_active_packets", 0.0)) or 0.0),
                "rollout_candidate_packets": float(rollout.get("rollout_candidate_packets", 0.0) or 0.0),
                "rollout_feasible_packets": float(rollout.get("rollout_feasible_packets", 0.0) or 0.0),
                "rollout_candidate_ratio": float(rollout.get("rollout_candidate_ratio", 0.0) or 0.0),
                "rollout_feasible_given_candidate": float(rollout.get("rollout_feasible_given_candidate", 0.0) or 0.0),
                "rollout_admitted_given_candidate_packet": float(rollout.get("rollout_admitted_given_candidate_packet", 0.0) or 0.0),
                "rollout_admitted_given_feasible": float(rollout.get("rollout_admitted_given_feasible", 0.0) or 0.0),
                "rollout_blocked_no_candidate": float(rollout.get("rollout_blocked_no_candidate", 0.0) or 0.0),
                "rollout_blocked_infeasible": float(rollout.get("rollout_blocked_infeasible", 0.0) or 0.0),
                "rollout_blocked_resource": float(rollout.get("rollout_blocked_resource", 0.0) or 0.0),
                "rollout_mean_intercell_interference_mw": float(
                    rollout.get("rollout_mean_intercell_interference_mw", 0.0) or 0.0
                ),
                "rollout_mean_intercell_interference_over_noise": float(
                    rollout.get("rollout_mean_intercell_interference_over_noise", 0.0) or 0.0
                ),
                "rollout_selected_action_intercell_cost_after_source_mask_mean": float(
                    rollout.get("rollout_selected_action_intercell_cost_after_source_mask_mean", 0.0) or 0.0
                ),
                "rollout_selected_action_intercell_cost_after_source_mask_over_noise_mean": float(
                    rollout.get("rollout_selected_action_intercell_cost_after_source_mask_over_noise_mean", 0.0) or 0.0
                ),
                "rollout_intercell_per_admitted_packet": float(
                    rollout.get("rollout_intercell_per_admitted_packet", 0.0) or 0.0
                ),
                "rollout_embb_rate_loss_due_to_intercell_ratio": float(
                    rollout.get("rollout_embb_rate_loss_due_to_intercell_ratio", 0.0) or 0.0
                ),
                "rollout_admission_bonus_pre_target": float(
                    rollout.get("rollout_admission_bonus_pre_target", 0.0) or 0.0
                ),
                "rollout_admission_bonus_tail": float(rollout.get("rollout_admission_bonus_tail", 0.0) or 0.0),
                "policy_loss": float(update["policy_loss"]),
                "value_loss": float(update["value_loss"]),
                "entropy": float(update["entropy"]),
                "approx_kl": float(update["approx_kl"]),
                "clip_fraction": float(update["clip_fraction"]),
                "explained_variance": float(update["explained_variance"]),
                "rollout_wall_time": float(rollout_wall_time),
                "update_wall_time": float(update_wall_time),
                "rollout_parallel_wall_time": float(rollout_wall_time),
                "rollout_worker_mean_time": float(rollout.get("rollout_worker_mean_time", rollout_wall_time) or 0.0),
                "rollout_worker_max_time": float(rollout.get("rollout_worker_max_time", rollout_wall_time) or 0.0),
            }
            reward_components = rollout.get("rollout_reward_components_mean", {})
            if isinstance(reward_components, dict) and reward_components:
                record["reward_components"] = {
                    str(key): float(value or 0.0)
                    for key, value in reward_components.items()
                }
            phase_metric_keys = [
                "phase0_mode_logprob_mean",
                "phase0_packet_logprob_mean",
                "phase0_owner_logprob_mean",
                "phase0_embb_power_logprob_mean",
                "phase0_mode_entropy_mean",
                "phase0_packet_entropy_mean",
                "phase0_owner_entropy_mean",
                "phase0_embb_power_entropy_mean",
                "phaseA_mode_logprob_mean",
                "phaseA_packet_logprob_mean",
                "phaseA_owner_logprob_mean",
                "phaseA_embb_power_logprob_mean",
                "phaseA_mode_entropy_mean",
                "phaseA_packet_entropy_mean",
                "phaseA_owner_entropy_mean",
                "phaseA_embb_power_entropy_mean",
            ]
            for key in phase_metric_keys:
                if key in rollout:
                    record[key] = float(rollout.get(key, 0.0) or 0.0)
            rollout_histogram_keys = [
                "phaseA_count",
                "phaseA_has_candidate_ratio",
                "phaseA_mode_KEEP_ratio",
                "phaseA_mode_OVERLAY_ratio",
                "phaseA_mode_PUNCTURE_ratio",
                "phaseA_packet_0_ratio",
                "phaseA_valid_packet_ratio",
                "phaseA_keep_given_candidate",
                "phaseA_pkt0_given_candidate",
                "phaseA_nonkeep_given_candidate",
                "phaseA_feasible_candidate_count_mean",
                "rollout_total_feasible_candidates",
                "rollout_admitted_given_candidate",
            ]
            for key in rollout_histogram_keys:
                if key in rollout:
                    record[key] = float(rollout.get(key, 0.0) or 0.0)
            if iteration % 10 == 0:
                recent = history[-9:] + [record]
                reward_window = [
                    float(item.get("rollout_reward", 0.0) or 0.0)
                    for item in recent
                    if "rollout_reward" in item
                ]
                if reward_window:
                    record["rollout_reward_avg10"] = float(np.mean(np.asarray(reward_window, dtype=float)))
                recent20 = history[-19:] + [record]
                reward_window20 = [
                    float(item.get("rollout_reward", 0.0) or 0.0)
                    for item in recent20
                    if "rollout_reward" in item
                ]
                if reward_window20:
                    record["rollout_reward_avg20"] = float(np.mean(np.asarray(reward_window20, dtype=float)))

            if iteration % int(max(eval_every, 1)) == 0:
                eval_start = perf_counter()
                deterministic_eval = _evaluate_policy(
                    env,
                    model,
                    episodes=int(max(eval_episodes, 1)),
                    seed=100_000 + run_seed,
                    target_load=target_load,
                    nominal_urllc_poisson_rate=nominal_urllc_poisson_rate,
                    deterministic_action=True,
                    metric_prefix="eval_deterministic",
                )
                stochastic_eval = _evaluate_policy(
                    env,
                    model,
                    episodes=int(max(eval_episodes, 1)),
                    seed=200_000 + run_seed,
                    target_load=target_load,
                    nominal_urllc_poisson_rate=nominal_urllc_poisson_rate,
                    deterministic_action=False,
                    metric_prefix="eval_stochastic",
                )
                record.update(deterministic_eval)
                record.update(stochastic_eval)
                eval_alias_suffixes = [
                    "embb_rate_mbps",
                    "avg_embb_rate_mbps",
                    "embb_min_rate_satisfaction",
                    "embb_min_rate_shortfall",
                    "admission",
                    "admitted_packets",
                    "active_packets",
                    "embb_blocked_users",
                    "phase0_blocked_users",
                    "phase0_partial_minrate_users",
                    "phase0_refill_rb_count",
                    "phase0_refill_gain_mbps",
                    "phase0_refill_intercell_delta_over_noise",
                    "final_blocked_users",
                    "phaseA_newly_blocked_users",
                    "urllc_blocked_users",
                    "total_power",
                    "overlay_ratio",
                    "puncture_ratio",
                    "step_reward_sum",
                    "terminal_reward_sum",
                    "total_reward",
                    "planning_embb_rate_delta_reward",
                    "urgency_bonus",
                    "embb_damage",
                    "power_penalty",
                    "step_embb_rate_deficit_penalty",
                    "terminal_urllc_admission",
                    "terminal_embb_minrate_bonus",
                    "terminal_embb_rate_guardrail_penalty",
                    "terminal_power_budget_penalty",
                    "overlay_count",
                    "puncturing_count",
                    "power_violation",
                    "aggregate_embb_rate",
                    "aggregate_embb_reference_rate",
                    "embb_rate_ratio",
                    "embb_jain_after",
                    "embb_5th_percentile_after",
                    "embb_rate_target_ratio",
                    "embb_guardrail_violation_amount",
                    "embb_guardrail_active_count",
                    "phase0_owner_change_ratio_vs_snapshot_raw",
                    "phase0_owner_change_ratio_vs_snapshot_executed",
                    "phase0_owner_fallback_to_candidate0_ratio",
                    "phase0_owner_invalid_option_ratio",
                    "phase0_owner_null_selected_ratio",
                    "phase0_owner_invalid_to_snapshot_ratio",
                    "phase0_owner_invalid_to_non_snapshot_ratio",
                    "phase0_owner_restored_to_snapshot_ratio",
                    "phase0_owner_replaced_with_non_snapshot_ratio",
                    "phase0_owner_non_null_ratio_raw",
                    "phase0_owner_non_null_ratio_executed",
                    "phase0_owner_changed_and_effective_ratio",
                    "phase0_owner_effective_rate_gain_vs_snapshot_mean",
                    "phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps",
                    "phase0_owner_change_harmful_ratio",
                    "owner_snapshot_fallback_taken",
                    "step_embb_rate_deficit_penalty_mean",
                    "step_embb_deficit_target_ratio",
                    "step_embb_rate_ratio",
                    "step_embb_rate_deficit_amount",
                    "step_embb_rate_deficit_active_count",
                    "step_embb_rate_deficit_active_ratio",
                    "total_embb_related_reward",
                    "total_urllc_related_reward",
                    "total_power_related_reward",
                    "power_excess_ratio",
                    "power_excess_mean",
                    "admission_ratio",
                    "admission_bonus_pre_target",
                    "admission_bonus_tail",
                    "total_arrivals",
                    "candidate_packets",
                    "feasible_packets",
                    "admitted_packets_breakdown",
                    "candidate_ratio",
                    "feasible_given_candidate",
                    "admitted_given_candidate",
                    "admitted_given_feasible",
                    "blocked_no_candidate",
                    "blocked_infeasible",
                    "blocked_resource",
                ]
                for suffix in eval_alias_suffixes:
                    deterministic_key = f"eval_deterministic_{suffix}"
                    if deterministic_key in record:
                        record[f"eval_{suffix}"] = float(record[deterministic_key])
                record["eval_admission_gap"] = float(
                    record.get("eval_deterministic_admission", 0.0) - record.get("eval_stochastic_admission", 0.0)
                )
                if bool(getattr(cfg.training, "clean_stress_eval_enabled", False)):
                    record.update(
                        _evaluate_stress_suite(
                            env,
                            model,
                            episodes_per_lambda=int(getattr(cfg.training, "clean_stress_eval_episodes_per_lambda", 2) or 2),
                            seed=200_000 + run_seed,
                            target_load=float(getattr(cfg.training, "clean_stress_eval_target_load", 24.0) or 24.0),
                            lambdas=list(getattr(cfg.training, "clean_stress_eval_lambdas", []) or []),
                        )
                    )
                eval_wall_time = float(perf_counter() - eval_start)
            record["eval_wall_time"] = float(eval_wall_time)
            record["iteration_sec"] = float(perf_counter() - iter_start)

            history.append(record)
            _append_clean_history_jsonl(history_jsonl_path, record)
            last_completed_iteration = int(iteration)
            line = (
                f"[CLEAN-MAPPO] iter {iteration}/{iterations} | load={actual_load:.1f} "
                f"| rollout_load={record['rollout_sampled_load_mean']:.1f} "
                f"| rollout_lambda={record['rollout_sampled_urllc_poisson_rate_mean']:.3f} "
                f"| envs={int(record['num_rollout_envs'])} "
                f"| rollout_env_steps={record['rollout_env_steps_per_env']:.2f} "
                f"| eff_eps={record['effective_episodes_per_iter']:.3f} "
                f"| rollout_t={record['rollout_wall_time']:.2f}s "
                f"| update_t={record['update_wall_time']:.2f}s "
                f"| rollout_reward={record['rollout_reward']:.4f} "
                f"| rollout_embb={record['rollout_embb_rate_mbps']:.3f} Mbps "
                f"| rollout_adm={record['rollout_admission']:.4f} "
                f"| admitted={record['rollout_admitted_packets']:.2f}/{record['rollout_active_packets']:.2f} "
                f"| cand_pkt={record.get('rollout_candidate_packets', 0.0):.2f}/{record.get('rollout_total_arrivals', record['rollout_active_packets']):.2f} "
                f"| feas_pkt={record.get('rollout_feasible_packets', 0.0):.2f} "
                f"| cand={record.get('phaseA_has_candidate_ratio', 0.0):.3f} "
                f"| phaseA_keep={record.get('phaseA_mode_KEEP_ratio', 0.0):.3f} "
                f"| keep|cand={record.get('phaseA_keep_given_candidate', 0.0):.3f} "
                f"| phaseA_pkt0={record.get('phaseA_packet_0_ratio', 0.0):.3f} "
                f"| pkt0|cand={record.get('phaseA_pkt0_given_candidate', 0.0):.3f} "
                f"| ph0_blk={record.get('rollout_phase0_blocked_users', record.get('rollout_embb_blocked_users', 0.0)):.2f} "
                f"| ph0_partial={record.get('rollout_phase0_partial_minrate_users', 0.0):.2f} "
                f"| ph0_refill={record.get('rollout_phase0_refill_gain_mbps', 0.0):.2f} "
                f"| fin_blk={record.get('rollout_final_blocked_users', 0.0):.2f} "
                f"| pi={record['policy_loss']:.4f} vf={record['value_loss']:.4f} ent={record['entropy']:.4f} "
                f"| kl={record['approx_kl']:.5f} clip={record['clip_fraction']:.3f} ev={record['explained_variance']:.3f}"
            )
            if "eval_embb_rate_mbps" in record:
                line += (
                    f" | eval_embb={record['eval_embb_rate_mbps']:.3f} Mbps"
                    f" | eval_minrate={record['eval_embb_min_rate_satisfaction']:.4f}"
                    f" | eval_adm={record['eval_admission']:.4f}"
                    f" | eval_pkt={record['eval_admitted_packets']:.2f}/{record['eval_active_packets']:.2f}"
                    f" | eval_cand_pkt={record.get('eval_candidate_packets', 0.0):.2f}"
                    f" | eval_feas_pkt={record.get('eval_feasible_packets', 0.0):.2f}"
                    f" | eval_ph0_blk={record.get('eval_phase0_blocked_users', record.get('eval_embb_blocked_users', 0.0)):.2f}"
                    f" | eval_ph0_partial={record.get('eval_phase0_partial_minrate_users', 0.0):.2f}"
                    f" | eval_ph0_refill={record.get('eval_phase0_refill_gain_mbps', 0.0):.2f}"
                    f" | eval_fin_blk={record.get('eval_final_blocked_users', 0.0):.2f}"
                    f" | eval_phaseA_blk={record.get('eval_phaseA_newly_blocked_users', 0.0):.2f}"
                    f" | eval_jain={record.get('eval_embb_jain_after', 0.0):.3f}"
                    f" | eval_p5={record.get('eval_embb_5th_percentile_after', 0.0):.3f}"
                    f" | eval_owner(raw/exe)={record.get('eval_phase0_owner_change_ratio_vs_snapshot_raw', 0.0):.3f}/"
                    f"{record.get('eval_phase0_owner_change_ratio_vs_snapshot_executed', 0.0):.3f}"
                    f" | eval_owner_restore={record.get('eval_phase0_owner_restored_to_snapshot_ratio', 0.0):.3f}"
                    f" | eval_owner_rate_gain={record.get('eval_phase0_owner_effective_rate_gain_vs_snapshot_mean', 0.0):.3f}"
                    f" | eval_owner_rate_gain_per_change={record.get('eval_phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps', 0.0):.3f} Mbps"
                    f" | eval_det_adm={record.get('eval_deterministic_admission', 0.0):.4f}"
                    f" | eval_sto_adm={record.get('eval_stochastic_admission', 0.0):.4f}"
                    f" | eval_r={record['eval_total_reward']:.4f}"
                    f" | eval_step={record['eval_step_reward_sum']:.4f}"
                    f" | eval_term={record['eval_terminal_reward_sum']:.4f}"
                )
                if "stress_eval_embb_rate_mbps" in record:
                    line += (
                        f" | stress_embb={record['stress_eval_embb_rate_mbps']:.3f} Mbps"
                        f" | stress_minrate={record['stress_eval_embb_min_rate_satisfaction']:.4f}"
                        f" | stress_adm={record['stress_eval_admission']:.4f}"
                        f" | stress_pkt={record['stress_eval_admitted_packets']:.2f}"
                    )
                actual_target_load = float(target_load if target_load is not None else actual_load)
                latest_eval_path = checkpoint_dir / f"{cfg.training.run_name}_clean_latest_eval.pt"
                eval_metadata = _clean_eval_metadata(record, iteration)
                _save_clean_checkpoint(
                    latest_eval_path,
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    history=history,
                    target_load=actual_target_load,
                    metadata=eval_metadata,
                    extra_state={
                        "last_iteration": int(iteration),
                        "run_seed": int(run_seed),
                        "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                        "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                        "best_eval_iter": int(best_eval_iter),
                        "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                        "best_stress_eval_iter": int(best_stress_eval_iter),
                        "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
                    },
                )
                minrate_sat = float(record["eval_embb_min_rate_satisfaction"])
                minrate_shortfall = float(record["eval_embb_min_rate_shortfall"])
                eval_total_power = float(record.get("eval_total_power", 0.0) or 0.0)
                primary_checkpoint_preference = str(
                    getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") or "best_throughput"
                )
                if primary_checkpoint_preference == "best_block_power_balanced":
                    eval_embb_blocked_users = float(record.get("eval_embb_blocked_users", 0.0) or 0.0)
                    eval_urllc_blocked_users = float(record.get("eval_urllc_blocked_users", 0.0) or 0.0)
                    throughput_mbps = float(record["eval_embb_rate_mbps"])
                    throughput_floor_ratio = float(
                        getattr(cfg.training, "clean_eval_throughput_floor_ratio", 0.0) or 0.0
                    )
                    throughput_floor_penalty_weight = float(
                        getattr(cfg.training, "clean_eval_throughput_floor_penalty_weight", 0.0) or 0.0
                    )
                    throughput_ratio = float(record.get("eval_embb_rate_ratio", 0.0) or 0.0)
                    throughput_floor_deficit = (
                        max(throughput_floor_ratio - throughput_ratio, 0.0)
                        if throughput_floor_ratio > 0.0
                        else 0.0
                    )
                    eval_score = (
                        -float(getattr(cfg.training, "clean_eval_embb_blocked_weight", 0.0) or 0.0)
                        * eval_embb_blocked_users
                        - float(getattr(cfg.training, "clean_eval_urllc_blocked_weight", 0.0) or 0.0)
                        * eval_urllc_blocked_users
                        - float(getattr(cfg.training, "clean_eval_power_weight", 0.0) or 0.0)
                        * eval_total_power
                        + float(getattr(cfg.training, "clean_eval_throughput_weight", 0.0) or 0.0)
                        * throughput_mbps
                        + float(getattr(cfg.training, "clean_eval_minrate_weight", 0.0) or 0.0)
                        * minrate_sat
                        - throughput_floor_penalty_weight * throughput_floor_deficit
                        - 40.0 * minrate_shortfall
                    )
                elif primary_checkpoint_preference == "best_phase0_clean_frontier":
                    throughput_mbps = float(record["eval_embb_rate_mbps"])
                    admission_ratio = float(
                        record.get("eval_admission_ratio", record.get("eval_admission", 0.0)) or 0.0
                    )
                    phase0_blocked_users = float(record.get("eval_phase0_blocked_users", 0.0) or 0.0)
                    final_blocked_users = float(record.get("eval_final_blocked_users", 0.0) or 0.0)
                    phasea_new_blocked_users = float(record.get("eval_phaseA_newly_blocked_users", 0.0) or 0.0)
                    intercell_over_noise = float(record.get("eval_mean_intercell_interference_over_noise", 0.0) or 0.0)
                    intercell_norm = float(
                        getattr(cfg.training, "clean_eval_intercell_over_noise_normalizer", 1.0e4) or 1.0e4
                    )
                    eval_score = (
                        float(getattr(cfg.training, "clean_eval_throughput_weight", 0.0) or 0.0) * throughput_mbps
                        + float(getattr(cfg.training, "clean_eval_admission_weight", 0.0) or 0.0) * admission_ratio
                        + float(getattr(cfg.training, "clean_eval_minrate_weight", 0.0) or 0.0) * minrate_sat
                        - float(getattr(cfg.training, "clean_eval_phase0_blocked_weight", 0.0) or 0.0)
                        * phase0_blocked_users
                        - float(getattr(cfg.training, "clean_eval_final_blocked_weight", 0.0) or 0.0)
                        * final_blocked_users
                        - float(getattr(cfg.training, "clean_eval_phasea_new_blocked_weight", 0.0) or 0.0)
                        * phasea_new_blocked_users
                        - float(getattr(cfg.training, "clean_eval_intercell_over_noise_weight", 0.0) or 0.0)
                        * (intercell_over_noise / max(intercell_norm, 1.0e-9))
                        - 40.0 * minrate_shortfall
                    )
                elif primary_checkpoint_preference == "best_phase0_result_first_frontier":
                    throughput_mbps = float(record["eval_embb_rate_mbps"])
                    admission_ratio = float(
                        record.get("eval_admission_ratio", record.get("eval_admission", 0.0)) or 0.0
                    )
                    phase0_blocked_users = float(record.get("eval_phase0_blocked_users", 0.0) or 0.0)
                    phase0_partial_minrate_users = float(record.get("eval_phase0_partial_minrate_users", 0.0) or 0.0)
                    phase0_refill_gain_mbps = float(record.get("eval_phase0_refill_gain_mbps", 0.0) or 0.0)
                    phase0_refill_intercell_delta = float(
                        record.get("eval_phase0_refill_intercell_delta_over_noise", 0.0) or 0.0
                    )
                    final_blocked_users = float(record.get("eval_final_blocked_users", 0.0) or 0.0)
                    phasea_new_blocked_users = float(record.get("eval_phaseA_newly_blocked_users", 0.0) or 0.0)
                    intercell_over_noise = float(record.get("eval_mean_intercell_interference_over_noise", 0.0) or 0.0)
                    intercell_norm = float(
                        getattr(cfg.training, "clean_eval_intercell_over_noise_normalizer", 1.0e4) or 1.0e4
                    )
                    jain_after = float(record.get("eval_embb_jain_after", 0.0) or 0.0)
                    p5_mbps = float(record.get("eval_embb_5th_percentile_after", 0.0) or 0.0)
                    p5_norm = float(getattr(cfg.training, "clean_eval_5th_percentile_normalizer_mbps", 1.0) or 1.0)
                    throughput_ratio = float(record.get("eval_embb_rate_ratio", 0.0) or 0.0)
                    throughput_floor_ratio = float(
                        getattr(cfg.training, "clean_eval_throughput_floor_ratio", 0.0) or 0.0
                    )
                    throughput_floor_penalty_weight = float(
                        getattr(cfg.training, "clean_eval_throughput_floor_penalty_weight", 0.0) or 0.0
                    )
                    throughput_floor_deficit = (
                        max(throughput_floor_ratio - throughput_ratio, 0.0)
                        if throughput_floor_ratio > 0.0
                        else 0.0
                    )
                    eval_score = (
                        float(getattr(cfg.training, "clean_eval_throughput_weight", 0.0) or 0.0) * throughput_mbps
                        + float(getattr(cfg.training, "clean_eval_admission_weight", 0.0) or 0.0) * admission_ratio
                        + float(getattr(cfg.training, "clean_eval_minrate_weight", 0.0) or 0.0) * minrate_sat
                        + float(getattr(cfg.training, "clean_eval_phase0_refill_gain_weight", 0.0) or 0.0)
                        * phase0_refill_gain_mbps
                        + float(getattr(cfg.training, "clean_eval_jain_weight", 0.0) or 0.0) * jain_after
                        + float(getattr(cfg.training, "clean_eval_5th_percentile_weight", 0.0) or 0.0)
                        * (p5_mbps / max(p5_norm, 1.0e-9))
                        - float(getattr(cfg.training, "clean_eval_phase0_blocked_weight", 0.0) or 0.0)
                        * phase0_blocked_users
                        - float(getattr(cfg.training, "clean_eval_phase0_partial_minrate_weight", 0.0) or 0.0)
                        * phase0_partial_minrate_users
                        - float(getattr(cfg.training, "clean_eval_final_blocked_weight", 0.0) or 0.0)
                        * final_blocked_users
                        - float(getattr(cfg.training, "clean_eval_phasea_new_blocked_weight", 0.0) or 0.0)
                        * phasea_new_blocked_users
                        - float(getattr(cfg.training, "clean_eval_intercell_over_noise_weight", 0.0) or 0.0)
                        * (intercell_over_noise / max(intercell_norm, 1.0e-9))
                        - float(getattr(cfg.training, "clean_eval_phase0_refill_intercell_weight", 0.0) or 0.0)
                        * (phase0_refill_intercell_delta / max(intercell_norm, 1.0e-9))
                        - throughput_floor_penalty_weight * throughput_floor_deficit
                        - 40.0 * minrate_shortfall
                    )
                else:
                    minrate_gate = 0.92
                    minrate_deficit = max(minrate_gate - minrate_sat, 0.0)
                    eval_score = (
                        float(record["eval_embb_rate_mbps"])
                        + 28.0 * float(record["eval_admission"])
                        + 18.0 * minrate_sat
                        - 120.0 * minrate_deficit
                        - 40.0 * minrate_shortfall
                    )
                    power_weight = float(getattr(cfg.training, "balanced_checkpoint_power_penalty_weight", 0.0) or 0.0)
                    if power_weight > 1.0e-12:
                        eval_score -= 25.0 * power_weight * eval_total_power
                if iteration in milestone_eval_iters:
                    milestone_path = checkpoint_dir / f"{cfg.training.run_name}_clean_eval_iter_{int(iteration)}.pt"
                    milestone_metadata = _clean_eval_metadata(record, iteration, eval_score=eval_score)
                    milestone_metadata["checkpoint_kind"] = "eval_milestone"
                    _save_clean_checkpoint(
                        milestone_path,
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        history=history,
                        target_load=actual_target_load,
                        metadata=milestone_metadata,
                        extra_state={
                            "last_iteration": int(iteration),
                            "run_seed": int(run_seed),
                            "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                            "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                            "best_eval_iter": int(best_eval_iter),
                            "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                            "best_stress_eval_iter": int(best_stress_eval_iter),
                            "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
                        },
                    )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_iter = int(iteration)
                    best_eval_path = checkpoint_dir / f"{cfg.training.run_name}_clean_best_eval.pt"
                    _save_clean_checkpoint(
                        best_eval_path,
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        history=history,
                        target_load=actual_target_load,
                        metadata=_clean_eval_metadata(record, iteration, eval_score=eval_score),
                        extra_state={
                            "last_iteration": int(iteration),
                            "run_seed": int(run_seed),
                            "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                            "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                            "best_eval_iter": int(best_eval_iter),
                            "best_eval_score": float(eval_score),
                            "best_stress_eval_iter": int(best_stress_eval_iter),
                            "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
                        },
                    )
                if "stress_eval_embb_rate_mbps" in record:
                    stress_minrate_sat = float(record["stress_eval_embb_min_rate_satisfaction"])
                    stress_minrate_shortfall = float(record["stress_eval_embb_min_rate_shortfall"])
                    stress_admission = float(record["stress_eval_admission"])
                    stress_admitted_packets = float(record["stress_eval_admitted_packets"])
                    stress_total_power = float(record.get("stress_eval_total_power", 0.0) or 0.0)
                    stress_overlay_ratio = float(record.get("stress_eval_overlay_ratio", 0.0) or 0.0)
                    stress_score = (
                        float(getattr(cfg.training, "clean_stress_eval_embb_rate_weight", 1.0) or 1.0)
                        * float(record["stress_eval_embb_rate_mbps"])
                        + float(getattr(cfg.training, "clean_stress_eval_admission_weight", 22.0) or 22.0)
                        * stress_admission
                        + float(getattr(cfg.training, "clean_stress_eval_minrate_weight", 28.0) or 28.0)
                        * stress_minrate_sat
                        - float(getattr(cfg.training, "clean_stress_eval_minrate_shortfall_penalty_weight", 42.0) or 42.0)
                        * stress_minrate_shortfall
                        + float(getattr(cfg.training, "clean_stress_eval_packet_weight", 0.10) or 0.10)
                        * stress_admitted_packets
                        - float(getattr(cfg.training, "clean_stress_eval_power_penalty_weight", 0.0) or 0.0)
                        * stress_total_power
                        - float(getattr(cfg.training, "clean_stress_eval_overlay_penalty_weight", 0.0) or 0.0)
                        * stress_overlay_ratio
                    )
                    if stress_score > best_stress_eval_score:
                        best_stress_eval_score = stress_score
                        best_stress_eval_iter = int(iteration)
                        best_stress_eval_path = checkpoint_dir / f"{cfg.training.run_name}_clean_best_stress_eval.pt"
                        _save_clean_checkpoint(
                            best_stress_eval_path,
                            cfg=cfg,
                            model=model,
                            optimizer=optimizer,
                            history=history,
                            target_load=actual_target_load,
                            metadata={
                                "iteration": float(iteration),
                                "stress_eval_score": float(stress_score),
                                "stress_eval_embb_rate_mbps": float(record["stress_eval_embb_rate_mbps"]),
                                "stress_eval_embb_min_rate_satisfaction": float(record["stress_eval_embb_min_rate_satisfaction"]),
                                "stress_eval_embb_min_rate_shortfall": float(record["stress_eval_embb_min_rate_shortfall"]),
                                "stress_eval_admission": float(record["stress_eval_admission"]),
                                "stress_eval_admitted_packets": float(record["stress_eval_admitted_packets"]),
                                "stress_eval_total_power": float(record.get("stress_eval_total_power", 0.0) or 0.0),
                                "stress_eval_overlay_ratio": float(record.get("stress_eval_overlay_ratio", 0.0) or 0.0),
                                "stress_eval_puncture_ratio": float(record.get("stress_eval_puncture_ratio", 0.0) or 0.0),
                                "stress_eval_lambda_min": float(record.get("stress_eval_lambda_min", 0.0) or 0.0),
                                "stress_eval_lambda_max": float(record.get("stress_eval_lambda_max", 0.0) or 0.0),
                            },
                            extra_state={
                                "last_iteration": int(iteration),
                                "run_seed": int(run_seed),
                                "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                                "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                                "best_eval_iter": int(best_eval_iter),
                                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                                "best_stress_eval_iter": int(best_stress_eval_iter),
                                "best_stress_eval_score": float(stress_score),
                            },
                        )
            should_print = (
                iteration == 1
                or iteration == int(max(iterations, 1))
                or (iteration % progress_every == 0)
                or ("eval_embb_rate_mbps" in record)
            )
            if should_print:
                print(f"[{_taipei_timestamp()} Asia/Taipei] {line}", flush=True)
            if (iteration % history_flush_every == 0) or ("eval_embb_rate_mbps" in record):
                _write_clean_history(history_path, history)
    except KeyboardInterrupt:
        interrupted = True
        interrupt_iter = max(int(last_completed_iteration), 0)
        interrupt_path = checkpoint_dir / f"{cfg.training.run_name}_clean_interrupted.pt"
        _save_clean_checkpoint(
            interrupt_path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            history=history,
            target_load=float(target_load if target_load is not None else actual_load),
            metadata={
                "interrupted": 1.0,
                "iteration": float(interrupt_iter),
                "best_eval_iter": float(best_eval_iter),
                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                "best_stress_eval_iter": float(best_stress_eval_iter),
                "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
            },
            extra_state={
                "last_iteration": int(interrupt_iter),
                "run_seed": int(run_seed),
                "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                "best_eval_iter": int(best_eval_iter),
                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                "best_stress_eval_iter": int(best_stress_eval_iter),
                "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
            },
        )
        _write_clean_history(history_path, history)
        print(
            f"[{_taipei_timestamp()} Asia/Taipei] [CLEAN-MAPPO] interrupted at iter {interrupt_iter}; "
            f"saved {interrupt_path.name} and latest clean_history.json",
            flush=True,
        )
    if not interrupted:
        final_path = checkpoint_dir / f"{cfg.training.run_name}_clean_final.pt"
        _save_clean_checkpoint(
            final_path,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            history=history,
            target_load=float(target_load if target_load is not None else actual_load),
            metadata={
                "best_eval_iter": float(best_eval_iter),
                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                "best_stress_eval_iter": float(best_stress_eval_iter),
                "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
            },
            extra_state={
                "last_iteration": int(last_completed_iteration),
                "run_seed": int(run_seed),
                "cumulative_rollout_episode_count": int(cumulative_rollout_episode_count),
                "cumulative_rollout_agent_transitions": float(cumulative_rollout_agent_transitions),
                "best_eval_iter": int(best_eval_iter),
                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                "best_stress_eval_iter": int(best_stress_eval_iter),
                "best_stress_eval_score": float(best_stress_eval_score if np.isfinite(best_stress_eval_score) else float("-inf")),
            },
        )
    _write_clean_history(history_path, history)
    return {
        "history": history,
        "checkpoint_path": str(
            (checkpoint_dir / f"{cfg.training.run_name}_clean_interrupted.pt")
            if interrupted
            else (checkpoint_dir / f"{cfg.training.run_name}_clean_final.pt")
        ),
        "history_path": str(history_path),
        "history_jsonl_path": str(history_jsonl_path),
        "checkpoint_dir": str(checkpoint_dir),
        "experiment": experiment_label(cfg.training.experiment_line),
        "interrupted": bool(interrupted),
        "last_iteration": int(last_completed_iteration),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal clean MAPPO trainer for debugging learning dynamics.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--rollout-horizon-env-steps", type=int, default=None)
    parser.add_argument("--num-rollout-envs", type=int, default=None)
    parser.add_argument("--parallel-rollout-workers", type=int, default=None)
    parser.add_argument("--rollout-worker-device", type=str, default=None)
    parser.add_argument("--disable-parallel-rollout", action="store_true")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-load", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--urllc-activation-prob", type=float, default=None, help="Force nominal per-user URLLC activation probability control for rollout and eval.")
    parser.add_argument("--urllc-activation-prob-min", type=float, default=None, help="Minimum per-user URLLC activation probability sampled per episode during rollout.")
    parser.add_argument("--urllc-activation-prob-max", type=float, default=None, help="Maximum per-user URLLC activation probability sampled per episode during rollout.")
    parser.add_argument("--urllc-activation-prob-sampling", type=str, default="uniform", choices=["uniform", "high_bias"], help="Sampling strategy for per-episode URLLC activation probability when min/max are provided.")
    parser.add_argument("--urllc-poisson-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--urllc-poisson-rate-min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--urllc-poisson-rate-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--urllc-poisson-rate-sampling", type=str, default=None, choices=["uniform", "high_bias"], help=argparse.SUPPRESS)
    parser.add_argument("--random-loads", type=str, default="", help="Comma-separated per-UAV loads sampled per episode during rollout.")
    parser.add_argument("--random-urllc-rate-range", type=str, default="", help="Comma-separated min,max per-user URLLC activation probability sampled per episode during rollout.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume clean trainer state from a *_clean_interrupted.pt / *_clean_final.pt checkpoint.")
    parser.add_argument("--embb-power-scale-max", type=float, default=None, help="Override cfg.env.embb_power_scale_max during this run/resume.")
    args = parser.parse_args()
    if args.experiment is not None:
        normalized_experiment = normalize_experiment_line(args.experiment)
        if normalized_experiment not in EXPERIMENT_CHOICES:
            parser.error(
                f"argument --experiment: invalid choice: {args.experiment!r} "
                f"(normalized to {normalized_experiment!r})"
            )
    random_loads = [float(x.strip()) for x in str(args.random_loads or "").split(",") if x.strip()]
    rate_range = [float(x.strip()) for x in str(args.random_urllc_rate_range or "").split(",") if x.strip()]
    if rate_range and len(rate_range) != 2:
        raise ValueError("--random-urllc-rate-range expects two comma-separated values: min,max")
    activation_prob = args.urllc_activation_prob
    if activation_prob is None:
        activation_prob = args.urllc_poisson_rate
    activation_prob_min = args.urllc_activation_prob_min
    activation_prob_max = args.urllc_activation_prob_max
    if activation_prob_min is None:
        activation_prob_min = args.urllc_poisson_rate_min
    if activation_prob_max is None:
        activation_prob_max = args.urllc_poisson_rate_max
    activation_prob_sampling = str(
        args.urllc_activation_prob_sampling
        if args.urllc_activation_prob_sampling is not None
        else (args.urllc_poisson_rate_sampling if args.urllc_poisson_rate_sampling is not None else "uniform")
    )
    if activation_prob_min is not None or activation_prob_max is not None:
        if activation_prob_min is None or activation_prob_max is None:
            raise ValueError("--urllc-activation-prob-min and --urllc-activation-prob-max must be provided together")
        rate_range = [float(activation_prob_min), float(activation_prob_max)]
    run_clean_mappo(
        experiment=args.experiment,
        iterations=args.iterations,
        rollout_horizon=args.rollout_horizon,
        rollout_horizon_env_steps=args.rollout_horizon_env_steps,
        num_rollout_envs=args.num_rollout_envs,
        parallel_rollout_workers=args.parallel_rollout_workers,
        rollout_worker_device=args.rollout_worker_device,
        disable_parallel_rollout=bool(args.disable_parallel_rollout),
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        target_load=args.target_load,
        device=args.device,
        random_loads=random_loads or None,
        urllc_poisson_rate_range=rate_range or None,
        urllc_poisson_rate_sampling=activation_prob_sampling,
        urllc_poisson_rate=activation_prob,
        progress_every=args.progress_every,
        resume_from=args.resume_from,
        embb_power_scale_max=args.embb_power_scale_max,
    )
