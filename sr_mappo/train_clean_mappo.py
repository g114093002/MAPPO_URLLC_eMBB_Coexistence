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
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .networks import SRMAPPOActorCritic
from .trainer import configure_env_for_users_per_uav
from .types import MODE_KEEP, AgentObservation, HybridAction


@dataclass
class CleanStepRecord:
    env_id: int
    agent_id: str
    timestep: int
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
        "total_power": float(summary.get("total_power", 0.0) or 0.0),
        "overlay_ratio": float(summary.get("overlay_ratio", 0.0) or 0.0),
        "puncture_ratio": float(summary.get("puncture_ratio", 0.0) or 0.0),
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
            "terminal_urllc_admission": float(reward_term_totals.get("terminal_urllc_admission", 0.0) or 0.0),
            "terminal_embb_minrate_bonus": float(
                reward_term_totals.get("terminal_embb_min_rate_satisfaction_bonus", 0.0) or 0.0
            ),
            "terminal_power_budget_penalty": float(
                reward_term_totals.get("terminal_total_power_budget_penalty", 0.0) or 0.0
            ),
            "overlay_count": float(full.get("overlay_count", 0.0) or 0.0),
            "puncturing_count": float(full.get("puncture_count", 0.0) or 0.0),
            "power_violation": float(power_violation),
        }
    )
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
        "schedule_success_weight": float(getattr(reward, "schedule_success_weight", 0.0) or 0.0),
        "urgency_reward_weight": float(getattr(reward, "urgency_reward_weight", 0.0) or 0.0),
        "embb_damage_weight": float(getattr(reward, "embb_damage_weight", 0.0) or 0.0),
        "power_penalty_scale": float(getattr(reward, "power_penalty_scale", 0.0) or 0.0),
        "overlay_power_surcharge_weight": float(getattr(reward, "overlay_power_surcharge_weight", 0.0) or 0.0),
        "terminal_embb_rate_weight": float(getattr(reward, "terminal_embb_rate_weight", 0.0) or 0.0),
        "terminal_urllc_admission_weight": float(getattr(reward, "terminal_urllc_admission_weight", 0.0) or 0.0),
        "terminal_embb_min_rate_satisfaction_bonus_weight": float(
            getattr(reward, "terminal_embb_min_rate_satisfaction_bonus_weight", 0.0) or 0.0
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
) -> Dict[str, float]:
    results: List[Dict[str, float]] = []
    _print_consistency_log(
        "EVAL CONFIG",
        _consistency_snapshot(
            env,
            target_load=target_load,
            urllc_poisson_rate=nominal_urllc_poisson_rate,
            deterministic_action=True,
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
                    deterministic=True,
                )
                actions = _to_action_dict(env, observations, output)
                observations, _rewards, dones, _infos = env.step(actions, prebuilt_observations=observations)
                step_reward_terms = _extract_shared_reward_terms(_infos, env.agent_ids)
                for key, value in step_reward_terms.items():
                    reward_term_totals[str(key)] = reward_term_totals.get(str(key), 0.0) + float(value)
                actor_hidden = output.actor_hidden
                critic_hidden = output.critic_hidden
                done = all(dones.values())
            results.append(_evaluate_episode_summary(env, reward_term_totals))

    return {
        "eval_embb_rate_mbps": _mean_summary(results, "embb_rate_mbps"),
        "eval_avg_embb_rate_mbps": _mean_summary(results, "avg_embb_rate_mbps"),
        "eval_embb_min_rate_satisfaction": _mean_summary(results, "embb_min_rate_satisfaction"),
        "eval_embb_min_rate_shortfall": _mean_summary(results, "embb_min_rate_shortfall"),
        "eval_admission": _mean_summary(results, "admission"),
        "eval_admitted_packets": _mean_summary(results, "admitted_packets"),
        "eval_active_packets": _mean_summary(results, "active_packets"),
        "eval_total_power": _mean_summary(results, "total_power"),
        "eval_overlay_ratio": _mean_summary(results, "overlay_ratio"),
        "eval_puncture_ratio": _mean_summary(results, "puncture_ratio"),
        "eval_step_reward_sum": _mean_summary(results, "step_reward_sum"),
        "eval_terminal_reward_sum": _mean_summary(results, "terminal_reward_sum"),
        "eval_total_reward": _mean_summary(results, "total_reward"),
        "eval_planning_embb_rate_delta_reward": _mean_summary(results, "planning_embb_rate_delta_reward"),
        "eval_urgency_bonus": _mean_summary(results, "urgency_bonus"),
        "eval_embb_damage": _mean_summary(results, "embb_damage"),
        "eval_power_penalty": _mean_summary(results, "power_penalty"),
        "eval_terminal_urllc_admission": _mean_summary(results, "terminal_urllc_admission"),
        "eval_terminal_embb_minrate_bonus": _mean_summary(results, "terminal_embb_minrate_bonus"),
        "eval_terminal_power_budget_penalty": _mean_summary(results, "terminal_power_budget_penalty"),
        "eval_overlay_count": _mean_summary(results, "overlay_count"),
        "eval_puncturing_count": _mean_summary(results, "puncturing_count"),
        "eval_power_violation": _mean_summary(results, "power_violation"),
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
        env_step_cursor[env_idx] = 0

    def _finalize_env_episode(env_idx: int) -> None:
        env = envs[env_idx]
        summary = _summarize_episode(env)
        summary["episode_reward_mean"] = float(episode_reward_sum[env_idx] / max(episode_step_count[env_idx], 1))
        summary["episode_reward_components_mean"] = {
            key: float(value / max(episode_step_count[env_idx], 1))
            for key, value in episode_reward_terms_sum[env_idx].items()
        }
        episode_summaries.append(summary)

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
                _finalize_env_episode(env_idx)
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
            _finalize_env_episode(env_idx)

    return {
        "records": records,
        "next_seed": int(seed + episode_serial),
        "episodes": episode_summaries,
        "rollout_episode_count": float(len(episode_summaries)),
        "rollout_reward_mean": _mean_summary(episode_summaries, "episode_reward_mean"),
        "rollout_reward_components_mean": _mean_nested_summary(episode_summaries, "episode_reward_components_mean"),
        "rollout_embb_rate_mbps": _mean_summary(episode_summaries, "embb_rate_mbps"),
        "rollout_admission": _mean_summary(episode_summaries, "admission"),
        "rollout_admitted_packets": _mean_summary(episode_summaries, "admitted_packets"),
        "rollout_active_packets": _mean_summary(episode_summaries, "active_packets"),
        "sampled_load_mean": float(np.mean(np.asarray(sampled_loads, dtype=float))) if sampled_loads else 0.0,
        "sampled_urllc_poisson_rate_mean": float(np.mean(np.asarray(sampled_rates, dtype=float))) if sampled_rates else 0.0,
        "rollout_total_agent_transitions": float(steps_collected),
        "rollout_env_steps_per_env": _rollout_horizon_env_steps_per_env(
            rollout_horizon_agent_transitions=steps_collected,
            num_rollout_envs=num_envs,
            agent_count=agent_count,
        ),
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
    return {
        "batch": batch,
        "records": merged_records,
        "next_seed": int(seed + num_envs),
        "episodes": merged_episodes,
        "rollout_episode_count": float(len(merged_episodes)),
        "rollout_reward_mean": _mean_summary(merged_episodes, "episode_reward_mean"),
        "rollout_reward_components_mean": _mean_nested_summary(merged_episodes, "episode_reward_components_mean"),
        "rollout_embb_rate_mbps": _mean_summary(merged_episodes, "embb_rate_mbps"),
        "rollout_admission": _mean_summary(merged_episodes, "admission"),
        "rollout_admitted_packets": _mean_summary(merged_episodes, "admitted_packets"),
        "rollout_active_packets": _mean_summary(merged_episodes, "active_packets"),
        "sampled_load_mean": float(np.mean(np.asarray(sampled_loads, dtype=float))) if sampled_loads else 0.0,
        "sampled_urllc_poisson_rate_mean": float(np.mean(np.asarray(sampled_rates, dtype=float))) if sampled_rates else 0.0,
        "rollout_total_agent_transitions": float(total_transitions),
        "rollout_env_steps_per_env": float(np.mean(np.asarray(env_steps_list, dtype=float))) if env_steps_list else 0.0,
        "rollout_parallel_wall_time": float(perf_counter() - rollout_start),
        "rollout_worker_mean_time": float(np.mean(np.asarray(worker_times, dtype=float))) if worker_times else 0.0,
        "rollout_worker_max_time": float(np.max(np.asarray(worker_times, dtype=float))) if worker_times else 0.0,
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
    history: List[Dict[str, object]],
    target_load: float,
    metadata: Optional[Dict[str, float]] = None,
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
    torch.save(payload, path)


def _write_clean_history(path: Path, history: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_clean_history_jsonl(path: Path, record: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


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
        "eval_admitted_packets": float(record["eval_admitted_packets"]),
        "eval_active_packets": float(record["eval_active_packets"]),
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
    history_jsonl_path.write_text("", encoding="utf-8")
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
    try:
        for iteration in range(1, int(max(iterations, 1)) + 1):
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
                record.update(
                    _evaluate_policy(
                        env,
                        model,
                        episodes=int(max(eval_episodes, 1)),
                        seed=100_000 + run_seed,
                        target_load=target_load,
                        nominal_urllc_poisson_rate=nominal_urllc_poisson_rate,
                    )
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
                f"| pi={record['policy_loss']:.4f} vf={record['value_loss']:.4f} ent={record['entropy']:.4f} "
                f"| kl={record['approx_kl']:.5f} clip={record['clip_fraction']:.3f} ev={record['explained_variance']:.3f}"
            )
            if "eval_embb_rate_mbps" in record:
                line += (
                    f" | eval_embb={record['eval_embb_rate_mbps']:.3f} Mbps"
                    f" | eval_minrate={record['eval_embb_min_rate_satisfaction']:.4f}"
                    f" | eval_adm={record['eval_admission']:.4f}"
                    f" | eval_pkt={record['eval_admitted_packets']:.2f}/{record['eval_active_packets']:.2f}"
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
                    history=history,
                    target_load=actual_target_load,
                    metadata=eval_metadata,
                )
                minrate_sat = float(record["eval_embb_min_rate_satisfaction"])
                minrate_shortfall = float(record["eval_embb_min_rate_shortfall"])
                eval_total_power = float(record.get("eval_total_power", 0.0) or 0.0)
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
                        history=history,
                        target_load=actual_target_load,
                        metadata=milestone_metadata,
                    )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_iter = int(iteration)
                    best_eval_path = checkpoint_dir / f"{cfg.training.run_name}_clean_best_eval.pt"
                    _save_clean_checkpoint(
                        best_eval_path,
                        cfg=cfg,
                        model=model,
                        history=history,
                        target_load=actual_target_load,
                        metadata=_clean_eval_metadata(record, iteration, eval_score=eval_score),
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
            history=history,
            target_load=float(target_load if target_load is not None else actual_load),
            metadata={
                "best_eval_iter": float(best_eval_iter),
                "best_eval_score": float(best_eval_score if np.isfinite(best_eval_score) else float("-inf")),
                "best_stress_eval_iter": float(best_stress_eval_iter),
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
    parser.add_argument("--experiment", type=str, default=None, choices=EXPERIMENT_CHOICES)
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
    parser.add_argument("--urllc-poisson-rate", type=float, default=None, help="Force nominal per-user URLLC poisson rate for rollout and eval.")
    parser.add_argument("--urllc-poisson-rate-min", type=float, default=None, help="Minimum per-user URLLC poisson rate sampled per episode during rollout.")
    parser.add_argument("--urllc-poisson-rate-max", type=float, default=None, help="Maximum per-user URLLC poisson rate sampled per episode during rollout.")
    parser.add_argument("--urllc-poisson-rate-sampling", type=str, default="uniform", choices=["uniform", "high_bias"], help="Sampling strategy for per-episode URLLC poisson rate when min/max are provided.")
    parser.add_argument("--random-loads", type=str, default="", help="Comma-separated per-UAV loads sampled per episode during rollout.")
    parser.add_argument("--random-urllc-rate-range", type=str, default="", help="Comma-separated min,max per-user URLLC poisson rate sampled per episode during rollout.")
    args = parser.parse_args()
    random_loads = [float(x.strip()) for x in str(args.random_loads or "").split(",") if x.strip()]
    rate_range = [float(x.strip()) for x in str(args.random_urllc_rate_range or "").split(",") if x.strip()]
    if rate_range and len(rate_range) != 2:
        raise ValueError("--random-urllc-rate-range expects two comma-separated values: min,max")
    if args.urllc_poisson_rate_min is not None or args.urllc_poisson_rate_max is not None:
        if args.urllc_poisson_rate_min is None or args.urllc_poisson_rate_max is None:
            raise ValueError("--urllc-poisson-rate-min and --urllc-poisson-rate-max must be provided together")
        rate_range = [float(args.urllc_poisson_rate_min), float(args.urllc_poisson_rate_max)]
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
        urllc_poisson_rate_sampling=str(args.urllc_poisson_rate_sampling or "uniform"),
        urllc_poisson_rate=args.urllc_poisson_rate,
        progress_every=args.progress_every,
    )
