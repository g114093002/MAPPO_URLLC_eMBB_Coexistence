"""MAPPO-style trainer and workflow helpers for the sibling SR-MAPPO package."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from .bc import GreedyWarmStartTrainer, collect_greedy_bc_dataset
from .buffer import SharedRolloutBuffer
from .load_aware import nearest_reference_load
from .types import HybridAction, MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, ShieldedAction


@dataclass
class TrainerStats:
    rollout_steps: float
    mean_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    best_mode_aux_loss: float
    overlay_aux_loss: float
    best_packet_aux_loss: float
    phase_a_embb_power_anchor_loss: float


def _delta_to_pre_tanh(power_delta: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(power_delta, dtype=np.float32), -0.999999, 0.999999)
    return np.arctanh(clipped).astype(np.float32)


def _ensure_env_base_profile(env) -> Dict[str, float]:
    if hasattr(env, '_sr_mappo_base_profile'):
        return env._sr_mappo_base_profile

    base_embb_per_uav = max(1, int(np.ceil(env.sys_cfg.num_embb_users / env.sys_cfg.num_uavs)))
    base_urllc_per_uav = max(1, int(np.ceil(env.sys_cfg.num_urllc_users / env.sys_cfg.num_uavs)))
    env._sr_mappo_base_profile = {
        'base_embb_per_uav': float(base_embb_per_uav),
        'base_urllc_per_uav': float(base_urllc_per_uav),
        'base_total_per_uav': float(base_embb_per_uav + base_urllc_per_uav),
        'base_poisson_rate': float(env.sim_cfg.urllc_poisson_rate),
        'embb_power_dbm': float(env.embb_cfg.power_limits[0]) if env.embb_cfg.power_limits else 23.0,
        'urllc_power_dbm': float(env.urllc_cfg.power_limits[0]) if env.urllc_cfg.power_limits else 24.0,
    }
    return env._sr_mappo_base_profile


def configure_env_for_users_per_uav(env, target_users_per_uav: float) -> float:
    profile = _ensure_env_base_profile(env)
    base_total = max(profile['base_total_per_uav'], 1.0)
    scale = float(target_users_per_uav) / base_total

    total_users = max(1, int(round(profile['base_total_per_uav'] * env.sys_cfg.num_uavs * scale)))
    urllc_ratio = float(getattr(env.sim_cfg, 'urllc_user_ratio', 0.0))
    urllc_ratio = float(np.clip(urllc_ratio, 0.0, 0.95))
    if urllc_ratio > 0.0:
        env.sys_cfg.num_urllc_users = max(1, int(round(total_users * urllc_ratio)))
        env.sys_cfg.num_embb_users = max(1, total_users - env.sys_cfg.num_urllc_users)
    else:
        env.sys_cfg.num_embb_users = max(1, int(round(profile['base_embb_per_uav'] * env.sys_cfg.num_uavs * scale)))
        env.sys_cfg.num_urllc_users = max(1, int(round(profile['base_urllc_per_uav'] * env.sys_cfg.num_uavs * scale)))
    env.sys_cfg.refresh_derived_params()

    env.embb_cfg.power_limits = [profile['embb_power_dbm']] * env.sys_cfg.num_embb_users
    env.urllc_cfg.power_limits = [profile['urllc_power_dbm']] * env.sys_cfg.num_urllc_users
    if not bool(getattr(env.sim_cfg, 'fixed_urllc_poisson_rate', False)):
        env.sim_cfg.urllc_poisson_rate = max(1e-6, profile['base_poisson_rate'] * scale)
    env.channel_model.reset_topology()
    env.simulation.static_association = None
    return float((env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / env.sys_cfg.num_uavs)


def available_curriculum_loads(cfg, iteration: int) -> List[float]:
    loads = list(getattr(cfg.training, 'curriculum_loads', []))
    if not loads:
        return []
    if not getattr(cfg.training, 'use_load_curriculum', True):
        return loads
    stage = 1 + max(iteration - 1, 0) // max(cfg.training.curriculum_stage_iterations, 1)
    return loads[: min(stage, len(loads))]


def choose_training_load(cfg, iteration: int) -> Optional[float]:
    loads = available_curriculum_loads(cfg, iteration)
    if not loads:
        return None
    if len(loads) == 1:
        return float(loads[0])
    hardest_bias = float(np.clip(getattr(cfg.training, 'hardest_load_sampling_bias', 0.55), 0.0, 1.0))
    second_bias = float(np.clip(getattr(cfg.training, 'second_hardest_load_sampling_bias', 0.20), 0.0, 1.0 - hardest_bias))
    draw = np.random.rand()
    if draw < hardest_bias:
        return float(loads[-1])
    if len(loads) >= 2 and draw < hardest_bias + second_bias:
        return float(loads[-2])
    return float(np.random.choice(loads[:-2] if len(loads) > 2 else loads))


def teacher_guidance_scale(cfg, iteration: int) -> float:
    total_iterations = max(int(cfg.training.total_iterations), 1)
    start_iter = int(round(float(cfg.training.teacher_guidance_decay_start_frac) * total_iterations))
    end_iter = int(round(float(cfg.training.teacher_guidance_decay_end_frac) * total_iterations))
    final_scale = float(cfg.training.teacher_guidance_final_scale)
    final_scale = float(np.clip(final_scale, 0.0, 1.0))

    if iteration <= start_iter:
        return 1.0
    if end_iter <= start_iter:
        return final_scale
    if iteration >= end_iter:
        return final_scale

    progress = float(iteration - start_iter) / float(max(end_iter - start_iter, 1))
    return float(1.0 + (final_scale - 1.0) * np.clip(progress, 0.0, 1.0))


def teacher_distill_coef(cfg, iteration: int) -> float:
    if not bool(getattr(cfg.training, "use_teacher_distillation", False)):
        return 0.0
    total_iterations = max(int(cfg.training.total_iterations), 1)
    end_frac = float(np.clip(getattr(cfg.training, "teacher_distill_end_frac", 0.0) or 0.0, 0.0, 1.0))
    start_coef = float(getattr(cfg.training, "teacher_distill_coef_start", 0.0) or 0.0)
    end_coef = float(getattr(cfg.training, "teacher_distill_coef_end", 0.0) or 0.0)
    if end_frac <= 1.0e-9:
        return end_coef
    frac = float(max(iteration - 1, 0) / max(total_iterations - 1, 1))
    progress = float(np.clip(frac / max(end_frac, 1.0e-9), 0.0, 1.0))
    return float(start_coef + (end_coef - start_coef) * progress)


def greedy_reference_bc_coef(cfg, iteration: int) -> float:
    if not bool(getattr(cfg.training, "use_greedy_reference_bc", False)):
        return 0.0
    total_iterations = max(int(cfg.training.total_iterations), 1)
    start_coef = float(getattr(cfg.training, "greedy_bc_coef_start", 0.0) or 0.0)
    end_coef = float(getattr(cfg.training, "greedy_bc_coef_end", 0.0) or 0.0)
    end_frac = float(np.clip(getattr(cfg.training, "greedy_bc_end_frac", 0.0) or 0.0, 0.0, 1.0))
    warmup_iters = max(int(getattr(cfg.training, "greedy_bc_warmup_iters", 0) or 0), 0)
    if iteration <= warmup_iters:
        return start_coef
    if end_frac <= 1.0e-9:
        return end_coef
    end_iter = max(int(round(end_frac * total_iterations)), warmup_iters + 1)
    if iteration >= end_iter:
        return end_coef
    progress = float(iteration - warmup_iters) / float(max(end_iter - warmup_iters, 1))
    return float(start_coef + (end_coef - start_coef) * np.clip(progress, 0.0, 1.0))


def phase_a_embb_power_runtime_enabled(cfg, iteration: int) -> bool:
    if not bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)):
        return False
    start_iteration = max(int(getattr(cfg.training, "phase_a_embb_power_start_iteration", 0) or 0), 0)
    if start_iteration <= 0:
        return True
    return int(iteration) >= start_iteration


def phase_a_embb_power_anchor_enabled(cfg, iteration: int) -> bool:
    if not bool(getattr(cfg.training, "use_phase_a_embb_power_anchor", False)):
        return False
    if not bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)):
        return False
    start_iteration = max(int(getattr(cfg.training, "phase_a_embb_power_anchor_start_iteration", 0) or 0), 0)
    if start_iteration <= 0:
        return True
    return int(iteration) >= start_iteration


def set_phase_a_embb_power_runtime(env, model, enabled: bool) -> None:
    enabled = bool(enabled)
    if env is not None:
        setattr(env, "phase_a_embb_power_enabled", enabled)
    if model is not None:
        setattr(model, "phase_a_embb_power_enabled", enabled)


def _timing_enabled(cfg) -> bool:
    return bool(getattr(cfg.training, "enable_timing_logs", False))


def _tw_timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")


def _trainer_timing_log(cfg, message: str) -> None:
    if not _timing_enabled(cfg):
        return
    timestamp = _tw_timestamp()
    print(f"[{timestamp}] [SR-MAPPO][TIMING] {message}", flush=True)


def _resolve_training_device(requested: str) -> str:
    req = str(requested or "auto").strip().lower()
    if req in {"", "auto", "default"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        return req if torch.cuda.is_available() else "cpu"
    return req


def _value_for_load(mapping, actual_load: float, default: float) -> float:
    if not mapping:
        return float(default)
    normalized = {float(key): float(value) for key, value in dict(mapping).items()}
    bucket = nearest_reference_load(float(actual_load), normalized.keys())
    return float(normalized.get(bucket, default))


def _summary_float(summary: Optional[Dict], *keys: str, default: float = 0.0) -> float:
    if not isinstance(summary, dict):
        return float(default)
    for key in keys:
        if key in summary:
            try:
                return float(summary.get(key, default))
            except Exception:
                continue
    return float(default)


def _split_actor_critic_parameters(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    actor_params: list[torch.nn.Parameter] = []
    critic_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("critic_encoder.") or name.startswith("value_head."):
            critic_params.append(param)
        else:
            actor_params.append(param)
    return actor_params, critic_params


def _stable_phase_schedule(cfg, iteration: int, base_actor_lr: float, base_critic_lr: float) -> tuple[float, float, float, float]:
    actor_lr = float(base_actor_lr)
    critic_lr = float(base_critic_lr)
    entropy_coef = float(getattr(cfg.training, "entropy_coef", 0.0) or 0.0)
    clip_ratio = float(getattr(cfg.training, "clip_ratio", 0.2) or 0.2)
    stable_start = max(int(getattr(cfg.training, "stable_phase_start_iteration", 0) or 0), 0)
    if stable_start <= 0 or int(iteration) < stable_start:
        return actor_lr, critic_lr, entropy_coef, clip_ratio

    actor_scale = float(np.clip(getattr(cfg.training, "stable_phase_actor_lr_scale", 1.0) or 1.0, 0.0, 1.0))
    actor_lr = float(base_actor_lr * actor_scale)
    final_entropy = float(max(getattr(cfg.training, "stable_phase_entropy_coef_final", 0.0) or 0.0, 0.0))
    final_clip_ratio = float(max(getattr(cfg.training, "stable_phase_clip_ratio_final", clip_ratio) or clip_ratio, 1.0e-3))
    total_iterations = max(int(getattr(cfg.training, "total_iterations", stable_start) or stable_start), stable_start)
    if total_iterations <= stable_start:
        progress = 1.0
    else:
        progress = float(np.clip((int(iteration) - stable_start) / max(total_iterations - stable_start, 1), 0.0, 1.0))
    entropy_coef = float(entropy_coef + (final_entropy - entropy_coef) * progress)
    clip_ratio = float(clip_ratio + (final_clip_ratio - clip_ratio) * progress)
    return actor_lr, critic_lr, entropy_coef, clip_ratio


def _balanced_checkpoint_score(summary: Optional[Dict], cfg) -> tuple[float, float, float]:
    compare_selected = {}
    if isinstance(summary, dict):
        compare_selected = dict(summary.get("compare_selected_baseline") or {})
    throughput_ratio = _summary_float(
        compare_selected,
        "mean_rate_ratio",
        default=_summary_float(
            summary,
            "policy_throughput_vs_throughput_feasible_oracle",
            default=_summary_float(
                summary,
                "policy_throughput_vs_throughput_only_greedy",
                default=_summary_float(summary, "policy_throughput_vs_channel_only_greedy", default=0.0),
            ),
        ),
    )
    per_load = list(compare_selected.get("per_load", []) or [])
    admission_ratios: list[float] = []
    for item in per_load:
        if not isinstance(item, dict):
            continue
        policy_adm = float(item.get("policy_mean_scheduled_ratio", 0.0) or 0.0)
        greedy_adm = float(item.get("greedy_mean_scheduled_ratio", 0.0) or 0.0)
        if greedy_adm > 1.0e-9:
            admission_ratios.append(float(np.clip(policy_adm / greedy_adm, 0.0, 1.0)))
        else:
            admission_ratios.append(1.0 if policy_adm <= 1.0e-9 else 1.0)
    if admission_ratios:
        admission_ratio = float(np.mean(admission_ratios))
    else:
        policy_adm = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
        greedy_adm = _summary_float(compare_selected, "greedy_mean_scheduled_ratio", default=0.0)
        if greedy_adm > 1.0e-9:
            admission_ratio = float(np.clip(policy_adm / greedy_adm, 0.0, 1.0))
        else:
            admission_ratio = float(np.clip(policy_adm, 0.0, 1.0))
    throughput_weight = float(getattr(cfg.training, "balanced_checkpoint_throughput_weight", 0.80) or 0.80)
    admission_weight = float(getattr(cfg.training, "balanced_checkpoint_admission_weight", 0.20) or 0.20)
    power_ratio = _summary_float(
        compare_selected,
        "mean_power_ratio",
        default=_summary_float(
            summary,
            "policy_power_vs_throughput_feasible_oracle",
            default=_summary_float(
                summary,
                "policy_power_vs_throughput_only_greedy",
                default=_summary_float(summary, "policy_power_vs_channel_only_greedy", default=1.0),
            ),
        ),
    )
    power_penalty_weight = float(getattr(cfg.training, "balanced_checkpoint_power_penalty_weight", 0.0) or 0.0)
    power_penalty = float(max(power_ratio - 1.0, 0.0))
    # Throughput-first balanced score used for checkpoint selection:
    # score = throughput_weight * normalized_throughput
    #       + admission_weight * normalized_admission
    #       - power_penalty_weight * max(power_ratio - 1.0, 0.0)
    score = throughput_weight * throughput_ratio + admission_weight * admission_ratio - power_penalty_weight * power_penalty
    return float(score), float(throughput_ratio), float(admission_ratio)


def _service_interference_balanced_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """Composite checkpoint score for service+interference repair experiments.

    score =
      1.0 * normalized_embb_rate
    + 1.5 * urllc_admission
    + 3.0 * embb_service_ratio
    + 2.0 * embb_min_rate_satisfaction_ratio
    - 2.0 * embb_rate_loss_due_to_intercell_ratio
    - 1.0 * phase_a_embb_power_raw_saturation_ratio
    - 1.0 * phase_a_embb_power_cap_hit_ratio
    - 0.5 * normalized_total_power
    """
    if not isinstance(summary, dict):
        return float("-inf")
    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(summary, "policy_mean_embb_service_ratio", default=0.0)
    min_rate_ratio = _summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0)
    inter_loss_ratio = _summary_float(summary, "policy_mean_embb_rate_loss_due_to_intercell_ratio", default=0.0)
    sat = _summary_float(summary, "policy_mean_phase_a_embb_power_raw_saturation_ratio", default=0.0)
    cap = _summary_float(summary, "policy_mean_phase_a_embb_power_cap_hit_ratio", default=0.0)
    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(summary, "greedy_mean_power", default=0.0)
    # Normalizers: keep score magnitude stable without depending on simulator-only algo configs.
    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    power_norm = float(greedy_power) if greedy_power > 1.0e-12 else 1.0
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))
    normalized_power = float(np.clip(power / max(power_norm, 1.0e-9), 0.0, 5.0))
    score = (
        1.0 * normalized_rate
        + 1.5 * admission
        + 3.0 * service_ratio
        + 2.0 * min_rate_ratio
        - 2.0 * inter_loss_ratio
        - 1.0 * sat
        - 1.0 * cap
        - 0.5 * normalized_power
    )
    return float(score)


def _service_power_interference_balanced_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """Composite checkpoint score for service+power+interference repair experiments.

    score =
      2.5 * embb_service_ratio
    + 2.0 * embb_min_rate_satisfaction_ratio
    + 1.0 * normalized_embb_rate
    + 1.5 * urllc_admission
    - 2.5 * embb_rate_loss_due_to_intercell_ratio
    - 1.5 * max(0, total_power/greedy_power - 1.1)
    - 1.0 * phase_a_cap_hit_ratio
    - 1.0 * phase_a_raw_saturation_ratio
    + 0.5 * admitted_urllc_reliability
    """
    if not isinstance(summary, dict):
        return float("-inf")
    service_ratio = _summary_float(summary, "policy_mean_embb_service_ratio", default=0.0)
    min_rate_ratio = _summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0)
    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    inter_loss_ratio = _summary_float(summary, "policy_mean_embb_rate_loss_due_to_intercell_ratio", default=0.0)
    sat = _summary_float(summary, "policy_mean_phase_a_embb_power_raw_saturation_ratio", default=0.0)
    cap = _summary_float(summary, "policy_mean_phase_a_embb_power_cap_hit_ratio", default=0.0)
    reliability = _summary_float(summary, "policy_mean_admitted_urllc_reliability", default=_summary_float(summary, "policy_mean_reliability", default=0.0))
    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(summary, "greedy_mean_power", default=0.0)

    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))
    power_ratio = float(power / max(greedy_power, 1.0e-9)) if greedy_power > 1.0e-12 else 1.0
    power_over = max(power_ratio - 1.10, 0.0)
    score = (
        2.5 * service_ratio
        + 2.0 * min_rate_ratio
        + 1.0 * normalized_rate
        + 1.5 * admission
        - 2.5 * inter_loss_ratio
        - 1.5 * power_over
        - 1.0 * cap
        - 1.0 * sat
        + 0.5 * reliability
    )
    return float(score)


def _service_gain_interference_balanced_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """Checkpoint score (v3): reward service/min-rate gains vs selected greedy baseline, while penalizing
    intercell loss, power-over-greedy, Phase-A saturation/cap-hit, and admission shortfall.

    score =
      2.0 * normalized_embb_rate
    + 2.0 * urllc_admission
    + 3.0 * embb_service_ratio
    + 4.0 * max(embb_service_ratio - greedy_service_ratio, 0)
    + 2.0 * max(embb_min_rate_satisfaction_ratio - greedy_min_rate_satisfaction_ratio, 0)
    - 3.0 * max(greedy_service_ratio - embb_service_ratio, 0)
    - 3.0 * embb_rate_loss_due_to_intercell_ratio
    - 2.0 * max(0, total_power/greedy_power - 1.05)
    - 2.0 * phase_a_raw_saturation_ratio
    - 2.0 * phase_a_cap_hit_ratio
    - 2.0 * max(0, 0.68 - urllc_admission)
    """
    if not isinstance(summary, dict):
        return float("-inf")
    compare_selected = summary.get("compare_selected_baseline", {})
    if not isinstance(compare_selected, dict):
        compare_selected = {}

    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(summary, "policy_mean_embb_service_ratio", default=0.0)
    min_rate_ratio = _summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0)
    greedy_service_ratio = _summary_float(compare_selected, "greedy_mean_embb_service_ratio", default=0.0)
    greedy_min_rate_ratio = _summary_float(compare_selected, "greedy_mean_embb_min_rate_satisfaction_ratio", default=0.0)
    inter_loss_ratio = _summary_float(summary, "policy_mean_embb_rate_loss_due_to_intercell_ratio", default=0.0)
    sat = _summary_float(summary, "policy_mean_phase_a_embb_power_raw_saturation_ratio", default=0.0)
    cap = _summary_float(summary, "policy_mean_phase_a_embb_power_cap_hit_ratio", default=0.0)

    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(compare_selected, "greedy_mean_power", default=_summary_float(summary, "greedy_mean_power", default=0.0))
    power_ratio = float(power / max(greedy_power, 1.0e-9))
    power_over = float(max(power_ratio - 1.05, 0.0))

    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))
    service_gain = float(max(service_ratio - greedy_service_ratio, 0.0))
    min_gain = float(max(min_rate_ratio - greedy_min_rate_ratio, 0.0))
    service_shortfall = float(max(greedy_service_ratio - service_ratio, 0.0))
    admission_shortfall = float(max(0.68 - admission, 0.0))

    score = (
        2.0 * normalized_rate
        + 2.0 * admission
        + 3.0 * service_ratio
        + 4.0 * service_gain
        + 2.0 * min_gain
        - 3.0 * service_shortfall
        - 3.0 * inter_loss_ratio
        - 2.0 * power_over
        - 2.0 * sat
        - 2.0 * cap
        - 2.0 * admission_shortfall
    )
    return float(score)

def _balanced_intercell_aware_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """Short-horizon balanced checkpoint score (intercell-aware; avoids throughput-only selection).

    balanced_score =
      + 1.0 * urllc_admission
      + 0.8 * embb_service_ratio
      + 0.6 * embb_minrate_ratio
      + 0.5 * normalized_embb_rate
      - 1.0 * embb_rate_loss_due_to_intercell_ratio
      - 0.8 * normalized_mean_intercell
      - 0.2 * normalized_total_power
      + 0.4 * useful_deviation_ratio
      - 0.4 * harmful_deviation_ratio
    """
    if not isinstance(summary, dict):
        return float("-inf")
    compare_selected = summary.get("compare_selected_baseline", {})
    if not isinstance(compare_selected, dict):
        compare_selected = {}

    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(summary, "policy_mean_embb_service_ratio", default=0.0)
    min_rate_ratio = _summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0)
    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    inter_loss_ratio = _summary_float(summary, "policy_mean_embb_rate_loss_due_to_intercell_ratio", default=0.0)
    mean_intercell_mw = _summary_float(summary, "policy_mean_mean_intercell_interference_mw", default=0.0)
    greedy_mean_intercell_mw = _summary_float(compare_selected, "greedy_mean_mean_intercell_interference_mw", default=mean_intercell_mw)
    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(compare_selected, "greedy_mean_power", default=_summary_float(summary, "greedy_mean_power", default=0.0))
    useful = _summary_float(summary, "policy_mean_useful_deviation_ratio", default=0.0)
    harmful = _summary_float(summary, "policy_mean_harmful_deviation_ratio", default=0.0)

    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))
    normalized_power = float(np.clip(power / max(greedy_power, 1.0e-9), 0.0, 5.0))
    normalized_mean_intercell = float(np.clip(mean_intercell_mw / max(greedy_mean_intercell_mw, 1.0e-9), 0.0, 5.0))
    score = (
        1.0 * admission
        + 0.8 * service_ratio
        + 0.6 * min_rate_ratio
        + 0.5 * normalized_rate
        - 1.0 * inter_loss_ratio
        - 0.8 * normalized_mean_intercell
        - 0.2 * normalized_power
        + 0.4 * useful
        - 0.4 * harmful
    )
    return float(score)

def _owner_frozen_action_intercell_balanced_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """v4 checkpoint score aligned with coexistence objectives (not throughput-only).

    score =
      + 2.0 * urllc_admission
      + 2.0 * embb_service_ratio_after_puncture_deduction
      + 1.5 * embb_min_rate_satisfaction_after_puncture_deduction
      + 1.0 * normalized_embb_rate
      - 2.0 * intercell_per_admitted_packet_normalized
      - 0.5 * total_power_ratio_vs_greedy
    """
    if not isinstance(summary, dict):
        return float("-inf")
    compare_selected = summary.get("compare_selected_baseline", {})
    if not isinstance(compare_selected, dict):
        compare_selected = {}

    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(
        summary,
        "policy_mean_embb_service_ratio_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_service_ratio", default=0.0),
    )
    min_rate_ratio = _summary_float(
        summary,
        "policy_mean_embb_min_rate_satisfaction_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0),
    )
    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))

    intercell = _summary_float(summary, "policy_mean_intercell_per_admitted_packet", default=0.0)
    greedy_intercell = _summary_float(compare_selected, "greedy_mean_intercell_per_admitted_packet", default=intercell)
    intercell_norm = float(np.clip(intercell / max(greedy_intercell, 1.0e-12), 0.0, 5.0))

    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(compare_selected, "greedy_mean_power", default=_summary_float(summary, "greedy_mean_power", default=0.0))
    power_ratio = float(np.clip(power / max(greedy_power, 1.0e-9), 0.0, 5.0))

    score = (
        2.0 * admission
        + 2.0 * service_ratio
        + 1.5 * min_rate_ratio
        + 1.0 * normalized_rate
        - 2.0 * intercell_norm
        - 0.5 * power_ratio
    )
    return float(score)


def _v5_balanced_intercell_admission_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """v5 checkpoint score: admission/service first with intercell-per-admission and power control."""
    if not isinstance(summary, dict):
        return float("-inf")
    compare_selected = summary.get("compare_selected_baseline", {})
    if not isinstance(compare_selected, dict):
        compare_selected = {}

    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(
        summary,
        "policy_mean_embb_service_ratio_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_service_ratio", default=0.0),
    )
    min_rate_ratio = _summary_float(
        summary,
        "policy_mean_embb_min_rate_satisfaction_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0),
    )
    embb_rate = _summary_float(summary, "policy_mean_embb_rate", default=_summary_float(summary, "policy_throughput_score", default=0.0))
    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))

    intercell = _summary_float(summary, "policy_mean_intercell_per_admitted_packet", default=0.0)
    greedy_intercell = _summary_float(compare_selected, "greedy_mean_intercell_per_admitted_packet", default=intercell)
    normalized_intercell = float(np.clip(intercell / max(greedy_intercell, 1.0e-12), 0.0, 5.0))

    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(compare_selected, "greedy_mean_power", default=_summary_float(summary, "greedy_mean_power", default=0.0))
    power_ratio = float(np.clip(power / max(greedy_power, 1.0e-9), 0.0, 5.0))

    score = (
        2.5 * admission
        + 1.5 * service_ratio
        + 1.0 * min_rate_ratio
        + 0.5 * normalized_rate
        - 1.5 * normalized_intercell
        - 0.5 * power_ratio
    )
    return float(score)


def _v6_balanced_puncture_accounting_checkpoint_score(summary: Optional[Dict], cfg) -> float:
    """v6 checkpoint score: puncture-accounting-aware balance vs selected greedy baseline."""
    if not isinstance(summary, dict):
        return float("-inf")
    compare_selected = summary.get("compare_selected_baseline", {})
    if not isinstance(compare_selected, dict):
        compare_selected = {}

    admission = _summary_float(summary, "policy_mean_scheduled_ratio", default=0.0)
    service_ratio = _summary_float(
        summary,
        "policy_mean_embb_service_ratio_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_service_ratio", default=0.0),
    )
    min_rate_ratio = _summary_float(
        summary,
        "policy_mean_embb_min_rate_satisfaction_after_puncture_deduction",
        default=_summary_float(summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0),
    )
    embb_rate = _summary_float(
        summary,
        "policy_mean_embb_rate",
        default=_summary_float(summary, "policy_throughput_score", default=0.0),
    )
    rate_norm = float(getattr(cfg.reward, "terminal_embb_rate_normalizer", 5.0e6) or 5.0e6)
    normalized_rate = float(np.clip(embb_rate / max(rate_norm, 1.0e-9), 0.0, 5.0))

    policy_puncture_ratio = _summary_float(summary, "policy_mean_admission_via_puncture_ratio", default=0.0)
    greedy_puncture_ratio = _summary_float(
        compare_selected,
        "greedy_mean_admission_via_puncture_ratio",
        default=policy_puncture_ratio,
    )
    puncture_ratio_gap = float(abs(policy_puncture_ratio - greedy_puncture_ratio))

    intercell = _summary_float(summary, "policy_mean_intercell_per_admitted_packet", default=0.0)
    greedy_intercell = _summary_float(
        compare_selected,
        "greedy_mean_intercell_per_admitted_packet",
        default=intercell,
    )
    normalized_intercell = float(np.clip(intercell / max(greedy_intercell, 1.0e-12), 0.0, 5.0))

    power = _summary_float(summary, "policy_mean_power", default=0.0)
    greedy_power = _summary_float(
        compare_selected,
        "greedy_mean_power",
        default=_summary_float(summary, "greedy_mean_power", default=0.0),
    )
    power_ratio = float(np.clip(power / max(greedy_power, 1.0e-9), 0.0, 5.0))

    score = (
        2.0 * admission
        + 1.5 * service_ratio
        + 1.0 * min_rate_ratio
        + 0.5 * normalized_rate
        - 0.8 * puncture_ratio_gap
        - 1.0 * normalized_intercell
        - 0.5 * power_ratio
    )
    return float(score)


def _teacher_policy_actions(env, observations, policy_name: str):
    normalized = str(policy_name or "channel_only_greedy").strip().lower()
    if normalized not in {"channel_only", "channel_only_greedy"}:
        normalized = "channel_only_greedy"
    # Reuse the existing channel-only greedy helper rather than introducing a
    # second handcrafted teacher implementation.
    from .evaluate import _channel_only_actions

    return _channel_only_actions(env, observations)


def _teacher_distillation_targets(env, observations, cfg):
    num_agents = len(env.agent_ids)
    admission_target = np.zeros(num_agents, dtype=np.int64)
    mode_target = np.zeros(num_agents, dtype=np.int64)
    admission_weight = np.zeros(num_agents, dtype=np.float32)
    mode_weight = np.zeros(num_agents, dtype=np.float32)

    if not bool(getattr(cfg.training, "use_teacher_distillation", False)):
        return admission_target, mode_target, admission_weight, mode_weight

    planning_phase = all(
        bool(observations[agent_id].metadata.get("planning_phase", 0.0))
        for agent_id in env.agent_ids
    )
    if planning_phase:
        return admission_target, mode_target, admission_weight, mode_weight

    teacher_policy = str(getattr(cfg.training, "teacher_policy", "channel_only_greedy") or "channel_only_greedy").strip().lower()
    actual_load = float(env._current_actual_load())
    base_load_weight = _value_for_load(
        getattr(cfg.training, "teacher_load_weights", {}),
        actual_load,
        default=1.0,
    )
    puncture_load_floor = float(getattr(cfg.training, "teacher_prefer_puncture_load_floor", 0.0) or 0.0)

    if teacher_policy == "hard_safe_puncture_anchor":
        if not bool(getattr(env, "_hard_mode_anchor_stage_active", lambda: False)()):
            return admission_target, mode_target, admission_weight, mode_weight
        for idx, agent_id in enumerate(env.agent_ids):
            obs = observations[agent_id]
            if any(bool(env._candidate_supports_safe_puncture_anchor(candidate)) for candidate in obs.candidates):
                mode_target[idx] = 1
                mode_weight[idx] = float(base_load_weight)
        return admission_target, mode_target, admission_weight, mode_weight

    teacher_actions = _teacher_policy_actions(
        env,
        observations,
        teacher_policy,
    )
    minislot, rb = env._current_cell()
    resolved = env._resolve_executed_actions(
        teacher_actions,
        observations,
        minislot=minislot,
        rb=rb,
    )

    for idx, agent_id in enumerate(env.agent_ids):
        teacher_action = resolved[agent_id]
        teacher_mode = int(teacher_action.action.mode)
        is_admit = bool(teacher_mode in {MODE_OVERLAY, MODE_PUNCTURE} and teacher_action.candidate is not None)
        is_feasible = bool(teacher_action.candidate is not None) if is_admit else True
        positive_gap = float(teacher_action.utility) > 1.0e-9

        if bool(getattr(cfg.training, "teacher_only_feasible_action", True)) and not is_feasible:
            continue
        if bool(getattr(cfg.training, "teacher_only_positive_gap", True)) and not positive_gap:
            continue

        admission_target[idx] = 1 if is_admit else 0
        admission_weight[idx] = float(base_load_weight)
        if is_admit:
            mode_target[idx] = 0 if teacher_mode == MODE_OVERLAY else 1
            mode_multiplier = 1.0
            if actual_load >= puncture_load_floor - 1.0e-9 and teacher_mode == MODE_PUNCTURE:
                mode_multiplier = _value_for_load(
                    getattr(cfg.training, "teacher_prefer_puncture_weights_by_load", {}),
                    actual_load,
                    default=1.0,
                )
            mode_weight[idx] = float(base_load_weight * mode_multiplier)

    return admission_target, mode_target, admission_weight, mode_weight


def _greedy_reference_bc_targets(env, observations, cfg):
    num_agents = len(env.agent_ids)
    mode_target = np.zeros(num_agents, dtype=np.int64)
    packet_target = np.zeros(num_agents, dtype=np.int64)
    owner_target = np.zeros(num_agents, dtype=np.int64)
    mode_weight = np.zeros(num_agents, dtype=np.float32)
    packet_weight = np.zeros(num_agents, dtype=np.float32)
    owner_weight = np.zeros(num_agents, dtype=np.float32)
    phase_a_mask = np.zeros(num_agents, dtype=np.float32)

    if not bool(getattr(cfg.training, "use_greedy_reference_bc", False)):
        return (
            mode_target,
            packet_target,
            owner_target,
            mode_weight,
            packet_weight,
            owner_weight,
            phase_a_mask,
        )

    actual_load = float(env._current_actual_load())
    load_weight = _value_for_load(
        getattr(cfg.training, "greedy_bc_load_weights", {}),
        actual_load,
        default=1.0,
    )
    only_when_feasible = bool(getattr(cfg.training, "greedy_bc_only_when_feasible", True))
    only_positive_gap = bool(getattr(cfg.training, "greedy_bc_only_positive_gap", True))

    for idx, agent_id in enumerate(env.agent_ids):
        obs = observations[agent_id]
        planning_phase = bool(obs.metadata.get("planning_phase", 0.0))
        phase_a_mask[idx] = 0.0 if planning_phase else 1.0
        ref = obs.greedy_reference
        if ref is None:
            continue
        owner_space = str(getattr(cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()

        if planning_phase:
            valid_owner = bool(
                0 <= int(ref.embb_owner_option) < len(obs.masks.embb_owner_mask)
                and float(obs.masks.embb_owner_mask[int(ref.embb_owner_option)]) > 0.5
            )
            if owner_space == "global_owner_id_no_null":
                owner_active = bool(np.any(np.asarray(obs.masks.embb_owner_mask) > 0.5))
            else:
                owner_active = bool(np.any(np.asarray(obs.masks.embb_owner_mask)[1:] > 0.5))
            if valid_owner and owner_active:
                owner_target[idx] = int(ref.embb_owner_option)
                owner_weight[idx] = float(load_weight)
            continue

        positive_gap = float(getattr(obs, "greedy_reference_utility", 0.0)) > 1.0e-9
        if only_positive_gap and not positive_gap:
            continue

        ref_mode = int(ref.mode)
        ref_packet = int(ref.packet_option)
        valid_mode = bool(0 <= ref_mode < len(obs.masks.mode_mask) and float(obs.masks.mode_mask[ref_mode]) > 0.5)
        valid_packet = bool(
            0 <= ref_mode < obs.masks.packet_mask.shape[0]
            and 0 <= ref_packet < obs.masks.packet_mask.shape[1]
            and float(obs.masks.packet_mask[ref_mode, ref_packet]) > 0.5
        )
        if only_when_feasible and (not valid_mode or not valid_packet):
            continue

        mode_target[idx] = ref_mode
        packet_target[idx] = ref_packet
        mode_weight[idx] = float(load_weight if valid_mode else 0.0)
        packet_weight[idx] = float(load_weight if valid_packet else 0.0)

        valid_owner = bool(
            0 <= int(ref.embb_owner_option) < len(obs.masks.embb_owner_mask)
            and float(obs.masks.embb_owner_mask[int(ref.embb_owner_option)]) > 0.5
        )
        if owner_space == "global_owner_id_no_null":
            owner_active = bool(np.any(np.asarray(obs.masks.embb_owner_mask) > 0.5))
        else:
            owner_active = bool(np.any(np.asarray(obs.masks.embb_owner_mask)[1:] > 0.5))
        if valid_owner and owner_active:
            owner_target[idx] = int(ref.embb_owner_option)
            owner_weight[idx] = float(load_weight)

    return (
        mode_target,
        packet_target,
        owner_target,
        mode_weight,
        packet_weight,
        owner_weight,
        phase_a_mask,
    )


def _phase_a_embb_power_anchor_targets(env, observations, cfg, iteration: int):
    num_agents = len(env.agent_ids)
    anchor_target = np.zeros(num_agents, dtype=np.float32)
    anchor_weight = np.zeros(num_agents, dtype=np.float32)

    if not phase_a_embb_power_anchor_enabled(cfg, iteration):
        return anchor_target, anchor_weight
    if not bool(getattr(env, "phase_a_embb_power_enabled", False)):
        return anchor_target, anchor_weight

    actual_load = float(env._current_actual_load())
    load_weight = _value_for_load(
        getattr(cfg.training, "phase_a_embb_power_anchor_load_weights", {}),
        actual_load,
        default=1.0,
    )
    min_retention = float(getattr(cfg.training, "phase_a_embb_power_anchor_min_retention", 0.0) or 0.0)
    positive_gap_only = bool(getattr(cfg.training, "phase_a_embb_power_anchor_positive_gap_only", True))

    for idx, agent_id in enumerate(env.agent_ids):
        obs = observations[agent_id]
        if bool(obs.metadata.get("planning_phase", 0.0)):
            continue
        candidates = list(obs.candidates or [])
        if not candidates:
            continue
        candidate, mode, _score = env._best_local_candidate_frontier_throughput_admission(
            candidates,
            actual_load=actual_load,
        )
        if candidate is None or mode == MODE_KEEP:
            continue
        utility_gap = float(candidate.utility_for_mode(mode))
        if positive_gap_only and utility_gap <= 1.0e-9:
            continue

        retention = float(env._selected_embb_retention_ratio(candidate, mode))
        uav_idx = int(obs.metadata.get("uav_index", env._agent_index_map[agent_id][0]))
        rb_idx = int(obs.metadata.get("rb_index", env._agent_index_map[agent_id][1]))
        minislot = int(obs.metadata.get("minislot_index", env._current_cell()[0]))
        owner = int(candidate.embb_owner_for_mode(mode))
        if owner < 0:
            owner = int(env._actual_embb_owner_for_cell(uav_idx, rb_idx, minislot))
        if owner < 0:
            continue
        base_rate = float(env._base_rate_for_cell(uav_idx, owner, rb_idx))
        if base_rate <= 1.0e-9:
            continue

        loss_ratio = float(candidate.loss_for_mode(mode) / max(base_rate, 1.0e-9))
        overlay_margin = float(candidate.overlay_utility - candidate.puncture_utility)
        target = 0.0
        confidence = 0.0

        if mode == MODE_PUNCTURE:
            if utility_gap > 1.0e-9 and (
                actual_load >= 20.0
                or loss_ratio >= 0.30
                or retention < max(min_retention, 0.82)
            ):
                target = 1.0
                confidence = 1.0 + 0.50 * min(loss_ratio, 1.0)
        elif mode == MODE_OVERLAY:
            if (
                actual_load >= 15.0
                and utility_gap > 1.0e-9
                and retention < max(min_retention + 0.03, 0.88)
            ):
                target = 1.0
                confidence = 0.85
            elif (
                retention >= max(min_retention, 0.92)
                and overlay_margin > 1.0e-9
                and actual_load <= 15.0
            ):
                target = -1.0
                confidence = 0.60 + 0.40 * min(retention, 1.0)

        if abs(target) <= 1.0e-9 or confidence <= 1.0e-9:
            continue
        anchor_target[idx] = float(target)
        anchor_weight[idx] = float(load_weight * confidence)

    return anchor_target, anchor_weight


def sic_curriculum_db(cfg, iteration: int) -> float:
    total_iterations = max(int(cfg.training.total_iterations), 1)
    end_frac = float(np.clip(cfg.training.sic_curriculum_end_frac, 0.0, 1.0))
    end_iter = int(round(end_frac * total_iterations))
    start_db = float(cfg.training.sic_curriculum_start_db)
    end_db = float(cfg.training.sic_curriculum_end_db)
    if end_iter <= 1:
        return end_db
    if iteration >= end_iter:
        return end_db
    progress = float(iteration - 1) / float(max(end_iter - 1, 1))
    return float(start_db + (end_db - start_db) * np.clip(progress, 0.0, 1.0))


def _compute_aux_targets(
    observations: Dict[str, object],
    agent_ids: List[str],
    env=None,
    target_policy: str = "best_utility",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_mode_targets = []
    overlay_available_targets = []
    best_packet_targets = []
    policy = str(target_policy or "best_utility").strip().lower()
    for agent_id in agent_ids:
        obs = observations[agent_id]
        overlay_available_targets.append(float(any(candidate.overlay_feasible for candidate in obs.candidates)))
        best_mode = MODE_KEEP
        best_packet = 0
        if policy in {"throughput_feasible_oracle", "oracle", "throughput_first"} and env is not None:
            candidate, best_mode, _best_score = env._best_local_candidate_throughput_first(obs.candidates)
            if candidate is not None and best_mode != MODE_KEEP:
                try:
                    best_packet = obs.candidates.index(candidate) + 1
                except ValueError:
                    best_packet = 0
            else:
                best_mode = MODE_KEEP
                best_packet = 0
        elif policy == "load_aware_balanced_oracle" and env is not None:
            candidate, best_mode, _best_score = env._best_local_candidate_load_aware_balanced(
                obs.candidates,
                actual_load=env._current_actual_load(),
            )
            if candidate is not None and best_mode != MODE_KEEP:
                try:
                    best_packet = obs.candidates.index(candidate) + 1
                except ValueError:
                    best_packet = 0
            else:
                best_mode = MODE_KEEP
                best_packet = 0
        elif policy in {"frontier_throughput_admission_oracle", "frontier_oracle", "tp_admission_frontier"} and env is not None:
            candidate, best_mode, _best_score = env._best_local_candidate_frontier_throughput_admission(
                obs.candidates,
                actual_load=env._current_actual_load(),
            )
            if candidate is not None and best_mode != MODE_KEEP:
                try:
                    best_packet = obs.candidates.index(candidate) + 1
                except ValueError:
                    best_packet = 0
            else:
                best_mode = MODE_KEEP
                best_packet = 0
        else:
            best_utility = 0.0
            for option_idx, candidate in enumerate(obs.candidates, start=1):
                utility = float(candidate.best_utility)
                if np.isfinite(utility) and utility > best_utility:
                    best_utility = utility
                    best_mode = int(candidate.best_mode)
                    best_packet = option_idx
            if best_utility <= 0.0:
                best_mode = MODE_KEEP
                best_packet = 0
        best_mode_targets.append(int(best_mode))
        best_packet_targets.append(int(best_packet))
    return (
        np.asarray(best_mode_targets, dtype=np.int64),
        np.asarray(overlay_available_targets, dtype=np.float32),
        np.asarray(best_packet_targets, dtype=np.int64),
    )


class SRMAPPOTrainer:
    """A practical trainer skeleton for mixed-action recurrent MAPPO."""

    def __init__(self, env, model, cfg):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.device_requested = str(getattr(cfg.training, "device", "auto") or "auto")
        self.device_resolved = _resolve_training_device(self.device_requested)
        # Keep downstream components on the same resolved device.
        self.cfg.training.device = str(self.device_resolved)
        self.device = torch.device(self.device_resolved)
        self.model.to(self.device)
        actor_params, critic_params = _split_actor_critic_parameters(self.model)
        self.base_actor_lr = float(cfg.training.learning_rate)
        self.base_critic_lr = float(cfg.training.learning_rate)
        self.actor_group_index = 0
        self.critic_group_index = 1
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": self.base_actor_lr, "name": "actor"},
                {"params": critic_params, "lr": self.base_critic_lr, "name": "critic"},
            ]
        )
        self.buffer = SharedRolloutBuffer(
            num_agents=len(env.agent_ids),
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dim=cfg.network.recurrent_hidden_dim,
        )
        _ensure_env_base_profile(env)

    def set_optimizer_lrs(self, actor_lr: float, critic_lr: Optional[float] = None) -> None:
        critic_lr = float(actor_lr if critic_lr is None else critic_lr)
        self.optimizer.param_groups[self.actor_group_index]["lr"] = float(actor_lr)
        self.optimizer.param_groups[self.critic_group_index]["lr"] = float(critic_lr)

    def collect_rollout(self, horizon: Optional[int] = None, seed: Optional[int] = None, iteration: int = 1):
        horizon = horizon or self.cfg.training.rollout_horizon
        seed = self.cfg.training.train_seed if seed is None else seed
        collect_detailed_metrics = not bool(getattr(self.cfg.training, "fast_metrics", True))
        episode_seed = int(seed)
        observations, _info = self.env.reset(seed=episode_seed)
        actor_hidden, critic_hidden = self.model.initial_state(batch_size=len(self.env.agent_ids), device=self.device)
        self.buffer.reset()
        mode_entropy_values = []
        packet_entropy_values = []
        reward_term_totals = {}
        episodes_completed = 0
        embb_total_rates = []
        total_agent_decisions = 0
        planning_agent_decisions = 0
        owner_head_active_total = 0
        embb_power_head_active_total = 0
        phase_a_embb_power_head_active_total = 0
        phase_a_mode_decisions_total = 0
        raw_overlay_count = 0
        raw_puncture_count = 0
        executed_overlay_count = 0
        executed_puncture_count = 0
        shield_corrections_total = 0
        mode_rewrite_total = 0
        owner_rewrite_total = 0
        packet_rewrite_total = 0
        power_projection_total = 0
        any_safety_rewrite_total = 0
        mode_correction_count = 0
        packet_invalid_count = 0
        mask_invalid_count = 0
        joint_rewrite_count = 0
        raw_owner_non_null_total = 0
        executed_owner_non_null_total = 0
        raw_embb_power_nonzero_total = 0
        executed_embb_power_nonzero_total = 0
        phase_a_raw_embb_power_nonzero_total = 0
        phase_a_executed_embb_power_nonzero_total = 0
        phase_a_raw_embb_power_delta_sum = 0.0
        phase_a_executed_embb_power_delta_sum = 0.0
        phase_a_embb_power_clipped_total = 0
        phase_a_embb_power_invalid_or_masked_total = 0
        phase_a_embb_power_anchor_binding_count = 0
        phase_a_embb_power_anchor_binding_denom = 0
        raw_executed_mode_gap_total = 0
        raw_executed_packet_gap_total = 0
        raw_executed_power_gap_total = 0
        raw_executed_owner_gap_total = 0
        raw_executed_embb_power_gap_total = 0
        raw_executed_any_gap_total = 0
        owner_policy_entropy_sum = 0.0
        owner_policy_top1_prob_sum = 0.0
        owner_policy_snapshot_prob_sum = 0.0
        owner_policy_non_snapshot_prob_sum = 0.0
        owner_policy_stat_count = 0
        sampled_owner_option_sum = 0.0
        sampled_owner_option_total = 0
        sampled_owner_option_nonzero_count = 0
        sampled_owner_option_valid_count = 0
        sampled_owner_option_snapshot_comparable_count = 0
        sampled_owner_option_equals_snapshot_count = 0
        sampled_owner_option_non_snapshot_count = 0
        ph0_owner_snapshot_comparable_count = 0
        ph0_owner_raw_same_as_snapshot_count = 0
        ph0_owner_raw_non_snapshot_count = 0
        ph0_owner_raw_null_count = 0
        owner_decode_debug_examples: List[Tuple[int, int, int, List[int]]] = []
        episode_summaries = []
        obs_build_sec = 0.0
        policy_forward_sec = 0.0
        greedy_bc_target_sec = 0.0
        env_step_sec = 0.0
        buffer_add_sec = 0.0

        def _planning_owner_context(agent_id: str, obs) -> Tuple[int, int, int, List[int]]:
            uav_idx, _rb_idx = self.env._agent_index_map.get(agent_id, (0, 0))
            rb_idx = int(obs.metadata.get("rb_index", getattr(self.env, "_current_planning_rb", lambda: 0)()))
            snapshot_owner = -1
            snapshot_map = getattr(self.env, "phase0_snapshot_owner_per_uav_rb", None)
            if snapshot_map is not None:
                arr = np.asarray(snapshot_map, dtype=int)
                if 0 <= uav_idx < arr.shape[0] and 0 <= rb_idx < arr.shape[1]:
                    snapshot_owner = int(arr[uav_idx, rb_idx])
            candidates: List[int] = []
            cand_map = getattr(self.env, "embb_owner_candidates_by_uav_rb", None)
            if cand_map is not None:
                try:
                    candidates = [int(v) for v in cand_map[uav_idx][rb_idx]]
                except Exception:
                    candidates = []
            return int(uav_idx), int(rb_idx), int(snapshot_owner), candidates

        def _decode_raw_owner_from_option(agent_id: str, obs, raw_option: int) -> Tuple[int, bool]:
            owner_space = str(getattr(self.cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
            owner_mask = np.asarray(obs.masks.embb_owner_mask, dtype=float)
            _uav, _rb, _snapshot_owner, candidates = _planning_owner_context(agent_id, obs)
            if owner_space == "global_owner_id_no_null":
                raw_valid = (
                    0 <= int(raw_option) < owner_mask.size
                    and float(owner_mask[int(raw_option)]) > 0.5
                    and int(raw_option) < int(self.env.sys_cfg.num_embb_users)
                )
                return (int(raw_option) if raw_valid else -1), False
            raw_null = int(raw_option) == 0
            raw_valid = (
                0 <= int(raw_option) < owner_mask.size
                and float(owner_mask[int(raw_option)]) > 0.5
                and (raw_null or (int(raw_option) - 1) < len(candidates))
            )
            if raw_valid and int(raw_option) > 0 and (int(raw_option) - 1) < len(candidates):
                return int(candidates[int(raw_option) - 1]), raw_null
            return -1, raw_null

        steps = 0
        done = False
        agent_index = {agent_id: idx for idx, agent_id in enumerate(self.env.agent_ids)}
        # Precompute whether expensive auxiliary targets are actually needed this rollout.
        distill_coef_now = float(teacher_distill_coef(self.cfg, int(iteration)))
        greedy_bc_coef_now = float(greedy_reference_bc_coef(self.cfg, int(iteration)))
        need_teacher_distill_targets = bool(
            distill_coef_now > 1.0e-12
            and (
                float(getattr(self.cfg.training, "teacher_admission_loss_weight", 0.0) or 0.0) > 1.0e-12
                or float(getattr(self.cfg.training, "teacher_mode_loss_weight", 0.0) or 0.0) > 1.0e-12
            )
        )
        need_greedy_bc_targets = bool(
            greedy_bc_coef_now > 1.0e-12
            and bool(getattr(self.cfg.training, "use_greedy_reference_bc", False))
            and (
                float(getattr(self.cfg.training, "greedy_bc_mode_weight", 0.0) or 0.0) > 1.0e-12
                or float(getattr(self.cfg.training, "greedy_bc_packet_weight", 0.0) or 0.0) > 1.0e-12
                or float(getattr(self.cfg.training, "greedy_bc_owner_weight", 0.0) or 0.0) > 1.0e-12
            )
        )
        need_phase_a_anchor_targets = bool(
            bool(getattr(self.cfg.training, "use_phase_a_embb_power_anchor", False))
            and float(getattr(self.cfg.training, "phase_a_embb_power_anchor_weight", 0.0) or 0.0) > 1.0e-12
            and phase_a_embb_power_anchor_enabled(self.cfg, int(iteration))
        )

        while steps < horizon:
            obs_build_start = perf_counter()
            local_obs_np = np.stack([observations[agent_id].local_obs for agent_id in self.env.agent_ids]).astype(np.float32)
            global_obs_np = np.stack([observations[agent_id].global_obs for agent_id in self.env.agent_ids]).astype(np.float32)
            mode_mask_np = np.stack([observations[agent_id].masks.mode_mask for agent_id in self.env.agent_ids]).astype(np.float32)
            packet_mask_np = np.stack([observations[agent_id].masks.packet_mask for agent_id in self.env.agent_ids]).astype(np.float32)
            embb_owner_mask_np = np.stack([observations[agent_id].masks.embb_owner_mask for agent_id in self.env.agent_ids]).astype(np.float32)
            aux_best_mode_target, aux_overlay_target, aux_best_packet_target = _compute_aux_targets(
                observations,
                self.env.agent_ids,
                env=self.env,
                target_policy=self.cfg.training.aux_target_policy,
            )
            num_agents = len(self.env.agent_ids)
            if need_teacher_distill_targets:
                teacher_admission_target, teacher_mode_target, teacher_admission_weight, teacher_mode_weight = (
                    _teacher_distillation_targets(
                        self.env,
                        observations,
                        self.cfg,
                    )
                )
            else:
                teacher_admission_target = np.zeros(num_agents, dtype=np.int64)
                teacher_mode_target = np.zeros(num_agents, dtype=np.int64)
                teacher_admission_weight = np.zeros(num_agents, dtype=np.float32)
                teacher_mode_weight = np.zeros(num_agents, dtype=np.float32)
            obs_build_sec += perf_counter() - obs_build_start

            greedy_bc_start = perf_counter()
            if need_greedy_bc_targets:
                (
                    greedy_bc_mode_target,
                    greedy_bc_packet_target,
                    greedy_bc_owner_target,
                    greedy_bc_mode_weight,
                    greedy_bc_packet_weight,
                    greedy_bc_owner_weight,
                    phase_a_mask,
                ) = _greedy_reference_bc_targets(
                    self.env,
                    observations,
                    self.cfg,
                )
            else:
                greedy_bc_mode_target = np.zeros(num_agents, dtype=np.int64)
                greedy_bc_packet_target = np.zeros(num_agents, dtype=np.int64)
                greedy_bc_owner_target = np.zeros(num_agents, dtype=np.int64)
                greedy_bc_mode_weight = np.zeros(num_agents, dtype=np.float32)
                greedy_bc_packet_weight = np.zeros(num_agents, dtype=np.float32)
                greedy_bc_owner_weight = np.zeros(num_agents, dtype=np.float32)
                phase_a_mask = np.asarray(
                    [0.0 if bool(observations[agent_id].metadata.get("planning_phase", 0.0)) else 1.0 for agent_id in self.env.agent_ids],
                    dtype=np.float32,
                )
            if need_phase_a_anchor_targets:
                (
                    phase_a_embb_power_anchor_target,
                    phase_a_embb_power_anchor_weight,
                ) = _phase_a_embb_power_anchor_targets(
                    self.env,
                    observations,
                    self.cfg,
                    iteration=int(iteration),
                )
            else:
                phase_a_embb_power_anchor_target = np.zeros(num_agents, dtype=np.float32)
                phase_a_embb_power_anchor_weight = np.zeros(num_agents, dtype=np.float32)
            greedy_bc_target_sec += perf_counter() - greedy_bc_start

            obs_tensor_start = perf_counter()
            local_obs = torch.from_numpy(local_obs_np).to(self.device)
            global_obs = torch.from_numpy(global_obs_np).to(self.device)
            mode_mask = torch.from_numpy(mode_mask_np).to(self.device)
            packet_mask = torch.from_numpy(packet_mask_np).to(self.device)
            embb_owner_mask = torch.from_numpy(embb_owner_mask_np).to(self.device)
            obs_build_sec += perf_counter() - obs_tensor_start

            prev_actor_hidden = actor_hidden.detach().cpu().numpy().squeeze(0)
            prev_critic_hidden = critic_hidden.detach().cpu().numpy().squeeze(0)
            policy_forward_start = perf_counter()
            output = self.model.act(
                local_obs=local_obs,
                global_obs=global_obs,
                mode_mask=mode_mask,
                packet_mask=packet_mask,
                embb_owner_mask=embb_owner_mask,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                deterministic=False,
            )
            mode_entropy_values.append(float(output.mode_entropy.mean().item()))
            packet_entropy_values.append(float(output.packet_entropy.mean().item()))
            owner_logits_np = None
            if collect_detailed_metrics and getattr(output, "embb_owner_logits", None) is not None:
                try:
                    owner_logits_np = output.embb_owner_logits.detach().cpu().numpy()
                except Exception:
                    owner_logits_np = None

            joint_actions = {}
            for idx, agent_id in enumerate(self.env.agent_ids):
                joint_actions[agent_id] = HybridAction(
                    mode=int(output.mode[idx].item()),
                    packet_option=int(output.packet_option[idx].item()),
                    power_delta=0.0,
                    embb_owner_option=int(output.embb_owner_option[idx].item()),
                    embb_power_delta=float(output.embb_power_delta[idx].item()),
                )
                if not collect_detailed_metrics:
                    continue
                head_activity = self.env.action_head_activity(observations[agent_id])
                if head_activity["planning_phase"] and head_activity["owner_active"]:
                    _uav_idx, _rb_idx, snapshot_owner, candidates = _planning_owner_context(agent_id, observations[agent_id])
                    owner_space = str(getattr(self.cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
                    owner_mask = np.asarray(observations[agent_id].masks.embb_owner_mask, dtype=float)
                    sampled_option = int(joint_actions[agent_id].embb_owner_option)
                    sampled_owner_option_total += 1
                    sampled_owner_option_sum += float(sampled_option)
                    sampled_owner_option_nonzero_count += int(sampled_option != 0)
                    sampled_owner_option_valid_count += int(
                        0 <= sampled_option < owner_mask.size and float(owner_mask[sampled_option]) > 0.5
                    )
                    snapshot_option_idx = -1
                    if snapshot_owner >= 0:
                        if owner_space == "global_owner_id_no_null":
                            if 0 <= int(snapshot_owner) < owner_mask.size:
                                snapshot_option_idx = int(snapshot_owner)
                        else:
                            for opt_i, owner_id in enumerate(candidates, start=1):
                                if int(owner_id) == int(snapshot_owner):
                                    snapshot_option_idx = int(opt_i)
                                    break
                    if snapshot_option_idx >= 0:
                        sampled_owner_option_snapshot_comparable_count += 1
                        if sampled_option == snapshot_option_idx:
                            sampled_owner_option_equals_snapshot_count += 1
                        else:
                            sampled_owner_option_non_snapshot_count += 1

                    raw_owner, raw_null = _decode_raw_owner_from_option(
                        agent_id,
                        observations[agent_id],
                        sampled_option,
                    )
                    if snapshot_owner >= 0:
                        ph0_owner_snapshot_comparable_count += 1
                        if raw_null or raw_owner < 0:
                            ph0_owner_raw_null_count += 1
                        elif int(raw_owner) == int(snapshot_owner):
                            ph0_owner_raw_same_as_snapshot_count += 1
                        else:
                            ph0_owner_raw_non_snapshot_count += 1

                    if owner_logits_np is not None and idx < owner_logits_np.shape[0]:
                        logits_row = np.asarray(owner_logits_np[idx], dtype=float)
                        logits_row = logits_row - float(np.max(logits_row))
                        probs = np.exp(logits_row)
                        probs = probs / max(float(np.sum(probs)), 1.0e-12)
                        owner_policy_entropy_sum += float(
                            -np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)))
                        )
                        owner_policy_top1_prob_sum += float(np.max(probs))
                        owner_policy_stat_count += 1

                        snapshot_action_idx = -1
                        if snapshot_owner >= 0:
                            if owner_space == "global_owner_id_no_null":
                                if 0 <= int(snapshot_owner) < probs.size:
                                    snapshot_action_idx = int(snapshot_owner)
                            else:
                                for opt_i, owner_id in enumerate(candidates, start=1):
                                    if int(owner_id) == int(snapshot_owner):
                                        snapshot_action_idx = int(opt_i)
                                        break
                        if 0 <= snapshot_action_idx < probs.size:
                            owner_policy_snapshot_prob_sum += float(probs[snapshot_action_idx])

                        non_snapshot_indices: List[int] = []
                        if owner_space == "global_owner_id_no_null":
                            valid = np.where(owner_mask > 0.5)[0]
                            for oi in valid.tolist():
                                if oi >= int(self.env.sys_cfg.num_embb_users):
                                    continue
                                if int(oi) == int(snapshot_owner):
                                    continue
                                non_snapshot_indices.append(int(oi))
                        else:
                            valid = np.where(owner_mask > 0.5)[0]
                            for oi in valid.tolist():
                                if oi <= 0 or oi >= len(owner_mask):
                                    continue
                                cand_idx = int(oi) - 1
                                if not (0 <= cand_idx < len(candidates)):
                                    continue
                                if int(candidates[cand_idx]) == int(snapshot_owner):
                                    continue
                                non_snapshot_indices.append(int(oi))
                        if non_snapshot_indices:
                            owner_policy_non_snapshot_prob_sum += float(np.sum(probs[non_snapshot_indices]))

            planning_phase = all(
                bool(observations[agent_id].metadata.get("planning_phase", 0.0))
                for agent_id in self.env.agent_ids
            )
            if (not planning_phase) and (not bool(getattr(self.env.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))):
                # Avoid artificial autonomy drops when Phase-A power is disabled: keep raw action consistent
                # with the execution path (which will zero the Phase-A eMBB power delta when inactive).
                for agent_id in self.env.agent_ids:
                    joint_actions[agent_id].embb_power_delta = 0.0
            if planning_phase:
                resolved = {
                    agent_id: self.env._raw_action_to_shielded_action(joint_actions[agent_id], observations[agent_id])
                    for agent_id in self.env.agent_ids
                }
            else:
                minislot, rb = self.env._current_cell()
                resolved = self.env._resolve_executed_actions(
                    joint_actions,
                    observations,
                    minislot=minislot,
                    rb=rb,
                )
            executed_mode_actions = np.asarray(
                [resolved[agent_id].action.mode for agent_id in self.env.agent_ids],
                dtype=np.int64,
            )
            executed_packet_actions = np.asarray(
                [resolved[agent_id].action.packet_option for agent_id in self.env.agent_ids],
                dtype=np.int64,
            )
            executed_power_delta = np.asarray(
                [resolved[agent_id].action.power_delta for agent_id in self.env.agent_ids],
                dtype=np.float32,
            )
            executed_embb_owner_actions = np.asarray(
                [resolved[agent_id].action.embb_owner_option for agent_id in self.env.agent_ids],
                dtype=np.int64,
            )
            executed_embb_power_delta = np.asarray(
                [resolved[agent_id].action.embb_power_delta for agent_id in self.env.agent_ids],
                dtype=np.float32,
            )
            for agent_id in self.env.agent_ids:
                total_agent_decisions += 1
                head_activity = self.env.action_head_activity(observations[agent_id])
                if head_activity["planning_phase"]:
                    planning_agent_decisions += 1
                if not collect_detailed_metrics:
                    diff_flags = self.env.action_diff_flags(joint_actions[agent_id], resolved[agent_id].action)
                    raw_executed_mode_gap_total += int(diff_flags["mode"])
                    raw_executed_packet_gap_total += int(diff_flags["packet"])
                    raw_executed_power_gap_total += int(diff_flags["power"])
                    raw_executed_owner_gap_total += int(diff_flags["owner"])
                    raw_executed_embb_power_gap_total += int(diff_flags["embb_power"])
                    raw_executed_any_gap_total += int(any(diff_flags.values()))
                    if not head_activity["planning_phase"]:
                        phase_a_mode_decisions_total += 1
                        raw_overlay_count += int(int(joint_actions[agent_id].mode) == MODE_OVERLAY)
                        raw_puncture_count += int(int(joint_actions[agent_id].mode) == MODE_PUNCTURE)
                        executed_overlay_count += int(int(resolved[agent_id].action.mode) == MODE_OVERLAY)
                        executed_puncture_count += int(int(resolved[agent_id].action.mode) == MODE_PUNCTURE)
                        mode_rewritten = bool(diff_flags["mode"])
                        owner_rewritten = bool(diff_flags["owner"])
                        packet_invalid_rewritten = bool(
                            resolved[agent_id].packet_invalid_fallback
                            or resolved[agent_id].mask_invalid_fallback
                        )
                        power_projection_applied = bool(diff_flags["power"] or diff_flags["embb_power"])
                        any_safety_rewrite = bool(
                            mode_rewritten
                            or owner_rewritten
                            or packet_invalid_rewritten
                            or resolved[agent_id].used_greedy_fallback
                            or resolved[agent_id].collision_rewritten
                            or resolved[agent_id].joint_reliability_rewritten
                        )
                        mode_rewrite_total += int(mode_rewritten)
                        owner_rewrite_total += int(owner_rewritten)
                        packet_rewrite_total += int(packet_invalid_rewritten)
                        power_projection_total += int(power_projection_applied)
                        any_safety_rewrite_total += int(any_safety_rewrite)
                        shield_corrections_total += int(any_safety_rewrite)
                        mode_correction_count += int(bool(resolved[agent_id].mode_corrected))
                        packet_invalid_count += int(bool(resolved[agent_id].packet_invalid_fallback))
                        mask_invalid_count += int(bool(resolved[agent_id].mask_invalid_fallback))
                        joint_rewrite_count += int(bool(resolved[agent_id].joint_reliability_rewritten))
                    continue
                if head_activity["owner_active"]:
                    owner_head_active_total += 1
                    owner_space = str(getattr(self.cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
                    owner_mask = np.asarray(observations[agent_id].masks.embb_owner_mask, dtype=float)
                    if owner_space == "global_owner_id_no_null":
                        raw_idx = int(joint_actions[agent_id].embb_owner_option)
                        exe_idx = int(resolved[agent_id].action.embb_owner_option)
                        raw_owner_non_null_total += int(0 <= raw_idx < owner_mask.size and owner_mask[raw_idx] > 0.5)
                        executed_owner_non_null_total += int(0 <= exe_idx < owner_mask.size and owner_mask[exe_idx] > 0.5)
                    else:
                        raw_owner_non_null_total += int(int(joint_actions[agent_id].embb_owner_option) > 0)
                        executed_owner_non_null_total += int(int(resolved[agent_id].action.embb_owner_option) > 0)
                if head_activity["embb_power_active"]:
                    embb_power_head_active_total += 1
                    raw_embb_power_nonzero_total += int(abs(float(joint_actions[agent_id].embb_power_delta)) > 1e-3)
                    executed_embb_power_nonzero_total += int(abs(float(resolved[agent_id].action.embb_power_delta)) > 1e-3)
                if head_activity["phase_a_embb_power_active"]:
                    idx = agent_index[agent_id]
                    phase_a_embb_power_head_active_total += 1
                    phase_a_raw_embb_power_nonzero_total += int(abs(float(joint_actions[agent_id].embb_power_delta)) > 1e-3)
                    phase_a_executed_embb_power_nonzero_total += int(abs(float(resolved[agent_id].action.embb_power_delta)) > 1e-3)
                    phase_a_raw_embb_power_delta_sum += float(joint_actions[agent_id].embb_power_delta)
                    phase_a_executed_embb_power_delta_sum += float(resolved[agent_id].action.embb_power_delta)
                    phase_a_embb_power_anchor_binding_denom += 1
                    phase_a_embb_power_anchor_binding_count += int(
                        idx < len(phase_a_embb_power_anchor_weight)
                        and float(phase_a_embb_power_anchor_weight[idx]) > 1e-9
                    )
                    power_info = dict(getattr(resolved[agent_id], "phase_a_embb_power_info", {}) or {})
                    phase_a_embb_power_clipped_total += int(
                        bool(power_info.get("delta_was_clipped", False))
                        or bool(power_info.get("scale_was_clipped", False))
                    )
                    phase_a_embb_power_invalid_or_masked_total += int(bool(power_info.get("invalid_or_masked", False)))
                diff_flags = self.env.action_diff_flags(joint_actions[agent_id], resolved[agent_id].action)
                raw_executed_mode_gap_total += int(diff_flags["mode"])
                raw_executed_packet_gap_total += int(diff_flags["packet"])
                raw_executed_power_gap_total += int(diff_flags["power"])
                raw_executed_owner_gap_total += int(diff_flags["owner"])
                raw_executed_embb_power_gap_total += int(diff_flags["embb_power"])
                raw_executed_any_gap_total += int(any(diff_flags.values()))
                if not head_activity["planning_phase"]:
                    phase_a_mode_decisions_total += 1
                    raw_overlay_count += int(int(joint_actions[agent_id].mode) == MODE_OVERLAY)
                    raw_puncture_count += int(int(joint_actions[agent_id].mode) == MODE_PUNCTURE)
                    executed_overlay_count += int(int(resolved[agent_id].action.mode) == MODE_OVERLAY)
                    executed_puncture_count += int(int(resolved[agent_id].action.mode) == MODE_PUNCTURE)
                    mode_rewritten = bool(diff_flags["mode"])
                    owner_rewritten = bool(diff_flags["owner"])
                    packet_invalid_rewritten = bool(
                        resolved[agent_id].packet_invalid_fallback
                        or resolved[agent_id].mask_invalid_fallback
                    )
                    power_projection_applied = bool(diff_flags["power"] or diff_flags["embb_power"])
                    any_safety_rewrite = bool(
                        mode_rewritten
                        or owner_rewritten
                        or packet_invalid_rewritten
                        or resolved[agent_id].used_greedy_fallback
                        or resolved[agent_id].collision_rewritten
                        or resolved[agent_id].joint_reliability_rewritten
                    )
                    mode_rewrite_total += int(mode_rewritten)
                    owner_rewrite_total += int(owner_rewritten)
                    packet_rewrite_total += int(packet_invalid_rewritten)
                    power_projection_total += int(power_projection_applied)
                    any_safety_rewrite_total += int(any_safety_rewrite)
                    shield_corrections_total += int(any_safety_rewrite)
                    mode_correction_count += int(bool(resolved[agent_id].mode_corrected))
                    packet_invalid_count += int(bool(resolved[agent_id].packet_invalid_fallback))
                    mask_invalid_count += int(bool(resolved[agent_id].mask_invalid_fallback))
                    joint_rewrite_count += int(bool(resolved[agent_id].joint_reliability_rewritten))
            executed_power_pre_tanh = _delta_to_pre_tanh(executed_power_delta)
            executed_embb_power_pre_tanh = _delta_to_pre_tanh(executed_embb_power_delta)
            executed_eval = self.model.evaluate_actions(
                local_obs=local_obs,
                global_obs=global_obs,
                mode_actions=torch.from_numpy(executed_mode_actions).to(self.device),
                packet_actions=torch.from_numpy(executed_packet_actions).to(self.device),
                power_pre_tanh=torch.from_numpy(executed_power_pre_tanh).reshape(-1, 1).to(self.device),
                embb_owner_actions=torch.from_numpy(executed_embb_owner_actions).to(self.device),
                embb_power_pre_tanh=torch.from_numpy(executed_embb_power_pre_tanh).reshape(-1, 1).to(self.device),
                mode_mask=mode_mask,
                packet_mask=packet_mask,
                embb_owner_mask=embb_owner_mask,
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
            )
            policy_forward_sec += perf_counter() - policy_forward_start

            env_step_start = perf_counter()
            next_obs, rewards_dict, dones_dict, _infos = self.env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            env_step_sec += perf_counter() - env_step_start
            ref_info = _infos[self.env.agent_ids[0]]
            if collect_detailed_metrics:
                for agent_id in self.env.agent_ids:
                    step_info = _infos.get(agent_id, {})
                    if not bool(step_info.get("planning_phase", False)):
                        continue
                    if len(owner_decode_debug_examples) >= 10:
                        continue
                if "raw_embb_owner_option" not in step_info:
                    continue
                snapshot_owner_id = int(step_info.get("snapshot_owner_id", -1))
                sampled_option = int(step_info.get("raw_embb_owner_option", -1))
                decoded_raw_owner_id = int(step_info.get("decoded_raw_owner_id", -1))
                valid_owner_mask: List[int] = []
                try:
                    owner_mask = np.asarray(observations[agent_id].masks.embb_owner_mask, dtype=float)
                    valid_owner_mask = [int(i) for i in np.where(owner_mask > 0.5)[0].tolist()]
                except Exception:
                    valid_owner_mask = []
                    owner_decode_debug_examples.append(
                        (snapshot_owner_id, sampled_option, decoded_raw_owner_id, valid_owner_mask)
                    )
            for key, value in ref_info.get('reward_terms', {}).items():
                reward_term_totals[key] = reward_term_totals.get(key, 0.0) + float(value)
            rewards_np = np.asarray([rewards_dict[agent_id] for agent_id in self.env.agent_ids], dtype=np.float32)
            dones_np = np.asarray([float(dones_dict[agent_id]) for agent_id in self.env.agent_ids], dtype=np.float32)

            buffer_add_start = perf_counter()
            self.buffer.add_step(
                local_obs=local_obs_np,
                global_obs=global_obs_np,
                mode_mask=mode_mask_np,
                packet_mask=packet_mask_np,
                embb_owner_mask=embb_owner_mask_np,
                mode_actions=executed_mode_actions,
                packet_actions=executed_packet_actions,
                power_pre_tanh=executed_power_pre_tanh,
                power_delta=executed_power_delta,
                embb_owner_actions=executed_embb_owner_actions,
                embb_power_pre_tanh=executed_embb_power_pre_tanh,
                embb_power_delta=executed_embb_power_delta,
                old_log_prob=executed_eval["log_prob"].detach().cpu().numpy(),
                values=executed_eval["value"].detach().cpu().numpy(),
                rewards=rewards_np,
                dones=dones_np,
                actor_hidden=prev_actor_hidden,
                critic_hidden=prev_critic_hidden,
                aux_best_mode_target=aux_best_mode_target,
                aux_overlay_feasible_target=aux_overlay_target,
                aux_best_packet_target=aux_best_packet_target,
                teacher_admission_target=teacher_admission_target,
                teacher_mode_target=teacher_mode_target,
                teacher_admission_weight=teacher_admission_weight,
                teacher_mode_weight=teacher_mode_weight,
                greedy_bc_mode_target=greedy_bc_mode_target,
                greedy_bc_packet_target=greedy_bc_packet_target,
                greedy_bc_owner_target=greedy_bc_owner_target,
                greedy_bc_mode_weight=greedy_bc_mode_weight,
                greedy_bc_packet_weight=greedy_bc_packet_weight,
                greedy_bc_owner_weight=greedy_bc_owner_weight,
                phase_a_embb_power_anchor_target=phase_a_embb_power_anchor_target,
                phase_a_embb_power_anchor_weight=phase_a_embb_power_anchor_weight,
                phase_a_mask=phase_a_mask,
            )
            buffer_add_sec += perf_counter() - buffer_add_start

            actor_hidden = output.actor_hidden.detach()
            critic_hidden = output.critic_hidden.detach()
            observations = next_obs
            done = all(dones_dict.values())
            steps += 1

            if done and steps < horizon:
                try:
                    episode_summary = dict(self.env.summarize_episode())
                    embb_total_rates.append(float(episode_summary.get("embb_total_rate", 0.0)))
                    episode_summaries.append(episode_summary)
                except Exception:
                    embb_total_rates.append(0.0)
                episodes_completed += 1
                episode_seed += 1
                observations, _info = self.env.reset(seed=episode_seed)
                actor_hidden, critic_hidden = self.model.initial_state(
                    batch_size=len(self.env.agent_ids),
                    device=self.device,
                )
                done = False

        if done:
            try:
                episode_summary = dict(self.env.summarize_episode())
                embb_total_rates.append(float(episode_summary.get("embb_total_rate", 0.0)))
                episode_summaries.append(episode_summary)
            except Exception:
                embb_total_rates.append(0.0)

        if done:
            last_values = np.zeros(len(self.env.agent_ids), dtype=np.float32)
        else:
            local_obs_np = np.stack([observations[agent_id].local_obs for agent_id in self.env.agent_ids]).astype(np.float32)
            global_obs_np = np.stack([observations[agent_id].global_obs for agent_id in self.env.agent_ids]).astype(np.float32)
            mode_mask_np = np.stack([observations[agent_id].masks.mode_mask for agent_id in self.env.agent_ids]).astype(np.float32)
            packet_mask_np = np.stack([observations[agent_id].masks.packet_mask for agent_id in self.env.agent_ids]).astype(np.float32)
            bootstrap_start = perf_counter()
            bootstrap = self.model.act(
                local_obs=torch.from_numpy(local_obs_np).to(self.device),
                global_obs=torch.from_numpy(global_obs_np).to(self.device),
                mode_mask=torch.from_numpy(mode_mask_np).to(self.device),
                packet_mask=torch.from_numpy(packet_mask_np).to(self.device),
                actor_hidden=actor_hidden,
                critic_hidden=critic_hidden,
                deterministic=True,
            )
            last_values = bootstrap.value.detach().cpu().numpy()
            policy_forward_sec += perf_counter() - bootstrap_start

        self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.cfg.training.gamma,
            gae_lambda=self.cfg.training.gae_lambda,
        )
        summary = self.buffer.summary()
        mode_actions = np.asarray(self.buffer.mode_actions, dtype=np.int64)
        advantages = np.asarray(self.buffer.advantages, dtype=np.float32)

        def _mode_adv(mode_id: int) -> float:
            mask = mode_actions == mode_id
            if not np.any(mask):
                return 0.0
            return float(np.mean(advantages[mask]))

        summary.update({
            'mean_mode_entropy': float(np.mean(mode_entropy_values)) if mode_entropy_values else 0.0,
            'mean_packet_entropy': float(np.mean(packet_entropy_values)) if packet_entropy_values else 0.0,
            'advantage_keep': _mode_adv(0),
            'advantage_overlay': _mode_adv(1),
            'advantage_puncture': _mode_adv(2),
            'action_ratio_keep': float(np.mean(mode_actions == 0)) if mode_actions.size > 0 else 0.0,
            'action_ratio_overlay': float(np.mean(mode_actions == 1)) if mode_actions.size > 0 else 0.0,
            'action_ratio_puncture': float(np.mean(mode_actions == 2)) if mode_actions.size > 0 else 0.0,
            'episodes_completed': float(episodes_completed + (1 if steps > 0 else 0)),
            'mean_embb_total_rate': float(np.mean(embb_total_rates)) if embb_total_rates else 0.0,
            'planning_phase_ratio': float(planning_agent_decisions / max(total_agent_decisions, 1)),
            'owner_head_active_ratio': float(owner_head_active_total / max(total_agent_decisions, 1)),
            'embb_power_head_active_ratio': float(embb_power_head_active_total / max(total_agent_decisions, 1)),
            'phase_a_embb_power_head_active_ratio': float(phase_a_embb_power_head_active_total / max(total_agent_decisions, 1)),
            'raw_owner_non_null_ratio': float(raw_owner_non_null_total / max(owner_head_active_total, 1)),
            'executed_owner_non_null_ratio': float(executed_owner_non_null_total / max(owner_head_active_total, 1)),
            'raw_embb_power_nonzero_ratio': float(raw_embb_power_nonzero_total / max(embb_power_head_active_total, 1)),
            'executed_embb_power_nonzero_ratio': float(executed_embb_power_nonzero_total / max(embb_power_head_active_total, 1)),
            'phase_a_raw_embb_power_nonzero_ratio': float(phase_a_raw_embb_power_nonzero_total / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_executed_embb_power_nonzero_ratio': float(phase_a_executed_embb_power_nonzero_total / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_raw_embb_power_mean_delta': float(phase_a_raw_embb_power_delta_sum / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_executed_embb_power_mean_delta': float(phase_a_executed_embb_power_delta_sum / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_embb_power_clip_ratio': float(phase_a_embb_power_clipped_total / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_embb_power_invalid_or_masked_ratio': float(phase_a_embb_power_invalid_or_masked_total / max(phase_a_embb_power_head_active_total, 1)),
            'phase_a_embb_power_anchor_binding_ratio': float(
                phase_a_embb_power_anchor_binding_count / max(phase_a_embb_power_anchor_binding_denom, 1)
            ),
            'owner_policy_entropy_mean': float(owner_policy_entropy_sum / max(owner_policy_stat_count, 1)),
            'owner_policy_top1_prob_mean': float(owner_policy_top1_prob_sum / max(owner_policy_stat_count, 1)),
            'owner_policy_snapshot_prob_mean': float(owner_policy_snapshot_prob_sum / max(owner_policy_stat_count, 1)),
            'owner_policy_non_snapshot_prob_mean': float(owner_policy_non_snapshot_prob_sum / max(owner_policy_stat_count, 1)),
            'sampled_embb_owner_option_mean': float(sampled_owner_option_sum / max(sampled_owner_option_total, 1)),
            'sampled_embb_owner_option_nonzero_ratio': float(
                sampled_owner_option_nonzero_count / max(sampled_owner_option_total, 1)
            ),
            'sampled_embb_owner_option_equals_snapshot_option_ratio': float(
                sampled_owner_option_equals_snapshot_count / max(sampled_owner_option_snapshot_comparable_count, 1)
            ),
            'sampled_embb_owner_option_valid_ratio': float(
                sampled_owner_option_valid_count / max(sampled_owner_option_total, 1)
            ),
            'rollout_sampled_owner_option_non_snapshot_ratio': float(
                sampled_owner_option_non_snapshot_count / max(sampled_owner_option_snapshot_comparable_count, 1)
            ),
            'rollout_sampled_owner_option_snapshot_ratio': float(
                sampled_owner_option_equals_snapshot_count / max(sampled_owner_option_snapshot_comparable_count, 1)
            ),
            'owner_decode_debug_examples': list(owner_decode_debug_examples[:10]),
            'owner_sampled_option_same_as_snapshot_ratio': float(
                ph0_owner_raw_same_as_snapshot_count / max(ph0_owner_snapshot_comparable_count, 1)
            ),
            'owner_sampled_option_non_snapshot_ratio': float(
                ph0_owner_raw_non_snapshot_count / max(ph0_owner_snapshot_comparable_count, 1)
            ),
            'owner_sampled_option_null_ratio': float(
                ph0_owner_raw_null_count / max(ph0_owner_snapshot_comparable_count, 1)
            ),
            'raw_executed_any_gap_ratio': float(raw_executed_any_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_mode_gap_ratio': float(raw_executed_mode_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_packet_gap_ratio': float(raw_executed_packet_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_power_gap_ratio': float(raw_executed_power_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_owner_gap_ratio': float(raw_executed_owner_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_embb_power_gap_ratio': float(raw_executed_embb_power_gap_total / max(total_agent_decisions, 1)),
            'raw_overlay_ratio': float(raw_overlay_count / max(phase_a_mode_decisions_total, 1)),
            'raw_puncture_ratio': float(raw_puncture_count / max(phase_a_mode_decisions_total, 1)),
            'executed_overlay_ratio': float(executed_overlay_count / max(phase_a_mode_decisions_total, 1)),
            'executed_puncture_ratio': float(executed_puncture_count / max(phase_a_mode_decisions_total, 1)),
            'shield_correction_ratio': float(shield_corrections_total / max(phase_a_mode_decisions_total, 1)),
            'mode_rewrite_ratio': float(mode_rewrite_total / max(phase_a_mode_decisions_total, 1)),
            'owner_rewrite_ratio': float(owner_rewrite_total / max(phase_a_mode_decisions_total, 1)),
            'packet_rewrite_ratio': float(packet_rewrite_total / max(phase_a_mode_decisions_total, 1)),
            'power_projection_ratio': float(power_projection_total / max(phase_a_mode_decisions_total, 1)),
            'any_safety_rewrite_ratio': float(any_safety_rewrite_total / max(phase_a_mode_decisions_total, 1)),
            'policy_autonomy_ratio': float(1.0 - any_safety_rewrite_total / max(phase_a_mode_decisions_total, 1)),
            'policy_autonomy_excluding_power_projection': float(
                1.0 - any_safety_rewrite_total / max(phase_a_mode_decisions_total, 1)
            ),
            'mode_correction_ratio': float(mode_correction_count / max(phase_a_mode_decisions_total, 1)),
            'packet_invalid_ratio': float(packet_invalid_count / max(phase_a_mode_decisions_total, 1)),
            'mask_invalid_ratio': float(mask_invalid_count / max(phase_a_mode_decisions_total, 1)),
            'joint_reliability_rewrite_ratio': float(joint_rewrite_count / max(phase_a_mode_decisions_total, 1)),
            'raw_vs_executed_mode_gap': float(
                abs(float(raw_overlay_count - executed_overlay_count) / max(phase_a_mode_decisions_total, 1))
                + abs(float(raw_puncture_count - executed_puncture_count) / max(phase_a_mode_decisions_total, 1))
            ),
            'obs_build_sec': float(obs_build_sec),
            'policy_forward_sec': float(policy_forward_sec),
            'greedy_bc_target_sec': float(greedy_bc_target_sec),
            'env_step_sec': float(env_step_sec),
            'buffer_add_sec': float(buffer_add_sec),
        })
        if episode_summaries:
            summary.update({
                'mean_phase0_owner_non_null_ratio_raw': float(
                    np.mean([item.get('phase0_owner_non_null_ratio_raw', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_non_null_ratio_executed': float(
                    np.mean([item.get('phase0_owner_non_null_ratio_executed', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_ratio_vs_snapshot_raw': float(
                    np.mean([item.get('phase0_owner_change_ratio_vs_snapshot_raw', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_ratio_vs_snapshot_executed': float(
                    np.mean([item.get('phase0_owner_change_ratio_vs_snapshot_executed', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_fallback_to_candidate0_ratio': float(
                    np.mean([item.get('phase0_owner_fallback_to_candidate0_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_invalid_option_ratio': float(
                    np.mean([item.get('phase0_owner_invalid_option_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_null_selected_ratio': float(
                    np.mean([item.get('phase0_owner_null_selected_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_invalid_to_null_ratio': float(
                    np.mean([item.get('phase0_owner_invalid_to_null_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_invalid_to_snapshot_ratio': float(
                    np.mean([item.get('phase0_owner_invalid_to_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_invalid_to_non_snapshot_ratio': float(
                    np.mean([item.get('phase0_owner_invalid_to_non_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_restored_to_snapshot_ratio': float(
                    np.mean([item.get('phase0_owner_restored_to_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_kept_null_ratio': float(
                    np.mean([item.get('phase0_owner_kept_null_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_replaced_with_non_snapshot_ratio': float(
                    np.mean([item.get('phase0_owner_replaced_with_non_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_raw_same_as_snapshot_ratio': float(
                    np.mean([item.get('ph0_owner_raw_same_as_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_raw_non_snapshot_ratio': float(
                    np.mean([item.get('ph0_owner_raw_non_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_raw_null_ratio': float(
                    np.mean([item.get('ph0_owner_raw_null_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_exec_same_as_snapshot_ratio': float(
                    np.mean([item.get('ph0_owner_exec_same_as_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_exec_non_snapshot_ratio': float(
                    np.mean([item.get('ph0_owner_exec_non_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_ph0_owner_reverted_to_snapshot_ratio': float(
                    np.mean([item.get('ph0_owner_reverted_to_snapshot_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_changed_and_effective_ratio': float(
                    np.mean([item.get('phase0_owner_changed_and_effective_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_budget_used': float(
                    np.mean([item.get('phase0_owner_change_budget_used', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_budget_allowed': float(
                    np.mean([item.get('phase0_owner_change_budget_allowed', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_budget_clipped_ratio': float(
                    np.mean([item.get('phase0_owner_change_budget_clipped_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_kept_topk_ratio': float(
                    np.mean([item.get('phase0_owner_change_kept_topk_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_change_dropped_over_budget_ratio': float(
                    np.mean([item.get('phase0_owner_change_dropped_over_budget_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase0_owner_effective_change_count': float(
                    np.mean([item.get('phase0_owner_effective_change_count', 0.0) for item in episode_summaries])
                ),
                'mean_embb_user_rate': float(np.mean([item.get('embb_user_rate_mean', 0.0) for item in episode_summaries])),
                'mean_embb_service_ratio': float(np.mean([item.get('embb_service_ratio', 0.0) for item in episode_summaries])),
                'mean_embb_positive_rate_ratio': float(np.mean([item.get('embb_positive_rate_ratio', item.get('embb_service_ratio', 0.0)) for item in episode_summaries])),
                'mean_urllc_admission_rate': float(np.mean([item.get('urllc_admission_rate', 0.0) for item in episode_summaries])),
                'mean_total_power': float(np.mean([item.get('total_power', 0.0) for item in episode_summaries])),
                'mean_throughput_per_watt': float(np.mean([item.get('throughput_per_watt', 0.0) for item in episode_summaries])),
                'mean_avg_throughput_per_served_embb_user': float(
                    np.mean([item.get('avg_throughput_per_served_embb_user', 0.0) for item in episode_summaries])
                ),
                'mean_phase_a_embb_power_zeroed_inactive_head_ratio': float(
                    np.mean([item.get('phase_a_embb_power_zeroed_inactive_head_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase_a_embb_power_zeroed_keep_mode_ratio': float(
                    np.mean([item.get('phase_a_embb_power_zeroed_keep_mode_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase_a_embb_power_zeroed_no_candidate_ratio': float(
                    np.mean([item.get('phase_a_embb_power_zeroed_no_candidate_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase_a_embb_power_zeroed_invalid_owner_ratio': float(
                    np.mean([item.get('phase_a_embb_power_zeroed_invalid_owner_ratio', 0.0) for item in episode_summaries])
                ),
                'mean_phase_a_embb_power_write_ratio': float(
                    np.mean([item.get('phase_a_embb_power_write_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_pre_clip_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_pre_clip_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_post_clip_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_post_clip_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_post_quant_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_post_quant_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_post_projection_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_post_projection_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_post_owner_validation_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_post_owner_validation_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_final_executed_mean_delta': float(
                    np.mean([item.get('phase_a_embb_power_final_executed_mean_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_clip_ratio': float(
                    np.mean([item.get('phase_a_embb_power_clip_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_quantized_ratio': float(
                    np.mean([item.get('phase_a_embb_power_quantized_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_projection_ratio': float(
                    np.mean([item.get('phase_a_embb_power_projection_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_owner_invalid_ratio': float(
                    np.mean([item.get('phase_a_embb_power_owner_invalid_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_no_candidate_ratio': float(
                    np.mean([item.get('phase_a_embb_power_no_candidate_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_keep_mode_zero_ratio': float(
                    np.mean([item.get('phase_a_embb_power_keep_mode_zero_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_cap_hit_ratio': float(
                    np.mean([item.get('phase_a_embb_power_cap_hit_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_floor_hit_ratio': float(
                    np.mean([item.get('phase_a_embb_power_floor_hit_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_sign_flip_ratio': float(
                    np.mean([item.get('phase_a_embb_power_sign_flip_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_abs_shrink_ratio': float(
                    np.mean([item.get('phase_a_embb_power_abs_shrink_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_mean_abs_raw_delta': float(
                    np.mean([item.get('phase_a_embb_power_mean_abs_raw_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_mean_abs_executed_delta': float(
                    np.mean([item.get('phase_a_embb_power_mean_abs_executed_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_projection_l2_mean': float(
                    np.mean([item.get('phase_a_embb_power_projection_l2_mean', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_pre_vs_final_l1_mean': float(
                    np.mean([item.get('phase_a_embb_power_pre_vs_final_l1_mean', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_pre_vs_final_sign_consistency': float(
                    np.mean([item.get('phase_a_embb_power_pre_vs_final_sign_consistency', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_effective_nonzero_ratio': float(
                    np.mean([item.get('phase_a_embb_power_effective_nonzero_ratio', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_floor_binding_strength': float(
                    np.mean([item.get('phase_a_embb_power_floor_binding_strength', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_cap_binding_strength': float(
                    np.mean([item.get('phase_a_embb_power_cap_binding_strength', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_proj_delta_l1': float(
                    np.mean([item.get('phase_a_embb_power_proj_delta_l1', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_proj_delta_l2': float(
                    np.mean([item.get('phase_a_embb_power_proj_delta_l2', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_pre_to_floor_delta': float(
                    np.mean([item.get('phase_a_embb_power_pre_to_floor_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_pre_to_cap_delta': float(
                    np.mean([item.get('phase_a_embb_power_pre_to_cap_delta', 0.0) for item in episode_summaries])
                ),
                'phase_a_embb_power_final_minus_proj_delta': float(
                    np.mean([item.get('phase_a_embb_power_final_minus_proj_delta', 0.0) for item in episode_summaries])
                ),
                'owner_dropped_raw_churn_ratio': float(
                    np.mean([item.get('owner_dropped_raw_churn_ratio', item.get('phase0_owner_change_dropped_over_budget_ratio', 0.0)) for item in episode_summaries])
                ),
                'owner_candidate_relaxed_ratio': float(
                    np.mean([item.get('owner_candidate_relaxed_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_candidate_fallback_used_ratio': float(
                    np.mean([item.get('owner_candidate_fallback_used_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_objective_gain_pre_filter_mean': float(
                    np.mean([item.get('owner_objective_gain_pre_filter_mean', 0.0) for item in episode_summaries])
                ),
                'owner_objective_gain_post_filter_mean': float(
                    np.mean([item.get('owner_objective_gain_post_filter_mean', 0.0) for item in episode_summaries])
                ),
                'owner_obj_mean': float(
                    np.mean([item.get('owner_obj_mean', 0.0) for item in episode_summaries])
                ),
                'owner_obj_std': float(
                    np.mean([item.get('owner_obj_std', 0.0) for item in episode_summaries])
                ),
                'owner_gate_threshold': float(
                    np.mean([item.get('owner_gate_threshold', 0.0) for item in episode_summaries])
                ),
                'owner_candidate_after_gate_ratio': float(
                    np.mean([item.get('owner_candidate_after_gate_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_positive_candidate_count_mean': float(
                    np.mean([item.get('owner_positive_candidate_count_mean', 0.0) for item in episode_summaries])
                ),
                'owner_neg_accept_clipped_ratio': float(
                    np.mean([item.get('owner_neg_accept_clipped_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_neg_rejected_by_quota_ratio': float(
                    np.mean([item.get('owner_neg_rejected_by_quota_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_neg_accept_ratio': float(
                    np.mean([item.get('owner_neg_accept_ratio', item.get('owner_negative_but_accepted_ratio', 0.0)) for item in episode_summaries])
                ),
                'owner_pos_selected_ratio': float(
                    np.mean([item.get('owner_pos_selected_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_neg_selected_count': float(
                    np.mean([item.get('owner_neg_selected_count', 0.0) for item in episode_summaries])
                ),
                'owner_pos_selected_count': float(
                    np.mean([item.get('owner_pos_selected_count', 0.0) for item in episode_summaries])
                ),
                'owner_selected_positive_count_mean': float(
                    np.mean([item.get('owner_selected_positive_count_mean', item.get('owner_pos_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_selected_negative_count_mean': float(
                    np.mean([item.get('owner_selected_negative_count_mean', item.get('owner_neg_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_selected_count': float(
                    np.mean([item.get('owner_selected_count', 0.0) for item in episode_summaries])
                ),
                'owner_final_selected_count': float(
                    np.mean([item.get('owner_final_selected_count', item.get('owner_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_final_pos_selected_count': float(
                    np.mean([item.get('owner_final_pos_selected_count', item.get('owner_pos_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_final_neg_selected_count': float(
                    np.mean([item.get('owner_final_neg_selected_count', item.get('owner_neg_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_final_safe_relax_selected_count': float(
                    np.mean([item.get('owner_final_safe_relax_selected_count', item.get('owner_safe_relax_selected_count_mean', 0.0)) for item in episode_summaries])
                ),
                'owner_final_keep_set_size': float(
                    np.mean([item.get('owner_final_keep_set_size', item.get('owner_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_allowed_k': float(
                    np.mean([item.get('owner_allowed_k', 0.0) for item in episode_summaries])
                ),
                'owner_selection_fill_ratio': float(
                    np.mean([item.get('owner_selection_fill_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_positive_shortage_ratio': float(
                    np.mean([item.get('owner_positive_shortage_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_negative_blocked_due_to_quota_ratio': float(
                    np.mean([item.get('owner_negative_blocked_due_to_quota_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relaxed_used_ratio': float(
                    np.mean([item.get('owner_safe_relaxed_used_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relaxed_candidate_count': float(
                    np.mean([item.get('owner_safe_relaxed_candidate_count', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relaxed_selected_count': float(
                    np.mean([item.get('owner_safe_relaxed_selected_count', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relax_selected_count_mean': float(
                    np.mean([item.get('owner_safe_relax_selected_count_mean', item.get('owner_safe_relaxed_selected_count', 0.0)) for item in episode_summaries])
                ),
                'owner_safe_relaxed_avg_objective': float(
                    np.mean([item.get('owner_safe_relaxed_avg_objective', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relaxed_service_delta_mean': float(
                    np.mean([item.get('owner_safe_relaxed_service_delta_mean', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relaxed_intercell_delta_mean': float(
                    np.mean([item.get('owner_safe_relaxed_intercell_delta_mean', 0.0) for item in episode_summaries])
                ),
                'owner_near_zero_objective_ratio': float(
                    np.mean([item.get('owner_near_zero_objective_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_positive_after_relax_ratio': float(
                    np.mean([item.get('owner_positive_after_relax_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relax_disabled_ratio': float(
                    np.mean([item.get('owner_safe_relax_disabled_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_safe_relax_off_ratio': float(
                    np.mean([item.get('owner_safe_relax_off_ratio', item.get('owner_safe_relax_disabled_ratio', 0.0)) for item in episode_summaries])
                ),
                'owner_negative_but_accepted_ratio': float(
                    np.mean([item.get('owner_negative_but_accepted_ratio', 0.0) for item in episode_summaries])
                ),
                'owner_neg_accepted_with_positive_candidate_ratio': float(
                    np.mean([item.get('owner_neg_accepted_with_positive_candidate_ratio', 0.0) for item in episode_summaries])
                ),
                'phaseA_positive_ratio': float(
                    np.mean([item.get('phaseA_positive_ratio', 0.0) for item in episode_summaries])
                ),
                'phaseA_zero_action_ratio': float(
                    np.mean([item.get('phaseA_zero_action_ratio', 0.0) for item in episode_summaries])
                ),
                'phaseA_power_reduction_l2_penalty': float(
                    np.mean([item.get('phaseA_power_reduction_l2_penalty', 0.0) for item in episode_summaries])
                ),
                'phaseA_power_saturation_penalty': float(
                    np.mean([item.get('phaseA_power_saturation_penalty', 0.0) for item in episode_summaries])
                ),
                'embb_service_floor_hinge_penalty': float(
                    np.mean([item.get('embb_service_floor_hinge_penalty', 0.0) for item in episode_summaries])
                ),
                'phaseA_delta_lt_neg09_ratio': float(
                    np.mean([item.get('phaseA_delta_lt_neg09_ratio', 0.0) for item in episode_summaries])
                ),
                'phaseA_delta_mean': float(
                    np.mean([item.get('phaseA_delta_mean', 0.0) for item in episode_summaries])
                ),
                'phaseA_delta_p10': float(
                    np.mean([item.get('phaseA_delta_p10', 0.0) for item in episode_summaries])
                ),
                'phaseA_delta_p50': float(
                    np.mean([item.get('phaseA_delta_p50', 0.0) for item in episode_summaries])
                ),
                'phaseA_delta_p90': float(
                    np.mean([item.get('phaseA_delta_p90', 0.0) for item in episode_summaries])
                ),
                'phase0_owner_candidate_positive_objective_ratio': float(
                    np.mean([item.get('phase0_owner_candidate_positive_objective_ratio', 0.0) for item in episode_summaries])
                ),
                'phase0_owner_accepted_positive_objective_ratio': float(
                    np.mean([item.get('phase0_owner_accepted_positive_objective_ratio', 0.0) for item in episode_summaries])
                ),
                'phaseA_executed_abs_delta_mean': float(
                    np.mean([item.get('phaseA_executed_abs_delta_mean', item.get('phase_a_embb_power_mean_abs_executed_delta', 0.0)) for item in episode_summaries])
                ),
                'reward_term_terminal_owner_raw_churn_penalty': float(
                    np.mean([item.get('reward_term_terminal_owner_raw_churn_penalty', 0.0) for item in episode_summaries])
                ),
                'reward_term_terminal_phase_a_effective_nonzero_floor_penalty': float(
                    np.mean([item.get('reward_term_terminal_phase_a_effective_nonzero_floor_penalty', 0.0) for item in episode_summaries])
                ),
                'terminal_debug_embb_total_rate': float(
                    np.mean([item.get('terminal_debug_embb_total_rate', 0.0) for item in episode_summaries])
                ),
                'terminal_debug_greedy_embb_total_rate': float(
                    np.mean([item.get('terminal_debug_greedy_embb_total_rate', 0.0) for item in episode_summaries])
                ),
                'terminal_debug_embb_rate_gain_vs_greedy': float(
                    np.mean([item.get('terminal_debug_embb_rate_gain_vs_greedy', 0.0) for item in episode_summaries])
                ),
                'terminal_debug_embb_rate_gain_weight': float(
                    np.mean([item.get('terminal_debug_embb_rate_gain_weight', 0.0) for item in episode_summaries])
                ),
            })
        if steps > 0:
            for key, value in reward_term_totals.items():
                summary[f'reward_term_{key}'] = float(value / steps)
        if hasattr(self.env, "timing_counters"):
            for key, value in dict(self.env.timing_counters()).items():
                summary[f'env_{key}'] = float(value)
        return summary

    def update(
        self,
        ppo_epochs: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        teacher_scale: float = 1.0,
        distill_coef: float = 0.0,
        greedy_bc_coef: float = 0.0,
        actual_load: Optional[float] = None,
        entropy_coef: Optional[float] = None,
        clip_ratio: Optional[float] = None,
    ) -> TrainerStats:
        ppo_epochs = ppo_epochs or self.cfg.training.ppo_epochs
        minibatch_size = minibatch_size or self.cfg.training.minibatch_size
        teacher_scale = float(np.clip(teacher_scale, 0.0, 1.0))
        distill_coef = float(max(distill_coef, 0.0))
        greedy_bc_coef = float(max(greedy_bc_coef, 0.0))
        actual_load = float(self.env._current_actual_load() if actual_load is None else actual_load)
        entropy_coef = float(self.cfg.training.entropy_coef if entropy_coef is None else entropy_coef)
        clip_ratio = float(self.cfg.training.clip_ratio if clip_ratio is None else clip_ratio)
        batch = self.buffer.as_torch(self.device)
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_best_mode_aux = 0.0
        total_overlay_aux = 0.0
        total_best_packet_aux = 0.0
        total_phase_a_embb_power_anchor_loss = 0.0
        update_steps = 0
        skipped_updates = 0

        num_samples = batch.local_obs.shape[0]
        indices = np.arange(num_samples)
        for _epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_actor_hidden = batch.actor_hidden[mb_idx].unsqueeze(0)
                mb_critic_hidden = batch.critic_hidden[mb_idx].unsqueeze(0)
                outputs = self.model.evaluate_actions(
                    local_obs=batch.local_obs[mb_idx],
                    global_obs=batch.global_obs[mb_idx],
                    mode_actions=batch.mode_actions[mb_idx],
                    packet_actions=batch.packet_actions[mb_idx],
                    power_pre_tanh=batch.power_pre_tanh[mb_idx],
                    embb_owner_actions=batch.embb_owner_actions[mb_idx],
                    embb_power_pre_tanh=batch.embb_power_pre_tanh[mb_idx],
                    mode_mask=batch.mode_mask[mb_idx],
                    packet_mask=batch.packet_mask[mb_idx],
                    embb_owner_mask=batch.embb_owner_mask[mb_idx],
                    actor_hidden=mb_actor_hidden,
                    critic_hidden=mb_critic_hidden,
                )

                log_ratio = outputs['log_prob'] - batch.old_log_prob[mb_idx]
                ratio = torch.exp(log_ratio)
                mb_advantages = advantages[mb_idx]
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = torch.mean((outputs['value'] - batch.returns[mb_idx]) ** 2)
                entropy = outputs['entropy'].mean()
                best_mode_aux_loss = F.cross_entropy(outputs['best_mode_logits'], batch.aux_best_mode_target[mb_idx])
                overlay_aux_loss = F.binary_cross_entropy_with_logits(
                    outputs['overlay_feasible_logit'],
                    batch.aux_overlay_feasible_target[mb_idx],
                )
                aux_packet_logits = self.model.compute_packet_logits(
                    outputs['actor_latent'],
                    batch.aux_best_mode_target[mb_idx],
                    batch.packet_mask[mb_idx],
                )
                best_packet_aux_loss = F.cross_entropy(aux_packet_logits, batch.aux_best_packet_target[mb_idx])
                mode_logits = outputs['mode_logits']
                admit_logits = torch.stack(
                    [
                        mode_logits[:, MODE_KEEP],
                        torch.logsumexp(mode_logits[:, [MODE_OVERLAY, MODE_PUNCTURE]], dim=-1),
                    ],
                    dim=-1,
                )
                admission_distill_loss_raw = F.cross_entropy(
                    admit_logits,
                    batch.teacher_admission_target[mb_idx],
                    reduction='none',
                )
                admission_weights = batch.teacher_admission_weight[mb_idx]
                if torch.sum(admission_weights) > 1.0e-9:
                    admission_distill_loss = torch.sum(admission_distill_loss_raw * admission_weights) / torch.sum(admission_weights)
                else:
                    admission_distill_loss = torch.zeros((), device=self.device)

                mode_distill_logits = mode_logits[:, [MODE_OVERLAY, MODE_PUNCTURE]]
                mode_distill_loss_raw = F.cross_entropy(
                    mode_distill_logits,
                    batch.teacher_mode_target[mb_idx],
                    reduction='none',
                )
                mode_weights = batch.teacher_mode_weight[mb_idx]
                if torch.sum(mode_weights) > 1.0e-9:
                    mode_distill_loss = torch.sum(mode_distill_loss_raw * mode_weights) / torch.sum(mode_weights)
                else:
                    mode_distill_loss = torch.zeros((), device=self.device)

                greedy_packet_logits = self.model.compute_packet_logits(
                    outputs['actor_latent'],
                    batch.greedy_bc_mode_target[mb_idx],
                    batch.packet_mask[mb_idx],
                )
                greedy_owner_logits = self.model.compute_embb_owner_logits(
                    outputs['actor_latent'],
                    batch.embb_owner_mask[mb_idx],
                )
                greedy_mode_bc_raw = F.cross_entropy(
                    mode_logits,
                    batch.greedy_bc_mode_target[mb_idx],
                    reduction='none',
                )
                greedy_packet_bc_raw = F.cross_entropy(
                    greedy_packet_logits,
                    batch.greedy_bc_packet_target[mb_idx],
                    reduction='none',
                )
                greedy_owner_bc_raw = F.cross_entropy(
                    greedy_owner_logits,
                    batch.greedy_bc_owner_target[mb_idx],
                    reduction='none',
                )
                greedy_mode_weights = batch.greedy_bc_mode_weight[mb_idx]
                greedy_packet_weights = batch.greedy_bc_packet_weight[mb_idx]
                greedy_owner_weights = batch.greedy_bc_owner_weight[mb_idx]
                if torch.sum(greedy_mode_weights) > 1.0e-9:
                    greedy_mode_bc_loss = torch.sum(greedy_mode_bc_raw * greedy_mode_weights) / torch.sum(greedy_mode_weights)
                else:
                    greedy_mode_bc_loss = torch.zeros((), device=self.device)
                if torch.sum(greedy_packet_weights) > 1.0e-9:
                    greedy_packet_bc_loss = torch.sum(greedy_packet_bc_raw * greedy_packet_weights) / torch.sum(greedy_packet_weights)
                else:
                    greedy_packet_bc_loss = torch.zeros((), device=self.device)
                if torch.sum(greedy_owner_weights) > 1.0e-9:
                    greedy_owner_bc_loss = torch.sum(greedy_owner_bc_raw * greedy_owner_weights) / torch.sum(greedy_owner_weights)
                else:
                    greedy_owner_bc_loss = torch.zeros((), device=self.device)

                frontier_anchor_loss = torch.zeros((), device=self.device)
                if bool(getattr(self.cfg.training, "use_frontier_mode_anchor", False)):
                    puncture_floor = _value_for_load(
                        getattr(self.cfg.training, "frontier_puncture_floor_by_load", {}),
                        actual_load,
                        default=0.0,
                    )
                    overlay_ceiling = _value_for_load(
                        getattr(self.cfg.training, "frontier_overlay_ceiling_by_load", {}),
                        actual_load,
                        default=1.0,
                    )
                    phase_a_active = batch.phase_a_mask[mb_idx] > 0.5
                    if torch.any(phase_a_active):
                        phase_mode_logits = mode_logits[phase_a_active]
                        phase_mode_probs = torch.softmax(phase_mode_logits, dim=-1)
                        admit_mass = phase_mode_probs[:, MODE_OVERLAY] + phase_mode_probs[:, MODE_PUNCTURE]
                        admit_mass_sum = torch.sum(admit_mass)
                        if admit_mass_sum > 1.0e-9:
                            overlay_ratio = torch.sum(phase_mode_probs[:, MODE_OVERLAY]) / admit_mass_sum
                            puncture_ratio = torch.sum(phase_mode_probs[:, MODE_PUNCTURE]) / admit_mass_sum
                            puncture_shortfall = torch.clamp(torch.tensor(puncture_floor, device=self.device) - puncture_ratio, min=0.0)
                            overlay_excess = torch.clamp(overlay_ratio - torch.tensor(overlay_ceiling, device=self.device), min=0.0)
                            frontier_anchor_loss = float(
                                getattr(self.cfg.training, "frontier_mode_anchor_weight", 0.0) or 0.0
                            ) * (puncture_shortfall + overlay_excess)

                phase_a_embb_power_anchor_loss = torch.zeros((), device=self.device)
                if bool(getattr(self.cfg.training, "use_phase_a_embb_power_anchor", False)):
                    anchor_weights = batch.phase_a_embb_power_anchor_weight[mb_idx]
                    if torch.sum(anchor_weights) > 1.0e-9:
                        anchor_targets = batch.phase_a_embb_power_anchor_target[mb_idx]
                        anchor_prediction = outputs["embb_power_delta_mean"]
                        anchor_raw = F.smooth_l1_loss(
                            anchor_prediction,
                            anchor_targets,
                            reduction='none',
                        )
                        phase_a_embb_power_anchor_loss = (
                            torch.sum(anchor_raw * anchor_weights) / torch.sum(anchor_weights)
                        )

                phase_a_embb_power_saturation_reg = torch.zeros((), device=self.device)
                pre_tanh_l1_w = float(getattr(self.cfg.training, "phase_a_embb_power_pre_tanh_l1_reg_weight", 0.0) or 0.0)
                tail_w = float(getattr(self.cfg.training, "phase_a_embb_power_tanh_tail_reg_weight", 0.0) or 0.0)
                tail_thr = float(getattr(self.cfg.training, "phase_a_embb_power_tanh_tail_threshold", 0.6) or 0.6)
                if (pre_tanh_l1_w > 0.0 or tail_w > 0.0) and batch.phase_a_mask.numel() > 0:
                    phase_a_active = batch.phase_a_mask[mb_idx] > 0.5
                    if torch.any(phase_a_active):
                        # IMPORTANT: regularize the *current policy mean* (not the stored executed action),
                        # so the penalty has gradient and can actually reduce Phase-A saturation.
                        pre = outputs["embb_power_mean"][phase_a_active]
                        if pre_tanh_l1_w > 0.0:
                            phase_a_embb_power_saturation_reg = phase_a_embb_power_saturation_reg + float(pre_tanh_l1_w) * torch.mean(torch.abs(pre))
                        if tail_w > 0.0:
                            tanh_out = torch.tanh(pre)
                            tail = torch.clamp(torch.abs(tanh_out) - float(tail_thr), min=0.0)
                            phase_a_embb_power_saturation_reg = phase_a_embb_power_saturation_reg + float(tail_w) * torch.mean(tail * tail)

                loss = (
                    policy_loss
                    + self.cfg.training.value_coef * value_loss
                    - entropy_coef * entropy
                    + (teacher_scale * self.cfg.training.aux_best_mode_coef) * best_mode_aux_loss
                    + self.cfg.training.aux_overlay_feasible_coef * overlay_aux_loss
                    + (teacher_scale * self.cfg.training.aux_best_packet_coef) * best_packet_aux_loss
                    + distill_coef
                    * (
                        float(getattr(self.cfg.training, "teacher_admission_loss_weight", 0.0) or 0.0) * admission_distill_loss
                        + float(getattr(self.cfg.training, "teacher_mode_loss_weight", 0.0) or 0.0) * mode_distill_loss
                    )
                    + greedy_bc_coef
                    * (
                        float(getattr(self.cfg.training, "greedy_bc_mode_weight", 0.0) or 0.0) * greedy_mode_bc_loss
                        + float(getattr(self.cfg.training, "greedy_bc_packet_weight", 0.0) or 0.0) * greedy_packet_bc_loss
                        + float(getattr(self.cfg.training, "greedy_bc_owner_weight", 0.0) or 0.0) * greedy_owner_bc_loss
                    )
                    + float(getattr(self.cfg.training, "phase_a_embb_power_anchor_weight", 0.0) or 0.0)
                    * phase_a_embb_power_anchor_loss
                    + frontier_anchor_loss
                    + phase_a_embb_power_saturation_reg
                )
                critical_tensors = (
                    outputs['log_prob'],
                    outputs['entropy'],
                    outputs['value'],
                    outputs['best_mode_logits'],
                    outputs['overlay_feasible_logit'],
                    aux_packet_logits,
                    mode_logits,
                    admit_logits,
                    admission_distill_loss,
                    mode_distill_loss,
                    greedy_packet_logits,
                    greedy_owner_logits,
                    greedy_mode_bc_loss,
                    greedy_packet_bc_loss,
                    greedy_owner_bc_loss,
                    phase_a_embb_power_anchor_loss,
                    frontier_anchor_loss,
                    loss,
                )
                if not all(torch.isfinite(tensor).all() for tensor in critical_tensors):
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_updates += 1
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                grads_finite = True
                for param in self.model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grads_finite = False
                        break
                if not grads_finite:
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_updates += 1
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                self.optimizer.step()
                params_finite = all(torch.isfinite(param).all() for param in self.model.parameters())
                if not params_finite:
                    raise FloatingPointError("Non-finite model parameters detected after optimizer step.")

                with torch.no_grad():
                    approx_kl = torch.mean(batch.old_log_prob[mb_idx] - outputs['log_prob']).abs().item()
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += float(approx_kl)
                total_best_mode_aux += float(best_mode_aux_loss.item())
                total_overlay_aux += float(overlay_aux_loss.item())
                total_best_packet_aux += float(best_packet_aux_loss.item())
                total_phase_a_embb_power_anchor_loss += float(phase_a_embb_power_anchor_loss.item())
                update_steps += 1

        rollout_summary = self.buffer.summary()
        denom = max(update_steps, 1)
        return TrainerStats(
            rollout_steps=rollout_summary['num_steps'],
            mean_reward=rollout_summary['mean_reward'],
            policy_loss=total_policy_loss / denom,
            value_loss=total_value_loss / denom,
            entropy=total_entropy / denom,
            approx_kl=total_kl / denom,
            best_mode_aux_loss=total_best_mode_aux / denom,
            overlay_aux_loss=total_overlay_aux / denom,
            best_packet_aux_loss=total_best_packet_aux / denom,
            phase_a_embb_power_anchor_loss=total_phase_a_embb_power_anchor_loss / denom,
        )

    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': asdict(self.cfg),
            'extra': extra or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> Dict:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload['model_state_dict'])
        optimizer_state = payload.get('optimizer_state_dict')
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except (ValueError, RuntimeError):
                pass
        extra = payload.get('extra', {})
        runtime_enabled = bool(
            extra.get(
                'phase_a_embb_power_runtime_enabled',
                getattr(self.cfg.env, "allow_phase_a_embb_power_adjustment", False),
            )
        )
        set_phase_a_embb_power_runtime(self.env, self.model, runtime_enabled)
        return extra


def build_default_components(cfg):
    from .compare import _build_main_like_configs
    from .env import SRMAPPOPhaseAEnv
    from .networks import SRMAPPOActorCritic

    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()

    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)
    # Enable training-only admission-guard jitter to reduce threshold overfitting.
    env.admission_guard_training_jitter_enabled = True
    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, cfg)
    set_phase_a_embb_power_runtime(env, model, bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)))
    trainer = SRMAPPOTrainer(env, model, cfg)
    return env, model, trainer


def set_training_fallback_warmup(env, cfg, iteration: int) -> None:
    warmup_iters = int(getattr(cfg.training, "greedy_fallback_warmup_iterations", 0) or 0)
    enabled = int(iteration) <= max(warmup_iters, 0)
    env.rl_cfg.shield.enable_greedy_fallback = bool(enabled)
    env.rl_cfg.shield.allow_mode_correction = bool(enabled)


def disable_eval_fallback(env) -> tuple[bool, bool]:
    previous = (
        bool(getattr(env.rl_cfg.shield, "enable_greedy_fallback", False)),
        bool(getattr(env.rl_cfg.shield, "allow_mode_correction", False)),
    )
    env.rl_cfg.shield.enable_greedy_fallback = False
    env.rl_cfg.shield.allow_mode_correction = False
    return previous


def restore_eval_fallback(env, previous: tuple[bool, bool]) -> None:
    env.rl_cfg.shield.enable_greedy_fallback = bool(previous[0])
    env.rl_cfg.shield.allow_mode_correction = bool(previous[1])


def run_training_loop(cfg, evaluation_fn=None, resume_path: Optional[Path] = None):
    env, model, trainer = build_default_components(cfg)
    history = []
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_every = 20
    selection_mode = str(getattr(cfg.training, "selection_mode", "dual_metric") or "dual_metric").strip().lower()
    selection_admission_floor = float(getattr(cfg.training, "selection_admission_floor", 0.0) or 0.0)
    load_aware_objective = bool(getattr(cfg.training, "load_aware_objective", False))
    low_damage_objective = bool(getattr(cfg.training, "low_damage_admission_objective", False))
    start_iteration = 1
    resume_extra = {}
    dead_phase_a_eval_streak = 0
    frozen_owner_eval_streak = 0
    owner_restore_collapse_eval_streak = 0
    if resume_path is not None:
        resume_extra = trainer.load_checkpoint(resume_path)
        start_iteration = int(resume_extra.get('iteration', 0)) + 1
        history = list(resume_extra.get('history', []) or [])

    summary = {
        "device_requested": str(getattr(trainer, "device_requested", getattr(cfg.training, "device", "auto"))),
        "device_resolved": str(getattr(trainer, "device_resolved", getattr(cfg.training, "device", "cpu"))),
        "cuda_available": bool(torch.cuda.is_available()),
        "phase0_owner_change_budget_mode": str(getattr(cfg.env, "phase0_owner_change_budget_mode", "unknown")),
        "allow_phase_a_embb_power_adjustment": bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)),
        "urllc_poisson_rate": float(getattr(env.sim_cfg, "urllc_poisson_rate", 0.0)),
        "fixed_urllc_poisson_rate": bool(getattr(env.sim_cfg, "fixed_urllc_poisson_rate", False)),
        "phase_a_embb_power_runtime_enabled": bool(getattr(env, "_phase_a_embb_power_runtime_enabled", lambda: False)()),
        "use_phase_a_embb_power_anchor": bool(getattr(cfg.training, "use_phase_a_embb_power_anchor", False)),
        "owner_snapshot_in_observation": bool(
            bool(getattr(cfg.env, "owner_snapshot_in_observation", True))
            and (
                bool(getattr(cfg.env, "include_greedy_reference_in_obs", False))
                or bool(getattr(cfg.training, "use_greedy_reference_bc", False))
                or str(getattr(cfg.training, "bc_teacher_policy", "") or "").strip().lower() == "greedy_reference"
            )
        ),
        "owner_snapshot_used_for_init": bool(getattr(cfg.env, "owner_snapshot_used_for_init", True)),
        "owner_snapshot_used_for_fallback": bool(getattr(cfg.env, "owner_snapshot_used_for_fallback", True)),
        "owner_snapshot_used_for_reward": bool(getattr(cfg.env, "owner_snapshot_used_for_reward", True)),
        "enable_feasibility_shield": bool(getattr(cfg.shield, "enable_feasibility_shield", False)),
        "apply_joint_reliability_rewrite": bool(getattr(cfg.shield, "apply_joint_reliability_rewrite", False)),
        "enable_greedy_fallback": bool(getattr(cfg.shield, "enable_greedy_fallback", False)),
        "use_greedy_reference_bc": bool(getattr(cfg.training, "use_greedy_reference_bc", False)),
        "teacher_distill_coef": float(teacher_distill_coef(cfg, start_iteration)),
        "use_teacher_distillation": bool(getattr(cfg.training, "use_teacher_distillation", False)),
    }
    print("SR-MAPPO effective config summary (startup):")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if str(getattr(cfg.training, "experiment_line", "") or "").strip().lower() == "phase0_joint_full_power_recovery_no_snapshot_debug":
        if (
            bool(summary.get("owner_snapshot_in_observation", False))
            or bool(summary.get("owner_snapshot_used_for_init", False))
            or bool(summary.get("owner_snapshot_used_for_fallback", False))
            or bool(summary.get("owner_snapshot_used_for_reward", False))
        ):
            print(
                "[FATAL] Snapshot leakage is still present in MAPPO owner pipeline: "
                f"owner_snapshot_in_observation={bool(summary.get('owner_snapshot_in_observation'))}, "
                f"owner_snapshot_used_for_init={bool(summary.get('owner_snapshot_used_for_init'))}, "
                f"owner_snapshot_used_for_fallback={bool(summary.get('owner_snapshot_used_for_fallback'))}, "
                f"owner_snapshot_used_for_reward={bool(summary.get('owner_snapshot_used_for_reward'))}",
                flush=True,
            )
            raise RuntimeError("Snapshot leakage is still present in MAPPO owner pipeline.")

    def _checkpoint_extra(
        iteration: int,
        record: Optional[Dict] = None,
        evaluation: Optional[Dict] = None,
        selection_metric: Optional[str] = None,
    ) -> Dict:
        snapshot_history = list(history)
        if record is not None:
            snapshot_history.append(record)
        payload = {
            'iteration': iteration,
            'bc': bc_stats,
            'history': snapshot_history,
        }
        if record is not None:
            payload['record'] = record
            if any(key in record for key in ('evaluation_kind', 'evaluation_loads', 'evaluation_episodes_per_load', 'evaluation_compare_modes')):
                payload['evaluation_config'] = {
                    'kind': str(record.get('evaluation_kind', 'none')),
                    'loads': [float(load) for load in (record.get('evaluation_loads', []) or [])],
                    'episodes_per_load': int(record.get('evaluation_episodes_per_load', 0) or 0),
                    'compare_modes': list(record.get('evaluation_compare_modes', []) or []),
                }
            payload['phase_a_embb_power_runtime_enabled'] = bool(
                ((record.get('control') or {}).get('phase_a_embb_power_runtime_enabled', False))
            )
            power_metric_source = (
                evaluation
                or record.get('checkpoint_evaluation')
                or record.get('evaluation')
                or record.get('rollout')
                or {}
            )
            payload['phase_a_embb_power_changed_count'] = _summary_float(
                power_metric_source,
                'policy_mean_phase_a_embb_power_changed_count',
                'phase_a_embb_power_changed_count',
                default=0.0,
            )
            payload['phase_a_embb_power_changed_ratio'] = _summary_float(
                power_metric_source,
                'policy_mean_phase_a_embb_power_changed_ratio',
                'phase_a_embb_power_changed_ratio',
                default=0.0,
            )
            payload['phase_a_embb_power_mean_raw_delta'] = _summary_float(
                power_metric_source,
                'policy_mean_phase_a_embb_power_mean_raw_delta',
                'phase_a_embb_power_mean_raw_delta',
                default=0.0,
            )
            payload['phase_a_embb_power_mean_executed_delta'] = _summary_float(
                power_metric_source,
                'policy_mean_phase_a_embb_power_mean_executed_delta',
                'phase_a_embb_power_mean_executed_delta',
                default=0.0,
            )
            payload['phase_a_embb_power_exercised'] = bool(
                payload['phase_a_embb_power_changed_count'] > 1.0e-9
                or abs(payload['phase_a_embb_power_mean_executed_delta']) > 1.0e-9
            )
        if evaluation is not None:
            payload['evaluation'] = evaluation
        if selection_metric is not None:
            payload['selection_metric'] = selection_metric
        return payload

    def _bc_before_reset(target_env, episode_idx: int):
        bc_loads = list(getattr(cfg.training, 'bc_loads', []))
        if bc_loads:
            configure_env_for_users_per_uav(target_env, bc_loads[episode_idx % len(bc_loads)])

    if resume_path is None and cfg.training.bc_episodes > 0 and cfg.training.bc_epochs > 0:
        bc_dataset = collect_greedy_bc_dataset(
            env,
            episodes=cfg.training.bc_episodes,
            seed=cfg.training.train_seed,
            before_reset=_bc_before_reset,
            teacher_policy=cfg.training.bc_teacher_policy,
        )
        bc_trainer = GreedyWarmStartTrainer(model, device=cfg.training.device)
        bc_stats = bc_trainer.fit(
            bc_dataset,
            epochs=cfg.training.bc_epochs,
            batch_size=cfg.training.bc_batch_size,
            learning_rate=cfg.training.bc_learning_rate,
        )
        bc_stats["bc_teacher_policy"] = str(cfg.training.bc_teacher_policy)
    else:
        bc_stats = {
            'bc_samples': 0.0,
            'bc_epochs': 0.0,
            'bc_loss': 0.0,
            'bc_teacher_policy': str(cfg.training.bc_teacher_policy),
        }

    best_reward_score = float('-inf')
    best_throughput_score = float('-inf')
    best_balanced_score = float('-inf')
    best_balanced_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_vs_original_score = float('-inf')
    best_vs_matched_score = float('-inf')
    best_vs_throughput_feasible_score = float('-inf')
    best_vs_throughput_only_score = float('-inf')
    best_vs_channel_only_score = float('-inf')
    best_floor_throughput_score = float('-inf')
    best_multiload_frontier_score = float('-inf')
    best_floor_fallback_key = (float("-inf"), float("-inf"))
    best_summary = None
    best_reward_summary = None
    best_throughput_summary = None
    best_balanced_summary = None
    best_service_interference_balanced_score = float("-inf")
    best_service_interference_balanced_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_service_interference_balanced_summary = None
    best_service_power_interference_balanced_score = float("-inf")
    best_service_power_interference_balanced_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_service_power_interference_balanced_summary = None
    best_service_gain_interference_balanced_score = float("-inf")
    best_service_gain_interference_balanced_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_service_gain_interference_balanced_summary = None
    best_balanced_intercell_aware_score = float("-inf")
    best_balanced_intercell_aware_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_balanced_intercell_aware_summary = None
    best_admission_service_intercell_score = float("-inf")
    best_admission_service_intercell_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_admission_service_intercell_summary = None
    best_owner_frozen_action_intercell_balanced_score = float("-inf")
    best_owner_frozen_action_intercell_balanced_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_owner_frozen_action_intercell_balanced_summary = None
    best_v5_balanced_intercell_admission_score = float("-inf")
    best_v5_balanced_intercell_admission_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_v5_balanced_intercell_admission_summary = None
    best_v6_balanced_puncture_accounting_score = float("-inf")
    best_v6_balanced_puncture_accounting_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_v6_balanced_puncture_accounting_summary = None
    best_vs_original_summary = None
    best_vs_matched_summary = None
    best_vs_throughput_feasible_summary = None
    best_vs_throughput_only_summary = None
    best_vs_channel_only_summary = None
    best_floor_throughput_summary = None
    best_multiload_frontier_summary = None
    primary_checkpoint_preference = str(
        getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") or "best_throughput"
    ).strip().lower()
    has_loadwise_selection_constraints = bool(
        dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_service_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_minrate_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}) or {})
        or float(getattr(cfg.training, "selection_reliability_floor", 0.0) or 0.0) > 0.0
        or float(getattr(cfg.training, "selection_puncture_ratio_ceiling", 1.0) or 1.0) < 1.0 - 1e-9
    )
    has_selection_constraints = bool(selection_admission_floor > 0.0 or has_loadwise_selection_constraints)
    best_throughput_warmup_iterations = max(
        int(getattr(cfg.training, "best_throughput_warmup_iterations", 0) or 0),
        0,
    )
    best_throughput_min_delta = float(getattr(cfg.training, "best_throughput_min_delta", 0.0) or 0.0)
    light_eval_every = max(int(getattr(cfg.training, "light_eval_every", 0) or 0), 0)
    full_eval_every = max(int(getattr(cfg.training, "full_eval_every", 0) or 0), 0)
    full_eval_enabled_during_training = bool(getattr(cfg.training, "full_eval_enabled_during_training", False))

    def _build_training_eval_cfg(base_cfg, force_full_compare: bool = False):
        eval_cfg = deepcopy(base_cfg)
        if force_full_compare:
            eval_kind = "full_eval_training"
        else:
            light_eval_loads = [
                float(load)
                for load in (
                    getattr(base_cfg.training, "light_eval_loads", [])
                    or []
                )
            ]
            light_eval_episodes_per_load = int(
                getattr(base_cfg.training, "light_eval_episodes_per_load", 0) or 0
            )
            if light_eval_loads or light_eval_episodes_per_load > 0:
                if light_eval_loads:
                    eval_cfg.training.eval_loads = light_eval_loads
                if light_eval_episodes_per_load > 0:
                    eval_cfg.training.eval_episodes_per_load = light_eval_episodes_per_load
                eval_kind = "light_eval_sparse"
            else:
                checkpoint_eval_scope = str(
                    getattr(base_cfg.training, "checkpoint_eval_scope", "representative_load") or "representative_load"
                ).strip().lower()
                eval_kind = "light_eval_representative"
                if checkpoint_eval_scope == "all_loads":
                    checkpoint_eval_loads = [
                        float(load)
                        for load in (
                            getattr(base_cfg.training, "checkpoint_eval_loads", [])
                            or getattr(base_cfg.training, "eval_loads", [])
                            or []
                        )
                    ]
                    checkpoint_eval_episodes_per_load = int(
                        getattr(base_cfg.training, "checkpoint_eval_episodes_per_load", 1) or 1
                    )
                    if checkpoint_eval_loads:
                        eval_cfg.training.eval_loads = checkpoint_eval_loads
                    eval_cfg.training.eval_episodes_per_load = checkpoint_eval_episodes_per_load
                    eval_kind = "light_eval_all_loads"
        compare_modes = (
            ["selected", "original", "matched", "throughput_feasible", "throughput_only", "channel_only"]
            if force_full_compare
            else list(getattr(eval_cfg.training, "eval_compare_modes", []) or ["selected"])
        )
        eval_cfg.training.eval_compare_modes = list(compare_modes)
        return eval_cfg, eval_kind, compare_modes, force_full_compare

    def _run_training_evaluation(model, target_cfg, force_full_compare: bool):
        if force_full_compare:
            try:
                return evaluation_fn(env, model, target_cfg, force_full_compare=True)
            except TypeError:
                return evaluation_fn(env, model, target_cfg)
        return evaluation_fn(env, model, target_cfg)

    total_episodes_completed = 0
    for iteration in range(start_iteration, cfg.training.total_iterations + 1):
        iteration_start = perf_counter()
        checkpoint_sec = 0.0
        phase_a_power_runtime = phase_a_embb_power_runtime_enabled(cfg, iteration)
        set_phase_a_embb_power_runtime(env, model, phase_a_power_runtime)
        actor_lr, critic_lr, entropy_coef, clip_ratio = _stable_phase_schedule(
            cfg,
            iteration,
            trainer.base_actor_lr,
            trainer.base_critic_lr,
        )
        trainer.set_optimizer_lrs(actor_lr, critic_lr)

        def _save_checkpoint_timed(path: Path, extra: Optional[Dict] = None) -> None:
            nonlocal checkpoint_sec
            checkpoint_start = perf_counter()
            trainer.save_checkpoint(path, extra=extra)
            checkpoint_sec += perf_counter() - checkpoint_start

        chosen_load = choose_training_load(cfg, iteration)
        actual_load = None
        if chosen_load is not None:
            actual_load = configure_env_for_users_per_uav(env, chosen_load)

        env.algo_cfg.embb_min_sic_snir_db = sic_curriculum_db(cfg, iteration)
        set_training_fallback_warmup(env, cfg, iteration)
        env.training_progress_frac = float((iteration - 1) / max(cfg.training.total_iterations - 1, 1))
        env.current_training_iteration = int(iteration)

        rollout_start = perf_counter()
        rollout_stats = trainer.collect_rollout(
            horizon=cfg.training.rollout_horizon,
            seed=cfg.training.train_seed + iteration,
            iteration=iteration,
        )
        episodes_this_iteration = max(
            int(round(float(rollout_stats.get("episodes_completed", 0.0) or 0.0))),
            0,
        )
        total_episodes_completed += int(episodes_this_iteration)
        rollout_sec = perf_counter() - rollout_start
        guidance_scale = teacher_guidance_scale(cfg, iteration)
        distill_coef = teacher_distill_coef(cfg, iteration)
        greedy_bc_coef = greedy_reference_bc_coef(cfg, iteration)
        update_start = perf_counter()
        update_stats = trainer.update(
            ppo_epochs=cfg.training.ppo_epochs,
            minibatch_size=cfg.training.minibatch_size,
            teacher_scale=guidance_scale,
            distill_coef=distill_coef,
            greedy_bc_coef=greedy_bc_coef,
            actual_load=actual_load,
            entropy_coef=entropy_coef,
            clip_ratio=clip_ratio,
        )
        update_sec = perf_counter() - update_start
        record = {
            'iteration': iteration,
            'episodes_this_iteration': int(episodes_this_iteration),
            'episodes_completed_total': int(total_episodes_completed),
            'curriculum_target_load': chosen_load,
            'curriculum_actual_load': actual_load,
            'teacher_guidance_scale': guidance_scale,
            'teacher_distill_coef': distill_coef,
            'greedy_reference_bc_coef': greedy_bc_coef,
            'rollout': rollout_stats,
            'update': asdict(update_stats),
            'control': {
                'phase': str(cfg.env.phase),
                'learn_embb_baseline': bool(cfg.env.learn_embb_baseline),
                'learn_phase0_embb_power': bool(getattr(cfg.env, 'learn_phase0_embb_power', True)),
                'allow_phase_a_embb_power_adjustment': bool(cfg.env.allow_phase_a_embb_power_adjustment),
                'phase_a_embb_power_runtime_enabled': bool(phase_a_power_runtime),
                'phase_a_embb_power_anchor_enabled': bool(
                    phase_a_power_runtime and phase_a_embb_power_anchor_enabled(cfg, iteration)
                ),
                'actor_learning_rate': float(actor_lr),
                'critic_learning_rate': float(critic_lr),
                'entropy_coef': float(entropy_coef),
                'clip_ratio': float(clip_ratio),
                'enable_action_masking': bool(cfg.shield.enable_action_masking),
                'enable_feasibility_shield': bool(cfg.shield.enable_feasibility_shield),
                'apply_joint_reliability_rewrite': bool(cfg.shield.apply_joint_reliability_rewrite),
                'enable_greedy_fallback': bool(cfg.shield.enable_greedy_fallback),
            },
        }
        eval_sec = 0.0

        effective_light_eval_every = light_eval_every if light_eval_every > 0 else max(int(cfg.training.eval_every), 1)
        should_final_eval = iteration == cfg.training.total_iterations
        should_light_eval = effective_light_eval_every > 0 and iteration % effective_light_eval_every == 0
        should_full_eval = (
            full_eval_enabled_during_training
            and full_eval_every > 0
            and not should_final_eval
            and iteration % full_eval_every == 0
        )

        if evaluation_fn is not None and (should_light_eval or should_full_eval or should_final_eval):
            eval_start = perf_counter()
            fallback_state = disable_eval_fallback(env)
            previous_training_progress = float(getattr(env, "training_progress_frac", 1.0))
            previous_training_iteration = int(getattr(env, "current_training_iteration", 1) or 1)
            env.training_progress_frac = 1.0
            env.current_training_iteration = int(iteration)
            training_eval_cfg, evaluation_kind, evaluation_compare_modes, force_full_compare = _build_training_eval_cfg(
                cfg,
                force_full_compare=bool(should_full_eval),
            )
            try:
                eval_summary = _run_training_evaluation(model, training_eval_cfg, force_full_compare=force_full_compare)
                checkpoint_eval_summary = eval_summary
            finally:
                env.current_training_iteration = previous_training_iteration
                env.training_progress_frac = previous_training_progress
                restore_eval_fallback(env, fallback_state)
            eval_sec = perf_counter() - eval_start
            record['evaluation'] = eval_summary
            record['checkpoint_evaluation'] = checkpoint_eval_summary
            record['evaluation_kind'] = evaluation_kind
            record['evaluation_loads'] = [
                float(load) for load in (getattr(training_eval_cfg.training, 'eval_loads', []) or [])
            ]
            record['evaluation_episodes_per_load'] = int(
                getattr(training_eval_cfg.training, 'eval_episodes_per_load', getattr(training_eval_cfg.training, 'eval_episodes', 1)) or 1
            )
            record['evaluation_compare_modes'] = list(evaluation_compare_modes)
            reward_score = float(checkpoint_eval_summary.get('policy_score', update_stats.mean_reward))
            throughput_score = float(
                checkpoint_eval_summary.get(
                    'policy_throughput_score',
                    checkpoint_eval_summary.get('policy_mean_embb_rate', 0.0),
                )
            )
            admission_score = float(checkpoint_eval_summary.get('policy_mean_scheduled_ratio', 0.0))
            balanced_score, balanced_throughput_ratio, balanced_admission_ratio = _balanced_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            service_interference_balanced_score = _service_interference_balanced_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            service_power_interference_balanced_score = _service_power_interference_balanced_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            service_gain_interference_balanced_score = _service_gain_interference_balanced_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            balanced_intercell_aware_score = _balanced_intercell_aware_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            owner_frozen_action_intercell_balanced_score = _owner_frozen_action_intercell_balanced_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            v5_balanced_intercell_admission_score = _v5_balanced_intercell_admission_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            v6_balanced_puncture_accounting_score = _v6_balanced_puncture_accounting_checkpoint_score(
                checkpoint_eval_summary,
                cfg,
            )
            weighted_floor_violation = float(checkpoint_eval_summary.get('weighted_floor_violation', 0.0))
            weighted_power_ceiling_violation = float(checkpoint_eval_summary.get('weighted_power_ceiling_violation', 0.0))
            vs_original_score = float(checkpoint_eval_summary.get('policy_score_vs_original_greedy', float('-inf')))
            vs_matched_score = float(checkpoint_eval_summary.get('policy_score_vs_matched_greedy', float('-inf')))
            vs_throughput_feasible_score = float(
                checkpoint_eval_summary.get('policy_score_vs_throughput_feasible_oracle', float('-inf'))
            )
            vs_throughput_only_score = float(
                checkpoint_eval_summary.get('policy_score_vs_throughput_only_greedy', float('-inf'))
            )
            vs_channel_only_score = float(checkpoint_eval_summary.get('policy_score_vs_channel_only_greedy', float('-inf')))
            multiload_frontier_score = float(checkpoint_eval_summary.get('multiload_frontier_score', float('-inf')))
            multiload_frontier_pass = bool(
                checkpoint_eval_summary.get('multiload_frontier_all_loads_pass_constraints', 0.0) >= 1.0
            )
            non_worse = bool(checkpoint_eval_summary.get('non_worse_than_greedy', 0.0) >= 1.0)
            eligible = non_worse if cfg.training.keep_best_non_worse_than_greedy else True
            if has_loadwise_selection_constraints:
                passes_selection_constraints = bool(
                    checkpoint_eval_summary.get('all_loads_pass_selection_constraints', 0.0) >= 1.0
                )
            elif selection_admission_floor > 0.0:
                passes_selection_constraints = admission_score >= selection_admission_floor - 1e-9
            else:
                passes_selection_constraints = True
            throughput_best_before = best_throughput_score
            throughput_warmup_satisfied = iteration >= max(best_throughput_warmup_iterations, 1)
            throughput_improved = throughput_score > best_throughput_score + best_throughput_min_delta
            record['checkpoint_metrics'] = {
                'reward_score': float(reward_score),
                'throughput_score': float(throughput_score),
                'admission_score': float(admission_score),
                'balanced_score': float(balanced_score),
                'balanced_throughput_ratio': float(balanced_throughput_ratio),
                'balanced_admission_ratio': float(balanced_admission_ratio),
                'eligible': float(eligible),
                'passes_selection_constraints': float(passes_selection_constraints),
                'throughput_best_before': float(throughput_best_before),
                'best_throughput_warmup_iterations': float(best_throughput_warmup_iterations),
                'best_throughput_min_delta': float(best_throughput_min_delta),
                'throughput_warmup_satisfied': float(throughput_warmup_satisfied),
                'throughput_improved_vs_best': float(throughput_improved),
                'service_interference_balanced_score': float(service_interference_balanced_score),
                'service_power_interference_balanced_score': float(service_power_interference_balanced_score),
                'service_gain_interference_balanced_score': float(service_gain_interference_balanced_score),
                'balanced_intercell_aware_score': float(balanced_intercell_aware_score),
                'owner_frozen_action_intercell_balanced_score': float(owner_frozen_action_intercell_balanced_score),
                'v5_balanced_intercell_admission_score': float(v5_balanced_intercell_admission_score),
                'v6_balanced_puncture_accounting_score': float(v6_balanced_puncture_accounting_score),
            }
            eval_phase_a_changed_count = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase_a_embb_power_changed_count',
                'phase_a_embb_power_changed_count',
                default=0.0,
            )
            eval_phase_a_changed_ratio = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase_a_embb_power_changed_ratio',
                'phase_a_embb_power_changed_ratio',
                default=0.0,
            )
            eval_phase_a_raw_delta = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase_a_embb_power_mean_raw_delta',
                'phase_a_embb_power_mean_raw_delta',
                default=0.0,
            )
            eval_phase_a_executed_delta = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase_a_embb_power_mean_executed_delta',
                'phase_a_embb_power_mean_executed_delta',
                default=0.0,
            )
            record['checkpoint_metrics'].update({
                'phase_a_embb_power_changed_count': float(eval_phase_a_changed_count),
                'phase_a_embb_power_changed_ratio': float(eval_phase_a_changed_ratio),
                'phase_a_embb_power_mean_raw_delta': float(eval_phase_a_raw_delta),
                'phase_a_embb_power_mean_executed_delta': float(eval_phase_a_executed_delta),
            })
            eval_phase_a_eff_nz = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase_a_embb_power_effective_nonzero_ratio',
                'phase_a_embb_power_effective_nonzero_ratio',
                default=0.0,
            )
            eval_phase0_owner_change_raw = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_change_ratio_vs_snapshot_raw',
                'phase0_owner_change_ratio_vs_snapshot_raw',
                default=0.0,
            )
            eval_phase0_owner_change_exe = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_change_ratio_vs_snapshot_executed',
                'phase0_owner_change_ratio_vs_snapshot_executed',
                default=0.0,
            )
            eval_phase0_owner_restored = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_restored_to_snapshot_ratio',
                'phase0_owner_restored_to_snapshot_ratio',
                default=0.0,
            )
            eval_phase0_owner_replaced = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_replaced_with_non_snapshot_ratio',
                'phase0_owner_replaced_with_non_snapshot_ratio',
                default=0.0,
            )
            eval_phase0_owner_invalid_to_snapshot = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_invalid_to_snapshot_ratio',
                'phase0_owner_invalid_to_snapshot_ratio',
                default=0.0,
            )
            eval_phase0_owner_invalid_to_non_snapshot = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_invalid_to_non_snapshot_ratio',
                'phase0_owner_invalid_to_non_snapshot_ratio',
                default=0.0,
            )
            eval_phase0_owner_nn_raw = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_non_null_ratio_raw',
                'phase0_owner_non_null_ratio_raw',
                default=0.0,
            )
            eval_phase0_owner_nn_exe = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_non_null_ratio_executed',
                'phase0_owner_non_null_ratio_executed',
                default=0.0,
            )
            eval_phase0_owner_effective = _summary_float(
                checkpoint_eval_summary,
                'policy_mean_phase0_owner_changed_and_effective_ratio',
                'phase0_owner_changed_and_effective_ratio',
                default=0.0,
            )
            if float(eval_phase_a_eff_nz) < 0.05:
                dead_phase_a_eval_streak += 1
            else:
                dead_phase_a_eval_streak = 0
            if float(eval_phase0_owner_change_exe) < 0.05 or float(eval_phase0_owner_effective) < 0.05:
                frozen_owner_eval_streak += 1
            else:
                frozen_owner_eval_streak = 0
            if (
                float(eval_phase0_owner_change_raw) > 0.50
                and float(eval_phase0_owner_change_exe) < 0.05
                and float(eval_phase0_owner_restored) > 0.80
            ):
                owner_restore_collapse_eval_streak += 1
            else:
                owner_restore_collapse_eval_streak = 0
            if dead_phase_a_eval_streak >= 3:
                warning = (
                    "Phase-A head is effectively dead: "
                    f"policy_mean_phaseA_eff_nz={float(eval_phase_a_eff_nz):.3f} "
                    f"(streak={dead_phase_a_eval_streak})"
                )
                record.setdefault('warnings', []).append(warning)
                print(f"[{_tw_timestamp()}] [SR-MAPPO][WARN] {warning}", flush=True)
            if frozen_owner_eval_streak >= 3:
                warning = (
                    "Phase-0 owner head is effectively frozen: "
                    f"raw_change={float(eval_phase0_owner_change_raw):.3f} "
                    f"exe_change={float(eval_phase0_owner_change_exe):.3f} "
                    f"effective={float(eval_phase0_owner_effective):.3f} "
                    f"restored={float(eval_phase0_owner_restored):.3f} "
                    f"replaced_non_snapshot={float(eval_phase0_owner_replaced):.3f} "
                    f"invalid_to_snapshot={float(eval_phase0_owner_invalid_to_snapshot):.3f} "
                    f"invalid_to_non_snapshot={float(eval_phase0_owner_invalid_to_non_snapshot):.3f} "
                    f"owner_nn(raw/exe)={float(eval_phase0_owner_nn_raw):.3f}/{float(eval_phase0_owner_nn_exe):.3f} "
                    f"(streak={frozen_owner_eval_streak})"
                )
                record.setdefault('warnings', []).append(warning)
                print(f"[{_tw_timestamp()}] [SR-MAPPO][WARN] {warning}", flush=True)
            if owner_restore_collapse_eval_streak >= 3:
                warning = (
                    "phase0 owner head is active in raw action but collapsed by execution restore path: "
                    f"policy_mean_owner_raw_change={float(eval_phase0_owner_change_raw):.3f} "
                    f"policy_mean_owner_exe_change={float(eval_phase0_owner_change_exe):.3f} "
                    f"policy_mean_owner_restored={float(eval_phase0_owner_restored):.3f} "
                    f"(streak={owner_restore_collapse_eval_streak})"
                )
                record.setdefault('warnings', []).append(warning)
                print(f"[{_tw_timestamp()}] [SR-MAPPO][WARN] {warning}", flush=True)
            if bool(phase_a_power_runtime) and eval_phase_a_changed_ratio <= 1.0e-9:
                warning = (
                    "Phase-A eMBB power runtime enabled but no nonzero Phase-A eMBB power changes "
                    f"were observed in {evaluation_kind} at iteration {iteration}."
                )
                record.setdefault('warnings', []).append(warning)
                timestamp = _tw_timestamp()
                print(f"[{timestamp}] [SR-MAPPO][WARN] {warning}", flush=True)
            if eligible and reward_score > best_reward_score:
                best_reward_score = reward_score
                best_reward_summary = checkpoint_eval_summary
                if selection_mode != "throughput_only" and not has_selection_constraints and not load_aware_objective:
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='policy_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_reward.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score',
                    ),
                )
            if eligible and throughput_warmup_satisfied and throughput_improved:
                best_throughput_score = throughput_score
                best_throughput_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['throughput_updated'] = 1.0
                if selection_mode == "throughput_only" and not has_selection_constraints and not load_aware_objective:
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='policy_throughput_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_throughput.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_throughput_score',
                    ),
                )
            else:
                record['checkpoint_metrics']['throughput_updated'] = 0.0
            balanced_key = (
                float(passes_selection_constraints),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(balanced_score),
                float(throughput_score),
            )
            balanced_improved = balanced_key > best_balanced_key
            record['checkpoint_metrics']['balanced_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and balanced_improved:
                best_balanced_key = balanced_key
                best_balanced_score = balanced_score
                best_balanced_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['balanced_updated'] = 1.0
                if primary_checkpoint_preference == "best_balanced":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='balanced_throughput_admission_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_balanced.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='balanced_throughput_admission_score',
                    ),
                )

            # Service+interference balanced checkpoint (for service recovery + intercell repair debug).
            service_interference_key = (
                float(passes_selection_constraints),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(service_interference_balanced_score),
                float(throughput_score),
            )
            service_interference_improved = service_interference_key > best_service_interference_balanced_key
            record['checkpoint_metrics']['service_interference_balanced_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and service_interference_improved:
                best_service_interference_balanced_key = service_interference_key
                best_service_interference_balanced_score = float(service_interference_balanced_score)
                best_service_interference_balanced_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['service_interference_balanced_updated'] = 1.0
                if primary_checkpoint_preference == "best_service_interference_balanced":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='service_interference_balanced_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_service_interference_balanced.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='service_interference_balanced_score',
                    ),
                )

            # Service+power+interference balanced checkpoint (v2): explicitly penalize power-over-greedy and keep reliability in view.
            service_power_interference_key = (
                float(passes_selection_constraints),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(service_power_interference_balanced_score),
                float(throughput_score),
            )
            service_power_interference_improved = service_power_interference_key > best_service_power_interference_balanced_key
            record['checkpoint_metrics']['service_power_interference_balanced_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and service_power_interference_improved:
                best_service_power_interference_balanced_key = service_power_interference_key
                best_service_power_interference_balanced_score = float(service_power_interference_balanced_score)
                best_service_power_interference_balanced_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['service_power_interference_balanced_updated'] = 1.0
                if primary_checkpoint_preference == "best_service_power_interference_balanced":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='service_power_interference_balanced_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_service_power_interference_balanced.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='service_power_interference_balanced_score',
                    ),
                )

            # Service gain vs greedy + interference balanced checkpoint (v3).
            service_gain_interference_key = (
                float(passes_selection_constraints),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(service_gain_interference_balanced_score),
                float(throughput_score),
            )
            service_gain_interference_improved = service_gain_interference_key > best_service_gain_interference_balanced_key
            record['checkpoint_metrics']['service_gain_interference_balanced_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and service_gain_interference_improved:
                best_service_gain_interference_balanced_key = service_gain_interference_key
                best_service_gain_interference_balanced_score = float(service_gain_interference_balanced_score)
                best_service_gain_interference_balanced_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['service_gain_interference_balanced_updated'] = 1.0
                if primary_checkpoint_preference == "best_service_gain_interference_balanced":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='service_gain_interference_balanced_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_service_gain_interference_balanced.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='service_gain_interference_balanced_score',
                    ),
                )

            # Always keep a short-horizon debug checkpoint (overwritten each eval).
            _save_checkpoint_timed(
                checkpoint_dir / f"{cfg.training.run_name}_latest_debug.pt",
                extra=_checkpoint_extra(
                    iteration,
                    record=record,
                    evaluation=checkpoint_eval_summary,
                    selection_metric='latest_debug',
                ),
            )

            # Short-horizon balanced intercell-aware checkpoints (do not require long training).
            compare_selected = checkpoint_eval_summary.get("compare_selected_baseline", {})
            if not isinstance(compare_selected, dict):
                compare_selected = {}
            per_load = list(compare_selected.get("per_load", []) or [])
            small_tol = 0.02
            admission_ok = 0
            service_ok = 0
            for item in per_load:
                if not isinstance(item, dict):
                    continue
                policy_adm = float(item.get("policy_mean_scheduled_ratio", 0.0) or 0.0)
                greedy_adm = float(item.get("greedy_mean_scheduled_ratio", 0.0) or 0.0)
                if policy_adm >= greedy_adm - small_tol:
                    admission_ok += 1
                policy_srv = float(item.get("policy_mean_embb_service_ratio", 0.0) or 0.0)
                greedy_srv = float(item.get("greedy_mean_embb_service_ratio", 0.0) or 0.0)
                if policy_srv >= greedy_srv - small_tol:
                    service_ok += 1
            phase_a_eff_nz = float(checkpoint_eval_summary.get("policy_mean_phase_a_embb_power_effective_nonzero_ratio", 0.0) or 0.0)
            policy_inter_loss = float(checkpoint_eval_summary.get("policy_mean_embb_rate_loss_due_to_intercell_ratio", 0.0) or 0.0)
            greedy_inter_loss = float(compare_selected.get("greedy_mean_embb_rate_loss_due_to_intercell_ratio", checkpoint_eval_summary.get("greedy_mean_embb_rate_loss_due_to_intercell_ratio", 0.0)) or 0.0)
            policy_mean_intercell = float(checkpoint_eval_summary.get("policy_mean_mean_intercell_interference_mw", 0.0) or 0.0)
            greedy_mean_intercell = float(compare_selected.get("greedy_mean_mean_intercell_interference_mw", checkpoint_eval_summary.get("greedy_mean_mean_intercell_interference_mw", 0.0)) or 0.0)
            intercell_ok = (greedy_mean_intercell <= 1.0e-9) or (policy_mean_intercell <= 1.05 * greedy_mean_intercell + 1.0e-9) or (policy_inter_loss <= greedy_inter_loss + 1.0e-9)
            debug_gate_pass = bool(admission_ok >= 3 and service_ok >= 3 and intercell_ok and phase_a_eff_nz > 0.03)

            balanced_intercell_key = (
                float(debug_gate_pass),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(balanced_intercell_aware_score),
                float(throughput_score),
            )
            balanced_intercell_improved = balanced_intercell_key > best_balanced_intercell_aware_key
            record['checkpoint_metrics']['balanced_intercell_aware_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and balanced_intercell_improved:
                best_balanced_intercell_aware_key = balanced_intercell_key
                best_balanced_intercell_aware_score = float(balanced_intercell_aware_score)
                best_balanced_intercell_aware_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['balanced_intercell_aware_updated'] = 1.0
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_balanced_intercell_aware.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='balanced_intercell_aware_score',
                    ),
                )

            admission_service_intercell_score = float(
                2.0 * float(admission_score)
                + 1.5 * float(_summary_float(checkpoint_eval_summary, "policy_mean_embb_service_ratio", default=0.0))
                + 1.0 * float(_summary_float(checkpoint_eval_summary, "policy_mean_embb_min_rate_satisfaction_ratio", default=0.0))
                - 2.0 * float(policy_inter_loss)
            )
            admission_service_key = (
                float(debug_gate_pass),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(admission_service_intercell_score),
                float(throughput_score),
            )
            admission_service_improved = admission_service_key > best_admission_service_intercell_key
            record['checkpoint_metrics']['admission_service_intercell_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and admission_service_improved:
                best_admission_service_intercell_key = admission_service_key
                best_admission_service_intercell_score = float(admission_service_intercell_score)
                best_admission_service_intercell_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['admission_service_intercell_updated'] = 1.0
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_admission_service_intercell.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='admission_service_intercell_score',
                    ),
                )

            owner_frozen_key = (
                float(debug_gate_pass),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(owner_frozen_action_intercell_balanced_score),
                float(throughput_score),
            )
            owner_frozen_improved = owner_frozen_key > best_owner_frozen_action_intercell_balanced_key
            record['checkpoint_metrics']['owner_frozen_action_intercell_balanced_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and owner_frozen_improved:
                best_owner_frozen_action_intercell_balanced_key = owner_frozen_key
                best_owner_frozen_action_intercell_balanced_score = float(owner_frozen_action_intercell_balanced_score)
                best_owner_frozen_action_intercell_balanced_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['owner_frozen_action_intercell_balanced_updated'] = 1.0
                if primary_checkpoint_preference == "best_owner_frozen_action_intercell_balanced":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='owner_frozen_action_intercell_balanced_score',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_owner_frozen_action_intercell_balanced.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='owner_frozen_action_intercell_balanced_score',
                    ),
                )
            v5_balanced_key = (
                float(debug_gate_pass),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(v5_balanced_intercell_admission_score),
                float(throughput_score),
            )
            v5_balanced_improved = v5_balanced_key > best_v5_balanced_intercell_admission_key
            record['checkpoint_metrics']['v5_balanced_intercell_admission_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and v5_balanced_improved:
                best_v5_balanced_intercell_admission_key = v5_balanced_key
                best_v5_balanced_intercell_admission_score = float(v5_balanced_intercell_admission_score)
                best_v5_balanced_intercell_admission_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['v5_balanced_intercell_admission_updated'] = 1.0
                if primary_checkpoint_preference == "best_v5_balanced_intercell_admission":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='best_v5_balanced_intercell_admission',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_v5_balanced_intercell_admission.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='best_v5_balanced_intercell_admission',
                    ),
                )
            v6_balanced_key = (
                float(debug_gate_pass),
                -float(weighted_floor_violation + weighted_power_ceiling_violation),
                float(v6_balanced_puncture_accounting_score),
                float(throughput_score),
            )
            v6_balanced_improved = v6_balanced_key > best_v6_balanced_puncture_accounting_key
            record['checkpoint_metrics']['v6_balanced_puncture_accounting_updated'] = 0.0
            if eligible and throughput_warmup_satisfied and v6_balanced_improved:
                best_v6_balanced_puncture_accounting_key = v6_balanced_key
                best_v6_balanced_puncture_accounting_score = float(v6_balanced_puncture_accounting_score)
                best_v6_balanced_puncture_accounting_summary = checkpoint_eval_summary
                record['checkpoint_metrics']['v6_balanced_puncture_accounting_updated'] = 1.0
                if primary_checkpoint_preference == "best_v6_balanced_puncture_accounting":
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='best_v6_balanced_puncture_accounting',
                        ),
                    )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_v6_balanced_puncture_accounting.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='best_v6_balanced_puncture_accounting',
                    ),
                )
            if eligible and has_selection_constraints and passes_selection_constraints and throughput_score > best_floor_throughput_score:
                best_floor_throughput_score = throughput_score
                best_floor_throughput_summary = checkpoint_eval_summary
                if primary_checkpoint_preference not in {
                    "best_multiload_frontier",
                    "best_multiload_tp_power",
                    "best_balanced",
                    "best_v5_balanced_intercell_admission",
                    "best_v6_balanced_puncture_accounting",
                }:
                    best_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_floor_throughput.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_throughput_score_with_selection_constraints',
                    ),
                )
                if primary_checkpoint_preference not in {
                    "best_multiload_frontier",
                    "best_multiload_tp_power",
                    "best_balanced",
                    "best_v5_balanced_intercell_admission",
                    "best_v6_balanced_puncture_accounting",
                }:
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric='policy_throughput_score_with_selection_constraints',
                        ),
                    )
            if (
                eligible
                and has_selection_constraints
                and best_floor_throughput_score == float("-inf")
            ):
                fallback_key = (-(weighted_floor_violation + weighted_power_ceiling_violation), throughput_score)
                if fallback_key > best_floor_fallback_key:
                    best_floor_fallback_key = fallback_key
                    if primary_checkpoint_preference not in {
                        "best_multiload_frontier",
                        "best_multiload_tp_power",
                        "best_balanced",
                        "best_v5_balanced_intercell_admission",
                        "best_v6_balanced_puncture_accounting",
                    }:
                        best_summary = checkpoint_eval_summary
                        _save_checkpoint_timed(
                            checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                            extra=_checkpoint_extra(
                                iteration,
                                record=record,
                                evaluation=checkpoint_eval_summary,
                                selection_metric='policy_floor_violation_then_throughput',
                            ),
                        )
            if eligible and np.isfinite(vs_original_score) and vs_original_score > best_vs_original_score:
                best_vs_original_score = vs_original_score
                best_vs_original_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_vs_original_greedy.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score_vs_original_greedy',
                    ),
                )
            if eligible and np.isfinite(vs_matched_score) and vs_matched_score > best_vs_matched_score:
                best_vs_matched_score = vs_matched_score
                best_vs_matched_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_vs_matched_greedy.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score_vs_matched_greedy',
                    ),
                )
            if (
                eligible
                and np.isfinite(vs_throughput_feasible_score)
                and vs_throughput_feasible_score > best_vs_throughput_feasible_score
            ):
                best_vs_throughput_feasible_score = vs_throughput_feasible_score
                best_vs_throughput_feasible_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_vs_throughput_feasible_oracle.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score_vs_throughput_feasible_oracle',
                    ),
                )
            if eligible and np.isfinite(vs_throughput_only_score) and vs_throughput_only_score > best_vs_throughput_only_score:
                best_vs_throughput_only_score = vs_throughput_only_score
                best_vs_throughput_only_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_vs_throughput_only_greedy.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score_vs_throughput_only_greedy',
                    ),
                )
            if eligible and np.isfinite(vs_channel_only_score) and vs_channel_only_score > best_vs_channel_only_score:
                best_vs_channel_only_score = vs_channel_only_score
                best_vs_channel_only_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_vs_channel_only_greedy.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='policy_score_vs_channel_only_greedy',
                    ),
                )
            if (
                str(getattr(cfg.training, "checkpoint_eval_scope", "representative_load") or "representative_load").strip().lower() == "all_loads"
                and eligible
                and multiload_frontier_pass
                and np.isfinite(multiload_frontier_score)
                and multiload_frontier_score > best_multiload_frontier_score
            ):
                best_multiload_frontier_score = multiload_frontier_score
                best_multiload_frontier_summary = checkpoint_eval_summary
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_multiload_tp_power.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='multiload_tp_power_score',
                    ),
                )
                _save_checkpoint_timed(
                    checkpoint_dir / f"{cfg.training.run_name}_best_multiload_frontier.pt",
                    extra=_checkpoint_extra(
                        iteration,
                        record=record,
                        evaluation=checkpoint_eval_summary,
                        selection_metric='multiload_frontier_score',
                    ),
                )
                if primary_checkpoint_preference in {"best_multiload_frontier", "best_multiload_tp_power"}:
                    best_summary = checkpoint_eval_summary
                    _save_checkpoint_timed(
                        checkpoint_dir / f"{cfg.training.run_name}_best.pt",
                        extra=_checkpoint_extra(
                            iteration,
                            record=record,
                            evaluation=checkpoint_eval_summary,
                            selection_metric=(
                                'multiload_tp_power_score'
                                if primary_checkpoint_preference == "best_multiload_tp_power"
                                else 'multiload_frontier_score'
                            ),
                        ),
                    )

        if iteration % cfg.training.checkpoint_every == 0:
            _save_checkpoint_timed(
                checkpoint_dir / f"{cfg.training.run_name}_iter{iteration}.pt",
                extra=_checkpoint_extra(iteration, record=record),
            )

        iteration_total_sec = perf_counter() - iteration_start
        record['timing'] = {
            'rollout_sec': float(rollout_sec),
            'update_sec': float(update_sec),
            'eval_sec': float(eval_sec),
            'checkpoint_sec': float(checkpoint_sec),
            'iteration_total_sec': float(iteration_total_sec),
            'obs_build_sec': float(rollout_stats.get('obs_build_sec', 0.0)),
            'policy_forward_sec': float(rollout_stats.get('policy_forward_sec', 0.0)),
            'greedy_bc_target_sec': float(rollout_stats.get('greedy_bc_target_sec', 0.0)),
            'env_step_sec': float(rollout_stats.get('env_step_sec', 0.0)),
            'buffer_add_sec': float(rollout_stats.get('buffer_add_sec', 0.0)),
        }

        history.append(record)

        if iteration % progress_every == 0 or iteration == 1 or iteration == cfg.training.total_iterations:
            rollout_reward = float(rollout_stats.get('mean_reward', 0.0))
            rollout_embb_rate = float(rollout_stats.get('mean_embb_total_rate', 0.0)) / 1.0e6
            latest_eval = record.get('evaluation')
            if latest_eval is None and best_reward_summary is not None:
                latest_eval = best_reward_summary
            timestamp = _tw_timestamp()
            msg = (
                f"[{timestamp}] [SR-MAPPO] iter {iteration}/{cfg.training.total_iterations} | "
                f"ep(iter)={int(episodes_this_iteration)} | "
                f"ep(total)={int(total_episodes_completed)} | "
                f"target_load={chosen_load if chosen_load is not None else 'na'} | "
                f"actual_load={actual_load if actual_load is not None else 'na'} | "
                f"rollout_reward={rollout_reward:.4f} | "
                f"rollout_embb_rate={rollout_embb_rate:.3f} Mbps | "
                f"teacher_scale={guidance_scale:.3f} | "
                f"distill_coef={distill_coef:.3f} | "
                f"greedy_bc_coef={greedy_bc_coef:.3f}"
            )
            msg += (
                f" | mode(raw ov/pu)={float(rollout_stats.get('raw_overlay_ratio', 0.0)):.3f}/{float(rollout_stats.get('raw_puncture_ratio', 0.0)):.3f}"
                f" | mode(exe ov/pu)={float(rollout_stats.get('executed_overlay_ratio', 0.0)):.3f}/{float(rollout_stats.get('executed_puncture_ratio', 0.0)):.3f}"
                f" | rewrite={float(rollout_stats.get('shield_correction_ratio', 0.0)):.3f}"
                f" | rw(mode/owner/pkt/safe)={float(rollout_stats.get('mode_rewrite_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_rewrite_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('packet_rewrite_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('any_safety_rewrite_ratio', 0.0)):.3f}"
                f" | power_proj={float(rollout_stats.get('power_projection_ratio', 0.0)):.3f}"
                f" | joint_rw={float(rollout_stats.get('joint_reliability_rewrite_ratio', 0.0)):.3f}"
                f" | mode_fix={float(rollout_stats.get('mode_correction_ratio', 0.0)):.3f}"
                f" | pkt_invalid={float(rollout_stats.get('packet_invalid_ratio', 0.0)):.3f}"
                f" | mask_invalid={float(rollout_stats.get('mask_invalid_ratio', 0.0)):.3f}"
                f" | raw_exec_mode_gap={float(rollout_stats.get('raw_vs_executed_mode_gap', 0.0)):.3f}"
            )
            msg += (
                f" | owner_nonnull(raw/exe)={float(rollout_stats.get('raw_owner_non_null_ratio', 0.0)):.3f}/{float(rollout_stats.get('executed_owner_non_null_ratio', 0.0)):.3f}"
                f" | ph0_owner_nn(raw/exe)={float(rollout_stats.get('mean_phase0_owner_non_null_ratio_raw', 0.0)):.3f}/{float(rollout_stats.get('mean_phase0_owner_non_null_ratio_executed', 0.0)):.3f}"
                f" | ph0_owner_chg(raw/exe)={float(rollout_stats.get('mean_phase0_owner_change_ratio_vs_snapshot_raw', 0.0)):.3f}/{float(rollout_stats.get('mean_phase0_owner_change_ratio_vs_snapshot_executed', 0.0)):.3f}"
                f" | ph0_owner_budget(allow/clip)={float(rollout_stats.get('mean_phase0_owner_change_budget_allowed', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phase0_owner_change_budget_clipped_ratio', 0.0)):.3f}"
                f" | ph0_owner_fb0={float(rollout_stats.get('mean_phase0_owner_fallback_to_candidate0_ratio', 0.0)):.3f}"
                f" | ph0_owner_invalid={float(rollout_stats.get('mean_phase0_owner_invalid_option_ratio', 0.0)):.3f}"
                f" | ph0_owner_null={float(rollout_stats.get('mean_phase0_owner_null_selected_ratio', 0.0)):.3f}"
                f" | ph0_owner_restore={float(rollout_stats.get('mean_phase0_owner_restored_to_snapshot_ratio', 0.0)):.3f}"
                f" | ph0_owner_replace_non_snapshot={float(rollout_stats.get('mean_phase0_owner_replaced_with_non_snapshot_ratio', 0.0)):.3f}"
                f" | ph0_owner_invalid_to_null={float(rollout_stats.get('mean_phase0_owner_invalid_to_null_ratio', 0.0)):.3f}"
                f" | ph0_owner_invalid_to_snapshot={float(rollout_stats.get('mean_phase0_owner_invalid_to_snapshot_ratio', 0.0)):.3f}"
                f" | ph0_owner_invalid_to_non_snapshot={float(rollout_stats.get('mean_phase0_owner_invalid_to_non_snapshot_ratio', 0.0)):.3f}"
                f" | ph0_owner_raw(same/nonSnap/null)={float(rollout_stats.get('mean_ph0_owner_raw_same_as_snapshot_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_ph0_owner_raw_non_snapshot_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_ph0_owner_raw_null_ratio', 0.0)):.3f}"
                f" | ph0_owner_exec(same/nonSnap/revert)={float(rollout_stats.get('mean_ph0_owner_exec_same_as_snapshot_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_ph0_owner_exec_non_snapshot_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_ph0_owner_reverted_to_snapshot_ratio', 0.0)):.3f}"
                f" | owner_pol(ent/top1/snap/nonSnap)={float(rollout_stats.get('owner_policy_entropy_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_policy_top1_prob_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_policy_snapshot_prob_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_policy_non_snapshot_prob_mean', 0.0)):.3f}"
                f" | owner_opt(mean/non0/valid/snap/nonSnap)={float(rollout_stats.get('sampled_embb_owner_option_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('sampled_embb_owner_option_nonzero_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('sampled_embb_owner_option_valid_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('rollout_sampled_owner_option_snapshot_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('rollout_sampled_owner_option_non_snapshot_ratio', 0.0)):.3f}"
                f" | owner_effective_change={float(rollout_stats.get('mean_phase0_owner_changed_and_effective_ratio', 0.0)):.3f}"
                f" | embb_service_ratio={float(rollout_stats.get('mean_embb_service_ratio', 0.0)):.3f}"
                f" | embb_positive_rate_ratio={float(rollout_stats.get('mean_embb_positive_rate_ratio', 0.0)):.3f}"
                f" | phaseA_embb_pwr(raw/exe)={float(rollout_stats.get('phase_a_raw_embb_power_nonzero_ratio', 0.0)):.3f}/{float(rollout_stats.get('phase_a_executed_embb_power_nonzero_ratio', 0.0)):.3f}"
                f" | phaseA_embb_delta(raw/exe)={float(rollout_stats.get('phase_a_raw_embb_power_mean_delta', 0.0)):.3f}/{float(rollout_stats.get('phase_a_executed_embb_power_mean_delta', 0.0)):.3f}"
                f" | phaseA_embb_clip={float(rollout_stats.get('phase_a_embb_power_clip_ratio', 0.0)):.3f}"
                f" | raw_exec_gap={float(rollout_stats.get('raw_executed_any_gap_ratio', 0.0)):.3f}"
                f" | autonomy={float(rollout_stats.get('policy_autonomy_ratio', 0.0)):.3f}"
                f" | autonomy_excl_pwr={float(rollout_stats.get('policy_autonomy_excluding_power_projection', 0.0)):.3f}"
                f" | phaseA_write={float(rollout_stats.get('mean_phase_a_embb_power_write_ratio', 0.0)):.3f}"
                f" | owner_raw_churn_pen={float(rollout_stats.get('reward_term_terminal_owner_raw_churn_penalty', 0.0)):.3f}"
                f" | owner_drop_churn={float(rollout_stats.get('owner_dropped_raw_churn_ratio', 0.0)):.3f}"
                f" | owner_pos_cand={float(rollout_stats.get('phase0_owner_candidate_positive_objective_ratio', 0.0)):.3f}"
                f" | owner_pos_acc={float(rollout_stats.get('phase0_owner_accepted_positive_objective_ratio', 0.0)):.3f}"
                f" | owner_relax={float(rollout_stats.get('owner_candidate_relaxed_ratio', 0.0)):.3f}"
                f" | owner_fb={float(rollout_stats.get('owner_candidate_fallback_used_ratio', 0.0)):.3f}"
                f" | owner_obj(pre/post)={float(rollout_stats.get('owner_objective_gain_pre_filter_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_objective_gain_post_filter_mean', 0.0)):.3f}"
                f" | owner_obj(mu/std/thr)={float(rollout_stats.get('owner_obj_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_obj_std', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_gate_threshold', 0.0)):.3f}"
                f" | owner_after_gate={float(rollout_stats.get('owner_candidate_after_gate_ratio', 0.0)):.3f}"
                f" | owner_neg_acc={float(rollout_stats.get('owner_negative_but_accepted_ratio', 0.0)):.3f}"
                f" | owner_neg_acc_when_pos={float(rollout_stats.get('owner_neg_accepted_with_positive_candidate_ratio', 0.0)):.3f}"
                f" | owner_neg_clip={float(rollout_stats.get('owner_neg_accept_clipped_ratio', 0.0)):.3f}"
                f" | owner_neg_quota_rej={float(rollout_stats.get('owner_neg_rejected_by_quota_ratio', 0.0)):.3f}"
                f" | owner_pos_sel={float(rollout_stats.get('owner_pos_selected_ratio', 0.0)):.3f}"
                f" | owner_sel(pos/neg)={float(rollout_stats.get('owner_final_pos_selected_count', rollout_stats.get('owner_pos_selected_count', 0.0))):.2f}/"
                f"{float(rollout_stats.get('owner_final_neg_selected_count', rollout_stats.get('owner_neg_selected_count', 0.0))):.2f}"
                f" | owner_sel/allow={float(rollout_stats.get('owner_selected_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_allowed_k', 0.0)):.2f}"
                f" | owner_final(sel/pos/neg/sr/keep)={float(rollout_stats.get('owner_final_selected_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_final_pos_selected_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_final_neg_selected_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_final_safe_relax_selected_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_final_keep_set_size', 0.0)):.2f}"
                f" | owner_fill={float(rollout_stats.get('owner_selection_fill_ratio', 0.0)):.3f}"
                f" | owner_pos_short={float(rollout_stats.get('owner_positive_shortage_ratio', 0.0)):.3f}"
                f" | owner_neg_blk={float(rollout_stats.get('owner_negative_blocked_due_to_quota_ratio', 0.0)):.3f}"
                f" | owner_safe_relax(used/cand/sel)={float(rollout_stats.get('owner_safe_relaxed_used_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_safe_relaxed_candidate_count', 0.0)):.2f}/"
                f"{float(rollout_stats.get('owner_safe_relaxed_selected_count', 0.0)):.2f}"
                f" | owner_safe_relax(obj/srv/intf)={float(rollout_stats.get('owner_safe_relaxed_avg_objective', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_safe_relaxed_service_delta_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('owner_safe_relaxed_intercell_delta_mean', 0.0)):.3f}"
                f" | owner_obj_near0={float(rollout_stats.get('owner_near_zero_objective_ratio', 0.0)):.3f}"
                f" | owner_pos_after_relax={float(rollout_stats.get('owner_positive_after_relax_ratio', 0.0)):.3f}"
                f" | owner_safe_relax_off={float(rollout_stats.get('owner_safe_relax_off_ratio', rollout_stats.get('owner_safe_relax_disabled_ratio', 0.0))):.3f}"
                f" | phaseA_pos={float(rollout_stats.get('phaseA_positive_ratio', 0.0)):.3f}"
                f" | phaseA_zero={float(rollout_stats.get('phaseA_zero_action_ratio', 0.0)):.3f}"
                f" | phaseA_delta(mu/p10/p50/p90)={float(rollout_stats.get('phaseA_delta_mean', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phaseA_delta_p10', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phaseA_delta_p50', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phaseA_delta_p90', 0.0)):.3f}"
                f" | phaseA_lt-0.9={float(rollout_stats.get('phaseA_delta_lt_neg09_ratio', 0.0)):.3f}"
                f" | phaseA_pen(l2/sat/svc)={float(rollout_stats.get('phaseA_power_reduction_l2_penalty', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phaseA_power_saturation_penalty', 0.0)):.3f}/"
                f"{float(rollout_stats.get('embb_service_floor_hinge_penalty', 0.0)):.3f}"
                f" | phaseA_eff_floor_pen={float(rollout_stats.get('reward_term_terminal_phase_a_effective_nonzero_floor_penalty', 0.0)):.3f}"
                f" | phaseA_abs_exec={float(rollout_stats.get('phaseA_executed_abs_delta_mean', 0.0)):.3f}"
            )
            msg += (
                f" | phaseA_pow_path(pre/clip/quant/proj/final)="
                f"{float(rollout_stats.get('phase_a_embb_power_pre_clip_mean_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_post_clip_mean_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_post_quant_mean_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_post_projection_mean_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_final_executed_mean_delta', 0.0)):.3f}"
                f" | phaseA_pow_ratios(clip/quant/proj/cap/floor)="
                f"{float(rollout_stats.get('phase_a_embb_power_clip_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_quantized_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_projection_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_cap_hit_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_floor_hit_ratio', 0.0)):.3f}"
                f" | phaseA_pow_abs(raw/exe)="
                f"{float(rollout_stats.get('phase_a_embb_power_mean_abs_raw_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_mean_abs_executed_delta', 0.0)):.3f}"
                f" | phaseA_pow_bind(floor/cap)={float(rollout_stats.get('phase_a_embb_power_floor_binding_strength', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_cap_binding_strength', 0.0)):.3f}"
                f" | phaseA_pow_proj_delta(l1/l2)={float(rollout_stats.get('phase_a_embb_power_proj_delta_l1', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_proj_delta_l2', 0.0)):.3f}"
                f" | phaseA_pow_floorcap_delta={float(rollout_stats.get('phase_a_embb_power_pre_to_floor_delta', 0.0)):.3f}/"
                f"{float(rollout_stats.get('phase_a_embb_power_pre_to_cap_delta', 0.0)):.3f}"
                f" | phaseA_pow_final_minus_proj={float(rollout_stats.get('phase_a_embb_power_final_minus_proj_delta', 0.0)):.3f}"
                f" | phaseA_pow_shrink={float(rollout_stats.get('phase_a_embb_power_abs_shrink_ratio', 0.0)):.3f}"
                f" | phaseA_pow_signflip={float(rollout_stats.get('phase_a_embb_power_sign_flip_ratio', 0.0)):.3f}"
                f" | phaseA_pow_anchor_bind={float(rollout_stats.get('phase_a_embb_power_anchor_binding_ratio', 0.0)):.3f}"
                f" | phaseA_pow_eff_nz={float(rollout_stats.get('phase_a_embb_power_effective_nonzero_ratio', 0.0)):.3f}"
            )
            msg += (
                f" | phaseA_zero(inactive/keep/noCand/invOwner)="
                f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_inactive_head_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_keep_mode_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_no_candidate_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_invalid_owner_ratio', 0.0)):.3f}"
            )
            msg += (
                f" | phaseA_keep(gate/try/succ/noOwner/projBlk)="
                f"{float(rollout_stats.get('mean_phaseA_zero_by_keep_due_to_mode_gate_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phaseA_keep_power_write_attempt_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phaseA_keep_power_write_success_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phaseA_power_write_blocked_no_owner_ratio', 0.0)):.3f}/"
                f"{float(rollout_stats.get('mean_phaseA_power_write_blocked_projection_ratio', 0.0)):.3f}"
            )
            msg += (
                f" | rollout_tpW={float(rollout_stats.get('mean_throughput_per_watt', 0.0)) / 1.0e6:.3f} Mb/W"
                f" | rollout_served_rate={float(rollout_stats.get('mean_avg_throughput_per_served_embb_user', 0.0)) / 1.0e6:.3f} Mbps"
                f" | term_dbg(rate/base/gain/w)="
                f"{float(rollout_stats.get('terminal_debug_embb_total_rate', 0.0)) / 1.0e6:.3f}/"
                f"{float(rollout_stats.get('terminal_debug_greedy_embb_total_rate', 0.0)) / 1.0e6:.3f}/"
                f"{float(rollout_stats.get('terminal_debug_embb_rate_gain_vs_greedy', 0.0)):.3f}/"
                f"{float(rollout_stats.get('terminal_debug_embb_rate_gain_weight', 0.0)):.3f}"
                f" | phaseA_pwr_active={int(bool(record.get('control', {}).get('phase_a_embb_power_runtime_enabled', False)))}"
                f" | phaseA_pwr_anchor={int(bool(record.get('control', {}).get('phase_a_embb_power_anchor_enabled', False)))}"
            )
            if latest_eval is not None:
                if 'mean_rate_ratio' in latest_eval:
                    msg += (
                        f" | eval_rate_ratio={float(latest_eval.get('mean_rate_ratio', 0.0)):.4f}"
                        f" | eval_power_ratio={float(latest_eval.get('mean_power_ratio', 1.0)):.4f}"
                        f" | eval_admission_gap={float(latest_eval.get('mean_admission_gap', 0.0)):.4f}"
                    )
                else:
                    msg += (
                        f" | eval_policy_score={float(latest_eval.get('policy_score', 0.0)):.4f}"
                        f" | eval_policy_embb_rate={float(latest_eval.get('policy_mean_embb_rate', 0.0)) / 1.0e6:.3f} Mbps"
                        f" | eval_policy_user_rate={float(latest_eval.get('policy_mean_embb_user_rate', 0.0)) / 1.0e6:.3f} Mbps"
                        f" | eval_policy_admission={float(latest_eval.get('policy_mean_scheduled_ratio', 0.0)):.4f}"
                        f" | eval_policy_power={float(latest_eval.get('policy_mean_power', 0.0)):.3f}"
                        f" | eval_tpW={float(latest_eval.get('policy_mean_throughput_per_watt', 0.0)) / 1.0e6:.3f} Mb/W"
                        f" | eval_phaseA_count={float(latest_eval.get('policy_mean_phase_a_embb_power_changed_count', 0.0)):.3f}"
                        f" | eval_phaseA_pwr={float(latest_eval.get('policy_mean_phase_a_embb_power_changed_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_raw={float(latest_eval.get('policy_mean_phase_a_embb_power_mean_raw_delta', 0.0)):.3f}"
                        f" | eval_phaseA_delta={float(latest_eval.get('policy_mean_phase_a_embb_power_mean_executed_delta', 0.0)):.3f}"
                        f" | eval_phaseA_path(pre/clip/proj/final)={float(latest_eval.get('policy_mean_phase_a_embb_power_pre_clip_mean_delta', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_post_clip_mean_delta', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_post_projection_mean_delta', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_final_executed_mean_delta', 0.0)):.3f}"
                        f" | eval_phaseA_ratios(clip/proj/cap/floor)={float(latest_eval.get('policy_mean_phase_a_embb_power_clip_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_projection_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_cap_hit_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_floor_hit_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_anchor_bind={float(latest_eval.get('policy_mean_phase_a_embb_power_anchor_binding_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_eff_nz={float(latest_eval.get('policy_mean_phase_a_embb_power_effective_nonzero_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_write={float(latest_eval.get('policy_mean_phase_a_embb_power_write_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_zero_keep={float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_keep_mode_ratio', 0.0)):.3f}"
                        f" | eval_owner(raw/exe)={float(latest_eval.get('policy_mean_phase0_owner_change_ratio_vs_snapshot_raw', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase0_owner_change_ratio_vs_snapshot_executed', 0.0)):.3f}"
                        f" | eval_owner_budget(allow/clip)={float(latest_eval.get('policy_mean_phase0_owner_change_budget_allowed', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase0_owner_change_budget_clipped_ratio', 0.0)):.3f}"
                        f" | eval_owner_non_snapshot(raw/exe)={float(latest_eval.get('policy_mean_ph0_owner_raw_non_snapshot_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_ph0_owner_exec_non_snapshot_ratio', 0.0)):.3f}"
                        f" | eval_owner_restore={float(latest_eval.get('policy_mean_phase0_owner_restored_to_snapshot_ratio', 0.0)):.3f}"
                        f" | eval_owner_keep_null={float(latest_eval.get('policy_mean_phase0_owner_kept_null_ratio', 0.0)):.3f}"
                        f" | eval_owner_replace={float(latest_eval.get('policy_mean_phase0_owner_replaced_with_non_snapshot_ratio', 0.0)):.3f}"
                        f" | eval_owner_invalid_to_snapshot={float(latest_eval.get('policy_mean_phase0_owner_invalid_to_snapshot_ratio', 0.0)):.3f}"
                        f" | eval_owner_invalid_to_non_snapshot={float(latest_eval.get('policy_mean_phase0_owner_invalid_to_non_snapshot_ratio', 0.0)):.3f}"
                        f" | eval_owner_nn(raw/exe)={float(latest_eval.get('policy_mean_phase0_owner_non_null_ratio_raw', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase0_owner_non_null_ratio_executed', 0.0)):.3f}"
                        f" | eval_owner_effective={float(latest_eval.get('policy_mean_phase0_owner_changed_and_effective_ratio', 0.0)):.3f}"
                        f" | eval_embb_pos={float(latest_eval.get('policy_mean_embb_positive_rate_ratio', 0.0)):.3f}"
                        f" | eval_embb_srv={float(latest_eval.get('policy_mean_embb_service_ratio', 0.0)):.3f}"
                        f" | eval_embb_min={float(latest_eval.get('policy_mean_embb_min_rate_satisfaction_ratio', 0.0)):.3f}"
                        f" | eval_served_embb={float(latest_eval.get('policy_mean_embb_served_user_count', 0.0)):.2f}"
                        f" | eval_srv_floor_pen={float(latest_eval.get('policy_mean_terminal_embb_service_floor_penalty', 0.0)):.3f}"
                        f" | eval_min_floor_pen={float(latest_eval.get('policy_mean_terminal_embb_min_rate_floor_penalty', 0.0)):.3f}"
                        f" | eval_adm_over_srv_pen={float(latest_eval.get('policy_mean_urllc_admission_over_service_tradeoff_penalty', 0.0)):.3f}"
                        f" | eval_owner_srv_gain={float(latest_eval.get('policy_mean_phase0_owner_effective_service_gain_ratio', 0.0)):.3f}"
                        f" | eval_owner_rate_gain={float(latest_eval.get('policy_mean_phase0_owner_effective_rate_gain_vs_snapshot_mean', 0.0)):.3f}"
                        f" | eval_phaseA_sat={float(latest_eval.get('policy_mean_phase_a_embb_power_raw_saturation_ratio', 0.0)):.3f}"
                        f" | eval_phaseA_final_std={float(latest_eval.get('policy_mean_phase_a_embb_power_final_std', 0.0)):.4f}"
                        f" | eval_ph0_pwr_chg={float(latest_eval.get('policy_mean_planning_embb_power_changed_ratio', 0.0)):.3f}"
                        f" | eval_tp_per_served={float(latest_eval.get('policy_mean_avg_throughput_per_served_embb_user', 0.0)) / 1.0e6:.3f} Mbps"
                        f" | eval_snapshot_leak={float(latest_eval.get('policy_mean_owner_snapshot_leak_detected', 0.0)):.3f}"
                    )
                    if 'policy_score_vs_original_greedy' in latest_eval:
                        msg += (
                            f" | eval_vs_orig={float(latest_eval.get('policy_score_vs_original_greedy', 0.0)):.4f}"
                            f" | eval_vs_matched={float(latest_eval.get('policy_score_vs_matched_greedy', 0.0)):.4f}"
                            f" | eval_vs_throughput_feasible={float(latest_eval.get('policy_score_vs_throughput_feasible_oracle', 0.0)):.4f}"
                            f" | eval_vs_throughput_only={float(latest_eval.get('policy_score_vs_throughput_only_greedy', 0.0)):.4f}"
                            f" | eval_vs_channel={float(latest_eval.get('policy_score_vs_channel_only_greedy', 0.0)):.4f}"
                        )
            if 'evaluation_kind' in record:
                msg += (
                    f" | eval_kind={record['evaluation_kind']}"
                    f" | eval_loads={record.get('evaluation_loads', [])}"
                    f" | eval_eps={int(record.get('evaluation_episodes_per_load', 0) or 0)}"
                    f" | eval_modes={record.get('evaluation_compare_modes', [])}"
                )
            checkpoint_metrics = dict(record.get('checkpoint_metrics', {}) or {})
            if checkpoint_metrics:
                msg += (
                    f" | tp_score={float(checkpoint_metrics.get('throughput_score', 0.0)) / 1.0e6:.3f} Mbps"
                    f" | tp_best={float(best_throughput_score) / 1.0e6 if np.isfinite(best_throughput_score) else float('nan'):.3f} Mbps"
                    f" | tp_updated={int(checkpoint_metrics.get('throughput_updated', 0.0))}"
                    f" | bal_score={float(checkpoint_metrics.get('balanced_score', 0.0)):.4f}"
                    f" | bal_updated={int(checkpoint_metrics.get('balanced_updated', 0.0))}"
                )
            msg += (
                f" | actor_lr={float(actor_lr):.2e}"
                f" | critic_lr={float(critic_lr):.2e}"
                f" | entropy={float(entropy_coef):.4f}"
                f" | clip={float(clip_ratio):.3f}"
            )
            print(msg, flush=True)
            owner_decode_examples = list(rollout_stats.get("owner_decode_debug_examples", []) or [])
            if (
                float(rollout_stats.get("owner_policy_non_snapshot_prob_mean", 0.0)) > 0.2
                and float(rollout_stats.get("mean_ph0_owner_raw_non_snapshot_ratio", 0.0)) <= 1.0e-9
                and owner_decode_examples
            ):
                print(
                    "[SR-MAPPO][owner-debug] first10(snapshot_owner_id, sampled_embb_owner_option, decoded_raw_owner_id, valid_owner_mask)="
                    + str(owner_decode_examples[:10]),
                    flush=True,
                )
            if latest_eval is not None:
                eval_snapshot_leak = float(latest_eval.get('policy_mean_owner_snapshot_leak_detected', 0.0))
                if eval_snapshot_leak > 1.0e-6:
                    print(
                        f"[{timestamp}] [SR-MAPPO][WARN] Snapshot leakage detected in owner pipeline during eval: "
                        f"policy_mean_owner_snapshot_leak_detected={eval_snapshot_leak:.3f}",
                        flush=True,
                    )
                sat = float(latest_eval.get('policy_mean_phase_a_embb_power_raw_saturation_ratio', 0.0))
                final_std = float(latest_eval.get('policy_mean_phase_a_embb_power_final_std', 0.0))
                diversity = float(latest_eval.get('policy_mean_phase_a_embb_power_cellwise_diversity', 0.0))
                if sat > 0.80 and final_std < 0.02:
                    print(
                        f"[{timestamp}] [SR-MAPPO][WARN] Phase-A power head is saturated and projection collapses diversity. "
                        f"raw_saturation_ratio={sat:.3f}, final_delta_std={final_std:.4f}, cellwise_diversity={diversity:.4f}",
                        flush=True,
                    )
                eval_phase_a_raw_nz = float(latest_eval.get('policy_mean_phase_a_raw_embb_power_nonzero_ratio', 0.0))
                eval_phase_a_exe_nz = float(latest_eval.get('policy_mean_phase_a_executed_embb_power_nonzero_ratio', 0.0))
                if eval_phase_a_raw_nz > 0.50 and eval_phase_a_exe_nz < 0.05:
                    print(
                        f"[{timestamp}] [SR-MAPPO][WARN] Phase-A power is being suppressed after raw policy output. "
                        f"raw_nonzero_ratio={eval_phase_a_raw_nz:.3f}, executed_nonzero_ratio={eval_phase_a_exe_nz:.3f} | "
                        f"suppress_reason(inactive/no_embb/no_owner/invalid_owner/cap/floor/unknown)="
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_inactive_head_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_no_embb_active_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_no_owner_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_invalid_owner_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_cap_projection_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_floor_projection_ratio', 0.0)):.3f}/"
                        f"{float(latest_eval.get('policy_mean_phase_a_embb_power_zeroed_unknown_ratio', 0.0)):.3f}",
                        flush=True,
                    )
                eval_owner_effective = float(latest_eval.get('policy_mean_phase0_owner_changed_and_effective_ratio', 0.0))
                eval_tp_per_served = float(latest_eval.get('policy_mean_avg_throughput_per_served_embb_user', 0.0))
                if eval_owner_effective > 0.5 and eval_tp_per_served < 1.0e6:
                    print(
                        f"[{timestamp}] [SR-MAPPO][WARN] Owner is changing aggressively but not creating useful throughput gains: "
                        f"eval_owner_effective={eval_owner_effective:.3f}, "
                        f"eval_avg_throughput_per_served_embb_user={eval_tp_per_served / 1.0e6:.3f} Mbps",
                        flush=True,
                    )
            if bool(record.get('control', {}).get('phase_a_embb_power_runtime_enabled', False)) and float(
                rollout_stats.get('phase_a_executed_embb_power_nonzero_ratio', 0.0)
            ) <= 1.0e-9:
                print(
                    f"[{timestamp}] [SR-MAPPO][WARN] Phase-A eMBB power runtime enabled but rollout executed no nonzero Phase-A eMBB power changes. "
                    f"zeroed(inactive/keep/no_candidate/invalid_owner)="
                    f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_inactive_head_ratio', 0.0)):.3f}/"
                    f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_keep_mode_ratio', 0.0)):.3f}/"
                    f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_no_candidate_ratio', 0.0)):.3f}/"
                    f"{float(rollout_stats.get('mean_phase_a_embb_power_zeroed_invalid_owner_ratio', 0.0)):.3f}",
                    flush=True,
                )
            shield_correction_ratio = float(rollout_stats.get('shield_correction_ratio', 0.0))
            raw_vs_executed_mode_gap = float(rollout_stats.get('raw_vs_executed_mode_gap', 0.0))
            policy_autonomy_ratio = float(rollout_stats.get('policy_autonomy_ratio', 0.0))
            if shield_correction_ratio > 0.15 or raw_vs_executed_mode_gap > 0.15:
                warning = (
                    "Phase-A training-time action rewriting is high: "
                    f"shield_correction_ratio={shield_correction_ratio:.3f}, "
                    f"raw_vs_executed_mode_gap={raw_vs_executed_mode_gap:.3f}, "
                    f"mode_correction_ratio={float(rollout_stats.get('mode_correction_ratio', 0.0)):.3f}, "
                    f"joint_reliability_rewrite_ratio={float(rollout_stats.get('joint_reliability_rewrite_ratio', 0.0)):.3f}, "
                    f"packet_invalid_ratio={float(rollout_stats.get('packet_invalid_ratio', 0.0)):.3f}, "
                    f"mask_invalid_ratio={float(rollout_stats.get('mask_invalid_ratio', 0.0)):.3f}"
                )
                record.setdefault('warnings', []).append(warning)
                print(f"[{timestamp}] [SR-MAPPO][WARN] {warning}", flush=True)
            if policy_autonomy_ratio < 0.30:
                warning = (
                    "Policy autonomy is low: "
                    f"policy_autonomy_ratio={policy_autonomy_ratio:.3f} (target >= 0.30) | "
                    f"raw_exec_gap={float(rollout_stats.get('raw_executed_any_gap_ratio', 0.0)):.3f} | "
                    f"shield_correction_ratio={shield_correction_ratio:.3f}"
                )
                record.setdefault('warnings', []).append(warning)
                print(f"[{timestamp}] [SR-MAPPO][WARN] {warning}", flush=True)
            _trainer_timing_log(
                cfg,
                f"iter={iteration} eval_kind={record.get('evaluation_kind', 'none')} "
                f"loads={record.get('evaluation_loads', [])} "
                f"eps={int(record.get('evaluation_episodes_per_load', 0) or 0)} "
                f"compare_modes={record.get('evaluation_compare_modes', [])} "
                f"rollout={rollout_sec:.3f}s update={update_sec:.3f}s "
                f"eval={eval_sec:.3f}s ckpt={checkpoint_sec:.3f}s total={iteration_total_sec:.3f}s | "
                f"collect(obs={float(rollout_stats.get('obs_build_sec', 0.0)):.3f}s, "
                f"forward={float(rollout_stats.get('policy_forward_sec', 0.0)):.3f}s, "
                f"greedy_bc={float(rollout_stats.get('greedy_bc_target_sec', 0.0)):.3f}s, "
                f"env={float(rollout_stats.get('env_step_sec', 0.0)):.3f}s, "
                f"buffer={float(rollout_stats.get('buffer_add_sec', 0.0)):.3f}s) | "
                f"env(obs={float(rollout_stats.get('env_build_observations_total_sec', 0.0)):.3f}s, "
                f"step={float(rollout_stats.get('env_step_total_sec', 0.0)):.3f}s)"
            )

    if best_summary is None and best_multiload_frontier_summary is not None:
        best_summary = best_multiload_frontier_summary
    if best_summary is None and best_balanced_summary is not None:
        best_summary = best_balanced_summary
    if best_summary is None and best_throughput_summary is not None:
        best_summary = best_throughput_summary

    final_extra = {
        'history': history,
        'bc': bc_stats,
        'best': best_summary,
        'best_reward': best_reward_summary,
        'best_throughput': best_throughput_summary,
        'best_balanced': best_balanced_summary,
        'best_service_interference_balanced': best_service_interference_balanced_summary,
        'best_service_power_interference_balanced': best_service_power_interference_balanced_summary,
        'best_service_gain_interference_balanced': best_service_gain_interference_balanced_summary,
        'best_v6_balanced_puncture_accounting': best_v6_balanced_puncture_accounting_summary,
        'best_vs_original_greedy': best_vs_original_summary,
        'best_vs_matched_greedy': best_vs_matched_summary,
        'best_vs_throughput_feasible_oracle': best_vs_throughput_feasible_summary,
        'best_vs_throughput_only_greedy': best_vs_throughput_only_summary,
        'best_vs_channel_only_greedy': best_vs_channel_only_summary,
        'best_floor_throughput': best_floor_throughput_summary,
        'best_multiload_frontier': best_multiload_frontier_summary,
        'best_multiload_tp_power': best_multiload_frontier_summary,
    }
    trainer.save_checkpoint(checkpoint_dir / f"{cfg.training.run_name}_final.pt", extra=final_extra)
    return {
        'bc': bc_stats,
        'history': history,
        'best': best_summary,
        'best_balanced': best_balanced_summary,
        'best_service_interference_balanced': best_service_interference_balanced_summary,
        'best_service_power_interference_balanced': best_service_power_interference_balanced_summary,
        'best_service_gain_interference_balanced': best_service_gain_interference_balanced_summary,
        'best_v6_balanced_puncture_accounting': best_v6_balanced_puncture_accounting_summary,
        'best_floor_throughput': best_floor_throughput_summary,
        'best_multiload_frontier': best_multiload_frontier_summary,
        'best_multiload_tp_power': best_multiload_frontier_summary,
        'best_vs_throughput_feasible_oracle': best_vs_throughput_feasible_summary,
        'checkpoint_dir': str(checkpoint_dir.resolve()),
    }


if __name__ == '__main__':
    import argparse

    from .config import SRMAPPOConfig
    from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset

    parser = argparse.ArgumentParser(description="Train SR-MAPPO.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override total training iterations.",
    )
    args = parser.parse_args()

    cfg = SRMAPPOConfig()
    if args.experiment:
        cfg = apply_experiment_preset(cfg, args.experiment)
    if args.iterations is not None:
        cfg.training.total_iterations = int(max(args.iterations, 1))

    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    run_training_loop(cfg, resume_path=resume_path)

