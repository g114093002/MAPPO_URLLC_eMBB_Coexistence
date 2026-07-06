"""Unified baseline runner for greedy, random, MAPPO, and IPPO comparisons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import _bootstrap  # noqa: F401
from .compare import _build_main_like_configs
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .evaluate import (
    _hard_feasible_throughput_actions,
    _myopic_throughput_actions,
    _policy_actions,
    _run_original_greedy_episode,
    _run_original_greedy_normal_v1_episode,
    _run_original_greedy_normal_v2_episode,
    _throughput_only_actions,
)
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, AgentObservation, CandidatePacket, HybridAction, ShieldedAction

if TYPE_CHECKING:
    import torch
    from .ippo_baseline import IPPOBaselineTrainer
    from .networks import SRMAPPOActorCritic


SUPPORTED_POLICIES = {
    "random_scheduler",
    "naive_random",
    "pure_puncturing",
    "pure_superposition",
    "greedy",
    "myopic_throughput_greedy",
    "hard_feasible_throughput_greedy",
    "throughput_only_greedy",
    "global_frontier_greedy",
    "original",
    "original_greedy_normal_v1",
    "original_greedy_normal_v2",
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
        cfg = deepcopy(config)
        partial_reuse_enabled = str(os.getenv("SR_MAPPO_ENABLE_PARTIAL_REUSE", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if partial_reuse_enabled:
            cfg.env.apply_partial_reuse_to_fixed_baseline_state = True
        return cfg
    cfg = SRMAPPOConfig()
    partial_reuse_enabled = str(os.getenv("SR_MAPPO_ENABLE_PARTIAL_REUSE", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if partial_reuse_enabled:
        cfg.env.apply_partial_reuse_to_fixed_baseline_state = True
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


def _ensure_hard_feasible_phase0_minrate_policy(
    config_dict: Dict[str, object],
    cfg: SRMAPPOConfig,
) -> None:
    env_overrides = dict(config_dict.get("env", {}) or {})
    # Keep each UE attached to a single serving UAV, then allocate RBs with a
    # min-rate-first greedy owner policy inside that fixed association.
    env_overrides.setdefault("fixed_embb_baseline_policy", "per_uav_minrate_optimizer")
    env_overrides.setdefault("phase0_embb_baseline_minrate_first", True)
    env_overrides.setdefault("phase0_embb_uav_assignment_mode", "throughput")
    config_dict["env"] = env_overrides
    if hasattr(cfg, "env"):
        if not getattr(cfg.env, "fixed_embb_baseline_policy", ""):
            cfg.env.fixed_embb_baseline_policy = "per_uav_minrate_optimizer"
        if str(getattr(cfg.env, "fixed_embb_baseline_policy", "") or "").strip().lower() == "global_sumrate_only":
            cfg.env.fixed_embb_baseline_policy = "per_uav_minrate_optimizer"
        cfg.env.phase0_embb_baseline_minrate_first = True
        cfg.env.phase0_embb_uav_assignment_mode = "throughput"


def _build_env(cfg: SRMAPPOConfig, component_overrides: Optional[Dict[str, Dict[str, object]]] = None) -> SRMAPPOPhaseAEnv:
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()
    overrides = dict(component_overrides or {})
    system_overrides = dict(overrides.get("system", {}) or {})
    channel_uses_override = system_overrides.pop("channel_uses_per_minislot", None)
    permissive_system_override_keys = {
        "nested_load_from_max_users_enabled",
        "nested_load_max_total_users",
        "nested_load_max_embb_users",
        "nested_load_max_urllc_users",
        "force_serving_hints_association",
    }
    for name, target in (
        ("system", sys_cfg),
        ("urllc", urllc_cfg),
        ("embb", embb_cfg),
        ("algorithm", algo_cfg),
        ("simulation", sim_cfg),
    ):
        source_overrides = system_overrides if name == "system" else dict(overrides.get(name, {}) or {})
        for key, value in source_overrides.items():
            if hasattr(target, key) or (name == "system" and key in permissive_system_override_keys):
                setattr(target, key, value)
    if hasattr(sys_cfg, "refresh_derived_params"):
        sys_cfg.refresh_derived_params()
    if channel_uses_override is not None:
        sys_cfg.channel_uses_per_minislot = int(max(channel_uses_override, 1))
    sim_cfg.verbose = False
    sim_cfg.plot_results = False
    return SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)


def _baseline_candidate_pool_upper_bound(
    component_overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> int:
    sys_cfg, _urllc_cfg, _embb_cfg, _algo_cfg, _sim_cfg = _build_main_like_configs()
    overrides = dict(component_overrides or {})
    system_overrides = dict(overrides.get("system", {}) or {})
    for key, value in system_overrides.items():
        if hasattr(sys_cfg, key):
            setattr(sys_cfg, key, value)
    if hasattr(sys_cfg, "refresh_derived_params"):
        sys_cfg.refresh_derived_params()
    num_urllc_users = int(max(getattr(sys_cfg, "num_urllc_users", 0) or 0, 0))
    num_minislots = int(max(getattr(sys_cfg, "num_minislots", 1) or 1, 1))
    return int(max(num_urllc_users * num_minislots, 1))


def _enable_phase_a_joint_minrate_protection(cfg: SRMAPPOConfig) -> None:
    cfg.env.phase_a_protect_phase0_satisfied_embb_users = True
    cfg.shield.apply_joint_minrate_rewrite = True


def _configure_eval_env(
    env: SRMAPPOPhaseAEnv,
    *,
    total_load: Optional[float],
    mix_ratio: Optional[float],
    explicit_mix_weights: Optional[Tuple[float, float]] = None,
) -> float:
    from .trainer import configure_env_for_users_per_uav

    if mix_ratio is not None:
        env.sim_cfg.urllc_user_ratio = float(np.clip(mix_ratio, 0.0, 0.95))
    if explicit_mix_weights is not None:
        env.sim_cfg.explicit_user_mix_weights = (
            float(explicit_mix_weights[0]),
            float(explicit_mix_weights[1]),
        )
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


def _greedy_gate_mode_packet_pairs(
    env: SRMAPPOPhaseAEnv,
    obs: AgentObservation,
    *,
    uav_idx: int,
    rb_idx: int,
    minislot_idx: int,
    overlay_retention_threshold: float,
    puncture_gain_margin_bps: float,
) -> List[Tuple[int, int, CandidatePacket]]:
    packet_mask = np.asarray(obs.masks.packet_mask, dtype=float)
    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
    gated: List[Tuple[int, int, CandidatePacket]] = []
    current_global_embb_throughput = float(_current_global_embb_throughput(env))
    puncture_loss_history_bps = [
        float(x)
        for x in list(getattr(env, "_baseline_recent_puncture_losses_bps", []) or [])
        if float(x) >= 0.0
    ]
    puncture_loss_window = 10
    puncture_loss_alpha = 1.3
    recent_puncture_avg_bps = (
        float(np.mean(puncture_loss_history_bps[-puncture_loss_window:]))
        if len(puncture_loss_history_bps) >= puncture_loss_window
        else None
    )
    for packet_option, candidate in enumerate(obs.candidates, start=1):
        overlay_allowed = False
        puncture_allowed = False
        overlay_tp = None
        puncture_tp = None
        if (
            MODE_OVERLAY < mode_mask.size
            and packet_mask.ndim == 2
            and MODE_OVERLAY < packet_mask.shape[0]
            and packet_option < packet_mask.shape[1]
            and mode_mask[MODE_OVERLAY] > 0.5
            and packet_mask[MODE_OVERLAY, packet_option] > 0.5
            and candidate.is_mode_feasible(MODE_OVERLAY)
        ):
            overlay_tp = float(
                env._global_embb_throughput_if_apply_candidate_cell(
                    int(uav_idx),
                    int(rb_idx),
                    int(minislot_idx),
                    candidate,
                    MODE_OVERLAY,
                    0.0,
                )
            )
            overlay_retention = float(getattr(candidate, "overlay_retention", 0.0) or 0.0)
            if overlay_retention + 1.0e-12 >= float(overlay_retention_threshold):
                overlay_allowed = True
        if (
            MODE_PUNCTURE < mode_mask.size
            and packet_mask.ndim == 2
            and MODE_PUNCTURE < packet_mask.shape[0]
            and packet_option < packet_mask.shape[1]
            and mode_mask[MODE_PUNCTURE] > 0.5
            and packet_mask[MODE_PUNCTURE, packet_option] > 0.5
            and candidate.is_mode_feasible(MODE_PUNCTURE)
        ):
            puncture_tp = float(
                env._global_embb_throughput_if_apply_candidate_cell(
                    int(uav_idx),
                    int(rb_idx),
                    int(minislot_idx),
                    candidate,
                    MODE_PUNCTURE,
                    0.0,
                )
            )
            puncture_loss_bps = float(current_global_embb_throughput - float(puncture_tp))
            loss_within_dynamic_cap = True
            if recent_puncture_avg_bps is not None:
                dynamic_cap_bps = float(recent_puncture_avg_bps) * float(puncture_loss_alpha)
                loss_within_dynamic_cap = bool(float(puncture_loss_bps) <= float(dynamic_cap_bps) + 1.0e-12)
            if loss_within_dynamic_cap:
                if overlay_allowed and overlay_tp is not None:
                    if float(puncture_tp) >= float(overlay_tp) + float(puncture_gain_margin_bps) - 1.0e-12:
                        puncture_allowed = True
                else:
                    puncture_allowed = True
        if overlay_allowed:
            gated.append((int(packet_option), MODE_OVERLAY, candidate))
        if puncture_tp is not None and puncture_allowed:
            gated.append((int(packet_option), MODE_PUNCTURE, candidate))
    return gated


def _sanitize_puncture_loss_history(history_like: Iterable[float]) -> List[float]:
    return [float(x) for x in list(history_like or []) if float(x) >= 0.0]


def _puncture_dynamic_cap_from_history(history_like: Iterable[float]) -> Optional[float]:
    history = _sanitize_puncture_loss_history(history_like)
    window = 10
    alpha = 1.3
    if len(history) < window:
        return None
    return float(np.mean(np.asarray(history[-window:], dtype=float))) * float(alpha)


def _puncture_dynamic_cap_bps(env: SRMAPPOPhaseAEnv) -> Optional[float]:
    return _puncture_dynamic_cap_from_history(getattr(env, "_baseline_recent_puncture_losses_bps", []) or [])


def _append_puncture_loss_history(env: SRMAPPOPhaseAEnv, loss_bps: float) -> None:
    puncture_history = _sanitize_puncture_loss_history(getattr(env, "_baseline_recent_puncture_losses_bps", []) or [])
    if float(loss_bps) >= 0.0:
        puncture_history.append(float(loss_bps))
    if len(puncture_history) > 10:
        puncture_history = puncture_history[-10:]
    env._baseline_recent_puncture_losses_bps = puncture_history


def _resolved_puncture_exceeds_dynamic_cap(
    env: SRMAPPOPhaseAEnv,
    *,
    uav_idx: int,
    rb_idx: int,
    minislot_idx: int,
    candidate: CandidatePacket,
) -> bool:
    cap_bps = _puncture_dynamic_cap_bps(env)
    if cap_bps is None:
        return False
    before_bps = float(_current_global_embb_throughput(env))
    after_bps = float(
        env._global_embb_throughput_if_apply_candidate_cell(
            int(uav_idx),
            int(rb_idx),
            int(minislot_idx),
            candidate,
            MODE_PUNCTURE,
            0.0,
        )
    )
    puncture_loss_bps = float(before_bps - after_bps)
    return bool(puncture_loss_bps > float(cap_bps) + 1.0e-12)


def _resolved_puncture_cap_debug(
    env: SRMAPPOPhaseAEnv,
    *,
    uav_idx: int,
    rb_idx: int,
    minislot_idx: int,
    candidate: CandidatePacket,
) -> Tuple[Optional[float], float]:
    cap_bps = _puncture_dynamic_cap_bps(env)
    before_bps = float(_current_global_embb_throughput(env))
    after_bps = float(
        env._global_embb_throughput_if_apply_candidate_cell(
            int(uav_idx),
            int(rb_idx),
            int(minislot_idx),
            candidate,
            MODE_PUNCTURE,
            0.0,
        )
    )
    puncture_loss_bps = float(before_bps - after_bps)
    return cap_bps, puncture_loss_bps


def _mode_rank_tuple(
    env: SRMAPPOPhaseAEnv,
    global_embb_throughput: float,
    candidate: CandidatePacket,
    mode: int,
    *,
    puncturing_selection_rule: str = "max_embb_sum_rate",
    superposition_selection_rule: str = "max_embb_sum_rate",
) -> Tuple[float, ...]:
    # Baseline compare-time ranking is intentionally simple: once a candidate/mode
    # has already passed feasibility gates, prefer the action that leaves the
    # largest global eMBB throughput. Keep only deterministic tie-breakers for
    # exact-equality cases.
    return (
        float(global_embb_throughput),
        -float(int(candidate.packet_id)),
        -float(int(mode)),
    )


def _mode_matching_weight(
    global_embb_throughput: float,
) -> float:
    return float(global_embb_throughput)


def _select_best_mode_for_candidate(
    env: SRMAPPOPhaseAEnv,
    candidate: CandidatePacket,
    allowed_modes: Iterable[int],
    *,
    global_embb_evaluator: Callable[[CandidatePacket, int], float],
    puncturing_selection_rule: str = "max_embb_sum_rate",
    superposition_selection_rule: str = "max_embb_sum_rate",
) -> Optional[Tuple[int, Tuple[float, ...], float]]:
    best: Optional[Tuple[int, Tuple[float, ...], float]] = None
    for mode in allowed_modes:
        if not candidate.is_mode_feasible(int(mode)):
            continue
        global_embb_throughput = float(global_embb_evaluator(candidate, int(mode)))
        rank = _mode_rank_tuple(
            env,
            float(global_embb_throughput),
            candidate,
            int(mode),
            puncturing_selection_rule=puncturing_selection_rule,
            superposition_selection_rule=superposition_selection_rule,
        )
        weight = _mode_matching_weight(float(global_embb_throughput))
        if best is None or rank > best[1]:
            best = (int(mode), rank, float(weight))
    return best


def _minislot_global_matching_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    *,
    allowed_modes: Iterable[int],
    puncturing_selection_rule: str = "max_embb_sum_rate",
    superposition_selection_rule: str = "max_embb_sum_rate",
    overlay_retention_threshold: float = 0.0,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    if planning_phase:
        for agent_id, obs in observations.items():
            actions[agent_id] = _planning_baseline_action(env, obs)
        return actions

    current_minislot, current_rb = env._current_cell()
    for uav_idx, agent_id in enumerate(env.agent_ids):
        obs = observations[agent_id]
        feasible = _feasible_mode_packet_pairs(obs, allowed_modes)
        if float(overlay_retention_threshold) > 0.0:
            feasible = [
                (packet_option, mode, candidate)
                for packet_option, mode, candidate in feasible
                if int(mode) != MODE_OVERLAY
                or float(getattr(candidate, "overlay_retention", 0.0) or 0.0)
                >= float(overlay_retention_threshold) - 1.0e-12
            ]
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue

        best_action: Optional[HybridAction] = None
        best_rank: Optional[Tuple[float, ...]] = None
        for packet_option, mode, candidate in feasible:
            global_embb_throughput = float(
                env._global_embb_throughput_if_apply_candidate_cell(
                    int(uav_idx),
                    int(current_rb),
                    int(current_minislot),
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
                puncturing_selection_rule=puncturing_selection_rule,
                superposition_selection_rule=superposition_selection_rule,
            )
            if best_rank is None or tuple(rank) > tuple(best_rank):
                best_rank = tuple(rank)
                best_action = HybridAction(
                    mode=int(mode),
                    packet_option=int(packet_option),
                    power_delta=0.0,
                )
        actions[agent_id] = (
            best_action
            if best_action is not None
            else HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
        )
    return actions


def _keep_shielded_action() -> ShieldedAction:
    return ShieldedAction(
        action=HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
        candidate=None,
        utility=0.0,
    )


def _joint_resolved_global_embb_throughput(
    env: SRMAPPOPhaseAEnv,
    resolved: Dict[str, ShieldedAction],
    *,
    minislot: int,
    rb: int,
) -> float:
    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()
    try:
        for uav_idx, agent_id in enumerate(env.agent_ids):
            shielded = resolved.get(str(agent_id))
            if shielded is None or shielded.candidate is None:
                continue
            mode_int = int(shielded.action.mode)
            if mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
                continue
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(uav_idx),
                rb_idx=int(rb),
                minislot=int(minislot),
                candidate=shielded.candidate,
                mode_int=int(mode_int),
            )
        return float(_current_global_embb_throughput(env))
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


def _random_policy_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    rng: np.random.Generator,
) -> Tuple[Dict[str, HybridAction], Dict[str, ShieldedAction]]:
    actions: Dict[str, HybridAction] = {}
    resolved: Dict[str, ShieldedAction] = {}
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    current_minislot: Optional[int] = None
    current_rb: Optional[int] = None
    available_packet_ids: List[int] = []
    if not planning_phase:
        current_minislot, current_rb = env._current_cell()
        available_packet_ids = list(env._available_packet_ids(int(current_minislot)))
    for agent_id, obs in observations.items():
        if planning_phase:
            actions[agent_id] = _planning_baseline_action(env, obs)
            resolved[agent_id] = env._raw_action_to_shielded_action(actions[agent_id], obs)
            continue
        uav_idx, _rb_idx = env._agent_index_map[agent_id]
        if not available_packet_ids or current_minislot is None or current_rb is None:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            resolved[agent_id] = _keep_shielded_action()
            continue
        packet_id = int(available_packet_ids[int(rng.integers(0, len(available_packet_ids)))])
        mode = int(MODE_OVERLAY if float(rng.random()) < 0.5 else MODE_PUNCTURE)
        action = HybridAction(mode=int(mode), packet_option=0, power_delta=0.0)
        actions[agent_id] = action
        candidates = env._enumerate_candidates_for_cell(
            int(uav_idx),
            int(current_rb),
            int(current_minislot),
            packet_ids=[int(packet_id)],
        )
        candidate = next((item for item in candidates if int(item.packet_id) == int(packet_id)), None)
        if candidate is None or not bool(candidate.is_mode_feasible(int(mode))):
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            resolved[agent_id] = _keep_shielded_action()
        else:
            resolved[agent_id] = ShieldedAction(
                action=HybridAction(mode=int(mode), packet_option=0, power_delta=0.0),
                candidate=candidate,
                utility=float(candidate.utility_for_mode(int(mode))),
            )
    if not planning_phase and current_minislot is not None and current_rb is not None:
        if not env.rl_cfg.env.multi_rb_agents and bool(getattr(env.rl_cfg.shield, "apply_joint_reliability_rewrite", True)):
            resolved = env._enforce_joint_reliability(int(current_minislot), int(current_rb), observations, resolved)
        if not env.rl_cfg.env.multi_rb_agents and bool(getattr(env.rl_cfg.shield, "apply_joint_minrate_rewrite", False)):
            resolved = env._enforce_joint_minrate(int(current_minislot), int(current_rb), observations, resolved)
        resolved = env._sanitize_phase_a_embb_power_actions(
            resolved,
            minislot=int(current_minislot),
            rb=int(current_rb),
        )
    return actions, resolved


def _naive_random_action_and_resolution(
    env: SRMAPPOPhaseAEnv,
    obs: AgentObservation,
    rng: np.random.Generator,
) -> Tuple[HybridAction, ShieldedAction]:
    if not obs.candidates:
        keep = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
        return keep, env._raw_action_to_shielded_action(keep, obs)

    packet_option = int(rng.integers(1, len(obs.candidates) + 1))
    mode = int(MODE_OVERLAY if float(rng.random()) < 0.5 else MODE_PUNCTURE)
    action = HybridAction(mode=mode, packet_option=packet_option, power_delta=0.0)

    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
    packet_mask = np.asarray(obs.masks.packet_mask, dtype=float)
    candidate = obs.candidates[packet_option - 1]
    locally_feasible = bool(
        mode < mode_mask.size
        and mode_mask[mode] > 0.5
        and packet_mask.ndim == 2
        and mode < packet_mask.shape[0]
        and packet_option < packet_mask.shape[1]
        and packet_mask[mode, packet_option] > 0.5
        and candidate.is_mode_feasible(mode)
    )
    if not locally_feasible:
        keep = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
        return action, ShieldedAction(
            action=keep,
            candidate=None,
            utility=0.0,
            packet_invalid_fallback=True,
        )
    return action, env._raw_action_to_shielded_action(action, obs)


def _puncture_rank_key(
    env: SRMAPPOPhaseAEnv,
    candidate: CandidatePacket,
    rule: str,
) -> Tuple[float, ...]:
    rule = str(rule or "max_embb_sum_rate").strip().lower()
    reference_rate = _reference_rate(env)
    embb_sum_rate = float(reference_rate - candidate.puncture_loss)
    if rule in {"max_embb_sum_rate", "max_sum_rate", "sum_rate", "min_embb_rate_loss", "min_rate_loss", "min_loss"}:
        return (embb_sum_rate, -float(candidate.puncture_loss), float(candidate.puncture_reliability))
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
    return _minislot_global_matching_actions(
        env,
        observations,
        allowed_modes=(MODE_PUNCTURE,),
        puncturing_selection_rule=selection_rule,
    )


def _superposition_rank_key(
    env: SRMAPPOPhaseAEnv,
    candidate: CandidatePacket,
    rule: str,
) -> Tuple[float, ...]:
    rule = str(rule or "max_embb_sum_rate").strip().lower()
    reference_rate = _reference_rate(env)
    embb_sum_rate = float(reference_rate - candidate.overlay_loss)
    if rule in {"max_embb_sum_rate", "max_sum_rate", "sum_rate", "min_embb_degradation", "minimum_embb_degradation", "min_degradation"}:
        return (embb_sum_rate, -float(candidate.overlay_loss), float(candidate.overlay_retention))
    if rule in {"min_embb_degradation", "minimum_embb_degradation", "min_degradation"}:
        return (-float(candidate.overlay_loss), float(candidate.overlay_retention), float(candidate.overlay_reliability))
    return (embb_sum_rate, -float(candidate.overlay_loss), float(candidate.overlay_retention))


def _pure_superposition_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    selection_rule: str,
) -> Dict[str, HybridAction]:
    overlay_retention_threshold = float(
        getattr(env.rl_cfg.env, "compare_pure_superposition_overlay_retention_threshold", 0.0) or 0.0
    )
    return _minislot_global_matching_actions(
        env,
        observations,
        allowed_modes=(MODE_OVERLAY,),
        superposition_selection_rule=selection_rule,
        overlay_retention_threshold=overlay_retention_threshold,
    )


def _greedy_actions(env: SRMAPPOPhaseAEnv, observations: Dict[str, AgentObservation]) -> Dict[str, HybridAction]:
    overlay_retention_threshold = float(
        getattr(env.rl_cfg.env, "compare_greedy_overlay_retention_threshold", 0.0) or 0.0
    )
    return _sequential_single_action_selector(
        env,
        observations,
        allowed_modes=(MODE_OVERLAY, MODE_PUNCTURE),
        overlay_retention_threshold=overlay_retention_threshold,
    )


def _append_jsonl_debug_record(path_like: str | os.PathLike[str], record: Dict[str, object]) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    def _normalize(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): _normalize(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [_normalize(v) for v in value]
        if isinstance(value, list):
            return [_normalize(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    serializable = {str(key): _normalize(value) for key, value in record.items()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def _best_sequential_entry(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    *,
    allowed_modes: Iterable[int],
    selected_agents: set[str],
    used_packets: set[int],
    current_minislot: int,
    current_rb: int,
    puncturing_selection_rule: str = "max_embb_sum_rate",
    superposition_selection_rule: str = "max_embb_sum_rate",
    overlay_retention_threshold: float = 0.0,
) -> Optional[Dict[str, object]]:
    best_entry: Optional[Dict[str, object]] = None
    for uav_idx, agent_id in enumerate(env.agent_ids):
        if str(agent_id) in selected_agents:
            continue
        obs = observations[agent_id]
        for packet_option, mode, candidate in _feasible_mode_packet_pairs(obs, allowed_modes):
            if (
                int(mode) == MODE_OVERLAY
                and float(overlay_retention_threshold) > 0.0
                and float(getattr(candidate, "overlay_retention", 0.0) or 0.0)
                < float(overlay_retention_threshold) - 1.0e-12
            ):
                continue
            packet_id = int(candidate.packet_id)
            if packet_id in used_packets:
                continue
            global_embb_throughput = float(
                env._global_embb_throughput_if_apply_candidate_cell(
                    int(uav_idx),
                    int(current_rb),
                    int(current_minislot),
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
                puncturing_selection_rule=puncturing_selection_rule,
                superposition_selection_rule=superposition_selection_rule,
            )
            entry = {
                "agent_id": str(agent_id),
                "uav": int(uav_idx),
                "packet_option": int(packet_option),
                "packet_id": int(packet_id),
                "candidate": candidate,
                "mode": int(mode),
                "rank": tuple(float(x) for x in rank),
                "global_embb_throughput": float(global_embb_throughput),
            }
            if best_entry is None or tuple(entry["rank"]) > tuple(best_entry["rank"]):
                best_entry = entry
    return best_entry


def _maybe_log_greedy_invariant(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    *,
    selected_agents: set[str],
    used_packets: set[int],
    current_minislot: int,
    current_rb: int,
    subiter: int,
    puncturing_selection_rule: str,
    superposition_selection_rule: str,
    chosen_entry: Optional[Dict[str, object]] = None,
) -> None:
    log_path = str(os.getenv("SR_MAPPO_GREEDY_INVARIANT_LOG", "") or "").strip()
    if not log_path:
        return
    puncture_best = _best_sequential_entry(
        env,
        observations,
        allowed_modes=(MODE_PUNCTURE,),
        selected_agents=selected_agents,
        used_packets=used_packets,
        current_minislot=current_minislot,
        current_rb=current_rb,
        puncturing_selection_rule=puncturing_selection_rule,
        superposition_selection_rule=superposition_selection_rule,
    )
    overlay_best = _best_sequential_entry(
        env,
        observations,
        allowed_modes=(MODE_OVERLAY,),
        selected_agents=selected_agents,
        used_packets=used_packets,
        current_minislot=current_minislot,
        current_rb=current_rb,
        puncturing_selection_rule=puncturing_selection_rule,
        superposition_selection_rule=superposition_selection_rule,
    )
    greedy_best = _best_sequential_entry(
        env,
        observations,
        allowed_modes=(MODE_OVERLAY, MODE_PUNCTURE),
        selected_agents=selected_agents,
        used_packets=used_packets,
        current_minislot=current_minislot,
        current_rb=current_rb,
        puncturing_selection_rule=puncturing_selection_rule,
        superposition_selection_rule=superposition_selection_rule,
    )
    if greedy_best is None and puncture_best is None and overlay_best is None:
        return

    def _entry_summary(entry: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if entry is None:
            return None
        candidate = entry.get("candidate")
        owner = int(candidate.embb_owner_for_mode(int(entry["mode"]))) if candidate is not None else -1
        return {
            "agent_id": str(entry["agent_id"]),
            "uav": int(entry["uav"]),
            "packet_option": int(entry["packet_option"]),
            "packet_id": int(entry["packet_id"]),
            "mode": int(entry["mode"]),
            "owner": int(owner),
            "rank": [float(x) for x in tuple(entry["rank"])],
            "global_embb_throughput": float(entry["global_embb_throughput"]),
        }

    greedy_thr = float(greedy_best["global_embb_throughput"]) if greedy_best is not None else float("-inf")
    puncture_thr = float(puncture_best["global_embb_throughput"]) if puncture_best is not None else float("-inf")
    overlay_thr = float(overlay_best["global_embb_throughput"]) if overlay_best is not None else float("-inf")
    best_single_thr = max(puncture_thr, overlay_thr)
    violation = bool(greedy_thr + 1.0e-9 < best_single_thr)

    record = {
        "minislot": int(current_minislot),
        "rb": int(current_rb),
        "subiter": int(subiter),
        "selected_agents": sorted(str(agent_id) for agent_id in selected_agents),
        "used_packets": sorted(int(packet_id) for packet_id in used_packets),
        "greedy_best": _entry_summary(greedy_best),
        "overlay_best": _entry_summary(overlay_best),
        "puncture_best": _entry_summary(puncture_best),
        "best_single_throughput": float(best_single_thr),
        "violation": bool(violation),
        "chosen_entry": _entry_summary(chosen_entry),
    }
    _append_jsonl_debug_record(log_path, record)


def _sequential_single_action_selector(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    *,
    allowed_modes: Iterable[int],
    puncturing_selection_rule: str = "max_embb_sum_rate",
    superposition_selection_rule: str = "max_embb_sum_rate",
    overlay_retention_threshold: float = 0.0,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    env._last_baseline_selection_order = []
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    if planning_phase:
        for agent_id, obs in observations.items():
            actions[agent_id] = _planning_baseline_action(env, obs)
        return actions

    current_minislot, current_rb = env._current_cell()
    for agent_id in observations:
        actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)

    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()

    selected_agents: set[str] = set()
    used_packets: set[int] = set()

    try:
        subiter = 0
        while True:
            best_entry = _best_sequential_entry(
                env,
                observations,
                allowed_modes=allowed_modes,
                selected_agents=selected_agents,
                used_packets=used_packets,
                current_minislot=int(current_minislot),
                current_rb=int(current_rb),
                puncturing_selection_rule=puncturing_selection_rule,
                superposition_selection_rule=superposition_selection_rule,
                overlay_retention_threshold=overlay_retention_threshold,
            )

            if best_entry is None:
                break

            if set(int(mode) for mode in allowed_modes) == {MODE_OVERLAY, MODE_PUNCTURE}:
                _maybe_log_greedy_invariant(
                    env,
                    observations,
                    selected_agents=selected_agents,
                    used_packets=used_packets,
                    current_minislot=int(current_minislot),
                    current_rb=int(current_rb),
                    subiter=int(subiter),
                    puncturing_selection_rule=puncturing_selection_rule,
                    superposition_selection_rule=superposition_selection_rule,
                    chosen_entry=best_entry,
                )

            agent_id = str(best_entry["agent_id"])
            candidate = best_entry["candidate"]
            mode = int(best_entry["mode"])
            actions[agent_id] = HybridAction(
                mode=mode,
                packet_option=int(best_entry["packet_option"]),
                power_delta=0.0,
            )
            env._last_baseline_selection_order.append(str(agent_id))
            selected_agents.add(agent_id)
            used_packets.add(int(best_entry["packet_id"]))
            if int(mode) == MODE_PUNCTURE:
                puncture_before_bps = float(_current_global_embb_throughput(env))
                puncture_after_bps = float(
                    env._global_embb_throughput_if_apply_candidate_cell(
                        int(best_entry["uav"]),
                        int(current_rb),
                        int(current_minislot),
                        candidate,
                        MODE_PUNCTURE,
                        0.0,
                    )
                )
                _append_puncture_loss_history(env, float(puncture_before_bps - puncture_after_bps))
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(best_entry["uav"]),
                rb_idx=int(current_rb),
                minislot=int(current_minislot),
                candidate=candidate,
                mode_int=mode,
            )

            if len(selected_agents) >= len(env.agent_ids):
                break
            subiter += 1
        return actions
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


def _sequential_greedy_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    env._last_baseline_selection_order = []
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    if planning_phase:
        for agent_id, obs in observations.items():
            actions[agent_id] = _planning_baseline_action(env, obs)
        return actions

    current_minislot, current_rb = env._current_cell()
    for agent_id in observations:
        actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)

    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()

    selected_agents: set[str] = set()
    used_packets: set[int] = set()
    overlay_retention_threshold = 0.90
    puncture_gain_margin_bps = 0.5e6

    try:
        while True:
            best_entry: Optional[Dict[str, object]] = None
            for uav_idx, agent_id in enumerate(env.agent_ids):
                if str(agent_id) in selected_agents:
                    continue
                obs = observations[agent_id]
                for packet_option, mode, candidate in _greedy_gate_mode_packet_pairs(
                    env,
                    obs,
                    uav_idx=int(uav_idx),
                    rb_idx=int(current_rb),
                    minislot_idx=int(current_minislot),
                    overlay_retention_threshold=float(overlay_retention_threshold),
                    puncture_gain_margin_bps=float(puncture_gain_margin_bps),
                ):
                    packet_id = int(candidate.packet_id)
                    if packet_id in used_packets:
                        continue
                    global_embb_throughput = float(
                        env._global_embb_throughput_if_apply_candidate_cell(
                            int(uav_idx),
                            int(current_rb),
                            int(current_minislot),
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
                    entry = {
                        "agent_id": str(agent_id),
                        "uav": int(uav_idx),
                        "packet_option": int(packet_option),
                        "packet_id": int(packet_id),
                        "candidate": candidate,
                        "mode": int(mode),
                        "rank": tuple(float(x) for x in rank),
                    }
                    if best_entry is None or tuple(entry["rank"]) > tuple(best_entry["rank"]):
                        best_entry = entry

            if best_entry is None:
                break

            agent_id = str(best_entry["agent_id"])
            candidate = best_entry["candidate"]
            mode = int(best_entry["mode"])
            actions[agent_id] = HybridAction(
                mode=mode,
                packet_option=int(best_entry["packet_option"]),
                power_delta=0.0,
            )
            env._last_baseline_selection_order.append(str(agent_id))
            selected_agents.add(agent_id)
            used_packets.add(int(best_entry["packet_id"]))
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(best_entry["uav"]),
                rb_idx=int(current_rb),
                minislot=int(current_minislot),
                candidate=candidate,
                mode_int=mode,
            )

            if len(selected_agents) >= len(env.agent_ids):
                break
        return actions
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


def _sequential_pure_puncturing_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    selection_rule: str,
) -> Dict[str, HybridAction]:
    return _sequential_single_action_selector(
        env,
        observations,
        allowed_modes=(MODE_PUNCTURE,),
        puncturing_selection_rule=selection_rule,
    )


def _sequential_pure_superposition_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    selection_rule: str,
) -> Dict[str, HybridAction]:
    return _sequential_single_action_selector(
        env,
        observations,
        allowed_modes=(MODE_OVERLAY,),
        superposition_selection_rule=selection_rule,
    )


def _sequential_random_scheduler_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    rng: np.random.Generator,
) -> Dict[str, HybridAction]:
    actions: Dict[str, HybridAction] = {}
    env._last_baseline_selection_order = []
    planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
    if planning_phase:
        for agent_id, obs in observations.items():
            actions[agent_id] = _planning_baseline_action(env, obs)
        return actions

    for agent_id in observations:
        actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)

    selected_agents: set[str] = set()
    used_packets: set[int] = set()
    current_minislot, current_rb = env._current_cell()

    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()

    try:
        while True:
            feasible_entries: List[Tuple[str, int, int, CandidatePacket, int]] = []
            for uav_idx, agent_id in enumerate(env.agent_ids):
                if str(agent_id) in selected_agents:
                    continue
                obs = observations[agent_id]
                for packet_option, mode, candidate in _feasible_mode_packet_pairs(obs, (MODE_OVERLAY, MODE_PUNCTURE)):
                    if int(candidate.packet_id) in used_packets:
                        continue
                    feasible_entries.append((str(agent_id), int(uav_idx), int(packet_option), candidate, int(mode)))
            if not feasible_entries:
                break
            chosen_agent_id, chosen_uav_idx, packet_option, candidate, mode = feasible_entries[int(rng.integers(0, len(feasible_entries)))]
            actions[str(chosen_agent_id)] = HybridAction(
                mode=int(mode),
                packet_option=int(packet_option),
                power_delta=0.0,
            )
            env._last_baseline_selection_order.append(str(chosen_agent_id))
            selected_agents.add(str(chosen_agent_id))
            used_packets.add(int(candidate.packet_id))
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(chosen_uav_idx),
                rb_idx=int(current_rb),
                minislot=int(current_minislot),
                candidate=candidate,
                mode_int=int(mode),
            )
            if len(selected_agents) >= len(env.agent_ids):
                break
        return actions
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


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


def _run_legacy_original_greedy_policy(
    env: SRMAPPOPhaseAEnv,
    *,
    seed: int,
    runner: Callable[[SRMAPPOPhaseAEnv, int, int], Dict[str, float]],
) -> Dict[str, object]:
    summary = dict(runner(env, int(seed), 0) or {})
    summary.setdefault("embb_user_count", float(getattr(env.sys_cfg, "num_embb_users", 0) or 0))
    return _standardize_metrics(
        summary,
        runtime_sec=0.0,
        avg_urllc_sinr_linear=0.0,
        episode_count=1.0,
        episode_step_count=0.0,
        episode_reward_mean=0.0,
        episode_reward_total_sum=0.0,
        logged_step_reward_sum=0.0,
        logged_terminal_reward_sum=0.0,
        embb_user_tx_powers=_extract_final_embb_user_tx_powers(env),
    )


def _collect_candidate_violation_counts(observations: Dict[str, AgentObservation]) -> Dict[str, int]:
    counts = {
        "urllc_reliability": 0,
        "embb_min_rate": 0,
        "power": 0,
        "collision": 0,
        "already_scheduled": 0,
        "intercell": 0,
        "deadline": 0,
        "overlay_gain_ratio": 0,
        "overlay_margin": 0,
        "overlay_retention_gate": 0,
        "overlay_positive_gate": 0,
        "overlay_owner_missing": 0,
        "overlay_reliability": 0,
        "overlay_sic": 0,
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
            counts["overlay_gain_ratio"] += int(bool(getattr(candidate, "cause_gain_ratio_unqualified", False)))
            counts["overlay_margin"] += int(bool(getattr(candidate, "cause_overlay_margin_blocked", False)))
            counts["overlay_retention_gate"] += int(
                bool(getattr(candidate, "cause_overlay_retention_gate_blocked", False))
            )
            counts["overlay_positive_gate"] += int(
                bool(getattr(candidate, "cause_overlay_positive_gate_blocked", False))
            )
            counts["overlay_owner_missing"] += int(
                bool(getattr(candidate, "cause_no_overlay_owner_available", False))
            )
            counts["overlay_reliability"] += int(
                bool(getattr(candidate, "cause_overlay_reliability_failed", False))
            )
            counts["overlay_sic"] += int(bool(getattr(candidate, "cause_overlay_sic_failed", False)))
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


def _append_per_admission_damage_sample(
    samples: List[Dict[str, object]],
    *,
    env: SRMAPPOPhaseAEnv,
    method: str,
    seed: int,
    episode: int,
    decision_step: int,
    minislot: int,
    uav_idx: int,
    rb_idx: int,
    candidate: CandidatePacket,
    mode_int: int,
    embb_sum_rate_before: float,
    embb_sum_rate_after: float,
) -> None:
    if candidate is None or mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
        return
    actual_power = float(env._project_actual_power(float(candidate.required_power_for_mode(int(mode_int))), 0.0))
    packet_bits = float(env._packet_bits_for_user(int(candidate.source_user)))
    channel_uses = int(getattr(env.sys_cfg, "channel_uses_per_minislot", 1) or 1)
    gamma_th = float(
        env.capacity_model.min_power_for_reliability(
            float(getattr(env.urllc_cfg, "target_error_probability", 0.0) or 0.0),
            packet_bits,
            channel_uses,
            0.0,
        )
    )
    reliability = float(candidate.reliability_for_mode(mode_int))
    reliability_target = 1.0 - float(getattr(env.urllc_cfg, "target_error_probability", 0.0) or 0.0)
    urllc_sinr = float(candidate.overlay_urllc_snir if mode_int == MODE_OVERLAY else candidate.puncture_urllc_snir)
    effective_total_interference = float(
        max(
            (actual_power * float(getattr(candidate, "channel_gain", 0.0) or 0.0) / max(urllc_sinr, 1.0e-15))
            - float(getattr(env.sys_cfg, "noise_power", 0.0) or 0.0),
            0.0,
        )
    )
    local_interference = float(getattr(candidate, "overlay_local_interference_power", 0.0) or 0.0)
    intercell_interference = float(getattr(candidate, "overlay_intercell_interference_power", 0.0) or 0.0)
    residual_interference = float(getattr(candidate, "overlay_residual_sic_interference_power", 0.0) or 0.0)
    if mode_int == MODE_PUNCTURE:
        local_interference = 0.0
        intercell_interference = float(effective_total_interference)
        residual_interference = 0.0
    elif (local_interference + intercell_interference) <= 0.0 and effective_total_interference > 0.0:
        # Fall back to the effective interference implied by the solved SINR.
        intercell_interference = float(effective_total_interference)
    selected_mode_stats = env._mode_local_outcome_stats(
        candidate=candidate,
        mode=int(mode_int),
        uav_idx=int(uav_idx),
        rb_idx=int(rb_idx if rb_idx >= 0 else getattr(candidate, "rb_index", -1)),
        minislot=int(minislot),
        actual_power_override=float(actual_power),
    )
    embb_sum_rate_before = float(embb_sum_rate_before)
    embb_sum_rate_after = float(embb_sum_rate_after)
    embb_loss = float(embb_sum_rate_before - embb_sum_rate_after)
    samples.append(
        {
            "method": str(method),
            "seed": int(seed),
            "episode": int(episode),
            "decision_step": int(decision_step),
            "decision_minislot": int(minislot),
            "packet_id": int(candidate.packet_id),
            "selected_packet_id": int(candidate.packet_id),
            "uav_idx": int(uav_idx),
            "selected_uav": int(uav_idx),
            "rb_idx": int(rb_idx if rb_idx >= 0 else getattr(candidate, "rb_index", -1)),
            "selected_rb": int(rb_idx if rb_idx >= 0 else getattr(candidate, "rb_index", -1)),
            "minislot": int(minislot),
            "source_user": int(getattr(candidate, "source_user", -1)),
            "embb_owner": int(candidate.embb_owner_for_mode(mode_int)),
            "selected_embb_owner": int(candidate.embb_owner_for_mode(mode_int)),
            "mode": "overlay" if mode_int == MODE_OVERLAY else "puncturing",
            "selected_mode": "overlay" if mode_int == MODE_OVERLAY else "puncturing",
            "mode_int": int(mode_int),
            "required_power": float(candidate.required_power_for_mode(mode_int)),
            "actual_power": float(actual_power),
            "channel_gain": float(getattr(candidate, "channel_gain", 0.0) or 0.0),
            "reliability": float(reliability),
            "reliability_target": float(reliability_target),
            "reliability_margin": float(reliability - reliability_target),
            "urllc_sinr": urllc_sinr,
            "gamma_th": gamma_th,
            "noise_power": float(getattr(env.sys_cfg, "noise_power", 0.0) or 0.0),
            "effective_total_interference_power": float(effective_total_interference),
            "local_interference_power": float(local_interference),
            "intercell_interference_power": float(intercell_interference),
            "residual_sic_interference_power": float(residual_interference),
            "overlay_pre_sic_snir": float(getattr(candidate, "overlay_pre_sic_snir", 0.0) or 0.0),
            "post_sic_snir": float(getattr(candidate, "post_sic_snir", 0.0) or 0.0),
            "base_embb_snir": float(getattr(candidate, "base_embb_snir", 0.0) or 0.0),
            "base_embb_signal_power": float(getattr(candidate, "base_embb_signal_power", 0.0) or 0.0),
            "base_embb_intercell_power": float(getattr(candidate, "base_embb_intercell_power", 0.0) or 0.0),
            "owner_rate_after": float(selected_mode_stats.get("owner_rate_after", 0.0) or 0.0),
            "owner_minrate_margin": float(selected_mode_stats.get("owner_minrate_margin", 0.0) or 0.0),
            "owner_minrate_ok": float(selected_mode_stats.get("owner_minrate_ok", 0.0) or 0.0),
            "owner_service_ok": float(selected_mode_stats.get("owner_service_ok", 0.0) or 0.0),
            "sic_margin": float(selected_mode_stats.get("sic_margin", 0.0) or 0.0),
            "urllc_reliability_satisfied": int(reliability + 1.0e-12 >= reliability_target),
            "global_embb_rate_before_action": embb_sum_rate_before,
            "embb_sum_rate_before_action": embb_sum_rate_before,
            "global_embb_rate_after_action": embb_sum_rate_after,
            "embb_sum_rate_after_action": embb_sum_rate_after,
            "embb_rate_loss_due_to_action": embb_loss,
        }
    )


def _current_global_embb_throughput(env: SRMAPPOPhaseAEnv) -> float:
    embb_rates_eff, _embb_power_alloc_eff, _ov_eff, _pu_eff = env._compute_episode_embb_metrics(
        ignore_intercell=False,
        apply_local_puncture_deduction=True,
        apply_embb_source_mask=True,
    )
    return float(np.sum(np.asarray(embb_rates_eff, dtype=float)))


def _apply_candidate_to_env_state_for_accounting(
    env: SRMAPPOPhaseAEnv,
    *,
    uav_idx: int,
    rb_idx: int,
    minislot: int,
    candidate: CandidatePacket,
    mode_int: int,
) -> None:
    packet_id = int(candidate.packet_id)
    actual_power = float(env._project_actual_power(float(candidate.required_power_for_mode(int(mode_int))), 0.0))
    env.packet_grid[uav_idx, rb_idx, minislot] = int(packet_id)
    env.mode_grid[uav_idx, rb_idx, minislot] = int(mode_int)
    env.embb_owner_grid[uav_idx, rb_idx, minislot] = (
        -1 if int(mode_int) == MODE_PUNCTURE else int(candidate.embb_owner_for_mode(int(mode_int)))
    )
    if packet_id < env.scheduled_power.shape[0]:
        env.scheduled_power[packet_id, :] = 0.0
        env.scheduled_power[packet_id, uav_idx] = float(actual_power)
    if packet_id < env.scheduled_uavs.size:
        env.scheduled_uavs[packet_id] = int(uav_idx)
    if packet_id < env.scheduled_reliabilities.size:
        env.scheduled_reliabilities[packet_id] = float(candidate.reliability_for_mode(int(mode_int)))


def _log_resolved_admission_damage_samples(
    *,
    env: SRMAPPOPhaseAEnv,
    resolved: Dict[str, ShieldedAction],
    method: str,
    seed: int,
    episode: int,
    decision_step: int,
    minislot: int,
    rb: int,
    samples: List[Dict[str, object]],
) -> None:
    admitted_items: List[Tuple[int, int, CandidatePacket, int]] = []
    ordered_agent_ids = list(getattr(env, "_last_baseline_selection_order", []) or [])
    if not ordered_agent_ids:
        ordered_agent_ids = list(env.agent_ids)
    seen_agent_ids: set[str] = set()
    for agent_id in ordered_agent_ids:
        if str(agent_id) in seen_agent_ids:
            continue
        seen_agent_ids.add(str(agent_id))
        agent_idx = int(env.agent_ids.index(agent_id))
        shielded = resolved.get(agent_id)
        if shielded is None or shielded.candidate is None:
            continue
        mode_int = int(shielded.action.mode)
        if mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
            continue
        admitted_items.append((int(agent_idx), int(rb), shielded.candidate, mode_int))
    for agent_idx, agent_id in enumerate(env.agent_ids):
        if str(agent_id) in seen_agent_ids:
            continue
        shielded = resolved.get(agent_id)
        if shielded is None or shielded.candidate is None:
            continue
        mode_int = int(shielded.action.mode)
        if mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
            continue
        admitted_items.append((int(agent_idx), int(rb), shielded.candidate, mode_int))
    if not admitted_items:
        return

    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()
    current_throughput = float(_current_global_embb_throughput(env))

    try:
        for uav_idx, rb_idx, candidate, mode_int in admitted_items:
            before = float(current_throughput)
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(uav_idx),
                rb_idx=int(rb_idx),
                minislot=int(minislot),
                candidate=candidate,
                mode_int=int(mode_int),
            )
            after = float(_current_global_embb_throughput(env))
            _append_per_admission_damage_sample(
                samples,
                env=env,
                method=str(method),
                seed=int(seed),
                episode=int(episode),
                decision_step=int(decision_step),
                minislot=int(minislot),
                uav_idx=int(uav_idx),
                rb_idx=int(rb_idx),
                candidate=candidate,
                mode_int=int(mode_int),
                embb_sum_rate_before=float(before),
                embb_sum_rate_after=float(after),
            )
            current_throughput = float(after)
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved


def _resolve_actions_sequentially_in_order(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    joint_actions: Dict[str, HybridAction],
    *,
    minislot: int,
    rb: int,
    ordered_agent_ids: List[str],
) -> Dict[str, ShieldedAction]:
    puncture_history_local = _sanitize_puncture_loss_history(
        getattr(env, "_baseline_recent_puncture_losses_bps", []) or []
    )
    resolved_final: Dict[str, ShieldedAction] = {
        str(agent_id): env._raw_action_to_shielded_action(
            HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
            observations[str(agent_id)],
        )
        for agent_id in env.agent_ids
    }
    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()
    puncture_debug_enabled = str(os.getenv("SR_MAPPO_DEBUG_PUNCTURE_CAP", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        seen: set[str] = set()
        for agent_id in ordered_agent_ids:
            if str(agent_id) in seen or str(agent_id) not in joint_actions:
                continue
            seen.add(str(agent_id))
            single_joint = {
                str(aid): HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
                for aid in env.agent_ids
            }
            single_joint[str(agent_id)] = joint_actions[str(agent_id)]
            resolved_single = env._resolve_executed_actions(
                single_joint,
                observations,
                minislot=int(minislot),
                rb=int(rb),
            )
            shielded = resolved_single[str(agent_id)]
            debug_cap_bps: Optional[float] = None
            debug_loss_bps: Optional[float] = None
            if shielded.candidate is not None and int(shielded.action.mode) == MODE_PUNCTURE:
                debug_cap_bps = _puncture_dynamic_cap_from_history(puncture_history_local)
                before_bps = float(_current_global_embb_throughput(env))
                after_bps = float(
                    env._global_embb_throughput_if_apply_candidate_cell(
                        int(env.agent_ids.index(str(agent_id))),
                        int(rb),
                        int(minislot),
                        shielded.candidate,
                        MODE_PUNCTURE,
                        0.0,
                    )
                )
                debug_loss_bps = float(before_bps - after_bps)
            if (
                shielded.candidate is not None
                and int(shielded.action.mode) == MODE_PUNCTURE
                and debug_cap_bps is not None
                and debug_loss_bps is not None
                and float(debug_loss_bps) > float(debug_cap_bps) + 1.0e-12
            ):
                shielded = env._raw_action_to_shielded_action(
                    HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
                    observations[str(agent_id)],
                )
            if puncture_debug_enabled and debug_loss_bps is not None:
                dropped = int(shielded.candidate is None or int(shielded.action.mode) != MODE_PUNCTURE)
                packet_id = int(resolved_single[str(agent_id)].candidate.packet_id) if resolved_single[str(agent_id)].candidate is not None else -1
                print(
                    "[SR-MAPPO][PUNCTURE-CAP] "
                    f"agent={agent_id} step_minislot={int(minislot)} rb={int(rb)} "
                    f"packet={packet_id} "
                    f"predicted_loss_mbps={float(debug_loss_bps)/1.0e6:.6f} "
                    f"cap_mbps={('None' if debug_cap_bps is None else f'{float(debug_cap_bps)/1.0e6:.6f}')} "
                    f"history_len={len(puncture_history_local)} "
                    f"dropped={dropped}",
                    flush=True,
                )
            resolved_final[str(agent_id)] = shielded
            if shielded.candidate is None:
                continue
            mode_int = int(shielded.action.mode)
            if mode_int not in {MODE_OVERLAY, MODE_PUNCTURE}:
                continue
            uav_idx = int(env.agent_ids.index(str(agent_id)))
            if int(mode_int) == MODE_PUNCTURE:
                predicted_before_bps = float(_current_global_embb_throughput(env))
                predicted_after_bps = float(
                    env._global_embb_throughput_if_apply_candidate_cell(
                        int(uav_idx),
                        int(rb),
                        int(minislot),
                        shielded.candidate,
                        MODE_PUNCTURE,
                        0.0,
                    )
                )
                puncture_loss_bps = float(predicted_before_bps - predicted_after_bps)
                if puncture_loss_bps >= 0.0:
                    puncture_history_local.append(float(puncture_loss_bps))
                    if len(puncture_history_local) > 10:
                        puncture_history_local = puncture_history_local[-10:]
                    env._baseline_recent_puncture_losses_bps = list(puncture_history_local)
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(uav_idx),
                rb_idx=int(rb),
                minislot=int(minislot),
                candidate=shielded.candidate,
                mode_int=int(mode_int),
            )
        return resolved_final
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


def _apply_greedy_puncture_cap_to_resolved_actions(
    env: SRMAPPOPhaseAEnv,
    observations: Dict[str, AgentObservation],
    resolved: Dict[str, ShieldedAction],
    *,
    minislot: int,
    rb: int,
    ordered_agent_ids: List[str],
) -> Dict[str, ShieldedAction]:
    adjusted = dict(resolved)
    packet_grid_saved = np.asarray(env.packet_grid, dtype=int).copy()
    mode_grid_saved = np.asarray(env.mode_grid, dtype=int).copy()
    owner_grid_saved = np.asarray(env.embb_owner_grid, dtype=int).copy()
    scheduled_uavs_saved = np.asarray(env.scheduled_uavs, dtype=int).copy()
    scheduled_reliabilities_saved = np.asarray(env.scheduled_reliabilities, dtype=float).copy()
    scheduled_power_saved = np.asarray(env.scheduled_power, dtype=float).copy()
    try:
        seen: set[str] = set()
        for agent_id in ordered_agent_ids:
            if str(agent_id) in seen or str(agent_id) not in adjusted:
                continue
            seen.add(str(agent_id))
            shielded = adjusted[str(agent_id)]
            candidate = shielded.candidate
            if candidate is None or int(shielded.action.mode) not in {MODE_OVERLAY, MODE_PUNCTURE}:
                continue
            uav_idx = int(env.agent_ids.index(str(agent_id)))
            if int(shielded.action.mode) == MODE_PUNCTURE:
                cap_bps = _puncture_dynamic_cap_bps(env)
                before_bps = float(_current_global_embb_throughput(env))
                after_bps = float(
                    env._global_embb_throughput_if_apply_candidate_cell(
                        int(uav_idx),
                        int(rb),
                        int(minislot),
                        candidate,
                        MODE_PUNCTURE,
                        0.0,
                    )
                )
                puncture_loss_bps = float(before_bps - after_bps)
                if cap_bps is not None and puncture_loss_bps > float(cap_bps) + 1.0e-12:
                    adjusted[str(agent_id)] = env._raw_action_to_shielded_action(
                        HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
                        observations[str(agent_id)],
                    )
                    continue
                _append_puncture_loss_history(env, puncture_loss_bps)
            _apply_candidate_to_env_state_for_accounting(
                env,
                uav_idx=int(uav_idx),
                rb_idx=int(rb),
                minislot=int(minislot),
                candidate=candidate,
                mode_int=int(shielded.action.mode),
            )
        return adjusted
    finally:
        env.packet_grid = packet_grid_saved
        env.mode_grid = mode_grid_saved
        env.embb_owner_grid = owner_grid_saved
        env.scheduled_uavs = scheduled_uavs_saved
        env.scheduled_reliabilities = scheduled_reliabilities_saved
        env.scheduled_power = scheduled_power_saved


def _append_embb_user_minrate_trace_for_minislot(
    trace_samples: List[Dict[str, object]],
    *,
    env: SRMAPPOPhaseAEnv,
    method: str,
    seed: int,
    episode: int,
    decision_step: int,
    minislot: int,
) -> None:
    rates_bps = np.asarray(
        env._compute_embb_rates_for_minislot(
            int(minislot),
            ignore_intercell=False,
            apply_local_puncture_deduction=True,
            apply_embb_source_mask=True,
        ),
        dtype=float,
    )
    r_min_bps = float(
        getattr(env.embb_cfg, "min_rate_per_user_bps", getattr(env.embb_cfg, "min_rate", 0.0)) or 0.0
    )
    r_min_mbps = float(r_min_bps / 1.0e6)
    for embb_user_id, rate_bps in enumerate(rates_bps):
        rate_bps = float(rate_bps or 0.0)
        if rate_bps <= 0.0:
            continue
        trace_samples.append(
            {
                "method": str(method),
                "seed": int(seed),
                "episode": int(episode),
                "embb_user_id": int(embb_user_id),
                "decision_step": int(decision_step),
                "minislot_index": int(minislot),
                "episode_total_minislots": int(env.sys_cfg.num_minislots),
                "time_granularity": "minislot-level",
                "r_min_mbps": float(r_min_mbps),
                "embb_rate_bps": float(rate_bps),
                "embb_rate_mbps": float(rate_bps / 1.0e6),
                "is_satisfied": int(rate_bps + 1.0e-9 >= r_min_bps),
            }
        )


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
    urllc_reliability_violation_count: float = 0.0,
    safe_admitted_urllc_count: float = 0.0,
    per_admission_embb_damage_samples: Optional[List[Dict[str, object]]] = None,
    embb_user_minrate_trace: Optional[List[Dict[str, object]]] = None,
    embb_user_tx_powers: Optional[List[float]] = None,
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
    embb_served_user_count = float(
        summary.get(
            "embb_served_user_count",
            summary.get("embb_served_users", 0.0),
        )
        or 0.0
    )
    phase0_blocked_user_count_raw = summary.get("phase0_minrate_blocked_user_count", None)
    phase0_blocked_user_count = None
    try:
        if phase0_blocked_user_count_raw not in (None, "", "None"):
            phase0_blocked_user_count = float(phase0_blocked_user_count_raw)
    except Exception:
        phase0_blocked_user_count = None
    phase_a_total_decisions = float(summary.get("phase_a_total_decisions", 0.0) or 0.0)
    overlay_action_count = float(summary.get("overlay_count", 0.0) or 0.0)
    puncturing_action_count = float(summary.get("puncture_count", 0.0) or 0.0)
    keep_action_count = max(phase_a_total_decisions - overlay_action_count - puncturing_action_count, 0.0)
    if phase0_blocked_user_count is not None:
        embb_blocked_user_count = max(float(phase0_blocked_user_count), 0.0)
        embb_served_user_count = max(embb_user_count - embb_blocked_user_count, 0.0)
    else:
        embb_blocked_user_count = max(embb_user_count - embb_served_user_count, 0.0)
    urllc_blocked_user_count = max(active_packets - scheduled_packets, 0.0)
    minrate_violation_count = max(int(round(embb_user_count * (1.0 - minrate_ok_ratio))), 0)
    avg_sinr_db = 10.0 * np.log10(max(avg_urllc_sinr_linear, 1.0e-15)) if avg_urllc_sinr_linear > 0.0 else 0.0
    logged_total_reward_sum = float(logged_step_reward_sum) + float(logged_terminal_reward_sum)
    safe_admission_ratio = float(safe_admitted_urllc_count / max(active_packets, 1.0)) if active_packets > 0.0 else 1.0
    urllc_reliability_violation_ratio = (
        float(urllc_reliability_violation_count / max(scheduled_packets, 1.0))
        if scheduled_packets > 0.0
        else 0.0
    )
    embb_minrate_violation_ratio = float(minrate_violation_count / max(embb_user_count, 1.0)) if embb_user_count > 0.0 else 0.0

    return {
        "total_embb_throughput": embb_rate,
        "total_urllc_arrivals": active_packets,
        "admitted_urllc_count": scheduled_packets,
        "scheduled_urllc_packets": scheduled_packets,
        "urllc_admission_ratio": float(summary.get("urllc_admission_rate", 0.0) or 0.0),
        "dropped_urllc_packets": float(urllc_blocked_user_count),
        "urllc_blocked_user_count": float(urllc_blocked_user_count),
        "average_urllc_sinr": float(avg_sinr_db),
        "average_urllc_sinr_db": float(avg_sinr_db),
        "average_embb_rate": avg_embb_rate,
        "embb_served_user_count": float(embb_served_user_count),
        "embb_blocked_user_count": float(embb_blocked_user_count),
        "embb_minimum_rate_violation_count": float(minrate_violation_count),
        "embb_min_rate_violation_ratio": float(embb_minrate_violation_ratio),
        "urllc_reliability_violation_count": float(urllc_reliability_violation_count),
        "urllc_reliability_violation_ratio": float(urllc_reliability_violation_ratio),
        "safe_admitted_urllc_count": float(safe_admitted_urllc_count),
        "safe_admission_ratio": float(safe_admission_ratio),
        "average_power_consumption": float(summary.get("total_power", 0.0) or 0.0),
        "total_power": float(summary.get("total_power", 0.0) or 0.0),
        "embb_power": float(summary.get("embb_power", 0.0) or 0.0),
        "urllc_power": float(summary.get("urllc_power", 0.0) or 0.0),
        "embb_user_tx_powers": list(embb_user_tx_powers or []),
        "phase_a_total_decisions": float(phase_a_total_decisions),
        "keep_action_count": float(keep_action_count),
        "overlay_action_count": float(overlay_action_count),
        "puncturing_action_count": float(puncturing_action_count),
        "keep_action_ratio": float(keep_action_count / max(phase_a_total_decisions, 1.0)),
        "overlay_action_ratio": float(overlay_action_count / max(phase_a_total_decisions, 1.0)),
        "puncturing_action_ratio": float(puncturing_action_count / max(phase_a_total_decisions, 1.0)),
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
        "per_admission_embb_damage_samples": list(per_admission_embb_damage_samples or []),
        "embb_user_minrate_trace": list(embb_user_minrate_trace or []),
        "raw_summary": dict(summary),
    }


def _rollout_with_selector(
    env: SRMAPPOPhaseAEnv,
    *,
    seed: int,
    selector: Callable[
        [SRMAPPOPhaseAEnv, Dict[str, AgentObservation]],
        Dict[str, HybridAction] | Tuple[Dict[str, HybridAction], Dict[str, ShieldedAction]],
    ],
    debug_reward_terms: bool = False,
    debug_reward_steps: bool = False,
    reward_trace_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    from .trainer import disable_eval_fallback, restore_eval_fallback

    observations, _info = env.reset(seed=seed)
    env._baseline_recent_puncture_losses_bps = []
    done = False
    avg_sinr_samples: List[float] = []
    constraint_violations: Dict[str, int] = {}
    correction_count = 0
    episode_reward_sum = 0.0
    episode_step_count = 0
    urllc_reliability_violation_count = 0
    safe_admitted_urllc_count = 0
    per_admission_embb_damage_samples: List[Dict[str, object]] = []
    embb_user_minrate_trace: List[Dict[str, object]] = []
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
            selector_output = selector(env, observations)
            if isinstance(selector_output, tuple):
                joint_actions, resolved_override = selector_output
            else:
                joint_actions = selector_output
                resolved_override = None
            if planning_phase:
                if resolved_override is not None:
                    resolved = resolved_override
                else:
                    resolved = {
                        aid: env._raw_action_to_shielded_action(joint_actions[aid], observations[aid])
                        for aid in env.agent_ids
                    }
            else:
                minislot, rb = env._current_cell()
                if resolved_override is not None:
                    resolved = resolved_override
                else:
                    resolved = env._resolve_executed_actions(
                        joint_actions,
                        observations,
                        minislot=int(minislot),
                        rb=int(rb),
                    )
                _log_resolved_admission_damage_samples(
                    env=env,
                    resolved=resolved,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
                    rb=int(rb),
                    samples=per_admission_embb_damage_samples,
                )
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
                    mode_int = int(shielded.action.mode)
                    if mode_int in {MODE_OVERLAY, MODE_PUNCTURE}:
                        reliability = float(candidate.reliability_for_mode(mode_int))
                        target_success = 1.0 - float(getattr(env.urllc_cfg, "target_error_probability", 0.0) or 0.0)
                        reliability_ok = bool(reliability + 1.0e-12 >= target_success)
                        minrate_ok = not bool(candidate.cause_embb_retention_below_threshold)
                        if not reliability_ok:
                            urllc_reliability_violation_count += 1
                        if reliability_ok and minrate_ok:
                            safe_admitted_urllc_count += 1
                    if int(shielded.action.mode) == MODE_OVERLAY:
                        avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                    elif int(shielded.action.mode) == MODE_PUNCTURE:
                        avg_sinr_samples.append(float(candidate.puncture_urllc_snir))

            observations, rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            if (not planning_phase) and int(rb) >= int(env.sys_cfg.num_subcarriers) - 1:
                _append_embb_user_minrate_trace_for_minislot(
                    embb_user_minrate_trace,
                    env=env,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
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
        urllc_reliability_violation_count=float(urllc_reliability_violation_count),
        safe_admitted_urllc_count=float(safe_admitted_urllc_count),
        per_admission_embb_damage_samples=per_admission_embb_damage_samples,
        embb_user_minrate_trace=embb_user_minrate_trace,
        reward_term_stats=finalized_reward_term_stats,
        reward_term_rankings=reward_term_rankings,
        step_reward_term_stats=finalized_step_reward_term_stats,
        step_reward_term_rankings=step_reward_term_rankings,
        terminal_reward_term_stats=finalized_terminal_reward_term_stats,
        terminal_reward_term_rankings=terminal_reward_term_rankings,
        logged_step_reward_sum=logged_step_reward_sum,
        logged_terminal_reward_sum=logged_terminal_reward_sum,
        reward_trace_steps=reward_trace_steps if debug_reward_steps else None,
        embb_user_tx_powers=_extract_final_embb_user_tx_powers(env),
    )


def _load_mappo_model(
    cfg: SRMAPPOConfig,
    checkpoint_path: str | Path,
    component_overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[SRMAPPOPhaseAEnv, SRMAPPOActorCritic]:
    import torch
    from .networks import SRMAPPOActorCritic

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
    model_cfg.env.urllc_arrival_mode = str(getattr(cfg.env, "urllc_arrival_mode", "bernoulli") or "bernoulli")
    model_cfg.env.urllc_bernoulli_tx_prob = float(
        getattr(cfg.env, "urllc_bernoulli_tx_prob", 0.5) or 0.5
    )
    _enable_phase_a_joint_minrate_protection(model_cfg)
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


def _extract_final_embb_user_tx_powers(env: SRMAPPOPhaseAEnv) -> List[float]:
    try:
        allocator_powers = np.asarray(getattr(env.allocator, "embb_user_tx_power", []), dtype=float)
    except Exception:
        allocator_powers = np.asarray([], dtype=float)
    if allocator_powers.size > 0:
        return [float(x) for x in allocator_powers.tolist()]

    try:
        embb_result = dict(getattr(env, "embb_result", {}) or {})
        result_powers = np.asarray(embb_result.get("user_tx_powers", []), dtype=float)
    except Exception:
        result_powers = np.asarray([], dtype=float)
    if result_powers.size > 0:
        return [float(x) for x in result_powers.tolist()]
    return []


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
    from .trainer import disable_eval_fallback, restore_eval_fallback

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
    urllc_reliability_violation_count = 0
    safe_admitted_urllc_count = 0
    per_admission_embb_damage_samples: List[Dict[str, object]] = []
    embb_user_minrate_trace: List[Dict[str, object]] = []
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
                _log_resolved_admission_damage_samples(
                    env=env,
                    resolved=resolved,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
                    rb=int(rb),
                    samples=per_admission_embb_damage_samples,
                )
                for shielded in resolved.values():
                    candidate = shielded.candidate
                    if candidate is None:
                        continue
                    mode_int = int(shielded.action.mode)
                    if mode_int in {MODE_OVERLAY, MODE_PUNCTURE}:
                        reliability = float(candidate.reliability_for_mode(mode_int))
                        target_success = 1.0 - float(getattr(env.urllc_cfg, "target_error_probability", 0.0) or 0.0)
                        reliability_ok = bool(reliability + 1.0e-12 >= target_success)
                        minrate_ok = not bool(candidate.cause_embb_retention_below_threshold)
                        if not reliability_ok:
                            urllc_reliability_violation_count += 1
                        if reliability_ok and minrate_ok:
                            safe_admitted_urllc_count += 1
                    if int(shielded.action.mode) == MODE_OVERLAY:
                        avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                    elif int(shielded.action.mode) == MODE_PUNCTURE:
                        avg_sinr_samples.append(float(candidate.puncture_urllc_snir))
            observations, rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            if (not planning_phase) and int(rb) >= int(env.sys_cfg.num_subcarriers) - 1:
                _append_embb_user_minrate_trace_for_minislot(
                    embb_user_minrate_trace,
                    env=env,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
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
        urllc_reliability_violation_count=float(urllc_reliability_violation_count),
        safe_admitted_urllc_count=float(safe_admitted_urllc_count),
        per_admission_embb_damage_samples=per_admission_embb_damage_samples,
        embb_user_minrate_trace=embb_user_minrate_trace,
        reward_term_stats=finalized_reward_term_stats,
        reward_term_rankings=reward_term_rankings,
        step_reward_term_stats=finalized_step_reward_term_stats,
        step_reward_term_rankings=step_reward_term_rankings,
        terminal_reward_term_stats=finalized_terminal_reward_term_stats,
        terminal_reward_term_rankings=terminal_reward_term_rankings,
        logged_step_reward_sum=logged_step_reward_sum,
        logged_terminal_reward_sum=logged_terminal_reward_sum,
        reward_trace_steps=reward_trace_steps if debug_reward_steps else None,
        embb_user_tx_powers=_extract_final_embb_user_tx_powers(env),
    )


def _rollout_with_naive_random(
    env: SRMAPPOPhaseAEnv,
    *,
    seed: int,
    debug_reward_terms: bool = False,
    debug_reward_steps: bool = False,
    reward_trace_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    from .trainer import disable_eval_fallback, restore_eval_fallback

    observations, _info = env.reset(seed=seed)
    env._baseline_recent_puncture_losses_bps = []
    done = False
    rng = np.random.default_rng(int(seed))
    avg_sinr_samples: List[float] = []
    start = perf_counter()
    episode_reward_sum = 0.0
    episode_step_count = 0
    reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    step_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    terminal_reward_term_stats_accum: Dict[str, Dict[str, float]] = {}
    reward_trace_steps: List[Dict[str, object]] = []
    fallback_state = disable_eval_fallback(env)
    constraint_violations: Dict[str, int] = {}
    urllc_reliability_violation_count = 0
    safe_admitted_urllc_count = 0
    per_admission_embb_damage_samples: List[Dict[str, object]] = []
    embb_user_minrate_trace: List[Dict[str, object]] = []
    try:
        while not done:
            _merge_counts(constraint_violations, _collect_candidate_violation_counts(observations))
            planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in env.agent_ids)
            joint_actions: Dict[str, HybridAction] = {}
            resolved: Dict[str, ShieldedAction] = {}
            minislot = -1
            rb = -1
            if not planning_phase:
                minislot, rb = env._current_cell()
            for agent_id, obs in observations.items():
                if planning_phase:
                    action = _planning_baseline_action(env, obs)
                    joint_actions[agent_id] = action
                    resolved[agent_id] = env._raw_action_to_shielded_action(action, obs)
                    continue
                action, shielded = _naive_random_action_and_resolution(env, obs, rng)
                joint_actions[agent_id] = action
                resolved[agent_id] = shielded
                candidate = shielded.candidate
                if candidate is None:
                    continue
                mode_int = int(shielded.action.mode)
                if mode_int in {MODE_OVERLAY, MODE_PUNCTURE}:
                    reliability = float(candidate.reliability_for_mode(mode_int))
                    target_success = 1.0 - float(getattr(env.urllc_cfg, "target_error_probability", 0.0) or 0.0)
                    reliability_ok = bool(reliability + 1.0e-12 >= target_success)
                    minrate_ok = not bool(candidate.cause_embb_retention_below_threshold)
                    if not reliability_ok:
                        urllc_reliability_violation_count += 1
                    if reliability_ok and minrate_ok:
                        safe_admitted_urllc_count += 1
                if mode_int == MODE_OVERLAY:
                    avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                elif mode_int == MODE_PUNCTURE:
                    avg_sinr_samples.append(float(candidate.puncture_urllc_snir))

            if not planning_phase:
                _log_resolved_admission_damage_samples(
                    env=env,
                    resolved=resolved,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
                    rb=int(rb),
                    samples=per_admission_embb_damage_samples,
                )

            observations, rewards, dones, _infos = env.step(
                joint_actions,
                prebuilt_observations=observations,
                pre_resolved_actions=resolved,
            )
            if (not planning_phase) and int(rb) >= int(env.sys_cfg.num_subcarriers) - 1:
                _append_embb_user_minrate_trace_for_minislot(
                    embb_user_minrate_trace,
                    env=env,
                    method=str(getattr(env, "_eval_method_name", "")),
                    seed=int(seed),
                    episode=1,
                    decision_step=int(episode_step_count),
                    minislot=int(minislot),
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
        urllc_reliability_violation_count=float(urllc_reliability_violation_count),
        safe_admitted_urllc_count=float(safe_admitted_urllc_count),
        per_admission_embb_damage_samples=per_admission_embb_damage_samples,
        embb_user_minrate_trace=embb_user_minrate_trace,
        reward_term_stats=finalized_reward_term_stats,
        reward_term_rankings=reward_term_rankings,
        step_reward_term_stats=finalized_step_reward_term_stats,
        step_reward_term_rankings=step_reward_term_rankings,
        terminal_reward_term_stats=finalized_terminal_reward_term_stats,
        terminal_reward_term_rankings=terminal_reward_term_rankings,
        logged_step_reward_sum=logged_step_reward_sum,
        logged_terminal_reward_sum=logged_terminal_reward_sum,
        reward_trace_steps=reward_trace_steps if debug_reward_steps else None,
        embb_user_tx_powers=_extract_final_embb_user_tx_powers(env),
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
    if normalized == "hard_feasible_throughput_greedy" and isinstance(config_dict, dict):
        _ensure_hard_feasible_phase0_minrate_policy(config_dict, cfg)
    if normalized in {
        "mappo",
        "mappo_overlay_forced",
        "mappo_puncture_forced",
        "greedy",
        "myopic_throughput_greedy",
        "hard_feasible_throughput_greedy",
        "throughput_only_greedy",
        "global_frontier_greedy",
    }:
        _enable_phase_a_joint_minrate_protection(cfg)
    reward_checkpoint_path = None
    if isinstance(config_dict, dict):
        reward_checkpoint_path = config_dict.get("reward_checkpoint_path")
    if reward_checkpoint_path:
        import torch

        reward_payload = torch.load(Path(str(reward_checkpoint_path)).expanduser(), map_location="cpu", weights_only=False)
        reward_checkpoint_cfg = reward_payload.get("cfg")
        if isinstance(reward_checkpoint_cfg, SRMAPPOConfig):
            cfg.reward = deepcopy(reward_checkpoint_cfg.reward)
        elif isinstance(reward_checkpoint_cfg, dict):
            reward_cfg = dict(reward_checkpoint_cfg.get("reward", {}) or {})
            for key, value in reward_cfg.items():
                if hasattr(cfg.reward, key):
                    setattr(cfg.reward, key, value)
    component_overrides = {
        name: dict(config_dict.get(name, {}) or {})
        for name in ("system", "simulation", "algorithm", "urllc", "embb")
    }
    if normalized in {"greedy", "pure_puncturing", "pure_superposition", "random_scheduler"}:
        cfg.action.max_candidate_packets = max(
            int(getattr(cfg.action, "max_candidate_packets", 0) or 0),
            int(_baseline_candidate_pool_upper_bound(component_overrides=component_overrides)),
        )
    total_load = config_dict.get("total_load") if isinstance(config_dict, dict) else None
    mix_ratio = config_dict.get("mix_ratio") if isinstance(config_dict, dict) else None
    explicit_mix_weights_raw = config_dict.get("explicit_mix_weights") if isinstance(config_dict, dict) else None
    explicit_mix_weights = None
    if isinstance(explicit_mix_weights_raw, (tuple, list)) and len(explicit_mix_weights_raw) >= 2:
        explicit_mix_weights = (
            float(explicit_mix_weights_raw[0]),
            float(explicit_mix_weights_raw[1]),
        )

    if normalized in {"mappo", "mappo_overlay_forced", "mappo_puncture_forced"}:
        forced_mode = None
        if normalized == "mappo_overlay_forced":
            forced_mode = MODE_OVERLAY
        elif normalized == "mappo_puncture_forced":
            forced_mode = MODE_PUNCTURE
        checkpoint_path = config_dict.get("checkpoint_path")
        if not checkpoint_path and bool(config_dict.get("train", False)):
            from .trainer import run_training_loop

            result = run_training_loop(cfg, evaluation_fn=None)
            training_curve = _extract_training_curve_from_history(list(result.get("history", []) or []))
            checkpoint_dir = Path(result["checkpoint_dir"])
            final_path = checkpoint_dir / f"{cfg.training.run_name}_final.pt"
            env, model = _load_mappo_model(cfg, final_path, component_overrides=component_overrides)
            env.enable_mode_downstream_logging = True
            env.mode_downstream_horizons = (5, 10)
            _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio, explicit_mix_weights=explicit_mix_weights)
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
        env._eval_method_name = normalized
        _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio, explicit_mix_weights=explicit_mix_weights)
        metrics = _rollout_with_mappo_model(
            env,
            model,
            seed=seed,
            forced_mode=forced_mode,
            debug_reward_terms=debug_reward_terms,
            debug_reward_steps=debug_reward_steps,
            reward_trace_context=reward_trace_context,
        )
        import torch

        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        history = list(((payload.get("extra") or {}).get("history") or []))
        metrics["training_reward_curve"] = _extract_training_curve_from_history(history)
        return metrics

    if normalized == "naive_random":
        env = _build_env(cfg, component_overrides=component_overrides)
        env.enable_mode_downstream_logging = True
        env.mode_downstream_horizons = (5, 10)
        env._eval_method_name = normalized
        _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio, explicit_mix_weights=explicit_mix_weights)
        return _rollout_with_naive_random(
            env,
            seed=seed,
            debug_reward_terms=debug_reward_terms,
            debug_reward_steps=debug_reward_steps,
            reward_trace_context=reward_trace_context,
        )

    if normalized == "ippo":
        from .ippo_baseline import IPPOBaselineTrainer

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
    env._eval_method_name = normalized
    _configure_eval_env(env, total_load=total_load, mix_ratio=mix_ratio, explicit_mix_weights=explicit_mix_weights)
    rng = np.random.default_rng(int(seed))

    if normalized in {"original", "original_greedy_normal_v1", "original_greedy_normal_v2"}:
        runner = _run_original_greedy_episode
        if normalized == "original_greedy_normal_v1":
            runner = _run_original_greedy_normal_v1_episode
        elif normalized == "original_greedy_normal_v2":
            runner = _run_original_greedy_normal_v2_episode
        return _run_legacy_original_greedy_policy(env, seed=seed, runner=runner)

    if normalized == "random_scheduler":
        selector = lambda e, o: _random_policy_actions(e, o, rng)
    elif normalized == "pure_puncturing":
        selection_rule = str(config_dict.get("puncturing_selection_rule", "max_embb_sum_rate"))
        selector = lambda e, o: _pure_puncturing_actions(e, o, selection_rule)
    elif normalized == "pure_superposition":
        selection_rule = str(config_dict.get("superposition_selection_rule", "max_embb_sum_rate"))
        selector = lambda e, o: _pure_superposition_actions(e, o, selection_rule)
    elif normalized in {"greedy", "myopic_throughput_greedy"}:
        selector = _greedy_actions
    elif normalized == "hard_feasible_throughput_greedy":
        selector = lambda e, o: _hard_feasible_throughput_actions(e, o)[0]
    elif normalized == "throughput_only_greedy":
        selector = lambda e, o: _throughput_only_actions(e, o)[0]
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
