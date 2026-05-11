"""Phase-A SR-MAPPO environment for URLLC coexistence over a fixed eMBB baseline."""

from copy import deepcopy
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from capacity_models import CapacityModels
from config import AlgorithmConfig, SimulationConfig, SystemConfig, URLLCConfig, eMBBConfig
from resource_allocator import ResourceAllocator
from simulation import MultiUAVSimulation, create_simulation

from .config import SRMAPPOConfig
from .load_aware import (
    load_aware_reward_schedule,
    nearest_reference_load,
    overlay_retention_gate_for_load,
    power_ratio_ceiling_for_load,
    puncture_loss_ceiling_for_load,
)
from .shield import FeasibilityShield
from .types import (
    MODE_KEEP,
    MODE_NAMES,
    MODE_OVERLAY,
    MODE_PUNCTURE,
    ActionMaskBundle,
    AgentObservation,
    CandidatePacket,
    HybridAction,
    ShieldedAction,
)


class SRMAPPOPhaseAEnv:
    """MAPPO-ready environment focused on URLLC coexistence decisions."""

    def __init__(
        self,
        sys_cfg: SystemConfig,
        urllc_cfg: URLLCConfig,
        embb_cfg: eMBBConfig,
        algo_cfg: AlgorithmConfig,
        sim_cfg: SimulationConfig,
        rl_cfg: Optional[SRMAPPOConfig] = None,
    ):
        self.sys_cfg = deepcopy(sys_cfg)
        self.urllc_cfg = deepcopy(urllc_cfg)
        self.embb_cfg = deepcopy(embb_cfg)
        self.algo_cfg = deepcopy(algo_cfg)
        self.sim_cfg = deepcopy(sim_cfg)
        self.rl_cfg = rl_cfg or SRMAPPOConfig()
        self.training_progress_frac = 1.0
        self.phase_a_embb_power_enabled = bool(getattr(self.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))

        if self.rl_cfg.env.multi_rb_agents:
            agent_ids = []
            self._agent_index_map = {}
            self._agent_id_by_uav_rb = {}
            for uav_idx in range(self.sys_cfg.num_uavs):
                for rb_idx in range(self.sys_cfg.num_subcarriers):
                    agent_id = f"uav_{uav_idx}_rb_{rb_idx}"
                    agent_ids.append(agent_id)
                    self._agent_index_map[agent_id] = (uav_idx, rb_idx)
                    self._agent_id_by_uav_rb[(uav_idx, rb_idx)] = agent_id
            self.agent_ids = agent_ids
        else:
            self.agent_ids = [f"uav_{idx}" for idx in range(self.sys_cfg.num_uavs)]
            self._agent_index_map = {agent_id: (idx, 0) for idx, agent_id in enumerate(self.agent_ids)}
            self._agent_id_by_uav_rb = {(idx, 0): agent_id for idx, agent_id in enumerate(self.agent_ids)}
        self.simulation = MultiUAVSimulation(
            self.sys_cfg,
            self.urllc_cfg,
            self.embb_cfg,
            self.algo_cfg,
            self.sim_cfg,
        )
        self.channel_model = self.simulation.channel_model
        self.capacity_model = CapacityModels(self.urllc_cfg, self.embb_cfg)
        self.allocator = ResourceAllocator(
            self.sys_cfg,
            self.urllc_cfg,
            self.embb_cfg,
            self.algo_cfg,
        )
        self.shield = FeasibilityShield(self.rl_cfg)

        self.local_obs_dim = self._local_obs_dim()
        self.global_obs_dim = self._global_obs_dim()
        self.num_packet_options = self.rl_cfg.action.max_candidate_packets + 1

        if self.rl_cfg.env.multi_rb_agents:
            self._cell_schedule = list(range(self.sys_cfg.num_minislots))
        else:
            self._cell_schedule = [
                (minislot, rb)
                for minislot in range(self.sys_cfg.num_minislots)
                for rb in range(self.sys_cfg.num_subcarriers)
            ]
        self._embb_plan_schedule = list(range(self.sys_cfg.num_subcarriers))
        self._reset_episode_state()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, AgentObservation], Dict]:
        """Reset a new slot-level episode."""
        _reset_t0 = perf_counter()
        self.current_reset_seed = int(self.sim_cfg.random_seed if seed is None else seed)
        if seed is not None:
            np.random.seed(seed)
        self.sys_cfg.random_seed = self.current_reset_seed
        if hasattr(self.channel_model, "set_seed"):
            self.channel_model.set_seed(self.current_reset_seed)
        self._reset_episode_state()
        _slot_ctx_t0 = perf_counter()
        self._prepare_slot_context()
        self.profile_prepare_slot_context_sec = float(perf_counter() - _slot_ctx_t0)
        needs_greedy_reference = bool(self.rl_cfg.reward.use_greedy_terminal_reference)
        if not needs_greedy_reference:
            # Some terminal penalties compare against a per-episode greedy reference (e.g., power over greedy).
            # Enable the reference metrics only when those penalties are active.
            needs_greedy_reference = bool(
                float(getattr(self.rl_cfg.reward, "terminal_power_ratio_penalty_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.rl_cfg.reward, "terminal_total_power_over_greedy_penalty_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.rl_cfg.reward, "terminal_embb_power_over_greedy_penalty_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.rl_cfg.reward, "terminal_embb_service_gain_vs_greedy_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.rl_cfg.reward, "terminal_embb_minrate_gain_vs_greedy_weight", 0.0) or 0.0) > 0.0
                or float(getattr(self.rl_cfg.reward, "terminal_embb_service_vs_greedy_shortfall_penalty_weight", 0.0) or 0.0) > 0.0
            )
        if needs_greedy_reference:
            self._prepare_original_greedy_reference()
        observations = self._build_observations()
        self.profile_reset_total_sec = float(perf_counter() - _reset_t0)
        info = {
            "slot_index": 0,
            "active_packets": int(self.packet_arrivals_by_minislot[0]) if self.packet_arrivals_by_minislot.size > 0 else 0,
            "cell_order_length": len(self._cell_schedule),
        }
        return observations, info

    def step(
        self,
        joint_actions: Dict[str, HybridAction],
    ) -> Tuple[Dict[str, AgentObservation], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        """Advance one joint (RB, minislot) decision step for all UAVs."""
        step_start = perf_counter()
        self.step_calls += 1
        try:
            if self.episode_done:
                raise RuntimeError("Episode already finished. Call reset() first.")
            if self.rl_cfg.env.learn_embb_baseline and not self.planning_done:
                return self._step_embb_planning(joint_actions)

            current_obs_start = perf_counter()
            observations = self._build_observations()
            current_obs_elapsed = perf_counter() - current_obs_start
            self.build_observations_current_step_calls += 1
            self.build_observations_current_step_total_sec += current_obs_elapsed
            for agent_id in self.agent_ids:
                candidates = observations[agent_id].candidates
                self.overlay_candidate_pairs += len(candidates)
                self.overlay_feasible_pairs += sum(int(candidate.overlay_feasible) for candidate in candidates)
                self.phase_a_feasible_candidate_pairs += sum(
                    int(bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible))
                    for candidate in candidates
                )
            if self.rl_cfg.env.multi_rb_agents:
                minislot = self._cell_schedule[self.current_cell_index]
                rb = None
            else:
                minislot, rb = self._cell_schedule[self.current_cell_index]
            shielded = self._resolve_executed_actions(
                joint_actions,
                observations,
                minislot=minislot,
                rb=rb,
            )

            team_reward = 0.0
            step_info = {}
            shared_reward_terms = {}
            for agent_id in self.agent_ids:
                uav_idx, rb_idx = self._agent_index_map[agent_id]
                outcome = self._apply_agent_action(
                    uav_idx=uav_idx,
                    agent_id=agent_id,
                    minislot=minislot,
                    rb=rb_idx if rb is None else rb,
                    observation=observations[agent_id],
                    shielded_action=shielded[agent_id],
                )
                step_info[agent_id] = outcome
                team_reward += outcome["reward"]
                for key, value in outcome.get("reward_terms", {}).items():
                    shared_reward_terms[key] = shared_reward_terms.get(key, 0.0) + float(value)

            # Non-carryover URLLC policy: packets that arrive in this minislot must be served now.
            # If still unscheduled at minislot end, they are dropped immediately (no cross-minislot queue).
            self._drop_unscheduled_packets_of_minislot(minislot)

            self.current_cell_index += 1
            if self._share_budget_exhausted():
                self.episode_done = True
            if self.rl_cfg.env.early_terminate_when_all_packets_scheduled and not self.unscheduled_packet_ids:
                self.episode_done = True
            if self.current_cell_index >= len(self._cell_schedule):
                self.episode_done = True

            terminal_reward_terms = {}
            if self.episode_done:
                summary = self.summarize_episode()
                load_reward_weights = self._get_load_aware_reward_weights(self._current_actual_load())
                if self.rl_cfg.reward.use_greedy_terminal_reference:
                    baseline_embb_total_rate = max(float(getattr(self, "original_greedy_embb_total_rate", 0.0)), 1e-9)
                    embb_rate_gain = float(summary['embb_total_rate'] / baseline_embb_total_rate) - 1.0
                    terminal_reward_terms["terminal_embb_rate_gain_vs_greedy"] = load_reward_weights["terminal_embb_rate_weight"] * embb_rate_gain
                    team_reward += terminal_reward_terms["terminal_embb_rate_gain_vs_greedy"]
                    baseline_fairness = float(getattr(self, "original_greedy_jain_fairness", 0.0))
                    fairness_gain = float(summary.get('jain_fairness', 0.0) - baseline_fairness)
                    terminal_reward_terms["terminal_embb_fairness_gain_vs_greedy"] = (
                        self.rl_cfg.reward.terminal_embb_fairness_weight * fairness_gain
                    )
                    team_reward += terminal_reward_terms["terminal_embb_fairness_gain_vs_greedy"]
                else:
                    embb_rate_scaled = float(summary['embb_total_rate'] / max(self.rl_cfg.reward.terminal_embb_rate_normalizer, 1e-9))
                    terminal_reward_terms["terminal_embb_rate"] = load_reward_weights["terminal_embb_rate_weight"] * embb_rate_scaled
                    team_reward += terminal_reward_terms["terminal_embb_rate"]
                    fairness = float(summary.get('jain_fairness', 0.0))
                    terminal_reward_terms["terminal_embb_fairness"] = self.rl_cfg.reward.terminal_embb_fairness_weight * fairness
                    team_reward += terminal_reward_terms["terminal_embb_fairness"]
                admission_ratio = float(summary.get('urllc_admission_rate', 0.0))
                terminal_reward_terms["terminal_urllc_admission"] = (
                    load_reward_weights["terminal_urllc_admission_weight"] * admission_ratio
                )
                team_reward += terminal_reward_terms["terminal_urllc_admission"]
                # Reliability-first terminal penalty: violating URLLC reliability floor should be
                # much more costly than gaining admission ratio.
                admitted_reliability = float(
                    summary.get("admitted_urllc_reliability", summary.get("urllc_success_rate", 1.0))
                )
                reliability_floor = float(
                    getattr(self.rl_cfg.reward, "terminal_urllc_reliability_floor", 0.0) or 0.0
                )
                reliability_shortfall_w = float(
                    getattr(
                        self.rl_cfg.reward,
                        "terminal_urllc_reliability_shortfall_penalty_weight",
                        0.0,
                    )
                    or 0.0
                )
                reliability_hard_violation_penalty = float(
                    getattr(
                        self.rl_cfg.reward,
                        "terminal_urllc_reliability_hard_violation_penalty",
                        0.0,
                    )
                    or 0.0
                )
                if reliability_floor > 0.0 and admitted_reliability < reliability_floor - 1.0e-12:
                    shortfall = reliability_floor - admitted_reliability
                    if reliability_shortfall_w > 0.0:
                        penalty = reliability_shortfall_w * (shortfall * shortfall)
                        terminal_reward_terms["terminal_urllc_reliability_shortfall_penalty"] = -penalty
                        team_reward += terminal_reward_terms["terminal_urllc_reliability_shortfall_penalty"]
                    if reliability_hard_violation_penalty > 0.0:
                        terminal_reward_terms["terminal_urllc_reliability_hard_violation_penalty"] = (
                            -reliability_hard_violation_penalty
                        )
                        team_reward += terminal_reward_terms[
                            "terminal_urllc_reliability_hard_violation_penalty"
                        ]
                admission_collapse_penalty_weight = float(
                    getattr(self.rl_cfg.reward, "terminal_admission_collapse_penalty_weight", 0.0) or 0.0
                )
                if admission_collapse_penalty_weight > 0.0:
                    collapse_floor = float(self._current_admission_collapse_floor(self._current_actual_load()))
                    collapse_shortfall = max(collapse_floor - admission_ratio, 0.0)
                    if collapse_floor > 0.0 and collapse_shortfall > 0.0:
                        terminal_reward_terms["terminal_admission_collapse_penalty"] = (
                            -admission_collapse_penalty_weight * collapse_shortfall
                        )
                        team_reward += terminal_reward_terms["terminal_admission_collapse_penalty"]
                admission_target = float(load_reward_weights["terminal_urllc_admission_target"])
                admission_penalty = float(getattr(self.rl_cfg.reward, "terminal_urllc_admission_penalty", 0.0))
                if admission_target > 0.0 and admission_penalty > 0.0:
                    shortfall = max(admission_target - admission_ratio, 0.0)
                    terminal_reward_terms["terminal_urllc_admission_shortfall"] = -admission_penalty * shortfall
                    team_reward += terminal_reward_terms["terminal_urllc_admission_shortfall"]
                unscheduled_penalty = float(load_reward_weights["terminal_unscheduled_penalty"])
                if self.rl_cfg.env.keep_unscheduled_packets_as_terminal_penalty and unscheduled_penalty > 0.0:
                    unscheduled_ratio = float(summary.get("unscheduled_ratio", 0.0))
                    terminal_reward_terms["terminal_unscheduled_packets"] = -unscheduled_penalty * unscheduled_ratio
                    team_reward += terminal_reward_terms["terminal_unscheduled_packets"]
                scheduled_packets_reward_weight = float(getattr(self.rl_cfg.reward, "terminal_scheduled_packets_reward_weight", 0.0))
                if scheduled_packets_reward_weight > 0.0:
                    scheduled_packets_per_uav = float(summary.get("scheduled_packets_per_uav", 0.0))
                    terminal_reward_terms["terminal_scheduled_packets_reward"] = (
                        scheduled_packets_reward_weight * scheduled_packets_per_uav
                    )
                    team_reward += terminal_reward_terms["terminal_scheduled_packets_reward"]
                zero_admission_active_penalty = float(getattr(self.rl_cfg.reward, "terminal_zero_admission_active_penalty", 0.0))
                if (
                    zero_admission_active_penalty > 0.0
                    and float(summary.get("active_packets", 0.0)) > 0.0
                    and float(summary.get("scheduled_packets", 0.0)) <= 0.0
                ):
                    terminal_reward_terms["terminal_zero_admission_active_penalty"] = -zero_admission_active_penalty
                    team_reward += terminal_reward_terms["terminal_zero_admission_active_penalty"]
                no_coexistence_penalty = float(getattr(self.rl_cfg.reward, "terminal_no_coexistence_with_feasible_penalty", 0.0))
                if (
                    no_coexistence_penalty > 0.0
                    and float(summary.get("overlay_count", 0.0)) <= 0.0
                    and float(summary.get("puncture_count", 0.0)) <= 0.0
                    and float(summary.get("phase_a_feasible_candidate_pairs", 0.0)) > 0.0
                ):
                    terminal_reward_terms["terminal_no_coexistence_with_feasible_penalty"] = -no_coexistence_penalty
                    team_reward += terminal_reward_terms["terminal_no_coexistence_with_feasible_penalty"]
                embb_min_penalty = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_penalty", 0.0))
                if embb_min_penalty > 0.0:
                    avg_shortfall = float(summary.get("embb_min_rate_shortfall", 0.0))
                    terminal_reward_terms["terminal_embb_min_rate_shortfall"] = -embb_min_penalty * avg_shortfall
                    team_reward += terminal_reward_terms["terminal_embb_min_rate_shortfall"]
                throughput_per_watt_weight = float(getattr(self.rl_cfg.reward, "terminal_throughput_per_watt_weight", 0.0) or 0.0)
                if throughput_per_watt_weight > 0.0:
                    throughput_per_watt = float(summary.get("throughput_per_watt", 0.0))
                    throughput_per_watt_scaled = throughput_per_watt / max(
                        float(getattr(self.rl_cfg.reward, "terminal_throughput_per_watt_normalizer", 1.0e6) or 1.0e6),
                        1.0e-9,
                    )
                    terminal_reward_terms["terminal_throughput_per_watt"] = throughput_per_watt_weight * throughput_per_watt_scaled
                    team_reward += terminal_reward_terms["terminal_throughput_per_watt"]
                served_user_rate_weight = float(getattr(self.rl_cfg.reward, "terminal_served_user_rate_weight", 0.0) or 0.0)
                if served_user_rate_weight > 0.0:
                    served_user_rate = float(summary.get("avg_throughput_per_served_embb_user", 0.0))
                    served_user_rate_scaled = served_user_rate / max(
                        float(getattr(self.rl_cfg.reward, "terminal_served_user_rate_normalizer", 1.0e6) or 1.0e6),
                        1.0e-9,
                    )
                    terminal_reward_terms["terminal_served_user_rate"] = served_user_rate_weight * served_user_rate_scaled
                    team_reward += terminal_reward_terms["terminal_served_user_rate"]
                thin_service_penalty_weight = float(getattr(self.rl_cfg.reward, "terminal_thin_service_penalty_weight", 0.0) or 0.0)
                if thin_service_penalty_weight > 0.0:
                    thin_service_fraction = float(summary.get("thin_service_fraction", 0.0))
                    terminal_reward_terms["terminal_thin_service_penalty"] = -thin_service_penalty_weight * thin_service_fraction
                    team_reward += terminal_reward_terms["terminal_thin_service_penalty"]
                puncture_loss_penalty = float(getattr(self.rl_cfg.reward, "terminal_puncture_loss_penalty_weight", 0.0))
                if puncture_loss_penalty > 0.0:
                    avg_puncture_loss = float(summary.get("avg_puncture_embb_loss", 0.0)) / 1.0e6
                    terminal_reward_terms["terminal_puncture_loss_penalty"] = -puncture_loss_penalty * avg_puncture_loss
                    team_reward += terminal_reward_terms["terminal_puncture_loss_penalty"]
                overlay_retention_bonus = float(getattr(self.rl_cfg.reward, "terminal_overlay_retention_bonus", 0.0))
                if overlay_retention_bonus > 0.0:
                    avg_overlay_retention = float(summary.get("avg_overlay_retention", 0.0))
                    terminal_reward_terms["terminal_overlay_retention_bonus"] = overlay_retention_bonus * avg_overlay_retention
                    team_reward += terminal_reward_terms["terminal_overlay_retention_bonus"]
                early_puncture_collapse_penalty = float(getattr(self.rl_cfg.reward, "early_puncture_collapse_penalty_weight", 0.0))
                early_puncture_collapse_end_frac = float(getattr(self.rl_cfg.reward, "early_puncture_collapse_end_frac", 0.0))
                training_progress = float(np.clip(getattr(self, "training_progress_frac", 1.0), 0.0, 1.0))
                if (
                    early_puncture_collapse_penalty > 0.0
                    and early_puncture_collapse_end_frac > 0.0
                    and training_progress < early_puncture_collapse_end_frac - 1e-12
                    and float(summary.get("phase_a_total_decisions", 0.0)) > 0.0
                    and float(summary.get("overlay_feasible_pairs", 0.0)) > 0.0
                ):
                    overlay_ratio_floor = float(
                        getattr(self.rl_cfg.reward, "early_puncture_collapse_overlay_ratio_floor", 0.0) or 0.0
                    )
                    puncture_ratio_ceiling = float(
                        getattr(self.rl_cfg.reward, "early_puncture_collapse_puncture_ratio_ceiling", 1.0) or 1.0
                    )
                    overlay_shortfall = max(overlay_ratio_floor - float(summary.get("overlay_ratio", 0.0)), 0.0)
                    puncture_excess = max(float(summary.get("puncture_ratio", 0.0)) - puncture_ratio_ceiling, 0.0)
                    collapse_severity = overlay_shortfall + puncture_excess
                    if collapse_severity > 0.0:
                        warmup_scale = max(1.0 - training_progress / max(early_puncture_collapse_end_frac, 1.0e-9), 0.0)
                        terminal_reward_terms["terminal_early_puncture_collapse_penalty"] = (
                            -early_puncture_collapse_penalty * warmup_scale * collapse_severity
                        )
                        team_reward += terminal_reward_terms["terminal_early_puncture_collapse_penalty"]
                load_adaptive_mode_target_weight = float(getattr(self.rl_cfg.reward, "load_adaptive_mode_target_weight", 0.0))
                load_adaptive_start_load = float(getattr(self.rl_cfg.reward, "load_adaptive_start_load", 15.0) or 15.0)
                actual_load = float(self._current_actual_load())
                coexist_count = float(summary.get("overlay_count", 0.0) + summary.get("puncture_count", 0.0))
                if (
                    load_adaptive_mode_target_weight > 0.0
                    and actual_load >= load_adaptive_start_load - 1e-9
                    and float(summary.get("active_packets", 0.0)) > 0.0
                    and coexist_count > 0.0
                ):
                    puncture_floor = self._current_load_adaptive_puncture_floor(actual_load)
                    overlay_ceiling = self._current_load_adaptive_overlay_ceiling(actual_load)
                    puncture_ratio = float(summary.get("puncture_ratio", 0.0))
                    overlay_ratio = float(summary.get("overlay_ratio", 0.0))
                    puncture_shortfall = max(puncture_floor - puncture_ratio, 0.0)
                    overlay_excess = max(overlay_ratio - overlay_ceiling, 0.0)
                    if puncture_shortfall > 0.0:
                        terminal_reward_terms["terminal_load_adaptive_puncture_floor_penalty"] = (
                            -load_adaptive_mode_target_weight * puncture_shortfall
                        )
                        team_reward += terminal_reward_terms["terminal_load_adaptive_puncture_floor_penalty"]
                    if overlay_excess > 0.0:
                        terminal_reward_terms["terminal_load_adaptive_overlay_ceiling_penalty"] = (
                            -load_adaptive_mode_target_weight * overlay_excess
                        )
                        team_reward += terminal_reward_terms["terminal_load_adaptive_overlay_ceiling_penalty"]
                frontier_mode_bonus_weight = float(getattr(self.rl_cfg.reward, "frontier_mode_bonus_weight", 0.0) or 0.0)
                frontier_mode_penalty_weight = float(getattr(self.rl_cfg.reward, "frontier_mode_penalty_weight", 0.0) or 0.0)
                frontier_puncture_floor = self._current_frontier_puncture_floor(actual_load)
                frontier_overlay_ceiling = self._current_frontier_overlay_ceiling(actual_load)
                if (
                    (frontier_mode_bonus_weight > 0.0 or frontier_mode_penalty_weight > 0.0)
                    and coexist_count > 0.0
                    and (
                        dict(getattr(self.rl_cfg.training, "frontier_puncture_floor_by_load", {}) or {})
                        or dict(getattr(self.rl_cfg.training, "frontier_overlay_ceiling_by_load", {}) or {})
                    )
                ):
                    puncture_ratio = float(summary.get("puncture_ratio", 0.0))
                    overlay_ratio = float(summary.get("overlay_ratio", 0.0))
                    puncture_shortfall = max(frontier_puncture_floor - puncture_ratio, 0.0)
                    overlay_excess = max(overlay_ratio - frontier_overlay_ceiling, 0.0)
                    if puncture_shortfall <= 1.0e-9 and overlay_excess <= 1.0e-9 and frontier_mode_bonus_weight > 0.0:
                        terminal_reward_terms["terminal_frontier_mode_bonus"] = frontier_mode_bonus_weight
                        team_reward += terminal_reward_terms["terminal_frontier_mode_bonus"]
                    elif frontier_mode_penalty_weight > 0.0:
                        terminal_reward_terms["terminal_frontier_mode_penalty"] = (
                            -frontier_mode_penalty_weight * (puncture_shortfall + overlay_excess)
                        )
                        team_reward += terminal_reward_terms["terminal_frontier_mode_penalty"]
                power_ratio_penalty = float(getattr(self.rl_cfg.reward, "terminal_power_ratio_penalty_weight", 0.0))
                if power_ratio_penalty > 0.0:
                    baseline_power = max(float(getattr(self, "original_greedy_metrics", {}).get("total_power", 0.0)), 1e-9)
                    power_ratio = float(summary.get("total_power", 0.0)) / baseline_power if baseline_power > 0.0 else 1.0
                    terminal_reward_terms["terminal_power_ratio_penalty"] = -power_ratio_penalty * max(power_ratio - 1.0, 0.0)
                    team_reward += terminal_reward_terms["terminal_power_ratio_penalty"]

                # Separation-oriented reward terms (optional; defaults are 0).
                pos_bonus_w = float(getattr(self.rl_cfg.reward, "embb_positive_rate_bonus_weight", 0.0) or 0.0)
                srv_bonus_w = float(getattr(self.rl_cfg.reward, "embb_service_ratio_bonus_weight", 0.0) or 0.0)
                pwr_overuse_w = float(getattr(self.rl_cfg.reward, "power_overuse_penalty_weight", 0.0) or 0.0)
                owner_util_w = float(getattr(self.rl_cfg.reward, "owner_change_utilization_bonus_weight", 0.0) or 0.0)
                owner_eff_w = float(getattr(self.rl_cfg.reward, "owner_effective_change_bonus_weight", 0.0) or 0.0)
                owner_restore_w = float(getattr(self.rl_cfg.reward, "owner_restored_to_snapshot_penalty_weight", 0.0) or 0.0)
                if not bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_reward", True)):
                    owner_util_w = 0.0
                    owner_eff_w = 0.0
                    owner_restore_w = 0.0

                # Use corrected effective eMBB KPIs (local puncture airtime deducted) for reward terms.
                service_ratio_corrected = float(
                    summary.get("embb_service_ratio_after_puncture_deduction", summary.get("embb_service_ratio", 0.0))
                )
                min_rate_ratio_corrected = float(
                    summary.get(
                        "embb_min_rate_satisfaction_after_puncture_deduction",
                        summary.get("embb_min_rate_satisfaction_ratio", 0.0),
                    )
                )
                if pos_bonus_w > 0.0:
                    terminal_reward_terms["terminal_embb_positive_rate_bonus"] = pos_bonus_w * float(summary.get("embb_positive_rate_ratio", 0.0))
                    team_reward += terminal_reward_terms["terminal_embb_positive_rate_bonus"]
                if srv_bonus_w > 0.0:
                    terminal_reward_terms["terminal_embb_service_ratio_bonus"] = srv_bonus_w * float(service_ratio_corrected)
                    team_reward += terminal_reward_terms["terminal_embb_service_ratio_bonus"]
                if pwr_overuse_w > 0.0:
                    terminal_reward_terms["terminal_power_overuse_penalty"] = -pwr_overuse_w * float(summary.get("total_power", 0.0))
                    team_reward += terminal_reward_terms["terminal_power_overuse_penalty"]

                # Service/throughput recovery terms (terminal).
                service_term_w = float(getattr(self.rl_cfg.reward, "terminal_embb_service_ratio_bonus_weight", 0.0) or 0.0)
                served_rate_term_w = float(getattr(self.rl_cfg.reward, "terminal_avg_served_embb_rate_bonus_weight", 0.0) or 0.0)
                served_rate_norm = float(getattr(self.rl_cfg.reward, "terminal_avg_served_embb_rate_bonus_normalizer", 1.0e6) or 1.0e6)
                min_rate_sat_w = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_satisfaction_bonus_weight", 0.0) or 0.0)
                if service_term_w > 0.0:
                    terminal_reward_terms["terminal_embb_service_ratio_bonus_v2"] = service_term_w * float(service_ratio_corrected)
                    team_reward += terminal_reward_terms["terminal_embb_service_ratio_bonus_v2"]
                if served_rate_term_w > 0.0:
                    served_rate = float(summary.get("avg_throughput_per_served_embb_user", 0.0))
                    ratio = float(served_rate / max(served_rate_norm, 1.0e-9))
                    ratio = float(min(max(ratio, 0.0), 2.0))
                    terminal_reward_terms["terminal_avg_served_embb_rate_bonus"] = served_rate_term_w * ratio
                    team_reward += terminal_reward_terms["terminal_avg_served_embb_rate_bonus"]
                if min_rate_sat_w > 0.0:
                    terminal_reward_terms["terminal_embb_min_rate_satisfaction_bonus"] = (
                        min_rate_sat_w * float(min_rate_ratio_corrected)
                    )
                    team_reward += terminal_reward_terms["terminal_embb_min_rate_satisfaction_bonus"]

                # Service-preserving terminal gates (quadratic penalties below floors) + small bonuses.
                srv_floor = float(getattr(self.rl_cfg.reward, "terminal_embb_service_floor", 0.0) or 0.0)
                srv_floor_w = float(getattr(self.rl_cfg.reward, "terminal_embb_service_floor_penalty_weight", 0.0) or 0.0)
                min_floor = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_floor", 0.0) or 0.0)
                min_floor_w = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_floor_penalty_weight", 0.0) or 0.0)
                srv_bonus_w2 = float(getattr(self.rl_cfg.reward, "terminal_embb_service_bonus_weight", 0.0) or 0.0)
                min_bonus_w2 = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_bonus_weight", 0.0) or 0.0)
                service_ratio = float(service_ratio_corrected)
                min_rate_ratio = float(min_rate_ratio_corrected)

                # Optional load-aware floors (nearest load key).
                def _nearest_load_floor(mapping, default: float) -> float:
                    try:
                        items = dict(mapping or {})
                    except Exception:
                        items = {}
                    if not items:
                        return float(default)
                    try:
                        load_val = float(summary.get("actual_load", actual_load if actual_load is not None else 0.0))
                    except Exception:
                        load_val = 0.0
                    best_k = None
                    best_d = float("inf")
                    for k, v in items.items():
                        try:
                            kk = float(k)
                            dd = abs(load_val - kk)
                        except Exception:
                            continue
                        if best_k is None or dd < best_d:
                            best_k = kk
                            best_d = dd
                    if best_k is None:
                        return float(default)
                    try:
                        return float(items.get(best_k, default))
                    except Exception:
                        return float(default)

                srv_floor_by_load = getattr(self.rl_cfg.reward, "terminal_embb_service_floor_by_load", {}) or {}
                min_floor_by_load = getattr(self.rl_cfg.reward, "terminal_embb_min_rate_floor_by_load", {}) or {}
                srv_floor_used = _nearest_load_floor(srv_floor_by_load, srv_floor)
                min_floor_used = _nearest_load_floor(min_floor_by_load, min_floor)
                self.terminal_embb_service_floor_used = float(srv_floor_used)
                self.terminal_embb_min_rate_floor_used = float(min_floor_used)
                if srv_floor_w > 0.0 and srv_floor > 0.0:
                    gap = max(srv_floor_used - service_ratio, 0.0)
                    if gap > 0.0:
                        penalty = float(srv_floor_w * (gap * gap))
                        terminal_reward_terms["terminal_embb_service_floor_penalty"] = float(penalty)
                        team_reward -= float(penalty)
                if min_floor_w > 0.0 and min_floor > 0.0:
                    gap = max(min_floor_used - min_rate_ratio, 0.0)
                    if gap > 0.0:
                        penalty = float(min_floor_w * (gap * gap))
                        terminal_reward_terms["terminal_embb_min_rate_floor_penalty"] = float(penalty)
                        team_reward -= float(penalty)
                if srv_bonus_w2 > 0.0:
                    bonus = float(srv_bonus_w2 * service_ratio)
                    terminal_reward_terms["terminal_embb_service_bonus"] = float(bonus)
                    team_reward += float(bonus)
                if min_bonus_w2 > 0.0:
                    bonus = float(min_bonus_w2 * min_rate_ratio)
                    terminal_reward_terms["terminal_embb_min_rate_bonus"] = float(bonus)
                    team_reward += float(bonus)

                # Relative service gains vs a per-episode greedy reference (terminal).
                srv_gain_w = float(getattr(self.rl_cfg.reward, "terminal_embb_service_gain_vs_greedy_weight", 0.0) or 0.0)
                min_gain_w = float(getattr(self.rl_cfg.reward, "terminal_embb_minrate_gain_vs_greedy_weight", 0.0) or 0.0)
                srv_short_w = float(getattr(self.rl_cfg.reward, "terminal_embb_service_vs_greedy_shortfall_penalty_weight", 0.0) or 0.0)
                if srv_gain_w > 0.0 or min_gain_w > 0.0 or srv_short_w > 0.0:
                    ref = getattr(self, "original_greedy_metrics", {}) or {}
                    try:
                        greedy_served = float(ref.get("embb_served_users", ref.get("embb_served_user_count", 0.0)) or 0.0)
                    except Exception:
                        greedy_served = 0.0
                    greedy_service_ratio = float(greedy_served / max(int(self.sys_cfg.num_embb_users), 1))
                    greedy_min_rate_ratio = float(greedy_service_ratio)
                    try:
                        rates = np.asarray(ref.get("embb_user_rate", []), dtype=float)
                        embb_min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", getattr(self.embb_cfg, "min_rate", 0.0)) or 0.0)
                        if rates.size > 0 and embb_min_rate > 0.0:
                            greedy_min_rate_ratio = float(np.mean(rates >= (embb_min_rate - 1.0e-9)))
                    except Exception:
                        pass
                    service_gain = max(service_ratio - greedy_service_ratio, 0.0)
                    min_gain = max(min_rate_ratio - greedy_min_rate_ratio, 0.0)
                    service_shortfall = max(greedy_service_ratio - service_ratio, 0.0)
                    if srv_gain_w > 0.0 and service_gain > 0.0:
                        bonus = float(srv_gain_w * service_gain)
                        terminal_reward_terms["terminal_embb_service_gain_vs_greedy_bonus"] = float(bonus)
                        team_reward += float(bonus)
                    if min_gain_w > 0.0 and min_gain > 0.0:
                        bonus = float(min_gain_w * min_gain)
                        terminal_reward_terms["terminal_embb_minrate_gain_vs_greedy_bonus"] = float(bonus)
                        team_reward += float(bonus)
                    if srv_short_w > 0.0 and service_shortfall > 0.0:
                        penalty = float(srv_short_w * (service_shortfall * service_shortfall))
                        terminal_reward_terms["terminal_embb_service_vs_greedy_shortfall_penalty"] = float(penalty)
                        team_reward -= float(penalty)

                # Extra tradeoff penalty: high URLLC admission while violating a minimum eMBB service floor.
                tradeoff_w = float(getattr(self.rl_cfg.reward, "urllc_admission_over_service_tradeoff_penalty_weight", 0.0) or 0.0)
                tradeoff_floor = float(getattr(self.rl_cfg.reward, "urllc_admission_over_service_service_floor", 0.0) or 0.0)
                if tradeoff_w > 0.0 and tradeoff_floor > 0.0:
                    deficit = max(tradeoff_floor - service_ratio, 0.0)
                    if deficit > 0.0:
                        admission = float(summary.get("urllc_admission_rate", 0.0))
                        penalty = tradeoff_w * admission * (deficit * deficit)
                        terminal_reward_terms["urllc_admission_over_service_tradeoff_penalty"] = -float(penalty)
                        team_reward += terminal_reward_terms["urllc_admission_over_service_tradeoff_penalty"]

                # Owner effectiveness shaping (terminal; uses snapshot counterfactual metrics, but does not alter
                # observation/init/fallback pipelines).
                owner_neg_srv_w = float(getattr(self.rl_cfg.reward, "owner_negative_service_gain_penalty_weight", 0.0) or 0.0)
                owner_neg_rate_w = float(getattr(self.rl_cfg.reward, "owner_negative_rate_gain_penalty_weight", 0.0) or 0.0)
                owner_pos_srv_w = float(getattr(self.rl_cfg.reward, "owner_positive_service_gain_bonus_weight", 0.0) or 0.0)
                owner_pos_rate_w = float(getattr(self.rl_cfg.reward, "owner_positive_rate_gain_bonus_weight", 0.0) or 0.0)
                # Backward-compatible knobs (older presets).
                owner_srv_gain_w = float(getattr(self.rl_cfg.reward, "owner_effective_service_gain_bonus_weight", 0.0) or 0.0)
                owner_changed_unserved_w = float(getattr(self.rl_cfg.reward, "owner_changed_but_no_service_penalty_weight", 0.0) or 0.0)
                owner_same_small_w = float(getattr(self.rl_cfg.reward, "owner_same_as_snapshot_small_penalty_weight", 0.0) or 0.0)
                if not bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_reward", True)):
                    owner_neg_srv_w = 0.0
                    owner_srv_gain_w = 0.0
                    owner_neg_rate_w = 0.0
                    owner_pos_srv_w = 0.0
                    owner_pos_rate_w = 0.0
                    owner_changed_unserved_w = 0.0
                    owner_same_small_w = 0.0

                owner_service_gain = float(summary.get("phase0_owner_effective_service_gain_ratio", 0.0))
                owner_rate_gain = float(summary.get("phase0_owner_effective_rate_gain_vs_snapshot_mean", 0.0))

                # New quality-gated shaping (preferred).
                if owner_neg_srv_w > 0.0 and owner_service_gain < 0.0:
                    terminal_reward_terms["terminal_owner_negative_service_gain_penalty"] = -owner_neg_srv_w * abs(owner_service_gain)
                    team_reward += terminal_reward_terms["terminal_owner_negative_service_gain_penalty"]
                if owner_pos_srv_w > 0.0 and owner_service_gain > 0.0:
                    terminal_reward_terms["terminal_owner_positive_service_gain_bonus"] = owner_pos_srv_w * owner_service_gain
                    team_reward += terminal_reward_terms["terminal_owner_positive_service_gain_bonus"]
                if owner_neg_rate_w > 0.0 and owner_rate_gain < 0.0:
                    terminal_reward_terms["terminal_owner_negative_rate_gain_penalty"] = -owner_neg_rate_w * abs(owner_rate_gain)
                    team_reward += terminal_reward_terms["terminal_owner_negative_rate_gain_penalty"]
                if owner_pos_rate_w > 0.0 and owner_rate_gain > 0.0:
                    terminal_reward_terms["terminal_owner_positive_rate_gain_bonus"] = owner_pos_rate_w * owner_rate_gain
                    team_reward += terminal_reward_terms["terminal_owner_positive_rate_gain_bonus"]

                # Backward-compatible bonus (older presets).
                if owner_srv_gain_w > 0.0 and owner_service_gain > 0.0:
                    terminal_reward_terms["terminal_owner_effective_service_gain_bonus"] = owner_srv_gain_w * owner_service_gain
                    team_reward += terminal_reward_terms["terminal_owner_effective_service_gain_bonus"]
                if owner_changed_unserved_w > 0.0:
                    unserved = float(summary.get("phase0_owner_changed_but_unserved_ratio", 0.0))
                    if unserved > 0.0:
                        terminal_reward_terms["terminal_owner_changed_but_no_service_penalty"] = -owner_changed_unserved_w * unserved
                        team_reward += terminal_reward_terms["terminal_owner_changed_but_no_service_penalty"]
                if owner_same_small_w > 0.0:
                    same = float(summary.get("phase0_owner_same_as_snapshot_ratio", 0.0))
                    if same > 0.0:
                        terminal_reward_terms["terminal_owner_same_as_snapshot_small_penalty"] = -owner_same_small_w * same
                        team_reward += terminal_reward_terms["terminal_owner_same_as_snapshot_small_penalty"]

                # Soft URLLC admission floor penalty (no snapshot/baseline reference to avoid leakage).
                adm_floor_w_scalar = float(getattr(self.rl_cfg.reward, "terminal_admission_floor_soft_penalty_weight", 0.0) or 0.0)
                adm_floor_scalar = float(getattr(self.rl_cfg.reward, "terminal_admission_floor_soft_penalty_floor", 0.65) or 0.65)
                adm_floor_by_load = getattr(self.rl_cfg.reward, "terminal_admission_floor_soft_penalty_floor_by_load", {}) or {}
                adm_w_by_load = getattr(self.rl_cfg.reward, "terminal_admission_floor_soft_penalty_weight_by_load", {}) or {}
                adm_floor_used = _nearest_load_floor(adm_floor_by_load, adm_floor_scalar)
                adm_w_used = _nearest_load_floor(adm_w_by_load, adm_floor_w_scalar)
                self.terminal_admission_floor_used = float(adm_floor_used)
                self.terminal_admission_floor_weight_used = float(adm_w_used)
                if adm_w_used > 0.0 and adm_floor_used > 0.0:
                    short = max(adm_floor_used - float(summary.get("urllc_admission_rate", 0.0)), 0.0)
                    if short > 0.0:
                        penalty = float(adm_w_used * (short * short))
                        terminal_reward_terms["terminal_admission_floor_soft_penalty"] = float(penalty)
                        team_reward -= float(penalty)

                # Phase-A power saturation/diversity soft penalties.
                sat_w = float(getattr(self.rl_cfg.reward, "terminal_power_saturation_penalty_weight", 0.0) or 0.0)
                cap_w = float(getattr(self.rl_cfg.reward, "terminal_phase_a_cap_hit_penalty_weight", 0.0) or 0.0)
                low_div_w = float(getattr(self.rl_cfg.reward, "terminal_phase_a_low_diversity_penalty_weight", 0.0) or 0.0)
                div_floor = float(getattr(self.rl_cfg.reward, "terminal_phase_a_diversity_floor", 0.02) or 0.02)
                sat_floor = float(getattr(self.rl_cfg.reward, "terminal_phase_a_raw_saturation_penalty_floor", 0.0) or 0.0)
                cap_floor = float(getattr(self.rl_cfg.reward, "terminal_phase_a_cap_hit_penalty_floor", 0.0) or 0.0)
                if sat_w > 0.0:
                    sat_ratio = float(summary.get("phase_a_embb_power_raw_saturation_ratio", 0.0))
                    sat_excess = max(sat_ratio - sat_floor, 0.0)
                    terminal_reward_terms["terminal_phase_a_raw_saturation_penalty"] = -sat_w * float(sat_excess)
                    team_reward += terminal_reward_terms["terminal_phase_a_raw_saturation_penalty"]
                if cap_w > 0.0:
                    cap_ratio = float(summary.get("phase_a_embb_power_cap_hit_ratio", 0.0))
                    cap_excess = max(cap_ratio - cap_floor, 0.0)
                    terminal_reward_terms["terminal_phase_a_cap_hit_penalty"] = -cap_w * float(cap_excess)
                    team_reward += terminal_reward_terms["terminal_phase_a_cap_hit_penalty"]
                if low_div_w > 0.0 and div_floor > 0.0:
                    final_std = float(summary.get("phase_a_embb_power_final_std", 0.0))
                    low = max(div_floor - final_std, 0.0)
                    if low > 0.0:
                        terminal_reward_terms["terminal_phase_a_low_diversity_penalty"] = -low_div_w * low
                        team_reward += terminal_reward_terms["terminal_phase_a_low_diversity_penalty"]

                # Phase-A power regularization (v2): discourage raw saturation and cap-hit, encourage diversity,
                # and keep the write ratio near a small target (avoid "always write everywhere").
                pa_sat_w = float(getattr(self.rl_cfg.reward, "phase_a_power_raw_saturation_penalty_weight", 0.0) or 0.0)
                pa_cap_w = float(getattr(self.rl_cfg.reward, "phase_a_power_cap_hit_penalty_weight", 0.0) or 0.0)
                pa_smooth_w = float(getattr(self.rl_cfg.reward, "phase_a_power_smooth_delta_penalty_weight", 0.0) or 0.0)
                pa_div_w = float(getattr(self.rl_cfg.reward, "phase_a_power_diversity_bonus_weight", 0.0) or 0.0)
                pa_target_write = float(getattr(self.rl_cfg.reward, "phase_a_power_target_write_ratio", 0.0) or 0.0)
                pa_write_w = float(getattr(self.rl_cfg.reward, "phase_a_power_write_ratio_penalty_weight", 0.0) or 0.0)
                if pa_sat_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_raw_saturation_penalty_v2"] = -pa_sat_w * float(
                        summary.get("phase_a_embb_power_raw_saturation_ratio", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_raw_saturation_penalty_v2"]
                if pa_cap_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_cap_hit_penalty_v2"] = -pa_cap_w * float(
                        summary.get("phase_a_embb_power_cap_hit_ratio", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_cap_hit_penalty_v2"]
                if pa_smooth_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_smooth_delta_penalty"] = -pa_smooth_w * float(
                        summary.get("phase_a_embb_power_mean_abs_raw_delta", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_smooth_delta_penalty"]
                if pa_write_w > 0.0 and pa_target_write > 0.0:
                    write_ratio = float(summary.get("phase_a_embb_power_write_ratio", 0.0))
                    write_gap = abs(write_ratio - pa_target_write)
                    terminal_reward_terms["terminal_phase_a_write_ratio_penalty"] = -pa_write_w * (write_gap * write_gap)
                    team_reward += terminal_reward_terms["terminal_phase_a_write_ratio_penalty"]
                if pa_div_w > 0.0:
                    final_std = float(summary.get("phase_a_embb_power_final_std", 0.0))
                    terminal_reward_terms["terminal_phase_a_diversity_bonus"] = pa_div_w * float(min(final_std, 0.15))
                    team_reward += terminal_reward_terms["terminal_phase_a_diversity_bonus"]
                pa_interf_bonus_w = float(getattr(self.rl_cfg.reward, "phase_a_interference_reduction_bonus_weight", 0.0) or 0.0)
                pa_change_pen_w = float(getattr(self.rl_cfg.reward, "phase_a_power_change_penalty_weight", 0.0) or 0.0)
                if pa_interf_bonus_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_interference_reduction_bonus"] = pa_interf_bonus_w * float(
                        summary.get("phase_a_power_intercell_reduction_mean", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_interference_reduction_bonus"]
                if pa_change_pen_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_power_change_penalty"] = -pa_change_pen_w * float(
                        summary.get("phase_a_embb_power_mean_abs_executed_delta", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_power_change_penalty"]
                pa_red_l2_w = float(getattr(self.rl_cfg.reward, "phase_a_power_reduction_l2_penalty_weight", 0.0) or 0.0)
                pa_sat2_w = float(getattr(self.rl_cfg.reward, "phase_a_power_saturation_penalty_weight", 0.0) or 0.0)
                pa_sat2_thr = float(getattr(self.rl_cfg.reward, "phase_a_power_saturation_threshold", 0.9) or 0.9)
                embb_floor_w = float(getattr(self.rl_cfg.reward, "embb_service_floor_hinge_penalty_weight", 0.0) or 0.0)
                embb_floor_t = float(getattr(self.rl_cfg.reward, "embb_service_floor_target", 0.55) or 0.55)
                if pa_red_l2_w > 0.0:
                    terminal_reward_terms["phaseA_power_reduction_l2_penalty"] = -pa_red_l2_w * float(
                        summary.get("phaseA_negative_delta_l2_mean", 0.0)
                    )
                    team_reward += terminal_reward_terms["phaseA_power_reduction_l2_penalty"]
                if pa_sat2_w > 0.0:
                    sat_ratio2 = float(summary.get("phaseA_negative_delta_saturation_ratio", 0.0))
                    terminal_reward_terms["phaseA_power_saturation_penalty"] = -pa_sat2_w * sat_ratio2
                    team_reward += terminal_reward_terms["phaseA_power_saturation_penalty"]
                if embb_floor_w > 0.0:
                    service_ratio_now = float(
                        summary.get("embb_service_ratio_after_puncture_deduction", summary.get("embb_service_ratio", 0.0))
                    )
                    service_gap2 = max(embb_floor_t - service_ratio_now, 0.0)
                    terminal_reward_terms["embb_service_floor_hinge_penalty"] = -embb_floor_w * (service_gap2 * service_gap2)
                    team_reward += terminal_reward_terms["embb_service_floor_hinge_penalty"]

                # Extra Phase-A penalties requested by interference repair presets.
                pa_l2_w = float(getattr(self.rl_cfg.reward, "phase_a_power_delta_l2_penalty_weight", 0.0) or 0.0)
                pa_flat_w = float(getattr(self.rl_cfg.reward, "phase_a_power_cellwise_flattening_penalty_weight", 0.0) or 0.0)
                div_floor2 = float(getattr(self.rl_cfg.reward, "terminal_phase_a_diversity_floor", 0.0) or 0.0)
                if pa_l2_w > 0.0:
                    terminal_reward_terms["terminal_phase_a_delta_l2_penalty"] = -pa_l2_w * float(
                        summary.get("phase_a_embb_power_mean_raw_delta_l2", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_delta_l2_penalty"]
                if pa_flat_w > 0.0 and div_floor2 > 0.0:
                    final_std = float(summary.get("phase_a_embb_power_final_std", 0.0))
                    short = max(div_floor2 - final_std, 0.0)
                    terminal_reward_terms["terminal_phase_a_cellwise_flattening_penalty"] = -pa_flat_w * short
                    team_reward += terminal_reward_terms["terminal_phase_a_cellwise_flattening_penalty"]

                # Inter-cell aware penalties (terminal).
                inter_norm = float(getattr(self.rl_cfg.reward, "terminal_intercell_penalty_normalizer", 1.0e-7) or 1.0e-7)
                inter_w = float(getattr(self.rl_cfg.reward, "terminal_intercell_penalty_weight", 0.0) or 0.0)
                inter_w = max(inter_w, float(getattr(self.rl_cfg.reward, "terminal_mean_intercell_mw_penalty_weight", 0.0) or 0.0))
                inter_pu_w = float(getattr(self.rl_cfg.reward, "terminal_puncture_intercell_penalty_weight", 0.0) or 0.0)
                inter_ov_w = float(getattr(self.rl_cfg.reward, "terminal_overlay_intercell_penalty_weight", 0.0) or 0.0)
                inter_loss_ratio_w = float(getattr(self.rl_cfg.reward, "terminal_intercell_rate_loss_ratio_penalty_weight", 0.0) or 0.0)
                inter_loss_ratio_w = max(inter_loss_ratio_w, float(getattr(self.rl_cfg.reward, "terminal_intercell_loss_ratio_penalty_weight", 0.0) or 0.0))
                if inter_loss_ratio_w > 0.0:
                    terminal_reward_terms["terminal_intercell_rate_loss_ratio_penalty"] = -inter_loss_ratio_w * float(
                        summary.get("embb_rate_loss_due_to_intercell_ratio", 0.0)
                    )
                    team_reward += terminal_reward_terms["terminal_intercell_rate_loss_ratio_penalty"]
                if inter_w > 0.0:
                    terminal_reward_terms["terminal_intercell_penalty"] = -inter_w * float(
                        summary.get("mean_intercell_interference_mw", 0.0)
                    ) / max(inter_norm, 1.0e-12)
                    team_reward += terminal_reward_terms["terminal_intercell_penalty"]
                if inter_pu_w > 0.0:
                    terminal_reward_terms["terminal_puncture_intercell_penalty"] = -inter_pu_w * float(
                        summary.get("puncture_intercell_interference_mw", 0.0)
                    ) / max(inter_norm, 1.0e-12)
                    team_reward += terminal_reward_terms["terminal_puncture_intercell_penalty"]
                if inter_ov_w > 0.0:
                    terminal_reward_terms["terminal_overlay_intercell_penalty"] = -inter_ov_w * float(
                        summary.get("overlay_intercell_interference_mw", 0.0)
                    ) / max(inter_norm, 1.0e-12)
                    team_reward += terminal_reward_terms["terminal_overlay_intercell_penalty"]

                # Extra intercell/power-over-greedy penalties (terminal; requires per-episode greedy reference metrics).
                intercell_power_w = float(getattr(self.rl_cfg.reward, "terminal_intercell_power_penalty_weight", 0.0) or 0.0)
                total_over_w = float(getattr(self.rl_cfg.reward, "terminal_total_power_over_greedy_penalty_weight", 0.0) or 0.0)
                embb_over_w = float(getattr(self.rl_cfg.reward, "terminal_embb_power_over_greedy_penalty_weight", 0.0) or 0.0)
                allowed_default = float(getattr(self.rl_cfg.reward, "terminal_power_over_greedy_allowed_ratio", 1.10) or 1.10)
                allowed_map = getattr(self.rl_cfg.env, "max_total_power_ratio_to_greedy_by_load", {}) or {}
                allowed_ratio = _nearest_load_floor(allowed_map, allowed_default)
                baseline_power = max(float(getattr(self, "original_greedy_metrics", {}).get("total_power", 0.0)), 1.0e-9)
                if intercell_power_w > 0.0:
                    penalty = float(intercell_power_w * float(summary.get("mean_intercell_interference_mw", 0.0)) / max(inter_norm, 1.0e-12))
                    terminal_reward_terms["terminal_intercell_power_penalty"] = float(penalty)
                    team_reward -= float(penalty)
                if (total_over_w > 0.0 or embb_over_w > 0.0) and baseline_power > 0.0:
                    total_ratio = float(summary.get("total_power", 0.0)) / baseline_power
                    over = max(total_ratio - float(allowed_ratio), 0.0)
                    if total_over_w > 0.0 and over > 0.0:
                        penalty = float(total_over_w * (over * over))
                        terminal_reward_terms["terminal_total_power_over_greedy_penalty"] = float(penalty)
                        team_reward -= float(penalty)
                    if embb_over_w > 0.0:
                        embb_ratio = float(summary.get("embb_power", 0.0)) / baseline_power
                        embb_over = max(embb_ratio - float(allowed_ratio), 0.0)
                        if embb_over > 0.0:
                            penalty = float(embb_over_w * (embb_over * embb_over))
                            terminal_reward_terms["terminal_embb_power_over_greedy_penalty"] = float(penalty)
                            team_reward -= float(penalty)

                # Served-user-count bonus (terminal; load-aware target).
                served_w = float(getattr(self.rl_cfg.reward, "terminal_embb_served_user_count_weight", 0.0) or 0.0)
                served_map = getattr(self.rl_cfg.reward, "terminal_embb_served_user_count_normalizer_by_load", {}) or {}
                served_target = _nearest_load_floor(served_map, 0.0)
                self.terminal_embb_served_user_target = float(served_target)
                if served_w > 0.0 and served_target > 0.0:
                    served_count = float(summary.get("embb_served_user_count", 0.0))
                    score = min(served_count / max(served_target, 1.0e-9), 1.0)
                    bonus = float(served_w * float(score))
                    terminal_reward_terms["terminal_embb_served_user_count_bonus"] = float(bonus)
                    team_reward += float(bonus)
                served_def_w = float(getattr(self.rl_cfg.reward, "terminal_embb_served_user_deficit_penalty_weight", 0.0) or 0.0)
                if served_def_w > 0.0 and served_target > 0.0:
                    served_count = float(summary.get("embb_served_user_count", 0.0))
                    deficit = max(served_target - served_count, 0.0) / max(served_target, 1.0e-9)
                    if deficit > 0.0:
                        penalty = float(served_def_w * (deficit * deficit))
                        terminal_reward_terms["terminal_embb_served_user_deficit_penalty"] = float(penalty)
                        team_reward -= float(penalty)
                if owner_restore_w > 0.0:
                    restored = float(summary.get("phase0_owner_restored_to_snapshot_ratio", 0.0))
                    if restored > 0.0:
                        terminal_reward_terms["terminal_owner_restored_to_snapshot_penalty"] = -owner_restore_w * restored
                        team_reward += terminal_reward_terms["terminal_owner_restored_to_snapshot_penalty"]
                if owner_eff_w > 0.0:
                    eff_change = float(summary.get("phase0_owner_changed_and_effective_ratio", 0.0))
                    # Scale by service ratio so we don't reward arbitrary owner churn if it doesn't improve served coverage.
                    terminal_reward_terms["terminal_owner_effective_change_bonus"] = owner_eff_w * eff_change * float(
                        summary.get("embb_service_ratio_after_puncture_deduction", summary.get("embb_service_ratio", 0.0))
                    )
                    team_reward += terminal_reward_terms["terminal_owner_effective_change_bonus"]
                if owner_util_w > 0.0:
                    owner_change = float(summary.get("phase0_owner_change_ratio_vs_snapshot_executed", 0.0))
                    if owner_change > 0.05:
                        terminal_reward_terms["terminal_owner_change_utilization_bonus"] = (
                            owner_util_w
                            * float(owner_change - 0.05)
                            * float(summary.get("urllc_admission_rate", 0.0))
                            * float(summary.get("embb_service_ratio_after_puncture_deduction", summary.get("embb_service_ratio", 0.0)))
                        )
                        team_reward += terminal_reward_terms["terminal_owner_change_utilization_bonus"]
                owner_pos_obj_w = float(getattr(self.rl_cfg.reward, "owner_positive_objective_gain_bonus_weight", 0.0) or 0.0)
                owner_neg_obj_w = float(getattr(self.rl_cfg.reward, "owner_negative_objective_gain_penalty_weight", 0.0) or 0.0)
                owner_harm_w = float(getattr(self.rl_cfg.reward, "owner_harmful_change_penalty_weight", 0.0) or 0.0)
                if owner_pos_obj_w > 0.0:
                    gain = max(float(summary.get("phase0_owner_objective_gain_accepted_mean", 0.0)), 0.0)
                    terminal_reward_terms["terminal_owner_positive_objective_gain_bonus"] = float(owner_pos_obj_w * gain)
                    team_reward += terminal_reward_terms["terminal_owner_positive_objective_gain_bonus"]
                if owner_neg_obj_w > 0.0:
                    neg = max(-float(summary.get("phase0_owner_objective_gain_accepted_mean", 0.0)), 0.0)
                    terminal_reward_terms["terminal_owner_negative_objective_gain_penalty"] = float(-owner_neg_obj_w * neg)
                    team_reward += terminal_reward_terms["terminal_owner_negative_objective_gain_penalty"]
                if owner_harm_w > 0.0:
                    harm = float(summary.get("phase0_owner_harmful_accepted_ratio", 0.0))
                    terminal_reward_terms["terminal_owner_harmful_change_penalty"] = float(-owner_harm_w * harm)
                    team_reward += terminal_reward_terms["terminal_owner_harmful_change_penalty"]
                owner_raw_churn_drop_w = float(getattr(self.rl_cfg.reward, "owner_dropped_raw_churn_penalty_weight", 0.0) or 0.0)
                if owner_raw_churn_drop_w > 0.0:
                    dropped_raw_churn = float(summary.get("owner_dropped_raw_churn_ratio", 0.0))
                    terminal_reward_terms["terminal_owner_raw_churn_penalty"] = float(-owner_raw_churn_drop_w * dropped_raw_churn)
                    team_reward += terminal_reward_terms["terminal_owner_raw_churn_penalty"]

                # Owner forced-change shaping (executed owner map vs snapshot).
                owner_change_bonus_w = float(getattr(self.rl_cfg.reward, "owner_change_bonus_weight", 0.0) or 0.0)
                owner_change_target = float(getattr(self.rl_cfg.reward, "owner_change_target_ratio", 0.20) or 0.20)
                owner_underuse_w = float(getattr(self.rl_cfg.reward, "owner_change_underuse_penalty_weight", 0.0) or 0.0)
                owner_same_w = float(getattr(self.rl_cfg.reward, "owner_same_as_snapshot_penalty_weight", 0.0) or 0.0)
                if owner_change_bonus_w > 0.0 or owner_underuse_w > 0.0 or owner_same_w > 0.0:
                    if bool(getattr(self.rl_cfg.env, "disable_snapshot_imitation", True)):
                        owner_change_bonus_w = 0.0
                        owner_underuse_w = 0.0
                        owner_same_w = 0.0
                    if not bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_reward", True)):
                        owner_change_bonus_w = 0.0
                        owner_underuse_w = 0.0
                        owner_same_w = 0.0
                    executed_change = float(summary.get("phase0_owner_change_ratio_vs_snapshot_executed", 0.0))
                    # Encourage a moderate executed change ratio (not zero, not full random).
                    band = 0.15
                    if owner_change_bonus_w > 0.0:
                        closeness = max(1.0 - abs(executed_change - owner_change_target) / max(band, 1e-9), 0.0)
                        terminal_reward_terms["terminal_owner_change_bonus"] = owner_change_bonus_w * closeness
                        team_reward += terminal_reward_terms["terminal_owner_change_bonus"]
                    if owner_underuse_w > 0.0:
                        under = max(owner_change_target - executed_change, 0.0)
                        terminal_reward_terms["terminal_owner_change_underuse_penalty"] = -owner_underuse_w * under
                        team_reward += terminal_reward_terms["terminal_owner_change_underuse_penalty"]
                    if owner_same_w > 0.0:
                        if float(summary.get("phase0_owner_candidate_positive_objective_ratio", 0.0)) <= 1.0e-12:
                            owner_same_w = 0.0
                        same = float(executed_change <= 1.0e-9)
                        terminal_reward_terms["terminal_owner_same_as_snapshot_penalty"] = -owner_same_w * same
                        team_reward += terminal_reward_terms["terminal_owner_same_as_snapshot_penalty"]

                phasea_eff_floor_w = float(getattr(self.rl_cfg.reward, "terminal_phase_a_effective_nonzero_floor_penalty_weight", 0.0) or 0.0)
                phasea_eff_floor = float(getattr(self.rl_cfg.reward, "terminal_phase_a_effective_nonzero_floor", 0.15) or 0.15)
                if phasea_eff_floor_w > 0.0 and phasea_eff_floor > 0.0:
                    phasea_eff_nz = float(summary.get("phase_a_embb_power_effective_nonzero_ratio", 0.0))
                    phasea_eff_deficit = max(phasea_eff_floor - phasea_eff_nz, 0.0)
                    terminal_reward_terms["terminal_phase_a_effective_nonzero_floor_penalty"] = float(
                        -phasea_eff_floor_w * phasea_eff_deficit
                    )
                    team_reward += terminal_reward_terms["terminal_phase_a_effective_nonzero_floor_penalty"]
                phasea_abs_floor_w = float(getattr(self.rl_cfg.reward, "terminal_phase_a_abs_delta_floor_penalty_weight", 0.0) or 0.0)
                phasea_abs_floor = float(getattr(self.rl_cfg.reward, "terminal_phase_a_abs_delta_floor", 0.04) or 0.04)
                if phasea_abs_floor_w > 0.0 and phasea_abs_floor > 0.0:
                    phasea_abs_exec = float(summary.get("phaseA_executed_abs_delta_mean", summary.get("phase_a_embb_power_mean_abs_executed_delta", 0.0)))
                    phasea_abs_deficit = max(phasea_abs_floor - phasea_abs_exec, 0.0)
                    terminal_reward_terms["terminal_phase_a_abs_delta_floor_penalty"] = float(-phasea_abs_floor_w * phasea_abs_deficit)
                    team_reward += terminal_reward_terms["terminal_phase_a_abs_delta_floor_penalty"]

            for key, value in terminal_reward_terms.items():
                shared_reward_terms[key] = shared_reward_terms.get(key, 0.0) + float(value)
            for key, value in shared_reward_terms.items():
                self.episode_reward_term_totals[key] = self.episode_reward_term_totals.get(key, 0.0) + float(value)

            rewards = {agent_id: team_reward for agent_id in self.agent_ids}
            dones = {agent_id: self.episode_done for agent_id in self.agent_ids}
            infos = {
                agent_id: {
                    **step_info[agent_id],
                    "reward_terms": dict(shared_reward_terms),
                    "team_reward": team_reward,
                    "overlay_count_total": int(np.sum(self.mode_grid == MODE_OVERLAY)),
                    "puncture_count_total": int(np.sum(self.mode_grid == MODE_PUNCTURE)),
                    "scheduled_packets_total": int(np.count_nonzero(self.scheduled_uavs >= 0)),
                    "remaining_packets": int(len(self.unscheduled_packet_ids)),
                }
                for agent_id in self.agent_ids
            }

            if self.episode_done:
                next_obs = {agent_id: observations[agent_id] for agent_id in self.agent_ids}
            else:
                next_obs_start = perf_counter()
                next_obs = self._build_observations()
                next_obs_elapsed = perf_counter() - next_obs_start
                self.build_observations_next_step_calls += 1
                self.build_observations_next_step_total_sec += next_obs_elapsed
            return next_obs, rewards, dones, infos
        finally:
            self.step_total_sec += perf_counter() - step_start

    def _raw_action_to_shielded_action(
        self,
        action: HybridAction,
        observation: AgentObservation,
    ) -> ShieldedAction:
        mode = int(action.mode)
        packet_option = int(action.packet_option)
        candidate = None
        if mode != MODE_KEEP and 0 < packet_option <= len(observation.candidates):
            candidate = observation.candidates[packet_option - 1]
        return ShieldedAction(
            action=HybridAction(
                mode=mode,
                packet_option=packet_option,
                power_delta=float(action.power_delta),
                embb_owner_option=int(action.embb_owner_option),
                embb_power_delta=float(action.embb_power_delta),
            ),
            candidate=candidate,
            utility=candidate.utility_for_mode(mode) if candidate is not None else 0.0,
        )

    def _resolve_executed_actions(
        self,
        joint_actions: Dict[str, HybridAction],
        observations: Dict[str, AgentObservation],
        minislot: Optional[int] = None,
        rb: Optional[int] = None,
    ) -> Dict[str, ShieldedAction]:
        shielded: Dict[str, ShieldedAction] = {}
        for agent_id in self.agent_ids:
            raw_action = joint_actions.get(agent_id, HybridAction())
            if self.rl_cfg.shield.enable_feasibility_shield:
                shielded[agent_id] = self.shield.sanitize_action(raw_action, observations[agent_id])
            else:
                shielded[agent_id] = self._raw_action_to_shielded_action(raw_action, observations[agent_id])
        if (
            not self.rl_cfg.env.multi_rb_agents
            and bool(getattr(self.rl_cfg.shield, "apply_joint_reliability_rewrite", True))
            and minislot is not None
            and rb is not None
        ):
            shielded = self._enforce_joint_reliability(minislot, rb, observations, shielded)
        if minislot is not None and rb is not None:
            shielded = self._sanitize_phase_a_embb_power_actions(shielded, minislot=minislot, rb=rb)
        return shielded

    def action_head_activity(self, observation: AgentObservation) -> Dict[str, bool]:
        """Return which MAPPO action heads are semantically active for this observation."""
        planning_phase = bool(observation.metadata.get("planning_phase", 0.0))
        owner_mask = np.asarray(observation.masks.embb_owner_mask, dtype=float)
        owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if owner_space == "global_owner_id_no_null":
            owner_active = bool(self.rl_cfg.env.learn_embb_baseline and np.any(owner_mask > 0.5))
        else:
            owner_active = bool(
                self.rl_cfg.env.learn_embb_baseline
                and owner_mask.size > 1
                and np.any(owner_mask[1:] > 0.5)
            )
        phase0_power_active = bool(
            planning_phase
            and owner_active
            and bool(getattr(self.rl_cfg.env, "learn_phase0_embb_power", True))
        )
        phase_a_power_active = bool(
            (not planning_phase)
            and self.rl_cfg.env.learn_embb_baseline
            and bool(self._phase_a_embb_power_runtime_enabled())
        )
        embb_power_active = bool(
            self.rl_cfg.env.learn_embb_baseline
            and (phase0_power_active or phase_a_power_active)
        )
        return {
            "planning_phase": planning_phase,
            "owner_active": owner_active,
            "embb_power_active": embb_power_active,
            "phase_a_embb_power_active": phase_a_power_active,
        }

    @staticmethod
    def action_diff_flags(raw_action: HybridAction, executed_action: HybridAction, tol: float = 1e-6) -> Dict[str, bool]:
        """Compare raw and executed actions component-wise."""
        return {
            "mode": int(raw_action.mode) != int(executed_action.mode),
            "packet": int(raw_action.packet_option) != int(executed_action.packet_option),
            "power": abs(float(raw_action.power_delta) - float(executed_action.power_delta)) > tol,
            "owner": int(raw_action.embb_owner_option) != int(executed_action.embb_owner_option),
            "embb_power": abs(float(raw_action.embb_power_delta) - float(executed_action.embb_power_delta)) > tol,
        }

    def _phase_a_embb_power_runtime_enabled(self) -> bool:
        return bool(
            getattr(
                self,
                "phase_a_embb_power_enabled",
                getattr(self.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False),
            )
        )

    def _phase_a_embb_power_residual_alpha(self) -> float:
        """Residual step size for Phase-A eMBB power repair (multiplicative)."""
        alpha = float(getattr(self.rl_cfg.action, "phase_a_embb_power_residual_alpha", 0.0) or 0.0)
        if alpha <= 0.0:
            alpha = float(getattr(self.rl_cfg.action, "embb_power_delta_limit", 0.05) or 0.05)
        return float(np.clip(alpha, 1.0e-6, 1.0))

    def _project_phase_a_embb_power_delta(
        self,
        embb_power_delta: float,
        *,
        base_scale: float = 1.0,
    ) -> Tuple[float, Dict[str, object]]:
        raw_delta = float(embb_power_delta)
        # Phase-A eMBB power delta pipeline (explicit stages for diagnostics):
        #   raw delta -> clip -> quantize -> projection(delta->scale, cap/floor) -> executed delta.
        clipped_delta = float(np.clip(raw_delta, -1.0, 1.0))
        delta_was_clipped = abs(clipped_delta - raw_delta) > 1e-9
        quantized_delta = clipped_delta
        used_discrete_bin = False
        raw_values = list(getattr(self.rl_cfg.env, "phase_a_embb_power_delta_values", []) or [])
        if raw_values:
            grid = np.clip(np.asarray(raw_values, dtype=float), -1.0, 1.0)
            if grid.size > 0:
                quantized_delta = float(grid[np.argmin(np.abs(grid - clipped_delta))])
                used_discrete_bin = abs(quantized_delta - clipped_delta) > 1e-9
        # Small-step residual multiplicative control to avoid cap saturation:
        #   executed_scale = base_scale * (1 + alpha * tanh(delta))
        alpha = float(self._phase_a_embb_power_residual_alpha())
        base_scale = float(base_scale)
        if not np.isfinite(base_scale) or base_scale <= 0.0:
            base_scale = 1.0
        requested_scale = float(base_scale) * (1.0 + alpha * float(np.tanh(quantized_delta)))
        projection_disabled = bool(getattr(self.rl_cfg.env, "disable_phase_a_embb_power_projection_for_debug", False))
        if projection_disabled:
            clipped_scale = float(requested_scale)
            scale_was_clipped = False
            eff_min = float(getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0) or 0.0)
            eff_max = float(getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0) or 0.0)
        else:
            bound_relax = float(getattr(self.rl_cfg.env, "phase_a_embb_power_scale_bound_relax", 1.0) or 1.0)
            bound_relax = float(max(bound_relax, 1.0))
            floor_relax = float(getattr(self.rl_cfg.env, "phase_a_embb_power_scale_floor_relax", 0.0) or 0.0)
            cap_relax = float(getattr(self.rl_cfg.env, "phase_a_embb_power_scale_cap_relax", 0.0) or 0.0)
            if floor_relax < 1.0:
                floor_relax = bound_relax
            if cap_relax < 1.0:
                cap_relax = bound_relax
            base_min = float(getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0) or 0.0)
            base_max = float(getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0) or 0.0)
            # Relax bounds around the neutral scale=1.0. When relax>1, widen the feasible window symmetrically
            # (in distance-from-1 space) so the executed deltas are less likely to be flattened by cap/floor.
            eff_min = float(1.0 - floor_relax * (1.0 - base_min))
            eff_max = float(1.0 + cap_relax * (base_max - 1.0))
            eff_min = float(max(eff_min, 1e-3))
            eff_max = float(max(eff_max, eff_min + 1e-6))
            clipped_scale = float(np.clip(
                requested_scale,
                eff_min,
                eff_max,
            ))
            scale_was_clipped = abs(clipped_scale - requested_scale) > 1e-9
        hit_upper_bound = (not projection_disabled) and (abs(clipped_scale - float(eff_max)) <= 1e-9) and (requested_scale > clipped_scale + 1e-9)
        hit_lower_bound = (not projection_disabled) and (abs(clipped_scale - float(eff_min)) <= 1e-9) and (requested_scale < clipped_scale - 1e-9)
        if abs(float(alpha)) > 1e-12 and abs(float(base_scale)) > 1e-12:
            executed_delta = float(((clipped_scale / float(base_scale)) - 1.0) / float(alpha))
        else:
            executed_delta = 0.0
        executed_delta = float(np.clip(executed_delta, -1.0, 1.0))
        return executed_delta, {
            "raw_delta": raw_delta,
            "clipped_delta": clipped_delta,
            "quantized_delta": quantized_delta,
            "executed_delta": executed_delta,
            "requested_scale": float(requested_scale),
            "executed_scale": float(clipped_scale),
            "residual_alpha": float(alpha),
            "base_scale": float(base_scale),
            "effective_scale_min": float(eff_min),
            "effective_scale_max": float(eff_max),
            "delta_was_clipped": bool(delta_was_clipped),
            "used_discrete_bin": bool(used_discrete_bin),
            "scale_was_clipped": bool(scale_was_clipped),
            "hit_upper_bound": bool(hit_upper_bound),
            "hit_lower_bound": bool(hit_lower_bound),
            "projection_disabled": bool(projection_disabled),
            "invalid_or_masked": False,
        }

    def _sanitize_phase_a_embb_power_actions(
        self,
        shielded: Dict[str, ShieldedAction],
        minislot: Optional[int],
        rb: Optional[int],
    ) -> Dict[str, ShieldedAction]:
        if minislot is None or rb is None:
            return shielded
        # IMPORTANT: when `multi_rb_agents=True`, `self.agent_ids` contains one agent per (uav, rb).
        # This sanitizer is already called per (minislot, rb), so we must only process agents that
        # correspond to the current `rb` to avoid double-counting and repeated zeroing.
        if bool(getattr(self.rl_cfg.env, "multi_rb_agents", False)):
            agent_ids = []
            for uav_idx in range(self.sys_cfg.num_uavs):
                agent_id = self._agent_id_by_uav_rb.get((uav_idx, int(rb)))
                if agent_id is not None:
                    agent_ids.append(agent_id)
        else:
            agent_ids = list(self.agent_ids)

        for agent_id in agent_ids:
            shielded_action = shielded.get(agent_id)
            if shielded_action is None:
                continue
            info: Dict[str, object] = dict(getattr(shielded_action, "phase_a_embb_power_info", {}) or {})
            info.setdefault("invalid_or_masked", False)
            info.pop("zeroed_reason", None)
            uav_idx, rb_idx = self._agent_index_map[agent_id]
            cell_rb = int(rb_idx if self.rl_cfg.env.multi_rb_agents else rb)

            # Owner validation / transmission-activity gating happens here, before we attempt any projection.
            # Phase-A eMBB power is a *repair* controller. It is only allowed to write when the selected
            # URLLC action actually admits a packet (handled at execution time). Here we do the minimal
            # local feasibility gating so KEEP-mode decisions do not pass through projection.
            if not self.rl_cfg.env.learn_embb_baseline or not self._phase_a_embb_power_runtime_enabled():
                shielded_action.action.embb_power_delta = 0.0
                info["invalid_or_masked"] = True
                info["zeroed_reason"] = "inactive_head"
                shielded_action.phase_a_embb_power_info = info
                continue
            allow_phase_a_power_on_keep = bool(getattr(self.rl_cfg.env, "allow_phase_a_power_on_keep", False))
            if int(shielded_action.action.mode) == MODE_KEEP and not allow_phase_a_power_on_keep:
                shielded_action.action.embb_power_delta = 0.0
                info["invalid_or_masked"] = True
                info["zeroed_reason"] = "keep_mode"
                shielded_action.phase_a_embb_power_info = info
                continue
            current_owner = -1
            if self.owner_per_uav_rb is not None:
                current_owner = int(self.owner_per_uav_rb[uav_idx, cell_rb])
            if current_owner < 0:
                shielded_action.action.embb_power_delta = 0.0
                info["invalid_or_masked"] = True
                info["zeroed_reason"] = "no_owner"
                shielded_action.phase_a_embb_power_info = info
                continue
            if current_owner >= int(self.sys_cfg.num_embb_users):
                shielded_action.action.embb_power_delta = 0.0
                info["invalid_or_masked"] = True
                info["zeroed_reason"] = "invalid_owner"
                shielded_action.phase_a_embb_power_info = info
                continue

            # Clip happens inside _project_phase_a_embb_power_delta(): raw delta is clipped to [-1, 1].
            # Quantization happens inside _project_phase_a_embb_power_delta(): optionally snap to a discrete delta bin.
            # Feasibility projection / shrink happens inside _project_phase_a_embb_power_delta(): convert delta->scale
            # and apply (embb_power_scale_min, embb_power_scale_max) unless projection is disabled for debug.
            base_scale = 1.0
            try:
                base_scale = float(self.embb_power_scale_grid[uav_idx, cell_rb, minislot])
            except Exception:
                base_scale = 1.0
            # Negative-only repair: Phase-A may ONLY reduce eMBB power (never increase).
            # Clamp raw positive deltas to 0 before projection.
            negative_only = bool(getattr(self.rl_cfg.env, "phase_a_negative_only_embb_power_repair", True))
            policy_raw_delta = float(shielded_action.action.embb_power_delta)
            info["policy_raw_delta"] = float(policy_raw_delta)
            if policy_raw_delta > 0.0:
                self.phase_a_power_raw_positive_count += 1
            delta_for_projection = float(policy_raw_delta)
            if negative_only:
                max_boost = float(getattr(self.rl_cfg.env, "phase_a_positive_boost_cap", 0.10) or 0.10)
                max_boost = float(np.clip(max_boost, 0.0, 1.0))
                delta_for_projection = float(np.clip(delta_for_projection, -1.0, max_boost))
                if policy_raw_delta > 0.0 and delta_for_projection <= 1.0e-12:
                    self.phase_a_power_positive_clamped_to_zero_count += 1
                # Keep positive moves rare: cap positive executed attempts to <=20% of total actions.
                if delta_for_projection > 0.0:
                    projected_positive_ratio = float(
                        (self.phase_a_power_positive_executed_count + 1) / max(self.phase_a_total_decisions + 1, 1)
                    )
                    if projected_positive_ratio > 0.20:
                        self.phase_a_power_positive_clamped_to_zero_count += 1
                        delta_for_projection = 0.0
            if policy_raw_delta < 0.0:
                self.phase_a_power_negative_candidate_count += 1

            executed_delta, projection_info = self._project_phase_a_embb_power_delta(
                delta_for_projection,
                base_scale=base_scale,
            )

            # Enforce: negative-dominant with bounded positive boost.
            try:
                alpha = float(projection_info.get("residual_alpha", self._phase_a_embb_power_residual_alpha()) or self._phase_a_embb_power_residual_alpha())
            except Exception:
                alpha = float(self._phase_a_embb_power_residual_alpha())
            try:
                executed_scale = float(projection_info.get("executed_scale", base_scale) or base_scale)
            except Exception:
                executed_scale = float(base_scale)
            if negative_only:
                max_boost = float(getattr(self.rl_cfg.env, "phase_a_positive_boost_cap", 0.10) or 0.10)
                max_boost = float(np.clip(max_boost, 0.0, 1.0))
                # Hard cap: allow only tiny positive boost around base scale.
                max_step_scale = float(base_scale) * float(1.0 + max_boost)
                executed_scale = float(min(executed_scale, max_step_scale))
                # Small-step guard: do not downscale more than a fixed fraction per decision.
                max_downscale = float(getattr(self.rl_cfg.env, "phase_a_embb_power_max_downscale_per_step", 0.05) or 0.05)
                max_downscale = float(np.clip(max_downscale, 0.0, 1.0))
                min_step_scale = float(base_scale) * float(1.0 - max_downscale)
                executed_scale = float(max(executed_scale, min_step_scale))

                # Service/min-rate hard guard (proxy): if we're already near the global floor, reject further downscale.
                floor = float(getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0) or 0.0)
                guard_margin = float(getattr(self.rl_cfg.env, "phase_a_power_guard_floor_margin", 0.02) or 0.02)
                if policy_raw_delta < 0.0 and float(base_scale) <= float(floor + guard_margin) + 1e-12:
                    self.phase_a_power_service_guard_reject_count += 1
                    self.phase_a_power_minrate_guard_reject_count += 1
                    executed_scale = float(base_scale)

            # Recompute executed_delta from the enforced executed_scale.
            if abs(float(alpha)) > 1.0e-12 and abs(float(base_scale)) > 1.0e-12:
                executed_delta = float(((executed_scale / float(base_scale)) - 1.0) / float(alpha))
            else:
                executed_delta = 0.0
            if negative_only:
                max_boost = float(getattr(self.rl_cfg.env, "phase_a_positive_boost_cap", 0.10) or 0.10)
                max_boost = float(np.clip(max_boost, 0.0, 1.0))
                executed_delta = float(np.clip(executed_delta, -1.0, max_boost))
                if executed_delta > 0.0:
                    projected_positive_ratio = float(
                        (self.phase_a_power_positive_executed_count + 1) / max(self.phase_a_total_decisions + 1, 1)
                    )
                    if projected_positive_ratio > 0.20:
                        executed_delta = 0.0
            projection_info["executed_scale"] = float(executed_scale)
            projection_info["executed_delta"] = float(executed_delta)

            if executed_delta < -1.0e-12:
                self.phase_a_power_negative_executed_count += 1
                self.phase_a_power_negative_executed_delta_sum += float(executed_delta)
            elif executed_delta > 1.0e-12:
                self.phase_a_power_positive_executed_count += 1
            else:
                self.phase_a_power_zero_action_count += 1
            self.phase_a_executed_delta_values.append(float(executed_delta))
            shielded_action.action.embb_power_delta = float(executed_delta)
            info.update(projection_info)
            # If the raw delta is non-trivial but projection collapses it to ~0, record the reason.
            raw_delta = float(info.get("raw_delta", 0.0) or 0.0)
            if abs(raw_delta) > 1.0e-3 and abs(float(executed_delta)) <= 1.0e-12:
                reason = "unknown"
                if bool(info.get("hit_upper_bound", False)):
                    reason = "cap_projection"
                elif bool(info.get("hit_lower_bound", False)):
                    reason = "floor_projection"
                else:
                    pass
                info["zeroed_reason"] = reason
            shielded_action.phase_a_embb_power_info = info
        return shielded

    def _step_embb_planning(
        self,
        joint_actions: Dict[str, HybridAction],
    ) -> Tuple[Dict[str, AgentObservation], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        rb = self._current_planning_rb()
        learn_phase0_embb_power = bool(getattr(self.rl_cfg.env, "learn_phase0_embb_power", True))
        guard_stats = {
            "owner_change_count": 0.0,
            "owner_change_ratio": 0.0,
            "rewrite_count": 0.0,
            "projected_embb_rate_ratio": 1.0,
            "projected_embb_power_ratio": 1.0,
            "rate_floor_violation": 0.0,
            "power_ceiling_violation": 0.0,
            "guard_violation": 0.0,
        }
        planning_owner_decode_info: Dict[str, Dict[str, float]] = {}
        for uav_idx in range(self.sys_cfg.num_uavs):
            if self.rl_cfg.env.multi_rb_agents:
                agent_id = self._agent_id_by_uav_rb.get((uav_idx, rb))
            else:
                agent_id = self.agent_ids[uav_idx]
            if agent_id is None:
                continue
            action = joint_actions.get(agent_id, HybridAction())
            embb_mask = self._build_embb_owner_mask(uav_idx, rb)
            raw_option = int(action.embb_owner_option)
            owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
            raw_null = False
            raw_valid = False
            raw_invalid = False
            option = raw_option
            owner = -1
            current_owner = int(self.owner_per_uav_rb[uav_idx, rb]) if self.owner_per_uav_rb is not None else -1
            snapshot_owner = int(self.phase0_snapshot_owner_per_uav_rb[uav_idx, rb]) if self.phase0_snapshot_owner_per_uav_rb is not None else -1
            if not bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_fallback", True)):
                snapshot_owner = -1
            raw_owner = -1
            freeze_owner = bool(getattr(self.rl_cfg.env, "freeze_phase0_owner_to_snapshot", False))
            if freeze_owner:
                # Debug: freeze Phase-0 owner to the baseline snapshot (disable owner learning).
                raw_valid = True
                raw_owner = int(snapshot_owner) if 0 <= int(snapshot_owner) < int(self.sys_cfg.num_embb_users) else -1
                if self.phase0_raw_owner_per_uav_rb is not None:
                    self.phase0_raw_owner_per_uav_rb[uav_idx, rb] = int(raw_owner)
                self.phase0_owner_raw_non_null_count += int(raw_owner >= 0)
                owner = int(raw_owner)
            elif self.embb_owner_candidates_by_uav_rb:
                candidates = self.embb_owner_candidates_by_uav_rb[uav_idx][rb]
                fallback_policy = str(getattr(self.rl_cfg.env, "phase0_owner_fallback_policy", "candidate0") or "candidate0").strip().lower()

                if owner_space == "global_owner_id_no_null":
                    # New semantics: embb_owner_option is a direct eMBB owner id in [0..global_embb_owner_dim-1].
                    raw_valid = 0 <= raw_option < embb_mask.size and embb_mask[raw_option] > 0.5
                    raw_invalid = not raw_valid
                    if raw_invalid:
                        self.phase0_owner_invalid_option_count += 1
                    raw_owner = int(raw_option) if (0 <= raw_option < int(self.sys_cfg.num_embb_users)) else -1
                    if self.phase0_raw_owner_per_uav_rb is not None:
                        self.phase0_raw_owner_per_uav_rb[uav_idx, rb] = int(raw_owner if raw_valid else -1)
                    self.phase0_owner_raw_non_null_count += int(raw_owner >= 0 and raw_valid)

                    if raw_valid and 0 <= raw_option < int(self.sys_cfg.num_embb_users):
                        owner = int(raw_option)
                    else:
                        # Invalid -> replace with a valid owner, preferring non-snapshot.
                        valid_owners = [int(c) for c in candidates if 0 <= int(c) < int(self.sys_cfg.num_embb_users)]
                        snapshot_ok = (snapshot_owner in valid_owners) if bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_fallback", True)) else False
                        non_snapshot = (
                            [o for o in valid_owners if o != int(snapshot_owner)]
                            if bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_fallback", True)) else list(valid_owners)
                        )
                        chosen = -1
                        if fallback_policy in {"sample_valid_non_snapshot", "resample_valid_non_snapshot"}:
                            if non_snapshot:
                                chosen = int(np.random.choice(np.asarray(non_snapshot, dtype=int)))
                            elif snapshot_ok:
                                chosen = int(snapshot_owner)
                        elif fallback_policy == "keep_snapshot" and snapshot_ok:
                            chosen = int(snapshot_owner)
                        elif fallback_policy == "keep_current" and (current_owner in valid_owners):
                            chosen = int(current_owner)
                        elif fallback_policy == "resample_valid" and valid_owners:
                            chosen = int(np.random.choice(np.asarray(valid_owners, dtype=int)))
                        elif fallback_policy == "candidate0" and valid_owners:
                            chosen = int(valid_owners[0])
                            self.phase0_owner_fallback_to_candidate0_count += 1
                        else:
                            # Safe last resort: pick snapshot if possible, otherwise any valid owner, otherwise 0.
                            if snapshot_ok:
                                chosen = int(snapshot_owner)
                            elif valid_owners:
                                chosen = int(valid_owners[0])
                            else:
                                chosen = 0 if int(self.sys_cfg.num_embb_users) > 0 else -1

                        owner = int(chosen) if (0 <= int(chosen) < int(self.sys_cfg.num_embb_users)) else -1
                        if raw_invalid:
                            if owner == int(snapshot_owner) and snapshot_ok:
                                self.phase0_owner_invalid_to_snapshot_count += 1
                                self.owner_snapshot_fallback_taken = True
                            elif owner >= 0:
                                self.phase0_owner_invalid_to_non_snapshot_count += 1
                else:
                    # Legacy semantics: embb_owner_option is {0(null), 1..M(candidates)}.
                    raw_null = raw_option == 0
                    raw_valid = 0 <= raw_option < embb_mask.size and embb_mask[raw_option] > 0.5
                    raw_invalid = (not raw_null) and (not raw_valid)
                    if raw_null:
                        self.phase0_owner_null_selected_count += 1
                    if raw_invalid:
                        self.phase0_owner_invalid_option_count += 1
                    option = raw_option if raw_valid else 0

                    if raw_option > 0 and raw_option - 1 < len(candidates) and raw_valid:
                        raw_owner = int(candidates[raw_option - 1])
                    if self.phase0_raw_owner_per_uav_rb is not None:
                        self.phase0_raw_owner_per_uav_rb[uav_idx, rb] = int(raw_owner)
                    self.phase0_owner_raw_non_null_count += int(raw_owner >= 0)
                    if option > 0 and option - 1 < len(candidates):
                        owner = int(candidates[option - 1])
                    else:
                        if (
                            fallback_policy == "candidate0"
                            and bool(getattr(self.rl_cfg.env, "force_embb_owner_per_rb", True))
                            and candidates
                        ):
                            owner = int(candidates[0])
                            self.phase0_owner_fallback_to_candidate0_count += 1
                        elif fallback_policy == "keep_snapshot" and snapshot_owner >= 0:
                            owner = int(snapshot_owner)
                            self.owner_snapshot_fallback_taken = True
                        elif fallback_policy == "keep_current" and current_owner >= 0:
                            owner = int(current_owner)
                        elif fallback_policy == "resample_valid" and candidates:
                            valid_options = np.where(embb_mask > 0)[0]
                            positive = valid_options[valid_options > 0]
                            if positive.size > 0:
                                chosen_option = int(np.random.choice(positive))
                                if chosen_option - 1 < len(candidates):
                                    owner = int(candidates[chosen_option - 1])
                            elif valid_options.size > 0:
                                chosen_option = int(np.random.choice(valid_options))
                                if chosen_option > 0 and chosen_option - 1 < len(candidates):
                                    owner = int(candidates[chosen_option - 1])
                        elif fallback_policy == "keep_null":
                            owner = -1
                        elif fallback_policy == "sample_valid_non_snapshot" and candidates:
                            snapshot_excluded = [int(c) for c in candidates if int(c) != int(snapshot_owner)]
                            if snapshot_excluded:
                                owner = int(np.random.choice(np.asarray(snapshot_excluded, dtype=int)))
                            else:
                                owner = -1
                        elif fallback_policy == "none":
                            owner = -1
                        elif bool(getattr(self.rl_cfg.env, "force_embb_owner_per_rb", True)) and candidates:
                            owner = int(candidates[0])
                            self.phase0_owner_fallback_to_candidate0_count += 1
            if owner < 0 or owner >= self.sys_cfg.num_embb_users:
                owner = -1
            self.owner_per_uav_rb[uav_idx, rb] = owner
            planning_owner_decode_info[agent_id] = {
                "raw_embb_owner_option": float(raw_option),
                "decoded_raw_owner_id": float(raw_owner),
                "snapshot_owner_id": float(snapshot_owner),
                "raw_option_valid": float(bool(raw_valid)),
                "decoded_raw_owner_equals_snapshot": float(
                    bool(raw_owner >= 0 and snapshot_owner >= 0 and int(raw_owner) == int(snapshot_owner))
                ),
            }
            self.planning_total_decisions += 1
            self.planning_owner_non_null_count += int(owner >= 0)
            self.phase0_owner_executed_non_null_count += int(owner >= 0)
            # Execution-path attribution (episode-level; aggregated into summary).
            if snapshot_owner >= 0:
                self.phase0_owner_snapshot_comparable_count += 1
                if raw_owner < 0:
                    self.phase0_owner_raw_null_count += 1
                elif int(raw_owner) == int(snapshot_owner):
                    self.phase0_owner_raw_same_as_snapshot_count += 1
                else:
                    self.phase0_owner_raw_non_snapshot_count += 1
                if owner < 0:
                    pass
                elif int(owner) == int(snapshot_owner):
                    self.phase0_owner_exec_same_as_snapshot_count += 1
                else:
                    self.phase0_owner_exec_non_snapshot_count += 1
                raw_is_change = (raw_owner >= 0 and raw_owner != snapshot_owner)
                executed_is_change = (owner >= 0 and owner != snapshot_owner)
                if raw_is_change and not executed_is_change and owner == snapshot_owner:
                    self.phase0_owner_restored_to_snapshot_count += 1
                    self.phase0_owner_reverted_to_snapshot_count += 1
                    self.owner_snapshot_fallback_taken = True
                if owner < 0:
                    self.phase0_owner_kept_null_count += 1
                    if raw_invalid:
                        self.phase0_owner_invalid_to_null_count += 1
                if (raw_owner < 0 or raw_invalid) and executed_is_change:
                    self.phase0_owner_replaced_with_non_snapshot_count += 1
                if executed_is_change:
                    self.phase0_owner_changed_and_effective_count += 1
            if self.embb_owner_grid is not None:
                self.embb_owner_grid[uav_idx, rb, :] = owner

            if learn_phase0_embb_power:
                clipped_delta = float(np.clip(action.embb_power_delta, -1.0, 1.0))
                scale = 1.0 + self.rl_cfg.action.embb_power_delta_limit * clipped_delta
                scale = float(np.clip(
                    scale,
                    self.rl_cfg.env.embb_power_scale_min,
                    self.rl_cfg.env.embb_power_scale_max,
                ))
                self.planning_embb_power_nonzero_count += int(abs(clipped_delta) > 1e-3)
                self.planning_embb_power_changed_count += int(abs(scale - 1.0) > 1e-3)
                self.embb_power_scale[uav_idx] = scale
                if self.embb_power_scale_per_uav_rb is not None:
                    self.embb_power_scale_per_uav_rb[uav_idx, rb] = scale
            else:
                self.embb_power_scale[uav_idx] = 1.0
                if self.embb_power_scale_per_uav_rb is not None:
                    self.embb_power_scale_per_uav_rb[uav_idx, rb] = 1.0

        if self.phase0_snapshot_owner_per_uav_rb is not None:
            guard_stats = self._apply_phase0_owner_guard(rb)

        self.planning_index += 1
        if self.planning_index >= len(self._embb_plan_schedule):
            self.planning_done = True
            self.current_cell_index = 0
            self._finalize_embb_baseline_from_policy()

        reward = 0.0
        try:
            embb_projection = self._project_embb_baseline_from_owner_map(
                self.owner_per_uav_rb,
                self.embb_power_scale_per_uav_rb,
            )
            total_rate = float(embb_projection["total_rate"])
            delta_rate = total_rate - float(self.planning_prev_total_rate)
            self.planning_prev_total_rate = total_rate
            reward = self.rl_cfg.reward.planning_embb_rate_weight * (
                delta_rate / max(self.rl_cfg.reward.terminal_embb_rate_normalizer, 1e-9)
            )
        except Exception:
            reward = 0.0

        next_obs_start = perf_counter()
        observations = self._build_observations()
        next_obs_elapsed = perf_counter() - next_obs_start
        self.build_observations_next_step_calls += 1
        self.build_observations_next_step_total_sec += next_obs_elapsed
        rewards = {agent_id: reward for agent_id in self.agent_ids}
        dones = {agent_id: False for agent_id in self.agent_ids}
        infos = {
            agent_id: {
                "planning_phase": True,
                "planning_rb": float(rb),
                "planning_reward": float(reward),
                "planning_owner_change_count": float(guard_stats["owner_change_count"]),
                "planning_owner_change_ratio": float(guard_stats["owner_change_ratio"]),
                "planning_owner_rewrite_count": float(guard_stats["rewrite_count"]),
                "planning_projected_embb_rate_ratio": float(guard_stats["projected_embb_rate_ratio"]),
                "planning_projected_embb_power_ratio": float(guard_stats["projected_embb_power_ratio"]),
                "planning_owner_rate_floor_violation": float(guard_stats["rate_floor_violation"]),
                "planning_owner_power_ceiling_violation": float(guard_stats["power_ceiling_violation"]),
                "planning_owner_guard_violation": float(guard_stats["guard_violation"]),
                "planning_owner_change_budget_used": float(guard_stats.get("owner_change_budget_used", 0.0)),
                "planning_owner_change_budget_allowed": float(guard_stats.get("owner_change_budget_allowed", 0.0)),
                "planning_owner_change_budget_clipped_ratio": float(guard_stats.get("owner_change_budget_clipped_ratio", 0.0)),
                "planning_owner_change_kept_topk_ratio": float(guard_stats.get("owner_change_kept_topk_ratio", 1.0)),
                "planning_owner_change_dropped_over_budget_ratio": float(guard_stats.get("owner_change_dropped_over_budget_ratio", 0.0)),
                **planning_owner_decode_info.get(agent_id, {}),
            }
            for agent_id in self.agent_ids
        }
        return observations, rewards, dones, infos

    def sample_random_actions(self, observations: Dict[str, AgentObservation]) -> Dict[str, HybridAction]:
        """Sample uniformly from the currently valid masked action set."""
        joint_actions = {}
        for agent_id, obs in observations.items():
            mode_options = np.where(obs.masks.mode_mask > 0)[0]
            mode = int(np.random.choice(mode_options))
            packet_options = np.where(obs.masks.packet_mask[mode] > 0)[0]
            packet_option = 0 if mode == MODE_KEEP else int(np.random.choice(packet_options[packet_options > 0]))
            embb_options = np.where(obs.masks.embb_owner_mask > 0)[0]
            embb_owner_option = int(np.random.choice(embb_options))
            joint_actions[agent_id] = HybridAction(
                mode=mode,
                packet_option=packet_option,
                power_delta=float(np.random.uniform(-1.0, 1.0)),
                embb_owner_option=embb_owner_option,
                embb_power_delta=float(np.random.uniform(-1.0, 1.0)),
            )
        return joint_actions

    def _prepare_slot_context(self) -> None:
        total_users = self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users
        nested_enabled = bool(getattr(self.sys_cfg, "nested_load_from_max_users_enabled", False))
        nested_max_total = int(getattr(self.sys_cfg, "nested_load_max_total_users", 0) or 0)
        nested_max_embb = int(getattr(self.sys_cfg, "nested_load_max_embb_users", 0) or 0)
        nested_max_urllc = int(getattr(self.sys_cfg, "nested_load_max_urllc_users", 0) or 0)
        if nested_max_total <= 0:
            nested_max_total = int(total_users)
        if nested_max_embb <= 0:
            nested_max_embb = int(self.sys_cfg.num_embb_users)
        if nested_max_urllc <= 0:
            nested_max_urllc = int(self.sys_cfg.num_urllc_users)
        generate_users = int(max(total_users, nested_max_total)) if nested_enabled else int(total_users)

        def _nested_user_indices() -> np.ndarray:
            if not nested_enabled:
                return np.arange(int(total_users), dtype=int)
            cur_u = int(self.sys_cfg.num_urllc_users)
            cur_e = int(self.sys_cfg.num_embb_users)
            max_u = int(max(nested_max_urllc, cur_u))
            max_e = int(max(nested_max_embb, cur_e))
            # Episode-wise nested-load subset:
            # 1) Per episode, randomly permute the mother-scene pools (URLLC/eMBB).
            # 2) For each load, take class-wise prefix from the same permutation.
            # This preserves "higher load contains lower load" within an episode,
            # while allowing the mother-scene user composition to vary across episodes.
            ur_pool = np.arange(0, max_u, dtype=int)
            em_pool = (np.arange(0, max_e, dtype=int) + max_u).astype(int, copy=False)
            ur_pool = ur_pool[ur_pool < int(generate_users)]
            em_pool = em_pool[em_pool < int(generate_users)]
            if cur_u > ur_pool.size:
                cur_u = int(ur_pool.size)
            if cur_e > em_pool.size:
                cur_e = int(em_pool.size)
            ur_perm = np.random.permutation(ur_pool) if ur_pool.size > 0 else ur_pool
            em_perm = np.random.permutation(em_pool) if em_pool.size > 0 else em_pool
            urllc_idx = ur_perm[:cur_u] if cur_u > 0 and ur_perm.size > 0 else np.asarray([], dtype=int)
            embb_idx = em_perm[:cur_e] if cur_e > 0 and em_perm.size > 0 else np.asarray([], dtype=int)
            # Keep URLLC first, then eMBB, preserving the expected index partition semantics.
            urllc_idx = np.sort(np.asarray(urllc_idx, dtype=int))
            embb_idx = np.sort(np.asarray(embb_idx, dtype=int))
            idx = np.concatenate([urllc_idx, embb_idx]).astype(int, copy=False)
            if idx.size < int(total_users):
                fallback = np.arange(int(generate_users), dtype=int)
                seen = set(int(v) for v in idx.tolist())
                extra = [int(v) for v in fallback.tolist() if int(v) not in seen]
                need = int(total_users - idx.size)
                if need > 0 and extra:
                    idx = np.concatenate([idx, np.asarray(extra[:need], dtype=int)])
            return np.asarray(idx[: int(total_users)], dtype=int)

        def _subset_topology(topology: Optional[Dict], user_indices: np.ndarray) -> Optional[Dict]:
            if not isinstance(topology, dict):
                return topology
            out: Dict[str, object] = {}
            for key, value in topology.items():
                arr = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
                if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] >= int(np.max(user_indices, initial=-1) + 1):
                    if key in {"user_positions", "horizontal_distances", "distances", "serving_hints"}:
                        out[key] = arr[user_indices].copy()
                    else:
                        out[key] = arr.copy()
                else:
                    out[key] = deepcopy(value)
            return out

        freeze_assoc = bool(getattr(self.rl_cfg.env, "freeze_association_across_episodes", False))
        freeze_channel = bool(getattr(self.rl_cfg.env, "freeze_channel_gains_across_episodes", False))
        cached_assoc = getattr(self, "_fixed_association_cache", None)
        cached_channel = getattr(self, "_fixed_channel_gains_cache", None)
        nested_indices = _nested_user_indices()
        if freeze_assoc and cached_assoc is not None:
            association = np.asarray(cached_assoc, dtype=int).copy()
        else:
            if nested_enabled:
                assoc_full = self.channel_model.get_association_from_large_scale(
                    int(generate_users),
                    int(self.sys_cfg.num_uavs),
                )
                association = np.asarray(assoc_full, dtype=int)[nested_indices].copy()
            else:
                if self.simulation.static_association is None:
                    association = self.simulation._prepare_static_association()
                else:
                    association = self.simulation.static_association.copy()
            if freeze_assoc:
                self._fixed_association_cache = np.asarray(association, dtype=int).copy()
        self.best_uav_per_user = association

        if freeze_channel and cached_channel is not None:
            self.channel_gains_mag_sq = np.asarray(cached_channel, dtype=float).copy()
            cached_topology = getattr(self, "_fixed_last_topology_cache", None)
            if cached_topology is not None:
                self.last_topology = deepcopy(cached_topology)
        else:
            if nested_enabled:
                channel_gains_full = self.channel_model.generate_channel_gains(
                    int(generate_users),
                    self.sys_cfg.num_uavs,
                    self.sys_cfg.num_subcarriers,
                    fading_type=self.sim_cfg.csi_generation_method,
                    rician_k=self.sim_cfg.rician_k_factor,
                )
                channel_gains = np.asarray(channel_gains_full, dtype=complex)[nested_indices, :, :]
                self.channel_gains_mag_sq = np.abs(channel_gains) ** 2
                self.last_topology = _subset_topology(
                    getattr(self.channel_model, "last_topology", None),
                    nested_indices,
                )
            else:
                channel_gains = self.channel_model.generate_channel_gains(
                    total_users,
                    self.sys_cfg.num_uavs,
                    self.sys_cfg.num_subcarriers,
                    fading_type=self.sim_cfg.csi_generation_method,
                    rician_k=self.sim_cfg.rician_k_factor,
                )
                self.channel_gains_mag_sq = np.abs(channel_gains) ** 2
                self.last_topology = getattr(self.channel_model, "last_topology", None)
            if freeze_channel:
                self._fixed_channel_gains_cache = np.asarray(self.channel_gains_mag_sq, dtype=float).copy()
                self._fixed_last_topology_cache = deepcopy(self.last_topology)
        self.planning_index = 0
        self.planning_done = not bool(self.rl_cfg.env.learn_embb_baseline)
        self.embb_power_scale = np.ones(self.sys_cfg.num_uavs, dtype=float)
        self.embb_power_scale_per_uav_rb = np.ones(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers),
            dtype=float,
        )
        self.planning_prev_total_rate = 0.0

        if self.rl_cfg.env.learn_embb_baseline:
            baseline_policy = str(getattr(self.rl_cfg.env, "fixed_embb_baseline_policy", "greedy")).lower()
            snapshot_result = self._build_fixed_embb_baseline(baseline_policy)
            self.phase0_snapshot_result = snapshot_result
            self.phase0_snapshot_owner_per_uav_rb = np.asarray(
                snapshot_result["owner_per_uav_rb"],
                dtype=int,
            ).copy()
            self.owner_init_from_snapshot = bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_init", True))
            if bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_init", True)):
                self.embb_selected_uavs = np.asarray(
                    snapshot_result["best_uav_per_user"],
                    dtype=int,
                ).copy()
            else:
                # Neutral init: do not take any baseline-derived owner/UAV hint.
                # Use the static association prepared by the simulator instead.
                embb_start = int(self.sys_cfg.num_urllc_users)
                self.embb_selected_uavs = np.asarray(self.best_uav_per_user[embb_start:], dtype=int).copy()
                # Also prepare a non-snapshot neutral owner map for filling missing entries during planning
                # projections. This prevents partial planning stages from collapsing projected throughput.
                neutral_result = self._build_deterministic_embb_baseline()
                self.phase0_neutral_owner_per_uav_rb = np.asarray(
                    neutral_result.get("owner_per_uav_rb", np.full_like(self.phase0_snapshot_owner_per_uav_rb, -1)),
                    dtype=int,
                ).copy()
            snapshot_projection = self._project_embb_baseline_from_owner_map(
                self.phase0_snapshot_owner_per_uav_rb,
                np.ones((self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers), dtype=float),
            )
            self.phase0_snapshot_embb_total_rate = float(snapshot_projection["total_rate"])
            self.phase0_snapshot_embb_total_power = float(snapshot_projection["total_power"])
            share_mode = str(getattr(self.rl_cfg.env, "greedy_urllc_share_mode", "none") or "none").strip().lower()
            share_ratio = float(getattr(self.rl_cfg.env, "greedy_urllc_share_ratio", 0.0) or 0.0)
            share_ratio = float(np.clip(share_ratio, 0.0, 1.0))
            if share_mode == "fixed_share" and share_ratio > 0.0:
                # Share semantics (v8 greedy ablation):
                # total eMBB loss budget per episode, not per-candidate local ratio cap.
                self.greedy_embb_loss_share_cap_ratio = float("inf")
                self.greedy_urllc_budget_bps = float(max(share_ratio * self.phase0_snapshot_embb_total_rate, 0.0))
            else:
                self.greedy_embb_loss_share_cap_ratio = float("inf")
                self.greedy_urllc_budget_bps = float("inf")
            self.greedy_urllc_budget_used_bps = 0.0
            try:
                rates = np.asarray(snapshot_projection.get("rates", np.zeros(self.sys_cfg.num_embb_users, dtype=float)), dtype=float)
                self.phase0_snapshot_embb_service_count = int(np.count_nonzero(rates > 1.0e-9))
                min_rate = float(getattr(self.embb_cfg, "min_rate", 0.0) or 0.0)
                if min_rate > 0.0:
                    self.phase0_snapshot_embb_min_rate_count = int(np.count_nonzero(rates >= (min_rate - 1.0e-9)))
                else:
                    self.phase0_snapshot_embb_min_rate_count = int(np.count_nonzero(rates > 1.0e-9))
            except Exception:
                self.phase0_snapshot_embb_service_count = 0
                self.phase0_snapshot_embb_min_rate_count = 0
            self.embb_result = None
            self.owner_per_uav_rb = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers),
                -1,
                dtype=int,
            )
            self.embb_base_rb_rates = np.zeros(self.sys_cfg.num_subcarriers, dtype=float)
            self.embb_base_rb_rates_per_uav_rb = np.zeros(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers),
                dtype=float,
            )
            self.embb_owner_grid = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
                -1,
                dtype=int,
            )
            self.embb_owner_candidates_by_uav_rb = self._build_embb_owner_candidates()
        else:
            self.phase0_snapshot_result = None
            self.phase0_snapshot_owner_per_uav_rb = None
            self.phase0_snapshot_embb_total_rate = 0.0
            self.phase0_snapshot_embb_total_power = 0.0
            self.greedy_embb_loss_share_cap_ratio = float("inf")
            self.greedy_urllc_budget_bps = float("inf")
            self.greedy_urllc_budget_used_bps = 0.0
            baseline_policy = str(getattr(self.rl_cfg.env, "fixed_embb_baseline_policy", "greedy")).lower()
            embb_result = self._build_fixed_embb_baseline(baseline_policy)
            self.embb_result = embb_result
            self.owner_per_uav_rb = embb_result["owner_per_uav_rb"].copy()
            self.embb_selected_uavs = embb_result["best_uav_per_user"].copy()
            self.embb_base_rb_rates = self.allocator.embb_base_rb_rates.copy()
            self.embb_base_rb_rates_per_uav_rb = self.allocator.embb_base_rb_rates_per_uav_rb.copy()
            self.embb_owner_grid = np.repeat(
                self.owner_per_uav_rb[:, :, None],
                self.sys_cfg.num_minislots,
                axis=2,
            ).astype(int, copy=True)

        _arrival_t0 = perf_counter()
        poisson_rate = getattr(
            self.sim_cfg,
            "urllc_poisson_rate",
            max(self.sim_cfg.urllc_arrival_prob * self.sys_cfg.num_urllc_users, 0.0),
        )
        if bool(getattr(self.rl_cfg.env, "urllc_poisson_rate_is_per_user", False)):
            poisson_rate = float(poisson_rate) * float(max(self.sys_cfg.num_urllc_users, 0))
        packet_counts_by_minislot = np.random.poisson(
            poisson_rate / max(self.sys_cfg.num_minislots, 1),
            size=self.sys_cfg.num_minislots,
        ).astype(int)
        self.packet_release_minislots = np.repeat(
            np.arange(self.sys_cfg.num_minislots, dtype=int),
            packet_counts_by_minislot,
        )
        self.num_packets = int(self.packet_release_minislots.size)
        if self.num_packets > 0:
            self.packet_sources = np.random.choice(
                self.sys_cfg.num_urllc_users,
                size=self.num_packets,
                replace=True,
            )
            shuffle_order = np.random.permutation(self.num_packets)
            self.packet_sources = self.packet_sources[shuffle_order]
            self.packet_release_minislots = self.packet_release_minislots[shuffle_order]
            max_packets = int(getattr(self.rl_cfg.env, "urllc_max_packets_per_episode", 64) or 0)
            if max_packets > 0 and self.num_packets > max_packets:
                self.packet_sources = self.packet_sources[:max_packets]
                self.packet_release_minislots = self.packet_release_minislots[:max_packets]
                self.num_packets = int(max_packets)
        else:
            self.packet_sources = np.asarray([], dtype=int)
            self.packet_release_minislots = np.asarray([], dtype=int)
        self.packet_arrivals_by_minislot = np.bincount(
            self.packet_release_minislots,
            minlength=self.sys_cfg.num_minislots,
        ).astype(int) if self.num_packets > 0 else np.zeros(self.sys_cfg.num_minislots, dtype=int)
        self.packet_associated_uavs = (
            self.best_uav_per_user[self.packet_sources]
            if self.num_packets > 0 else np.asarray([], dtype=int)
        )
        self.profile_arrival_generation_sec = float(perf_counter() - _arrival_t0)
        self.unscheduled_packet_ids = set(range(self.num_packets))
        self.packet_infeasible_streak = np.zeros(self.num_packets, dtype=int)
        self.packet_last_seen_minislot = np.full(self.num_packets, -1, dtype=int)
        self.packet_last_feasible_minislot = np.full(self.num_packets, -1, dtype=int)
        self._carryover_tracking_minislot = -1
        self._carryover_seen_in_minislot = np.zeros(self.num_packets, dtype=bool)
        self._carryover_feasible_in_minislot = np.zeros(self.num_packets, dtype=bool)

        self.scheduled_power = np.zeros((self.num_packets, self.sys_cfg.num_uavs), dtype=float)
        self.scheduled_uavs = np.full(self.num_packets, -1, dtype=int)
        self.scheduled_reliabilities = np.full(self.num_packets, np.nan, dtype=float)
        self.mode_grid = np.full(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
            MODE_KEEP,
            dtype=int,
        )
        self.executed_local_puncture_mask = np.zeros(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
            dtype=bool,
        )
        self.packet_grid = np.full(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
            -1,
            dtype=int,
        )
        self.embb_power_scale_grid = np.ones(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
            dtype=float,
        )
        self.overlay_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.puncture_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.scheduled_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.selected_overlay_retentions = []
        self.selected_puncture_losses = []
        self.selected_overlay_losses = []
        self.overlay_candidate_pairs = 0
        self.overlay_feasible_pairs = 0
        self.overlay_selected_pairs = 0
        self.phase_a_feasible_candidate_pairs = 0
        self.overlay_success_ema = np.zeros(self.sys_cfg.num_uavs, dtype=float)
        self.puncture_loss_ema = np.zeros(self.sys_cfg.num_uavs, dtype=float)
        self._last_joint_reliabilities = {}
        self._last_primary_assignment = {}

        if not nested_enabled:
            self.last_topology = getattr(self.channel_model, "last_topology", None)
        self.associated_embb_counts = np.bincount(
            self.embb_selected_uavs,
            minlength=self.sys_cfg.num_uavs,
        )
        self.associated_urllc_counts = np.bincount(
            self.best_uav_per_user[:self.sys_cfg.num_urllc_users],
            minlength=self.sys_cfg.num_uavs,
        )

    def _reset_episode_state(self) -> None:
        self.current_cell_index = 0
        self.episode_done = False
        self.planning_index = 0
        self.planning_done = False
        self.best_uav_per_user = None
        self.channel_gains_mag_sq = None
        self.embb_result = None
        self.owner_per_uav_rb = None
        self.phase0_raw_owner_per_uav_rb = None
        self.embb_owner_grid = None
        self.embb_selected_uavs = None
        self.embb_base_rb_rates = None
        self.embb_base_rb_rates_per_uav_rb = None
        self.embb_power_scale = np.ones(self.sys_cfg.num_uavs, dtype=float)
        self.embb_power_scale_per_uav_rb = np.ones(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers),
            dtype=float,
        )
        self.phase0_raw_owner_per_uav_rb = np.full(
            (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers),
            -1,
            dtype=int,
        )
        self.planning_prev_total_rate = 0.0
        self.embb_owner_candidates_by_uav_rb = None
        self.phase0_snapshot_result = None
        self.phase0_snapshot_owner_per_uav_rb = None
        self.phase0_neutral_owner_per_uav_rb = None
        self.phase0_snapshot_embb_total_rate = 0.0
        self.phase0_snapshot_embb_total_power = 0.0
        self.greedy_urllc_budget_bps = float("inf")
        self.greedy_urllc_budget_used_bps = 0.0
        self.greedy_embb_loss_share_cap_ratio = float("inf")
        self.profile_reset_total_sec = 0.0
        self.profile_prepare_slot_context_sec = 0.0
        self.profile_arrival_generation_sec = 0.0
        self.profile_hf_action_calls = 0
        self.profile_hf_prefilter_sec = 0.0
        self.profile_hf_eval_sec = 0.0
        self.profile_hf_fastpath_sec = 0.0
        # Snapshot leakage runtime flags (diagnostics only).
        self.owner_init_from_snapshot = False
        self.owner_snapshot_fallback_taken = False
        self.num_packets = 0
        self.packet_sources = np.asarray([], dtype=int)
        self.packet_release_minislots = np.asarray([], dtype=int)
        self.packet_arrivals_by_minislot = np.zeros(self.sys_cfg.num_minislots, dtype=int)
        self.packet_associated_uavs = np.asarray([], dtype=int)
        self.unscheduled_packet_ids = set()
        self.packet_infeasible_streak = np.asarray([], dtype=int)
        self.packet_last_seen_minislot = np.asarray([], dtype=int)
        self.packet_last_feasible_minislot = np.asarray([], dtype=int)
        self._carryover_tracking_minislot = -1
        self._carryover_seen_in_minislot = np.asarray([], dtype=bool)
        self._carryover_feasible_in_minislot = np.asarray([], dtype=bool)
        self.scheduled_power = np.zeros((0, self.sys_cfg.num_uavs), dtype=float)
        self.scheduled_uavs = np.asarray([], dtype=int)
        self.scheduled_reliabilities = np.asarray([], dtype=float)
        shape = (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots)
        self.mode_grid = np.full(shape, MODE_KEEP, dtype=int)
        self.executed_local_puncture_mask = np.zeros(shape, dtype=bool)
        self.packet_grid = np.full(shape, -1, dtype=int)
        self.embb_power_scale_grid = np.ones(shape, dtype=float)
        self.overlay_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.puncture_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.scheduled_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.selected_overlay_retentions = []
        self.selected_puncture_losses = []
        self.selected_overlay_losses = []
        self.overlay_candidate_pairs = 0
        self.overlay_feasible_pairs = 0
        self.overlay_selected_pairs = 0
        self.phase_a_feasible_candidate_pairs = 0
        self.overlay_success_ema = np.zeros(self.sys_cfg.num_uavs, dtype=float)
        self.puncture_loss_ema = np.zeros(self.sys_cfg.num_uavs, dtype=float)
        self.associated_embb_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.associated_urllc_counts = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        self.planning_total_decisions = 0
        self.planning_owner_non_null_count = 0
        self.phase0_owner_raw_non_null_count = 0
        self.phase0_owner_executed_non_null_count = 0
        self.phase0_owner_change_vs_snapshot_raw_count = 0
        self.phase0_owner_change_vs_snapshot_executed_count = 0
        self.phase0_owner_fallback_to_candidate0_count = 0
        self.phase0_owner_invalid_option_count = 0
        self.phase0_owner_null_selected_count = 0
        self.phase0_owner_invalid_to_null_count = 0
        self.phase0_owner_invalid_to_snapshot_count = 0
        self.phase0_owner_invalid_to_non_snapshot_count = 0
        self.phase0_owner_restored_to_snapshot_count = 0
        self.phase0_owner_kept_null_count = 0
        self.phase0_owner_replaced_with_non_snapshot_count = 0
        self.phase0_owner_snapshot_comparable_count = 0
        self.phase0_owner_raw_same_as_snapshot_count = 0
        self.phase0_owner_raw_non_snapshot_count = 0
        self.phase0_owner_raw_null_count = 0
        self.phase0_owner_exec_same_as_snapshot_count = 0
        self.phase0_owner_exec_non_snapshot_count = 0
        self.phase0_owner_reverted_to_snapshot_count = 0
        self.phase0_owner_changed_and_effective_count = 0
        self.phase0_owner_changed_but_unserved_count = 0
        self.phase0_owner_same_as_snapshot_count = 0
        self.phase0_owner_executed_change_count = 0.0
        self.phase0_owner_positive_service_gain_change_count = 0.0
        self.phase0_owner_effective_rate_gain_sum = 0.0
        self.phase0_owner_effective_rate_gain_count = 0
        self.planning_embb_power_nonzero_count = 0
        self.planning_embb_power_changed_count = 0
        self.planning_owner_rewrite_count = 0
        self.planning_owner_rate_floor_violation_count = 0
        self.planning_owner_power_ceiling_violation_count = 0
        self.planning_owner_guard_violation_count = 0
        self.phase0_owner_change_budget_checks = 0
        self.phase0_owner_change_budget_used_sum = 0.0
        self.phase0_owner_change_budget_allowed_sum = 0.0
        self.phase0_owner_change_budget_clipped_ratio_sum = 0.0
        self.phase0_owner_change_kept_topk_ratio_sum = 0.0
        self.phase0_owner_change_dropped_over_budget_ratio_sum = 0.0
        self.phase0_owner_raw_changed_count_sum = 0.0
        self.phase0_owner_allowed_k_sum = 0.0
        self.phase0_owner_executed_changed_count_sum = 0.0
        self.phase0_owner_dropped_count_sum = 0.0
        self.phase0_owner_budget_min_one_rule_applied_count = 0
        self.phase0_owner_budget_min_one_rule_eligible_count = 0
        self.phase0_owner_change_detail_records = []
        self.phase0_owner_min_one_blocked_by_no_positive_candidate_count = 0
        self.phase0_owner_candidate_count = 0
        self.phase0_owner_candidate_positive_objective_count = 0
        self.phase0_owner_candidate_relaxed_count = 0
        self.phase0_owner_candidate_fallback_used_count = 0
        self.phase0_owner_objective_gain_pre_filter_sum = 0.0
        self.phase0_owner_objective_gain_post_filter_sum = 0.0
        self.phase0_owner_objective_gain_post_filter_count = 0
        self.phase0_owner_gate_obj_mean_sum = 0.0
        self.phase0_owner_gate_obj_std_sum = 0.0
        self.phase0_owner_gate_threshold_sum = 0.0
        self.phase0_owner_candidate_after_gate_count = 0
        self.phase0_owner_negative_but_accepted_count = 0
        self.phase0_owner_neg_accept_clipped_ratio_sum = 0.0
        self.phase0_owner_neg_rejected_by_quota_ratio_sum = 0.0
        self.phase0_owner_pos_selected_count_sum = 0.0
        self.phase0_owner_neg_selected_count_sum = 0.0
        self.phase0_owner_selection_decision_count = 0
        self.phase0_owner_selection_allowed_sum = 0.0
        self.phase0_owner_selection_selected_sum = 0.0
        self.phase0_owner_positive_shortage_count = 0
        self.phase0_owner_negative_blocked_due_to_quota_count = 0
        self.phase0_owner_safe_relaxed_used_count = 0
        self.phase0_owner_safe_relaxed_candidate_count_sum = 0.0
        self.phase0_owner_safe_relaxed_selected_count_sum = 0.0
        self.phase0_owner_safe_relaxed_objective_sum = 0.0
        self.phase0_owner_safe_relaxed_service_delta_sum = 0.0
        self.phase0_owner_safe_relaxed_intercell_delta_sum = 0.0
        self.phase0_owner_near_zero_objective_count = 0
        self.phase0_owner_positive_after_relax_count = 0
        self.phase0_owner_safe_relax_disabled_count = 0
        self.phase0_owner_neg_accepted_with_positive_candidate_count = 0
        self.phase0_owner_steps_with_positive_candidate = 0
        self.phase0_owner_accepted_positive_objective_count = 0
        self.phase0_owner_rejected_nonpositive_objective_count = 0
        self.phase0_owner_objective_gain_sum = 0.0
        self.phase0_owner_objective_gain_accepted_sum = 0.0
        self.phase0_owner_effective_rate_gain_accepted_sum = 0.0
        self.phase0_owner_intercell_reduction_accepted_sum = 0.0
        self.phase0_owner_service_gain_accepted_sum = 0.0
        self.phase0_owner_minrate_gain_accepted_sum = 0.0
        self.phase0_owner_harmful_accepted_count = 0
        self.planning_projected_embb_rate_ratio_sum = 0.0
        self.planning_projected_embb_rate_ratio_min = float("inf")
        self.planning_projected_embb_power_ratio_sum = 0.0
        self.planning_projected_embb_power_ratio_max = 0.0
        self.planning_projected_metric_count = 0
        self.phase_a_total_decisions = 0
        # Phase-A per-load debug counters (for diagnosing admission collapse / feasibility bottlenecks).
        self.phase_a_candidate_total = 0
        self.phase_a_feasible_overlay_candidate_total = 0
        self.phase_a_feasible_puncture_candidate_total = 0
        self.phase_a_selected_keep_total = 0
        self.phase_a_selected_overlay_total = 0
        self.phase_a_selected_puncture_total = 0
        self.phase_a_rejected_intercell_total = 0
        self.phase_a_rejected_min_rate_total = 0
        self.phase_a_rejected_power_guard_total = 0
        self.phase_a_rejected_collision_total = 0
        self.phase_a_rejected_deadline_total = 0
        self.phase_a_rejected_other_total = 0
        self.phase_a_rejected_other_gain_ratio_total = 0
        self.phase_a_rejected_other_overlay_margin_total = 0
        self.phase_a_rejected_other_overlay_positive_gate_total = 0
        self.phase_a_rejected_other_no_overlay_owner_total = 0
        self.phase_a_rejected_other_overlay_reliability_total = 0
        self.phase_a_rejected_other_overlay_sic_total = 0
        # Greedy hard-feasible candidate diagnostics (for report-side root-cause analysis).
        self.greedy_hf_decision_count = 0
        self.greedy_hf_candidate_evaluated_total = 0
        self.greedy_hf_candidate_reject_reliability_total = 0
        self.greedy_hf_candidate_reject_power_total = 0
        self.greedy_hf_candidate_reject_min_rate_total = 0
        self.greedy_hf_candidate_reject_share_cap_total = 0
        self.greedy_hf_candidate_feasible_total = 0
        self.greedy_hf_no_candidate_total = 0
        self.greedy_hf_all_rejected_total = 0
        self.greedy_hf_budget_exhausted_keep_total = 0
        self.greedy_hf_selected_overlay_total = 0
        self.greedy_hf_selected_puncture_total = 0
        self.greedy_hf_selected_keep_total = 0
        self.greedy_hf_prefilter_pair_total = 0
        self.greedy_hf_prefilter_block_mode_mask_total = 0
        self.greedy_hf_prefilter_block_packet_mask_total = 0
        self.greedy_hf_prefilter_block_mode_infeasible_total = 0
        self.greedy_hf_no_candidate_block_mode_mask_total = 0
        self.greedy_hf_no_candidate_block_packet_mask_total = 0
        self.greedy_hf_no_candidate_block_mode_infeasible_total = 0
        self.greedy_hf_no_candidate_empty_observation_total = 0
        self.greedy_hf_no_candidate_mask_block_total = 0
        # Step-level intercell-aware reward components (must be non-zero when enabled).
        self.step_intercell_penalty_sum = 0.0
        self.step_intercell_penalty_count = 0
        self.step_intercell_penalty_active_count = 0
        self.selected_action_intercell_cost_values = []
        # Action-level intercell cost diagnostics:
        # - before_source_mask includes punctured other-cell eMBB as (incorrect) sources (legacy debug).
        # - after_source_mask excludes punctured other-cell eMBB sources (correct semantics).
        self.selected_action_intercell_cost_before_source_mask_values = []
        self.selected_action_intercell_cost_after_source_mask_values = []
        self.selected_action_intercell_cost_after_source_mask_admit_sum = 0.0
        self.selected_action_intercell_cost_after_source_mask_admit_count = 0
        # Interference-aware admission shaping (step reward; running baseline within episode).
        self.low_interference_admission_bonus_sum = 0.0
        self.high_intercell_admission_penalty_sum = 0.0
        self.high_intercell_admission_budget_ema = None
        # Action-level intercell guard runtime (Phase-A).
        self.action_intercell_guard_running_min = float("inf")
        self.action_intercell_guard_total_cell_count = 0
        self.action_intercell_guard_active_cell_count = 0
        self.action_intercell_guard_masked_option_count = 0
        # v5 guard diagnostics (local-min reference).
        self.action_intercell_guard_candidate_total_count = 0
        self.action_intercell_guard_candidate_high_count = 0
        self.action_intercell_guard_selected_violation_count = 0
        self.action_intercell_guard_local_min_cost_sum = 0.0
        self.action_intercell_guard_local_min_cost_count = 0
        self.action_intercell_guard_selected_excess_sum = 0.0
        self.action_intercell_guard_selected_excess_count = 0
        # Phase-A power admission gate diagnostics.
        self.phase_a_power_zeroed_non_admission_count = 0
        self.phase_a_power_write_on_admission_count = 0
        self.phase_a_power_write_on_keep_count = 0
        self.phase_a_keep_power_write_attempt_count = 0
        self.phase_a_keep_power_write_success_count = 0
        self.phase_a_zero_by_keep_due_to_mode_gate_count = 0
        self.phase_a_power_write_blocked_no_owner_count = 0
        self.phase_a_power_write_blocked_projection_count = 0
        self.phase_a_embb_power_write_count = 0
        self.phase_a_embb_power_changed_count = 0
        self.phase_a_embb_power_change_sum = 0.0
        self.phase_a_embb_power_projection_count = 0
        self.phase_a_embb_power_delta_clipped_count = 0
        self.phase_a_embb_power_quantized_count = 0
        self.phase_a_embb_power_scale_clipped_count = 0
        self.phase_a_embb_power_cap_hit_count = 0
        self.phase_a_embb_power_floor_hit_count = 0
        self.phase_a_embb_power_invalid_or_masked_count = 0
        self.phase_a_embb_power_zeroed_inactive_head_count = 0
        self.phase_a_embb_power_zeroed_keep_mode_count = 0
        self.phase_a_embb_power_zeroed_no_candidate_count = 0
        self.phase_a_embb_power_zeroed_no_embb_active_count = 0
        self.phase_a_embb_power_zeroed_no_owner_count = 0
        self.phase_a_embb_power_zeroed_invalid_owner_count = 0
        self.phase_a_embb_power_zeroed_cap_projection_count = 0
        self.phase_a_embb_power_zeroed_floor_projection_count = 0
        self.phase_a_embb_power_zeroed_unknown_count = 0
        self.phase_a_embb_power_raw_delta_sum = 0.0
        self.phase_a_embb_power_executed_delta_sum = 0.0
        self.phase_a_embb_power_pre_clip_delta_sum = 0.0
        self.phase_a_embb_power_post_clip_delta_sum = 0.0
        self.phase_a_embb_power_post_quant_delta_sum = 0.0
        self.phase_a_embb_power_post_projection_delta_sum = 0.0
        self.phase_a_embb_power_post_owner_validation_delta_sum = 0.0
        self.phase_a_embb_power_final_executed_delta_sum = 0.0
        self.phase_a_embb_power_sign_flip_count = 0
        self.phase_a_embb_power_pre_vs_final_abs_diff_sum = 0.0
        self.phase_a_embb_power_pre_vs_final_sq_diff_sum = 0.0
        self.phase_a_embb_power_pre_vs_final_sign_consistent_count = 0
        self.phase_a_embb_power_pre_vs_final_sign_consistent_denom = 0
        self.phase_a_embb_power_effective_nonzero_count = 0
        self.phase_a_embb_power_mean_abs_raw_delta_sum = 0.0
        self.phase_a_embb_power_mean_abs_executed_delta_sum = 0.0
        self.phase_a_embb_power_raw_delta_sq_sum = 0.0
        self.phase_a_embb_power_raw_saturation_count = 0
        self.phase_a_embb_power_final_delta_sq_sum = 0.0
        self.phase_a_embb_power_executed_scale_sum = 0.0
        self.phase_a_embb_power_executed_scale_sq_sum = 0.0
        self.phase_a_embb_power_floor_binding_strength_sum = 0.0
        self.phase_a_embb_power_cap_binding_strength_sum = 0.0
        self.phase_a_embb_power_proj_delta_abs_sum = 0.0
        self.phase_a_embb_power_proj_delta_sq_sum = 0.0
        self.phase_a_embb_power_pre_to_floor_delta_sum = 0.0
        self.phase_a_embb_power_pre_to_cap_delta_sum = 0.0
        self.phase_a_embb_power_final_minus_proj_abs_sum = 0.0
        # Phase-A negative-only power repair diagnostics (requested by coexistence debug).
        self.phase_a_power_raw_positive_count = 0
        self.phase_a_power_positive_clamped_to_zero_count = 0
        self.phase_a_power_positive_executed_count = 0
        self.phase_a_power_negative_candidate_count = 0
        self.phase_a_power_negative_executed_count = 0
        self.phase_a_power_negative_executed_delta_sum = 0.0
        self.phase_a_executed_delta_values = []
        self.phase_a_power_zero_action_count = 0
        self.phase_a_power_service_guard_reject_count = 0
        self.phase_a_power_minrate_guard_reject_count = 0
        self.phase_a_power_reliability_guard_reject_count = 0
        self.phase_a_power_total_power_reduction_sum = 0.0
        self.phase_a_power_intercell_reduction_sum = 0.0
        self.urllc_power_projection_count = 0
        self.urllc_power_delta_clipped_count = 0
        self.urllc_power_quantized_count = 0
        self.urllc_power_cap_hit_count = 0
        self.urllc_power_floor_hit_count = 0
        self.urllc_raw_power_delta_sum = 0.0
        self.urllc_executed_power_delta_sum = 0.0
        self.puncture_candidate_total = 0
        self.puncture_candidate_pruned_by_loss_ceiling_count = 0
        self.puncture_candidate_overlay_suppressed_count = 0
        self.selected_overlay_admission_count = 0
        self.selected_puncture_admission_count = 0
        self.executed_puncture_action_count = 0
        self.both_modes_feasible_count = 0
        self.safe_puncture_available_count = 0
        self.mode_balance_good_overlay_available_count = 0
        self.mode_balance_overlay_chosen_when_good_count = 0
        self.mode_balance_puncture_chosen_when_good_overlay_count = 0
        self.mode_balance_selected_overlay_cost_values = []
        self.mode_balance_selected_puncture_cost_values = []
        self.overlay_chosen_when_safe_puncture_available_count = 0
        self.puncture_chosen_when_safe_puncture_available_count = 0
        # v5 puncture-recovery diagnostics.
        self.feasible_puncture_available_count = 0
        self.puncture_chosen_when_feasible_count = 0
        self.overlay_chosen_when_lower_intercell_puncture_available_count = 0
        self.missed_feasible_puncture_count = 0
        self.teacher_mode_agreement_count = 0
        self.mode_anchor_active_count = 0
        self.selected_intercell_interference_sum = 0.0
        self.selected_intercell_interference_count = 0
        self.selected_intercell_interference_nonzero_count = 0
        self.selected_overlay_intercell_interference_sum = 0.0
        self.selected_overlay_intercell_interference_count = 0
        self.selected_puncture_intercell_interference_sum = 0.0
        self.selected_puncture_intercell_interference_count = 0
        # Reward-side floors/targets used (for diagnostics/report).
        self.terminal_embb_service_floor_used = 0.0
        self.terminal_embb_min_rate_floor_used = 0.0
        self.terminal_embb_served_user_target = 0.0
        self.terminal_admission_floor_used = 0.0
        self.terminal_admission_floor_weight_used = 0.0
        self.episode_reward_term_totals = {}
        self.last_topology = None
        self._last_joint_reliabilities = {}
        self._last_primary_assignment = {}
        self.current_reset_seed = int(getattr(self, "current_reset_seed", self.sim_cfg.random_seed))
        self.original_greedy_embb_total_rate = 0.0
        self.original_greedy_jain_fairness = 0.0
        self.original_greedy_metrics = {}
        self.phase0_snapshot_embb_service_count = 0
        self.phase0_snapshot_embb_min_rate_count = 0
        self.phase0_owner_guard_checks = 0
        self.phase0_owner_guard_rewrite_count = 0
        self.phase0_owner_guard_service_violation_count = 0
        self.phase0_owner_guard_min_rate_violation_count = 0
        self.phase0_owner_guard_accepted_positive_service_gain_count = 0
        self.phase0_owner_guard_accepted_negative_service_gain_count = 0
        self._rb_summary_cache = {}
        self.build_observations_calls = 0
        self.build_observations_total_sec = 0.0
        self.build_observations_current_step_calls = 0
        self.build_observations_current_step_total_sec = 0.0
        self.build_observations_next_step_calls = 0
        self.build_observations_next_step_total_sec = 0.0
        self.step_calls = 0
        self.step_total_sec = 0.0

    def timing_counters(self) -> Dict[str, float]:
        return {
            "build_observations_calls": float(self.build_observations_calls),
            "build_observations_total_sec": float(self.build_observations_total_sec),
            "build_observations_current_step_calls": float(self.build_observations_current_step_calls),
            "build_observations_current_step_total_sec": float(self.build_observations_current_step_total_sec),
            "build_observations_next_step_calls": float(self.build_observations_next_step_calls),
            "build_observations_next_step_total_sec": float(self.build_observations_next_step_total_sec),
            "step_calls": float(self.step_calls),
            "step_total_sec": float(self.step_total_sec),
        }

    def _prepare_original_greedy_reference(self) -> None:
        rng_state = np.random.get_state()
        try:
            sys_cfg = deepcopy(self.sys_cfg)
            urllc_cfg = deepcopy(self.urllc_cfg)
            embb_cfg = deepcopy(self.embb_cfg)
            algo_cfg = deepcopy(self.algo_cfg)
            sim_cfg = deepcopy(self.sim_cfg)
            sim_cfg.random_seed = int(self.current_reset_seed)
            simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
            reference_mode = str(getattr(self.rl_cfg.reward, "greedy_terminal_reference_mode", "original") or "original").strip().lower()
            if reference_mode in {"original_greedy_normal_v2", "normal_v2"}:
                result = simulation.run_single_allocation_normal_v2(slot_index=0)
            elif reference_mode in {"original_greedy_normal_v1", "original_greedy_lite", "normal_v1"}:
                result = simulation.run_single_allocation_normal_v1(slot_index=0)
            else:
                result = simulation.run_single_allocation(slot_index=0)
            metrics = result.get("metrics", {})
            self.original_greedy_metrics = metrics
            self.original_greedy_embb_total_rate = float(metrics.get("embb_total_rate", 0.0))
            self.original_greedy_jain_fairness = float(metrics.get("jain_fairness", 0.0))
        finally:
            np.random.set_state(rng_state)

    def _effective_owner_map(self, owner_per_uav_rb: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if owner_per_uav_rb is None:
            owner_per_uav_rb = self.owner_per_uav_rb
        if owner_per_uav_rb is None:
            return None
        effective = np.asarray(owner_per_uav_rb, dtype=int).copy()
        snapshot = self.phase0_snapshot_owner_per_uav_rb
        if (
            bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_fallback", True))
            and snapshot is not None
            and snapshot.shape == effective.shape
        ):
            missing = effective < 0
            effective[missing] = snapshot[missing]
        else:
            # No-snapshot debug: still fill missing entries with a neutral (non-snapshot) owner map so
            # planning-time projections remain well-defined while the policy gradually writes RBs.
            neutral = getattr(self, "phase0_neutral_owner_per_uav_rb", None)
            if neutral is not None and np.asarray(neutral).shape == effective.shape:
                missing = effective < 0
                effective[missing] = np.asarray(neutral, dtype=int)[missing]
        return effective

    def _project_embb_baseline_from_owner_map(
        self,
        owner_per_uav_rb: np.ndarray,
        power_scale_per_uav_rb: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray | float]:
        num_embb = self.sys_cfg.num_embb_users
        num_uavs = self.sys_cfg.num_uavs
        num_rbs = self.sys_cfg.num_subcarriers
        embb_rb_alloc = np.zeros((num_embb, num_rbs), dtype=int)
        alpha_e = np.zeros((num_embb, num_uavs, num_rbs), dtype=int)

        for uav_idx in range(num_uavs):
            for rb_idx in range(num_rbs):
                owner = int(owner_per_uav_rb[uav_idx, rb_idx])
                if 0 <= owner < num_embb:
                    embb_rb_alloc[owner, rb_idx] = 1
                    alpha_e[owner, uav_idx, rb_idx] = 1

        max_power_per_user = np.zeros(num_embb, dtype=float)
        embb_tx_powers = np.zeros(num_embb, dtype=float)
        power_scale_matrix = (
            np.asarray(power_scale_per_uav_rb, dtype=float)
            if power_scale_per_uav_rb is not None else None
        )
        best_uav_per_user = (
            np.asarray(self.embb_selected_uavs, dtype=int)
            if self.embb_selected_uavs is not None else
            np.asarray(self.best_uav_per_user[self.sys_cfg.num_urllc_users:], dtype=int)
        )
        for embb_idx in range(num_embb):
            power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
            max_power_per_user[embb_idx] = min(
                self.allocator._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                self.algo_cfg.power_upper_bound,
            )
            uav_idx = int(best_uav_per_user[embb_idx]) if best_uav_per_user.size > embb_idx else 0
            assigned_rbs = np.where(embb_rb_alloc[embb_idx] > 0)[0]
            if power_scale_matrix is not None and assigned_rbs.size > 0:
                scale = float(np.mean(power_scale_matrix[uav_idx, assigned_rbs]))
            else:
                scale = float(self.embb_power_scale[uav_idx]) if self.embb_power_scale is not None else 1.0
            scale = float(np.clip(
                scale,
                self.rl_cfg.env.embb_power_scale_min,
                self.rl_cfg.env.embb_power_scale_max,
            ))
            embb_tx_powers[embb_idx] = max_power_per_user[embb_idx] * scale

        embb_state = self.allocator._compute_embb_state(
            embb_rb_alloc,
            self.channel_gains_mag_sq,
            best_uav_per_user,
            embb_tx_powers,
        )
        return {
            "rb_allocation": embb_rb_alloc,
            "alpha_e": alpha_e,
            "power_allocation": embb_state["power_allocation"],
            "rates": embb_state["rates"],
            "total_rate": float(np.sum(embb_state["rates"])),
            "total_power": float(np.sum(embb_state["power_allocation"])),
            "owner_per_rb": embb_state["owner_per_rb"],
            "base_rb_rates": embb_state["base_rb_rates"],
            "base_rb_rates_per_uav_rb": embb_state["base_rb_rates_per_uav_rb"],
            "user_tx_powers": embb_tx_powers,
        }

    def _record_planning_projected_metrics(self, rate_ratio: float, power_ratio: float) -> None:
        self.planning_projected_metric_count += 1
        self.planning_projected_embb_rate_ratio_sum += float(rate_ratio)
        self.planning_projected_embb_rate_ratio_min = min(
            float(self.planning_projected_embb_rate_ratio_min),
            float(rate_ratio),
        )
        self.planning_projected_embb_power_ratio_sum += float(power_ratio)
        self.planning_projected_embb_power_ratio_max = max(
            float(self.planning_projected_embb_power_ratio_max),
            float(power_ratio),
        )

    def _phase0_mean_intercell_power_for_owner_map(
        self,
        owner_per_uav_rb: Optional[np.ndarray],
        power_scale_per_uav_rb: Optional[np.ndarray] = None,
    ) -> float:
        """Compute mean eMBB-only intercell interference power for a Phase-0 owner map (no URLLC)."""
        if owner_per_uav_rb is None:
            return 0.0
        owner_per_uav_rb = np.asarray(owner_per_uav_rb, dtype=int)
        if owner_per_uav_rb.ndim != 2:
            return 0.0
        num_uavs, num_rbs = owner_per_uav_rb.shape
        scale = np.asarray(power_scale_per_uav_rb, dtype=float) if power_scale_per_uav_rb is not None else None
        total = 0.0
        count = 0
        for victim_uav in range(num_uavs):
            for rb_idx in range(num_rbs):
                intercell = 0.0
                for other_uav in range(num_uavs):
                    if other_uav == victim_uav:
                        continue
                    owner = int(owner_per_uav_rb[other_uav, rb_idx])
                    if owner < 0 or owner >= int(self.sys_cfg.num_embb_users):
                        continue
                    embb_idx = int(self.sys_cfg.num_urllc_users + owner)
                    other_power = float(self.allocator._get_embb_per_rb_power(owner))
                    if scale is not None and scale.shape == owner_per_uav_rb.shape:
                        other_power *= float(scale[other_uav, rb_idx])
                    gain = float(self.channel_gains_mag_sq[embb_idx, victim_uav, rb_idx])
                    intercell += float(other_power * gain)
                total += float(intercell)
                count += 1
        return float(total / max(count, 1))

    def _apply_phase0_owner_guard(self, rb: int) -> Dict[str, float]:
        snapshot = self.phase0_snapshot_owner_per_uav_rb
        effective_owner_map = self._effective_owner_map()
        if snapshot is None or effective_owner_map is None:
            summary = {
                "owner_change_count": 0.0,
                "owner_change_ratio": 0.0,
                "rewrite_count": 0.0,
                "projected_embb_rate_ratio": 1.0,
                "projected_embb_power_ratio": 1.0,
                "rate_floor_violation": 0.0,
                "power_ceiling_violation": 0.0,
                "guard_violation": 0.0,
            }

        projection = self._project_embb_baseline_from_owner_map(
            effective_owner_map,
            self.embb_power_scale_per_uav_rb,
        )
        snapshot_rate = float(self.phase0_snapshot_embb_total_rate)
        snapshot_power = float(self.phase0_snapshot_embb_total_power)
        rate_ratio = (
            float(projection["total_rate"]) / max(snapshot_rate, 1e-9)
            if snapshot_rate > 1e-9 else 1.0
        )
        if snapshot_power > 1e-9:
            power_ratio = float(projection["total_power"]) / snapshot_power
        else:
            power_ratio = 1.0 if float(projection["total_power"]) <= 1e-9 else float("inf")

        total_owner_cells = max(self.sys_cfg.num_uavs * self.sys_cfg.num_subcarriers, 1)
        owner_change_count = int(np.count_nonzero(effective_owner_map != snapshot))
        owner_change_ratio = float(owner_change_count / total_owner_cells)

        guard_enabled = bool(getattr(self.rl_cfg.env, "phase0_owner_guard_enabled", False))
        change_limit = float(getattr(self.rl_cfg.env, "phase0_owner_max_change_ratio", 1.0))
        if bool(getattr(self.rl_cfg.env, "phase0_owner_change_warmup_enabled", False)):
            start_ratio = float(getattr(self.rl_cfg.env, "phase0_owner_change_ratio_start", 0.05) or 0.05)
            end_ratio = float(getattr(self.rl_cfg.env, "phase0_owner_change_ratio_end", 0.30) or 0.30)
            warmup_iters = max(int(getattr(self.rl_cfg.env, "phase0_owner_change_warmup_iters", 1000) or 1000), 1)
            current_iter = max(int(getattr(self, "current_training_iteration", 1) or 1), 1)
            progress = float(np.clip((current_iter - 1) / max(warmup_iters, 1), 0.0, 1.0))
            change_limit = float(start_ratio + (end_ratio - start_ratio) * progress)
        change_limit = float(np.clip(change_limit, 0.0, 1.0))
        rate_floor = float(getattr(self.rl_cfg.env, "phase0_owner_projected_rate_floor_ratio", 0.0))
        power_ceiling = float(getattr(self.rl_cfg.env, "phase0_owner_projected_power_ceiling_ratio", float("inf")))
        service_guard = bool(getattr(self.rl_cfg.env, "phase0_owner_service_preserve_guard", False))
        service_guard = service_guard or bool(getattr(self.rl_cfg.env, "phase0_owner_service_gain_guard", False))
        service_guard = service_guard or bool(
            getattr(self.rl_cfg.env, "phase0_owner_allow_change_only_if_projected_service_not_worse", False)
        )
        min_rate_guard = bool(getattr(self.rl_cfg.env, "phase0_owner_rate_preserve_guard", False))
        min_rate_guard = min_rate_guard or bool(
            getattr(self.rl_cfg.env, "phase0_owner_allow_change_only_if_projected_minrate_not_worse", False)
        )
        service_floor_ratio = float(getattr(self.rl_cfg.env, "phase0_owner_service_floor_count_ratio", 1.0) or 1.0)
        min_rate_floor_ratio = float(getattr(self.rl_cfg.env, "phase0_owner_min_rate_floor_count_ratio", 1.0) or 1.0)

        proj_rates = np.asarray(projection.get("rates", np.zeros(self.sys_cfg.num_embb_users, dtype=float)), dtype=float)
        projected_service_count = int(np.count_nonzero(proj_rates > 1.0e-9))
        embb_min_rate = float(getattr(self.embb_cfg, "min_rate", 0.0) or 0.0)
        if embb_min_rate > 0.0:
            projected_min_rate_count = int(np.count_nonzero(proj_rates >= (embb_min_rate - 1.0e-9)))
        else:
            projected_min_rate_count = projected_service_count
        snapshot_service_count = int(getattr(self, "phase0_snapshot_embb_service_count", 0) or 0)
        snapshot_min_rate_count = int(getattr(self, "phase0_snapshot_embb_min_rate_count", 0) or 0)

        # Snapshot/greedy must NEVER be used as a hard imitation target (baseline only).
        # When enabled (default), disable snapshot-based "not worse than snapshot" guard terms and avoid
        # restore-to-snapshot rewrites.
        disable_snapshot_imitation = bool(getattr(self.rl_cfg.env, "disable_snapshot_imitation", True))
        if disable_snapshot_imitation:
            rate_floor = 0.0
            service_guard = False
            min_rate_guard = False

        # Budget limiter: apply when guard is enabled OR warmup limiter is enabled.
        budget_limiter_enabled = bool(guard_enabled or bool(getattr(self.rl_cfg.env, "phase0_owner_change_warmup_enabled", False)))
        change_violation = budget_limiter_enabled and owner_change_ratio > (change_limit + 1e-12)
        rate_violation = guard_enabled and rate_ratio < (rate_floor - 1e-12)
        power_violation = guard_enabled and power_ratio > (power_ceiling + 1e-12)
        service_violation = guard_enabled and service_guard and snapshot_service_count > 0 and (
            projected_service_count < int(np.ceil(service_floor_ratio * snapshot_service_count - 1.0e-12))
        )
        min_rate_violation = guard_enabled and min_rate_guard and snapshot_min_rate_count > 0 and (
            projected_min_rate_count < int(np.ceil(min_rate_floor_ratio * snapshot_min_rate_count - 1.0e-12))
        )
        guard_violation = change_violation or rate_violation or power_violation or service_violation or min_rate_violation
        self.phase0_owner_guard_checks += 1
        self.phase0_owner_guard_service_violation_count += int(bool(service_violation))
        self.phase0_owner_guard_min_rate_violation_count += int(bool(min_rate_violation))

        rewrite_count = 0
        budget_clipped_ratio = 0.0
        kept_topk_ratio = 1.0
        dropped_over_budget_ratio = 0.0
        # Soft curriculum limiter: if owner change budget is exceeded, keep top-K changed cells and
        # revert the rest to snapshot (no hard fallback-to-snapshot semantics).
        if change_violation:
            changed_cells = list(zip(*np.where(effective_owner_map != snapshot)))
            proposed_change_count = int(len(changed_cells))
            allowed_change_count = int(np.floor(change_limit * total_owner_cells + 1.0e-9))
            allowed_change_count = int(np.clip(allowed_change_count, 0, total_owner_cells))
            if proposed_change_count > allowed_change_count:
                scored_cells: List[Tuple[float, int, int]] = []
                candidate_records: List[Dict[str, object]] = []
                w_service = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_service", 1.0) or 1.0)
                w_minrate = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_minrate", 0.75) or 0.75)
                w_rate = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_rate", 0.50) or 0.50)
                w_intercell = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_intercell", 0.25) or 0.25)
                w_power = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_power", 0.20) or 0.20)
                w_harm = float(getattr(self.rl_cfg.env, "phase0_owner_objective_w_harm", 0.50) or 0.50)
                obj_eps = float(getattr(self.rl_cfg.env, "phase0_owner_objective_eps", 1.0e-9) or 1.0e-9)
                adaptive_k = float(getattr(self.rl_cfg.env, "owner_objective_adaptive_k", 0.7) or 0.7)
                service_drop_tol = float(getattr(self.rl_cfg.env, "owner_service_drop_tol", 0.01) or 0.01)
                intercell_increase_tol = float(getattr(self.rl_cfg.env, "owner_intercell_increase_tol", 0.02) or 0.02)
                positive_candidate_count = 0
                objective_scores: List[float] = []
                for uav_idx, rb_idx in changed_cells:
                    new_owner = int(effective_owner_map[uav_idx, rb_idx])
                    old_owner = int(snapshot[uav_idx, rb_idx])
                    new_gain, _new_pwr, new_rate = self._planning_owner_summary(uav_idx, rb_idx, new_owner)
                    old_gain, _old_pwr, old_rate = self._planning_owner_summary(uav_idx, rb_idx, old_owner)
                    embb_min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0) or 0.0)
                    minrate_gain = float((new_rate >= embb_min_rate - 1.0e-9) - (old_rate >= embb_min_rate - 1.0e-9))
                    service_gain = float((new_rate > 1.0e-9) - (old_rate > 1.0e-9))
                    rate_gain_raw = float(new_rate - old_rate)
                    normalized_rate_gain = float(rate_gain_raw / max(embb_min_rate, 1.0e6))
                    intercell_before = float(old_gain)
                    intercell_after = float(new_gain)
                    intercell_reduction = float(intercell_before - intercell_after)
                    power_before = float(_old_pwr)
                    power_after = float(_new_pwr)
                    power_delta = float(power_after - power_before)
                    final_effective_rate_gain = float(rate_gain_raw)
                    score = float(
                        w_service * service_gain
                        + w_minrate * minrate_gain
                        + w_rate * normalized_rate_gain
                        + w_intercell * intercell_reduction
                        - w_power * max(power_delta, 0.0)
                        - w_harm * max(0.0, -final_effective_rate_gain / max(embb_min_rate, 1.0e6))
                    )
                    positive_objective = bool(score > obj_eps)
                    positive_candidate_count += int(positive_objective)
                    objective_scores.append(float(score))
                    candidate_records.append({
                        "cell": int(uav_idx * self.sys_cfg.num_subcarriers + rb_idx),
                        "uav": int(uav_idx),
                        "rb": int(rb_idx),
                        "old_owner": int(old_owner),
                        "new_owner": int(new_owner),
                        "service_gain": float(service_gain),
                        "minrate_gain": float(minrate_gain),
                        "rate_gain": float(rate_gain_raw),
                        "intercell_before": float(intercell_before),
                        "intercell_after": float(intercell_after),
                        "intercell_reduction": float(intercell_reduction),
                        "power_before": float(power_before),
                        "power_after": float(power_after),
                        "power_delta": float(power_delta),
                        "owner_objective_gain": float(score),
                        "accepted_by_positive_objective_gate": bool(positive_objective),
                        "accepted_by_relaxed_objective_gate": False,
                        "dropped_reason": "pending",
                        "final_effective_rate_gain": float(final_effective_rate_gain),
                    })
                    self.phase0_owner_candidate_count += 1
                    self.phase0_owner_candidate_positive_objective_count += int(positive_objective)
                    self.phase0_owner_objective_gain_sum += float(score)
                    self.phase0_owner_objective_gain_pre_filter_sum += float(score)
                if objective_scores:
                    obj_mean = float(np.mean(np.asarray(objective_scores, dtype=float)))
                    obj_std = float(np.std(np.asarray(objective_scores, dtype=float)))
                else:
                    obj_mean = 0.0
                    obj_std = 0.0
                relax_margin = float(getattr(self.rl_cfg.env, "owner_objective_relax_margin", 0.5) or 0.5)
                gate_threshold = float(obj_mean - adaptive_k * obj_std)
                gate_threshold_relaxed = float(gate_threshold - relax_margin)
                self.phase0_owner_gate_obj_mean_sum += float(obj_mean)
                self.phase0_owner_gate_obj_std_sum += float(obj_std)
                self.phase0_owner_gate_threshold_sum += float(gate_threshold_relaxed)
                for item in candidate_records:
                    score = float(item.get("owner_objective_gain", 0.0))
                    service_gain = float(item.get("service_gain", 0.0))
                    intercell_before = float(item.get("intercell_before", 0.0))
                    intercell_after = float(item.get("intercell_after", 0.0))
                    adaptive_pass = bool(
                        (score > gate_threshold_relaxed)
                        and (service_gain >= -service_drop_tol)
                        and ((intercell_after - intercell_before) <= intercell_increase_tol)
                    )
                    item["accepted_by_relaxed_objective_gate"] = bool(adaptive_pass)
                    self.phase0_owner_candidate_relaxed_count += int(adaptive_pass)
                    self.phase0_owner_near_zero_objective_count += int(abs(score) < 0.2)
                    self.phase0_owner_positive_after_relax_count += int(adaptive_pass and score >= 0.0)
                    if adaptive_pass:
                        scored_cells.append((score, int(item["uav"]), int(item["rb"])))
                self.phase0_owner_candidate_after_gate_count += int(len(scored_cells))
                scored_cells.sort(key=lambda x: x[0], reverse=True)
                keep_set = set()
                fallback_used = False
                has_positive_candidate_step = bool(positive_candidate_count > 0)
                self.phase0_owner_steps_with_positive_candidate += int(has_positive_candidate_step)
                safe_relaxed_cells_set: Set[Tuple[int, int]] = set()
                safe_relaxed_selected_meta: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
                if len(scored_cells) > 0:
                    pos_cells = [(s, u, r) for (s, u, r) in scored_cells if float(s) >= 0.0]
                    neg_cells = [(s, u, r) for (s, u, r) in scored_cells if float(s) < 0.0]
                    pos_cells.sort(key=lambda x: x[0], reverse=True)
                    neg_cells.sort(key=lambda x: x[0], reverse=True)
                    max_neg_ratio = float(getattr(self.rl_cfg.env, "owner_max_negative_accept_ratio", 0.30) or 0.30)
                    max_neg_ratio = float(np.clip(max_neg_ratio, 0.0, 1.0))
                    neg_quota = int(np.floor(float(allowed_change_count) * max_neg_ratio)) if allowed_change_count > 0 else 0
                    pos_selected = pos_cells[:allowed_change_count]
                    remaining_slots = max(allowed_change_count - len(pos_selected), 0)
                    neg_selected: List[Tuple[float, int, int]] = []
                    safe_relaxed_selected: List[Tuple[float, int, int]] = []
                    # Hard episode-level guard: once any positive owner candidate has appeared
                    # in the episode, do not allow safe-relax negative fallback.
                    has_positive_candidate_episode = bool(self.phase0_owner_candidate_positive_objective_count > 0)
                    safe_relax_allowed = bool((not has_positive_candidate_step) and (not has_positive_candidate_episode))
                    if has_positive_candidate_step:
                        # Hard rule: whenever any positive candidate exists, prohibit all negative/safe-relax selection.
                        self.phase0_owner_safe_relax_disabled_count += 1
                    # Safe relaxed fallback: only when positive pool is effectively absent.
                    if (
                        safe_relax_allowed
                        and allowed_change_count >= 1
                        and len(pos_selected) == 0
                        and len(neg_cells) > 0
                        and len(neg_selected) == 0
                    ):
                        safe_pool = []
                        for score_n, u_n, r_n in neg_cells:
                            for item in candidate_records:
                                if int(item.get("uav", -1)) == int(u_n) and int(item.get("rb", -1)) == int(r_n):
                                    service_delta_n = float(item.get("service_gain", 0.0))
                                    intercell_delta_n = float(item.get("intercell_after", 0.0) - item.get("intercell_before", 0.0))
                                    if service_delta_n >= -service_drop_tol and intercell_delta_n <= intercell_increase_tol:
                                        safe_pool.append((float(score_n), int(u_n), int(r_n), service_delta_n, intercell_delta_n))
                                    break
                        self.phase0_owner_safe_relaxed_candidate_count_sum += float(len(safe_pool))
                        if safe_pool:
                            safe_pool.sort(key=lambda x: x[0], reverse=True)
                            best = safe_pool[0]
                            safe_relaxed_selected = [(best[0], best[1], best[2])]
                            safe_relaxed_cells_set = {(int(best[1]), int(best[2]))}
                            safe_relaxed_selected_meta[(int(best[1]), int(best[2]))] = (
                                float(best[0]), float(best[3]), float(best[4])
                            )
                    keep_set = {(u, r) for _, u, r in (pos_selected + neg_selected + safe_relaxed_selected)}
                    selected_negative_count = int(len(neg_selected) + len(safe_relaxed_selected))
                    safe_relax_selected_count = int(len(safe_relaxed_selected))
                    if has_positive_candidate_step:
                        assert selected_negative_count == 0, (
                            f"safe_relax gating violated: positive_candidate_count={positive_candidate_count} "
                            f"but selected_negative_count={selected_negative_count}"
                        )
                        assert safe_relax_selected_count == 0, (
                            f"safe_relax gating violated: positive_candidate_count={positive_candidate_count} "
                            f"but safe_relax_selected_count={safe_relax_selected_count}"
                        )
                    dropped_neg = int(max(min(remaining_slots, len(neg_cells)) - len(neg_selected), 0))
                    neg_pool = int(len(neg_cells))
                    clipped_ratio = float(dropped_neg / max(neg_pool, 1))
                    self.phase0_owner_neg_accept_clipped_ratio_sum += clipped_ratio
                    self.phase0_owner_neg_rejected_by_quota_ratio_sum += clipped_ratio
                    self.phase0_owner_selection_decision_count += 1
                    self.phase0_owner_selection_allowed_sum += float(allowed_change_count)
                    self.phase0_owner_positive_shortage_count += int(len(pos_cells) < allowed_change_count)
                    self.phase0_owner_negative_blocked_due_to_quota_count += int(dropped_neg > 0)
                else:
                    if has_positive_candidate_step:
                        # Hard rule: when any positive candidate exists, never fallback to negative.
                        self.phase0_owner_safe_relax_disabled_count += 1
                        keep_set = set()
                        self.phase0_owner_selection_decision_count += 1
                        self.phase0_owner_selection_allowed_sum += float(allowed_change_count)
                        self.phase0_owner_positive_shortage_count += int(0 < allowed_change_count)
                    else:
                        fallback_candidates: List[Tuple[float, int, int]] = []
                        for item in candidate_records:
                            if float(item.get("service_gain", 0.0)) >= (-2.0 * service_drop_tol):
                                fallback_candidates.append(
                                    (float(item.get("owner_objective_gain", -np.inf)), int(item["uav"]), int(item["rb"]))
                                )
                        fallback_candidates.sort(key=lambda x: x[0], reverse=True)
                        if allowed_change_count == 1:
                            # Strict behavior for k=1: do not fill with negative fallback when no positive exists.
                            keep_set = set()
                            if len(fallback_candidates) > 0:
                                self.phase0_owner_neg_accept_clipped_ratio_sum += 1.0
                                self.phase0_owner_neg_rejected_by_quota_ratio_sum += 1.0
                                self.phase0_owner_negative_blocked_due_to_quota_count += 1
                        elif fallback_candidates:
                            fallback_used = True
                            self.phase0_owner_candidate_fallback_used_count += 1
                            top_u, top_r = fallback_candidates[0][1], fallback_candidates[0][2]
                            keep_set = {(int(top_u), int(top_r))}
                        else:
                            min_one_eligible = bool(change_limit > 0.0 and change_limit < (1.0 / max(total_owner_cells, 1)))
                            if min_one_eligible:
                                self.phase0_owner_min_one_blocked_by_no_positive_candidate_count += 1
                        self.phase0_owner_selection_decision_count += 1
                        self.phase0_owner_selection_allowed_sum += float(allowed_change_count)
                        self.phase0_owner_positive_shortage_count += int(0 < allowed_change_count)
                if has_positive_candidate_step:
                    positive_keep_set = set()
                    for item in candidate_records:
                        if (
                            bool(item.get("accepted_by_relaxed_objective_gate", False))
                            and float(item.get("owner_objective_gain", 0.0)) >= 0.0
                        ):
                            positive_keep_set.add((int(item["uav"]), int(item["rb"])))
                    keep_set = {k for k in keep_set if k in positive_keep_set}
                selected_items = [
                    item for item in candidate_records
                    if (int(item["uav"]), int(item["rb"])) in keep_set
                ]
                if has_positive_candidate_step:
                    # Final hard filter before output: when any positive candidate exists,
                    # selected list must contain non-negative objective candidates only.
                    selected_items = [
                        item for item in selected_items
                        if float(item.get("owner_objective_gain", 0.0)) >= 0.0
                    ]
                    keep_set = {(int(item["uav"]), int(item["rb"])) for item in selected_items}
                final_selected_count = int(len(selected_items))
                final_pos_selected_count = int(sum(1 for item in selected_items if float(item.get("owner_objective_gain", 0.0)) >= 0.0))
                final_neg_selected_count = int(sum(1 for item in selected_items if float(item.get("owner_objective_gain", 0.0)) < 0.0))
                final_safe_relax_selected_items = [
                    item for item in selected_items
                    if (int(item["uav"]), int(item["rb"])) in safe_relaxed_cells_set
                ]
                final_safe_relax_selected_count = int(len(final_safe_relax_selected_items))
                if has_positive_candidate_step:
                    final_neg_selected_count = 0
                    final_safe_relax_selected_count = 0
                # Final execution-layer guard: enforce selected list invariants immediately
                # before owner map write-back.
                keep_set = {(int(item["uav"]), int(item["rb"])) for item in selected_items}
                if has_positive_candidate_step:
                    assert all(float(item.get("owner_objective_gain", 0.0)) >= 0.0 for item in selected_items), (
                        "execution-layer guard violated: negative selected item exists while positive candidates are present"
                    )
                    assert final_neg_selected_count == 0, (
                        f"execution-layer guard violated: final_neg_selected_count={final_neg_selected_count}"
                    )
                    assert final_safe_relax_selected_count == 0, (
                        f"execution-layer guard violated: final_safe_relax_selected_count={final_safe_relax_selected_count}"
                    )
                # Final-only accounting for selected/accepted metrics.
                self.phase0_owner_pos_selected_count_sum += float(final_pos_selected_count)
                self.phase0_owner_neg_selected_count_sum += float(final_neg_selected_count)
                self.phase0_owner_selection_selected_sum += float(final_selected_count)
                if final_safe_relax_selected_count > 0:
                    self.phase0_owner_safe_relaxed_used_count += 1
                    self.phase0_owner_safe_relaxed_selected_count_sum += float(final_safe_relax_selected_count)
                    for item in final_safe_relax_selected_items:
                        key_sr = (int(item["uav"]), int(item["rb"]))
                        meta = safe_relaxed_selected_meta.get(key_sr, (0.0, 0.0, 0.0))
                        self.phase0_owner_safe_relaxed_objective_sum += float(meta[0])
                        self.phase0_owner_safe_relaxed_service_delta_sum += float(meta[1])
                        self.phase0_owner_safe_relaxed_intercell_delta_sum += float(meta[2])
                prev_map = np.asarray(self.owner_per_uav_rb, dtype=int).copy()
                for uav_idx, rb_idx in changed_cells:
                    if (int(uav_idx), int(rb_idx)) not in keep_set:
                        self.owner_per_uav_rb[uav_idx, rb_idx] = int(snapshot[uav_idx, rb_idx])
                        if self.embb_owner_grid is not None:
                            self.embb_owner_grid[uav_idx, rb_idx, :] = int(snapshot[uav_idx, rb_idx])
                if candidate_records:
                    for item in candidate_records:
                        k = (int(item["uav"]), int(item["rb"]))
                        accepted = k in keep_set
                        item["accepted_by_positive_objective_gate"] = bool(accepted and bool(item.get("accepted_by_positive_objective_gate", False)))
                        if accepted:
                            item["dropped_reason"] = ""
                            self.phase0_owner_accepted_positive_objective_count += int(item["accepted_by_positive_objective_gate"])
                            self.phase0_owner_negative_but_accepted_count += int(float(item["owner_objective_gain"]) < 0.0)
                            if has_positive_candidate_step and float(item["owner_objective_gain"]) < 0.0:
                                self.phase0_owner_neg_accepted_with_positive_candidate_count += 1
                            self.phase0_owner_objective_gain_post_filter_sum += float(item["owner_objective_gain"])
                            self.phase0_owner_objective_gain_post_filter_count += 1
                            self.phase0_owner_objective_gain_accepted_sum += float(item["owner_objective_gain"])
                            self.phase0_owner_effective_rate_gain_accepted_sum += float(item["final_effective_rate_gain"])
                            self.phase0_owner_intercell_reduction_accepted_sum += float(item["intercell_reduction"])
                            self.phase0_owner_service_gain_accepted_sum += float(item["service_gain"])
                            self.phase0_owner_minrate_gain_accepted_sum += float(item["minrate_gain"])
                            self.phase0_owner_harmful_accepted_count += int(float(item["final_effective_rate_gain"]) < -1.0e-9)
                        else:
                            if float(item["owner_objective_gain"]) <= obj_eps:
                                item["dropped_reason"] = "nonpositive_objective"
                                self.phase0_owner_rejected_nonpositive_objective_count += 1
                            else:
                                item["dropped_reason"] = "over_budget_topk"
                        if fallback_used and accepted:
                            item["dropped_reason"] = ""
                    self.phase0_owner_change_detail_records.extend(candidate_records[: min(64, len(candidate_records))])
                    if len(self.phase0_owner_change_detail_records) > 512:
                        self.phase0_owner_change_detail_records = self.phase0_owner_change_detail_records[:512]
                rewrite_count += int(np.count_nonzero(prev_map != np.asarray(self.owner_per_uav_rb, dtype=int)))
                effective_owner_map = self._effective_owner_map()
                owner_change_count = int(np.count_nonzero(effective_owner_map != snapshot))
                owner_change_ratio = float(owner_change_count / total_owner_cells)
                kept = int(owner_change_count)
                dropped = int(max(proposed_change_count - kept, 0))
                budget_clipped_ratio = float(dropped / max(proposed_change_count, 1))
                kept_topk_ratio = float(kept / max(proposed_change_count, 1))
                dropped_over_budget_ratio = float(dropped / max(proposed_change_count, 1))
        # Owner-change budget diagnostics (count-level; no behavior change).
        raw_owner_map_local = np.asarray(
            getattr(self, "phase0_raw_owner_per_uav_rb", np.asarray(self.owner_per_uav_rb, dtype=int)),
            dtype=int,
        )
        raw_changed_count = int(np.count_nonzero(raw_owner_map_local != snapshot))
        executed_changed_count = int(owner_change_count)
        allowed_k = int(np.floor(change_limit * total_owner_cells + 1.0e-9))
        allowed_k = int(np.clip(allowed_k, 0, total_owner_cells))
        dropped_count = int(max(raw_changed_count - executed_changed_count, 0))
        min_one_eligible = bool(change_limit > 0.0 and change_limit < (1.0 / max(total_owner_cells, 1)))
        min_one_applied = bool(min_one_eligible and allowed_k >= 1)
        if min_one_eligible:
            self.phase0_owner_budget_min_one_rule_eligible_count += 1
            self.phase0_owner_budget_min_one_rule_applied_count += int(min_one_applied)
        self.phase0_owner_raw_changed_count_sum += float(raw_changed_count)
        self.phase0_owner_allowed_k_sum += float(allowed_k)
        self.phase0_owner_executed_changed_count_sum += float(executed_changed_count)
        self.phase0_owner_dropped_count_sum += float(dropped_count)
        # Keep compact per-decision detail for report-side owner effectiveness audit.
        try:
            changed_cells_final = list(zip(*np.where(np.asarray(self.owner_per_uav_rb, dtype=int) != snapshot)))
            local_records = []
            for uav_idx, rb_idx in changed_cells_final:
                old_owner = int(snapshot[uav_idx, rb_idx])
                new_owner = int(self.owner_per_uav_rb[uav_idx, rb_idx])
                if new_owner == old_owner:
                    continue
                new_gain, _new_pwr, new_rate = self._planning_owner_summary(uav_idx, rb_idx, new_owner)
                old_gain, _old_pwr, old_rate = self._planning_owner_summary(uav_idx, rb_idx, old_owner)
                embb_min_rate_local = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0) or 0.0)
                service_before = bool(old_rate > 1.0e-9)
                service_after = bool(new_rate > 1.0e-9)
                rate_gain = float(new_rate - old_rate)
                intercell_risk = float(new_gain / max(float(np.mean(self.channel_gains_mag_sq[:, uav_idx, rb_idx])) + 1.0e-12, 1.0e-12))
                local_records.append({
                    "cell_index": int(uav_idx * self.sys_cfg.num_subcarriers + rb_idx),
                    "uav": int(uav_idx),
                    "rb": int(rb_idx),
                    "old_owner": int(old_owner),
                    "new_owner": int(new_owner),
                    "local_service_gain": float(float(service_after) - float(service_before)),
                    "projected_rate_gain": float(rate_gain),
                    "intercell_risk_penalty": float(intercell_risk),
                    "final_effective_rate_gain": float(rate_gain),
                    "served_before": bool(service_before),
                    "served_after": bool(service_after),
                    "harmful": bool(rate_gain < -1.0e-9),
                })
            local_records.sort(key=lambda item: abs(float(item.get("projected_rate_gain", 0.0))), reverse=True)
            if local_records:
                self.phase0_owner_change_detail_records.extend(local_records[: min(32, len(local_records))])
                if len(self.phase0_owner_change_detail_records) > 256:
                    self.phase0_owner_change_detail_records = self.phase0_owner_change_detail_records[:256]
        except Exception:
            pass

        if guard_violation:
            self.planning_owner_guard_violation_count += 1
            self.planning_owner_rate_floor_violation_count += int(rate_violation)
            self.planning_owner_power_ceiling_violation_count += int(power_violation)
            rewrite_on_violation = bool(getattr(self.rl_cfg.env, "phase0_owner_rewrite_to_snapshot_on_violation", True))
            # Budget-only violations are already handled by top-K clipping above; avoid re-expanding
            # owner changes via additional fallback rewrites in this case.
            if change_violation and not (rate_violation or power_violation or service_violation or min_rate_violation):
                rewrite_on_violation = False
            if bool(disable_snapshot_imitation) and not (
                change_violation and not (rate_violation or power_violation or service_violation or min_rate_violation)
            ):
                # Hard guard violations still need repair, but must not restore-to-snapshot by default.
                rewrite_on_violation = True
            if rewrite_on_violation:
                fallback = str(getattr(self.rl_cfg.env, "phase0_owner_guard_violation_fallback", "snapshot") or "snapshot").strip().lower()
                if bool(disable_snapshot_imitation) and fallback == "snapshot":
                    fallback = "best_valid"
                previous_column = np.asarray(self.owner_per_uav_rb[:, rb], dtype=int).copy()
                if (
                    fallback == "best_valid"
                    and self.embb_owner_candidates_by_uav_rb
                    and self.owner_per_uav_rb is not None
                ):
                    # Best-effort recovery: choose a valid owner map for this RB column that maximizes a simple
                    # service/min-rate/sum-rate objective under the current power scaling. This is used only when
                    # the guard detects a violation and would otherwise restore to snapshot.
                    trial_owner_map = np.asarray(self._effective_owner_map(self.owner_per_uav_rb), dtype=int).copy()
                    if bool(disable_snapshot_imitation):
                        # Start from the current effective map (no restore-to-snapshot default).
                        trial_owner_map[:, rb] = np.asarray(self.owner_per_uav_rb[:, rb], dtype=int)
                    else:
                        # Legacy: start from snapshot for stability.
                        trial_owner_map[:, rb] = np.asarray(snapshot[:, rb], dtype=int)
                    for uav_idx in range(int(self.sys_cfg.num_uavs)):
                        candidates = list(self.embb_owner_candidates_by_uav_rb[uav_idx][rb]) if self.embb_owner_candidates_by_uav_rb else []
                        best_owner = int(trial_owner_map[uav_idx, rb])
                        best_obj = float("-inf")
                        for owner in candidates:
                            cand = int(owner)
                            if cand < 0 or cand >= int(self.sys_cfg.num_embb_users):
                                continue
                            trial2 = trial_owner_map.copy()
                            trial2[uav_idx, rb] = cand
                            proj2 = self._project_embb_baseline_from_owner_map(
                                trial2,
                                self.embb_power_scale_per_uav_rb,
                            )
                            rates = np.asarray(proj2.get("rates", np.zeros(self.sys_cfg.num_embb_users, dtype=float)), dtype=float)
                            service_count = int(np.count_nonzero(rates > 1.0e-9))
                            embb_min_rate = float(getattr(self.embb_cfg, "min_rate", 0.0) or 0.0)
                            if embb_min_rate > 0.0:
                                min_count = int(np.count_nonzero(rates >= (embb_min_rate - 1.0e-9)))
                            else:
                                min_count = int(service_count)
                            total_rate = float(proj2.get("total_rate", 0.0))
                            obj = 2.0 * float(service_count) + 1.0 * float(min_count) + 0.2 * float(total_rate / 1.0e6)
                            if obj > best_obj:
                                best_obj = float(obj)
                                best_owner = int(cand)
                        trial_owner_map[uav_idx, rb] = int(best_owner)

                    self.owner_per_uav_rb[:, rb] = np.asarray(trial_owner_map[:, rb], dtype=int)
                    if self.embb_owner_grid is not None:
                        self.embb_owner_grid[:, rb, :] = np.asarray(trial_owner_map[:, rb], dtype=int)[:, None]
                else:
                    if bool(disable_snapshot_imitation):
                        # No snapshot imitation: fall back to the pre-guard column (hard feasibility only).
                        self.owner_per_uav_rb[:, rb] = previous_column
                        if self.embb_owner_grid is not None:
                            self.embb_owner_grid[:, rb, :] = np.asarray(previous_column, dtype=int)[:, None]
                    else:
                        # Legacy: restore to snapshot.
                        self.owner_per_uav_rb[:, rb] = snapshot[:, rb]
                        if self.embb_owner_grid is not None:
                            self.embb_owner_grid[:, rb, :] = snapshot[:, rb][:, None]

                rewrite_count = int(np.count_nonzero(previous_column != np.asarray(self.owner_per_uav_rb[:, rb], dtype=int)))
                if rewrite_count > 0:
                    self.planning_owner_rewrite_count += int(rewrite_count)
                    self.phase0_owner_guard_rewrite_count += int(rewrite_count)
                effective_owner_map = self._effective_owner_map()
                projection = self._project_embb_baseline_from_owner_map(
                    effective_owner_map,
                    self.embb_power_scale_per_uav_rb,
                )
                rate_ratio = (
                    float(projection["total_rate"]) / max(snapshot_rate, 1e-9)
                    if snapshot_rate > 1e-9 else 1.0
                )
                if snapshot_power > 1e-9:
                    power_ratio = float(projection["total_power"]) / snapshot_power
                else:
                    power_ratio = 1.0 if float(projection["total_power"]) <= 1e-9 else float("inf")
                owner_change_count = int(np.count_nonzero(effective_owner_map != snapshot))
                owner_change_ratio = float(owner_change_count / total_owner_cells)

        # Accepted service gain attribution (after any rewrite).
        try:
            proj_rates = np.asarray(projection.get("rates", np.zeros(self.sys_cfg.num_embb_users, dtype=float)), dtype=float)
            projected_service_count = int(np.count_nonzero(proj_rates > 1.0e-9))
            if projected_service_count >= snapshot_service_count:
                self.phase0_owner_guard_accepted_positive_service_gain_count += 1
            else:
                self.phase0_owner_guard_accepted_negative_service_gain_count += 1
        except Exception:
            pass

        self._record_planning_projected_metrics(rate_ratio, power_ratio)
        self.phase0_owner_change_budget_checks += 1
        self.phase0_owner_change_budget_used_sum += float(owner_change_ratio)
        self.phase0_owner_change_budget_allowed_sum += float(change_limit)
        self.phase0_owner_change_budget_clipped_ratio_sum += float(budget_clipped_ratio)
        self.phase0_owner_change_kept_topk_ratio_sum += float(kept_topk_ratio)
        self.phase0_owner_change_dropped_over_budget_ratio_sum += float(dropped_over_budget_ratio)
        return {
            "owner_change_count": float(owner_change_count),
            "owner_change_ratio": float(owner_change_ratio),
            "rewrite_count": float(rewrite_count),
            "projected_embb_rate_ratio": float(rate_ratio),
            "projected_embb_power_ratio": float(power_ratio),
            "rate_floor_violation": float(rate_violation),
            "power_ceiling_violation": float(power_violation),
            "guard_violation": float(guard_violation),
            "owner_change_budget_used": float(owner_change_ratio),
            "owner_change_budget_allowed": float(change_limit),
            "owner_change_budget_clipped_ratio": float(budget_clipped_ratio),
            "owner_change_kept_topk_ratio": float(kept_topk_ratio),
            "owner_change_dropped_over_budget_ratio": float(dropped_over_budget_ratio),
            "raw_changed_count": float(raw_changed_count),
            "allowed_k": float(allowed_k),
            "executed_changed_count": float(executed_changed_count),
            "dropped_count": float(dropped_count),
            "budget_min_one_rule_eligible": float(min_one_eligible),
            "budget_min_one_rule_applied": float(min_one_applied),
        }

    def _build_observations(self) -> Dict[str, AgentObservation]:
        build_start = perf_counter()
        self.build_observations_calls += 1
        try:
            if self.rl_cfg.env.learn_embb_baseline and not self.planning_done:
                return self._build_planning_observations()
            observations = {}
            self._rb_summary_cache = {}
            attach_greedy_reference = self._should_attach_greedy_reference()
            if self.rl_cfg.env.multi_rb_agents:
                minislot = self._cell_schedule[self.current_cell_index]
                progress_summary = self._phase_a_progress_summary(minislot)
                candidates_by_rb: Dict[int, List[List[CandidatePacket]]] = {}
                primary_by_rb: Dict[int, Dict[int, int]] = {}
                for rb in range(self.sys_cfg.num_subcarriers):
                    raw_candidates_by_uav: List[List[CandidatePacket]] = []
                    for uav_idx in range(self.sys_cfg.num_uavs):
                        raw_candidates_by_uav.append(self._enumerate_candidates_for_cell(uav_idx, rb, minislot))
                    self._annotate_candidate_contention(raw_candidates_by_uav)
                    candidates_by_rb[rb] = raw_candidates_by_uav
                    primary_by_rb[rb] = self._candidate_primary_assignment(raw_candidates_by_uav, minislot)
                self._last_primary_assignment = {
                    (rb, packet_id): uav_idx
                    for rb, mapping in primary_by_rb.items()
                    for packet_id, uav_idx in mapping.items()
                }
                for agent_id in self.agent_ids:
                    uav_idx, rb = self._agent_index_map[agent_id]
                    raw_candidates = candidates_by_rb[rb][uav_idx]
                    candidates = self._select_candidate_subset(raw_candidates, minislot, uav_idx)
                    greedy_candidate = None
                    greedy_mode = MODE_KEEP
                    greedy_utility = 0.0
                    greedy_packet_option = 0
                    greedy_owner_option = 0
                    if attach_greedy_reference:
                        greedy_candidate, greedy_mode, greedy_utility = self._best_local_candidate(candidates)
                        if greedy_candidate is not None:
                            greedy_packet_option = candidates.index(greedy_candidate) + 1
                            owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
                            greedy_owner = int(greedy_candidate.embb_owner_for_mode(greedy_mode))
                            greedy_owner_option = greedy_owner if owner_space == "global_owner_id_no_null" else int(greedy_owner + 1)
                    greedy_reference_action = (
                        HybridAction(
                            mode=greedy_mode,
                            packet_option=greedy_packet_option,
                            power_delta=0.0,
                            embb_owner_option=greedy_owner_option,
                        )
                        if attach_greedy_reference else None
                    )
                    local_obs = self._build_local_obs(uav_idx, rb, minislot, candidates)
                    global_obs = self._build_global_obs(rb, minislot)
                    masks = self._build_masks(
                        candidates,
                        uav_idx=int(uav_idx),
                        rb=int(rb),
                        minislot=int(minislot),
                        greedy_reference=greedy_reference_action,
                    )
                    observations[agent_id] = AgentObservation(
                        local_obs=local_obs,
                        global_obs=global_obs,
                        masks=masks,
                        candidates=candidates[: self.rl_cfg.action.max_candidate_packets],
                        greedy_reference=greedy_reference_action,
                        greedy_reference_utility=greedy_utility if attach_greedy_reference else 0.0,
                        metadata={
                            "uav_index": float(uav_idx),
                            "rb_index": float(rb),
                            "minislot_index": float(minislot),
                            "remaining_packets": float(len(self.unscheduled_packet_ids)),
                            **progress_summary,
                        },
                    )
                return observations

            minislot, rb = self._current_cell()
            progress_summary = self._phase_a_progress_summary(minislot)
            raw_candidates_by_uav: List[List[CandidatePacket]] = []
            for uav_idx in range(self.sys_cfg.num_uavs):
                raw_candidates_by_uav.append(self._enumerate_candidates_for_cell(uav_idx, rb, minislot))
            self._annotate_candidate_contention(raw_candidates_by_uav)
            base_assignment = self._candidate_primary_assignment(raw_candidates_by_uav, minislot)
            self._last_primary_assignment = {
                (rb, packet_id): uav_idx for packet_id, uav_idx in base_assignment.items()
            }
            for uav_idx, agent_id in enumerate(self.agent_ids):
                candidates = self._select_candidate_subset(raw_candidates_by_uav[uav_idx], minislot, uav_idx)
                greedy_candidate = None
                greedy_mode = MODE_KEEP
                greedy_utility = 0.0
                greedy_packet_option = 0
                greedy_owner_option = 0
                if attach_greedy_reference:
                    greedy_candidate, greedy_mode, greedy_utility = self._best_local_candidate(candidates)
                    if greedy_candidate is not None:
                        greedy_packet_option = candidates.index(greedy_candidate) + 1
                        owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
                        greedy_owner = int(greedy_candidate.embb_owner_for_mode(greedy_mode))
                        greedy_owner_option = greedy_owner if owner_space == "global_owner_id_no_null" else int(greedy_owner + 1)
                greedy_reference_action = (
                    HybridAction(
                        mode=greedy_mode,
                        packet_option=greedy_packet_option,
                        power_delta=0.0,
                        embb_owner_option=greedy_owner_option,
                    )
                    if attach_greedy_reference else None
                )
                local_obs = self._build_local_obs(uav_idx, rb, minislot, candidates)
                global_obs = self._build_global_obs(rb, minislot)
                masks = self._build_masks(
                    candidates,
                    uav_idx=int(uav_idx),
                    rb=int(rb),
                    minislot=int(minislot),
                    greedy_reference=greedy_reference_action,
                )
                observations[agent_id] = AgentObservation(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    masks=masks,
                    candidates=candidates[: self.rl_cfg.action.max_candidate_packets],
                    greedy_reference=greedy_reference_action,
                    greedy_reference_utility=greedy_utility if attach_greedy_reference else 0.0,
                    metadata={
                        "uav_index": float(uav_idx),
                        "rb_index": float(rb),
                        "minislot_index": float(minislot),
                        "remaining_packets": float(len(self.unscheduled_packet_ids)),
                        **progress_summary,
                    },
                )
            return observations
        finally:
            self.build_observations_total_sec += perf_counter() - build_start

    def _build_planning_observations(self) -> Dict[str, AgentObservation]:
        observations = {}
        self._rb_summary_cache = {}
        rb = self._current_planning_rb()
        owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if owner_space == "global_owner_id_no_null":
            embb_dim = int(getattr(self.rl_cfg.action, "global_embb_owner_dim", 0) or 0)
            embb_dim = max(embb_dim, 1)
        else:
            embb_dim = self.rl_cfg.action.max_embb_candidates + int(self.rl_cfg.action.include_null_embb_option)
        attach_greedy_reference = self._should_attach_greedy_reference()
        for agent_id in self.agent_ids:
            uav_idx, rb_idx = self._agent_index_map[agent_id]
            planning_rb = rb if not self.rl_cfg.env.multi_rb_agents else rb_idx
            embb_mask = self._build_embb_owner_mask(uav_idx, planning_rb) if planning_rb == rb else None
            mode_mask = np.zeros(self.rl_cfg.action.num_mode_actions, dtype=np.float32)
            mode_mask[MODE_KEEP] = 1.0
            packet_mask = np.zeros((self.rl_cfg.action.num_mode_actions, self.num_packet_options), dtype=np.float32)
            packet_mask[:, 0] = 1.0
            provisional_obs = AgentObservation(
                local_obs=self._build_planning_local_obs(uav_idx, planning_rb),
                global_obs=self._build_planning_global_obs(rb),
                masks=ActionMaskBundle(
                    mode_mask=mode_mask,
                    packet_mask=packet_mask,
                    embb_owner_mask=(
                        embb_mask
                        if embb_mask is not None else (
                            np.zeros(embb_dim, dtype=np.float32)
                            if owner_space == "global_owner_id_no_null" else np.pad(
                                np.array([1.0], dtype=np.float32),
                                (0, max(embb_dim - 1, 0)),
                                mode='constant',
                                constant_values=0.0,
                            )
                        )
                    ),
                ),
                candidates=[],
                greedy_reference=None,
                greedy_reference_utility=0.0,
                metadata={
                    "uav_index": float(uav_idx),
                    "rb_index": float(planning_rb),
                    "minislot_index": -1.0,
                    "planning_phase": 1.0,
                },
            )
            greedy_reference = self._planning_teacher_action(provisional_obs) if attach_greedy_reference else None
            observations[agent_id] = AgentObservation(
                local_obs=provisional_obs.local_obs,
                global_obs=provisional_obs.global_obs,
                masks=provisional_obs.masks,
                candidates=[],
                greedy_reference=greedy_reference,
                greedy_reference_utility=0.0,
                metadata=provisional_obs.metadata,
            )
        return observations

    def _planning_owner_summary(self, uav_idx: int, rb: int, embb_idx: int) -> Tuple[float, float, float]:
        if embb_idx < 0:
            return 0.0, 0.0, 0.0
        user_idx = self.sys_cfg.num_urllc_users + embb_idx
        gain = float(self.channel_gains_mag_sq[user_idx, uav_idx, rb])
        power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
        max_power = min(
            self.allocator._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
            self.algo_cfg.power_upper_bound,
        )
        scale = float(self.embb_power_scale_per_uav_rb[uav_idx, rb])
        scale = float(np.clip(
            scale,
            self.rl_cfg.env.embb_power_scale_min,
            self.rl_cfg.env.embb_power_scale_max,
        ))
        assigned_count = int(np.sum(self.owner_per_uav_rb[uav_idx, :] == embb_idx)) if self.owner_per_uav_rb is not None else 0
        if self.owner_per_uav_rb is not None and int(self.owner_per_uav_rb[uav_idx, rb]) != embb_idx:
            assigned_count += 1
        assigned_count = max(assigned_count, 1)
        per_rb_power = float(max_power * scale / assigned_count)
        snir = per_rb_power * gain / max(self.sys_cfg.noise_power, 1e-15)
        rate = float(self.capacity_model.shannon_capacity(snir, self.sys_cfg.subcarrier_bw))
        return gain, per_rb_power, rate

    def _planning_packet_pressure(self, uav_idx: int) -> Tuple[float, float]:
        if self.num_packets <= 0:
            return 0.0, 0.0
        packet_mask = np.asarray(self.packet_associated_uavs == uav_idx, dtype=bool)
        if not np.any(packet_mask):
            return 0.0, 0.0
        total_ratio = float(np.mean(packet_mask))
        early_cutoff = max(int(np.ceil(self.sys_cfg.num_minislots / 4.0)), 1)
        early_mask = packet_mask & (self.packet_release_minislots < early_cutoff)
        early_ratio = float(np.count_nonzero(early_mask) / max(int(np.count_nonzero(packet_mask)), 1))
        return total_ratio, early_ratio

    def _build_planning_local_obs(self, uav_idx: int, rb: int) -> np.ndarray:
        progress = self.planning_index / max(len(self._embb_plan_schedule) - 1, 1)
        remaining_rbs = float(len(self._embb_plan_schedule) - 1 - self.planning_index)
        owner = int(self.owner_per_uav_rb[uav_idx, rb]) if self.owner_per_uav_rb is not None else -1
        candidates = []
        if self.embb_owner_candidates_by_uav_rb:
            candidates = list(self.embb_owner_candidates_by_uav_rb[uav_idx][rb])
        max_obs_candidates = self.rl_cfg.action.max_candidate_packets
        candidate_gains = []
        candidate_rates = []
        candidate_counts = []
        for embb_idx in candidates[:max_obs_candidates]:
            gain, _per_rb_power, rate = self._planning_owner_summary(uav_idx, rb, embb_idx)
            candidate_gains.append(gain)
            candidate_rates.append(rate)
            candidate_counts.append(int(np.sum(self.owner_per_uav_rb[uav_idx, :] == embb_idx)) if self.owner_per_uav_rb is not None else 0)
        min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0))
        target_per_rb = min_rate / max(self.sys_cfg.num_subcarriers, 1)
        best_rate = max(candidate_rates) if candidate_rates else 0.0
        best_margin = (best_rate - target_per_rb) / max(target_per_rb, 1e-9) if target_per_rb > 0.0 else 0.0
        packet_ratio, early_packet_ratio = self._planning_packet_pressure(uav_idx)
        assigned_ratio = (
            float(np.count_nonzero(self.owner_per_uav_rb[uav_idx, :] >= 0)) / max(self.sys_cfg.num_subcarriers, 1)
            if self.owner_per_uav_rb is not None else 0.0
        )
        current_scale = float(self.embb_power_scale_per_uav_rb[uav_idx, rb])
        max_scale = max(self.rl_cfg.env.embb_power_scale_max, 1e-9)
        features = [
            progress,
            -1.0,
            rb / max(self.sys_cfg.num_subcarriers - 1, 1),
            uav_idx / max(self.sys_cfg.num_uavs - 1, 1),
            remaining_rbs / max(len(self._embb_plan_schedule) - 1, 1),
            float(owner >= 0),
            (owner + 1) / max(self.sys_cfg.num_embb_users, 1),
            best_rate / 1.0e7,
            np.clip(best_margin, -1.0, 1.0),
            current_scale / max_scale,
            self.associated_embb_counts[uav_idx] / max(self.sys_cfg.num_embb_users, 1),
            assigned_ratio,
            float(np.mean(candidate_rates) / 1.0e7) if candidate_rates else 0.0,
            len(candidates) / max(self.rl_cfg.action.max_embb_candidates, 1),
            float(np.mean(np.log10(np.asarray(candidate_gains) + 1e-15)) / 15.0) if candidate_gains else 0.0,
            float(np.max(np.log10(np.asarray(candidate_gains) + 1e-15)) / 15.0) if candidate_gains else 0.0,
            float(np.mean(candidate_counts) / max(self.sys_cfg.num_subcarriers, 1)) if candidate_counts else 0.0,
            packet_ratio,
            early_packet_ratio,
        ]
        features.extend([0.0] * self._phase_a_progress_obs_dim())
        current_owner = owner
        for rank, embb_idx in enumerate(candidates[:max_obs_candidates]):
            gain, per_rb_power, rate = self._planning_owner_summary(uav_idx, rb, embb_idx)
            assigned_count = int(np.sum(self.owner_per_uav_rb[uav_idx, :] == embb_idx)) if self.owner_per_uav_rb is not None else 0
            margin = (rate - target_per_rb) / max(target_per_rb, 1e-9) if target_per_rb > 0.0 else 0.0
            features.extend([
                1.0,
                (embb_idx + 1) / max(self.sys_cfg.num_embb_users, 1),
                np.log10(gain + 1e-15) / 15.0,
                rate / 1.0e7,
                float(embb_idx == current_owner),
                assigned_count / max(self.sys_cfg.num_subcarriers, 1),
                per_rb_power / max(self.algo_cfg.power_upper_bound, 1e-12),
                np.clip(margin, -1.0, 1.0),
                current_scale / max_scale,
                rank / max(max_obs_candidates - 1, 1),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ])
        for _ in range(max_obs_candidates - min(len(candidates), max_obs_candidates)):
            features.extend([0.0] * 16)
        features.extend([0.0] * self._rb_summary_feature_dim())
        return np.asarray(features, dtype=np.float32)

    def _build_planning_global_obs(self, rb: int) -> np.ndarray:
        progress = self.planning_index / max(len(self._embb_plan_schedule) - 1, 1)
        remaining_rbs = float(len(self._embb_plan_schedule) - 1 - self.planning_index)
        assigned_total = (
            float(np.count_nonzero(self.owner_per_uav_rb >= 0)) / max(self.sys_cfg.num_uavs * self.sys_cfg.num_subcarriers, 1)
            if self.owner_per_uav_rb is not None else 0.0
        )
        features = [
            progress,
            -1.0,
            rb / max(self.sys_cfg.num_subcarriers - 1, 1),
            assigned_total,
            0.0,
            remaining_rbs / max(len(self._embb_plan_schedule) - 1, 1),
        ]
        for uav_idx in range(self.sys_cfg.num_uavs):
            candidates = []
            if self.embb_owner_candidates_by_uav_rb:
                candidates = list(self.embb_owner_candidates_by_uav_rb[uav_idx][rb])
            candidate_rates = [self._planning_owner_summary(uav_idx, rb, embb_idx)[2] for embb_idx in candidates[:self.rl_cfg.action.max_candidate_packets]]
            candidate_gains = [self._planning_owner_summary(uav_idx, rb, embb_idx)[0] for embb_idx in candidates[:self.rl_cfg.action.max_candidate_packets]]
            packet_ratio, _early_ratio = self._planning_packet_pressure(uav_idx)
            assigned_ratio = (
                float(np.count_nonzero(self.owner_per_uav_rb[uav_idx, :] >= 0)) / max(self.sys_cfg.num_subcarriers, 1)
                if self.owner_per_uav_rb is not None else 0.0
            )
            current_owner = int(self.owner_per_uav_rb[uav_idx, rb]) if self.owner_per_uav_rb is not None else -1
            features.extend([
                self.associated_embb_counts[uav_idx] / max(self.sys_cfg.num_embb_users, 1),
                assigned_ratio,
                float(current_owner >= 0),
                float(self.embb_power_scale_per_uav_rb[uav_idx, rb]) / max(self.rl_cfg.env.embb_power_scale_max, 1e-9),
                float(max(candidate_rates) / 1.0e7) if candidate_rates else 0.0,
                float(np.mean(candidate_rates) / 1.0e7) if candidate_rates else 0.0,
                float(np.mean(np.log10(np.asarray(candidate_gains) + 1e-15)) / 15.0) if candidate_gains else 0.0,
                packet_ratio,
            ])
        return np.asarray(features, dtype=np.float32)

    def _build_embb_owner_mask(self, uav_idx: int, rb: int) -> np.ndarray:
        owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if owner_space == "global_owner_id_no_null":
            embb_dim = int(getattr(self.rl_cfg.action, "global_embb_owner_dim", 0) or 0)
            embb_dim = max(embb_dim, 1)
            mask = np.zeros(embb_dim, dtype=np.float32)
            if self.embb_owner_candidates_by_uav_rb:
                candidates = self.embb_owner_candidates_by_uav_rb[uav_idx][rb]
                for owner in candidates:
                    idx = int(owner)
                    if 0 <= idx < embb_dim and idx < int(self.sys_cfg.num_embb_users):
                        mask[idx] = 1.0
            if not np.any(mask > 0.5):
                # Minimal safe fallback: allow owner 0 if it exists.
                if int(self.sys_cfg.num_embb_users) > 0:
                    mask[0] = 1.0
            return mask

        embb_dim = self.rl_cfg.action.max_embb_candidates + int(self.rl_cfg.action.include_null_embb_option)
        mask = np.zeros(embb_dim, dtype=np.float32)
        mask[0] = 1.0
        if not self.embb_owner_candidates_by_uav_rb:
            return mask
        candidates = self.embb_owner_candidates_by_uav_rb[uav_idx][rb]
        if self.rl_cfg.env.force_embb_owner_per_rb and candidates:
            mask[0] = 0.0
        for idx in range(min(len(candidates), embb_dim - 1)):
            mask[idx + 1] = 1.0
        return mask

    def _overlay_margin(self, candidate: CandidatePacket) -> float:
        if not candidate.overlay_feasible:
            return float("-inf")
        return float(candidate.overlay_utility - candidate.puncture_utility)

    def _candidate_age(self, candidate: CandidatePacket, minislot: int) -> int:
        if candidate.packet_id >= self.packet_release_minislots.size:
            return 0
        return int(max(minislot - int(self.packet_release_minislots[candidate.packet_id]), 0))

    def _candidate_selection_key(self, candidate: CandidatePacket, minislot: int, uav_idx: int) -> Tuple[float, ...]:
        return (
            self._candidate_age(candidate, minislot),
            float(candidate.overlay_feasible),
            self._overlay_margin(candidate),
            candidate.best_utility,
            candidate.overlay_retention,
            -candidate.puncture_loss,
        )

    def _candidate_primary_assignment(self, candidates_by_uav: List[List[CandidatePacket]], minislot: int) -> Dict[int, int]:
        assignment: Dict[int, Tuple[Tuple[float, ...], int]] = {}
        for uav_idx, candidate_list in enumerate(candidates_by_uav):
            for candidate in candidate_list:
                key = (
                    float(candidate.overlay_feasible),
                    candidate.best_utility,
                    self._overlay_margin(candidate),
                    self._candidate_age(candidate, minislot),
                )
                previous = assignment.get(int(candidate.packet_id))
                if previous is None or key > previous[0]:
                    assignment[int(candidate.packet_id)] = (key, uav_idx)
        return {packet_id: uav_idx for packet_id, (_key, uav_idx) in assignment.items()}

    def _build_joint_candidate_views(self, candidates_by_uav: List[List[CandidatePacket]], minislot: int) -> List[List[CandidatePacket]]:
        max_candidates = self.rl_cfg.action.max_candidate_packets
        primary_assignment = self._candidate_primary_assignment(candidates_by_uav, minislot)
        rb = int(self._current_cell()[1])
        self._last_primary_assignment = {
            (rb, packet_id): uav_idx for packet_id, uav_idx in primary_assignment.items()
        }
        selected_by_uav: List[List[CandidatePacket]] = [[] for _ in range(self.sys_cfg.num_uavs)]
        globally_reserved = set()

        primary_counts = {
            uav_idx: sum(1 for candidate in candidates_by_uav[uav_idx] if primary_assignment.get(int(candidate.packet_id)) == uav_idx)
            for uav_idx in range(self.sys_cfg.num_uavs)
        }
        order = sorted(
            range(self.sys_cfg.num_uavs),
            key=lambda idx: (primary_counts[idx], len(candidates_by_uav[idx])),
        )

        for uav_idx in order:
            primary_candidates = [
                candidate
                for candidate in candidates_by_uav[uav_idx]
                if primary_assignment.get(int(candidate.packet_id)) == uav_idx
            ]
            selected = self._select_candidate_subset(primary_candidates, minislot, uav_idx)
            selected_by_uav[uav_idx].extend(selected[:max_candidates])
            globally_reserved.update(int(candidate.packet_id) for candidate in selected_by_uav[uav_idx])

        for uav_idx in order:
            if len(selected_by_uav[uav_idx]) >= max_candidates:
                continue
            selected_ids = {int(candidate.packet_id) for candidate in selected_by_uav[uav_idx]}
            unique_backups = [
                candidate
                for candidate in candidates_by_uav[uav_idx]
                if int(candidate.packet_id) not in selected_ids
                and int(candidate.packet_id) not in globally_reserved
            ]
            unique_backups.sort(
                key=lambda candidate: self._candidate_selection_key(candidate, minislot, uav_idx),
                reverse=True,
            )
            for candidate in unique_backups:
                if len(selected_by_uav[uav_idx]) >= max_candidates:
                    break
                selected_by_uav[uav_idx].append(candidate)
                selected_ids.add(int(candidate.packet_id))
                globally_reserved.add(int(candidate.packet_id))

        for uav_idx in order:
            if len(selected_by_uav[uav_idx]) >= max_candidates:
                continue
            selected_ids = {int(candidate.packet_id) for candidate in selected_by_uav[uav_idx]}
            overlap_backups = [
                candidate
                for candidate in candidates_by_uav[uav_idx]
                if int(candidate.packet_id) not in selected_ids
            ]
            overlap_backups.sort(
                key=lambda candidate: self._candidate_selection_key(candidate, minislot, uav_idx),
                reverse=True,
            )
            for candidate in overlap_backups:
                if len(selected_by_uav[uav_idx]) >= max_candidates:
                    break
                selected_by_uav[uav_idx].append(candidate)

        for uav_idx in range(self.sys_cfg.num_uavs):
            selected_by_uav[uav_idx].sort(
                key=lambda candidate: self._candidate_selection_key(candidate, minislot, uav_idx),
                reverse=True,
            )
            selected_by_uav[uav_idx] = selected_by_uav[uav_idx][:max_candidates]
        return selected_by_uav

    def _estimate_candidate_action_intercell_cost_after_source_mask(
        self,
        uav_idx: int,
        rb: int,
        minislot: int,
        candidate: CandidatePacket,
        mode: int,
    ) -> float:
        """Estimate selected-action victim intercell cost for a (candidate, mode) action.

        Uses the same semantics as `selected_action_intercell_cost_after_source_mask` (victim-side
        incoming-interference delta, including URLLC interference and corrected eMBB source masking).
        """
        if candidate is None:
            return 0.0
        packet_id = int(getattr(candidate, "packet_id", -1))
        if packet_id < 0 or packet_id >= int(getattr(self.scheduled_uavs, "size", 0) or 0):
            return 0.0
        if (
            self.mode_grid is None
            or self.packet_grid is None
            or self.scheduled_uavs is None
            or self.scheduled_power is None
            or self.channel_gains_mag_sq is None
        ):
            return 0.0

        victims = [other for other in range(self.sys_cfg.num_uavs) if other != uav_idx]
        if not victims:
            return 0.0
        try:
            pre = [
                float(self._compute_intercell_interference(other, rb, minislot, apply_embb_source_mask=True))
                for other in victims
            ]
        except Exception:
            return 0.0

        saved_packet = int(self.packet_grid[uav_idx, rb, minislot])
        saved_mode = int(self.mode_grid[uav_idx, rb, minislot])
        saved_owner = -1
        if self.embb_owner_grid is not None:
            saved_owner = int(self.embb_owner_grid[uav_idx, rb, minislot])
        saved_sched_uav = int(self.scheduled_uavs[packet_id])
        saved_sched_power = 0.0
        try:
            saved_sched_power = float(self.scheduled_power[packet_id, uav_idx])
        except Exception:
            saved_sched_power = 0.0
        try:
            self.packet_grid[uav_idx, rb, minislot] = int(packet_id)
            self.mode_grid[uav_idx, rb, minislot] = int(mode)
            if self.embb_owner_grid is not None:
                self.embb_owner_grid[uav_idx, rb, minislot] = (
                    -1 if int(mode) == MODE_PUNCTURE else int(candidate.embb_owner_for_mode(int(mode)))
                )
            self.scheduled_uavs[packet_id] = int(uav_idx)
            try:
                self.scheduled_power[packet_id, uav_idx] = float(candidate.required_power_for_mode(int(mode)))
            except Exception:
                self.scheduled_power[packet_id, uav_idx] = 0.0
            post = [
                float(self._compute_intercell_interference(other, rb, minislot, apply_embb_source_mask=True))
                for other in victims
            ]
            return float(sum(max(p - q, 0.0) for p, q in zip(post, pre)) / max(len(victims), 1))
        finally:
            self.packet_grid[uav_idx, rb, minislot] = int(saved_packet)
            self.mode_grid[uav_idx, rb, minislot] = int(saved_mode)
            if self.embb_owner_grid is not None:
                self.embb_owner_grid[uav_idx, rb, minislot] = int(saved_owner)
            self.scheduled_uavs[packet_id] = int(saved_sched_uav)
            try:
                self.scheduled_power[packet_id, uav_idx] = float(saved_sched_power)
            except Exception:
                pass

    def _build_masks(
        self,
        candidates: List[CandidatePacket],
        *,
        uav_idx: Optional[int] = None,
        rb: Optional[int] = None,
        minislot: Optional[int] = None,
        greedy_reference: Optional[HybridAction] = None,
    ) -> ActionMaskBundle:
        relax_hf_prefilter_mask = bool(getattr(self.rl_cfg.env, "greedy_hf_relax_prefilter_mask", False))
        mode_mask = np.zeros(self.rl_cfg.action.num_mode_actions, dtype=np.float32)
        mode_mask[MODE_KEEP] = 1.0
        if relax_hf_prefilter_mask:
            if len(candidates) > 0:
                mode_mask[MODE_PUNCTURE] = 1.0
                mode_mask[MODE_OVERLAY] = 1.0
        else:
            if any(c.puncture_feasible for c in candidates):
                mode_mask[MODE_PUNCTURE] = 1.0
            if any(c.overlay_feasible for c in candidates):
                mode_mask[MODE_OVERLAY] = 1.0

        packet_mask = np.zeros((self.rl_cfg.action.num_mode_actions, self.num_packet_options), dtype=np.float32)
        packet_mask[MODE_KEEP, 0] = 1.0
        if relax_hf_prefilter_mask:
            for idx, _candidate in enumerate(candidates, start=1):
                packet_mask[MODE_OVERLAY, idx] = 1.0
                packet_mask[MODE_PUNCTURE, idx] = 1.0
        else:
            for idx, candidate in enumerate(candidates, start=1):
                if candidate.overlay_feasible:
                    packet_mask[MODE_OVERLAY, idx] = 1.0
                if candidate.puncture_feasible:
                    packet_mask[MODE_PUNCTURE, idx] = 1.0

        has_feasible_admit = bool(
            np.any(packet_mask[MODE_OVERLAY, 1:] > 0.5)
            or np.any(packet_mask[MODE_PUNCTURE, 1:] > 0.5)
        )
        if (
            bool(getattr(self.rl_cfg.env, "disallow_keep_when_urllc_pending", True))
            and len(candidates) > 0
            and has_feasible_admit
        ):
            # Reliability-first semantics:
            # KEEP is disallowed only when there exists at least one feasible admit candidate.
            # If no feasible admit candidate exists, KEEP remains available to avoid forcing
            # reliability-violating admissions.
            mode_mask[MODE_KEEP] = 0.0
            packet_mask[MODE_KEEP, 0] = 0.0

        # Optional action-level intercell guard: suppress high-intercell candidates early so the policy
        # sees the constraint as a (soft) hard pressure instead of only a terminal penalty.
        running_ratio_cfg = getattr(self.rl_cfg.env, "action_intercell_guard_ratio_to_running_min", 1.25)
        use_running_min_guard = running_ratio_cfg is not None
        if (
            bool(getattr(self.rl_cfg.env, "enable_action_intercell_guard", False))
            and use_running_min_guard
            and uav_idx is not None
            and rb is not None
            and minislot is not None
            and candidates
        ):
            inter_norm = float(getattr(self.rl_cfg.reward, "terminal_intercell_penalty_normalizer", 1.0e-7) or 1.0e-7)
            running_min = float(getattr(self, "action_intercell_guard_running_min", float("inf")))
            if not np.isfinite(running_min):
                running_min = float("inf")

            greedy_cost = float("inf")
            if greedy_reference is not None:
                try:
                    g_mode = int(getattr(greedy_reference, "mode", MODE_KEEP))
                    g_opt = int(getattr(greedy_reference, "packet_option", 0))
                except Exception:
                    g_mode, g_opt = MODE_KEEP, 0
                if g_mode in {MODE_OVERLAY, MODE_PUNCTURE} and 1 <= g_opt <= len(candidates):
                    greedy_cost = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidates[g_opt - 1],
                            int(g_mode),
                        )
                    )

            baseline = min(float(running_min), float(greedy_cost))
            if not np.isfinite(baseline):
                baseline = float("inf")
            baseline = float(max(baseline, inter_norm))
            guard_ratio = float(running_ratio_cfg or 1.25)
            guard_ratio = float(max(guard_ratio, 1.0))
            threshold = float(guard_ratio * baseline)

            keep_best = bool(getattr(self.rl_cfg.env, "action_intercell_guard_keep_best_feasible", True))
            masked_any = False
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                feasible_options = [idx for idx in range(1, len(candidates) + 1) if packet_mask[mode, idx] > 0.5]
                if not feasible_options:
                    continue
                costs = {}
                for idx in feasible_options:
                    cost = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidates[idx - 1],
                            int(mode),
                        )
                    )
                    costs[idx] = cost
                    if cost > threshold + 1.0e-12:
                        packet_mask[mode, idx] = 0.0
                        masked_any = True
                        self.action_intercell_guard_masked_option_count += 1
                remaining = [idx for idx in feasible_options if packet_mask[mode, idx] > 0.5]
                if keep_best and (not remaining):
                    best_idx = min(feasible_options, key=lambda key: float(costs.get(key, float("inf"))))
                    packet_mask[mode, best_idx] = 1.0
            if masked_any:
                self.action_intercell_guard_active_cell_count += 1
            self.action_intercell_guard_total_cell_count += 1

            # Update mode mask based on post-guard feasibility.
            if not np.any(packet_mask[MODE_OVERLAY, 1:] > 0.5):
                mode_mask[MODE_OVERLAY] = 0.0
            if not np.any(packet_mask[MODE_PUNCTURE, 1:] > 0.5):
                mode_mask[MODE_PUNCTURE] = 0.0
        owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if owner_space == "global_owner_id_no_null":
            # Phase-A action masking: owner head is inactive here.
            embb_dim = int(getattr(self.rl_cfg.action, "global_embb_owner_dim", 0) or 0)
            embb_dim = max(embb_dim, 1)
            embb_owner_mask = np.zeros(embb_dim, dtype=np.float32)
        else:
            embb_dim = self.rl_cfg.action.max_embb_candidates + int(self.rl_cfg.action.include_null_embb_option)
            embb_owner_mask = np.zeros(embb_dim, dtype=np.float32)
            embb_owner_mask[0] = 1.0
        return ActionMaskBundle(mode_mask=mode_mask, packet_mask=packet_mask, embb_owner_mask=embb_owner_mask)

    def _urgent_packet_count_for_uav(self, uav_idx: int, minislot: int) -> int:
        urgent_packets = 0
        max_latency = float(getattr(self.urllc_cfg, "max_latency_minislots", self.sys_cfg.num_minislots))
        for packet_id in self._available_packet_ids(minislot):
            if (
                self.num_packets > 0
                and (not bool(getattr(self.rl_cfg.env, "use_all_uavs_as_candidate_servers", False)))
                and int(self.packet_associated_uavs[packet_id]) != uav_idx
            ):
                continue
            release = int(self.packet_release_minislots[packet_id])
            age = max(minislot - release, 0)
            if age >= max_latency - 1:
                urgent_packets += 1
        return int(urgent_packets)

    def _rb_summary_feature_dim(self) -> int:
        return 9 * self.sys_cfg.num_subcarriers

    def _rb_summary_block(self, uav_idx: int, rb: int, minislot: int) -> List[float]:
        owner = int(self.owner_per_uav_rb[uav_idx, rb]) if self.owner_per_uav_rb is not None else -1
        base_rate = self._base_rate_for_cell(uav_idx, owner, rb)
        min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0))
        target_per_rb = min_rate / max(self.sys_cfg.num_subcarriers, 1)
        min_rate_margin = (base_rate - target_per_rb) / max(target_per_rb, 1e-9) if target_per_rb > 0.0 else 0.0

        raw_candidates = self._enumerate_candidates_for_cell(uav_idx, rb, minislot)
        summary_candidates = self._apply_low_damage_candidate_constraints(
            raw_candidates,
            self._current_actual_load(),
        )
        overlay_candidates = [candidate for candidate in summary_candidates if candidate.overlay_feasible]
        puncture_candidates = [candidate for candidate in summary_candidates if candidate.puncture_feasible]

        best_overlay_utility = max((candidate.overlay_utility for candidate in overlay_candidates), default=0.0)
        best_puncture_utility = max((candidate.puncture_utility for candidate in puncture_candidates), default=0.0)
        best_overlay_retention = max((candidate.overlay_retention for candidate in overlay_candidates), default=0.0)
        best_puncture_loss = min((candidate.puncture_loss for candidate in puncture_candidates), default=0.0)

        urgent_packet_ids = set()
        max_latency = float(getattr(self.urllc_cfg, "max_latency_minislots", self.sys_cfg.num_minislots))
        for candidate in raw_candidates:
            packet_id = int(candidate.packet_id)
            if packet_id in urgent_packet_ids:
                continue
            release = int(self.packet_release_minislots[packet_id]) if packet_id < self.packet_release_minislots.size else 0
            age = max(minislot - release, 0)
            if age >= max_latency - 1:
                urgent_packet_ids.add(packet_id)

        packet_norm = max(self.num_packets, 1)
        return [
            base_rate / 1.0e7,
            len(overlay_candidates) / packet_norm,
            len(puncture_candidates) / packet_norm,
            np.clip(best_overlay_utility / 1.0e6, -10.0, 10.0),
            np.clip(best_puncture_utility / 1.0e6, -10.0, 10.0),
            float(np.clip(best_overlay_retention, 0.0, 1.0)),
            best_puncture_loss / 1.0e7,
            len(urgent_packet_ids) / packet_norm,
            float(np.clip(min_rate_margin, -1.0, 1.0)),
        ]

    def _build_minislot_rb_summary_features(self, uav_idx: int, minislot: int) -> List[float]:
        cache_key = (int(self.current_cell_index), int(uav_idx), int(minislot))
        cached = self._rb_summary_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        features: List[float] = []
        for summary_rb in range(self.sys_cfg.num_subcarriers):
            features.extend(self._rb_summary_block(uav_idx, summary_rb, minislot))
        self._rb_summary_cache[cache_key] = tuple(features)
        return features

    def _build_local_obs(self, uav_idx: int, rb: int, minislot: int, candidates: List[CandidatePacket]) -> np.ndarray:
        progress = self.current_cell_index / max(len(self._cell_schedule) - 1, 1)
        remaining_cells = float(len(self._cell_schedule) - 1 - self.current_cell_index)
        owner = int(self.owner_per_uav_rb[uav_idx, rb])
        base_rate = self._base_rate_for_cell(uav_idx, owner, rb)
        embb_per_rb_power = self._embb_per_rb_power_for_owner(uav_idx, owner, rb, minislot) if owner >= 0 else 0.0
        min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0))
        target_per_rb = min_rate / max(self.sys_cfg.num_subcarriers, 1)
        min_rate_margin = (base_rate - target_per_rb) / max(target_per_rb, 1e-9)
        urgent_packets = self._urgent_packet_count_for_uav(uav_idx, minislot)
        progress_summary = self._phase_a_progress_summary(minislot)
        features = [
            progress,
            minislot / max(self.sys_cfg.num_minislots - 1, 1),
            rb / max(self.sys_cfg.num_subcarriers - 1, 1),
            uav_idx / max(self.sys_cfg.num_uavs - 1, 1),
            remaining_cells / max(len(self._cell_schedule) - 1, 1),
            float(owner >= 0),
            (owner + 1) / max(self.sys_cfg.num_embb_users, 1),
            base_rate / 1.0e7,
            np.clip(min_rate_margin, -1.0, 1.0),
            embb_per_rb_power / max(self.algo_cfg.power_upper_bound, 1e-12),
            self.associated_embb_counts[uav_idx] / max(self.sys_cfg.num_embb_users, 1),
            self.associated_urllc_counts[uav_idx] / max(self.sys_cfg.num_urllc_users, 1),
            self.scheduled_counts[uav_idx] / max(self.num_packets, 1),
            self.overlay_counts[uav_idx] / max(len(self._cell_schedule), 1),
            self.puncture_counts[uav_idx] / max(len(self._cell_schedule), 1),
            self.overlay_success_ema[uav_idx],
            self.puncture_loss_ema[uav_idx],
            urgent_packets / max(self.num_packets, 1),
            len(self._available_packet_ids(minislot)) / max(self.num_packets, 1),
        ]
        features.extend(self._phase_a_progress_obs_features(progress_summary))
        for candidate in candidates:
            features.extend(self._candidate_feature_block(uav_idx, candidate, minislot))
        missing = self.rl_cfg.action.max_candidate_packets - len(candidates)
        for _ in range(missing):
            features.extend([0.0] * 16)
        features.extend(self._build_minislot_rb_summary_features(uav_idx, minislot))
        return np.asarray(features, dtype=np.float32)

    def _build_global_obs(self, rb: int, minislot: int) -> np.ndarray:
        progress = self.current_cell_index / max(len(self._cell_schedule) - 1, 1)
        current_available = len(self._available_packet_ids(minislot))
        remaining_cells = float(len(self._cell_schedule) - 1 - self.current_cell_index)
        features = [
            progress,
            minislot / max(self.sys_cfg.num_minislots - 1, 1),
            rb / max(self.sys_cfg.num_subcarriers - 1, 1),
            current_available / max(self.num_packets, 1),
            np.count_nonzero(self.scheduled_uavs >= 0) / max(self.num_packets, 1),
            remaining_cells / max(len(self._cell_schedule) - 1, 1),
        ]
        for uav_idx in range(self.sys_cfg.num_uavs):
            urgent_packets = self._urgent_packet_count_for_uav(uav_idx, minislot)
            features.extend([
                self.associated_embb_counts[uav_idx] / max(self.sys_cfg.num_embb_users, 1),
                self.associated_urllc_counts[uav_idx] / max(self.sys_cfg.num_urllc_users, 1),
                self.scheduled_counts[uav_idx] / max(self.num_packets, 1),
                self.overlay_counts[uav_idx] / max(len(self._cell_schedule), 1),
                self.puncture_counts[uav_idx] / max(len(self._cell_schedule), 1),
                self.overlay_success_ema[uav_idx],
                self.puncture_loss_ema[uav_idx],
                urgent_packets / max(self.num_packets, 1),
            ])
        return np.asarray(features, dtype=np.float32)
    def _candidate_feature_block(self, uav_idx: int, candidate: CandidatePacket, minislot: int) -> List[float]:
        release = int(self.packet_release_minislots[candidate.packet_id]) if candidate.packet_id < self.packet_release_minislots.size else 0
        age = float(max(minislot - release, 0))
        max_latency = float(getattr(self.urllc_cfg, "max_latency_minislots", self.sys_cfg.num_minislots))
        slack = max(max_latency - age - 1.0, 0.0)
        return [
            1.0,
            (candidate.source_user + 1) / max(self.sys_cfg.num_urllc_users, 1),
            np.log10(candidate.channel_gain + 1e-15) / 15.0,
            float(candidate.puncture_feasible),
            float(candidate.overlay_feasible),
            candidate.puncture_power / max(self.algo_cfg.power_upper_bound, 1e-12),
            candidate.overlay_power / max(self.algo_cfg.power_upper_bound, 1e-12),
            candidate.puncture_reliability - (1.0 - self.urllc_cfg.target_error_probability),
            candidate.overlay_reliability - (1.0 - self.urllc_cfg.target_error_probability),
            candidate.puncture_loss / 1.0e7,
            candidate.overlay_retention,
            np.clip(candidate.puncture_utility / 1.0e6, -10.0, 10.0),
            np.clip(candidate.overlay_utility / 1.0e6, -10.0, 10.0),
            age / max(self.sys_cfg.num_minislots, 1),
            slack / max(max_latency, 1.0),
            release / max(self.sys_cfg.num_minislots, 1),
        ]

    def _annotate_candidate_contention(self, candidates_by_uav: List[List[CandidatePacket]]) -> None:
        # Association is fixed: each packet belongs to exactly one UAV.
        # Contention metadata is therefore disabled and forced to zero.
        for candidate_list in candidates_by_uav:
            for candidate in candidate_list:
                candidate.feasible_uav_count = 1
                candidate.overlay_uav_count = 1 if candidate.overlay_feasible else 0
                candidate.puncture_uav_count = 1 if candidate.puncture_feasible else 0
                candidate.contention_score = 0.0

    def _evaluate_candidates_for_cell(self, uav_idx: int, rb: int, minislot: int) -> List[CandidatePacket]:
        candidates = self._enumerate_candidates_for_cell(uav_idx, rb, minislot)
        candidates = self._apply_low_damage_candidate_constraints(candidates, self._current_actual_load())
        self._annotate_candidate_contention([candidates])
        return self._select_candidate_subset(candidates, minislot, uav_idx)

    def _overlay_owner_candidates_for_cell(self, uav_idx: int) -> List[int]:
        if self.embb_selected_uavs is None:
            return []
        return [
            embb_owner
            for embb_owner in range(self.sys_cfg.num_embb_users)
            if int(self.embb_selected_uavs[embb_owner]) == uav_idx
        ]

    def _enumerate_candidates_for_cell(self, uav_idx: int, rb: int, minislot: int) -> List[CandidatePacket]:
        candidates: List[CandidatePacket] = []
        if (
            bool(getattr(self.rl_cfg.env, "allow_packet_carryover_across_minislots", False))
            and int(getattr(self, "num_packets", 0) or 0) > 0
        ):
            if int(getattr(self, "_carryover_tracking_minislot", -1)) != int(minislot):
                self._carryover_tracking_minislot = int(minislot)
                self._carryover_seen_in_minislot = np.zeros(self.num_packets, dtype=bool)
                self._carryover_feasible_in_minislot = np.zeros(self.num_packets, dtype=bool)
        owner = int(self.owner_per_uav_rb[uav_idx, rb])
        channel_uses = self.sys_cfg.channel_uses_per_minislot
        base_rate = self._base_rate_for_cell(uav_idx, owner, rb)
        puncture_embb_user_idx = self.sys_cfg.num_urllc_users + owner if owner >= 0 else -1

        for packet_id in self._available_packet_ids(minislot):
            if self.num_packets > 0 and int(self.packet_associated_uavs[packet_id]) != uav_idx:
                continue
            source_user = int(self.packet_sources[packet_id])
            urllc_gain = float(self.channel_gains_mag_sq[source_user, uav_idx, rb])
            if urllc_gain <= 1e-12:
                continue

            max_power_w = min(
                self.allocator._dbm_to_watts(
                    self.urllc_cfg.power_limits[min(source_user, len(self.urllc_cfg.power_limits) - 1)]
                ),
                self.algo_cfg.power_upper_bound,
            )
            overload_penalty = 0.0
            packet_bits = self._packet_bits_for_user(source_user)

            intercell_punct = self._compute_intercell_interference(uav_idx, rb, minislot)
            puncture_power = self.allocator._bisection_search_urllc_power(
                urllc_gain,
                packet_bits,
                self.urllc_cfg.target_error_probability,
                max_power_w,
                channel_uses,
                interference_power=intercell_punct,
            )
            puncture_snir = puncture_power * urllc_gain / max(self.sys_cfg.noise_power + intercell_punct, 1e-15)
            puncture_error = self.capacity_model.decoding_error_probability(puncture_snir, packet_bits, channel_uses)
            puncture_reliability = 1.0 - puncture_error
            puncture_feasible = puncture_error <= self.urllc_cfg.target_error_probability
            puncture_loss = base_rate
            puncture_utility = self._safe_metric(
                self.allocator._compute_action_utility(
                    embb_rate_delta=-puncture_loss,
                    urllc_reliability=puncture_reliability,
                    power=puncture_power,
                    overload_penalty=overload_penalty,
                ),
                default=-1.0e6,
            )

            overlay_feasible = False
            overlay_power = 0.0
            overlay_snir = 0.0
            overlay_reliability = 0.0
            overlay_loss = base_rate
            overlay_retention = 0.0
            overlay_utility = float("-inf")
            overlay_no_intercell_feasible = False
            overlay_error = 1.0
            post_sic_snir = 0.0
            base_embb_snir = 0.0
            best_overlay_snir = 0.0
            best_post_sic_snir = 0.0
            best_base_embb_snir = 0.0
            best_base_embb_signal = 0.0
            best_base_embb_intercell = 0.0
            min_post_sic = 10 ** (self.algo_cfg.embb_min_sic_snir_db / 10.0)
            intercell_noma = 0.0
            noma_interference = 0.0
            gain_ratio = 0.0
            overlay_embb_owner = -1
            overlay_embb_user_idx = -1
            best_overlay_score = float("-inf")
            gain_ratio_ok_found = False
            cause_required_power_exceeds_budget = False
            cause_urllc_sinr_unachievable = False
            cause_embb_retention_below_threshold = False
            cause_cross_uav_interference_too_high = False
            cause_gain_ratio_unqualified = False
            cause_overlay_margin_blocked = False
            cause_overlay_positive_gate_blocked = False
            cause_no_overlay_owner_available = False
            cause_overlay_reliability_failed = False
            cause_overlay_sic_failed = False
            overlay_owner_candidates = self._overlay_owner_candidates_for_cell(uav_idx)
            if not overlay_owner_candidates:
                cause_no_overlay_owner_available = True
            for overlay_owner in overlay_owner_candidates:
                embb_user_idx = self.sys_cfg.num_urllc_users + overlay_owner
                embb_gain = float(self.channel_gains_mag_sq[embb_user_idx, uav_idx, rb])
                embb_per_rb_power = self._embb_per_rb_power_for_owner(uav_idx, overlay_owner, rb, minislot)
                local_gain_ratio = urllc_gain / max(embb_gain, 1e-12)
                if local_gain_ratio < self.algo_cfg.min_noma_gain_ratio:
                    continue

                gain_ratio_ok_found = True
                intercell_noma_local = self._compute_intercell_interference(uav_idx, rb, minislot)
                # Interference for URLLC decoding should include the local eMBB signal
                # plus inter-cell interference. Do not double-count local eMBB inside
                # intercell_noma_local.
                local_noma_interference = embb_per_rb_power * embb_gain
                noma_interference_local = intercell_noma_local + local_noma_interference
                overlay_power_local = self.allocator._bisection_search_urllc_power(
                    urllc_gain,
                    packet_bits,
                    self.urllc_cfg.target_error_probability,
                    max_power_w,
                    channel_uses,
                    interference_power=noma_interference_local,
                )
                overlay_snir_local = overlay_power_local * urllc_gain / max(
                    self.sys_cfg.noise_power + noma_interference_local,
                    1e-15,
                )
                if overlay_snir_local > best_overlay_snir:
                    best_overlay_snir = float(overlay_snir_local)
                overlay_error_local = self.capacity_model.decoding_error_probability(
                    overlay_snir_local,
                    packet_bits,
                    channel_uses,
                )
                overlay_reliability_local = 1.0 - overlay_error_local
                post_sic_snir_local = embb_per_rb_power * embb_gain / max(
                    self.sys_cfg.noise_power
                    + intercell_noma_local
                    + self.algo_cfg.sic_residual_factor * overlay_power_local * urllc_gain,
                    1e-15,
                )
                if post_sic_snir_local > best_post_sic_snir:
                    best_post_sic_snir = float(post_sic_snir_local)
                overlay_feasible_local = (
                    overlay_error_local <= self.urllc_cfg.target_error_probability and
                    post_sic_snir_local >= min_post_sic
                )
                cause_overlay_reliability_failed = (
                    cause_overlay_reliability_failed
                    or (overlay_error_local > self.urllc_cfg.target_error_probability)
                )
                cause_overlay_sic_failed = (
                    cause_overlay_sic_failed
                    or (post_sic_snir_local < min_post_sic)
                )

                overlay_power_no_intercell = self.allocator._bisection_search_urllc_power(
                    urllc_gain,
                    packet_bits,
                    self.urllc_cfg.target_error_probability,
                    max_power_w,
                    channel_uses,
                    interference_power=local_noma_interference,
                )
                overlay_snir_no_intercell = overlay_power_no_intercell * urllc_gain / max(
                    self.sys_cfg.noise_power + local_noma_interference,
                    1e-15,
                )
                overlay_error_no_intercell = self.capacity_model.decoding_error_probability(
                    overlay_snir_no_intercell,
                    packet_bits,
                    channel_uses,
                )
                post_sic_no_intercell = embb_per_rb_power * embb_gain / max(
                    self.sys_cfg.noise_power
                    + self.algo_cfg.sic_residual_factor * overlay_power_no_intercell * urllc_gain,
                    1e-15,
                )
                overlay_no_intercell_feasible_local = (
                    overlay_error_no_intercell <= self.urllc_cfg.target_error_probability and
                    post_sic_no_intercell >= min_post_sic
                )

                cause_required_power_exceeds_budget = (
                    cause_required_power_exceeds_budget or
                    (
                        overlay_power_local >= 0.999 * max_power_w and
                        overlay_error_local > self.urllc_cfg.target_error_probability
                    )
                )
                cause_urllc_sinr_unachievable = (
                    cause_urllc_sinr_unachievable or
                    (overlay_error_local > self.urllc_cfg.target_error_probability)
                )
                cause_embb_retention_below_threshold = (
                    cause_embb_retention_below_threshold or
                    (post_sic_snir_local < min_post_sic)
                )
                cause_cross_uav_interference_too_high = (
                    cause_cross_uav_interference_too_high or
                    ((not overlay_feasible_local) and overlay_no_intercell_feasible_local and intercell_noma_local > 0.0)
                )

                base_embb_snir = embb_per_rb_power * embb_gain / max(
                    self.sys_cfg.noise_power + intercell_noma_local,
                    1e-15,
                )
                if base_embb_snir > best_base_embb_snir:
                    best_base_embb_snir = float(base_embb_snir)
                    best_base_embb_signal = float(embb_per_rb_power * embb_gain)
                    best_base_embb_intercell = float(intercell_noma_local)
                base_rate_local = self.capacity_model.shannon_capacity(
                    base_embb_snir,
                    self.sys_cfg.subcarrier_bw,
                ) / max(self.sys_cfg.num_minislots, 1)
                overlay_retention_local = min(
                    0.95,
                    max(
                        self.algo_cfg.noma_retention_factor,
                        np.log2(1.0 + post_sic_snir_local) / max(np.log2(1.0 + base_embb_snir), 1e-12),
                    ),
                )
                overlay_loss_local = base_rate_local * (1.0 - overlay_retention_local)
                overlay_utility_local = self._safe_metric(
                    self.allocator._compute_action_utility(
                        embb_rate_delta=-overlay_loss_local,
                        urllc_reliability=overlay_reliability_local,
                        power=overlay_power_local,
                        overload_penalty=overload_penalty,
                    ),
                    default=-1.0e6,
                )
                retained_rate_local = base_rate_local * overlay_retention_local
                # Soft overlay gates: keep feasibility but penalize poor retention/margin.
                retention_gap = max(self.rl_cfg.env.min_overlay_retention - overlay_retention_local, 0.0)
                ratio_gap = max(self.rl_cfg.env.min_overlay_retained_rate_ratio - (retained_rate_local / max(base_rate_local, 1e-12)), 0.0)
                puncture_retained_rate = max(base_rate_local - puncture_loss, 0.0)
                margin_gap = max(puncture_retained_rate - retained_rate_local, 0.0)
                overlay_utility_local -= (
                    self.rl_cfg.env.soft_overlay_retention_penalty * retention_gap
                    + self.rl_cfg.env.soft_overlay_ratio_penalty * ratio_gap
                    + self.rl_cfg.env.soft_overlay_margin_penalty * margin_gap / max(base_rate_local, 1e-12)
                )
                if not self.rl_cfg.env.allow_negative_overlay_margin and retained_rate_local <= puncture_retained_rate:
                    cause_overlay_margin_blocked = True
                    continue
                if self.rl_cfg.env.enforce_throughput_positive_overlay and retained_rate_local <= puncture_retained_rate:
                    overlay_feasible_local = False
                    cause_overlay_positive_gate_blocked = True
                    continue
                overlay_score_local = (
                    retained_rate_local / 1.0e6
                    + 0.10 * overlay_retention_local
                    + 0.10 * (overlay_utility_local / 1.0e6)
                    - 0.05 * (overlay_power_local / max(max_power_w, 1e-9))
                )

                if overlay_feasible_local and (
                    (not overlay_feasible) or
                    (overlay_score_local > best_overlay_score + 1e-12) or
                    (
                        abs(overlay_score_local - best_overlay_score) <= 1e-12 and
                        overlay_utility_local > overlay_utility + 1e-12
                    ) or
                    (
                        abs(overlay_score_local - best_overlay_score) <= 1e-12 and
                        abs(overlay_utility_local - overlay_utility) <= 1e-12 and
                        overlay_retention_local > overlay_retention + 1e-12
                    ) or
                    (
                        abs(overlay_score_local - best_overlay_score) <= 1e-12 and
                        abs(overlay_utility_local - overlay_utility) <= 1e-12 and
                        abs(overlay_retention_local - overlay_retention) <= 1e-12 and
                        overlay_power_local < overlay_power
                    )
                ):
                    best_overlay_score = float(overlay_score_local)
                    overlay_feasible = True
                    overlay_power = float(overlay_power_local)
                    overlay_snir = float(overlay_snir_local)
                    overlay_reliability = float(overlay_reliability_local)
                    overlay_loss = float(overlay_loss_local)
                    overlay_retention = float(overlay_retention_local)
                    overlay_utility = float(overlay_utility_local)
                    overlay_error = float(overlay_error_local)
                    post_sic_snir = float(post_sic_snir_local)
                    intercell_noma = float(intercell_noma_local)
                    noma_interference = float(noma_interference_local)
                    overlay_no_intercell_feasible = bool(overlay_no_intercell_feasible_local)
                    gain_ratio = float(local_gain_ratio)
                    overlay_embb_owner = int(overlay_owner)
                    overlay_embb_user_idx = int(embb_user_idx)

            if (not gain_ratio_ok_found) and overlay_owner_candidates:
                cause_gain_ratio_unqualified = True

            known_structural_causes = bool(
                cause_required_power_exceeds_budget
                or cause_urllc_sinr_unachievable
                or cause_embb_retention_below_threshold
                or cause_cross_uav_interference_too_high
                or cause_gain_ratio_unqualified
                or cause_overlay_margin_blocked
                or cause_overlay_positive_gate_blocked
                or cause_no_overlay_owner_available
                or cause_overlay_reliability_failed
                or cause_overlay_sic_failed
            )
            cause_other_structural_reason = bool((not overlay_feasible) and (not known_structural_causes))

            candidates.append(
                CandidatePacket(
                    packet_id=packet_id,
                    source_user=source_user,
                    associated_uav=int(self.packet_associated_uavs[packet_id]) if self.num_packets > 0 else -1,
                    channel_gain=urllc_gain,
                    puncture_feasible=puncture_feasible,
                    overlay_feasible=overlay_feasible,
                    puncture_power=float(puncture_power),
                    overlay_power=float(overlay_power),
                    puncture_reliability=float(puncture_reliability),
                    overlay_reliability=float(overlay_reliability),
                    puncture_loss=float(puncture_loss),
                    overlay_loss=float(overlay_loss),
                    overlay_retention=float(overlay_retention),
                    puncture_utility=float(puncture_utility),
                    overlay_utility=float(overlay_utility),
                    fixed_embb_owner=int(owner),
                    fixed_embb_user_idx=int(puncture_embb_user_idx),
                    overlay_embb_owner=int(overlay_embb_owner),
                    overlay_embb_user_idx=int(overlay_embb_user_idx),
                    overlay_urllc_snir=float(best_overlay_snir),
                    post_sic_snir=float(best_post_sic_snir),
                    base_embb_snir=float(best_base_embb_snir),
                    base_embb_signal_power=float(best_base_embb_signal),
                    base_embb_intercell_power=float(best_base_embb_intercell),
                    rb_index=int(rb),
                    cause_urllc_sinr_unachievable=bool((not overlay_feasible) and cause_urllc_sinr_unachievable),
                    cause_embb_retention_below_threshold=bool((not overlay_feasible) and cause_embb_retention_below_threshold),
                    cause_required_power_exceeds_budget=bool((not overlay_feasible) and cause_required_power_exceeds_budget),
                    cause_rb_minislot_collision=False,
                    cause_packet_already_scheduled_elsewhere=False,
                    cause_cross_uav_interference_too_high=bool((not overlay_feasible) and cause_cross_uav_interference_too_high),
                    cause_deadline_or_release_violation=False,
                    cause_other_structural_reason=bool((not overlay_feasible) and cause_other_structural_reason),
                    cause_gain_ratio_unqualified=bool((not overlay_feasible) and cause_gain_ratio_unqualified),
                    cause_overlay_margin_blocked=bool((not overlay_feasible) and cause_overlay_margin_blocked),
                    cause_overlay_positive_gate_blocked=bool((not overlay_feasible) and cause_overlay_positive_gate_blocked),
                    cause_no_overlay_owner_available=bool((not overlay_feasible) and cause_no_overlay_owner_available),
                    cause_overlay_reliability_failed=bool((not overlay_feasible) and cause_overlay_reliability_failed),
                    cause_overlay_sic_failed=bool((not overlay_feasible) and cause_overlay_sic_failed),
                )
            )
            if (
                bool(getattr(self.rl_cfg.env, "allow_packet_carryover_across_minislots", False))
                and 0 <= int(packet_id) < int(getattr(self, "num_packets", 0) or 0)
            ):
                self._carryover_seen_in_minislot[int(packet_id)] = True
                if bool(puncture_feasible) or bool(overlay_feasible):
                    self._carryover_feasible_in_minislot[int(packet_id)] = True

        return candidates

    def _select_candidate_subset(self, candidates: List[CandidatePacket], minislot: int, uav_idx: int) -> List[CandidatePacket]:
        max_candidates = self.rl_cfg.action.max_candidate_packets
        if len(candidates) <= max_candidates:
            return candidates

        urgent_quota = min(max_candidates, max(1, int(np.ceil(0.375 * max_candidates))))
        urgent_candidates = sorted(
            candidates,
            key=lambda candidate: (
                self._candidate_age(candidate, minislot),
                candidate.puncture_feasible or candidate.overlay_feasible,
                candidate.best_utility,
                candidate.overlay_feasible,
            ),
            reverse=True,
        )
        selected = urgent_candidates[:urgent_quota]
        selected_ids = {candidate.packet_id for candidate in selected}

        overlay_candidates = [candidate for candidate in candidates if candidate.overlay_feasible]
        overlay_candidates.sort(
            key=lambda candidate: (
                self._candidate_age(candidate, minislot),
                self._overlay_margin(candidate),
                candidate.overlay_utility,
                candidate.overlay_retention,
                -candidate.overlay_loss,
                candidate.best_utility,
            ),
            reverse=True,
        )

        min_overlay = min(
            len(overlay_candidates),
            max_candidates - len(selected),
            max(0, self.rl_cfg.action.min_overlay_candidate_slots - sum(int(candidate.overlay_feasible) for candidate in selected)),
            max(0, int(np.ceil(max_candidates * self.rl_cfg.action.overlay_candidate_share)) - sum(int(candidate.overlay_feasible) for candidate in selected)),
        )
        for candidate in overlay_candidates:
            if len(selected) >= max_candidates or min_overlay <= 0:
                break
            if candidate.packet_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.packet_id)
            min_overlay -= 1

        remaining = [candidate for candidate in candidates if candidate.packet_id not in selected_ids]
        remaining.sort(
            key=lambda candidate: (
                self._candidate_age(candidate, minislot),
                self._overlay_margin(candidate),
                candidate.best_utility,
                candidate.overlay_feasible,
                candidate.overlay_utility,
                -candidate.puncture_loss,
            ),
            reverse=True,
        )
        selected.extend(remaining[: max_candidates - len(selected)])
        selected.sort(
            key=lambda candidate: (
                self._overlay_margin(candidate),
                candidate.best_utility,
                candidate.overlay_feasible,
                candidate.overlay_utility,
                -candidate.puncture_loss,
            ),
            reverse=True,
        )
        return selected[:max_candidates]

    def _apply_agent_action(self, uav_idx: int, agent_id: str, minislot: int, rb: int, observation: AgentObservation, shielded_action) -> Dict[str, float]:
        reward = 0.0
        chosen_mode_name = MODE_NAMES[shielded_action.action.mode]
        scheduled_packet = -1
        actual_power = 0.0
        reward_terms = {}
        self.phase_a_total_decisions += 1
        # Pre-action victim inter-cell interference (used for action-level intercell cost + optional penalty).
        # We measure *incoming* intercell interference at neighbor UAV receivers on the same (rb, minislot),
        # before and after applying the selected action for this (uav_idx, rb, minislot). This gives a
        # meaningful delta signal even when puncture disables local eMBB (outgoing baseline deltas can cancel).
        step_intercell_w = float(getattr(self.rl_cfg.reward, "step_intercell_outgoing_delta_penalty_weight", 0.0) or 0.0)
        pre_victim_intercell_after_mask = None
        pre_victim_intercell_before_mask = None
        try:
            pre_victim_intercell_after_mask = [
                float(self._compute_intercell_interference(other_uav, rb, minislot, apply_embb_source_mask=True))
                for other_uav in range(self.sys_cfg.num_uavs)
                if other_uav != uav_idx
            ]
            pre_victim_intercell_before_mask = [
                float(self._compute_intercell_interference(other_uav, rb, minislot, apply_embb_source_mask=False))
                for other_uav in range(self.sys_cfg.num_uavs)
                if other_uav != uav_idx
            ]
        except Exception:
            pre_victim_intercell_after_mask = None
            pre_victim_intercell_before_mask = None
        # Per-load debug: feasibility landscape at this decision cell.
        try:
            candidates = list(observation.candidates or [])
        except Exception:
            candidates = []
        running_guard_ratio_cfg = getattr(self.rl_cfg.env, "action_intercell_guard_ratio_to_running_min", 1.25)
        local_min_guard_enabled = bool(
            bool(getattr(self.rl_cfg.env, "enable_action_intercell_guard", False))
            and running_guard_ratio_cfg is None
        )
        local_guard_ratio = float(getattr(self.rl_cfg.env, "action_intercell_guard_ratio_to_local_min", 1.25) or 1.25)
        local_guard_ratio = float(max(local_guard_ratio, 1.0))
        guard_local_threshold = None
        local_min_cost = float("inf")
        guard_selected_excess = 0.0
        guard_selected_cost_by_mode_packet: Dict[Tuple[int, int], float] = {}

        target_rel = float(1.0 - self.urllc_cfg.target_error_probability)

        def _is_safe_puncture_candidate(local_candidate: Optional[CandidatePacket]) -> bool:
            if local_candidate is None or not bool(getattr(local_candidate, "puncture_feasible", False)):
                return False
            puncture_rel = float(getattr(local_candidate, "puncture_reliability", 0.0) or 0.0)
            if puncture_rel < target_rel - 1.0e-9:
                return False
            if bool(getattr(local_candidate, "cause_embb_retention_below_threshold", False)):
                return False
            return True

        feasible_puncture_candidates = [candidate for candidate in candidates if _is_safe_puncture_candidate(candidate)]
        min_safe_puncture_intercell_cost = float("inf")
        if feasible_puncture_candidates:
            self.feasible_puncture_available_count += 1

        if (
            local_min_guard_enabled
            and uav_idx is not None
            and rb is not None
            and minislot is not None
            and candidates
        ):
            local_costs = []
            for candidate in candidates:
                packet_id = int(getattr(candidate, "packet_id", -1))
                if packet_id < 0:
                    continue
                if bool(getattr(candidate, "overlay_feasible", False)):
                    overlay_cost = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidate,
                            MODE_OVERLAY,
                        )
                    )
                    local_costs.append(overlay_cost)
                    key = (MODE_OVERLAY, packet_id)
                    prev = guard_selected_cost_by_mode_packet.get(key, float("inf"))
                    guard_selected_cost_by_mode_packet[key] = float(min(prev, overlay_cost))
                if bool(getattr(candidate, "puncture_feasible", False)):
                    puncture_cost = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidate,
                            MODE_PUNCTURE,
                        )
                    )
                    local_costs.append(puncture_cost)
                    key = (MODE_PUNCTURE, packet_id)
                    prev = guard_selected_cost_by_mode_packet.get(key, float("inf"))
                    guard_selected_cost_by_mode_packet[key] = float(min(prev, puncture_cost))
                    if candidate in feasible_puncture_candidates:
                        min_safe_puncture_intercell_cost = float(min(min_safe_puncture_intercell_cost, puncture_cost))
            finite_costs = [float(cost) for cost in local_costs if np.isfinite(float(cost))]
            if finite_costs:
                local_min_cost = float(min(finite_costs))
                guard_local_threshold = float(local_min_cost * local_guard_ratio + 1.0e-12)
                high_count = int(sum(1 for cost in finite_costs if float(cost) > guard_local_threshold))
                self.action_intercell_guard_candidate_total_count += int(len(finite_costs))
                self.action_intercell_guard_candidate_high_count += int(high_count)
                self.action_intercell_guard_local_min_cost_sum += float(local_min_cost)
                self.action_intercell_guard_local_min_cost_count += 1
                self.action_intercell_guard_total_cell_count += 1
                if high_count > 0:
                    self.action_intercell_guard_active_cell_count += 1

        self.phase_a_candidate_total += int(len(candidates))
        ov_feasible = sum(int(bool(getattr(c, "overlay_feasible", False))) for c in candidates)
        pu_feasible = sum(int(bool(getattr(c, "puncture_feasible", False))) for c in candidates)
        self.phase_a_feasible_overlay_candidate_total += int(ov_feasible)
        self.phase_a_feasible_puncture_candidate_total += int(pu_feasible)
        # Rejection-reason aggregation (counts candidates that flagged each cause).
        for c in candidates:
            if bool(getattr(c, "cause_cross_uav_interference_too_high", False)):
                self.phase_a_rejected_intercell_total += 1
            if bool(getattr(c, "cause_embb_retention_below_threshold", False)):
                self.phase_a_rejected_min_rate_total += 1
            if bool(getattr(c, "cause_required_power_exceeds_budget", False)):
                self.phase_a_rejected_power_guard_total += 1
            if bool(getattr(c, "cause_rb_minislot_collision", False)) or bool(getattr(c, "cause_packet_already_scheduled_elsewhere", False)):
                self.phase_a_rejected_collision_total += 1
            if bool(getattr(c, "cause_deadline_or_release_violation", False)):
                self.phase_a_rejected_deadline_total += 1
            if bool(getattr(c, "cause_other_structural_reason", False)):
                self.phase_a_rejected_other_total += 1
            if bool(getattr(c, "cause_gain_ratio_unqualified", False)):
                self.phase_a_rejected_other_gain_ratio_total += 1
            if bool(getattr(c, "cause_overlay_margin_blocked", False)):
                self.phase_a_rejected_other_overlay_margin_total += 1
            if bool(getattr(c, "cause_overlay_positive_gate_blocked", False)):
                self.phase_a_rejected_other_overlay_positive_gate_total += 1
            if bool(getattr(c, "cause_no_overlay_owner_available", False)):
                self.phase_a_rejected_other_no_overlay_owner_total += 1
            if bool(getattr(c, "cause_overlay_reliability_failed", False)):
                self.phase_a_rejected_other_overlay_reliability_total += 1
            if bool(getattr(c, "cause_overlay_sic_failed", False)):
                self.phase_a_rejected_other_overlay_sic_total += 1
        both_modes_feasible_available = any(
            bool(candidate.overlay_feasible) and bool(candidate.puncture_feasible)
            for candidate in observation.candidates
        )
        safe_puncture_available = any(
            self._candidate_supports_safe_puncture_anchor(candidate)
            for candidate in observation.candidates
        )
        mode_anchor_active = bool(safe_puncture_available)
        if both_modes_feasible_available:
            self.both_modes_feasible_count += 1
        if safe_puncture_available:
            self.safe_puncture_available_count += 1
            chosen_mode = int(shielded_action.action.mode)
            if chosen_mode == MODE_OVERLAY:
                self.overlay_chosen_when_safe_puncture_available_count += 1
            elif chosen_mode == MODE_PUNCTURE:
                self.puncture_chosen_when_safe_puncture_available_count += 1
        if mode_anchor_active:
            self.mode_anchor_active_count += 1
            if int(shielded_action.action.mode) == MODE_PUNCTURE:
                self.teacher_mode_agreement_count += 1
        phase_a_power_info = dict(getattr(shielded_action, "phase_a_embb_power_info", {}) or {})
        if self.rl_cfg.env.learn_embb_baseline and self._phase_a_embb_power_runtime_enabled():
            self.phase_a_embb_power_invalid_or_masked_count += int(bool(phase_a_power_info.get("invalid_or_masked", False)))
            # Suppress-reason accounting MUST be per executed decision (denominator=`phase_a_total_decisions`).
            # Do not increment these counters inside the sanitizer, because the sanitizer may be invoked more than
            # once per environment step (and that would create ratios > 1).
            reason = str(phase_a_power_info.get("zeroed_reason", "") or "").strip().lower()
            if reason:
                if reason == "inactive_head":
                    self.phase_a_embb_power_zeroed_inactive_head_count += 1
                elif reason in {"keep_mode_block", "keep_mode", "keep"}:
                    self.phase_a_embb_power_zeroed_keep_mode_count += 1
                    self.phase_a_zero_by_keep_due_to_mode_gate_count += 1
                elif reason == "no_candidate":
                    self.phase_a_embb_power_zeroed_no_candidate_count += 1
                elif reason == "no_embb_active":
                    self.phase_a_embb_power_zeroed_no_embb_active_count += 1
                elif reason == "no_owner":
                    self.phase_a_embb_power_zeroed_no_owner_count += 1
                    self.phase_a_power_write_blocked_no_owner_count += 1
                elif reason == "invalid_owner":
                    self.phase_a_embb_power_zeroed_invalid_owner_count += 1
                elif reason == "cap_projection":
                    self.phase_a_embb_power_zeroed_cap_projection_count += 1
                    self.phase_a_power_write_blocked_projection_count += 1
                elif reason == "floor_projection":
                    self.phase_a_embb_power_zeroed_floor_projection_count += 1
                    self.phase_a_power_write_blocked_projection_count += 1
                else:
                    self.phase_a_embb_power_zeroed_unknown_count += 1

        good_overlay_retention_threshold = float(
            getattr(self.rl_cfg.env, "good_overlay_retention_threshold", 0.85) or 0.85
        )
        good_overlay_intercell_ratio_to_local_min = float(
            getattr(self.rl_cfg.env, "good_overlay_intercell_ratio_to_local_min", 1.5) or 1.5
        )
        good_overlay_intercell_ratio_to_local_min = float(max(good_overlay_intercell_ratio_to_local_min, 1.0))
        good_overlay_candidates = []
        if np.isfinite(local_min_cost):
            for local_candidate in candidates:
                if not bool(getattr(local_candidate, "overlay_feasible", False)):
                    continue
                if float(getattr(local_candidate, "overlay_retention", 0.0) or 0.0) < good_overlay_retention_threshold - 1.0e-12:
                    continue
                local_packet_id = int(getattr(local_candidate, "packet_id", -1))
                overlay_cost = float(
                    guard_selected_cost_by_mode_packet.get((MODE_OVERLAY, local_packet_id), float("inf"))
                )
                if not np.isfinite(overlay_cost):
                    continue
                if overlay_cost <= float(local_min_cost) * good_overlay_intercell_ratio_to_local_min + 1.0e-12:
                    good_overlay_candidates.append((local_candidate, overlay_cost))
        good_overlay_available = bool(len(good_overlay_candidates) > 0)
        if good_overlay_available:
            self.mode_balance_good_overlay_available_count += 1

        if shielded_action.candidate is None:
            self.phase_a_selected_keep_total += 1
            if feasible_puncture_candidates:
                self.missed_feasible_puncture_count += 1
            feasible_candidates = [
                candidate for candidate in observation.candidates
                if bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible)
            ]
            if feasible_candidates:
                reward -= self.rl_cfg.reward.keep_feasible_penalty
                reward_terms["keep_feasible_penalty"] = -self.rl_cfg.reward.keep_feasible_penalty
                keep_urgent_weight = float(getattr(self.rl_cfg.reward, "keep_urgent_penalty_weight", 0.0))
                if keep_urgent_weight > 0.0:
                    max_urgency = max(self._candidate_urgency(candidate.packet_id, minislot) for candidate in feasible_candidates)
                    urgent_penalty = keep_urgent_weight * max_urgency
                    if urgent_penalty > 0.0:
                        reward -= urgent_penalty
                        reward_terms["keep_urgent_penalty"] = -urgent_penalty
            admission_band_bonus_weight = float(getattr(self.rl_cfg.reward, "admission_band_bonus_weight", 0.0) or 0.0)
            admission_band_penalty_weight = float(getattr(self.rl_cfg.reward, "admission_band_penalty_weight", 0.0) or 0.0)
            target_admission_mid = float(self._current_target_admission_mid(self._current_actual_load()))
            target_admission_tol = float(self._current_target_admission_tol(self._current_actual_load()))
            if (admission_band_bonus_weight > 0.0 or admission_band_penalty_weight > 0.0) and target_admission_mid > 0.0:
                current_scheduled_packets = int(np.count_nonzero(self.scheduled_uavs >= 0))
                projected_admission_ratio = float(current_scheduled_packets / max(self.num_packets, 1)) if self.num_packets > 0 else 1.0
                admission_gap = abs(projected_admission_ratio - target_admission_mid)
                if target_admission_tol > 1.0e-9 and admission_gap <= target_admission_tol + 1.0e-9:
                    band_score = 1.0 - float(admission_gap / max(target_admission_tol, 1.0e-9))
                    if admission_band_bonus_weight > 0.0 and band_score > 0.0:
                        reward += admission_band_bonus_weight * band_score
                        reward_terms["admission_band_bonus"] = reward_terms.get("admission_band_bonus", 0.0) + admission_band_bonus_weight * band_score
                elif admission_band_penalty_weight > 0.0:
                    excess_gap = max(admission_gap - target_admission_tol, 0.0)
                    penalty = admission_band_penalty_weight * excess_gap
                    reward -= penalty
                    reward_terms["admission_band_penalty"] = reward_terms.get("admission_band_penalty", 0.0) - penalty
            if shielded_action.used_greedy_fallback:
                reward -= self.rl_cfg.reward.invalid_action_penalty
                reward_terms["invalid_action_penalty"] = reward_terms.get("invalid_action_penalty", 0.0) - self.rl_cfg.reward.invalid_action_penalty
            if shielded_action.collision_rewritten:
                reward -= self.rl_cfg.reward.collision_rewrite_penalty
                reward_terms["collision_rewrite_penalty"] = reward_terms.get("collision_rewrite_penalty", 0.0) - self.rl_cfg.reward.collision_rewrite_penalty
            if shielded_action.joint_reliability_rewritten:
                reward -= self.rl_cfg.reward.invalid_action_penalty
                reward_terms["joint_reliability_rewrite_penalty"] = reward_terms.get("joint_reliability_rewrite_penalty", 0.0) - self.rl_cfg.reward.invalid_action_penalty
            if feasible_puncture_candidates:
                target_floor = float(getattr(self.rl_cfg.env, "admission_target_soft_floor", 0.0) or 0.0)
                current_scheduled_packets = int(np.count_nonzero(self.scheduled_uavs >= 0))
                current_admission_ratio = float(current_scheduled_packets / max(self.num_packets, 1)) if self.num_packets > 0 else 1.0
                penalty_weight = float(getattr(self.rl_cfg.reward, "missed_feasible_puncture_penalty_weight", 0.0) or 0.0)
                if penalty_weight > 0.0 and current_admission_ratio < target_floor - 1.0e-9:
                    reward -= penalty_weight
                    reward_terms["missed_feasible_puncture_penalty"] = (
                        reward_terms.get("missed_feasible_puncture_penalty", 0.0) - penalty_weight
                    )

            # KEEP does not admit a URLLC packet. We still write the execution grids for consistent
            # episode metrics. Optionally, Phase-A residual eMBB power repair may still write on KEEP.
            self.packet_grid[uav_idx, rb, minislot] = -1
            self.mode_grid[uav_idx, rb, minislot] = MODE_KEEP
            if getattr(self, "executed_local_puncture_mask", None) is not None:
                self.executed_local_puncture_mask[uav_idx, rb, minislot] = False
            embb_owner = -1
            if self.owner_per_uav_rb is not None:
                embb_owner = int(self.owner_per_uav_rb[uav_idx, rb])
            self.embb_owner_grid[uav_idx, rb, minislot] = int(embb_owner)
            if self.rl_cfg.env.learn_embb_baseline and bool(self._phase_a_embb_power_runtime_enabled()):
                allow_phase_a_power_on_keep = bool(getattr(self.rl_cfg.env, "allow_phase_a_power_on_keep", False))
                if allow_phase_a_power_on_keep:
                    self.phase_a_keep_power_write_attempt_count += 1
                    self.phase_a_power_write_on_keep_count += 1
                    power_info = dict(phase_a_power_info)
                    current_owner = int(embb_owner)
                    if current_owner < 0 or current_owner >= int(self.sys_cfg.num_embb_users):
                        shielded_action.action.embb_power_delta = 0.0
                    else:
                        previous_scale = float(self.embb_power_scale_grid[uav_idx, rb, minislot])
                        alpha = float(
                            power_info.get(
                                "residual_alpha",
                                self._phase_a_embb_power_residual_alpha(),
                            ) or self._phase_a_embb_power_residual_alpha()
                        )
                        clipped_delta = float(np.clip(
                            float(power_info.get("executed_delta", shielded_action.action.embb_power_delta)),
                            -1.0,
                            1.0,
                        ))
                        embb_scale = float(
                            power_info.get(
                                "executed_scale",
                                previous_scale * (1.0 + alpha * float(np.tanh(clipped_delta))),
                            )
                        )
                        if not bool(power_info.get("projection_disabled", False)):
                            eff_min = float(
                                power_info.get(
                                    "effective_scale_min",
                                    getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0),
                                )
                            )
                            eff_max = float(
                                power_info.get(
                                    "effective_scale_max",
                                    getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0),
                                )
                            )
                            embb_scale = float(np.clip(embb_scale, eff_min, eff_max))

                        self.embb_power_scale_grid[uav_idx, rb, minislot] = embb_scale
                        self.phase_a_embb_power_write_count += 1
                        self.phase_a_keep_power_write_success_count += 1
                        delta_scale = abs(embb_scale - previous_scale)
                        self.phase_a_embb_power_changed_count += int(delta_scale > 1e-3)
                        self.phase_a_embb_power_change_sum += float(delta_scale)

                        try:
                            if float(embb_scale) < float(previous_scale) - 1e-12:
                                per_rb_power = float(self.allocator._get_embb_per_rb_power(current_owner))
                                power_reduction = float(per_rb_power * (float(previous_scale) - float(embb_scale)))
                                self.phase_a_power_total_power_reduction_sum += float(max(power_reduction, 0.0))
                                embb_idx = int(self.sys_cfg.num_urllc_users + current_owner)
                                intercell_reduction = 0.0
                                for victim_uav in range(self.sys_cfg.num_uavs):
                                    if victim_uav == uav_idx:
                                        continue
                                    gain = float(self.channel_gains_mag_sq[embb_idx, victim_uav, rb])
                                    intercell_reduction += float(power_reduction * gain)
                                self.phase_a_power_intercell_reduction_sum += float(max(intercell_reduction, 0.0))
                        except Exception:
                            pass

                        self.phase_a_embb_power_projection_count += 1
                        self.phase_a_embb_power_delta_clipped_count += int(bool(power_info.get("delta_was_clipped", False)))
                        self.phase_a_embb_power_quantized_count += int(bool(power_info.get("used_discrete_bin", False)))
                        self.phase_a_embb_power_scale_clipped_count += int(bool(power_info.get("scale_was_clipped", False)))
                        self.phase_a_embb_power_cap_hit_count += int(bool(power_info.get("hit_upper_bound", False)))
                        self.phase_a_embb_power_floor_hit_count += int(bool(power_info.get("hit_lower_bound", False)))
                        self.phase_a_embb_power_raw_delta_sum += float(power_info.get("raw_delta", 0.0))
                        self.phase_a_embb_power_executed_delta_sum += float(power_info.get("executed_delta", 0.0))

                        raw_delta = float(power_info.get("raw_delta", 0.0))
                        clipped_delta_stage = float(power_info.get("clipped_delta", raw_delta))
                        quant_delta_stage = float(power_info.get("quantized_delta", clipped_delta_stage))
                        projection_delta_stage = float(power_info.get("executed_delta", quant_delta_stage))
                        if abs(alpha) > 1e-12 and abs(previous_scale) > 1e-12:
                            final_executed_delta = float(((embb_scale / previous_scale) - 1.0) / alpha)
                        else:
                            final_executed_delta = 0.0
                        final_executed_delta = float(np.clip(final_executed_delta, -1.0, 1.0))
                        requested_scale = float(
                            power_info.get(
                                "requested_scale",
                                power_info.get(
                                    "requested_total_scale",
                                    previous_scale * (1.0 + alpha * float(np.tanh(quant_delta_stage))),
                                ),
                            )
                        )
                        eff_min = float(power_info.get("effective_scale_min", getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0)))
                        eff_max = float(power_info.get("effective_scale_max", getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0)))
                        self.phase_a_embb_power_pre_clip_delta_sum += raw_delta
                        self.phase_a_embb_power_post_clip_delta_sum += clipped_delta_stage
                        self.phase_a_embb_power_post_quant_delta_sum += quant_delta_stage
                        self.phase_a_embb_power_post_projection_delta_sum += projection_delta_stage
                        self.phase_a_embb_power_post_owner_validation_delta_sum += projection_delta_stage
                        self.phase_a_embb_power_final_executed_delta_sum += final_executed_delta
                        self.phase_a_embb_power_mean_abs_raw_delta_sum += abs(raw_delta)
                        self.phase_a_embb_power_mean_abs_executed_delta_sum += abs(final_executed_delta)
                        self.phase_a_embb_power_raw_delta_sq_sum += float(raw_delta * raw_delta)
                        self.phase_a_embb_power_raw_saturation_count += int(abs(raw_delta) > 0.98)
                        self.phase_a_embb_power_final_delta_sq_sum += float(final_executed_delta * final_executed_delta)
                        self.phase_a_embb_power_executed_scale_sum += float(embb_scale)
                        self.phase_a_embb_power_executed_scale_sq_sum += float(embb_scale * embb_scale)
                        diff = float(raw_delta - final_executed_delta)
                        self.phase_a_embb_power_pre_vs_final_abs_diff_sum += abs(diff)
                        self.phase_a_embb_power_pre_vs_final_sq_diff_sum += float(diff * diff)
                        self.phase_a_embb_power_sign_flip_count += int(raw_delta * final_executed_delta < -1.0e-12)
                        raw_sign_denom = abs(raw_delta) > 1e-6
                        self.phase_a_embb_power_pre_vs_final_sign_consistent_denom += int(raw_sign_denom)
                        self.phase_a_embb_power_pre_vs_final_sign_consistent_count += int(
                            raw_sign_denom and (raw_delta * final_executed_delta >= -1.0e-12)
                        )
                        self.phase_a_embb_power_effective_nonzero_count += int(abs(final_executed_delta) > 1e-3)
                        self.phase_a_embb_power_floor_binding_strength_sum += float(max(eff_min - requested_scale, 0.0))
                        self.phase_a_embb_power_cap_binding_strength_sum += float(max(requested_scale - eff_max, 0.0))
                        proj_delta = float(projection_delta_stage - quant_delta_stage)
                        self.phase_a_embb_power_proj_delta_abs_sum += abs(proj_delta)
                        self.phase_a_embb_power_proj_delta_sq_sum += float(proj_delta * proj_delta)
                        if bool(power_info.get("hit_lower_bound", False)):
                            self.phase_a_embb_power_pre_to_floor_delta_sum += float(max(projection_delta_stage - quant_delta_stage, 0.0))
                        if bool(power_info.get("hit_upper_bound", False)):
                            self.phase_a_embb_power_pre_to_cap_delta_sum += float(max(quant_delta_stage - projection_delta_stage, 0.0))
                        self.phase_a_embb_power_final_minus_proj_abs_sum += float(abs(final_executed_delta - projection_delta_stage))
                else:
                    self.phase_a_power_zeroed_non_admission_count += 1
                    shielded_action.action.embb_power_delta = 0.0
            return {
                "mode": chosen_mode_name,
                "scheduled_packet": scheduled_packet,
                "reward": reward,
                "power": actual_power,
                "reward_terms": reward_terms,
                "greedy_reference_utility": observation.greedy_reference_utility,
                "used_greedy_fallback": float(shielded_action.used_greedy_fallback),
            }

        candidate = shielded_action.candidate
        mode = shielded_action.action.mode
        if int(mode) == MODE_OVERLAY:
            self.phase_a_selected_overlay_total += 1
        elif int(mode) == MODE_PUNCTURE:
            self.phase_a_selected_puncture_total += 1
        else:
            self.phase_a_selected_keep_total += 1
        scheduled_packet = candidate.packet_id
        reward, actual_power, reward_terms, projection_info = self._counterfactual_local_reward(
            candidate,
            mode,
            minislot=minislot,
            power_delta=shielded_action.action.power_delta,
        )
        mode_int = int(mode)
        packet_id = int(getattr(candidate, "packet_id", -1))
        selected_overlay_cost = float(
            guard_selected_cost_by_mode_packet.get((MODE_OVERLAY, packet_id), float("inf"))
        )
        selected_puncture_cost = float(
            guard_selected_cost_by_mode_packet.get((MODE_PUNCTURE, packet_id), float("inf"))
        )
        if mode_int == MODE_OVERLAY and np.isfinite(selected_overlay_cost):
            self.mode_balance_selected_overlay_cost_values.append(float(selected_overlay_cost))
        if mode_int == MODE_PUNCTURE and np.isfinite(selected_puncture_cost):
            self.mode_balance_selected_puncture_cost_values.append(float(selected_puncture_cost))

        # v6 mode-balance soft shaping: keep puncture recovery, but prevent puncture overuse when
        # a high-quality overlay candidate exists with acceptable intercell cost.
        if mode_int == MODE_OVERLAY and good_overlay_available:
            overlay_good_match = any(
                int(getattr(local_candidate, "packet_id", -1)) == packet_id
                for local_candidate, _overlay_cost in good_overlay_candidates
            )
            if overlay_good_match:
                self.mode_balance_overlay_chosen_when_good_count += 1
                overlay_bonus_weight = float(
                    getattr(self.rl_cfg.reward, "overlay_good_candidate_bonus_weight", 0.0) or 0.0
                )
                if overlay_bonus_weight > 0.0:
                    bonus = overlay_bonus_weight * float(
                        np.clip(getattr(candidate, "overlay_retention", 0.0), 0.0, 1.0)
                    )
                    reward += bonus
                    reward_terms["overlay_balance_bonus"] = (
                        reward_terms.get("overlay_balance_bonus", 0.0) + bonus
                    )
        if mode_int == MODE_PUNCTURE and good_overlay_available:
            self.mode_balance_puncture_chosen_when_good_overlay_count += 1
            puncture_cost_for_compare = float(selected_puncture_cost)
            if not np.isfinite(puncture_cost_for_compare):
                try:
                    puncture_cost_for_compare = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidate,
                            MODE_PUNCTURE,
                        )
                    )
                except Exception:
                    puncture_cost_for_compare = float("inf")
            has_strong_good_overlay = False
            if np.isfinite(puncture_cost_for_compare):
                for local_candidate, overlay_cost in good_overlay_candidates:
                    if float(getattr(local_candidate, "overlay_retention", 0.0) or 0.0) < 0.9 - 1.0e-12:
                        continue
                    if float(overlay_cost) <= puncture_cost_for_compare * 1.25 + 1.0e-12:
                        has_strong_good_overlay = True
                        break
            if has_strong_good_overlay:
                penalty_weight = float(
                    getattr(
                        self.rl_cfg.reward,
                        "puncture_when_good_overlay_available_penalty_weight",
                        0.0,
                    ) or 0.0
                )
                if penalty_weight > 0.0:
                    reward -= penalty_weight
                    reward_terms["puncture_when_good_overlay_available_penalty"] = (
                        reward_terms.get("puncture_when_good_overlay_available_penalty", 0.0)
                        - penalty_weight
                    )

        if feasible_puncture_candidates and mode_int != MODE_PUNCTURE:
            self.missed_feasible_puncture_count += 1
        if feasible_puncture_candidates and mode_int == MODE_PUNCTURE:
            self.puncture_chosen_when_feasible_count += 1

        # v5: local-min intercell guard soft pressure (selected-action excess vs decision-local reference).
        if local_min_guard_enabled and guard_local_threshold is not None and mode_int in {MODE_OVERLAY, MODE_PUNCTURE}:
            selected_cost = float(
                guard_selected_cost_by_mode_packet.get((mode_int, packet_id), float("inf"))
            )
            if np.isfinite(selected_cost):
                guard_selected_excess = float(max(selected_cost - float(guard_local_threshold), 0.0))
                self.action_intercell_guard_selected_excess_sum += float(guard_selected_excess)
                self.action_intercell_guard_selected_excess_count += 1
                if guard_selected_excess > 0.0:
                    self.action_intercell_guard_selected_violation_count += 1

        # v5 puncture recovery shaping (conservative).
        if mode_int == MODE_PUNCTURE:
            if _is_safe_puncture_candidate(candidate):
                safe_bonus_weight = float(getattr(self.rl_cfg.reward, "safe_puncture_bonus_weight", 0.0) or 0.0)
                if safe_bonus_weight > 0.0:
                    reward += safe_bonus_weight
                    reward_terms["safe_puncture_bonus"] = (
                        reward_terms.get("safe_puncture_bonus", 0.0) + safe_bonus_weight
                    )
        elif mode_int == MODE_OVERLAY and feasible_puncture_candidates:
            selected_overlay_cost = float(
                guard_selected_cost_by_mode_packet.get((MODE_OVERLAY, packet_id), float("inf"))
            )
            if not np.isfinite(selected_overlay_cost):
                try:
                    selected_overlay_cost = float(
                        self._estimate_candidate_action_intercell_cost_after_source_mask(
                            int(uav_idx),
                            int(rb),
                            int(minislot),
                            candidate,
                            MODE_OVERLAY,
                        )
                    )
                except Exception:
                    selected_overlay_cost = float("inf")
            if np.isfinite(selected_overlay_cost) and np.isfinite(min_safe_puncture_intercell_cost):
                if selected_overlay_cost > float(min_safe_puncture_intercell_cost) * 1.25 + 1.0e-12:
                    self.overlay_chosen_when_lower_intercell_puncture_available_count += 1
                    overlay_penalty_weight = float(
                        getattr(
                            self.rl_cfg.reward,
                            "overlay_when_lower_intercell_puncture_available_penalty_weight",
                            getattr(self.rl_cfg.reward, "overlay_when_safe_puncture_penalty_weight", 0.0),
                        ) or 0.0
                    )
                    if overlay_penalty_weight > 0.0:
                        reward -= overlay_penalty_weight
                        reward_terms["overlay_when_lower_intercell_puncture_available_penalty"] = (
                            reward_terms.get("overlay_when_lower_intercell_puncture_available_penalty", 0.0)
                            - overlay_penalty_weight
                        )

        self.urllc_power_projection_count += 1
        self.urllc_power_delta_clipped_count += int(bool(projection_info.get("delta_was_clipped", False)))
        self.urllc_power_quantized_count += int(bool(projection_info.get("used_discrete_bin", False)))
        self.urllc_power_cap_hit_count += int(bool(projection_info.get("hit_upper_bound", False)))
        self.urllc_power_floor_hit_count += int(bool(projection_info.get("hit_feasible_floor", False)))
        self.urllc_raw_power_delta_sum += float(projection_info.get("raw_delta", 0.0))
        self.urllc_executed_power_delta_sum += float(projection_info.get("executed_delta", 0.0))
        if shielded_action.used_greedy_fallback:
            reward -= self.rl_cfg.reward.invalid_action_penalty
            reward_terms["invalid_action_penalty"] = reward_terms.get("invalid_action_penalty", 0.0) - self.rl_cfg.reward.invalid_action_penalty
        if shielded_action.collision_rewritten:
            reward -= self.rl_cfg.reward.collision_rewrite_penalty
            reward_terms["collision_rewrite_penalty"] = reward_terms.get("collision_rewrite_penalty", 0.0) - self.rl_cfg.reward.collision_rewrite_penalty
        if shielded_action.joint_reliability_rewritten:
            reward -= self.rl_cfg.reward.invalid_action_penalty
            reward_terms["joint_reliability_rewrite_penalty"] = reward_terms.get("joint_reliability_rewrite_penalty", 0.0) - self.rl_cfg.reward.invalid_action_penalty

        self.packet_grid[uav_idx, rb, minislot] = candidate.packet_id
        self.mode_grid[uav_idx, rb, minislot] = mode
        is_executed_puncture = bool(int(mode) == MODE_PUNCTURE)
        if getattr(self, "executed_local_puncture_mask", None) is not None:
            self.executed_local_puncture_mask[uav_idx, rb, minislot] = bool(is_executed_puncture)
        if is_executed_puncture:
            self.executed_puncture_action_count += 1
        self.embb_owner_grid[uav_idx, rb, minislot] = (
            -1 if mode == MODE_PUNCTURE else int(candidate.embb_owner_for_mode(mode))
        )

        # Selected-action intercell interference diagnostics.
        # Note: `CandidatePacket.base_embb_intercell_power` is computed during candidate evaluation as the
        # intercell NOMA interference term for the selected cell. We accumulate it at execution time for
        # fast report plots (overlay/puncture split).
        try:
            intercell_w = float(getattr(candidate, "base_embb_intercell_power", 0.0) or 0.0)
        except Exception:
            intercell_w = 0.0
        self.selected_intercell_interference_sum += intercell_w
        self.selected_intercell_interference_count += 1
        self.selected_intercell_interference_nonzero_count += int(intercell_w > 0.0)
        if int(mode) == MODE_OVERLAY:
            self.selected_overlay_intercell_interference_sum += intercell_w
            self.selected_overlay_intercell_interference_count += 1
        elif int(mode) == MODE_PUNCTURE:
            self.selected_puncture_intercell_interference_sum += intercell_w
            self.selected_puncture_intercell_interference_count += 1
        # Phase-A eMBB power repair: only write on *admitted* URLLC actions (overlay/puncture).
        if (
            self.rl_cfg.env.learn_embb_baseline
            and bool(self._phase_a_embb_power_runtime_enabled())
            and mode_int in {MODE_OVERLAY, MODE_PUNCTURE}
        ):
            self.phase_a_power_write_on_admission_count += 1
            power_info = dict(phase_a_power_info)
            previous_scale = float(self.embb_power_scale_grid[uav_idx, rb, minislot])
            alpha = float(power_info.get("residual_alpha", self._phase_a_embb_power_residual_alpha()) or self._phase_a_embb_power_residual_alpha())
            clipped_delta = float(np.clip(
                float(power_info.get("executed_delta", shielded_action.action.embb_power_delta)),
                -1.0,
                1.0,
            ))
            embb_scale = float(
                power_info.get("executed_scale", previous_scale * (1.0 + alpha * float(np.tanh(clipped_delta))))
            )
            if not bool(power_info.get("projection_disabled", False)):
                eff_min = float(power_info.get("effective_scale_min", getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0)))
                eff_max = float(power_info.get("effective_scale_max", getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0)))
                embb_scale = float(np.clip(embb_scale, eff_min, eff_max))

            self.embb_power_scale_grid[uav_idx, rb, minislot] = embb_scale
            self.phase_a_embb_power_write_count += 1
            delta_scale = abs(embb_scale - previous_scale)
            self.phase_a_embb_power_change_sum += float(delta_scale)
            self.phase_a_embb_power_changed_count += int(delta_scale > 1e-3)

            # Negative-only repair diagnostics: quantify actual power + intercell reduction when we downscale.
            try:
                if float(embb_scale) < float(previous_scale) - 1e-12:
                    owner = int(self.owner_per_uav_rb[uav_idx, rb]) if self.owner_per_uav_rb is not None else -1
                    if 0 <= owner < int(self.sys_cfg.num_embb_users):
                        per_rb_power = float(self.allocator._get_embb_per_rb_power(owner))
                        power_reduction = float(per_rb_power * (float(previous_scale) - float(embb_scale)))
                        self.phase_a_power_total_power_reduction_sum += float(max(power_reduction, 0.0))
                        embb_idx = int(self.sys_cfg.num_urllc_users + owner)
                        intercell_reduction = 0.0
                        for victim_uav in range(self.sys_cfg.num_uavs):
                            if victim_uav == uav_idx:
                                continue
                            gain = float(self.channel_gains_mag_sq[embb_idx, victim_uav, rb])
                            intercell_reduction += float(power_reduction * gain)
                        self.phase_a_power_intercell_reduction_sum += float(max(intercell_reduction, 0.0))
            except Exception:
                pass

            self.phase_a_embb_power_projection_count += 1
            self.phase_a_embb_power_delta_clipped_count += int(bool(power_info.get("delta_was_clipped", False)))
            self.phase_a_embb_power_quantized_count += int(bool(power_info.get("used_discrete_bin", False)))
            self.phase_a_embb_power_scale_clipped_count += int(bool(power_info.get("scale_was_clipped", False)))
            self.phase_a_embb_power_cap_hit_count += int(bool(power_info.get("hit_upper_bound", False)))
            self.phase_a_embb_power_floor_hit_count += int(bool(power_info.get("hit_lower_bound", False)))

            self.phase_a_embb_power_raw_delta_sum += float(power_info.get("raw_delta", 0.0))
            self.phase_a_embb_power_executed_delta_sum += float(power_info.get("executed_delta", 0.0))

            raw_delta = float(power_info.get("raw_delta", 0.0))
            clipped_delta_stage = float(power_info.get("clipped_delta", raw_delta))
            quant_delta_stage = float(power_info.get("quantized_delta", clipped_delta_stage))
            projection_delta_stage = float(power_info.get("executed_delta", quant_delta_stage))

            if abs(alpha) > 1e-12 and abs(previous_scale) > 1e-12:
                final_executed_delta = float(((embb_scale / previous_scale) - 1.0) / alpha)
            else:
                final_executed_delta = 0.0
            final_executed_delta = float(np.clip(final_executed_delta, -1.0, 1.0))

            requested_scale = float(
                power_info.get(
                    "requested_scale",
                    power_info.get("requested_total_scale", previous_scale * (1.0 + alpha * float(np.tanh(quant_delta_stage)))),
                )
            )
            eff_min = float(power_info.get("effective_scale_min", getattr(self.rl_cfg.env, "embb_power_scale_min", 0.0)))
            eff_max = float(power_info.get("effective_scale_max", getattr(self.rl_cfg.env, "embb_power_scale_max", 0.0)))

            self.phase_a_embb_power_pre_clip_delta_sum += raw_delta
            self.phase_a_embb_power_post_clip_delta_sum += clipped_delta_stage
            self.phase_a_embb_power_post_quant_delta_sum += quant_delta_stage
            self.phase_a_embb_power_post_projection_delta_sum += projection_delta_stage
            self.phase_a_embb_power_post_owner_validation_delta_sum += projection_delta_stage
            self.phase_a_embb_power_final_executed_delta_sum += final_executed_delta
            self.phase_a_embb_power_mean_abs_raw_delta_sum += abs(raw_delta)
            self.phase_a_embb_power_mean_abs_executed_delta_sum += abs(final_executed_delta)
            self.phase_a_embb_power_raw_delta_sq_sum += float(raw_delta * raw_delta)
            self.phase_a_embb_power_raw_saturation_count += int(abs(raw_delta) > 0.98)
            self.phase_a_embb_power_final_delta_sq_sum += float(final_executed_delta * final_executed_delta)
            self.phase_a_embb_power_executed_scale_sum += float(embb_scale)
            self.phase_a_embb_power_executed_scale_sq_sum += float(embb_scale * embb_scale)

            diff = float(raw_delta - final_executed_delta)
            self.phase_a_embb_power_pre_vs_final_abs_diff_sum += abs(diff)
            self.phase_a_embb_power_pre_vs_final_sq_diff_sum += float(diff * diff)
            self.phase_a_embb_power_sign_flip_count += int(raw_delta * final_executed_delta < -1.0e-12)
            raw_sign_denom = abs(raw_delta) > 1e-6
            self.phase_a_embb_power_pre_vs_final_sign_consistent_denom += int(raw_sign_denom)
            self.phase_a_embb_power_pre_vs_final_sign_consistent_count += int(
                raw_sign_denom and (raw_delta * final_executed_delta >= -1.0e-12)
            )
            self.phase_a_embb_power_effective_nonzero_count += int(abs(final_executed_delta) > 1e-3)

            self.phase_a_embb_power_floor_binding_strength_sum += float(max(eff_min - requested_scale, 0.0))
            self.phase_a_embb_power_cap_binding_strength_sum += float(max(requested_scale - eff_max, 0.0))

            proj_delta = float(projection_delta_stage - quant_delta_stage)
            self.phase_a_embb_power_proj_delta_abs_sum += abs(proj_delta)
            self.phase_a_embb_power_proj_delta_sq_sum += float(proj_delta * proj_delta)
            if bool(power_info.get("hit_lower_bound", False)):
                self.phase_a_embb_power_pre_to_floor_delta_sum += float(max(projection_delta_stage - quant_delta_stage, 0.0))
            if bool(power_info.get("hit_upper_bound", False)):
                self.phase_a_embb_power_pre_to_cap_delta_sum += float(max(quant_delta_stage - projection_delta_stage, 0.0))
            self.phase_a_embb_power_final_minus_proj_abs_sum += float(abs(final_executed_delta - projection_delta_stage))
        self.scheduled_power[candidate.packet_id, uav_idx] = actual_power
        self.scheduled_uavs[candidate.packet_id] = uav_idx
        self.scheduled_reliabilities[candidate.packet_id] = float(
            self._last_joint_reliabilities.get(agent_id, candidate.reliability_for_mode(mode))
        )
        self.unscheduled_packet_ids.discard(candidate.packet_id)
        # NOTE:
        # greedy_urllc_budget_used_bps is an eMBB-loss budget tracker (global per-episode cap),
        # not a URLLC-throughput tracker. It must only be updated by the selected action's
        # projected eMBB loss in hard_feasible_throughput_greedy_action().
        #
        # Do not accumulate packet-throughput units here; that mixes units and effectively
        # breaks the intended "total eMBB loss share" constraint.
        self.scheduled_counts[uav_idx] += 1
        if mode == MODE_OVERLAY:
            self.overlay_counts[uav_idx] += 1
            self.selected_overlay_retentions.append(float(candidate.overlay_retention))
            self.selected_overlay_losses.append(float(candidate.overlay_loss))
            self.overlay_selected_pairs += 1
            self.overlay_success_ema[uav_idx] = 0.9 * self.overlay_success_ema[uav_idx] + 0.1 * float(candidate.overlay_retention)
            self.selected_overlay_admission_count += 1
        elif mode == MODE_PUNCTURE:
            self.puncture_counts[uav_idx] += 1
            self.selected_puncture_losses.append(float(candidate.puncture_loss))
            self.selected_puncture_admission_count += 1
            puncture_admission_bonus_weight = float(getattr(self.rl_cfg.reward, "puncture_admission_bonus_weight", 0.0))
            if puncture_admission_bonus_weight > 0.0:
                reward += puncture_admission_bonus_weight
                reward_terms["puncture_admission_bonus"] = (
                    reward_terms.get("puncture_admission_bonus", 0.0) + puncture_admission_bonus_weight
                )
            loss_norm = float(candidate.puncture_loss / 1.0e6)
            self.puncture_loss_ema[uav_idx] = 0.9 * self.puncture_loss_ema[uav_idx] + 0.1 * loss_norm

        # Action-level intercell cost (victim incoming-interference delta caused by this action).
        # Keep BOTH variants:
        # - before_source_mask: legacy (punctured other-cell eMBB still counted as sources) for debug.
        # - after_source_mask: corrected (punctured other-cell eMBB excluded from sources) used for reward/plots.
        inter_norm = float(getattr(self.rl_cfg.reward, "terminal_intercell_penalty_normalizer", 1.0e-7) or 1.0e-7)
        delta_victim_intercell_after_mask = 0.0
        delta_victim_intercell_before_mask = 0.0
        if pre_victim_intercell_after_mask is not None and len(pre_victim_intercell_after_mask) > 0:
            try:
                post_after = [
                    float(self._compute_intercell_interference(other_uav, rb, minislot, apply_embb_source_mask=True))
                    for other_uav in range(self.sys_cfg.num_uavs)
                    if other_uav != uav_idx
                ]
                delta_victim_intercell_after_mask = float(
                    sum(max(p - q, 0.0) for p, q in zip(post_after, pre_victim_intercell_after_mask)) / max(len(post_after), 1)
                )
            except Exception:
                delta_victim_intercell_after_mask = 0.0
        if pre_victim_intercell_before_mask is not None and len(pre_victim_intercell_before_mask) > 0:
            try:
                post_before = [
                    float(self._compute_intercell_interference(other_uav, rb, minislot, apply_embb_source_mask=False))
                    for other_uav in range(self.sys_cfg.num_uavs)
                    if other_uav != uav_idx
                ]
                delta_victim_intercell_before_mask = float(
                    sum(max(p - q, 0.0) for p, q in zip(post_before, pre_victim_intercell_before_mask)) / max(len(post_before), 1)
                )
            except Exception:
                delta_victim_intercell_before_mask = 0.0

        self.selected_action_intercell_cost_before_source_mask_values.append(float(delta_victim_intercell_before_mask))
        self.selected_action_intercell_cost_after_source_mask_values.append(float(delta_victim_intercell_after_mask))
        # Backward-compat key: keep selected_action_intercell_cost_* mapped to the corrected (after-mask) semantics.
        self.selected_action_intercell_cost_values.append(float(delta_victim_intercell_after_mask))
        if int(scheduled_packet) >= 0:
            self.selected_action_intercell_cost_after_source_mask_admit_sum += float(delta_victim_intercell_after_mask)
            self.selected_action_intercell_cost_after_source_mask_admit_count += 1
            if bool(getattr(self.rl_cfg.env, "enable_action_intercell_guard", False)) and np.isfinite(delta_victim_intercell_after_mask):
                self.action_intercell_guard_running_min = float(min(
                    float(self.action_intercell_guard_running_min),
                    float(delta_victim_intercell_after_mask),
                ))

        self.step_intercell_penalty_count += 1
        if delta_victim_intercell_after_mask > 0.0:
            self.step_intercell_penalty_active_count += 1
        # Optional step-level penalty (action selection should see this signal).
        if step_intercell_w > 0.0:
            penalty = float(step_intercell_w * float(delta_victim_intercell_after_mask) / max(inter_norm, 1.0e-12))
            if penalty > 0.0:
                reward -= penalty
                reward_terms["step_intercell_victim_delta_penalty"] = reward_terms.get("step_intercell_victim_delta_penalty", 0.0) - penalty
            self.step_intercell_penalty_sum += float(penalty)

        # Optional action-level intercell penalty (explicitly aligned with selected_action_intercell_cost_after_source_mask).
        action_intercell_w = float(getattr(self.rl_cfg.reward, "step_action_intercell_penalty_weight", 0.0) or 0.0)
        if action_intercell_w > 0.0:
            normalizer = float(getattr(self.rl_cfg.reward, "step_action_intercell_penalty_normalizer", 1.0e-10) or 1.0e-10)
            if normalizer <= 0.0:
                normalizer = float(inter_norm)
            if local_min_guard_enabled and guard_local_threshold is not None:
                penalty_base = float(max(guard_selected_excess, 0.0))
            else:
                penalty_base = float(delta_victim_intercell_after_mask)
            penalty = float(action_intercell_w * penalty_base / max(normalizer, 1.0e-12))
            if penalty > 0.0:
                reward -= penalty
                reward_terms["step_action_intercell_penalty"] = reward_terms.get("step_action_intercell_penalty", 0.0) - penalty

        # Interference-aware admission shaping: reward admissions that are reliable AND low-interference.
        admitted_packet = int(int(scheduled_packet) >= 0)
        reliability_ok = 0
        if admitted_packet > 0 and shielded_action.candidate is not None:
            try:
                target_rel = float(1.0 - self.urllc_cfg.target_error_probability)
            except Exception:
                target_rel = 0.0
            try:
                rel = float(shielded_action.candidate.reliability_for_mode(int(shielded_action.action.mode)))
            except Exception:
                rel = float(target_rel)
            reliability_ok = int(rel >= target_rel - 1.0e-9) if target_rel > 0.0 else 1

        low_bonus_w = float(getattr(self.rl_cfg.reward, "low_interference_admission_bonus_weight", 0.0) or 0.0)
        if low_bonus_w > 0.0 and admitted_packet > 0 and reliability_ok > 0:
            scale = float(getattr(self.rl_cfg.reward, "low_interference_admission_intercell_scale", 0.0) or 0.0)
            if scale <= 0.0:
                # Running normalization (avoid hard-coded absolute budget).
                scale = float(max(self.high_intercell_admission_budget_ema or 0.0, inter_norm, 1.0e-12))
            bonus = float(low_bonus_w * float(np.exp(-float(delta_victim_intercell_after_mask) / max(scale, 1.0e-12))))
            if bonus > 0.0:
                reward += bonus
                reward_terms["low_interference_admission_bonus"] = reward_terms.get("low_interference_admission_bonus", 0.0) + bonus
                self.low_interference_admission_bonus_sum += float(bonus)

        high_pen_w = float(getattr(self.rl_cfg.reward, "high_intercell_admission_penalty_weight", 0.0) or 0.0)
        if high_pen_w > 0.0 and admitted_packet > 0:
            beta = float(getattr(self.rl_cfg.reward, "high_intercell_admission_budget_ema_beta", 0.95) or 0.95)
            beta = float(np.clip(beta, 0.0, 0.9999))
            budget = self.high_intercell_admission_budget_ema
            if budget is None or not np.isfinite(float(budget)):
                budget = float(delta_victim_intercell_after_mask)
            budget = float(budget)
            excess = float(max(float(delta_victim_intercell_after_mask) - budget, 0.0))
            penalty = float(high_pen_w * excess)
            if penalty > 0.0:
                reward -= penalty
                reward_terms["high_intercell_admission_penalty"] = reward_terms.get("high_intercell_admission_penalty", 0.0) - penalty
                self.high_intercell_admission_penalty_sum += float(penalty)
            # Update running baseline after applying this decision (avoid look-ahead).
            self.high_intercell_admission_budget_ema = float(beta * budget + (1.0 - beta) * float(delta_victim_intercell_after_mask))

        return {
            "mode": chosen_mode_name,
            "scheduled_packet": scheduled_packet,
            "reward": reward,
            "power": actual_power,
            "reward_terms": reward_terms,
            "greedy_reference_utility": observation.greedy_reference_utility,
            "used_greedy_fallback": float(shielded_action.used_greedy_fallback),
        }

    def _candidate_urgency(self, packet_id: int, minislot: int) -> float:
        release = int(self.packet_release_minislots[packet_id]) if packet_id < self.packet_release_minislots.size else 0
        age = float(max(minislot - release, 0))
        max_latency = float(getattr(self.urllc_cfg, "max_latency_minislots", self.sys_cfg.num_minislots))
        return float(np.clip((age + 1.0) / max(max_latency, 1.0), 0.0, 1.0))

    def _current_actual_load(self) -> float:
        return float((self.sys_cfg.num_embb_users + self.sys_cfg.num_urllc_users) / max(self.sys_cfg.num_uavs, 1))

    def _fixed_share_reference_rate_bps_for_current_load(self) -> float:
        mapping = getattr(self.rl_cfg.env, "greedy_share_reference_pre_mbps_by_load", {}) or {}
        if not isinstance(mapping, dict) or not mapping:
            return 0.0
        try:
            actual = float(self._current_actual_load())
            keys = [float(k) for k in mapping.keys()]
            nearest = min(keys, key=lambda k: abs(k - actual))
            mbps = float(mapping.get(nearest, 0.0))
            if mbps <= 0.0:
                return 0.0
            return float(mbps * 1.0e6)
        except Exception:
            return 0.0

    def _get_load_aware_reward_weights(self, actual_load: Optional[float] = None) -> Dict[str, float]:
        weights = {
            "terminal_embb_rate_weight": float(self.rl_cfg.reward.terminal_embb_rate_weight),
            "terminal_urllc_admission_weight": float(self.rl_cfg.reward.terminal_urllc_admission_weight),
            "terminal_urllc_admission_target": float(self.rl_cfg.reward.terminal_urllc_admission_target),
            "terminal_unscheduled_penalty": float(self.rl_cfg.reward.terminal_unscheduled_penalty),
            "puncture_extra_penalty": float(self.rl_cfg.reward.puncture_extra_penalty),
            "missed_overlay_penalty": float(self.rl_cfg.reward.missed_overlay_penalty),
            "overlay_gain_weight": float(self.rl_cfg.reward.overlay_gain_weight),
        }
        if not bool(getattr(self.rl_cfg.training, "load_aware_objective", False)):
            return weights
        schedule = load_aware_reward_schedule(
            self._current_actual_load() if actual_load is None else float(actual_load)
        )
        weights.update(schedule)
        return weights


    def _current_puncture_loss_ceiling(self, actual_load: Optional[float] = None) -> float:
        ceiling = puncture_loss_ceiling_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "puncture_loss_ceiling_by_load", {}),
            fallback=float("inf"),
        )
        # Accept both Mbps-style ceilings (e.g. 0.55) and raw-rate ceilings
        # (e.g. 0.80e6) in experiment presets. Candidate puncture loss is compared
        # in Mbps below, so convert large raw-rate ceilings to Mbps here.
        if np.isfinite(ceiling) and ceiling > 1.0e3:
            return float(ceiling / 1.0e6)
        return float(ceiling)

    def _current_overlay_retention_gate(self, actual_load: Optional[float] = None) -> float:
        return overlay_retention_gate_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "overlay_retention_gate_by_load", {}),
            fallback=0.0,
        )

    def _load_bucket_value(
        self,
        mapping: Optional[Dict[float, float]],
        fallback: float,
        actual_load: Optional[float] = None,
    ) -> float:
        if not mapping:
            return float(fallback)
        normalized = {float(key): float(value) for key, value in dict(mapping).items()}
        bucket = nearest_reference_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            normalized.keys(),
        )
        return float(normalized.get(bucket, fallback))

    def _current_load_adaptive_puncture_floor(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.reward, "load_adaptive_puncture_floor_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _current_load_adaptive_overlay_ceiling(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.reward, "load_adaptive_overlay_ceiling_by_load", {}),
            fallback=1.0,
            actual_load=actual_load,
        )

    def _current_target_admission_mid(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.reward, "target_admission_mid_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _current_target_admission_tol(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.reward, "target_admission_tol_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _current_frontier_puncture_floor(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.training, "frontier_puncture_floor_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _current_frontier_overlay_ceiling(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.training, "frontier_overlay_ceiling_by_load", {}),
            fallback=1.0,
            actual_load=actual_load,
        )

    def _current_frontier_oracle_admission_floor(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.training, "frontier_oracle_admission_floor_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _current_target_quota_ratio(self, actual_load: Optional[float] = None) -> float:
        actual = self._current_actual_load() if actual_load is None else float(actual_load)
        frontier_floor = float(self._current_frontier_oracle_admission_floor(actual))
        if frontier_floor > 0.0:
            return float(np.clip(frontier_floor, 0.0, 1.0))
        target_mid = float(self._current_target_admission_mid(actual))
        if target_mid > 0.0:
            return float(np.clip(target_mid, 0.0, 1.0))
        return 0.0

    def _hard_mode_anchor_stage_active(self) -> bool:
        end_frac = float(np.clip(getattr(self.rl_cfg.training, "early_mode_anchor_end_frac", 0.0) or 0.0, 0.0, 1.0))
        if end_frac <= 1.0e-9:
            return False
        progress = float(np.clip(getattr(self, "training_progress_frac", 1.0), 0.0, 1.0))
        return bool(progress < end_frac - 1.0e-9)

    def _hard_mode_anchor_strength(self) -> float:
        end_frac = float(np.clip(getattr(self.rl_cfg.training, "early_mode_anchor_end_frac", 0.0) or 0.0, 0.0, 1.0))
        if end_frac <= 1.0e-9:
            return 0.0
        progress = float(np.clip(getattr(self, "training_progress_frac", 1.0), 0.0, 1.0))
        if progress < end_frac - 1.0e-9:
            return 1.0
        restore_end = min(2.0 * end_frac, 1.0)
        if progress >= restore_end - 1.0e-9:
            return 0.0
        restore_progress = float((progress - end_frac) / max(restore_end - end_frac, 1.0e-9))
        return float(np.clip(1.0 - restore_progress, 0.0, 1.0))

    def _candidate_supports_safe_puncture_anchor(self, candidate: Optional[CandidatePacket]) -> bool:
        if candidate is None:
            return False
        if not (bool(candidate.overlay_feasible) and bool(candidate.puncture_feasible)):
            return False
        puncture_reliability_target = float(1.0 - self.urllc_cfg.target_error_probability)
        if float(candidate.puncture_reliability) < puncture_reliability_target - 1.0e-9:
            return False
        loss_threshold = float(
            getattr(self.rl_cfg.training, "mode_anchor_safe_puncture_loss_threshold", 0.0) or 0.0
        )
        if loss_threshold > 0.0 and float(candidate.puncture_loss) > loss_threshold + 1.0e-9:
            return False
        overlay_margin_override = float(
            getattr(self.rl_cfg.training, "mode_anchor_overlay_margin_override", 0.0) or 0.0
        )
        if float(candidate.overlay_utility) > float(candidate.puncture_utility) + overlay_margin_override + 1.0e-9:
            return False
        return True

    def _phase_a_progress_summary(self, minislot: int) -> Dict[str, float]:
        del minislot
        total_packets = max(int(self.num_packets), 1)
        total_cells = max(int(len(self._cell_schedule)), 1)
        admitted_so_far = int(np.count_nonzero(self.scheduled_uavs >= 0))
        remaining_active_packets = max(int(len(self.unscheduled_packet_ids)), 0)
        remaining_cells = max(int(len(self._cell_schedule) - self.current_cell_index), 1)
        target_quota_ratio = float(self._current_target_quota_ratio(self._current_actual_load()))
        target_quota_packets = int(np.ceil(target_quota_ratio * total_packets - 1.0e-9)) if target_quota_ratio > 0.0 else 0
        target_quota_packets = int(np.clip(target_quota_packets, 0, total_packets))
        quota_gap_packets = max(target_quota_packets - admitted_so_far, 0)
        quota_gap_normalized = float(quota_gap_packets / total_packets)
        required_admit_rate_per_remaining_cell = float(np.clip(quota_gap_packets / max(remaining_cells, 1), 0.0, 1.0))
        overlay_count = int(np.sum(self.mode_grid == MODE_OVERLAY))
        puncture_count = int(np.sum(self.mode_grid == MODE_PUNCTURE))
        reference_embb_rate = float(self._reference_embb_total_rate_for_local_shaping())
        cumulative_embb_loss = float(np.sum(self.selected_overlay_losses) + np.sum(self.selected_puncture_losses))
        cumulative_embb_rate = max(reference_embb_rate - cumulative_embb_loss, 0.0)
        return {
            "target_quota_packets": float(target_quota_packets),
            "target_quota_normalized": float(target_quota_ratio),
            "admitted_so_far_packets": float(admitted_so_far),
            "admitted_so_far_normalized": float(admitted_so_far / total_packets),
            "remaining_active_packets": float(remaining_active_packets),
            "remaining_active_packets_normalized": float(remaining_active_packets / total_packets),
            "quota_gap_packets": float(quota_gap_packets),
            "quota_gap_normalized": quota_gap_normalized,
            "remaining_cells": float(remaining_cells),
            "remaining_cells_normalized": float(remaining_cells / total_cells),
            "required_admit_rate_per_remaining_cell": required_admit_rate_per_remaining_cell,
            "overlay_count_so_far_normalized": float(overlay_count / total_cells),
            "puncture_count_so_far_normalized": float(puncture_count / total_cells),
            "cumulative_embb_rate_so_far_normalized": float(cumulative_embb_rate / max(reference_embb_rate, 1.0e-9)),
            "cumulative_embb_loss_so_far_normalized": float(np.clip(cumulative_embb_loss / max(reference_embb_rate, 1.0e-9), 0.0, 1.5)),
        }

    def _phase_a_progress_obs_features(self, progress_summary: Dict[str, float]) -> List[float]:
        features: List[float] = []
        if bool(getattr(self.rl_cfg.env, "include_frontier_progress_obs", False)):
            features.extend([
                float(progress_summary.get("target_quota_normalized", 0.0)),
                float(progress_summary.get("quota_gap_normalized", 0.0)),
                float(progress_summary.get("required_admit_rate_per_remaining_cell", 0.0)),
            ])
        if bool(getattr(self.rl_cfg.env, "include_quota_progress_obs", False)):
            features.extend([
                float(progress_summary.get("admitted_so_far_normalized", 0.0)),
                float(progress_summary.get("remaining_active_packets_normalized", 0.0)),
                float(progress_summary.get("remaining_cells_normalized", 0.0)),
                float(progress_summary.get("overlay_count_so_far_normalized", 0.0)),
                float(progress_summary.get("puncture_count_so_far_normalized", 0.0)),
                float(progress_summary.get("cumulative_embb_rate_so_far_normalized", 0.0)),
                float(progress_summary.get("cumulative_embb_loss_so_far_normalized", 0.0)),
            ])
        return features

    def _phase_a_progress_obs_dim(self) -> int:
        dim = 0
        if bool(getattr(self.rl_cfg.env, "include_frontier_progress_obs", False)):
            dim += 3
        if bool(getattr(self.rl_cfg.env, "include_quota_progress_obs", False)):
            dim += 7
        return dim

    def _current_admission_collapse_floor(self, actual_load: Optional[float] = None) -> float:
        return self._load_bucket_value(
            getattr(self.rl_cfg.reward, "terminal_admission_collapse_floor_by_load", {}),
            fallback=0.0,
            actual_load=actual_load,
        )

    def _should_attach_greedy_reference(self) -> bool:
        # Greedy snapshot leakage guard: even if higher-level training wants a greedy reference
        # (BC/teacher forcing), allow debug experiments to hard-disable any baseline-derived hints
        # entering the policy forward path.
        if not bool(getattr(self.rl_cfg.env, "owner_snapshot_in_observation", True)):
            return False
        return bool(
            bool(getattr(self.rl_cfg.env, "include_greedy_reference_in_obs", False))
            or bool(getattr(self.rl_cfg.training, "use_greedy_reference_bc", False))
            or str(getattr(self.rl_cfg.training, "bc_teacher_policy", "") or "").strip().lower() == "greedy_reference"
        )

    def _current_power_ratio_ceiling(self, actual_load: Optional[float] = None) -> float:
        return power_ratio_ceiling_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "selection_power_ratio_ceiling_by_load", {}),
            fallback=float("inf"),
        )

    def _reference_embb_total_rate_for_local_shaping(self) -> float:
        snapshot_rate = float(getattr(self, "phase0_snapshot_embb_total_rate", 0.0))
        current_base_rate = 0.0
        if self.embb_base_rb_rates is not None:
            current_base_rate = float(np.sum(self.embb_base_rb_rates))
        return float(max(snapshot_rate, current_base_rate, 1.0e-9))

    def throughput_only_greedy_action(
        self,
        observation: AgentObservation,
    ) -> Tuple[HybridAction, Dict[str, float]]:
        """Pick the one-step action with the highest immediate aggregate eMBB throughput.

        This helper is intentionally one-dimensional:
        maximize resulting aggregate eMBB throughput, or equivalently minimize
        immediate aggregate eMBB throughput loss. URLLC admission is not an
        objective here; it only appears as an action-feasibility consequence.
        When `MODE_KEEP` is available, it is evaluated as an explicit no-op /
        reject action and selected whenever it preserves more eMBB throughput
        than every feasible admit action.
        """

        planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
        if planning_phase:
            action = self._planning_teacher_action(observation)
            return action, {
                "phase_a_decision": 0.0,
                "noop_available": 0.0,
                "noop_selected": 0.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "selected_loss": 0.0,
                "selected_retention": 1.0,
                "selected_throughput": 0.0,
                "rejected_when_noop_better": 0.0,
                "feasible_admit_count": 0.0,
                "no_op_better_than_best_admit": 0.0,
                "no_op_tied_with_best_admit": 0.0,
                "current_env_requires_feasible_admission_only": 0.0,
            }

        reference_rate = self._reference_embb_total_rate_for_local_shaping()
        mode_mask = np.asarray(observation.masks.mode_mask, dtype=float)
        packet_mask = np.asarray(observation.masks.packet_mask, dtype=float)

        keep_available = bool(
            mode_mask.size > MODE_KEEP
            and mode_mask[MODE_KEEP] > 0.5
            and packet_mask.ndim == 2
            and packet_mask.shape[0] > MODE_KEEP
            and packet_mask.shape[1] > 0
            and packet_mask[MODE_KEEP, 0] > 0.5
        )

        feasible_admit_count = 0
        reject_reliability_count = 0
        reject_power_count = 0
        reject_min_rate_count = 0
        reject_share_cap_count = 0
        candidate_evaluated_count = 0
        best_action: Optional[HybridAction] = None
        best_mode = MODE_KEEP
        best_loss = float("inf")
        best_throughput = float("-inf")
        best_retention = 0.0
        best_packet_option = 0
        best_key: Optional[Tuple[float, float, float, float, float]] = None
        best_admit_throughput = float("-inf")
        best_admit_loss = float("inf")

        if keep_available:
            best_action = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            best_mode = MODE_KEEP
            best_loss = 0.0
            best_throughput = float(reference_rate)
            best_retention = 1.0
            best_packet_option = 0
            # Exact-throughput ties are resolved deterministically in favor of
            # KEEP so we do not inject an implicit admission objective.
            best_key = (
                float(reference_rate),
                1.0,
                0.0,
                0.0,
                0.0,
            )

        for packet_option, candidate in enumerate(observation.candidates, start=1):
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                if (
                    mode_mask.size <= mode
                    or mode_mask[mode] <= 0.5
                    or packet_mask.ndim != 2
                    or packet_mask.shape[0] <= mode
                    or packet_mask.shape[1] <= packet_option
                    or packet_mask[mode, packet_option] <= 0.5
                    or not candidate.is_mode_feasible(mode)
                ):
                    continue
                feasible_admit_count += 1
                loss = float(candidate.loss_for_mode(mode))
                throughput = float(reference_rate - loss)
                retention = float(np.clip(throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                intercell_cost = 0.0
                try:
                    uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                    rb_idx = int(observation.metadata.get("rb_index", -1.0))
                    minislot_idx = int(observation.metadata.get("minislot_index", -1.0))
                    if uav_idx >= 0 and rb_idx >= 0 and minislot_idx >= 0:
                        intercell_cost = float(
                            self._estimate_candidate_action_intercell_cost_after_source_mask(
                                uav_idx,
                                rb_idx,
                                minislot_idx,
                                candidate,
                                int(mode),
                            )
                        )
                except Exception:
                    intercell_cost = 0.0
                admit_key = (
                    -(loss + intercell_cost),
                    throughput,
                    0.0,
                    -float(packet_option),
                    -float(mode),
                    -loss,
                )
                if throughput > best_admit_throughput + 1.0e-12:
                    best_admit_throughput = throughput
                    best_admit_loss = loss
                elif abs(throughput - best_admit_throughput) <= 1.0e-12 and loss < best_admit_loss:
                    best_admit_loss = loss
                if best_key is None or admit_key > best_key:
                    best_key = admit_key
                    best_action = HybridAction(
                        mode=int(mode),
                        packet_option=int(packet_option),
                        power_delta=0.0,
                    )
                    best_mode = int(mode)
                    best_loss = loss
                    best_throughput = throughput
                    best_retention = retention
                    best_packet_option = int(packet_option)

        if feasible_admit_count > 0 and best_action is not None and int(best_mode) == MODE_KEEP:
            best_action = None
            best_key = None
            best_mode = MODE_KEEP
            best_loss = float("inf")
            best_throughput = float("-inf")
            best_retention = 0.0
            best_packet_option = 0
            for packet_option, candidate in enumerate(observation.candidates, start=1):
                for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                    if (
                        mode_mask.size <= mode
                        or mode_mask[mode] <= 0.5
                        or packet_mask.ndim != 2
                        or packet_mask.shape[0] <= mode
                        or packet_mask.shape[1] <= packet_option
                        or packet_mask[mode, packet_option] <= 0.5
                        or not candidate.is_mode_feasible(mode)
                    ):
                        continue
                    loss = float(candidate.loss_for_mode(mode))
                    throughput = float(reference_rate - loss)
                    retention = float(np.clip(throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                    intercell_cost = 0.0
                    try:
                        uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                        rb_idx = int(observation.metadata.get("rb_index", -1.0))
                        minislot_idx = int(observation.metadata.get("minislot_index", -1.0))
                        if uav_idx >= 0 and rb_idx >= 0 and minislot_idx >= 0:
                            intercell_cost = float(
                                self._estimate_candidate_action_intercell_cost_after_source_mask(
                                    uav_idx,
                                    rb_idx,
                                    minislot_idx,
                                    candidate,
                                    int(mode),
                                )
                            )
                    except Exception:
                        intercell_cost = 0.0
                    admit_key = (
                        -(loss + intercell_cost),
                        throughput,
                        0.0,
                        -float(packet_option),
                        -float(mode),
                        -loss,
                    )
                    if best_key is None or admit_key > best_key:
                        best_key = admit_key
                        best_action = HybridAction(
                            mode=int(mode),
                            packet_option=int(packet_option),
                            power_delta=0.0,
                        )
                        best_mode = int(mode)
                        best_loss = loss
                        best_throughput = throughput
                        best_retention = retention
                        best_packet_option = int(packet_option)

        if best_action is None:
            best_action = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            best_mode = MODE_KEEP
            best_loss = 0.0
            best_throughput = float(reference_rate)
            best_retention = 1.0
            best_packet_option = 0

        no_op_better_than_best_admit = bool(
            keep_available
            and feasible_admit_count > 0
            and reference_rate > best_admit_throughput + 1.0e-12
        )
        no_op_tied_with_best_admit = bool(
            keep_available
            and feasible_admit_count > 0
            and abs(reference_rate - best_admit_throughput) <= 1.0e-12
        )

        return best_action, {
            "phase_a_decision": 1.0,
            "noop_available": float(keep_available),
            "noop_selected": float(best_mode == MODE_KEEP),
            "admit_selected": float(best_mode != MODE_KEEP),
            "overlay_selected": float(best_mode == MODE_OVERLAY),
            "puncture_selected": float(best_mode == MODE_PUNCTURE),
            "selected_loss": float(best_loss),
            "selected_retention": float(best_retention),
            "selected_throughput": float(best_throughput),
            "rejected_when_noop_better": float(no_op_better_than_best_admit and best_mode == MODE_KEEP),
            "feasible_admit_count": float(feasible_admit_count),
            "no_op_better_than_best_admit": float(no_op_better_than_best_admit),
            "no_op_tied_with_best_admit": float(no_op_tied_with_best_admit),
            "current_env_requires_feasible_admission_only": float((not keep_available) and feasible_admit_count > 0),
        }

    def myopic_throughput_greedy_action(
        self,
        observation: AgentObservation,
        *,
        throughput_rel_tol: float = 1.0e-2,
        throughput_abs_tol: float = 2.0e5,
        overlay_retention_tiebreak_floor: float = 0.90,
        urgent_keep_forbidden_threshold: float = 0.50,
    ) -> Tuple[HybridAction, Dict[str, float]]:
        """Single-objective myopic greedy baseline (hard-feasible, weak tie-breaks).

        Primary objective:
            maximize immediate aggregate eMBB throughput (equivalently minimize immediate eMBB loss).

        Hard constraints:
            admit actions must be feasible per masks + candidate feasibility flags.

        KEEP/no-op is an explicit candidate. When the best admit action is within a small throughput
        tolerance (epsilon) of KEEP, we treat it as a near-tie and apply weak tie-breaks:
            1) lower required URLLC power
            2) prefer overlay if overlay retention is above a floor
            3) higher admitted reliability
            4) higher urgency (deadline pressure)

        Admission is not an optimization target; it only arises as a consequence of the selected
        near-tie action under the throughput-first objective.
        """

        planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
        if planning_phase:
            action = self._planning_teacher_action(observation)
            return action, {
                "phase_a_decision": 0.0,
                "noop_available": 0.0,
                "noop_selected": 0.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "selected_loss": 0.0,
                "selected_retention": 1.0,
                "selected_throughput": 0.0,
                "rejected_when_noop_better": 0.0,
                "feasible_admit_count": 0.0,
                "no_op_better_than_best_admit": 0.0,
                "no_op_tied_with_best_admit": 0.0,
                "current_env_requires_feasible_admission_only": 0.0,
            }

        minislot = int(observation.metadata.get("minislot_index", 0.0))
        reference_rate = float(self._reference_embb_total_rate_for_local_shaping())
        share_mode = str(getattr(self.rl_cfg.env, "greedy_urllc_share_mode", "none") or "none").strip().lower()
        share_ratio = float(np.clip(float(getattr(self.rl_cfg.env, "greedy_urllc_share_ratio", 0.0) or 0.0), 0.0, 1.0))
        if share_mode == "fixed_share" and share_ratio > 0.0:
            greedy_embb_throughput_floor = float(reference_rate * (1.0 - share_ratio))
        else:
            greedy_embb_throughput_floor = 0.0
        share_mode = str(getattr(self.rl_cfg.env, "greedy_urllc_share_mode", "none") or "none").strip().lower()
        share_ratio = float(np.clip(float(getattr(self.rl_cfg.env, "greedy_urllc_share_ratio", 0.0) or 0.0), 0.0, 1.0))
        if share_mode == "fixed_share" and share_ratio > 0.0:
            greedy_embb_throughput_floor = float(reference_rate * (1.0 - share_ratio))
        else:
            greedy_embb_throughput_floor = 0.0
        mode_mask = np.asarray(observation.masks.mode_mask, dtype=float)
        packet_mask = np.asarray(observation.masks.packet_mask, dtype=float)

        keep_available = bool(
            mode_mask.size > MODE_KEEP
            and mode_mask[MODE_KEEP] > 0.5
            and packet_mask.ndim == 2
            and packet_mask.shape[0] > MODE_KEEP
            and packet_mask.shape[1] > 0
            and packet_mask[MODE_KEEP, 0] > 0.5
        )

        options: List[Dict[str, object]] = []
        feasible_admit_count = 0
        max_feasible_urgency = 0.0

        if keep_available:
            options.append({
                "action": HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
                "mode": int(MODE_KEEP),
                "packet_option": 0,
                "loss": 0.0,
                "throughput": float(reference_rate),
                "retention": 1.0,
                "power": 0.0,
                "reliability": 1.0,
                "urgency": 0.0,
                "overlay_prefer": 0.0,
            })

        for packet_option, candidate in enumerate(observation.candidates, start=1):
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                if (
                    mode_mask.size <= mode
                    or mode_mask[mode] <= 0.5
                    or packet_mask.ndim != 2
                    or packet_mask.shape[0] <= mode
                    or packet_mask.shape[1] <= packet_option
                    or packet_mask[mode, packet_option] <= 0.5
                    or not candidate.is_mode_feasible(mode)
                ):
                    continue
                feasible_admit_count += 1
                loss = float(candidate.loss_for_mode(mode))
                throughput = float(reference_rate - loss)
                retention = float(np.clip(throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                options.append({
                    "action": HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0),
                    "mode": int(mode),
                    "packet_option": int(packet_option),
                    "loss": loss,
                    "global_loss": loss,
                    "throughput": throughput,
                    "retention": retention,
                    "power": float(candidate.required_power_for_mode(mode)),
                    "reliability": float(candidate.reliability_for_mode(mode)),
                    "urgency": float(self._candidate_urgency(candidate.packet_id, minislot)),
                    "overlay_prefer": float(
                        mode == MODE_OVERLAY and float(candidate.overlay_retention) >= float(overlay_retention_tiebreak_floor)
                    ),
                })
                max_feasible_urgency = max(max_feasible_urgency, float(self._candidate_urgency(candidate.packet_id, minislot)))

        keep_forbidden_by_urgency = bool(
            keep_available
            and feasible_admit_count > 0
            and max_feasible_urgency >= float(urgent_keep_forbidden_threshold) - 1.0e-12
        )
        # Mandatory-admit policy: once there is any feasible admit option, do not allow KEEP.
        if feasible_admit_count > 0:
            options = [item for item in options if int(item.get("mode", MODE_KEEP)) != MODE_KEEP]
        elif keep_forbidden_by_urgency:
            options = [item for item in options if int(item.get("mode", MODE_KEEP)) != MODE_KEEP]

        if not options:
            best_action = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            return best_action, {
                "phase_a_decision": 1.0,
                "noop_available": 0.0,
                "noop_selected": 1.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "selected_loss": 0.0,
                "selected_retention": 1.0,
                "selected_throughput": float(reference_rate),
                "rejected_when_noop_better": 0.0,
                "feasible_admit_count": float(feasible_admit_count),
                "no_op_better_than_best_admit": 0.0,
                "no_op_tied_with_best_admit": 0.0,
                "current_env_requires_feasible_admission_only": 1.0,
            }

        best_throughput = float(max(float(item["throughput"]) for item in options))
        eps = float(max(float(throughput_abs_tol), float(throughput_rel_tol) * max(reference_rate, 1.0)))
        near_best = [item for item in options if best_throughput - float(item["throughput"]) <= eps + 1.0e-12]

        def _tie_key(item: Dict[str, object]) -> tuple:
            mode = int(item["mode"])
            intercell_cost = 0.0
            try:
                packet_idx = int(item.get("packet_option", 0))
                if packet_idx > 0:
                    candidate = observation.candidates[packet_idx - 1]
                    uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                    rb_idx = int(observation.metadata.get("rb_index", -1.0))
                    minislot_idx = int(observation.metadata.get("minislot_index", -1.0))
                    if uav_idx >= 0 and rb_idx >= 0 and minislot_idx >= 0:
                        intercell_cost = float(
                            self._estimate_candidate_action_intercell_cost_after_source_mask(
                                uav_idx,
                                rb_idx,
                                minislot_idx,
                                candidate,
                                mode,
                            )
                        )
            except Exception:
                intercell_cost = 0.0
            global_loss = float(item.get("global_loss", item.get("loss", 0.0))) + float(intercell_cost)
            return (
                float(global_loss),
                -float(item["throughput"]),
                float(item["loss"]),
                float(intercell_cost),
                float(item["power"]),
                -float(item["overlay_prefer"]),
                -float(item["reliability"]),
                -float(item["urgency"]),
                1 if mode == MODE_KEEP else 0,
                -mode,
                -int(item["packet_option"]),
            )

        chosen = min(near_best, key=_tie_key)
        action: HybridAction = chosen["action"]  # type: ignore[assignment]
        chosen_mode = int(chosen["mode"])

        best_admit_tp = float("-inf")
        if feasible_admit_count > 0:
            best_admit_tp = float(max(
                float(item["throughput"])
                for item in options
                if int(item["mode"]) in (MODE_OVERLAY, MODE_PUNCTURE)
            ))

        no_op_better_than_best_admit = bool(
            keep_available
            and feasible_admit_count > 0
            and reference_rate > best_admit_tp + 1.0e-12
        )
        no_op_tied_with_best_admit = bool(
            keep_available
            and feasible_admit_count > 0
            and abs(reference_rate - best_admit_tp) <= eps + 1.0e-12
        )

        return action, {
            "phase_a_decision": 1.0,
            "noop_available": float(keep_available),
            "noop_selected": float(chosen_mode == MODE_KEEP),
            "admit_selected": float(chosen_mode != MODE_KEEP),
            "overlay_selected": float(chosen_mode == MODE_OVERLAY),
            "puncture_selected": float(chosen_mode == MODE_PUNCTURE),
            "selected_loss": float(chosen.get("loss", 0.0)),
            "selected_retention": float(chosen.get("retention", 1.0)),
            "selected_throughput": float(chosen.get("throughput", reference_rate)),
            "rejected_when_noop_better": float(no_op_better_than_best_admit and chosen_mode == MODE_KEEP),
            "feasible_admit_count": float(feasible_admit_count),
            "no_op_better_than_best_admit": float(no_op_better_than_best_admit),
            "no_op_tied_with_best_admit": float(no_op_tied_with_best_admit),
            "current_env_requires_feasible_admission_only": float((not keep_available) and feasible_admit_count > 0),
            "keep_forbidden_by_urgency": float(keep_forbidden_by_urgency),
        }

    def throughput_feasible_oracle_action(
        self,
        observation: AgentObservation,
    ) -> Tuple[HybridAction, Dict[str, float]]:
        """Constrained throughput oracle: best eMBB throughput among feasible-admit actions only.

        Definition:
            argmax_{a in A_feasible_admit} immediate_eMBB_throughput(a)

        KEEP/no-op is not part of the comparison set and is used only as a hard fallback
        when there is no feasible admit action.
        """

        planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
        if planning_phase:
            action = self._planning_teacher_action(observation)
            return action, {
                "phase_a_decision": 0.0,
                "noop_available": 0.0,
                "noop_selected": 0.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "selected_loss": 0.0,
                "selected_retention": 1.0,
                "selected_throughput": 0.0,
                "rejected_when_noop_better": 0.0,
                "feasible_admit_count": 0.0,
                "no_op_better_than_best_admit": 0.0,
                "no_op_tied_with_best_admit": 0.0,
                "current_env_requires_feasible_admission_only": 1.0,
            }

        reference_rate_dynamic = float(self._reference_embb_total_rate_for_local_shaping())
        reference_rate_fixed = float(self._fixed_share_reference_rate_bps_for_current_load())
        reference_rate = float(reference_rate_fixed if reference_rate_fixed > 0.0 else reference_rate_dynamic)
        share_mode = str(getattr(self.rl_cfg.env, "greedy_urllc_share_mode", "none") or "none").strip().lower()
        share_ratio = float(np.clip(float(getattr(self.rl_cfg.env, "greedy_urllc_share_ratio", 0.0) or 0.0), 0.0, 1.0))
        mode_mask = np.asarray(observation.masks.mode_mask, dtype=float)
        packet_mask = np.asarray(observation.masks.packet_mask, dtype=float)

        feasible_admit_count = 0
        candidate_evaluated_count = 0
        reject_reliability_count = 0
        reject_power_count = 0
        reject_min_rate_count = 0
        reject_share_cap_count = 0
        best_action: Optional[HybridAction] = None
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_mode = MODE_KEEP
        best_loss = float("inf")
        best_throughput = float("-inf")
        best_retention = 0.0

        for packet_option, candidate in enumerate(observation.candidates, start=1):
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                if (
                    mode_mask.size <= mode
                    or mode_mask[mode] <= 0.5
                    or packet_mask.ndim != 2
                    or packet_mask.shape[0] <= mode
                    or packet_mask.shape[1] <= packet_option
                    or packet_mask[mode, packet_option] <= 0.5
                    or not candidate.is_mode_feasible(mode)
                ):
                    continue
                feasible_admit_count += 1
                loss = float(candidate.loss_for_mode(mode))
                throughput = float(reference_rate - loss)
                retention = float(np.clip(throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                # Deterministic tie-break only.
                key = (
                    throughput,
                    -loss,
                    float(mode == MODE_OVERLAY),
                    -float(packet_option),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_action = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
                    best_mode = int(mode)
                    best_loss = loss
                    best_throughput = throughput
                    best_retention = retention

        if best_action is None:
            best_action = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            best_mode = MODE_KEEP
            best_loss = 0.0
            best_throughput = float(reference_rate)
            best_retention = 1.0

        return best_action, {
            "phase_a_decision": 1.0,
            "noop_available": 0.0,
            "noop_selected": float(best_mode == MODE_KEEP),
            "admit_selected": float(best_mode != MODE_KEEP),
            "overlay_selected": float(best_mode == MODE_OVERLAY),
            "puncture_selected": float(best_mode == MODE_PUNCTURE),
            "selected_loss": float(best_loss),
            "selected_retention": float(best_retention),
            "selected_throughput": float(best_throughput),
            "rejected_when_noop_better": 0.0,
            "feasible_admit_count": float(feasible_admit_count),
            "no_op_better_than_best_admit": 0.0,
            "no_op_tied_with_best_admit": 0.0,
            "current_env_requires_feasible_admission_only": 1.0,
            "greedy_policy_name": "throughput_feasible_oracle",
            "greedy_requires_feasible_admission_only": 1.0,
        }

    def hard_feasible_throughput_greedy_action(
        self,
        observation: AgentObservation,
    ) -> Tuple[HybridAction, Dict[str, float]]:
        """Hard-feasible throughput greedy (admit-only; KEEP iff none feasible).

        This baseline enumerates admit actions only and chooses the feasible admit action with
        the highest resulting aggregate eMBB throughput. KEEP/no-op is used only when there is
        no feasible admit action.

        Feasible admit action requirements:
          a) URLLC reliability >= target reliability
          b) action mode feasible (overlay_feasible / puncture_feasible)
          c) required transmit power does not exceed the power upper bound
        """

        self.profile_hf_action_calls += 1
        planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
        if planning_phase:
            action = self._planning_teacher_action(observation)
            return action, {
                "phase_a_decision": 0.0,
                "noop_available": 0.0,
                "noop_selected": 0.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "selected_loss": 0.0,
                "selected_retention": 1.0,
                "selected_throughput": 0.0,
                "rejected_when_noop_better": 0.0,
                "selected_reliability": 0.0,
                "selected_embb_min_rate_ok": 1.0,
                "feasible_admit_count": 0.0,
                "keep_selected_due_to_no_feasible_admit": 0.0,
                "no_op_better_than_best_admit": 0.0,
                "no_op_tied_with_best_admit": 0.0,
                "current_env_requires_feasible_admission_only": 1.0,
                "greedy_policy_name": "hard_feasible_throughput_greedy",
                "greedy_requires_feasible_admission_only": 1.0,
                "greedy_hf_no_candidate_ratio": 0.0,
                "greedy_hf_all_rejected_ratio": 0.0,
                "greedy_hf_budget_exhausted_keep_ratio": 0.0,
            }

        reference_rate = float(self._reference_embb_total_rate_for_local_shaping())
        share_mode = str(getattr(self.rl_cfg.env, "greedy_urllc_share_mode", "none") or "none").strip().lower()
        share_ratio = float(np.clip(float(getattr(self.rl_cfg.env, "greedy_urllc_share_ratio", 0.0) or 0.0), 0.0, 1.0))
        greedy_embb_throughput_floor = float(reference_rate * (1.0 - share_ratio)) if (share_mode == "fixed_share" and share_ratio > 0.0) else 0.0
        mode_mask = np.asarray(observation.masks.mode_mask, dtype=float)
        packet_mask = np.asarray(observation.masks.packet_mask, dtype=float)
        reliability_target = float(1.0 - self.urllc_cfg.target_error_probability)
        power_upper_bound = float(self.algo_cfg.power_upper_bound)
        slot_duration_s = 1.0e-3
        packet_lengths = list(getattr(self.urllc_cfg, "packet_lengths", []) or [])
        packet_bits = float(np.mean(np.asarray(packet_lengths, dtype=float))) if packet_lengths else 160.0
        urllc_packet_throughput_bps_est = float(packet_bits / max(slot_duration_s, 1.0e-12))

        feasible_admit_count = 0
        candidate_evaluated_count = 0
        reject_reliability_count = 0
        reject_power_count = 0
        reject_min_rate_count = 0
        reject_share_cap_count = 0
        best_action: Optional[HybridAction] = None
        best_key: Optional[Tuple[float, float, float, float]] = None
        best_mode = MODE_KEEP
        best_loss = float("inf")
        best_throughput = float("-inf")
        best_retention = 0.0
        best_reliability = 0.0
        best_min_rate_ok = 1.0

        # Stage-1 cheap prefilter: keep lowest-loss candidate-mode pairs only.
        _prefilter_t0 = perf_counter()
        candidate_mode_pairs_all: List[Tuple[int, object, int, float]] = []
        prefilter_mode_mask_block_count = 0
        prefilter_packet_mask_block_count = 0
        prefilter_mode_infeasible_block_count = 0
        for packet_option, candidate in enumerate(observation.candidates, start=1):
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                self.greedy_hf_prefilter_pair_total += 1
                if mode_mask.size <= mode or mode_mask[mode] <= 0.5:
                    prefilter_mode_mask_block_count += 1
                    self.greedy_hf_prefilter_block_mode_mask_total += 1
                if (
                    packet_mask.ndim != 2
                    or packet_mask.shape[0] <= mode
                    or packet_mask.shape[1] <= packet_option
                    or packet_mask[mode, packet_option] <= 0.5
                ):
                    prefilter_packet_mask_block_count += 1
                    self.greedy_hf_prefilter_block_packet_mask_total += 1
                if not candidate.is_mode_feasible(mode):
                    prefilter_mode_infeasible_block_count += 1
                    self.greedy_hf_prefilter_block_mode_infeasible_total += 1
                    continue
                est_loss = float(candidate.loss_for_mode(mode))
                candidate_mode_pairs_all.append((packet_option, candidate, int(mode), est_loss))

        prefilter_topk = int(getattr(self.rl_cfg.env, "greedy_hf_prefilter_topk", 0) or 0)
        candidate_mode_pairs = list(candidate_mode_pairs_all)
        prefilter_truncated = False
        candidate_mode_pairs_tail: List[Tuple[int, object, int, float]] = []
        if prefilter_topk > 0 and len(candidate_mode_pairs) > prefilter_topk:
            candidate_mode_pairs.sort(key=lambda x: x[3])
            candidate_mode_pairs_tail = candidate_mode_pairs[prefilter_topk:]
            candidate_mode_pairs = candidate_mode_pairs[:prefilter_topk]
            prefilter_truncated = True
        no_candidate_before_prefilter = float(len(candidate_mode_pairs_all) <= 0)
        if no_candidate_before_prefilter > 0.5:
            if len(observation.candidates) <= 0:
                self.greedy_hf_no_candidate_empty_observation_total += 1
            else:
                self.greedy_hf_no_candidate_mask_block_total += 1
            self.greedy_hf_no_candidate_block_mode_mask_total += int(prefilter_mode_mask_block_count)
            self.greedy_hf_no_candidate_block_packet_mask_total += int(prefilter_packet_mask_block_count)
            self.greedy_hf_no_candidate_block_mode_infeasible_total += int(prefilter_mode_infeasible_block_count)
        self.profile_hf_prefilter_sec += float(perf_counter() - _prefilter_t0)

        # Stage-2 full feasibility checks on prefiltered pairs.
        # Keep share-cap checks in a separate prefix pass so we can reuse
        # the same candidate ordering/feasibility work efficiently.
        _eval_t0 = perf_counter()
        rescan_used = False
        # User policy: feasible URLLC must be admitted; do not reject via share-cap budget.
        share_cap_enabled = False

        feasible_candidates_nonshare: List[Tuple[int, object, int, float, float, float, float]] = []
        # tuple: (packet_option, candidate, mode, loss, reliability, intercell_cost, global_sum_throughput_bps)

        def _eval_pairs_nonshare(pairs: List[Tuple[int, object, int, float]]) -> None:
            nonlocal candidate_evaluated_count
            nonlocal reject_reliability_count, reject_power_count, reject_min_rate_count
            for packet_option, candidate, mode, _est_loss in pairs:
                candidate_evaluated_count += 1

                reliability = float(candidate.reliability_for_mode(mode))
                if reliability < reliability_target - 1.0e-9:
                    reject_reliability_count += 1
                    continue

                required_power = float(candidate.required_power_for_mode(mode))
                if required_power > power_upper_bound + 1.0e-12:
                    reject_power_count += 1
                    continue

                # Relax min-rate hard gate here to avoid pathological all-reject
                # behavior under heavy URLLC arrival. eMBB protection is still
                # reflected in the global throughput objective and resulting tradeoff.

                loss = float(candidate.loss_for_mode(mode))
                intercell_cost = 0.0
                try:
                    uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                    rb_idx = int(observation.metadata.get("rb_index", -1.0))
                    minislot_idx = int(observation.metadata.get("minislot_index", -1.0))
                    if uav_idx >= 0 and rb_idx >= 0 and minislot_idx >= 0:
                        intercell_cost = float(
                            self._estimate_candidate_action_intercell_cost_after_source_mask(
                                uav_idx,
                                rb_idx,
                                minislot_idx,
                                candidate,
                                int(mode),
                            )
                        )
                except Exception:
                    intercell_cost = 0.0
                global_sum_tp = float(
                    self._global_sum_throughput_if_apply_candidate_action(
                        observation=observation,
                        candidate=candidate,
                        mode=int(mode),
                        power_delta=0.0,
                    )
                )
                feasible_candidates_nonshare.append(
                    (
                        int(packet_option),
                        candidate,
                        int(mode),
                        float(loss),
                        float(reliability),
                        float(intercell_cost),
                        float(global_sum_tp),
                    )
                )

        _eval_pairs_nonshare(candidate_mode_pairs)

        # Fast+safe fallback:
        # If top-k produced no feasible admit and we truncated candidates, evaluate
        # the remaining tail once to avoid false "no-feasible" caused by prefilter.
        if len(feasible_candidates_nonshare) <= 0 and prefilter_truncated and candidate_mode_pairs_tail:
            rescan_used = True
            _eval_pairs_nonshare(candidate_mode_pairs_tail)

        # Prefix-style share-cap screening over the same sorted feasible list.
        # Because objective is throughput=max(reference-loss), lower loss dominates;
        # once a loss violates remaining share budget, larger losses cannot recover.
        if feasible_candidates_nonshare:
            feasible_candidates_nonshare.sort(
                key=lambda item: (
                    -float(item[6]),  # strict objective: maximize global sum throughput
                    float(item[5]),  # tie-break: lower inter-cell side effect
                    float(item[3]),  # tie-break: lower local eMBB loss
                    -float(item[2] == MODE_OVERLAY),  # deterministic tie-break (overlay preferred)
                    float(item[0]),  # then smaller packet index
                )
            )

            if not share_cap_enabled:
                packet_option, _candidate, mode, loss, reliability, _intercell_cost, _global_tp = feasible_candidates_nonshare[0]
                feasible_admit_count = int(len(feasible_candidates_nonshare))
                best_action = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
                best_mode = int(mode)
                best_loss = float(loss)
                best_throughput = float(reference_rate - loss)
                best_retention = float(np.clip(best_throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                best_reliability = float(reliability)
                best_min_rate_ok = 1.0
            else:
                remaining_budget = float(self.greedy_urllc_budget_bps - self.greedy_urllc_budget_used_bps)
                remaining_budget = max(remaining_budget, 0.0)
                for idx, (packet_option, _candidate, mode, loss, reliability, _intercell_cost, _global_tp) in enumerate(feasible_candidates_nonshare):
                    projected_used = float(self.greedy_urllc_budget_used_bps + loss)
                    if projected_used > float(self.greedy_urllc_budget_bps) + 1.0e-9:
                        reject_share_cap_count += int(len(feasible_candidates_nonshare) - idx)
                        break
                    if loss > remaining_budget + 1.0e-9:
                        reject_share_cap_count += int(len(feasible_candidates_nonshare) - idx)
                        break

                    feasible_admit_count = int(len(feasible_candidates_nonshare) - idx)
                    best_action = HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0)
                    best_mode = int(mode)
                    best_loss = float(loss)
                    best_throughput = float(reference_rate - loss)
                    best_retention = float(np.clip(best_throughput / max(reference_rate, 1.0e-9), 0.0, 1.0))
                    best_reliability = float(reliability)
                    best_min_rate_ok = 1.0
                    break

                if best_action is None:
                    # Non-share feasible exists, but all blocked by share cap/floor.
                    feasible_admit_count = 0
        self.profile_hf_eval_sec += float(perf_counter() - _eval_t0)
        all_rejected_after_eval = float(
            len(candidate_mode_pairs_all) > 0 and feasible_admit_count <= 0
        )
        top1_global_loss = float("nan")
        top2_global_loss = float("nan")
        top1_global_tp = float("nan")
        top2_global_tp = float("nan")
        top1_mode = float(MODE_KEEP)
        top2_mode = float(MODE_KEEP)
        top12_gap = float("nan")
        if feasible_candidates_nonshare:
            top1 = feasible_candidates_nonshare[0]
            top1_global_loss = float(top1[3] + top1[5])
            top1_global_tp = float(top1[6])
            top1_mode = float(top1[2])
            if len(feasible_candidates_nonshare) > 1:
                top2 = feasible_candidates_nonshare[1]
                top2_global_loss = float(top2[3] + top2[5])
                top2_global_tp = float(top2[6])
                top2_mode = float(top2[2])
                top12_gap = float(top2_global_loss - top1_global_loss)

        keep_selected_due_to_no_feasible_admit = 0.0
        if best_action is None:
            keep_selected_due_to_no_feasible_admit = 1.0
            best_action = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            best_mode = MODE_KEEP
            best_loss = 0.0
            best_throughput = float(reference_rate)
            best_retention = 1.0
            best_reliability = 1.0
            best_min_rate_ok = 1.0

        # Episode-level greedy feasibility diagnostics.
        self.greedy_hf_decision_count += 1
        self.greedy_hf_candidate_evaluated_total += int(candidate_evaluated_count)
        self.greedy_hf_candidate_reject_reliability_total += int(reject_reliability_count)
        self.greedy_hf_candidate_reject_power_total += int(reject_power_count)
        self.greedy_hf_candidate_reject_min_rate_total += int(reject_min_rate_count)
        self.greedy_hf_candidate_reject_share_cap_total += int(reject_share_cap_count)
        self.greedy_hf_candidate_feasible_total += int(feasible_admit_count)
        self.greedy_hf_no_candidate_total += int(no_candidate_before_prefilter > 0.5)
        self.greedy_hf_all_rejected_total += int(all_rejected_after_eval > 0.5)
        if best_mode == MODE_OVERLAY:
            self.greedy_hf_selected_overlay_total += 1
        elif best_mode == MODE_PUNCTURE:
            self.greedy_hf_selected_puncture_total += 1
        else:
            self.greedy_hf_selected_keep_total += 1
        # Consume episode-level eMBB loss budget only when we actually admit URLLC.
        if best_mode in (MODE_OVERLAY, MODE_PUNCTURE) and np.isfinite(self.greedy_urllc_budget_bps):
            self.greedy_urllc_budget_used_bps = float(
                min(float(self.greedy_urllc_budget_bps), float(self.greedy_urllc_budget_used_bps + max(best_loss, 0.0)))
            )

        return best_action, {
            "phase_a_decision": 1.0,
            "noop_available": 0.0,
            "noop_selected": float(best_mode == MODE_KEEP),
            "admit_selected": float(best_mode != MODE_KEEP),
            "overlay_selected": float(best_mode == MODE_OVERLAY),
            "puncture_selected": float(best_mode == MODE_PUNCTURE),
            "selected_loss": float(best_loss),
            "selected_retention": float(best_retention),
            "selected_throughput": float(best_throughput),
            "rejected_when_noop_better": 0.0,
            "selected_reliability": float(best_reliability),
            "selected_embb_min_rate_ok": float(best_min_rate_ok),
            "feasible_admit_count": float(feasible_admit_count),
            "keep_selected_due_to_no_feasible_admit": float(keep_selected_due_to_no_feasible_admit),
            "no_op_better_than_best_admit": 0.0,
            "no_op_tied_with_best_admit": 0.0,
            "current_env_requires_feasible_admission_only": 1.0,
            "greedy_policy_name": "hard_feasible_throughput_greedy",
            "greedy_requires_feasible_admission_only": 1.0,
            # Explicit greedy diagnostics (per-decision; aggregated in eval/report).
            "greedy_feasible_admit_count": float(feasible_admit_count),
            "greedy_no_feasible_admit_ratio": float(feasible_admit_count <= 0),
            "greedy_admit_selected_ratio": float(best_mode != MODE_KEEP),
            "greedy_keep_only_when_no_feasible_admit_ratio": float(keep_selected_due_to_no_feasible_admit),
            "greedy_selected_embb_throughput": float(best_throughput),
            "greedy_selected_urllc_reliability": float(best_reliability),
            "greedy_selected_embb_min_rate_ok": float(best_min_rate_ok),
            "greedy_hf_candidate_evaluated_count": float(candidate_evaluated_count),
            "greedy_hf_candidate_reject_reliability_count": float(reject_reliability_count),
            "greedy_hf_candidate_reject_power_count": float(reject_power_count),
            "greedy_hf_candidate_reject_min_rate_count": float(reject_min_rate_count),
            "greedy_hf_candidate_reject_share_cap_count": float(reject_share_cap_count),
            "greedy_hf_rescan_used": float(rescan_used),
            "greedy_hf_prefilter_truncated": float(prefilter_truncated),
            "greedy_hf_candidates_total_before_prefilter": float(len(candidate_mode_pairs_all)),
            "greedy_hf_no_candidate_ratio": float(no_candidate_before_prefilter),
            "greedy_hf_all_rejected_ratio": float(all_rejected_after_eval),
            "greedy_hf_budget_exhausted_keep_ratio": 0.0,
            "greedy_hf_top1_global_loss": float(top1_global_loss),
            "greedy_hf_top2_global_loss": float(top2_global_loss),
            "greedy_hf_top1_global_sum_throughput_bps": float(top1_global_tp),
            "greedy_hf_top2_global_sum_throughput_bps": float(top2_global_tp),
            "greedy_hf_top1_mode": float(top1_mode),
            "greedy_hf_top2_mode": float(top2_mode),
            "greedy_hf_top12_global_loss_gap": float(top12_gap),
        }

    def _global_sum_throughput_if_apply_candidate_action(
        self,
        observation: AgentObservation,
        candidate: Optional[CandidatePacket],
        mode: int,
        power_delta: float = 0.0,
    ) -> float:
            """Strict global objective evaluator for a hypothetical single action.

            Applies the candidate action to the current in-episode state, recomputes
            full-system throughput (eMBB effective throughput + URLLC throughput),
            then restores state.
            """
            if candidate is None or int(mode) == MODE_KEEP:
                embb_rates_eff, _embb_power_alloc_eff, _ov_eff, _pu_eff = self._compute_episode_embb_metrics(
                    ignore_intercell=False,
                    apply_local_puncture_deduction=True,
                    apply_embb_source_mask=True,
                )
                embb_total_eff = float(np.sum(embb_rates_eff))
                return float(embb_total_eff + self._current_scheduled_urllc_throughput_bps())

            try:
                uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                rb_idx = int(observation.metadata.get("rb_index", -1.0))
                minislot = int(observation.metadata.get("minislot_index", -1.0))
            except Exception:
                return float("-inf")
            if uav_idx < 0 or rb_idx < 0 or minislot < 0:
                return float("-inf")

            packet_id = int(getattr(candidate, "packet_id", -1))
            if packet_id < 0 or packet_id >= int(getattr(self, "num_packets", 0) or 0):
                return float("-inf")

            required_power = float(candidate.required_power_for_mode(int(mode)))
            actual_power = float(self._project_actual_power(required_power, float(power_delta)))

            # Snapshot mutable state.
            prev_packet_cell = int(self.packet_grid[uav_idx, rb_idx, minislot])
            prev_mode_cell = int(self.mode_grid[uav_idx, rb_idx, minislot])
            prev_owner_cell = int(self.embb_owner_grid[uav_idx, rb_idx, minislot])
            prev_scheduled_uav = int(self.scheduled_uavs[packet_id]) if packet_id < self.scheduled_uavs.size else -1
            prev_scheduled_rel = float(self.scheduled_reliabilities[packet_id]) if packet_id < self.scheduled_reliabilities.size else float("nan")
            prev_scheduled_power_row = self.scheduled_power[packet_id, :].copy() if packet_id < self.scheduled_power.shape[0] else None

            try:
                self.packet_grid[uav_idx, rb_idx, minislot] = int(packet_id)
                self.mode_grid[uav_idx, rb_idx, minislot] = int(mode)
                self.embb_owner_grid[uav_idx, rb_idx, minislot] = (
                    -1 if int(mode) == MODE_PUNCTURE else int(candidate.embb_owner_for_mode(int(mode)))
                )
                if packet_id < self.scheduled_power.shape[0]:
                    self.scheduled_power[packet_id, :] = 0.0
                    self.scheduled_power[packet_id, uav_idx] = float(actual_power)
                if packet_id < self.scheduled_uavs.size:
                    self.scheduled_uavs[packet_id] = int(uav_idx)
                if packet_id < self.scheduled_reliabilities.size:
                    self.scheduled_reliabilities[packet_id] = float(candidate.reliability_for_mode(int(mode)))

                embb_rates_eff, _embb_power_alloc_eff, _ov_eff, _pu_eff = self._compute_episode_embb_metrics(
                    ignore_intercell=False,
                    apply_local_puncture_deduction=True,
                    apply_embb_source_mask=True,
                )
                embb_total_eff = float(np.sum(embb_rates_eff))
                urllc_tp = float(self._current_scheduled_urllc_throughput_bps())
                return float(embb_total_eff + urllc_tp)
            finally:
                self.packet_grid[uav_idx, rb_idx, minislot] = int(prev_packet_cell)
                self.mode_grid[uav_idx, rb_idx, minislot] = int(prev_mode_cell)
                self.embb_owner_grid[uav_idx, rb_idx, minislot] = int(prev_owner_cell)
                if packet_id < self.scheduled_uavs.size:
                    self.scheduled_uavs[packet_id] = int(prev_scheduled_uav)
                if packet_id < self.scheduled_reliabilities.size:
                    self.scheduled_reliabilities[packet_id] = float(prev_scheduled_rel)
                if prev_scheduled_power_row is not None and packet_id < self.scheduled_power.shape[0]:
                    self.scheduled_power[packet_id, :] = prev_scheduled_power_row

    def _current_scheduled_urllc_throughput_bps(self) -> float:
            """Compute current scheduled URLLC throughput in bps (1 ms slot)."""
            if self.num_packets <= 0 or self.scheduled_uavs.size <= 0:
                return 0.0
            slot_duration_s = 1.0e-3
            total_bits = 0.0
            for packet_id in range(min(self.num_packets, self.scheduled_uavs.size)):
                if int(self.scheduled_uavs[packet_id]) < 0:
                    continue
                source_user = int(self.packet_sources[packet_id]) if packet_id < self.packet_sources.size else 0
                total_bits += float(self._packet_bits_for_user(source_user))
            return float(total_bits / max(slot_duration_s, 1.0e-12))

    def _embb_min_rate_ok_if_apply_candidate_action(
        self,
        observation: AgentObservation,
        candidate: Optional[CandidatePacket],
        mode: int,
        power_delta: float = 0.0,
    ) -> bool:
            """Hard-check whether a hypothetical action keeps all eMBB users above min-rate."""
            embb_min_rate = float(
                getattr(self.embb_cfg, "min_rate_per_user_bps", getattr(self.embb_cfg, "min_rate", 0.0)) or 0.0
            )
            if embb_min_rate <= 0.0:
                return True

            if candidate is None or int(mode) == MODE_KEEP:
                embb_rates_eff, _embb_power_alloc_eff, _ov_eff, _pu_eff = self._compute_episode_embb_metrics(
                    ignore_intercell=False,
                    apply_local_puncture_deduction=True,
                    apply_embb_source_mask=True,
                )
                rates = np.asarray(embb_rates_eff, dtype=float)
                if rates.size <= 0:
                    return True
                return bool(np.all(rates >= (embb_min_rate - 1.0e-9)))

            try:
                uav_idx = int(observation.metadata.get("uav_idx", -1.0))
                rb_idx = int(observation.metadata.get("rb_index", -1.0))
                minislot = int(observation.metadata.get("minislot_index", -1.0))
            except Exception:
                return False
            if uav_idx < 0 or rb_idx < 0 or minislot < 0:
                return False

            packet_id = int(getattr(candidate, "packet_id", -1))
            if packet_id < 0 or packet_id >= int(getattr(self, "num_packets", 0) or 0):
                return False

            required_power = float(candidate.required_power_for_mode(int(mode)))
            actual_power = float(self._project_actual_power(required_power, float(power_delta)))

            prev_packet_cell = int(self.packet_grid[uav_idx, rb_idx, minislot])
            prev_mode_cell = int(self.mode_grid[uav_idx, rb_idx, minislot])
            prev_owner_cell = int(self.embb_owner_grid[uav_idx, rb_idx, minislot])
            prev_scheduled_uav = int(self.scheduled_uavs[packet_id]) if packet_id < self.scheduled_uavs.size else -1
            prev_scheduled_rel = float(self.scheduled_reliabilities[packet_id]) if packet_id < self.scheduled_reliabilities.size else float("nan")
            prev_scheduled_power_row = self.scheduled_power[packet_id, :].copy() if packet_id < self.scheduled_power.shape[0] else None

            try:
                self.packet_grid[uav_idx, rb_idx, minislot] = int(packet_id)
                self.mode_grid[uav_idx, rb_idx, minislot] = int(mode)
                self.embb_owner_grid[uav_idx, rb_idx, minislot] = (
                    -1 if int(mode) == MODE_PUNCTURE else int(candidate.embb_owner_for_mode(int(mode)))
                )
                if packet_id < self.scheduled_power.shape[0]:
                    self.scheduled_power[packet_id, :] = 0.0
                    self.scheduled_power[packet_id, uav_idx] = float(actual_power)
                if packet_id < self.scheduled_uavs.size:
                    self.scheduled_uavs[packet_id] = int(uav_idx)
                if packet_id < self.scheduled_reliabilities.size:
                    self.scheduled_reliabilities[packet_id] = float(candidate.reliability_for_mode(int(mode)))

                embb_rates_eff, _embb_power_alloc_eff, _ov_eff, _pu_eff = self._compute_episode_embb_metrics(
                    ignore_intercell=False,
                    apply_local_puncture_deduction=True,
                    apply_embb_source_mask=True,
                )
                rates = np.asarray(embb_rates_eff, dtype=float)
                if rates.size <= 0:
                    return True
                return bool(np.all(rates >= (embb_min_rate - 1.0e-9)))
            finally:
                self.packet_grid[uav_idx, rb_idx, minislot] = int(prev_packet_cell)
                self.mode_grid[uav_idx, rb_idx, minislot] = int(prev_mode_cell)
                self.embb_owner_grid[uav_idx, rb_idx, minislot] = int(prev_owner_cell)
                if packet_id < self.scheduled_uavs.size:
                    self.scheduled_uavs[packet_id] = int(prev_scheduled_uav)
                if packet_id < self.scheduled_reliabilities.size:
                    self.scheduled_reliabilities[packet_id] = float(prev_scheduled_rel)
                if prev_scheduled_power_row is not None and packet_id < self.scheduled_power.shape[0]:
                    self.scheduled_power[packet_id, :] = prev_scheduled_power_row

    def throughput_biased_greedy_action(
        self,
        observation: AgentObservation,
        admission_band: Tuple[float, float] = (0.20, 0.50),
    ) -> Tuple[HybridAction, Dict[str, float]]:
        """Throughput-first greedy with a weak admission-band preference (legacy heuristic)."""

        planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
        if planning_phase:
            action = self._planning_teacher_action(observation)
            return action, {
                "phase_a_decision": 0.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "feasible_admit_count": 0.0,
                "band_hit": 0.0,
                "band_fallback_used": 0.0,
                "keep_selected_due_to_no_feasible_admit": 0.0,
                "selected_loss": 0.0,
                "selected_throughput": 0.0,
                "selected_admission": 0.0,
            }

        band_min = float(admission_band[0])
        band_max = float(admission_band[1])

        reference_rate = self._reference_embb_total_rate_for_local_shaping()
        mode_mask = np.asarray(observation.masks.mode_mask, dtype=float)
        packet_mask = np.asarray(observation.masks.packet_mask, dtype=float)

        feasible = []

        def _band_distance(x: float) -> float:
            if x < band_min:
                return band_min - x
            if x > band_max:
                return x - band_max
            return 0.0

        for packet_option, candidate in enumerate(observation.candidates, start=1):
            for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                if (
                    mode_mask.size <= mode
                    or mode_mask[mode] <= 0.5
                    or packet_mask.ndim != 2
                    or packet_mask.shape[0] <= mode
                    or packet_mask.shape[1] <= packet_option
                    or packet_mask[mode, packet_option] <= 0.5
                    or not candidate.is_mode_feasible(mode)
                ):
                    continue

                loss = float(candidate.loss_for_mode(mode))
                throughput = float(reference_rate - loss)
                admission = 1.0
                feasible.append({
                    "action": HybridAction(mode=int(mode), packet_option=int(packet_option), power_delta=0.0),
                    "mode": int(mode),
                    "loss": loss,
                    "throughput": throughput,
                    "admission": admission,
                    "band_distance": _band_distance(admission),
                })

        feasible_admit_count = len(feasible)
        if feasible_admit_count == 0:
            return HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0), {
                "phase_a_decision": 1.0,
                "admit_selected": 0.0,
                "overlay_selected": 0.0,
                "puncture_selected": 0.0,
                "feasible_admit_count": 0.0,
                "band_hit": 0.0,
                "band_fallback_used": 0.0,
                "keep_selected_due_to_no_feasible_admit": 1.0,
                "selected_loss": 0.0,
                "selected_throughput": reference_rate,
                "selected_admission": 0.0,
            }

        band_candidates = [item for item in feasible if band_min <= float(item["admission"]) <= band_max]
        if band_candidates:
            best = max(band_candidates, key=lambda item: float(item["throughput"]))
            band_hit = 1.0
            band_fallback_used = 0.0
        else:
            min_dist = min(float(item["band_distance"]) for item in feasible)
            closest = [item for item in feasible if abs(float(item["band_distance"]) - min_dist) <= 1.0e-12]
            best = max(closest, key=lambda item: float(item["throughput"]))
            band_hit = 0.0
            band_fallback_used = 1.0

        best_mode = int(best["mode"])
        return best["action"], {
            "phase_a_decision": 1.0,
            "admit_selected": 1.0,
            "overlay_selected": float(best_mode == MODE_OVERLAY),
            "puncture_selected": float(best_mode == MODE_PUNCTURE),
            "feasible_admit_count": float(feasible_admit_count),
            "band_hit": float(band_hit),
            "band_fallback_used": float(band_fallback_used),
            "keep_selected_due_to_no_feasible_admit": 0.0,
            "selected_loss": float(best["loss"]),
            "selected_throughput": float(best["throughput"]),
            "selected_admission": float(best["admission"]),
        }

    def _selected_embb_retention_ratio(self, candidate: Optional[CandidatePacket], mode: int) -> float:
        if candidate is None or mode == MODE_KEEP:
            return 1.0
        reference_rate = self._reference_embb_total_rate_for_local_shaping()
        retained_rate = max(reference_rate - float(candidate.loss_for_mode(mode)), 0.0)
        return float(np.clip(retained_rate / max(reference_rate, 1.0e-9), 0.0, 1.0))

    def _best_local_embb_retention_ratio(self, candidate: Optional[CandidatePacket]) -> float:
        if candidate is None:
            return 1.0
        best_retention = 1.0
        if bool(candidate.overlay_feasible):
            best_retention = max(best_retention, float(np.clip(candidate.overlay_retention, 0.0, 1.0)))
        if bool(candidate.puncture_feasible):
            best_retention = max(best_retention, self._selected_embb_retention_ratio(candidate, MODE_PUNCTURE))
        return float(np.clip(best_retention, 0.0, 1.0))

    def _selected_utility_gap(self, candidate: Optional[CandidatePacket], mode: int) -> float:
        if candidate is None or mode == MODE_KEEP:
            return 0.0
        # KEEP is the zero-action reference, so positive utility means the one-step
        # admission/mode decision is locally throughput-safe enough to help.
        return float(candidate.utility_for_mode(mode))

    def _apply_low_damage_candidate_constraints(
        self,
        candidates: List[CandidatePacket],
        actual_load: Optional[float] = None,
    ) -> List[CandidatePacket]:
        if not bool(getattr(self.rl_cfg.training, "low_damage_admission_objective", False)):
            return candidates
        actual = self._current_actual_load() if actual_load is None else float(actual_load)
        puncture_loss_ceiling_mbps = self._current_puncture_loss_ceiling(actual)
        overlay_gate = self._current_overlay_retention_gate(actual)
        filtered: List[CandidatePacket] = []
        for candidate in candidates:
            if candidate.puncture_feasible:
                self.puncture_candidate_total += 1
                puncture_loss_mbps = float(candidate.puncture_loss) / 1.0e6
                if puncture_loss_mbps > puncture_loss_ceiling_mbps + 1e-12:
                    candidate.puncture_feasible = False
                    candidate.puncture_utility = float("-inf")
                    self.puncture_candidate_pruned_by_loss_ceiling_count += 1
                elif candidate.overlay_feasible and float(candidate.overlay_retention) >= overlay_gate:
                    candidate.puncture_feasible = False
                    candidate.puncture_utility = float("-inf")
                    self.puncture_candidate_overlay_suppressed_count += 1
            if candidate.overlay_feasible or candidate.puncture_feasible:
                filtered.append(candidate)
        return filtered

    def _counterfactual_local_reward(
            self,
            candidate: Optional[CandidatePacket],
            mode: int,
            minislot: int,
            power_delta: float = 0.0,
        ) -> Tuple[float, float, Dict[str, float], Dict[str, object]]:
            if candidate is None or mode == MODE_KEEP:
                return 0.0, 0.0, {}, {}

            required_power = candidate.required_power_for_mode(mode)
            actual_power, projection_info = self._project_actual_power(required_power, float(power_delta), return_info=True)
            requested_power = float(projection_info["requested_power"])
            actual_power = float(actual_power)
            base_rate_scale = max(candidate.puncture_loss, 1e-9)
            damage_norm = float(np.clip(candidate.loss_for_mode(mode) / base_rate_scale, 0.0, 1.5))
            urgency = self._candidate_urgency(candidate.packet_id, minislot)
            projection_norm = float(np.clip(abs(actual_power - requested_power) / max(self.algo_cfg.power_upper_bound, 1e-12), 0.0, 1.0))
            power_norm = float(np.clip(actual_power / max(self.algo_cfg.power_upper_bound, 1e-12), 0.0, 1.5))
            load_reward_weights = self._get_load_aware_reward_weights(self._current_actual_load())
            mode_anchor_strength = float(self._hard_mode_anchor_strength())
            safe_puncture_available = bool(self._candidate_supports_safe_puncture_anchor(candidate))
            effective_overlay_gain_weight = float(load_reward_weights["overlay_gain_weight"] * (1.0 - mode_anchor_strength))
            effective_overlay_margin_weight = float(self.rl_cfg.reward.overlay_margin_weight * (1.0 - mode_anchor_strength))
            effective_missed_overlay_penalty = float(load_reward_weights["missed_overlay_penalty"] * (1.0 - mode_anchor_strength))
            effective_puncture_extra_penalty = float(
                load_reward_weights["puncture_extra_penalty"] * (0.20 + 0.80 * (1.0 - mode_anchor_strength))
            )

            reward_terms = {
                "schedule_success": self.rl_cfg.reward.schedule_success_weight,
                "urgency_bonus": self.rl_cfg.reward.urgency_reward_weight * urgency,
                "embb_damage": -self.rl_cfg.reward.embb_damage_weight * damage_norm,
                "power_penalty": -self.rl_cfg.reward.power_penalty_scale * power_norm,
                "power_projection_penalty": -self.rl_cfg.reward.power_projection_penalty * projection_norm,
            }
            urllc_tx_power_penalty_scale = float(getattr(self.rl_cfg.reward, "urllc_tx_power_penalty_scale", 0.0))
            if urllc_tx_power_penalty_scale > 0.0:
                reward_terms["urllc_tx_power_penalty"] = -urllc_tx_power_penalty_scale * power_norm

            admitted_packet_count = 1.0
            selected_retention = self._selected_embb_retention_ratio(candidate, mode)
            utility_gap = self._selected_utility_gap(candidate, mode)
            safe_retention_threshold = float(getattr(self.rl_cfg.reward, "safe_embb_retention_threshold", 0.0) or 0.0)
            unsafe_retention_threshold = float(getattr(self.rl_cfg.reward, "unsafe_embb_retention_threshold", 0.0) or 0.0)
            safe_admission_bonus_weight = float(getattr(self.rl_cfg.reward, "safe_admission_bonus_weight", 0.0) or 0.0)
            unsafe_admission_penalty_weight = float(getattr(self.rl_cfg.reward, "unsafe_admission_penalty_weight", 0.0) or 0.0)
            puncture_safe_bonus_extra_weight = float(getattr(self.rl_cfg.reward, "puncture_safe_bonus_extra_weight", 0.0) or 0.0)
            negative_gap_admission_penalty_weight = float(
                getattr(self.rl_cfg.reward, "negative_gap_admission_penalty_weight", 0.0) or 0.0
            )
            safe_admission = (
                safe_retention_threshold > 0.0
                and selected_retention >= safe_retention_threshold - 1e-9
                and utility_gap > 1e-9
            )
            unsafe_admission = (
                (unsafe_retention_threshold > 0.0 and selected_retention < unsafe_retention_threshold - 1e-9)
                or utility_gap <= 1e-9
            )
            if safe_admission_bonus_weight > 0.0 and safe_admission:
                reward_terms["safe_admission_bonus"] = safe_admission_bonus_weight * admitted_packet_count
            if unsafe_admission_penalty_weight > 0.0 and unsafe_admission:
                reward_terms["unsafe_admission_penalty"] = -unsafe_admission_penalty_weight * admitted_packet_count
            if negative_gap_admission_penalty_weight > 0.0 and utility_gap <= 1e-9:
                reward_terms["negative_gap_admission_penalty"] = (
                    -negative_gap_admission_penalty_weight * admitted_packet_count
                )
            if puncture_safe_bonus_extra_weight > 0.0 and mode == MODE_PUNCTURE and safe_admission:
                reward_terms["puncture_safe_bonus_extra"] = (
                    puncture_safe_bonus_extra_weight * admitted_packet_count
                )
            local_embb_opportunity_cost_weight = float(
                getattr(self.rl_cfg.reward, "local_embb_opportunity_cost_weight", 0.0) or 0.0
            )
            if (
                bool(getattr(self.rl_cfg.reward, "use_local_embb_opportunity_cost_term", False))
                and local_embb_opportunity_cost_weight > 0.0
            ):
                selected_embb_retention = self._selected_embb_retention_ratio(candidate, mode)
                best_embb_retention = self._best_local_embb_retention_ratio(candidate)
                opportunity_gap = max(best_embb_retention - selected_embb_retention, 0.0)
                quota_gate = 1.0
                if bool(getattr(self.rl_cfg.reward, "local_embb_opportunity_cost_gate_by_quota", True)):
                    progress_summary = self._phase_a_progress_summary(minislot)
                    quota_pressure = float(
                        np.clip(progress_summary.get("required_admit_rate_per_remaining_cell", 0.0), 0.0, 1.0)
                    )
                    quota_gate = float(0.25 + 0.75 * (1.0 - quota_pressure))
                if opportunity_gap > 1.0e-9:
                    reward_terms["local_embb_opportunity_cost"] = (
                        -local_embb_opportunity_cost_weight * quota_gate * opportunity_gap * admitted_packet_count
                    )
            admission_band_bonus_weight = float(getattr(self.rl_cfg.reward, "admission_band_bonus_weight", 0.0) or 0.0)
            admission_band_penalty_weight = float(getattr(self.rl_cfg.reward, "admission_band_penalty_weight", 0.0) or 0.0)
            target_admission_mid = float(self._current_target_admission_mid(self._current_actual_load()))
            target_admission_tol = float(self._current_target_admission_tol(self._current_actual_load()))
            if (admission_band_bonus_weight > 0.0 or admission_band_penalty_weight > 0.0) and target_admission_mid > 0.0:
                current_scheduled_packets = int(np.count_nonzero(self.scheduled_uavs >= 0))
                projected_scheduled_packets = current_scheduled_packets + int(mode != MODE_KEEP)
                projected_admission_ratio = float(projected_scheduled_packets / max(self.num_packets, 1)) if self.num_packets > 0 else 1.0
                admission_gap = abs(projected_admission_ratio - target_admission_mid)
                if target_admission_tol > 1.0e-9 and admission_gap <= target_admission_tol + 1.0e-9:
                    band_score = 1.0 - float(admission_gap / max(target_admission_tol, 1.0e-9))
                    if admission_band_bonus_weight > 0.0 and band_score > 0.0:
                        reward_terms["admission_band_bonus"] = admission_band_bonus_weight * band_score
                elif admission_band_penalty_weight > 0.0:
                    excess_gap = max(admission_gap - target_admission_tol, 0.0)
                    reward_terms["admission_band_penalty"] = -admission_band_penalty_weight * excess_gap

            if mode == MODE_OVERLAY:
                overlay_gain_norm = float(np.clip((candidate.puncture_loss - candidate.overlay_loss) / max(candidate.puncture_loss, 1e-9), -1.0, 1.0))
                overlay_margin_norm = float(np.clip((candidate.overlay_utility - candidate.puncture_utility) / 1.0e6, -1.0, 1.0))
                reward_terms["overlay_gain"] = effective_overlay_gain_weight * overlay_gain_norm
                reward_terms["overlay_margin"] = effective_overlay_margin_weight * max(overlay_margin_norm, 0.0)
                safe_puncture_preference_penalty_weight = float(
                    getattr(self.rl_cfg.reward, "safe_puncture_preference_penalty_weight", 0.0) or 0.0
                )
                overlay_margin_needed_to_override_puncture = float(
                    getattr(self.rl_cfg.reward, "overlay_margin_needed_to_override_puncture", 0.0) or 0.0
                )
                puncture_loss_safe_threshold = float(
                    getattr(self.rl_cfg.reward, "puncture_loss_safe_threshold", 0.0) or 0.0
                )
                puncture_reliability_target = float(1.0 - self.urllc_cfg.target_error_probability)
                puncture_reliable = bool(float(candidate.puncture_reliability) >= puncture_reliability_target - 1.0e-9)
                puncture_loss_safe = bool(
                    puncture_loss_safe_threshold <= 0.0
                    or float(candidate.puncture_loss) <= puncture_loss_safe_threshold + 1.0e-9
                )
                overlay_not_clearly_better = bool(
                    float(candidate.overlay_utility)
                    <= float(candidate.puncture_utility) + overlay_margin_needed_to_override_puncture + 1.0e-9
                )
                if (
                    safe_puncture_preference_penalty_weight > 0.0
                    and bool(candidate.overlay_feasible)
                    and bool(candidate.puncture_feasible)
                    and puncture_reliable
                    and puncture_loss_safe
                    and overlay_not_clearly_better
                ):
                    overlay_margin_shortfall = max(
                        overlay_margin_needed_to_override_puncture
                        - float(candidate.overlay_utility - candidate.puncture_utility),
                        0.0,
                    )
                    if overlay_margin_needed_to_override_puncture > 1.0e-9:
                        preference_penalty_scale = float(
                            np.clip(
                                overlay_margin_shortfall / overlay_margin_needed_to_override_puncture,
                                0.0,
                                1.0,
                            )
                        )
                    else:
                        preference_penalty_scale = 1.0
                    reward_terms["safe_puncture_preference_penalty"] = (
                        -safe_puncture_preference_penalty_weight * preference_penalty_scale
                    )
                overlay_when_safe_puncture_penalty_weight = float(
                    getattr(self.rl_cfg.reward, "overlay_when_safe_puncture_penalty_weight", 0.0) or 0.0
                )
                if overlay_when_safe_puncture_penalty_weight > 0.0 and safe_puncture_available and mode_anchor_strength > 0.0:
                    reward_terms["overlay_when_safe_puncture_penalty"] = (
                        -overlay_when_safe_puncture_penalty_weight * mode_anchor_strength
                    )
                overlay_retention_gate_bonus_weight = float(
                    getattr(self.rl_cfg.reward, "overlay_retention_gate_bonus_weight", 0.0)
                )
                if overlay_retention_gate_bonus_weight > 0.0 and candidate.overlay_feasible:
                    overlay_retention_threshold = max(
                        float(getattr(self.rl_cfg.reward, "overlay_retention_gate_bonus_threshold", 0.0) or 0.0),
                        float(self._current_overlay_retention_gate(self._current_actual_load())),
                    )
                    if float(candidate.overlay_retention) >= overlay_retention_threshold - 1e-9:
                        reward_terms["overlay_retention_gate_bonus"] = (
                            overlay_retention_gate_bonus_weight * float(candidate.overlay_retention)
                        )
            elif mode == MODE_PUNCTURE:
                reward_terms["puncture_extra_penalty"] = -effective_puncture_extra_penalty
                puncture_mode_usage_penalty = float(getattr(self.rl_cfg.reward, "puncture_mode_usage_penalty", 0.0))
                if puncture_mode_usage_penalty > 0.0:
                    reward_terms["puncture_mode_usage_penalty"] = -puncture_mode_usage_penalty
                selected_puncture_loss_penalty_weight = float(
                    getattr(self.rl_cfg.reward, "selected_puncture_loss_penalty_weight", 0.0)
                )
                if selected_puncture_loss_penalty_weight > 0.0:
                    reward_terms["selected_puncture_loss_penalty"] = (
                        -selected_puncture_loss_penalty_weight * (float(candidate.puncture_loss) / 1.0e6)
                    )
                safe_puncture_bonus_weight = float(
                    getattr(self.rl_cfg.reward, "safe_puncture_bonus_weight", 0.0) or 0.0
                )
                if safe_puncture_bonus_weight > 0.0 and safe_puncture_available and mode_anchor_strength > 0.0:
                    reward_terms["safe_puncture_bonus"] = (
                        safe_puncture_bonus_weight * mode_anchor_strength
                    )
                if candidate.overlay_feasible and candidate.overlay_utility > candidate.puncture_utility:
                    overlay_gap_norm = float(np.clip((candidate.overlay_utility - candidate.puncture_utility) / 1.0e6, 0.0, 1.0))
                    reward_terms["missed_overlay_penalty"] = -effective_missed_overlay_penalty * overlay_gap_norm

            reward = float(sum(reward_terms.values()))
            return reward, float(actual_power), reward_terms, projection_info

    def _project_actual_power(self, required_power: float, power_delta: float, return_info: bool = False):
            raw_power_delta = float(power_delta)
            clipped_power_delta = float(np.clip(raw_power_delta, -1.0, 1.0))
            delta_was_clipped = abs(clipped_power_delta - raw_power_delta) > 1e-9
            quantized_power_delta = clipped_power_delta
            used_discrete_bin = False
            if self.rl_cfg.action.bootstrap_with_discrete_power and self.rl_cfg.action.initial_power_bins > 1:
                bins = int(self.rl_cfg.action.initial_power_bins)
                grid = np.linspace(-1.0, 1.0, bins)
                quantized_power_delta = float(grid[np.argmin(np.abs(grid - clipped_power_delta))])
                used_discrete_bin = abs(quantized_power_delta - clipped_power_delta) > 1e-9

            power_upper_bound = float(self.algo_cfg.power_upper_bound)
            pos_scale = float(max(getattr(self.rl_cfg.action, "power_delta_pos_scale", 1.0), 0.0))
            neg_scale = float(max(getattr(self.rl_cfg.action, "power_delta_neg_scale", 1.0), 1.0e-12))
            delta = quantized_power_delta
            if delta >= 0.0:
                headroom = max(power_upper_bound - required_power, 0.0)
                positive_fraction = float(np.clip(pos_scale * delta, 0.0, 1.0))
                requested_power = float(required_power + positive_fraction * headroom)
            else:
                requested_power = float(required_power * (1.0 + neg_scale * delta))
            requested_power = float(max(0.0, requested_power))

            actual_power = max(0.0, min(requested_power, power_upper_bound))
            hit_upper_bound = actual_power + 1e-12 < requested_power
            hit_feasible_floor = False
            if self.rl_cfg.shield.force_power_to_feasible_minimum:
                floored_power = max(actual_power, required_power)
                hit_feasible_floor = floored_power > actual_power + 1e-12
                actual_power = floored_power

            if actual_power >= required_power - 1e-12:
                headroom = max(power_upper_bound - required_power, 0.0)
                if headroom > 1e-12 and pos_scale > 1e-12:
                    executed_delta = float((actual_power - required_power) / headroom / pos_scale)
                    executed_delta = float(np.clip(executed_delta, 0.0, 1.0))
                else:
                    executed_delta = 0.0
            elif required_power > 1e-12:
                executed_delta = float((actual_power / required_power - 1.0) / neg_scale)
                executed_delta = float(np.clip(executed_delta, -1.0, 0.0))
            else:
                executed_delta = 0.0

            info = {
                "required_power": float(required_power),
                "raw_delta": raw_power_delta,
                "clipped_delta": clipped_power_delta,
                "quantized_delta": quantized_power_delta,
                "delta_was_clipped": bool(delta_was_clipped),
                "used_discrete_bin": bool(used_discrete_bin),
                "requested_power": float(requested_power),
                "actual_power": float(actual_power),
                "hit_upper_bound": bool(hit_upper_bound),
                "hit_feasible_floor": bool(hit_feasible_floor),
                "executed_delta": float(executed_delta),
            }
            if return_info:
                return float(actual_power), info
            return float(actual_power)

    def _build_joint_options_for_agent(self, shielded_action, observation: AgentObservation, uav_idx: int, rb: int, minislot: int) -> List:
            options = [
                ShieldedAction(
                    action=HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0),
                    candidate=None,
                    utility=0.0,
                )
            ]
            candidate_pool = list(observation.candidates)

            def _candidate_priority(candidate: CandidatePacket) -> Tuple[float, float, float, float]:
                release = int(self.packet_release_minislots[candidate.packet_id]) if candidate.packet_id < self.packet_release_minislots.size else 0
                age = float(max(minislot - release, 0))
                feasible = float(candidate.overlay_feasible or candidate.puncture_feasible)
                overlay_margin = float(candidate.overlay_utility - candidate.puncture_utility) if candidate.overlay_feasible else float("-inf")
                return (
                    age,
                    feasible,
                    float(candidate.overlay_feasible),
                    overlay_margin,
                )

            candidate_pool.sort(key=_candidate_priority, reverse=True)
            seen = set()
            raw_packet_id = getattr(shielded_action.candidate, "packet_id", None)
            raw_mode = int(shielded_action.action.mode)
            raw_power_delta = float(shielded_action.action.power_delta)
            for candidate in candidate_pool:
                packet_option = 0
                for idx, obs_candidate in enumerate(observation.candidates, start=1):
                    if obs_candidate.packet_id == candidate.packet_id:
                        packet_option = idx
                        break
                if packet_option == 0:
                    continue
                for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                    if not candidate.is_mode_feasible(mode):
                        continue
                    signature = (candidate.packet_id, mode)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    option_power_delta = raw_power_delta if (raw_packet_id == candidate.packet_id and raw_mode == mode) else 0.0
                    options.append(
                        ShieldedAction(
                            action=HybridAction(mode=mode, packet_option=packet_option, power_delta=option_power_delta),
                            candidate=candidate,
                            utility=float(candidate.utility_for_mode(mode)),
                            mode_corrected=(candidate.packet_id != getattr(shielded_action.candidate, 'packet_id', None)) or (mode != shielded_action.action.mode),
                        )
                    )
            return options

    def _joint_combo_reliability(self, rb: int, minislot: int, combo: Dict[str, object]) -> Tuple[bool, Dict[str, float]]:
            active = []
            for uav_idx, agent_id in enumerate(self.agent_ids):
                shielded_action = combo[agent_id]
                if shielded_action.candidate is None or shielded_action.action.mode == MODE_KEEP:
                    continue
                candidate = shielded_action.candidate
                active.append({
                    'uav_idx': uav_idx,
                    'candidate': candidate,
                    'mode': int(shielded_action.action.mode),
                    'power': self._project_actual_power(
                        candidate.required_power_for_mode(int(shielded_action.action.mode)),
                        shielded_action.action.power_delta,
                    ),
                })

            reliabilities: Dict[str, float] = {}
            for action in active:
                uav_idx = action['uav_idx']
                candidate = action['candidate']
                mode = action['mode']
                source_user = int(candidate.source_user)
                urllc_gain = float(self.channel_gains_mag_sq[source_user, uav_idx, rb])
                local_interference = 0.0
                embb_owner = int(candidate.embb_owner_for_mode(mode))
                if mode == MODE_OVERLAY and embb_owner >= 0:
                    embb_user_idx = int(candidate.embb_user_idx_for_mode(mode))
                    embb_per_rb_power = self._embb_per_rb_power_for_owner(uav_idx, embb_owner, rb, minislot)
                    embb_gain = float(self.channel_gains_mag_sq[embb_user_idx, uav_idx, rb])
                    local_interference = embb_per_rb_power * embb_gain

                intercell = 0.0
                for other_uav in range(self.sys_cfg.num_uavs):
                    if other_uav == uav_idx:
                        continue
                    other_action = combo[self.agent_ids[other_uav]]
                    other_mode = int(other_action.action.mode)
                    other_owner = int(self.owner_per_uav_rb[other_uav, rb])
                    if other_action.candidate is not None and other_mode == MODE_OVERLAY:
                        other_owner = int(other_action.candidate.embb_owner_for_mode(other_mode))
                    if other_owner >= 0 and other_mode != MODE_PUNCTURE:
                        other_embb_idx = self.sys_cfg.num_urllc_users + other_owner
                        other_embb_power = self._embb_per_rb_power_for_owner(other_uav, other_owner, rb, minislot)
                        embb_cross_gain = float(self.channel_gains_mag_sq[other_embb_idx, uav_idx, rb])
                        intercell += other_embb_power * embb_cross_gain
                    if other_action.candidate is not None and other_mode != MODE_KEEP:
                        other_power = self._project_actual_power(
                            other_action.candidate.required_power_for_mode(other_mode),
                            other_action.action.power_delta,
                        )
                        cross_gain = float(self.channel_gains_mag_sq[int(other_action.candidate.source_user), uav_idx, rb])
                        intercell += other_power * cross_gain

                snir = action['power'] * urllc_gain / max(self.sys_cfg.noise_power + local_interference + intercell, 1e-15)
                packet_bits = self._packet_bits_for_user(source_user)
                error_prob = self.capacity_model.decoding_error_probability(
                    snir,
                    packet_bits,
                    self.sys_cfg.channel_uses_per_minislot,
                )
                reliability = float(1.0 - error_prob)
                reliabilities[self.agent_ids[uav_idx]] = reliability
                if reliability < (1.0 - self.urllc_cfg.target_error_probability):
                    return False, reliabilities

            return True, reliabilities

    def _enforce_joint_reliability(self, minislot: int, rb: int, observations: Dict[str, AgentObservation], shielded: Dict[str, object]) -> Dict[str, object]:
            if self.rl_cfg.env.multi_rb_agents:
                self._last_joint_reliabilities = {}
                return shielded
            active_agents = [
                agent_id
                for agent_id in self.agent_ids
                if shielded[agent_id].candidate is not None and shielded[agent_id].action.mode != MODE_KEEP
            ]
            if len(active_agents) == 0:
                self._last_joint_reliabilities = {}
                return shielded

            option_lists = []
            for uav_idx, agent_id in enumerate(self.agent_ids):
                option_lists.append(
                    self._build_joint_options_for_agent(
                        shielded[agent_id],
                        observations[agent_id],
                        uav_idx,
                        rb,
                        minislot,
                    )
                )
            best_combo = None
            best_key = None
            best_reliabilities = {}
            original_packet_counts = {}
            for agent_id in self.agent_ids:
                original = shielded[agent_id]
                if original.candidate is None or original.action.mode == MODE_KEEP:
                    continue
                packet_id = int(original.candidate.packet_id)
                original_packet_counts[packet_id] = original_packet_counts.get(packet_id, 0) + 1
            original_duplicate_packets = {packet_id for packet_id, count in original_packet_counts.items() if count > 1}

            def _search(agent_idx: int, current: Dict[str, object]):
                nonlocal best_combo, best_key, best_reliabilities
                if agent_idx >= len(self.agent_ids):
                    feasible, reliabilities = self._joint_combo_reliability(rb, minislot, current)
                    if not feasible:
                        return
                    scheduled_count = sum(
                        1 for agent_id in self.agent_ids
                        if current[agent_id].candidate is not None and current[agent_id].action.mode != MODE_KEEP
                    )
                    total_embb_loss = float(sum(
                        current[agent_id].candidate.loss_for_mode(current[agent_id].action.mode)
                        for agent_id in self.agent_ids
                        if current[agent_id].candidate is not None and current[agent_id].action.mode != MODE_KEEP
                    ))
                    overlay_count = sum(
                        1 for agent_id in self.agent_ids
                        if current[agent_id].candidate is not None and current[agent_id].action.mode == MODE_OVERLAY
                    )
                    total_utility = float(sum(current[agent_id].utility for agent_id in self.agent_ids))
                    puncture_count = sum(
                        1 for agent_id in self.agent_ids
                        if current[agent_id].candidate is not None and current[agent_id].action.mode == MODE_PUNCTURE
                    )
                    total_overlay_margin = float(sum(
                        max(
                            current[agent_id].candidate.overlay_utility - current[agent_id].candidate.puncture_utility,
                            0.0,
                        )
                        for agent_id in self.agent_ids
                        if current[agent_id].candidate is not None and current[agent_id].action.mode == MODE_OVERLAY
                    ))
                    primary_match_count = sum(
                        1
                        for local_agent_id in self.agent_ids
                        if current[local_agent_id].candidate is not None
                        and current[local_agent_id].action.mode != MODE_KEEP
                        and self._last_primary_assignment.get(
                            (
                                int(current[local_agent_id].candidate.rb_index),
                                int(current[local_agent_id].candidate.packet_id),
                            ),
                            -1,
                        ) == self._agent_index_map[local_agent_id][0]
                    )
                    norm_embb_loss = float(total_embb_loss / 1.0e7)
                    norm_overlay_margin = float(total_overlay_margin / 1.0e6)
                    norm_utility = float(total_utility / 1.0e6)
                    score = (
                        self.rl_cfg.env.joint_schedule_weight * float(scheduled_count)
                        + self.rl_cfg.env.joint_primary_match_bonus * float(primary_match_count)
                        + self.rl_cfg.env.joint_overlay_bonus * float(overlay_count)
                        + self.rl_cfg.env.joint_overlay_margin_weight * norm_overlay_margin
                        - self.rl_cfg.env.joint_puncture_penalty * float(puncture_count)
                        - self.rl_cfg.env.joint_embb_loss_weight * norm_embb_loss
                        + 0.10 * norm_utility
                    )
                    key = (
                        score,
                        scheduled_count,
                        primary_match_count,
                        overlay_count,
                        -puncture_count,
                        -norm_embb_loss,
                        norm_utility,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_reliabilities = dict(reliabilities)
                        best_combo = {
                            agent_id: ShieldedAction(
                                action=HybridAction(
                                    mode=current[agent_id].action.mode,
                                    packet_option=current[agent_id].action.packet_option,
                                    power_delta=current[agent_id].action.power_delta,
                                    embb_owner_option=current[agent_id].action.embb_owner_option,
                                    embb_power_delta=current[agent_id].action.embb_power_delta,
                                ),
                                candidate=current[agent_id].candidate,
                                utility=current[agent_id].utility,
                                used_greedy_fallback=current[agent_id].used_greedy_fallback,
                                collision_rewritten=current[agent_id].collision_rewritten,
                                mode_corrected=current[agent_id].mode_corrected,
                                packet_invalid_fallback=current[agent_id].packet_invalid_fallback,
                                mask_invalid_fallback=current[agent_id].mask_invalid_fallback,
                            )
                            for agent_id in self.agent_ids
                        }
                    return

                agent_id = self.agent_ids[agent_idx]
                for option in option_lists[agent_idx]:
                    if option.candidate is not None and option.action.mode != MODE_KEEP:
                        option_packet_id = int(option.candidate.packet_id)
                        packet_conflict = any(
                            current_action.candidate is not None
                            and current_action.action.mode != MODE_KEEP
                            and int(current_action.candidate.packet_id) == option_packet_id
                            for current_action in current.values()
                        )
                        if packet_conflict:
                            continue
                    current[agent_id] = option
                    _search(agent_idx + 1, current)
                current.pop(agent_id, None)

            _search(0, {})
            if best_combo is None:
                self._last_joint_reliabilities = {}
                return shielded

            self._last_joint_reliabilities = dict(best_reliabilities)
            adjusted = {}
            for agent_id in self.agent_ids:
                original = shielded[agent_id]
                final = best_combo[agent_id]
                original_packet = (
                    int(original.candidate.packet_id)
                    if original.candidate is not None and original.action.mode != MODE_KEEP
                    else None
                )
                final_packet = (
                    int(final.candidate.packet_id)
                    if final.candidate is not None and final.action.mode != MODE_KEEP
                    else None
                )
                changed = (
                    original.action.mode != final.action.mode
                    or original.action.packet_option != final.action.packet_option
                    or abs(float(original.action.power_delta) - float(final.action.power_delta)) > 1e-8
                )
                final.used_greedy_fallback = original.used_greedy_fallback or final.used_greedy_fallback
                final.collision_rewritten = (
                    original.collision_rewritten
                    or final.collision_rewritten
                    or (original_packet in original_duplicate_packets and original_packet != final_packet)
                )
                final.packet_invalid_fallback = original.packet_invalid_fallback or final.packet_invalid_fallback
                final.mask_invalid_fallback = original.mask_invalid_fallback or final.mask_invalid_fallback
                final.mode_corrected = original.mode_corrected or final.mode_corrected
                final.joint_reliability_rewritten = changed
                adjusted[agent_id] = final
            return adjusted

    def _best_local_candidate(self, candidates: List[CandidatePacket]) -> Tuple[Optional[CandidatePacket], int, float]:
            if not candidates:
                return None, MODE_KEEP, 0.0
            best = max(candidates, key=lambda item: item.best_utility)
            if not np.isfinite(best.best_utility) or best.best_mode == MODE_KEEP:
                return None, MODE_KEEP, 0.0
            return best, best.best_mode, best.best_utility

    def _retention_proxy_for_mode(self, candidate: CandidatePacket, mode: int) -> float:
            if mode == MODE_OVERLAY:
                return float(candidate.overlay_retention)
            if mode != MODE_PUNCTURE:
                return 0.0
            owner = int(candidate.embb_owner_for_mode(mode))
            rb = int(candidate.rb_index if candidate.rb_index >= 0 else self._current_cell()[1])
            base_rate = self._base_rate_for_cell(int(candidate.associated_uav), owner, rb)
            if base_rate <= 1e-9:
                return 0.0
            return float(np.clip(1.0 - candidate.loss_for_mode(mode) / max(base_rate, 1e-9), 0.0, 1.0))

    def _best_local_candidate_throughput_first(
            self,
            candidates: List[CandidatePacket],
        ) -> Tuple[Optional[CandidatePacket], int, float]:
            best_candidate = None
            best_mode = MODE_KEEP
            best_key: Optional[Tuple[float, ...]] = None

            for candidate in candidates:
                for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                    if not candidate.is_mode_feasible(mode):
                        continue
                    embb_loss = float(candidate.loss_for_mode(mode))
                    retention = self._retention_proxy_for_mode(candidate, mode)
                    required_power = float(candidate.required_power_for_mode(mode))
                    reliability = float(candidate.reliability_for_mode(mode))
                    key = (
                        -embb_loss,
                        retention,
                        -required_power,
                        reliability,
                        1.0 if mode == MODE_OVERLAY else 0.0,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_candidate = candidate
                        best_mode = mode

            if best_candidate is None or best_key is None:
                return None, MODE_KEEP, 0.0
            return best_candidate, best_mode, float(best_key[0])

    def _best_local_candidate_load_aware_balanced(
            self,
            candidates: List[CandidatePacket],
            actual_load: Optional[float] = None,
        ) -> Tuple[Optional[CandidatePacket], int, float]:
            actual = self._current_actual_load() if actual_load is None else float(actual_load)
            load_bucket = nearest_reference_load(actual)
            feasible_entries = []
            puncture_entries = []
            overlay_entries = []

            for candidate in candidates:
                for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                    if not candidate.is_mode_feasible(mode):
                        continue
                    loss = float(candidate.loss_for_mode(mode))
                    retention = float(self._retention_proxy_for_mode(candidate, mode))
                    power = float(candidate.required_power_for_mode(mode))
                    reliability = float(candidate.reliability_for_mode(mode))
                    utility = float(candidate.utility_for_mode(mode))
                    gain_proxy = float((1.0 - np.clip(loss / max(candidate.puncture_loss, 1e-9), 0.0, 1.0)) + retention)
                    entry = {
                        "candidate": candidate,
                        "mode": mode,
                        "loss": loss,
                        "retention": retention,
                        "power": power,
                        "reliability": reliability,
                        "utility": utility,
                        "gain_proxy": gain_proxy,
                        "overlay_retention": float(candidate.overlay_retention if mode == MODE_OVERLAY else 0.0),
                    }
                    feasible_entries.append(entry)
                    if mode == MODE_OVERLAY:
                        overlay_entries.append(entry)
                    else:
                        puncture_entries.append(entry)

            if not feasible_entries:
                return None, MODE_KEEP, 0.0
            max_loss = max(float(item["loss"]) for item in feasible_entries) if feasible_entries else 1.0

            def _pick_best(entries: List[Dict], key_fn):
                if not entries:
                    return None
                return max(entries, key=key_fn)

            if load_bucket <= 10.0:
                if overlay_entries:
                    best = _pick_best(
                        overlay_entries,
                        lambda item: (
                            item["gain_proxy"],
                            item["retention"],
                            item["utility"],
                            -item["loss"],
                            -item["power"],
                            item["reliability"],
                        ),
                    )
                    return best["candidate"], best["mode"], float(best["gain_proxy"])
                if puncture_entries:
                    min_loss = min(item["loss"] for item in puncture_entries)
                    low_loss = [
                        item for item in puncture_entries
                        if item["loss"] <= min_loss * 1.15 + 1e-9
                    ]
                    best = _pick_best(
                        low_loss or puncture_entries,
                        lambda item: (
                            item["gain_proxy"],
                            -item["loss"],
                            item["reliability"],
                            -item["power"],
                        ),
                    )
                    return best["candidate"], best["mode"], float(best["gain_proxy"])

            if load_bucket == 15.0:
                best = _pick_best(
                    feasible_entries,
                    lambda item: (
                        item["gain_proxy"]
                        - 0.9 * item["loss"] / max(max_loss, 1.0e-9)
                        + 0.7 * item["overlay_retention"]
                        + 0.2 * (1.0 if item["mode"] == MODE_OVERLAY else 0.0),
                        item["retention"],
                        -item["loss"],
                        -item["power"],
                    ),
                )
                balanced_score = (
                    best["gain_proxy"]
                    - 0.9 * best["loss"] / max(max_loss, 1.0e-9)
                    + 0.7 * best["overlay_retention"]
                    + 0.2 * (1.0 if best["mode"] == MODE_OVERLAY else 0.0)
                )
                return best["candidate"], best["mode"], float(balanced_score)

            if overlay_entries:
                best = _pick_best(
                    overlay_entries,
                    lambda item: (
                        item["retention"],
                        -item["loss"],
                        item["reliability"],
                        -item["power"],
                    ),
                )
                return best["candidate"], best["mode"], float(best["retention"])

            puncture_entries.sort(key=lambda item: item["loss"])
            puncture_losses = np.asarray([item["loss"] for item in puncture_entries], dtype=float)
            low_loss_threshold = float(np.quantile(puncture_losses, 0.35)) if puncture_losses.size > 1 else float(puncture_losses[0])
            low_loss_entries = [
                item for item in puncture_entries
                if item["loss"] <= low_loss_threshold + 1e-9
            ]
            best = _pick_best(
                low_loss_entries or puncture_entries,
                lambda item: (
                    -item["loss"],
                    item["reliability"],
                    -item["power"],
                ),
            )
            return best["candidate"], best["mode"], float(-best["loss"])

    def _best_local_candidate_frontier_throughput_admission(
            self,
            candidates: List[CandidatePacket],
            actual_load: Optional[float] = None,
        ) -> Tuple[Optional[CandidatePacket], int, float]:
            actual = self._current_actual_load() if actual_load is None else float(actual_load)
            admission_floor = float(self._current_frontier_oracle_admission_floor(actual))
            current_admitted = int(np.count_nonzero(self.scheduled_uavs >= 0))
            current_admission_ratio = float(current_admitted / max(self.num_packets, 1)) if self.num_packets > 0 else 1.0
            below_admission_floor = current_admission_ratio < admission_floor - 1.0e-9

            if below_admission_floor:
                candidate, mode, score = self._best_local_candidate_load_aware_balanced(
                    candidates,
                    actual_load=actual,
                )
                if candidate is not None and mode != MODE_KEEP:
                    return candidate, mode, float(score)

            feasible_entries = []
            for candidate in candidates:
                for mode in (MODE_OVERLAY, MODE_PUNCTURE):
                    if not candidate.is_mode_feasible(mode):
                        continue
                    utility = float(candidate.utility_for_mode(mode))
                    if utility <= 1.0e-9:
                        continue
                    retention = float(self._retention_proxy_for_mode(candidate, mode))
                    loss = float(candidate.loss_for_mode(mode))
                    power = float(candidate.required_power_for_mode(mode))
                    reliability = float(candidate.reliability_for_mode(mode))
                    high_load_bonus = 0.0
                    if actual >= 15.0 and mode == MODE_PUNCTURE:
                        high_load_bonus = 0.10
                    score = (
                        1.20 * utility / 1.0e6
                        + 0.55 * retention
                        - 0.18 * loss / 1.0e6
                        - 0.08 * power / max(self.algo_cfg.power_upper_bound, 1.0e-9)
                        + 0.05 * reliability
                        + high_load_bonus
                    )
                    feasible_entries.append((score, candidate, mode))

            if feasible_entries:
                score, candidate, mode = max(feasible_entries, key=lambda item: item[0])
                return candidate, mode, float(score)

            return self._best_local_candidate_throughput_first(candidates)

    def _planning_teacher_action(self, observation: AgentObservation) -> HybridAction:
            valid_owner_options = np.where(observation.masks.embb_owner_mask > 0)[0]
            embb_owner_option = 0
            positive_options = valid_owner_options[valid_owner_options > 0]
            if positive_options.size > 0:
                embb_owner_option = int(positive_options[0])
            elif valid_owner_options.size > 0:
                embb_owner_option = int(valid_owner_options[0])
            return HybridAction(
                mode=MODE_KEEP,
                packet_option=0,
                power_delta=0.0,
                embb_owner_option=embb_owner_option,
                embb_power_delta=0.0,
            )

    def bc_teacher_action(
            self,
            agent_id: str,
            observation: AgentObservation,
            teacher_policy: str = "greedy_reference",
        ) -> HybridAction:
            planning_phase = bool(observation.metadata.get("planning_phase", 0.0) > 0.5)
            if planning_phase:
                return self._planning_teacher_action(observation)

            policy = str(teacher_policy or "greedy_reference").strip().lower()
            if policy in {"throughput_feasible_oracle", "oracle", "throughput_first"}:
                candidate, mode, _score = self._best_local_candidate_throughput_first(observation.candidates)
            elif policy == "load_aware_balanced_oracle":
                candidate, mode, _score = self._best_local_candidate_load_aware_balanced(
                    observation.candidates,
                    actual_load=self._current_actual_load(),
                )
            elif policy in {"frontier_throughput_admission_oracle", "frontier_oracle", "tp_admission_frontier"}:
                candidate, mode, _score = self._best_local_candidate_frontier_throughput_admission(
                    observation.candidates,
                    actual_load=self._current_actual_load(),
                )
            else:
                candidate, mode, _score = self._best_local_candidate(observation.candidates)

            packet_option = 0
            if candidate is not None and mode != MODE_KEEP:
                try:
                    packet_option = observation.candidates.index(candidate) + 1
                except ValueError:
                    packet_option = 0

            return HybridAction(
                mode=int(mode),
                packet_option=int(packet_option),
                power_delta=0.0,
                embb_owner_option=0,
                embb_power_delta=0.0,
            )

    def _base_rate_for_cell(self, uav_idx: int, owner: int, rb: int) -> float:
            if owner < 0:
                return 0.0
            owner_uav = int(self.embb_selected_uavs[owner])
            if owner_uav != uav_idx:
                return 0.0
            if self.embb_base_rb_rates_per_uav_rb is not None:
                return float(
                    self.embb_base_rb_rates_per_uav_rb[uav_idx, rb] / max(self.sys_cfg.num_minislots, 1)
                )
            return float(self.embb_base_rb_rates[rb] / max(self.sys_cfg.num_minislots, 1))

    def _packet_bits_for_user(self, user_idx: int) -> int:
            lengths = self.urllc_cfg.packet_lengths
            return int(lengths[user_idx % len(lengths)])

    @staticmethod
    def _safe_metric(value: float, default: float = 0.0, low: float = -1.0e6, high: float = 1.0e6) -> float:
            if not np.isfinite(value):
                return float(default)
            return float(np.clip(value, low, high))

    def _actual_embb_owner_for_cell(self, uav_idx: int, rb_idx: int, minislot: int) -> int:
            if self.embb_owner_grid is not None:
                return int(self.embb_owner_grid[uav_idx, rb_idx, minislot])
            return int(self.owner_per_uav_rb[uav_idx, rb_idx])

    def _baseline_embb_owner_for_cell(self, uav_idx: int, rb_idx: int, minislot: int) -> int:
            if self.owner_per_uav_rb is not None:
                return int(self.owner_per_uav_rb[uav_idx, rb_idx])
            return int(self._actual_embb_owner_for_cell(uav_idx, rb_idx, minislot))

    def _executed_local_puncture_for_cell(self, uav_idx: int, rb_idx: int, minislot: int) -> bool:
            mask = getattr(self, "executed_local_puncture_mask", None)
            if (
                isinstance(mask, np.ndarray)
                and mask.ndim == 3
                and mask.shape[0] == self.sys_cfg.num_uavs
                and mask.shape[1] == self.sys_cfg.num_subcarriers
                and mask.shape[2] == self.sys_cfg.num_minislots
            ):
                return bool(mask[uav_idx, rb_idx, minislot])
            return bool(int(self.mode_grid[uav_idx, rb_idx, minislot]) == MODE_PUNCTURE)

    def _embb_per_rb_power_for_owner(
            self,
            uav_idx: int,
            embb_owner: int,
            rb_idx: int,
            minislot: Optional[int] = None,
        ) -> float:
            if embb_owner < 0:
                return 0.0
            power_limit_idx = min(embb_owner, len(self.embb_cfg.power_limits) - 1)
            max_power = min(
                self.allocator._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                self.algo_cfg.power_upper_bound,
            )
            scale = float(self.embb_power_scale[uav_idx])
            if self.embb_power_scale_grid is not None:
                grid_minislot = 0 if minislot is None else int(np.clip(minislot, 0, self.sys_cfg.num_minislots - 1))
                scale = float(self.embb_power_scale_grid[uav_idx, rb_idx, grid_minislot])
            scale = float(np.clip(
                scale,
                self.rl_cfg.env.embb_power_scale_min,
                self.rl_cfg.env.embb_power_scale_max,
            ))
            assigned_count = int(np.sum(self.owner_per_uav_rb[uav_idx, :] == embb_owner)) if self.owner_per_uav_rb is not None else 0
            if self.owner_per_uav_rb is not None and int(self.owner_per_uav_rb[uav_idx, rb_idx]) != embb_owner:
                assigned_count += 1
            assigned_count = max(assigned_count, 1)
            return float(max_power * scale / assigned_count)

    def _compute_intercell_interference(
        self,
        uav_idx: int,
        rb_idx: int,
        minislot: int,
        *,
        apply_embb_source_mask: bool = True,
    ) -> float:
            intercell = 0.0
            for other_uav in range(self.sys_cfg.num_uavs):
                if other_uav == uav_idx:
                    continue

                other_owner = self._actual_embb_owner_for_cell(other_uav, rb_idx, minislot)
                if other_owner < 0 and (not apply_embb_source_mask) and bool(
                    self._executed_local_puncture_for_cell(other_uav, rb_idx, minislot)
                ):
                    other_owner = self._baseline_embb_owner_for_cell(other_uav, rb_idx, minislot)
                other_locally_punctured = bool(self._executed_local_puncture_for_cell(other_uav, rb_idx, minislot))
                if other_owner >= 0 and ((not apply_embb_source_mask) or (not other_locally_punctured)):
                    other_embb_idx = self.sys_cfg.num_urllc_users + other_owner
                    other_embb_power = self.allocator._get_embb_per_rb_power(other_owner)
                    if self.embb_power_scale_grid is not None:
                        other_embb_power *= float(self.embb_power_scale_grid[other_uav, rb_idx, minislot])
                    embb_cross_gain = float(self.channel_gains_mag_sq[other_embb_idx, uav_idx, rb_idx])
                    intercell += other_embb_power * embb_cross_gain

                other_packet = int(self.packet_grid[other_uav, rb_idx, minislot])
                if (
                    other_packet >= 0
                    and other_packet < self.scheduled_power.shape[0]
                    and int(self.scheduled_uavs[other_packet]) == other_uav
                ):
                    other_user = int(self.packet_sources[other_packet])
                    other_power = float(self.scheduled_power[other_packet, other_uav])
                    urllc_cross_gain = float(self.channel_gains_mag_sq[other_user, uav_idx, rb_idx])
                    intercell += other_power * urllc_cross_gain

            return float(intercell)

    def _compute_episode_embb_metrics(
        self,
        *,
        ignore_intercell: bool = False,
        apply_local_puncture_deduction: bool = True,
        apply_embb_source_mask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
            num_embb = self.sys_cfg.num_embb_users
            num_minislots = max(self.sys_cfg.num_minislots, 1)
            embb_rates = np.zeros(num_embb, dtype=float)
            embb_power_alloc = np.zeros((num_embb, self.sys_cfg.num_uavs), dtype=float)
            overlay_rate_sum = 0.0
            puncture_rate_sum = 0.0

            for uav_idx in range(self.sys_cfg.num_uavs):
                for rb_idx in range(self.sys_cfg.num_subcarriers):
                    for minislot in range(self.sys_cfg.num_minislots):
                        embb_owner = self._actual_embb_owner_for_cell(uav_idx, rb_idx, minislot)
                        if embb_owner < 0 and bool(self._executed_local_puncture_for_cell(uav_idx, rb_idx, minislot)):
                            embb_owner = self._baseline_embb_owner_for_cell(uav_idx, rb_idx, minislot)
                        if embb_owner < 0:
                            continue
                        locally_punctured = bool(self._executed_local_puncture_for_cell(uav_idx, rb_idx, minislot))
                        if bool(apply_local_puncture_deduction) and locally_punctured:
                            # Local puncture removes the eMBB transmission on this (RB, minislot).
                            continue
                        mode = int(self.mode_grid[uav_idx, rb_idx, minislot])

                        embb_user_idx = self.sys_cfg.num_urllc_users + embb_owner
                        embb_per_rb_power = self.allocator._get_embb_per_rb_power(embb_owner)
                        embb_per_rb_power *= float(self.embb_power_scale_grid[uav_idx, rb_idx, minislot])
                        embb_gain = float(self.channel_gains_mag_sq[embb_user_idx, uav_idx, rb_idx])

                        intercell = 0.0
                        if not bool(ignore_intercell):
                            for other_uav in range(self.sys_cfg.num_uavs):
                                if other_uav == uav_idx:
                                    continue
                                other_owner = self._actual_embb_owner_for_cell(other_uav, rb_idx, minislot)
                                if other_owner < 0 and (not apply_embb_source_mask) and bool(
                                    self._executed_local_puncture_for_cell(other_uav, rb_idx, minislot)
                                ):
                                    other_owner = self._baseline_embb_owner_for_cell(other_uav, rb_idx, minislot)
                                other_locally_punctured = bool(self._executed_local_puncture_for_cell(other_uav, rb_idx, minislot))
                                if other_owner >= 0 and ((not apply_embb_source_mask) or (not other_locally_punctured)):
                                    other_embb_idx = self.sys_cfg.num_urllc_users + other_owner
                                    other_embb_power = self.allocator._get_embb_per_rb_power(other_owner)
                                    other_embb_power *= float(self.embb_power_scale_grid[other_uav, rb_idx, minislot])
                                    embb_cross_gain = float(self.channel_gains_mag_sq[other_embb_idx, uav_idx, rb_idx])
                                    intercell += other_embb_power * embb_cross_gain

                                other_packet = int(self.packet_grid[other_uav, rb_idx, minislot])
                                if (
                                    other_packet >= 0 and
                                    other_packet < self.scheduled_power.shape[0] and
                                    int(self.scheduled_uavs[other_packet]) == other_uav
                                ):
                                    other_user = int(self.packet_sources[other_packet])
                                    other_power = float(self.scheduled_power[other_packet, other_uav])
                                    urllc_cross_gain = float(self.channel_gains_mag_sq[other_user, uav_idx, rb_idx])
                                    intercell += other_power * urllc_cross_gain

                        local_residual = 0.0
                        packet_id = int(self.packet_grid[uav_idx, rb_idx, minislot])
                        if mode == MODE_OVERLAY and packet_id >= 0:
                            source_user = int(self.packet_sources[packet_id])
                            urllc_power = float(self.scheduled_power[packet_id, uav_idx])
                            urllc_gain_local = float(self.channel_gains_mag_sq[source_user, uav_idx, rb_idx])
                            local_residual = self.algo_cfg.sic_residual_factor * urllc_power * urllc_gain_local

                        snir = embb_per_rb_power * embb_gain / max(
                            self.sys_cfg.noise_power + intercell + local_residual,
                            1e-15,
                        )
                        cell_rate = self.capacity_model.shannon_capacity(
                            snir,
                            self.sys_cfg.subcarrier_bw,
                        ) / num_minislots
                        embb_rates[embb_owner] += cell_rate
                        embb_power_alloc[embb_owner, uav_idx] += embb_per_rb_power / num_minislots
                        if mode == MODE_OVERLAY:
                            overlay_rate_sum += float(cell_rate)
                        elif mode == MODE_PUNCTURE:
                            puncture_rate_sum += float(cell_rate)

            return embb_rates, embb_power_alloc, float(overlay_rate_sum), float(puncture_rate_sum)

    def summarize_episode(self) -> Dict[str, float]:
            """Summarize the finished episode from the executed cell-wise decisions."""
            # IMPORTANT: keep legacy eMBB metrics for backward-compatible debugging, but ALSO compute
            # corrected effective eMBB metrics that deduct local puncture airtime and exclude punctured
            # other-cell eMBB from intercell interference sources.
            #
            # Legacy (debug): local puncture does NOT deduct eMBB airtime; punctured other-cell eMBB may still
            # be counted as an intercell source in some terms.
            embb_rates, embb_power_alloc, overlay_rate_with_intercell, puncture_rate_with_intercell = self._compute_episode_embb_metrics(
                ignore_intercell=False,
                apply_local_puncture_deduction=False,
                apply_embb_source_mask=False,
            )
            embb_total_rate = float(np.sum(embb_rates))
            embb_user_rate = float(np.mean(embb_rates)) if embb_rates.size > 0 else 0.0
            embb_service_ratio = float(np.mean(embb_rates > 0)) if embb_rates.size > 0 else 0.0
            embb_served_users = int(np.count_nonzero(embb_rates > 0.0)) if embb_rates.size > 0 else 0

            # Corrected effective (used for main service/min-rate KPIs + reward shaping).
            embb_rates_eff, embb_power_alloc_eff, overlay_rate_with_intercell_eff, puncture_rate_with_intercell_eff = self._compute_episode_embb_metrics(
                ignore_intercell=False,
                apply_local_puncture_deduction=True,
                apply_embb_source_mask=True,
            )
            embb_total_rate_eff = float(np.sum(embb_rates_eff))
            embb_service_ratio_eff = float(np.mean(embb_rates_eff > 0)) if embb_rates_eff.size > 0 else 0.0

            # Counterfactual: no-intercell upper bound (legacy semantics; report-only).
            embb_rates_no_intercell, _p_alloc0, overlay_rate_no_intercell, puncture_rate_no_intercell = self._compute_episode_embb_metrics(
                ignore_intercell=True,
                apply_local_puncture_deduction=False,
                apply_embb_source_mask=False,
            )
            embb_rate_without_intercell_est = float(np.sum(embb_rates_no_intercell))
            embb_rate_with_intercell = float(embb_total_rate)
            embb_rate_loss_due_to_intercell = float(max(embb_rate_without_intercell_est - embb_rate_with_intercell, 0.0))
            embb_rate_loss_due_to_intercell_ratio = float(
                embb_rate_loss_due_to_intercell / max(embb_rate_without_intercell_est, 1.0e-9)
            )
            overlay_rate_loss_due_to_intercell = float(max(overlay_rate_no_intercell - overlay_rate_with_intercell, 0.0))
            puncture_rate_loss_due_to_intercell = float(max(puncture_rate_no_intercell - puncture_rate_with_intercell, 0.0))

            # Counterfactual: no-intercell WITH the same local puncture mask (preferred for interpretation).
            embb_rates_no_intercell_same_mask, _p_alloc1, overlay_rate_no_intercell_eff, puncture_rate_no_intercell_eff = self._compute_episode_embb_metrics(
                ignore_intercell=True,
                apply_local_puncture_deduction=True,
                apply_embb_source_mask=True,
            )
            no_intercell_rate_with_same_puncture_mask = float(np.sum(embb_rates_no_intercell_same_mask))
            embb_rate_with_intercell_after_puncture_deduction = float(embb_total_rate_eff)
            intercell_rate_loss_with_same_puncture_mask = float(
                max(no_intercell_rate_with_same_puncture_mask - embb_rate_with_intercell_after_puncture_deduction, 0.0)
            )

            # Local puncture airtime deduction diagnostics (isolate local puncture effect; keep source mask corrected).
            embb_rates_before_local_deduction, _p_alloc2, _ov2, _pu2 = self._compute_episode_embb_metrics(
                ignore_intercell=False,
                apply_local_puncture_deduction=False,
                apply_embb_source_mask=True,
            )
            embb_rate_raw_before_local_puncture_deduction = float(np.sum(embb_rates_before_local_deduction))
            embb_rate_after_local_puncture_deduction = float(embb_rate_with_intercell_after_puncture_deduction)
            embb_rate_loss_due_to_local_puncture = float(
                max(embb_rate_raw_before_local_puncture_deduction - embb_rate_after_local_puncture_deduction, 0.0)
            )
            embb_rate_loss_due_to_local_puncture_ratio = float(
                embb_rate_loss_due_to_local_puncture / max(embb_rate_raw_before_local_puncture_deduction, 1.0e-9)
            )
            scheduled_packets = int(np.count_nonzero(self.scheduled_uavs >= 0))
            scheduled_packets_per_uav = float(scheduled_packets / max(self.sys_cfg.num_uavs, 1))
            active_packets = int(self.num_packets)
            admission_ratio = float(scheduled_packets / max(active_packets, 1)) if active_packets > 0 else 1.0
            unscheduled_ratio = float(max(active_packets - scheduled_packets, 0) / max(active_packets, 1)) if active_packets > 0 else 0.0
            admitted_reliability = self._compute_admitted_urllc_reliability()
            empty_admission_case = float(active_packets > 0 and scheduled_packets <= 0)
            if active_packets > 0:
                if scheduled_packets > 0 and np.isfinite(admitted_reliability):
                    effective_urllc_success_over_arrivals = float(
                        admitted_reliability * scheduled_packets / max(active_packets, 1)
                    )
                else:
                    effective_urllc_success_over_arrivals = 0.0
            else:
                effective_urllc_success_over_arrivals = 1.0

            # URLLC throughput estimate (slot-based).
            # The simulator schedules packets at the slot level; assume 1 ms/slot for reporting.
            # Estimate: scheduled packets × avg packet bits / 1 ms slot.
            slot_duration_s = 1.0e-3
            packet_lengths = list(getattr(self.urllc_cfg, "packet_lengths", []) or [])
            packet_bits = float(np.mean(np.asarray(packet_lengths, dtype=float))) if packet_lengths else 160.0
            urllc_throughput_bps_slot_est = float(scheduled_packets * packet_bits / max(slot_duration_s, 1.0e-12))
            urllc_throughput_mbps_slot_est = float(urllc_throughput_bps_slot_est / 1.0e6)

            overlay_count = int(np.sum(self.mode_grid == MODE_OVERLAY))
            puncture_count = int(np.sum(self.mode_grid == MODE_PUNCTURE))
            coexist_count = overlay_count + puncture_count
            overlay_ratio = float(overlay_count / max(coexist_count, 1))
            puncture_ratio = float(puncture_count / max(coexist_count, 1))
            overlay_selection_ratio = float(overlay_count / max(self.phase_a_total_decisions, 1))
            puncture_selection_ratio = float(puncture_count / max(self.phase_a_total_decisions, 1))
            overlay_utilization = float(self.overlay_selected_pairs / max(self.overlay_feasible_pairs, 1))

            # eMBB transmission-activity masks (per (uav, rb, minislot)):
            # has_owner: baseline eMBB owner exists (even if punctured this minislot).
            # tx_active: has_owner AND not punctured locally (executed_local_puncture_mask).
            embb_has_owner_count = 0
            embb_tx_active_count = 0
            embb_tx_excluded_by_puncture_count = 0
            punctured_cells_with_local_embb_owner_count = 0
            for _uav in range(self.sys_cfg.num_uavs):
                for _rb in range(self.sys_cfg.num_subcarriers):
                    for _ms in range(self.sys_cfg.num_minislots):
                        _owner = int(self._baseline_embb_owner_for_cell(_uav, _rb, _ms))
                        if _owner < 0:
                            continue
                        embb_has_owner_count += 1
                        _locally_punctured = bool(self._executed_local_puncture_for_cell(_uav, _rb, _ms))
                        if _locally_punctured:
                            embb_tx_excluded_by_puncture_count += 1
                            punctured_cells_with_local_embb_owner_count += 1
                        else:
                            embb_tx_active_count += 1

            local_punctured_embb_airtime_ratio = float(embb_tx_excluded_by_puncture_count / max(embb_has_owner_count, 1))
            intercell_source_mask_excluded_by_puncture_ratio = float(
                embb_tx_excluded_by_puncture_count / max(embb_has_owner_count, 1)
            )
            intercell_source_active_ratio = float(embb_tx_active_count / max(embb_has_owner_count, 1))
            if (
                int(getattr(self, "executed_puncture_action_count", 0)) > 0
                and punctured_cells_with_local_embb_owner_count > 0
            ):
                assert local_punctured_embb_airtime_ratio > 0.0, (
                    "Local puncture accounting invariant violated: puncture executed on active local eMBB "
                    "but local_punctured_embb_airtime_ratio is zero."
                )
                assert embb_rate_loss_due_to_local_puncture >= -1.0e-9, (
                    "Local puncture accounting invariant violated: embb_rate_loss_due_to_local_puncture < 0."
                )

            # Intercell eMBB interference power diagnostics: before/after applying the puncture source mask.
            # (eMBB-only; URLLC sources excluded here by design.)
            intercell_power_before_puncture_source_mask_mean = 0.0
            intercell_power_after_puncture_source_mask_mean = 0.0
            intercell_power_reduction_from_source_mask_mean = 0.0
            try:
                before_vals = []
                after_vals = []
                for victim_uav in range(self.sys_cfg.num_uavs):
                    for rb_idx in range(self.sys_cfg.num_subcarriers):
                        for minislot in range(self.sys_cfg.num_minislots):
                            before = 0.0
                            after = 0.0
                            for other_uav in range(self.sys_cfg.num_uavs):
                                if other_uav == victim_uav:
                                    continue
                                other_owner = int(self._baseline_embb_owner_for_cell(other_uav, rb_idx, minislot))
                                if other_owner < 0:
                                    continue
                                other_embb_idx = self.sys_cfg.num_urllc_users + other_owner
                                other_embb_power = float(self.allocator._get_embb_per_rb_power(other_owner))
                                if self.embb_power_scale_grid is not None:
                                    other_embb_power *= float(self.embb_power_scale_grid[other_uav, rb_idx, minislot])
                                embb_cross_gain = float(self.channel_gains_mag_sq[other_embb_idx, victim_uav, rb_idx])
                                term = float(other_embb_power * embb_cross_gain)
                                before += term
                                if not bool(self._executed_local_puncture_for_cell(other_uav, rb_idx, minislot)):
                                    after += term
                            before_vals.append(float(before))
                            after_vals.append(float(after))
                if before_vals:
                    intercell_power_before_puncture_source_mask_mean = float(np.mean(np.asarray(before_vals, dtype=float)))
                    intercell_power_after_puncture_source_mask_mean = float(np.mean(np.asarray(after_vals, dtype=float)))
                    intercell_power_reduction_from_source_mask_mean = float(
                        intercell_power_before_puncture_source_mask_mean - intercell_power_after_puncture_source_mask_mean
                    )
            except Exception:
                intercell_power_before_puncture_source_mask_mean = 0.0
                intercell_power_after_puncture_source_mask_mean = 0.0
                intercell_power_reduction_from_source_mask_mean = 0.0

            total_cells = self.sys_cfg.num_uavs * self.sys_cfg.num_subcarriers * self.sys_cfg.num_minislots
            embb_active_cells = self.embb_owner_grid >= 0
            urllc_active_cells = self.packet_grid >= 0
            embb_only_cells = np.count_nonzero(embb_active_cells & ~urllc_active_cells)
            idle_cells = np.count_nonzero(~embb_active_cells & ~urllc_active_cells)
            embb_only_fraction = float(embb_only_cells / max(total_cells, 1))
            overlay_fraction = float(overlay_count / max(total_cells, 1))
            puncture_fraction = float(puncture_count / max(total_cells, 1))
            idle_fraction = float(idle_cells / max(total_cells, 1))
            minislot_utilization = 1.0 - idle_fraction
            local_puncture_mask_nonzero_count = int(
                np.count_nonzero(
                    np.asarray(
                        getattr(self, "executed_local_puncture_mask", np.zeros_like(self.mode_grid)),
                        dtype=bool,
                    )
                )
            )
            executed_puncture_action_count = int(getattr(self, "executed_puncture_action_count", 0))

            urllc_slot_avg_power = float(np.sum(self.scheduled_power) / max(self.sys_cfg.num_minislots, 1))
            embb_slot_avg_power = float(np.sum(embb_power_alloc))
            total_power = float(urllc_slot_avg_power + embb_slot_avg_power)
            throughput_per_watt = float(embb_total_rate / max(total_power, 1.0e-9))
            avg_throughput_per_served_embb_user = float(embb_total_rate / max(embb_served_users, 1))
            jain_fairness = self._compute_jain_fairness(embb_rates)
            embb_min_rate = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 0.0))
            if embb_rates.size > 0 and embb_min_rate > 0.0:
                shortfall = np.clip((embb_min_rate - embb_rates) / max(embb_min_rate, 1e-9), 0.0, 1.0)
                embb_min_rate_shortfall = float(np.mean(shortfall))
                embb_min_rate_satisfaction_ratio = float(np.mean(embb_rates >= embb_min_rate))
            else:
                embb_min_rate_shortfall = 0.0
                embb_min_rate_satisfaction_ratio = float(embb_service_ratio)

            # Corrected (after local puncture airtime deduction) service/min-rate KPIs.
            if embb_rates_eff.size > 0 and embb_min_rate > 0.0:
                embb_min_rate_satisfaction_after_puncture_deduction = float(np.mean(embb_rates_eff >= embb_min_rate))
            else:
                embb_min_rate_satisfaction_after_puncture_deduction = float(embb_service_ratio_eff)
            embb_service_ratio_after_puncture_deduction = float(embb_service_ratio_eff)
            thin_service_threshold = float(getattr(self.rl_cfg.reward, "terminal_thin_service_threshold_bps", 0.0) or 0.0)
            if embb_rates.size > 0 and thin_service_threshold > 0.0:
                served_mask = embb_rates > 0.0
                if np.any(served_mask):
                    thin_service_fraction = float(np.mean(embb_rates[served_mask] < thin_service_threshold))
                else:
                    thin_service_fraction = 0.0
            else:
                thin_service_fraction = 0.0
            final_owner_map = self._effective_owner_map()
            snapshot_owner_map = self.phase0_snapshot_owner_per_uav_rb
            if final_owner_map is not None and snapshot_owner_map is not None and final_owner_map.shape == snapshot_owner_map.shape:
                planning_owner_change_count = int(np.count_nonzero(final_owner_map != snapshot_owner_map))
            else:
                planning_owner_change_count = 0
            planning_owner_change_ratio = float(
                planning_owner_change_count / max(self.sys_cfg.num_uavs * self.sys_cfg.num_subcarriers, 1)
            )
            planning_owner_non_null_ratio = float(self.planning_owner_non_null_count / max(self.planning_total_decisions, 1))
            planning_embb_power_nonzero_ratio = float(self.planning_embb_power_nonzero_count / max(self.planning_total_decisions, 1))
            planning_embb_power_changed_ratio = float(self.planning_embb_power_changed_count / max(self.planning_total_decisions, 1))
            planning_owner_rewrite_ratio = float(self.planning_owner_rewrite_count / max(self.planning_total_decisions, 1))
            snapshot_rb_owner = self.phase0_snapshot_owner_per_uav_rb
            executed_rb_owner = self.owner_per_uav_rb
            raw_rb_owner = self.phase0_raw_owner_per_uav_rb
            denom_cells = max(self.sys_cfg.num_uavs * self.sys_cfg.num_subcarriers, 1)
            if (
                snapshot_rb_owner is not None
                and executed_rb_owner is not None
                and snapshot_rb_owner.shape == executed_rb_owner.shape
            ):
                phase0_owner_change_ratio_vs_snapshot_executed = float(
                    np.count_nonzero(executed_rb_owner != snapshot_rb_owner) / denom_cells
                )
                phase0_owner_non_null_ratio_executed = float(
                    np.count_nonzero(executed_rb_owner >= 0) / denom_cells
                )
            else:
                phase0_owner_change_ratio_vs_snapshot_executed = 0.0
                phase0_owner_non_null_ratio_executed = 0.0

            if (
                snapshot_rb_owner is not None
                and raw_rb_owner is not None
                and snapshot_rb_owner.shape == raw_rb_owner.shape
            ):
                phase0_owner_change_ratio_vs_snapshot_raw = float(
                    np.count_nonzero(raw_rb_owner != snapshot_rb_owner) / denom_cells
                )
                phase0_owner_non_null_ratio_raw = float(
                    np.count_nonzero(raw_rb_owner >= 0) / denom_cells
                )
            else:
                phase0_owner_change_ratio_vs_snapshot_raw = 0.0
                phase0_owner_non_null_ratio_raw = 0.0
            phase0_owner_fallback_to_candidate0_ratio = float(
                self.phase0_owner_fallback_to_candidate0_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_invalid_option_ratio = float(
                self.phase0_owner_invalid_option_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_null_selected_ratio = float(
                self.phase0_owner_null_selected_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_invalid_to_null_ratio = float(
                self.phase0_owner_invalid_to_null_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_invalid_to_snapshot_ratio = float(
                self.phase0_owner_invalid_to_snapshot_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_invalid_to_non_snapshot_ratio = float(
                self.phase0_owner_invalid_to_non_snapshot_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_restored_to_snapshot_ratio = float(
                self.phase0_owner_restored_to_snapshot_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_kept_null_ratio = float(
                self.phase0_owner_kept_null_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_replaced_with_non_snapshot_ratio = float(
                self.phase0_owner_replaced_with_non_snapshot_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_raw_same_as_snapshot_ratio = float(
                self.phase0_owner_raw_same_as_snapshot_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_raw_non_snapshot_ratio = float(
                self.phase0_owner_raw_non_snapshot_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_raw_null_ratio = float(
                self.phase0_owner_raw_null_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_exec_same_as_snapshot_ratio = float(
                self.phase0_owner_exec_same_as_snapshot_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_exec_non_snapshot_ratio = float(
                self.phase0_owner_exec_non_snapshot_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_reverted_to_snapshot_ratio = float(
                self.phase0_owner_reverted_to_snapshot_count / max(self.phase0_owner_snapshot_comparable_count, 1)
            )
            phase0_owner_guard_rewrite_ratio = float(
                self.phase0_owner_guard_rewrite_count / max(self.planning_total_decisions, 1)
            )
            phase0_owner_service_violation_ratio = float(
                self.phase0_owner_guard_service_violation_count / max(self.phase0_owner_guard_checks, 1)
            )
            phase0_owner_rate_violation_ratio = float(
                self.phase0_owner_guard_min_rate_violation_count / max(self.phase0_owner_guard_checks, 1)
            )
            phase0_owner_accepted_positive_service_gain_ratio = float(
                self.phase0_owner_guard_accepted_positive_service_gain_count / max(self.phase0_owner_guard_checks, 1)
            )
            phase0_owner_accepted_negative_service_gain_ratio = float(
                self.phase0_owner_guard_accepted_negative_service_gain_count / max(self.phase0_owner_guard_checks, 1)
            )
            # Redefined "changed_and_effective": compare executed owner vs snapshot, and mark "effective"
            # only if the final projected eMBB allocation actually uses that RB (alpha_e == 1).
            phase0_owner_changed_and_effective_ratio = 0.0
            phase0_owner_changed_but_unserved_ratio = 0.0
            phase0_owner_same_as_snapshot_ratio = 0.0
            phase0_owner_effective_service_gain_ratio = 0.0
            phase0_owner_effective_rate_gain_vs_snapshot_mean = 0.0
            phase0_owner_effective_rate_gain_vs_snapshot_cells_mean = 0.0
            owner_change_counterfactual_service_gain = 0.0
            owner_change_counterfactual_intercell_gain = 0.0
            owner_change_counterfactual_objective_gain = 0.0
            owner_rejected_by_snapshot_imitation_ratio = 0.0
            owner_rejected_by_hard_feasibility_ratio = 0.0
            owner_accepted_low_intercell_non_greedy_ratio = 0.0
            denom_owner_cells = max(int(self.planning_total_decisions), 1)
            snapshot_rb_owner = self.phase0_snapshot_owner_per_uav_rb
            if (
                snapshot_rb_owner is not None
                and self.owner_per_uav_rb is not None
                and snapshot_rb_owner.shape == self.owner_per_uav_rb.shape
                and int(self.sys_cfg.num_embb_users) > 0
            ):
                executed_effective = self._effective_owner_map(self.owner_per_uav_rb)
                executed_proj = self._project_embb_baseline_from_owner_map(
                    executed_effective,
                    self.embb_power_scale_per_uav_rb,
                )
                snapshot_proj = self._project_embb_baseline_from_owner_map(
                    np.asarray(snapshot_rb_owner, dtype=int),
                    np.ones((self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers), dtype=float),
                )
                alpha_e = np.asarray(executed_proj.get("alpha_e"), dtype=int)
                owners = np.asarray(executed_effective, dtype=int)
                snapshot_owners = np.asarray(snapshot_rb_owner, dtype=int)
                changed_mask = owners != snapshot_owners
                executed_owner_change_count = int(np.count_nonzero(changed_mask & (owners >= 0)))
                positive_service_gain_owner_changes = 0

                num_uavs, num_rbs = int(owners.shape[0]), int(owners.shape[1])
                uav_grid = np.repeat(np.arange(num_uavs, dtype=int)[:, None], num_rbs, axis=1).ravel()
                rb_grid = np.repeat(np.arange(num_rbs, dtype=int)[None, :], num_uavs, axis=0).ravel()
                owner_flat = owners.ravel()
                valid_owner = (owner_flat >= 0) & (owner_flat < int(self.sys_cfg.num_embb_users))
                effective_flat = np.zeros(owner_flat.shape, dtype=bool)
                if alpha_e.ndim == 3 and alpha_e.shape[0] >= int(self.sys_cfg.num_embb_users):
                    idx = np.where(valid_owner)[0]
                    effective_flat[idx] = alpha_e[owner_flat[idx], uav_grid[idx], rb_grid[idx]] > 0
                effective_mask = effective_flat.reshape(owners.shape)

                changed_and_effective_mask = changed_mask & effective_mask & (owners >= 0)
                changed_but_unserved_mask = changed_mask & (~effective_mask) & (owners >= 0)
                same_as_snapshot_mask = (~changed_mask) & effective_mask & (owners >= 0)

                changed_and_effective_count = int(np.count_nonzero(changed_and_effective_mask))
                changed_but_unserved_count = int(np.count_nonzero(changed_but_unserved_mask))
                same_as_snapshot_count = int(np.count_nonzero(same_as_snapshot_mask))
                phase0_owner_changed_and_effective_ratio = float(changed_and_effective_count / denom_owner_cells)
                phase0_owner_changed_but_unserved_ratio = float(changed_but_unserved_count / denom_owner_cells)
                phase0_owner_same_as_snapshot_ratio = float(same_as_snapshot_count / denom_owner_cells)

                # Service/rate gains vs snapshot (diagnostics only; snapshot not used for policy).
                exec_rates = np.asarray(executed_proj.get("rates"), dtype=float)
                snap_rates = np.asarray(snapshot_proj.get("rates"), dtype=float)
                exec_service = float(np.mean(exec_rates > 1.0e-12)) if exec_rates.size else 0.0
                snap_service = float(np.mean(snap_rates > 1.0e-12)) if snap_rates.size else 0.0
                phase0_owner_effective_service_gain_ratio = float((exec_service - snap_service) / max(snap_service, 1.0e-9))
                snap_total_rate = float(snapshot_proj.get("total_rate", 0.0))
                exec_total_rate = float(executed_proj.get("total_rate", 0.0))
                phase0_owner_effective_rate_gain_vs_snapshot_mean = float((exec_total_rate - snap_total_rate) / max(snap_total_rate, 1.0e-9))

                # Counterfactual gains (baseline snapshot is allowed as a comparison, but NEVER as an imitation target).
                snap_intercell = float(self._phase0_mean_intercell_power_for_owner_map(snapshot_rb_owner, np.ones((self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers), dtype=float)))
                exec_intercell = float(self._phase0_mean_intercell_power_for_owner_map(executed_effective, self.embb_power_scale_per_uav_rb))
                owner_change_counterfactual_service_gain = float(phase0_owner_effective_service_gain_ratio)
                owner_change_counterfactual_intercell_gain = float((snap_intercell - exec_intercell) / max(snap_intercell, 1.0e-12))
                snap_power = float(snapshot_proj.get("total_power", 0.0))
                exec_power = float(executed_proj.get("total_power", 0.0))
                power_delta_ratio = float((exec_power - snap_power) / max(snap_power, 1.0e-9))
                # Simple normalized objective gain (debug): service + rate - intercell - power.
                owner_change_counterfactual_objective_gain = float(
                    1.0 * owner_change_counterfactual_service_gain
                    + 0.5 * float(phase0_owner_effective_rate_gain_vs_snapshot_mean)
                    + 0.5 * float(owner_change_counterfactual_intercell_gain)
                    - 0.2 * float(power_delta_ratio)
                )
                owner_rejected_by_snapshot_imitation_ratio = 0.0 if bool(getattr(self.rl_cfg.env, "disable_snapshot_imitation", True)) else float(
                    self.phase0_owner_restored_to_snapshot_count / max(self.phase0_owner_guard_checks, 1)
                )
                owner_rejected_by_hard_feasibility_ratio = float(self.planning_owner_guard_violation_count / max(self.phase0_owner_guard_checks, 1))
                owner_accepted_low_intercell_non_greedy_ratio = float(
                    (phase0_owner_changed_and_effective_ratio > 0.0) and (snap_intercell - exec_intercell > 0.0)
                )

                executed_base = np.asarray(executed_proj.get("base_rb_rates_per_uav_rb"), dtype=float)
                snapshot_base = np.asarray(snapshot_proj.get("base_rb_rates_per_uav_rb"), dtype=float)
                if executed_base.shape == snapshot_base.shape == owners.shape:
                    per_cell_gain = executed_base - snapshot_base
                    gain_cells = per_cell_gain[changed_and_effective_mask]
                    if gain_cells.size:
                        phase0_owner_effective_rate_gain_vs_snapshot_cells_mean = float(np.mean(gain_cells) / 1.0e6)
                    positive_service_gain_owner_changes = int(
                        np.count_nonzero(changed_and_effective_mask & (per_cell_gain > 1.0e-12))
                    )

                # Store counts for legacy log fields.
                self.phase0_owner_changed_and_effective_count = float(changed_and_effective_count)
                self.phase0_owner_changed_but_unserved_count = float(changed_but_unserved_count)
                self.phase0_owner_same_as_snapshot_count = float(same_as_snapshot_count)
                self.phase0_owner_executed_change_count = float(executed_owner_change_count)
                self.phase0_owner_positive_service_gain_change_count = float(positive_service_gain_owner_changes)
            phase0_owner_effective_change_count = float(self.phase0_owner_changed_and_effective_count)
            planning_projected_embb_rate_ratio_mean = float(
                self.planning_projected_embb_rate_ratio_sum / max(self.planning_projected_metric_count, 1)
            )
            planning_projected_embb_rate_ratio_min = (
                float(self.planning_projected_embb_rate_ratio_min)
                if self.planning_projected_metric_count > 0 and np.isfinite(self.planning_projected_embb_rate_ratio_min)
                else 0.0
            )
            planning_projected_embb_power_ratio_mean = float(
                self.planning_projected_embb_power_ratio_sum / max(self.planning_projected_metric_count, 1)
            )
            planning_projected_embb_power_ratio_max = (
                float(self.planning_projected_embb_power_ratio_max)
                if self.planning_projected_metric_count > 0 else 0.0
            )
            phase0_owner_change_budget_used = float(
                self.phase0_owner_change_budget_used_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_change_budget_allowed = float(
                self.phase0_owner_change_budget_allowed_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_change_budget_clipped_ratio = float(
                self.phase0_owner_change_budget_clipped_ratio_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_change_kept_topk_ratio = float(
                self.phase0_owner_change_kept_topk_ratio_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_change_dropped_over_budget_ratio = float(
                self.phase0_owner_change_dropped_over_budget_ratio_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_raw_changed_count_mean = float(
                self.phase0_owner_raw_changed_count_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_allowed_k_mean = float(
                self.phase0_owner_allowed_k_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_executed_changed_count_mean = float(
                self.phase0_owner_executed_changed_count_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_dropped_count_mean = float(
                self.phase0_owner_dropped_count_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_budget_min_one_rule_eligible_ratio = float(
                self.phase0_owner_budget_min_one_rule_eligible_count / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_budget_min_one_rule_applied_ratio = float(
                self.phase0_owner_budget_min_one_rule_applied_count / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_min_one_blocked_by_no_positive_candidate_ratio = float(
                self.phase0_owner_min_one_blocked_by_no_positive_candidate_count / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_candidate_positive_objective_ratio = float(
                self.phase0_owner_candidate_positive_objective_count / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_candidate_relaxed_ratio = float(
                self.phase0_owner_candidate_relaxed_count / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_candidate_fallback_used_ratio = float(
                self.phase0_owner_candidate_fallback_used_count / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_obj_mean = float(
                self.phase0_owner_gate_obj_mean_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_obj_std = float(
                self.phase0_owner_gate_obj_std_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_gate_threshold = float(
                self.phase0_owner_gate_threshold_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_candidate_after_gate_ratio = float(
                self.phase0_owner_candidate_after_gate_count / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_positive_candidate_count_mean = float(
                self.phase0_owner_candidate_positive_objective_count / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_neg_accept_clipped_ratio = float(
                self.phase0_owner_neg_accept_clipped_ratio_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_neg_rejected_by_quota_ratio = float(
                self.phase0_owner_neg_rejected_by_quota_ratio_sum / max(self.phase0_owner_change_budget_checks, 1)
            )
            phase0_owner_pos_selected_count_mean = float(
                self.phase0_owner_pos_selected_count_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_neg_selected_count_mean = float(
                self.phase0_owner_neg_selected_count_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_selected_negative_count_mean = float(
                (self.phase0_owner_neg_selected_count_sum + self.phase0_owner_safe_relaxed_selected_count_sum)
                / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_selected_count = float(
                self.phase0_owner_selection_selected_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_final_selected_count_mean = float(phase0_owner_selected_count)
            phase0_owner_final_pos_selected_count_mean = float(phase0_owner_pos_selected_count_mean)
            phase0_owner_final_neg_selected_count_mean = float(phase0_owner_neg_selected_count_mean)
            phase0_owner_final_keep_set_size_mean = float(phase0_owner_selected_count)
            phase0_owner_allowed_k = float(
                self.phase0_owner_selection_allowed_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_pos_selected_ratio = float(
                self.phase0_owner_pos_selected_count_sum / max(self.phase0_owner_selection_selected_sum, 1.0)
            )
            phase0_owner_selection_fill_ratio = float(
                self.phase0_owner_selection_selected_sum / max(self.phase0_owner_selection_allowed_sum, 1.0)
            )
            phase0_owner_neg_accept_ratio = float(
                self.phase0_owner_negative_but_accepted_count / max(self.phase0_owner_executed_changed_count_sum, 1.0)
            )
            phase0_owner_positive_shortage_ratio = float(
                self.phase0_owner_positive_shortage_count / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_negative_blocked_due_to_quota_ratio = float(
                self.phase0_owner_negative_blocked_due_to_quota_count / max(self.phase0_owner_selection_decision_count, 1)
            )
            owner_safe_relaxed_used_ratio = float(
                self.phase0_owner_safe_relaxed_used_count / max(self.phase0_owner_selection_decision_count, 1)
            )
            owner_safe_relaxed_candidate_count = float(
                self.phase0_owner_safe_relaxed_candidate_count_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            owner_safe_relaxed_selected_count = float(
                self.phase0_owner_safe_relaxed_selected_count_sum / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_final_safe_relax_selected_count_mean = float(owner_safe_relaxed_selected_count)
            owner_safe_relaxed_avg_objective = float(
                self.phase0_owner_safe_relaxed_objective_sum / max(self.phase0_owner_safe_relaxed_selected_count_sum, 1.0)
            )
            owner_safe_relaxed_service_delta_mean = float(
                self.phase0_owner_safe_relaxed_service_delta_sum / max(self.phase0_owner_safe_relaxed_selected_count_sum, 1.0)
            )
            owner_safe_relaxed_intercell_delta_mean = float(
                self.phase0_owner_safe_relaxed_intercell_delta_sum / max(self.phase0_owner_safe_relaxed_selected_count_sum, 1.0)
            )
            owner_near_zero_objective_ratio = float(
                self.phase0_owner_near_zero_objective_count / max(self.phase0_owner_candidate_count, 1)
            )
            owner_positive_after_relax_ratio = float(
                self.phase0_owner_positive_after_relax_count / max(self.phase0_owner_candidate_count, 1)
            )
            owner_safe_relax_disabled_ratio = float(
                self.phase0_owner_safe_relax_disabled_count / max(self.phase0_owner_selection_decision_count, 1)
            )
            phase0_owner_accepted_positive_objective_ratio = float(
                self.phase0_owner_accepted_positive_objective_count / max(self.phase0_owner_executed_changed_count_sum, 1.0)
            )
            phase0_owner_rejected_nonpositive_objective_ratio = float(
                self.phase0_owner_rejected_nonpositive_objective_count / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_objective_gain_mean = float(
                self.phase0_owner_objective_gain_sum / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_objective_gain_pre_filter_mean = float(
                self.phase0_owner_objective_gain_pre_filter_sum / max(self.phase0_owner_candidate_count, 1)
            )
            phase0_owner_objective_gain_post_filter_mean = float(
                self.phase0_owner_objective_gain_post_filter_sum / max(self.phase0_owner_objective_gain_post_filter_count, 1)
            )
            phase0_owner_negative_but_accepted_ratio = float(
                self.phase0_owner_negative_but_accepted_count / max(self.phase0_owner_executed_changed_count_sum, 1.0)
            )
            phase0_owner_neg_accepted_with_positive_candidate_ratio = float(
                self.phase0_owner_neg_accepted_with_positive_candidate_count
                / max(self.phase0_owner_steps_with_positive_candidate, 1)
            )
            phase0_owner_objective_gain_accepted_mean = float(
                self.phase0_owner_objective_gain_accepted_sum / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            phase0_owner_effective_rate_gain_accepted_mean = float(
                self.phase0_owner_effective_rate_gain_accepted_sum / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            phase0_owner_intercell_reduction_accepted_mean = float(
                self.phase0_owner_intercell_reduction_accepted_sum / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            phase0_owner_service_gain_accepted_mean = float(
                self.phase0_owner_service_gain_accepted_sum / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            phase0_owner_minrate_gain_accepted_mean = float(
                self.phase0_owner_minrate_gain_accepted_sum / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            phase0_owner_harmful_accepted_ratio = float(
                self.phase0_owner_harmful_accepted_count / max(self.phase0_owner_accepted_positive_objective_count, 1)
            )
            owner_change_detail_top = list(self.phase0_owner_change_detail_records[:32])
            harmful_count = int(sum(1 for item in owner_change_detail_top if bool(item.get("harmful", False))))
            owner_change_harmful_ratio = float(harmful_count / max(len(owner_change_detail_top), 1))
            phase_a_embb_power_write_ratio = float(self.phase_a_embb_power_write_count / max(self.phase_a_total_decisions, 1))
            phase_a_embb_power_changed_ratio = float(self.phase_a_embb_power_changed_count / max(self.phase_a_total_decisions, 1))
            phase_a_power_zeroed_non_admission_ratio = float(
                self.phase_a_power_zeroed_non_admission_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_write_on_admission_ratio = float(
                self.phase_a_power_write_on_admission_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_write_on_keep_ratio = float(
                self.phase_a_power_write_on_keep_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_zero_by_keep_due_to_mode_gate_ratio = float(
                self.phase_a_zero_by_keep_due_to_mode_gate_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_keep_power_write_attempt_ratio = float(
                self.phase_a_keep_power_write_attempt_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_keep_power_write_success_ratio = float(
                self.phase_a_keep_power_write_success_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_write_blocked_no_owner_ratio = float(
                self.phase_a_power_write_blocked_no_owner_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_write_blocked_projection_ratio = float(
                self.phase_a_power_write_blocked_projection_count / max(self.phase_a_total_decisions, 1)
            )
            action_intercell_guard_active_ratio = float(
                self.action_intercell_guard_active_cell_count / max(self.action_intercell_guard_total_cell_count, 1)
            )
            action_intercell_guard_candidate_active_ratio = float(
                self.action_intercell_guard_candidate_high_count / max(self.action_intercell_guard_candidate_total_count, 1)
            )
            action_intercell_guard_selected_violation_ratio = float(
                self.action_intercell_guard_selected_violation_count / max(self.action_intercell_guard_selected_excess_count, 1)
            )
            action_intercell_guard_local_min_cost_mean = float(
                self.action_intercell_guard_local_min_cost_sum / max(self.action_intercell_guard_local_min_cost_count, 1)
            )
            action_intercell_guard_selected_excess_mean = float(
                self.action_intercell_guard_selected_excess_sum / max(self.action_intercell_guard_selected_excess_count, 1)
            )
            phase_a_embb_power_mean_abs_change = float(self.phase_a_embb_power_change_sum / max(self.phase_a_embb_power_write_count, 1))
            phase_a_embb_power_mean_raw_delta = float(
                self.phase_a_embb_power_raw_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_mean_executed_delta = float(
                self.phase_a_embb_power_executed_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_clip_ratio = float(
                (self.phase_a_embb_power_delta_clipped_count + self.phase_a_embb_power_scale_clipped_count)
                / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_quantized_ratio = float(
                self.phase_a_embb_power_quantized_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_projection_ratio = float(
                self.phase_a_embb_power_scale_clipped_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_cap_hit_ratio = float(
                self.phase_a_embb_power_cap_hit_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_floor_hit_ratio = float(
                self.phase_a_embb_power_floor_hit_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_pre_clip_mean_delta = float(
                self.phase_a_embb_power_pre_clip_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_post_clip_mean_delta = float(
                self.phase_a_embb_power_post_clip_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_post_quant_mean_delta = float(
                self.phase_a_embb_power_post_quant_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_post_projection_mean_delta = float(
                self.phase_a_embb_power_post_projection_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_post_owner_validation_mean_delta = float(
                self.phase_a_embb_power_post_owner_validation_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_final_executed_mean_delta = float(
                self.phase_a_embb_power_final_executed_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_sign_flip_ratio = float(
                self.phase_a_embb_power_sign_flip_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_mean_abs_raw_delta = float(
                self.phase_a_embb_power_mean_abs_raw_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_mean_abs_executed_delta = float(
                self.phase_a_embb_power_mean_abs_executed_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_mean_raw_delta_l2 = float(
                self.phase_a_embb_power_raw_delta_sq_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_abs_shrink_ratio = float(
                self.phase_a_embb_power_mean_abs_executed_delta_sum / max(self.phase_a_embb_power_mean_abs_raw_delta_sum, 1.0e-12)
            )
            phase_a_embb_power_projection_l2_mean = float(np.sqrt(
                self.phase_a_embb_power_pre_vs_final_sq_diff_sum / max(self.phase_a_embb_power_projection_count, 1)
            ))
            phase_a_embb_power_pre_vs_final_l1_mean = float(
                self.phase_a_embb_power_pre_vs_final_abs_diff_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_pre_vs_final_sign_consistency = float(
                self.phase_a_embb_power_pre_vs_final_sign_consistent_count
                / max(self.phase_a_embb_power_pre_vs_final_sign_consistent_denom, 1)
            )
            phase_a_embb_power_effective_nonzero_ratio = float(
                self.phase_a_embb_power_effective_nonzero_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_floor_binding_strength = float(
                self.phase_a_embb_power_floor_binding_strength_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_cap_binding_strength = float(
                self.phase_a_embb_power_cap_binding_strength_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_proj_delta_l1 = float(
                self.phase_a_embb_power_proj_delta_abs_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_proj_delta_l2 = float(np.sqrt(
                self.phase_a_embb_power_proj_delta_sq_sum / max(self.phase_a_embb_power_projection_count, 1)
            ))
            phase_a_embb_power_pre_to_floor_delta = float(
                self.phase_a_embb_power_pre_to_floor_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_pre_to_cap_delta = float(
                self.phase_a_embb_power_pre_to_cap_delta_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_final_minus_proj_delta = float(
                self.phase_a_embb_power_final_minus_proj_abs_sum / max(self.phase_a_embb_power_projection_count, 1)
            )
            phase_a_embb_power_invalid_or_masked_ratio = float(
                self.phase_a_embb_power_invalid_or_masked_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_raw_saturation_ratio = float(
                self.phase_a_embb_power_raw_saturation_count / max(self.phase_a_embb_power_projection_count, 1)
            )
            # Diversity diagnostics over effective writes.
            # final_std: std of final executed deltas
            # cellwise_diversity: std of final executed scales
            final_mean = float(self.phase_a_embb_power_final_executed_delta_sum / max(self.phase_a_embb_power_projection_count, 1))
            final_var = float(
                (self.phase_a_embb_power_final_delta_sq_sum / max(self.phase_a_embb_power_projection_count, 1)) - final_mean * final_mean
            )
            phase_a_embb_power_final_std = float(np.sqrt(max(final_var, 0.0)))
            scale_mean = float(self.phase_a_embb_power_executed_scale_sum / max(self.phase_a_embb_power_projection_count, 1))
            scale_var = float(
                (self.phase_a_embb_power_executed_scale_sq_sum / max(self.phase_a_embb_power_projection_count, 1)) - scale_mean * scale_mean
            )
            phase_a_embb_power_cellwise_diversity = float(np.sqrt(max(scale_var, 0.0)))
            phase_a_embb_power_zeroed_inactive_head_ratio = float(
                self.phase_a_embb_power_zeroed_inactive_head_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_keep_mode_ratio = float(
                self.phase_a_embb_power_zeroed_keep_mode_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_no_candidate_ratio = float(
                self.phase_a_embb_power_zeroed_no_candidate_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_no_embb_active_ratio = float(
                self.phase_a_embb_power_zeroed_no_embb_active_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_no_owner_ratio = float(
                self.phase_a_embb_power_zeroed_no_owner_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_invalid_owner_ratio = float(
                self.phase_a_embb_power_zeroed_invalid_owner_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_cap_projection_ratio = float(
                self.phase_a_embb_power_zeroed_cap_projection_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_floor_projection_ratio = float(
                self.phase_a_embb_power_zeroed_floor_projection_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_embb_power_zeroed_unknown_ratio = float(
                self.phase_a_embb_power_zeroed_unknown_count / max(self.phase_a_total_decisions, 1)
            )

            # Phase-A negative-only power repair diagnostics.
            phase_a_power_raw_positive_ratio = float(self.phase_a_power_raw_positive_count / max(self.phase_a_total_decisions, 1))
            phase_a_power_positive_ratio = float(self.phase_a_power_positive_executed_count / max(self.phase_a_total_decisions, 1))
            phase_a_power_positive_clamped_to_zero_ratio = float(
                self.phase_a_power_positive_clamped_to_zero_count / max(self.phase_a_power_raw_positive_count, 1)
            )
            phase_a_zero_action_ratio = float(self.phase_a_power_zero_action_count / max(self.phase_a_total_decisions, 1))
            phase_a_exec_delta_arr = np.asarray(getattr(self, "phase_a_executed_delta_values", []), dtype=float)
            if phase_a_exec_delta_arr.size > 0:
                phaseA_delta_mean = float(np.mean(phase_a_exec_delta_arr))
                phaseA_delta_p10 = float(np.percentile(phase_a_exec_delta_arr, 10))
                phaseA_delta_p50 = float(np.percentile(phase_a_exec_delta_arr, 50))
                phaseA_delta_p90 = float(np.percentile(phase_a_exec_delta_arr, 90))
                phaseA_delta_lt_neg09_ratio = float(np.mean(phase_a_exec_delta_arr <= -0.9))
                phase_a_negative_delta_l2_mean = float(np.mean(np.square(np.maximum(0.0, -phase_a_exec_delta_arr))))
                phase_a_negative_delta_saturation_ratio = float(np.mean(np.maximum(0.0, -phase_a_exec_delta_arr) > float(
                    getattr(self.rl_cfg.reward, "phase_a_power_saturation_threshold", 0.9) or 0.9
                )))
            else:
                phaseA_delta_mean = 0.0
                phaseA_delta_p10 = 0.0
                phaseA_delta_p50 = 0.0
                phaseA_delta_p90 = 0.0
                phaseA_delta_lt_neg09_ratio = 0.0
                phase_a_negative_delta_l2_mean = 0.0
                phase_a_negative_delta_saturation_ratio = 0.0
            embb_service_floor_target = float(getattr(self.rl_cfg.reward, "embb_service_floor_target", 0.55) or 0.55)
            embb_service_gap = max(embb_service_floor_target - float(embb_service_ratio), 0.0)
            phaseA_power_reduction_l2_penalty = float(
                (float(getattr(self.rl_cfg.reward, "phase_a_power_reduction_l2_penalty_weight", 0.0) or 0.0))
                * phase_a_negative_delta_l2_mean
            )
            phaseA_power_saturation_penalty = float(
                (float(getattr(self.rl_cfg.reward, "phase_a_power_saturation_penalty_weight", 0.0) or 0.0))
                * phase_a_negative_delta_saturation_ratio
            )
            embb_service_floor_hinge_penalty = float(
                (float(getattr(self.rl_cfg.reward, "embb_service_floor_hinge_penalty_weight", 0.0) or 0.0))
                * (embb_service_gap * embb_service_gap)
            )
            phase_a_power_negative_candidate_ratio = float(
                self.phase_a_power_negative_candidate_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_negative_executed_ratio = float(
                self.phase_a_power_negative_executed_count / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_mean_negative_delta = float(
                self.phase_a_power_negative_executed_delta_sum / max(self.phase_a_power_negative_executed_count, 1)
            )
            phase_a_power_service_guard_reject_ratio = float(
                self.phase_a_power_service_guard_reject_count / max(self.phase_a_power_negative_candidate_count, 1)
            )
            phase_a_power_minrate_guard_reject_ratio = float(
                self.phase_a_power_minrate_guard_reject_count / max(self.phase_a_power_negative_candidate_count, 1)
            )
            phase_a_power_reliability_guard_reject_ratio = float(
                self.phase_a_power_reliability_guard_reject_count / max(self.phase_a_power_negative_candidate_count, 1)
            )
            phase_a_power_total_power_reduction_mean = float(
                self.phase_a_power_total_power_reduction_sum / max(self.phase_a_total_decisions, 1)
            )
            phase_a_power_intercell_reduction_mean = float(
                self.phase_a_power_intercell_reduction_sum / max(self.phase_a_total_decisions, 1)
            )
            urllc_projection_count = max(self.urllc_power_projection_count, 1)
            power_delta_clipped_ratio = float(self.urllc_power_delta_clipped_count / urllc_projection_count)
            power_quantized_ratio = float(self.urllc_power_quantized_count / urllc_projection_count)
            power_cap_hit_ratio = float(self.urllc_power_cap_hit_count / urllc_projection_count)
            power_floor_hit_ratio = float(self.urllc_power_floor_hit_count / urllc_projection_count)
            mean_raw_power_delta = float(self.urllc_raw_power_delta_sum / urllc_projection_count)
            mean_executed_power_delta = float(self.urllc_executed_power_delta_sum / urllc_projection_count)
            admission_via_overlay_ratio = float(self.selected_overlay_admission_count / max(scheduled_packets, 1))
            admission_via_puncture_ratio = float(self.selected_puncture_admission_count / max(scheduled_packets, 1))
            low_interference_admission_bonus_mean = float(self.low_interference_admission_bonus_sum / max(scheduled_packets, 1))
            high_intercell_admission_penalty_mean = float(self.high_intercell_admission_penalty_sum / max(scheduled_packets, 1))
            puncture_candidate_pruned_by_loss_ceiling_ratio = float(self.puncture_candidate_pruned_by_loss_ceiling_count / max(self.puncture_candidate_total, 1))
            feasible_puncture_available_ratio = float(self.feasible_puncture_available_count / max(self.phase_a_total_decisions, 1))
            puncture_chosen_when_feasible_ratio = float(
                self.puncture_chosen_when_feasible_count / max(self.feasible_puncture_available_count, 1)
            )
            overlay_chosen_when_lower_intercell_puncture_available_ratio = float(
                self.overlay_chosen_when_lower_intercell_puncture_available_count / max(self.feasible_puncture_available_count, 1)
            )
            missed_feasible_puncture_ratio = float(
                self.missed_feasible_puncture_count / max(self.feasible_puncture_available_count, 1)
            )
            both_modes_feasible_ratio = float(self.both_modes_feasible_count / max(self.phase_a_total_decisions, 1))
            safe_puncture_available_ratio = float(self.safe_puncture_available_count / max(self.phase_a_total_decisions, 1))
            overlay_chosen_when_safe_puncture_available_ratio = float(
                self.overlay_chosen_when_safe_puncture_available_count / max(self.safe_puncture_available_count, 1)
            )
            puncture_chosen_when_safe_puncture_available_ratio = float(
                self.puncture_chosen_when_safe_puncture_available_count / max(self.safe_puncture_available_count, 1)
            )
            teacher_mode_agreement_ratio = float(
                self.teacher_mode_agreement_count / max(self.mode_anchor_active_count, 1)
            )
            mode_anchor_active_ratio = float(self.mode_anchor_active_count / max(self.phase_a_total_decisions, 1))
            owner_snapshot_in_observation = bool(getattr(self.rl_cfg.env, "owner_snapshot_in_observation", True))
            owner_snapshot_used_for_init = bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_init", True))
            owner_snapshot_used_for_fallback = bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_fallback", True))
            owner_snapshot_used_for_reward = bool(getattr(self.rl_cfg.env, "owner_snapshot_used_for_reward", True))
            owner_snapshot_leak_detected = bool(
                owner_snapshot_in_observation
                or owner_snapshot_used_for_init
                or owner_snapshot_used_for_fallback
                or owner_snapshot_used_for_reward
                or bool(getattr(self, "owner_init_from_snapshot", False))
                or bool(getattr(self, "owner_snapshot_fallback_taken", False))
                or bool(getattr(self.rl_cfg.reward, "use_greedy_terminal_reference", False))
                or bool(getattr(self.rl_cfg.env, "include_greedy_reference_in_obs", False))
            )

            intercell_count = int(self.selected_intercell_interference_count)
            if intercell_count > 0:
                mean_intercell_interference_power = float(self.selected_intercell_interference_sum / intercell_count)
                intercell_interference_nonzero_ratio = float(self.selected_intercell_interference_nonzero_count / intercell_count)
            else:
                mean_intercell_interference_power = 0.0
                intercell_interference_nonzero_ratio = 0.0
            mean_intercell_interference_mw = float(mean_intercell_interference_power * 1.0e3)
            mean_intercell_interference_dbm = float(10.0 * np.log10(max(mean_intercell_interference_mw, 1.0e-12)))
            overlay_intercell_interference_mw = float(
                (self.selected_overlay_intercell_interference_sum / max(self.selected_overlay_intercell_interference_count, 1)) * 1.0e3
            )
            puncture_intercell_interference_mw = float(
                (self.selected_puncture_intercell_interference_sum / max(self.selected_puncture_intercell_interference_count, 1)) * 1.0e3
            )

            # Step-level intercell-aware reward diagnostics (used by trainer/eval/report).
            step_intercell_penalty_mean = float(self.step_intercell_penalty_sum / max(self.step_intercell_penalty_count, 1))
            intercell_penalty_active_ratio = float(self.step_intercell_penalty_active_count / max(self.step_intercell_penalty_count, 1))
            if self.selected_action_intercell_cost_values:
                vals = np.asarray(self.selected_action_intercell_cost_values, dtype=float)
                selected_action_intercell_cost_mean = float(np.mean(vals))
                selected_action_intercell_cost_p95 = float(np.percentile(vals, 95))
            else:
                selected_action_intercell_cost_mean = 0.0
                selected_action_intercell_cost_p95 = 0.0

            intercell_per_admitted_packet = float(
                self.selected_action_intercell_cost_after_source_mask_admit_sum / max(scheduled_packets, 1)
            )

            if self.selected_action_intercell_cost_before_source_mask_values:
                vals = np.asarray(self.selected_action_intercell_cost_before_source_mask_values, dtype=float)
                selected_action_intercell_cost_before_source_mask_mean = float(np.mean(vals))
                selected_action_intercell_cost_before_source_mask_p95 = float(np.percentile(vals, 95))
            else:
                selected_action_intercell_cost_before_source_mask_mean = 0.0
                selected_action_intercell_cost_before_source_mask_p95 = 0.0

            if self.selected_action_intercell_cost_after_source_mask_values:
                vals = np.asarray(self.selected_action_intercell_cost_after_source_mask_values, dtype=float)
                selected_action_intercell_cost_after_source_mask_mean = float(np.mean(vals))
                selected_action_intercell_cost_after_source_mask_p95 = float(np.percentile(vals, 95))
            else:
                selected_action_intercell_cost_after_source_mask_mean = 0.0
                selected_action_intercell_cost_after_source_mask_p95 = 0.0

            # Per-load feasibility diagnostics (mean counts per decision cell).
            phase_a_feasible_overlay_candidates_mean = float(self.phase_a_feasible_overlay_candidate_total / max(self.phase_a_total_decisions, 1))
            phase_a_feasible_puncture_candidates_mean = float(self.phase_a_feasible_puncture_candidate_total / max(self.phase_a_total_decisions, 1))
            phase_a_selected_keep_ratio = float(self.phase_a_selected_keep_total / max(self.phase_a_total_decisions, 1))
            phase_a_selected_overlay_ratio = float(self.phase_a_selected_overlay_total / max(self.phase_a_total_decisions, 1))
            phase_a_selected_puncture_ratio = float(self.phase_a_selected_puncture_total / max(self.phase_a_total_decisions, 1))

            return {
                'embb_total_rate': embb_total_rate,
                'embb_user_rate_mean': embb_user_rate,
                'embb_service_ratio': embb_service_ratio,
                # Single definition: embb_positive_rate_ratio == embb_service_ratio.
                'embb_positive_rate_ratio': embb_service_ratio,
                'embb_service_ratio_after_puncture_deduction': float(embb_service_ratio_after_puncture_deduction),
                'embb_served_users': float(embb_served_users),
                'embb_user_count': float(int(self.sys_cfg.num_embb_users)),
                'urllc_user_count': float(int(self.sys_cfg.num_urllc_users)),
                'embb_urllc_user_ratio': float(int(self.sys_cfg.num_embb_users) / max(int(self.sys_cfg.num_urllc_users), 1)),
                'phase': str(self.rl_cfg.env.phase),
                'learn_embb_baseline': float(bool(self.rl_cfg.env.learn_embb_baseline)),
                'learn_phase0_embb_power': float(bool(getattr(self.rl_cfg.env, "learn_phase0_embb_power", True))),
                'allow_phase_a_embb_power_adjustment': float(bool(self.rl_cfg.env.allow_phase_a_embb_power_adjustment)),
                'allow_phase_a_power_on_keep': float(bool(getattr(self.rl_cfg.env, "allow_phase_a_power_on_keep", False))),
                'phase_a_embb_power_runtime_enabled': float(bool(self._phase_a_embb_power_runtime_enabled())),
                # Snapshot leakage diagnostics (should be all False in no-snapshot debug experiments).
                'owner_snapshot_in_observation': float(owner_snapshot_in_observation),
                'owner_snapshot_used_for_init': float(owner_snapshot_used_for_init),
                'owner_snapshot_used_for_fallback': float(owner_snapshot_used_for_fallback),
                'owner_snapshot_used_for_reward': float(owner_snapshot_used_for_reward),
                'owner_init_from_snapshot': float(bool(getattr(self, "owner_init_from_snapshot", False))),
                'owner_snapshot_fallback_taken': float(bool(getattr(self, "owner_snapshot_fallback_taken", False))),
                'owner_snapshot_leak_detected': float(owner_snapshot_leak_detected),
                'enable_action_masking': float(bool(self.rl_cfg.shield.enable_action_masking)),
                'enable_feasibility_shield': float(bool(self.rl_cfg.shield.enable_feasibility_shield)),
                'apply_joint_reliability_rewrite': float(bool(self.rl_cfg.shield.apply_joint_reliability_rewrite)),
                'enable_greedy_fallback': float(bool(self.rl_cfg.shield.enable_greedy_fallback)),
                'urllc_admission_rate': admission_ratio,
                'urllc_success_rate': admitted_reliability,
                'admitted_urllc_reliability': admitted_reliability,
                'effective_urllc_success_over_arrivals': float(effective_urllc_success_over_arrivals),
                'empty_admission_case': float(empty_admission_case),
                # URLLC throughput definition inputs (slot-based; useful for report-side recomputation).
                'urllc_slot_duration_s': float(slot_duration_s),
                'urllc_packet_bits_mean': float(packet_bits),
                'greedy_urllc_budget_bps': float(self.greedy_urllc_budget_bps if np.isfinite(self.greedy_urllc_budget_bps) else 0.0),
                'greedy_urllc_budget_used_bps': float(self.greedy_urllc_budget_used_bps),
                'greedy_urllc_budget_utilization_ratio': float(
                    self.greedy_urllc_budget_used_bps / max(self.greedy_urllc_budget_bps, 1.0e-9)
                ) if np.isfinite(self.greedy_urllc_budget_bps) and self.greedy_urllc_budget_bps > 0.0 else 0.0,
                'profile_reset_total_sec': float(self.profile_reset_total_sec),
                'profile_prepare_slot_context_sec': float(self.profile_prepare_slot_context_sec),
                'profile_arrival_generation_sec': float(self.profile_arrival_generation_sec),
                'profile_hf_action_calls': float(self.profile_hf_action_calls),
                'profile_hf_prefilter_sec': float(self.profile_hf_prefilter_sec),
                'profile_hf_eval_sec': float(self.profile_hf_eval_sec),
                'profile_hf_fastpath_sec': float(self.profile_hf_fastpath_sec),
                'greedy_embb_loss_share_cap_ratio': float(
                    self.greedy_embb_loss_share_cap_ratio if np.isfinite(self.greedy_embb_loss_share_cap_ratio) else 0.0
                ),
                'greedy_hf_decision_count': float(self.greedy_hf_decision_count),
                'greedy_hf_candidate_evaluated_per_decision': float(
                    self.greedy_hf_candidate_evaluated_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_candidate_feasible_per_decision': float(
                    self.greedy_hf_candidate_feasible_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_reject_reliability_per_decision': float(
                    self.greedy_hf_candidate_reject_reliability_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_reject_power_per_decision': float(
                    self.greedy_hf_candidate_reject_power_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_reject_min_rate_per_decision': float(
                    self.greedy_hf_candidate_reject_min_rate_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_reject_share_cap_per_decision': float(
                    self.greedy_hf_candidate_reject_share_cap_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_reject_reliability_ratio': float(
                    self.greedy_hf_candidate_reject_reliability_total / max(self.greedy_hf_candidate_evaluated_total, 1)
                ),
                'greedy_hf_reject_power_ratio': float(
                    self.greedy_hf_candidate_reject_power_total / max(self.greedy_hf_candidate_evaluated_total, 1)
                ),
                'greedy_hf_reject_min_rate_ratio': float(
                    self.greedy_hf_candidate_reject_min_rate_total / max(self.greedy_hf_candidate_evaluated_total, 1)
                ),
                'greedy_hf_reject_share_cap_ratio': float(
                    self.greedy_hf_candidate_reject_share_cap_total / max(self.greedy_hf_candidate_evaluated_total, 1)
                ),
                'greedy_hf_feasible_ratio': float(
                    self.greedy_hf_candidate_feasible_total / max(self.greedy_hf_candidate_evaluated_total, 1)
                ),
                'greedy_hf_selected_overlay_ratio': float(
                    self.greedy_hf_selected_overlay_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_selected_puncture_ratio': float(
                    self.greedy_hf_selected_puncture_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_selected_keep_ratio': float(
                    self.greedy_hf_selected_keep_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_no_candidate_per_decision': float(
                    self.greedy_hf_no_candidate_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_all_rejected_per_decision': float(
                    self.greedy_hf_all_rejected_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_budget_exhausted_keep_per_decision': float(
                    self.greedy_hf_budget_exhausted_keep_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_prefilter_pair_per_decision': float(
                    self.greedy_hf_prefilter_pair_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_prefilter_block_mode_mask_per_decision': float(
                    self.greedy_hf_prefilter_block_mode_mask_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_prefilter_block_packet_mask_per_decision': float(
                    self.greedy_hf_prefilter_block_packet_mask_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_prefilter_block_mode_infeasible_per_decision': float(
                    self.greedy_hf_prefilter_block_mode_infeasible_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_prefilter_block_mode_mask_ratio': float(
                    self.greedy_hf_prefilter_block_mode_mask_total / max(self.greedy_hf_prefilter_pair_total, 1)
                ),
                'greedy_hf_prefilter_block_packet_mask_ratio': float(
                    self.greedy_hf_prefilter_block_packet_mask_total / max(self.greedy_hf_prefilter_pair_total, 1)
                ),
                'greedy_hf_prefilter_block_mode_infeasible_ratio': float(
                    self.greedy_hf_prefilter_block_mode_infeasible_total / max(self.greedy_hf_prefilter_pair_total, 1)
                ),
                'greedy_hf_no_candidate_block_mode_mask_per_no_candidate': float(
                    self.greedy_hf_no_candidate_block_mode_mask_total / max(self.greedy_hf_no_candidate_total, 1)
                ),
                'greedy_hf_no_candidate_block_packet_mask_per_no_candidate': float(
                    self.greedy_hf_no_candidate_block_packet_mask_total / max(self.greedy_hf_no_candidate_total, 1)
                ),
                'greedy_hf_no_candidate_block_mode_infeasible_per_no_candidate': float(
                    self.greedy_hf_no_candidate_block_mode_infeasible_total / max(self.greedy_hf_no_candidate_total, 1)
                ),
                'greedy_hf_no_candidate_empty_observation_per_decision': float(
                    self.greedy_hf_no_candidate_empty_observation_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_no_candidate_mask_block_per_decision': float(
                    self.greedy_hf_no_candidate_mask_block_total / max(self.greedy_hf_decision_count, 1)
                ),
                'greedy_hf_no_candidate_empty_observation_given_no_candidate_ratio': float(
                    self.greedy_hf_no_candidate_empty_observation_total / max(self.greedy_hf_no_candidate_total, 1)
                ),
                'greedy_hf_no_candidate_mask_block_given_no_candidate_ratio': float(
                    self.greedy_hf_no_candidate_mask_block_total / max(self.greedy_hf_no_candidate_total, 1)
                ),
                'greedy_hf_no_candidate_given_no_feasible_ratio': float(
                    self.greedy_hf_no_candidate_total / max(
                        self.greedy_hf_no_candidate_total
                        + self.greedy_hf_all_rejected_total
                        + self.greedy_hf_budget_exhausted_keep_total,
                        1,
                    )
                ),
                'greedy_hf_all_rejected_given_no_feasible_ratio': float(
                    self.greedy_hf_all_rejected_total / max(
                        self.greedy_hf_no_candidate_total
                        + self.greedy_hf_all_rejected_total
                        + self.greedy_hf_budget_exhausted_keep_total,
                        1,
                    )
                ),
                'greedy_hf_budget_exhausted_given_no_feasible_ratio': float(
                    self.greedy_hf_budget_exhausted_keep_total / max(
                        self.greedy_hf_no_candidate_total
                        + self.greedy_hf_all_rejected_total
                        + self.greedy_hf_budget_exhausted_keep_total,
                        1,
                    )
                ),
                # New explicit names (slot-based estimates).
                'urllc_throughput_bps_slot_est': float(urllc_throughput_bps_slot_est),
                'urllc_throughput_mbps_slot_est': float(urllc_throughput_mbps_slot_est),
                # Backward-compatible aliases (deprecated): keep the same values.
                'urllc_throughput_bps_est': float(urllc_throughput_bps_slot_est),
                'urllc_throughput_mbps_est': float(urllc_throughput_mbps_slot_est),
                'unscheduled_ratio': unscheduled_ratio,
                'embb_min_rate_shortfall': embb_min_rate_shortfall,
                'embb_min_rate_satisfaction_ratio': float(embb_min_rate_satisfaction_ratio),
                'embb_min_rate_satisfaction_after_puncture_deduction': float(embb_min_rate_satisfaction_after_puncture_deduction),
                'embb_served_user_count': float(embb_served_users),
                'total_power': total_power,
                'embb_power': embb_slot_avg_power,
                'urllc_power': urllc_slot_avg_power,
                'throughput_per_watt': throughput_per_watt,
                'avg_throughput_per_served_embb_user': avg_throughput_per_served_embb_user,
                # Inter-cell interference rate loss diagnostics (counterfactual, report-only semantics).
                'embb_rate_with_intercell': float(embb_rate_with_intercell),
                'embb_rate_without_intercell_est': float(embb_rate_without_intercell_est),
                'embb_rate_loss_due_to_intercell': float(embb_rate_loss_due_to_intercell),
                'embb_rate_loss_due_to_intercell_ratio': float(embb_rate_loss_due_to_intercell_ratio),
                'overlay_rate_with_intercell': float(overlay_rate_with_intercell),
                'overlay_rate_without_intercell_est': float(overlay_rate_no_intercell),
                'overlay_rate_loss_due_to_intercell': float(overlay_rate_loss_due_to_intercell),
                'puncture_rate_with_intercell': float(puncture_rate_with_intercell),
                'puncture_rate_without_intercell_est': float(puncture_rate_no_intercell),
                'puncture_rate_loss_due_to_intercell': float(puncture_rate_loss_due_to_intercell),
                # Corrected effective eMBB throughput diagnostics (puncture-deducted + masked intercell sources).
                'embb_rate_with_intercell_after_puncture_deduction': float(embb_rate_with_intercell_after_puncture_deduction),
                'no_intercell_rate_with_same_puncture_mask': float(no_intercell_rate_with_same_puncture_mask),
                'intercell_rate_loss_with_same_puncture_mask': float(intercell_rate_loss_with_same_puncture_mask),
                # Local puncture airtime deduction diagnostics.
                'local_punctured_embb_airtime_ratio': float(local_punctured_embb_airtime_ratio),
                'embb_rate_raw_before_local_puncture_deduction': float(embb_rate_raw_before_local_puncture_deduction),
                'embb_rate_after_local_puncture_deduction': float(embb_rate_after_local_puncture_deduction),
                'embb_rate_loss_due_to_local_puncture': float(embb_rate_loss_due_to_local_puncture),
                'embb_rate_loss_due_to_local_puncture_ratio': float(embb_rate_loss_due_to_local_puncture_ratio),
                'local_puncture_mask_nonzero_count': float(local_puncture_mask_nonzero_count),
                'executed_puncture_action_count': float(executed_puncture_action_count),
                'thin_service_fraction': thin_service_fraction,
                # Floors/targets used by the current reward preset (useful for fast debug plots).
                'terminal_embb_service_floor_used': float(getattr(self, "terminal_embb_service_floor_used", 0.0) or 0.0),
                'terminal_embb_min_rate_floor_used': float(getattr(self, "terminal_embb_min_rate_floor_used", 0.0) or 0.0),
                'terminal_embb_served_user_target': float(getattr(self, "terminal_embb_served_user_target", 0.0) or 0.0),
                'terminal_admission_floor_used': float(getattr(self, "terminal_admission_floor_used", 0.0) or 0.0),
                'terminal_admission_floor_weight_used': float(getattr(self, "terminal_admission_floor_weight_used", 0.0) or 0.0),
                'mean_intercell_interference_power': float(mean_intercell_interference_power),
                'mean_intercell_interference_mw': float(mean_intercell_interference_mw),
                'mean_intercell_interference_dbm': float(mean_intercell_interference_dbm),
                # Intercell source-mask diagnostics (eMBB puncture removes the transmitter in that minislot).
                'intercell_source_mask_excluded_by_puncture_ratio': float(intercell_source_mask_excluded_by_puncture_ratio),
                'intercell_source_active_ratio': float(intercell_source_active_ratio),
                'intercell_power_before_puncture_source_mask_mean': float(intercell_power_before_puncture_source_mask_mean),
                'intercell_power_after_puncture_source_mask_mean': float(intercell_power_after_puncture_source_mask_mean),
                'intercell_power_reduction_from_source_mask_mean': float(intercell_power_reduction_from_source_mask_mean),
                'intercell_interference_nonzero_ratio': float(intercell_interference_nonzero_ratio),
                'overlay_intercell_interference_mw': float(overlay_intercell_interference_mw),
                'puncture_intercell_interference_mw': float(puncture_intercell_interference_mw),
                'intercell_penalty_active_ratio': float(intercell_penalty_active_ratio),
                'intercell_reward_component_mean': float(-step_intercell_penalty_mean),
                'selected_action_intercell_cost_mean': float(selected_action_intercell_cost_mean),
                'selected_action_intercell_cost_p95': float(selected_action_intercell_cost_p95),
                'selected_action_intercell_cost_after_source_mask_mean': float(selected_action_intercell_cost_after_source_mask_mean),
                'selected_action_intercell_cost_after_source_mask_p95': float(selected_action_intercell_cost_after_source_mask_p95),
                'selected_action_intercell_cost_before_source_mask_mean': float(selected_action_intercell_cost_before_source_mask_mean),
                'selected_action_intercell_cost_before_source_mask_p95': float(selected_action_intercell_cost_before_source_mask_p95),
                'intercell_per_admitted_packet': float(intercell_per_admitted_packet),
                'phase_a_feasible_overlay_candidates_mean': float(phase_a_feasible_overlay_candidates_mean),
                'phase_a_feasible_puncture_candidates_mean': float(phase_a_feasible_puncture_candidates_mean),
                'phase_a_selected_keep_ratio': float(phase_a_selected_keep_ratio),
                'phase_a_selected_overlay_ratio': float(phase_a_selected_overlay_ratio),
                'phase_a_selected_puncture_ratio': float(phase_a_selected_puncture_ratio),
                'feasible_overlay_good_ratio': float(
                    self.mode_balance_good_overlay_available_count / max(self.phase_a_total_decisions, 1)
                ),
                'overlay_chosen_when_good_ratio': float(
                    self.mode_balance_overlay_chosen_when_good_count / max(self.mode_balance_good_overlay_available_count, 1)
                ),
                'puncture_chosen_when_good_overlay_available_ratio': float(
                    self.mode_balance_puncture_chosen_when_good_overlay_count / max(self.mode_balance_good_overlay_available_count, 1)
                ),
                'mode_balance_selected_overlay_cost_mean': float(
                    np.mean(np.asarray(self.mode_balance_selected_overlay_cost_values, dtype=float))
                ) if self.mode_balance_selected_overlay_cost_values else 0.0,
                'mode_balance_selected_puncture_cost_mean': float(
                    np.mean(np.asarray(self.mode_balance_selected_puncture_cost_values, dtype=float))
                ) if self.mode_balance_selected_puncture_cost_values else 0.0,
                'phase_a_rejected_intercell_per_decision': float(self.phase_a_rejected_intercell_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_min_rate_per_decision': float(self.phase_a_rejected_min_rate_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_power_guard_per_decision': float(self.phase_a_rejected_power_guard_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_collision_per_decision': float(self.phase_a_rejected_collision_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_deadline_per_decision': float(self.phase_a_rejected_deadline_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_per_decision': float(self.phase_a_rejected_other_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_gain_ratio_per_decision': float(self.phase_a_rejected_other_gain_ratio_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_overlay_margin_per_decision': float(self.phase_a_rejected_other_overlay_margin_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_overlay_positive_gate_per_decision': float(self.phase_a_rejected_other_overlay_positive_gate_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_no_overlay_owner_per_decision': float(self.phase_a_rejected_other_no_overlay_owner_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_overlay_reliability_per_decision': float(self.phase_a_rejected_other_overlay_reliability_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_overlay_sic_per_decision': float(self.phase_a_rejected_other_overlay_sic_total / max(self.phase_a_total_decisions, 1)),
                'phase_a_rejected_other_gain_ratio_given_other_ratio': float(self.phase_a_rejected_other_gain_ratio_total / max(self.phase_a_rejected_other_total, 1)),
                'phase_a_rejected_other_overlay_margin_given_other_ratio': float(self.phase_a_rejected_other_overlay_margin_total / max(self.phase_a_rejected_other_total, 1)),
                'phase_a_rejected_other_overlay_positive_gate_given_other_ratio': float(self.phase_a_rejected_other_overlay_positive_gate_total / max(self.phase_a_rejected_other_total, 1)),
                'phase_a_rejected_other_no_overlay_owner_given_other_ratio': float(self.phase_a_rejected_other_no_overlay_owner_total / max(self.phase_a_rejected_other_total, 1)),
                'phase_a_rejected_other_overlay_reliability_given_other_ratio': float(self.phase_a_rejected_other_overlay_reliability_total / max(self.phase_a_rejected_other_total, 1)),
                'phase_a_rejected_other_overlay_sic_given_other_ratio': float(self.phase_a_rejected_other_overlay_sic_total / max(self.phase_a_rejected_other_total, 1)),
                'overlay_ratio': overlay_ratio,
                'puncture_ratio': puncture_ratio,
                'overlay_count': float(overlay_count),
                'puncture_count': float(puncture_count),
                'overlay_candidate_pairs': float(self.overlay_candidate_pairs),
                'overlay_feasible_pairs': float(self.overlay_feasible_pairs),
                'overlay_selected_pairs': float(self.overlay_selected_pairs),
                'overlay_utilization_ratio': float(overlay_utilization),
                'scheduled_packets': float(scheduled_packets),
                'scheduled_packets_per_uav': scheduled_packets_per_uav,
                'active_packets': float(active_packets),
                'avg_overlay_retention': float(np.mean(self.selected_overlay_retentions)) if self.selected_overlay_retentions else 0.0,
                'avg_puncture_embb_loss': float(np.mean(self.selected_puncture_losses)) if self.selected_puncture_losses else 0.0,
                'avg_overlay_embb_loss': float(np.mean(self.selected_overlay_losses)) if self.selected_overlay_losses else 0.0,
                'admission_via_overlay_ratio': admission_via_overlay_ratio,
                'admission_via_puncture_ratio': admission_via_puncture_ratio,
                'low_interference_admission_bonus_mean': float(low_interference_admission_bonus_mean),
                'high_intercell_admission_penalty_mean': float(high_intercell_admission_penalty_mean),
                'overlay_selection_ratio': overlay_selection_ratio,
                'puncture_selection_ratio': puncture_selection_ratio,
                'both_modes_feasible_ratio': both_modes_feasible_ratio,
                'safe_puncture_available_ratio': safe_puncture_available_ratio,
                'feasible_puncture_available_ratio': feasible_puncture_available_ratio,
                'puncture_chosen_when_feasible_ratio': puncture_chosen_when_feasible_ratio,
                'overlay_chosen_when_lower_intercell_puncture_available_ratio': overlay_chosen_when_lower_intercell_puncture_available_ratio,
                'missed_feasible_puncture_ratio': missed_feasible_puncture_ratio,
                'overlay_chosen_when_safe_puncture_available_ratio': overlay_chosen_when_safe_puncture_available_ratio,
                'puncture_chosen_when_safe_puncture_available_ratio': puncture_chosen_when_safe_puncture_available_ratio,
                'teacher_mode_agreement_ratio': teacher_mode_agreement_ratio,
                'mode_anchor_active_ratio': mode_anchor_active_ratio,
                'puncture_candidate_pruned_by_loss_ceiling_ratio': puncture_candidate_pruned_by_loss_ceiling_ratio,
                'phase_a_feasible_candidate_pairs': float(self.phase_a_feasible_candidate_pairs),
                'jain_fairness': jain_fairness,
                'embb_only_fraction': embb_only_fraction,
                'overlay_fraction': overlay_fraction,
                'puncture_fraction': puncture_fraction,
                'idle_fraction': idle_fraction,
                'minislot_utilization': float(minislot_utilization),
                'planning_total_decisions': float(self.planning_total_decisions),
                'planning_owner_non_null_count': float(self.planning_owner_non_null_count),
                'planning_embb_power_nonzero_count': float(self.planning_embb_power_nonzero_count),
                'planning_embb_power_changed_count': float(self.planning_embb_power_changed_count),
                'planning_owner_change_count': float(planning_owner_change_count),
                'planning_owner_non_null_ratio': planning_owner_non_null_ratio,
                'planning_owner_change_ratio': planning_owner_change_ratio,
                'planning_embb_power_nonzero_ratio': planning_embb_power_nonzero_ratio,
                'planning_embb_power_changed_ratio': planning_embb_power_changed_ratio,
                'planning_owner_rewrite_count': float(self.planning_owner_rewrite_count),
                'planning_owner_rewrite_ratio': planning_owner_rewrite_ratio,
                'phase0_owner_non_null_ratio_raw': phase0_owner_non_null_ratio_raw,
                'phase0_owner_non_null_ratio_executed': phase0_owner_non_null_ratio_executed,
                'phase0_owner_change_ratio_vs_snapshot_raw': phase0_owner_change_ratio_vs_snapshot_raw,
                'phase0_owner_change_ratio_vs_snapshot_executed': phase0_owner_change_ratio_vs_snapshot_executed,
                'phase0_owner_fallback_to_candidate0_ratio': phase0_owner_fallback_to_candidate0_ratio,
                'phase0_owner_invalid_option_ratio': phase0_owner_invalid_option_ratio,
                'phase0_owner_null_selected_ratio': phase0_owner_null_selected_ratio,
                'phase0_owner_invalid_to_null_ratio': phase0_owner_invalid_to_null_ratio,
                'phase0_owner_invalid_to_snapshot_ratio': phase0_owner_invalid_to_snapshot_ratio,
                'phase0_owner_invalid_to_non_snapshot_ratio': phase0_owner_invalid_to_non_snapshot_ratio,
                'phase0_owner_restored_to_snapshot_ratio': phase0_owner_restored_to_snapshot_ratio,
                'phase0_owner_kept_null_ratio': phase0_owner_kept_null_ratio,
                'phase0_owner_replaced_with_non_snapshot_ratio': phase0_owner_replaced_with_non_snapshot_ratio,
                'ph0_owner_raw_same_as_snapshot_ratio': phase0_owner_raw_same_as_snapshot_ratio,
                'ph0_owner_raw_non_snapshot_ratio': phase0_owner_raw_non_snapshot_ratio,
                'ph0_owner_raw_null_ratio': phase0_owner_raw_null_ratio,
                'ph0_owner_exec_same_as_snapshot_ratio': phase0_owner_exec_same_as_snapshot_ratio,
                'ph0_owner_exec_non_snapshot_ratio': phase0_owner_exec_non_snapshot_ratio,
                'ph0_owner_reverted_to_snapshot_ratio': phase0_owner_reverted_to_snapshot_ratio,
                'phase0_owner_guard_rewrite_ratio': float(phase0_owner_guard_rewrite_ratio),
                'phase0_owner_service_violation_ratio': float(phase0_owner_service_violation_ratio),
                'phase0_owner_rate_violation_ratio': float(phase0_owner_rate_violation_ratio),
                'phase0_owner_change_budget_used': float(phase0_owner_change_budget_used),
                'phase0_owner_change_budget_allowed': float(phase0_owner_change_budget_allowed),
                'phase0_owner_change_budget_clipped_ratio': float(phase0_owner_change_budget_clipped_ratio),
                'phase0_owner_change_kept_topk_ratio': float(phase0_owner_change_kept_topk_ratio),
                'phase0_owner_change_dropped_over_budget_ratio': float(phase0_owner_change_dropped_over_budget_ratio),
                'owner_dropped_raw_churn_ratio': float(phase0_owner_change_dropped_over_budget_ratio),
                'phase0_owner_raw_changed_count_mean': float(phase0_owner_raw_changed_count_mean),
                'phase0_owner_allowed_k_mean': float(phase0_owner_allowed_k_mean),
                'phase0_owner_executed_changed_count_mean': float(phase0_owner_executed_changed_count_mean),
                'phase0_owner_dropped_count_mean': float(phase0_owner_dropped_count_mean),
                'phase0_owner_budget_min_one_rule_eligible_ratio': float(phase0_owner_budget_min_one_rule_eligible_ratio),
                'phase0_owner_budget_min_one_rule_applied_ratio': float(phase0_owner_budget_min_one_rule_applied_ratio),
                'phase0_owner_min_one_blocked_by_no_positive_candidate_ratio': float(phase0_owner_min_one_blocked_by_no_positive_candidate_ratio),
                'phase0_owner_accepted_positive_service_gain_ratio': float(phase0_owner_accepted_positive_service_gain_ratio),
                'phase0_owner_accepted_negative_service_gain_ratio': float(phase0_owner_accepted_negative_service_gain_ratio),
                'phase0_owner_candidate_positive_objective_ratio': float(phase0_owner_candidate_positive_objective_ratio),
                'owner_positive_candidate_count_mean': float(phase0_owner_positive_candidate_count_mean),
                'owner_candidate_relaxed_ratio': float(phase0_owner_candidate_relaxed_ratio),
                'owner_candidate_fallback_used_ratio': float(phase0_owner_candidate_fallback_used_ratio),
                'owner_obj_mean': float(phase0_owner_obj_mean),
                'owner_obj_std': float(phase0_owner_obj_std),
                'owner_gate_threshold': float(phase0_owner_gate_threshold),
                'owner_candidate_after_gate_ratio': float(phase0_owner_candidate_after_gate_ratio),
                'owner_neg_accept_ratio': float(phase0_owner_neg_accept_ratio),
                'owner_neg_accept_clipped_ratio': float(phase0_owner_neg_accept_clipped_ratio),
                'owner_neg_rejected_by_quota_ratio': float(phase0_owner_neg_rejected_by_quota_ratio),
                'owner_pos_selected_ratio': float(phase0_owner_pos_selected_ratio),
                'owner_neg_selected_count': float(phase0_owner_neg_selected_count_mean),
                'owner_pos_selected_count': float(phase0_owner_pos_selected_count_mean),
                'owner_selected_positive_count_mean': float(phase0_owner_pos_selected_count_mean),
                'owner_selected_negative_count_mean': float(phase0_owner_selected_negative_count_mean),
                'owner_selected_count': float(phase0_owner_selected_count),
                'owner_allowed_k': float(phase0_owner_allowed_k),
                'owner_final_selected_count': float(phase0_owner_final_selected_count_mean),
                'owner_final_pos_selected_count': float(phase0_owner_final_pos_selected_count_mean),
                'owner_final_neg_selected_count': float(phase0_owner_final_neg_selected_count_mean),
                'owner_final_safe_relax_selected_count': float(phase0_owner_final_safe_relax_selected_count_mean),
                'owner_final_keep_set_size': float(phase0_owner_final_keep_set_size_mean),
                'owner_selection_fill_ratio': float(phase0_owner_selection_fill_ratio),
                'owner_positive_shortage_ratio': float(phase0_owner_positive_shortage_ratio),
                'owner_negative_blocked_due_to_quota_ratio': float(phase0_owner_negative_blocked_due_to_quota_ratio),
                'owner_safe_relaxed_used_ratio': float(owner_safe_relaxed_used_ratio),
                'owner_safe_relaxed_candidate_count': float(owner_safe_relaxed_candidate_count),
                'owner_safe_relaxed_selected_count': float(owner_safe_relaxed_selected_count),
                'owner_safe_relax_selected_count_mean': float(owner_safe_relaxed_selected_count),
                'owner_safe_relaxed_avg_objective': float(owner_safe_relaxed_avg_objective),
                'owner_safe_relaxed_service_delta_mean': float(owner_safe_relaxed_service_delta_mean),
                'owner_safe_relaxed_intercell_delta_mean': float(owner_safe_relaxed_intercell_delta_mean),
                'owner_near_zero_objective_ratio': float(owner_near_zero_objective_ratio),
                'owner_positive_after_relax_ratio': float(owner_positive_after_relax_ratio),
                'owner_safe_relax_disabled_ratio': float(owner_safe_relax_disabled_ratio),
                'owner_safe_relax_off_ratio': float(owner_safe_relax_disabled_ratio),
                'phase0_owner_accepted_positive_objective_ratio': float(phase0_owner_accepted_positive_objective_ratio),
                'phase0_owner_rejected_nonpositive_objective_ratio': float(phase0_owner_rejected_nonpositive_objective_ratio),
                'phase0_owner_objective_gain_mean': float(phase0_owner_objective_gain_mean),
                'owner_objective_gain_pre_filter_mean': float(phase0_owner_objective_gain_pre_filter_mean),
                'owner_objective_gain_post_filter_mean': float(phase0_owner_objective_gain_post_filter_mean),
                'owner_negative_but_accepted_ratio': float(phase0_owner_negative_but_accepted_ratio),
                'owner_neg_accepted_with_positive_candidate_ratio': float(phase0_owner_neg_accepted_with_positive_candidate_ratio),
                'phase0_owner_objective_gain_accepted_mean': float(phase0_owner_objective_gain_accepted_mean),
                'phase0_owner_effective_rate_gain_accepted_mean': float(phase0_owner_effective_rate_gain_accepted_mean),
                'phase0_owner_intercell_reduction_accepted_mean': float(phase0_owner_intercell_reduction_accepted_mean),
                'phase0_owner_service_gain_accepted_mean': float(phase0_owner_service_gain_accepted_mean),
                'phase0_owner_minrate_gain_accepted_mean': float(phase0_owner_minrate_gain_accepted_mean),
                'phase0_owner_harmful_accepted_ratio': float(phase0_owner_harmful_accepted_ratio),
                'phase0_owner_changed_and_effective_ratio': phase0_owner_changed_and_effective_ratio,
                'phase0_owner_changed_but_unserved_ratio': float(phase0_owner_changed_but_unserved_ratio),
                'phase0_owner_same_as_snapshot_ratio': float(phase0_owner_same_as_snapshot_ratio),
                'phase0_owner_executed_change_count': float(getattr(self, "phase0_owner_executed_change_count", 0.0) or 0.0),
                'phase0_owner_positive_service_gain_change_count': float(getattr(self, "phase0_owner_positive_service_gain_change_count", 0.0) or 0.0),
                'owner_change_service_conversion_ratio': float(
                    (float(getattr(self, "phase0_owner_positive_service_gain_change_count", 0.0) or 0.0))
                    / max(float(getattr(self, "phase0_owner_executed_change_count", 0.0) or 0.0), 1.0)
                ),
                'phase0_owner_effective_service_gain_ratio': float(phase0_owner_effective_service_gain_ratio),
                # Backward-compatible alias (legacy dashboards referenced this shorter key name).
                'owner_effective_service_gain_ratio': float(phase0_owner_effective_service_gain_ratio),
                'phase0_owner_effective_rate_gain_vs_snapshot_mean': float(phase0_owner_effective_rate_gain_vs_snapshot_mean),
                'phase0_owner_change_harmful_ratio': float(owner_change_harmful_ratio),
                'phase0_owner_change_detail_top': owner_change_detail_top,
                'owner_change_counterfactual_service_gain': float(owner_change_counterfactual_service_gain),
                'owner_change_counterfactual_intercell_gain': float(owner_change_counterfactual_intercell_gain),
                'owner_change_counterfactual_objective_gain': float(owner_change_counterfactual_objective_gain),
                'owner_rejected_by_snapshot_imitation_ratio': float(owner_rejected_by_snapshot_imitation_ratio),
                'owner_rejected_by_hard_feasibility_ratio': float(owner_rejected_by_hard_feasibility_ratio),
                'owner_accepted_low_intercell_non_greedy_ratio': float(owner_accepted_low_intercell_non_greedy_ratio),
                # Mean per-cell base-rate gain (only over changed+effective cells), in Mbps.
                'phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps': float(phase0_owner_effective_rate_gain_vs_snapshot_cells_mean),
                'phase0_owner_effective_change_count': phase0_owner_effective_change_count,
                'planning_projected_embb_rate_ratio_mean': planning_projected_embb_rate_ratio_mean,
                'planning_projected_embb_rate_ratio_min': planning_projected_embb_rate_ratio_min,
                'planning_projected_embb_power_ratio_mean': planning_projected_embb_power_ratio_mean,
                'planning_projected_embb_power_ratio_max': planning_projected_embb_power_ratio_max,
                'planning_owner_rate_floor_violation_count': float(self.planning_owner_rate_floor_violation_count),
                'planning_owner_power_ceiling_violation_count': float(self.planning_owner_power_ceiling_violation_count),
                'planning_owner_guard_violation_count': float(self.planning_owner_guard_violation_count),
                'phase_a_total_decisions': float(self.phase_a_total_decisions),
                'phase_a_embb_power_write_count': float(self.phase_a_embb_power_write_count),
                'phase_a_embb_power_changed_count': float(self.phase_a_embb_power_changed_count),
                'phase_a_embb_power_write_ratio': phase_a_embb_power_write_ratio,
                'phase_a_embb_power_changed_ratio': phase_a_embb_power_changed_ratio,
                'phase_a_power_zeroed_non_admission_count': float(self.phase_a_power_zeroed_non_admission_count),
                'phase_a_power_zeroed_non_admission_ratio': float(phase_a_power_zeroed_non_admission_ratio),
                'phase_a_power_write_on_admission_ratio': float(phase_a_power_write_on_admission_ratio),
                'phase_a_power_write_on_keep_ratio': float(phase_a_power_write_on_keep_ratio),
                'phaseA_zero_by_keep_due_to_mode_gate_ratio': float(phase_a_zero_by_keep_due_to_mode_gate_ratio),
                'phaseA_keep_power_write_attempt_ratio': float(phase_a_keep_power_write_attempt_ratio),
                'phaseA_keep_power_write_success_ratio': float(phase_a_keep_power_write_success_ratio),
                'phaseA_power_write_blocked_no_owner_ratio': float(phase_a_power_write_blocked_no_owner_ratio),
                'phaseA_power_write_blocked_projection_ratio': float(phase_a_power_write_blocked_projection_ratio),
                'phase_a_keep_power_write_attempt_count': float(self.phase_a_keep_power_write_attempt_count),
                'phase_a_keep_power_write_success_count': float(self.phase_a_keep_power_write_success_count),
                'phase_a_zero_by_keep_due_to_mode_gate_count': float(self.phase_a_zero_by_keep_due_to_mode_gate_count),
                'phase_a_power_write_blocked_no_owner_count': float(self.phase_a_power_write_blocked_no_owner_count),
                'phase_a_power_write_blocked_projection_count': float(self.phase_a_power_write_blocked_projection_count),
                'action_intercell_guard_active_ratio': float(action_intercell_guard_active_ratio),
                'action_intercell_guard_candidate_active_ratio': float(action_intercell_guard_candidate_active_ratio),
                'action_intercell_guard_selected_violation_ratio': float(action_intercell_guard_selected_violation_ratio),
                'action_intercell_guard_local_min_cost_mean': float(action_intercell_guard_local_min_cost_mean),
                'action_intercell_guard_selected_excess_mean': float(action_intercell_guard_selected_excess_mean),
                'action_intercell_guard_masked_option_count': float(self.action_intercell_guard_masked_option_count),
                'phase_a_embb_power_mean_abs_change': phase_a_embb_power_mean_abs_change,
                'phase_a_embb_power_mean_raw_delta': phase_a_embb_power_mean_raw_delta,
                'phase_a_embb_power_mean_executed_delta': phase_a_embb_power_mean_executed_delta,
                'phase_a_embb_power_pre_clip_mean_delta': phase_a_embb_power_pre_clip_mean_delta,
                'phase_a_embb_power_post_clip_mean_delta': phase_a_embb_power_post_clip_mean_delta,
                'phase_a_embb_power_post_quant_mean_delta': phase_a_embb_power_post_quant_mean_delta,
                'phase_a_embb_power_post_projection_mean_delta': phase_a_embb_power_post_projection_mean_delta,
                'phase_a_embb_power_post_owner_validation_mean_delta': phase_a_embb_power_post_owner_validation_mean_delta,
                'phase_a_embb_power_final_executed_mean_delta': phase_a_embb_power_final_executed_mean_delta,
                'phase_a_embb_power_clip_ratio': phase_a_embb_power_clip_ratio,
                'phase_a_embb_power_quantized_ratio': phase_a_embb_power_quantized_ratio,
                'phase_a_embb_power_projection_ratio': phase_a_embb_power_projection_ratio,
                'phase_a_embb_power_cap_hit_ratio': phase_a_embb_power_cap_hit_ratio,
                'phase_a_embb_power_floor_hit_ratio': phase_a_embb_power_floor_hit_ratio,
                'phase_a_embb_power_sign_flip_ratio': phase_a_embb_power_sign_flip_ratio,
                'phase_a_embb_power_abs_shrink_ratio': phase_a_embb_power_abs_shrink_ratio,
                'phase_a_embb_power_projection_l2_mean': phase_a_embb_power_projection_l2_mean,
                'phase_a_embb_power_pre_vs_final_l1_mean': phase_a_embb_power_pre_vs_final_l1_mean,
                'phase_a_embb_power_pre_vs_final_sign_consistency': phase_a_embb_power_pre_vs_final_sign_consistency,
                'phase_a_embb_power_effective_nonzero_ratio': phase_a_embb_power_effective_nonzero_ratio,
                'phase_a_embb_power_raw_saturation_ratio': phase_a_embb_power_raw_saturation_ratio,
                'phase_a_embb_power_final_std': phase_a_embb_power_final_std,
                'phase_a_embb_power_cellwise_diversity': phase_a_embb_power_cellwise_diversity,
                # Phase-A negative-only repair (new; independent of legacy cap/floor diagnostics).
                'phase_a_power_raw_positive_ratio': float(phase_a_power_raw_positive_ratio),
                'phaseA_positive_ratio': float(phase_a_power_positive_ratio),
                'phase_a_power_positive_clamped_to_zero_ratio': float(phase_a_power_positive_clamped_to_zero_ratio),
                'phaseA_zero_action_ratio': float(phase_a_zero_action_ratio),
                'phaseA_delta_lt_neg09_ratio': float(phaseA_delta_lt_neg09_ratio),
                'phaseA_delta_mean': float(phaseA_delta_mean),
                'phaseA_delta_p10': float(phaseA_delta_p10),
                'phaseA_delta_p50': float(phaseA_delta_p50),
                'phaseA_delta_p90': float(phaseA_delta_p90),
                'phaseA_negative_delta_l2_mean': float(phase_a_negative_delta_l2_mean),
                'phaseA_negative_delta_saturation_ratio': float(phase_a_negative_delta_saturation_ratio),
                'phaseA_power_reduction_l2_penalty': float(phaseA_power_reduction_l2_penalty),
                'phaseA_power_saturation_penalty': float(phaseA_power_saturation_penalty),
                'embb_service_floor_hinge_penalty': float(embb_service_floor_hinge_penalty),
                'phase_a_power_negative_candidate_ratio': float(phase_a_power_negative_candidate_ratio),
                'phase_a_power_negative_executed_ratio': float(phase_a_power_negative_executed_ratio),
                'phaseA_executed_abs_delta_mean': float(phase_a_embb_power_mean_abs_executed_delta),
                'phase_a_power_mean_negative_delta': float(phase_a_power_mean_negative_delta),
                'phase_a_power_service_guard_reject_ratio': float(phase_a_power_service_guard_reject_ratio),
                'phase_a_power_minrate_guard_reject_ratio': float(phase_a_power_minrate_guard_reject_ratio),
                'phase_a_power_reliability_guard_reject_ratio': float(phase_a_power_reliability_guard_reject_ratio),
                'phase_a_power_total_power_reduction_mean': float(phase_a_power_total_power_reduction_mean),
                'phase_a_power_intercell_reduction_mean': float(phase_a_power_intercell_reduction_mean),
                'phase_a_embb_power_floor_binding_strength': phase_a_embb_power_floor_binding_strength,
                'phase_a_embb_power_cap_binding_strength': phase_a_embb_power_cap_binding_strength,
                'phase_a_embb_power_proj_delta_l1': phase_a_embb_power_proj_delta_l1,
                'phase_a_embb_power_proj_delta_l2': phase_a_embb_power_proj_delta_l2,
                'phase_a_embb_power_pre_to_floor_delta': phase_a_embb_power_pre_to_floor_delta,
                'phase_a_embb_power_pre_to_cap_delta': phase_a_embb_power_pre_to_cap_delta,
                'phase_a_embb_power_final_minus_proj_delta': phase_a_embb_power_final_minus_proj_delta,
                'phase_a_embb_power_mean_abs_raw_delta': phase_a_embb_power_mean_abs_raw_delta,
                'phase_a_embb_power_mean_abs_executed_delta': phase_a_embb_power_mean_abs_executed_delta,
                'phase_a_embb_power_mean_raw_delta_l2': phase_a_embb_power_mean_raw_delta_l2,
                'phase_a_embb_power_invalid_or_masked_ratio': phase_a_embb_power_invalid_or_masked_ratio,
                'phase_a_embb_power_zeroed_inactive_head_count': float(self.phase_a_embb_power_zeroed_inactive_head_count),
                'phase_a_embb_power_zeroed_keep_mode_count': float(self.phase_a_embb_power_zeroed_keep_mode_count),
                'phase_a_embb_power_zeroed_no_candidate_count': float(self.phase_a_embb_power_zeroed_no_candidate_count),
                'phase_a_embb_power_zeroed_no_embb_active_count': float(self.phase_a_embb_power_zeroed_no_embb_active_count),
                'phase_a_embb_power_zeroed_no_owner_count': float(self.phase_a_embb_power_zeroed_no_owner_count),
                'phase_a_embb_power_zeroed_invalid_owner_count': float(self.phase_a_embb_power_zeroed_invalid_owner_count),
                'phase_a_embb_power_zeroed_cap_projection_count': float(self.phase_a_embb_power_zeroed_cap_projection_count),
                'phase_a_embb_power_zeroed_floor_projection_count': float(self.phase_a_embb_power_zeroed_floor_projection_count),
                'phase_a_embb_power_zeroed_unknown_count': float(self.phase_a_embb_power_zeroed_unknown_count),
                'phase_a_embb_power_zeroed_inactive_head_ratio': phase_a_embb_power_zeroed_inactive_head_ratio,
                'phase_a_embb_power_zeroed_keep_mode_ratio': phase_a_embb_power_zeroed_keep_mode_ratio,
                'phase_a_embb_power_zeroed_no_candidate_ratio': phase_a_embb_power_zeroed_no_candidate_ratio,
                'phase_a_embb_power_zeroed_no_embb_active_ratio': phase_a_embb_power_zeroed_no_embb_active_ratio,
                'phase_a_embb_power_zeroed_no_owner_ratio': phase_a_embb_power_zeroed_no_owner_ratio,
                'phase_a_embb_power_zeroed_invalid_owner_ratio': phase_a_embb_power_zeroed_invalid_owner_ratio,
                'phase_a_embb_power_zeroed_cap_projection_ratio': phase_a_embb_power_zeroed_cap_projection_ratio,
                'phase_a_embb_power_zeroed_floor_projection_ratio': phase_a_embb_power_zeroed_floor_projection_ratio,
                'phase_a_embb_power_zeroed_unknown_ratio': phase_a_embb_power_zeroed_unknown_ratio,
                'phase_a_embb_power_owner_invalid_ratio': phase_a_embb_power_zeroed_invalid_owner_ratio,
                'phase_a_embb_power_no_candidate_ratio': phase_a_embb_power_zeroed_no_candidate_ratio,
                'phase_a_embb_power_keep_mode_zero_ratio': phase_a_embb_power_zeroed_keep_mode_ratio,
                'phase_a_embb_power_no_owner_zero_ratio': phase_a_embb_power_zeroed_no_owner_ratio,
                'phase_a_embb_power_no_embb_active_zero_ratio': phase_a_embb_power_zeroed_no_embb_active_ratio,
                'phase_a_embb_power_cap_projection_zero_ratio': phase_a_embb_power_zeroed_cap_projection_ratio,
                'phase_a_embb_power_floor_projection_zero_ratio': phase_a_embb_power_zeroed_floor_projection_ratio,
                'phase_a_embb_power_unknown_zero_ratio': phase_a_embb_power_zeroed_unknown_ratio,
                # Diagnostic aliases (requested naming) for fast log scanning.
                'phaseA_pow_ratio_preclip_mean': phase_a_embb_power_pre_clip_mean_delta,
                'phaseA_pow_ratio_postclip_mean': phase_a_embb_power_post_clip_mean_delta,
                'phaseA_pow_ratio_postproj_mean': phase_a_embb_power_post_projection_mean_delta,
                'phaseA_pow_ratio_final_mean': phase_a_embb_power_final_executed_mean_delta,
                'phaseA_pow_cap_hit_ratio': phase_a_embb_power_cap_hit_ratio,
                'phaseA_pow_floor_hit_ratio': phase_a_embb_power_floor_hit_ratio,
                'phaseA_pow_projection_l2_mean': phase_a_embb_power_projection_l2_mean,
                'phaseA_pow_pre_vs_final_l1_mean': phase_a_embb_power_pre_vs_final_l1_mean,
                'phaseA_pow_pre_vs_final_sign_consistency': phase_a_embb_power_pre_vs_final_sign_consistency,
                'phaseA_pow_effective_nonzero_ratio': phase_a_embb_power_effective_nonzero_ratio,
                'phaseA_pow_floor_binding_strength': phase_a_embb_power_floor_binding_strength,
                'phaseA_pow_cap_binding_strength': phase_a_embb_power_cap_binding_strength,
                'phaseA_pow_proj_delta_l1': phase_a_embb_power_proj_delta_l1,
                'phaseA_pow_proj_delta_l2': phase_a_embb_power_proj_delta_l2,
                'phaseA_pow_pre_to_floor_delta': phase_a_embb_power_pre_to_floor_delta,
                'phaseA_pow_pre_to_cap_delta': phase_a_embb_power_pre_to_cap_delta,
                'phaseA_pow_final_minus_proj_delta': phase_a_embb_power_final_minus_proj_delta,
                'power_delta_clipped_ratio': power_delta_clipped_ratio,
                'power_quantized_ratio': power_quantized_ratio,
                'power_cap_hit_ratio': power_cap_hit_ratio,
                'power_floor_hit_ratio': power_floor_hit_ratio,
                'mean_raw_power_delta': mean_raw_power_delta,
                'mean_executed_power_delta': mean_executed_power_delta,
                'build_observations_calls': float(self.build_observations_calls),
                'build_observations_total_sec': float(self.build_observations_total_sec),
                'build_observations_current_step_calls': float(self.build_observations_current_step_calls),
                'build_observations_current_step_total_sec': float(self.build_observations_current_step_total_sec),
                'build_observations_next_step_calls': float(self.build_observations_next_step_calls),
                'build_observations_next_step_total_sec': float(self.build_observations_next_step_total_sec),
                'step_calls': float(self.step_calls),
                'step_total_sec': float(self.step_total_sec),
            }
            reward_term_keys = [
                'safe_admission_bonus',
                'unsafe_admission_penalty',
                'negative_gap_admission_penalty',
                'local_embb_opportunity_cost',
                'admission_band_bonus',
                'admission_band_penalty',
                'overlay_gain',
                'overlay_margin',
                'safe_puncture_preference_penalty',
                'safe_puncture_bonus',
                'overlay_balance_bonus',
                'puncture_when_good_overlay_available_penalty',
                'overlay_when_safe_puncture_penalty',
                'overlay_when_lower_intercell_puncture_available_penalty',
                'missed_feasible_puncture_penalty',
                'overlay_retention_gate_bonus',
                'missed_overlay_penalty',
                'terminal_admission_floor_soft_penalty',
                'terminal_embb_service_floor_penalty',
                'terminal_embb_min_rate_floor_penalty',
                'terminal_embb_service_bonus',
                'terminal_embb_min_rate_bonus',
                'terminal_avg_served_embb_rate_bonus',
                'terminal_embb_service_gain_vs_greedy_bonus',
                'terminal_embb_minrate_gain_vs_greedy_bonus',
                'terminal_embb_service_vs_greedy_shortfall_penalty',
                'urllc_admission_over_service_tradeoff_penalty',
                'terminal_owner_effective_service_gain_bonus',
                'terminal_owner_negative_service_gain_penalty',
                'terminal_owner_positive_service_gain_bonus',
                'terminal_owner_negative_rate_gain_penalty',
                'terminal_owner_positive_rate_gain_bonus',
                'terminal_owner_changed_but_no_service_penalty',
                'terminal_owner_same_as_snapshot_small_penalty',
                'terminal_phase_a_raw_saturation_penalty_v2',
                'terminal_phase_a_cap_hit_penalty_v2',
                'terminal_phase_a_smooth_delta_penalty',
                'terminal_phase_a_write_ratio_penalty',
                'terminal_phase_a_diversity_bonus',
                'terminal_phase_a_delta_l2_penalty',
                'terminal_phase_a_cellwise_flattening_penalty',
                'terminal_intercell_rate_loss_ratio_penalty',
                'terminal_intercell_penalty',
                'terminal_puncture_intercell_penalty',
                'terminal_overlay_intercell_penalty',
                'terminal_embb_served_user_count_bonus',
                'terminal_embb_served_user_deficit_penalty',
                'terminal_intercell_power_penalty',
                'terminal_total_power_over_greedy_penalty',
                'terminal_embb_power_over_greedy_penalty',
                'terminal_unscheduled_packets',
                'terminal_zero_admission_active_penalty',
                'terminal_frontier_mode_bonus',
                'terminal_frontier_mode_penalty',
                'terminal_admission_collapse_penalty',
            ]
            summary.update(
                {
                    key: float(self.episode_reward_term_totals.get(key, 0.0))
                    for key in reward_term_keys
                }
            )
            return summary

    def _compute_admitted_urllc_reliability(self) -> float:
            if self.num_packets <= 0:
                return 1.0

            if self.scheduled_reliabilities.size == self.num_packets:
                mask = np.isfinite(self.scheduled_reliabilities) & (self.scheduled_uavs >= 0)
                if np.any(mask):
                    return float(np.mean(self.scheduled_reliabilities[mask]))

            reliabilities = []
            channel_uses = self.sys_cfg.channel_uses_per_minislot
            decisions = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
                'EMPTY',
                dtype='<U5',
            )
            decisions[self.mode_grid == MODE_OVERLAY] = 'NOMA'
            decisions[self.mode_grid == MODE_PUNCTURE] = 'PUNCT'

            for packet_id in range(self.num_packets):
                uav_idx = int(self.scheduled_uavs[packet_id]) if packet_id < self.scheduled_uavs.size else -1
                if uav_idx < 0:
                    continue

                positions = np.argwhere(self.packet_grid[uav_idx] == packet_id)
                if positions.size == 0:
                    continue
                rb_idx, minislot = positions[0]
                rb_idx = int(rb_idx)
                minislot = int(minislot)

                source_user = int(self.packet_sources[packet_id])
                packet_bits = self._packet_bits_for_user(source_user)
                urllc_gain = float(self.channel_gains_mag_sq[source_user, uav_idx, rb_idx])
                actual_power = float(self.scheduled_power[packet_id, uav_idx])
                mode = int(self.mode_grid[uav_idx, rb_idx, minislot])

                local_interference = 0.0
                embb_owner = self._actual_embb_owner_for_cell(uav_idx, rb_idx, minislot)
                if mode == MODE_OVERLAY and embb_owner >= 0:
                    embb_user_idx = self.sys_cfg.num_urllc_users + embb_owner
                    embb_per_rb_power = self.allocator._get_embb_per_rb_power(embb_owner)
                    embb_per_rb_power *= float(self.embb_power_scale_grid[uav_idx, rb_idx, minislot])
                    embb_gain = float(self.channel_gains_mag_sq[embb_user_idx, uav_idx, rb_idx])
                    local_interference = embb_per_rb_power * embb_gain

                intercell = 0.0
                for other_uav in range(self.sys_cfg.num_uavs):
                    if other_uav == uav_idx:
                        continue
                    other_owner = self._actual_embb_owner_for_cell(other_uav, rb_idx, minislot)
                    other_mode = int(self.mode_grid[other_uav, rb_idx, minislot])
                    if other_owner >= 0 and other_mode != MODE_PUNCTURE:
                        other_embb_idx = self.sys_cfg.num_urllc_users + other_owner
                        other_embb_power = self.allocator._get_embb_per_rb_power(other_owner)
                        other_embb_power *= float(self.embb_power_scale_grid[other_uav, rb_idx, minislot])
                        embb_cross_gain = float(self.channel_gains_mag_sq[other_embb_idx, uav_idx, rb_idx])
                        intercell += other_embb_power * embb_cross_gain

                    other_packet = int(self.packet_grid[other_uav, rb_idx, minislot])
                    if (
                        other_packet >= 0 and
                        other_packet < self.scheduled_power.shape[0] and
                        int(self.scheduled_uavs[other_packet]) == other_uav
                    ):
                        other_urllc_user = int(self.packet_sources[other_packet])
                        urllc_cross_gain = float(self.channel_gains_mag_sq[other_urllc_user, uav_idx, rb_idx])
                        intercell += float(self.scheduled_power[other_packet, other_uav]) * urllc_cross_gain
                snir = actual_power * urllc_gain / max(
                    self.sys_cfg.noise_power + local_interference + intercell,
                    1e-15,
                )
                error_prob = self.capacity_model.decoding_error_probability(
                    snir,
                    packet_bits,
                    channel_uses,
                )
                reliabilities.append(float(1.0 - error_prob))

            if not reliabilities:
                return float("nan") if self.num_packets > 0 else 1.0
            return float(np.mean(reliabilities))

    @staticmethod
    def _compute_jain_fairness(rates: np.ndarray) -> float:
            rates = np.asarray(rates, dtype=float)
            if rates.size == 0:
                return 0.0
            denom = rates.size * np.sum(rates ** 2)
            if denom <= 0:
                return 0.0
            return float((np.sum(rates) ** 2) / denom)

    def _available_packet_ids(self, minislot: Optional[int] = None) -> List[int]:
            if minislot is None:
                minislot, _rb = self._current_cell()
            if self.packet_release_minislots.size == 0:
                return []
            carryover_enabled = bool(getattr(self.rl_cfg.env, "allow_packet_carryover_across_minislots", False))
            if carryover_enabled:
                return [
                    packet_id
                    for packet_id in sorted(self.unscheduled_packet_ids)
                    if int(self.packet_release_minislots[packet_id]) <= int(minislot)
                ]
            return [
                packet_id
                for packet_id in sorted(self.unscheduled_packet_ids)
                if self.packet_release_minislots[packet_id] == minislot
            ]

    def _drop_unscheduled_packets_of_minislot(self, minislot: int) -> None:
            if bool(getattr(self.rl_cfg.env, "allow_packet_carryover_across_minislots", False)):
                if self.packet_release_minislots.size == 0 or not self.unscheduled_packet_ids:
                    return
                # Finalize carryover prune once per minislot (after the last RB of this minislot).
                if not bool(getattr(self.rl_cfg.env, "multi_rb_agents", False)):
                    _ms, _rb = self._cell_schedule[self.current_cell_index]
                    if int(_rb) < int(self.sys_cfg.num_subcarriers) - 1:
                        return
                if int(getattr(self, "_carryover_tracking_minislot", -1)) != int(minislot):
                    self._carryover_tracking_minislot = int(minislot)
                    self._carryover_seen_in_minislot = np.zeros(self.num_packets, dtype=bool)
                    self._carryover_feasible_in_minislot = np.zeros(self.num_packets, dtype=bool)

                max_infeasible = int(getattr(self.rl_cfg.env, "carryover_max_consecutive_infeasible", 0) or 0)
                ttl = int(getattr(self.rl_cfg.env, "carryover_packet_ttl_minislots", 0) or 0)
                min_age = int(getattr(self.rl_cfg.env, "carryover_prune_min_age_minislots", 0) or 0)
                stale_ids = []
                for packet_id in list(self.unscheduled_packet_ids):
                    if packet_id < 0 or packet_id >= self.packet_release_minislots.size:
                        continue
                    release = int(self.packet_release_minislots[packet_id])
                    if release > int(minislot):
                        continue
                    age = int(minislot) - release
                    if packet_id < self.packet_last_seen_minislot.size:
                        self.packet_last_seen_minislot[packet_id] = int(minislot)
                    feasible_now = bool(
                        packet_id < self._carryover_feasible_in_minislot.size
                        and self._carryover_feasible_in_minislot[packet_id]
                    )
                    if feasible_now:
                        if packet_id < self.packet_infeasible_streak.size:
                            self.packet_infeasible_streak[packet_id] = 0
                        if packet_id < self.packet_last_feasible_minislot.size:
                            self.packet_last_feasible_minislot[packet_id] = int(minislot)
                    else:
                        if packet_id < self.packet_infeasible_streak.size:
                            self.packet_infeasible_streak[packet_id] += 1

                    drop_by_ttl = bool(ttl > 0 and age >= ttl)
                    drop_by_infeasible = bool(
                        max_infeasible > 0
                        and age >= min_age
                        and packet_id < self.packet_infeasible_streak.size
                        and int(self.packet_infeasible_streak[packet_id]) >= max_infeasible
                    )
                    if drop_by_ttl or drop_by_infeasible:
                        stale_ids.append(packet_id)
                for packet_id in stale_ids:
                    self.unscheduled_packet_ids.discard(packet_id)
                return
            if self.packet_release_minislots.size == 0 or not self.unscheduled_packet_ids:
                return
            stale_ids = [
                packet_id
                for packet_id in self.unscheduled_packet_ids
                if int(self.packet_release_minislots[packet_id]) == int(minislot)
            ]
            if not stale_ids:
                return
            for packet_id in stale_ids:
                self.unscheduled_packet_ids.discard(packet_id)

    def _current_cell(self) -> Tuple[int, int]:
            if self.current_cell_index >= len(self._cell_schedule):
                current = self._cell_schedule[-1]
            else:
                current = self._cell_schedule[self.current_cell_index]
            if self.rl_cfg.env.multi_rb_agents:
                return int(current), 0
            return current

    def _share_budget_exhausted(self) -> bool:
            if not bool(getattr(self.rl_cfg.env, "early_terminate_when_share_budget_exhausted", True)):
                return False
            budget = float(getattr(self, "greedy_urllc_budget_bps", float("inf")))
            used = float(getattr(self, "greedy_urllc_budget_used_bps", 0.0))
            if not np.isfinite(budget) or budget <= 0.0:
                return False
            return bool(used >= budget - 1.0e-12)

    def _current_planning_rb(self) -> int:
            if self.planning_index >= len(self._embb_plan_schedule):
                return self._embb_plan_schedule[-1]
            return self._embb_plan_schedule[self.planning_index]

    def _build_embb_owner_candidates(self) -> List[List[List[int]]]:
            num_uavs = self.sys_cfg.num_uavs
            num_rbs = self.sys_cfg.num_subcarriers
            num_embb = self.sys_cfg.num_embb_users
            embb_start = self.sys_cfg.num_urllc_users
            candidates_by_uav = []
            if num_embb <= 0:
                return [[[] for _ in range(num_rbs)] for _ in range(num_uavs)]

            owner_space = str(getattr(self.rl_cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
            if owner_space == "global_owner_id_no_null":
                dim = int(getattr(self.rl_cfg.action, "global_embb_owner_dim", 0) or 0)
                dim = max(dim, 1)
                # Keep the candidate semantics consistent with the legacy per-RB top-K list, but expose
                # candidates as *global owner ids* (no null bin). This avoids the degenerate behavior
                # where every UAV/RB can choose any user, which tends to collapse throughput.
                num_embb_effective = min(num_embb, dim)
                for uav_idx in range(num_uavs):
                    embb_indices = np.where(self.embb_selected_uavs == uav_idx)[0]
                    embb_indices = embb_indices[embb_indices < num_embb_effective]
                    per_rb = []
                    for rb in range(num_rbs):
                        gains = []
                        for embb_idx in embb_indices:
                            user_idx = embb_start + int(embb_idx)
                            gain = float(self.channel_gains_mag_sq[user_idx, uav_idx, rb])
                            gains.append((gain, int(embb_idx)))
                        gains.sort(key=lambda item: item[0], reverse=True)
                        top = [embb_idx for _gain, embb_idx in gains[: self.rl_cfg.action.max_embb_candidates]]
                        per_rb.append(top)
                    candidates_by_uav.append(per_rb)
                return candidates_by_uav

            for uav_idx in range(num_uavs):
                embb_indices = np.where(self.embb_selected_uavs == uav_idx)[0]
                embb_indices = embb_indices[embb_indices < num_embb]
                per_rb = []
                for rb in range(num_rbs):
                    gains = []
                    for embb_idx in embb_indices:
                        user_idx = embb_start + embb_idx
                        gain = float(self.channel_gains_mag_sq[user_idx, uav_idx, rb])
                        gains.append((gain, int(embb_idx)))
                    gains.sort(key=lambda item: item[0], reverse=True)
                    top = [embb_idx for _gain, embb_idx in gains[: self.rl_cfg.action.max_embb_candidates]]
                    per_rb.append(top)
                candidates_by_uav.append(per_rb)
            return candidates_by_uav

    def _build_fixed_embb_baseline(self, baseline_policy: str) -> Dict[str, np.ndarray]:
            policy = str(baseline_policy or "greedy").strip().lower()
            if policy == "deterministic_max_gain":
                return self._build_deterministic_embb_baseline()
            if policy == "balanced_round_robin":
                return self._build_balanced_round_robin_embb_baseline()
            return self.allocator.allocate_embb_greedy(
                self.channel_gains_mag_sq,
                associated_uavs=self.best_uav_per_user,
            )

    def _build_deterministic_embb_baseline(self) -> Dict[str, np.ndarray]:
            num_embb = self.sys_cfg.num_embb_users
            num_uavs = self.sys_cfg.num_uavs
            num_rbs = self.sys_cfg.num_subcarriers
            embb_start = self.sys_cfg.num_urllc_users

            embb_rb_alloc = np.zeros((num_embb, num_rbs), dtype=int)
            alpha_e = np.zeros((num_embb, num_uavs, num_rbs), dtype=int)
            owner_per_uav_rb = np.full((num_uavs, num_rbs), -1, dtype=int)
            best_uav_per_user = self.best_uav_per_user[embb_start:].copy()

            max_power_per_user = np.zeros(num_embb, dtype=float)
            for embb_idx in range(num_embb):
                power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
                max_power_per_user[embb_idx] = min(
                    self.allocator._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                    self.algo_cfg.power_upper_bound,
                )

            for uav_idx in range(num_uavs):
                candidate_users = np.where(best_uav_per_user == uav_idx)[0]
                if candidate_users.size == 0:
                    continue
                for rb_idx in range(num_rbs):
                    best_user = -1
                    best_gain = -np.inf
                    for embb_idx in candidate_users:
                        user_idx = embb_start + embb_idx
                        gain = float(self.channel_gains_mag_sq[user_idx, uav_idx, rb_idx])
                        if gain > best_gain:
                            best_gain = gain
                            best_user = int(embb_idx)
                    if best_user >= 0:
                        embb_rb_alloc[best_user, rb_idx] = 1
                        alpha_e[best_user, uav_idx, rb_idx] = 1
                        owner_per_uav_rb[uav_idx, rb_idx] = best_user

            embb_tx_powers = np.zeros(num_embb, dtype=float)
            for embb_idx in range(num_embb):
                quota = int(np.sum(embb_rb_alloc[embb_idx, :]))
                if quota <= 0:
                    continue
                load_fraction = quota / max(num_rbs, 1)
                embb_tx_powers[embb_idx] = max_power_per_user[embb_idx] * load_fraction

            self.allocator.embb_owner_per_uav_rb = owner_per_uav_rb.copy()
            self.allocator.embb_selected_uavs = best_uav_per_user.copy()
            self.allocator.alpha_e_allocation = alpha_e.copy()
            self.allocator.rb_allocation = embb_rb_alloc.copy()
            self.allocator.embb_user_tx_power = embb_tx_powers.copy()

            baseline = self.allocator._compute_embb_state(
                embb_rb_alloc,
                self.channel_gains_mag_sq,
                best_uav_per_user,
                embb_tx_powers,
            )

            self.allocator.embb_base_rb_rates = baseline["base_rb_rates"].copy()
            self.allocator.embb_base_rb_rates_per_uav_rb = baseline["base_rb_rates_per_uav_rb"].copy()
            self.allocator.embb_power_allocation = baseline["power_allocation"].copy()
            self.allocator.embb_owner_per_rb = baseline["owner_per_rb"].copy()

            return {
                "rb_allocation": embb_rb_alloc,
                "alpha_e": alpha_e,
                "power_allocation": baseline["power_allocation"],
                "rates": baseline["rates"],
                "total_rate": np.sum(baseline["rates"]),
                "owner_per_rb": baseline["owner_per_rb"],
                "owner_per_uav_rb": owner_per_uav_rb,
                "best_uav_per_user": best_uav_per_user,
                "base_rb_rates": baseline["base_rb_rates"],
                "base_rb_rates_per_uav_rb": baseline["base_rb_rates_per_uav_rb"],
                "user_tx_powers": embb_tx_powers,
            }

    def _build_balanced_round_robin_embb_baseline(self) -> Dict[str, np.ndarray]:
            num_embb = self.sys_cfg.num_embb_users
            num_uavs = self.sys_cfg.num_uavs
            num_rbs = self.sys_cfg.num_subcarriers
            embb_start = self.sys_cfg.num_urllc_users

            embb_rb_alloc = np.zeros((num_embb, num_rbs), dtype=int)
            alpha_e = np.zeros((num_embb, num_uavs, num_rbs), dtype=int)
            owner_per_uav_rb = np.full((num_uavs, num_rbs), -1, dtype=int)
            best_uav_per_user = np.asarray(self.best_uav_per_user[embb_start:], dtype=int).copy()

            max_power_per_user = np.zeros(num_embb, dtype=float)
            for embb_idx in range(num_embb):
                power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
                max_power_per_user[embb_idx] = min(
                    self.allocator._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                    self.algo_cfg.power_upper_bound,
                )

            for uav_idx in range(num_uavs):
                candidate_users = np.where(best_uav_per_user == uav_idx)[0].astype(int)
                if candidate_users.size == 0:
                    continue
                candidate_users = np.sort(candidate_users)
                start_offset = int(uav_idx % max(candidate_users.size, 1))
                rotated_users = np.roll(candidate_users, -start_offset)
                for rb_idx in range(num_rbs):
                    embb_idx = int(rotated_users[rb_idx % rotated_users.size])
                    embb_rb_alloc[embb_idx, rb_idx] = 1
                    alpha_e[embb_idx, uav_idx, rb_idx] = 1
                    owner_per_uav_rb[uav_idx, rb_idx] = embb_idx

            embb_tx_powers = np.zeros(num_embb, dtype=float)
            for embb_idx in range(num_embb):
                quota = int(np.sum(embb_rb_alloc[embb_idx, :]))
                if quota <= 0:
                    continue
                load_fraction = quota / max(num_rbs, 1)
                embb_tx_powers[embb_idx] = max_power_per_user[embb_idx] * load_fraction

            self.allocator.embb_owner_per_uav_rb = owner_per_uav_rb.copy()
            self.allocator.embb_selected_uavs = best_uav_per_user.copy()
            self.allocator.alpha_e_allocation = alpha_e.copy()
            self.allocator.rb_allocation = embb_rb_alloc.copy()
            self.allocator.embb_user_tx_power = embb_tx_powers.copy()

            baseline = self.allocator._compute_embb_state(
                embb_rb_alloc,
                self.channel_gains_mag_sq,
                best_uav_per_user,
                embb_tx_powers,
            )

            self.allocator.embb_base_rb_rates = baseline["base_rb_rates"].copy()
            self.allocator.embb_base_rb_rates_per_uav_rb = baseline["base_rb_rates_per_uav_rb"].copy()
            self.allocator.embb_power_allocation = baseline["power_allocation"].copy()
            self.allocator.embb_owner_per_rb = baseline["owner_per_rb"].copy()

            return {
                "rb_allocation": embb_rb_alloc,
                "alpha_e": alpha_e,
                "power_allocation": baseline["power_allocation"],
                "rates": baseline["rates"],
                "total_rate": np.sum(baseline["rates"]),
                "owner_per_rb": baseline["owner_per_rb"],
                "owner_per_uav_rb": owner_per_uav_rb,
                "best_uav_per_user": best_uav_per_user,
                "base_rb_rates": baseline["base_rb_rates"],
                "base_rb_rates_per_uav_rb": baseline["base_rb_rates_per_uav_rb"],
                "user_tx_powers": embb_tx_powers,
            }

    def _finalize_embb_baseline_from_policy(self) -> None:
            finalized_owner_map = self.owner_per_uav_rb.copy()
            if (
                bool(getattr(self.rl_cfg.env, "phase0_owner_guard_enabled", False))
                and self.phase0_snapshot_owner_per_uav_rb is not None
            ):
                finalized_owner_map = self._effective_owner_map(self.owner_per_uav_rb)
                self.owner_per_uav_rb = finalized_owner_map.copy()

            embb_projection = self._project_embb_baseline_from_owner_map(
                finalized_owner_map,
                self.embb_power_scale_per_uav_rb,
            )

            self.allocator.embb_owner_per_uav_rb = finalized_owner_map.copy()
            self.allocator.embb_selected_uavs = self.embb_selected_uavs.copy()
            self.allocator.alpha_e_allocation = embb_projection["alpha_e"]
            self.allocator.rb_allocation = embb_projection["rb_allocation"]
            self.allocator.embb_user_tx_power = embb_projection["user_tx_powers"].copy()
            self.allocator.embb_power_allocation = embb_projection['power_allocation']
            self.allocator.embb_base_rb_rates = embb_projection['base_rb_rates']
            self.allocator.embb_base_rb_rates_per_uav_rb = embb_projection['base_rb_rates_per_uav_rb']

            self.embb_result = {
                'rb_allocation': embb_projection['rb_allocation'],
                'alpha_e': embb_projection['alpha_e'],
                'power_allocation': embb_projection['power_allocation'],
                'rates': embb_projection['rates'],
                'total_rate': embb_projection['total_rate'],
                'owner_per_rb': embb_projection['owner_per_rb'],
                'owner_per_uav_rb': finalized_owner_map.copy(),
                'best_uav_per_user': self.embb_selected_uavs.copy(),
                'base_rb_rates': embb_projection['base_rb_rates'],
                'base_rb_rates_per_uav_rb': embb_projection['base_rb_rates_per_uav_rb'],
                'user_tx_powers': embb_projection['user_tx_powers'],
            }
            self.embb_base_rb_rates = embb_projection['base_rb_rates']
            self.embb_base_rb_rates_per_uav_rb = embb_projection['base_rb_rates_per_uav_rb']
            self.embb_owner_grid = np.repeat(
                finalized_owner_map[:, :, None],
                self.sys_cfg.num_minislots,
                axis=2,
            ).astype(int, copy=True)
            if self.embb_power_scale_per_uav_rb is not None:
                self.embb_power_scale_grid = np.repeat(
                    self.embb_power_scale_per_uav_rb[:, :, None],
                    self.sys_cfg.num_minislots,
                    axis=2,
                ).astype(float, copy=True)

    def _local_obs_dim(self) -> int:
            base_features = 19 + self._phase_a_progress_obs_dim()
            candidate_features = 16
            rb_summary_features = self._rb_summary_feature_dim()
            return base_features + candidate_features * self.rl_cfg.action.max_candidate_packets + rb_summary_features

    def _global_obs_dim(self) -> int:
            base_features = 6
            per_uav_features = 8
            return base_features + per_uav_features * self.sys_cfg.num_uavs
