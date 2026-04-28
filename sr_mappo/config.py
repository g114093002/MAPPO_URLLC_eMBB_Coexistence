"""Configuration objects for Shielded Recurrent Action-Masked MAPPO."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HybridActionConfig:
    """Hybrid action space specification."""

    max_candidate_packets: int = 8
    max_embb_candidates: int = 12
    # Phase-0 owner action space:
    # - "candidate_option_with_null": legacy {0(null), 1..M(candidates)}
    # - "global_owner_id_no_null": owner id directly in [0..global_embb_owner_dim-1], masked to valid eMBB owners
    embb_owner_action_space: str = "candidate_option_with_null"
    # Only used when embb_owner_action_space == "global_owner_id_no_null".
    # Must be >= the maximum eMBB UE count the run can reach across curriculum/eval loads.
    global_embb_owner_dim: int = 32
    num_mode_actions: int = 3
    continuous_power: bool = True
    power_delta_limit: float = 0.45
    power_delta_pos_scale: float = 1.0
    power_delta_neg_scale: float = 1.0
    embb_power_delta_limit: float = 0.45
    # Phase-A eMBB power uses a small residual multiplicative update:
    # executed_scale = base_scale * (1 + alpha * tanh(raw_delta)), alpha in (0, 1).
    # Keep this small to avoid immediate cap/floor saturation and to preserve sign sensitivity.
    phase_a_embb_power_residual_alpha: float = 0.05
    bootstrap_with_discrete_power: bool = False
    initial_power_bins: int = 8
    include_null_packet_option: bool = True
    include_null_embb_option: bool = True
    min_overlay_candidate_slots: int = 1
    overlay_candidate_share: float = 0.375


@dataclass
class RewardConfig:
    """Minimal step-first reward coefficients for SR-MAPPO."""

    schedule_success_weight: float = 0.25
    embb_damage_weight: float = 0.16
    puncture_extra_penalty: float = 0.05
    overlay_gain_weight: float = 0.08
    overlay_margin_weight: float = 0.15
    missed_overlay_penalty: float = 0.12
    safe_puncture_preference_penalty_weight: float = 0.0
    overlay_margin_needed_to_override_puncture: float = 0.0
    puncture_loss_safe_threshold: float = 0.0
    safe_puncture_bonus_weight: float = 0.0
    overlay_good_candidate_bonus_weight: float = 0.0
    puncture_when_good_overlay_available_penalty_weight: float = 0.0
    overlay_when_safe_puncture_penalty_weight: float = 0.0
    # v5 alias: prefer explicit naming for intercell-aware overlay penalty when safer puncture exists.
    overlay_when_lower_intercell_puncture_available_penalty_weight: float = 0.0
    missed_feasible_puncture_penalty_weight: float = 0.0
    power_penalty_scale: float = 0.01
    # Step-level intercell-aware penalty: penalize outgoing intercell interference deltas versus eMBB-only baseline.
    # Normalized by `terminal_intercell_penalty_normalizer`.
    step_intercell_outgoing_delta_penalty_weight: float = 0.0
    # Step-level action intercell penalty: penalize selected-action victim-side intercell cost
    # (selected_action_intercell_cost_after_source_mask). Normalized by `step_action_intercell_penalty_normalizer`.
    step_action_intercell_penalty_weight: float = 0.0
    step_action_intercell_penalty_normalizer: float = 1.0e-10
    # Interference-aware admission shaping (step-level; off by default).
    low_interference_admission_bonus_weight: float = 0.0
    # If <=0, env uses a running normalization/budget (recommended).
    low_interference_admission_intercell_scale: float = 0.0
    high_intercell_admission_penalty_weight: float = 0.0
    # Running baseline for the intercell budget (within episode).
    high_intercell_admission_budget_ema_beta: float = 0.95
    keep_feasible_penalty: float = 0.10
    invalid_action_penalty: float = 0.06
    collision_rewrite_penalty: float = 0.20
    power_projection_penalty: float = 0.05
    urgency_reward_weight: float = 0.15
    keep_urgent_penalty_weight: float = 0.25
    admission_quota_pressure_weight: float = 0.15
    terminal_unscheduled_penalty: float = 8.00
    terminal_embb_rate_weight: float = 1.50
    terminal_urllc_admission_weight: float = 4.00
    terminal_urllc_admission_target: float = 0.78
    terminal_urllc_admission_penalty: float = 12.00
    terminal_embb_fairness_weight: float = 0.00
    terminal_embb_min_rate_penalty: float = 3.00
    terminal_embb_rate_normalizer: float = 5.0e6
    terminal_throughput_per_watt_weight: float = 0.0
    terminal_throughput_per_watt_normalizer: float = 1.0e6
    terminal_served_user_rate_weight: float = 0.0
    terminal_served_user_rate_normalizer: float = 1.0e6
    terminal_thin_service_penalty_weight: float = 0.0
    terminal_thin_service_threshold_bps: float = 0.0
    terminal_puncture_loss_penalty_weight: float = 0.0
    terminal_power_ratio_penalty_weight: float = 0.0
    terminal_overlay_retention_bonus: float = 0.0
    terminal_scheduled_packets_reward_weight: float = 0.0
    terminal_zero_admission_active_penalty: float = 0.0
    terminal_no_coexistence_with_feasible_penalty: float = 0.0
    selected_puncture_loss_penalty_weight: float = 0.0
    urllc_tx_power_penalty_scale: float = 0.0
    puncture_mode_usage_penalty: float = 0.0
    overlay_retention_gate_bonus_weight: float = 0.0
    overlay_retention_gate_bonus_threshold: float = 0.0
    early_puncture_collapse_penalty_weight: float = 0.0
    early_puncture_collapse_overlay_ratio_floor: float = 0.0
    early_puncture_collapse_puncture_ratio_ceiling: float = 1.0
    early_puncture_collapse_end_frac: float = 0.0
    load_adaptive_mode_target_weight: float = 0.0
    load_adaptive_puncture_floor_by_load: Dict[float, float] = field(default_factory=dict)
    load_adaptive_overlay_ceiling_by_load: Dict[float, float] = field(default_factory=dict)
    load_adaptive_start_load: float = 15.0
    puncture_admission_bonus_weight: float = 0.0
    safe_admission_bonus_weight: float = 0.0
    unsafe_admission_penalty_weight: float = 0.0
    puncture_safe_bonus_extra_weight: float = 0.0
    safe_embb_retention_threshold: float = 0.0
    unsafe_embb_retention_threshold: float = 0.0
    negative_gap_admission_penalty_weight: float = 0.0
    use_local_embb_opportunity_cost_term: bool = False
    local_embb_opportunity_cost_weight: float = 0.0
    local_embb_opportunity_cost_gate_by_quota: bool = True
    admission_band_bonus_weight: float = 0.0
    admission_band_penalty_weight: float = 0.0
    target_admission_mid_by_load: Dict[float, float] = field(default_factory=dict)
    target_admission_tol_by_load: Dict[float, float] = field(default_factory=dict)
    frontier_mode_bonus_weight: float = 0.0
    frontier_mode_penalty_weight: float = 0.0
    terminal_admission_collapse_penalty_weight: float = 0.0
    terminal_admission_collapse_floor_by_load: Dict[float, float] = field(default_factory=dict)
    planning_embb_rate_weight: float = 1.50
    planning_embb_service_weight: float = 0.0
    planning_embb_min_rate_weight: float = 0.0
    planning_embb_fairness_weight: float = 0.0
    planning_cell_edge_weight: float = 0.0
    use_greedy_terminal_reference: bool = False
    greedy_terminal_reference_mode: str = "original"

    # Separation-oriented reward terms (all off by default; enable via experiment preset).
    embb_positive_rate_bonus_weight: float = 0.0
    embb_service_ratio_bonus_weight: float = 0.0
    power_overuse_penalty_weight: float = 0.0
    owner_change_utilization_bonus_weight: float = 0.0
    owner_effective_change_bonus_weight: float = 0.0
    owner_restored_to_snapshot_penalty_weight: float = 0.0

    # Owner forced-change shaping (off by default; enable via experiment preset).
    owner_change_bonus_weight: float = 0.0
    owner_change_target_ratio: float = 0.20
    owner_change_underuse_penalty_weight: float = 0.0
    owner_same_as_snapshot_penalty_weight: float = 0.0

    # Service/throughput recovery terminal bonuses (off by default; enable via experiment preset).
    terminal_embb_service_ratio_bonus_weight: float = 0.0
    terminal_avg_served_embb_rate_bonus_weight: float = 0.0
    terminal_avg_served_embb_rate_bonus_normalizer: float = 1.0e6
    terminal_embb_min_rate_satisfaction_bonus_weight: float = 0.0

    # Soft admission floor penalty to preserve URLLC admission when optimizing service/throughput.
    terminal_admission_floor_soft_penalty_weight: float = 0.0
    terminal_admission_floor_soft_penalty_floor: float = 0.65
    # Load-aware admission floors/weights (optional; overrides scalar floor/weight when provided).
    terminal_admission_floor_soft_penalty_floor_by_load: Dict[float, float] = field(default_factory=dict)
    terminal_admission_floor_soft_penalty_weight_by_load: Dict[float, float] = field(default_factory=dict)

    # Phase-A power saturation/diversity penalties (off by default; enable via experiment preset).
    terminal_power_saturation_penalty_weight: float = 0.0
    terminal_phase_a_cap_hit_penalty_weight: float = 0.0
    terminal_phase_a_low_diversity_penalty_weight: float = 0.0
    terminal_phase_a_diversity_floor: float = 0.02
    # Optional hinge floors for terminal Phase-A penalties (0.0 => penalize from 0).
    terminal_phase_a_raw_saturation_penalty_floor: float = 0.0
    terminal_phase_a_cap_hit_penalty_floor: float = 0.0

    # Service-preserving terminal gates/bonuses (off by default; enable via experiment preset).
    terminal_embb_service_floor: float = 0.0
    terminal_embb_service_floor_penalty_weight: float = 0.0
    terminal_embb_min_rate_floor: float = 0.0
    terminal_embb_min_rate_floor_penalty_weight: float = 0.0
    terminal_embb_service_bonus_weight: float = 0.0
    terminal_embb_min_rate_bonus_weight: float = 0.0
    urllc_admission_over_service_tradeoff_penalty_weight: float = 0.0
    urllc_admission_over_service_service_floor: float = 0.0

    # Load-aware service floors (optional; used by service recovery presets).
    # Keys are representative average UE loads per UAV (e.g., 5, 10, 15, 20, 25).
    terminal_embb_service_floor_by_load: Dict[float, float] = field(default_factory=dict)
    terminal_embb_min_rate_floor_by_load: Dict[float, float] = field(default_factory=dict)

    # Served-user-count bonus (terminal; optional).
    terminal_embb_served_user_count_weight: float = 0.0
    terminal_embb_served_user_count_normalizer_by_load: Dict[float, float] = field(default_factory=dict)
    # Served-user deficit penalty (terminal; optional; uses the same load-aware targets).
    terminal_embb_served_user_deficit_penalty_weight: float = 0.0

    # Relative-to-greedy service gains (terminal; optional; uses per-episode greedy reference metrics).
    terminal_embb_service_gain_vs_greedy_weight: float = 0.0
    terminal_embb_minrate_gain_vs_greedy_weight: float = 0.0
    terminal_embb_service_vs_greedy_shortfall_penalty_weight: float = 0.0

    # Inter-cell aware terminal penalties (optional; diagnostics keys exist in env summary).
    terminal_intercell_penalty_weight: float = 0.0
    terminal_puncture_intercell_penalty_weight: float = 0.0
    terminal_overlay_intercell_penalty_weight: float = 0.0
    terminal_intercell_penalty_normalizer: float = 1.0e-7
    terminal_intercell_rate_loss_ratio_penalty_weight: float = 0.0
    # Aliases (newer naming) for clarity in experiments (wired in env).
    terminal_intercell_loss_ratio_penalty_weight: float = 0.0
    terminal_mean_intercell_mw_penalty_weight: float = 0.0
    # Extra power/interference penalties relative to the per-episode greedy reference (optional).
    terminal_intercell_power_penalty_weight: float = 0.0
    terminal_total_power_over_greedy_penalty_weight: float = 0.0
    terminal_embb_power_over_greedy_penalty_weight: float = 0.0
    terminal_power_over_greedy_allowed_ratio: float = 1.10

    # Owner effectiveness shaping (terminal; defaults off; enable via experiment preset).
    owner_effective_service_gain_bonus_weight: float = 0.0
    owner_negative_rate_gain_penalty_weight: float = 0.0
    owner_changed_but_no_service_penalty_weight: float = 0.0
    owner_same_as_snapshot_small_penalty_weight: float = 0.0
    # Owner change quality gate (terminal; prefer these in newer presets).
    owner_negative_service_gain_penalty_weight: float = 0.0
    owner_positive_service_gain_bonus_weight: float = 0.0
    owner_positive_rate_gain_bonus_weight: float = 0.0
    owner_positive_objective_gain_bonus_weight: float = 0.0
    owner_negative_objective_gain_penalty_weight: float = 0.0
    owner_harmful_change_penalty_weight: float = 0.0
    owner_dropped_raw_churn_penalty_weight: float = 0.0

    # Phase-A power regularization (terminal; defaults off; enable via experiment preset).
    phase_a_power_raw_saturation_penalty_weight: float = 0.0
    phase_a_power_cap_hit_penalty_weight: float = 0.0
    phase_a_power_smooth_delta_penalty_weight: float = 0.0
    phase_a_power_diversity_bonus_weight: float = 0.0
    phase_a_power_target_write_ratio: float = 0.0
    phase_a_power_write_ratio_penalty_weight: float = 0.0
    phase_a_power_delta_l2_penalty_weight: float = 0.0
    phase_a_power_cellwise_flattening_penalty_weight: float = 0.0
    terminal_phase_a_effective_nonzero_floor_penalty_weight: float = 0.0
    terminal_phase_a_effective_nonzero_floor: float = 0.15
    terminal_phase_a_abs_delta_floor_penalty_weight: float = 0.0
    terminal_phase_a_abs_delta_floor: float = 0.04


@dataclass
class ShieldConfig:
    """Safety and feasibility controls."""

    enable_action_masking: bool = True
    enable_feasibility_shield: bool = True
    apply_joint_reliability_rewrite: bool = True
    enable_greedy_fallback: bool = False
    force_power_to_feasible_minimum: bool = True
    resolve_packet_collisions: bool = False
    allow_mode_correction: bool = False
    force_overlay_when_better: bool = False
    force_overlay_utility_margin: float = 0.0


@dataclass
class EnvAdapterConfig:
    """How the RL environment is layered on top of the current simulator."""

    phase: str = "phase_a_coexistence_only"
    max_packets_per_step_view: int = 6
    normalize_local_observation: bool = True
    normalize_global_observation: bool = True
    early_terminate_when_all_packets_scheduled: bool = False
    keep_unscheduled_packets_as_terminal_penalty: bool = True
    use_all_uavs_as_candidate_servers: bool = False
    include_greedy_reference_in_obs: bool = False
    multi_rb_agents: bool = False
    learn_embb_baseline: bool = False
    learn_phase0_embb_power: bool = True
    force_embb_owner_per_rb: bool = True
    # Debug: freeze Phase-0 owner learning to the baseline snapshot owner map.
    freeze_phase0_owner_to_snapshot: bool = False
    # Phase-0 owner fallback behavior when the policy selects a null/invalid owner option.
    # - "candidate0": fall back to the first baseline candidate (legacy behavior when `force_embb_owner_per_rb=True`)
    # - "keep_snapshot": keep the fixed snapshot owner (no implicit greedy fallback)
    # - "keep_current": keep the current owner map entry (no implicit greedy fallback)
    # - "resample_valid": sample a valid non-null owner option from the current mask/candidates
    # - "keep_null": keep owner=-1 (do not restore snapshot)
    # - "sample_valid_non_snapshot": sample a valid owner that differs from snapshot; fallback to null
    # - "none": allow owner=-1 (may reduce feasible eMBB transmission)
    phase0_owner_fallback_policy: str = "candidate0"
    # Snapshot leakage control for Phase-0 owner pipeline.
    # These default to True for backward compatibility, but can be forced False in debug experiments
    # to ensure the MAPPO owner pipeline does not reference any fixed baseline snapshot.
    owner_snapshot_in_observation: bool = True
    owner_snapshot_used_for_init: bool = True
    owner_snapshot_used_for_fallback: bool = True
    owner_snapshot_used_for_reward: bool = True
    # Snapshot/greedy is baseline-only; never use restore-to-snapshot as a hard target.
    disable_snapshot_imitation: bool = True
    allow_phase_a_embb_power_adjustment: bool = False
    allow_phase_a_power_on_keep: bool = False
    include_frontier_progress_obs: bool = False
    include_quota_progress_obs: bool = False
    phase_a_embb_power_delta_values: List[float] = field(default_factory=list)
    disable_phase_a_embb_power_projection_for_debug: bool = False
    phase_a_embb_power_scale_bound_relax: float = 1.0
    # When <= 0 or < 1.0, defaults to phase_a_embb_power_scale_bound_relax at runtime.
    phase_a_embb_power_scale_floor_relax: float = 0.0
    phase_a_embb_power_scale_cap_relax: float = 0.0
    # Phase-A eMBB power repair: negative-only (do not allow power increases).
    phase_a_negative_only_embb_power_repair: bool = True
    phase_a_embb_power_max_downscale_per_step: float = 0.05
    phase_a_power_guard_floor_margin: float = 0.02
    phase0_owner_guard_enabled: bool = False
    phase0_owner_max_change_ratio: float = 1.0
    phase0_owner_change_warmup_enabled: bool = False
    phase0_owner_change_ratio_start: float = 0.05
    phase0_owner_change_ratio_end: float = 0.30
    phase0_owner_change_warmup_iters: int = 1000
    phase0_owner_objective_eps: float = 1.0e-9
    phase0_owner_objective_w_service: float = 1.0
    phase0_owner_objective_w_minrate: float = 0.75
    phase0_owner_objective_w_rate: float = 0.50
    phase0_owner_objective_w_intercell: float = 0.25
    phase0_owner_objective_w_power: float = 0.20
    phase0_owner_objective_w_harm: float = 0.50
    owner_objective_relax_eps: float = 0.02
    owner_service_drop_tol: float = 0.01
    owner_intercell_increase_tol: float = 0.02
    phase0_owner_change_budget_mode: str = "committed_only"  # "committed_only" or "full_snapshot_legacy"
    phase0_owner_projected_rate_floor_ratio: float = 0.0
    phase0_owner_projected_power_ceiling_ratio: float = float("inf")
    phase0_owner_rewrite_to_snapshot_on_violation: bool = True
    # Phase-0 owner service/rate preserving guards (snapshot-only internal projections; never exposed to policy).
    phase0_owner_service_preserve_guard: bool = False
    phase0_owner_rate_preserve_guard: bool = False
    # Aliases for newer experiment presets (kept for clarity; wired as OR with the preserve guards).
    phase0_owner_service_gain_guard: bool = False
    phase0_owner_allow_change_only_if_projected_service_not_worse: bool = False
    phase0_owner_allow_change_only_if_projected_minrate_not_worse: bool = False
    phase0_owner_guard_scope: str = "per_rb"  # "per_rb" or "global"
    phase0_owner_service_floor_count_ratio: float = 1.00
    phase0_owner_min_rate_floor_count_ratio: float = 1.00
    # When guard violation happens and rewrite_to_snapshot_on_violation=True, choose how to recover.
    # - "snapshot": restore to snapshot owners (legacy)
    # - "best_valid": search a best-effort valid owner map for this RB column (service/min-rate/sum-rate objective)
    phase0_owner_guard_violation_fallback: str = "snapshot"

    # Inter-cell aware power projection (optional; used by service+interference repair presets).
    intercell_aware_power_projection: bool = False
    max_total_power_ratio_to_greedy_by_load: Dict[float, float] = field(default_factory=dict)
    # Action-level intercell guard (Phase-A): mask/penalize high-intercell candidates to avoid relying only on terminal penalties.
    enable_action_intercell_guard: bool = False
    action_intercell_guard_ratio_to_running_min: float = 1.25
    action_intercell_guard_ratio_to_local_min: float = 1.25
    action_intercell_guard_keep_best_feasible: bool = True
    good_overlay_retention_threshold: float = 0.85
    good_overlay_intercell_ratio_to_local_min: float = 1.5
    # Supported snapshot builders: deterministic_max_gain, balanced_round_robin, greedy.
    fixed_embb_baseline_policy: str = "deterministic_max_gain"
    embb_power_scale_min: float = 0.80
    embb_power_scale_max: float = 1.10
    # Overlay quality gates (throughput-aware). These make overlay feasible
    # only if it preserves enough eMBB rate in the current cell.
    min_overlay_retention: float = 0.0
    min_overlay_retained_rate_ratio: float = 0.0
    allow_negative_overlay_margin: bool = True
    enforce_throughput_positive_overlay: bool = False
    soft_overlay_retention_penalty: float = 0.10
    soft_overlay_ratio_penalty: float = 0.10
    soft_overlay_margin_penalty: float = 0.10
    joint_schedule_weight: float = 1.00
    joint_primary_match_bonus: float = 0.35
    joint_overlay_bonus: float = 0.35
    joint_overlay_margin_weight: float = 0.20
    joint_puncture_penalty: float = 0.55
    joint_embb_loss_weight: float = 1.60
    joint_contention_penalty: float = 0.35
    admission_target_soft_floor: float = 0.55


@dataclass
class RecurrentNetworkConfig:
    """Network sizes for the shared actor and centralized critic."""

    # When False, the policy/value networks become memoryless feedforward MLP encoders
    # while preserving the (actor_hidden, critic_hidden) interface expected by the trainer.
    use_recurrent: bool = True
    local_encoder_dim: int = 192
    global_encoder_dim: int = 192
    recurrent_hidden_dim: int = 192
    actor_hidden_dim: int = 192
    critic_hidden_dim: int = 192
    min_power_log_std: float = -5.0
    max_power_log_std: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyperparameters and workflow settings."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.015
    value_coef: float = 0.5
    aux_best_mode_coef: float = 0.00
    aux_overlay_feasible_coef: float = 0.12
    aux_best_packet_coef: float = 0.00
    teacher_guidance_decay_start_frac: float = 0.10
    teacher_guidance_decay_end_frac: float = 0.35
    teacher_guidance_final_scale: float = 0.00
    max_grad_norm: float = 10.0
    learning_rate: float = 2e-4
    device: str = "cpu"

    total_iterations: int = 5000
    rollout_horizon: int = 256
    ppo_epochs: int = 4
    minibatch_size: int = 256
    bc_episodes: int = 0
    bc_epochs: int = 0
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    bc_teacher_policy: str = "greedy_reference"
    greedy_fallback_warmup_iterations: int = 60
    aux_target_policy: str = "best_utility"
    train_seed: int = 42

    eval_every: int = 50
    eval_episodes: int = 20
    checkpoint_every: int = 50
    enable_timing_logs: bool = False
    eval_compare_modes: List[str] = field(default_factory=lambda: ["selected", "original", "matched", "throughput_feasible", "throughput_only", "channel_only"])
    early_mode_anchor_end_frac: float = 0.0
    mode_anchor_safe_puncture_loss_threshold: float = 0.0
    mode_anchor_overlay_margin_override: float = 0.0
    light_eval_every: int = 0
    full_eval_every: int = 0
    light_eval_loads: List[float] = field(default_factory=list)
    light_eval_episodes_per_load: int = 0
    full_eval_enabled_during_training: bool = False
    # Report-side speed toggle: when True, `sr_mappo.report` generates only
    # a minimal set of debug plots (core KPIs + a small lambda sweep).
    report_fast_debug: bool = False
    report_lambda_sweep_load: float = 15.0
    report_lambda_sweep_values: List[float] = field(default_factory=lambda: [4.0, 8.0, 12.0, 16.0])
    report_lambda_sweep_episodes_per_lambda: int = 50
    experiment_line: str = "manual"
    selection_mode: str = "dual_metric"
    selection_baseline_mode: str = "matched_fixed_embb"
    selection_admission_floor: float = 0.0
    selection_admission_floor_by_load: Dict[float, float] = field(default_factory=dict)
    selection_admission_floor_ratio_to_baseline: float = 0.0
    checkpoint_eval_scope: str = "representative_load"
    checkpoint_eval_loads: List[float] = field(default_factory=list)
    checkpoint_eval_episodes_per_load: int = 1
    phase_a_embb_power_start_iteration: int = 0
    best_throughput_warmup_iterations: int = 0
    best_throughput_min_delta: float = 0.0
    selection_score_weights_by_load: Dict[float, float] = field(default_factory=dict)
    selection_throughput_ratio_floor_by_load: Dict[float, float] = field(default_factory=dict)
    selection_reliability_floor: float = 0.0
    selection_power_ratio_ceiling_by_load: Dict[float, float] = field(default_factory=dict)
    selection_puncture_ratio_ceiling: float = 1.0
    selection_puncture_ratio_floor_by_load: Dict[float, float] = field(default_factory=dict)
    selection_overlay_ratio_ceiling_by_load: Dict[float, float] = field(default_factory=dict)
    load_aware_objective: bool = False
    low_damage_admission_objective: bool = False
    greedy_baseline_mode: str = "matched_fixed_embb"

    sic_curriculum_start_db: float = -4.0
    sic_curriculum_end_db: float = 0.0
    sic_curriculum_end_frac: float = 0.50
    checkpoint_dir: str = "checkpoints"
    run_name: str = "sr_mappo_phase_a"
    save_best_only: bool = True
    keep_best_non_worse_than_greedy: bool = False
    non_worse_power_tolerance: float = 1.20
    non_worse_rate_ratio: float = 0.98
    non_worse_admission_gap: float = -0.01
    required_non_worse_fraction: float = 0.83

    use_load_curriculum: bool = True
    curriculum_stage_iterations: int = 40
    curriculum_loads: List[float] = field(default_factory=lambda: list(range(8, 19)))
    bc_loads: List[float] = field(default_factory=lambda: list(range(8, 19)))
    eval_loads: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0])
    coarse_eval_loads: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0])
    dense_eval_loads: List[float] = field(default_factory=lambda: [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0])
    eval_episodes_per_load: int = 6
    eval_replicas: int = 5
    puncture_loss_ceiling_by_load: Dict[float, float] = field(default_factory=dict)
    overlay_retention_gate_by_load: Dict[float, float] = field(default_factory=dict)
    use_teacher_distillation: bool = False
    teacher_policy: str = "channel_only_greedy"
    teacher_distill_coef_start: float = 0.0
    teacher_distill_coef_end: float = 0.0
    teacher_distill_end_frac: float = 0.0
    teacher_admission_loss_weight: float = 0.0
    teacher_mode_loss_weight: float = 0.0
    teacher_only_positive_gap: bool = True
    teacher_only_feasible_action: bool = True
    teacher_load_weights: Dict[float, float] = field(default_factory=dict)
    teacher_prefer_puncture_load_floor: float = 0.0
    teacher_prefer_puncture_weights_by_load: Dict[float, float] = field(default_factory=dict)
    primary_checkpoint_preference: str = "best_throughput"
    require_primary_checkpoint_match: bool = False
    use_greedy_reference_bc: bool = False
    greedy_bc_coef_start: float = 0.0
    greedy_bc_coef_end: float = 0.0
    greedy_bc_end_frac: float = 0.0
    greedy_bc_warmup_iters: int = 0
    greedy_bc_mode_weight: float = 0.0
    greedy_bc_packet_weight: float = 0.0
    greedy_bc_owner_weight: float = 0.0
    greedy_bc_only_when_feasible: bool = True
    greedy_bc_only_positive_gap: bool = True
    greedy_bc_load_weights: Dict[float, float] = field(default_factory=dict)
    use_frontier_mode_anchor: bool = False
    frontier_mode_anchor_weight: float = 0.0
    frontier_puncture_floor_by_load: Dict[float, float] = field(default_factory=dict)
    frontier_overlay_ceiling_by_load: Dict[float, float] = field(default_factory=dict)
    frontier_oracle_admission_floor_by_load: Dict[float, float] = field(default_factory=dict)
    use_phase_a_embb_power_anchor: bool = True
    phase_a_embb_power_anchor_weight: float = 0.50
    phase_a_embb_power_anchor_start_iteration: int = 0
    phase_a_embb_power_anchor_load_weights: Dict[float, float] = field(default_factory=dict)
    phase_a_embb_power_anchor_min_retention: float = 0.0
    phase_a_embb_power_anchor_positive_gap_only: bool = True
    # Phase-A power saturation regularization (loss-level; optional).
    # Applies only on Phase-A decisions (phase_a_mask==1).
    phase_a_embb_power_pre_tanh_l1_reg_weight: float = 0.0
    phase_a_embb_power_tanh_tail_reg_weight: float = 0.0
    phase_a_embb_power_tanh_tail_threshold: float = 0.6
    multiload_frontier_power_penalty_weight: float = 0.0
    stable_phase_start_iteration: int = 0
    stable_phase_actor_lr_scale: float = 1.0
    stable_phase_entropy_coef_final: float = 0.0
    stable_phase_clip_ratio_final: float = 0.2
    balanced_checkpoint_throughput_weight: float = 0.80
    balanced_checkpoint_admission_weight: float = 0.20
    balanced_checkpoint_power_penalty_weight: float = 0.0
    hardest_load_sampling_bias: float = 0.70
    second_hardest_load_sampling_bias: float = 0.20


@dataclass
class SRMAPPOConfig:
    """Top-level config for the SR-MAPPO package."""

    name: str = "Shielded Recurrent Action-Masked MAPPO"
    action: HybridActionConfig = field(default_factory=HybridActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    env: EnvAdapterConfig = field(default_factory=EnvAdapterConfig)
    network: RecurrentNetworkConfig = field(default_factory=RecurrentNetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def cfg_from_dict(data: Dict[str, Any] | None) -> SRMAPPOConfig:
    """Rebuild a config object from a checkpoint payload."""
    cfg = SRMAPPOConfig()
    if not data:
        return cfg
    if "name" in data:
        cfg.name = str(data["name"])
    for section in ("action", "reward", "shield", "env", "network", "training"):
        section_data = data.get(section, {})
        section_obj = getattr(cfg, section)
        if not isinstance(section_data, dict):
            continue
        for key, value in section_data.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
    # Backward compatibility: older v32 checkpoints were saved before the
    # phase-A progress-observation toggles were serialized. Those runs did
    # train with both frontier and quota progress features enabled, so we
    # restore that behavior when loading legacy checkpoints for report/eval.
    run_name = str(getattr(cfg.training, "run_name", "") or "")
    experiment_line = str(getattr(cfg.training, "experiment_line", "") or "")
    is_legacy_v32_progress_run = (
        "throughput_full_mappo_v32_frontier_progress_obs_tp_stable_v1" in experiment_line
        or run_name.startswith("sr_mappo_tp_full_mappo_v32_frontier_progress_obs_tp_stable_v1")
    )
    env_data = data.get("env", {}) if isinstance(data.get("env", {}), dict) else {}
    if is_legacy_v32_progress_run:
        if "include_frontier_progress_obs" not in env_data:
            cfg.env.include_frontier_progress_obs = True
        if "include_quota_progress_obs" not in env_data:
            cfg.env.include_quota_progress_obs = True
    return cfg

