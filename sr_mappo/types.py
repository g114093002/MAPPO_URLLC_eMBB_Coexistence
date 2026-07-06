"""Shared dataclasses and constants for the SR-MAPPO package."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


MODE_KEEP = 0
MODE_OVERLAY = 1
MODE_PUNCTURE = 2
MODE_NAMES = {
    MODE_KEEP: "KEEP",
    MODE_OVERLAY: "NOMA",
    MODE_PUNCTURE: "PUNCT",
}


@dataclass
class HybridAction:
    """One UAV action for a single (RB, minislot) decision step."""

    mode: int = MODE_KEEP
    packet_option: int = 0
    power_delta: float = 0.0
    embb_owner_option: int = 0
    embb_power_delta: float = 0.0


@dataclass
class CandidatePacket:
    """Per-packet local coexistence diagnostics for one UAV cell."""

    packet_id: int
    source_user: int
    associated_uav: int
    channel_gain: float
    puncture_feasible: bool
    overlay_feasible: bool
    puncture_power: float
    overlay_power: float
    puncture_reliability: float
    overlay_reliability: float
    puncture_loss: float
    overlay_loss: float
    overlay_retention: float
    puncture_utility: float
    overlay_utility: float
    fixed_embb_owner: int = -1
    fixed_embb_user_idx: int = -1
    overlay_embb_owner: int = -1
    overlay_embb_user_idx: int = -1
    cause_urllc_sinr_unachievable: bool = False
    cause_embb_retention_below_threshold: bool = False
    cause_required_power_exceeds_budget: bool = False
    cause_rb_minislot_collision: bool = False
    cause_packet_already_scheduled_elsewhere: bool = False
    cause_cross_uav_interference_too_high: bool = False
    cause_deadline_or_release_violation: bool = False
    cause_other_structural_reason: bool = False
    # Decomposition of `cause_other_structural_reason` for report-side root-cause analysis.
    cause_gain_ratio_unqualified: bool = False
    cause_overlay_margin_blocked: bool = False
    cause_overlay_retention_gate_blocked: bool = False
    cause_overlay_positive_gate_blocked: bool = False
    cause_no_overlay_owner_available: bool = False
    cause_overlay_reliability_failed: bool = False
    cause_overlay_sic_failed: bool = False
    feasible_uav_count: int = 1
    overlay_uav_count: int = 0
    puncture_uav_count: int = 0
    contention_score: float = 0.0
    overlay_urllc_snir: float = 0.0
    puncture_urllc_snir: float = 0.0
    overlay_pre_sic_snir: float = 0.0
    overlay_noise_power: float = 0.0
    overlay_intercell_interference_power: float = 0.0
    overlay_local_interference_power: float = 0.0
    overlay_residual_sic_interference_power: float = 0.0
    post_sic_snir: float = 0.0
    base_embb_snir: float = 0.0
    base_embb_signal_power: float = 0.0
    base_embb_intercell_power: float = 0.0
    rb_index: int = -1

    @property
    def best_mode(self) -> int:
        if self.overlay_feasible and self.overlay_utility >= self.puncture_utility:
            return MODE_OVERLAY
        if self.puncture_feasible:
            return MODE_PUNCTURE
        return MODE_KEEP

    @property
    def best_utility(self) -> float:
        values = []
        if self.puncture_feasible:
            values.append(self.puncture_utility)
        if self.overlay_feasible:
            values.append(self.overlay_utility)
        if not values:
            return float("-inf")
        return float(max(values))

    def utility_for_mode(self, mode: int) -> float:
        if mode == MODE_OVERLAY:
            return self.overlay_utility
        if mode == MODE_PUNCTURE:
            return self.puncture_utility
        return 0.0

    def required_power_for_mode(self, mode: int) -> float:
        if mode == MODE_OVERLAY:
            return self.overlay_power
        if mode == MODE_PUNCTURE:
            return self.puncture_power
        return 0.0

    def loss_for_mode(self, mode: int) -> float:
        if mode == MODE_OVERLAY:
            return self.overlay_loss
        if mode == MODE_PUNCTURE:
            return self.puncture_loss
        return 0.0

    def reliability_for_mode(self, mode: int) -> float:
        if mode == MODE_OVERLAY:
            return self.overlay_reliability
        if mode == MODE_PUNCTURE:
            return self.puncture_reliability
        return 1.0

    def embb_owner_for_mode(self, mode: int) -> int:
        if mode == MODE_OVERLAY:
            return self.overlay_embb_owner
        if mode == MODE_PUNCTURE:
            return self.fixed_embb_owner
        return self.fixed_embb_owner

    def embb_user_idx_for_mode(self, mode: int) -> int:
        if mode == MODE_OVERLAY:
            return self.overlay_embb_user_idx
        if mode == MODE_PUNCTURE:
            return self.fixed_embb_user_idx
        return self.fixed_embb_user_idx

    def is_mode_feasible(self, mode: int) -> bool:
        if mode == MODE_OVERLAY:
            return self.overlay_feasible
        if mode == MODE_PUNCTURE:
            return self.puncture_feasible
        return True


@dataclass
class ActionMaskBundle:
    """Discrete masks for one UAV at the current decision cell."""

    mode_mask: np.ndarray
    packet_mask: np.ndarray
    embb_owner_mask: np.ndarray


@dataclass
class AgentObservation:
    """Observation package for one UAV agent."""

    local_obs: np.ndarray
    global_obs: np.ndarray
    masks: ActionMaskBundle
    candidates: List[CandidatePacket] = field(default_factory=list)
    greedy_reference: Optional[HybridAction] = None
    greedy_reference_utility: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class ShieldedAction:
    """Action after masking, shielding, and collision resolution."""

    action: HybridAction
    candidate: Optional[CandidatePacket]
    utility: float
    used_greedy_fallback: bool = False
    collision_rewritten: bool = False
    mode_corrected: bool = False
    packet_invalid_fallback: bool = False
    mask_invalid_fallback: bool = False
    joint_reliability_rewritten: bool = False
    phase_a_embb_power_info: Dict[str, float] = field(default_factory=dict)
