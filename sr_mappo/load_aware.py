
"""Shared load-aware reward, gating, and selection helpers for SR-MAPPO."""

from __future__ import annotations

from typing import Dict, Iterable

REFERENCE_LOADS = (5.0, 10.0, 15.0, 20.0, 25.0)

_REWARD_SCHEDULE = {
    5.0: {"terminal_embb_rate_weight": 7.0, "terminal_urllc_admission_weight": 0.8, "terminal_urllc_admission_target": 0.55, "terminal_unscheduled_penalty": 0.8, "puncture_extra_penalty": 0.10, "missed_overlay_penalty": 0.18, "overlay_gain_weight": 0.10},
    10.0: {"terminal_embb_rate_weight": 6.0, "terminal_urllc_admission_weight": 1.2, "terminal_urllc_admission_target": 0.62, "terminal_unscheduled_penalty": 1.0, "puncture_extra_penalty": 0.12, "missed_overlay_penalty": 0.22, "overlay_gain_weight": 0.15},
    15.0: {"terminal_embb_rate_weight": 5.0, "terminal_urllc_admission_weight": 1.8, "terminal_urllc_admission_target": 0.68, "terminal_unscheduled_penalty": 1.2, "puncture_extra_penalty": 0.16, "missed_overlay_penalty": 0.28, "overlay_gain_weight": 0.20},
    20.0: {"terminal_embb_rate_weight": 4.2, "terminal_urllc_admission_weight": 2.4, "terminal_urllc_admission_target": 0.72, "terminal_unscheduled_penalty": 1.5, "puncture_extra_penalty": 0.20, "missed_overlay_penalty": 0.35, "overlay_gain_weight": 0.25},
    25.0: {"terminal_embb_rate_weight": 3.8, "terminal_urllc_admission_weight": 2.8, "terminal_urllc_admission_target": 0.74, "terminal_unscheduled_penalty": 1.7, "puncture_extra_penalty": 0.24, "missed_overlay_penalty": 0.40, "overlay_gain_weight": 0.30},
}

_SELECTION_FORMULA = {
    5.0: {"throughput": 4.0, "admission": 0.5, "puncture_loss": 0.5, "overlay_retention": 0.2, "power": 0.05, "mix": 0.10},
    10.0: {"throughput": 3.5, "admission": 0.8, "puncture_loss": 0.8, "overlay_retention": 0.3, "power": 0.05, "mix": 0.15},
    15.0: {"throughput": 2.5, "admission": 1.5, "puncture_loss": 1.5, "overlay_retention": 0.5, "power": 0.05, "mix": 0.20},
    20.0: {"throughput": 1.8, "admission": 2.5, "puncture_loss": 2.0, "overlay_retention": 0.8, "power": 0.05, "mix": 0.25},
    25.0: {"throughput": 1.5, "admission": 3.0, "puncture_loss": 2.2, "overlay_retention": 1.0, "power": 0.05, "mix": 0.30},
}

_LOW_DAMAGE_SELECTION_FORMULA = {
    5.0: {"throughput": 4.0, "admission": 0.8, "puncture_loss": 1.5, "overlay_retention": 0.5, "power": 0.3, "mix": 0.10},
    10.0: {"throughput": 3.5, "admission": 1.2, "puncture_loss": 2.0, "overlay_retention": 0.8, "power": 0.4, "mix": 0.15},
    15.0: {"throughput": 2.8, "admission": 1.8, "puncture_loss": 2.8, "overlay_retention": 1.0, "power": 0.5, "mix": 0.20},
    20.0: {"throughput": 2.0, "admission": 2.2, "puncture_loss": 3.2, "overlay_retention": 1.2, "power": 0.6, "mix": 0.25},
    25.0: {"throughput": 1.8, "admission": 2.4, "puncture_loss": 3.5, "overlay_retention": 1.3, "power": 0.7, "mix": 0.30},
}

def nearest_reference_load(actual_load: float, reference_loads: Iterable[float] | None = None) -> float:
    loads = [float(value) for value in (reference_loads or REFERENCE_LOADS)]
    actual = float(actual_load)
    return min(loads, key=lambda value: (abs(value - actual), value))

def value_for_load(actual_load: float, mapping: Dict[float, float] | None, fallback: float = 0.0) -> float:
    if not mapping:
        return float(fallback)
    normalized = {float(key): float(value) for key, value in mapping.items()}
    bucket = nearest_reference_load(actual_load, normalized.keys())
    return float(normalized.get(bucket, fallback))

def load_aware_reward_schedule(actual_load: float) -> Dict[str, float]:
    return dict(_REWARD_SCHEDULE[nearest_reference_load(actual_load)])

def puncture_loss_ceiling_for_load(actual_load: float, ceiling_by_load: Dict[float, float] | None, fallback: float = float('inf')) -> float:
    return float(value_for_load(actual_load, ceiling_by_load, fallback))

def overlay_retention_gate_for_load(actual_load: float, gate_by_load: Dict[float, float] | None, fallback: float = 0.0) -> float:
    return float(value_for_load(actual_load, gate_by_load, fallback))

def power_ratio_ceiling_for_load(actual_load: float, ceiling_by_load: Dict[float, float] | None, fallback: float = float('inf')) -> float:
    return float(value_for_load(actual_load, ceiling_by_load, fallback))

def load_aware_selection_score(actual_load: float, throughput_excess: float, admission_gap: float, puncture_loss_gap: float, overlay_retention_gap: float, power_ratio: float, *, low_damage: bool = False) -> float:
    weights = (_LOW_DAMAGE_SELECTION_FORMULA if low_damage else _SELECTION_FORMULA)[nearest_reference_load(actual_load)]
    power_ratio_excess = max(float(power_ratio) - 1.0, 0.0)
    return float(weights["throughput"] * float(throughput_excess) + weights["admission"] * float(admission_gap) - weights["puncture_loss"] * float(puncture_loss_gap) + weights["overlay_retention"] * float(overlay_retention_gap) - weights["power"] * power_ratio_excess)

def load_aware_score_mix(actual_load: float, *, low_damage: bool = False) -> float:
    weights = (_LOW_DAMAGE_SELECTION_FORMULA if low_damage else _SELECTION_FORMULA)[nearest_reference_load(actual_load)]
    return float(weights["mix"])

def selection_floor_for_load(actual_load: float, floor_by_load: Dict[float, float] | None = None, fallback_floor: float = 0.0) -> float:
    return float(value_for_load(actual_load, floor_by_load, fallback_floor))
