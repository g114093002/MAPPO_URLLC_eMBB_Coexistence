"""Evaluation helpers for comparing SR-MAPPO against the original Greedy simulator baseline."""

from copy import deepcopy
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch

from . import _bootstrap  # noqa: F401
from .baseline_catalog import (
    baseline_label as _shared_baseline_label,
    baseline_metadata as _shared_baseline_metadata,
    baseline_narrative as _shared_baseline_narrative,
    normalize_baseline_mode as _shared_normalize_baseline_mode,
)
from .config import SRMAPPOConfig
from .load_aware import (
    load_aware_score_mix,
    load_aware_selection_score,
    nearest_reference_load,
    power_ratio_ceiling_for_load,
    selection_floor_for_load,
)
from .trainer import (
    _phase_a_embb_power_anchor_targets,
    configure_env_for_users_per_uav,
    disable_eval_fallback,
    phase_a_embb_power_anchor_enabled,
    restore_eval_fallback,
)
from .types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, HybridAction
from simulation import create_simulation


def _timing_enabled(cfg: SRMAPPOConfig) -> bool:
    return bool(getattr(cfg.training, "enable_timing_logs", False))


def _eval_log(cfg: SRMAPPOConfig, message: str) -> None:
    if not _timing_enabled(cfg):
        return
    timestamp = np.datetime64("now")
    print(f"[{timestamp}] [SR-MAPPO][EVAL] {message}", flush=True)


def _policy_actions(env, model, observations, actor_hidden, critic_hidden):
    local_obs = torch.from_numpy(np.stack([observations[agent_id].local_obs for agent_id in env.agent_ids]).astype(np.float32)).to(model.power_log_std.device)
    global_obs = torch.from_numpy(np.stack([observations[agent_id].global_obs for agent_id in env.agent_ids]).astype(np.float32)).to(model.power_log_std.device)
    mode_mask = torch.from_numpy(np.stack([observations[agent_id].masks.mode_mask for agent_id in env.agent_ids]).astype(np.float32)).to(model.power_log_std.device)
    packet_mask = torch.from_numpy(np.stack([observations[agent_id].masks.packet_mask for agent_id in env.agent_ids]).astype(np.float32)).to(model.power_log_std.device)
    embb_owner_mask = torch.from_numpy(np.stack([observations[agent_id].masks.embb_owner_mask for agent_id in env.agent_ids]).astype(np.float32)).to(model.power_log_std.device)
    output = model.act(
        local_obs=local_obs,
        global_obs=global_obs,
        mode_mask=mode_mask,
        packet_mask=packet_mask,
        embb_owner_mask=embb_owner_mask,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        deterministic=False,
    )
    joint_actions = {}
    for idx, agent_id in enumerate(env.agent_ids):
        joint_actions[agent_id] = HybridAction(
            mode=int(output.mode[idx].item()),
            packet_option=int(output.packet_option[idx].item()),
            power_delta=float(output.power_delta[idx].item()),
            embb_owner_option=int(output.embb_owner_option[idx].item()),
            embb_power_delta=float(output.embb_power_delta[idx].item()),
        )
    planning_phase = all(
        bool(observations[agent_id].metadata.get("planning_phase", 0.0))
        for agent_id in env.agent_ids
    )
    if (not planning_phase) and (not bool(getattr(env.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))):
        # Keep raw actions consistent with the execution path when Phase-A power is disabled.
        for agent_id in env.agent_ids:
            joint_actions[agent_id].embb_power_delta = 0.0
    return joint_actions, output.actor_hidden.detach(), output.critic_hidden.detach()


def _greedy_actions(env, observations):
    actions = {}
    for agent_id in env.agent_ids:
        ref = observations[agent_id].greedy_reference
        actions[agent_id] = ref if ref is not None else HybridAction()
    return actions


def _channel_only_actions(env, observations):
    """Deliberately weak, conservative, puncture-biased channel heuristic baseline.

    This baseline is intentionally weaker than the greedy reference. It only looks at
    the current observation candidates, narrows the choice to the top-2 by channel
    gain, uses a 70/30 stochastic tie-break, and only allows overlay when retention
    is very strong and clearly better than puncture. The goal is to provide a weaker
    channel-only control line rather than a strong handcrafted baseline.
    """
    actions = {}
    for agent_id, obs in observations.items():
        feasible = [
            (idx, candidate)
            for idx, candidate in enumerate(obs.candidates, start=1)
            if bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible)
        ]
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue
        feasible.sort(key=lambda item: float(item[1].channel_gain), reverse=True)
        shortlist = feasible[:2]
        if len(shortlist) == 1:
            option_idx, best = shortlist[0]
        else:
            base_seed = int(getattr(env, "current_reset_seed", 0))
            cell_index = int(getattr(env, "current_cell_index", 0))
            agent_hash = sum(ord(ch) for ch in str(agent_id))
            rng = np.random.default_rng(
                (base_seed * 73856093 + cell_index * 19349663 + agent_hash * 83492791) & 0xFFFFFFFF
            )
            option_idx, best = shortlist[0] if float(rng.random()) < 0.70 else shortlist[1]
        overlay_margin = float(best.overlay_utility) - float(best.puncture_utility)
        puncture_scale = 0.10 * max(abs(float(best.puncture_utility)), 1.0)
        allow_overlay = bool(
            best.overlay_feasible
            and float(best.overlay_retention) >= 0.93
            and overlay_margin >= puncture_scale
        )
        if allow_overlay:
            mode = MODE_OVERLAY
        elif bool(best.puncture_feasible):
            mode = MODE_PUNCTURE
        elif bool(best.overlay_feasible):
            mode = MODE_OVERLAY
        else:
            mode = MODE_KEEP
        actions[agent_id] = HybridAction(
            mode=mode,
            packet_option=int(option_idx),
            power_delta=0.0,
        )
    return actions


def _throughput_only_actions(env, observations):
    """One-dimensional greedy baseline: maximize immediate aggregate eMBB throughput.

    Every legal action is compared on the same scalar objective:
    resulting aggregate eMBB throughput after the current decision.
    That makes this equivalent to minimizing immediate aggregate eMBB
    throughput loss. URLLC admission is not rewarded here; it only
    survives as a feasibility consequence of the chosen action.
    """

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.throughput_only_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _myopic_throughput_actions(env, observations):
    """Main coexistence greedy baseline: myopic throughput-first greedy with weak tie-breaks."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.myopic_throughput_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _throughput_feasible_actions(env, observations):
    """Coexistence oracle: best eMBB throughput among feasible admit actions only."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.throughput_feasible_oracle_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _hard_feasible_throughput_actions(env, observations):
    """Hard-feasible throughput greedy: admit-only; KEEP iff none feasible."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.hard_feasible_throughput_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics

def _throughput_biased_actions(env, observations):
    """Throughput-first greedy with admission band constraint."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.throughput_biased_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _normalize_baseline_mode(mode: str | None) -> str:
    return _shared_normalize_baseline_mode(mode, default="original")


def _greedy_baseline_mode(cfg: SRMAPPOConfig) -> str:
    return _normalize_baseline_mode(getattr(cfg.training, "greedy_baseline_mode", "original"))


def _baseline_label(mode: str | None) -> str:
    return _shared_baseline_label(mode)


def _baseline_metadata(mode: str | None) -> Dict[str, object]:
    return _shared_baseline_metadata(mode)


def _baseline_narrative(
    mode: str | None,
    *,
    greedy_requires_feasible_admission_only: bool = False,
) -> Dict[str, str]:
    return _shared_baseline_narrative(
        mode,
        greedy_requires_feasible_admission_only=greedy_requires_feasible_admission_only,
    )


def _selection_baseline_mode(cfg: SRMAPPOConfig) -> str:
    baseline = str(
        getattr(
            cfg.training,
            "selection_baseline_mode",
            getattr(cfg.training, "greedy_baseline_mode", "original"),
        ) or "original"
    ).strip().lower()
    probe_cfg = deepcopy(cfg)
    probe_cfg.training.greedy_baseline_mode = baseline
    return _greedy_baseline_mode(probe_cfg)


def _normalized_eval_compare_modes(cfg: SRMAPPOConfig, force_full_compare: bool = False) -> List[str]:
    if force_full_compare:
        return ["selected", "original", "matched", "hard_feasible_throughput", "throughput_feasible", "throughput_biased", "throughput_only", "channel_only"]
    raw_modes = list(
        getattr(cfg.training, "eval_compare_modes", [])
        or ["selected", "original", "matched", "hard_feasible_throughput", "throughput_feasible", "throughput_biased", "throughput_only", "channel_only"]
    )
    aliases = {
        "selected_only": "selected",
        "selected": "selected",
        "original": "original",
        "matched": "matched",
        "hard_feasible": "hard_feasible_throughput",
        "hard_feasible_throughput": "hard_feasible_throughput",
        "hard_feasible_throughput_greedy": "hard_feasible_throughput",
        "throughput_feasible": "throughput_feasible",
        "throughput_feasible_oracle": "throughput_feasible",
        "throughput_biased": "throughput_biased",
        "throughput_biased_greedy": "throughput_biased",
        "throughput_only": "throughput_only",
        "throughput_only_greedy": "throughput_only",
        "channel_only": "channel_only",
    }
    modes: List[str] = []
    for raw_mode in raw_modes:
        normalized = aliases.get(str(raw_mode or "").strip().lower())
        if normalized and normalized not in modes:
            modes.append(normalized)
    if "selected" not in modes:
        modes.insert(0, "selected")
    return modes or ["selected"]


def _empty_compare_summary(baseline_mode: str) -> Dict[str, float]:
    normalized = _normalize_baseline_mode(baseline_mode)
    summary = {
        "policy_mean_reward": 0.0,
        "policy_mean_scheduled_packets": 0.0,
        "policy_mean_scheduled_ratio": 0.0,
        "policy_mean_reliability": 0.0,
        "policy_mean_power": 0.0,
        "policy_mean_overlay": 0.0,
        "policy_mean_puncture": 0.0,
        "policy_mean_embb_rate": 0.0,
        "greedy_mean_reward": 0.0,
        "greedy_mean_scheduled_packets": 0.0,
        "greedy_mean_scheduled_ratio": 0.0,
        "greedy_mean_reliability": 0.0,
        "greedy_mean_power": 0.0,
        "greedy_mean_overlay": 0.0,
        "greedy_mean_puncture": 0.0,
        "greedy_mean_embb_rate": 0.0,
        "greedy_noop_selected_ratio": 0.0,
        "greedy_admit_selected_ratio": 0.0,
        "greedy_overlay_ratio": 0.0,
        "greedy_puncture_ratio": 0.0,
        "greedy_avg_embb_retention": 0.0,
        "greedy_avg_embb_loss": 0.0,
        "greedy_avg_selected_throughput": 0.0,
        "greedy_avg_rejected_urllc_when_noop_better": 0.0,
        "greedy_noop_available_ratio": 0.0,
        "greedy_noop_better_ratio": 0.0,
        "greedy_requires_feasible_admission_only": 0.0,
        "mean_rate_ratio": 0.0,
        "min_rate_ratio": 0.0,
        "mean_admission_gap": 0.0,
        "mean_power_ratio": 1.0,
        "gain_fraction": 0.0,
        "gain10_fraction": 0.0,
        "non_worse_fraction": 0.0,
        "policy_score": float("-inf"),
        "weighted_selection_score": float("-inf"),
        "loadwise_selection_score": [],
        "loadwise_admission_floor": [],
        "loadwise_floor_pass": [],
        "loadwise_floor_violation": [],
        "loadwise_puncture_loss_gap": [],
        "loadwise_overlay_retention_gap": [],
        "loadwise_power_ratio_ceiling": [],
        "loadwise_power_ceiling_pass": [],
        "loadwise_power_ceiling_violation": [],
        "weighted_floor_violation": 0.0,
        "weighted_power_ceiling_violation": 0.0,
        "all_loads_pass_admission_floor": 1.0,
        "all_loads_pass_power_ceiling": 1.0,
        "all_loads_pass_selection_constraints": 1.0,
        "greedy_score": 0.0,
        "score_margin": float("-inf"),
        "non_worse_than_greedy": 0.0,
        "per_load": [],
        "eval_loads": [],
        "greedy_baseline": normalized,
        "comparison_baseline_key": normalized,
        "comparison_baseline_label": _baseline_label(normalized),
        "selection_admission_floor_ratio_to_baseline": 0.0,
        "selection_basis": "skipped_compare_baseline",
    }
    summary.update(_baseline_metadata(normalized))
    summary.update(_baseline_narrative(normalized))
    return summary


def _selection_floor_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return float(
        selection_floor_for_load(
            load,
            getattr(cfg.training, "selection_admission_floor_by_load", {}),
            fallback_floor=float(getattr(cfg.training, "selection_admission_floor", 0.0) or 0.0),
        )
    )


def _selection_floor_ratio_to_baseline(cfg: SRMAPPOConfig) -> float:
    return float(getattr(cfg.training, "selection_admission_floor_ratio_to_baseline", 0.0) or 0.0)


def _effective_selection_floor(cfg: SRMAPPOConfig, load: float, baseline_admission: float) -> float:
    absolute_floor = _selection_floor_for_load(cfg, load)
    relative_floor = _selection_floor_ratio_to_baseline(cfg) * float(np.clip(baseline_admission, 0.0, 1.0))
    return float(max(absolute_floor, relative_floor))


def _selection_power_ceiling_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return float(
        power_ratio_ceiling_for_load(
            load,
            getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}),
            fallback=float("inf"),
        )
    )


def _selection_puncture_ratio_ceiling(cfg: SRMAPPOConfig) -> float:
    return float(getattr(cfg.training, "selection_puncture_ratio_ceiling", 1.0) or 1.0)


def _selection_throughput_ratio_floor_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return _selection_value_for_load(
        load,
        getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}),
        fallback=0.0,
    )


def _selection_reliability_floor(cfg: SRMAPPOConfig) -> float:
    return float(getattr(cfg.training, "selection_reliability_floor", 0.0) or 0.0)


def _selection_value_for_load(load: float, mapping: Dict[float, float] | None, fallback: float) -> float:
    if not mapping:
        return float(fallback)
    normalized = {float(key): float(value) for key, value in dict(mapping).items()}
    bucket = nearest_reference_load(float(load), normalized.keys())
    return float(normalized.get(bucket, fallback))


def _selection_puncture_ratio_floor_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return _selection_value_for_load(
        load,
        getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}),
        fallback=0.0,
    )


def _selection_overlay_ratio_ceiling_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return _selection_value_for_load(
        load,
        getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}),
        fallback=1.0,
    )


def _mode_ratios_from_counts(overlay_count: float, puncture_count: float) -> tuple[float, float]:
    total = float(overlay_count) + float(puncture_count)
    if total <= 1.0e-9:
        return 0.0, 0.0
    overlay_ratio = float(overlay_count / total)
    puncture_ratio = float(puncture_count / total)
    return overlay_ratio, puncture_ratio


def _selection_constraint_metrics(
    cfg: SRMAPPOConfig,
    actual_load: float,
    baseline_admission: float,
    policy_admission: float,
    policy_effective_urllc_success: float,
    rate_ratio: float,
    policy_overlay_count: float,
    policy_puncture_count: float,
    power_ratio: float,
) -> Dict[str, float]:
    admission_floor = _effective_selection_floor(cfg, actual_load, baseline_admission)
    admission_violation = float(max(admission_floor - policy_admission, 0.0))
    throughput_ratio_floor = _selection_throughput_ratio_floor_for_load(cfg, actual_load)
    throughput_ratio_floor_violation = float(max(throughput_ratio_floor - rate_ratio, 0.0))
    reliability_floor = _selection_reliability_floor(cfg)
    policy_effective_urllc_success = float(
        0.0 if not np.isfinite(policy_effective_urllc_success) else policy_effective_urllc_success
    )
    reliability_violation = float(max(reliability_floor - policy_effective_urllc_success, 0.0))
    overlay_ratio, puncture_ratio = _mode_ratios_from_counts(policy_overlay_count, policy_puncture_count)

    puncture_floor = _selection_puncture_ratio_floor_for_load(cfg, actual_load)
    puncture_floor_violation = float(max(puncture_floor - puncture_ratio, 0.0))

    overlay_ceiling = _selection_overlay_ratio_ceiling_for_load(cfg, actual_load)
    overlay_ceiling_violation = float(max(overlay_ratio - overlay_ceiling, 0.0))

    puncture_ratio_ceiling = _selection_puncture_ratio_ceiling(cfg)
    puncture_ceiling_violation = float(max(puncture_ratio - puncture_ratio_ceiling, 0.0))

    selection_side_violation = float(
        admission_violation
        + throughput_ratio_floor_violation
        + reliability_violation
        + puncture_floor_violation
        + overlay_ceiling_violation
        + puncture_ceiling_violation
    )
    selection_side_pass = bool(selection_side_violation <= 1.0e-9)

    power_ceiling = _selection_power_ceiling_for_load(cfg, actual_load)
    power_ceiling_violation = float(max(power_ratio - power_ceiling, 0.0)) if np.isfinite(power_ceiling) else 0.0
    power_ceiling_pass = bool(power_ceiling_violation <= 1.0e-9)

    summary = {
        "selection_admission_floor": float(admission_floor),
        "selection_throughput_ratio_floor": float(throughput_ratio_floor),
        "selection_reliability_floor": float(reliability_floor),
        "selection_power_ratio_ceiling": float(power_ceiling),
        "floor_pass": float(selection_side_pass),
        "floor_violation": float(selection_side_violation),
        "power_ceiling_pass": float(power_ceiling_pass),
        "power_ceiling_violation": float(power_ceiling_violation),
    }
    return summary


def _checkpoint_eval_scope(cfg: SRMAPPOConfig) -> str:
    scope = str(getattr(cfg.training, "checkpoint_eval_scope", "representative_load") or "representative_load").strip().lower()
    return scope if scope in {"representative_load", "all_loads"} else "representative_load"


def _checkpoint_eval_loads(cfg: SRMAPPOConfig) -> List[float]:
    if _checkpoint_eval_scope(cfg) == "all_loads":
        loads = list(getattr(cfg.training, "checkpoint_eval_loads", []) or getattr(cfg.training, "eval_loads", []) or [])
        return [float(load) for load in loads]
    eval_loads = list(getattr(cfg.training, "eval_loads", []) or [])
    if eval_loads:
        return [float(eval_loads[-1])]
    return []


def _checkpoint_eval_episodes_per_load(cfg: SRMAPPOConfig) -> int:
    return int(getattr(cfg.training, "checkpoint_eval_episodes_per_load", 1) or 1)


def _primary_checkpoint_preference(cfg: SRMAPPOConfig) -> str:
    preference = str(getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") or "best_throughput").strip().lower()
    return preference if preference in {"best_throughput", "best_balanced", "best_v5_balanced_intercell_admission", "best_multiload_frontier", "best_multiload_tp_power"} else "best_throughput"


def _require_primary_checkpoint_match(cfg: SRMAPPOConfig) -> bool:
    return bool(getattr(cfg.training, "require_primary_checkpoint_match", False))


def _primary_checkpoint_match_warning(cfg: SRMAPPOConfig, checkpoint_reason: str) -> str:
    if _primary_checkpoint_preference(cfg) not in {"best_balanced", "best_v5_balanced_intercell_admission", "best_multiload_frontier", "best_multiload_tp_power"}:
        return ""
    if not _require_primary_checkpoint_match(cfg):
        return ""
    if any(token in str(checkpoint_reason) for token in ("best_balanced", "best_v5_balanced_intercell_admission", "multiload_frontier", "multiload_tp_power")):
        return ""
    return "primary_checkpoint_preference requested but not found; fell back to best_throughput"


def _selection_score_weight_for_load(cfg: SRMAPPOConfig, load: float) -> float:
    return _selection_value_for_load(
        load,
        getattr(cfg.training, "selection_score_weights_by_load", {}),
        fallback=1.0,
    )


def _weighted_mean(values: List[float], weights: List[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    numer = float(sum(float(v) * float(w) for v, w in zip(values, weights)))
    denom = float(sum(float(w) for w in weights))
    if denom <= 1.0e-9:
        return float(default)
    return float(numer / denom)


def _balanced_rescreen_score(summary: Dict[str, float], cfg: SRMAPPOConfig) -> tuple[float, float, float, float]:
    compare_selected = dict(summary.get("compare_selected_baseline") or {})
    throughput_ratio = float(
        compare_selected.get(
            "mean_rate_ratio",
            summary.get(
                "policy_throughput_vs_throughput_feasible_oracle",
                summary.get(
                    "policy_throughput_vs_throughput_only_greedy",
                    summary.get("policy_throughput_vs_channel_only_greedy", 0.0),
                ),
            ),
        )
        or 0.0
    )
    per_load = list(compare_selected.get("per_load", []) or [])
    admission_ratios: List[float] = []
    for item in per_load:
        if not isinstance(item, dict):
            continue
        policy_adm = float(item.get("policy_mean_scheduled_ratio", 0.0) or 0.0)
        greedy_adm = float(item.get("greedy_mean_scheduled_ratio", 0.0) or 0.0)
        if greedy_adm > 1.0e-9:
            admission_ratios.append(float(np.clip(policy_adm / greedy_adm, 0.0, 1.0)))
        else:
            admission_ratios.append(1.0 if policy_adm <= 1.0e-9 else 1.0)
    admission_ratio = float(np.mean(admission_ratios)) if admission_ratios else float(
        np.clip(summary.get("policy_mean_scheduled_ratio", 0.0), 0.0, 1.0)
    )
    power_ratio = float(
        compare_selected.get(
            "mean_power_ratio",
            summary.get(
                "policy_power_vs_throughput_feasible_oracle",
                summary.get(
                    "policy_power_vs_throughput_only_greedy",
                    summary.get("policy_power_vs_channel_only_greedy", 1.0),
                ),
            ),
        )
        or 1.0
    )
    throughput_weight = float(getattr(cfg.training, "balanced_checkpoint_throughput_weight", 0.80) or 0.80)
    admission_weight = float(getattr(cfg.training, "balanced_checkpoint_admission_weight", 0.20) or 0.20)
    power_penalty_weight = float(getattr(cfg.training, "balanced_checkpoint_power_penalty_weight", 0.0) or 0.0)
    power_penalty = float(max(power_ratio - 1.0, 0.0))
    score = throughput_weight * throughput_ratio + admission_weight * admission_ratio - power_penalty_weight * power_penalty
    return float(score), float(throughput_ratio), float(admission_ratio), float(power_ratio)


def _match_per_load_item(per_load: List[Dict[str, float]], target_load: float) -> Dict[str, float] | None:
    if not per_load:
        return None
    target = float(target_load)
    return min(
        per_load,
        key=lambda item: (
            abs(float(item.get("target_load", item.get("actual_load", target))) - target),
            abs(float(item.get("actual_load", target)) - target),
        ),
    )


def _multiload_frontier_metrics(summary: Dict[str, float], cfg: SRMAPPOConfig) -> Dict[str, object]:
    compare_summary = (
        summary.get("compare_selected_baseline")
        or summary.get("compare_throughput_feasible_oracle")
        or summary.get("compare_throughput_only")
        or summary.get("compare_channel_only")
        or summary
    )
    per_load = list(compare_summary.get("per_load", []) or [])
    loads = _checkpoint_eval_loads(cfg)
    if not loads:
        loads = [float(item.get("actual_load", item.get("target_load", 0.0))) for item in per_load]

    score_loads: List[float] = []
    throughput_ratios: List[float] = []
    capped_admission_ratios: List[float] = []
    power_ratios: List[float] = []
    weights: List[float] = []
    loadwise_pass: List[float] = []

    for load in loads:
        item = _match_per_load_item(per_load, load)
        if item is None:
            continue
        policy_adm = float(item.get("policy_mean_scheduled_ratio", 0.0))
        greedy_adm = float(item.get("greedy_mean_scheduled_ratio", 0.0))
        if greedy_adm <= 1.0e-9:
            admission_ratio_to_greedy = 1.0
        else:
            admission_ratio_to_greedy = float(policy_adm / max(greedy_adm, 1.0e-9))
        score_loads.append(float(load))
        throughput_ratios.append(float(item.get("rate_ratio", 0.0)))
        capped_admission_ratios.append(float(min(admission_ratio_to_greedy, 1.0)))
        power_ratios.append(float(item.get("power_ratio", 1.0)))
        weights.append(_selection_score_weight_for_load(cfg, load))
        loadwise_pass.append(
            float(
                float(item.get("floor_pass", 1.0)) >= 1.0 - 1e-9
                and float(item.get("power_ceiling_pass", 1.0)) >= 1.0 - 1e-9
            )
        )

    weighted_throughput_ratio = _weighted_mean(throughput_ratios, weights, default=0.0)
    weighted_capped_admission_ratio = _weighted_mean(capped_admission_ratios, weights, default=0.0)
    weighted_power_ratio = _weighted_mean(power_ratios, weights, default=1.0)
    worst_load_throughput_ratio = float(min(throughput_ratios)) if throughput_ratios else 0.0
    power_penalty_weight = float(getattr(cfg.training, "multiload_frontier_power_penalty_weight", 0.0) or 0.0)
    power_penalty = float(power_penalty_weight * max(weighted_power_ratio - 1.0, 0.0))
    score = float(
        0.70 * weighted_throughput_ratio
        + 0.20 * worst_load_throughput_ratio
        + 0.10 * weighted_capped_admission_ratio
        - power_penalty
    )
    all_constraints_pass = float(np.mean(loadwise_pass) >= 1.0 - 1e-9) if loadwise_pass else 0.0

    return {
        "checkpoint_eval_scope": _checkpoint_eval_scope(cfg),
        "checkpoint_eval_loads": [float(load) for load in score_loads],
        "checkpoint_eval_episodes_per_load": _checkpoint_eval_episodes_per_load(cfg),
        "primary_checkpoint_preference": _primary_checkpoint_preference(cfg),
        "multiload_frontier_score": score,
        "multiload_frontier_weighted_throughput_ratio": float(weighted_throughput_ratio),
        "multiload_frontier_weighted_capped_admission_ratio": float(weighted_capped_admission_ratio),
        "multiload_frontier_weighted_power_ratio": float(weighted_power_ratio),
        "multiload_frontier_power_penalty": float(power_penalty),
        "multiload_frontier_worst_load_throughput_ratio": float(worst_load_throughput_ratio),
        "multiload_frontier_all_loads_pass_constraints": float(all_constraints_pass),
        "multiload_frontier_loadwise_throughput_ratio": [float(value) for value in throughput_ratios],
        "multiload_frontier_loadwise_capped_admission_ratio": [float(value) for value in capped_admission_ratios],
        "multiload_frontier_loadwise_power_ratio": [float(value) for value in power_ratios],
        "multiload_frontier_loadwise_pass": [float(value) for value in loadwise_pass],
    }


def _load_frozen_greedy_payload(cfg: SRMAPPOConfig) -> Dict:
    raw_path = str(getattr(cfg.training, "frozen_greedy_metrics_path", "") or "").strip()
    if not raw_path:
        raise FileNotFoundError("greedy_baseline_mode='frozen_json' requires training.frozen_greedy_metrics_path")
    payload_path = Path(raw_path).expanduser()
    if not payload_path.exists():
        raise FileNotFoundError(f"Frozen greedy metrics not found: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _frozen_greedy_metrics_by_load(payload: Dict) -> Dict[float, Dict]:
    metrics = payload.get("greedy_metrics", {})
    loads = metrics.get("loads", []) or payload.get("loads", [])
    lookup: Dict[float, Dict] = {}
    for idx, load in enumerate(loads):
        key = float(load)
        lookup[key] = {
            metric_name: metric_values[idx]
            for metric_name, metric_values in metrics.items()
            if isinstance(metric_values, list) and len(metric_values) == len(loads)
        }
    return lookup


def _normalize_urllc_success_metrics(
    active_packets: float,
    scheduled_packets: float,
    admitted_reliability: float,
) -> tuple[float, float, float]:
    if not np.isfinite(admitted_reliability):
        admitted_reliability = float("nan") if active_packets > 0 and scheduled_packets <= 0 else (
            1.0 if active_packets <= 0 else 0.0
        )
    if active_packets <= 0:
        effective_success = 1.0
    elif scheduled_packets <= 0 or not np.isfinite(admitted_reliability):
        effective_success = 0.0
    else:
        effective_success = float(admitted_reliability * scheduled_packets / max(active_packets, 1.0))
    empty_admission_case = float(active_packets > 0 and scheduled_packets <= 0)
    return float(admitted_reliability), float(effective_success), float(empty_admission_case)


def _run_original_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    sys_cfg = deepcopy(env.sys_cfg)
    urllc_cfg = deepcopy(env.urllc_cfg)
    embb_cfg = deepcopy(env.embb_cfg)
    algo_cfg = deepcopy(env.algo_cfg)
    sim_cfg = deepcopy(env.sim_cfg)
    sim_cfg.random_seed = int(seed)

    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    result = simulation.run_single_allocation(slot_index=slot_index)
    metrics = result["metrics"]
    allocation = result["allocation"]
    active_packets = float(metrics.get("active_urllc_users", 0.0))
    scheduled_packets = float(metrics.get("scheduled_urllc_users", 0.0))
    admission = float(metrics.get("urllc_admission_rate", np.nan))
    reliability = float(metrics.get("urllc_success_rate", np.nan))
    rho_actions = allocation.get("rho_action_list", []) or []
    varpi_actions = allocation.get("varpi_action_list", []) or []
    overlay_count = float(len(rho_actions))
    puncture_count = float(len(varpi_actions))
    total_mode = max(overlay_count + puncture_count, 1.0)

    if not np.isfinite(admission):
        admission = 1.0 if active_packets <= 0 else float(scheduled_packets / max(active_packets, 1.0))
    reliability, effective_success, empty_admission_case = _normalize_urllc_success_metrics(
        active_packets,
        scheduled_packets,
        reliability,
    )

    return {
        "seed": float(seed),
        "team_reward": 0.0,
        "cell_order_length": 0.0,
        "embb_total_rate": float(metrics["embb_total_rate"]),
        "embb_user_rate_mean": float(metrics.get("embb_user_rate_mean", 0.0)),
        "urllc_admission_rate": admission,
        "urllc_success_rate": reliability,
        "admitted_urllc_reliability": reliability,
        "effective_urllc_success_over_arrivals": effective_success,
        "empty_admission_case": empty_admission_case,
        "scheduled_ratio": admission,
        "scheduled_packets": scheduled_packets,
        "active_packets": active_packets,
        "overlay_count": overlay_count,
        "puncture_count": puncture_count,
        "overlay_ratio": float(overlay_count / total_mode),
        "puncture_ratio": float(puncture_count / total_mode),
        "total_power": float(metrics["total_power"]),
    }


def _run_original_greedy_normal_v1_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    sys_cfg = deepcopy(env.sys_cfg)
    urllc_cfg = deepcopy(env.urllc_cfg)
    embb_cfg = deepcopy(env.embb_cfg)
    algo_cfg = deepcopy(env.algo_cfg)
    sim_cfg = deepcopy(env.sim_cfg)
    sim_cfg.random_seed = int(seed)

    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    result = simulation.run_single_allocation_normal_v1(slot_index=slot_index)
    metrics = result["metrics"]
    allocation = result["allocation"]
    active_packets = float(metrics.get("active_urllc_users", 0.0))
    scheduled_packets = float(metrics.get("scheduled_urllc_users", 0.0))
    admission = float(metrics.get("urllc_admission_rate", np.nan))
    reliability = float(metrics.get("urllc_success_rate", np.nan))
    rho_actions = allocation.get("rho_action_list", []) or []
    varpi_actions = allocation.get("varpi_action_list", []) or []
    overlay_count = float(len(rho_actions))
    puncture_count = float(len(varpi_actions))
    total_mode = max(overlay_count + puncture_count, 1.0)

    if not np.isfinite(admission):
        admission = 1.0 if active_packets <= 0 else float(scheduled_packets / max(active_packets, 1.0))
    reliability, effective_success, empty_admission_case = _normalize_urllc_success_metrics(
        active_packets,
        scheduled_packets,
        reliability,
    )

    return {
        "seed": float(seed),
        "team_reward": 0.0,
        "cell_order_length": 0.0,
        "embb_total_rate": float(metrics["embb_total_rate"]),
        "embb_user_rate_mean": float(metrics.get("embb_user_rate_mean", 0.0)),
        "urllc_admission_rate": admission,
        "urllc_success_rate": reliability,
        "admitted_urllc_reliability": reliability,
        "effective_urllc_success_over_arrivals": effective_success,
        "empty_admission_case": empty_admission_case,
        "scheduled_ratio": admission,
        "scheduled_packets": scheduled_packets,
        "active_packets": active_packets,
        "overlay_count": overlay_count,
        "puncture_count": puncture_count,
        "overlay_ratio": float(overlay_count / total_mode),
        "puncture_ratio": float(puncture_count / total_mode),
        "total_power": float(metrics["total_power"]),
    }


def _run_original_greedy_normal_v2_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    sys_cfg = deepcopy(env.sys_cfg)
    urllc_cfg = deepcopy(env.urllc_cfg)
    embb_cfg = deepcopy(env.embb_cfg)
    algo_cfg = deepcopy(env.algo_cfg)
    sim_cfg = deepcopy(env.sim_cfg)
    sim_cfg.random_seed = int(seed)

    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    result = simulation.run_single_allocation_normal_v2(slot_index=slot_index)
    metrics = result["metrics"]
    allocation = result["allocation"]
    active_packets = float(metrics.get("active_urllc_users", 0.0))
    scheduled_packets = float(metrics.get("scheduled_urllc_users", 0.0))
    admission = float(metrics.get("urllc_admission_rate", np.nan))
    reliability = float(metrics.get("urllc_success_rate", np.nan))
    rho_actions = allocation.get("rho_action_list", []) or []
    varpi_actions = allocation.get("varpi_action_list", []) or []
    overlay_count = float(len(rho_actions))
    puncture_count = float(len(varpi_actions))
    total_mode = max(overlay_count + puncture_count, 1.0)

    if not np.isfinite(admission):
        admission = 1.0 if active_packets <= 0 else float(scheduled_packets / max(active_packets, 1.0))
    reliability, effective_success, empty_admission_case = _normalize_urllc_success_metrics(
        active_packets,
        scheduled_packets,
        reliability,
    )

    return {
        "seed": float(seed),
        "team_reward": 0.0,
        "cell_order_length": 0.0,
        "embb_total_rate": float(metrics["embb_total_rate"]),
        "embb_user_rate_mean": float(metrics.get("embb_user_rate_mean", 0.0)),
        "urllc_admission_rate": admission,
        "urllc_success_rate": reliability,
        "admitted_urllc_reliability": reliability,
        "effective_urllc_success_over_arrivals": effective_success,
        "empty_admission_case": empty_admission_case,
        "scheduled_ratio": admission,
        "scheduled_packets": scheduled_packets,
        "active_packets": active_packets,
        "overlay_count": overlay_count,
        "puncture_count": puncture_count,
        "overlay_ratio": float(overlay_count / total_mode),
        "puncture_ratio": float(puncture_count / total_mode),
        "total_power": float(metrics["total_power"]),
    }


def _run_matched_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="reference")


def _run_channel_only_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="channel_only")


def _run_throughput_feasible_oracle_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="throughput_feasible")

def _run_hard_feasible_throughput_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="hard_feasible_throughput")

def _run_throughput_biased_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="throughput_biased")

def _run_throughput_only_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="throughput_only")


def _run_myopic_throughput_greedy_episode(env, seed: int, slot_index: int = 0) -> Dict[str, float]:
    del slot_index
    return rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy="myopic_throughput")


def rollout_episode(
    env,
    model=None,
    seed: int = 42,
    use_greedy: bool = False,
    greedy_policy: str = "reference",
) -> Dict[str, float]:
    episode_start = perf_counter()
    previous_greedy_obs = bool(getattr(env.rl_cfg.env, "include_greedy_reference_in_obs", False))
    previous_training_progress = float(getattr(env, "training_progress_frac", 1.0))
    previous_training_iteration = int(getattr(env, "current_training_iteration", 1) or 1)
    fallback_state = disable_eval_fallback(env)
    normalized_greedy_policy = str(greedy_policy or "reference").strip().lower()
    if model is not None:
        setattr(
            env,
            "phase_a_embb_power_enabled",
            bool(getattr(model, "phase_a_embb_power_enabled", getattr(env.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))),
        )
    env.rl_cfg.env.include_greedy_reference_in_obs = bool(use_greedy and normalized_greedy_policy == "reference")
    env.training_progress_frac = 1.0
    env.current_training_iteration = int(getattr(env, "current_training_iteration", previous_training_iteration))
    try:
        observations, info = env.reset(seed=seed)
        total_team_reward = 0.0
        total_power = 0.0
        done = False
        actor_hidden = critic_hidden = None
        total_agent_decisions = 0
        shield_corrections = 0
        raw_executed_mode_gap_total = 0
        raw_executed_packet_gap_total = 0
        raw_executed_power_gap_total = 0
        raw_executed_owner_gap_total = 0
        raw_executed_embb_power_gap_total = 0
        urllc_power_abs_change_sum = 0.0
        embb_power_abs_change_sum = 0.0
        greedy_phase_a_decisions = 0
        greedy_noop_selected = 0.0
        greedy_admit_selected = 0.0
        greedy_overlay_selected = 0.0
        greedy_puncture_selected = 0.0
        greedy_selected_retention_sum = 0.0
        greedy_selected_loss_sum = 0.0
        greedy_selected_throughput_sum = 0.0
        greedy_selected_reliability_sum = 0.0
        greedy_selected_embb_min_rate_ok_sum = 0.0
        greedy_feasible_admit_count_sum = 0.0
        greedy_no_feasible_admit_sum = 0.0
        greedy_keep_only_when_no_feasible_admit_sum = 0.0
        greedy_rejected_when_noop_better_sum = 0.0
        greedy_noop_available_sum = 0.0
        greedy_noop_better_sum = 0.0
        greedy_requires_feasible_only = 0.0
        phase_a_embb_power_anchor_binding_count = 0
        phase_a_embb_power_anchor_binding_denom = 0
        phase_a_head_active_total = 0
        phase_a_raw_embb_power_nonzero_total = 0
        phase_a_executed_embb_power_nonzero_total = 0
        phase_a_eligible_total = 0
        phase_a_raw_nonzero_eligible_total = 0
        phase_a_executed_nonzero_eligible_total = 0
        greedy_agreement_total = 0
        greedy_agreement_count = 0
        useful_deviation_count = 0
        harmful_deviation_count = 0
        if not use_greedy and model is not None:
            actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids), device=model.power_log_std.device)

        while not done:
            greedy_debug = {}
            if use_greedy:
                if normalized_greedy_policy == "channel_only":
                    joint_actions = _channel_only_actions(env, observations)
                elif normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput"}:
                    joint_actions, greedy_debug = _hard_feasible_throughput_actions(env, observations)
                elif normalized_greedy_policy == "throughput_feasible":
                    joint_actions, greedy_debug = _throughput_feasible_actions(env, observations)
                elif normalized_greedy_policy == "throughput_biased":
                    joint_actions, greedy_debug = _throughput_biased_actions(env, observations)
                elif normalized_greedy_policy in {"myopic", "myopic_throughput"}:
                    joint_actions, greedy_debug = _myopic_throughput_actions(env, observations)
                elif normalized_greedy_policy == "throughput_only":
                    joint_actions, greedy_debug = _throughput_only_actions(env, observations)
                else:
                    joint_actions = _greedy_actions(env, observations)

                if greedy_debug:
                    planning_phase = all(
                        bool(observations[agent_id].metadata.get("planning_phase", 0.0))
                        for agent_id in env.agent_ids
                    )
                    if not planning_phase:
                        for agent_id in env.agent_ids:
                            debug = greedy_debug.get(agent_id)
                            if not debug:
                                continue
                            greedy_phase_a_decisions += int(debug.get("phase_a_decision", 0.0) > 0.5)
                            greedy_noop_selected += float(debug.get("noop_selected", 0.0))
                            greedy_admit_selected += float(debug.get("admit_selected", 0.0))
                            greedy_overlay_selected += float(debug.get("overlay_selected", 0.0))
                            greedy_puncture_selected += float(debug.get("puncture_selected", 0.0))
                            greedy_selected_retention_sum += float(debug.get("selected_retention", 0.0))
                            greedy_selected_loss_sum += float(debug.get("selected_loss", 0.0))
                            greedy_selected_throughput_sum += float(debug.get("selected_throughput", 0.0))
                            greedy_selected_reliability_sum += float(
                                debug.get("selected_reliability", debug.get("reliability", 0.0))
                            )
                            greedy_selected_embb_min_rate_ok_sum += float(debug.get("selected_embb_min_rate_ok", 1.0))
                            greedy_feasible_admit_count_sum += float(debug.get("feasible_admit_count", 0.0))
                            if float(debug.get("feasible_admit_count", 0.0)) <= 0.5:
                                greedy_no_feasible_admit_sum += 1.0
                            greedy_keep_only_when_no_feasible_admit_sum += float(
                                debug.get("keep_selected_due_to_no_feasible_admit", 0.0)
                            )
                            greedy_rejected_when_noop_better_sum += float(debug.get("rejected_when_noop_better", 0.0))
                            greedy_noop_available_sum += float(debug.get("noop_available", 0.0))
                            greedy_noop_better_sum += float(debug.get("no_op_better_than_best_admit", 0.0))
                            greedy_requires_feasible_only = max(
                                greedy_requires_feasible_only,
                                float(debug.get("current_env_requires_feasible_admission_only", 0.0)),
                            )
            else:
                joint_actions, actor_hidden, critic_hidden = _policy_actions(env, model, observations, actor_hidden, critic_hidden)

            # Shield/autonomy diagnostics: compare raw actions vs executed (masked+shielded) actions.
            planning_phase = all(
                bool(observations[agent_id].metadata.get("planning_phase", 0.0))
                for agent_id in env.agent_ids
            )
            if not planning_phase and phase_a_embb_power_anchor_enabled(env.rl_cfg, iteration=1):
                anchor_target, anchor_weight = _phase_a_embb_power_anchor_targets(env, observations, env.rl_cfg, iteration=1)
                for idx, agent_id in enumerate(env.agent_ids):
                    head_activity = env.action_head_activity(observations[agent_id])
                    if not head_activity.get("phase_a_embb_power_active", False):
                        continue
                    phase_a_embb_power_anchor_binding_denom += 1
                    if idx < len(anchor_weight) and float(anchor_weight[idx]) > 1e-9:
                        phase_a_embb_power_anchor_binding_count += 1
            if planning_phase:
                resolved = {
                    agent_id: env._raw_action_to_shielded_action(joint_actions[agent_id], observations[agent_id])
                    for agent_id in env.agent_ids
                }
            else:
                minislot, rb = env._current_cell()
                resolved = env._resolve_executed_actions(
                    joint_actions,
                    observations,
                    minislot=minislot,
                    rb=rb,
                )

            for agent_id in env.agent_ids:
                raw = joint_actions[agent_id]
                final = resolved[agent_id].action
                diff_flags = env.action_diff_flags(raw, final)
                raw_executed_mode_gap_total += int(diff_flags["mode"])
                raw_executed_packet_gap_total += int(diff_flags["packet"])
                raw_executed_power_gap_total += int(diff_flags["power"])
                raw_executed_owner_gap_total += int(diff_flags["owner"])
                raw_executed_embb_power_gap_total += int(diff_flags["embb_power"])
                urllc_power_abs_change_sum += float(abs(float(raw.power_delta) - float(final.power_delta)))
                embb_power_abs_change_sum += float(abs(float(raw.embb_power_delta) - float(final.embb_power_delta)))
                total_agent_decisions += 1
                if any(diff_flags.values()):
                    shield_corrections += 1
                if (not planning_phase):
                    try:
                        head_activity = env.action_head_activity(observations[agent_id])
                    except Exception:
                        head_activity = {}
                    if bool(head_activity.get("phase_a_embb_power_active", False)):
                        phase_a_head_active_total += 1
                        phase_a_raw_embb_power_nonzero_total += int(abs(float(raw.embb_power_delta)) > 1e-3)
                        phase_a_executed_embb_power_nonzero_total += int(abs(float(final.embb_power_delta)) > 1e-3)
                        # Eligibility: exclude inherently-ineligible cells (inactive/no-owner/no-eMBB) from the denom.
                        try:
                            pinfo = dict(getattr(resolved[agent_id], "phase_a_embb_power_info", {}) or {})
                        except Exception:
                            pinfo = {}
                        reason = str(pinfo.get("zeroed_reason", "") or "").strip().lower()
                        if reason not in {"inactive_head", "no_embb_active", "no_owner", "invalid_owner"}:
                            phase_a_eligible_total += 1
                            phase_a_raw_nonzero_eligible_total += int(abs(float(raw.embb_power_delta)) > 1e-3)
                            phase_a_executed_nonzero_eligible_total += int(abs(float(final.embb_power_delta)) > 1e-3)
                    greedy_ref = observations[agent_id].greedy_reference
                    if greedy_ref is not None:
                        greedy_agreement_total += 1
                        agree_core = (
                            int(final.mode) == int(greedy_ref.mode)
                            and int(final.packet_option) == int(greedy_ref.packet_option)
                        )
                        greedy_agreement_count += int(bool(agree_core))
                        if not bool(agree_core):
                            try:
                                util = float(resolved[agent_id].utility)
                            except Exception:
                                util = 0.0
                            gutil = float(getattr(observations[agent_id], "greedy_reference_utility", 0.0) or 0.0)
                            if util >= gutil + 1.0e-9:
                                useful_deviation_count += 1
                            elif util <= gutil - 1.0e-9:
                                harmful_deviation_count += 1

            observations, rewards, dones, infos = env.step(joint_actions)
            ref_info = infos[env.agent_ids[0]]
            total_team_reward += float(ref_info.get('team_reward', 0.0))
            total_power += sum(float(infos[agent_id].get('power', 0.0)) for agent_id in env.agent_ids)
            done = all(dones.values())

        summary = env.summarize_episode()
        # Unify `embb_positive_rate_ratio` to be exactly `embb_service_ratio` (single definition).
        summary["embb_positive_rate_ratio"] = float(summary.get("embb_service_ratio", 0.0))
        # Ratios use the eligible denominator (exclude inactive/no-owner/no-eMBB cells) to avoid masking suppression.
        phase_a_raw_nonzero_ratio = float(phase_a_raw_nonzero_eligible_total / max(phase_a_eligible_total, 1))
        phase_a_executed_nonzero_ratio = float(phase_a_executed_nonzero_eligible_total / max(phase_a_eligible_total, 1))
        action_agreement_with_greedy = float(greedy_agreement_count / max(greedy_agreement_total, 1))
        useful_deviation_ratio = float(useful_deviation_count / max(greedy_agreement_total, 1))
        harmful_deviation_ratio = float(harmful_deviation_count / max(greedy_agreement_total, 1))
        summary.update({
            'seed': float(seed),
            'team_reward': float(total_team_reward),
            'cell_order_length': float(info.get('cell_order_length', 0)),
            'scheduled_ratio': float(summary.get('urllc_admission_rate', 1.0)),
            'admitted_urllc_reliability': float(summary.get('admitted_urllc_reliability', summary.get('urllc_success_rate', np.nan))),
            'effective_urllc_success_over_arrivals': float(
                summary.get('effective_urllc_success_over_arrivals', summary.get('urllc_success_rate', np.nan))
            ),
            'empty_admission_case': float(summary.get('empty_admission_case', 0.0)),
            'scheduled_packets': float(summary.get('scheduled_packets', 0.0)),
            'active_packets': float(summary.get('active_packets', 0.0)),
            'overlay_count': float(summary.get('overlay_count', 0.0)),
            'puncture_count': float(summary.get('puncture_count', 0.0)),
            'total_power': float(summary.get('total_power', total_power)),
            'episode_sec': float(perf_counter() - episode_start),
            'shield_correction_ratio': float(shield_corrections / max(total_agent_decisions, 1)),
            'raw_executed_any_gap_ratio': float(shield_corrections / max(total_agent_decisions, 1)),
            'raw_executed_mode_gap_ratio': float(raw_executed_mode_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_packet_gap_ratio': float(raw_executed_packet_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_power_gap_ratio': float(raw_executed_power_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_owner_gap_ratio': float(raw_executed_owner_gap_total / max(total_agent_decisions, 1)),
            'raw_executed_embb_power_gap_ratio': float(raw_executed_embb_power_gap_total / max(total_agent_decisions, 1)),
            'policy_autonomy_ratio': float(1.0 - shield_corrections / max(total_agent_decisions, 1)),
            'shield_mode_changed_ratio': float(raw_executed_mode_gap_total / max(total_agent_decisions, 1)),
            'shield_packet_changed_ratio': float(raw_executed_packet_gap_total / max(total_agent_decisions, 1)),
            'shield_owner_changed_ratio': float(raw_executed_owner_gap_total / max(total_agent_decisions, 1)),
            'shield_urllc_power_changed_ratio': float(raw_executed_power_gap_total / max(total_agent_decisions, 1)),
            'shield_embb_power_changed_ratio': float(raw_executed_embb_power_gap_total / max(total_agent_decisions, 1)),
            'shield_mean_abs_urllc_power_delta_change': float(urllc_power_abs_change_sum / max(total_agent_decisions, 1)),
            'shield_mean_abs_embb_power_delta_change': float(embb_power_abs_change_sum / max(total_agent_decisions, 1)),
            'phase_a_embb_power_anchor_binding_ratio': float(
                phase_a_embb_power_anchor_binding_count / max(phase_a_embb_power_anchor_binding_denom, 1)
            ),
            'phaseA_pow_anchor_binding_ratio': float(
                phase_a_embb_power_anchor_binding_count / max(phase_a_embb_power_anchor_binding_denom, 1)
            ),
            'phase_a_raw_embb_power_nonzero_ratio': float(phase_a_raw_nonzero_ratio),
            'phase_a_executed_embb_power_nonzero_ratio': float(phase_a_executed_nonzero_ratio),
            'phase_a_raw_embb_power_nonzero_count': float(phase_a_raw_embb_power_nonzero_total),
            'phase_a_executed_embb_power_nonzero_count': float(phase_a_executed_embb_power_nonzero_total),
            'phase_a_embb_power_head_active_count': float(phase_a_head_active_total),
            'phase_a_embb_power_eligible_count': float(phase_a_eligible_total),
            'action_agreement_with_greedy': float(action_agreement_with_greedy),
            'useful_deviation_ratio': float(useful_deviation_ratio),
            'harmful_deviation_ratio': float(harmful_deviation_ratio),
            'phase0_owner_non_null_ratio_raw': float(summary.get('phase0_owner_non_null_ratio_raw', 0.0)),
            'phase0_owner_non_null_ratio_executed': float(summary.get('phase0_owner_non_null_ratio_executed', 0.0)),
            'phase0_owner_change_ratio_vs_snapshot_raw': float(summary.get('phase0_owner_change_ratio_vs_snapshot_raw', 0.0)),
            'phase0_owner_change_ratio_vs_snapshot_executed': float(summary.get('phase0_owner_change_ratio_vs_snapshot_executed', 0.0)),
            'phase0_owner_change_budget_used': float(summary.get('phase0_owner_change_budget_used', 0.0)),
            'phase0_owner_change_budget_allowed': float(summary.get('phase0_owner_change_budget_allowed', 0.0)),
            'phase0_owner_change_budget_clipped_ratio': float(summary.get('phase0_owner_change_budget_clipped_ratio', 0.0)),
            'phase0_owner_change_kept_topk_ratio': float(summary.get('phase0_owner_change_kept_topk_ratio', 0.0)),
            'phase0_owner_change_dropped_over_budget_ratio': float(summary.get('phase0_owner_change_dropped_over_budget_ratio', 0.0)),
            'phase0_owner_raw_changed_count_mean': float(summary.get('phase0_owner_raw_changed_count_mean', 0.0)),
            'phase0_owner_allowed_k_mean': float(summary.get('phase0_owner_allowed_k_mean', 0.0)),
            'phase0_owner_executed_changed_count_mean': float(summary.get('phase0_owner_executed_changed_count_mean', 0.0)),
            'phase0_owner_dropped_count_mean': float(summary.get('phase0_owner_dropped_count_mean', 0.0)),
            'ph0_owner_raw_non_snapshot_ratio': float(summary.get('ph0_owner_raw_non_snapshot_ratio', 0.0)),
            'ph0_owner_exec_non_snapshot_ratio': float(summary.get('ph0_owner_exec_non_snapshot_ratio', 0.0)),
            'phase0_owner_fallback_to_candidate0_ratio': float(summary.get('phase0_owner_fallback_to_candidate0_ratio', 0.0)),
            'phase0_owner_invalid_option_ratio': float(summary.get('phase0_owner_invalid_option_ratio', 0.0)),
            'phase0_owner_null_selected_ratio': float(summary.get('phase0_owner_null_selected_ratio', 0.0)),
            'phase0_owner_invalid_to_null_ratio': float(summary.get('phase0_owner_invalid_to_null_ratio', 0.0)),
            'phase0_owner_invalid_to_snapshot_ratio': float(summary.get('phase0_owner_invalid_to_snapshot_ratio', 0.0)),
            'phase0_owner_invalid_to_non_snapshot_ratio': float(summary.get('phase0_owner_invalid_to_non_snapshot_ratio', 0.0)),
            'phase0_owner_restored_to_snapshot_ratio': float(summary.get('phase0_owner_restored_to_snapshot_ratio', 0.0)),
            'phase0_owner_kept_null_ratio': float(summary.get('phase0_owner_kept_null_ratio', 0.0)),
            'phase0_owner_replaced_with_non_snapshot_ratio': float(summary.get('phase0_owner_replaced_with_non_snapshot_ratio', 0.0)),
            'phase0_owner_changed_and_effective_ratio': float(summary.get('phase0_owner_changed_and_effective_ratio', 0.0)),
            'phase0_owner_effective_change_count': float(summary.get('phase0_owner_effective_change_count', 0.0)),
            'phase0_owner_same_as_snapshot_ratio': float(summary.get('phase0_owner_same_as_snapshot_ratio', 0.0)),
            'phase0_owner_effective_rate_gain_vs_snapshot_mean': float(summary.get('phase0_owner_effective_rate_gain_vs_snapshot_mean', 0.0)),
            'phase0_owner_change_harmful_ratio': float(summary.get('phase0_owner_change_harmful_ratio', 0.0)),
            'phase_a_embb_power_write_ratio': float(summary.get('phase_a_embb_power_write_ratio', 0.0)),
            'phase_a_embb_power_zeroed_keep_mode_ratio': float(summary.get('phase_a_embb_power_zeroed_keep_mode_ratio', 0.0)),
            'phase_a_power_raw_positive_ratio': float(summary.get('phase_a_power_raw_positive_ratio', 0.0)),
            'phase_a_power_positive_clamped_to_zero_ratio': float(summary.get('phase_a_power_positive_clamped_to_zero_ratio', 0.0)),
            'phase_a_power_negative_executed_ratio': float(summary.get('phase_a_power_negative_executed_ratio', 0.0)),
            # Snapshot leakage diagnostics (episode-level; carried through eval/report).
            'owner_snapshot_leak_detected': float(summary.get('owner_snapshot_leak_detected', 0.0)),
            'owner_snapshot_in_observation': float(summary.get('owner_snapshot_in_observation', 0.0)),
            'owner_snapshot_used_for_init': float(summary.get('owner_snapshot_used_for_init', 0.0)),
            'owner_snapshot_used_for_fallback': float(summary.get('owner_snapshot_used_for_fallback', 0.0)),
            'owner_snapshot_used_for_reward': float(summary.get('owner_snapshot_used_for_reward', 0.0)),
            'owner_init_from_snapshot': float(summary.get('owner_init_from_snapshot', 0.0)),
            'owner_snapshot_fallback_taken': float(summary.get('owner_snapshot_fallback_taken', 0.0)),
            'intervention_severity': float(
                (raw_executed_mode_gap_total / max(total_agent_decisions, 1))
                + (raw_executed_packet_gap_total / max(total_agent_decisions, 1))
                + 0.5 * (raw_executed_owner_gap_total / max(total_agent_decisions, 1))
                + (urllc_power_abs_change_sum / max(total_agent_decisions, 1))
                + (embb_power_abs_change_sum / max(total_agent_decisions, 1))
            ),
        })
        if use_greedy and normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput", "throughput_only", "throughput_biased", "myopic", "myopic_throughput"}:
            if normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput"}:
                baseline_key = "hard_feasible_throughput_greedy"
            elif normalized_greedy_policy == "throughput_biased":
                baseline_key = "throughput_biased_greedy"
            elif normalized_greedy_policy == "throughput_only":
                baseline_key = "throughput_only_greedy"
            else:
                baseline_key = "myopic_throughput_greedy"
            decision_denom = max(greedy_phase_a_decisions, 1)
            summary.update({
                **_baseline_metadata(baseline_key),
                'greedy_noop_selected_ratio': float(greedy_noop_selected / decision_denom),
                'greedy_admit_selected_ratio': float(greedy_admit_selected / decision_denom),
                'greedy_overlay_ratio': float(greedy_overlay_selected / decision_denom),
                'greedy_puncture_ratio': float(greedy_puncture_selected / decision_denom),
                'greedy_avg_embb_retention': float(greedy_selected_retention_sum / decision_denom),
                'greedy_avg_embb_loss': float(greedy_selected_loss_sum / decision_denom),
                'greedy_avg_selected_throughput': float(greedy_selected_throughput_sum / decision_denom),
                'greedy_selected_embb_throughput': float(greedy_selected_throughput_sum / decision_denom),
                'greedy_feasible_admit_count': float(greedy_feasible_admit_count_sum / decision_denom),
                'greedy_no_feasible_admit_ratio': float(greedy_no_feasible_admit_sum / decision_denom),
                'greedy_keep_only_when_no_feasible_admit_ratio': float(
                    greedy_keep_only_when_no_feasible_admit_sum / max(greedy_noop_selected, 1.0)
                ),
                'greedy_selected_urllc_reliability': float(greedy_selected_reliability_sum / decision_denom),
                'greedy_selected_embb_min_rate_ok': float(greedy_selected_embb_min_rate_ok_sum / decision_denom),
                'greedy_avg_rejected_urllc_when_noop_better': float(
                    greedy_rejected_when_noop_better_sum / decision_denom
                ),
                'greedy_noop_available_ratio': float(greedy_noop_available_sum / decision_denom),
                'greedy_noop_better_ratio': float(greedy_noop_better_sum / decision_denom),
                'greedy_requires_feasible_admission_only': float(greedy_requires_feasible_only),
            })
            summary.update(
                _baseline_narrative(
                    baseline_key,
                    greedy_requires_feasible_admission_only=bool(greedy_requires_feasible_only),
                )
            )
        _eval_log(
            env.rl_cfg,
            f"rollout_episode mode={'greedy:' + normalized_greedy_policy if use_greedy else 'policy'} "
            f"seed={seed} sec={float(summary['episode_sec']):.3f}",
        )
        return summary
    finally:
        env.current_training_iteration = previous_training_iteration
        env.training_progress_frac = previous_training_progress
        env.rl_cfg.env.include_greedy_reference_in_obs = previous_greedy_obs
        restore_eval_fallback(env, fallback_state)


def evaluate_policy_only(env, model, cfg) -> Dict[str, float]:
    """Evaluate SR-MAPPO on a fixed load sweep using the true environment reward."""
    eval_start = perf_counter()
    eval_loads = list(getattr(cfg.training, 'eval_loads', []))
    if not eval_loads:
        eval_loads = [configure_env_for_users_per_uav(env, _ensure_default_load(env))]

    episodes = int(getattr(cfg.training, 'eval_episodes_per_load', cfg.training.eval_episodes))
    per_load: List[Dict[str, float]] = []
    for load_idx, target_load in enumerate(eval_loads):
        load_start = perf_counter()
        actual_load = configure_env_for_users_per_uav(env, float(target_load))
        policy_metrics: List[Dict[str, float]] = []
        for episode_idx in range(episodes):
            seed = int(cfg.training.train_seed + 5000 + 100 * load_idx + episode_idx)
            policy_metrics.append(rollout_episode(env, model=model, seed=seed, use_greedy=False))

        per_load.append({
            'target_load': float(target_load),
            'actual_load': float(actual_load),
            'policy_mean_reward': _mean(policy_metrics, 'team_reward'),
            'policy_mean_scheduled_packets': _mean(policy_metrics, 'scheduled_packets'),
            'policy_mean_scheduled_ratio': _mean(policy_metrics, 'scheduled_ratio', 1.0),
            'policy_mean_reliability': _mean(policy_metrics, 'admitted_urllc_reliability', np.nan),
            'policy_mean_admitted_urllc_reliability': _mean(policy_metrics, 'admitted_urllc_reliability', np.nan),
            'policy_mean_effective_urllc_success_over_arrivals': _mean(
                policy_metrics, 'effective_urllc_success_over_arrivals', 1.0
            ),
            'policy_mean_empty_admission_case_ratio': _mean(policy_metrics, 'empty_admission_case', 0.0),
            'policy_mean_power': _mean(policy_metrics, 'total_power'),
            'policy_mean_overlay': _mean(policy_metrics, 'overlay_count'),
            'policy_mean_puncture': _mean(policy_metrics, 'puncture_count'),
            'policy_mean_overlay_ratio': _mean(policy_metrics, 'overlay_ratio'),
            'policy_mean_puncture_ratio': _mean(policy_metrics, 'puncture_ratio'),
            'policy_mean_embb_rate': _mean(policy_metrics, 'embb_total_rate'),
            'policy_mean_embb_user_rate': _mean(policy_metrics, 'embb_user_rate_mean'),
            'policy_mean_embb_service_ratio': _mean(policy_metrics, 'embb_service_ratio'),
            'policy_mean_embb_min_rate_satisfaction_ratio': _mean(policy_metrics, 'embb_min_rate_satisfaction_ratio'),
            'policy_mean_embb_served_user_count': _mean(policy_metrics, 'embb_served_user_count'),
            # Kept for backward compatibility: equals `policy_mean_embb_service_ratio`.
            'policy_mean_embb_positive_rate_ratio': _mean(policy_metrics, 'embb_positive_rate_ratio'),
            'policy_mean_terminal_embb_service_floor_penalty': _mean(policy_metrics, 'terminal_embb_service_floor_penalty'),
            'policy_mean_terminal_embb_min_rate_floor_penalty': _mean(policy_metrics, 'terminal_embb_min_rate_floor_penalty'),
            'policy_mean_terminal_embb_service_bonus': _mean(policy_metrics, 'terminal_embb_service_bonus'),
            'policy_mean_terminal_embb_min_rate_bonus': _mean(policy_metrics, 'terminal_embb_min_rate_bonus'),
            'policy_mean_terminal_avg_served_embb_rate_bonus': _mean(policy_metrics, 'terminal_avg_served_embb_rate_bonus'),
            'policy_mean_urllc_admission_over_service_tradeoff_penalty': _mean(policy_metrics, 'urllc_admission_over_service_tradeoff_penalty'),
            'policy_mean_throughput_per_watt': _mean(policy_metrics, 'throughput_per_watt'),
            'policy_mean_avg_throughput_per_served_embb_user': _mean(policy_metrics, 'avg_throughput_per_served_embb_user'),
            'policy_mean_phase_a_embb_power_runtime_enabled': _mean(policy_metrics, 'phase_a_embb_power_runtime_enabled'),
            'policy_mean_phase_a_embb_power_changed_count': _mean(policy_metrics, 'phase_a_embb_power_changed_count'),
            'policy_mean_phase_a_embb_power_changed_ratio': _mean(policy_metrics, 'phase_a_embb_power_changed_ratio'),
            'policy_mean_phase_a_embb_power_mean_raw_delta': _mean(policy_metrics, 'phase_a_embb_power_mean_raw_delta'),
            'policy_mean_phase_a_embb_power_mean_executed_delta': _mean(policy_metrics, 'phase_a_embb_power_mean_executed_delta'),
            'policy_mean_phase_a_embb_power_pre_clip_mean_delta': _mean(policy_metrics, 'phase_a_embb_power_pre_clip_mean_delta'),
            'policy_mean_phase_a_embb_power_post_clip_mean_delta': _mean(policy_metrics, 'phase_a_embb_power_post_clip_mean_delta'),
            'policy_mean_phase_a_embb_power_post_projection_mean_delta': _mean(policy_metrics, 'phase_a_embb_power_post_projection_mean_delta'),
            'policy_mean_phase_a_embb_power_final_executed_mean_delta': _mean(policy_metrics, 'phase_a_embb_power_final_executed_mean_delta'),
            'policy_mean_phase_a_embb_power_clip_ratio': _mean(policy_metrics, 'phase_a_embb_power_clip_ratio'),
            'policy_mean_phase_a_embb_power_projection_ratio': _mean(policy_metrics, 'phase_a_embb_power_projection_ratio'),
            'policy_mean_phase_a_embb_power_cap_hit_ratio': _mean(policy_metrics, 'phase_a_embb_power_cap_hit_ratio'),
            'policy_mean_phase_a_embb_power_floor_hit_ratio': _mean(policy_metrics, 'phase_a_embb_power_floor_hit_ratio'),
            'policy_mean_phase_a_embb_power_projection_l2_mean': _mean(policy_metrics, 'phase_a_embb_power_projection_l2_mean'),
            'policy_mean_phase_a_embb_power_pre_vs_final_l1_mean': _mean(policy_metrics, 'phase_a_embb_power_pre_vs_final_l1_mean'),
            'policy_mean_phase_a_embb_power_pre_vs_final_sign_consistency': _mean(policy_metrics, 'phase_a_embb_power_pre_vs_final_sign_consistency'),
            'policy_mean_phase_a_embb_power_anchor_binding_ratio': _mean(policy_metrics, 'phase_a_embb_power_anchor_binding_ratio'),
            'policy_mean_phase_a_embb_power_effective_nonzero_ratio': _mean(policy_metrics, 'phase_a_embb_power_effective_nonzero_ratio'),
            'policy_mean_phase_a_embb_power_write_ratio': _mean(policy_metrics, 'phase_a_embb_power_write_ratio'),
            'policy_mean_phase_a_embb_power_zeroed_keep_mode_ratio': _mean(policy_metrics, 'phase_a_embb_power_zeroed_keep_mode_ratio'),
            'policy_mean_phase_a_embb_power_raw_saturation_ratio': _mean(policy_metrics, 'phase_a_embb_power_raw_saturation_ratio'),
            'policy_mean_phase_a_embb_power_final_std': _mean(policy_metrics, 'phase_a_embb_power_final_std'),
            'policy_mean_phase_a_embb_power_cellwise_diversity': _mean(policy_metrics, 'phase_a_embb_power_cellwise_diversity'),
            'policy_mean_phase_a_embb_power_mean_raw_delta_l2': _mean(policy_metrics, 'phase_a_embb_power_mean_raw_delta_l2'),
            'policy_mean_phase_a_embb_power_floor_binding_strength': _mean(policy_metrics, 'phase_a_embb_power_floor_binding_strength'),
            'policy_mean_phase_a_embb_power_cap_binding_strength': _mean(policy_metrics, 'phase_a_embb_power_cap_binding_strength'),
            'policy_mean_phase_a_embb_power_proj_delta_l1': _mean(policy_metrics, 'phase_a_embb_power_proj_delta_l1'),
            'policy_mean_phase_a_embb_power_proj_delta_l2': _mean(policy_metrics, 'phase_a_embb_power_proj_delta_l2'),
            'policy_mean_phase_a_embb_power_pre_to_floor_delta': _mean(policy_metrics, 'phase_a_embb_power_pre_to_floor_delta'),
            'policy_mean_phase_a_embb_power_pre_to_cap_delta': _mean(policy_metrics, 'phase_a_embb_power_pre_to_cap_delta'),
            'policy_mean_phase_a_embb_power_final_minus_proj_delta': _mean(policy_metrics, 'phase_a_embb_power_final_minus_proj_delta'),
            'policy_mean_phase0_owner_non_null_ratio_raw': _mean(policy_metrics, 'phase0_owner_non_null_ratio_raw'),
            'policy_mean_phase0_owner_non_null_ratio_executed': _mean(policy_metrics, 'phase0_owner_non_null_ratio_executed'),
            'policy_mean_phase0_owner_change_ratio_vs_snapshot_raw': _mean(policy_metrics, 'phase0_owner_change_ratio_vs_snapshot_raw'),
            'policy_mean_phase0_owner_change_ratio_vs_snapshot_executed': _mean(policy_metrics, 'phase0_owner_change_ratio_vs_snapshot_executed'),
            'policy_mean_phase0_owner_change_budget_used': _mean(policy_metrics, 'phase0_owner_change_budget_used'),
            'policy_mean_phase0_owner_change_budget_allowed': _mean(policy_metrics, 'phase0_owner_change_budget_allowed'),
            'policy_mean_phase0_owner_change_budget_clipped_ratio': _mean(policy_metrics, 'phase0_owner_change_budget_clipped_ratio'),
            'policy_mean_phase0_owner_change_kept_topk_ratio': _mean(policy_metrics, 'phase0_owner_change_kept_topk_ratio'),
            'policy_mean_phase0_owner_change_dropped_over_budget_ratio': _mean(policy_metrics, 'phase0_owner_change_dropped_over_budget_ratio'),
            'policy_mean_phase0_owner_raw_changed_count_mean': _mean(policy_metrics, 'phase0_owner_raw_changed_count_mean'),
            'policy_mean_phase0_owner_allowed_k_mean': _mean(policy_metrics, 'phase0_owner_allowed_k_mean'),
            'policy_mean_phase0_owner_executed_changed_count_mean': _mean(policy_metrics, 'phase0_owner_executed_changed_count_mean'),
            'policy_mean_phase0_owner_dropped_count_mean': _mean(policy_metrics, 'phase0_owner_dropped_count_mean'),
            'policy_mean_phase0_owner_candidate_positive_objective_ratio': _mean(policy_metrics, 'phase0_owner_candidate_positive_objective_ratio'),
            'policy_mean_phase0_owner_accepted_positive_objective_ratio': _mean(policy_metrics, 'phase0_owner_accepted_positive_objective_ratio'),
            'policy_mean_phase0_owner_rejected_nonpositive_objective_ratio': _mean(policy_metrics, 'phase0_owner_rejected_nonpositive_objective_ratio'),
            'policy_mean_phase0_owner_objective_gain_accepted_mean': _mean(policy_metrics, 'phase0_owner_objective_gain_accepted_mean'),
            'policy_mean_phase0_owner_harmful_accepted_ratio': _mean(policy_metrics, 'phase0_owner_harmful_accepted_ratio'),
            'policy_mean_ph0_owner_raw_non_snapshot_ratio': _mean(policy_metrics, 'ph0_owner_raw_non_snapshot_ratio'),
            'policy_mean_ph0_owner_exec_non_snapshot_ratio': _mean(policy_metrics, 'ph0_owner_exec_non_snapshot_ratio'),
            'policy_mean_phase0_owner_fallback_to_candidate0_ratio': _mean(policy_metrics, 'phase0_owner_fallback_to_candidate0_ratio'),
            'policy_mean_phase0_owner_invalid_option_ratio': _mean(policy_metrics, 'phase0_owner_invalid_option_ratio'),
            'policy_mean_phase0_owner_null_selected_ratio': _mean(policy_metrics, 'phase0_owner_null_selected_ratio'),
            'policy_mean_phase0_owner_invalid_to_null_ratio': _mean(policy_metrics, 'phase0_owner_invalid_to_null_ratio'),
            'policy_mean_phase0_owner_invalid_to_snapshot_ratio': _mean(policy_metrics, 'phase0_owner_invalid_to_snapshot_ratio'),
            'policy_mean_phase0_owner_invalid_to_non_snapshot_ratio': _mean(policy_metrics, 'phase0_owner_invalid_to_non_snapshot_ratio'),
            'policy_mean_phase0_owner_restored_to_snapshot_ratio': _mean(policy_metrics, 'phase0_owner_restored_to_snapshot_ratio'),
            'policy_mean_phase0_owner_kept_null_ratio': _mean(policy_metrics, 'phase0_owner_kept_null_ratio'),
            'policy_mean_phase0_owner_replaced_with_non_snapshot_ratio': _mean(policy_metrics, 'phase0_owner_replaced_with_non_snapshot_ratio'),
            'policy_mean_phase0_owner_changed_and_effective_ratio': _mean(policy_metrics, 'phase0_owner_changed_and_effective_ratio'),
            'policy_mean_phase0_owner_effective_change_count': _mean(policy_metrics, 'phase0_owner_effective_change_count'),
            'policy_mean_phase0_owner_changed_but_unserved_ratio': _mean(policy_metrics, 'phase0_owner_changed_but_unserved_ratio'),
            'policy_mean_phase0_owner_same_as_snapshot_ratio': _mean(policy_metrics, 'phase0_owner_same_as_snapshot_ratio'),
            'policy_mean_phase0_owner_effective_service_gain_ratio': _mean(policy_metrics, 'phase0_owner_effective_service_gain_ratio'),
            'policy_mean_phase0_owner_effective_rate_gain_vs_snapshot_mean': _mean(policy_metrics, 'phase0_owner_effective_rate_gain_vs_snapshot_mean'),
            'policy_mean_phase0_owner_change_harmful_ratio': _mean(policy_metrics, 'phase0_owner_change_harmful_ratio'),
            'policy_mean_phase_a_power_raw_positive_ratio': _mean(policy_metrics, 'phase_a_power_raw_positive_ratio'),
            'policy_mean_phase_a_power_positive_clamped_to_zero_ratio': _mean(policy_metrics, 'phase_a_power_positive_clamped_to_zero_ratio'),
            'policy_mean_phase_a_power_negative_executed_ratio': _mean(policy_metrics, 'phase_a_power_negative_executed_ratio'),
            'policy_mean_embb_rate_loss_due_to_intercell_ratio': _mean(policy_metrics, 'embb_rate_loss_due_to_intercell_ratio'),
            'policy_mean_planning_embb_power_changed_ratio': _mean(policy_metrics, 'planning_embb_power_changed_ratio'),
            'policy_mean_owner_snapshot_leak_detected': _mean(policy_metrics, 'owner_snapshot_leak_detected'),
            'policy_mean_owner_init_from_snapshot': _mean(policy_metrics, 'owner_init_from_snapshot'),
            'policy_mean_owner_snapshot_fallback_taken': _mean(policy_metrics, 'owner_snapshot_fallback_taken'),
            'policy_mean_unscheduled_ratio': _mean(policy_metrics, 'unscheduled_ratio'),
            'policy_mean_fairness': _mean(policy_metrics, 'jain_fairness'),
            'policy_mean_embb_min_rate_shortfall': _mean(policy_metrics, 'embb_min_rate_shortfall'),
            'policy_mean_both_modes_feasible_ratio': _mean(policy_metrics, 'both_modes_feasible_ratio'),
            'policy_mean_safe_puncture_available_ratio': _mean(policy_metrics, 'safe_puncture_available_ratio'),
            'policy_mean_overlay_chosen_when_safe_puncture_available_ratio': _mean(policy_metrics, 'overlay_chosen_when_safe_puncture_available_ratio'),
            'policy_mean_puncture_chosen_when_safe_puncture_available_ratio': _mean(policy_metrics, 'puncture_chosen_when_safe_puncture_available_ratio'),
            'policy_mean_teacher_mode_agreement_ratio': _mean(policy_metrics, 'teacher_mode_agreement_ratio'),
            'policy_mean_mode_anchor_active_ratio': _mean(policy_metrics, 'mode_anchor_active_ratio'),
        })
        _eval_log(
            cfg,
            f"evaluate_policy_only load={float(target_load):.1f} actual={float(actual_load):.2f} "
            f"episodes={episodes} total_sec={perf_counter() - load_start:.3f} "
            f"avg_episode_sec={_mean(policy_metrics, 'episode_sec'):.3f} "
            f"phaseA_pwr_changed_count={_mean(policy_metrics, 'phase_a_embb_power_changed_count'):.3f} "
            f"phaseA_pwr_changed={_mean(policy_metrics, 'phase_a_embb_power_changed_ratio'):.3f} "
            f"phaseA_delta={_mean(policy_metrics, 'phase_a_embb_power_mean_executed_delta'):.3f} "
            f"phaseA_path(pre/clip/proj/final)={_mean(policy_metrics, 'phase_a_embb_power_pre_clip_mean_delta'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_post_clip_mean_delta'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_post_projection_mean_delta'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_final_executed_mean_delta'):.3f} "
            f"phaseA_ratios(clip/proj/cap/floor)={_mean(policy_metrics, 'phase_a_embb_power_clip_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_projection_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_cap_hit_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_floor_hit_ratio'):.3f} "
            f"phaseA_anchor_bind={_mean(policy_metrics, 'phase_a_embb_power_anchor_binding_ratio'):.3f} "
            f"phaseA_eff_nz={_mean(policy_metrics, 'phase_a_embb_power_effective_nonzero_ratio'):.3f} "
            f"phaseA_bind(floor/cap)={_mean(policy_metrics, 'phase_a_embb_power_floor_binding_strength'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_cap_binding_strength'):.3f} "
            f"phaseA_proj_delta(l1/l2)={_mean(policy_metrics, 'phase_a_embb_power_proj_delta_l1'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_embb_power_proj_delta_l2'):.3f} "
            f"owner_budget(raw/allowed/exe/drop)={_mean(policy_metrics, 'phase0_owner_raw_changed_count_mean'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_allowed_k_mean'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_executed_changed_count_mean'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_dropped_count_mean'):.3f} "
            f"owner_obj(pos/acc/rej/acc_gain/harm)={_mean(policy_metrics, 'phase0_owner_candidate_positive_objective_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_accepted_positive_objective_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_rejected_nonpositive_objective_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_objective_gain_accepted_mean'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_harmful_accepted_ratio'):.3f} "
            f"owner_ratio(raw/exe/eff/same)={_mean(policy_metrics, 'phase0_owner_change_ratio_vs_snapshot_raw'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_change_ratio_vs_snapshot_executed'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_changed_and_effective_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_same_as_snapshot_ratio'):.3f} "
            f"owner_gain(mean/harmful)={_mean(policy_metrics, 'phase0_owner_effective_rate_gain_vs_snapshot_mean'):.3f}/"
            f"{_mean(policy_metrics, 'phase0_owner_change_harmful_ratio'):.3f} "
            f"phaseA(raw_pos/clamped/neg_exe)={_mean(policy_metrics, 'phase_a_power_raw_positive_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_power_positive_clamped_to_zero_ratio'):.3f}/"
            f"{_mean(policy_metrics, 'phase_a_power_negative_executed_ratio'):.3f}",
        )

    summary = {
        'policy_mean_reward': float(np.mean([item['policy_mean_reward'] for item in per_load])) if per_load else 0.0,
        'policy_mean_scheduled_packets': float(np.mean([item['policy_mean_scheduled_packets'] for item in per_load])) if per_load else 0.0,
        'policy_mean_scheduled_ratio': float(np.mean([item['policy_mean_scheduled_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_reliability': _mean(per_load, 'policy_mean_admitted_urllc_reliability', np.nan),
        'policy_mean_admitted_urllc_reliability': _mean(per_load, 'policy_mean_admitted_urllc_reliability', np.nan),
        'policy_mean_effective_urllc_success_over_arrivals': _mean(
            per_load, 'policy_mean_effective_urllc_success_over_arrivals', 1.0
        ),
        'policy_mean_empty_admission_case_ratio': _mean(per_load, 'policy_mean_empty_admission_case_ratio', 0.0),
        'policy_mean_power': float(np.mean([item['policy_mean_power'] for item in per_load])) if per_load else 0.0,
        'policy_mean_overlay': float(np.mean([item['policy_mean_overlay'] for item in per_load])) if per_load else 0.0,
        'policy_mean_puncture': float(np.mean([item['policy_mean_puncture'] for item in per_load])) if per_load else 0.0,
        'policy_mean_overlay_ratio': float(np.mean([item['policy_mean_overlay_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_puncture_ratio': float(np.mean([item['policy_mean_puncture_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_embb_rate': float(np.mean([item['policy_mean_embb_rate'] for item in per_load])) if per_load else 0.0,
        'policy_mean_embb_user_rate': float(np.mean([item['policy_mean_embb_user_rate'] for item in per_load])) if per_load else 0.0,
        'policy_mean_embb_service_ratio': float(np.mean([item['policy_mean_embb_service_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_embb_min_rate_satisfaction_ratio': float(np.mean([item.get('policy_mean_embb_min_rate_satisfaction_ratio', 0.0) for item in per_load])) if per_load else 0.0,
        'policy_mean_terminal_embb_service_floor_penalty': float(np.mean([item.get('policy_mean_terminal_embb_service_floor_penalty', 0.0) for item in per_load])) if per_load else 0.0,
        'policy_mean_terminal_embb_min_rate_floor_penalty': float(np.mean([item.get('policy_mean_terminal_embb_min_rate_floor_penalty', 0.0) for item in per_load])) if per_load else 0.0,
        'policy_mean_urllc_admission_over_service_tradeoff_penalty': float(np.mean([item.get('policy_mean_urllc_admission_over_service_tradeoff_penalty', 0.0) for item in per_load])) if per_load else 0.0,
        'policy_mean_throughput_per_watt': float(np.mean([item['policy_mean_throughput_per_watt'] for item in per_load])) if per_load else 0.0,
        'policy_mean_avg_throughput_per_served_embb_user': float(
            np.mean([item['policy_mean_avg_throughput_per_served_embb_user'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_runtime_enabled': float(
            np.mean([item['policy_mean_phase_a_embb_power_runtime_enabled'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_changed_count': float(
            np.mean([item['policy_mean_phase_a_embb_power_changed_count'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_changed_ratio': float(
            np.mean([item['policy_mean_phase_a_embb_power_changed_ratio'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_mean_raw_delta': float(
            np.mean([item['policy_mean_phase_a_embb_power_mean_raw_delta'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_mean_executed_delta': float(
            np.mean([item['policy_mean_phase_a_embb_power_mean_executed_delta'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_pre_clip_mean_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_pre_clip_mean_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_post_clip_mean_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_post_clip_mean_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_post_projection_mean_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_post_projection_mean_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_final_executed_mean_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_final_executed_mean_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_clip_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_clip_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_projection_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_projection_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_cap_hit_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_cap_hit_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_floor_hit_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_floor_hit_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_projection_l2_mean': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_projection_l2_mean', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_pre_vs_final_l1_mean': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_pre_vs_final_l1_mean', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_pre_vs_final_sign_consistency': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_pre_vs_final_sign_consistency', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_anchor_binding_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_anchor_binding_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_effective_nonzero_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_effective_nonzero_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_write_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_write_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_zeroed_keep_mode_ratio': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_zeroed_keep_mode_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_floor_binding_strength': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_floor_binding_strength', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_cap_binding_strength': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_cap_binding_strength', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_proj_delta_l1': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_proj_delta_l1', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_proj_delta_l2': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_proj_delta_l2', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_pre_to_floor_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_pre_to_floor_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_pre_to_cap_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_pre_to_cap_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase_a_embb_power_final_minus_proj_delta': float(
            np.mean([item.get('policy_mean_phase_a_embb_power_final_minus_proj_delta', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase0_owner_change_ratio_vs_snapshot_raw': float(
            np.mean([item.get('policy_mean_phase0_owner_change_ratio_vs_snapshot_raw', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase0_owner_change_ratio_vs_snapshot_executed': float(
            np.mean([item.get('policy_mean_phase0_owner_change_ratio_vs_snapshot_executed', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase0_owner_change_budget_allowed': float(
            np.mean([item.get('policy_mean_phase0_owner_change_budget_allowed', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_phase0_owner_change_budget_clipped_ratio': float(
            np.mean([item.get('policy_mean_phase0_owner_change_budget_clipped_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_ph0_owner_raw_non_snapshot_ratio': float(
            np.mean([item.get('policy_mean_ph0_owner_raw_non_snapshot_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_ph0_owner_exec_non_snapshot_ratio': float(
            np.mean([item.get('policy_mean_ph0_owner_exec_non_snapshot_ratio', 0.0) for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_unscheduled_ratio': float(np.mean([item['policy_mean_unscheduled_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_embb_min_rate_shortfall': float(np.mean([item['policy_mean_embb_min_rate_shortfall'] for item in per_load])) if per_load else 0.0,
        'policy_mean_fairness': float(np.mean([item['policy_mean_fairness'] for item in per_load])) if per_load else 0.0,
        'policy_mean_both_modes_feasible_ratio': float(np.mean([item['policy_mean_both_modes_feasible_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_safe_puncture_available_ratio': float(np.mean([item['policy_mean_safe_puncture_available_ratio'] for item in per_load])) if per_load else 0.0,
        'policy_mean_overlay_chosen_when_safe_puncture_available_ratio': float(
            np.mean([item['policy_mean_overlay_chosen_when_safe_puncture_available_ratio'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_puncture_chosen_when_safe_puncture_available_ratio': float(
            np.mean([item['policy_mean_puncture_chosen_when_safe_puncture_available_ratio'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_teacher_mode_agreement_ratio': float(
            np.mean([item['policy_mean_teacher_mode_agreement_ratio'] for item in per_load])
        ) if per_load else 0.0,
        'policy_mean_mode_anchor_active_ratio': float(
            np.mean([item['policy_mean_mode_anchor_active_ratio'] for item in per_load])
        ) if per_load else 0.0,
        'policy_score': float(np.mean([item['policy_mean_reward'] for item in per_load])) if per_load else 0.0,
        'policy_throughput_score': float(np.mean([item['policy_mean_embb_rate'] for item in per_load])) if per_load else 0.0,
        'non_worse_than_greedy': 1.0,
        'eval_loads': [float(load) for load in eval_loads],
        'per_load': per_load,
        'selection_basis': 'mean_team_reward_over_fixed_eval_loads',
        'throughput_selection_basis': 'mean_embb_throughput_over_fixed_eval_loads',
    }
    _eval_log(cfg, f"evaluate_policy_only total_sec={perf_counter() - eval_start:.3f}")
    return summary


def evaluate_dual_selection(env, model, cfg, force_full_compare: bool = False) -> Dict[str, float]:
    """Evaluate on absolute reward/throughput plus configurable Greedy comparisons."""
    total_start = perf_counter()
    compare_modes = _normalized_eval_compare_modes(cfg, force_full_compare=force_full_compare)
    selected_baseline_mode = _selection_baseline_mode(cfg)
    baseline_cache: Dict[str, Dict[str, float]] = {}

    absolute_start = perf_counter()
    absolute_summary = evaluate_policy_only(env, model, cfg)
    absolute_sec = perf_counter() - absolute_start

    selected_start = perf_counter()
    selected_cfg = deepcopy(cfg)
    selected_cfg.training.greedy_baseline_mode = selected_baseline_mode
    selected_summary = evaluate_against_greedy(env, model, selected_cfg)
    baseline_cache[selected_baseline_mode] = selected_summary
    selected_sec = perf_counter() - selected_start

    def _resolve_compare(mode_name: str, baseline_mode: str) -> tuple[Dict[str, float], float]:
        if mode_name not in compare_modes:
            if _normalize_baseline_mode(baseline_mode) == selected_baseline_mode:
                return selected_summary, 0.0
            return _empty_compare_summary(baseline_mode), 0.0
        normalized_mode = _normalize_baseline_mode(baseline_mode)
        if normalized_mode == selected_baseline_mode:
            return selected_summary, 0.0
        cached = baseline_cache.get(normalized_mode)
        if cached is not None:
            return cached, 0.0
        compare_cfg = deepcopy(cfg)
        compare_cfg.training.greedy_baseline_mode = normalized_mode
        compare_start = perf_counter()
        compare_summary = evaluate_against_greedy(env, model, compare_cfg)
        compare_sec = perf_counter() - compare_start
        baseline_cache[normalized_mode] = compare_summary
        return compare_summary, compare_sec

    original_summary, original_sec = _resolve_compare("original", "original")
    matched_summary, matched_sec = _resolve_compare("matched", "matched_fixed_embb")
    throughput_feasible_summary, throughput_feasible_sec = _resolve_compare(
        "throughput_feasible",
        "throughput_feasible_oracle",
    )
    throughput_only_summary, throughput_only_sec = _resolve_compare("throughput_only", "throughput_only_greedy")
    channel_only_summary, channel_only_sec = _resolve_compare("channel_only", "channel_only_greedy")

    summary = dict(absolute_summary)
    summary["selected_baseline_key"] = selected_baseline_mode
    summary["selected_baseline_label"] = _baseline_label(selected_baseline_mode)
    summary.update(_baseline_metadata(selected_baseline_mode))
    summary["compare_selected_baseline"] = selected_summary
    summary["selected_baseline_mode"] = selected_baseline_mode
    summary["compare_original"] = original_summary
    summary["compare_matched"] = matched_summary
    summary["compare_throughput_feasible_oracle"] = throughput_feasible_summary
    summary["compare_throughput_only"] = throughput_only_summary
    summary["compare_channel_only"] = channel_only_summary
    summary["policy_score_selected_baseline"] = float(selected_summary.get("policy_score", 0.0))
    summary["policy_throughput_selected_baseline"] = float(selected_summary.get("policy_mean_embb_rate", 0.0))
    summary["weighted_selection_score"] = float(selected_summary.get("weighted_selection_score", 0.0))
    summary["loadwise_selection_score"] = list(selected_summary.get("loadwise_selection_score", []))
    summary["loadwise_admission_floor"] = list(selected_summary.get("loadwise_admission_floor", []))
    summary["loadwise_floor_pass"] = list(selected_summary.get("loadwise_floor_pass", []))
    summary["loadwise_floor_violation"] = list(selected_summary.get("loadwise_floor_violation", []))
    summary["loadwise_puncture_loss_gap"] = list(selected_summary.get("loadwise_puncture_loss_gap", []))
    summary["loadwise_overlay_retention_gap"] = list(selected_summary.get("loadwise_overlay_retention_gap", []))
    summary["weighted_floor_violation"] = float(selected_summary.get("weighted_floor_violation", 0.0))
    summary["weighted_power_ceiling_violation"] = float(selected_summary.get("weighted_power_ceiling_violation", 0.0))
    summary["all_loads_pass_admission_floor"] = float(selected_summary.get("all_loads_pass_admission_floor", 1.0))
    summary["all_loads_pass_power_ceiling"] = float(selected_summary.get("all_loads_pass_power_ceiling", 1.0))
    summary["all_loads_pass_selection_constraints"] = float(selected_summary.get("all_loads_pass_selection_constraints", 1.0))
    summary["loadwise_power_ratio_ceiling"] = list(selected_summary.get("loadwise_power_ratio_ceiling", []))
    summary["loadwise_power_ceiling_pass"] = list(selected_summary.get("loadwise_power_ceiling_pass", []))
    summary["loadwise_power_ceiling_violation"] = list(selected_summary.get("loadwise_power_ceiling_violation", []))
    summary["policy_score_vs_original_greedy"] = float(original_summary.get("policy_score", float("-inf")))
    summary["policy_score_vs_matched_greedy"] = float(matched_summary.get("policy_score", float("-inf")))
    summary["policy_score_vs_throughput_feasible_oracle"] = float(
        throughput_feasible_summary.get("policy_score", float("-inf"))
    )
    summary["policy_score_vs_throughput_only_greedy"] = float(throughput_only_summary.get("policy_score", float("-inf")))
    summary["policy_score_vs_channel_only_greedy"] = float(channel_only_summary.get("policy_score", float("-inf")))
    summary["policy_throughput_vs_original_greedy"] = float(original_summary.get("mean_rate_ratio", 0.0))
    summary["policy_throughput_vs_matched_greedy"] = float(matched_summary.get("mean_rate_ratio", 0.0))
    summary["policy_throughput_vs_throughput_feasible_oracle"] = float(
        throughput_feasible_summary.get("mean_rate_ratio", 0.0)
    )
    summary["policy_throughput_vs_throughput_only_greedy"] = float(throughput_only_summary.get("mean_rate_ratio", 0.0))
    summary["policy_throughput_vs_channel_only_greedy"] = float(channel_only_summary.get("mean_rate_ratio", 0.0))
    summary["policy_power_vs_original_greedy"] = float(original_summary.get("mean_power_ratio", 1.0))
    summary["policy_power_vs_matched_greedy"] = float(matched_summary.get("mean_power_ratio", 1.0))
    summary["policy_power_vs_throughput_feasible_oracle"] = float(
        throughput_feasible_summary.get("mean_power_ratio", 1.0)
    )
    summary["policy_power_vs_throughput_only_greedy"] = float(throughput_only_summary.get("mean_power_ratio", 1.0))
    summary["policy_power_vs_channel_only_greedy"] = float(channel_only_summary.get("mean_power_ratio", 1.0))
    summary["policy_non_worse_vs_original_greedy"] = float(original_summary.get("non_worse_than_greedy", 0.0))
    summary["policy_non_worse_vs_matched_greedy"] = float(matched_summary.get("non_worse_than_greedy", 0.0))
    summary["policy_non_worse_vs_throughput_feasible_oracle"] = float(
        throughput_feasible_summary.get("non_worse_than_greedy", 0.0)
    )
    summary["policy_non_worse_vs_throughput_only_greedy"] = float(
        throughput_only_summary.get("non_worse_than_greedy", 0.0)
    )
    summary["policy_non_worse_vs_channel_only_greedy"] = float(channel_only_summary.get("non_worse_than_greedy", 0.0))
    summary["selection_mode"] = str(getattr(cfg.training, "selection_mode", "dual_metric"))
    summary["selection_baseline_mode"] = selected_baseline_mode
    summary["eval_compare_modes"] = list(compare_modes)
    if bool(getattr(cfg.training, "load_aware_objective", False)):
        summary["policy_score"] = float(selected_summary.get("policy_score", summary.get("policy_score", 0.0)))
        summary["policy_throughput_score"] = float(
            selected_summary.get("policy_mean_embb_rate", summary.get("policy_throughput_score", 0.0))
        )
        summary["policy_mean_scheduled_ratio"] = float(
            selected_summary.get("policy_mean_scheduled_ratio", summary.get("policy_mean_scheduled_ratio", 0.0))
        )
    frontier_source = (
        selected_summary
        if list(selected_summary.get("per_load", []) or [])
        else (
            throughput_only_summary
            if list(throughput_only_summary.get("per_load", []) or [])
            else selected_summary
        )
    )
    summary.update(_multiload_frontier_metrics(frontier_source, cfg))
    _eval_log(
        cfg,
        f"evaluate_dual_selection compare_modes={compare_modes} "
        f"policy_only={absolute_sec:.3f}s selected={selected_sec:.3f}s "
        f"original={original_sec:.3f}s matched={matched_sec:.3f}s "
        f"throughput_feasible={throughput_feasible_sec:.3f}s "
        f"throughput_only={throughput_only_sec:.3f}s channel_only={channel_only_sec:.3f}s "
        f"total={perf_counter() - total_start:.3f}s",
    )
    return summary


def _mean(items, key, default=0.0):
    if not items:
        return float(default)
    arr = np.asarray([item.get(key, default) for item in items], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _evaluate_one_load(env, model, cfg, target_load: float, seed_base: int) -> Dict[str, float]:
    actual_load = configure_env_for_users_per_uav(env, target_load)
    policy_metrics: List[Dict[str, float]] = []
    greedy_metrics: List[Dict[str, float]] = []
    episodes = int(getattr(cfg.training, 'eval_episodes_per_load', cfg.training.eval_episodes))
    greedy_mode = _greedy_baseline_mode(cfg)
    for episode_idx in range(episodes):
        seed = seed_base + episode_idx
        policy_metrics.append(rollout_episode(env, model=model, seed=seed, use_greedy=False))
        if greedy_mode == "original":
            greedy_metrics.append(_run_original_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "original_greedy_normal_v1":
            greedy_metrics.append(_run_original_greedy_normal_v1_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "original_greedy_normal_v2":
            greedy_metrics.append(_run_original_greedy_normal_v2_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "matched_fixed_embb":
            greedy_metrics.append(_run_matched_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "throughput_feasible_oracle":
            greedy_metrics.append(_run_throughput_feasible_oracle_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "throughput_biased_greedy":
            greedy_metrics.append(_run_throughput_biased_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "hard_feasible_throughput_greedy":
            greedy_metrics.append(_run_hard_feasible_throughput_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "myopic_throughput_greedy":
            greedy_metrics.append(_run_myopic_throughput_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "throughput_only_greedy":
            greedy_metrics.append(_run_throughput_only_greedy_episode(env, seed=seed, slot_index=episode_idx))
        elif greedy_mode == "channel_only_greedy":
            greedy_metrics.append(_run_channel_only_greedy_episode(env, seed=seed, slot_index=episode_idx))
        else:
            raise RuntimeError("Frozen greedy metrics should be handled outside _evaluate_one_load().")

    policy_embb = _mean(policy_metrics, 'embb_total_rate')
    greedy_embb = _mean(greedy_metrics, 'embb_total_rate')
    policy_adm = _mean(policy_metrics, 'scheduled_ratio', 1.0)
    greedy_adm = _mean(greedy_metrics, 'scheduled_ratio', 1.0)
    policy_reliability = _mean(policy_metrics, 'admitted_urllc_reliability', np.nan)
    greedy_reliability = _mean(greedy_metrics, 'admitted_urllc_reliability', np.nan)
    policy_effective_success = _mean(policy_metrics, 'effective_urllc_success_over_arrivals', 1.0)
    greedy_effective_success = _mean(greedy_metrics, 'effective_urllc_success_over_arrivals', 1.0)
    policy_power = _mean(policy_metrics, 'total_power')
    greedy_power = _mean(greedy_metrics, 'total_power')
    policy_overlay = _mean(policy_metrics, 'overlay_count')
    policy_puncture = _mean(policy_metrics, 'puncture_count')
    greedy_overlay = _mean(greedy_metrics, 'overlay_count')
    policy_puncture_loss = _mean(policy_metrics, 'avg_puncture_embb_loss')
    greedy_puncture_loss = _mean(greedy_metrics, 'avg_puncture_embb_loss')
    policy_overlay_retention = _mean(policy_metrics, 'avg_overlay_retention')
    greedy_overlay_retention = _mean(greedy_metrics, 'avg_overlay_retention')

    rate_ratio = policy_embb / max(greedy_embb, 1e-9)
    admission_gap = policy_adm - greedy_adm
    power_ratio = policy_power / max(greedy_power, 1e-9)
    overlay_gap = policy_overlay - greedy_overlay
    throughput_excess = float(rate_ratio - 1.0)
    puncture_loss_gap = float(policy_puncture_loss - greedy_puncture_loss)
    overlay_retention_gap = float(policy_overlay_retention - greedy_overlay_retention)
    low_damage_objective = bool(getattr(cfg.training, 'low_damage_admission_objective', False))
    if bool(getattr(cfg.training, 'load_aware_objective', False)):
        score = load_aware_selection_score(
            actual_load,
            throughput_excess,
            admission_gap,
            puncture_loss_gap,
            overlay_retention_gap,
            power_ratio,
            low_damage=low_damage_objective,
        )
    else:
        score = (
            4.0 * throughput_excess
            + 2.0 * max(throughput_excess - 0.10, 0.0)
            + 0.50 * admission_gap
            - 0.05 * (power_ratio - 1.0)
            + 0.02 * overlay_gap
        )
    selection_metrics = _selection_constraint_metrics(
        cfg,
        actual_load,
        greedy_adm,
        policy_adm,
        policy_effective_success,
        rate_ratio,
        policy_overlay,
        policy_puncture,
        power_ratio,
    )
    weighted_contribution = float(load_aware_score_mix(actual_load, low_damage=low_damage_objective) * score) if bool(getattr(cfg.training, 'load_aware_objective', False)) else 0.0
    non_worse = (
        policy_embb >= cfg.training.non_worse_rate_ratio * greedy_embb
        and policy_adm >= greedy_adm + cfg.training.non_worse_admission_gap
        and policy_power <= cfg.training.non_worse_power_tolerance * greedy_power
    )

    summary = {
        'target_load': float(target_load),
        'actual_load': float(actual_load),
        'policy_mean_reward': _mean(policy_metrics, 'team_reward'),
        'policy_mean_scheduled_packets': _mean(policy_metrics, 'scheduled_packets'),
        'policy_mean_scheduled_ratio': policy_adm,
        'policy_mean_reliability': policy_reliability,
        'policy_mean_admitted_urllc_reliability': policy_reliability,
        'policy_mean_effective_urllc_success_over_arrivals': policy_effective_success,
        'policy_mean_empty_admission_case_ratio': _mean(policy_metrics, 'empty_admission_case', 0.0),
        'policy_mean_power': policy_power,
        'policy_mean_overlay': policy_overlay,
        'policy_mean_puncture': policy_puncture,
        'policy_mean_embb_rate': policy_embb,
        'policy_mean_embb_service_ratio': _mean(policy_metrics, 'embb_service_ratio'),
        'policy_mean_embb_min_rate_satisfaction_ratio': _mean(policy_metrics, 'embb_min_rate_satisfaction_ratio'),
        'policy_mean_embb_service_ratio_after_puncture_deduction': _mean(
            policy_metrics,
            'embb_service_ratio_after_puncture_deduction',
            _mean(policy_metrics, 'embb_service_ratio', 0.0),
        ),
        'policy_mean_embb_min_rate_satisfaction_after_puncture_deduction': _mean(
            policy_metrics,
            'embb_min_rate_satisfaction_after_puncture_deduction',
            _mean(policy_metrics, 'embb_min_rate_satisfaction_ratio', 0.0),
        ),
        'policy_mean_embb_served_user_count': _mean(policy_metrics, 'embb_served_user_count'),
        'policy_mean_embb_rate_loss_due_to_intercell_ratio': _mean(policy_metrics, 'embb_rate_loss_due_to_intercell_ratio', 0.0),
        'policy_mean_selected_action_intercell_cost_after_source_mask_mean': _mean(
            policy_metrics,
            'selected_action_intercell_cost_after_source_mask_mean',
            _mean(policy_metrics, 'selected_action_intercell_cost_mean', 0.0),
        ),
        'policy_mean_selected_action_intercell_cost_after_source_mask_p95': _mean(
            policy_metrics,
            'selected_action_intercell_cost_after_source_mask_p95',
            _mean(policy_metrics, 'selected_action_intercell_cost_p95', 0.0),
        ),
        'policy_mean_intercell_per_admitted_packet': _mean(policy_metrics, 'intercell_per_admitted_packet', 0.0),
        'policy_mean_phase_a_embb_power_raw_saturation_ratio': _mean(policy_metrics, 'phase_a_embb_power_raw_saturation_ratio', 0.0),
        'policy_mean_phase_a_embb_power_cap_hit_ratio': _mean(policy_metrics, 'phase_a_embb_power_cap_hit_ratio', 0.0),
        'policy_mean_phase_a_embb_power_floor_hit_ratio': _mean(policy_metrics, 'phase_a_embb_power_floor_hit_ratio', 0.0),
        'policy_mean_phase_a_embb_power_final_std': _mean(policy_metrics, 'phase_a_embb_power_final_std', 0.0),
        'greedy_mean_reward': _mean(greedy_metrics, 'team_reward'),
        'greedy_mean_scheduled_packets': _mean(greedy_metrics, 'scheduled_packets'),
        'greedy_mean_scheduled_ratio': greedy_adm,
        'greedy_mean_reliability': greedy_reliability,
        'greedy_mean_admitted_urllc_reliability': greedy_reliability,
        'greedy_mean_effective_urllc_success_over_arrivals': greedy_effective_success,
        'greedy_mean_empty_admission_case_ratio': _mean(greedy_metrics, 'empty_admission_case', 0.0),
        'greedy_mean_power': greedy_power,
        'greedy_mean_overlay': greedy_overlay,
        'greedy_mean_puncture': _mean(greedy_metrics, 'puncture_count'),
        'greedy_mean_embb_rate': greedy_embb,
        'greedy_mean_embb_service_ratio': _mean(greedy_metrics, 'embb_service_ratio'),
        'greedy_mean_embb_min_rate_satisfaction_ratio': _mean(greedy_metrics, 'embb_min_rate_satisfaction_ratio'),
        'greedy_mean_embb_service_ratio_after_puncture_deduction': _mean(
            greedy_metrics,
            'embb_service_ratio_after_puncture_deduction',
            _mean(greedy_metrics, 'embb_service_ratio', 0.0),
        ),
        'greedy_mean_embb_min_rate_satisfaction_after_puncture_deduction': _mean(
            greedy_metrics,
            'embb_min_rate_satisfaction_after_puncture_deduction',
            _mean(greedy_metrics, 'embb_min_rate_satisfaction_ratio', 0.0),
        ),
        'greedy_mean_embb_served_user_count': _mean(greedy_metrics, 'embb_served_user_count'),
        'greedy_mean_embb_rate_loss_due_to_intercell_ratio': _mean(greedy_metrics, 'embb_rate_loss_due_to_intercell_ratio', 0.0),
        'greedy_mean_selected_action_intercell_cost_after_source_mask_mean': _mean(
            greedy_metrics,
            'selected_action_intercell_cost_after_source_mask_mean',
            _mean(greedy_metrics, 'selected_action_intercell_cost_mean', 0.0),
        ),
        'greedy_mean_selected_action_intercell_cost_after_source_mask_p95': _mean(
            greedy_metrics,
            'selected_action_intercell_cost_after_source_mask_p95',
            _mean(greedy_metrics, 'selected_action_intercell_cost_p95', 0.0),
        ),
        'greedy_mean_intercell_per_admitted_packet': _mean(greedy_metrics, 'intercell_per_admitted_packet', 0.0),
        'greedy_noop_selected_ratio': _mean(greedy_metrics, 'greedy_noop_selected_ratio'),
        'greedy_admit_selected_ratio': _mean(greedy_metrics, 'greedy_admit_selected_ratio'),
        'greedy_overlay_ratio': _mean(greedy_metrics, 'greedy_overlay_ratio'),
        'greedy_puncture_ratio': _mean(greedy_metrics, 'greedy_puncture_ratio'),
        'greedy_avg_embb_retention': _mean(greedy_metrics, 'greedy_avg_embb_retention'),
        'greedy_avg_embb_loss': _mean(greedy_metrics, 'greedy_avg_embb_loss'),
        'greedy_avg_selected_throughput': _mean(greedy_metrics, 'greedy_avg_selected_throughput'),
        'greedy_avg_rejected_urllc_when_noop_better': _mean(
            greedy_metrics,
            'greedy_avg_rejected_urllc_when_noop_better',
        ),
        'greedy_noop_available_ratio': _mean(greedy_metrics, 'greedy_noop_available_ratio'),
        'greedy_noop_better_ratio': _mean(greedy_metrics, 'greedy_noop_better_ratio'),
        'greedy_requires_feasible_admission_only': _mean(
            greedy_metrics,
            'greedy_requires_feasible_admission_only',
        ),
        'rate_ratio': float(rate_ratio),
        'admission_gap': float(admission_gap),
        'power_ratio': float(power_ratio),
        'avg_puncture_loss_gap': puncture_loss_gap,
        'avg_overlay_retention_gap': overlay_retention_gap,
        'throughput_excess': throughput_excess,
        **selection_metrics,
        'weighted_selection_contribution': weighted_contribution,
        'score': float(score),
        'non_worse': float(non_worse),
        'greedy_baseline': greedy_mode,
    }
    try:
        policy_intercell = float(summary.get('policy_mean_intercell_per_admitted_packet', 0.0) or 0.0)
        greedy_intercell = float(summary.get('greedy_mean_intercell_per_admitted_packet', 0.0) or 0.0)
        summary['intercell_excess_vs_greedy_ratio'] = float(
            (policy_intercell - greedy_intercell) / max(greedy_intercell, 1.0e-12)
        )
        summary['intercell_per_admitted_packet_mappo'] = float(policy_intercell)
        summary['intercell_per_admitted_packet_greedy'] = float(greedy_intercell)
        summary['admission_intercell_efficiency'] = float(policy_adm / max(policy_intercell + 1.0e-12, 1.0e-12))
    except Exception:
        summary['intercell_excess_vs_greedy_ratio'] = 0.0
        summary['intercell_per_admitted_packet_mappo'] = 0.0
        summary['intercell_per_admitted_packet_greedy'] = 0.0
        summary['admission_intercell_efficiency'] = 0.0
    summary.update(_baseline_metadata(greedy_mode))
    summary.update(
        _baseline_narrative(
            greedy_mode,
            greedy_requires_feasible_admission_only=_mean(greedy_metrics, 'greedy_requires_feasible_admission_only', 0.0) > 0.5,
        )
    )
    return summary


def evaluate_against_greedy(env, model, cfg) -> Dict[str, float]:
    eval_start = perf_counter()
    previous_greedy_obs = bool(getattr(env.rl_cfg.env, 'include_greedy_reference_in_obs', False))
    env.rl_cfg.env.include_greedy_reference_in_obs = False
    try:
        eval_loads = list(getattr(cfg.training, 'eval_loads', []))
        if not eval_loads:
            eval_loads = [configure_env_for_users_per_uav(env, _ensure_default_load(env))]

        per_load = []
        greedy_mode = _greedy_baseline_mode(cfg)
        if greedy_mode == "frozen_json":
            frozen_payload = _load_frozen_greedy_payload(cfg)
            frozen_lookup = _frozen_greedy_metrics_by_load(frozen_payload)
            episodes = int(getattr(cfg.training, 'eval_episodes_per_load', cfg.training.eval_episodes))
            for load_idx, target_load in enumerate(eval_loads):
                load_start = perf_counter()
                actual_load = configure_env_for_users_per_uav(env, float(target_load))
                policy_metrics: List[Dict[str, float]] = []
                for episode_idx in range(episodes):
                    seed = cfg.training.train_seed + 1000 + 100 * load_idx + episode_idx
                    policy_metrics.append(rollout_episode(env, model=model, seed=seed, use_greedy=False))
                frozen = frozen_lookup.get(float(target_load))
                if frozen is None:
                    raise KeyError(f"Frozen greedy metrics do not contain load {target_load}")
                policy_embb = _mean(policy_metrics, 'embb_total_rate')
                policy_adm = _mean(policy_metrics, 'scheduled_ratio', 1.0)
                policy_reliability = _mean(policy_metrics, 'admitted_urllc_reliability', np.nan)
                policy_effective_success = _mean(policy_metrics, 'effective_urllc_success_over_arrivals', 1.0)
                policy_power = _mean(policy_metrics, 'total_power')
                policy_overlay = _mean(policy_metrics, 'overlay_count')
                policy_puncture = _mean(policy_metrics, 'puncture_count')
                greedy_embb = float(frozen.get('embb_rate', 0.0))
                greedy_adm = float(frozen.get('urllc_admission', 1.0))
                greedy_reliability = float(
                    frozen.get('admitted_urllc_reliability', frozen.get('urllc_reliability', np.nan))
                )
                greedy_effective_success = float(
                    frozen.get(
                        'effective_urllc_success_over_arrivals',
                        1.0 if float(frozen.get('active_packets', 0.0)) <= 0.0 else 0.0
                    )
                )
                greedy_power = float(frozen.get('total_power', 0.0))
                greedy_overlay = float(frozen.get('overlay_count', 0.0))
                policy_puncture_loss = _mean(policy_metrics, 'avg_puncture_embb_loss')
                greedy_puncture_loss = float(frozen.get('avg_puncture_loss', 0.0))
                policy_overlay_retention = _mean(policy_metrics, 'avg_overlay_retention')
                greedy_overlay_retention = float(frozen.get('avg_overlay_retention', 0.0))
                rate_ratio = policy_embb / max(greedy_embb, 1e-9)
                admission_gap = policy_adm - greedy_adm
                power_ratio = policy_power / max(greedy_power, 1e-9)
                overlay_gap = policy_overlay - greedy_overlay
                throughput_excess = float(rate_ratio - 1.0)
                puncture_loss_gap = float(policy_puncture_loss - greedy_puncture_loss)
                overlay_retention_gap = float(policy_overlay_retention - greedy_overlay_retention)
                low_damage_objective = bool(getattr(cfg.training, 'low_damage_admission_objective', False))
                if bool(getattr(cfg.training, 'load_aware_objective', False)):
                    score = load_aware_selection_score(
                        actual_load,
                        throughput_excess,
                        admission_gap,
                        puncture_loss_gap,
                        overlay_retention_gap,
                        power_ratio,
                        low_damage=low_damage_objective,
                    )
                else:
                    score = (
                        4.0 * throughput_excess
                        + 2.0 * max(throughput_excess - 0.10, 0.0)
                        + 0.50 * admission_gap
                        - 0.05 * (power_ratio - 1.0)
                        + 0.02 * overlay_gap
                    )
                policy_puncture = _mean(policy_metrics, 'puncture_count')
                selection_metrics = _selection_constraint_metrics(
                    cfg,
                    actual_load,
                    greedy_adm,
                    policy_adm,
                    policy_effective_success,
                    rate_ratio,
                    policy_overlay,
                    policy_puncture,
                    power_ratio,
                )
                weighted_contribution = float(load_aware_score_mix(actual_load, low_damage=low_damage_objective) * score) if bool(getattr(cfg.training, 'load_aware_objective', False)) else 0.0
                non_worse = (
                    policy_embb >= cfg.training.non_worse_rate_ratio * greedy_embb
                    and policy_adm >= greedy_adm + cfg.training.non_worse_admission_gap
                    and policy_power <= cfg.training.non_worse_power_tolerance * greedy_power
                )
                per_load_summary = {
                    'target_load': float(target_load),
                    'actual_load': float(actual_load),
                    'policy_mean_reward': _mean(policy_metrics, 'team_reward'),
                    'policy_mean_scheduled_packets': _mean(policy_metrics, 'scheduled_packets'),
                    'policy_mean_scheduled_ratio': policy_adm,
                    'policy_mean_reliability': policy_reliability,
                    'policy_mean_admitted_urllc_reliability': policy_reliability,
                    'policy_mean_effective_urllc_success_over_arrivals': policy_effective_success,
                    'policy_mean_empty_admission_case_ratio': _mean(policy_metrics, 'empty_admission_case', 0.0),
                    'policy_mean_power': policy_power,
                    'policy_mean_overlay': policy_overlay,
                    'policy_mean_puncture': policy_puncture,
                    'policy_mean_embb_rate': policy_embb,
                    'greedy_mean_reward': 0.0,
                    'greedy_mean_scheduled_packets': float(frozen.get('scheduled_packets', 0.0)),
                    'greedy_mean_scheduled_ratio': greedy_adm,
                    'greedy_mean_reliability': greedy_reliability,
                    'greedy_mean_admitted_urllc_reliability': greedy_reliability,
                    'greedy_mean_effective_urllc_success_over_arrivals': greedy_effective_success,
                    'greedy_mean_empty_admission_case_ratio': float(frozen.get('empty_admission_case', 0.0)),
                    'greedy_mean_power': greedy_power,
                    'greedy_mean_overlay': greedy_overlay,
                    'greedy_mean_puncture': float(frozen.get('puncture_count', 0.0)),
                    'greedy_mean_embb_rate': greedy_embb,
                    'greedy_noop_selected_ratio': float(frozen.get('greedy_noop_selected_ratio', 0.0)),
                    'greedy_admit_selected_ratio': float(frozen.get('greedy_admit_selected_ratio', 0.0)),
                    'greedy_overlay_ratio': float(frozen.get('greedy_overlay_ratio', 0.0)),
                    'greedy_puncture_ratio': float(frozen.get('greedy_puncture_ratio', 0.0)),
                    'greedy_avg_embb_retention': float(frozen.get('greedy_avg_embb_retention', 0.0)),
                    'greedy_avg_embb_loss': float(frozen.get('greedy_avg_embb_loss', 0.0)),
                    'greedy_avg_selected_throughput': float(frozen.get('greedy_avg_selected_throughput', 0.0)),
                    'greedy_avg_rejected_urllc_when_noop_better': float(
                        frozen.get('greedy_avg_rejected_urllc_when_noop_better', 0.0)
                    ),
                    'greedy_noop_available_ratio': float(frozen.get('greedy_noop_available_ratio', 0.0)),
                    'greedy_noop_better_ratio': float(frozen.get('greedy_noop_better_ratio', 0.0)),
                    'greedy_requires_feasible_admission_only': float(
                        frozen.get('greedy_requires_feasible_admission_only', 0.0)
                    ),
                    'rate_ratio': float(rate_ratio),
                    'admission_gap': float(admission_gap),
                    'power_ratio': float(power_ratio),
                    'avg_puncture_loss_gap': puncture_loss_gap,
                    'avg_overlay_retention_gap': overlay_retention_gap,
                    'throughput_excess': throughput_excess,
                    **selection_metrics,
                    'weighted_selection_contribution': weighted_contribution,
                    'score': float(score),
                    'non_worse': float(non_worse),
                    'greedy_baseline': 'frozen_json',
                }
                frozen_mode = str(frozen_payload.get('greedy_baseline_mode', '') or '').strip().lower()
                per_load_summary.update(_baseline_metadata(frozen_mode))
                per_load_summary.update(
                    _baseline_narrative(
                        frozen_mode,
                        greedy_requires_feasible_admission_only=float(frozen.get('greedy_requires_feasible_admission_only', 0.0)) > 0.5,
                    )
                )
                per_load.append(per_load_summary)
                _eval_log(
                    cfg,
                    f"evaluate_against_greedy baseline=frozen_json load={float(target_load):.1f} "
                    f"actual={float(actual_load):.2f} episodes={episodes} total_sec={perf_counter() - load_start:.3f} "
                    f"avg_policy_episode_sec={_mean(policy_metrics, 'episode_sec'):.3f}",
                )
        else:
            for load_idx, target_load in enumerate(eval_loads):
                load_start = perf_counter()
                per_load_item = _evaluate_one_load(
                    env,
                    model,
                    cfg,
                    float(target_load),
                    cfg.training.train_seed + 1000 + 100 * load_idx,
                )
                per_load.append(per_load_item)
                _eval_log(
                    cfg,
                    f"evaluate_against_greedy baseline={greedy_mode} load={float(target_load):.1f} "
                    f"actual={float(per_load_item.get('actual_load', target_load)):.2f} "
                    f"episodes={int(getattr(cfg.training, 'eval_episodes_per_load', cfg.training.eval_episodes))} "
                    f"total_sec={perf_counter() - load_start:.3f}",
                )

        mean_rate_ratio = float(np.mean([item['rate_ratio'] for item in per_load]))
        min_rate_ratio = float(np.min([item['rate_ratio'] for item in per_load]))
        mean_admission_gap = float(np.mean([item['admission_gap'] for item in per_load]))
        mean_power_ratio = float(np.mean([item['power_ratio'] for item in per_load]))
        non_worse_fraction = float(np.mean([item['non_worse'] for item in per_load]))
        gain_fraction = float(np.mean([item['rate_ratio'] >= 1.0 for item in per_load]))
        gain10_fraction = float(np.mean([item['rate_ratio'] >= 1.10 for item in per_load]))
        has_loadwise_constraints = bool(
            dict(getattr(cfg.training, 'selection_admission_floor_by_load', {}) or {})
            or dict(getattr(cfg.training, 'selection_power_ratio_ceiling_by_load', {}) or {})
            or dict(getattr(cfg.training, 'selection_throughput_ratio_floor_by_load', {}) or {})
            or dict(getattr(cfg.training, 'selection_puncture_ratio_floor_by_load', {}) or {})
            or dict(getattr(cfg.training, 'selection_overlay_ratio_ceiling_by_load', {}) or {})
            or _selection_reliability_floor(cfg) > 0.0
            or _selection_floor_ratio_to_baseline(cfg) > 0.0
            or _selection_puncture_ratio_ceiling(cfg) < 1.0 - 1e-9
        )
        admission_only_pass_by_load = [
            float(item.get('policy_mean_scheduled_ratio', 0.0) >= item.get('selection_admission_floor', 0.0) - 1e-9)
            for item in per_load
        ]
        if bool(getattr(cfg.training, 'load_aware_objective', False)):
            selection_score = float(np.sum([item['weighted_selection_contribution'] for item in per_load]))
            loadwise_selection_score = [float(item['score']) for item in per_load]
            loadwise_admission_floor = [float(item['selection_admission_floor']) for item in per_load]
            loadwise_floor_pass = [float(item['floor_pass']) for item in per_load]
            low_damage_objective = bool(getattr(cfg.training, 'low_damage_admission_objective', False))
            loadwise_floor_violation = [float(item['floor_violation']) for item in per_load]
            loadwise_puncture_loss_gap = [float(item['avg_puncture_loss_gap']) for item in per_load]
            loadwise_overlay_retention_gap = [float(item['avg_overlay_retention_gap']) for item in per_load]
            loadwise_power_ratio_ceiling = [float(item.get('selection_power_ratio_ceiling', float('inf'))) for item in per_load]
            loadwise_power_ceiling_pass = [float(item.get('power_ceiling_pass', 1.0)) for item in per_load]
            loadwise_power_ceiling_violation = [float(item.get('power_ceiling_violation', 0.0)) for item in per_load]
            weighted_floor_violation = float(np.sum([
                load_aware_score_mix(item['actual_load'], low_damage=low_damage_objective) * float(item['floor_violation'])
                for item in per_load
            ]))
            weighted_power_ceiling_violation = float(np.sum([
                load_aware_score_mix(item['actual_load'], low_damage=low_damage_objective) * float(item.get('power_ceiling_violation', 0.0))
                for item in per_load
            ]))
            all_loads_pass_floor = float(np.mean(loadwise_floor_pass) >= 1.0 - 1e-9)
            all_loads_pass_power_ceiling = float(np.mean(loadwise_power_ceiling_pass) >= 1.0 - 1e-9)
        else:
            selection_score = (
                3.0 * (mean_rate_ratio - 1.0)
                + 2.0 * (min_rate_ratio - 1.0)
                + 1.2 * gain_fraction
                + 1.8 * gain10_fraction
                + 0.5 * mean_admission_gap
                - 0.05 * (mean_power_ratio - 1.0)
            )
            loadwise_selection_score = [float(item['score']) for item in per_load] if has_loadwise_constraints else []
            loadwise_admission_floor = [float(item['selection_admission_floor']) for item in per_load] if has_loadwise_constraints else []
            loadwise_floor_pass = [float(item['floor_pass']) for item in per_load] if has_loadwise_constraints else []
            loadwise_floor_violation = [float(item['floor_violation']) for item in per_load] if has_loadwise_constraints else []
            loadwise_puncture_loss_gap = [float(item['avg_puncture_loss_gap']) for item in per_load] if has_loadwise_constraints else []
            loadwise_overlay_retention_gap = [float(item['avg_overlay_retention_gap']) for item in per_load] if has_loadwise_constraints else []
            loadwise_power_ratio_ceiling = [float(item.get('selection_power_ratio_ceiling', float('inf'))) for item in per_load] if has_loadwise_constraints else []
            loadwise_power_ceiling_pass = [float(item.get('power_ceiling_pass', 1.0)) for item in per_load] if has_loadwise_constraints else []
            loadwise_power_ceiling_violation = [float(item.get('power_ceiling_violation', 0.0)) for item in per_load] if has_loadwise_constraints else []
            weighted_floor_violation = float(np.mean(loadwise_floor_violation)) if loadwise_floor_violation else 0.0
            weighted_power_ceiling_violation = float(np.mean(loadwise_power_ceiling_violation)) if loadwise_power_ceiling_violation else 0.0
            all_loads_pass_floor = float(np.mean(loadwise_floor_pass) >= 1.0 - 1e-9) if loadwise_floor_pass else 1.0
            all_loads_pass_power_ceiling = float(np.mean(loadwise_power_ceiling_pass) >= 1.0 - 1e-9) if loadwise_power_ceiling_pass else 1.0
        all_loads_pass_admission_floor = float(np.mean(admission_only_pass_by_load) >= 1.0 - 1e-9) if admission_only_pass_by_load else 1.0

        summary = {
            'policy_mean_reward': float(np.mean([item['policy_mean_reward'] for item in per_load])),
            'policy_mean_scheduled_packets': float(np.mean([item['policy_mean_scheduled_packets'] for item in per_load])),
            'policy_mean_scheduled_ratio': float(np.mean([item['policy_mean_scheduled_ratio'] for item in per_load])),
            'policy_mean_reliability': _mean(per_load, 'policy_mean_admitted_urllc_reliability', np.nan),
            'policy_mean_admitted_urllc_reliability': _mean(per_load, 'policy_mean_admitted_urllc_reliability', np.nan),
            'policy_mean_effective_urllc_success_over_arrivals': _mean(
                per_load, 'policy_mean_effective_urllc_success_over_arrivals', 1.0
            ),
            'policy_mean_empty_admission_case_ratio': _mean(per_load, 'policy_mean_empty_admission_case_ratio', 0.0),
            'policy_mean_power': float(np.mean([item['policy_mean_power'] for item in per_load])),
            'policy_mean_overlay': float(np.mean([item['policy_mean_overlay'] for item in per_load])),
            'policy_mean_puncture': float(np.mean([item['policy_mean_puncture'] for item in per_load])),
            'policy_mean_embb_rate': float(np.mean([item['policy_mean_embb_rate'] for item in per_load])),
            'greedy_mean_reward': float(np.mean([item['greedy_mean_reward'] for item in per_load])),
            'greedy_mean_scheduled_packets': float(np.mean([item['greedy_mean_scheduled_packets'] for item in per_load])),
            'greedy_mean_scheduled_ratio': float(np.mean([item['greedy_mean_scheduled_ratio'] for item in per_load])),
            'greedy_mean_reliability': _mean(per_load, 'greedy_mean_admitted_urllc_reliability', np.nan),
            'greedy_mean_admitted_urllc_reliability': _mean(per_load, 'greedy_mean_admitted_urllc_reliability', np.nan),
            'greedy_mean_effective_urllc_success_over_arrivals': _mean(
                per_load, 'greedy_mean_effective_urllc_success_over_arrivals', 1.0
            ),
            'greedy_mean_empty_admission_case_ratio': _mean(per_load, 'greedy_mean_empty_admission_case_ratio', 0.0),
            'greedy_mean_power': float(np.mean([item['greedy_mean_power'] for item in per_load])),
            'greedy_mean_overlay': float(np.mean([item['greedy_mean_overlay'] for item in per_load])),
            'greedy_mean_puncture': float(np.mean([item['greedy_mean_puncture'] for item in per_load])),
            'greedy_mean_embb_rate': float(np.mean([item['greedy_mean_embb_rate'] for item in per_load])),
            'greedy_noop_selected_ratio': float(np.mean([item.get('greedy_noop_selected_ratio', 0.0) for item in per_load])),
            'greedy_admit_selected_ratio': float(np.mean([item.get('greedy_admit_selected_ratio', 0.0) for item in per_load])),
            'greedy_overlay_ratio': float(np.mean([item.get('greedy_overlay_ratio', 0.0) for item in per_load])),
            'greedy_puncture_ratio': float(np.mean([item.get('greedy_puncture_ratio', 0.0) for item in per_load])),
            'greedy_avg_embb_retention': float(np.mean([item.get('greedy_avg_embb_retention', 0.0) for item in per_load])),
            'greedy_avg_embb_loss': float(np.mean([item.get('greedy_avg_embb_loss', 0.0) for item in per_load])),
            'greedy_avg_selected_throughput': float(
                np.mean([item.get('greedy_avg_selected_throughput', 0.0) for item in per_load])
            ),
            'greedy_avg_rejected_urllc_when_noop_better': float(
                np.mean([item.get('greedy_avg_rejected_urllc_when_noop_better', 0.0) for item in per_load])
            ),
            'greedy_noop_available_ratio': float(
                np.mean([item.get('greedy_noop_available_ratio', 0.0) for item in per_load])
            ),
            'greedy_noop_better_ratio': float(
                np.mean([item.get('greedy_noop_better_ratio', 0.0) for item in per_load])
            ),
            'greedy_requires_feasible_admission_only': float(
                np.max([item.get('greedy_requires_feasible_admission_only', 0.0) for item in per_load])
            ),
            'mean_rate_ratio': mean_rate_ratio,
            'min_rate_ratio': min_rate_ratio,
            'mean_admission_gap': mean_admission_gap,
            'mean_power_ratio': mean_power_ratio,
            'gain_fraction': gain_fraction,
            'gain10_fraction': gain10_fraction,
            'non_worse_fraction': non_worse_fraction,
            'policy_score': selection_score,
            'weighted_selection_score': selection_score,
            'loadwise_selection_score': loadwise_selection_score,
            'loadwise_admission_floor': loadwise_admission_floor,
            'loadwise_floor_pass': loadwise_floor_pass,
            'loadwise_floor_violation': loadwise_floor_violation,
            'loadwise_puncture_loss_gap': loadwise_puncture_loss_gap,
            'loadwise_overlay_retention_gap': loadwise_overlay_retention_gap,
            'loadwise_power_ratio_ceiling': loadwise_power_ratio_ceiling,
            'loadwise_power_ceiling_pass': loadwise_power_ceiling_pass,
            'loadwise_power_ceiling_violation': loadwise_power_ceiling_violation,
            'weighted_floor_violation': weighted_floor_violation,
            'weighted_power_ceiling_violation': weighted_power_ceiling_violation,
            'all_loads_pass_admission_floor': all_loads_pass_admission_floor,
            'all_loads_pass_power_ceiling': all_loads_pass_power_ceiling,
            'all_loads_pass_selection_constraints': float(
                all_loads_pass_floor >= 1.0 - 1e-9
                and all_loads_pass_power_ceiling >= 1.0 - 1e-9
            ),
            'greedy_score': 0.0,
            'score_margin': float(selection_score),
            'non_worse_than_greedy': float(non_worse_fraction >= cfg.training.required_non_worse_fraction),
            'per_load': per_load,
            'eval_loads': [float(load) for load in eval_loads],
            'greedy_baseline': greedy_mode,
            'comparison_baseline_key': greedy_mode,
            'comparison_baseline_label': _baseline_label(greedy_mode),
            'selection_admission_floor_ratio_to_baseline': _selection_floor_ratio_to_baseline(cfg),
            'selection_basis': ('low_damage_weighted_score' if bool(getattr(cfg.training, 'low_damage_admission_objective', False)) else 'load_aware_weighted_score') if bool(getattr(cfg.training, 'load_aware_objective', False)) else 'throughput_heavy_comparison_score',
        }
        summary.update(_baseline_metadata(greedy_mode))
        summary.update(
            _baseline_narrative(
                greedy_mode,
                greedy_requires_feasible_admission_only=_mean(per_load, 'greedy_requires_feasible_admission_only', 0.0) > 0.5,
            )
        )
        summary.update(_multiload_frontier_metrics(summary, cfg))
        _eval_log(
            cfg,
            f"evaluate_against_greedy baseline={greedy_mode} total_sec={perf_counter() - eval_start:.3f}",
        )
        return summary
    finally:
        env.rl_cfg.env.include_greedy_reference_in_obs = previous_greedy_obs


def rescreen_checkpoints_for_report(
    cfg: SRMAPPOConfig | None = None,
    coarse_eval_episodes: int = 4,
    final_eval_episodes: int = 6,
    shortlist_size: int = 6,
) -> Dict[str, object]:
    from .trainer import build_default_components

    cfg = cfg or SRMAPPOConfig()
    base_cfg = deepcopy(cfg)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = []
    for path in sorted(checkpoint_dir.glob(f"{cfg.training.run_name}_iter*.pt")):
        try:
            iteration = int(path.stem.rsplit("iter", 1)[1])
        except (IndexError, ValueError):
            continue
        if iteration % max(int(cfg.training.checkpoint_every), 1) == 0:
            candidate_paths.append(path)
    for extra_name in [
        f"{cfg.training.run_name}_best.pt",
        f"{cfg.training.run_name}_best_reward.pt",
        f"{cfg.training.run_name}_best_throughput.pt",
        f"{cfg.training.run_name}_best_multiload_tp_power.pt",
        f"{cfg.training.run_name}_best_multiload_frontier.pt",
        f"{cfg.training.run_name}_best_vs_original_greedy.pt",
        f"{cfg.training.run_name}_best_vs_matched_greedy.pt",
        f"{cfg.training.run_name}_best_vs_throughput_feasible_oracle.pt",
        f"{cfg.training.run_name}_best_vs_throughput_only_greedy.pt",
        f"{cfg.training.run_name}_best_vs_channel_only_greedy.pt",
        f"{cfg.training.run_name}_final.pt",
    ]:
        extra_path = checkpoint_dir / extra_name
        if extra_path.exists():
            candidate_paths.append(extra_path)

    # Deduplicate while preserving order.
    seen = set()
    candidate_paths = [path for path in candidate_paths if not (path.name in seen or seen.add(path.name))]
    if not candidate_paths:
        raise FileNotFoundError(f"No checkpoints found for {cfg.training.run_name} in {checkpoint_dir}")

    env, model, trainer = build_default_components(base_cfg)

    def _reward_score(summary: Dict[str, float]) -> tuple:
        return (
            float(summary.get("policy_score", 0.0)),
            float(summary.get("policy_mean_embb_rate", 0.0)),
            -float(summary.get("policy_mean_unscheduled_ratio", 0.0)),
            -float(summary.get("policy_mean_embb_min_rate_shortfall", 0.0)),
        )

    def _load_aware_score(summary: Dict[str, float]) -> tuple:
        return (
            float(summary.get("all_loads_pass_selection_constraints", summary.get("all_loads_pass_admission_floor", 0.0))),
            -float(summary.get("weighted_floor_violation", 0.0) + summary.get("weighted_power_ceiling_violation", 0.0)),
            float(summary.get("weighted_selection_score", summary.get("policy_score", 0.0))),
            float(summary.get("policy_mean_embb_rate", 0.0)),
        )

    def _throughput_score(summary: Dict[str, float]) -> tuple:
        return (
            float(summary.get("policy_throughput_score", 0.0)),
            float(summary.get("policy_score", 0.0)),
            float(summary.get("policy_mean_scheduled_ratio", 0.0)),
            -float(summary.get("policy_mean_unscheduled_ratio", 0.0)),
        )

    def _floor_throughput_score(summary: Dict[str, float]) -> tuple:
        return (
            float(summary.get("all_loads_pass_selection_constraints", summary.get("all_loads_pass_admission_floor", 0.0))),
            -float(summary.get("weighted_floor_violation", 0.0)),
            -float(summary.get("weighted_power_ceiling_violation", 0.0)),
            float(summary.get("policy_throughput_score", 0.0)),
            float(summary.get("policy_score", 0.0)),
            float(summary.get("policy_mean_scheduled_ratio", 0.0)),
            -float(summary.get("policy_mean_unscheduled_ratio", 0.0)),
        )

    def _multiload_frontier_score(summary: Dict[str, float]) -> tuple:
        return (
            float(summary.get("multiload_frontier_all_loads_pass_constraints", 0.0)),
            float(summary.get("multiload_frontier_score", 0.0)),
            float(summary.get("multiload_frontier_weighted_throughput_ratio", 0.0)),
            -float(summary.get("multiload_frontier_weighted_power_ratio", 1.0)),
            float(summary.get("multiload_frontier_weighted_capped_admission_ratio", 0.0)),
            float(summary.get("policy_throughput_score", 0.0)),
        )

    def _passes_admission_floor(summary: Dict[str, float]) -> bool:
        puncture_ratio_ceiling = _selection_puncture_ratio_ceiling(cfg)
        has_loadwise_constraints = bool(
            dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {})
            or dict(getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}) or {})
            or dict(getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}) or {})
            or dict(getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}) or {})
            or dict(getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}) or {})
            or _selection_reliability_floor(cfg) > 0.0
            or _selection_floor_ratio_to_baseline(cfg) > 0.0
            or puncture_ratio_ceiling < 1.0 - 1e-9
        )
        if bool(getattr(cfg.training, "low_damage_admission_objective", False)):
            return bool(summary.get("all_loads_pass_selection_constraints", 0.0) >= 1.0)
        if bool(getattr(cfg.training, "load_aware_objective", False)) or has_loadwise_constraints:
            return bool(summary.get("all_loads_pass_selection_constraints", summary.get("all_loads_pass_admission_floor", 0.0)) >= 1.0)
        floor = float(getattr(cfg.training, "selection_admission_floor", 0.0) or 0.0)
        if floor <= 0.0:
            return True
        return float(summary.get("policy_mean_scheduled_ratio", 0.0)) >= floor - 1e-9

    coarse_cfg = deepcopy(base_cfg)
    coarse_cfg.training.eval_episodes_per_load = int(coarse_eval_episodes)
    coarse_results: List[Dict[str, object]] = []
    for path in candidate_paths:
        extra = trainer.load_checkpoint(path)
        summary = evaluate_dual_selection(env, model, coarse_cfg, force_full_compare=True)
        coarse_results.append(
            {
                "checkpoint": path.name,
                "iteration": extra.get("iteration"),
                "summary": summary,
                "reward_score_tuple": _reward_score(summary),
                "throughput_score_tuple": _throughput_score(summary),
                "multiload_frontier_score_tuple": _multiload_frontier_score(summary),
                "vs_original_score_tuple": (
                    float(summary.get("policy_score_vs_original_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_original_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_matched_score_tuple": (
                    float(summary.get("policy_score_vs_matched_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_matched_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_throughput_feasible_score_tuple": (
                    float(summary.get("policy_score_vs_throughput_feasible_oracle", 0.0)),
                    float(summary.get("policy_throughput_vs_throughput_feasible_oracle", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_throughput_only_score_tuple": (
                    float(summary.get("policy_score_vs_throughput_only_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_throughput_only_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_channel_only_score_tuple": (
                    float(summary.get("policy_score_vs_channel_only_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_channel_only_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
            }
        )

    if bool(getattr(cfg.training, "load_aware_objective", False)):
        coarse_results.sort(key=lambda item: _load_aware_score(item["summary"]), reverse=True)
    else:
        coarse_results.sort(key=lambda item: item["throughput_score_tuple"], reverse=True)
    shortlisted = coarse_results[: max(1, int(shortlist_size))]

    final_cfg = deepcopy(base_cfg)
    final_cfg.training.eval_episodes_per_load = int(final_eval_episodes)
    final_results: List[Dict[str, object]] = []
    for item in shortlisted:
        path = checkpoint_dir / item["checkpoint"]
        extra = trainer.load_checkpoint(path)
        summary = evaluate_dual_selection(env, model, final_cfg, force_full_compare=True)
        final_results.append(
            {
                "checkpoint": path.name,
                "iteration": extra.get("iteration"),
                "summary": summary,
                "reward_score_tuple": _reward_score(summary),
                "throughput_score_tuple": _throughput_score(summary),
                "balanced_score_tuple": _balanced_rescreen_score(summary, cfg),
                "multiload_frontier_score_tuple": _multiload_frontier_score(summary),
                "vs_original_score_tuple": (
                    float(summary.get("policy_score_vs_original_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_original_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_matched_score_tuple": (
                    float(summary.get("policy_score_vs_matched_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_matched_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_throughput_feasible_score_tuple": (
                    float(summary.get("policy_score_vs_throughput_feasible_oracle", 0.0)),
                    float(summary.get("policy_throughput_vs_throughput_feasible_oracle", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_throughput_only_score_tuple": (
                    float(summary.get("policy_score_vs_throughput_only_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_throughput_only_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
                "vs_channel_only_score_tuple": (
                    float(summary.get("policy_score_vs_channel_only_greedy", 0.0)),
                    float(summary.get("policy_throughput_vs_channel_only_greedy", 0.0)),
                    float(summary.get("policy_mean_scheduled_ratio", 0.0)),
                ),
            }
        )

    reward_sorted = sorted(final_results, key=lambda item: item["reward_score_tuple"], reverse=True)
    throughput_sorted = sorted(final_results, key=lambda item: item["throughput_score_tuple"], reverse=True)
    balanced_sorted = sorted(final_results, key=lambda item: item["balanced_score_tuple"], reverse=True)
    floor_throughput_candidates = [item for item in final_results if _passes_admission_floor(item["summary"])]
    if bool(getattr(cfg.training, "load_aware_objective", False)):
        floor_throughput_sorted = sorted(floor_throughput_candidates, key=lambda item: _load_aware_score(item["summary"]), reverse=True)
        floor_fallback_sorted = sorted(final_results, key=lambda item: _load_aware_score(item["summary"]), reverse=True)
    else:
        floor_throughput_sorted = sorted(floor_throughput_candidates, key=lambda item: _floor_throughput_score(item["summary"]), reverse=True)
        floor_fallback_sorted = sorted(final_results, key=lambda item: _floor_throughput_score(item["summary"]), reverse=True)
    vs_original_sorted = sorted(final_results, key=lambda item: item["vs_original_score_tuple"], reverse=True)
    vs_matched_sorted = sorted(final_results, key=lambda item: item["vs_matched_score_tuple"], reverse=True)
    vs_throughput_feasible_sorted = sorted(final_results, key=lambda item: item["vs_throughput_feasible_score_tuple"], reverse=True)
    vs_throughput_only_sorted = sorted(final_results, key=lambda item: item["vs_throughput_only_score_tuple"], reverse=True)
    vs_channel_only_sorted = sorted(final_results, key=lambda item: item["vs_channel_only_score_tuple"], reverse=True)
    multiload_frontier_sorted = sorted(final_results, key=lambda item: item["multiload_frontier_score_tuple"], reverse=True)
    best_reward = reward_sorted[0]
    best_throughput = throughput_sorted[0]
    best_balanced = balanced_sorted[0]
    best_floor_throughput = floor_throughput_sorted[0] if floor_throughput_sorted else floor_fallback_sorted[0]
    best_multiload_frontier = multiload_frontier_sorted[0]
    best_vs_original = vs_original_sorted[0]
    best_vs_matched = vs_matched_sorted[0]
    best_vs_throughput_feasible = vs_throughput_feasible_sorted[0]
    best_vs_throughput_only = vs_throughput_only_sorted[0]
    best_vs_channel_only = vs_channel_only_sorted[0]
    report_best_reward_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_reward.pt"
    report_best_throughput_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_throughput.pt"
    report_best_balanced_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_balanced.pt"
    report_best_floor_throughput_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_floor_throughput.pt"
    report_best_multiload_tp_power_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_multiload_tp_power.pt"
    report_best_multiload_frontier_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_multiload_frontier.pt"
    report_best_vs_original_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_vs_original_greedy.pt"
    report_best_vs_matched_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_vs_matched_greedy.pt"
    report_best_vs_throughput_feasible_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_vs_throughput_feasible_oracle.pt"
    report_best_vs_throughput_only_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_vs_throughput_only_greedy.pt"
    report_best_vs_channel_only_path = checkpoint_dir / f"{cfg.training.run_name}_report_best_vs_channel_only_greedy.pt"
    report_best_path = checkpoint_dir / f"{cfg.training.run_name}_report_best.pt"
    shutil.copy2(checkpoint_dir / best_reward["checkpoint"], report_best_reward_path)
    shutil.copy2(checkpoint_dir / best_throughput["checkpoint"], report_best_throughput_path)
    shutil.copy2(checkpoint_dir / best_balanced["checkpoint"], report_best_balanced_path)
    shutil.copy2(checkpoint_dir / best_multiload_frontier["checkpoint"], report_best_multiload_tp_power_path)
    shutil.copy2(checkpoint_dir / best_multiload_frontier["checkpoint"], report_best_multiload_frontier_path)
    shutil.copy2(checkpoint_dir / best_vs_original["checkpoint"], report_best_vs_original_path)
    shutil.copy2(checkpoint_dir / best_vs_matched["checkpoint"], report_best_vs_matched_path)
    shutil.copy2(checkpoint_dir / best_vs_throughput_feasible["checkpoint"], report_best_vs_throughput_feasible_path)
    shutil.copy2(checkpoint_dir / best_vs_throughput_only["checkpoint"], report_best_vs_throughput_only_path)
    shutil.copy2(checkpoint_dir / best_vs_channel_only["checkpoint"], report_best_vs_channel_only_path)
    selection_mode = str(getattr(cfg.training, "selection_mode", "dual_metric") or "dual_metric").strip().lower()
    selection_baseline_mode = _selection_baseline_mode(cfg)
    primary_checkpoint_preference = _primary_checkpoint_preference(cfg)
    selection_admission_floor = float(getattr(cfg.training, "selection_admission_floor", 0.0) or 0.0)
    selection_admission_floor_ratio = _selection_floor_ratio_to_baseline(cfg)
    has_loadwise_selection_constraints = bool(
        dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}) or {})
        or _selection_reliability_floor(cfg) > 0.0
        or selection_admission_floor_ratio > 0.0
        or _selection_puncture_ratio_ceiling(cfg) < 1.0 - 1e-9
    )
    has_floor_passing_checkpoint = bool(floor_throughput_candidates)
    if primary_checkpoint_preference == "best_balanced":
        preferred_report_best = best_balanced
        preferred_report_best_reason = "best_balanced"
    elif _checkpoint_eval_scope(cfg) == "all_loads" and primary_checkpoint_preference in {"best_multiload_frontier", "best_multiload_tp_power"}:
        preferred_report_best = best_multiload_frontier
        preferred_report_best_reason = (
            "best_multiload_tp_power"
            if primary_checkpoint_preference == "best_multiload_tp_power"
            else "best_multiload_frontier"
        )
    elif bool(getattr(cfg.training, "load_aware_objective", False)):
        preferred_report_best = best_floor_throughput
        preferred_report_best_reason = "best_floor_throughput"
    elif has_loadwise_selection_constraints or selection_admission_floor > 0.0:
        preferred_report_best = best_floor_throughput
        preferred_report_best_reason = "best_floor_throughput"
    elif selection_mode == "throughput_only":
        preferred_report_best = best_throughput
        preferred_report_best_reason = "best_throughput"
    else:
        preferred_report_best = (
            best_vs_original
            if selection_baseline_mode in {"original", "original_greedy_normal_v1", "original_greedy_normal_v2"}
            else (
                best_vs_matched
                if selection_baseline_mode == "matched_fixed_embb"
                else (
                    best_vs_throughput_feasible
                    if selection_baseline_mode == "throughput_feasible_oracle"
                    else (
                        best_vs_throughput_only
                        if selection_baseline_mode == "throughput_only_greedy"
                        else best_vs_channel_only
                    )
                )
            )
        )
        preferred_report_best_reason = (
            "best_vs_original_greedy"
            if selection_baseline_mode in {"original", "original_greedy_normal_v1", "original_greedy_normal_v2"}
            else (
                "best_vs_matched_greedy"
                if selection_baseline_mode == "matched_fixed_embb"
                else (
                    "best_vs_throughput_feasible_oracle"
                    if selection_baseline_mode == "throughput_feasible_oracle"
                    else (
                        "best_vs_throughput_only_greedy"
                        if selection_baseline_mode == "throughput_only_greedy"
                        else "best_vs_channel_only_greedy"
                    )
                )
            )
        )
    if has_loadwise_selection_constraints or selection_admission_floor > 0.0 or has_floor_passing_checkpoint:
        shutil.copy2(checkpoint_dir / best_floor_throughput["checkpoint"], report_best_floor_throughput_path)
    elif report_best_floor_throughput_path.exists():
        report_best_floor_throughput_path.unlink()
    shutil.copy2(checkpoint_dir / preferred_report_best["checkpoint"], report_best_path)
    primary_checkpoint_match_warning = _primary_checkpoint_match_warning(cfg, preferred_report_best_reason)

    payload = {
        "coarse_eval_episodes": int(coarse_eval_episodes),
        "final_eval_episodes": int(final_eval_episodes),
        "shortlist_size": int(shortlist_size),
        "selected_checkpoint_reward": best_reward["checkpoint"],
        "selected_iteration_reward": best_reward["iteration"],
        "selected_reward_score_tuple": list(best_reward["reward_score_tuple"]),
        "selected_checkpoint_throughput": best_throughput["checkpoint"],
        "selected_iteration_throughput": best_throughput["iteration"],
        "selected_throughput_score_tuple": list(best_throughput["throughput_score_tuple"]),
        "selected_checkpoint_balanced": best_balanced["checkpoint"],
        "selected_iteration_balanced": best_balanced["iteration"],
        "selected_balanced_score_tuple": list(best_balanced["balanced_score_tuple"]),
        "selected_checkpoint_floor_throughput": best_floor_throughput["checkpoint"],
        "selected_iteration_floor_throughput": best_floor_throughput["iteration"],
        "selected_floor_throughput_score_tuple": list(best_floor_throughput["throughput_score_tuple"]),
        "selected_checkpoint_multiload_frontier": best_multiload_frontier["checkpoint"],
        "selected_iteration_multiload_frontier": best_multiload_frontier["iteration"],
        "selected_multiload_frontier_score_tuple": list(best_multiload_frontier["multiload_frontier_score_tuple"]),
        "selected_checkpoint_multiload_tp_power": best_multiload_frontier["checkpoint"],
        "selected_iteration_multiload_tp_power": best_multiload_frontier["iteration"],
        "selected_multiload_tp_power_score_tuple": list(best_multiload_frontier["multiload_frontier_score_tuple"]),
        "selection_admission_floor": selection_admission_floor,
        "selection_admission_floor_ratio_to_baseline": selection_admission_floor_ratio,
        "selection_admission_floor_by_load": {
            str(key): float(value)
            for key, value in dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {}).items()
        },
        "checkpoint_eval_scope": _checkpoint_eval_scope(cfg),
        "checkpoint_eval_loads": [float(load) for load in _checkpoint_eval_loads(cfg)],
        "checkpoint_eval_episodes_per_load": _checkpoint_eval_episodes_per_load(cfg),
        "primary_checkpoint_preference": primary_checkpoint_preference,
        "require_primary_checkpoint_match": _require_primary_checkpoint_match(cfg),
        "primary_checkpoint_match_warning": primary_checkpoint_match_warning,
        "selected_checkpoint_report_best": preferred_report_best["checkpoint"],
        "selected_iteration_report_best": preferred_report_best["iteration"],
        "selected_report_best_reason": preferred_report_best_reason,
        "selection_score_weights_by_load": {
            str(key): float(value)
            for key, value in dict(getattr(cfg.training, "selection_score_weights_by_load", {}) or {}).items()
        },
        "selection_throughput_ratio_floor_by_load": {
            str(key): float(value)
            for key, value in dict(getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}) or {}).items()
        },
        "selection_reliability_floor": float(getattr(cfg.training, "selection_reliability_floor", 0.0) or 0.0),
        "selection_puncture_ratio_floor_by_load": {
            str(key): float(value)
            for key, value in dict(getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}) or {}).items()
        },
        "selection_overlay_ratio_ceiling_by_load": {
            str(key): float(value)
            for key, value in dict(getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}) or {}).items()
        },
        "selected_weighted_selection_score": float(best_floor_throughput["summary"].get("weighted_selection_score", best_floor_throughput["summary"].get("policy_score", 0.0))),
        "selected_weighted_floor_violation": float(best_floor_throughput["summary"].get("weighted_floor_violation", 0.0)),
        "selected_weighted_power_ceiling_violation": float(best_floor_throughput["summary"].get("weighted_power_ceiling_violation", 0.0)),
        "selected_loadwise_selection_score": list(best_floor_throughput["summary"].get("loadwise_selection_score", [])),
        "selected_loadwise_floor_pass": list(best_floor_throughput["summary"].get("loadwise_floor_pass", [])),
        "has_floor_passing_checkpoint": bool(has_floor_passing_checkpoint),
        "selected_checkpoint_vs_original": best_vs_original["checkpoint"],
        "selected_iteration_vs_original": best_vs_original["iteration"],
        "selected_vs_original_score_tuple": list(best_vs_original["vs_original_score_tuple"]),
        "selected_checkpoint_vs_matched": best_vs_matched["checkpoint"],
        "selected_iteration_vs_matched": best_vs_matched["iteration"],
        "selected_vs_matched_score_tuple": list(best_vs_matched["vs_matched_score_tuple"]),
        "selected_checkpoint_vs_throughput_feasible": best_vs_throughput_feasible["checkpoint"],
        "selected_iteration_vs_throughput_feasible": best_vs_throughput_feasible["iteration"],
        "selected_vs_throughput_feasible_score_tuple": list(best_vs_throughput_feasible["vs_throughput_feasible_score_tuple"]),
        "selected_checkpoint_vs_throughput_only": best_vs_throughput_only["checkpoint"],
        "selected_iteration_vs_throughput_only": best_vs_throughput_only["iteration"],
        "selected_vs_throughput_only_score_tuple": list(best_vs_throughput_only["vs_throughput_only_score_tuple"]),
        "selected_checkpoint_vs_channel_only": best_vs_channel_only["checkpoint"],
        "selected_iteration_vs_channel_only": best_vs_channel_only["iteration"],
        "selected_vs_channel_only_score_tuple": list(best_vs_channel_only["vs_channel_only_score_tuple"]),
        "selection_basis_reward": "policy_score",
        "selection_basis_throughput": "policy_throughput_score",
        "selection_basis_multiload_frontier": "multiload_frontier_score",
        "selection_basis_multiload_tp_power": "multiload_tp_power_score",
        "selection_basis_vs_original": "policy_score_vs_original_greedy",
        "selection_basis_vs_matched": "policy_score_vs_matched_greedy",
        "selection_basis_vs_throughput_feasible": "policy_score_vs_throughput_feasible_oracle",
        "selection_basis_vs_throughput_only": "policy_score_vs_throughput_only_greedy",
        "selection_basis_vs_channel_only": "policy_score_vs_channel_only_greedy",
        "coarse_results": [
            {
                "checkpoint": item["checkpoint"],
                "iteration": item["iteration"],
                "reward_score_tuple": list(item["reward_score_tuple"]),
                "throughput_score_tuple": list(item["throughput_score_tuple"]),
                "multiload_frontier_score_tuple": list(item["multiload_frontier_score_tuple"]),
                "vs_original_score_tuple": list(item["vs_original_score_tuple"]),
                "vs_matched_score_tuple": list(item["vs_matched_score_tuple"]),
                "vs_throughput_feasible_score_tuple": list(item["vs_throughput_feasible_score_tuple"]),
                "vs_throughput_only_score_tuple": list(item["vs_throughput_only_score_tuple"]),
                "vs_channel_only_score_tuple": list(item["vs_channel_only_score_tuple"]),
                "summary": item["summary"],
            }
            for item in coarse_results
        ],
        "final_results": [
            {
                "checkpoint": item["checkpoint"],
                "iteration": item["iteration"],
                "reward_score_tuple": list(item["reward_score_tuple"]),
                "throughput_score_tuple": list(item["throughput_score_tuple"]),
                "multiload_frontier_score_tuple": list(item["multiload_frontier_score_tuple"]),
                "vs_original_score_tuple": list(item["vs_original_score_tuple"]),
                "vs_matched_score_tuple": list(item["vs_matched_score_tuple"]),
                "vs_throughput_feasible_score_tuple": list(item["vs_throughput_feasible_score_tuple"]),
                "vs_throughput_only_score_tuple": list(item["vs_throughput_only_score_tuple"]),
                "vs_channel_only_score_tuple": list(item["vs_channel_only_score_tuple"]),
                "summary": item["summary"],
            }
            for item in final_results
        ],
        "report_best_reward_path": str(report_best_reward_path),
        "report_best_throughput_path": str(report_best_throughput_path),
        "report_best_vs_original_path": str(report_best_vs_original_path),
        "report_best_vs_matched_path": str(report_best_vs_matched_path),
        "report_best_vs_throughput_feasible_path": str(report_best_vs_throughput_feasible_path),
        "report_best_vs_throughput_only_path": str(report_best_vs_throughput_only_path),
        "report_best_vs_channel_only_path": str(report_best_vs_channel_only_path),
        "report_best_path": str(report_best_path),
    }
    (results_dir / "report_checkpoint_rescreen.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _ensure_default_load(env) -> float:
    return float((env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / env.sys_cfg.num_uavs)
