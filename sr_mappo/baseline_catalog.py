from __future__ import annotations

from typing import Dict


BASELINE_ALIASES = {
    "channel_only": "channel_only_greedy",
    "throughput_only": "throughput_only_greedy",
    "tp_only": "throughput_only_greedy",
    "rate_loss_min": "rate_loss_min_greedy",
    "global_rate_loss": "rate_loss_min_greedy",
    "sumrate_minloss": "rate_loss_min_greedy",
    "global_sumrate_minloss": "rate_loss_min_greedy",
    "force_admit_minloss": "force_admit_minloss_greedy",
    "rate_loss_force_admit": "force_admit_minloss_greedy",
    "sumrate_force_admit": "force_admit_minloss_greedy",
    "myopic": "myopic_throughput_greedy",
    "myopic_tp": "myopic_throughput_greedy",
    "myopic_throughput": "myopic_throughput_greedy",
    "hard_feasible": "hard_feasible_throughput_greedy",
    "hard_feasible_throughput": "hard_feasible_throughput_greedy",
    "hard_feasible_tp": "hard_feasible_throughput_greedy",
    "global_frontier": "global_frontier_greedy",
    "global_greedy": "global_frontier_greedy",
    "throughput_feasible": "throughput_feasible_oracle",
    "coexistence_oracle": "throughput_feasible_oracle",
    "matched": "matched_fixed_embb",
    "matched_oracle": "matched_fixed_embb",
    "throughput_biased": "throughput_biased_greedy",
    "tp_biased": "throughput_biased_greedy",
    "original_greedy_lite": "original_greedy_normal_v1",
    "original_normal": "original_greedy_normal_v1",
    "original_greedy_normal": "original_greedy_normal_v1",
    "original_normal_v2": "original_greedy_normal_v2",
    "original_greedy_normal_v2": "original_greedy_normal_v2",
}

VALID_BASELINE_MODES = {
    "original",
    "original_greedy_normal_v1",
    "original_greedy_normal_v2",
    "myopic_throughput_greedy",
    "hard_feasible_throughput_greedy",
    "global_frontier_greedy",
    "matched_fixed_embb",
    "throughput_biased_greedy",
    "throughput_feasible_oracle",
    "throughput_only_greedy",
    "rate_loss_min_greedy",
    "force_admit_minloss_greedy",
    "channel_only_greedy",
    "frozen_json",
}

BASELINE_LABELS = {
    "original": "Original Greedy",
    "original_greedy_normal_v1": "Original Greedy Normal v1",
    "original_greedy_normal_v2": "Original Greedy Normal v2",
    "myopic_throughput_greedy": "Myopic Throughput-first Greedy (hard-feasible, weak tie-breaks)",
    "hard_feasible_throughput_greedy": "Hard-feasible Throughput Greedy (admit-only; KEEP iff none feasible)",
    "global_frontier_greedy": "Global Frontier Greedy (shared score; no mix-specific shaping)",
    "matched_fixed_embb": "Matched Fixed-Power Throughput Oracle",
    "throughput_biased_greedy": "Throughput-biased Greedy (admission-bounded)",
    "throughput_feasible_oracle": "Throughput-feasible Oracle",
    "throughput_only_greedy": "Throughput-only Greedy (eMBB-only ceiling)",
    "rate_loss_min_greedy": "Rate-loss-min Greedy (pure-sumrate owner + min-loss admit)",
    "force_admit_minloss_greedy": "Force-admit Min-loss Greedy (pure-sumrate owner + no-KEEP min-loss admit)",
    "channel_only_greedy": "Channel-only Greedy",
    "frozen_json": "Frozen Baseline",
}


def normalize_baseline_mode(mode: str | None, default: str = "original") -> str:
    value = str(mode or default).strip().lower()
    value = BASELINE_ALIASES.get(value, value)
    if value not in VALID_BASELINE_MODES:
        return default
    return value


def baseline_label(mode: str | None) -> str:
    normalized = normalize_baseline_mode(mode)
    return BASELINE_LABELS.get(normalized, BASELINE_LABELS["original"])


def baseline_metadata(
    mode: str | None,
    *,
    main_coexistence_mode: str = "myopic_throughput_greedy",
) -> Dict[str, object]:
    normalized = normalize_baseline_mode(mode)
    main_mode = normalize_baseline_mode(main_coexistence_mode)
    metadata = {
        "baseline_key": normalized,
        "comparison_baseline_key": normalized,
        "comparison_baseline_label": baseline_label(normalized),
        "baseline_objective_type": "legacy_greedy",
        "baseline_requires_admission_feasible_set": False,
        "baseline_allows_noop": False,
        "baseline_is_debug_ceiling": False,
        "baseline_is_main_coexistence_reference": normalized == main_mode,
    }
    if normalized == "throughput_only_greedy":
        metadata.update({
            "baseline_objective_type": "embb_only_greedy",
            "baseline_requires_admission_feasible_set": False,
            "baseline_allows_noop": True,
            "baseline_is_debug_ceiling": True,
            "baseline_is_main_coexistence_reference": False,
        })
    elif normalized == "rate_loss_min_greedy":
        metadata.update({
            "baseline_objective_type": "global_rate_loss_min_greedy",
            "baseline_requires_admission_feasible_set": False,
            "baseline_allows_noop": True,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": False,
        })
    elif normalized == "force_admit_minloss_greedy":
        metadata.update({
            "baseline_objective_type": "force_admit_minloss_greedy",
            "baseline_requires_admission_feasible_set": True,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": False,
        })
    elif normalized == "myopic_throughput_greedy":
        metadata.update({
            "baseline_objective_type": "myopic_throughput_greedy",
            "baseline_requires_admission_feasible_set": False,
            "baseline_allows_noop": True,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": normalized == main_mode,
        })
    elif normalized == "hard_feasible_throughput_greedy":
        metadata.update({
            "baseline_objective_type": "hard_feasible_throughput_greedy",
            "baseline_requires_admission_feasible_set": True,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": normalized == main_mode,
        })
    elif normalized == "global_frontier_greedy":
        metadata.update({
            "baseline_objective_type": "global_frontier_greedy",
            "baseline_requires_admission_feasible_set": True,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": normalized == main_mode,
        })
    elif normalized == "throughput_biased_greedy":
        metadata.update({
            "baseline_objective_type": "throughput_biased_greedy",
            "baseline_requires_admission_feasible_set": False,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": normalized == main_mode,
        })
    elif normalized == "throughput_feasible_oracle":
        metadata.update({
            "baseline_objective_type": "throughput_feasible_oracle",
            "baseline_requires_admission_feasible_set": True,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": normalized == main_mode,
        })
    elif normalized == "matched_fixed_embb":
        metadata.update({
            "baseline_objective_type": "matched_fixedpower_oracle",
            "baseline_requires_admission_feasible_set": True,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
            "baseline_is_main_coexistence_reference": True if main_mode == "matched_fixed_embb" else normalized == main_mode,
        })
    elif normalized == "channel_only_greedy":
        metadata.update({
            "baseline_objective_type": "channel_only_greedy",
            "baseline_requires_admission_feasible_set": False,
            "baseline_allows_noop": False,
            "baseline_is_debug_ceiling": False,
        })
    elif normalized == "frozen_json":
        metadata.update({
            "baseline_objective_type": "frozen_baseline",
        })
    return metadata


def baseline_narrative(
    mode: str | None,
    *,
    greedy_requires_feasible_admission_only: bool = False,
) -> Dict[str, str]:
    normalized = normalize_baseline_mode(mode)
    if normalized == "throughput_only_greedy":
        return {
            "greedy_objective": (
                "maximize aggregate eMBB throughput "
                "(equivalently minimize immediate eMBB throughput loss)"
            ),
            "greedy_admission_role": (
                "URLLC admission is not an optimization target; it only appears as a "
                "feasibility consequence of the eMBB-only throughput-maximizing action."
            ),
            "greedy_noop_policy": (
                "current env requires choosing among feasible admission actions only; "
                "the baseline therefore picks the feasible admit action with minimum eMBB damage."
                if bool(greedy_requires_feasible_admission_only)
                else (
                    "MODE_KEEP/no-op is part of the action comparison. "
                    "If reject/no-op preserves more eMBB throughput than every admit action, "
                    "URLLC is rejected."
                )
            ),
        }
    if normalized == "rate_loss_min_greedy":
        return {
            "greedy_objective": (
                "choose the Phase-0 eMBB owner that maximizes global eMBB sum-rate, "
                "then choose the URLLC admit action with the minimum global eMBB throughput loss"
            ),
            "greedy_admission_role": (
                "URLLC admission is evaluated through its global eMBB rate damage only; "
                "min-rate and service-floor shaping are not part of the action score."
            ),
            "greedy_noop_policy": (
                "MODE_KEEP/no-op is part of the comparison set. If every admit action causes "
                "more global eMBB damage than rejecting, KEEP is selected."
            ),
        }
    if normalized == "force_admit_minloss_greedy":
        return {
            "greedy_objective": (
                "choose the Phase-0 eMBB owner that maximizes global eMBB sum-rate, "
                "then force a hard-feasible URLLC admit action with the minimum global eMBB throughput loss"
            ),
            "greedy_admission_role": (
                "URLLC admission is mandatory whenever a hard-feasible packet/mode pair exists; "
                "admit actions are ranked only by their global eMBB throughput damage."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is not part of the comparison set. It is used only as a hard fallback "
                "when no feasible admit action exists."
            ),
        }
    if normalized == "myopic_throughput_greedy":
        return {
            "greedy_objective": (
                "maximize immediate aggregate eMBB throughput (myopic, single-objective)"
            ),
            "greedy_admission_role": (
                "URLLC admission is not an optimization target. Admit actions are only considered "
                "when they are hard-feasible; KEEP/no-op is an explicit candidate and may be chosen "
                "when it preserves more eMBB throughput."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is part of the comparison set. Near-ties in eMBB throughput are resolved "
                "via weak tie-breaks (e.g., lower power / overlay quality / reliability / urgency) without "
                "introducing an admission reward. Additionally, KEEP may be disallowed under hard deadline "
                "pressure (urgency threshold) as a feasibility/sanity constraint."
            ),
        }
    if normalized == "hard_feasible_throughput_greedy":
        return {
            "greedy_objective": (
                "maximize immediate aggregate eMBB throughput over hard-feasible admit actions only"
            ),
            "greedy_admission_role": (
                "URLLC admission is not an optimization target. Admit actions are considered only when they are "
                "hard-feasible (reliability/mode/min-rate/power constraints)."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is not part of the throughput comparison set. It is chosen only as a hard fallback "
                "when there is no feasible admit action."
            ),
        }
    if normalized == "global_frontier_greedy":
        return {
            "greedy_objective": (
                "maximize a single shared frontier score that balances retained eMBB throughput, "
                "URLLC admission pressure, reliability, power efficiency, and per-UAV balance"
            ),
            "greedy_admission_role": (
                "URLLC admission is part of the same global score rather than being handled by "
                "mix-specific shaping or a pure throughput-only rule."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is used only when there is no hard-feasible admit action. Among feasible "
                "admit actions, the baseline selects the highest global frontier score."
            ),
        }
    if normalized == "throughput_biased_greedy":
        return {
            "greedy_objective": (
                "maximize aggregate eMBB throughput with a bounded-admission preference"
            ),
            "greedy_admission_role": (
                "URLLC admission is not the primary optimization target. "
                "It is only constrained to stay in a moderate target band."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is not part of the main objective and is used only "
                "when no feasible admit action exists."
            ),
        }
    if normalized == "throughput_feasible_oracle":
        return {
            "greedy_objective": (
                "maximize aggregate eMBB throughput over the feasible-admit action set"
            ),
            "greedy_admission_role": (
                "URLLC admission is a hard feasibility constraint of the oracle action set, "
                "not a soft reward target."
            ),
            "greedy_noop_policy": (
                "KEEP/no-op is not part of the oracle objective; it is used only as a hard fallback "
                "when no feasible admit action exists."
            ),
        }
    if normalized == "matched_fixed_embb":
        return {
            "greedy_objective": (
                "maximize throughput within the matched fixed-power coexistence reference"
            ),
            "greedy_admission_role": (
                "URLLC admission is part of the matched coexistence reference construction. "
                "This baseline is used as the main coexistence reference rather than an eMBB-only ceiling."
            ),
            "greedy_noop_policy": (
                "Reject-all/no-op is not the intended optimization target of the matched fixed-power reference."
            ),
        }
    if normalized == "channel_only_greedy":
        return {
            "greedy_objective": "rank and serve candidates using the channel-only heuristic.",
            "greedy_admission_role": "URLLC admission follows the legacy channel heuristic.",
            "greedy_noop_policy": "No dedicated no-op ceiling semantics are attached to this legacy baseline.",
        }
    return {
        "greedy_objective": "follow the legacy coexistence greedy/reference policy.",
        "greedy_admission_role": "URLLC admission follows the legacy baseline semantics.",
        "greedy_noop_policy": "No dedicated no-op ceiling semantics are attached to this baseline.",
    }
