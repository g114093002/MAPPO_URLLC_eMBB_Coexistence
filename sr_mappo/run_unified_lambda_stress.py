from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .plot_unified_lambda_reward_breakdown import build_reward_breakdown
from .unified_policy_runner import MIX_PRESETS, run_policy


DEFAULT_POLICIES = [
    "greedy",
    "random_scheduler",
    "pure_puncturing",
    "pure_superposition",
]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _urllc_throughput_mbps(metrics: Dict[str, object]) -> float:
    summary = dict(metrics.get("raw_summary", {}) or {})
    packet_bits = float(summary.get("urllc_packet_bits_mean", 0.0) or 0.0)
    slot_duration_s = float(summary.get("urllc_slot_duration_s", 0.0) or 0.0)
    scheduled_packets = float(metrics.get("scheduled_urllc_packets", 0.0) or 0.0)
    if packet_bits <= 0.0 or slot_duration_s <= 0.0:
        return 0.0
    return float((scheduled_packets * packet_bits) / max(slot_duration_s, 1.0e-9) / 1.0e6)


def _activation_prob_key(value: float) -> str:
    return f"activation_prob_{value:g}"


def _policy_series_from_raw(raw_policy_runs: Dict[str, List[Dict[str, object]]], activation_probs: List[float]) -> Dict[str, List[float]]:
    series = {
        "urllc_activation_prob": [],
        "embb_rate_mbps": [],
        "episode_reward_total_sum": [],
        "episode_reward_mean": [],
        "logged_step_reward_sum": [],
        "logged_terminal_reward_sum": [],
        "reward_residual_unlogged": [],
        "total_power": [],
        "embb_power": [],
        "urllc_power": [],
        "embb_power_share": [],
        "urllc_power_share": [],
        "embb_minrate_satisfied_users": [],
        "embb_minrate_satisfaction_ratio": [],
        "num_feasible_modes_for_selected_pair": [],
        "overlay_feasible_count": [],
        "puncturing_feasible_count": [],
        "both_modes_feasible_count": [],
        "mode_regret_applied_count": [],
        "mode_regret_zero_count": [],
        "mean_mode_regret": [],
        "soft_paired_mode_regret_penalty_mean": [],
        "soft_paired_mode_regret_candidate_count": [],
        "soft_paired_mode_regret_applied_count": [],
        "soft_paired_mode_regret_rate": [],
        "opposite_mode_feasible_count": [],
        "opposite_mode_blocked_by_infeasible_count": [],
        "opposite_mode_blocked_by_embb_violation_count": [],
        "opposite_mode_lower_embb_loss_count": [],
        "mean_selected_embb_loss": [],
        "mean_opposite_embb_loss": [],
        "mean_mode_loss_gap": [],
        "mean_power_gap": [],
        "mean_reliability_gap": [],
        "mean_power_discount": [],
        "mean_reliability_discount": [],
        "mean_selected_overlay_embb_loss": [],
        "mean_selected_puncturing_embb_loss": [],
        "mean_alternative_overlay_embb_loss": [],
        "mean_alternative_puncturing_embb_loss": [],
        "selected_overlay_regretted_count": [],
        "selected_puncturing_regretted_count": [],
        "urllc_admission": [],
        "urllc_tp_mbps": [],
        "urllc_admitted_packets": [],
        "overlay_action_count": [],
        "overlay_action_ratio": [],
        "puncturing_action_count": [],
        "puncturing_action_ratio": [],
        "avg_embb_rate_mbps": [],
        "runtime_sec": [],
    }
    for activation_prob in activation_probs:
        per_seed_runs = list(raw_policy_runs.get(_activation_prob_key(activation_prob), []) or [])
        if not per_seed_runs:
            continue
        mean_embb = float(np.mean([float(item["total_embb_throughput"]) for item in per_seed_runs])) / 1.0e6
        mean_admission = float(np.mean([float(item["urllc_admission_ratio"]) for item in per_seed_runs]))
        mean_urllc_packets = float(np.mean([float(item["scheduled_urllc_packets"]) for item in per_seed_runs]))
        mean_urllc_tp = float(np.mean([_urllc_throughput_mbps(item) for item in per_seed_runs]))
        mean_overlay = float(np.mean([float(item["overlay_action_count"]) for item in per_seed_runs]))
        mean_puncture = float(np.mean([float(item["puncturing_action_count"]) for item in per_seed_runs]))
        mean_avg_embb = float(np.mean([float(item["average_embb_rate"]) for item in per_seed_runs])) / 1.0e6
        mean_runtime = float(np.mean([float(item["runtime"]) for item in per_seed_runs]))
        mean_episode_reward_total_sum = float(np.mean([float(item.get("episode_reward_total_sum", 0.0) or 0.0) for item in per_seed_runs]))
        mean_episode_reward_mean = float(np.mean([float(item.get("episode_reward_mean", 0.0) or 0.0) for item in per_seed_runs]))
        mean_logged_step_reward_sum = float(np.mean([float(item.get("logged_step_reward_sum", 0.0) or 0.0) for item in per_seed_runs]))
        mean_logged_terminal_reward_sum = float(np.mean([float(item.get("logged_terminal_reward_sum", 0.0) or 0.0) for item in per_seed_runs]))
        mean_reward_residual_unlogged = float(np.mean([float(item.get("reward_residual_unlogged", 0.0) or 0.0) for item in per_seed_runs]))

        total_powers: List[float] = []
        embb_powers: List[float] = []
        urllc_powers: List[float] = []
        embb_power_shares: List[float] = []
        urllc_power_shares: List[float] = []
        embb_minrate_satisfied_users: List[float] = []
        embb_minrate_satisfaction_ratios: List[float] = []
        feasible_mode_counts: List[float] = []
        overlay_feasible_counts: List[float] = []
        puncturing_feasible_counts: List[float] = []
        both_modes_feasible_counts: List[float] = []
        mode_regret_applied_counts: List[float] = []
        mode_regret_zero_counts: List[float] = []
        mean_mode_regrets: List[float] = []
        soft_paired_mode_regret_penalty_means: List[float] = []
        soft_paired_mode_regret_candidate_counts: List[float] = []
        soft_paired_mode_regret_applied_counts: List[float] = []
        soft_paired_mode_regret_rates: List[float] = []
        opposite_mode_feasible_counts: List[float] = []
        opposite_mode_blocked_by_infeasible_counts: List[float] = []
        opposite_mode_blocked_by_embb_violation_counts: List[float] = []
        opposite_mode_lower_embb_loss_counts: List[float] = []
        mean_selected_embb_losses: List[float] = []
        mean_opposite_embb_losses: List[float] = []
        mean_mode_loss_gaps: List[float] = []
        mean_power_gaps: List[float] = []
        mean_reliability_gaps: List[float] = []
        mean_power_discounts: List[float] = []
        mean_reliability_discounts: List[float] = []
        mean_selected_overlay_embb_losses: List[float] = []
        mean_selected_puncturing_embb_losses: List[float] = []
        mean_alternative_overlay_embb_losses: List[float] = []
        mean_alternative_puncturing_embb_losses: List[float] = []
        selected_overlay_regretted_counts: List[float] = []
        selected_puncturing_regretted_counts: List[float] = []
        overlay_action_ratios: List[float] = []
        puncturing_action_ratios: List[float] = []
        for item in per_seed_runs:
            summary = dict(item.get("raw_summary", {}) or {})
            total_power = float(summary.get("total_power", 0.0) or 0.0)
            embb_power = float(summary.get("embb_power", 0.0) or 0.0)
            urllc_power = float(summary.get("urllc_power", total_power - embb_power) or 0.0)
            total_powers.append(total_power)
            embb_powers.append(embb_power)
            urllc_powers.append(urllc_power)
            if total_power > 1.0e-9:
                embb_power_shares.append(float(embb_power / total_power))
                urllc_power_shares.append(float(urllc_power / total_power))
            else:
                embb_power_shares.append(0.0)
                urllc_power_shares.append(0.0)

            minrate_ratio = float(summary.get("embb_min_rate_satisfaction_ratio", 0.0) or 0.0)
            embb_user_count = float(summary.get("embb_user_count", 0.0) or 0.0)
            embb_minrate_satisfaction_ratios.append(minrate_ratio)
            embb_minrate_satisfied_users.append(float(int(round(minrate_ratio * embb_user_count))) if embb_user_count > 0.0 else 0.0)
            feasible_mode_counts.append(float(summary.get("num_feasible_modes_for_selected_pair", 0.0) or 0.0))
            overlay_feasible_counts.append(float(summary.get("overlay_feasible_count", 0.0) or 0.0))
            puncturing_feasible_counts.append(float(summary.get("puncturing_feasible_count", 0.0) or 0.0))
            both_modes_feasible_counts.append(float(summary.get("both_modes_feasible_count", 0.0) or 0.0))
            mode_regret_applied_counts.append(float(summary.get("mode_regret_applied_count", 0.0) or 0.0))
            mode_regret_zero_counts.append(float(summary.get("mode_regret_zero_count", 0.0) or 0.0))
            mean_mode_regrets.append(float(summary.get("mean_mode_regret", 0.0) or 0.0))
            soft_paired_mode_regret_penalty_means.append(float(summary.get("soft_paired_mode_regret_penalty_mean", 0.0) or 0.0))
            soft_paired_mode_regret_candidate_counts.append(float(summary.get("soft_paired_mode_regret_candidate_count", 0.0) or 0.0))
            soft_paired_mode_regret_applied_counts.append(float(summary.get("soft_paired_mode_regret_applied_count", 0.0) or 0.0))
            soft_paired_mode_regret_rates.append(float(summary.get("soft_paired_mode_regret_rate", 0.0) or 0.0))
            opposite_mode_feasible_counts.append(float(summary.get("opposite_mode_feasible_count", 0.0) or 0.0))
            opposite_mode_blocked_by_infeasible_counts.append(float(summary.get("opposite_mode_blocked_by_infeasible_count", 0.0) or 0.0))
            opposite_mode_blocked_by_embb_violation_counts.append(float(summary.get("opposite_mode_blocked_by_embb_violation_count", 0.0) or 0.0))
            opposite_mode_lower_embb_loss_counts.append(float(summary.get("opposite_mode_lower_embb_loss_count", 0.0) or 0.0))
            mean_selected_embb_losses.append(float(summary.get("mean_selected_embb_loss", 0.0) or 0.0))
            mean_opposite_embb_losses.append(float(summary.get("mean_opposite_embb_loss", 0.0) or 0.0))
            mean_mode_loss_gaps.append(float(summary.get("mean_mode_loss_gap", 0.0) or 0.0))
            mean_power_gaps.append(float(summary.get("mean_power_gap", 0.0) or 0.0))
            mean_reliability_gaps.append(float(summary.get("mean_reliability_gap", 0.0) or 0.0))
            mean_power_discounts.append(float(summary.get("mean_power_discount", 0.0) or 0.0))
            mean_reliability_discounts.append(float(summary.get("mean_reliability_discount", 0.0) or 0.0))
            mean_selected_overlay_embb_losses.append(float(summary.get("mean_selected_overlay_embb_loss", 0.0) or 0.0))
            mean_selected_puncturing_embb_losses.append(float(summary.get("mean_selected_puncturing_embb_loss", 0.0) or 0.0))
            mean_alternative_overlay_embb_losses.append(float(summary.get("mean_alternative_overlay_embb_loss", 0.0) or 0.0))
            mean_alternative_puncturing_embb_losses.append(float(summary.get("mean_alternative_puncturing_embb_loss", 0.0) or 0.0))
            selected_overlay_regretted_counts.append(float(summary.get("selected_overlay_regretted_count", 0.0) or 0.0))
            selected_puncturing_regretted_counts.append(float(summary.get("selected_puncturing_regretted_count", 0.0) or 0.0))

            action_total = float(item["overlay_action_count"]) + float(item["puncturing_action_count"])
            if action_total > 1.0e-9:
                overlay_action_ratios.append(float(float(item["overlay_action_count"]) / action_total))
                puncturing_action_ratios.append(float(float(item["puncturing_action_count"]) / action_total))
            else:
                overlay_action_ratios.append(0.0)
                puncturing_action_ratios.append(0.0)

        series["urllc_activation_prob"].append(float(activation_prob))
        series["embb_rate_mbps"].append(mean_embb)
        series["episode_reward_total_sum"].append(mean_episode_reward_total_sum)
        series["episode_reward_mean"].append(mean_episode_reward_mean)
        series["logged_step_reward_sum"].append(mean_logged_step_reward_sum)
        series["logged_terminal_reward_sum"].append(mean_logged_terminal_reward_sum)
        series["reward_residual_unlogged"].append(mean_reward_residual_unlogged)
        series["total_power"].append(float(np.mean(total_powers)))
        series["embb_power"].append(float(np.mean(embb_powers)))
        series["urllc_power"].append(float(np.mean(urllc_powers)))
        series["embb_power_share"].append(float(np.mean(embb_power_shares)))
        series["urllc_power_share"].append(float(np.mean(urllc_power_shares)))
        series["embb_minrate_satisfied_users"].append(float(np.mean(embb_minrate_satisfied_users)))
        series["embb_minrate_satisfaction_ratio"].append(float(np.mean(embb_minrate_satisfaction_ratios)))
        series["num_feasible_modes_for_selected_pair"].append(float(np.mean(feasible_mode_counts)))
        series["overlay_feasible_count"].append(float(np.mean(overlay_feasible_counts)))
        series["puncturing_feasible_count"].append(float(np.mean(puncturing_feasible_counts)))
        series["both_modes_feasible_count"].append(float(np.mean(both_modes_feasible_counts)))
        series["mode_regret_applied_count"].append(float(np.mean(mode_regret_applied_counts)))
        series["mode_regret_zero_count"].append(float(np.mean(mode_regret_zero_counts)))
        series["mean_mode_regret"].append(float(np.mean(mean_mode_regrets)))
        series["soft_paired_mode_regret_penalty_mean"].append(float(np.mean(soft_paired_mode_regret_penalty_means)))
        series["soft_paired_mode_regret_candidate_count"].append(float(np.mean(soft_paired_mode_regret_candidate_counts)))
        series["soft_paired_mode_regret_applied_count"].append(float(np.mean(soft_paired_mode_regret_applied_counts)))
        series["soft_paired_mode_regret_rate"].append(float(np.mean(soft_paired_mode_regret_rates)))
        series["opposite_mode_feasible_count"].append(float(np.mean(opposite_mode_feasible_counts)))
        series["opposite_mode_blocked_by_infeasible_count"].append(float(np.mean(opposite_mode_blocked_by_infeasible_counts)))
        series["opposite_mode_blocked_by_embb_violation_count"].append(float(np.mean(opposite_mode_blocked_by_embb_violation_counts)))
        series["opposite_mode_lower_embb_loss_count"].append(float(np.mean(opposite_mode_lower_embb_loss_counts)))
        series["mean_selected_embb_loss"].append(float(np.mean(mean_selected_embb_losses)))
        series["mean_opposite_embb_loss"].append(float(np.mean(mean_opposite_embb_losses)))
        series["mean_mode_loss_gap"].append(float(np.mean(mean_mode_loss_gaps)))
        series["mean_power_gap"].append(float(np.mean(mean_power_gaps)))
        series["mean_reliability_gap"].append(float(np.mean(mean_reliability_gaps)))
        series["mean_power_discount"].append(float(np.mean(mean_power_discounts)))
        series["mean_reliability_discount"].append(float(np.mean(mean_reliability_discounts)))
        series["mean_selected_overlay_embb_loss"].append(float(np.mean(mean_selected_overlay_embb_losses)))
        series["mean_selected_puncturing_embb_loss"].append(float(np.mean(mean_selected_puncturing_embb_losses)))
        series["mean_alternative_overlay_embb_loss"].append(float(np.mean(mean_alternative_overlay_embb_losses)))
        series["mean_alternative_puncturing_embb_loss"].append(float(np.mean(mean_alternative_puncturing_embb_losses)))
        series["selected_overlay_regretted_count"].append(
            float(np.mean(selected_overlay_regretted_counts))
        )
        series["selected_puncturing_regretted_count"].append(
            float(np.mean(selected_puncturing_regretted_counts))
        )
        series["urllc_admission"].append(mean_admission)
        series["urllc_tp_mbps"].append(mean_urllc_tp)
        series["urllc_admitted_packets"].append(mean_urllc_packets)
        series["overlay_action_count"].append(mean_overlay)
        series["overlay_action_ratio"].append(float(np.mean(overlay_action_ratios)))
        series["puncturing_action_count"].append(mean_puncture)
        series["puncturing_action_ratio"].append(float(np.mean(puncturing_action_ratios)))
        series["avg_embb_rate_mbps"].append(mean_avg_embb)
        series["runtime_sec"].append(mean_runtime)
    return series


def _write_progress(payload: Dict[str, object], out_dir: Path) -> None:
    _plot_metric_grid(payload.get("series", {}), out_dir)
    json_path = out_dir / "unified_lambda_stress_metrics.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(payload), handle, indent=2)


def _meta_compatible(existing: Dict[str, object], expected_meta: Dict[str, object]) -> bool:
    meta = dict(existing.get("meta", {}) or {})
    return (
        list(meta.get("mixes", [])) == list(expected_meta.get("mixes", []))
        and list(meta.get("loads", [])) == list(expected_meta.get("loads", []))
        and list(meta.get("activation_probs", meta.get("lambdas", []))) == list(expected_meta.get("activation_probs", []))
        and list(meta.get("seeds", [])) == list(expected_meta.get("seeds", []))
        and int(meta.get("episodes_per_seed", 1) or 1) == int(expected_meta.get("episodes_per_seed", 1) or 1)
    )


def _build_policy_config(
    args,
    *,
    policy: str,
    total_load: float,
    mix_ratio: float,
    activation_prob: float,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "total_load": float(total_load),
        "mix_ratio": float(mix_ratio),
        "simulation": {
            "fixed_urllc_poisson_rate": True,
            "urllc_poisson_rate": float(activation_prob),
            "urllc_user_ratio": float(mix_ratio),
        },
        "env": {
            "urllc_poisson_rate_is_per_user": True,
            "urllc_poisson_rate_is_slot_level": False,
            "urllc_arrival_mode": "bernoulli",
            "urllc_bernoulli_tx_prob": 0.5,
        },
    }
    if args.system_num_subcarriers is not None or args.system_num_minislots is not None:
        cfg["system"] = {}
        if args.system_num_subcarriers is not None:
            cfg["system"]["num_subcarriers"] = int(args.system_num_subcarriers)
        if args.system_num_minislots is not None:
            cfg["system"]["num_minislots"] = int(args.system_num_minislots)
    if policy in {"mappo", "mappo_overlay_forced", "mappo_puncture_forced"}:
        if not args.mappo_checkpoint_path:
            raise ValueError(f"Policy '{policy}' requires --mappo-checkpoint-path.")
        cfg["checkpoint_path"] = str(Path(args.mappo_checkpoint_path).expanduser())
    if policy == "ippo":
        if args.ippo_checkpoint_path:
            cfg["checkpoint_path"] = str(Path(args.ippo_checkpoint_path).expanduser())
            cfg["train"] = False
        else:
            cfg["train"] = True
            cfg["train_iterations"] = int(args.ippo_train_iterations)
        cfg["reward_scope"] = str(args.ippo_reward_scope)
    if policy == "pure_puncturing":
        cfg["puncturing_selection_rule"] = str(args.puncturing_selection_rule)
    if policy == "pure_superposition":
        cfg["superposition_selection_rule"] = str(args.superposition_selection_rule)
    if args.reward_from_checkpoint_path:
        cfg["reward_checkpoint_path"] = str(Path(args.reward_from_checkpoint_path).expanduser())
    return cfg


def _plot_metric_grid(series: Dict[str, Dict[str, Dict[str, List[float]]]], out_dir: Path) -> None:
    metric_specs = [
        ("embb_rate_mbps", "Aggregate eMBB throughput", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
    ]
    for mix_name, policy_series in series.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = np.asarray(axes).ravel()
        x_values = np.asarray(next(iter(policy_series.values())).get("urllc_activation_prob", []), dtype=float) if policy_series else np.asarray([], dtype=float)
        for ax, (metric_key, title, ylabel) in zip(axes, metric_specs):
            for policy_name, data in policy_series.items():
                y_values = np.asarray(data.get(metric_key, []), dtype=float)
                if x_values.size != y_values.size:
                    continue
                ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=policy_name)
            ax.set_title(f"{title} ({mix_name})")
            ax.set_xlabel("URLLC Activation Probability per Minislot")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"activation_prob_stress_{mix_name.replace(':', '')}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run URLLC activation-probability sweeps for unified baselines on fixed mixes and loads."
    )
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--mixes", default="7:3,5:5,3:7")
    parser.add_argument("--loads", default="20")
    parser.add_argument("--activation-probs", default="")
    parser.add_argument("--lambdas", default="")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--out-dir", default="sr_mappo/results/unified_lambda_stress")
    parser.add_argument("--mappo-checkpoint-path", default=None)
    parser.add_argument("--reward-from-checkpoint-path", default=None)
    parser.add_argument("--ippo-checkpoint-path", default=None)
    parser.add_argument("--ippo-train-iterations", type=int, default=30)
    parser.add_argument("--ippo-reward-scope", default="global")
    parser.add_argument("--puncturing-selection-rule", default="max_embb_sum_rate")
    parser.add_argument("--superposition-selection-rule", default="max_embb_sum_rate")
    parser.add_argument("--system-num-subcarriers", type=int, default=None)
    parser.add_argument("--system-num-minislots", type=int, default=None)
    parser.add_argument("--debug-reward-terms", action="store_true")
    parser.add_argument("--debug-reward-steps", action="store_true")
    parser.add_argument("--auto-plot-reward-breakdown", action="store_true")
    parser.add_argument("--reward-breakdown-out-dir", default=None)
    parser.add_argument("--reward-breakdown-top-k", type=int, default=20)
    parser.add_argument("--reward-breakdown-abs-threshold", type=float, default=0.05)
    parser.add_argument("--reward-breakdown-rel-threshold", type=float, default=0.01)
    args = parser.parse_args()

    policies = _parse_csv_strings(args.policies)
    mixes = _parse_csv_strings(args.mixes)
    loads = _parse_csv_floats(args.loads)
    activation_probs = _parse_csv_floats(args.activation_probs) if str(args.activation_probs).strip() else (
        _parse_csv_floats(args.lambdas) if str(args.lambdas).strip() else [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
    )
    seeds = [int(value) for value in _parse_csv_floats(args.seeds)]
    episodes_per_seed = int(max(args.episodes_per_seed, 1))

    invalid_mixes = [mix for mix in mixes if mix not in MIX_PRESETS]
    if invalid_mixes:
        raise ValueError(f"Unsupported mixes={invalid_mixes}. Allowed={sorted(MIX_PRESETS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, object] = {
        "meta": {
            "policies": policies,
            "mixes": mixes,
            "loads": loads,
            "activation_probs": activation_probs,
            "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
        },
        "series": {},
        "raw_runs": {},
    }
    json_path = out_dir / "unified_lambda_stress_metrics.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not _meta_compatible(existing, merged["meta"]):
            raise ValueError(
                f"Existing {json_path} is not compatible with current mixes/loads/activation_probs/seeds. "
                "Use a new --out-dir or remove the old metrics file."
            )
        merged = existing

    for mix_name in mixes:
        mix_ratio = float(MIX_PRESETS[mix_name])
        merged["series"].setdefault(mix_name, {})
        merged["raw_runs"].setdefault(mix_name, {})
        for policy in policies:
            raw_policy_runs = dict(merged["raw_runs"][mix_name].get(policy, {}) or {})
            merged["raw_runs"][mix_name][policy] = raw_policy_runs
            for activation_prob in activation_probs:
                ap_key = _activation_prob_key(activation_prob)
                if raw_policy_runs.get(ap_key):
                    print(
                        f"[ACTIVATION-STRESS][SKIP] mix={mix_name} policy={policy} activation_prob={activation_prob:.3f} already saved",
                        flush=True,
                    )
                    continue

                per_seed_runs: List[Dict[str, object]] = []
                for total_load in loads:
                    for seed in seeds:
                        for episode_idx in range(episodes_per_seed):
                            derived_seed = int(seed) if episodes_per_seed == 1 else int(seed) * 1000003 + int(episode_idx)
                            policy_cfg = _build_policy_config(
                                args,
                                policy=policy,
                                total_load=float(total_load),
                                mix_ratio=mix_ratio,
                                activation_prob=float(activation_prob),
                            )
                            trace_context = {
                                "policy": str(policy),
                                "mix": str(mix_name),
                                "load": float(total_load),
                                "urllc_activation_prob": float(activation_prob),
                                "seed": int(seed),
                                "episode_index_within_seed": int(episode_idx),
                                "episode_id": (
                                    f"{policy}|mix={mix_name}|load={float(total_load):g}|activation_prob={float(activation_prob):g}"
                                    f"|seed={int(seed)}|episode={int(episode_idx)}"
                                ),
                            }
                            metrics = run_policy(
                                policy,
                                policy_cfg,
                                int(derived_seed),
                                debug_reward_terms=bool(args.debug_reward_terms),
                                debug_reward_steps=bool(args.debug_reward_steps),
                                reward_trace_context=trace_context,
                            )
                            metrics["evaluated_load"] = float(total_load)
                            metrics["evaluated_urllc_activation_prob"] = float(activation_prob)
                            metrics["evaluated_lambda_per_user"] = float(activation_prob)
                            metrics["evaluated_mix"] = mix_name
                            metrics["evaluated_seed"] = int(seed)
                            metrics["evaluated_episode_index_within_seed"] = int(episode_idx)
                            metrics["evaluated_derived_seed"] = int(derived_seed)
                            per_seed_runs.append(metrics)

                raw_policy_runs[ap_key] = per_seed_runs
                series = _policy_series_from_raw(raw_policy_runs, activation_probs)
                merged["series"][mix_name][policy] = series
                _write_progress(merged, out_dir)
                mean_embb = float(series["embb_rate_mbps"][-1])
                mean_admission = float(series["urllc_admission"][-1])
                mean_urllc_packets = float(series["urllc_admitted_packets"][-1])
                print(
                    f"[ACTIVATION-STRESS] mix={mix_name} policy={policy} activation_prob={activation_prob:.3f} "
                    f"eMBB={mean_embb:.3f}Mbps admit={mean_admission:.4f} scheduled={mean_urllc_packets:.2f}",
                    flush=True,
                )

            merged["series"][mix_name][policy] = _policy_series_from_raw(raw_policy_runs, activation_probs)
            _write_progress(merged, out_dir)

    _write_progress(merged, out_dir)
    print(f"[LAMBDA-STRESS] wrote metrics to {json_path}", flush=True)
    if args.auto_plot_reward_breakdown:
        if len(mixes) != 1 or len(policies) != 1 or len(activation_probs) != 1:
            raise ValueError(
                "--auto-plot-reward-breakdown currently requires exactly one mix, one policy, and one activation probability value."
            )
        breakdown_out_dir = (
            Path(args.reward_breakdown_out_dir)
            if args.reward_breakdown_out_dir
            else out_dir / "reward_breakdown"
        )
        summary = build_reward_breakdown(
            metrics_path=json_path,
            mix=str(mixes[0]),
            policy=str(policies[0]),
            lambda_value=float(activation_probs[0]),
            seed=None,
            out_dir=breakdown_out_dir,
            top_k=int(args.reward_breakdown_top_k),
            abs_threshold=float(args.reward_breakdown_abs_threshold),
            rel_threshold=float(args.reward_breakdown_rel_threshold),
        )
        print(
            f"[ACTIVATION-STRESS] wrote reward breakdown plots to {breakdown_out_dir} "
            f"(episodes={summary['episode_count_for_lambda_bucket']}, residual={summary['reward_residual_unlogged']:.6f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
