from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .baseline_catalog import baseline_label, baseline_metadata


def dense_series() -> List[Tuple[str, str, str, str]]:
    return [
        ("original", baseline_label("original"), "tab:blue", "o"),
        ("original_greedy_normal_v1", baseline_label("original_greedy_normal_v1"), "tab:cyan", "P"),
        ("original_greedy_normal_v2", baseline_label("original_greedy_normal_v2"), "tab:pink", "X"),
        ("myopic_throughput_greedy", baseline_label("myopic_throughput_greedy"), "tab:brown", "h"),
        ("matched_fixed_embb", baseline_label("matched_fixed_embb"), "tab:green", "^"),
        ("throughput_only_greedy", baseline_label("throughput_only_greedy"), "tab:purple", "D"),
        ("channel_only_greedy", baseline_label("channel_only_greedy"), "tab:gray", "v"),
        ("rl", "MAPPO", "tab:orange", "s"),
        ("embb_only_ceiling", "eMBB-only ceiling", "black", "x"),
        ("throughput_feasible_oracle", baseline_label("throughput_feasible_oracle"), "tab:red", "*"),
        ("throughput_biased_greedy", baseline_label("throughput_biased_greedy"), "tab:brown", "h"),
    ]


def plot_dense_series() -> List[Tuple[str, str, str, str]]:
    return [
        ("myopic_throughput_greedy", baseline_label("myopic_throughput_greedy"), "tab:brown", "h"),
        ("rl", "MAPPO", "tab:orange", "s"),
    ]


def scalar_dist(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=float)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def normalize_dense_record(record: Dict, num_uavs: int) -> Dict[str, float]:
    embb_rate = float(record.get("embb_rate", record.get("embb_total_rate", 0.0)))
    admission = float(record.get("urllc_admission", record.get("urllc_admission_rate", 0.0)))
    admitted_reliability = float(
        record.get("admitted_urllc_reliability", record.get("urllc_reliability", record.get("urllc_success_rate", np.nan)))
    )
    effective_success = float(
        record.get("effective_urllc_success_over_arrivals", record.get("urllc_success_rate", 0.0))
    )
    total_power = float(record.get("total_power", 0.0))
    puncture_loss = float(record.get("avg_puncture_loss", record.get("avg_puncture_embb_loss", 0.0)))
    overlay_retention = float(record.get("avg_overlay_retention", 0.0))
    scheduled_packets = float(record.get("scheduled_packets", 0.0))
    return {
        "embb_rate": embb_rate,
        "urllc_admission": admission,
        "admitted_urllc_reliability": admitted_reliability,
        "urllc_reliability": admitted_reliability,
        "effective_urllc_success_over_arrivals": effective_success,
        "empty_admission_case": float(record.get("empty_admission_case", 0.0)),
        "total_power": total_power,
        "embb_power": float(record.get("embb_power", 0.0)),
        "urllc_power": float(record.get("urllc_power", 0.0)),
        "avg_puncture_loss": puncture_loss,
        "avg_overlay_retention": overlay_retention,
        "overlay_ratio": float(record.get("overlay_ratio", 0.0)),
        "puncture_ratio": float(record.get("puncture_ratio", 0.0)),
        "overlay_selection_ratio": float(record.get("overlay_selection_ratio", 0.0)),
        "puncture_selection_ratio": float(record.get("puncture_selection_ratio", 0.0)),
        "admission_via_overlay_ratio": float(record.get("admission_via_overlay_ratio", 0.0)),
        "admission_via_puncture_ratio": float(record.get("admission_via_puncture_ratio", 0.0)),
        "puncture_candidate_pruned_by_loss_ceiling_ratio": float(record.get("puncture_candidate_pruned_by_loss_ceiling_ratio", 0.0)),
        "overlay_candidate_pairs": float(record.get("overlay_candidate_pairs", 0.0)),
        "overlay_feasible_pairs": float(record.get("overlay_feasible_pairs", 0.0)),
        "overlay_selected_pairs": float(record.get("overlay_selected_pairs", 0.0)),
        "phase_a_total_decisions": float(record.get("phase_a_total_decisions", 0.0)),
        "scheduled_packets": scheduled_packets,
        "scheduled_packets_per_uav": float(scheduled_packets / max(num_uavs, 1)),
    }


_DENSE_METRIC_KEYS = [
    "embb_rate", "urllc_admission", "admitted_urllc_reliability", "urllc_reliability",
    "effective_urllc_success_over_arrivals", "empty_admission_case", "total_power", "embb_power", "urllc_power",
    "avg_puncture_loss", "avg_overlay_retention", "overlay_ratio", "puncture_ratio",
    "overlay_selection_ratio", "puncture_selection_ratio",
    "admission_via_overlay_ratio", "admission_via_puncture_ratio",
    "puncture_candidate_pruned_by_loss_ceiling_ratio", "overlay_candidate_pairs",
    "overlay_feasible_pairs", "overlay_selected_pairs", "phase_a_total_decisions",
    "scheduled_packets", "scheduled_packets_per_uav",
]


def build_method_dense_summary(
    method_key: str,
    method_name: str,
    loads: List[float],
    records_by_load: List[List[Dict[str, float]]],
    checkpoint_path: str,
    checkpoint_reason: str,
    run_name: str,
    protocol: Dict[str, object],
) -> Dict:
    summary = {
        "method_key": method_key,
        "method_name": method_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_reason": checkpoint_reason,
        "run_name": run_name,
        "loads": [float(load) for load in loads],
        "per_load": [],
        "stats": {metric: {field: [] for field in ("mean", "std", "min", "max", "p25", "p75")} for metric in _DENSE_METRIC_KEYS},
        "protocol": {
            **protocol,
            **(baseline_metadata(method_key) if method_key not in {"rl", "embb_only_ceiling"} else {}),
        },
        "audit_rows": [],
    }
    for load, records in zip(loads, records_by_load):
        load_entry = {
            "method_name": method_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_reason": checkpoint_reason,
            "run_name": run_name,
            "load_value": float(load),
            "records": records,
        }
        for metric in _DENSE_METRIC_KEYS:
            dist = scalar_dist([float(record.get(metric, 0.0)) for record in records])
            load_entry[f"{metric}_stats"] = dist
            for field, value in dist.items():
                summary["stats"][metric][field].append(value)
        load_entry.update({
            "aggregate_embb_throughput": float(load_entry["embb_rate_stats"]["mean"]),
            "admission_ratio": float(load_entry["urllc_admission_stats"]["mean"]),
            "total_power": float(load_entry["total_power_stats"]["mean"]),
            "avg_embb_loss_per_puncture": float(load_entry["avg_puncture_loss_stats"]["mean"]),
            "overlay_selection_ratio": float(load_entry["overlay_selection_ratio_stats"]["mean"]),
            "puncture_selection_ratio": float(load_entry["puncture_selection_ratio_stats"]["mean"]),
        })
        summary["per_load"].append(load_entry)
        summary["audit_rows"].append({
            "method_name": method_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_reason": checkpoint_reason,
            "run_name": run_name,
            "load_value": float(load),
            "aggregate_embb_throughput": float(load_entry["aggregate_embb_throughput"]),
            "admission_ratio": float(load_entry["admission_ratio"]),
            "total_power": float(load_entry["total_power"]),
            "avg_embb_loss_per_puncture": float(load_entry["avg_embb_loss_per_puncture"]),
            "overlay_selection_ratio": float(load_entry["overlay_selection_ratio"]),
            "puncture_selection_ratio": float(load_entry["puncture_selection_ratio"]),
        })
    for metric in _DENSE_METRIC_KEYS:
        summary[metric] = list(summary["stats"][metric]["mean"])
    return summary


def finalize_dense_bundle(method_bundle: Dict[str, Dict]) -> Dict[str, Dict]:
    baseline = method_bundle["original_greedy_normal_v2"]
    ceiling = method_bundle["embb_only_ceiling"]
    oracle = method_bundle["throughput_feasible_oracle"]
    for payload in method_bundle.values():
        payload["throughput_to_ceiling"] = []
        payload["throughput_gap_vs_original_greedy_normal_v2"] = []
        payload["throughput_gap_vs_throughput_feasible_oracle"] = []
        payload["admission_gap_vs_original_greedy_normal_v2"] = []
        payload["power_ratio_vs_original_greedy_normal_v2"] = []
        payload["power_normalized_throughput"] = []
        payload["puncture_damage_normalized_admission"] = []
        for idx, _load in enumerate(payload["loads"]):
            embb_rate = float(payload["embb_rate"][idx])
            total_power = float(payload["total_power"][idx])
            puncture_loss = float(payload["avg_puncture_loss"][idx])
            payload["throughput_to_ceiling"].append(float(embb_rate / max(float(ceiling["embb_rate"][idx]), 1e-9)))
            payload["throughput_gap_vs_original_greedy_normal_v2"].append(float((embb_rate - float(baseline["embb_rate"][idx])) / 1e6))
            payload["throughput_gap_vs_throughput_feasible_oracle"].append(float((embb_rate - float(oracle["embb_rate"][idx])) / 1e6))
            payload["admission_gap_vs_original_greedy_normal_v2"].append(float(payload["urllc_admission"][idx] - float(baseline["urllc_admission"][idx])))
            payload["power_ratio_vs_original_greedy_normal_v2"].append(float(total_power / max(float(baseline["total_power"][idx]), 1e-9)))
            payload["power_normalized_throughput"].append(float((embb_rate / 1e6) / max(total_power, 1e-9)))
            payload["puncture_damage_normalized_admission"].append(float(payload["urllc_admission"][idx] / max(1.0 + puncture_loss / 1e6, 1e-9)))
            payload["per_load"][idx]["loadwise_power_ratio_vs_original_greedy_normal_v2"] = float(payload["power_ratio_vs_original_greedy_normal_v2"][-1])
    return method_bundle


def extract_method_point_audit(dense_bundle: Dict[str, Dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method_key, _label, _color, _marker in dense_series():
        payload = dense_bundle.get(method_key)
        if payload:
            rows.extend(payload.get("audit_rows", []))
    return rows


def nearest_replica_proxy(dense_bundle: Dict[str, Dict]) -> Dict[str, object]:
    compare_methods = ["matched_fixed_embb", "rl"]
    compare_methods = [method for method in compare_methods if method in dense_bundle]
    loads = list(dense_bundle[compare_methods[0]]["loads"]) if compare_methods else []
    throughput_at_matched_admission = {method: [] for method in compare_methods}
    admission_at_matched_throughput = {method: [] for method in compare_methods}
    target_admission = []
    target_throughput = []
    for idx, _load in enumerate(loads):
        adm_target = float(np.median([dense_bundle[method]["urllc_admission"][idx] for method in compare_methods]))
        thr_target = float(np.median([dense_bundle[method]["embb_rate"][idx] for method in compare_methods]))
        target_admission.append(adm_target)
        target_throughput.append(thr_target)
        for method in compare_methods:
            records = dense_bundle[method]["per_load"][idx]["records"]
            best_adm = min(records, key=lambda record: abs(float(record["urllc_admission"]) - adm_target))
            best_thr = min(records, key=lambda record: abs(float(record["embb_rate"]) - thr_target))
            throughput_at_matched_admission[method].append(float(best_adm["embb_rate"]) / 1e6)
            admission_at_matched_throughput[method].append(float(best_thr["urllc_admission"]))
    return {
        "loads": loads,
        "target_admission": target_admission,
        "target_throughput": [float(value) / 1e6 for value in target_throughput],
        "throughput_at_matched_admission": throughput_at_matched_admission,
        "admission_at_matched_throughput": admission_at_matched_throughput,
    }


def build_low_damage_diagnostics(rl_dense: Dict, baseline_dense: Dict) -> Dict[str, List[float]]:
    loads = list(rl_dense["loads"])
    return {
        "loads": loads,
        "baseline_label": str(baseline_dense.get("method_name", baseline_label("matched_fixed_embb"))),
        "loadwise_selected_puncture_loss": [float(value) for value in rl_dense["avg_puncture_loss"]],
        "loadwise_selected_overlay_retention": [float(value) for value in rl_dense["avg_overlay_retention"]],
        "loadwise_power_ratio_vs_original_greedy_normal_v2": [float(value) for value in rl_dense["power_ratio_vs_original_greedy_normal_v2"]],
        "admission_via_overlay_ratio": [float(value) for value in rl_dense["admission_via_overlay_ratio"]],
        "admission_via_puncture_ratio": [float(value) for value in rl_dense["admission_via_puncture_ratio"]],
        "puncture_candidate_pruned_by_loss_ceiling_ratio": [float(value) for value in rl_dense["puncture_candidate_pruned_by_loss_ceiling_ratio"]],
        "baseline_puncture_loss": [float(value) for value in baseline_dense["avg_puncture_loss"]],
        "baseline_overlay_retention": [float(value) for value in baseline_dense["avg_overlay_retention"]],
    }


def _plot_dense_band(ax, dense_bundle: Dict[str, Dict], metric_key: str, transform, ylabel: str, title: str, style_fn, top_axis_fn):
    loads = list(np.asarray(next(iter(dense_bundle.values()))["loads"], dtype=float))
    for method_key, label, color, marker in plot_dense_series():
        payload = dense_bundle.get(method_key)
        if payload is None:
            continue
        mean = transform(np.asarray(payload["stats"][metric_key]["mean"], dtype=float))
        p25 = transform(np.asarray(payload["stats"][metric_key]["p25"], dtype=float))
        p75 = transform(np.asarray(payload["stats"][metric_key]["p75"], dtype=float))
        ax.plot(loads, mean, marker=marker, color=color, linewidth=1.6, label=label)
        ax.fill_between(loads, p25, p75, color=color, alpha=0.10)
    style_fn(ax, title, "Average UE load per UAV", ylabel)
    top_axis_fn(ax, loads)


def plot_dense_uncertainty_bands(dense_bundle: Dict[str, Dict], results_dir, style_fn, style_power_axis, top_axis_fn):
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    _plot_dense_band(axes[0], dense_bundle, "embb_rate", lambda x: x / 1e6, "Mbps", "Dense throughput with uncertainty band", style_fn, top_axis_fn)
    axes[0].legend(fontsize=8, ncol=3)
    _plot_dense_band(axes[1], dense_bundle, "urllc_admission", lambda x: x, "Admission ratio", "Dense admission with uncertainty band", style_fn, top_axis_fn)
    _plot_dense_band(axes[2], dense_bundle, "total_power", lambda x: x * 1e3, "mW", "Dense total power with uncertainty band", style_power_axis, top_axis_fn)
    fig.suptitle("Dense Load Sweep with Seed/Scenario Band", fontsize=14)
    path = results_dir / "14_dense_uncertainty_bands.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_normalized_gap_diagnostics(dense_bundle: Dict[str, Dict], results_dir, style_fn, top_axis_fn):
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    loads = dense_bundle["rl"]["loads"]
    panels = [
        ("throughput_to_ceiling", "Throughput / eMBB-only ceiling", "Ratio"),
        ("throughput_gap_vs_original_greedy_normal_v2", "Throughput gap vs Original Greedy Normal v2", "Mbps"),
        ("throughput_gap_vs_throughput_feasible_oracle", "Throughput gap vs Throughput-feasible Oracle", "Mbps"),
        ("admission_gap_vs_original_greedy_normal_v2", "Admission gap vs Original Greedy Normal v2", "Gap"),
        ("power_normalized_throughput", "Power-normalized throughput", "Mbps/W"),
        ("puncture_damage_normalized_admission", "Puncture-damage-normalized admission", "Admission / (1+loss Mbps)"),
    ]
    compare_methods = ["matched_fixed_embb", "rl"]
    for ax, (metric_key, title, ylabel) in zip(axes.flat, panels):
        for method_key, label, color, marker in plot_dense_series():
            if method_key not in compare_methods:
                continue
            payload = dense_bundle.get(method_key)
            if payload is None:
                continue
            ax.plot(loads, payload[metric_key], marker=marker, color=color, label=label)
        style_fn(ax, title, "Average UE load per UAV", ylabel)
        top_axis_fn(ax, loads)
        ax.legend(fontsize=8)
    fig.suptitle("Normalized-to-Ceiling and Gap Diagnostics", fontsize=14)
    path = results_dir / "15_normalized_gap_diagnostics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_matched_admission_diagnostics(matched_bundle: Dict[str, object], results_dir, style_fn):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    loads = matched_bundle["loads"]
    compare_methods = ["matched_fixed_embb", "rl"]
    for method_key, label, color, marker in plot_dense_series():
        if method_key not in compare_methods or method_key not in matched_bundle["throughput_at_matched_admission"]:
            continue
        axes[0].plot(loads, matched_bundle["throughput_at_matched_admission"][method_key], marker=marker, color=color, label=label)
        axes[1].plot(loads, matched_bundle["admission_at_matched_throughput"][method_key], marker=marker, color=color, label=label)
    style_fn(axes[0], "Throughput @ matched admission (nearest-replica proxy)", "Average UE load per UAV", "Mbps")
    style_fn(axes[1], "Admission @ matched throughput (nearest-replica proxy)", "Average UE load per UAV", "Admission ratio")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Matched-Admission / Matched-Throughput Views", fontsize=14)
    path = results_dir / "16_matched_admission_diagnostics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_method_decomposition_dense(dense_bundle: Dict[str, Dict], results_dir, style_fn, top_axis_fn):
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), constrained_layout=True)
    loads = dense_bundle["rl"]["loads"]
    panels = [
        ("admission_via_overlay_ratio", "Admission via overlay ratio", "Ratio"),
        ("admission_via_puncture_ratio", "Admission via puncture ratio", "Ratio"),
        ("avg_puncture_loss", "Selected puncture loss", "Mbps"),
        ("avg_overlay_retention", "Selected overlay retention", "Ratio"),
        ("overlay_candidate_pairs", "Overlay candidate count", "Pairs"),
        ("overlay_feasible_pairs", "Overlay feasible count", "Pairs"),
        ("overlay_selected_pairs", "Overlay selected count", "Pairs"),
        ("scheduled_packets_per_uav", "Scheduled URLLC packets / UAV", "Packets/UAV"),
        ("power_ratio_vs_original_greedy_normal_v2", "Power ratio vs Original Greedy Normal v2", "Ratio"),
        ("urllc_power", "URLLC transmit power", "mW"),
    ]
    compare_methods = ["matched_fixed_embb", "rl"]
    for ax, (metric_key, title, ylabel) in zip(axes.flat, panels):
        for method_key, label, color, marker in plot_dense_series():
            if method_key not in compare_methods:
                continue
            payload = dense_bundle.get(method_key)
            if payload is None:
                continue
            values = np.asarray(payload[metric_key], dtype=float)
            if metric_key == "avg_puncture_loss":
                values = values / 1e6
            elif metric_key == "urllc_power":
                values = values * 1e3
            ax.plot(loads, values, marker=marker, color=color, label=label)
        style_fn(ax, title, "Average UE load per UAV", ylabel)
        top_axis_fn(ax, loads)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Method-specific Decomposition (Dense Sweep)", fontsize=14)
    path = results_dir / "17_method_decomposition_dense.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_marginal_degradation_slopes(dense_bundle: Dict[str, Dict], results_dir, style_fn, style_power_axis):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    compare_methods = ["matched_fixed_embb", "rl"]
    base_loads = np.asarray(dense_bundle["rl"]["loads"], dtype=float)
    delta_x = base_loads[1:]
    for method_key, label, color, marker in plot_dense_series():
        if method_key not in compare_methods:
            continue
        payload = dense_bundle.get(method_key)
        if payload is None:
            continue
        axes[0].plot(delta_x, np.diff(np.asarray(payload["embb_rate"], dtype=float) / 1e6), marker=marker, color=color, label=label)
        axes[1].plot(delta_x, np.diff(np.asarray(payload["urllc_admission"], dtype=float)), marker=marker, color=color, label=label)
        axes[2].plot(delta_x, np.diff(np.asarray(payload["total_power"], dtype=float) * 1e3), marker=marker, color=color, label=label)
    style_fn(axes[0], "Marginal throughput degradation slope", "Load transition end-point", "Δ throughput (Mbps)")
    style_fn(axes[1], "Marginal admission slope", "Load transition end-point", "Δ admission")
    style_power_axis(axes[2], "Marginal power slope", "Load transition end-point", "Δ power (mW)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Marginal Degradation Slopes across Dense Loads", fontsize=14)
    path = results_dir / "18_marginal_degradation_slopes.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_low_damage_admission_diagnostics(low_damage: Dict, results_dir, style_fn):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    loads = low_damage["loads"]
    baseline_label_text = str(low_damage.get("baseline_label", baseline_label("matched_fixed_embb")))
    axes[0, 0].plot(loads, np.asarray(low_damage["loadwise_selected_puncture_loss"]) / 1e6, marker="o", label="MAPPO")
    axes[0, 0].plot(loads, np.asarray(low_damage["baseline_puncture_loss"]) / 1e6, marker="*", linestyle="--", label=baseline_label_text)
    style_fn(axes[0, 0], "Selected puncture loss", "Average UE load per UAV", "Mbps loss/action")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(loads, low_damage["loadwise_selected_overlay_retention"], marker="o", label="MAPPO")
    axes[0, 1].plot(loads, low_damage["baseline_overlay_retention"], marker="*", linestyle="--", label=baseline_label_text)
    style_fn(axes[0, 1], "Selected overlay retention", "Average UE load per UAV", "Retention ratio")
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot(loads, low_damage["loadwise_power_ratio_vs_original_greedy_normal_v2"], marker="o", color="tab:red")
    style_fn(axes[0, 2], "Power ratio vs Original Greedy Normal v2", "Average UE load per UAV", "Ratio")
    axes[1, 0].plot(loads, low_damage["admission_via_overlay_ratio"], marker="o", color="tab:green")
    style_fn(axes[1, 0], "Admission via overlay", "Average UE load per UAV", "Ratio")
    axes[1, 1].plot(loads, low_damage["admission_via_puncture_ratio"], marker="o", color="tab:purple")
    style_fn(axes[1, 1], "Admission via puncture", "Average UE load per UAV", "Ratio")
    axes[1, 2].plot(loads, low_damage["puncture_candidate_pruned_by_loss_ceiling_ratio"], marker="o", color="tab:brown")
    style_fn(axes[1, 2], "Puncture candidates pruned by loss ceiling", "Average UE load per UAV", "Ratio")
    fig.suptitle("Low-Damage Admission Diagnostics", fontsize=14)
    path = results_dir / "13_low_damage_admission_diagnostics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
