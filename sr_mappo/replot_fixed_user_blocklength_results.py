from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "greedy": "Greedy",
    "mappo": "MAPPO",
    "pure_superposition": "Greedy(pure superposition)",
    "pure_puncturing": "Greedy(pure puncturing)",
    "random_scheduler": "Greedy(random scheduler)",
}


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: object) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _resolve_policy_label(
    policy: str,
    policy_label_map: Dict[str, str],
) -> str:
    if policy in POLICY_LABELS:
        return POLICY_LABELS[policy]
    label = str(policy_label_map.get(str(policy), "") or "").strip()
    if label:
        return label
    return POLICY_LABELS.get(policy, policy)


def _resolve_metric_value(
    row: Dict[str, object],
    metric_key: str,
) -> float:
    value = _to_float(row.get(metric_key, 0.0))
    if str(metric_key) == "total_urllc_arrivals" and value <= 0.0:
        admitted = _to_float(row.get("admitted_urllc_count", 0.0))
        blocked = _to_float(row.get("urllc_blocked_user_count", 0.0))
        rebuilt = admitted + blocked
        if rebuilt > 0.0:
            return rebuilt
    return value


def _finalize_figure_with_bottom_legend(
    fig: plt.Figure,
    *,
    handles,
    labels,
    title: str,
    legend_ncol: int,
) -> None:
    fig.suptitle(title, y=0.98)
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=max(int(legend_ncol), 1),
            frameon=False,
        )
        fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.90])
    else:
        fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.90])


def _parse_embb_user_rates_after(row: Dict[str, object]) -> List[float]:
    raw = row.get("embb_user_rates_after", "[]")
    if isinstance(raw, list):
        values = raw
    else:
        text = str(raw or "[]").strip()
        if not text:
            return []
        try:
            values = json.loads(text)
        except Exception:
            return []
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    return [float(x) for x in arr.tolist()]


def _served_only_rate_metric(
    row: Dict[str, object],
    metric_key: str,
) -> float:
    aggregate_key = f"{metric_key}_served_only"
    if aggregate_key in row:
        return _to_float(row.get(aggregate_key, 0.0))
    rates = _parse_embb_user_rates_after(row)
    served = [float(x) for x in rates if float(x) > 1.0e-12]
    if not served:
        return 0.0
    if str(metric_key) == "embb_min_rate_after":
        return float(min(served))
    if str(metric_key) == "embb_5th_percentile_after":
        return float(np.percentile(np.asarray(served, dtype=float), 5.0))
    if str(metric_key) == "embb_median_rate_after":
        return float(np.percentile(np.asarray(served, dtype=float), 50.0))
    return _resolve_metric_value(row, metric_key)


def _aggregate_per_run_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, int, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            _to_int(row.get("embb_users", 0)),
            _to_int(row.get("urllc_users", 0)),
            _to_int(row.get("packet_bits", 0)),
            str(row.get("policy", "")).strip(),
        )
        grouped.setdefault(key, []).append(row)

    aggregated: List[Dict[str, object]] = []
    for _key, group_rows in grouped.items():
        base = dict(group_rows[0])
        numeric_keys = [
            "total_embb_throughput",
            "embb_sum_rate_after",
            "total_urllc_arrivals",
            "urllc_arrival_count",
            "average_power_consumption",
            "total_power",
            "embb_power",
            "urllc_power",
            "embb_served_user_count",
            "embb_blocked_user_count",
            "embb_blocked_count",
            "urllc_blocked_user_count",
            "urllc_blocked_count",
            "embb_minimum_rate_violation_count",
            "overlay_action_count",
            "overlay_count",
            "puncturing_action_count",
            "puncture_count",
            "phase_a_total_decisions",
            "keep_action_count",
            "keep_count",
            "urllc_admission_ratio",
            "admitted_urllc_count",
            "urllc_admitted_count",
            "embb_jain_after",
            "embb_min_rate_after",
            "embb_5th_percentile_after",
            "embb_median_rate_after",
        ]
        for metric in numeric_keys:
            vals = [_to_float(row.get(metric, 0.0)) for row in group_rows]
            base[metric] = float(np.mean(np.asarray(vals, dtype=float))) if vals else 0.0

        served_min_vals: List[float] = []
        served_p5_vals: List[float] = []
        served_median_vals: List[float] = []
        for row in group_rows:
            rates = _parse_embb_user_rates_after(row)
            served = [float(x) for x in rates if float(x) > 1.0e-12]
            if not served:
                served_min_vals.append(0.0)
                served_p5_vals.append(0.0)
                served_median_vals.append(0.0)
                continue
            arr = np.asarray(served, dtype=float)
            served_min_vals.append(float(np.min(arr)))
            served_p5_vals.append(float(np.percentile(arr, 5.0)))
            served_median_vals.append(float(np.percentile(arr, 50.0)))
        base["embb_min_rate_after_served_only"] = float(np.mean(np.asarray(served_min_vals, dtype=float))) if served_min_vals else 0.0
        base["embb_5th_percentile_after_served_only"] = float(np.mean(np.asarray(served_p5_vals, dtype=float))) if served_p5_vals else 0.0
        base["embb_median_rate_after_served_only"] = float(np.mean(np.asarray(served_median_vals, dtype=float))) if served_median_vals else 0.0
        aggregated.append(base)
    return aggregated


def _plot_metric_by_blocklength_panels(
    path: Path,
    rows_map: Dict[Tuple[int, int, str], Dict[str, object]],
    *,
    embb_users: int,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
    metric_key: str,
    overall_title: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, len(packet_bits_list), figsize=(max(12.0, len(packet_bits_list) * 4.2), 5.2), sharey=True)
    if len(packet_bits_list) == 1:
        axes = [axes]

    x = np.arange(len(urllc_users_list), dtype=float)
    width = 0.8 / max(len(policies), 1)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        str(policy): default_colors[idx % len(default_colors)] if default_colors else None
        for idx, policy in enumerate(policies)
    }
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for panel_idx, (ax, packet_bits) in enumerate(zip(axes, packet_bits_list)):
        for policy_idx, policy in enumerate(policies):
            values = [
                _resolve_metric_value(
                    rows_map[(int(urllc_users), int(packet_bits), str(policy))],
                    metric_key,
                )
                for urllc_users in urllc_users_list
            ]
            offset = (policy_idx - (len(policies) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=_resolve_policy_label(str(policy), policy_label_map),
                color=colors.get(str(policy), None),
                edgecolor="black",
                linewidth=0.6,
            )

        panel_tag = panel_labels[panel_idx] if panel_idx < len(panel_labels) else f"({panel_idx + 1})"
        ax.set_title(f"{panel_tag} B{int(packet_bits)}")
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    _finalize_figure_with_bottom_legend(
        fig,
        handles=handles[: len(policies)],
        labels=labels[: len(policies)],
        title=overall_title,
        legend_ncol=max(len(policies), 1),
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_arrival_block_overlay_by_blocklength_panels(
    path: Path,
    rows_map: Dict[Tuple[int, int, str], Dict[str, object]],
    *,
    embb_users: int,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, len(packet_bits_list), figsize=(max(12.0, len(packet_bits_list) * 4.6), 5.4), sharey=True)
    if len(packet_bits_list) == 1:
        axes = [axes]

    x = np.arange(len(urllc_users_list), dtype=float)
    width = 0.8 / max(len(policies), 1)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        str(policy): default_colors[idx % len(default_colors)] if default_colors else None
        for idx, policy in enumerate(policies)
    }
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for panel_idx, (ax, packet_bits) in enumerate(zip(axes, packet_bits_list)):
        arrival_values = []
        for urllc_users in urllc_users_list:
            per_policy_arrivals = [
                _resolve_metric_value(rows_map[(int(urllc_users), int(packet_bits), str(policy))], "total_urllc_arrivals")
                for policy in policies
            ]
            arrival_values.append(float(np.mean(np.asarray(per_policy_arrivals, dtype=float))))

        for policy_idx, policy in enumerate(policies):
            blocked_values = [
                _resolve_metric_value(rows_map[(int(urllc_users), int(packet_bits), str(policy))], "urllc_blocked_user_count")
                for urllc_users in urllc_users_list
            ]
            offset = (policy_idx - (len(policies) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                blocked_values,
                width=width,
                label=_resolve_policy_label(str(policy), policy_label_map),
                color=colors.get(str(policy), None),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )

        ax.plot(
            x,
            arrival_values,
            color="black",
            linewidth=2.0,
            linestyle="--",
            marker="o",
            markersize=5.5,
            label="Mean arrivals",
        )

        panel_tag = panel_labels[panel_idx] if panel_idx < len(panel_labels) else f"({panel_idx + 1})"
        ax.set_title(f"{panel_tag} B{int(packet_bits)}")
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("URLLC packets")
    handles, labels = axes[0].get_legend_handles_labels()
    _finalize_figure_with_bottom_legend(
        fig,
        handles=handles,
        labels=labels,
        title=f"URLLC Arrivals and Blocked Packets (fixed eMBB={embb_users})",
        legend_ncol=min(len(handles), 3),
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mode_ratio_lines_averaged_over_blocklength(
    path: Path,
    rows_map: Dict[Tuple[int, int, str], Dict[str, object]],
    *,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
) -> bool:
    mode_keys = [
        ("keep_action_count", "KEEP ratio", "(a) KEEP ratio"),
        ("overlay_action_count", "OVERLAY ratio", "(b) OVERLAY ratio"),
        ("puncturing_action_count", "PUNCTURE ratio", "(c) PUNCTURE ratio"),
    ]
    # Existing CSV results must contain a valid total decision count to recover KEEP ratios faithfully.
    for urllc_users in urllc_users_list:
        for packet_bits in packet_bits_list:
            for policy in policies:
                total = _to_float(rows_map[(int(urllc_users), int(packet_bits), str(policy))].get("phase_a_total_decisions", 0.0))
                if total <= 0.0:
                    return False

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    x = np.asarray(urllc_users_list, dtype=float)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        str(policy): default_colors[idx % len(default_colors)] if default_colors else None
        for idx, policy in enumerate(policies)
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    markers = {str(policy): marker_cycle[idx % len(marker_cycle)] for idx, policy in enumerate(policies)}

    for ax, (mode_key, ylabel, title) in zip(axes, mode_keys):
        for policy in policies:
            ratios: List[float] = []
            for urllc_users in urllc_users_list:
                per_block_ratios: List[float] = []
                for packet_bits in packet_bits_list:
                    row = rows_map[(int(urllc_users), int(packet_bits), str(policy))]
                    total = max(_to_float(row.get("phase_a_total_decisions", 0.0)), 1.0)
                    per_block_ratios.append(100.0 * _to_float(row.get(mode_key, 0.0)) / total)
                ratios.append(float(np.mean(np.asarray(per_block_ratios, dtype=float))))
            ax.plot(
                x,
                ratios,
                marker=markers.get(str(policy), "o"),
                linewidth=2.0,
                markersize=6.5,
                color=colors.get(str(policy), None),
                label=_resolve_policy_label(str(policy), policy_label_map),
            )
        ax.set_title(title)
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.0, 100.0)

    axes[0].set_ylabel("Average mode ratio (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    _finalize_figure_with_bottom_legend(
        fig,
        handles=handles[: len(policies)],
        labels=labels[: len(policies)],
        title="Mode Selection Ratios Averaged over B120, B150, and B180",
        legend_ncol=max(len(policies), 1),
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def _plot_metric_averaged_over_blocklength(
    path: Path,
    rows_map: Dict[Tuple[int, int, str], Dict[str, object]],
    *,
    embb_users: int,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
    metric_key: str,
    overall_title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(9.5, len(urllc_users_list) * 1.35), 5.2))
    x = np.arange(len(urllc_users_list), dtype=float)
    width = 0.8 / max(len(policies), 1)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        str(policy): default_colors[idx % len(default_colors)] if default_colors else None
        for idx, policy in enumerate(policies)
    }

    for policy_idx, policy in enumerate(policies):
        values = []
        for urllc_users in urllc_users_list:
            per_block_values = [
                _resolve_metric_value(
                    rows_map[(int(urllc_users), int(packet_bits), str(policy))],
                    metric_key,
                )
                for packet_bits in packet_bits_list
            ]
            values.append(float(np.mean(np.asarray(per_block_values, dtype=float))) if per_block_values else 0.0)
        offset = (policy_idx - (len(policies) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=_resolve_policy_label(str(policy), policy_label_map),
            color=colors.get(str(policy), None),
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xlabel("URLLC users")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    _finalize_figure_with_bottom_legend(
        fig,
        handles=handles[: len(policies)],
        labels=labels[: len(policies)],
        title=overall_title,
        legend_ncol=max(len(policies), 1),
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_rate_tail_summary_panels(
    path: Path,
    rows_map: Dict[Tuple[int, int, str], Dict[str, object]],
    *,
    embb_users: int,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
    metric_key: str,
    overall_title: str,
    ylabel: str,
    served_only: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(packet_bits_list), figsize=(max(12.0, len(packet_bits_list) * 4.2), 5.0), sharey=True)
    if len(packet_bits_list) == 1:
        axes = [axes]

    x = np.asarray(urllc_users_list, dtype=float)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        str(policy): default_colors[idx % len(default_colors)] if default_colors else None
        for idx, policy in enumerate(policies)
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P"]
    markers = {str(policy): marker_cycle[idx % len(marker_cycle)] for idx, policy in enumerate(policies)}
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for panel_idx, (ax, packet_bits) in enumerate(zip(axes, packet_bits_list)):
        for policy in policies:
            values = [
                (
                    _served_only_rate_metric(
                        rows_map[(int(urllc_users), int(packet_bits), str(policy))],
                        metric_key,
                    )
                    if served_only
                    else _resolve_metric_value(
                        rows_map[(int(urllc_users), int(packet_bits), str(policy))],
                        metric_key,
                    )
                )
                for urllc_users in urllc_users_list
            ]
            ax.plot(
                x,
                values,
                marker=markers.get(str(policy), "o"),
                linewidth=2.0,
                markersize=6.0,
                color=colors.get(str(policy), None),
                label=_resolve_policy_label(str(policy), policy_label_map),
            )
        panel_tag = panel_labels[panel_idx] if panel_idx < len(panel_labels) else f"({panel_idx + 1})"
        ax.set_title(f"{panel_tag} B{int(packet_bits)}")
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    _finalize_figure_with_bottom_legend(
        fig,
        handles=handles[: len(policies)],
        labels=labels[: len(policies)],
        title=overall_title,
        legend_ncol=max(len(policies), 1),
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot fixed-user blocklength comparison figures from existing CSV results.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--plots-dir-name", default="plots_reworked")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    aggregated_csv = results_dir / "aggregated_results.csv"
    if not aggregated_csv.exists():
        raise FileNotFoundError(f"Missing aggregated CSV: {aggregated_csv}")

    per_run_csv = results_dir / "per_run_results.csv"
    if per_run_csv.exists():
        rows = _aggregate_per_run_rows(_read_csv(per_run_csv))
    else:
        rows = _read_csv(aggregated_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {aggregated_csv}")

    embb_users_values = sorted({_to_int(row.get("embb_users", 0)) for row in rows})
    urllc_users_values = sorted({_to_int(row.get("urllc_users", 0)) for row in rows})
    packet_bits_values = sorted({_to_int(row.get("packet_bits", 0)) for row in rows})
    policies = sorted({str(row.get("policy", "")).strip() for row in rows if str(row.get("policy", "")).strip()})
    policy_label_map: Dict[str, str] = {}

    rows_map: Dict[Tuple[int, int, str], Dict[str, object]] = {}
    for row in rows:
        policy = str(row.get("policy", "")).strip()
        label = str(row.get("policy_label", "") or "").strip()
        if policy and label and policy not in policy_label_map:
            policy_label_map[policy] = label
        rows_map[(
            _to_int(row.get("urllc_users", 0)),
            _to_int(row.get("packet_bits", 0)),
            policy,
        )] = row

    plots_dir = results_dir / str(args.plots_dir_name)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for embb_users in embb_users_values:
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_sum_rate.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="total_embb_throughput",
            overall_title=f"eMBB Sum Rate under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="eMBB sum rate (bps)",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_total_power.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="average_power_consumption",
            overall_title=f"Total Power (W) under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="Total power",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_embb_power.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_power",
            overall_title=f"eMBB Power (W) under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="eMBB power",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_urllc_power.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="urllc_power",
            overall_title=f"URLLC Power (W) under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="URLLC power",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_urllc_arrivals.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="total_urllc_arrivals",
            overall_title=f"URLLC Total Arrivals under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="URLLC total arrivals",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_urllc_admitted_users.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="admitted_urllc_count",
            overall_title=f"Admitted URLLC Packets under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="Admitted URLLC packets",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_urllc_blocked_users.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="urllc_blocked_user_count",
            overall_title=f"URLLC Blocked Users under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="URLLC blocked users count",
        )
        _plot_metric_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_embb_blocked_users.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_blocked_user_count",
            overall_title=f"eMBB Blocked Users under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="eMBB blocked users count",
        )
        _plot_metric_averaged_over_blocklength(
            plots_dir / f"embb_{embb_users}_embb_power_avg_over_blocklength.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_power",
            overall_title=f"eMBB Power (W) (fixed eMBB={embb_users})",
            ylabel="Average eMBB power",
        )
        _plot_metric_averaged_over_blocklength(
            plots_dir / f"embb_{embb_users}_embb_blocked_users_avg_over_blocklength.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_blocked_user_count",
            overall_title=f"eMBB Blocked Users (fixed eMBB={embb_users})",
            ylabel="Average eMBB blocked users count",
        )
        _plot_rate_tail_summary_panels(
            plots_dir / f"embb_{embb_users}_jain_fairness.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_jain_after",
            overall_title=f"eMBB Jain Fairness under Different Block Lengths (fixed eMBB={embb_users})",
            ylabel="Jain fairness index",
        )
        _plot_rate_tail_summary_panels(
            plots_dir / f"embb_{embb_users}_min_rate_tail.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_min_rate_after",
            overall_title=f"Minimum Served eMBB Rate under Different Block Lengths (fixed eMBB={embb_users}, excluding 0 Mbps blocked users)",
            ylabel="Minimum served eMBB rate (Mbps)",
            served_only=True,
        )
        _plot_rate_tail_summary_panels(
            plots_dir / f"embb_{embb_users}_5th_percentile_rate.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_5th_percentile_after",
            overall_title=f"5th Percentile Served eMBB Rate under Different Block Lengths (fixed eMBB={embb_users}, excluding 0 Mbps blocked users)",
            ylabel="5th percentile eMBB rate (Mbps)",
            served_only=True,
        )
        _plot_rate_tail_summary_panels(
            plots_dir / f"embb_{embb_users}_median_rate.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
            metric_key="embb_median_rate_after",
            overall_title=f"Median Served eMBB Rate under Different Block Lengths (fixed eMBB={embb_users}, excluding 0 Mbps blocked users)",
            ylabel="Median served eMBB rate (Mbps)",
            served_only=True,
        )
        _plot_arrival_block_overlay_by_blocklength_panels(
            plots_dir / f"embb_{embb_users}_urllc_arrivals_vs_blocked.png",
            rows_map,
            embb_users=embb_users,
            urllc_users_list=urllc_users_values,
            packet_bits_list=packet_bits_values,
            policies=policies,
            policy_label_map=policy_label_map,
        )
    mode_ok = _plot_mode_ratio_lines_averaged_over_blocklength(
        plots_dir / f"embb_{embb_users_values[0]}_mode_ratios.png",
        rows_map,
        urllc_users_list=urllc_users_values,
        packet_bits_list=packet_bits_values,
        policies=policies,
        policy_label_map=policy_label_map,
    )
    note_path = plots_dir / "mode_plot_note.txt"
    if mode_ok:
        note_path.write_text(
            "Mode ratio figure successfully generated from existing CSV results.\n",
            encoding="utf-8",
        )
    else:
        note_path.write_text(
            "Mode ratio replot skipped because existing CSV has phase_a_total_decisions=0 for the saved rows,\n"
            "so KEEP/OVERLAY/PUNCTURE ratios cannot be reconstructed faithfully from this result set.\n",
            encoding="utf-8",
        )
    print(f"[REPLOT] wrote updated plots to {plots_dir}", flush=True)
    if mode_ok:
        print(f"[REPLOT] wrote mode ratio plot to {plots_dir / f'embb_{embb_users_values[0]}_mode_ratios.png'}", flush=True)
    else:
        print(f"[REPLOT] mode ratio plot skipped; see {note_path}", flush=True)


if __name__ == "__main__":
    main()
