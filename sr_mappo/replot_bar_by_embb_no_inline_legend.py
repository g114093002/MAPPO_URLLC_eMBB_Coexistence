from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pretty_policy_label(label: str) -> str:
    raw = str(label).strip()
    lowered = raw.lower()
    explicit = {
        "greedy": "Greedy",
        "pure_puncturing": "Pure Puncturing",
        "pure_superposition": "Pure Superposition",
        "random_scheduler": "Random Scheduler",
        "mappo": "MAPPO",
        "mappo (v17zza)": "MAPPO",
    }
    if lowered in explicit:
        return explicit[lowered]
    return raw.replace("_", " ").title()


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed = {
                str(key).strip().lstrip("\ufeff"): value
                for key, value in dict(row).items()
            }
            for key in ("embb_users", "urllc_users", "packet_bits"):
                if key in parsed and parsed[key] not in (None, "", "None"):
                    parsed[key] = int(round(float(parsed[key])))
            for key, value in list(parsed.items()):
                if key in {"policy", "policy_label"}:
                    continue
                if isinstance(value, str) and value not in ("", "None"):
                    try:
                        parsed[key] = float(value)
                    except Exception:
                        pass
            policy_label = str(parsed.get("policy_label", parsed.get("policy", "")) or "")
            parsed["policy_label"] = _pretty_policy_label(policy_label)
            rows.append(parsed)
        return rows


def _policy_order(rows: Iterable[Dict[str, object]]) -> List[str]:
    ordered: "OrderedDict[str, None]" = OrderedDict()
    for row in rows:
        ordered[str(row["policy_label"])] = None
    return list(ordered.keys())


def _legend_style_map(policy_labels: Iterable[str]) -> Dict[str, str]:
    palette = {
        "Greedy": "#4C78A8",
        "Pure Puncturing": "#F58518",
        "Pure Superposition": "#54A24B",
        "Random Scheduler": "#B279A2",
        "MAPPO": "#E45756",
    }
    style: Dict[str, str] = {}
    fallback = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    for idx, label in enumerate(policy_labels):
        style[str(label)] = palette.get(str(label), fallback[idx % len(fallback)])
    return style


def _save_standalone_legend(path: Path, policy_labels: List[str]) -> None:
    color_map = _legend_style_map(policy_labels)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map[label], edgecolor="black", linewidth=0.6)
        for label in policy_labels
    ]
    fig, ax = plt.subplots(figsize=(max(6.0, 2.3 * len(policy_labels)), 1.2))
    ax.axis("off")
    ax.legend(
        handles,
        policy_labels,
        loc="center",
        ncol=max(1, len(policy_labels)),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_grouped_bars(
    path: Path,
    title: str,
    ylabel: str,
    rows: List[Dict[str, object]],
    metric_key: str,
    policy_labels: List[str],
) -> None:
    color_map = _legend_style_map(policy_labels)
    urllc_values = sorted({int(row["urllc_users"]) for row in rows})
    width = 0.8 / max(len(policy_labels), 1)
    x = np.arange(len(urllc_values), dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for idx, label in enumerate(policy_labels):
        values: List[float] = []
        for urllc_users in urllc_values:
            match = next(
                row for row in rows
                if int(row["urllc_users"]) == int(urllc_users) and str(row["policy_label"]) == label
            )
            values.append(float(match.get(metric_key, 0.0) or 0.0))
        offset = (idx - (len(policy_labels) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color_map[label],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_title(title)
    ax.set_xlabel("URLLC Users")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in urllc_values])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mode_ratios(path: Path, rows: List[Dict[str, object]], policy_labels: List[str]) -> None:
    color_map = _legend_style_map(policy_labels)
    urllc_values = sorted({int(row["urllc_users"]) for row in rows})
    x = np.arange(len(urllc_values), dtype=float)
    mode_specs = [
        ("keep_ratio_pct", "KEEP Ratio (%)"),
        ("overlay_ratio_pct", "OVERLAY Ratio (%)"),
        ("puncture_ratio_pct", "PUNCTURE Ratio (%)"),
    ]
    width = 0.8 / max(len(policy_labels), 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    for ax, (metric_key, title) in zip(axes, mode_specs):
        for idx, label in enumerate(policy_labels):
            values: List[float] = []
            for urllc_users in urllc_values:
                match = next(
                    row for row in rows
                    if int(row["urllc_users"]) == int(urllc_users) and str(row["policy_label"]) == label
                )
                values.append(float(match.get(metric_key, 0.0) or 0.0))
            offset = (idx - (len(policy_labels) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=color_map[label],
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_title(title)
        ax.set_xlabel("URLLC Users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in urllc_values])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Ratio (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mode_ratios_bar_no_keep(path: Path, rows: List[Dict[str, object]], policy_labels: List[str]) -> None:
    color_map = _legend_style_map(policy_labels)
    urllc_values = sorted({int(row["urllc_users"]) for row in rows})
    x = np.arange(len(urllc_values), dtype=float)
    mode_specs = [
        ("overlay_ratio_pct", "Overlay"),
        ("puncture_ratio_pct", "Puncture"),
    ]
    width = 0.8 / max(len(policy_labels), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
    for ax, (metric_key, title) in zip(axes, mode_specs):
        for idx, label in enumerate(policy_labels):
            values: List[float] = []
            for urllc_users in urllc_values:
                match = next(
                    row for row in rows
                    if int(row["urllc_users"]) == int(urllc_users) and str(row["policy_label"]) == label
                )
                overlay = float(match.get("overlay_ratio_pct", 0.0) or 0.0)
                puncture = float(match.get("puncture_ratio_pct", 0.0) or 0.0)
                active = overlay + puncture
                if active <= 1.0e-12:
                    normalized = 0.0
                elif metric_key == "overlay_ratio_pct":
                    normalized = 100.0 * overlay / active
                else:
                    normalized = 100.0 * puncture / active
                values.append(float(normalized))
            offset = (idx - (len(policy_labels) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=color_map[label],
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_title(title)
        ax.set_xlabel("URLLC Users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in urllc_values])
        ax.set_ylim(0.0, 100.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Conditional Ratio (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_selection_lines_no_keep(path: Path, rows: List[Dict[str, object]], policy_labels: List[str]) -> None:
    color_map = _legend_style_map(policy_labels)
    urllc_values = sorted({int(row["urllc_users"]) for row in rows})
    mode_specs = [
        ("overlay_ratio_pct", "Overlay"),
        ("puncture_ratio_pct", "Puncture"),
    ]
    mode_styles = {
        "Overlay": ("o", "-"),
        "Puncture": ("s", "--"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label in policy_labels:
        policy_rows = [
            row for row in rows
            if str(row["policy_label"]) == label
        ]
        for metric_key, mode_name in mode_specs:
            values: List[float] = []
            for urllc_users in urllc_values:
                match = next(
                    row for row in policy_rows
                    if int(row["urllc_users"]) == int(urllc_users)
                )
                overlay = float(match.get("overlay_ratio_pct", 0.0) or 0.0)
                puncture = float(match.get("puncture_ratio_pct", 0.0) or 0.0)
                active = overlay + puncture
                if active <= 1.0e-12:
                    normalized = 0.0
                elif metric_key == "overlay_ratio_pct":
                    normalized = 100.0 * overlay / active
                else:
                    normalized = 100.0 * puncture / active
                values.append(float(normalized))
            marker, linestyle = mode_styles[mode_name]
            ax.plot(
                urllc_values,
                values,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6,
                color=color_map[label],
                label=f"{label} - {mode_name}",
            )
    ax.set_title("Mode Selection (Overlay / Puncture Only)")
    ax.set_xlabel("URLLC Users")
    ax.set_ylabel("Conditional Ratio (%)")
    ax.set_xticks(urllc_values)
    ax.set_ylim(0.0, 100.0)
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_selection_lines_with_keep(path: Path, rows: List[Dict[str, object]], policy_labels: List[str]) -> None:
    color_map = _legend_style_map(policy_labels)
    urllc_values = sorted({int(row["urllc_users"]) for row in rows})
    mode_specs = [
        ("keep_ratio_pct", "Keep"),
        ("overlay_ratio_pct", "Overlay"),
        ("puncture_ratio_pct", "Puncture"),
    ]
    mode_styles = {
        "Keep": ("^", ":"),
        "Overlay": ("o", "-"),
        "Puncture": ("s", "--"),
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for label in policy_labels:
        policy_rows = [
            row for row in rows
            if str(row["policy_label"]) == label
        ]
        for metric_key, mode_name in mode_specs:
            values: List[float] = []
            for urllc_users in urllc_values:
                match = next(
                    row for row in policy_rows
                    if int(row["urllc_users"]) == int(urllc_users)
                )
                values.append(float(match.get(metric_key, 0.0) or 0.0))
            marker, linestyle = mode_styles[mode_name]
            ax.plot(
                urllc_values,
                values,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6,
                color=color_map[label],
                label=f"{label} - {mode_name}",
            )
    ax.set_title("Mode Selection (Keep / Overlay / Puncture)")
    ax.set_xlabel("URLLC Users")
    ax.set_ylabel("Ratio (%)")
    ax.set_xticks(urllc_values)
    ax.set_ylim(0.0, 100.0)
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_note(path: Path) -> None:
    path.write_text(
        "Inline legends removed. See legend.png in the same folder for policy colors/order.\n",
        encoding="utf-8",
    )


def _replot_folder(folder: Path) -> None:
    plot_source = folder / "plot_source_metrics.csv"
    if not plot_source.exists():
        raise FileNotFoundError(f"Missing plot source: {plot_source}")
    rows = _read_rows(plot_source)
    if not rows:
        raise ValueError(f"No rows in {plot_source}")
    embb_users = int(rows[0]["embb_users"])
    policy_labels = _policy_order(rows)

    metric_specs = [
        ("sum_rate_bar.png", "sum_rate_mbps", f"eMBB={embb_users}: Sum Rate", "Sum Rate (Mbps)"),
        ("total_power_bar.png", "total_power", f"eMBB={embb_users}: Total Power", "Total Power"),
        ("embb_blocked_users_bar.png", "embb_blocked_users", f"eMBB={embb_users}: eMBB Blocked Users", "Blocked Users"),
        ("urllc_admitted_users_bar.png", "urllc_admitted_packets", f"eMBB={embb_users}: URLLC Admitted Packets", "Admitted Packets"),
        ("urllc_arrivals_bar.png", "urllc_arrivals", f"eMBB={embb_users}: URLLC Arrivals", "Arrivals"),
        ("urllc_blocked_users_bar.png", "urllc_blocked_users", f"eMBB={embb_users}: URLLC Blocked Users", "Blocked Users"),
        ("urllc_admission_ratio_bar.png", "urllc_admission_ratio", f"eMBB={embb_users}: URLLC Admission Ratio", "Admission Ratio"),
        ("embb_jain_bar.png", "embb_jain_fairness", f"eMBB={embb_users}: eMBB Jain Fairness", "Jain Fairness"),
        ("embb_5th_percentile_bar.png", "embb_5th_percentile_mbps", f"eMBB={embb_users}: eMBB 5th Percentile", "Rate (Mbps)"),
        ("embb_median_rate_bar.png", "embb_median_rate_mbps", f"eMBB={embb_users}: eMBB Median Rate", "Rate (Mbps)"),
    ]
    for filename, metric_key, title, ylabel in metric_specs:
        _plot_grouped_bars(folder / filename, title, ylabel, rows, metric_key, policy_labels)

    _plot_mode_ratios(folder / "mode_ratios_bar.png", rows, policy_labels)
    _plot_mode_ratios_bar_no_keep(folder / "mode_ratios_bar_no_keep.png", rows, policy_labels)
    _plot_mode_selection_lines_no_keep(folder / "mode_selection_lines_no_keep.png", rows, policy_labels)
    _plot_mode_selection_lines_with_keep(folder / "mode_selection_lines_with_keep.png", rows, policy_labels)

    power_source = folder / "power_split_source.csv"
    if power_source.exists():
        power_rows = _read_rows(power_source)
        _plot_grouped_bars(
            folder / "embb_power_bar.png",
            f"eMBB={embb_users}: eMBB Power",
            "eMBB Power",
            power_rows,
            "embb_power",
            _policy_order(power_rows),
        )
        _plot_grouped_bars(
            folder / "urllc_power_bar.png",
            f"eMBB={embb_users}: URLLC Power",
            "URLLC Power",
            power_rows,
            "urllc_power",
            _policy_order(power_rows),
        )
        _plot_grouped_bars(
            folder / "total_power_bar.png",
            f"eMBB={embb_users}: Total Power",
            "Total Power",
            power_rows,
            "total_power",
            _policy_order(power_rows),
        )

    _save_standalone_legend(folder / "legend.png", policy_labels)
    _write_note(folder / "note.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot embb_xx bar charts without inline legends.")
    parser.add_argument("plots_root", type=Path, help="Path to plots_bar_by_embb root.")
    args = parser.parse_args()

    plots_root = args.plots_root.expanduser().resolve()
    embb_dirs = sorted(path for path in plots_root.iterdir() if path.is_dir() and path.name.startswith("embb_"))
    if not embb_dirs:
        raise FileNotFoundError(f"No embb_* folders under {plots_root}")
    for folder in embb_dirs:
        _replot_folder(folder)
        print(folder)


if __name__ == "__main__":
    main()
