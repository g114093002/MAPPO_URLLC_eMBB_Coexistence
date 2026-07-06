from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key, value in row.items():
                clean_key = str(key).strip().lstrip("\ufeff")
                if clean_key in {"policy", "policy_label"}:
                    parsed[clean_key] = value
                    continue
                try:
                    parsed[clean_key] = float(value) if value not in ("", None, "None") else 0.0
                except Exception:
                    parsed[clean_key] = value
            rows.append(parsed)
        return rows


def _policy_order(rows: List[Dict[str, object]]) -> List[str]:
    ordered: "OrderedDict[str, None]" = OrderedDict()
    preferred = [
        "Greedy",
        "Pure Puncturing",
        "Pure Superposition",
        "Random Scheduler",
        "MAPPO",
    ]
    existing = {str(row.get("policy_label", "")) for row in rows}
    for label in preferred:
        if label in existing:
            ordered[label] = None
    for row in rows:
        ordered[str(row.get("policy_label", ""))] = None
    return list(ordered.keys())


def _policy_colors() -> Dict[str, str]:
    return {
        "Greedy": "#4C78A8",
        "Pure Puncturing": "#F58518",
        "Pure Superposition": "#54A24B",
        "Random Scheduler": "#B279A2",
        "MAPPO": "#E45756",
    }


def _annotate_segment(ax: plt.Axes, x: float, bottom: float, height: float) -> None:
    if height <= 0.0:
        return
    text_y = bottom + height / 2.0
    text_color = "white" if height >= 9.0 else "black"
    ax.text(
        x,
        text_y,
        f"{height:.1f}%",
        ha="center",
        va="center",
        fontsize=9,
        color=text_color,
    )


def _plot_no_keep(
    csv_path: Path,
    out_path: Path,
    fixed_label: str,
    title: str,
    ylabel: str,
) -> None:
    rows = _read_rows(csv_path)
    fixed_values = sorted({int(row["fixed_value"]) for row in rows})
    policy_labels = _policy_order(rows)
    colors = _policy_colors()

    fig, axes = plt.subplots(1, len(fixed_values), figsize=(6.0 * len(fixed_values), 5.0), sharey=True)
    if len(fixed_values) == 1:
        axes = [axes]

    for ax, fixed_value in zip(axes, fixed_values):
        subset = [row for row in rows if int(row["fixed_value"]) == fixed_value]
        x = np.arange(len(policy_labels), dtype=float)
        overlay_values: List[float] = []
        puncture_values: List[float] = []
        for label in policy_labels:
            match = next(row for row in subset if str(row["policy_label"]) == label)
            overlay = float(match.get("overlay_ratio_pct", 0.0) or 0.0)
            puncture = float(match.get("puncture_ratio_pct", 0.0) or 0.0)
            active = overlay + puncture
            overlay_norm = 100.0 * overlay / active if active > 1.0e-12 else 0.0
            puncture_norm = 100.0 * puncture / active if active > 1.0e-12 else 0.0
            overlay_values.append(overlay_norm)
            puncture_values.append(puncture_norm)

        ax.bar(
            x,
            overlay_values,
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.6,
            label="Overlay",
        )
        ax.bar(
            x,
            puncture_values,
            bottom=overlay_values,
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
            label="Puncture",
        )

        for idx, (overlay, puncture) in enumerate(zip(overlay_values, puncture_values)):
            _annotate_segment(ax, x[idx], 0.0, overlay)
            _annotate_segment(ax, x[idx], overlay, puncture)

        tick_colors = [colors.get(label, "black") for label in policy_labels]
        ax.set_title(f"{fixed_label} = {fixed_value}")
        ax.set_xticks(x)
        ax.set_xticklabels(policy_labels, rotation=20)
        for tick, color in zip(ax.get_xticklabels(), tick_colors):
            tick.set_color(color)
        ax.set_ylim(0.0, 100.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = OrderedDict()
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    fig.suptitle(title, y=1.02)
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=len(dedup), frameon=False)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot mode-average no-keep charts with percentage labels.")
    parser.add_argument("plots_root", type=Path, help="Path to plots_bar_by_embb root.")
    args = parser.parse_args()

    plots_root = args.plots_root.expanduser().resolve()
    _plot_no_keep(
        plots_root / "mode_average_by_embb.csv",
        plots_root / "mode_average_by_embb_no_keep.png",
        fixed_label="eMBB",
        title="URLLC Coexistence Mode Selection",
        ylabel="Mode ratio",
    )
    urllc_csv = plots_root / "mode_average_by_urllc.csv"
    if urllc_csv.exists():
        _plot_no_keep(
            urllc_csv,
            plots_root / "mode_average_by_urllc_no_keep.png",
            fixed_label="URLLC",
            title="URLLC Coexistence Mode Selection",
            ylabel="Mode ratio",
        )
    print(plots_root)


if __name__ == "__main__":
    main()
