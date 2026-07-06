from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

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
            parsed: Dict[str, object] = {
                str(key).strip().lstrip("\ufeff"): value
                for key, value in dict(row).items()
            }
            for key in ("embb_users", "urllc_users", "packet_bits"):
                if key in parsed and parsed[key] not in (None, "", "None"):
                    parsed[key] = int(round(float(str(parsed[key]))))
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


def _policy_order(rows: List[Dict[str, object]]) -> List[str]:
    ordered: "OrderedDict[str, None]" = OrderedDict()
    for row in rows:
        ordered[str(row["policy_label"])] = None
    return list(ordered.keys())


def _legend_style_map(policy_labels: List[str]) -> Dict[str, str]:
    palette = {
        "Greedy": "#4C78A8",
        "Pure Puncturing": "#F58518",
        "Pure Superposition": "#54A24B",
        "Random Scheduler": "#B279A2",
        "MAPPO": "#E45756",
    }
    fallback = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    return {
        label: palette.get(label, fallback[idx % len(fallback)])
        for idx, label in enumerate(policy_labels)
    }


METRIC_SPECS: Dict[str, Tuple[str, str, str, str]] = {
    "total_power": ("power_split_source.csv", "total_power", "Total Power", "Total Power"),
    "embb_power": ("power_split_source.csv", "embb_power", "eMBB Power", "eMBB Power"),
    "urllc_power": ("power_split_source.csv", "urllc_power", "URLLC Power", "URLLC Power"),
    "embb_blocked_users": ("plot_source_metrics.csv", "embb_blocked_users", "eMBB Blocked Users", "Blocked Users"),
    "urllc_blocked_users": ("plot_source_metrics.csv", "urllc_blocked_users", "URLLC Blocked Users", "Blocked Users"),
}


def _load_rows_by_embb(plots_root: Path, filename: str) -> Dict[int, List[Dict[str, object]]]:
    rows_by_embb: Dict[int, List[Dict[str, object]]] = {}
    for embb_users in (10, 20, 30):
        path = plots_root / f"embb_{embb_users}" / filename
        rows = _read_rows(path)
        if not rows:
            raise ValueError(f"No rows in {path}")
        rows_by_embb[embb_users] = rows
    return rows_by_embb


def _plot_metric(plots_root: Path, metric_name: str) -> Path:
    filename, metric_key, title, ylabel = METRIC_SPECS[metric_name]
    rows_by_embb = _load_rows_by_embb(plots_root, filename)
    all_rows: List[Dict[str, object]] = []
    for embb_users in sorted(rows_by_embb):
        all_rows.extend(rows_by_embb[embb_users])
    policy_labels = _policy_order(all_rows)
    color_map = _legend_style_map(policy_labels)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, embb_users in zip(axes, [10, 20, 30]):
        rows = rows_by_embb[embb_users]
        urllc_values = sorted({int(row["urllc_users"]) for row in rows})
        x = np.arange(len(urllc_values), dtype=float)
        width = 0.8 / max(len(policy_labels), 1)
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
                label=label,
            )
        ax.set_title(f"eMBB = {embb_users}")
        ax.set_xlabel("URLLC Users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in urllc_values])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(policy_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(title, y=1.08)
    fig.tight_layout()

    out_path = plots_root / f"{metric_name}_bar_combined.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine embb_10/20/30 bar charts for selected metrics.")
    parser.add_argument("plots_root", type=Path, help="Path to plots_bar_by_embb root.")
    parser.add_argument(
        "--metrics",
        default="total_power,embb_power,urllc_power,embb_blocked_users,urllc_blocked_users",
        help="Comma-separated metric names.",
    )
    args = parser.parse_args()

    plots_root = args.plots_root.expanduser().resolve()
    metric_names = [item.strip() for item in str(args.metrics).split(",") if item.strip()]
    for metric_name in metric_names:
        if metric_name not in METRIC_SPECS:
            raise ValueError(f"Unsupported metric: {metric_name}")
        print(_plot_metric(plots_root, metric_name))


if __name__ == "__main__":
    main()
