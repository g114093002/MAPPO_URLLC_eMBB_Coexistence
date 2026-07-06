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


def _plot_combined(plots_root: Path) -> Path:
    embb_dirs = [
        plots_root / "embb_10",
        plots_root / "embb_20",
        plots_root / "embb_30",
    ]
    rows_by_embb: Dict[int, List[Dict[str, object]]] = {}
    for folder in embb_dirs:
        rows = _read_rows(folder / "plot_source_metrics.csv")
        if not rows:
            raise ValueError(f"No rows in {folder / 'plot_source_metrics.csv'}")
        embb_users = int(rows[0]["embb_users"])
        rows_by_embb[embb_users] = rows

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
                values.append(float(match.get("sum_rate_mbps", 0.0) or 0.0))
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
    axes[0].set_ylabel("Sum Rate (Mbps)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(policy_labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("eMBB Sum Rate", y=1.08)
    fig.tight_layout()

    out_path = plots_root / "sum_rate_bar_combined.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine embb_10/20/30 sum-rate bar charts into one figure.")
    parser.add_argument("plots_root", type=Path, help="Path to plots_bar_by_embb root.")
    args = parser.parse_args()
    out_path = _plot_combined(args.plots_root.expanduser().resolve())
    print(out_path)


if __name__ == "__main__":
    main()
