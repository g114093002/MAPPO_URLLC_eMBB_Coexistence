from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SERIES_COLORS = {
    "zh": "#1f77b4",
    "greedy": "#d62728",
}

DROP_BOTH_COMPONENTS = {
    "terminal_embb_power_over_greedy_penalty",
}

DROP_GREEDY_ONLY_COMPONENTS = {
    "planning_embb_rate_delta_reward",
}

DROP_PLANNING_BOTH_COMPONENTS = {
    "planning_embb_rate_delta_reward",
}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pairs_to_map(pairs: List[List[object]] | List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in pairs:
        if not row or len(row) < 2:
            continue
        out[str(row[0])] = float(row[1] or 0.0)
    return out


def _filter_map(values: Dict[str, float], *, drop: set[str]) -> Dict[str, float]:
    return {key: val for key, val in values.items() if key not in drop}


def _filtered_component_maps(
    summary: Dict[str, object],
    *,
    policy: str,
    drop_planning_for_both: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    step_map = _pairs_to_map(list(summary.get("kept_step_components_by_sum", []) or []))
    terminal_map = _pairs_to_map(list(summary.get("kept_terminal_components_by_sum", []) or []))
    step_drop = set()
    if policy == "greedy":
        step_drop |= DROP_GREEDY_ONLY_COMPONENTS
    if drop_planning_for_both:
        step_drop |= DROP_PLANNING_BOTH_COMPONENTS
    step_map = _filter_map(step_map, drop=step_drop)
    terminal_map = _filter_map(terminal_map, drop=DROP_BOTH_COMPONENTS)
    return step_map, terminal_map


def _plot_summary_bars(
    zh: Dict[str, object],
    greedy: Dict[str, object],
    *,
    zh_step_map: Dict[str, float],
    zh_terminal_map: Dict[str, float],
    greedy_step_map: Dict[str, float],
    greedy_terminal_map: Dict[str, float],
    out_path: Path,
) -> None:
    labels = [
        "visible_reward_total_sum",
        "visible_step_reward_sum",
        "visible_terminal_reward_sum",
    ]
    zh_visible_step = float(sum(zh_step_map.values()))
    gr_visible_step = float(sum(greedy_step_map.values()))
    zh_visible_terminal = float(sum(zh_terminal_map.values()))
    gr_visible_terminal = float(sum(greedy_terminal_map.values()))
    zh_total = zh_visible_step + zh_visible_terminal
    gr_total = gr_visible_step + gr_visible_terminal
    zh_vals = [
        zh_total,
        zh_visible_step,
        zh_visible_terminal,
    ]
    gr_vals = [
        gr_total,
        gr_visible_step,
        gr_visible_terminal,
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2.0, zh_vals, width=width, color=SERIES_COLORS["zh"], label="zh")
    ax.bar(x + width / 2.0, gr_vals, width=width, color=SERIES_COLORS["greedy"], label="greedy")
    ax.set_xticks(x)
    ax.set_xticklabels(["total reward", "visible step reward", "visible terminal reward"], rotation=10)
    ax.set_ylabel("Reward")
    ax.set_title("Reward Summary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_component_compare(
    *,
    left_map: Dict[str, float],
    right_map: Dict[str, float],
    left_label: str,
    right_label: str,
    title: str,
    out_path: Path,
) -> None:
    keys: List[str] = []
    seen = set()
    for key in list(left_map.keys()) + list(right_map.keys()):
        if key not in seen:
            seen.add(key)
            keys.append(key)

    rows = [
        (
            key,
            float(left_map.get(key, 0.0) or 0.0),
            float(right_map.get(key, 0.0) or 0.0),
        )
        for key in keys
    ]
    rows = [row for row in rows if abs(row[1]) > 1.0e-12 or abs(row[2]) > 1.0e-12]

    y = np.arange(len(rows), dtype=float)
    width = 0.36
    fig_h = max(4.2, 0.5 * len(rows) + 1.5)

    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.barh(
        y - width / 2.0,
        [row[1] for row in rows],
        height=width,
        color=SERIES_COLORS["zh"],
        label=left_label,
    )
    ax.barh(
        y + width / 2.0,
        [row[2] for row in rows],
        height=width,
        color=SERIES_COLORS["greedy"],
        label=right_label,
    )
    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Reward")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two lambda reward breakdown summaries.")
    parser.add_argument("--zh-summary", required=True)
    parser.add_argument("--greedy-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--drop-planning-for-both",
        action="store_true",
        help="Exclude planning_embb_rate_delta_reward from both zh and greedy comparisons.",
    )
    args = parser.parse_args()

    zh = _load_json(Path(args.zh_summary))
    greedy = _load_json(Path(args.greedy_summary))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zh_step_map, zh_terminal_map = _filtered_component_maps(
        zh,
        policy="zh",
        drop_planning_for_both=bool(args.drop_planning_for_both),
    )
    greedy_step_map, greedy_terminal_map = _filtered_component_maps(
        greedy,
        policy="greedy",
        drop_planning_for_both=bool(args.drop_planning_for_both),
    )
    zh_total = float(sum(zh_step_map.values()) + sum(zh_terminal_map.values()))
    greedy_total = float(sum(greedy_step_map.values()) + sum(greedy_terminal_map.values()))

    _plot_summary_bars(
        zh,
        greedy,
        zh_step_map=zh_step_map,
        zh_terminal_map=zh_terminal_map,
        greedy_step_map=greedy_step_map,
        greedy_terminal_map=greedy_terminal_map,
        out_path=out_dir / "reward_summary_compare.png",
    )
    _plot_component_compare(
        left_map=zh_step_map,
        right_map=greedy_step_map,
        left_label="zh",
        right_label="greedy",
        title="Step Reward Components",
        out_path=out_dir / "step_components_compare.png",
    )
    _plot_component_compare(
        left_map=zh_terminal_map,
        right_map=greedy_terminal_map,
        left_label="zh",
        right_label="greedy",
        title="Terminal Reward Components",
        out_path=out_dir / "terminal_components_compare.png",
    )

    comparison = {
        "zh_reward_total_sum": zh_total,
        "greedy_reward_total_sum": greedy_total,
        "zh_step_sum_filtered": float(sum(zh_step_map.values())),
        "greedy_step_sum_filtered": float(sum(greedy_step_map.values())),
        "zh_terminal_sum_filtered": float(sum(zh_terminal_map.values())),
        "greedy_terminal_sum_filtered": float(sum(greedy_terminal_map.values())),
        "dropped_both_components": sorted(DROP_BOTH_COMPONENTS),
        "dropped_greedy_only_components": sorted(DROP_GREEDY_ONLY_COMPONENTS),
        "drop_planning_for_both": bool(args.drop_planning_for_both),
    }
    (out_dir / "reward_breakdown_compare_summary.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
