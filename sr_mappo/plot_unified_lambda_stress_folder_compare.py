from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_FOLDERS = [
    "unified_lambda_stress_mix55_with_mappo_pure_punturing",
    "unified_lambda_stress_mix55_with_mappo_greedy",
    "unified_lambda_stress_mix55_with_mappo_pure_superposition",
    "unified_lambda_stress_mix55_mappo_v16",
    "unified_lambda_stress_mix55_mappo_v15",
    "unified_lambda_stress_mix55_with_ippo_per_user_loss",
    "unified_lambda_stress_mix55_with_mappo",
]

METRIC_LABELS = {
    "avg_embb_rate_mbps": "Average eMBB Rate (Mbps/user)",
    "embb_rate_mbps": "Aggregate eMBB Rate (Mbps)",
    "episode_reward_total_sum": "Episode Reward Total Sum",
    "episode_reward_mean": "Episode Reward Mean",
    "embb_power": "eMBB Power (W)",
    "embb_minrate_satisfied_users": "eMBB Min-Rate Satisfied Users",
    "embb_minrate_satisfaction_ratio": "eMBB Min-Rate Satisfaction Ratio",
    "embb_power_share": "eMBB Power Share",
    "num_feasible_modes_for_selected_pair": "Feasible Modes for Selected Pair",
    "mode_regret_applied_count": "Mode-Regret Applied Count",
    "mode_regret_zero_count": "Mode-Regret Zero Count",
    "mean_mode_regret": "Mean Mode-Regret",
    "overlay_action_count": "Overlay Action Count",
    "overlay_action_ratio": "Overlay Action Ratio",
    "puncturing_action_count": "Puncturing Action Count",
    "puncturing_action_ratio": "Puncturing Action Ratio",
    "runtime_sec": "Runtime (sec)",
    "total_power": "Total Power (W)",
    "logged_step_reward_sum": "Logged Step Reward Sum",
    "logged_terminal_reward_sum": "Logged Terminal Reward Sum",
    "reward_residual_unlogged": "Unlogged Reward Residual",
    "urllc_admission": "URLLC Admission Ratio",
    "urllc_admitted_packets": "URLLC Admitted Packets",
    "urllc_power": "URLLC Power (W)",
    "urllc_power_share": "URLLC Power Share",
    "urllc_tp_mbps": "URLLC Throughput (Mbps)",
}

SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

DISPLAY_LABEL_OVERRIDES = {
    "mappo": "MAPPO",
    "hard_feasible_throughput_greedy": "Greedy",
    "pure_puncturing": "Pure Puncture",
    "pure_superposition": "Pure Overlay",
    "random_scheduler": "Random Scheduler",
}


def _sanitize_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _label_for_folder_policy(folder: str, policy: str, index_within_folder: int, total_in_folder: int) -> str:
    if policy in DISPLAY_LABEL_OVERRIDES:
        return DISPLAY_LABEL_OVERRIDES[policy]
    short_map = {
        "unified_lambda_stress_mix55_with_mappo_pure_punturing": "pure_puncturing",
        "unified_lambda_stress_mix55_with_mappo_greedy": "greedy",
        "unified_lambda_stress_mix55_with_mappo_pure_superposition": "pure_superposition",
        "unified_lambda_stress_mix55_mappo_v16": "mappo_v16",
        "unified_lambda_stress_mix55_mappo_v15": "mappo_v15",
        "unified_lambda_stress_mix55_with_ippo_per_user_loss": "ippo_per_user_loss",
        "unified_lambda_stress_mix55_with_mappo": "with_mappo",
    }
    folder_label = short_map.get(folder, folder)
    if total_in_folder == 1 and folder_label == policy:
        return folder_label
    if total_in_folder == 1 and policy in {"mappo", "ippo", "greedy", "pure_puncturing", "pure_superposition"}:
        return folder_label
    if folder_label == "with_mappo":
        return f"{folder_label}:{policy}"
    if total_in_folder > 1:
        return f"{folder_label}:{policy}"
    if folder_label.endswith(policy):
        return folder_label
    return f"{folder_label}:{policy}" if index_within_folder > 0 else folder_label


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_series(results_root: Path, folders: List[str]) -> Tuple[List[Dict], List[str]]:
    collected: List[Dict] = []
    notes: List[str] = []
    for folder in folders:
        metrics_path = results_root / folder / "unified_lambda_stress_metrics.json"
        payload = _load_metrics(metrics_path)
        mix_map = payload.get("series", {})
        if "5:5" not in mix_map:
            notes.append(f"{folder}: missing 5:5 mix, skipped")
            continue
        policies = list(mix_map["5:5"].keys())
        for idx, policy in enumerate(policies):
            metric_map = mix_map["5:5"][policy]
            collected.append(
                {
                    "folder": folder,
                    "policy": policy,
                    "label": _label_for_folder_policy(folder, policy, idx, len(policies)),
                    "metrics_path": str(metrics_path),
                    "metrics": metric_map,
                }
            )
        if len(policies) != 1:
            notes.append(f"{folder}: contains {len(policies)} series {policies}")
    return collected, notes


def _common_metric_keys(series_bundle: List[Dict]) -> List[str]:
    metric_sets = []
    for entry in series_bundle:
        metric_sets.append({key for key in entry["metrics"].keys() if key != "lambda_per_user_override"})
    common = set.intersection(*metric_sets) if metric_sets else set()
    ordered = [key for key in METRIC_LABELS.keys() if key in common]
    for key in sorted(common):
        if key not in ordered:
            ordered.append(key)
    return ordered


def _series_xy(entry: Dict, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(entry["metrics"]["lambda_per_user_override"], dtype=float)
    y = np.asarray(entry["metrics"][metric_key], dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    return x[finite_mask], y[finite_mask]


def _style_axis(ax, metric_key: str) -> None:
    ax.set_xlabel("Lambda per URLLC user")
    ax.set_ylabel(METRIC_LABELS.get(metric_key, metric_key))
    ax.set_title(METRIC_LABELS.get(metric_key, metric_key), fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.8)


def _plot_metric(out_path: Path, series_bundle: List[Dict], metric_key: str) -> None:
    plt.figure(figsize=(10.5, 6.2))
    for idx, entry in enumerate(series_bundle):
        x, y = _series_xy(entry, metric_key)
        if x.size == 0:
            continue
        plt.plot(
            x,
            y,
            label=entry["label"],
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[idx % len(SERIES_MARKERS)],
            linewidth=2.0,
            markersize=5.5,
        )
    _style_axis(plt.gca(), metric_key)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_dashboard(out_path: Path, series_bundle: List[Dict], metric_keys: List[str]) -> None:
    n_metrics = len(metric_keys)
    n_cols = 3
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.0 * n_rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx_metric, metric_key in enumerate(metric_keys):
        ax = axes_arr[idx_metric // n_cols, idx_metric % n_cols]
        for idx_series, entry in enumerate(series_bundle):
            x, y = _series_xy(entry, metric_key)
            if x.size == 0:
                continue
            ax.plot(
                x,
                y,
                label=entry["label"],
                color=SERIES_COLORS[idx_series % len(SERIES_COLORS)],
                marker=SERIES_MARKERS[idx_series % len(SERIES_MARKERS)],
                linewidth=1.8,
                markersize=4.8,
            )
        _style_axis(ax, metric_key)

    for idx_metric in range(n_metrics, n_rows * n_cols):
        axes_arr[idx_metric // n_cols, idx_metric % n_cols].axis("off")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_legend_only(out_path: Path, series_bundle: List[Dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 1.6))
    handles = []
    labels = []
    for idx, entry in enumerate(series_bundle):
        handle, = ax.plot(
            [],
            [],
            label=entry["label"],
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[idx % len(SERIES_MARKERS)],
            linewidth=2.0,
            markersize=6.0,
        )
        handles.append(handle)
        labels.append(entry["label"])
    ax.axis("off")
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=max(1, min(5, len(labels))),
        frameon=False,
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_inventory(out_path: Path, series_bundle: List[Dict], metric_keys: List[str], notes: List[str]) -> None:
    inventory = {
        "series": [],
        "notes": notes,
        "common_metrics": metric_keys,
    }
    for entry in series_bundle:
        row = {
            "label": entry["label"],
            "folder": entry["folder"],
            "policy": entry["policy"],
            "metrics_path": entry["metrics_path"],
            "lambda_points": len(entry["metrics"].get("lambda_per_user_override", [])),
            "lambda_max": (
                max(entry["metrics"].get("lambda_per_user_override", []))
                if entry["metrics"].get("lambda_per_user_override")
                else None
            ),
        }
        inventory["series"].append(row)
    out_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay unified lambda stress metrics across result folders.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Root folder containing result subdirectories.",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=DEFAULT_RESULT_FOLDERS,
        help="Result folders to compare.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "results"
            / "unified_lambda_stress_mix55_folder_compare"
        ),
        help="Output directory for generated plots.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    series_bundle, notes = _collect_series(results_root, list(args.folders))
    metric_keys = _common_metric_keys(series_bundle)

    for metric_key in metric_keys:
        out_path = out_dir / f"{_sanitize_filename(metric_key)}.png"
        _plot_metric(out_path, series_bundle, metric_key)

    _plot_dashboard(out_dir / "dashboard_all_metrics.png", series_bundle, metric_keys)
    _plot_legend_only(out_dir / "legend_only.png", series_bundle)
    _write_inventory(out_dir / "series_inventory.json", series_bundle, metric_keys, notes)

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
