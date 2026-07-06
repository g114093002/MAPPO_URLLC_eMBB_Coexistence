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


METRIC_LABELS = {
    "avg_embb_rate_mbps": "Average eMBB Rate (Mbps/user)",
    "embb_rate_mbps": "Aggregate eMBB Rate (Mbps)",
    "embb_power": "eMBB Power (W)",
    "embb_minrate_satisfied_users": "Satisfied eMBB Users",
    "embb_minrate_satisfaction_ratio": "eMBB Min-Rate Satisfaction Ratio",
    "embb_power_share": "eMBB Power Share",
    "overlay_action_count": "Overlay Action Count",
    "overlay_action_ratio": "Overlay Action Ratio",
    "puncturing_action_count": "Puncturing Action Count",
    "puncturing_action_ratio": "Puncturing Action Ratio",
    "runtime_sec": "Runtime (sec)",
    "total_power": "Total Power (W)",
    "urllc_admission": "URLLC Admission Ratio",
    "urllc_admitted_packets": "URLLC Admitted Packets",
    "urllc_power": "URLLC Power (W)",
    "urllc_tp_mbps": "URLLC Throughput (Mbps)",
    "urllc_power_share": "URLLC Power Share",
}

SERIES_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
]

SERIES_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _enrich_metric_map(payload: dict, mix: str, policy_name: str, metric_map: Dict) -> Dict:
    enriched = dict(metric_map)
    raw_runs = (
        dict(payload.get("raw_runs", {}) or {})
        .get(mix, {})
        .get(policy_name, {})
    )
    lambdas = list(metric_map.get("lambda_per_user_override", []) or [])
    if not raw_runs or not lambdas:
        return enriched

    extra = {
        "embb_power": [],
        "embb_minrate_satisfied_users": [],
        "embb_minrate_satisfaction_ratio": [],
        "embb_power_share": [],
        "overlay_action_ratio": [],
        "puncturing_action_ratio": [],
        "total_power": [],
        "urllc_power": [],
        "urllc_power_share": [],
    }
    for lam in lambdas:
        per_seed_runs = list(raw_runs.get(f"lambda_{lam:g}", []) or [])
        if not per_seed_runs:
            for key in extra:
                extra[key].append(float("nan"))
            continue
        satisfied_users: List[float] = []
        satisfied_ratio: List[float] = []
        embb_power_abs: List[float] = []
        embb_power_share: List[float] = []
        overlay_action_ratio: List[float] = []
        puncturing_action_ratio: List[float] = []
        total_power_abs: List[float] = []
        urllc_power_abs: List[float] = []
        urllc_power_share: List[float] = []
        for run in per_seed_runs:
            summary = dict(run.get("raw_summary", {}) or {})
            ratio = float(summary.get("embb_min_rate_satisfaction_ratio", 0.0) or 0.0)
            embb_user_count = float(summary.get("embb_user_count", 0.0) or 0.0)
            satisfied_users.append(ratio * embb_user_count)
            satisfied_ratio.append(ratio)

            total_power = float(summary.get("total_power", 0.0) or 0.0)
            embb_power = float(summary.get("embb_power", 0.0) or 0.0)
            urllc_power = float(summary.get("urllc_power", 0.0) or 0.0)
            overlay_count = float(summary.get("overlay_action_count", 0.0) or 0.0)
            puncture_count = float(summary.get("puncturing_action_count", 0.0) or 0.0)
            coexist_count = overlay_count + puncture_count
            total_power_abs.append(total_power)
            embb_power_abs.append(embb_power)
            urllc_power_abs.append(urllc_power)
            if coexist_count > 1.0e-12:
                overlay_action_ratio.append(overlay_count / coexist_count)
                puncturing_action_ratio.append(puncture_count / coexist_count)
            else:
                overlay_action_ratio.append(float("nan"))
                puncturing_action_ratio.append(float("nan"))
            if total_power > 1.0e-12:
                embb_power_share.append(embb_power / total_power)
                urllc_power_share.append(urllc_power / total_power)
            else:
                embb_power_share.append(float("nan"))
                urllc_power_share.append(float("nan"))
        extra["embb_power"].append(float(np.nanmean(np.asarray(embb_power_abs, dtype=float))))
        extra["embb_minrate_satisfied_users"].append(float(np.nanmean(np.asarray(satisfied_users, dtype=float))))
        extra["embb_minrate_satisfaction_ratio"].append(float(np.nanmean(np.asarray(satisfied_ratio, dtype=float))))
        extra["embb_power_share"].append(float(np.nanmean(np.asarray(embb_power_share, dtype=float))))
        extra["overlay_action_ratio"].append(float(np.nanmean(np.asarray(overlay_action_ratio, dtype=float))))
        extra["puncturing_action_ratio"].append(float(np.nanmean(np.asarray(puncturing_action_ratio, dtype=float))))
        extra["total_power"].append(float(np.nanmean(np.asarray(total_power_abs, dtype=float))))
        extra["urllc_power"].append(float(np.nanmean(np.asarray(urllc_power_abs, dtype=float))))
        extra["urllc_power_share"].append(float(np.nanmean(np.asarray(urllc_power_share, dtype=float))))

    enriched.update(extra)
    overlay_counts = np.asarray(enriched.get("overlay_action_count", []), dtype=float)
    puncture_counts = np.asarray(enriched.get("puncturing_action_count", []), dtype=float)
    if overlay_counts.size == puncture_counts.size and overlay_counts.size > 0:
        denom = overlay_counts + puncture_counts
        with np.errstate(invalid="ignore", divide="ignore"):
            overlay_ratio = np.where(denom > 1.0e-12, overlay_counts / denom, np.nan)
            puncture_ratio = np.where(denom > 1.0e-12, puncture_counts / denom, np.nan)
        if (
            "overlay_action_ratio" not in enriched
            or not np.any(np.isfinite(np.asarray(enriched.get("overlay_action_ratio", []), dtype=float)))
        ):
            enriched["overlay_action_ratio"] = overlay_ratio.tolist()
        if (
            "puncturing_action_ratio" not in enriched
            or not np.any(np.isfinite(np.asarray(enriched.get("puncturing_action_ratio", []), dtype=float)))
        ):
            enriched["puncturing_action_ratio"] = puncture_ratio.tolist()
    return enriched


def _collect_series(
    result_dirs: List[Path],
    mix: str,
    custom_labels: List[str] | None = None,
) -> Tuple[List[Dict], List[str]]:
    series_bundle: List[Dict] = []
    notes: List[str] = []
    resolved_labels = list(custom_labels or [])
    for result_dir_idx, result_dir in enumerate(result_dirs):
        metrics_path = result_dir / "unified_lambda_stress_metrics.json"
        payload = _load_metrics(metrics_path)
        mix_map = dict(payload.get("series", {}) or {})
        if mix not in mix_map:
            notes.append(f"{result_dir.name}: missing mix {mix}, skipped")
            continue
        policy_map = dict(mix_map[mix] or {})
        if len(policy_map) != 1:
            notes.append(f"{result_dir.name}: contains {len(policy_map)} policies {list(policy_map)}")
        for policy_name, metric_map in policy_map.items():
            label = (
                resolved_labels[result_dir_idx]
                if result_dir_idx < len(resolved_labels)
                else result_dir.name
            )
            if len(policy_map) > 1 or result_dir.name.lower() == policy_name.lower():
                label = f"{result_dir.name}:{policy_name}"
            metric_map = _enrich_metric_map(payload, mix, policy_name, metric_map)
            series_bundle.append(
                {
                    "label": label,
                    "result_dir": str(result_dir),
                    "policy": policy_name,
                    "metrics_path": str(metrics_path),
                    "metrics": metric_map,
                }
            )
    return series_bundle, notes


def _common_metric_keys(series_bundle: List[Dict]) -> List[str]:
    if not series_bundle:
        return []
    metric_sets = [{key for key in entry["metrics"] if key != "lambda_per_user_override"} for entry in series_bundle]
    common = set.intersection(*metric_sets)
    ordered = [key for key in METRIC_LABELS if key in common]
    for key in sorted(common):
        if key not in ordered:
            ordered.append(key)
    return ordered


def _series_xy(entry: Dict, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(entry["metrics"].get("lambda_per_user_override", []), dtype=float)
    y = np.asarray(entry["metrics"].get(metric_key, []), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


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
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_dashboard(out_path: Path, series_bundle: List[Dict], metric_keys: List[str]) -> None:
    n_metrics = len(metric_keys)
    n_cols = 3
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.0 * n_rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)

    legend_handles = None
    legend_labels = None
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
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for idx_metric in range(n_metrics, n_rows * n_cols):
        axes_arr[idx_metric // n_cols, idx_metric % n_cols].axis("off")

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(4, len(legend_labels)),
            frameon=False,
        )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_inventory(out_path: Path, series_bundle: List[Dict], metric_keys: List[str], notes: List[str]) -> None:
    payload = {
        "series": [],
        "notes": notes,
        "common_metrics": metric_keys,
    }
    for entry in series_bundle:
        payload["series"].append(
            {
                "label": entry["label"],
                "policy": entry["policy"],
                "result_dir": entry["result_dir"],
                "metrics_path": entry["metrics_path"],
                "lambda_points": len(entry["metrics"].get("lambda_per_user_override", [])),
            }
        )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple unified lambda-stress result folders on one plot.")
    parser.add_argument("--result-dirs", nargs="+", required=True, help="Result directories containing unified_lambda_stress_metrics.json")
    parser.add_argument("--mix", default="5:5", help="Mix key to compare, default 5:5")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional custom labels aligned with --result-dirs.")
    args = parser.parse_args()

    result_dirs = [Path(item).resolve() for item in args.result_dirs]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(args.labels or [])
    if labels and len(labels) != len(result_dirs):
        raise SystemExit("--labels count must match --result-dirs count")

    series_bundle, notes = _collect_series(result_dirs, args.mix, custom_labels=labels)
    if not series_bundle:
        raise SystemExit("No comparable series found.")

    metric_keys = _common_metric_keys(series_bundle)
    _plot_dashboard(out_dir / "dashboard_all_metrics.png", series_bundle, metric_keys)
    for metric_key in metric_keys:
        _plot_metric(out_dir / f"{metric_key}.png", series_bundle, metric_key)
    _write_inventory(out_dir / "series_inventory.json", series_bundle, metric_keys, notes)
    print(f"wrote comparison plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
