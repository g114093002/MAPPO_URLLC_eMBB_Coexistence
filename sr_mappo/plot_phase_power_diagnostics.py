from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_lambda_keys(raw_runs: Dict[str, list]) -> List[str]:
    return sorted(raw_runs.keys(), key=lambda key: float(str(key).split("_")[1]))


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _extract_series(metrics_path: Path, mix: str, policy: str) -> List[dict]:
    payload = _load_metrics(metrics_path)
    raw_runs = payload["raw_runs"][mix][policy]
    points: List[dict] = []
    for lambda_key in _sorted_lambda_keys(raw_runs):
        lam = float(lambda_key.split("_")[1])
        runs = raw_runs[lambda_key]
        phase0_values = []
        final_values = []
        puncture_values = []
        overlay_values = []
        phase_a_mean_delta_values = []
        for run in runs:
            summary = run.get("raw_summary", {})
            phase0 = summary.get("phase0_executed_embb_total_power")
            if phase0 is None:
                phase0 = summary.get("phase0_embb_power_mean")
            final = summary.get("embb_power")
            puncture_ratio = summary.get("phase_a_selected_puncture_ratio")
            overlay_ratio = summary.get("phase_a_selected_overlay_ratio")
            phase_a_mean_delta = summary.get("phase_a_embb_power_mean_executed_delta")
            if phase0 is None or final is None:
                continue
            phase0_values.append(float(phase0))
            final_values.append(float(final))
            puncture_values.append(float(puncture_ratio or 0.0))
            overlay_values.append(float(overlay_ratio or 0.0))
            phase_a_mean_delta_values.append(float(phase_a_mean_delta or 0.0))
        if not phase0_values:
            continue
        phase0_mean = _mean(phase0_values)
        final_mean = _mean(final_values)
        points.append(
            {
                "lambda": lam,
                "phase0_embb_power": phase0_mean,
                "final_embb_power_after_phase_a": final_mean,
                "phase_a_embb_power_delta": final_mean - phase0_mean,
                "phase_a_embb_power_mean_executed_delta": _mean(phase_a_mean_delta_values),
                "puncture_ratio": _mean(puncture_values),
                "overlay_ratio": _mean(overlay_values),
                "seed_count": len(phase0_values),
            }
        )
    return points


def _extract_greedy_series(metrics_path: Path, mix: str, policy: str) -> List[dict]:
    payload = _load_metrics(metrics_path)
    raw_runs = payload["raw_runs"][mix][policy]
    points: List[dict] = []
    for lambda_key in _sorted_lambda_keys(raw_runs):
        lam = float(lambda_key.split("_")[1])
        runs = raw_runs[lambda_key]
        embb_values = []
        for run in runs:
            summary = run.get("raw_summary", {})
            embb = summary.get("embb_power")
            if embb is not None:
                embb_values.append(float(embb))
        if embb_values:
            points.append({"lambda": lam, "greedy_embb_power": _mean(embb_values), "seed_count": len(embb_values)})
    return points


def _align_series(mappo_points: List[dict], greedy_points: List[dict]) -> List[dict]:
    greedy_by_lambda = {point["lambda"]: point for point in greedy_points}
    merged = []
    for point in mappo_points:
        merged_point = dict(point)
        greedy = greedy_by_lambda.get(point["lambda"])
        merged_point["greedy_embb_power"] = None if greedy is None else greedy["greedy_embb_power"]
        merged.append(merged_point)
    return merged


def _plot_phase_power(series: List[dict], out_path: Path) -> None:
    x = [point["lambda"] for point in series]
    plt.figure(figsize=(8.8, 5.4))
    plt.plot(x, [point["phase0_embb_power"] for point in series], marker="o", linewidth=2.2, label="phase0 embb power")
    plt.plot(
        x,
        [point["final_embb_power_after_phase_a"] for point in series],
        marker="o",
        linewidth=2.2,
        label="final embb power after phase a",
    )
    plt.plot(x, [point["greedy_embb_power"] for point in series], marker="o", linewidth=2.2, label="greedy embb power")
    plt.xlabel("lambda")
    plt.ylabel("eMBB power")
    plt.title("Phase-0 vs Phase-A eMBB Power")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_phase_delta(series: List[dict], out_path: Path) -> None:
    x = [point["lambda"] for point in series]
    y = [point["phase_a_embb_power_delta"] for point in series]
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(x, y, marker="o", linewidth=2.2, color="#c0392b")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    plt.xlabel("lambda")
    plt.ylabel("final embb power - phase0 embb power")
    plt.title("Phase-A eMBB Power Delta")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_mode_ratio_vs_delta(series: List[dict], out_path: Path) -> None:
    x = [point["lambda"] for point in series]
    delta = [point["phase_a_embb_power_delta"] for point in series]
    puncture = [point["puncture_ratio"] for point in series]
    overlay = [point["overlay_ratio"] for point in series]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(x, delta, marker="o", linewidth=2.2, color="#c0392b", label="phase a embb power delta")
    ax1.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("phase a embb power delta")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, puncture, marker="s", linewidth=2.0, color="#1f77b4", label="puncture ratio")
    ax2.plot(x, overlay, marker="^", linewidth=2.0, color="#2ca02c", label="overlay ratio")
    ax2.set_ylabel("mode ratio")
    ax2.set_ylim(0.0, 1.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.title("Mode Ratio vs Phase-A eMBB Power Delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(series: List[dict], out_path: Path) -> None:
    plt.figure(figsize=(6.8, 5.2))
    x = [point["puncture_ratio"] for point in series]
    y = [point["phase_a_embb_power_delta"] for point in series]
    labels = [point["lambda"] for point in series]
    plt.scatter(x, y, s=52, color="#1f77b4")
    for x0, y0, lam in zip(x, y, labels):
        plt.annotate(str(lam), (x0, y0), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    plt.xlabel("puncture ratio")
    plt.ylabel("phase a embb power delta")
    plt.title("Puncture Ratio vs Phase-A Delta")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappo-metrics", required=True)
    parser.add_argument("--greedy-metrics", required=True)
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--mappo-policy", default="mappo")
    parser.add_argument("--greedy-policy", default="greedy")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mappo_points = _extract_series(Path(args.mappo_metrics), args.mix, args.mappo_policy)
    greedy_points = _extract_greedy_series(Path(args.greedy_metrics), args.mix, args.greedy_policy)
    merged = _align_series(mappo_points, greedy_points)

    _plot_phase_power(merged, out_dir / "phase0_vs_phasea_vs_greedy_embb_power.png")
    _plot_phase_delta(merged, out_dir / "phasea_embb_power_delta_by_lambda.png")
    _plot_mode_ratio_vs_delta(merged, out_dir / "mode_ratio_vs_phasea_delta_by_lambda.png")
    _plot_scatter(merged, out_dir / "puncture_ratio_vs_phasea_delta_scatter.png")

    summary = {"series": merged}
    (out_dir / "phase_power_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PHASE-POWER] wrote diagnostics to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
