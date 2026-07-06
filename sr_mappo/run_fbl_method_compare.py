from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .run_fbl_reliability_sweep import _build_policy_config
from .run_unified_lambda_stress import _parse_csv_floats, _parse_csv_strings
from .unified_policy_runner import MIX_PRESETS, run_policy


DEFAULT_POLICIES = [
    "mappo",
    "greedy",
    "pure_superposition",
    "pure_puncturing",
    "naive_random",
]

POLICY_LABELS = {
    "mappo": "MAPPO",
    "greedy": "Greedy",
    "pure_superposition": "Overlay-only",
    "pure_puncturing": "Puncturing-only",
    "naive_random": "Naive Random",
}

METRICS = [
    "admitted_urllc_count",
    "total_urllc_arrivals",
    "urllc_admission_ratio",
    "urllc_reliability_violation_count",
    "urllc_reliability_violation_ratio",
    "embb_min_rate_violation_count",
    "embb_min_rate_violation_ratio",
    "safe_admitted_urllc_count",
    "safe_admission_ratio",
    "total_embb_throughput",
    "average_embb_throughput",
    "average_power_consumption",
]


def _mean_std_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bar_plot(path: Path, title: str, ylabel: str, methods: List[str], means: List[float], ci95: List[float]) -> None:
    xs = np.arange(len(methods))
    plt.figure(figsize=(9.0, 5.8))
    plt.bar(xs, means, yerr=ci95, capsize=6, alpha=0.9)
    plt.xticks(xs, methods, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _scatter_tradeoff(path: Path, methods: List[str], xvals: List[float], yvals: List[float]) -> None:
    plt.figure(figsize=(7.4, 6.0))
    for method, x, y in zip(methods, xvals, yvals):
        plt.scatter([x], [y], s=90, label=method)
        plt.annotate(method, (x, y), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("Admitted URLLC packet count")
    plt.ylabel("eMBB min-rate violation ratio")
    plt.title("Fig. 4  Admission-Violation Tradeoff")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-condition finite-blocklength method comparison.")
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--load", type=float, default=24.0)
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--target-error-probability", type=float, default=1e-5)
    parser.add_argument("--channel-uses", type=int, default=32)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default="sr_mappo/results/fbl_method_compare")
    parser.add_argument("--mappo-checkpoint-path", default=None)
    args = parser.parse_args()

    policies = _parse_csv_strings(args.policies)
    seeds = [int(round(x)) for x in _parse_csv_floats(args.seeds)]
    if str(args.mix) not in MIX_PRESETS:
        raise ValueError(f"Unsupported mix={args.mix!r}. Allowed={sorted(MIX_PRESETS)}")
    mix_ratio = float(MIX_PRESETS[str(args.mix)])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, object]] = []
    by_method: Dict[str, List[Dict[str, object]]] = {policy: [] for policy in policies}

    for policy in policies:
        for seed in seeds:
            cfg = _build_policy_config(
                args,
                policy=policy,
                total_load=float(args.load),
                mix_ratio=mix_ratio,
                lam=float(args.lambda_value),
                packet_bits=int(args.packet_bits),
                channel_uses=int(args.channel_uses),
                target_error_probability=float(args.target_error_probability),
            )
            result = run_policy(policy, cfg, int(seed))
            row = {
                "method": POLICY_LABELS.get(policy, policy),
                "seed": int(seed),
                "episode": 1,
                "packet_bits": int(args.packet_bits),
                "target_error_probability": float(args.target_error_probability),
                "channel_uses": int(args.channel_uses),
                "mix": str(args.mix),
                "load": float(args.load),
                "lambda": float(args.lambda_value),
                "total_urllc_arrivals": float(result.get("total_urllc_arrivals", 0.0) or 0.0),
                "admitted_urllc_count": float(result.get("admitted_urllc_count", 0.0) or 0.0),
                "urllc_admission_ratio": float(result.get("urllc_admission_ratio", 0.0) or 0.0),
                "urllc_reliability_violation_count": float(result.get("urllc_reliability_violation_count", 0.0) or 0.0),
                "urllc_reliability_violation_ratio": float(result.get("urllc_reliability_violation_ratio", 0.0) or 0.0),
                "embb_min_rate_violation_count": float(result.get("embb_minimum_rate_violation_count", 0.0) or 0.0),
                "embb_min_rate_violation_ratio": float(result.get("embb_min_rate_violation_ratio", 0.0) or 0.0),
                "safe_admitted_urllc_count": float(result.get("safe_admitted_urllc_count", 0.0) or 0.0),
                "safe_admission_ratio": float(result.get("safe_admission_ratio", 0.0) or 0.0),
                "total_embb_throughput": float(result.get("total_embb_throughput", 0.0) or 0.0),
                "average_embb_throughput": float(result.get("average_embb_rate", 0.0) or 0.0),
                "average_power_consumption": float(result.get("average_power_consumption", 0.0) or 0.0),
            }
            per_run_rows.append(row)
            by_method[policy].append(row)
            print(
                f"[FBL-METHOD-COMPARE] method={row['method']} seed={seed} "
                f"adm={row['admitted_urllc_count']:.2f} safe={row['safe_admitted_urllc_count']:.2f} "
                f"embb_viol={row['embb_min_rate_violation_ratio']:.4f}",
                flush=True,
            )

    per_run_csv = out_dir / "per_run_summary.csv"
    _write_csv(per_run_csv, per_run_rows, list(per_run_rows[0].keys()))

    aggregated_rows: List[Dict[str, object]] = []
    methods = [POLICY_LABELS.get(p, p) for p in policies]
    fig1_means, fig1_ci95 = [], []
    fig2_means, fig2_ci95 = [], []
    fig3_means, fig3_ci95 = [], []
    fig4_x, fig4_y = [], []
    for policy in policies:
        rows = by_method[policy]
        label = POLICY_LABELS.get(policy, policy)
        for metric in METRICS:
            stats = _mean_std_ci95([float(r[metric]) for r in rows])
            aggregated_rows.append(
                {
                    "method": label,
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "ci95": stats["ci95"],
                }
            )
        adm_stats = _mean_std_ci95([float(r["admitted_urllc_count"]) for r in rows])
        embb_violation_stats = _mean_std_ci95([float(r["embb_min_rate_violation_ratio"]) for r in rows])
        safe_adm_stats = _mean_std_ci95([float(r["safe_admission_ratio"]) for r in rows])
        fig1_means.append(adm_stats["mean"])
        fig1_ci95.append(adm_stats["ci95"])
        fig2_means.append(embb_violation_stats["mean"])
        fig2_ci95.append(embb_violation_stats["ci95"])
        fig3_means.append(safe_adm_stats["mean"])
        fig3_ci95.append(safe_adm_stats["ci95"])
        fig4_x.append(adm_stats["mean"])
        fig4_y.append(embb_violation_stats["mean"])

    aggregated_csv = out_dir / "aggregated_summary.csv"
    _write_csv(aggregated_csv, aggregated_rows, ["method", "metric", "mean", "std", "ci95"])

    _bar_plot(out_dir / "fig1_admitted_urllc_packet_count.png", "Fig. 1  Admitted URLLC Packet Count", "Admitted URLLC packet count", methods, fig1_means, fig1_ci95)
    _bar_plot(out_dir / "fig2_embb_min_rate_violation_ratio.png", "Fig. 2  eMBB Min-Rate Violation Ratio", "eMBB min-rate violation ratio", methods, fig2_means, fig2_ci95)
    _bar_plot(out_dir / "fig3_safe_admission_ratio.png", "Fig. 3  Safe Admission Ratio", "Safe admission ratio", methods, fig3_means, fig3_ci95)
    _scatter_tradeoff(out_dir / "fig4_admission_violation_tradeoff.png", methods, fig4_x, fig4_y)

    meta = {
        "policies": policies,
        "mix": args.mix,
        "load": float(args.load),
        "lambda": float(args.lambda_value),
        "packet_bits": int(args.packet_bits),
        "target_error_probability": float(args.target_error_probability),
        "channel_uses": int(args.channel_uses),
        "seeds": seeds,
        "per_run_csv": str(per_run_csv),
        "aggregated_csv": str(aggregated_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[FBL-METHOD-COMPARE] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
