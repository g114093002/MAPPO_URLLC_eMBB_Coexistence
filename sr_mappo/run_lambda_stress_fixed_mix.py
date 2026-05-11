from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .run_greedy_mix_share_grid import (
    MIX_BASE_EXPERIMENT,
    PROJECT_ROOT,
    RESULTS_DIR,
    _build_report_cmd_env,
    _load_full_metrics_payload,
    _load_greedy_metrics,
)


def _plot_lambda_stress(data_by_mix: dict, out_path: Path) -> None:
    mixes = ["10:0", "7:3", "5:5", "3:7"]
    mixes = [m for m in mixes if m in data_by_mix]
    if not mixes:
        return
    metrics = [
        ("embb_rate_pre_urllc_admission_mbps", "Aggregate eMBB throughput (pre-URLLC admission)", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
    ]
    color_map = {"10:0": "#1f77b4", "7:3": "#ff7f0e", "5:5": "#2ca02c", "3:7": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = np.asarray(axes).ravel()
    for ax, (k, title, ylab) in zip(axes, metrics):
        for mix in mixes:
            d = data_by_mix[mix]
            x = np.asarray(d["lambda_total_override"], dtype=float)
            y = np.asarray(d[k], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, color=color_map.get(mix, None), label=mix)
        ax.set_title(title)
        ax.set_xlabel("Total URLLC lambda override (pkts/slot)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)
    fig.suptitle("Greedy Lambda Stress Test (Fixed Mix Ratio)", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-mix greedy lambda stress test (total-system lambda sweep).")
    parser.add_argument(
        "--mixes",
        default="7:3,5:5,3:7",
        help="Comma-separated mix list from {10:0,7:3,5:5,3:7}.",
    )
    parser.add_argument("--episodes-per-load", type=int, default=100)
    parser.add_argument(
        "--loads",
        default="18",
        help="Comma-separated loads override. Default: 18",
    )
    parser.add_argument(
        "--lambdas",
        default="10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated total-system lambda overrides (pkts/slot).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "lambda_stress_fixed_mix"),
        help="Directory for archived metrics and plots.",
    )
    parser.add_argument("--paired-seed-base", type=int, default=None)
    parser.add_argument(
        "--parallel-mixes",
        action="store_true",
        help="Run mixes in parallel for each lambda.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_pool = {"10:0": 0.0, "7:3": 0.3, "5:5": 0.5, "3:7": 0.7}
    mix_list = [m.strip() for m in str(args.mixes).split(",") if m.strip()]
    invalid_mixes = [m for m in mix_list if m not in mix_pool]
    if invalid_mixes:
        raise ValueError(f"Unsupported mixes: {invalid_mixes}. Allowed: {list(mix_pool.keys())}")
    mixes = {m: mix_pool[m] for m in mix_list}
    lambdas = [float(x.strip()) for x in str(args.lambdas).split(",") if x.strip()]

    all_data: dict = {mix: {} for mix in mixes.keys()}
    detailed_payload = {
        "meta": {
            "mixes": list(mixes.keys()),
            "episodes_per_load": int(args.episodes_per_load),
            "loads": str(args.loads),
            "lambda_total_sweep": lambdas,
            "paired_seed_base": (int(args.paired_seed_base) if args.paired_seed_base is not None else None),
        },
        "grid": {},
    }

    for lam in lambdas:
        print(f"[LAMBDA] total_lambda={lam:.3f}", flush=True)
        procs: list[tuple[str, subprocess.Popen, Path]] = []
        for mix_label, ratio in mixes.items():
            mix_key = mix_label.replace(":", "_")
            run_experiment = MIX_BASE_EXPERIMENT.get(mix_label, "phase0_joint_full_power_service_interference_repair_v8_greedy_share20_debug")
            run_dir = out_dir / f"_run_lambda_{lam:g}_{mix_key}_{int(time.time()*1000)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd, env = _build_report_cmd_env(
                run_experiment,
                ratio=ratio,
                share=0.0,
                episodes_per_load=int(args.episodes_per_load),
                loads_override=str(args.loads),
                seed_base=args.paired_seed_base,
                urllc_poisson_rate_override=lam,
            )
            env["SR_MAPPO_RESULTS_DIR_OVERRIDE"] = str(run_dir)
            if args.parallel_mixes:
                procs.append((mix_label, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env), run_dir))
            else:
                print(f"[LAMBDA] run mix={mix_label} lambda={lam:.3f}", flush=True)
                subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
                procs.append((mix_label, None, run_dir))

        if args.parallel_mixes:
            for mix_label, proc, _run_dir in procs:
                assert proc is not None
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"parallel run failed: mix={mix_label}, lambda={lam:.3f}, rc={rc}")

        for mix_label, _proc, run_dir in procs:
            mix_key = mix_label.replace(":", "_")
            metrics_src = run_dir / "sr_mappo_report_metrics.json"
            metrics_dst = out_dir / f"lambda_{lam:g}_mix_{mix_key}_sr_mappo_report_metrics.json"
            shutil.copy2(metrics_src, metrics_dst)
            d = _load_greedy_metrics(metrics_dst)
            d["lambda_total_override"] = [lam for _ in d["loads"]]
            # This runner is intended for fixed load stress, keep one value if user passes one load.
            val_idx = 0
            if len(d["loads"]) > 1:
                try:
                    target_load = float(str(args.loads).split(",")[0])
                    val_idx = list(d["loads"]).index(target_load)
                except Exception:
                    val_idx = 0
            point = {k: (float(v[val_idx]) if isinstance(v, np.ndarray) and v.size else v) for k, v in d.items()}
            point["load"] = float(d["loads"][val_idx]) if len(d["loads"]) else None
            point["lambda_total_override"] = float(lam)
            all_data[mix_label].setdefault("lambda_total_override", []).append(float(lam))
            for key in [
                "embb_rate_mbps",
                "embb_rate_pre_urllc_admission_mbps",
                "urllc_admission",
                "urllc_tp_mbps",
                "urllc_admitted_packets",
                "budget_used_ratio",
                "avg_embb_loss",
                "embb_loss_per_admit",
                "embb_service_ratio",
                "embb_min_rate",
                "total_power_mw",
            ]:
                all_data[mix_label].setdefault(key, []).append(float(point.get(key, 0.0)))
            detailed_payload["grid"].setdefault(f"lambda_{lam:g}", {})
            detailed_payload["grid"][f"lambda_{lam:g}"][mix_label] = _load_full_metrics_payload(metrics_dst)

    summary_plot = out_dir / "lambda_stress_fixed_mix_comparison.png"
    _plot_lambda_stress(all_data, summary_plot)

    merged_json = out_dir / "lambda_stress_detailed_metrics.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": detailed_payload["meta"], "series": all_data, "grid": detailed_payload["grid"]}, f, indent=2)

    print(f"[LAMBDA] done. summary plot: {summary_plot}", flush=True)
    print(f"[LAMBDA] done. merged json: {merged_json}", flush=True)


if __name__ == "__main__":
    main()

