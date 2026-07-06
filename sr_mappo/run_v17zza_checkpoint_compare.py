from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_fixed_user_blocklength_compare import _build_policy_config, _parse_csv_ints
from .unified_policy_runner import run_policy


DEFAULT_CHECKPOINT_DIR = Path("checkpoints") / "clean_mappo"
DEFAULT_RUN_STEM = "sr_mappo_tp_full_mappo_v17zza_embb_qos_rebalance_v1_clean"
DEFAULT_ITERATIONS = [100, 500, 1000, 1500, 2000, 2500, 2800]
DEFAULT_METRICS = [
    "total_embb_throughput",
    "average_power_consumption",
    "embb_blocked_user_count",
    "urllc_blocked_user_count",
]


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_rows(rows: List[Dict[str, object]], metrics: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in metrics:
        out[key] = float(np.mean([float(row.get(key, 0.0) or 0.0) for row in rows])) if rows else 0.0
    return out


def _discover_checkpoint_specs(
    checkpoint_dir: Path,
    run_stem: str,
    iterations: List[int],
    include_best: bool,
    include_latest: bool,
) -> List[Tuple[str, Path, float]]:
    specs: List[Tuple[str, Path, float]] = []
    if include_best:
        path = checkpoint_dir / f"{run_stem}_best_eval.pt"
        if path.exists():
            specs.append(("best_eval", path, -1.0))
    for iteration in iterations:
        path = checkpoint_dir / f"{run_stem}_eval_iter_{int(iteration)}.pt"
        if path.exists():
            specs.append((f"iter_{int(iteration)}", path, float(iteration)))
    if include_latest:
        path = checkpoint_dir / f"{run_stem}_latest_eval.pt"
        if path.exists():
            specs.append(("latest_eval", path, float(max(iterations) + 100 if iterations else 9_999)))
    return specs


def _plot_metric_vs_checkpoint(
    path: Path,
    rows: List[Dict[str, object]],
    metric_key: str,
    ylabel: str,
) -> None:
    filtered = [row for row in rows if str(row.get("policy", "")) == "mappo"]
    if not filtered:
        return
    filtered.sort(key=lambda row: float(row.get("checkpoint_order", 0.0) or 0.0))
    labels = [str(row.get("checkpoint_label", "")) for row in filtered]
    values = [float(row.get(metric_key, 0.0) or 0.0) for row in filtered]
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.9), 4.8))
    ax.plot(x, values, marker="o", linewidth=2.0, color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Checkpoint")
    ax.set_title(f"{ylabel} vs v17zza checkpoint")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v17zza checkpoints against greedy on fixed scenario sweeps.")
    parser.add_argument("--embb-users", type=int, default=10)
    parser.add_argument("--urllc-users", default="10,20,30,40,50")
    parser.add_argument("--packet-bits", default="24")
    parser.add_argument("--channel-uses", type=int, default=None)
    parser.add_argument("--lambda-per-user", type=float, default=None)
    parser.add_argument("--target-error-probability", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--run-stem", default=DEFAULT_RUN_STEM)
    parser.add_argument("--iterations", default=",".join(str(v) for v in DEFAULT_ITERATIONS))
    parser.add_argument("--include-best-eval", action="store_true")
    parser.add_argument("--include-latest-eval", action="store_true")
    parser.add_argument("--out-dir", default="sr_mappo/results/v17zza_checkpoint_compare")
    args = parser.parse_args()

    urllc_users_list = _parse_csv_ints(args.urllc_users)
    packet_bits_list = _parse_csv_ints(args.packet_bits)
    iterations = _parse_csv_ints(args.iterations)
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_specs = _discover_checkpoint_specs(
        checkpoint_dir=checkpoint_dir,
        run_stem=str(args.run_stem),
        iterations=iterations,
        include_best=bool(args.include_best_eval),
        include_latest=bool(args.include_latest_eval),
    )
    if not checkpoint_specs:
        raise FileNotFoundError("No checkpoint specs found. Check --checkpoint-dir / --run-stem / --iterations.")

    per_run_rows: List[Dict[str, object]] = []

    # Greedy baseline once per scenario.
    for urllc_users in urllc_users_list:
        for packet_bits in packet_bits_list:
            greedy_cfg = _build_policy_config(
                policy="greedy",
                embb_users=int(args.embb_users),
                urllc_users=int(urllc_users),
                packet_bits=int(packet_bits),
                channel_uses=args.channel_uses,
                lambda_per_user=args.lambda_per_user,
                target_error_probability=args.target_error_probability,
                mappo_checkpoint_path=None,
            )
            result = run_policy("greedy", deepcopy(greedy_cfg), int(args.seed))
            per_run_rows.append(
                {
                    "policy": "greedy",
                    "checkpoint_label": "greedy",
                    "checkpoint_path": "",
                    "checkpoint_order": -2.0,
                    "seed": int(args.seed),
                    "embb_users": int(args.embb_users),
                    "urllc_users": int(urllc_users),
                    "packet_bits": int(packet_bits),
                    **{key: float(result.get(key, 0.0) or 0.0) for key in DEFAULT_METRICS},
                }
            )
            print(
                f"[CKPT-COMPARE] greedy U={urllc_users} B={packet_bits} "
                f"rate={float(result.get('total_embb_throughput', 0.0) or 0.0):.3e}",
                flush=True,
            )

    for checkpoint_label, checkpoint_path, checkpoint_order in checkpoint_specs:
        for urllc_users in urllc_users_list:
            for packet_bits in packet_bits_list:
                mappo_cfg = _build_policy_config(
                    policy="mappo",
                    embb_users=int(args.embb_users),
                    urllc_users=int(urllc_users),
                    packet_bits=int(packet_bits),
                    channel_uses=args.channel_uses,
                    lambda_per_user=args.lambda_per_user,
                    target_error_probability=args.target_error_probability,
                    mappo_checkpoint_path=str(checkpoint_path),
                )
                result = run_policy("mappo", deepcopy(mappo_cfg), int(args.seed))
                per_run_rows.append(
                    {
                        "policy": "mappo",
                        "checkpoint_label": str(checkpoint_label),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_order": float(checkpoint_order),
                        "seed": int(args.seed),
                        "embb_users": int(args.embb_users),
                        "urllc_users": int(urllc_users),
                        "packet_bits": int(packet_bits),
                        **{key: float(result.get(key, 0.0) or 0.0) for key in DEFAULT_METRICS},
                    }
                )
                print(
                    f"[CKPT-COMPARE] {checkpoint_label} U={urllc_users} B={packet_bits} "
                    f"rate={float(result.get('total_embb_throughput', 0.0) or 0.0):.3e}",
                    flush=True,
                )

    per_run_csv = out_dir / "per_run_checkpoint_compare.csv"
    _write_csv(per_run_csv, per_run_rows, list(per_run_rows[0].keys()))

    aggregated_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in per_run_rows:
        key = (str(row["policy"]), str(row["checkpoint_label"]))
        grouped.setdefault(key, []).append(row)

    for (policy, checkpoint_label), rows in grouped.items():
        means = _mean_rows(rows, DEFAULT_METRICS)
        aggregated_rows.append(
            {
                "policy": policy,
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(rows[0].get("checkpoint_path", "")),
                "checkpoint_order": float(rows[0].get("checkpoint_order", 0.0) or 0.0),
                "scenario_count": len(rows),
                **means,
            }
        )

    aggregated_rows.sort(key=lambda row: (str(row["policy"]), float(row.get("checkpoint_order", 0.0) or 0.0)))
    aggregated_csv = out_dir / "aggregated_checkpoint_compare.csv"
    _write_csv(aggregated_csv, aggregated_rows, list(aggregated_rows[0].keys()))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_vs_checkpoint(plots_dir / "embb_sum_rate_vs_checkpoint.png", aggregated_rows, "total_embb_throughput", "eMBB sum rate (bps)")
    _plot_metric_vs_checkpoint(plots_dir / "total_power_vs_checkpoint.png", aggregated_rows, "average_power_consumption", "Total power")
    _plot_metric_vs_checkpoint(plots_dir / "embb_blocked_vs_checkpoint.png", aggregated_rows, "embb_blocked_user_count", "eMBB blocked users")
    _plot_metric_vs_checkpoint(plots_dir / "urllc_blocked_vs_checkpoint.png", aggregated_rows, "urllc_blocked_user_count", "URLLC blocked users")

    summary = {
        "embb_users": int(args.embb_users),
        "urllc_users": urllc_users_list,
        "packet_bits": packet_bits_list,
        "seed": int(args.seed),
        "checkpoint_specs": [
            {"label": label, "path": str(path), "order": float(order)}
            for label, path, order in checkpoint_specs
        ],
        "per_run_csv": str(per_run_csv),
        "aggregated_csv": str(aggregated_csv),
        "plots_dir": str(plots_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[CKPT-COMPARE] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
