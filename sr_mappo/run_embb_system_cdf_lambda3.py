from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .run_unified_lambda_stress import _jsonify, _parse_csv_floats, _parse_csv_strings
from .unified_policy_runner import MIX_PRESETS, run_policy


DISPLAY_NAMES = {
    "mappo": "MAPPO",
    "greedy": "Greedy",
}


def _build_policy_config(
    args: argparse.Namespace,
    *,
    policy: str,
    total_load: float,
    mix_ratio: float,
    lam: float,
    packet_bits: int | None,
    target_error_probability: float | None,
    channel_uses: int | None,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "total_load": float(total_load),
        "mix_ratio": float(mix_ratio),
        "simulation": {
            "fixed_urllc_poisson_rate": True,
            "urllc_poisson_rate": float(lam),
            "urllc_user_ratio": float(mix_ratio),
        },
        "env": {
            "urllc_poisson_rate_is_per_user": True,
            "urllc_poisson_rate_is_slot_level": False,
        },
    }
    urllc_cfg: Dict[str, object] = {}
    if target_error_probability is not None:
        urllc_cfg["target_error_probability"] = float(target_error_probability)
    if packet_bits is not None and int(packet_bits) > 0:
        urllc_cfg["packet_lengths"] = [int(packet_bits), int(packet_bits), int(packet_bits)]
    if urllc_cfg:
        cfg["urllc"] = urllc_cfg

    system_cfg: Dict[str, object] = {}
    if channel_uses is not None and int(channel_uses) > 0:
        system_cfg["channel_uses_per_minislot"] = int(channel_uses)
    if args.system_num_subcarriers is not None:
        system_cfg["num_subcarriers"] = int(args.system_num_subcarriers)
    if args.system_num_minislots is not None:
        system_cfg["num_minislots"] = int(args.system_num_minislots)
    if system_cfg:
        cfg["system"] = system_cfg

    if policy.startswith("mappo"):
        if not args.mappo_checkpoint_path:
            raise ValueError(f"Policy '{policy}' requires --mappo-checkpoint-path.")
        cfg["checkpoint_path"] = str(Path(args.mappo_checkpoint_path).expanduser())
    return cfg


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric_cdf(
    out_path: Path,
    rows: List[Dict[str, object]],
    *,
    field: str,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    methods = []
    for row in rows:
        method = str(row.get("method", ""))
        if method not in methods:
            methods.append(method)
    for method in methods:
        vals = [
            float(row[field])
            for row in rows
            if str(row.get("method", "")) == method and np.isfinite(float(row.get(field, float("nan"))))
        ]
        if not vals:
            continue
        xs = np.sort(np.asarray(vals, dtype=float))
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(xs, ys, linewidth=2.2, label=DISPLAY_NAMES.get(method, method))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate episode-level system eMBB KPI CDFs at lambda=3."
    )
    parser.add_argument("--policies", default="mappo,greedy")
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--load", type=float, default=24.0)
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--packet-bits", type=int, default=0)
    parser.add_argument("--target-error-probability", type=float, default=-1.0)
    parser.add_argument("--channel-uses", type=int, default=0)
    parser.add_argument("--system-num-subcarriers", type=int, default=None)
    parser.add_argument("--system-num-minislots", type=int, default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default="sr_mappo/results/embb_system_cdf_lambda3")
    parser.add_argument("--mappo-checkpoint-path", default=None)
    args = parser.parse_args()

    mix_name = str(args.mix).strip()
    if mix_name not in MIX_PRESETS:
        raise ValueError(f"Unsupported mix={mix_name!r}. Allowed={sorted(MIX_PRESETS)}")

    policies = _parse_csv_strings(args.policies)
    seeds = [int(round(v)) for v in _parse_csv_floats(args.seeds)]
    mix_ratio = float(MIX_PRESETS[mix_name])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    summary_payload: Dict[str, object] = {
        "meta": {
            "mix": mix_name,
            "load": float(args.load),
            "lambda": float(args.lambda_value),
            "packet_bits": (None if int(args.packet_bits) <= 0 else int(args.packet_bits)),
            "target_error_probability": (None if float(args.target_error_probability) < 0.0 else float(args.target_error_probability)),
            "channel_uses": (None if int(args.channel_uses) <= 0 else int(args.channel_uses)),
            "system_num_subcarriers": args.system_num_subcarriers,
            "system_num_minislots": args.system_num_minislots,
            "seeds": list(seeds),
            "policies": list(policies),
        },
        "methods": {},
    }

    for policy in policies:
        method_rows: List[Dict[str, object]] = []
        for seed in seeds:
            cfg = _build_policy_config(
                args,
                policy=policy,
                total_load=float(args.load),
                mix_ratio=float(mix_ratio),
                lam=float(args.lambda_value),
                packet_bits=(None if int(args.packet_bits) <= 0 else int(args.packet_bits)),
                target_error_probability=(None if float(args.target_error_probability) < 0.0 else float(args.target_error_probability)),
                channel_uses=(None if int(args.channel_uses) <= 0 else int(args.channel_uses)),
            )
            result = run_policy(policy, cfg, int(seed))
            raw_summary = dict(result.get("raw_summary", {}) or {})
            row = {
                "method": str(policy),
                "seed": int(seed),
                "episode": 1,
                "aggregate_embb_throughput_mbps": float(result.get("total_embb_throughput", 0.0) or 0.0) / 1.0e6,
                "average_embb_rate_mbps_per_user": float(result.get("average_embb_rate", 0.0) or 0.0) / 1.0e6,
                "urllc_admission_ratio": float(result.get("urllc_admission_ratio", 0.0) or 0.0),
                "urllc_admitted_packets": float(result.get("scheduled_urllc_packets", 0.0) or 0.0),
                "overlay_action_count": float(result.get("overlay_action_count", 0.0) or 0.0),
                "puncturing_action_count": float(result.get("puncturing_action_count", 0.0) or 0.0),
                "runtime_sec": float(result.get("runtime", 0.0) or 0.0),
                "embb_min_rate_satisfaction_after_puncture_deduction": float(
                    raw_summary.get(
                        "embb_min_rate_satisfaction_after_puncture_deduction",
                        raw_summary.get("embb_min_rate_satisfaction_ratio", 0.0),
                    )
                    or 0.0
                ),
            }
            method_rows.append(row)
            episode_rows.append(row)
            print(
                f"[SYS-CDF] policy={policy} seed={int(seed)} "
                f"agg_embb={float(row['aggregate_embb_throughput_mbps']):.3f} Mbps "
                f"avg_embb={float(row['average_embb_rate_mbps_per_user']):.3f} Mbps/user "
                f"adm={float(row['urllc_admission_ratio']):.4f}",
                flush=True,
            )

        agg_vals = [float(r["aggregate_embb_throughput_mbps"]) for r in method_rows]
        avg_vals = [float(r["average_embb_rate_mbps_per_user"]) for r in method_rows]
        summary = {
            "method": str(policy),
            "num_episode_samples": int(len(method_rows)),
            "mean_aggregate_embb_throughput_mbps": float(np.mean(np.asarray(agg_vals, dtype=float))) if agg_vals else np.nan,
            "p10_aggregate_embb_throughput_mbps": _percentile(agg_vals, 10),
            "median_aggregate_embb_throughput_mbps": _percentile(agg_vals, 50),
            "mean_average_embb_rate_mbps_per_user": float(np.mean(np.asarray(avg_vals, dtype=float))) if avg_vals else np.nan,
            "p10_average_embb_rate_mbps_per_user": _percentile(avg_vals, 10),
            "median_average_embb_rate_mbps_per_user": _percentile(avg_vals, 50),
        }
        summary_payload["methods"][policy] = summary
        summary_rows.append(summary)

    episodes_csv = out_dir / "embb_system_metrics_lambda3.csv"
    summary_csv = out_dir / "embb_system_metrics_summary_lambda3.csv"
    agg_fig = out_dir / "fig_system_aggregate_embb_throughput_cdf_lambda3.png"
    avg_fig = out_dir / "fig_system_average_embb_rate_cdf_lambda3.png"
    summary_json = out_dir / "summary.json"

    _write_csv(
        episodes_csv,
        [
            "method",
            "seed",
            "episode",
            "aggregate_embb_throughput_mbps",
            "average_embb_rate_mbps_per_user",
            "urllc_admission_ratio",
            "urllc_admitted_packets",
            "overlay_action_count",
            "puncturing_action_count",
            "runtime_sec",
            "embb_min_rate_satisfaction_after_puncture_deduction",
        ],
        episode_rows,
    )
    _write_csv(
        summary_csv,
        [
            "method",
            "num_episode_samples",
            "mean_aggregate_embb_throughput_mbps",
            "p10_aggregate_embb_throughput_mbps",
            "median_aggregate_embb_throughput_mbps",
            "mean_average_embb_rate_mbps_per_user",
            "p10_average_embb_rate_mbps_per_user",
            "median_average_embb_rate_mbps_per_user",
        ],
        summary_rows,
    )
    _plot_metric_cdf(
        agg_fig,
        episode_rows,
        field="aggregate_embb_throughput_mbps",
        title="Episode-level aggregate eMBB throughput CDF at lambda = 3",
        xlabel="Aggregate eMBB throughput (Mbps)",
    )
    _plot_metric_cdf(
        avg_fig,
        episode_rows,
        field="average_embb_rate_mbps_per_user",
        title="Episode-level average eMBB rate CDF at lambda = 3",
        xlabel="Average eMBB rate over system (Mbps/user)",
    )

    summary_json.write_text(json.dumps(_jsonify(summary_payload), indent=2), encoding="utf-8")

    print("\n[SYS-CDF] Summary table", flush=True)
    for row in summary_rows:
        print(
            f"{DISPLAY_NAMES.get(str(row['method']), str(row['method'])):<12} "
            f"episodes={int(row['num_episode_samples'])} "
            f"agg_mean={float(row['mean_aggregate_embb_throughput_mbps']):.3f} "
            f"agg_p10={float(row['p10_aggregate_embb_throughput_mbps']):.3f} "
            f"avg_mean={float(row['mean_average_embb_rate_mbps_per_user']):.3f}",
            flush=True,
        )
    print(f"[SYS-CDF] wrote {episodes_csv}", flush=True)
    print(f"[SYS-CDF] wrote {summary_csv}", flush=True)
    print(f"[SYS-CDF] wrote {agg_fig}", flush=True)
    print(f"[SYS-CDF] wrote {avg_fig}", flush=True)


if __name__ == "__main__":
    main()
