from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .run_unified_lambda_stress import _jsonify, _parse_csv_floats, _parse_csv_strings
from .unified_policy_runner import MIX_PRESETS, run_policy


DEFAULT_POLICIES = ["mappo", "greedy"]


def _eps_key(value: float) -> str:
    return f"eps_{value:.0e}"


def _build_policy_config(
    args,
    *,
    policy: str,
    total_load: float,
    mix_ratio: float,
    lam: float,
    packet_bits: int,
    channel_uses: int,
    target_error_probability: float,
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
        "urllc": {
            "target_error_probability": float(target_error_probability),
            "packet_lengths": [int(packet_bits), int(packet_bits), int(packet_bits)],
        },
        "system": {
            "channel_uses_per_minislot": int(channel_uses),
        },
    }
    if policy == "mappo":
        if not args.mappo_checkpoint_path:
            raise ValueError("Policy 'mappo' requires --mappo-checkpoint-path.")
        cfg["checkpoint_path"] = str(Path(args.mappo_checkpoint_path).expanduser())
    return cfg


def _reliability_satisfaction_ratio(metrics: Dict[str, object]) -> float:
    summary = dict(metrics.get("raw_summary", {}) or {})
    scheduled = float(summary.get("scheduled_packets", 0.0) or 0.0)
    active = float(summary.get("active_packets", 0.0) or 0.0)
    violations = float(summary.get("urllc_constraint_violations", 0.0) or 0.0)
    if scheduled > 0.0:
        return float(max(scheduled - violations, 0.0) / max(scheduled, 1.0))
    return 0.0 if active > 0.0 else 1.0


def _aggregate_seed_runs(seed_runs: List[Dict[str, object]]) -> Dict[str, float]:
    def _mean_key(key: str) -> float:
        return float(np.mean([float(item.get(key, 0.0) or 0.0) for item in seed_runs])) if seed_runs else 0.0

    raw_summaries = [dict(item.get("raw_summary", {}) or {}) for item in seed_runs]
    admitted_reliability = []
    effective_success = []
    satisfaction_ratio = []
    for item, summary in zip(seed_runs, raw_summaries):
        admitted_reliability.append(float(summary.get("admitted_urllc_reliability", item.get("urllc_admission_ratio", 0.0)) or 0.0))
        effective_success.append(float(summary.get("effective_urllc_success_over_arrivals", 0.0) or 0.0))
        satisfaction_ratio.append(_reliability_satisfaction_ratio(item))
    return {
        "urllc_admission_ratio": _mean_key("urllc_admission_ratio"),
        "reliability_satisfaction_ratio": float(np.mean(np.asarray(satisfaction_ratio, dtype=float))) if satisfaction_ratio else 0.0,
        "admitted_urllc_reliability": float(np.mean(np.asarray(admitted_reliability, dtype=float))) if admitted_reliability else 0.0,
        "effective_urllc_success_over_arrivals": float(np.mean(np.asarray(effective_success, dtype=float))) if effective_success else 0.0,
        "embb_rate_mbps": _mean_key("total_embb_throughput") / 1.0e6,
        "runtime_sec": _mean_key("runtime"),
    }


def _plot_metric(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    x_values: List[float],
    series: Dict[int, Dict[str, List[float]]],
) -> None:
    channel_uses_values = list(sorted(series))
    cols = min(2, max(len(channel_uses_values), 1))
    rows = int(np.ceil(len(channel_uses_values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 4.8 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, channel_uses in zip(axes_flat, channel_uses_values):
        per_line = dict(series.get(channel_uses, {}) or {})
        for label, y_values in per_line.items():
            if len(y_values) != len(x_values):
                continue
            ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=label)
        ax.set_xscale("log")
        ax.set_title(f"channel uses = {int(channel_uses)}")
        ax.set_xlabel("Target error probability")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
    for ax in axes_flat[len(channel_uses_values):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite-blocklength URLLC reliability sweep over target error probability, packet size, and blocklength."
    )
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--mixes", default="5:5")
    parser.add_argument("--loads", default="24")
    parser.add_argument("--lambdas", default="3")
    parser.add_argument("--target-error-probs", default="1e-3,1e-4,1e-5,1e-6,1e-7")
    parser.add_argument("--packet-bits", default="120,150,180")
    parser.add_argument("--channel-uses", default="24,32,40")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default="sr_mappo/results/fbl_reliability_sweep")
    parser.add_argument("--mappo-checkpoint-path", default=None)
    args = parser.parse_args()

    policies = _parse_csv_strings(args.policies)
    mixes = _parse_csv_strings(args.mixes)
    loads = _parse_csv_floats(args.loads)
    lambdas = _parse_csv_floats(args.lambdas)
    target_error_probs = _parse_csv_floats(args.target_error_probs)
    packet_bits_values = [int(round(value)) for value in _parse_csv_floats(args.packet_bits)]
    channel_uses_values = [int(round(value)) for value in _parse_csv_floats(args.channel_uses)]
    seeds = [int(round(value)) for value in _parse_csv_floats(args.seeds)]

    invalid_mixes = [mix for mix in mixes if mix not in MIX_PRESETS]
    if invalid_mixes:
        raise ValueError(f"Unsupported mixes={invalid_mixes}. Allowed={sorted(MIX_PRESETS)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "meta": {
            "policies": list(policies),
            "mixes": list(mixes),
            "loads": list(loads),
            "lambdas": list(lambdas),
            "target_error_probs": list(target_error_probs),
            "packet_bits": list(packet_bits_values),
            "channel_uses": list(channel_uses_values),
            "seeds": list(seeds),
        },
        "aggregates": {},
        "raw_runs": {},
    }

    for mix_name in mixes:
        mix_ratio = float(MIX_PRESETS[mix_name])
        mix_bucket = payload["aggregates"].setdefault(mix_name, {})
        raw_mix_bucket = payload["raw_runs"].setdefault(mix_name, {})
        for total_load in loads:
            load_key = f"load_{float(total_load):g}"
            load_bucket = mix_bucket.setdefault(load_key, {})
            raw_load_bucket = raw_mix_bucket.setdefault(load_key, {})
            for lam in lambdas:
                lam_key = f"lambda_{float(lam):g}"
                lam_bucket = load_bucket.setdefault(lam_key, {})
                raw_lam_bucket = raw_load_bucket.setdefault(lam_key, {})
                for policy in policies:
                    policy_bucket = lam_bucket.setdefault(policy, {})
                    raw_policy_bucket = raw_lam_bucket.setdefault(policy, {})
                    for channel_uses in channel_uses_values:
                        block_bucket = policy_bucket.setdefault(str(int(channel_uses)), {})
                        raw_block_bucket = raw_policy_bucket.setdefault(str(int(channel_uses)), {})
                        for packet_bits in packet_bits_values:
                            packet_bucket = block_bucket.setdefault(str(int(packet_bits)), {})
                            raw_packet_bucket = raw_block_bucket.setdefault(str(int(packet_bits)), {})
                            for eps in target_error_probs:
                                eps_key = _eps_key(float(eps))
                                seed_runs = list(raw_packet_bucket.get(eps_key, []) or [])
                                if len(seed_runs) < len(seeds):
                                    seed_runs = []
                                    for seed in seeds:
                                        cfg = _build_policy_config(
                                            args,
                                            policy=policy,
                                            total_load=float(total_load),
                                            mix_ratio=float(mix_ratio),
                                            lam=float(lam),
                                            packet_bits=int(packet_bits),
                                            channel_uses=int(channel_uses),
                                            target_error_probability=float(eps),
                                        )
                                        result = run_policy(policy, cfg, int(seed))
                                        seed_runs.append(result)
                                        print(
                                            f"[FBL-SWEEP] mix={mix_name} load={float(total_load):g} lambda={float(lam):g} "
                                            f"policy={policy} n={int(channel_uses)} bits={int(packet_bits)} eps={float(eps):.0e} "
                                            f"adm={float(result.get('urllc_admission_ratio', 0.0) or 0.0):.4f} "
                                            f"rel_sat={_reliability_satisfaction_ratio(result):.4f}",
                                            flush=True,
                                        )
                                    raw_packet_bucket[eps_key] = seed_runs
                                packet_bucket[eps_key] = _aggregate_seed_runs(seed_runs)

                metric_specs = [
                    ("urllc_admission_ratio", "URLLC Admission Ratio", "Admission ratio"),
                    ("reliability_satisfaction_ratio", "URLLC Reliability Satisfaction Ratio", "Satisfaction ratio"),
                    ("admitted_urllc_reliability", "Admitted URLLC Reliability", "Reliability"),
                ]
                for metric_key, title, ylabel in metric_specs:
                    line_series: Dict[int, Dict[str, List[float]]] = {}
                    for channel_uses in channel_uses_values:
                        channel_series = line_series.setdefault(int(channel_uses), {})
                        for policy in policies:
                            for packet_bits in packet_bits_values:
                                label = f"{policy} | {int(packet_bits)} bits"
                                values: List[float] = []
                                for eps in target_error_probs:
                                    entry = (
                                        payload["aggregates"][mix_name][load_key][lam_key][policy][str(int(channel_uses))][str(int(packet_bits))][_eps_key(float(eps))]
                                    )
                                    values.append(float(entry.get(metric_key, 0.0) or 0.0))
                                channel_series[label] = values
                    stem = f"{metric_key}_{mix_name.replace(':','')}_load{float(total_load):g}_lambda{float(lam):g}".replace(".", "p")
                    _plot_metric(
                        out_path=out_dir / f"{stem}.png",
                        title=f"{title} | mix={mix_name} load={float(total_load):g} lambda={float(lam):g}",
                        ylabel=ylabel,
                        x_values=list(target_error_probs),
                        series=line_series,
                    )

    metrics_path = out_dir / "finite_blocklength_reliability_sweep.json"
    metrics_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
    print(f"[FBL-SWEEP] wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
