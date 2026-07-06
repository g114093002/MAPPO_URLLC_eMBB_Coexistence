from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Dict, List

import numpy as np

from .run_fbl_reliability_sweep import _build_policy_config
from .unified_policy_runner import MIX_PRESETS, run_policy


POLICY_LABELS = {
    "mappo": "MAPPO",
    "mappo_puncture_forced": "MAPPO puncturing-forced",
    "greedy": "Greedy",
}


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-admission eMBB damage logging under fixed FBL settings.")
    parser.add_argument("--policies", default="mappo,mappo_puncture_forced,greedy")
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--load", type=float, default=24.0)
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--packet-bits", type=int, default=24)
    parser.add_argument("--target-error-probability", type=float, default=1e-5)
    parser.add_argument("--channel-uses", type=int, default=32)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default="sr_mappo/results/per_admission_embb_damage_compare")
    parser.add_argument("--mappo-checkpoint-path", required=True)
    args = parser.parse_args()

    policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    if str(args.mix) not in MIX_PRESETS:
        raise ValueError(f"Unsupported mix={args.mix!r}. Allowed={sorted(MIX_PRESETS)}")
    mix_ratio = float(MIX_PRESETS[str(args.mix)])
    seeds = [int(round(float(x))) for x in str(args.seeds).split(",") if str(x).strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    admitted_counts_by_method: Dict[str, float] = {}

    for policy in policies:
        label = POLICY_LABELS.get(policy, policy)
        method_samples: List[Dict[str, object]] = []
        admitted_total = 0.0
        for seed in seeds:
            cfg = _build_policy_config(
                args,
                policy="mappo" if policy.startswith("mappo") else policy,
                total_load=float(args.load),
                mix_ratio=mix_ratio,
                lam=float(args.lambda_value),
                packet_bits=int(args.packet_bits),
                channel_uses=int(args.channel_uses),
                target_error_probability=float(args.target_error_probability),
            )
            result = run_policy(policy, cfg, seed)
            admitted_total += float(result.get("admitted_urllc_count", 0.0) or 0.0)
            for row in list(result.get("per_admission_embb_damage_samples", []) or []):
                fixed = dict(row)
                fixed["method"] = label
                method_samples.append(fixed)
                sample_rows.append(fixed)
        admitted_counts_by_method[label] = admitted_total
        losses = [float(row["embb_rate_loss_due_to_action"]) for row in method_samples]
        summary_rows.append(
            {
                "method": label,
                "num_admitted_samples": len(method_samples),
                "mean_embb_rate_loss_per_admission": float(np.mean(losses)) if losses else 0.0,
                "median_embb_rate_loss_per_admission": float(median(losses)) if losses else 0.0,
                "p90_embb_rate_loss_per_admission": float(np.percentile(np.asarray(losses, dtype=float), 90)) if losses else 0.0,
                "std_embb_rate_loss_per_admission": float(np.std(np.asarray(losses, dtype=float), ddof=1)) if len(losses) > 1 else 0.0,
                "total_embb_rate_loss_due_to_admitted_urllc": float(np.sum(losses)) if losses else 0.0,
            }
        )

    sample_fields = [
        "method",
        "seed",
        "episode",
        "decision_step",
        "selected_packet_id",
        "selected_uav",
        "selected_rb",
        "selected_embb_owner",
        "selected_mode",
        "urllc_sinr",
        "gamma_th",
        "urllc_reliability_satisfied",
        "embb_sum_rate_before_action",
        "embb_sum_rate_after_action",
        "embb_rate_loss_due_to_action",
    ]
    summary_fields = [
        "method",
        "num_admitted_samples",
        "mean_embb_rate_loss_per_admission",
        "median_embb_rate_loss_per_admission",
        "p90_embb_rate_loss_per_admission",
        "std_embb_rate_loss_per_admission",
        "total_embb_rate_loss_due_to_admitted_urllc",
    ]
    samples_csv = out_dir / "per_admission_embb_damage_samples.csv"
    summary_csv = out_dir / "per_admission_embb_damage_summary.csv"
    _write_csv(samples_csv, sample_rows, sample_fields)
    _write_csv(summary_csv, summary_rows, summary_fields)

    print("[PER-ADMISSION-DAMAGE] first 10 samples")
    for row in sample_rows[:10]:
        print(json.dumps(row, ensure_ascii=False))

    print("[PER-ADMISSION-DAMAGE] consistency checks")
    for row in sample_rows[:10]:
        before = float(row["embb_sum_rate_before_action"])
        after = float(row["embb_sum_rate_after_action"])
        loss = float(row["embb_rate_loss_due_to_action"])
        if before >= after:
            status = bool(loss >= 0.0)
        else:
            status = bool(abs(loss) <= 1.0e-12)
        print(
            f"step_check method={row['method']} mode={row['selected_mode']} before={before:.3f} after={after:.3f} loss={loss:.3f} ok={status}"
        )

    for item in summary_rows:
        admitted = admitted_counts_by_method.get(item["method"], 0.0)
        print(
            f"method_check method={item['method']} num_admitted_samples={item['num_admitted_samples']} admitted_urllc_count_sum={admitted:.0f}"
        )

    meta = {
        "mix": args.mix,
        "load": float(args.load),
        "lambda": float(args.lambda_value),
        "packet_bits": int(args.packet_bits),
        "target_error_probability": float(args.target_error_probability),
        "channel_uses": int(args.channel_uses),
        "policies": policies,
        "seeds": seeds,
        "samples_csv": str(samples_csv),
        "summary_csv": str(summary_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[PER-ADMISSION-DAMAGE] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
