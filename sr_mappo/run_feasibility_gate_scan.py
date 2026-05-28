from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _run_setting(
    *,
    label: str,
    experiment: str,
    loads: str,
    episodes_per_load: int,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    share_mode: str,
    share_ratio: float,
    sic_db: float,
    gain_ratio: float,
    sic_prior_enabled: bool,
) -> None:
    out_prefix = f"bench_gate_scan_{label}"
    env = os.environ.copy()
    env["SR_MAPPO_GREEDY_HF_SIC_PAIRING_PRIOR"] = "1" if sic_prior_enabled else "0"
    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.run_fair_mix_clean_greedy",
        "--experiment",
        experiment,
        "--loads",
        loads,
        "--episodes-per-load",
        str(int(episodes_per_load)),
        "--seed-base",
        str(int(seed_base)),
        "--mother-id",
        mother_id,
        "--feasible-graph-id",
        feasible_graph_id,
        "--share-mode",
        share_mode,
        "--share-ratio",
        str(float(share_ratio)),
        "--out-prefix",
        out_prefix,
        "--sic-override-db",
        f"{float(sic_db):.6f}",
        "--gain-ratio-override",
        f"{float(gain_ratio):.6f}",
        "--skip-cleanliness-audit",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def _metric_values(path: Path, key: str) -> List[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = payload.get("greedy", {}).get(key, [])
    if not isinstance(arr, list):
        return []
    return [float(x) for x in arr]


def _collect_triplet(label: str, episodes_per_load: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for mix_name in ["3_7", "5_5", "7_3"]:
        p = RESULTS_DIR / f"bench_gate_scan_{label}_{mix_name}_e{int(episodes_per_load)}" / "sr_mappo_report_metrics.json"
        feas = _metric_values(p, "feasible_pair_ratio")
        nof = _metric_values(p, "greedy_no_feasible_admit_ratio")
        ovf = _metric_values(p, "overlay_feasible_pairs")
        ovc = _metric_values(p, "overlay_candidate_pairs")
        embb = _metric_values(p, "embb_rate")
        rej_rel = _metric_values(p, "greedy_hf_reject_reliability_ratio")
        rej_power = _metric_values(p, "greedy_hf_reject_power_ratio")
        rej_min = _metric_values(p, "greedy_hf_reject_min_rate_ratio")
        rej_share = _metric_values(p, "greedy_hf_reject_share_cap_ratio")
        intercell = _metric_values(p, "phase_a_rejected_intercell_per_decision")

        def _mean(v: List[float]) -> float:
            return float(sum(v) / max(len(v), 1)) if v else 0.0

        out[mix_name] = {
            "feasible_pair_ratio_min": min(feas) if feas else 0.0,
            "feasible_pair_ratio_mean": _mean(feas),
            "no_feasible_admit_ratio_max": max(nof) if nof else 0.0,
            "overlay_feasible_pairs_min": min(ovf) if ovf else 0.0,
            "overlay_candidate_pairs_mean": _mean(ovc),
            "p_feasible_given_candidate_mean": _mean(
                [a / max(b, 1.0e-12) for a, b in zip(ovf, ovc)]
            ) if ovf and ovc and len(ovf) == len(ovc) else 0.0,
            "embb_rate_mean_mbps": _mean(embb) / 1.0e6 if embb else 0.0,
            "reject_reliability_ratio_mean": _mean(rej_rel),
            "reject_power_ratio_mean": _mean(rej_power),
            "reject_min_rate_ratio_mean": _mean(rej_min),
            "reject_share_cap_ratio_mean": _mean(rej_share),
            "intercell_reject_ratio_max": max(intercell) if intercell else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate sensitivity quick scan for feasibility-starved diagnosis.")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,24")
    ap.add_argument("--episodes-per-load", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=20264519)
    ap.add_argument("--mother-id", required=True)
    ap.add_argument("--feasible-graph-id", required=True)
    ap.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    ap.add_argument("--share-ratio", type=float, default=0.0)
    ap.add_argument("--sic-db", type=float, default=-2.0)
    args = ap.parse_args()

    settings: List[Tuple[str, float, bool]] = [
        ("base_g095_prior1", 0.95, True),
        ("g080_prior1", 0.80, True),
        ("g060_prior1", 0.60, True),
        ("g095_prior0", 0.95, False),
        ("g080_prior0", 0.80, False),
    ]

    summary = {}
    for label, gain, prior in settings:
        print(f"\n=== Running {label} | gain={gain:.2f} | sic_prior={int(prior)} ===")
        _run_setting(
            label=label,
            experiment=str(args.experiment),
            loads=str(args.loads),
            episodes_per_load=int(args.episodes_per_load),
            seed_base=int(args.seed_base),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
            share_mode=str(args.share_mode),
            share_ratio=float(args.share_ratio),
            sic_db=float(args.sic_db),
            gain_ratio=float(gain),
            sic_prior_enabled=bool(prior),
        )
        summary[label] = _collect_triplet(label, int(args.episodes_per_load))

    print("\n=== Gate scan summary (mix-level) ===")
    print(
        "setting | mix | feas_min | no_feas_max | ov_feas_min | P(feas|cand)_mean | "
        "rej_rel_mean | intercell_rej_max | embb_mean_Mbps"
    )
    print("-" * 150)
    for label, _g, _p in settings:
        trip = summary[label]
        for mix in ["3_7", "5_5", "7_3"]:
            m = trip[mix]
            print(
                f"{label} | {mix.replace('_',':')} | "
                f"{m['feasible_pair_ratio_min']:.4f} | {m['no_feasible_admit_ratio_max']:.4f} | "
                f"{m['overlay_feasible_pairs_min']:.2f} | {m['p_feasible_given_candidate_mean']:.4f} | "
                f"{m['reject_reliability_ratio_mean']:.4f} | {m['intercell_reject_ratio_max']:.4f} | "
                f"{m['embb_rate_mean_mbps']:.2f}"
            )

    out_path = RESULTS_DIR / "bench_gate_scan_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

