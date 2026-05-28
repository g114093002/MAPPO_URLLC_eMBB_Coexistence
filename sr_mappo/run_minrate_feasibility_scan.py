from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _run_setting(
    label: str,
    minrate_scale: float,
    experiment: str,
    loads: str,
    episodes_per_load: int,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    share_mode: str,
    share_ratio: float,
) -> None:
    out_prefix = f"bench_minratescan_{label}"
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
        "--embb-min-rate-scale",
        f"{float(minrate_scale):.6f}",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _read_metric(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = payload.get("greedy", {}).get(key, [])
    if not arr:
        return 0.0
    return float(arr[0])


def _collect_rows(setting_label: str, episodes_per_load: int) -> List[Tuple[str, float, float, float, float, float, float, float]]:
    rows = []
    for mix_name, req in [("3_7", 0.7), ("5_5", 0.5), ("7_3", 0.3)]:
        p = RESULTS_DIR / f"bench_minratescan_{setting_label}_{mix_name}_e{int(episodes_per_load)}" / "sr_mappo_report_metrics.json"
        row = (
            mix_name.replace("_", ":"),
            req,
            _read_metric(p, "realized_resource_ratio"),
            _read_metric(p, "feasible_pair_ratio"),
            _read_metric(p, "overlay_feasible_pairs"),
            _read_metric(p, "greedy_hf_prefilter_block_mode_infeasible_ratio"),
            _read_metric(p, "greedy_keep_only_when_no_feasible_admit_ratio"),
            _read_metric(p, "greedy_no_feasible_admit_ratio"),
        )
        rows.append(row)
    return rows


def _is_monotonic(rows: List[Tuple[str, float, float, float, float, float, float, float]]) -> bool:
    s = sorted(rows, key=lambda x: x[1], reverse=True)
    vals = [r[2] for r in s]
    return vals[0] > vals[1] and vals[1] > vals[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Single-variable min-rate feasibility lever scan (baseline/mild/strong).")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,12,15,18,21,24")
    ap.add_argument("--episodes-per-load", type=int, default=15)
    ap.add_argument("--seed-base", type=int, default=20260516)
    ap.add_argument("--mother-id", default="fair_mix_clean_mother_v1")
    ap.add_argument("--feasible-graph-id", default="fair_mix_clean_fg_v1")
    ap.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    ap.add_argument("--share-ratio", type=float, default=0.0)
    ap.add_argument("--baseline-minrate-scale", type=float, default=1.0)
    ap.add_argument("--mild-minrate-scale", type=float, default=0.9)
    ap.add_argument("--strong-minrate-scale", type=float, default=0.8)
    args = ap.parse_args()

    settings: List[Tuple[str, float]] = [
        ("baseline", float(args.baseline_minrate_scale)),
        ("mild", float(args.mild_minrate_scale)),
        ("strong", float(args.strong_minrate_scale)),
    ]

    for label, scale in settings:
        print(f"\n=== Running {label} (minrate_scale={scale:.3f}) ===")
        _run_setting(
            label=label,
            minrate_scale=scale,
            experiment=str(args.experiment),
            loads=str(args.loads),
            episodes_per_load=int(args.episodes_per_load),
            seed_base=int(args.seed_base),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
            share_mode=str(args.share_mode),
            share_ratio=float(args.share_ratio),
        )

    print("\n=== Min-rate scan summary ===")
    print(
        "setting | mix | requested_ratio | realized_resource_ratio | gap(realized-requested) | feasible_pair_ratio "
        "| overlay_feasible_pairs | mode_infeasible_ratio | no_feasible_keep_ratio | no_feasible_admit_ratio"
    )
    print("-" * 220)
    monotonic: Dict[str, bool] = {}
    for label, _ in settings:
        rows = _collect_rows(label, int(args.episodes_per_load))
        monotonic[label] = _is_monotonic(rows)
        for (mix, req, real_res, feas_ratio, ov_pairs, mode_inf, no_feas_keep, no_feas_admit) in rows:
            print(
                f"{label} | {mix} | {req:.4f} | {real_res:.4f} | {real_res-req:+.4f} | {feas_ratio:.4f} "
                f"| {ov_pairs:.2f} | {mode_inf:.4f} | {no_feas_keep:.4f} | {no_feas_admit:.4f}"
            )
    print("\n=== Monotonic criterion (0.7 > 0.5 > 0.3 on realized_resource_ratio) ===")
    for label, ok in monotonic.items():
        print(f"{label}: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
