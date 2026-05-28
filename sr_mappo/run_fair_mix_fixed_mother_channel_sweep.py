from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _run(cmd: List[str], env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(p.returncode)


def _read(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _adm_mean(path: Path) -> float:
    payload = _read(path)
    rows = payload.get("greedy", {}).get("greedy_episode_admitted_samples", [])
    vals: List[float] = []
    for r in rows:
        if isinstance(r, list):
            vals.extend(float(x) for x in r)
    return float(sum(vals) / max(len(vals), 1)) if vals else 0.0


def _embb_mean(path: Path) -> float:
    payload = _read(path)
    arr = payload.get("greedy", {}).get("embb_rate", [])
    if not isinstance(arr, list) or not arr:
        return 0.0
    vals = [float(x) for x in arr]
    return float(sum(vals) / max(len(vals), 1))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-mother channel-seed sweep for fair tri-mix.")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,15,24")
    ap.add_argument("--episodes-per-load", type=int, default=5)
    ap.add_argument("--seed-list", default="20260516,20261516,20262516")
    ap.add_argument("--mother-id", required=True)
    ap.add_argument("--feasible-graph-id", required=True)
    ap.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    ap.add_argument("--share-ratio", type=float, default=0.0)
    ap.add_argument("--out-prefix", default="bench_mix_fixedmother_channelsweep")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seed_list).split(",") if x.strip()]
    print(f"[INFO] seeds={seeds}")

    for idx, seed in enumerate(seeds):
        run_prefix = f"{args.out_prefix}_s{seed}"
        env = dict(**__import__("os").environ)
        # Keep geometry fixed but allow channel refresh across episodes/seeds.
        env["SR_MAPPO_REPORT_FORCE_FREEZE_ASSOC"] = "1"
        env["SR_MAPPO_REPORT_FORCE_FREEZE_CHANNEL"] = "0"

        cmd = [
            sys.executable,
            "-m",
            "sr_mappo.run_fair_mix_clean_greedy",
            "--experiment", str(args.experiment),
            "--loads", str(args.loads),
            "--episodes-per-load", str(int(args.episodes_per_load)),
            "--seed-base", str(int(seed)),
            "--mother-id", str(args.mother_id),
            "--feasible-graph-id", str(args.feasible_graph_id),
            "--share-mode", str(args.share_mode),
            "--share-ratio", str(float(args.share_ratio)),
            "--out-prefix", run_prefix,
            "--skip-cleanliness-audit",
        ]
        print("\n[RUN]", " ".join(cmd))
        rc = _run(cmd, env)
        if rc != 0:
            print(f"[FAIL] seed={seed} rc={rc}")
            continue

        p37 = RESULTS_DIR / f"{run_prefix}_3_7_e{int(args.episodes_per_load)}" / "sr_mappo_report_metrics.json"
        p55 = RESULTS_DIR / f"{run_prefix}_5_5_e{int(args.episodes_per_load)}" / "sr_mappo_report_metrics.json"
        p73 = RESULTS_DIR / f"{run_prefix}_7_3_e{int(args.episodes_per_load)}" / "sr_mappo_report_metrics.json"

        a37, a55, a73 = _adm_mean(p37), _adm_mean(p55), _adm_mean(p73)
        e37, e55, e73 = _embb_mean(p37), _embb_mean(p55), _embb_mean(p73)

        adm_pass = bool(a37 > a55 > a73)
        embb_pass = bool(e73 > e55 > e37)
        print(
            f"[SEED {seed}] admitted(3:7,5:5,7:3)=({a37:.2f},{a55:.2f},{a73:.2f}) pass={adm_pass} | "
            f"embbMbps(7:3,5:5,3:7)=({e73/1e6:.2f},{e55/1e6:.2f},{e37/1e6:.2f}) pass={embb_pass}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
