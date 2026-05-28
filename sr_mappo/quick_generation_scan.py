from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_profile(name: str, cmd: list[str], extra_env: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(extra_env)
    print(f"\n=== PROFILE: {name} ===")
    if extra_env:
        print("env_overrides:", extra_env)
    print("cmd:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(completed.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick scan for generation-side levers (fast debug mode).")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,15,21")
    ap.add_argument("--pilot-episodes-per-load", type=int, default=2)
    ap.add_argument("--episodes-per-load", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20260516)
    ap.add_argument("--seed-step", type=int, default=1000)
    ap.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    ap.add_argument("--share-ratio", type=float, default=0.0)
    ap.add_argument("--min-feasible-pair-ratio", type=float, default=0.03)
    ap.add_argument("--max-feasible-ratio-spread", type=float, default=2.5)
    ap.add_argument("--max-no-feasible-admit-ratio", type=float, default=0.92)
    args = ap.parse_args()

    base = [
        sys.executable,
        "-m",
        "sr_mappo.run_fair_mix_with_guardrail_v2",
        "--experiment",
        str(args.experiment),
        "--loads",
        str(args.loads),
        "--pilot-episodes-per-load",
        str(int(args.pilot_episodes_per_load)),
        "--episodes-per-load",
        str(int(args.episodes_per_load)),
        "--seed-base",
        str(int(args.seed_base)),
        "--seed-step",
        str(int(args.seed_step)),
        "--share-mode",
        str(args.share_mode),
        "--share-ratio",
        str(float(args.share_ratio)),
        "--max-attempts",
        str(int(args.max_attempts)),
        "--min-feasible-pair-ratio",
        str(float(args.min_feasible_pair_ratio)),
        "--max-feasible-ratio-spread",
        str(float(args.max_feasible_ratio_spread)),
        "--max-no-feasible-admit-ratio",
        str(float(args.max_no_feasible_admit_ratio)),
    ]

    profiles = [
        {
            "name": "lambda_per_user_down_22",
            "cmd": base
            + [
                "--mother-id-base",
                "quickscan_lambda22_mother",
                "--feasible-graph-id-base",
                "quickscan_lambda22_fg",
                "--out-prefix",
                "quickscan_lambda22",
            ],
            "env": {"SR_MAPPO_URLLC_POISSON_RATE_OVERRIDE": "22"},
        },
        {
            "name": "topology_guardrail_overlay_up",
            "cmd": base
            + [
                "--mother-id-base",
                "quickscan_topology_mother",
                "--feasible-graph-id-base",
                "quickscan_topology_fg",
                "--out-prefix",
                "quickscan_topology",
            ],
            "env": {"SR_MAPPO_SCENARIO_GUARDRAIL_MIN_OVERLAY_FEASIBLE_RATIO": "0.15"},
        },
        {
            "name": "nested_subset_no_fixed",
            "cmd": base
            + [
                "--mother-id-base",
                "quickscan_nested_mother",
                "--feasible-graph-id-base",
                "quickscan_nested_fg",
                "--out-prefix",
                "quickscan_nested",
            ],
            "env": {"SR_MAPPO_REPORT_NESTED_LOAD_SCENARIO": "0"},
        },
    ]

    rc_map: dict[str, int] = {}
    for p in profiles:
        rc_map[p["name"]] = _run_profile(p["name"], p["cmd"], p["env"])

    print("\n=== QUICK SCAN EXIT CODES ===")
    for k, v in rc_map.items():
        print(f"{k}: rc={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
