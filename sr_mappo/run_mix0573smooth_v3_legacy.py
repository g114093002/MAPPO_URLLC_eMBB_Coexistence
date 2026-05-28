from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _run_one(
    ratio: float,
    out_dir: Path,
    loads: str,
    episodes_per_load: int,
    seed_base: int,
    experiment: str,
    poisson_rate: float | None,
    fixed_poisson_rate: bool | None,
    poisson_per_user: bool | None,
    poisson_slot_level: bool | None,
    freeze_topology: bool,
    mother_id: str,
    feasible_graph_id: str,
) -> None:
    env = os.environ.copy()
    env["SR_MAPPO_REPORT_LEGACY_MIX_OVERRIDE"] = "1"
    env["SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE"] = f"{float(ratio):.6f}"
    env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(episodes_per_load))
    env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(loads)
    env["SR_MAPPO_REPORT_SEED_BASE"] = str(int(seed_base))
    if poisson_rate is not None:
        env["SR_MAPPO_REPORT_URLLC_POISSON_RATE_OVERRIDE"] = f"{float(poisson_rate):g}"
    if fixed_poisson_rate is not None:
        env["SR_MAPPO_REPORT_FIXED_URLLC_POISSON_RATE"] = "1" if bool(fixed_poisson_rate) else "0"
    if poisson_per_user is not None:
        env["SR_MAPPO_REPORT_URLLC_POISSON_PER_USER"] = "1" if bool(poisson_per_user) else "0"
    if poisson_slot_level is not None:
        env["SR_MAPPO_REPORT_URLLC_POISSON_SLOT_LEVEL"] = "1" if bool(poisson_slot_level) else "0"
    if freeze_topology:
        env["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
        env["SR_MAPPO_FEASIBLE_GRAPH_FREEZE"] = "1"
        env["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = str(mother_id)
        env["SR_MAPPO_FEASIBLE_GRAPH_ID"] = str(feasible_graph_id)

    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        str(experiment),
        "--fast",
        "--greedy-only",
        "--out-dir",
        str(out_dir),
    ]
    print(
        f"[LEGACY] ratio={float(ratio):.3f} experiment={experiment} "
        f"out_dir={out_dir} poisson_rate={poisson_rate!r} "
        f"fixed_poisson_rate={fixed_poisson_rate!r} poisson_per_user={poisson_per_user!r} "
        f"poisson_slot_level={poisson_slot_level!r} "
        f"freeze_topology={int(bool(freeze_topology))}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def _print_summary(out_paths: list[Path]) -> None:
    print("\n[SUMMARY] legacy mix0573smooth_v3")
    for p in out_paths:
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        greedy = payload.get("greedy", {})
        loads = greedy.get("loads", [])
        sched = greedy.get("scheduled_packets", [])
        embb = greedy.get("embb_rate", [])
        print(f"- {p.parent.name}")
        print(f"  loads={loads}")
        print(f"  scheduled_packets={sched}")
        print(f"  embb_rate={[float(x) / 1.0e6 for x in embb]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the legacy mix0573smooth_v3 tri-mix greedy report line.")
    parser.add_argument(
        "--experiment",
        default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug",
    )
    parser.add_argument("--loads", default="9,12,15,18,21,24")
    parser.add_argument("--episodes-per-load", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260518)
    parser.add_argument("--out-prefix", default="bench_owner_topk_mix0573smooth_v3_legacy")
    parser.add_argument(
        "--poisson-rate",
        type=float,
        default=1.0,
        help="Force URLLC Poisson rate override. Use a negative value to inherit the experiment preset.",
    )
    parser.add_argument(
        "--fixed-poisson-rate",
        default="1",
        choices=["inherit", "0", "1"],
        help="Override fixed_urllc_poisson_rate. 'inherit' keeps the experiment preset.",
    )
    parser.add_argument(
        "--poisson-per-user",
        default="1",
        choices=["inherit", "0", "1"],
        help="Override urllc_poisson_rate_is_per_user. '1' is closer to the historical mix37/mix55/mix73 presets.",
    )
    parser.add_argument(
        "--poisson-slot-level",
        default="1",
        choices=["inherit", "0", "1"],
        help="Interpret urllc_poisson_rate as minislot-level lambda. '1' better matches historical mix0573smooth_v3 arrival magnitudes.",
    )
    parser.add_argument(
        "--freeze-topology",
        action="store_true",
        help="Freeze mother topology and feasible graph. Disabled by default because old mix0573smooth_v3 e10 runs were not frozen this way.",
    )
    parser.add_argument("--mother-id", default="feas_geomfix4_mother_a0")
    parser.add_argument("--feasible-graph-id", default="feas_geomfix4_fg_a0")
    parser.add_argument("--single-mix", default="", choices=["", "3_7", "5_5", "7_3"])
    args = parser.parse_args()

    poisson_rate = None if float(args.poisson_rate) < 0.0 else float(args.poisson_rate)
    fixed_poisson_rate = None if args.fixed_poisson_rate == "inherit" else bool(int(args.fixed_poisson_rate))
    poisson_per_user = None if args.poisson_per_user == "inherit" else bool(int(args.poisson_per_user))
    poisson_slot_level = None if args.poisson_slot_level == "inherit" else bool(int(args.poisson_slot_level))

    all_mixes = [("3_7", 0.7), ("5_5", 0.5), ("7_3", 0.3)]
    mixes = [(k, v) for (k, v) in all_mixes if not args.single_mix or k == args.single_mix]

    out_paths: list[Path] = []
    for mix_name, ratio in mixes:
        out_dir = RESULTS_DIR / f"{args.out_prefix}_{mix_name}_e{int(args.episodes_per_load)}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_one(
            ratio=float(ratio),
            out_dir=out_dir,
            loads=str(args.loads),
            episodes_per_load=int(args.episodes_per_load),
            seed_base=int(args.seed_base),
            experiment=str(args.experiment),
            poisson_rate=poisson_rate,
            fixed_poisson_rate=fixed_poisson_rate,
            poisson_per_user=poisson_per_user,
            poisson_slot_level=poisson_slot_level,
            freeze_topology=bool(args.freeze_topology),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
        )
        out_paths.append(out_dir / "sr_mappo_report_metrics.json")

    _print_summary(out_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
