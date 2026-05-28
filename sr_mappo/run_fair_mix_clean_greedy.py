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


CANONICAL_GREEDY_MIX_EXPERIMENT_BY_RATIO = {
    0.0: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix100_debug",
    0.3: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix73_debug",
    0.5: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug",
    0.7: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix37_debug",
    1.0: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix010_debug",
}


def _canonical_mix_experiment(experiment: str, ratio: float) -> str:
    normalized = str(experiment or "").strip().lower()
    if not normalized:
        return str(experiment)
    if "phase0_joint_full_power_service_interference_repair_v8_greedy_" not in normalized:
        return str(experiment)
    for key, canonical in CANONICAL_GREEDY_MIX_EXPERIMENT_BY_RATIO.items():
        if abs(float(ratio) - float(key)) <= 1.0e-9:
            return canonical
    return str(experiment)


def _run_one(
    experiment: str,
    ratio: float,
    out_dir: Path,
    episodes_per_load: int,
    loads: str,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    enable_share: bool,
    share_mode: str,
    share_ratio: float,
    sic_override_db: float | None,
    gain_ratio_override: float | None,
    embb_min_rate_scale: float | None,
    fixed_embb_users: int | None,
    match_embb_across_mix: bool,
    match_embb_mode: str,
    match_embb_core_count: int,
) -> None:
    effective_experiment = _canonical_mix_experiment(experiment, ratio)
    env = os.environ.copy()
    env["SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE"] = f"{ratio:.6f}"
    env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(episodes_per_load))
    env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(loads)
    env["SR_MAPPO_REPORT_SEED_BASE"] = str(int(seed_base))
    env["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
    env["SR_MAPPO_FEASIBLE_GRAPH_FREEZE"] = "1"
    env["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = mother_id
    env["SR_MAPPO_FEASIBLE_GRAPH_ID"] = feasible_graph_id
    env["SR_MAPPO_REPORT_ENABLE_GREEDY_SHARE"] = "1" if enable_share else "0"
    env["SR_MAPPO_REPORT_GREEDY_SHARE_MODE_OVERRIDE"] = share_mode
    env["SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE"] = f"{float(share_ratio):.8f}"
    if sic_override_db is not None:
        env["SR_MAPPO_REPORT_GREEDY_HF_EMBB_MIN_SIC_SNIR_DB_OVERRIDE"] = f"{float(sic_override_db):.6f}"
    if gain_ratio_override is not None:
        env["SR_MAPPO_REPORT_GREEDY_HF_MIN_NOMA_GAIN_RATIO_OVERRIDE"] = f"{float(gain_ratio_override):.6f}"
    if embb_min_rate_scale is not None:
        env["SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE"] = f"{float(embb_min_rate_scale):.6f}"
    if fixed_embb_users is not None and int(fixed_embb_users) > 0:
        env["SR_MAPPO_REPORT_FIXED_EMBB_USERS"] = str(int(fixed_embb_users))
    if bool(match_embb_across_mix):
        env["SR_MAPPO_NESTED_MATCH_EMBB_ACROSS_MIX"] = "1"
        env["SR_MAPPO_NESTED_MATCH_EMBB_MODE"] = str(match_embb_mode).strip().lower()
        if int(match_embb_core_count) > 0:
            env["SR_MAPPO_NESTED_EMBB_CORE_COUNT"] = str(int(match_embb_core_count))

    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        effective_experiment,
        "--fast",
        "--greedy-only",
        "--out-dir",
        str(out_dir),
    ]
    print(
        f"[RUN] ratio={float(ratio):.3f} base_experiment={experiment} "
        f"effective_experiment={effective_experiment}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def _print_quick_mix_diagnostics(out_paths: list[str]) -> None:
    rows = []
    for path in out_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        greedy = payload.get("greedy", {}) if isinstance(payload, dict) else {}
        loads = list(payload.get("loads", []))
        embb = list(greedy.get("embb_rate", []))
        oc = list(greedy.get("overlay_candidate_pairs", []))
        of = list(greedy.get("overlay_feasible_pairs", []))
        nf = list(greedy.get("greedy_no_feasible_admit_ratio", []))
        mix = "?"
        name = p.parent.name
        if "_3_7_" in name:
            mix = "3:7"
        elif "_5_5_" in name:
            mix = "5:5"
        elif "_7_3_" in name:
            mix = "7:3"
        for i, ld in enumerate(loads):
            embb_mbps = float(embb[i]) / 1.0e6 if i < len(embb) else float("nan")
            oc_i = float(oc[i]) if i < len(oc) else float("nan")
            of_i = float(of[i]) if i < len(of) else float("nan")
            feas_ratio = float(of_i / max(oc_i, 1.0e-12)) if (oc_i == oc_i and of_i == of_i) else float("nan")
            no_feas = float(nf[i]) if i < len(nf) else float("nan")
            rows.append((mix, float(ld), embb_mbps, feas_ratio, no_feas))
    if not rows:
        return
    rows.sort(key=lambda x: (x[1], x[0]))
    print("\n[DIAG] quick per-load metrics (mix, load, embb_mbps, feasible_ratio, no_feasible_admit_ratio)")
    for mix, ld, embb_mbps, feas_ratio, no_feas in rows:
        print(
            f"- {mix} | load={ld:.1f} | embb={embb_mbps:.2f} | "
            f"feas={feas_ratio:.4f} | no_feas={no_feas:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean fair-mix greedy comparison with strict freeze and aligned switches.")
    parser.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    parser.add_argument("--loads", default="9,12,15,18,21,24")
    parser.add_argument("--episodes-per-load", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=20260516)
    parser.add_argument("--mother-id", default="fair_mix_clean_mother_v1")
    parser.add_argument("--feasible-graph-id", default="fair_mix_clean_fg_v1")
    parser.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    parser.add_argument("--share-ratio", type=float, default=0.0)
    parser.add_argument("--out-prefix", default="bench_mix_clean")
    parser.add_argument("--sic-override-db", type=float, default=None)
    parser.add_argument("--gain-ratio-override", type=float, default=None)
    parser.add_argument("--embb-min-rate-scale", type=float, default=None)
    parser.add_argument("--fixed-embb-users", type=int, default=None)
    parser.add_argument(
        "--match-embb-across-mix",
        action="store_true",
        help="Force tri-mix to share the same eMBB core subset across mixes "
             "(implemented via SR_MAPPO_NESTED_EMBB_CORE_COUNT).",
    )
    parser.add_argument(
        "--match-embb-mode",
        default="core",
        choices=["core", "exact"],
        help="`core`: shared eMBB core + mix-specific tail. "
             "`exact`: force exact same eMBB IDs across mixes "
             "(auto sets fixed eMBB count to core size).",
    )
    parser.add_argument(
        "--match-embb-core-count",
        type=int,
        default=0,
        help="Core eMBB count used when --match-embb-across-mix is set. "
             "0=auto infer from fixed nested pool total users (uses min eMBB among 3:7/5:5/7:3).",
    )
    parser.add_argument("--skip-cleanliness-audit", action="store_true")
    parser.add_argument(
        "--single-mix",
        default="",
        choices=["", "3_7", "5_5", "7_3"],
        help="Run only one mix. Empty(default) runs all three mixes.",
    )
    args = parser.parse_args()

    all_mixes = [("3_7", 0.7), ("5_5", 0.5), ("7_3", 0.3)]
    if str(args.single_mix).strip():
        mixes = [(k, v) for (k, v) in all_mixes if k == str(args.single_mix).strip()]
    else:
        mixes = list(all_mixes)
    out_paths = []
    inferred_core = int(args.match_embb_core_count or 0)
    fixed_embb_users_effective = (None if args.fixed_embb_users is None else int(args.fixed_embb_users))
    if bool(args.match_embb_across_mix) and inferred_core <= 0:
        # Auto infer a shared eMBB core size from nested fixed-pool total users.
        # min eMBB among URLLC ratios {0.7,0.5,0.3} is at 0.7 => floor(total*(1-0.7)) ~= 0.3*total.
        pool_total_env = os.getenv("SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS", "").strip()
        try:
            pool_total = int(pool_total_env) if pool_total_env else 0
        except Exception:
            pool_total = 0
        if pool_total > 0:
            inferred_core = max(1, int(round(float(pool_total) * 0.3)))
        else:
            # Safe default aligned with your common setup.
            inferred_core = 22
        print(f"[INFO] --match-embb-across-mix enabled; inferred embb core count={inferred_core}")
    if bool(args.match_embb_across_mix) and str(args.match_embb_mode).strip().lower() == "exact":
        # True exact mode should preserve mix semantics (cur_e/cur_u vary by ratio)
        # while removing eMBB subset randomness. Do NOT force fixed eMBB count here.
        fixed_embb_users_effective = (None if args.fixed_embb_users is None else int(args.fixed_embb_users))
        print(
            "[INFO] --match-embb-mode=exact enabled; "
            "using deterministic canonical eMBB-prefix selection in env "
            "(no fixed_embb_users override)."
        )
    for mix_name, ratio in mixes:
        out_dir = RESULTS_DIR / f"{args.out_prefix}_{mix_name}_e{int(args.episodes_per_load)}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_one(
            experiment=str(args.experiment),
            ratio=float(ratio),
            out_dir=out_dir,
            episodes_per_load=int(args.episodes_per_load),
            loads=str(args.loads),
            seed_base=int(args.seed_base),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
            enable_share=True,
            share_mode=str(args.share_mode),
            share_ratio=float(args.share_ratio),
            sic_override_db=(None if args.sic_override_db is None else float(args.sic_override_db)),
            gain_ratio_override=(None if args.gain_ratio_override is None else float(args.gain_ratio_override)),
            embb_min_rate_scale=(None if args.embb_min_rate_scale is None else float(args.embb_min_rate_scale)),
            fixed_embb_users=fixed_embb_users_effective,
            match_embb_across_mix=bool(args.match_embb_across_mix),
            match_embb_mode=str(args.match_embb_mode).strip().lower(),
            match_embb_core_count=int(inferred_core),
        )
        out_paths.append(str(out_dir / "sr_mappo_report_metrics.json"))

    _print_quick_mix_diagnostics(out_paths)

    if bool(args.skip_cleanliness_audit):
        print("Skipping cleanliness audit by request (--skip-cleanliness-audit).")
        return 0

    audit_cmd = [sys.executable, "-m", "sr_mappo.audit_fair_mix_cleanliness", *out_paths]
    print("Running audit:", " ".join(audit_cmd))
    completed = subprocess.run(audit_cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
