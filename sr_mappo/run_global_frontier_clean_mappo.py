from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .run_global_frontier_clean import (
    PROJECT_ROOT,
    RESULTS_DIR,
    _apply_clean_env,
    _normalize_total_system_loads,
)


def _run_one_mappo(
    *,
    experiment: str,
    out_dir: Path,
    checkpoint_path: str | None,
    checkpoint_kind: str | None,
    ratio: float,
    episodes_per_load: int,
    loads: str,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    preset: str,
    resample_mother_scene_each_episode: bool,
    embb_min_rate_scale: float | None,
    greedy_policy: str | None,
    frozen_greedy_json: str | None,
) -> None:
    # Reuse the canonical clean/global-frontier launcher to keep environment
    # overrides aligned with the greedy benchmark family, then swap only the
    # final command from greedy-only report to full MAPPO report.
    effective_resample_mother_scene_each_episode = bool(resample_mother_scene_each_episode) or int(episodes_per_load) > 1
    internal_loads, user_visible_loads, num_uavs = _normalize_total_system_loads(loads)
    env = {
        **os.environ,
        "SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE": f"{ratio:.6f}",
        "SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE": str(int(episodes_per_load)),
        "SR_MAPPO_REPORT_LOADS_OVERRIDE": str(internal_loads),
        "SR_MAPPO_REPORT_SEED_BASE": str(int(seed_base)),
        "SR_MAPPO_MOTHER_TOPOLOGY_FREEZE": "1",
        "SR_MAPPO_FEASIBLE_GRAPH_FREEZE": "1",
        "SR_MAPPO_MOTHER_TOPOLOGY_ID": str(mother_id),
        "SR_MAPPO_FEASIBLE_GRAPH_ID": str(feasible_graph_id),
        "SR_MAPPO_REPORT_SHARED_MOTHER_RESAMPLE_EACH_EPISODE": (
            "1" if effective_resample_mother_scene_each_episode else "0"
        ),
    }
    _apply_clean_env(env, internal_loads, preset=str(preset))
    if str(greedy_policy or "").strip():
        env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = str(greedy_policy).strip()
    if str(frozen_greedy_json or "").strip():
        env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = "frozen_json"
        env["SR_MAPPO_REPORT_FROZEN_GREEDY_JSON"] = str(frozen_greedy_json).strip()
    if embb_min_rate_scale is not None:
        env["SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE"] = f"{float(embb_min_rate_scale):.6f}"

    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        str(experiment),
        "--fast",
        "--out-dir",
        str(out_dir),
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint-path", str(checkpoint_path)])
    if checkpoint_kind:
        cmd.extend(["--checkpoint-kind", str(checkpoint_kind)])

    print(
        f"[RUN][global_frontier_clean_mappo] ratio={float(ratio):.3f} "
        f"experiment={experiment} load_semantics=total_system "
        f"input_loads={user_visible_loads} internal_per_uav_loads={internal_loads} "
        f"num_uavs={int(num_uavs)} out_dir={out_dir} "
        f"shared_mother_resample_each_episode={int(effective_resample_mother_scene_each_episode)}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SR-MAPPO on the same clean global-frontier benchmark family as the greedy launcher."
    )
    parser.add_argument(
        "--experiment",
        default="pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_globaltp_adm_quickcheck_v8_small_edit_unlock",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--checkpoint-kind",
        default="best_balanced",
        choices=[
            "best_throughput",
            "best_balanced",
            "best_v5_balanced_intercell_admission",
            "best_v6_balanced_puncture_accounting",
            "latest",
            "final",
            "best",
        ],
    )
    parser.add_argument("--loads", default="9,12,15,18,21,24")
    parser.add_argument("--episodes-per-load", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=246813579)
    parser.add_argument("--mother-id", default="fair_mix_clean_mother_v1")
    parser.add_argument("--feasible-graph-id", default="fair_mix_clean_fg_global_clean_v1")
    parser.add_argument("--out-prefix", default="global_frontier_clean_mappo")
    parser.add_argument("--preset", default="v10_rate")
    parser.add_argument("--single-mix", default="5_5", choices=["", "7_3", "5_5", "3_7"])
    parser.add_argument(
        "--resample-mother-scene-each-episode",
        action="store_true",
        help="Resample one shared mother scene per episode. This is also auto-enabled whenever episodes-per-load > 1.",
    )
    parser.add_argument("--embb-min-rate-scale", type=float, default=None)
    parser.add_argument("--greedy-policy", default=None)
    parser.add_argument(
        "--frozen-greedy-json",
        default=None,
        help="Reuse a previously frozen greedy baseline JSON instead of rerunning greedy in the report.",
    )
    args = parser.parse_args()

    all_mixes = [("7_3", 0.3), ("5_5", 0.5), ("3_7", 0.7)]
    if str(args.single_mix).strip():
        mixes = [(k, v) for (k, v) in all_mixes if k == str(args.single_mix).strip()]
    else:
        mixes = list(all_mixes)

    for mix_name, ratio in mixes:
        out_dir = RESULTS_DIR / f"{args.out_prefix}_{mix_name}_e{int(args.episodes_per_load)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_one_mappo(
            experiment=str(args.experiment),
            out_dir=out_dir,
            checkpoint_path=args.checkpoint_path,
            checkpoint_kind=args.checkpoint_kind,
            ratio=float(ratio),
            episodes_per_load=int(args.episodes_per_load),
            loads=str(args.loads),
            seed_base=int(args.seed_base),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
            preset=str(args.preset),
            resample_mother_scene_each_episode=bool(args.resample_mother_scene_each_episode),
            embb_min_rate_scale=args.embb_min_rate_scale,
            greedy_policy=args.greedy_policy,
            frozen_greedy_json=args.frozen_greedy_json,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
