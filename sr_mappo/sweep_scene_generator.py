from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_csv_floats(text: str) -> List[float]:
    return [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def _safe_name(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _run_scan(
    *,
    experiment: str,
    loads: str,
    seeds: str,
    episodes_per_seed: int,
    out_dir: Path,
    env_updates: Dict[str, str],
    scan_args: argparse.Namespace,
) -> Dict[str, object]:
    env = os.environ.copy()
    env.update(env_updates)
    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.scan_scene_quality",
        "--experiment",
        str(experiment),
        "--loads",
        str(loads),
        "--seeds",
        str(seeds),
        "--episodes-per-seed",
        str(int(episodes_per_seed)),
        "--out-dir",
        str(out_dir),
        "--min-episode-arrivals",
        str(float(scan_args.min_episode_arrivals)),
        "--min-overlay-proxy",
        str(float(scan_args.min_overlay_proxy)),
        "--min-embb-minrate-proxy",
        str(float(scan_args.min_embb_minrate_proxy)),
        "--min-agents-with-feasible-ratio",
        str(float(scan_args.min_agents_with_feasible_ratio)),
        "--min-both-modes-ratio",
        str(float(scan_args.min_both_modes_ratio)),
        "--max-uav-imbalance",
        str(float(scan_args.max_uav_imbalance)),
        "--guardrail-max-resamples",
        str(int(scan_args.guardrail_max_resamples)),
    ]
    if bool(scan_args.freeze_association):
        cmd.append("--freeze-association")
    if bool(scan_args.freeze_channel):
        cmd.append("--freeze-channel")

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    if int(completed.returncode) != 0:
        raise RuntimeError(f"scene quality scan failed: rc={completed.returncode}")

    payload_path = out_dir / "scene_quality_scan.json"
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _overall_summary(payload: Dict[str, object]) -> Dict[str, float]:
    rows = list(payload.get("rows", []))
    if not rows:
        return {
            "pass_ratio": 0.0,
            "learnability": 0.0,
            "overlay_proxy": 0.0,
            "both_modes_ratio": 0.0,
            "feasible_agents_ratio": 0.0,
            "uav_imbalance": 0.0,
        }

    def _mean(key: str) -> float:
        vals = [float(row.get(key, 0.0)) for row in rows]
        return float(sum(vals) / max(len(vals), 1))

    return {
        "pass_ratio": _mean("scene_pass"),
        "learnability": _mean("scene_learnability_score"),
        "overlay_proxy": _mean("guardrail_overlay_proxy"),
        "both_modes_ratio": _mean("both_modes_candidate_ratio"),
        "feasible_agents_ratio": _mean("agents_with_feasible_ratio"),
        "uav_imbalance": _mean("guardrail_uav_imbalance_proxy"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep scene-generation knobs and rank mother-scene learnability.")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,12,15,18")
    ap.add_argument("--seeds", default="20260516,20261516,20262516,20263516")
    ap.add_argument("--episodes-per-seed", type=int, default=2)
    ap.add_argument("--out-dir", default="sr_mappo/results/scene_generator_sweep")
    ap.add_argument("--freeze-association", action="store_true")
    ap.add_argument("--freeze-channel", action="store_true")
    ap.add_argument("--spread-scales", default="0.75,1.0,1.25")
    ap.add_argument("--min-spacings", default="0,20")
    ap.add_argument("--intra-max-dists", default="0,140")
    ap.add_argument("--inter-min-dists", default="0,60")
    ap.add_argument("--highload-spread-scales", default="1.0")
    ap.add_argument("--highload-threshold", type=float, default=0.75)
    ap.add_argument("--min-episode-arrivals", type=float, default=4.0)
    ap.add_argument("--min-overlay-proxy", type=float, default=0.10)
    ap.add_argument("--min-embb-minrate-proxy", type=float, default=0.40)
    ap.add_argument("--min-agents-with-feasible-ratio", type=float, default=0.35)
    ap.add_argument("--min-both-modes-ratio", type=float, default=0.05)
    ap.add_argument("--max-uav-imbalance", type=float, default=1.00)
    ap.add_argument("--guardrail-max-resamples", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spread_scales = _parse_csv_floats(args.spread_scales)
    min_spacings = _parse_csv_floats(args.min_spacings)
    intra_max_dists = _parse_csv_floats(args.intra_max_dists)
    inter_min_dists = _parse_csv_floats(args.inter_min_dists)
    highload_scales = _parse_csv_floats(args.highload_spread_scales)

    combinations = list(
        itertools.product(
            spread_scales,
            min_spacings,
            intra_max_dists,
            inter_min_dists,
            highload_scales,
        )
    )

    ranking_rows: List[Dict[str, object]] = []
    for idx, (spread, min_spacing, intra_max, inter_min, highload_scale) in enumerate(combinations, start=1):
        profile_name = (
            f"s{_safe_name(spread)}"
            f"_ms{_safe_name(min_spacing)}"
            f"_im{_safe_name(intra_max)}"
            f"_xm{_safe_name(inter_min)}"
            f"_hs{_safe_name(highload_scale)}"
        )
        profile_out = out_dir / profile_name
        env_updates = {
            "SR_MAPPO_USER_CLUSTER_SPREAD_SCALE": str(float(spread)),
            "SR_MAPPO_USER_MIN_SPACING": str(float(min_spacing)),
            "SR_MAPPO_USER_INTRA_CLUSTER_MAX_DIST": str(float(intra_max)),
            "SR_MAPPO_USER_INTER_CLUSTER_MIN_DIST": str(float(inter_min)),
            "SR_MAPPO_USER_CLUSTER_SPREAD_HIGHLOAD_THRESHOLD": str(float(args.highload_threshold)),
            "SR_MAPPO_USER_CLUSTER_SPREAD_HIGHLOAD_SCALE": str(float(highload_scale)),
        }
        print(
            f"[SCENE_SWEEP] ({idx}/{len(combinations)}) profile={profile_name} "
            f"spread={spread} minSpacing={min_spacing} intraMax={intra_max} interMin={inter_min} highloadScale={highload_scale}",
            flush=True,
        )
        payload = _run_scan(
            experiment=str(args.experiment),
            loads=str(args.loads),
            seeds=str(args.seeds),
            episodes_per_seed=int(args.episodes_per_seed),
            out_dir=profile_out,
            env_updates=env_updates,
            scan_args=args,
        )
        overall = _overall_summary(payload)
        ranking_rows.append(
            {
                "profile": profile_name,
                "spread_scale": float(spread),
                "min_spacing": float(min_spacing),
                "intra_max_dist": float(intra_max),
                "inter_min_dist": float(inter_min),
                "highload_spread_scale": float(highload_scale),
                **overall,
            }
        )

    ranking_rows.sort(
        key=lambda row: (
            float(row["pass_ratio"]),
            float(row["learnability"]),
            float(row["feasible_agents_ratio"]),
            float(row["both_modes_ratio"]),
        ),
        reverse=True,
    )

    summary = {
        "experiment": str(args.experiment),
        "loads": _parse_csv_floats(args.loads),
        "seeds": _parse_csv_ints(args.seeds),
        "episodes_per_seed": int(args.episodes_per_seed),
        "profiles": ranking_rows,
    }
    (out_dir / "scene_generator_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("[SCENE_SWEEP] top profiles")
    for row in ranking_rows[: min(10, len(ranking_rows))]:
        print(
            f"- {row['profile']} pass={float(row['pass_ratio']):.3f} "
            f"learnability={float(row['learnability']):.3f} "
            f"feasible_agents={float(row['feasible_agents_ratio']):.3f} "
            f"both_modes={float(row['both_modes_ratio']):.3f} "
            f"overlay_proxy={float(row['overlay_proxy']):.3f} "
            f"uav_imbalance={float(row['uav_imbalance']):.3f}"
        )
    print(f"[SCENE_SWEEP] wrote {out_dir / 'scene_generator_sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
