from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .compare import _build_main_like_configs, _configure_density_scenario
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset
from .types import HybridAction


def _parse_csv_floats(text: str) -> List[float]:
    return [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


@contextmanager
def _temporary_env(updates: Dict[str, str]):
    old: Dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            old[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_score(value: float, low: float, high: float, inverse: bool = False) -> float:
    if not np.isfinite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= high else 0.0
    clipped = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return float(1.0 - clipped) if inverse else clipped


def _collect_episode_observation_metrics(
    env: SRMAPPOPhaseAEnv,
    observations: Dict,
) -> Dict[str, float]:
    candidate_counts: List[int] = []
    feasible_counts: List[int] = []
    overlay_counts: List[int] = []
    puncture_counts: List[int] = []
    both_mode_counts: List[int] = []
    admit_agent_flags: List[float] = []
    multi_feasible_agent_flags: List[float] = []
    active_agent_flags: List[float] = []
    step_candidate_presence: List[float] = []
    step_feasible_presence: List[float] = []

    current_obs = observations
    while True:
        step_has_candidate = False
        step_has_feasible = False
        for obs in current_obs.values():
            candidates = list(obs.candidates)
            count_here = len(candidates)
            candidate_counts.append(count_here)
            feasible_here = sum(int(bool(c.overlay_feasible) or bool(c.puncture_feasible)) for c in candidates)
            overlay_here = sum(int(bool(c.overlay_feasible)) for c in candidates)
            puncture_here = sum(int(bool(c.puncture_feasible)) for c in candidates)
            both_here = sum(int(bool(c.overlay_feasible) and bool(c.puncture_feasible)) for c in candidates)
            feasible_counts.append(feasible_here)
            overlay_counts.append(overlay_here)
            puncture_counts.append(puncture_here)
            both_mode_counts.append(both_here)
            admit_agent_flags.append(float(feasible_here > 0))
            multi_feasible_agent_flags.append(float(feasible_here >= 2))
            active_agent_flags.append(float(count_here > 0))
            step_has_candidate = step_has_candidate or (count_here > 0)
            step_has_feasible = step_has_feasible or (feasible_here > 0)
        step_candidate_presence.append(float(step_has_candidate))
        step_feasible_presence.append(float(step_has_feasible))

        if bool(getattr(env, "episode_done", False)):
            break
        joint_keep = {agent_id: HybridAction() for agent_id in env.agent_ids}
        current_obs, _rewards, dones, _info = env.step(joint_keep)
        if all(bool(v) for v in dones.values()):
            break

    total_candidates = float(sum(candidate_counts))
    total_feasible = float(sum(feasible_counts))
    total_overlay = float(sum(overlay_counts))
    total_puncture = float(sum(puncture_counts))
    total_both = float(sum(both_mode_counts))
    arrivals = np.asarray(getattr(env, "packet_arrivals_by_minislot", []), dtype=float)
    total_arrivals = float(np.sum(arrivals)) if arrivals.size else 0.0
    active_slot0 = float(arrivals[0]) if arrivals.size > 0 else 0.0

    overlay_proxy = float(getattr(env, "guardrail_actual_overlay_feasible_ratio", 0.0) or 0.0)
    minrate_proxy = float(getattr(env, "guardrail_actual_embb_minrate_ratio", 0.0) or 0.0)
    imbalance_proxy = float(getattr(env, "guardrail_actual_uav_load_imbalance", 0.0) or 0.0)

    score_components = [
        0.30 * _normalize_score(overlay_proxy, 0.05, 0.25),
        0.20 * _normalize_score(minrate_proxy, 0.20, 0.70),
        0.15 * _normalize_score(
            float(np.mean(admit_agent_flags)) if admit_agent_flags else 0.0,
            0.20,
            0.80,
        ),
        0.15 * _normalize_score(
            float(total_both / max(total_candidates, 1.0)),
            0.02,
            0.20,
        ),
        0.10 * _normalize_score(
            float(np.mean(multi_feasible_agent_flags)) if multi_feasible_agent_flags else 0.0,
            0.10,
            0.60,
        ),
        0.10 * _normalize_score(imbalance_proxy, 0.15, 1.00, inverse=True),
    ]

    return {
        "episode_total_arrivals": total_arrivals,
        "slot0_arrivals": active_slot0,
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "candidate_count_max": float(np.max(candidate_counts)) if candidate_counts else 0.0,
        "step_has_candidate_ratio": float(np.mean(step_candidate_presence)) if step_candidate_presence else 0.0,
        "step_has_feasible_ratio": float(np.mean(step_feasible_presence)) if step_feasible_presence else 0.0,
        "feasible_candidate_ratio": float(total_feasible / max(total_candidates, 1.0)),
        "overlay_candidate_ratio": float(total_overlay / max(total_candidates, 1.0)),
        "puncture_candidate_ratio": float(total_puncture / max(total_candidates, 1.0)),
        "both_modes_candidate_ratio": float(total_both / max(total_candidates, 1.0)),
        "agents_with_feasible_ratio": float(np.mean(admit_agent_flags)) if admit_agent_flags else 0.0,
        "agents_with_multi_feasible_ratio": float(np.mean(multi_feasible_agent_flags)) if multi_feasible_agent_flags else 0.0,
        "agents_with_any_candidate_ratio": float(np.mean(active_agent_flags)) if active_agent_flags else 0.0,
        "guardrail_overlay_proxy": overlay_proxy,
        "guardrail_embb_minrate_proxy": minrate_proxy,
        "guardrail_uav_imbalance_proxy": imbalance_proxy,
        "guardrail_resample_count": float(getattr(env, "guardrail_resample_count", 0.0) or 0.0),
        "scene_learnability_score": float(sum(score_components)),
    }


def _scene_pass(metrics: Dict[str, float], args) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if float(metrics["episode_total_arrivals"]) < float(args.min_episode_arrivals):
        reasons.append("low_arrivals")
    if float(metrics["guardrail_overlay_proxy"]) < float(args.min_overlay_proxy):
        reasons.append("low_overlay_proxy")
    if float(metrics["guardrail_embb_minrate_proxy"]) < float(args.min_embb_minrate_proxy):
        reasons.append("low_embb_minrate_proxy")
    if float(metrics["agents_with_feasible_ratio"]) < float(args.min_agents_with_feasible_ratio):
        reasons.append("low_feasible_agent_ratio")
    if float(metrics["both_modes_candidate_ratio"]) < float(args.min_both_modes_ratio):
        reasons.append("low_both_modes_ratio")
    if float(metrics["guardrail_uav_imbalance_proxy"]) > float(args.max_uav_imbalance):
        reasons.append("high_uav_imbalance")
    return len(reasons) == 0, reasons


def _aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_seed: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)

    summary: List[Dict[str, object]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        pass_count = sum(int(bool(row["scene_pass"])) for row in seed_rows)
        summary.append(
            {
                "seed": int(seed),
                "cases": int(len(seed_rows)),
                "pass_count": int(pass_count),
                "pass_ratio": float(pass_count / max(len(seed_rows), 1)),
                "mean_learnability_score": float(
                    np.mean([float(row["scene_learnability_score"]) for row in seed_rows])
                ),
                "mean_overlay_proxy": float(
                    np.mean([float(row["guardrail_overlay_proxy"]) for row in seed_rows])
                ),
                "mean_both_modes_ratio": float(
                    np.mean([float(row["both_modes_candidate_ratio"]) for row in seed_rows])
                ),
                "mean_feasible_agent_ratio": float(
                    np.mean([float(row["agents_with_feasible_ratio"]) for row in seed_rows])
                ),
            }
        )
    return summary


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan mother seeds for scene learnability without running greedy or MAPPO.")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug", choices=EXPERIMENT_CHOICES)
    ap.add_argument("--loads", default="9,12,15,18,21,24")
    ap.add_argument("--seeds", default="20260516,20261516,20262516")
    ap.add_argument("--episodes-per-seed", type=int, default=3)
    ap.add_argument("--out-dir", default="sr_mappo/results/scene_quality_scan")
    ap.add_argument("--freeze-association", action="store_true")
    ap.add_argument("--freeze-channel", action="store_true")
    ap.add_argument(
        "--fixed-embb-baseline-policy",
        default="",
        choices=["", "global_sumrate_only", "deterministic_max_gain", "balanced_round_robin", "greedy"],
    )
    ap.add_argument("--fixed-subset-across-loads", action="store_true")
    ap.add_argument("--fixed-subset-across-episodes", action="store_true")
    ap.add_argument(
        "--embb-order-mode",
        default="",
        choices=["", "hybrid", "feasible", "global_throughput"],
    )
    ap.add_argument("--embb-balance-by-uav", type=int, choices=[0, 1], default=-1)
    ap.add_argument("--embb-fixed-prefix-only", action="store_true")
    ap.add_argument("--subset-pool-count", type=int, default=0)
    ap.add_argument("--subset-jitter-window", type=int, default=0)
    ap.add_argument("--min-episode-arrivals", type=float, default=4.0)
    ap.add_argument("--min-overlay-proxy", type=float, default=0.10)
    ap.add_argument("--min-embb-minrate-proxy", type=float, default=0.40)
    ap.add_argument("--min-agents-with-feasible-ratio", type=float, default=0.35)
    ap.add_argument("--min-both-modes-ratio", type=float, default=0.05)
    ap.add_argument("--max-uav-imbalance", type=float, default=1.00)
    ap.add_argument("--guardrail-max-resamples", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    loads = _parse_csv_floats(args.loads)
    seeds = _parse_csv_ints(args.seeds)

    cfg = apply_experiment_preset(SRMAPPOConfig(), str(args.experiment))
    cfg.env.freeze_association_across_episodes = bool(args.freeze_association)
    cfg.env.freeze_channel_gains_across_episodes = bool(args.freeze_channel)
    cfg.reward.use_greedy_terminal_reference = False
    if str(args.fixed_embb_baseline_policy).strip():
        cfg.env.fixed_embb_baseline_policy = str(args.fixed_embb_baseline_policy).strip().lower()
    if bool(args.fixed_subset_across_loads):
        cfg.env.nested_fixed_user_subset_across_loads = True
    if bool(args.fixed_subset_across_episodes):
        cfg.env.nested_fixed_user_subset_across_episodes = True

    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()

    env_updates = {
        "SR_MAPPO_SCENARIO_GUARDRAIL_ENABLED": "1",
        "SR_MAPPO_SCENARIO_GUARDRAIL_MAX_RESAMPLES": str(int(args.guardrail_max_resamples)),
        "SR_MAPPO_SCENARIO_GUARDRAIL_MIN_OVERLAY_FEASIBLE_RATIO": str(float(args.min_overlay_proxy)),
        "SR_MAPPO_SCENARIO_GUARDRAIL_MIN_EMBB_MINRATE_RATIO": str(float(args.min_embb_minrate_proxy)),
        "SR_MAPPO_SCENARIO_GUARDRAIL_MAX_UAV_LOAD_IMBALANCE": str(float(args.max_uav_imbalance)),
    }
    if str(args.embb_order_mode).strip():
        env_updates["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = str(args.embb_order_mode).strip().lower()
    if int(args.embb_balance_by_uav) in {0, 1}:
        env_updates["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = str(int(args.embb_balance_by_uav))
    if bool(args.embb_fixed_prefix_only):
        env_updates["SR_MAPPO_NESTED_EMBB_FIXED_PREFIX_ONLY"] = "1"
    if int(args.subset_pool_count) > 0:
        env_updates["SR_MAPPO_NESTED_SUBSET_POOL_COUNT"] = str(int(args.subset_pool_count))
    if int(args.subset_jitter_window) > 0:
        env_updates["SR_MAPPO_NESTED_SUBSET_JITTER_WINDOW"] = str(int(args.subset_jitter_window))

    rows: List[Dict[str, object]] = []
    with _temporary_env(env_updates):
        for load in loads:
            sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
                float(load),
                base_sys,
                base_urllc,
                base_embb,
                base_algo,
                base_sim,
            )
            env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)
            for seed in seeds:
                for episode_idx in range(max(int(args.episodes_per_seed), 1)):
                    episode_seed = int(seed + episode_idx)
                    observations, _ = env.reset(seed=episode_seed)
                    metrics = _collect_episode_observation_metrics(env, observations)
                    passed, reasons = _scene_pass(metrics, args)
                    row: Dict[str, object] = {
                        "experiment": str(args.experiment),
                        "load": float(load),
                        "seed": int(seed),
                        "episode_seed": int(episode_seed),
                        "scene_pass": bool(passed),
                        "reject_reasons": ",".join(reasons),
                        "same_assoc_hash": str(getattr(env, "same_assoc_hash", "")),
                        "same_channel_hash": str(getattr(env, "same_channel_hash", "")),
                        "mix_user_subset_hash": str(getattr(env, "mix_user_subset_hash", "")),
                        "same_feasible_graph_hash": str(getattr(env, "same_feasible_graph_hash", "")),
                    }
                    row.update(metrics)
                    rows.append(row)

    seed_summary = _aggregate_rows(rows)
    payload = {
        "experiment": str(args.experiment),
        "loads": [float(x) for x in loads],
        "seeds": [int(x) for x in seeds],
        "episodes_per_seed": int(args.episodes_per_seed),
        "thresholds": {
            "min_episode_arrivals": float(args.min_episode_arrivals),
            "min_overlay_proxy": float(args.min_overlay_proxy),
            "min_embb_minrate_proxy": float(args.min_embb_minrate_proxy),
            "min_agents_with_feasible_ratio": float(args.min_agents_with_feasible_ratio),
            "min_both_modes_ratio": float(args.min_both_modes_ratio),
            "max_uav_imbalance": float(args.max_uav_imbalance),
        },
        "seed_summary": seed_summary,
        "rows": rows,
    }

    _write_json(out_dir / "scene_quality_scan.json", payload)
    _write_csv(out_dir / "scene_quality_scan.csv", rows)

    print("[SCENE_SCAN] seed summary")
    for item in seed_summary:
        print(
            f"- seed={int(item['seed'])} pass_ratio={float(item['pass_ratio']):.3f} "
            f"learnability={float(item['mean_learnability_score']):.3f} "
            f"overlay_proxy={float(item['mean_overlay_proxy']):.3f} "
            f"both_modes={float(item['mean_both_modes_ratio']):.3f} "
            f"feasible_agents={float(item['mean_feasible_agent_ratio']):.3f}"
        )
    print(f"[SCENE_SCAN] wrote {out_dir / 'scene_quality_scan.json'}")
    print(f"[SCENE_SCAN] wrote {out_dir / 'scene_quality_scan.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
