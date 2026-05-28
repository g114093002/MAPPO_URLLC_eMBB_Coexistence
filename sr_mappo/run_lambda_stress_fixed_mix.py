from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SRMAPPOConfig
from .experiments import apply_experiment_preset
from .run_greedy_mix_share_grid import (
    PROJECT_ROOT,
    RESULTS_DIR,
    _build_report_cmd_env,
    _load_full_metrics_payload,
    _load_greedy_metrics,
)

DEFAULT_NEUTRAL_EXPERIMENT = (
    "phase0_joint_full_power_service_interference_repair_v8_phasea_antisaturation_debug"
)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _extract_scalar_metric(payload: dict, key: str, load: float | None = None) -> float | None:
    greedy = dict(payload.get("greedy") or {})
    loads = list(greedy.get("loads") or [])
    values = list(greedy.get(key) or [])
    if not values:
        return None
    if load is None or len(values) == 1 or not loads:
        try:
            return float(values[0])
        except Exception:
            return None
    try:
        load_idx = list(float(x) for x in loads).index(float(load))
    except Exception:
        load_idx = 0
    try:
        return float(values[load_idx])
    except Exception:
        return None


def _maybe_trigger_early_stop(
    state: dict,
    *,
    admitted_packets: float,
    admission_ratio: float,
    min_points: int,
    patience: int,
    admission_floor: float,
    rel_gain: float,
    abs_gain: float,
) -> tuple[bool, str | None]:
    points_seen = int(state.get("points_seen", 0)) + 1
    state["points_seen"] = points_seen

    best_packets = float(state.get("best_packets", float("-inf")))
    if best_packets == float("-inf"):
        state["best_packets"] = float(admitted_packets)
        state["plateau_count"] = 0
        return False, None

    improve_margin = max(float(abs_gain), float(rel_gain) * max(best_packets, 1.0))
    improved = float(admitted_packets) > best_packets + improve_margin
    if improved:
        state["best_packets"] = float(admitted_packets)
        state["plateau_count"] = 0
        return False, None

    if points_seen < int(min_points):
        state["plateau_count"] = 0
        return False, None

    if float(admission_ratio) <= float(admission_floor):
        plateau_count = int(state.get("plateau_count", 0)) + 1
        state["plateau_count"] = plateau_count
    else:
        state["plateau_count"] = 0
        plateau_count = 0

    if plateau_count >= int(patience):
        reason = (
            f"admitted_packets plateaued below gain margin for {plateau_count} lambdas "
            f"while admission stayed low ({float(admission_ratio):.4f} <= {float(admission_floor):.4f})"
        )
        state["stop_reason"] = reason
        return True, reason
    return False, None


def _resolve_pinned_checkpoint_path(experiment: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        resolved = Path(checkpoint_path).expanduser()
        if not resolved.is_absolute():
            resolved = (PROJECT_ROOT / resolved).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit checkpoint path not found: {resolved}")
        return resolved

    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment)
    run_name = str(getattr(cfg.training, "run_name", "") or "").strip()
    if not run_name:
        raise RuntimeError(f"Could not resolve run_name for experiment '{experiment}'.")

    final_path = CHECKPOINT_DIR / f"{run_name}_final.pt"
    if final_path.exists():
        return final_path

    raise FileNotFoundError(
        "Pinned lambda-stress checkpoint not found for experiment "
        f"'{experiment}': expected {final_path}. Refusing to fall back to unrelated latest_any checkpoint."
    )


def _resolve_effective_paired_seed_base(seed_base: int | None) -> int:
    if seed_base is not None:
        return int(seed_base)
    # Generate one seed for the whole sweep so every lambda point shares the
    # same mother scene / channel / subset stream unless the caller overrides it.
    return int(secrets.randbelow(1_000_000_000))


def _plot_lambda_stress(data_by_mix: dict, out_path: Path) -> None:
    mixes = ["10:0", "7:3", "5:5", "3:7"]
    mixes = [m for m in mixes if m in data_by_mix]
    if not mixes:
        return
    metrics = [
        ("embb_rate_mbps", "Aggregate eMBB throughput", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
    ]
    color_map = {"10:0": "#1f77b4", "7:3": "#ff7f0e", "5:5": "#2ca02c", "3:7": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = np.asarray(axes).ravel()
    for ax, (k, title, ylab) in zip(axes, metrics):
        for mix in mixes:
            d = data_by_mix[mix]
            x = np.asarray(d["lambda_per_user_override"], dtype=float)
            y = np.asarray(d[k], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2, color=color_map.get(mix, None), label=mix)
        ax.set_title(title)
        ax.set_xlabel("lambda0 (per URLLC user per slot)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)
    fig.suptitle("Greedy Lambda Stress Test (Fixed Mix Ratio, Per-User Per-Slot lambda0)", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-mix greedy lambda stress test (per-URLLC-user per-slot lambda0 sweep)."
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_NEUTRAL_EXPERIMENT,
        help="Neutral base experiment shared by all mixes. Mix differences come only from ratio override.",
    )
    parser.add_argument(
        "--mixes",
        default="7:3,5:5,3:7",
        help="Comma-separated mix list from {10:0,7:3,5:5,3:7}.",
    )
    parser.add_argument("--episodes-per-load", type=int, default=100)
    parser.add_argument(
        "--loads",
        default="20",
        help="Comma-separated per-UAV loads override. Default: 20 (total users 60 with 3 UAVs)",
    )
    parser.add_argument(
        "--lambdas",
        default="1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10",
        help="Comma-separated per-URLLC-user per-slot lambda0 overrides.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "lambda_stress_fixed_mix"),
        help="Directory for archived metrics and plots.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint path. If omitted, pin to the experiment's *_final checkpoint and fail if missing.",
    )
    parser.add_argument("--paired-seed-base", type=int, default=None)
    parser.add_argument(
        "--parallel-mixes",
        action="store_true",
        help="Run mixes in parallel for each lambda.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="If >0, stop a mix after this many consecutive lambda points with no meaningful admitted-packet gain in low-admission regime.",
    )
    parser.add_argument(
        "--early-stop-min-points",
        type=int,
        default=8,
        help="Minimum number of lambda points seen for a mix before early stop can trigger.",
    )
    parser.add_argument(
        "--early-stop-admission-floor",
        type=float,
        default=0.08,
        help="Only count plateau points toward early stop when URLLC admission ratio is at or below this floor.",
    )
    parser.add_argument(
        "--early-stop-rel-gain",
        type=float,
        default=0.03,
        help="Relative admitted-packet improvement needed to reset plateau counting.",
    )
    parser.add_argument(
        "--early-stop-abs-gain",
        type=float,
        default=24.0,
        help="Absolute admitted-packet improvement needed to reset plateau counting.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pinned_checkpoint = _resolve_pinned_checkpoint_path(str(args.experiment), args.checkpoint_path)
    effective_paired_seed_base = _resolve_effective_paired_seed_base(args.paired_seed_base)
    frozen_scene_tag = (
        f"lambda_stress_fixed_mix_seed{int(effective_paired_seed_base)}"
        f"_load{str(args.loads).replace(',', '_')}"
    )

    mix_pool = {"10:0": 0.0, "7:3": 0.3, "5:5": 0.5, "3:7": 0.7}
    mix_list = [m.strip() for m in str(args.mixes).split(",") if m.strip()]
    invalid_mixes = [m for m in mix_list if m not in mix_pool]
    if invalid_mixes:
        raise ValueError(f"Unsupported mixes: {invalid_mixes}. Allowed: {list(mix_pool.keys())}")
    mixes = {m: mix_pool[m] for m in mix_list}
    lambdas = [float(x.strip()) for x in str(args.lambdas).split(",") if x.strip()]

    all_data: dict = {mix: {} for mix in mixes.keys()}
    detailed_payload = {
        "meta": {
            "mixes": list(mixes.keys()),
            "episodes_per_load": int(args.episodes_per_load),
            "loads": str(args.loads),
            "lambda_per_user_sweep": lambdas,
            "paired_seed_base": int(effective_paired_seed_base),
            "checkpoint_path": str(pinned_checkpoint),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_points": int(args.early_stop_min_points),
            "early_stop_admission_floor": float(args.early_stop_admission_floor),
            "early_stop_rel_gain": float(args.early_stop_rel_gain),
            "early_stop_abs_gain": float(args.early_stop_abs_gain),
        },
        "grid": {},
        "early_stop": {},
    }
    active_mix_labels = list(mixes.keys())
    early_stop_state = {
        mix: {
            "points_seen": 0,
            "best_packets": float("-inf"),
            "plateau_count": 0,
            "stopped": False,
            "stop_lambda": None,
            "stop_reason": None,
        }
        for mix in mixes.keys()
    }

    for lam in lambdas:
        active_mix_labels = [mix for mix in active_mix_labels if not bool(early_stop_state[mix].get("stopped", False))]
        if not active_mix_labels:
            print("[LAMBDA] all mixes stopped early; ending sweep.", flush=True)
            break
        print(f"[LAMBDA] per_user_lambda0={lam:.3f}", flush=True)
        procs: list[tuple[str, subprocess.Popen, Path]] = []
        for mix_label in active_mix_labels:
            ratio = mixes[mix_label]
            mix_key = mix_label.replace(":", "_")
            run_dir = out_dir / f"_run_lambda_{lam:g}_{mix_key}_{int(time.time()*1000)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd, env = _build_report_cmd_env(
                str(args.experiment),
                ratio=ratio,
                share=0.0,
                episodes_per_load=int(args.episodes_per_load),
                loads_override=str(args.loads),
                seed_base=effective_paired_seed_base,
                urllc_poisson_rate_override=lam,
            )
            cmd.extend(["--checkpoint-path", str(pinned_checkpoint)])
            # Clean lambda-stress mode:
            # keep a single neutral greedy preset for every mix and remove
            # mix-specific helpers such as canonical preset realignment,
            # share caps, and 3:7-only assist knobs from the experiment lines.
            env["SR_MAPPO_REPORT_DISABLE_CANONICAL_GREEDY_MIX_REALIGN"] = "1"
            env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = "global_frontier"
            env["SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY"] = "global_sumrate_only"
            env["SR_MAPPO_REPORT_GREEDY_SHARE_MODE_OVERRIDE"] = "none"
            env["SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE"] = "0.0"
            env["SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS"] = ""
            env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
            # Feed the lambda through the report-side override path as well.
            # Some report flows rebuild configs internally and only consult the
            # SR_MAPPO_REPORT_* namespace when serializing the final metrics.
            env["SR_MAPPO_REPORT_URLLC_POISSON_RATE_OVERRIDE"] = f"{float(lam):g}"
            env["SR_MAPPO_REPORT_FIXED_URLLC_POISSON_RATE"] = "1"
            env["SR_MAPPO_REPORT_URLLC_POISSON_PER_USER"] = "1"
            env["SR_MAPPO_REPORT_URLLC_POISSON_SLOT_LEVEL"] = "1"
            env["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
            env["SR_MAPPO_FEASIBLE_GRAPH_FREEZE"] = "1"
            env["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = frozen_scene_tag
            env["SR_MAPPO_FEASIBLE_GRAPH_ID"] = frozen_scene_tag
            env["SR_MAPPO_REPORT_FORCE_FREEZE_ASSOC"] = "1"
            env["SR_MAPPO_REPORT_FORCE_FREEZE_CHANNEL"] = "1"
            env["SR_MAPPO_SCENARIO_GUARDRAIL_ENABLED"] = "0"
            # Episode-internal fast stop:
            # keep every lambda point, but end the current run early once the
            # remaining schedule has no overlay/puncture-feasible opportunity.
            env["SR_MAPPO_EARLY_TERMINATE_WHEN_NO_FUTURE_FEASIBLE_CANDIDATE"] = "1"
            env["SR_MAPPO_RESULTS_DIR_OVERRIDE"] = str(run_dir)
            if args.parallel_mixes:
                procs.append((mix_label, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env), run_dir))
            else:
                print(f"[LAMBDA] run mix={mix_label} lambda0={lam:.3f}", flush=True)
                subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
                procs.append((mix_label, None, run_dir))

        if args.parallel_mixes:
            for mix_label, proc, _run_dir in procs:
                assert proc is not None
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"parallel run failed: mix={mix_label}, lambda={lam:.3f}, rc={rc}")

        for mix_label, _proc, run_dir in procs:
            mix_key = mix_label.replace(":", "_")
            metrics_src = run_dir / "sr_mappo_report_metrics.json"
            metrics_dst = out_dir / f"lambda_{lam:g}_mix_{mix_key}_sr_mappo_report_metrics.json"
            shutil.copy2(metrics_src, metrics_dst)
            d = _load_greedy_metrics(metrics_dst)
            d["lambda_per_user_override"] = [lam for _ in d["loads"]]
            # This runner is intended for fixed load stress, keep one value if user passes one load.
            val_idx = 0
            if len(d["loads"]) > 1:
                try:
                    target_load = float(str(args.loads).split(",")[0])
                    val_idx = list(d["loads"]).index(target_load)
                except Exception:
                    val_idx = 0
            point = {k: (float(v[val_idx]) if isinstance(v, np.ndarray) and v.size else v) for k, v in d.items()}
            point["load"] = float(d["loads"][val_idx]) if len(d["loads"]) else None
            point["lambda_per_user_override"] = float(lam)
            all_data[mix_label].setdefault("lambda_per_user_override", []).append(float(lam))
            for key in [
                "embb_rate_mbps",
                "embb_rate_pre_urllc_admission_mbps",
                "urllc_admission",
                "urllc_tp_mbps",
                "urllc_admitted_packets",
                "budget_used_ratio",
                "avg_embb_loss",
                "embb_loss_per_admit",
                "embb_service_ratio",
                "embb_min_rate",
                "total_power_mw",
            ]:
                all_data[mix_label].setdefault(key, []).append(float(point.get(key, 0.0)))
            detailed_payload["grid"].setdefault(f"lambda_{lam:g}", {})
            full_payload = _load_full_metrics_payload(metrics_dst)
            detailed_payload["grid"][f"lambda_{lam:g}"][mix_label] = full_payload

            if int(args.early_stop_patience) > 0:
                load_for_stop = point.get("load", None)
                payload_admitted = _extract_scalar_metric(full_payload, "scheduled_packets", load=load_for_stop)
                payload_admission = _extract_scalar_metric(full_payload, "urllc_admission", load=load_for_stop)
                stop_now, reason = _maybe_trigger_early_stop(
                    early_stop_state[mix_label],
                    admitted_packets=float(
                        payload_admitted if payload_admitted is not None else point.get("urllc_admitted_packets", 0.0)
                    ),
                    admission_ratio=float(
                        payload_admission if payload_admission is not None else point.get("urllc_admission", 0.0)
                    ),
                    min_points=int(args.early_stop_min_points),
                    patience=int(args.early_stop_patience),
                    admission_floor=float(args.early_stop_admission_floor),
                    rel_gain=float(args.early_stop_rel_gain),
                    abs_gain=float(args.early_stop_abs_gain),
                )
                if stop_now:
                    early_stop_state[mix_label]["stopped"] = True
                    early_stop_state[mix_label]["stop_lambda"] = float(lam)
                    print(
                        f"[LAMBDA][EARLY-STOP] mix={mix_label} at lambda0={lam:.3f} | {reason}",
                        flush=True,
                    )

    detailed_payload["early_stop"] = {
        mix: {
            "points_seen": int(state.get("points_seen", 0)),
            "best_packets": (
                None if float(state.get("best_packets", float("-inf"))) == float("-inf")
                else float(state.get("best_packets", 0.0))
            ),
            "plateau_count": int(state.get("plateau_count", 0)),
            "stopped": bool(state.get("stopped", False)),
            "stop_lambda": (
                None if state.get("stop_lambda", None) is None else float(state.get("stop_lambda"))
            ),
            "stop_reason": state.get("stop_reason", None),
        }
        for mix, state in early_stop_state.items()
    }

    summary_plot = out_dir / "lambda_stress_fixed_mix_comparison.png"
    _plot_lambda_stress(all_data, summary_plot)

    merged_json = out_dir / "lambda_stress_detailed_metrics.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump({"meta": detailed_payload["meta"], "series": all_data, "grid": detailed_payload["grid"]}, f, indent=2)

    print(f"[LAMBDA] done. summary plot: {summary_plot}", flush=True)
    print(f"[LAMBDA] done. merged json: {merged_json}", flush=True)


if __name__ == "__main__":
    main()
