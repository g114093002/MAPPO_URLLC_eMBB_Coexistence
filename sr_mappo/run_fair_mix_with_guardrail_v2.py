from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _write_selected_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_rejected_json_path(selected_path: Path) -> Path:
    stem = selected_path.stem
    if stem.endswith("_selected"):
        stem = stem[: -len("_selected")]
    return selected_path.with_name(f"{stem}_rejected.json")


def _run_clean_mix_once(
    out_prefix: str,
    episodes_per_load: int,
    experiment: str,
    loads: str,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    share_mode: str,
    share_ratio: float,
) -> None:
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
        "--skip-cleanliness-audit",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _metric_head(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = payload.get("greedy", {}).get(key, [])
    return float(arr[0]) if arr else 0.0


def _metric_values(path: Path, key: str) -> List[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = payload.get("greedy", {}).get(key, [])
    if not isinstance(arr, list):
        return []
    return [float(x) for x in arr]


def _collect_triplet(out_prefix: str, episodes_per_load: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for mix_name in ["3_7", "5_5", "7_3"]:
        p = RESULTS_DIR / f"{out_prefix}_{mix_name}_e{int(episodes_per_load)}" / "sr_mappo_report_metrics.json"
        feas_vals = _metric_values(p, "feasible_pair_ratio")
        nofeas_vals = _metric_values(p, "greedy_no_feasible_admit_ratio")
        ovf_vals = _metric_values(p, "overlay_feasible_pairs")
        punc_vals = _metric_values(p, "puncture_count")
        intercell_vals = _metric_values(p, "phase_a_rejected_intercell_per_decision")
        req_vals = _metric_values(p, "requested_mix_ratio")
        real_vals = _metric_values(p, "realized_resource_ratio")
        out[mix_name] = {
            # Use load-aggregated stats instead of head(load0) only.
            "feasible_pair_ratio": min(feas_vals) if feas_vals else 0.0,
            "no_feasible_admit_ratio": max(nofeas_vals) if nofeas_vals else 0.0,
            "overlay_feasible_pairs": min(ovf_vals) if ovf_vals else 0.0,
            "puncture_count": (sum(punc_vals) / max(len(punc_vals), 1)) if punc_vals else 0.0,
            "phase_a_rejected_intercell_per_decision": max(intercell_vals) if intercell_vals else 0.0,
            "requested_mix_ratio": (sum(req_vals) / max(len(req_vals), 1)) if req_vals else 0.0,
            "realized_resource_ratio": (sum(real_vals) / max(len(real_vals), 1)) if real_vals else 0.0,
            "admitted_packets_mean": _admitted_packets_mean(p),
            "embb_rate_mean": _metric_mean(p, "embb_rate"),
        }
    return out


def _metric_mean(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arr = payload.get("greedy", {}).get(key, [])
    if not isinstance(arr, list) or not arr:
        return 0.0
    vals = [float(x) for x in arr]
    return float(sum(vals) / max(len(vals), 1))


def _admitted_packets_mean(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("greedy", {}).get("greedy_episode_admitted_samples", [])
    flat = []
    for r in rows:
        if isinstance(r, list):
            flat.extend(float(x) for x in r)
    return float(sum(flat) / max(len(flat), 1)) if flat else 0.0


def _admitted_monotonic_pass(m: Dict[str, Dict[str, float]]) -> bool:
    return bool(
        float(m["3_7"]["admitted_packets_mean"])
        > float(m["5_5"]["admitted_packets_mean"])
        > float(m["7_3"]["admitted_packets_mean"])
    )


def _embb_monotonic_pass(m: Dict[str, Dict[str, float]]) -> bool:
    # eMBB-heavy mix should yield higher eMBB throughput.
    # 7:3 (urllc=0.3) > 5:5 > 3:7 (urllc=0.7)
    return bool(
        float(m["7_3"]["embb_rate_mean"])
        > float(m["5_5"]["embb_rate_mean"])
        > float(m["3_7"]["embb_rate_mean"])
    )


def _guardrail_pass(
    m: Dict[str, Dict[str, float]],
    min_feasible_pair_ratio: float,
    max_feasible_ratio_spread: float,
    max_no_feasible_admit_ratio: float,
    min_overlay_feasible_pairs: float = 0.0,
    min_puncture_feasible_pairs: float = 0.0,
    max_intercell_interference_rejection_ratio: float = 1.0,
    require_admitted_monotonic: bool = False,
    require_embb_monotonic: bool = False,
) -> Tuple[bool, str]:
    feas = [m[k]["feasible_pair_ratio"] for k in ["3_7", "5_5", "7_3"]]
    nof = [m[k]["no_feasible_admit_ratio"] for k in ["3_7", "5_5", "7_3"]]
    ovf = [m[k]["overlay_feasible_pairs"] for k in ["3_7", "5_5", "7_3"]]
    punc = [m[k]["puncture_count"] for k in ["3_7", "5_5", "7_3"]]
    intercell_rej = [m[k]["phase_a_rejected_intercell_per_decision"] for k in ["3_7", "5_5", "7_3"]]
    feas_min = min(feas)
    feas_max = max(feas)
    ovf_min = min(ovf)
    punc_min = min(punc)
    intercell_rej_max = max(intercell_rej)
    spread = (feas_max / max(feas_min, 1.0e-12)) if feas_min > 0.0 else float("inf")
    nof_max = max(nof)

    if feas_min < float(min_feasible_pair_ratio):
        return False, f"min feasible_pair_ratio too low: {feas_min:.4f} < {min_feasible_pair_ratio:.4f}"
    if spread > float(max_feasible_ratio_spread):
        return False, f"feasible_pair_ratio spread too wide: {spread:.4f} > {max_feasible_ratio_spread:.4f}"
    if nof_max > float(max_no_feasible_admit_ratio):
        return False, f"no_feasible_admit_ratio too high: {nof_max:.4f} > {max_no_feasible_admit_ratio:.4f}"
    if ovf_min < float(min_overlay_feasible_pairs):
        return False, f"min overlay_feasible_pairs too low: {ovf_min:.4f} < {min_overlay_feasible_pairs:.4f}"
    # Note: direct puncture_feasible_pairs metric is not exposed in report metrics yet.
    # We use puncture_count as a conservative operational proxy for puncture-space viability.
    if punc_min < float(min_puncture_feasible_pairs):
        return False, f"min puncture_feasible_pairs(proxy=puncture_count) too low: {punc_min:.4f} < {min_puncture_feasible_pairs:.4f}"
    if intercell_rej_max > float(max_intercell_interference_rejection_ratio):
        return False, (
            "inter-cell interference rejection ratio too high: "
            f"{intercell_rej_max:.4f} > {max_intercell_interference_rejection_ratio:.4f}"
        )
    if bool(require_admitted_monotonic) and not _admitted_monotonic_pass(m):
        a37 = float(m["3_7"]["admitted_packets_mean"])
        a55 = float(m["5_5"]["admitted_packets_mean"])
        a73 = float(m["7_3"]["admitted_packets_mean"])
        return False, f"URLLC admitted not monotonic: 3:7={a37:.2f}, 5:5={a55:.2f}, 7:3={a73:.2f}"
    if bool(require_embb_monotonic) and not _embb_monotonic_pass(m):
        t37 = float(m["3_7"]["embb_rate_mean"]) / 1.0e6
        t55 = float(m["5_5"]["embb_rate_mean"]) / 1.0e6
        t73 = float(m["7_3"]["embb_rate_mean"]) / 1.0e6
        return False, f"eMBB throughput not monotonic: 7:3={t73:.2f}Mbps, 5:5={t55:.2f}Mbps, 3:7={t37:.2f}Mbps"
    return True, "pass"


def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario generation guardrail v2 for fair tri-mix comparisons.")
    ap.add_argument("--experiment", default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug")
    ap.add_argument("--loads", default="9,12,15,18,21,24")
    ap.add_argument("--episodes-per-load", type=int, default=15)
    ap.add_argument("--pilot-episodes-per-load", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=20260516)
    ap.add_argument("--seed-step", type=int, default=1000)
    ap.add_argument("--mother-id-base", default="fair_mix_guardrailv2_mother")
    ap.add_argument("--feasible-graph-id-base", default="fair_mix_guardrailv2_fg")
    ap.add_argument("--share-mode", default="none", choices=["none", "fixed_share"])
    ap.add_argument("--share-ratio", type=float, default=0.0)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--min-feasible-pair-ratio", type=float, default=0.12)
    ap.add_argument("--max-feasible-ratio-spread", type=float, default=1.5)
    ap.add_argument("--max-no-feasible-admit-ratio", type=float, default=0.75)
    ap.add_argument("--min-overlay-feasible-pairs", type=float, default=0.0)
    ap.add_argument("--min-puncture-feasible-pairs", type=float, default=0.0)
    ap.add_argument("--max-intercell-interference-rejection-ratio", type=float, default=1.0)
    ap.add_argument("--require-admitted-monotonic-pilot", action="store_true")
    ap.add_argument("--require-admitted-monotonic-full", action="store_true")
    ap.add_argument("--require-embb-monotonic-pilot", action="store_true")
    ap.add_argument("--require-embb-monotonic-full", action="store_true")
    ap.add_argument("--out-prefix", default="bench_mix_guardrailv2")
    ap.add_argument(
        "--continue-after-selected",
        action="store_true",
        help="Keep scanning until max-attempts and collect all passing scenarios instead of stopping at first pass.",
    )
    ap.add_argument(
        "--selected-output-json",
        default="",
        help="Optional path to save selected scenario metadata as JSON. "
             "If omitted, defaults to sr_mappo/results/<out-prefix>_selected.json",
    )
    args = ap.parse_args()

    selected = None
    selected_all: List[Dict] = []
    for attempt in range(int(args.max_attempts)):
        seed = int(args.seed_base) + attempt * int(args.seed_step)
        mother_id = f"{args.mother_id_base}_a{attempt}"
        fg_id = f"{args.feasible_graph_id_base}_a{attempt}"
        pilot_prefix = f"{args.out_prefix}_pilot_a{attempt}"

        for mix_name in ["3_7", "5_5", "7_3"]:
            d = RESULTS_DIR / f"{pilot_prefix}_{mix_name}_e{int(args.pilot_episodes_per_load)}"
            if d.exists():
                shutil.rmtree(d)

        print(f"\n[attempt {attempt}] pilot seed={seed} mother={mother_id} fg={fg_id}")
        _run_clean_mix_once(
            out_prefix=pilot_prefix,
            episodes_per_load=int(args.pilot_episodes_per_load),
            experiment=str(args.experiment),
            loads=str(args.loads),
            seed_base=seed,
            mother_id=mother_id,
            feasible_graph_id=fg_id,
            share_mode=str(args.share_mode),
            share_ratio=float(args.share_ratio),
        )
        pilot_metrics = _collect_triplet(pilot_prefix, int(args.pilot_episodes_per_load))
        ok, reason = _guardrail_pass(
            pilot_metrics,
            min_feasible_pair_ratio=float(args.min_feasible_pair_ratio),
            max_feasible_ratio_spread=float(args.max_feasible_ratio_spread),
            max_no_feasible_admit_ratio=float(args.max_no_feasible_admit_ratio),
            min_overlay_feasible_pairs=float(args.min_overlay_feasible_pairs),
            min_puncture_feasible_pairs=float(args.min_puncture_feasible_pairs),
            max_intercell_interference_rejection_ratio=float(args.max_intercell_interference_rejection_ratio),
            require_admitted_monotonic=bool(args.require_admitted_monotonic_pilot),
            require_embb_monotonic=bool(args.require_embb_monotonic_pilot),
        )
        print(f"[attempt {attempt}] guardrail_v2={ok} reason={reason}")
        if ok:
            selected_record = {
                "attempt": int(attempt),
                "seed_base": int(seed),
                "mother_id": str(mother_id),
                "feasible_graph_id": str(fg_id),
                "pilot_metrics": pilot_metrics,
            }
            selected_all.append(selected_record)
            if selected is None:
                selected = (seed, mother_id, fg_id, attempt)
            if not bool(args.continue_after_selected):
                break

    selected_json_path = (
        Path(str(args.selected_output_json)).expanduser()
        if str(args.selected_output_json).strip()
        else (RESULTS_DIR / f"{str(args.out_prefix)}_selected.json")
    )

    if selected is None:
        print("\n[FAIL] No scenario met guardrail v2 constraints within max attempts.")
        _write_selected_json(
            selected_json_path,
            {
                "selected_count": 0,
                "selected_scenarios": [],
                "scan": {
                    "max_attempts": int(args.max_attempts),
                    "loads": str(args.loads),
                    "pilot_episodes_per_load": int(args.pilot_episodes_per_load),
                },
            },
        )
        print(f"[SCAN] saved scan metadata: {selected_json_path}")
        return 2

    if bool(args.continue_after_selected):
        _write_selected_json(
            selected_json_path,
            {
                "selected_count": int(len(selected_all)),
                "selected_scenarios": selected_all,
                "scan": {
                    "max_attempts": int(args.max_attempts),
                    "loads": str(args.loads),
                    "pilot_episodes_per_load": int(args.pilot_episodes_per_load),
                    "episodes_per_load": int(args.episodes_per_load),
                },
                "guardrail_thresholds": {
                    "min_feasible_pair_ratio": float(args.min_feasible_pair_ratio),
                    "max_feasible_ratio_spread": float(args.max_feasible_ratio_spread),
                    "max_no_feasible_admit_ratio": float(args.max_no_feasible_admit_ratio),
                    "min_overlay_feasible_pairs": float(args.min_overlay_feasible_pairs),
                    "min_puncture_feasible_pairs": float(args.min_puncture_feasible_pairs),
                    "max_intercell_interference_rejection_ratio": float(args.max_intercell_interference_rejection_ratio),
                },
            },
        )
        print(f"\n[PASS] continue-after-selected mode: found {len(selected_all)} passing scenarios.")
        print(f"[SELECTED] saved scenario metadata: {selected_json_path}")
        print("[INFO] Skipping full run in continue-after-selected mode.")
        return 0

    seed, mother_id, fg_id, attempt = selected
    print(
        f"\n[SELECTED] attempt={attempt} seed={seed} mother_id={mother_id} feasible_graph_id={fg_id}\n"
        f"Running full episodes_per_load={int(args.episodes_per_load)}..."
    )
    _run_clean_mix_once(
        out_prefix=str(args.out_prefix),
        episodes_per_load=int(args.episodes_per_load),
        experiment=str(args.experiment),
        loads=str(args.loads),
        seed_base=int(seed),
        mother_id=str(mother_id),
        feasible_graph_id=str(fg_id),
        share_mode=str(args.share_mode),
        share_ratio=float(args.share_ratio),
    )
    full_metrics = _collect_triplet(str(args.out_prefix), int(args.episodes_per_load))
    ok, reason = _guardrail_pass(
        full_metrics,
        min_feasible_pair_ratio=float(args.min_feasible_pair_ratio),
        max_feasible_ratio_spread=float(args.max_feasible_ratio_spread),
        max_no_feasible_admit_ratio=float(args.max_no_feasible_admit_ratio),
        min_overlay_feasible_pairs=float(args.min_overlay_feasible_pairs),
        min_puncture_feasible_pairs=float(args.min_puncture_feasible_pairs),
        max_intercell_interference_rejection_ratio=float(args.max_intercell_interference_rejection_ratio),
        require_admitted_monotonic=bool(args.require_admitted_monotonic_full),
        require_embb_monotonic=bool(args.require_embb_monotonic_full),
    )
    print(f"[FULL] guardrail_v2={ok} reason={reason}")
    print(
        "\n[FULL] summary (mix -> requested, realized_resource, feasible_pair_ratio, "
        "no_feasible_admit_ratio, admitted_packets_mean, embb_rate_mbps)"
    )
    for mix_name in ["3_7", "5_5", "7_3"]:
        x = full_metrics[mix_name]
        print(
            f"{mix_name.replace('_',':')} -> req={x['requested_mix_ratio']:.4f}, "
            f"real={x['realized_resource_ratio']:.4f}, feas={x['feasible_pair_ratio']:.4f}, "
            f"no_feas={x['no_feasible_admit_ratio']:.4f}, adm={x['admitted_packets_mean']:.2f}, "
            f"embb={x['embb_rate_mean'] / 1.0e6:.2f}"
        )

    selected_payload = {
        "selected": {
            "attempt": int(attempt),
            "seed_base": int(seed),
            "mother_id": str(mother_id),
            "feasible_graph_id": str(fg_id),
            "out_prefix": str(args.out_prefix),
            "loads": str(args.loads),
            "pilot_episodes_per_load": int(args.pilot_episodes_per_load),
            "episodes_per_load": int(args.episodes_per_load),
        },
        "guardrail_thresholds": {
            "min_feasible_pair_ratio": float(args.min_feasible_pair_ratio),
            "max_feasible_ratio_spread": float(args.max_feasible_ratio_spread),
            "max_no_feasible_admit_ratio": float(args.max_no_feasible_admit_ratio),
            "min_overlay_feasible_pairs": float(args.min_overlay_feasible_pairs),
            "min_puncture_feasible_pairs": float(args.min_puncture_feasible_pairs),
            "max_intercell_interference_rejection_ratio": float(args.max_intercell_interference_rejection_ratio),
        },
        "requirements": {
            "require_admitted_monotonic_pilot": bool(args.require_admitted_monotonic_pilot),
            "require_admitted_monotonic_full": bool(args.require_admitted_monotonic_full),
            "require_embb_monotonic_pilot": bool(args.require_embb_monotonic_pilot),
            "require_embb_monotonic_full": bool(args.require_embb_monotonic_full),
        },
        "full_result": {
            "guardrail_pass": bool(ok),
            "reason": str(reason),
            "metrics": full_metrics,
        },
    }
    if ok:
        _write_selected_json(selected_json_path, selected_payload)
        print(f"[SELECTED] saved scenario metadata: {selected_json_path}")
        return 0

    rejected_json_path = _default_rejected_json_path(selected_json_path)
    _write_selected_json(rejected_json_path, selected_payload)
    print(f"[REJECTED] saved scenario metadata: {rejected_json_path}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
