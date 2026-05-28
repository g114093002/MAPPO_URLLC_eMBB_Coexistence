from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


KEYS_EXACT_MATCH = [
    "feasible_graph_freeze_enabled",
    "same_feasible_graph_hash",
    "same_channel_hash",
    "same_assoc_hash",
    "same_user_pool_hash",
]

KEYS_GREEDY_FLOW = [
    "greedy_hf_relaxed_candidate_ratio",
    "greedy_hf_selected_relaxed_ratio",
    "greedy_hf_final_gate_reject_ratio",
    "greedy_hf_final_gate_keep_ratio",
]


def _load_payload(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(x) -> List:
    if isinstance(x, list):
        return x
    return []


def _summary_scalar(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / max(len(values), 1))


def _flatten_mean_2d(values) -> float:
    if not isinstance(values, list):
        return 0.0
    flat: List[float] = []
    for row in values:
        if isinstance(row, list):
            for x in row:
                try:
                    flat.append(float(x))
                except (TypeError, ValueError):
                    continue
    if not flat:
        return 0.0
    return float(sum(flat) / max(len(flat), 1))


def _collect_row(payload: Dict, source_name: str) -> Dict:
    row: Dict[str, object] = {"source": source_name}
    greedy = payload.get("greedy", {})
    for key in KEYS_EXACT_MATCH + KEYS_GREEDY_FLOW + [
        "requested_mix_ratio",
        "realized_resource_ratio",
        "realized_power_ratio",
        "realized_served_users_ratio",
        "overlay_graph_hash",
        "sic_order_hash",
        "repair_sequence_hash",
        "mix_user_subset_hash",
    ]:
        vals = _as_list(greedy.get(key, []))
        if vals and isinstance(vals[0], str):
            row[key] = str(vals[0])
        else:
            row[key] = _summary_scalar(vals)
    row["urllc_admitted_packets_mean"] = _flatten_mean_2d(greedy.get("greedy_episode_admitted_samples", []))
    row["urllc_arrivals_packets_mean"] = _flatten_mean_2d(greedy.get("greedy_episode_arrivals_samples", []))
    return row


def _all_equal(values: List[object]) -> bool:
    if not values:
        return True
    first = values[0]
    return all(v == first for v in values[1:])


def _print_table(rows: List[Dict]) -> None:
    headers = [
        "source",
        "freeze",
        "overlay_hash",
        "sic_hash",
        "repair_hash",
        "same_feasible_graph_hash",
        "mix_subset_hash",
        "relaxed_ratio",
        "gate_reject_ratio",
        "req_mix",
        "realized_resource",
        "adm_pkt_mean",
    ]
    print(" | ".join(headers))
    print("-" * 160)
    for r in rows:
        print(
            " | ".join(
                [
                    str(r["source"]),
                    f"{float(r['feasible_graph_freeze_enabled']):.3f}",
                    str(r["overlay_graph_hash"]),
                    str(r["sic_order_hash"]),
                    str(r["repair_sequence_hash"]),
                    str(r["same_feasible_graph_hash"]),
                    str(r["mix_user_subset_hash"]),
                    f"{float(r['greedy_hf_relaxed_candidate_ratio']):.4f}",
                    f"{float(r['greedy_hf_final_gate_reject_ratio']):.4f}",
                    f"{float(r['requested_mix_ratio']):.4f}",
                    f"{float(r['realized_resource_ratio']):.4f}",
                    f"{float(r['urllc_admitted_packets_mean']):.2f}",
                ]
            )
        )


def _check_requested_order_monotonic(rows: List[Dict], key: str) -> Tuple[bool, str]:
    by_req = sorted(rows, key=lambda x: float(x["requested_mix_ratio"]), reverse=True)
    vals = [float(r.get(key, 0.0)) for r in by_req]
    reqs = [float(r.get("requested_mix_ratio", 0.0)) for r in by_req]
    if len(vals) < 3:
        return True, f"skipped (need 3 mixes, got {len(vals)})"
    ok = vals[0] > vals[1] > vals[2]
    msg = (
        f"requested {reqs[0]:.1f}>{reqs[1]:.1f}>{reqs[2]:.1f} "
        f"=> {key} {vals[0]:.4f}>{vals[1]:.4f}>{vals[2]:.4f}"
    )
    return ok, msg


def audit(paths: List[Path], tol: float = 1e-9, require_admitted_monotonic: bool = False) -> Tuple[bool, List[str]]:
    rows: List[Dict] = []
    for p in paths:
        payload = _load_payload(p)
        rows.append(_collect_row(payload, str(p)))

    _print_table(rows)

    errors: List[str] = []

    for key in KEYS_EXACT_MATCH:
        vals = [r[key] for r in rows]
        if key == "feasible_graph_freeze_enabled":
            if not all(abs(float(v) - 1.0) <= tol for v in vals):
                errors.append("feasible_graph_freeze_enabled must be 1 for all runs.")
            continue
        if not _all_equal(vals):
            errors.append(f"{key} mismatch across runs.")

    # Informational only: these often differ across mix ratios because user subsets differ.
    # Do not fail on them.
    for key in ["overlay_graph_hash", "sic_order_hash", "repair_sequence_hash", "mix_user_subset_hash"]:
        vals = [r[key] for r in rows]
        if not _all_equal(vals):
            print(f"[INFO] {key} differs across mixes (expected when mix user subsets differ).")

    # Informational only: greedy flow ratios naturally differ across mixes because
    # candidate composition and gate dynamics differ with URLLC ratio.
    for key in KEYS_GREEDY_FLOW:
        vals = [float(r[key]) for r in rows]
        vmax = max(vals) if vals else 0.0
        vmin = min(vals) if vals else 0.0
        if abs(vmax - vmin) > tol:
            print(f"[INFO] {key} differs across mixes (min={vmin:.6f}, max={vmax:.6f}).")

    mono_ok, mono_msg = _check_requested_order_monotonic(rows, "urllc_admitted_packets_mean")
    print("\n[MONO] URLLC admitted packets monotonic check:")
    print(f"- {mono_msg}")
    print(f"- result: {mono_ok}")
    if require_admitted_monotonic and not mono_ok:
        errors.append("URLLC admitted packets are not monotonic vs requested mix ratio.")

    if errors:
        print("\n[FAIL] Fair-mix cleanliness audit failed:")
        for e in errors:
            print(f"- {e}")
        return False, errors

    print("\n[PASS] Fair-mix cleanliness audit passed.")
    return True, []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether mix-comparison runs are clean/fair before KPI interpretation."
    )
    parser.add_argument("metrics", nargs="+", help="Paths to sr_mappo_report_metrics.json files.")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerance for float-equality checks.")
    parser.add_argument(
        "--require-admitted-monotonic",
        action="store_true",
        help="Fail audit when mean admitted URLLC packets do not satisfy 0.7 > 0.5 > 0.3.",
    )
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.metrics]
    ok, _ = audit(
        paths,
        tol=float(args.tol),
        require_admitted_monotonic=bool(args.require_admitted_monotonic),
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
