from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _as_float_list(x) -> List[float]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def _mix_label_from_path(p: Path) -> str:
    m = re.search(r"_(3_7|5_5|7_3)_e\d+", p.as_posix())
    return m.group(1).replace("_", ":") if m else p.parent.name


def _load_metrics(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    g = payload.get("greedy", {})
    loads = _as_float_list(g.get("load", []))
    if not loads:
        loads = _as_float_list(g.get("loads", []))
    if not loads:
        loads = _as_float_list(payload.get("loads", []))
    post = _as_float_list(g.get("embb_rate", []))
    pre = _as_float_list(g.get("embb_rate_pre_urllc_admission", []))
    feas_ratio = _as_float_list(g.get("feasible_pair_ratio", []))
    no_feas = _as_float_list(g.get("greedy_no_feasible_admit_ratio", []))
    cand = _as_float_list(g.get("candidate_pair_count", []))
    feas_cnt = _as_float_list(g.get("feasible_pair_count", []))
    ov_cand = _as_float_list(g.get("overlay_candidate_pairs", []))
    ov_feas = _as_float_list(g.get("overlay_feasible_pairs", []))
    per_uav_embb = g.get("per_uav_associated_embb", [])
    per_uav_urllc = g.get("per_uav_associated_urllc", [])

    n = min(len(loads), len(post), len(pre), len(feas_ratio), len(no_feas))
    rows = []
    for i in range(n):
        pre_i = pre[i]
        post_i = post[i]
        loss = pre_i - post_i
        loss_ratio = (loss / pre_i) if pre_i > 0 else 0.0
        cp = cand[i] if i < len(cand) else 0.0
        fp = feas_cnt[i] if i < len(feas_cnt) else 0.0
        ocp = ov_cand[i] if i < len(ov_cand) else 0.0
        ofp = ov_feas[i] if i < len(ov_feas) else 0.0
        p_feas_given_cand = (fp / cp) if cp > 0 else 0.0
        p_ov_feas_given_ov_cand = (ofp / ocp) if ocp > 0 else 0.0
        embb_uav = per_uav_embb[i] if i < len(per_uav_embb) and isinstance(per_uav_embb[i], list) else []
        urllc_uav = per_uav_urllc[i] if i < len(per_uav_urllc) and isinstance(per_uav_urllc[i], list) else []
        embb_uav = [float(x) for x in embb_uav]
        urllc_uav = [float(x) for x in urllc_uav]
        embb_max = max(embb_uav) if embb_uav else 0.0
        embb_sum = sum(embb_uav) if embb_uav else 0.0
        urllc_max = max(urllc_uav) if urllc_uav else 0.0
        urllc_sum = sum(urllc_uav) if urllc_uav else 0.0
        embb_uav_skew = (embb_max / embb_sum) if embb_sum > 0 else 0.0
        urllc_uav_skew = (urllc_max / urllc_sum) if urllc_sum > 0 else 0.0
        rows.append(
            {
                "load": loads[i],
                "embb_pre": pre_i,
                "embb_post": post_i,
                "embb_loss": loss,
                "embb_loss_ratio": loss_ratio,
                "feas_ratio": feas_ratio[i],
                "no_feas_ratio": no_feas[i],
                "cand": cp,
                "feas_cnt": fp,
                "p_feas_given_cand": p_feas_given_cand,
                "ov_cand": ocp,
                "ov_feas": ofp,
                "p_ov_feas_given_ov_cand": p_ov_feas_given_ov_cand,
                "per_uav_embb": embb_uav,
                "per_uav_urllc": urllc_uav,
                "embb_uav_skew": embb_uav_skew,
                "urllc_uav_skew": urllc_uav_skew,
            }
        )

    return {
        "path": str(path),
        "mix": _mix_label_from_path(path),
        "rows": rows,
        "mean": {
            "embb_pre": _mean([r["embb_pre"] for r in rows]),
            "embb_post": _mean([r["embb_post"] for r in rows]),
            "embb_loss": _mean([r["embb_loss"] for r in rows]),
            "embb_loss_ratio": _mean([r["embb_loss_ratio"] for r in rows]),
            "feas_ratio": _mean([r["feas_ratio"] for r in rows]),
            "no_feas_ratio": _mean([r["no_feas_ratio"] for r in rows]),
            "p_feas_given_cand": _mean([r["p_feas_given_cand"] for r in rows]),
            "p_ov_feas_given_ov_cand": _mean([r["p_ov_feas_given_ov_cand"] for r in rows]),
            "embb_uav_skew": _mean([r["embb_uav_skew"] for r in rows]),
            "urllc_uav_skew": _mean([r["urllc_uav_skew"] for r in rows]),
        },
    }


def _print_summary(items: List[Dict]) -> None:
    print("\n=== Mix Mean Summary ===")
    print("mix | embb_pre | embb_post | loss | loss_ratio | feas_ratio | no_feas | P(feas|cand) | P(ov_feas|ov_cand) | embb_uav_skew | urllc_uav_skew")
    print("-" * 120)
    for x in sorted(items, key=lambda t: t["mix"]):
        m = x["mean"]
        print(
            f"{x['mix']} | {m['embb_pre']/1e6:.2f} | {m['embb_post']/1e6:.2f} | {m['embb_loss']/1e6:.2f} | "
            f"{m['embb_loss_ratio']:.3f} | {m['feas_ratio']:.4f} | {m['no_feas_ratio']:.4f} | "
            f"{m['p_feas_given_cand']:.4f} | {m['p_ov_feas_given_ov_cand']:.4f} | "
            f"{m['embb_uav_skew']:.3f} | {m['urllc_uav_skew']:.3f}"
        )


def _print_load_order(items: List[Dict], key: str, title: str) -> None:
    loads = sorted({r["load"] for x in items for r in x["rows"]})
    print(f"\n=== Per-load Order: {title} ===")
    for ld in loads:
        vals: List[Tuple[str, float]] = []
        for x in items:
            row = next((r for r in x["rows"] if abs(r["load"] - ld) < 1e-9), None)
            if row is not None:
                vals.append((x["mix"], float(row[key])))
        vals.sort(key=lambda t: t[1], reverse=True)
        desc = " > ".join([f"{k}({v/1e6:.2f})" if 'embb' in key else f"{k}({v:.4f})" for k, v in vals])
        print(f"load={ld:.1f}: {desc}")


def _print_full_rows(items: List[Dict]) -> None:
    print("\n=== Detailed Rows ===")
    print(
        "mix | load | pre(Mbps) | post(Mbps) | loss(Mbps) | loss_ratio | feas_ratio | no_feas | "
        "P(feas|cand) | P(ov_feas|ov_cand) | embb_uav | urllc_uav | embb_uav_skew | urllc_uav_skew"
    )
    print("-" * 140)
    for x in sorted(items, key=lambda t: t["mix"]):
        for r in x["rows"]:
            print(
                f"{x['mix']} | {r['load']:.1f} | {r['embb_pre']/1e6:.2f} | {r['embb_post']/1e6:.2f} | "
                f"{r['embb_loss']/1e6:.2f} | {r['embb_loss_ratio']:.3f} | {r['feas_ratio']:.4f} | {r['no_feas_ratio']:.4f} | "
                f"{r['p_feas_given_cand']:.4f} | {r['p_ov_feas_given_ov_cand']:.4f} | "
                f"{r['per_uav_embb']} | {r['per_uav_urllc']} | {r['embb_uav_skew']:.3f} | {r['urllc_uav_skew']:.3f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose pre-URLLC eMBB baseline vs post-admission impacts from metrics JSON.")
    ap.add_argument("metrics", nargs="+", help="Paths to sr_mappo_report_metrics.json")
    args = ap.parse_args()

    items = [_load_metrics(Path(p).resolve()) for p in args.metrics]
    _print_summary(items)
    _print_load_order(items, "embb_pre", "eMBB Pre-Admission Throughput")
    _print_load_order(items, "embb_post", "eMBB Post-Admission Throughput")
    _print_load_order(items, "embb_loss", "Absolute eMBB Loss (pre-post)")
    _print_full_rows(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
