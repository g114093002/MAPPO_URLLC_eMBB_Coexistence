from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _share_sort_key(share_key: str) -> int:
    try:
        return int(share_key.replace("share", ""))
    except Exception:
        return 10**9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-share rejection status from merged mix/share detailed metrics."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to mix_share_grid_detailed_metrics_merged.json")
    parser.add_argument("--mix", type=str, default="3:7", help="Mix label, e.g. 3:7 / 7:3 / 5:5 / 10:0")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for csv/png")
    args = parser.parse_args()

    in_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    grid = payload.get("grid", {})
    mix_label = args.mix.strip()
    share_keys = sorted([k for k in grid.keys() if mix_label in grid.get(k, {})], key=_share_sort_key)
    if not share_keys:
        raise RuntimeError(f"No share data found for mix={mix_label} in {in_path}")

    # Use first share loads as canonical x-axis.
    first_node = grid[share_keys[0]][mix_label]["greedy"]
    loads = [float(x) for x in first_node.get("loads", [])]

    counts_csv = out_dir / f"mix_{mix_label.replace(':', '_')}_share_rejection_counts.csv"
    reasons_csv = out_dir / f"mix_{mix_label.replace(':', '_')}_share_rejection_reasons.csv"
    decomp_csv = out_dir / f"mix_{mix_label.replace(':', '_')}_share_no_feasible_decomposition.csv"
    plot_path = out_dir / f"mix_{mix_label.replace(':', '_')}_share_rejection_overview.png"
    prefilter_plot_path = out_dir / f"mix_{mix_label.replace(':', '_')}_share_prefilter_blocks.png"

    # Write counts CSV (packet-level status).
    with counts_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "share",
                "load",
                "arrived_packets",
                "admitted_packets",
                "rejected_packets",
                "admission_ratio",
            ]
        )
        for sk in share_keys:
            g = grid[sk][mix_label]["greedy"]
            arrived = [float(x) for x in g.get("active_packets", [])]
            admitted = [float(x) for x in g.get("scheduled_packets", [])]
            ratio = [float(x) for x in g.get("urllc_admission", [])]
            for ld, a, b, r in zip(loads, arrived, admitted, ratio):
                rej = max(a - b, 0.0)
                w.writerow([sk, ld, a, b, rej, r])

    # Write reason CSV (decision-level reasons; useful but not packet-count reasons).
    with reasons_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "share",
                "load",
                "no_feasible_admit_ratio",
                "reject_share_cap_ratio",
                "reject_reliability_ratio",
                "reject_power_ratio",
                "reject_min_rate_ratio",
                "feasible_ratio",
                "prefilter_block_mode_mask_per_decision",
                "prefilter_block_packet_mask_per_decision",
                "prefilter_block_mode_infeasible_per_decision",
                "candidate_evaluated_per_decision",
                "candidate_feasible_per_decision",
            ]
        )
        for sk in share_keys:
            g = grid[sk][mix_label]["greedy"]
            nof = [float(x) for x in g.get("greedy_no_feasible_admit_ratio", [])]
            rsc = [float(x) for x in g.get("greedy_hf_reject_share_cap_ratio", [])]
            rrel = [float(x) for x in g.get("greedy_hf_reject_reliability_ratio", [])]
            rpow = [float(x) for x in g.get("greedy_hf_reject_power_ratio", [])]
            rmin = [float(x) for x in g.get("greedy_hf_reject_min_rate_ratio", [])]
            feas = [float(x) for x in g.get("greedy_hf_feasible_ratio", [])]
            b_mode = [float(x) for x in g.get("greedy_hf_prefilter_block_mode_mask_per_decision", [0.0] * len(loads))]
            b_pkt = [float(x) for x in g.get("greedy_hf_prefilter_block_packet_mask_per_decision", [0.0] * len(loads))]
            b_inf = [float(x) for x in g.get("greedy_hf_prefilter_block_mode_infeasible_per_decision", [0.0] * len(loads))]
            c_eval = [float(x) for x in g.get("greedy_hf_candidate_evaluated_per_decision", [0.0] * len(loads))]
            c_feas = [float(x) for x in g.get("greedy_hf_candidate_feasible_per_decision", [0.0] * len(loads))]
            for row in zip(loads, nof, rsc, rrel, rpow, rmin, feas, b_mode, b_pkt, b_inf, c_eval, c_feas):
                w.writerow([sk, *row])

    # Write no-feasible decomposition CSV (same top-level denominator as no_feasible_admit_ratio).
    # This is the closest additive split for "why not admitted" under current telemetry.
    with decomp_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "share",
                "load",
                "no_feasible_admit_ratio",
                "no_feasible_due_to_no_candidate_ratio",
                "no_feasible_due_to_budget_exhausted_ratio",
                "no_feasible_due_to_all_rejected_ratio",
                "no_candidate_due_to_mask_block_ratio",
                "no_candidate_due_to_empty_observation_ratio",
                "no_candidate_due_to_mode_mask_ratio",
                "no_candidate_due_to_packet_mask_ratio",
                "no_candidate_due_to_mode_infeasible_ratio",
            ]
        )
        for sk in share_keys:
            g = grid[sk][mix_label]["greedy"]
            nof = [float(x) for x in g.get("greedy_no_feasible_admit_ratio", [0.0] * len(loads))]
            p_nc_given_nof = [float(x) for x in g.get("greedy_hf_no_candidate_given_no_feasible_ratio", [0.0] * len(loads))]
            p_bud_given_nof = [float(x) for x in g.get("greedy_hf_budget_exhausted_given_no_feasible_ratio", [0.0] * len(loads))]
            p_ar_given_nof = [float(x) for x in g.get("greedy_hf_all_rejected_given_no_feasible_ratio", [0.0] * len(loads))]

            p_mask_given_nc = [float(x) for x in g.get("greedy_hf_no_candidate_mask_block_given_no_candidate_ratio", [0.0] * len(loads))]
            p_empty_given_nc = [float(x) for x in g.get("greedy_hf_no_candidate_empty_observation_given_no_candidate_ratio", [0.0] * len(loads))]
            p_mode_mask_given_nc = [float(x) for x in g.get("greedy_hf_no_candidate_block_mode_mask_per_no_candidate", [0.0] * len(loads))]
            p_pkt_mask_given_nc = [float(x) for x in g.get("greedy_hf_no_candidate_block_packet_mask_per_no_candidate", [0.0] * len(loads))]
            p_mode_inf_given_nc = [float(x) for x in g.get("greedy_hf_no_candidate_block_mode_infeasible_per_no_candidate", [0.0] * len(loads))]

            for ld, a, b, c, d, e, m, o, pm, mi in zip(
                loads,
                nof,
                p_nc_given_nof,
                p_bud_given_nof,
                p_ar_given_nof,
                p_mask_given_nc,
                p_empty_given_nc,
                p_mode_mask_given_nc,
                p_pkt_mask_given_nc,
                p_mode_inf_given_nc,
            ):
                nof_no_candidate = a * b
                nof_budget = a * c
                nof_all_rejected = a * d
                nc_mask = nof_no_candidate * e
                nc_empty = nof_no_candidate * m
                nc_mode_mask = nof_no_candidate * o
                nc_pkt_mask = nof_no_candidate * pm
                nc_mode_inf = nof_no_candidate * mi
                w.writerow(
                    [
                        sk,
                        ld,
                        a,
                        nof_no_candidate,
                        nof_budget,
                        nof_all_rejected,
                        nc_mask,
                        nc_empty,
                        nc_mode_mask,
                        nc_pkt_mask,
                        nc_mode_inf,
                    ]
                )

    # Plot overview: packet-level + reason ratios.
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    c = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, sk in enumerate(share_keys):
        g = grid[sk][mix_label]["greedy"]
        arrived = [float(x) for x in g.get("active_packets", [])]
        admitted = [float(x) for x in g.get("scheduled_packets", [])]
        rejected = [max(a - b, 0.0) for a, b in zip(arrived, admitted)]
        ratio = [float(x) for x in g.get("urllc_admission", [])]
        nof = [float(x) for x in g.get("greedy_no_feasible_admit_ratio", [])]
        rsc = [float(x) for x in g.get("greedy_hf_reject_share_cap_ratio", [])]
        label = sk.replace("share", "share ")
        color = c[idx % len(c)]

        axes[0, 0].plot(loads, arrived, marker="o", color=color, label=label)
        axes[0, 1].plot(loads, admitted, marker="o", color=color, label=label)
        axes[1, 0].plot(loads, rejected, marker="o", color=color, label=label)
        axes[1, 1].plot(loads, [100.0 * x for x in ratio], marker="o", color=color, linestyle="-", label=f"{label} admission%")
        axes[1, 1].plot(loads, [100.0 * x for x in nof], marker="x", color=color, linestyle="--", label=f"{label} no_feasible%")
        axes[1, 1].plot(loads, [100.0 * x for x in rsc], marker="^", color=color, linestyle=":", label=f"{label} share_cap_reject%")

    axes[0, 0].set_title("URLLC arrived packets")
    axes[0, 1].set_title("URLLC admitted packets")
    axes[1, 0].set_title("URLLC rejected packets")
    axes[1, 1].set_title("Admission and rejection-related ratios")

    for ax in axes.ravel():
        ax.set_xlabel("Average UE load per UAV")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Packets")
    axes[0, 1].set_ylabel("Packets")
    axes[1, 0].set_ylabel("Packets")
    axes[1, 1].set_ylabel("Percent (%)")

    fig.suptitle(f"Per-share rejection status under eMBB:URLLC={mix_label}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    # Prefilter block breakdown (decision-level); this pinpoints why candidate pool is tiny.
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for idx, sk in enumerate(share_keys):
        g = grid[sk][mix_label]["greedy"]
        label = sk.replace("share", "share ")
        color = c[idx % len(c)]
        b_mode = [float(x) for x in g.get("greedy_hf_prefilter_block_mode_mask_per_decision", [0.0] * len(loads))]
        b_pkt = [float(x) for x in g.get("greedy_hf_prefilter_block_packet_mask_per_decision", [0.0] * len(loads))]
        b_inf = [float(x) for x in g.get("greedy_hf_prefilter_block_mode_infeasible_per_decision", [0.0] * len(loads))]
        cand_eval = [float(x) for x in g.get("greedy_hf_candidate_evaluated_per_decision", [0.0] * len(loads))]
        no_feas = [float(x) for x in g.get("greedy_no_feasible_admit_ratio", [0.0] * len(loads))]

        axes2[0].plot(loads, b_mode, marker="o", color=color, label=label)
        axes2[1].plot(loads, b_pkt, marker="o", color=color, label=label)
        axes2[2].plot(loads, b_inf, marker="o", color=color, label=label)
        # Overlay key sanity signals
        axes2[0].plot(loads, cand_eval, marker="x", linestyle="--", color=color, alpha=0.6, label=f"{label} cand_eval")
        axes2[1].plot(loads, [100.0 * x for x in no_feas], marker="^", linestyle=":", color=color, alpha=0.6, label=f"{label} no_feas%")

    axes2[0].set_title("Prefilter block: mode mask (per decision)")
    axes2[1].set_title("Prefilter block: packet mask (per decision)")
    axes2[2].set_title("Prefilter block: mode infeasible (per decision)")
    for ax in axes2:
        ax.set_xlabel("Average UE load per UAV")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes2[0].set_ylabel("Count/decision")
    axes2[1].set_ylabel("Count/decision or %")
    axes2[2].set_ylabel("Count/decision")
    fig2.suptitle(f"Prefilter blocking breakdown under eMBB:URLLC={mix_label}")
    fig2.tight_layout()
    fig2.savefig(prefilter_plot_path, dpi=220)
    plt.close(fig2)

    print(f"[ANALYZE] counts csv: {counts_csv}")
    print(f"[ANALYZE] reasons csv: {reasons_csv}")
    print(f"[ANALYZE] no-feasible decomp csv: {decomp_csv}")
    print(f"[ANALYZE] plot: {plot_path}")
    print(f"[ANALYZE] prefilter plot: {prefilter_plot_path}")


if __name__ == "__main__":
    main()
