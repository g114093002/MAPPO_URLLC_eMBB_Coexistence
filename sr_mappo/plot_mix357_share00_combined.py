from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_NUM_UAVS = 3


def _infer_num_uavs(payload: dict) -> int:
    candidates = (
        payload.get("num_uavs"),
        payload.get("scenario", {}).get("num_uavs") if isinstance(payload.get("scenario"), dict) else None,
        payload.get("meta", {}).get("num_uavs") if isinstance(payload.get("meta"), dict) else None,
        payload.get("summary", {}).get("num_uavs") if isinstance(payload.get("summary"), dict) else None,
    )
    for value in candidates:
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except Exception:
            continue
    return DEFAULT_NUM_UAVS


def _to_system_loads(loads: list[float], num_uavs: int) -> list[float]:
    return [float(load) * max(int(num_uavs), 1) for load in loads]


def _format_mix_label(mix_label: str) -> str:
    return f"eMBB:URLLC={str(mix_label).split('(', 1)[0].strip()}"


def _node_to_compact_metrics(node: dict) -> dict:
    greedy = node.get("greedy", {})
    loads = [float(x) for x in node.get("loads", greedy.get("loads", []))]
    embb_rate = [float(x) / 1e6 for x in greedy.get("embb_rate", [])]
    urllc_admission = [float(x) for x in greedy.get("urllc_admission", [])]
    urllc_tp = [float(x) for x in greedy.get("urllc_throughput_mbps_slot_est", [])]
    admitted_packets = [float(x) for x in greedy.get("scheduled_packets", [])]
    budget_used_ratio = [float(x) for x in greedy.get("greedy_urllc_budget_used_ratio", [])]
    avg_embb_loss = [float(x) for x in greedy.get("greedy_avg_embb_loss", [])]
    embb_service_ratio = [float(x) for x in greedy.get("embb_service_ratio", [])]
    embb_min_rate = [float(x) for x in greedy.get("embb_min_rate_satisfaction_ratio", [])]
    total_power_mw = [float(x) * 1e3 for x in greedy.get("total_power", [])]
    embb_loss_per_admit = []
    for loss, admit in zip(avg_embb_loss, admitted_packets):
        embb_loss_per_admit.append(loss / max(admit, 1.0))
    return {
        "loads": loads,
        "embb_rate_mbps": embb_rate,
        "urllc_admission": urllc_admission,
        "urllc_tp_mbps": urllc_tp,
        "urllc_admitted_packets": admitted_packets,
        "budget_used_ratio": budget_used_ratio,
        "avg_embb_loss": avg_embb_loss,
        "embb_loss_per_admit": embb_loss_per_admit,
        "embb_service_ratio": embb_service_ratio,
        "embb_min_rate": embb_min_rate,
        "total_power_mw": total_power_mw,
        "num_uavs": int(_infer_num_uavs(node)),
    }


def _load_one(path: Path) -> tuple[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    # Backward-compatible path: per-mix grid payload produced by mix-share runs.
    grid = payload.get("grid", {})
    if "share00" in grid:
        share0 = grid["share00"]
        if len(share0) != 1:
            raise ValueError(f"{path} share00 should contain exactly one mix.")
        mix_label = next(iter(share0.keys()))
        node = share0[mix_label]
        return mix_label, _node_to_compact_metrics(node)

    # New path: direct report payload (e.g., multislot10 experiment folders).
    # We compare greedy metrics across mixes to match the original share00 intent.
    if "greedy" in payload and "loads" in payload:
        greedy = payload.get("greedy", {})
        # Infer mix label from experiment name first, then fallback to filename.
        exp_name = str(payload.get("experiment_line", "") or "")
        mix_label = "unknown"
        if "mix37" in exp_name:
            mix_label = "3:7"
        elif "mix55" in exp_name:
            mix_label = "5:5"
        elif "mix73" in exp_name:
            mix_label = "7:3"
        else:
            stem = path.stem.lower()
            if "mix37" in stem:
                mix_label = "3:7"
            elif "mix55" in stem:
                mix_label = "5:5"
            elif "mix73" in stem:
                mix_label = "7:3"

        loads = [float(x) for x in payload.get("loads", greedy.get("loads", []))]
        embb_rate = [float(x) / 1e6 for x in greedy.get("embb_rate", [])]
        urllc_admission = [float(x) for x in greedy.get("urllc_admission", [])]
        urllc_tp = [float(x) for x in greedy.get("urllc_throughput_mbps_slot_est", [])]
        admitted_packets = [float(x) for x in greedy.get("scheduled_packets", [])]
        embb_service_ratio = [float(x) for x in greedy.get("embb_service_ratio", [])]
        embb_min_rate = [float(x) for x in greedy.get("embb_min_rate_satisfaction_ratio", [])]
        total_power_mw = [float(x) * 1e3 for x in greedy.get("total_power", [])]
        avg_embb_loss = [float(x) for x in greedy.get("avg_puncture_embb_loss", [0.0] * len(loads))]
        embb_loss_per_admit = []
        for loss, admit in zip(avg_embb_loss, admitted_packets):
            embb_loss_per_admit.append(loss / max(admit, 1.0))

        return mix_label, {
            "loads": loads,
            "embb_rate_mbps": embb_rate,
            "urllc_admission": urllc_admission,
            "urllc_tp_mbps": urllc_tp,
            "urllc_admitted_packets": admitted_packets,
            "budget_used_ratio": [0.0 for _ in loads],
            "avg_embb_loss": avg_embb_loss,
            "embb_loss_per_admit": embb_loss_per_admit,
            "embb_service_ratio": embb_service_ratio,
            "embb_min_rate": embb_min_rate,
            "total_power_mw": total_power_mw,
            "num_uavs": int(_infer_num_uavs(payload)),
        }

    raise ValueError(f"{path} is neither share-grid payload nor sr_mappo report payload.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine mix 3:7 / 5:5 / 7:3 share00 detailed metrics into one comparison figure."
    )
    parser.add_argument(
        "--inputs",
        nargs=3,
        required=True,
        help="Three mix_share_grid_detailed_metrics.json files (mix37, mix55, mix73).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sr_mappo/results/mix357_share00_combined_comparison.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()

    data = {"share00": {}}
    for p in args.inputs:
        mix_label, metrics = _load_one(Path(p).resolve())
        data["share00"][mix_label] = metrics

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mix_order = [m for m in ["10:0", "7:3", "5:5", "3:7"] if m in data["share00"]]
    metrics = [
        ("embb_rate_mbps", "Aggregate eMBB throughput", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
        ("embb_service_ratio", "eMBB service ratio", "Ratio"),
        ("embb_min_rate", "eMBB min-rate satisfaction", "Ratio"),
        ("total_power_mw", "Total transmit power", "mW"),
    ]
    color_map = {"10:0": "#1f77b4", "7:3": "#ff7f0e", "5:5": "#2ca02c", "3:7": "#d62728"}
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21.0, 5.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    legend_handles = None
    legend_labels = None
    for ax, (k, title, ylab) in zip(axes, metrics):
        xticks = None
        for mix_label in mix_order:
            d = data["share00"][mix_label]
            if xticks is None:
                xticks = _to_system_loads(d["loads"], int(d.get("num_uavs", DEFAULT_NUM_UAVS)))
            ax.plot(
                _to_system_loads(d["loads"], int(d.get("num_uavs", DEFAULT_NUM_UAVS))),
                d[k],
                marker="o",
                linewidth=2,
                color=color_map[mix_label],
                label=_format_mix_label(mix_label),
            )
        ax.set_title(title)
        ax.set_xlabel("Total system load")
        ax.set_ylabel(ylab)
        if xticks:
            ax.set_xticks(xticks)
        ax.grid(True, alpha=0.35)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Greedy Mix Comparison under Share=0%", fontsize=15)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
