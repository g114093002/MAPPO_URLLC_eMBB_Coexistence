from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_greedy_mix_share_grid import (
    _plot_grid,
    _plot_per_mix_share_comparison,
    _plot_per_share_mix_comparison,
    _plot_share_budget_hit_cdf_for_mix,
)
import matplotlib.pyplot as plt

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


def _share_key_from_ratio(share_ratio: float) -> str:
    return f"share{int(round(float(share_ratio) * 100)):02d}"


def _node_to_compact_metrics(node: dict) -> dict:
    g = node.get("greedy", {})
    loads = [float(x) for x in node.get("loads", g.get("loads", []))]
    embb_rate = [float(x) for x in g.get("embb_rate", [])]
    embb_rate_pre_urllc_admission = [
        float(x)
        for x in g.get(
            "embb_rate_pre_urllc_admission",
            g.get("embb_rate", []),
        )
    ]
    urllc_admission = [float(x) for x in g.get("urllc_admission", [])]
    urllc_tp = [float(x) for x in g.get("urllc_throughput_mbps_slot_est", [])]
    admitted_packets = [float(x) for x in g.get("scheduled_packets", [])]
    arrived_packets = [float(x) for x in g.get("active_packets", [])]
    feasible_admit_packets = [float(x) for x in g.get("greedy_feasible_admit_count", [])]
    budget_used_ratio = [float(x) for x in g.get("greedy_urllc_budget_used_ratio", [])]
    avg_embb_loss = [float(x) for x in g.get("greedy_avg_embb_loss", [])]
    embb_service_ratio = [float(x) for x in g.get("embb_service_ratio", [])]
    embb_min_rate = [float(x) for x in g.get("embb_min_rate_satisfaction_ratio", [])]
    total_power_mw = [float(x) * 1e3 for x in g.get("total_power", [])]
    embb_user_count = [float(x) for x in g.get("embb_user_count", [])]
    urllc_user_count = [float(x) for x in g.get("urllc_user_count", [])]
    embb_loss_per_admit = []
    for loss, admit in zip(avg_embb_loss, admitted_packets):
        denom = admit if admit > 0 else 1.0
        embb_loss_per_admit.append(loss / denom)
    return {
        "loads": loads,
        "embb_rate_mbps": [x / 1e6 for x in embb_rate],
        "embb_rate_pre_urllc_admission_mbps": [x / 1e6 for x in embb_rate_pre_urllc_admission],
        "urllc_admission": urllc_admission,
        "urllc_tp_mbps": urllc_tp,
        "urllc_admitted_packets": admitted_packets,
        "urllc_arrived_packets": arrived_packets,
        "urllc_feasible_admit_packets": feasible_admit_packets,
        "budget_used_ratio": budget_used_ratio,
        "avg_embb_loss": avg_embb_loss,
        "embb_loss_per_admit": embb_loss_per_admit,
        "embb_service_ratio": embb_service_ratio,
        "embb_min_rate": embb_min_rate,
        "total_power_mw": total_power_mw,
        "embb_user_count": embb_user_count,
        "urllc_user_count": urllc_user_count,
        "greedy_episode_arrivals_samples": g.get("greedy_episode_arrivals_samples", []),
        "greedy_episode_admitted_samples": g.get("greedy_episode_admitted_samples", []),
        "greedy_episode_budget_used_ratio_samples": g.get("greedy_episode_budget_used_ratio_samples", []),
        "num_uavs": int(_infer_num_uavs(node)),
    }


def _plot_arrival_feasible_admit_per_mix(data: dict, mix_label: str, out_path: Path, share_keys: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    share_labels = []
    for share_key in share_keys:
        if not share_key.startswith("share"):
            continue
        if mix_label not in data.get(share_key, {}):
            continue
        try:
            share_pct = int(share_key.replace("share", ""))
            share_labels.append((share_pct, share_key))
        except ValueError:
            continue
    share_labels.sort(key=lambda x: x[0])

    for idx, (_, share_key) in enumerate(share_labels):
        d = data[share_key][mix_label]
        loads = _to_system_loads(d.get("loads", []), int(d.get("num_uavs", DEFAULT_NUM_UAVS)))
        c = color_cycle[idx % len(color_cycle)]
        axes[0].plot(loads, d.get("urllc_arrived_packets", []), marker="o", color=c, label=f"share {int(share_key.replace('share',''))}%")
        admitted = d.get("urllc_admitted_packets", [])
        arrived = d.get("urllc_arrived_packets", [])
        not_admitted = [max(float(a) - float(b), 0.0) for a, b in zip(arrived, admitted)]
        axes[1].plot(loads, admitted, marker="o", color=c, label=f"share {int(share_key.replace('share',''))}%")
        axes[2].plot(loads, not_admitted, marker="o", color=c, label=f"share {int(share_key.replace('share',''))}%")

    axes[0].set_title("URLLC arrived packets")
    axes[0].set_xlabel("Total system load")
    axes[0].set_ylabel("Packets")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("URLLC admitted packets")
    axes[1].set_xlabel("Total system load")
    axes[1].set_ylabel("Packets")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("URLLC not-admitted packets")
    axes[2].set_xlabel("Total system load")
    axes[2].set_ylabel("Packets")
    axes[2].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=9)
        axes[1].legend(fontsize=9)
        axes[2].legend(fontsize=9)
    fig.suptitle(f"Arrival/Admitted/Not-admitted under eMBB:URLLC={mix_label}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_share_delta_vs_baseline_per_mix(
    data: dict,
    mix_label: str,
    out_path: Path,
    share_keys: list[str],
    baseline_share_key: str = "share05",
) -> None:
    if baseline_share_key not in data or mix_label not in data.get(baseline_share_key, {}):
        return
    base = data[baseline_share_key][mix_label]
    loads = _to_system_loads(base.get("loads", []), int(base.get("num_uavs", DEFAULT_NUM_UAVS)))
    metrics = [
        ("embb_rate_mbps", "eMBB throughput", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "ratio"),
        ("urllc_tp_mbps", "URLLC throughput", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "packets"),
        ("budget_used_ratio", "Share-budget used ratio", "ratio"),
        ("embb_loss_per_admit", "eMBB loss per admitted packet", "bps/packet"),
    ]
    candidates = []
    for sk in share_keys:
        if sk == baseline_share_key:
            continue
        if mix_label in data.get(sk, {}):
            candidates.append(sk)
    if not candidates:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    axes = axes.ravel()
    color_cycle = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, (k, title, unit) in zip(axes, metrics):
        y_base = base.get(k, [])
        for idx, sk in enumerate(candidates):
            y = data[sk][mix_label].get(k, [])
            if not y_base or not y:
                continue
            delta_pct = []
            for a, b in zip(y, y_base):
                denom = b if abs(b) > 1e-12 else 1e-12
                delta_pct.append((a - b) / denom * 100.0)
            ax.plot(
                loads,
                delta_pct,
                marker="o",
                color=color_cycle[idx % len(color_cycle)],
                label=f"{sk.replace('share', 'share ')} vs {baseline_share_key.replace('share', 'share ')}",
            )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        ax.set_title(f"{title} (% delta)")
        ax.set_xlabel("Total system load")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Share deltas vs {baseline_share_key.replace('share', 'share ')} under eMBB:URLLC={mix_label}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple mix_share_grid_detailed_metrics.json files and regenerate combined plots."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input detailed metrics json files (from different share runs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for merged json and plots.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}
    merged_payload = {
        "meta": {
            "source_files": [str(Path(p).resolve()) for p in args.inputs],
            "mixes": [],
            "shares": [],
            "episodes_per_load": None,
            "base_experiment": None,
            "paired_seed_enabled": None,
            "paired_seed_base": None,
            "respect_mix_preset": None,
        },
        "grid": {},
    }

    for input_path in args.inputs:
        p = Path(input_path).resolve()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        grid = data.get("grid", {})
        meta = data.get("meta", {})
        if merged_payload["meta"]["episodes_per_load"] is None:
            merged_payload["meta"]["episodes_per_load"] = meta.get("episodes_per_load")
        if merged_payload["meta"]["base_experiment"] is None:
            merged_payload["meta"]["base_experiment"] = meta.get("base_experiment")
        if merged_payload["meta"]["paired_seed_enabled"] is None:
            merged_payload["meta"]["paired_seed_enabled"] = meta.get("paired_seed_enabled")
        if merged_payload["meta"]["paired_seed_base"] is None:
            merged_payload["meta"]["paired_seed_base"] = meta.get("paired_seed_base")
        if merged_payload["meta"]["respect_mix_preset"] is None:
            merged_payload["meta"]["respect_mix_preset"] = meta.get("respect_mix_preset")

        for share_label, per_mix in grid.items():
            if share_label not in all_data:
                all_data[share_label] = {}
            if share_label not in merged_payload["grid"]:
                merged_payload["grid"][share_label] = {}
            for mix_label, node in per_mix.items():
                all_data[share_label][mix_label] = _node_to_compact_metrics(node)
                merged_payload["grid"][share_label][mix_label] = node

    share_keys_ordered = sorted(all_data.keys())
    mixes_present = sorted({m for sk in share_keys_ordered for m in all_data.get(sk, {}).keys()})

    merged_payload["meta"]["mixes"] = mixes_present
    merged_payload["meta"]["shares"] = [int(sk.replace("share", "")) / 100.0 for sk in share_keys_ordered]

    # Main grid
    _plot_grid(all_data, out_dir / "mix_share_grid_comparison.png")

    # Per-mix: compare shares
    for mix_label in mixes_present:
        mix_key = mix_label.replace(":", "_")
        _plot_per_mix_share_comparison(
            all_data,
            mix_label,
            out_dir / f"mix_{mix_key}_share_comparison.png",
            share_keys_ordered,
        )
        _plot_arrival_feasible_admit_per_mix(
            all_data,
            mix_label,
            out_dir / f"mix_{mix_key}_arrival_feasible_admit_comparison.png",
            share_keys_ordered,
        )
        _plot_share_budget_hit_cdf_for_mix(
            all_data,
            mix_label,
            out_dir / f"mix_{mix_key}_share_cap_packet_cdf.png",
        )
        _plot_share_delta_vs_baseline_per_mix(
            all_data,
            mix_label,
            out_dir / f"mix_{mix_key}_share_delta_vs_share05.png",
            share_keys_ordered,
            baseline_share_key="share05",
        )

    # Per-share: compare mixes
    mix_label_suffix = "_".join([m.replace(":", "_") for m in mixes_present]) if mixes_present else "none"
    for share_key in share_keys_ordered:
        _plot_per_share_mix_comparison(
            all_data,
            share_key,
            out_dir / f"{share_key}_mix_{mix_label_suffix}_comparison.png",
        )

    merged_json = out_dir / "mix_share_grid_detailed_metrics_merged.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2)

    print(f"[MERGE] done. merged json: {merged_json}")
    print(f"[MERGE] done. main plot: {out_dir / 'mix_share_grid_comparison.png'}")


if __name__ == "__main__":
    main()
