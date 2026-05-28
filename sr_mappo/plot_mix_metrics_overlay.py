from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_NUM_UAVS = 3


def _load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_num_uavs(payload: Dict) -> int:
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


def _to_system_loads(loads: List[float], num_uavs: int) -> List[float]:
    return [float(load) * max(int(num_uavs), 1) for load in loads]


def _format_load_label(load: float) -> str:
    return str(int(load)) if float(load).is_integer() else f"{float(load):g}"


def _format_mix_label(mix_label: str) -> str:
    compact = str(mix_label).split("(", 1)[0].strip()
    return f"eMBB:URLLC={compact}"


def _infer_mix_label(path: Path) -> str:
    name = str(path).lower()
    if "_3_7_" in name:
        return "3:7"
    if "_5_5_" in name:
        return "5:5"
    if "_7_3_" in name:
        return "7:3"
    return path.parent.name


def _mix_order_key(label: str) -> int:
    order = {"3:7": 0, "5:5": 1, "7:3": 2}
    return order.get(label.split(" ")[0], 99)


def _series(block: Dict, key: str, n: int) -> List[float]:
    vals = block.get(key, [])
    if not isinstance(vals, list):
        return [0.0] * n
    out = [float(v) for v in vals[:n]]
    if len(out) < n:
        out += [0.0] * (n - len(out))
    return out


def _metric_specs():
    return [
        ("realized_resource_ratio", "Realized Resource Ratio"),
        ("urllc_admission", "URLLC Admission Ratio"),
        ("admitted_urllc_reliability", "Admitted URLLC Reliability"),
        ("embb_rate", "eMBB Total Throughput (Mbps)"),
        ("embb_rate_pre_urllc_admission", "eMBB Pre-URLLC Throughput (Mbps)"),
        ("embb_min_rate_satisfaction_ratio", "eMBB Min-rate Satisfaction"),
        ("overlay_ratio", "Overlay Ratio"),
        ("overlay_feasible_pairs", "Overlay Feasible Pairs"),
        ("scheduled_packets", "Scheduled URLLC Packets"),
    ]


def _plot_standard_metric(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
    key: str,
    title: str,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for label, series_block in rows:
        y = _series(series_block, key, len(loads))
        if key in {"embb_rate", "embb_rate_pre_urllc_admission"}:
            y = [v / 1.0e6 for v in y]
        ax.plot(x, y, marker="o", linewidth=2.4, markersize=7, label=_format_mix_label(label))
    ax.set_title(title)
    ax.set_xlabel("Total system load")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_embb_retention(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for label, series_block in rows:
        post = np.asarray(_series(series_block, "embb_rate", len(loads)), dtype=float)
        pre = np.asarray(_series(series_block, "embb_rate_pre_urllc_admission", len(loads)), dtype=float)
        retention = np.divide(post, np.maximum(pre, 1.0e-9), out=np.zeros_like(post), where=pre > 1.0e-9)
        retention = np.clip(retention, 0.0, 1.5)
        ax.plot(x, retention, marker="o", linewidth=2.6, markersize=7, label=_format_mix_label(label))
    ax.set_title("eMBB Throughput Retention vs Load")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("post-URLLC / pre-URLLC")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_embb_loss(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for label, series_block in rows:
        post = np.asarray(_series(series_block, "embb_rate", len(loads)), dtype=float)
        pre = np.asarray(_series(series_block, "embb_rate_pre_urllc_admission", len(loads)), dtype=float)
        loss_mbps = np.maximum(pre - post, 0.0) / 1.0e6
        ax.plot(x, loss_mbps, marker="o", linewidth=2.6, markersize=7, label=_format_mix_label(label))
    ax.set_title("eMBB Throughput Loss vs Load")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("Throughput loss (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_embb_loss_per_packet(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for label, series_block in rows:
        post = np.asarray(_series(series_block, "embb_rate", len(loads)), dtype=float)
        pre = np.asarray(_series(series_block, "embb_rate_pre_urllc_admission", len(loads)), dtype=float)
        scheduled = np.asarray(_series(series_block, "scheduled_packets", len(loads)), dtype=float)
        loss_kbps = np.maximum(pre - post, 0.0) / 1.0e3
        loss_per_packet = np.divide(
            loss_kbps,
            np.maximum(scheduled, 1.0),
            out=np.zeros_like(loss_kbps),
            where=scheduled > 0.0,
        )
        ax.plot(x, loss_per_packet, marker="o", linewidth=2.6, markersize=7, label=_format_mix_label(label))
    ax.set_title("eMBB Loss per Scheduled URLLC Packet")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("Loss per packet (Kbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_urllc_throughput(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    for label, series_block in rows:
        y = _series(series_block, "urllc_throughput_mbps_slot_est", len(loads))
        ax.plot(x, y, marker="o", linewidth=2.6, markersize=7, label=_format_mix_label(label))
    ax.set_title("URLLC Throughput vs Load")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("URLLC throughput (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_power_split(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = np.asarray(_to_system_loads(loads, num_uavs), dtype=float)
    x_labels = [_format_load_label(l) for l in x]
    fig, axes = plt.subplots(1, len(rows), figsize=(16.5, 5.4), sharey=True, constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, series_block) in zip(axes, rows):
        embb = np.asarray(_series(series_block, "embb_power", len(loads)), dtype=float) * 1e3
        urllc = np.asarray(_series(series_block, "urllc_power", len(loads)), dtype=float) * 1e3
        ax.plot(x, embb, marker="o", linewidth=2.4, markersize=6, color="#4e79a7", label="eMBB power")
        ax.plot(x, urllc, marker="s", linewidth=2.4, markersize=6, color="#e15759", label="URLLC power")
        ax.set_title(_format_mix_label(label))
        ax.set_xlabel("Total system load")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(alpha=0.22, axis="y")
        ax.legend(loc="upper right", frameon=True, fontsize=9)
    axes[0].set_ylabel("Transmit power (mW)")
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_mode_selection(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = np.asarray(_to_system_loads(loads, num_uavs), dtype=float)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    colors = {"3:7": "#59a14f", "5:5": "#f28e2b", "7:3": "#4e79a7"}
    markers = {"3:7": "o", "5:5": "s", "7:3": "^"}
    for label, series_block in rows:
        puncture = np.asarray(_series(series_block, "puncture_selection_ratio", len(loads)), dtype=float)
        mix_key = label.split(" ")[0]
        ax.plot(
            x,
            puncture,
            marker=markers.get(mix_key, "o"),
            linewidth=2.4,
            markersize=6,
            color=colors.get(mix_key, "#e15759"),
            label=f"{_format_mix_label(label)} puncture",
        )
    ax.set_title("Puncture Selection Ratio vs Load")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("Puncture selection ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.0, 0.12)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _plot_min_rate_satisfied_users(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> Path:
    x = _to_system_loads(loads, num_uavs)
    x_labels = [_format_load_label(l) for l in x]
    fig, ax = plt.subplots(figsize=(12.8, 7.4), constrained_layout=True)
    colors = {"3:7": "#59a14f", "5:5": "#f28e2b", "7:3": "#4e79a7"}
    markers = {"3:7": "o", "5:5": "s", "7:3": "^"}
    for label, series_block in rows:
        served = np.asarray(_series(series_block, "embb_served_user_count", len(loads)), dtype=float)
        service_ratio = np.asarray(_series(series_block, "embb_service_ratio", len(loads)), dtype=float)
        sat_ratio = np.asarray(_series(series_block, "embb_min_rate_satisfaction_ratio", len(loads)), dtype=float)
        total_users = np.divide(served, service_ratio, out=np.zeros_like(served), where=service_ratio > 1e-9)
        satisfied = total_users * sat_ratio
        mix_key = label.split(" ")[0]
        ax.plot(
            x,
            satisfied,
            marker=markers.get(mix_key, "o"),
            linewidth=2.4,
            markersize=6,
            color=colors.get(mix_key, "#4e79a7"),
            label=_format_mix_label(label),
        )
    ax.set_title("Min-rate Satisfied eMBB Users vs Load")
    ax.set_xlabel("Total system load")
    ax.set_ylabel("Satisfied eMBB users (estimated count)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=10)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _build_plot(
    rows: List[Tuple[str, Dict]],
    loads: List[float],
    out_path: Path,
    num_uavs: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    for key, title in _metric_specs():
        _plot_standard_metric(rows, loads, out_path.with_name(f"{stem}_{key}.png"), num_uavs, key, title)
    _plot_embb_retention(rows, loads, out_path.with_name(f"{stem}_embb_retention.png"), num_uavs)
    _plot_embb_loss(rows, loads, out_path.with_name(f"{stem}_embb_loss.png"), num_uavs)
    _plot_embb_loss_per_packet(rows, loads, out_path.with_name(f"{stem}_embb_loss_per_packet.png"), num_uavs)
    _plot_urllc_throughput(rows, loads, out_path.with_name(f"{stem}_urllc_throughput_vs_load.png"), num_uavs)
    _plot_power_split(rows, loads, out_path.with_name(f"{stem}_power_split.png"), num_uavs)
    _plot_mode_selection(rows, loads, out_path.with_name(f"{stem}_mode_selection.png"), num_uavs)
    _plot_min_rate_satisfied_users(rows, loads, out_path.with_name(f"{stem}_min_rate_satisfied_users.png"), num_uavs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot 3 fair-mix metrics on one figure.")
    parser.add_argument("metrics", nargs=3, help="Three sr_mappo_report_metrics.json paths.")
    parser.add_argument(
        "--out",
        default="sr_mappo/results/mix_overlay_metrics.png",
        help="Output PNG base path. Script writes one PNG per metric.",
    )
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.metrics]
    payloads = [_load_metrics(p) for p in paths]
    loads = [float(x) for x in payloads[0].get("loads", [])]
    if not loads:
        raise SystemExit("No 'loads' found in metrics.")
    num_uavs = _infer_num_uavs(payloads[0])

    rows: List[Tuple[str, Dict]] = []
    for p, payload in zip(paths, payloads):
        label = _infer_mix_label(p)
        series_block = payload.get("sr_mappo") or payload.get("greedy", {})
        rows.append((label, series_block))

    rows.sort(key=lambda x: _mix_order_key(x[0]))

    out_path = Path(args.out).resolve()
    _build_plot(rows, loads, out_path, num_uavs)
    print(f"[OK] wrote figures under {out_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
