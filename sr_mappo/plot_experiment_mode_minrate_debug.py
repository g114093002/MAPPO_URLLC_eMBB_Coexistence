from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def _series(bundle: Dict[str, Any], key: str) -> np.ndarray:
    values = bundle.get(key, [])
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)


def _series_like(bundle: Dict[str, Any], key: str, length: int) -> np.ndarray:
    values = _series(bundle, key)
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros(length, dtype=float)
    if values.size > length:
        return values[:length]
    out = np.zeros(length, dtype=float)
    out[:values.size] = values
    return out


def _loads_and_label(payload: Dict[str, Any]) -> tuple[np.ndarray, str]:
    loads = np.asarray(payload.get("loads", []), dtype=float)
    num_uavs = int(payload.get("num_uavs", 1) or 1)
    if num_uavs > 1:
        return loads * float(num_uavs), "Total system load (all UAVs)"
    return loads, "Load"


def _embb_user_count(bundle: Dict[str, Any]) -> np.ndarray:
    direct = _series(bundle, "embb_user_count")
    if direct.size > 0:
        return direct
    served = _series(bundle, "embb_served_user_count")
    service_ratio = _series(bundle, "embb_service_ratio")
    if served.size == 0 or service_ratio.size == 0:
        return np.zeros(0, dtype=float)
    return np.divide(
        served,
        np.maximum(service_ratio, 1.0e-12),
        out=np.zeros_like(served, dtype=float),
        where=np.maximum(service_ratio, 1.0e-12) > 0.0,
    )


def _embb_user_count_like(bundle: Dict[str, Any], length: int) -> np.ndarray:
    values = _embb_user_count(bundle)
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros(length, dtype=float)
    if values.size > length:
        return values[:length]
    out = np.zeros(length, dtype=float)
    out[:values.size] = values
    return out


def _active_packets_like(bundle: Dict[str, Any], length: int) -> np.ndarray:
    direct = _series_like(bundle, "active_packets", length)
    if np.any(direct > 0.0):
        return direct
    admitted = _series_like(bundle, "scheduled_packets", length)
    admission_ratio = _series_like(bundle, "urllc_admission", length)
    return np.divide(
        admitted,
        np.maximum(admission_ratio, 1.0e-12),
        out=np.zeros_like(admitted, dtype=float),
        where=np.maximum(admission_ratio, 1.0e-12) > 0.0,
    )


def _plot(payload: Dict[str, Any], out_path: Path) -> Path:
    sr = payload.get("sr_mappo", {}) or {}
    greedy = payload.get("greedy", {}) or {}
    x, xlabel = _loads_and_label(payload)
    n = int(x.size)

    sr_embb_users = _embb_user_count_like(sr, n)
    gr_embb_users = _embb_user_count_like(greedy, n)
    sr_minrate_ratio = _series_like(sr, "embb_min_rate_satisfaction_ratio", n)
    gr_minrate_ratio = _series_like(greedy, "embb_min_rate_satisfaction_ratio", n)
    sr_minrate_users = sr_embb_users * sr_minrate_ratio
    gr_minrate_users = gr_embb_users * gr_minrate_ratio

    sr_admitted = _series_like(sr, "scheduled_packets", n)
    gr_admitted = _series_like(greedy, "scheduled_packets", n)
    sr_arrivals = _active_packets_like(sr, n)
    gr_arrivals = _active_packets_like(greedy, n)

    sr_overlay_ratio = _series_like(sr, "overlay_ratio", n)
    gr_overlay_ratio = _series_like(greedy, "overlay_ratio", n)
    sr_puncture_ratio = _series_like(sr, "puncture_ratio", n)
    gr_puncture_ratio = _series_like(greedy, "puncture_ratio", n)

    sr_admit_overlay_ratio = _series_like(sr, "admission_via_overlay_ratio", n)
    gr_admit_overlay_ratio = _series_like(greedy, "admission_via_overlay_ratio", n)
    sr_admit_puncture_ratio = _series_like(sr, "admission_via_puncture_ratio", n)
    gr_admit_puncture_ratio = _series_like(greedy, "admission_via_puncture_ratio", n)
    sr_overlay_admits = sr_admitted * sr_admit_overlay_ratio
    gr_overlay_admits = gr_admitted * gr_admit_overlay_ratio
    sr_puncture_admits = sr_admitted * sr_admit_puncture_ratio
    gr_puncture_admits = gr_admitted * gr_admit_puncture_ratio

    sr_overlay_pairs = _series_like(sr, "overlay_selected_pairs", n)
    gr_overlay_pairs = _series_like(greedy, "overlay_selected_pairs", n)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, sr_minrate_users, marker="s", linewidth=2.0, color="tab:orange", label="MAPPO min-rate-satisfied users")
    ax.plot(x, gr_minrate_users, marker="o", linewidth=2.0, linestyle="--", color="tab:brown", label="Greedy min-rate-satisfied users")
    ax.set_title("eMBB min-rate-satisfied user count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(x, sr_admitted, marker="s", linewidth=2.0, color="tab:orange", label="MAPPO admitted packets")
    ax.plot(x, gr_admitted, marker="o", linewidth=2.0, linestyle="--", color="tab:brown", label="Greedy admitted packets")
    ax.plot(x, sr_arrivals, marker="^", linewidth=1.5, color="tab:blue", alpha=0.5, label="MAPPO arrivals")
    ax.plot(x, gr_arrivals, marker="d", linewidth=1.5, linestyle="--", color="tab:gray", alpha=0.5, label="Greedy arrivals")
    ax.set_title("URLLC admitted packet count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Packets")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(x, sr_overlay_ratio, marker="s", linewidth=2.0, color="tab:orange", label="MAPPO overlay ratio")
    ax.plot(x, sr_puncture_ratio, marker="x", linewidth=2.0, color="tab:red", label="MAPPO puncture ratio")
    ax.plot(x, gr_overlay_ratio, marker="o", linewidth=2.0, linestyle="--", color="tab:brown", label="Greedy overlay ratio")
    ax.plot(x, gr_puncture_ratio, marker="d", linewidth=2.0, linestyle="--", color="tab:gray", label="Greedy puncture ratio")
    ax.set_title("Mode selection ratio")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(x, sr_overlay_admits, marker="s", linewidth=2.0, color="tab:orange", label="MAPPO overlay-admitted packets")
    ax.plot(x, sr_puncture_admits, marker="x", linewidth=2.0, color="tab:red", label="MAPPO puncture-admitted packets")
    ax.plot(x, gr_overlay_admits, marker="o", linewidth=2.0, linestyle="--", color="tab:brown", label="Greedy overlay-admitted packets")
    ax.plot(x, gr_puncture_admits, marker="d", linewidth=2.0, linestyle="--", color="tab:gray", label="Greedy puncture-admitted packets")
    ax.set_title("Admitted packets by mode")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Packets")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2, 0]
    ax.plot(x, sr_overlay_pairs, marker="s", linewidth=2.0, color="tab:orange", label="MAPPO overlay selected pairs")
    ax.plot(x, gr_overlay_pairs, marker="o", linewidth=2.0, linestyle="--", color="tab:brown", label="Greedy overlay selected pairs")
    ax.set_title("Overlay selection count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2, 1]
    width = 0.18
    x_idx = np.arange(len(x), dtype=float)
    ax.bar(x_idx - 1.5 * width, sr_minrate_ratio, width=width, color="tab:orange", alpha=0.8, label="MAPPO min-rate ratio")
    ax.bar(x_idx - 0.5 * width, gr_minrate_ratio, width=width, color="tab:brown", alpha=0.8, label="Greedy min-rate ratio")
    ax.bar(x_idx + 0.5 * width, sr_admit_overlay_ratio, width=width, color="tab:blue", alpha=0.8, label="MAPPO overlay share in admits")
    ax.bar(x_idx + 1.5 * width, gr_admit_overlay_ratio, width=width, color="tab:gray", alpha=0.8, label="Greedy overlay share in admits")
    ax.set_title("Min-rate ratio and overlay admit share")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ratio")
    ax.set_xticks(x_idx, [f"{int(v):d}" if float(v).is_integer() else f"{v:g}" for v in x])
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=False)

    fig.suptitle(
        f"{Path(str(payload.get('checkpoint', ''))).stem} | baseline={payload.get('selected_baseline_key', 'greedy')}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot min-rate/admit/mode diagnostics from a report metrics JSON.")
    parser.add_argument("metrics_json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json).expanduser().resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_path = Path(args.out).expanduser().resolve() if args.out else metrics_path.with_name("minrate_admit_mode_debug.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = _plot(payload, out_path)
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
