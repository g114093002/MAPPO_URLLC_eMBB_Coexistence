from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _user_count_by_load(payload: dict, user_kind: str) -> list[int]:
    counts: list[int] = []
    dist = payload.get("uav_ue_distribution", {})
    for load in payload["sr_mappo"]["loads"]:
        entry = dist[str(float(load))]
        counts.append(int(entry[f"{user_kind}_user_count"]))
    return counts


def _plot_min_rate_satisfied_count(payload: dict, out_path: Path) -> None:
    loads = np.asarray(payload["sr_mappo"]["loads"], dtype=float)
    embb_counts = np.asarray(_user_count_by_load(payload, "embb"), dtype=float)
    mappo_ratio = np.asarray(payload["sr_mappo"]["embb_min_rate_satisfaction_ratio"], dtype=float)
    greedy_ratio = np.asarray(payload["greedy"]["embb_min_rate_satisfaction_ratio"], dtype=float)

    mappo_count = mappo_ratio * embb_counts
    greedy_count = greedy_ratio * embb_counts

    plt.figure(figsize=(9, 5.5))
    plt.plot(loads, mappo_count, marker="o", linewidth=2.2, label="SR-MAPPO")
    plt.plot(loads, greedy_count, marker="s", linewidth=2.2, label="Global Frontier Greedy")
    plt.xticks(loads, [str(int(load * 3)) for load in loads])
    plt.xlabel("Total users in system")
    plt.ylabel("eMBB users meeting min-rate")
    plt.title("Min-rate satisfied eMBB user count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_admitted_urllc_packets(payload: dict, out_path: Path) -> None:
    loads = np.asarray(payload["sr_mappo"]["loads"], dtype=float)
    mappo_sched = np.asarray(payload["sr_mappo"]["scheduled_packets"], dtype=float)
    greedy_sched = np.asarray(payload["greedy"]["scheduled_packets"], dtype=float)

    plt.figure(figsize=(9, 5.5))
    plt.plot(loads, mappo_sched, marker="o", linewidth=2.2, label="SR-MAPPO")
    plt.plot(loads, greedy_sched, marker="s", linewidth=2.2, label="Global Frontier Greedy")
    plt.xticks(loads, [str(int(load * 3)) for load in loads])
    plt.xlabel("Total users in system")
    plt.ylabel("Admitted URLLC packets")
    plt.title("Admitted URLLC packet count")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot custom comparisons from sr_mappo_report_metrics.json.")
    parser.add_argument("metrics_json", type=str)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json).resolve()
    payload = _load_json(metrics_path)
    out_dir = metrics_path.parent

    _plot_min_rate_satisfied_count(payload, out_dir / "custom_min_rate_satisfied_count_compare.png")
    _plot_admitted_urllc_packets(payload, out_dir / "custom_admitted_urllc_packets_compare.png")
    print(str(out_dir / "custom_min_rate_satisfied_count_compare.png"))
    print(str(out_dir / "custom_admitted_urllc_packets_compare.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
