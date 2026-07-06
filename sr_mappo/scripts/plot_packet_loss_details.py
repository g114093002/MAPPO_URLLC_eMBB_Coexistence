from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-packet eMBB loss details.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_files = {
        "Greedy": input_dir / "greedy_u30_k48_seed42_packet_loss.csv",
        "Greedy(pure puncturing)": input_dir / "pure_puncturing_u30_k48_seed42_packet_loss.csv",
        "Greedy(pure superposition)": input_dir / "pure_superposition_u30_k48_seed42_packet_loss.csv",
    }

    loss_series: Dict[str, np.ndarray] = {}
    for label, path in policy_files.items():
        rows = _read_rows(path)
        loss_series[label] = np.asarray(
            [_safe_float(row.get("embb_rate_loss_due_to_action_mbps", 0.0)) for row in rows],
            dtype=float,
        )

    plt.figure(figsize=(9.0, 5.2))
    for label, values in loss_series.items():
        if values.size == 0:
            continue
        x = np.arange(1, values.size + 1, dtype=int)
        plt.plot(x, values, linewidth=1.5, label=label)
    plt.xlabel("Admitted packet order")
    plt.ylabel("eMBB rate loss per admitted packet (Mbps)")
    plt.title("Per-packet eMBB Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_packet_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9.0, 5.2))
    for label, values in loss_series.items():
        if values.size == 0:
            continue
        x = np.arange(1, values.size + 1, dtype=int)
        plt.plot(x, np.cumsum(values), linewidth=2.0, label=label)
    plt.xlabel("Admitted packet order")
    plt.ylabel("Cumulative eMBB rate loss (Mbps)")
    plt.title("Cumulative eMBB Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_packet_loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    data = [loss_series[label] for label in policy_files if loss_series[label].size > 0]
    labels = [label for label in policy_files if loss_series[label].size > 0]
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("eMBB rate loss per admitted packet (Mbps)")
    plt.title("Per-packet eMBB Loss Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_packet_loss_boxplot.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
