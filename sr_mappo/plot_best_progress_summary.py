from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_mix_label(path: Path) -> str:
    name = str(path).lower()
    if "_3_7_" in name:
        return "3:7 (URLLC=0.7)"
    if "_5_5_" in name:
        return "5:5 (URLLC=0.5)"
    if "_7_3_" in name:
        return "7:3 (URLLC=0.3)"
    return path.parent.name


def _series(greedy: Dict, key: str, n: int) -> List[float]:
    vals = greedy.get(key, [])
    if not isinstance(vals, list):
        return [0.0] * n
    out = [float(v) for v in vals[:n]]
    if len(out) < n:
        out += [0.0] * (n - len(out))
    return out


def _plot_panel(ax, rows: List[Tuple[str, Dict]], loads: List[float], key: str, title: str, value_scale: float = 1.0) -> None:
    x = list(range(len(loads)))
    x_labels = [str(int(l)) if float(l).is_integer() else str(l) for l in loads]
    for label, greedy in rows:
        y = [v / value_scale for v in _series(greedy, key, len(loads))]
        ax.plot(x, y, marker="o", linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Load")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.28)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot best-so-far mix progress on one figure.")
    parser.add_argument("metrics", nargs=3, help="Three sr_mappo_report_metrics.json paths.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.metrics]
    payloads = [_load_metrics(p) for p in paths]
    loads = [float(x) for x in payloads[0].get("loads", [])]
    if not loads:
        raise SystemExit("No loads found in metrics.")

    rows: List[Tuple[str, Dict]] = []
    for p, payload in zip(paths, payloads):
        rows.append((_infer_mix_label(p), payload.get("greedy", {})))

    order = {"3:7": 0, "5:5": 1, "7:3": 2}
    rows.sort(key=lambda x: order.get(x[0].split(" ")[0], 99))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    _plot_panel(axes[0], rows, loads, "embb_rate", "eMBB Total Throughput (Mbps)", value_scale=1.0e6)
    _plot_panel(axes[1], rows, loads, "urllc_admission", "URLLC Admission Ratio")
    _plot_panel(axes[2], rows, loads, "scheduled_packets", "Scheduled URLLC Packets")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
