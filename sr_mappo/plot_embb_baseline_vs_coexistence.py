from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot eMBB baseline-proxy vs coexistence throughput")
    p.add_argument("--input", required=True, help="Path to mix_share_grid_detailed_metrics_merged.json")
    p.add_argument("--mix", default="3:7", help="Mix label, e.g. 3:7")
    p.add_argument("--out", required=True, help="Output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    grid = data.get("grid", {})
    share_keys = sorted(grid.keys(), key=lambda s: int(s.replace("share", "")))

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = False
    for share_key in share_keys:
        mix_node = grid.get(share_key, {}).get(args.mix)
        if not mix_node:
            continue
        g = mix_node.get("greedy", {})
        loads = g.get("loads", [])
        embb = g.get("embb_rate", [])
        embb_wo_intf = g.get("embb_rate_without_intercell_est", [])
        if not loads or not embb or not embb_wo_intf:
            continue

        share_pct = int(share_key.replace("share", ""))
        embb_mbps = [v / 1e6 for v in embb]
        baseline_mbps = [v / 1e6 for v in embb_wo_intf]

        ax.plot(loads, embb_mbps, marker="o", label=f"share {share_pct}% coexistence")
        ax.plot(loads, baseline_mbps, marker="x", linestyle="--", label=f"share {share_pct}% baseline proxy (wo intercell)")
        plotted = True

    if not plotted:
        raise RuntimeError(f"No plottable data for mix {args.mix}")

    ax.set_title(f"eMBB Throughput: Baseline Proxy vs Coexistence (mix {args.mix})")
    ax.set_xlabel("Average UE load per UAV")
    ax.set_ylabel("Mbps")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
