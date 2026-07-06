from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_POWER_KEYS = [
    "power_penalty",
    "overlay_power_surcharge",
    "urllc_tx_power_penalty",
    "power_projection_penalty",
]

TERMINAL_POWER_KEYS = [
    "terminal_total_power_budget_penalty",
    "terminal_urllc_power_share_penalty",
    "terminal_total_power_over_greedy_penalty",
    "terminal_embb_power_over_greedy_penalty",
    "terminal_urllc_power_over_greedy_penalty",
    "terminal_intercell_power_penalty",
    "terminal_power_overuse_penalty",
    "terminal_phase_a_raw_saturation_penalty",
    "terminal_phase_a_cap_hit_penalty",
    "terminal_power_ratio_penalty",
]

SERIES_COLORS = {
    "v17zc": "#1f77b4",
    "v17zg": "#d62728",
}


def _load_metrics(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_runs(payload: Dict[str, object], *, mix: str, policy: str, lam: float) -> List[Dict[str, object]]:
    raw_runs = dict(payload.get("raw_runs", {}) or {})
    mix_runs = dict(raw_runs.get(mix, {}) or {})
    policy_runs = dict(mix_runs.get(policy, {}) or {})
    lam_key = f"lambda_{lam:g}"
    runs = list(policy_runs.get(lam_key, []) or [])
    if not runs:
        raise ValueError(f"No runs found for mix={mix}, policy={policy}, lambda={lam_key}")
    return [dict(run) for run in runs]


def _mean_component_map(runs: List[Dict[str, object]], stats_key: str) -> Dict[str, float]:
    keys = set()
    for run in runs:
        keys.update((run.get(stats_key, {}) or {}).keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals: List[float] = []
        for run in runs:
            payload = dict(dict(run.get(stats_key, {}) or {}).get(key, {}) or {})
            vals.append(float(payload.get("sum", 0.0) or 0.0))
        out[str(key)] = float(np.mean(np.asarray(vals, dtype=float))) if vals else 0.0
    return out


def _filtered_pairs(component_map: Dict[str, float], keys: List[str]) -> List[Tuple[str, float]]:
    pairs = [(key, float(component_map.get(key, 0.0) or 0.0)) for key in keys]
    return [pair for pair in pairs if abs(pair[1]) > 1.0e-12]


def _plot_group_compare(
    *,
    left_pairs: List[Tuple[str, float]],
    right_pairs: List[Tuple[str, float]],
    left_label: str,
    right_label: str,
    title: str,
    out_path: Path,
) -> List[Tuple[str, float, float]]:
    key_order: List[str] = []
    seen = set()
    for key, _ in left_pairs + right_pairs:
        if key not in seen:
            seen.add(key)
            key_order.append(key)

    rows: List[Tuple[str, float, float]] = []
    for key in key_order:
        left_val = next((value for name, value in left_pairs if name == key), 0.0)
        right_val = next((value for name, value in right_pairs if name == key), 0.0)
        rows.append((key, float(left_val), float(right_val)))

    labels = [row[0] for row in rows]
    y = np.arange(len(labels), dtype=float)
    width = 0.36

    fig_h = max(4.2, 0.52 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.barh(
        y - width / 2.0,
        [row[1] for row in rows],
        height=width,
        color=SERIES_COLORS.get(left_label, "#1f77b4"),
        alpha=0.9,
        label=left_label,
    )
    ax.barh(
        y + width / 2.0,
        [row[2] for row in rows],
        height=width,
        color=SERIES_COLORS.get(right_label, "#d62728"),
        alpha=0.9,
        label=right_label,
    )
    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean reward over 100 episodes")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v17zc and v17zg power-related reward penalties.")
    parser.add_argument("--zc-metrics", required=True)
    parser.add_argument("--zg-metrics", required=True)
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--zc-policy", default="mappo")
    parser.add_argument("--zg-policy", default="mappo")
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    zc_runs = _pick_runs(
        _load_metrics(Path(args.zc_metrics)),
        mix=str(args.mix),
        policy=str(args.zc_policy),
        lam=float(args.lambda_value),
    )
    zg_runs = _pick_runs(
        _load_metrics(Path(args.zg_metrics)),
        mix=str(args.mix),
        policy=str(args.zg_policy),
        lam=float(args.lambda_value),
    )

    zc_step = _filtered_pairs(_mean_component_map(zc_runs, "step_reward_term_stats"), STEP_POWER_KEYS)
    zg_step = _filtered_pairs(_mean_component_map(zg_runs, "step_reward_term_stats"), STEP_POWER_KEYS)
    zc_terminal = _filtered_pairs(_mean_component_map(zc_runs, "terminal_reward_term_stats"), TERMINAL_POWER_KEYS)
    zg_terminal = _filtered_pairs(_mean_component_map(zg_runs, "terminal_reward_term_stats"), TERMINAL_POWER_KEYS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_rows = _plot_group_compare(
        left_pairs=zc_step,
        right_pairs=zg_step,
        left_label="v17zc",
        right_label="v17zg",
        title="Step Power Penalties",
        out_path=out_dir / "step_power_penalty_compare.png",
    )
    terminal_rows = _plot_group_compare(
        left_pairs=zc_terminal,
        right_pairs=zg_terminal,
        left_label="v17zc",
        right_label="v17zg",
        title="Terminal Power Penalties",
        out_path=out_dir / "terminal_power_penalty_compare.png",
    )

    summary = {
        "mix": str(args.mix),
        "lambda_value": float(args.lambda_value),
        "zc_policy": str(args.zc_policy),
        "zg_policy": str(args.zg_policy),
        "step_rows": step_rows,
        "terminal_rows": terminal_rows,
    }
    (out_dir / "power_penalty_compare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
