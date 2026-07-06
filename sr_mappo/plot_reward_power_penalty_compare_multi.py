from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    "v17zg": "#d62728",
    "v17zh": "#1f77b4",
    "v17zi": "#2ca02c",
    "v17zc": "#9467bd",
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


def _filtered_rows(series_maps: Dict[str, Dict[str, float]], keys: Sequence[str]) -> List[Tuple[str, Dict[str, float]]]:
    rows: List[Tuple[str, Dict[str, float]]] = []
    for key in keys:
        values = {label: float(component_map.get(key, 0.0) or 0.0) for label, component_map in series_maps.items()}
        if any(abs(value) > 1.0e-12 for value in values.values()):
            rows.append((str(key), values))
    return rows


def _plot_multi_compare(
    *,
    rows: List[Tuple[str, Dict[str, float]]],
    labels: Sequence[str],
    title: str,
    out_path: Path,
) -> List[Dict[str, object]]:
    y = np.arange(len(rows), dtype=float)
    n_series = max(1, len(labels))
    width = min(0.78 / n_series, 0.26)
    offsets = (np.arange(n_series, dtype=float) - (n_series - 1) / 2.0) * width

    fig_h = max(4.2, 0.52 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    summary_rows: List[Dict[str, object]] = []

    for index, label in enumerate(labels):
        values = [float(value_map.get(label, 0.0) or 0.0) for _, value_map in rows]
        ax.barh(
            y + offsets[index],
            values,
            height=width,
            color=SERIES_COLORS.get(label, None),
            alpha=0.9,
            label=label,
        )

    for key, value_map in rows:
        row = {"key": key}
        row.update({label: float(value_map.get(label, 0.0) or 0.0) for label in labels})
        summary_rows.append(row)

    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([key for key, _ in rows])
    ax.set_xlabel("Mean reward per episode")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def _parse_input(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Expected label=path, got: {raw}")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-run power-related reward penalties.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Entries like v17zg=path/to/metrics.json")
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--policy", default="mappo")
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    labels: List[str] = []
    step_maps: Dict[str, Dict[str, float]] = {}
    terminal_maps: Dict[str, Dict[str, float]] = {}
    episodes_per_series: Dict[str, int] = {}

    for raw in args.inputs:
        label, path = _parse_input(str(raw))
        runs = _pick_runs(
            _load_metrics(path),
            mix=str(args.mix),
            policy=str(args.policy),
            lam=float(args.lambda_value),
        )
        labels.append(label)
        episodes_per_series[label] = len(runs)
        step_maps[label] = _mean_component_map(runs, "step_reward_term_stats")
        terminal_maps[label] = _mean_component_map(runs, "terminal_reward_term_stats")

    step_rows = _filtered_rows(step_maps, STEP_POWER_KEYS)
    terminal_rows = _filtered_rows(terminal_maps, TERMINAL_POWER_KEYS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_summary = _plot_multi_compare(
        rows=step_rows,
        labels=labels,
        title="Step Power Penalties",
        out_path=out_dir / "step_power_penalty_compare_multi.png",
    )
    terminal_summary = _plot_multi_compare(
        rows=terminal_rows,
        labels=labels,
        title="Terminal Power Penalties",
        out_path=out_dir / "terminal_power_penalty_compare_multi.png",
    )

    summary = {
        "mix": str(args.mix),
        "policy": str(args.policy),
        "lambda_value": float(args.lambda_value),
        "episodes_per_series": episodes_per_series,
        "step_rows": step_summary,
        "terminal_rows": terminal_summary,
    }
    (out_dir / "power_penalty_compare_multi_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
