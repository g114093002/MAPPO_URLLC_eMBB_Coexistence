from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def _magnitude_ratio(values: Iterable[float]) -> float:
    nonzero = [abs(float(value)) for value in values if float(value) != 0.0]
    if len(nonzero) <= 1:
        return 1.0
    return max(nonzero) / max(min(nonzero), 1e-12)


def _symlog_linthresh(values: Iterable[float]) -> float:
    nonzero = sorted(abs(float(value)) for value in values if float(value) != 0.0)
    if not nonzero:
        return 1.0
    smallest = nonzero[0]
    median = nonzero[len(nonzero) // 2]
    return max(1.0, min(median * 0.25, smallest * 10.0))


def _apply_symlog_if_needed(ax: plt.Axes, axis: str, values: Iterable[float]) -> None:
    ratio = _magnitude_ratio(values)
    if ratio < 100.0:
        return
    linthresh = _symlog_linthresh(values)
    if axis == "x":
        ax.set_xscale("symlog", linthresh=linthresh)
    else:
        ax.set_yscale("symlog", linthresh=linthresh)


def _split_series_by_scale(
    named_series: List[Tuple[str, np.ndarray]],
    *,
    ratio_threshold: float = 25.0,
) -> List[List[Tuple[str, np.ndarray]]]:
    magnitudes: List[Tuple[str, np.ndarray, float]] = []
    for key, values in named_series:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        magnitudes.append((key, values, max_abs))
    magnitudes.sort(key=lambda item: item[2], reverse=True)
    if len(magnitudes) <= 1:
        return [[(key, values) for key, values, _ in magnitudes]]

    groups: List[List[Tuple[str, np.ndarray]]] = []
    current: List[Tuple[str, np.ndarray]] = []
    current_ref = 0.0
    for key, values, max_abs in magnitudes:
        if not current:
            current = [(key, values)]
            current_ref = max_abs
            continue
        safe_ref = max(current_ref, 1e-12)
        if max_abs > 0.0 and safe_ref / max(max_abs, 1e-12) > ratio_threshold:
            groups.append(current)
            current = [(key, values)]
            current_ref = max_abs
        else:
            current.append((key, values))
            current_ref = max(current_ref, max_abs)
    if current:
        groups.append(current)
    return groups


def _load_metrics(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_run(
    payload: Dict[str, object],
    *,
    mix: str,
    policy: str,
    lam: float,
    seed: int | None,
) -> Dict[str, object]:
    raw_runs = dict(payload.get("raw_runs", {}) or {})
    mix_runs = dict(raw_runs.get(mix, {}) or {})
    policy_runs = dict(mix_runs.get(policy, {}) or {})
    lam_key = f"lambda_{lam:g}"
    runs = list(policy_runs.get(lam_key, []) or [])
    if not runs:
        raise ValueError(f"No run found for mix={mix}, policy={policy}, lambda={lam_key}")
    if seed is None:
        return dict(runs[0])
    for run in runs:
        if int(run.get("evaluated_seed", -1)) == int(seed):
            return dict(run)
    raise ValueError(f"No run found for seed={seed}")


def _pick_runs(
    payload: Dict[str, object],
    *,
    mix: str,
    policy: str,
    lam: float,
) -> List[Dict[str, object]]:
    raw_runs = dict(payload.get("raw_runs", {}) or {})
    mix_runs = dict(raw_runs.get(mix, {}) or {})
    policy_runs = dict(mix_runs.get(policy, {}) or {})
    lam_key = f"lambda_{lam:g}"
    runs = list(policy_runs.get(lam_key, []) or [])
    if not runs:
        raise ValueError(f"No runs found for mix={mix}, policy={policy}, lambda={lam_key}")
    return [dict(run) for run in runs]


def _stats_to_series(stats: Dict[str, object], field: str = "sum") -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for key, payload in dict(stats or {}).items():
        if not isinstance(payload, dict):
            continue
        rows.append((str(key), float(payload.get(field, 0.0) or 0.0)))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows


def _mean_stats_to_series(
    runs: List[Dict[str, object]],
    *,
    stats_key: str,
    field: str = "sum",
) -> List[Tuple[str, float]]:
    keys: List[str] = []
    seen = set()
    for run in runs:
        stats = dict(run.get(stats_key, {}) or {})
        for key in stats.keys():
            key_s = str(key)
            if key_s in seen:
                continue
            seen.add(key_s)
            keys.append(key_s)

    rows: List[Tuple[str, float]] = []
    for key in keys:
        values: List[float] = []
        for run in runs:
            payload = dict(dict(run.get(stats_key, {}) or {}).get(key, {}) or {})
            values.append(float(payload.get(field, 0.0) or 0.0))
        rows.append((key, float(np.mean(values)) if values else 0.0))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows


def _filter_small_series(
    series: List[Tuple[str, float]],
    *,
    abs_threshold: float,
    rel_threshold: float,
) -> List[Tuple[str, float]]:
    if not series:
        return []
    max_abs = max(abs(value) for _, value in series)
    cutoff = max(float(abs_threshold), float(rel_threshold) * max_abs)
    filtered = [item for item in series if abs(item[1]) >= cutoff]
    return filtered if filtered else series[:1]


def _plot_barh(
    series: List[Tuple[str, float]],
    title: str,
    out_path: Path,
    top_k: int = 20,
    abs_threshold: float = 0.0,
    rel_threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    filtered = _filter_small_series(
        series,
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )
    trimmed = list(filtered[: max(int(top_k), 1)])
    if not trimmed:
        trimmed = [("none", 0.0)]
    labels = [_pretty_label(item[0]) for item in trimmed][::-1]
    values = [item[1] for item in trimmed][::-1]
    colors = ["#d62728" if value < 0 else "#1f77b4" for value in values]

    fig_h = max(4.0, 0.38 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    _apply_symlog_if_needed(ax, "x", values)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Reward")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return trimmed


def _plot_sum_compare(run: Dict[str, object], out_path: Path) -> None:
    named_values = [
        ("episode_reward_total_sum", float(run.get("episode_reward_total_sum", 0.0) or 0.0), "#111111"),
        ("logged_step_reward_sum", float(run.get("logged_step_reward_sum", 0.0) or 0.0), "#1f77b4"),
        ("logged_terminal_reward_sum", float(run.get("logged_terminal_reward_sum", 0.0) or 0.0), "#d62728"),
        ("logged_total_reward_sum", float(run.get("logged_total_reward_sum", 0.0) or 0.0), "#2ca02c"),
        ("reward_residual_unlogged", float(run.get("reward_residual_unlogged", 0.0) or 0.0), "#ff7f0e"),
    ]

    groups_raw = _split_series_by_scale(
        [(label, np.asarray([value], dtype=float)) for label, value, _ in named_values],
        ratio_threshold=100.0,
    )
    value_map = {label: value for label, value, _ in named_values}
    color_map = {label: color for label, _, color in named_values}
    groups: List[List[Tuple[str, float, str]]] = []
    for group in groups_raw:
        groups.append([(label, value_map[label], color_map[label]) for label, _ in group])

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(10, 3.6 * len(groups) + 0.3),
        squeeze=False,
    )
    axes_flat = list(axes[:, 0])
    for idx, (ax, group) in enumerate(zip(axes_flat, groups)):
        labels = [_pretty_label(label) for label, _, _ in group]
        values = [value for _, value, _ in group]
        colors = [color for _, _, color in group]
        ax.bar(labels, values, color=colors, alpha=0.9)
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        _apply_symlog_if_needed(ax, "y", values)
        if idx == 0:
            ax.set_title("Episode Reward Sum Comparison")
        ax.set_ylabel("Reward sum")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_episode_lines(
    runs: List[Dict[str, object]],
    *,
    title: str,
    out_path: Path,
) -> None:
    episodes = np.arange(1, len(runs) + 1, dtype=int)
    named_series = [
        ("episode reward total", np.asarray([float(run.get("episode_reward_total_sum", 0.0) or 0.0) for run in runs], dtype=float), "#111111"),
        ("step reward sum", np.asarray([float(run.get("logged_step_reward_sum", 0.0) or 0.0) for run in runs], dtype=float), "#1f77b4"),
        ("terminal reward", np.asarray([float(run.get("logged_terminal_reward_sum", 0.0) or 0.0) for run in runs], dtype=float), "#d62728"),
    ]
    groups_raw = _split_series_by_scale([(label, values) for label, values, _ in named_series], ratio_threshold=100.0)
    color_map = {label: color for label, _, color in named_series}
    groups: List[List[Tuple[str, np.ndarray, str]]] = []
    for group in groups_raw:
        groups.append([(label, values, color_map[label]) for label, values in group])

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(11, 4.2 * len(groups) + 0.4),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes[:, 0])
    for idx, (ax, group) in enumerate(zip(axes_flat, groups)):
        all_values: List[float] = []
        for label, values, color in group:
            ax.plot(episodes, values, color=color, marker="o", linewidth=2.0, label=label)
            all_values.extend(values.tolist())
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        _apply_symlog_if_needed(ax, "y", all_values)
        if idx == 0:
            ax.set_title(title)
        ax.set_ylabel("Reward sum")
        ax.set_xlim(float(episodes[0]), float(episodes[-1]))
        ax.margins(x=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
    axes_flat[-1].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _collect_component_series(
    runs: List[Dict[str, object]],
    *,
    stats_key: str,
) -> Dict[str, np.ndarray]:
    component_names: List[str] = []
    seen = set()
    for run in runs:
        stats = dict(run.get(stats_key, {}) or {})
        for key in stats.keys():
            key_s = str(key)
            if key_s in seen:
                continue
            seen.add(key_s)
            component_names.append(key_s)

    series: Dict[str, np.ndarray] = {}
    for name in component_names:
        values: List[float] = []
        for run in runs:
            payload = dict(dict(run.get(stats_key, {}) or {}).get(name, {}) or {})
            values.append(float(payload.get("sum", 0.0) or 0.0))
        series[name] = np.asarray(values, dtype=float)
    return series


def _filter_component_lines(
    component_series: Dict[str, np.ndarray],
    *,
    abs_threshold: float,
    rel_threshold: float,
) -> Dict[str, np.ndarray]:
    if not component_series:
        return {}
    max_abs = max(float(np.max(np.abs(values))) if values.size else 0.0 for values in component_series.values())
    cutoff = max(float(abs_threshold), float(rel_threshold) * max_abs)
    kept = {
        key: values
        for key, values in component_series.items()
        if values.size and float(np.max(np.abs(values))) >= cutoff
    }
    if kept:
        return kept
    best_key = max(
        component_series.keys(),
        key=lambda key: float(np.max(np.abs(component_series[key]))) if component_series[key].size else 0.0,
    )
    return {best_key: component_series[best_key]}


def _plot_component_lines(
    runs: List[Dict[str, object]],
    *,
    stats_key: str,
    title: str,
    out_path: Path,
    abs_threshold: float,
    rel_threshold: float,
) -> List[str]:
    component_series = _collect_component_series(runs, stats_key=stats_key)
    kept = _filter_component_lines(
        component_series,
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )
    episodes = np.arange(1, len(runs) + 1, dtype=int)
    sorted_items = sorted(
        kept.items(),
        key=lambda item: float(np.max(np.abs(item[1]))) if item[1].size else 0.0,
        reverse=True,
    )
    groups = _split_series_by_scale(sorted_items, ratio_threshold=25.0)

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(13.5, 4.1 * len(groups) + 0.6),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes[:, 0])
    for idx, (ax, group) in enumerate(zip(axes_flat, groups)):
        group_values: List[float] = []
        for key, values in group:
            ax.plot(episodes, values, marker="o", linewidth=1.8, label=_pretty_label(key))
            group_values.extend(values.tolist())
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
        _apply_symlog_if_needed(ax, "y", group_values)
        if idx == 0:
            ax.set_title(title)
        ax.set_ylabel("Component reward sum")
        ax.set_xlim(float(episodes[0]), float(episodes[-1]))
        ax.margins(x=0.0)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=9,
            ncol=1,
            borderaxespad=0.0,
        )
    axes_flat[-1].set_xlabel("Episode")
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return list(kept.keys())


def build_reward_breakdown(
    *,
    metrics_path: str | Path,
    mix: str,
    policy: str,
    lambda_value: float,
    seed: int | None,
    out_dir: str | Path,
    top_k: int = 20,
    abs_threshold: float = 0.05,
    rel_threshold: float = 0.01,
) -> Dict[str, object]:
    payload = _load_metrics(Path(metrics_path))
    run = _pick_run(
        payload,
        mix=str(mix),
        policy=str(policy),
        lam=float(lambda_value),
        seed=seed,
    )
    runs = _pick_runs(
        payload,
        mix=str(mix),
        policy=str(policy),
        lam=float(lambda_value),
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    step_series = _mean_stats_to_series(runs, stats_key="step_reward_term_stats", field="sum")
    terminal_series = _mean_stats_to_series(runs, stats_key="terminal_reward_term_stats", field="sum")

    kept_step_series = _plot_barh(
        step_series,
        title="Step Reward Components",
        out_path=out_dir_path / "step_reward_components_sum.png",
        top_k=int(top_k),
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )
    kept_terminal_series = _plot_barh(
        terminal_series,
        title="Terminal Reward Components",
        out_path=out_dir_path / "terminal_reward_components_sum.png",
        top_k=int(top_k),
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )
    _plot_sum_compare(run, out_dir_path / "reward_sum_compare.png")
    _plot_episode_lines(
        runs,
        title="Reward",
        out_path=out_dir_path / "reward_sum_by_episode_lines.png",
    )
    kept_step_episode_components = _plot_component_lines(
        runs,
        stats_key="step_reward_term_stats",
        title="Step Reward",
        out_path=out_dir_path / "step_reward_components_by_episode_lines.png",
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )
    kept_terminal_episode_components = _plot_component_lines(
        runs,
        stats_key="terminal_reward_term_stats",
        title="Terminal Reward",
        out_path=out_dir_path / "terminal_reward_components_by_episode_lines.png",
        abs_threshold=float(abs_threshold),
        rel_threshold=float(rel_threshold),
    )

    summary = {
        "mix": str(mix),
        "policy": str(policy),
        "lambda_value": float(lambda_value),
        "seed": seed,
        "episode_reward_total_sum": float(run.get("episode_reward_total_sum", 0.0) or 0.0),
        "logged_step_reward_sum": float(run.get("logged_step_reward_sum", 0.0) or 0.0),
        "logged_terminal_reward_sum": float(run.get("logged_terminal_reward_sum", 0.0) or 0.0),
        "logged_total_reward_sum": float(run.get("logged_total_reward_sum", 0.0) or 0.0),
        "reward_residual_unlogged": float(run.get("reward_residual_unlogged", 0.0) or 0.0),
        "episode_count_for_lambda_bucket": int(len(runs)),
        "kept_step_components_by_sum": kept_step_series,
        "kept_terminal_components_by_sum": kept_terminal_series,
        "kept_step_components_by_episode": kept_step_episode_components,
        "kept_terminal_components_by_episode": kept_terminal_episode_components,
        "dropped_step_components_by_sum": [item for item in step_series if item not in kept_step_series],
        "dropped_terminal_components_by_sum": [item for item in terminal_series if item not in kept_terminal_series],
    }
    (out_dir_path / "reward_breakdown_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot step/terminal reward breakdown for one unified lambda-stress run.")
    parser.add_argument("--metrics-path", required=True, help="Path to unified_lambda_stress_metrics.json")
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--abs-threshold", type=float, default=0.05)
    parser.add_argument("--rel-threshold", type=float, default=0.01)
    args = parser.parse_args()
    summary = build_reward_breakdown(
        metrics_path=args.metrics_path,
        mix=str(args.mix),
        policy=str(args.policy),
        lambda_value=float(args.lambda_value),
        seed=args.seed,
        out_dir=args.out_dir,
        top_k=int(args.top_k),
        abs_threshold=float(args.abs_threshold),
        rel_threshold=float(args.rel_threshold),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
