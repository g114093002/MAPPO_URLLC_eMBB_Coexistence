from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_history(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def _collect_component_names(history: List[Dict[str, object]]) -> List[str]:
    names = set()
    for record in history:
        payload = record.get("reward_components", {})
        if isinstance(payload, dict):
            names.update(str(key) for key in payload.keys())
    return sorted(names)


def _build_x_axis(
    history: List[Dict[str, object]],
    *,
    x_axis: str,
) -> Tuple[np.ndarray, str]:
    if str(x_axis) == "episodes":
        values = []
        usable = True
        for item in history:
            raw = item.get("cumulative_rollout_episode_count", None)
            if raw is None:
                usable = False
                break
            try:
                values.append(float(raw))
            except Exception:
                usable = False
                break
        if usable and values:
            return np.asarray(values, dtype=float), "Cumulative rollout episodes"
    if str(x_axis) == "effective_episodes":
        values = []
        usable = True
        for idx, item in enumerate(history):
            raw = item.get("train_effective_episodes", None)
            if raw is None:
                usable = False
                break
            try:
                values.append(float(raw))
            except Exception:
                usable = False
                break
        if usable and values:
            return np.asarray(values, dtype=float), "Effective episodes"
    iterations = np.asarray(
        [float(item.get("iteration", idx + 1) or (idx + 1)) for idx, item in enumerate(history)],
        dtype=float,
    )
    return iterations, "Training iteration"


def _build_series(
    history: List[Dict[str, object]],
    component_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    iterations = np.asarray(
        [float(item.get("iteration", idx + 1) or (idx + 1)) for idx, item in enumerate(history)],
        dtype=float,
    )
    rollout_reward = np.asarray(
        [float(item.get("rollout_reward", 0.0) or 0.0) for item in history],
        dtype=float,
    )
    series: Dict[str, np.ndarray] = {}
    for name in component_names:
        values = []
        for record in history:
            payload = record.get("reward_components", {})
            if isinstance(payload, dict):
                values.append(float(payload.get(name, 0.0) or 0.0))
            else:
                values.append(0.0)
        series[name] = np.asarray(values, dtype=float)
    return iterations, rollout_reward, series


def _select_components(
    series: Dict[str, np.ndarray],
    *,
    top_k: int,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    ranked = sorted(
        series.keys(),
        key=lambda name: float(np.mean(np.abs(series[name]))),
        reverse=True,
    )
    selected = ranked[: max(int(top_k), 1)]
    selected_series = {name: series[name] for name in selected}
    dropped = [name for name in ranked if name not in selected]
    if dropped:
        other = np.zeros_like(next(iter(series.values())))
        for name in dropped:
            other = other + series[name]
        selected.append("other")
        selected_series["other"] = other
    return selected, selected_series


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_reward_decomposition(
    history_path: Path,
    *,
    out_dir: Path,
    top_k: int,
    x_axis: str = "iteration",
) -> Dict[str, Path]:
    history = _load_history(history_path)
    component_names = _collect_component_names(history)
    if not component_names:
        raise ValueError(
            f"No reward_components found in {history_path}. "
            "This history was likely generated before reward decomposition logging was added."
        )
    iterations, rollout_reward, series = _build_series(history, component_names)
    x_values, x_label = _build_x_axis(history, x_axis=str(x_axis))
    selected_names, selected_series = _select_components(series, top_k=top_k)
    component_sum = np.zeros_like(rollout_reward)
    for name in selected_names:
        component_sum = component_sum + selected_series[name]
    residual_total_minus_step = rollout_reward - component_sum

    pos_names = [name for name in selected_names if np.any(selected_series[name] > 0.0)]
    neg_names = [name for name in selected_names if np.any(selected_series[name] < 0.0)]

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    if pos_names:
        ax.stackplot(
            x_values,
            *[np.clip(selected_series[name], 0.0, None) for name in pos_names],
            labels=pos_names,
            alpha=0.85,
        )
    if neg_names:
        ax.stackplot(
            x_values,
            *[np.clip(selected_series[name], None, 0.0) for name in neg_names],
            labels=neg_names,
            alpha=0.70,
        )
    ax.plot(x_values, component_sum, color="tab:red", linewidth=1.5, linestyle="--", label="sum(shown_components)")

    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_title("Reward Components (Signed)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean reward contribution per step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    signed_png_path = out_dir / "reward_component_stacked_area.png"
    fig.savefig(signed_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    abs_series = {name: np.abs(selected_series[name]) for name in selected_names}
    abs_names = [name for name in selected_names if np.any(abs_series[name] > 0.0)]
    if abs_names:
        ax.stackplot(
            x_values,
            *[abs_series[name] for name in abs_names],
            labels=abs_names,
            alpha=0.85,
        )
    ax.set_title("Reward Components (Absolute Magnitude)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Absolute mean contribution per step")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    abs_png_path = out_dir / "reward_component_abs_magnitude_area.png"
    fig.savefig(abs_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_values, rollout_reward, color="black", linewidth=2.0, label="rollout_reward")
    if len(iterations) >= 10:
        reward_avg10 = np.convolve(rollout_reward, np.ones(10) / 10.0, mode="valid")
        ax.plot(
            x_values[9:],
            reward_avg10,
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label="rollout_reward_avg10",
        )
    if len(iterations) >= 20:
        reward_avg20 = np.convolve(rollout_reward, np.ones(20) / 20.0, mode="valid")
        ax.plot(
            x_values[19:],
            reward_avg20,
            color="tab:blue",
            linewidth=2.0,
            linestyle=":",
            label="rollout_reward_avg20",
        )
    ax.set_title("Total Training Reward")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    lines_png_path = out_dir / "reward_total_lines.png"
    fig.savefig(lines_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 7))
    for idx, name in enumerate(selected_names):
        values = selected_series[name]
        if len(values) >= 10:
            smoothed = np.convolve(values, np.ones(10) / 10.0, mode="valid")
            x_axis_values = x_values[9:]
        else:
            smoothed = values
            x_axis_values = x_values
        ax.plot(
            x_axis_values,
            smoothed,
            linewidth=1.8,
            label=name,
        )
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_title("Step Reward Components Mean Trend")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean reward contribution per step (avg10)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    step_overlay_png_path = out_dir / "step_reward_component_mean_overlay_avg10.png"
    fig.savefig(step_overlay_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    nonzero_names = [name for name in selected_names if np.any(np.abs(selected_series[name]) > 1.0e-12)]
    fig, ax = plt.subplots(figsize=(15, 7))
    for name in nonzero_names:
        values = selected_series[name]
        if len(values) >= 10:
            smoothed = np.convolve(values, np.ones(10) / 10.0, mode="valid")
            x_axis_values = x_values[9:]
        else:
            smoothed = values
            x_axis_values = x_values
        ax.plot(x_axis_values, smoothed, linewidth=1.8, label=name)
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_title("Step Reward Components Mean Trend (Nonzero Only)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean reward contribution per step (avg10)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    step_overlay_nonzero_png_path = out_dir / "step_reward_component_mean_overlay_avg10_nonzero_only.png"
    fig.savefig(step_overlay_nonzero_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ax_top, ax_bottom = axes
    ax_top.plot(x_values, rollout_reward, color="black", linewidth=2.0, label="total_rollout_reward")
    ax_top.plot(x_values, component_sum, color="tab:blue", linewidth=2.0, label="step_component_sum")
    ax_top.plot(
        x_values,
        residual_total_minus_step,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label="terminal_proxy = total - step_sum",
    )
    ax_top.set_title("Total Reward vs Step Sum vs Terminal Proxy")
    ax_top.set_ylabel("Reward")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(frameon=False, ncol=3)

    safe_total = np.where(np.abs(rollout_reward) > 1.0e-12, rollout_reward, np.nan)
    step_share = component_sum / safe_total
    residual_share = residual_total_minus_step / safe_total
    ax_bottom.plot(x_values, step_share, color="tab:blue", linewidth=1.8, label="step_sum / total")
    ax_bottom.plot(
        x_values,
        residual_share,
        color="tab:red",
        linewidth=1.8,
        linestyle="--",
        label="terminal_proxy / total",
    )
    ax_bottom.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax_bottom.axhline(1.0, color="gray", linewidth=1.0, alpha=0.35, linestyle=":")
    ax_bottom.set_xlabel(x_label)
    ax_bottom.set_ylabel("Fraction of total reward")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(frameon=False)
    fig.tight_layout()
    step_vs_terminal_png_path = out_dir / "step_sum_vs_terminal_proxy_vs_total.png"
    fig.savefig(step_vs_terminal_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Per-component line dashboard for easier trend inspection than stacked areas.
    n_components = len(selected_names)
    n_cols = 2
    n_rows = max(int(np.ceil(n_components / float(n_cols))), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.6 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    for idx, name in enumerate(selected_names):
        ax = axes_flat[idx]
        values = selected_series[name]
        ax.plot(x_values, values, color="tab:blue", linewidth=1.8, label=name)
        if len(values) >= 10:
            avg10 = np.convolve(values, np.ones(10) / 10.0, mode="valid")
            ax.plot(
                x_values[9:],
                avg10,
                color="tab:orange",
                linewidth=1.5,
                linestyle="--",
                label=f"{name}_avg10",
            )
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean reward contribution per step")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for idx in range(n_components, len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.tight_layout()
    dashboard_png_path = out_dir / "reward_component_lines_dashboard.png"
    fig.savefig(dashboard_png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    component_line_dir = out_dir / "reward_component_lines"
    component_line_dir.mkdir(parents=True, exist_ok=True)
    component_line_paths: Dict[str, Path] = {}
    for name in selected_names:
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
        fig, ax = plt.subplots(figsize=(13, 4.8))
        values = selected_series[name]
        ax.plot(x_values, values, color="tab:blue", linewidth=2.0, label=name)
        if len(values) >= 10:
            avg10 = np.convolve(values, np.ones(10) / 10.0, mode="valid")
            ax.plot(
                x_values[9:],
                avg10,
                color="tab:orange",
                linewidth=2.0,
                linestyle="--",
                label="avg10",
            )
        if len(values) >= 20:
            avg20 = np.convolve(values, np.ones(20) / 20.0, mode="valid")
            ax.plot(
                x_values[19:],
                avg20,
                color="tab:green",
                linewidth=1.8,
                linestyle=":",
                label="avg20",
            )
        ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
        ax.set_title(f"Reward Component: {name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean reward contribution per step")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        component_path = component_line_dir / f"{safe_name}.png"
        fig.savefig(component_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        component_line_paths[name] = component_path

    json_path = out_dir / "reward_component_stacked_area.json"

    _save_json(
        json_path,
        {
            "history_path": str(history_path),
            "top_k": int(top_k),
            "selected_components": selected_names,
            "iterations": iterations.tolist(),
            "x_axis": str(x_axis),
            "x_axis_values": x_values.tolist(),
            "x_axis_label": x_label,
            "rollout_reward": rollout_reward.tolist(),
            "shown_component_sum": component_sum.tolist(),
            "residual_total_minus_step": residual_total_minus_step.tolist(),
            "reward_components": {
                name: selected_series[name].tolist()
                for name in selected_names
            },
        },
    )
    return {
        "signed_png": signed_png_path,
        "abs_png": abs_png_path,
        "lines_png": lines_png_path,
        "step_overlay_png": step_overlay_png_path,
        "step_overlay_nonzero_png": step_overlay_nonzero_png_path,
        "step_vs_terminal_proxy_png": step_vs_terminal_png_path,
        "component_lines_dashboard_png": dashboard_png_path,
        "component_line_dir": component_line_dir,
        "json": json_path,
        "component_line_paths": component_line_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clean-training reward decomposition stacked area chart.")
    parser.add_argument("--history-path", required=True, help="Path to *_clean_history.json")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs")
    parser.add_argument("--top-k", type=int, default=10, help="Number of dominant reward components to show explicitly")
    parser.add_argument(
        "--x-axis",
        choices=["iteration", "episodes", "effective_episodes"],
        default="iteration",
        help="Horizontal axis for plots.",
    )
    args = parser.parse_args()

    outputs = plot_reward_decomposition(
        Path(args.history_path),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
        x_axis=str(args.x_axis),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
