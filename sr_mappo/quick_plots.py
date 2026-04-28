"""Lightweight plotting utilities for quick KPI debugging.

This module intentionally avoids running the full SR-MAPPO report pipeline.
It supports:
  - Core KPI vs load (MAPPO vs Greedy) from an existing `sr_mappo_report_metrics.json`.
  - A small URLLC arrival-rate sweep (lambda sweep) at a fixed average UE load by
    re-evaluating MAPPO (checkpoint) and Greedy (myopic_throughput_greedy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .compare import _build_main_like_configs, _configure_density_scenario
from .config import SRMAPPOConfig, cfg_from_dict
from .env import SRMAPPOPhaseAEnv
from .networks import SRMAPPOActorCritic
from .report import run_env_episode
from .trainer import set_phase_a_embb_power_runtime


_COLOR_MAPPO = "tab:orange"
_COLOR_GREEDY = "#8c564b"  # tab:brown


def _parse_csv_floats(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(tok.strip()) for tok in text.split(",") if tok.strip()]


def _load_metrics(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "sr_mappo" not in payload or "greedy" not in payload:
        raise KeyError("Expected keys `sr_mappo` and `greedy` in report metrics JSON.")
    return payload


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)


def _plot_two_series(
    ax,
    x: np.ndarray,
    y_mappo: np.ndarray,
    y_greedy: np.ndarray,
    y_label: str,
    title: str,
    xlabel: str,
    *,
    mappo_label: str = "MAPPO",
    greedy_label: str = "Greedy",
) -> None:
    ax.plot(
        x,
        y_mappo,
        color=_COLOR_MAPPO,
        marker="s",
        markersize=6,
        linewidth=2.0,
        label=mappo_label,
    )
    ax.plot(
        x,
        y_greedy,
        color=_COLOR_GREEDY,
        marker="o",
        markersize=6,
        linewidth=2.0,
        label=greedy_label,
    )
    _style_axis(ax, title, xlabel, y_label)


def plot_core_kpi_debug(metrics: Dict, out_path: Path) -> Path:
    """1x4 core KPI figure (MAPPO vs Greedy) vs load."""
    rl = metrics["sr_mappo"]
    greedy = metrics["greedy"]

    loads = np.asarray(rl["loads"], dtype=float)
    loads_g = np.asarray(greedy["loads"], dtype=float)
    if loads.shape != loads_g.shape or not np.allclose(loads, loads_g):
        raise ValueError("Load grids differ between `sr_mappo` and `greedy` series; cannot plot cleanly.")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    xlabel = "Average UE load per UAV"

    _plot_two_series(
        axes[0],
        loads,
        np.asarray(rl["embb_rate"], dtype=float) / 1e6,
        np.asarray(greedy["embb_rate"], dtype=float) / 1e6,
        "Mbps",
        "Aggregate eMBB throughput",
        xlabel,
    )
    _plot_two_series(
        axes[1],
        loads,
        np.asarray(rl["urllc_admission"], dtype=float),
        np.asarray(greedy["urllc_admission"], dtype=float),
        "Ratio",
        "URLLC admission ratio",
        xlabel,
    )
    _plot_two_series(
        axes[2],
        loads,
        np.asarray(rl["embb_positive_rate_ratio"], dtype=float),
        np.asarray(greedy["embb_positive_rate_ratio"], dtype=float),
        "Ratio",
        "eMBB service ratio",
        xlabel,
    )
    _plot_two_series(
        axes[3],
        loads,
        np.asarray(rl["total_power"], dtype=float) * 1e3,
        np.asarray(greedy["total_power"], dtype=float) * 1e3,
        "mW",
        "Total transmit power",
        xlabel,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend_.remove() if ax.get_legend() else None
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _set_lambda(sim_cfg, sys_cfg, lam: float) -> None:
    if hasattr(sim_cfg, "fixed_urllc_poisson_rate"):
        sim_cfg.fixed_urllc_poisson_rate = True
    if hasattr(sim_cfg, "urllc_poisson_rate"):
        sim_cfg.urllc_poisson_rate = float(lam)
    if hasattr(sim_cfg, "urllc_arrival_prob"):
        denom = float(getattr(sys_cfg, "num_urllc_users", 1) or 1)
        sim_cfg.urllc_arrival_prob = float(np.clip(float(lam) / denom, 0.0, 1.0))


def run_lambda_sweep(
    checkpoint_path: Path,
    experiment_line: str,
    *,
    fixed_load: float,
    lambdas: Iterable[float],
    episodes_per_lambda: int,
) -> Dict[str, object]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_cfg = cfg_from_dict(payload.get("cfg"))
    extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
    runtime_enabled = bool(
        extra.get(
            "phase_a_embb_power_runtime_enabled",
            getattr(checkpoint_cfg.env, "allow_phase_a_embb_power_adjustment", False),
        )
    )

    series = {"lambda": [], "mappo": {"urllc_admission": [], "embb_rate": [], "total_power": []}, "greedy": {"urllc_admission": [], "embb_rate": [], "total_power": []}}
    model = None
    cfg = checkpoint_cfg

    for idx, lam in enumerate(list(lambdas)):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            float(fixed_load), base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, "urllc_user_ratio"):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        _set_lambda(sim_cfg, sys_cfg, float(lam))

        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)
        if model is None:
            model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, cfg)
            model.load_state_dict(payload["model_state_dict"], strict=False)
            model.to(torch.device(cfg.training.device))
            model.eval()
        set_phase_a_embb_power_runtime(env, model, runtime_enabled)

        seed_base = 10_000 + 97 * idx
        mappo_runs = []
        greedy_runs = []
        for ep in range(max(int(episodes_per_lambda), 1)):
            mappo_runs.append(
                run_env_episode(
                    env,
                    model,
                    cfg,
                    seed=seed_base + ep,
                    collect_trace=False,
                    use_greedy=False,
                    cache_tag=f"lambda_sweep_{fixed_load}_{lam}",
                )
            )
            greedy_runs.append(
                run_env_episode(
                    env,
                    model=None,
                    cfg=cfg,
                    seed=seed_base + ep + 50_000,
                    collect_trace=False,
                    use_greedy=True,
                    greedy_policy="myopic_throughput",
                    cache_tag=f"lambda_sweep_{fixed_load}_{lam}_greedy",
                )
            )

        def _mean(runs: List[Dict], key: str) -> float:
            vals = np.asarray([float(item.get(key, np.nan)) for item in runs], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else float("nan")

        series["lambda"].append(float(lam))
        series["mappo"]["urllc_admission"].append(_mean(mappo_runs, "urllc_admission"))
        series["mappo"]["embb_rate"].append(_mean(mappo_runs, "embb_rate"))
        series["mappo"]["total_power"].append(_mean(mappo_runs, "total_power"))
        series["greedy"]["urllc_admission"].append(_mean(greedy_runs, "urllc_admission"))
        series["greedy"]["embb_rate"].append(_mean(greedy_runs, "embb_rate"))
        series["greedy"]["total_power"].append(_mean(greedy_runs, "total_power"))

    return {
        "fixed_load": float(fixed_load),
        "checkpoint": str(checkpoint_path),
        "experiment_line": str(experiment_line),
        "series": series,
    }


def plot_lambda_sweep(bundle: Dict[str, object], out_path: Path) -> Path:
    series = bundle["series"]
    x = np.asarray(series["lambda"], dtype=float)
    mappo = series["mappo"]
    greedy = series["greedy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    xlabel = "λ"

    _plot_two_series(
        axes[0],
        x,
        np.asarray(mappo["urllc_admission"], dtype=float),
        np.asarray(greedy["urllc_admission"], dtype=float),
        "Ratio",
        "URLLC admission ratio",
        xlabel,
    )
    _plot_two_series(
        axes[1],
        x,
        np.asarray(mappo["embb_rate"], dtype=float) / 1e6,
        np.asarray(greedy["embb_rate"], dtype=float) / 1e6,
        "Mbps",
        "Aggregate eMBB throughput",
        xlabel,
    )
    _plot_two_series(
        axes[2],
        x,
        np.asarray(mappo["total_power"], dtype=float) * 1e3,
        np.asarray(greedy["total_power"], dtype=float) * 1e3,
        "mW",
        "Total transmit power",
        xlabel,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend_.remove() if ax.get_legend() else None
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate minimal KPI debug plots (MAPPO vs Greedy).")
    parser.add_argument("--metrics", type=str, required=True, help="Path to `sr_mappo_report_metrics.json`.")
    parser.add_argument("--fixed-load", type=float, default=20.0, help="Fixed average UE load for lambda sweep.")
    parser.add_argument("--lambdas", type=str, default="4,8,12,16", help="Comma-separated lambda values.")
    parser.add_argument("--episodes-per-lambda", type=int, default=6, help="Episodes per lambda value.")
    parser.add_argument("--skip-core-kpi", action="store_true", help="Skip core KPI vs load plot.")
    parser.add_argument("--skip-lambda-sweep", action="store_true", help="Skip lambda sweep plot.")
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics).expanduser().resolve()
    metrics = _load_metrics(metrics_path)
    out_dir = metrics_path.parent

    if not args.skip_core_kpi:
        plot_core_kpi_debug(metrics, out_dir / "core_kpi_debug.png")

    if not args.skip_lambda_sweep:
        checkpoint = Path(str(metrics.get("checkpoint", ""))).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found (from metrics): {checkpoint}")
        experiment_line = str(metrics.get("experiment_line", "")).strip()
        if not experiment_line:
            raise ValueError("Missing `experiment_line` in metrics; cannot infer experiment preset for env config.")
        bundle = run_lambda_sweep(
            checkpoint,
            experiment_line,
            fixed_load=float(args.fixed_load),
            lambdas=_parse_csv_floats(args.lambdas),
            episodes_per_lambda=int(args.episodes_per_lambda),
        )
        plot_lambda_sweep(bundle, out_dir / "lambda_sweep.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
