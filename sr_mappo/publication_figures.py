from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch

from sr_mappo.compare import _build_main_like_configs, _configure_density_scenario
from sr_mappo.config import SRMAPPOConfig
from sr_mappo.env import SRMAPPOPhaseAEnv
from sr_mappo.networks import SRMAPPOActorCritic
from sr_mappo.report import _build_model_for_env, _select_checkpoint, _policy_actions
from sr_mappo.types import MODE_KEEP, MODE_NAMES, MODE_OVERLAY, MODE_PUNCTURE


PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
PUBLICATION_DIR = RESULTS_DIR / "publication_figures"

LOAD_SWEEP = [10.0, 20.0, 30.0, 40.0, 50.0]
REPRESENTATIVE_LOAD = 40.0
TRAFFIC_MIX_TOTAL_LOAD = 40.0
TRAFFIC_MIXES = [
    ("7:3", 0.7, 0.3),
    ("5:5", 0.5, 0.5),
    ("3:7", 0.3, 0.7),
]
RELIABILITY_TARGETS = [1e-3, 1e-4, 1e-5]
EPISODES_PER_LOAD = 10
EPISODES_PER_MIX = 10
EPISODES_PER_TARGET = 10
CHANNEL_BIN_COUNT = 5

MODE_COLORS = {
    MODE_KEEP: "#4e79a7",
    MODE_OVERLAY: "#59a14f",
    MODE_PUNCTURE: "#e15759",
}
METRIC_COLORS = {
    "throughput": "#1f77b4",
    "admission": "#2ca02c",
    "power": "#ff7f0e",
    "puncture": "#d62728",
    "overlay": "#2ca02c",
    "keep": "#7f7f7f",
}


def _apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> List[str]:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = PUBLICATION_DIR / f"{stem}.{ext}"
        try:
            fig.savefig(path, bbox_inches="tight")
            paths.append(str(path))
        except PermissionError:
            fallback = PUBLICATION_DIR / f"{stem}_latest.{ext}"
            fig.savefig(fallback, bbox_inches="tight")
            paths.append(str(fallback))
    plt.close(fig)
    return paths


def _set_axes_style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_plain_y_ticks(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _set_power_axes_style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    _set_axes_style(ax, xlabel, ylabel, title)
    _set_plain_y_ticks(ax)


def _load_to_lambda(load: float) -> float:
    base_sys, _u, _e, _a, base_sim = _build_main_like_configs()
    base_total_per_uav = (
        max(1, int(np.ceil(base_sys.num_embb_users / base_sys.num_uavs)))
        + max(1, int(np.ceil(base_sys.num_urllc_users / base_sys.num_uavs)))
    )
    return float(base_sim.urllc_poisson_rate * load / max(base_total_per_uav, 1.0))


def _build_ratio_scenario(label: str, embb_share: float, urllc_share: float) -> Tuple:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    total_users = int(round(TRAFFIC_MIX_TOTAL_LOAD * base_sys.num_uavs))
    embb_users = int(round(total_users * embb_share))
    urllc_users = max(total_users - embb_users, 1)
    sys_cfg = deepcopy(base_sys)
    urllc_cfg = deepcopy(base_urllc)
    embb_cfg = deepcopy(base_embb)
    algo_cfg = deepcopy(base_algo)
    sim_cfg = deepcopy(base_sim)

    sys_cfg.num_embb_users = embb_users
    sys_cfg.num_urllc_users = urllc_users
    sys_cfg.refresh_derived_params()
    urllc_cfg.power_limits = [26] * sys_cfg.num_urllc_users
    embb_cfg.power_limits = [23] * sys_cfg.num_embb_users

    base_lambda_per_urllc = _load_to_lambda(REPRESENTATIVE_LOAD) / max(36.0, 1.0)
    sim_cfg.urllc_poisson_rate = float(base_lambda_per_urllc * sys_cfg.num_urllc_users)
    sim_cfg.verbose = False
    return label, sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _build_target_scenario(load: float, target_error: float) -> Tuple:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )
    urllc_cfg.target_error_probability = float(target_error)
    sim_cfg.verbose = False
    return sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _make_env_and_model(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, checkpoint: Path):
    rl_cfg = SRMAPPOConfig()
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, rl_cfg)
    cfg, model = _build_model_for_env(env, checkpoint)
    return env, cfg, model


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _linear_trend(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size < 2 or np.allclose(np.std(x), 0.0):
        return x, np.full_like(x, np.mean(y) if y.size else 0.0)
    coeff = np.polyfit(x, y, deg=1)
    xs = np.linspace(np.min(x), np.max(x), 100)
    ys = np.polyval(coeff, xs)
    return xs, ys


def _pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    frontier = []
    best_y = -np.inf
    for x_val, y_val in ordered:
        if y_val >= best_y:
            frontier.append((x_val, y_val))
            best_y = y_val
    return frontier


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {key: _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(value) for value in obj]
    return obj


def _run_policy_episode(env: SRMAPPOPhaseAEnv, model: SRMAPPOActorCritic, seed: int) -> Dict:
    observations, _ = env.reset(seed=seed)
    actor_hidden, critic_hidden = model.initial_state(
        batch_size=len(env.agent_ids),
        device=model.power_log_std.device,
    )

    step_trace = []
    selected_dual_feasible = []
    selected_channel_modes = []
    done = False

    while not done:
        current_obs = observations
        minislot, rb = env._current_cell()
        joint_actions, actor_hidden, critic_hidden = _policy_actions(
            env, model, current_obs, actor_hidden, critic_hidden
        )
        pre_shield = {
            agent_id: env.shield.sanitize_action(joint_actions[agent_id], current_obs[agent_id])
            for agent_id in env.agent_ids
        }
        resolved = env._enforce_joint_reliability(minislot, rb, current_obs, pre_shield)

        trace_row = {
            "minislot": int(minislot),
            "rb": int(rb),
            "overlay_candidate_count": 0,
            "overlay_feasible_count": 0,
            "selected_overlay": 0,
            "selected_puncture": 0,
            "selected_keep": 0,
            "total_power_step": 0.0,
            "scheduled_packets_step": 0,
            "embb_rate_step_proxy": 0.0,
        }

        for uav_idx, agent_id in enumerate(env.agent_ids):
            obs = current_obs[agent_id]
            final = resolved[agent_id]
            trace_row["overlay_candidate_count"] += len(obs.candidates)
            trace_row["overlay_feasible_count"] += sum(int(c.overlay_feasible) for c in obs.candidates)

            owner = int(env.owner_per_uav_rb[uav_idx, rb])
            base_rate = env._base_rate_for_cell(uav_idx, owner, rb)

            if final.candidate is None or final.action.mode == MODE_KEEP:
                trace_row["selected_keep"] += 1
                trace_row["embb_rate_step_proxy"] += base_rate
                continue

            candidate = final.candidate
            actual_power = env._project_actual_power(
                candidate.required_power_for_mode(final.action.mode),
                final.action.power_delta,
            )
            trace_row["total_power_step"] += float(actual_power)
            trace_row["scheduled_packets_step"] += 1

            if final.action.mode == MODE_OVERLAY:
                trace_row["selected_overlay"] += 1
                trace_row["embb_rate_step_proxy"] += max(base_rate - candidate.overlay_loss, 0.0)
                if candidate.puncture_feasible:
                    selected_dual_feasible.append(
                        {
                            "overlay_loss": float(candidate.overlay_loss),
                            "puncture_loss": float(candidate.puncture_loss),
                            "selected_mode": "overlay",
                        }
                    )
            elif final.action.mode == MODE_PUNCTURE:
                trace_row["selected_puncture"] += 1
                trace_row["embb_rate_step_proxy"] += max(base_rate - candidate.puncture_loss, 0.0)
                if candidate.overlay_feasible:
                    selected_dual_feasible.append(
                        {
                            "overlay_loss": float(candidate.overlay_loss),
                            "puncture_loss": float(candidate.puncture_loss),
                            "selected_mode": "puncture",
                        }
                    )

            selected_channel_modes.append(
                {
                    "channel_gain_db": float(10.0 * np.log10(max(candidate.channel_gain, 1e-15))),
                    "mode": int(final.action.mode),
                }
            )

        observations, _rewards, dones, _infos = env.step(joint_actions)
        done = all(dones.values())
        step_trace.append(trace_row)

    summary = env.summarize_episode()
    scheduled_mask = np.isfinite(env.scheduled_reliabilities) & (env.scheduled_uavs >= 0)
    reliabilities = env.scheduled_reliabilities[scheduled_mask].astype(float).tolist()

    base_embb_power = np.asarray(env.embb_result["power_allocation"], dtype=float)
    episode = {
        "summary": summary,
        "step_trace": step_trace,
        "mode_grid": env.mode_grid.copy(),
        "packet_grid": env.packet_grid.copy(),
        "owner_per_uav_rb": env.owner_per_uav_rb.copy(),
        "scheduled_reliabilities": reliabilities,
        "selected_dual_feasible": selected_dual_feasible,
        "selected_channel_modes": selected_channel_modes,
        "scheduled_power": env.scheduled_power.copy(),
        "base_embb_power": base_embb_power.copy(),
        "packet_release_minislots": env.packet_release_minislots.copy(),
        "scheduled_uavs": env.scheduled_uavs.copy(),
        "packet_sources": env.packet_sources.copy(),
        "num_packets": int(env.num_packets),
    }
    return episode


def _aggregate_episodes(episodes: List[Dict]) -> Dict[str, float]:
    summaries = [episode["summary"] for episode in episodes]
    return {
        "embb_rate": _safe_mean([float(s["embb_total_rate"]) for s in summaries]),
        "urllc_admission": _safe_mean([float(s["urllc_admission_rate"]) for s in summaries], 1.0),
        "urllc_reliability": _safe_mean([float(s["urllc_success_rate"]) for s in summaries], 1.0),
        "total_power": _safe_mean([float(s["total_power"]) for s in summaries]),
        "overlay_ratio": _safe_mean([float(s["overlay_ratio"]) for s in summaries]),
        "puncture_ratio": _safe_mean([float(s["puncture_ratio"]) for s in summaries]),
        "embb_only_fraction": _safe_mean([float(s["embb_only_fraction"]) for s in summaries]),
        "overlay_fraction": _safe_mean([float(s["overlay_fraction"]) for s in summaries]),
        "puncture_fraction": _safe_mean([float(s["puncture_fraction"]) for s in summaries]),
        "avg_puncture_loss": _safe_mean([float(s["avg_puncture_embb_loss"]) for s in summaries]),
        "avg_overlay_retention": _safe_mean([float(s["avg_overlay_retention"]) for s in summaries]),
    }


def _evaluate_scenario(
    env: SRMAPPOPhaseAEnv,
    model: SRMAPPOActorCritic,
    episodes: int,
    seed_base: int,
) -> Dict:
    episode_list = [_run_policy_episode(env, model, seed_base + idx) for idx in range(episodes)]
    aggregate = _aggregate_episodes(episode_list)
    representative = max(
        episode_list,
        key=lambda episode: (
            float(episode["summary"]["overlay_ratio"]),
            float(episode["summary"]["embb_total_rate"]),
        ),
    )
    return {
        "aggregate": aggregate,
        "episodes": episode_list,
        "representative": representative,
    }


def _plot_traffic_mix_sensitivity(mix_results: List[Dict]) -> List[str]:
    labels = [item["label"] for item in mix_results]
    metrics = [
        ("embb_rate", "Aggregate eMBB Throughput", "Mbps", METRIC_COLORS["throughput"], 1e6),
        ("urllc_admission", "URLLC Admission Ratio", "", METRIC_COLORS["admission"], 1.0),
        ("total_power", "Total Transmit Power", "mW", METRIC_COLORS["power"], 1e-3),
        ("puncture_ratio", "Puncture Ratio", "", METRIC_COLORS["puncture"], 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, (key, title, unit, color, scale) in zip(axes.flatten(), metrics):
        values = [item["aggregate"][key] / scale for item in mix_results]
        bars = ax.bar(x, values, color=color, width=0.6)
        if key == "total_power":
            _set_power_axes_style(ax, "Traffic ratio (eMBB:URLLC)", unit or "Ratio", title)
        else:
            _set_axes_style(ax, "Traffic ratio (eMBB:URLLC)", unit or "Ratio", title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("SR-MAPPO Traffic Mix Sensitivity at Fixed 40 UE/UAV", y=1.02)
    return _save_figure(fig, "01_traffic_mix_sensitivity")


def _plot_reliability_cdf(reliabilities: List[float], target_error: float) -> List[str]:
    values = np.sort(np.asarray(reliabilities, dtype=float))
    cdf = np.linspace(0.0, 1.0, values.size, endpoint=True) if values.size else np.asarray([])
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.plot(values, cdf, color=METRIC_COLORS["throughput"])
    ax.axvline(1.0 - target_error, color=METRIC_COLORS["puncture"], linestyle="--", linewidth=1.2, label=f"Target = 1 - {target_error:.0e}")
    _set_axes_style(ax, "Admitted URLLC reliability", "CDF", "CDF of Admitted URLLC Reliability")
    ax.set_xlim(max(0.999, np.min(values) - 5e-5), 1.000001)
    ax.legend(frameon=False, loc="lower right")
    return _save_figure(fig, "02_urllc_reliability_cdf")


def _plot_violation_probability(target_results: List[Dict]) -> List[str]:
    x = np.arange(len(target_results))
    labels = [f"1e-{int(abs(np.log10(item['target_error'])))}" for item in target_results]
    y = [item["violation_probability"] for item in target_results]
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax.plot(x, y, marker="o", color=METRIC_COLORS["puncture"])
    _set_axes_style(ax, "Reliability target (error probability)", "Violation probability", "Reliability Violation Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    return _save_figure(fig, "03_reliability_violation_probability")


def _plot_mode_selection_behavior(load_results: List[Dict], channel_modes: List[Dict]) -> List[str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    overall_overlay = _safe_mean([item["aggregate"]["overlay_fraction"] for item in load_results])
    overall_puncture = _safe_mean([item["aggregate"]["puncture_fraction"] for item in load_results])
    overall_keep = _safe_mean([item["aggregate"]["embb_only_fraction"] for item in load_results])
    axes[0].bar(["KEEP", "OVERLAY", "PUNCTURE"], [overall_keep, overall_overlay, overall_puncture], color=[MODE_COLORS[MODE_KEEP], MODE_COLORS[MODE_OVERLAY], MODE_COLORS[MODE_PUNCTURE]])
    _set_axes_style(axes[0], "", "Ratio", "Overall Mode Ratio")

    loads = [item["load"] for item in load_results]
    axes[1].plot(loads, [item["aggregate"]["overlay_ratio"] for item in load_results], marker="o", color=MODE_COLORS[MODE_OVERLAY], label="Overlay")
    axes[1].plot(loads, [item["aggregate"]["puncture_ratio"] for item in load_results], marker="s", color=MODE_COLORS[MODE_PUNCTURE], label="Puncture")
    axes[1].plot(loads, [item["aggregate"]["embb_only_fraction"] for item in load_results], marker="^", color=MODE_COLORS[MODE_KEEP], label="eMBB-only")
    _set_axes_style(axes[1], "Average UE load per UAV", "Ratio", "Mode Ratio vs URLLC Load")
    axes[1].legend(frameon=False)

    gains = np.asarray([item["channel_gain_db"] for item in channel_modes], dtype=float)
    if gains.size:
        bins = np.quantile(gains, np.linspace(0.0, 1.0, CHANNEL_BIN_COUNT + 1))
        bins = np.unique(bins)
        if bins.size < 2:
            bins = np.asarray([np.min(gains) - 1e-6, np.max(gains) + 1e-6])
        centers, overlay_vals, puncture_vals = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (gains >= lo) & (gains <= hi if hi == bins[-1] else gains < hi)
            modes = [item["mode"] for item, keep in zip(channel_modes, mask) if keep]
            overlay_vals.append(np.mean(np.asarray(modes) == MODE_OVERLAY) if modes else 0.0)
            puncture_vals.append(np.mean(np.asarray(modes) == MODE_PUNCTURE) if modes else 0.0)
            centers.append((lo + hi) / 2)
        axes[2].plot(centers, overlay_vals, marker="o", color=MODE_COLORS[MODE_OVERLAY], label="Overlay")
        axes[2].plot(centers, puncture_vals, marker="s", color=MODE_COLORS[MODE_PUNCTURE], label="Puncture")
    _set_axes_style(axes[2], "Selected-packet channel gain (dB)", "Conditional ratio", "Mode Ratio vs Channel Quality")
    axes[2].legend(frameon=False)
    return _save_figure(fig, "04_mode_selection_behavior")


def _plot_damage_aware_behavior(points: List[Dict]) -> List[str]:
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    for mode_name, color in [("overlay", MODE_COLORS[MODE_OVERLAY]), ("puncture", MODE_COLORS[MODE_PUNCTURE])]:
        subset = [item for item in points if item["selected_mode"] == mode_name]
        if not subset:
            continue
        ax.scatter(
            [item["overlay_loss"] / 1e6 for item in subset],
            [item["puncture_loss"] / 1e6 for item in subset],
            s=18,
            alpha=0.65,
            label=mode_name.capitalize(),
            color=color,
        )
    diag_max = 0.0
    if points:
        diag_max = max(max(item["overlay_loss"], item["puncture_loss"]) for item in points) / 1e6
    ax.plot([0, diag_max], [0, diag_max], linestyle="--", color="#666666", linewidth=1.0)
    _set_axes_style(ax, "eMBB loss under overlay (Mbps)", "eMBB loss under puncture (Mbps)", "Damage-Aware Decision Behavior")
    ax.legend(frameon=False)
    return _save_figure(fig, "05_damage_aware_decisions")


def _plot_temporal_evolution(representative: Dict) -> List[str]:
    traces = representative["step_trace"]
    num_minislots = max(int(max(item["minislot"] for item in traces)) + 1, 1)
    x = np.arange(num_minislots)
    embb = np.zeros(num_minislots)
    cumulative = np.zeros(num_minislots)
    overlay = np.zeros(num_minislots)
    puncture = np.zeros(num_minislots)
    power = np.zeros(num_minislots)
    for minislot in x:
        slot_rows = [item for item in traces if item["minislot"] == minislot]
        embb[minislot] = sum(item["embb_rate_step_proxy"] for item in slot_rows) / 1e6
        cumulative[minislot] = sum(item["scheduled_packets_step"] for item in traces if item["minislot"] <= minislot)
        overlay[minislot] = sum(item["selected_overlay"] for item in slot_rows)
        puncture[minislot] = sum(item["selected_puncture"] for item in slot_rows)
        power[minislot] = sum(item["total_power_step"] for item in slot_rows)

    fig, axes = plt.subplots(4, 1, figsize=(7.6, 7.4), sharex=True, constrained_layout=True)
    axes[0].plot(x, embb, color=METRIC_COLORS["throughput"], marker="o")
    _set_axes_style(axes[0], "", "Mbps", "eMBB Throughput by Minislot")
    axes[1].plot(x, cumulative, color=METRIC_COLORS["admission"], marker="s")
    _set_axes_style(axes[1], "", "Packets", "Cumulative Admitted URLLC Packets")
    axes[2].bar(x - 0.15, overlay, width=0.3, color=MODE_COLORS[MODE_OVERLAY], label="Overlay")
    axes[2].bar(x + 0.15, puncture, width=0.3, color=MODE_COLORS[MODE_PUNCTURE], label="Puncture")
    _set_axes_style(axes[2], "", "Count", "Mode Selection Count by Minislot")
    axes[2].legend(frameon=False)
    axes[3].plot(x, power * 1e3, color=METRIC_COLORS["power"], marker="^")
    _set_power_axes_style(axes[3], "Minislot index", "mW", "Transmit Power by Minislot")
    return _save_figure(fig, "06_temporal_evolution_within_slot")


def _plot_resource_utilization_heatmap(representative: Dict) -> List[str]:
    mode_grid = np.asarray(representative["mode_grid"], dtype=int)
    owners = np.asarray(representative["owner_per_uav_rb"], dtype=int)
    num_uavs = mode_grid.shape[0]
    fig, axes = plt.subplots(1, num_uavs, figsize=(4.0 * num_uavs, 3.8), constrained_layout=True)
    if num_uavs == 1:
        axes = [axes]
    cmap = matplotlib.colors.ListedColormap(["#4e79a7", "#59a14f", "#e15759"])
    bounds = [MODE_KEEP - 0.5, MODE_KEEP + 0.5, MODE_OVERLAY + 0.5, MODE_PUNCTURE + 0.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    for uav_idx, ax in enumerate(axes):
        grid = mode_grid[uav_idx].T
        ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"UAV {uav_idx + 1}")
        ax.set_xlabel("RB index")
        ax.set_ylabel("Minislot index")
        overlay_pct = np.mean(mode_grid[uav_idx] == MODE_OVERLAY)
        puncture_pct = np.mean(mode_grid[uav_idx] == MODE_PUNCTURE)
        ax.text(0.02, 1.02, f"Overlay {overlay_pct:.1%} | Puncture {puncture_pct:.1%}", transform=ax.transAxes, fontsize=8)
        for rb in range(grid.shape[1]):
            owner = owners[uav_idx, rb]
            ax.text(rb, -0.45, f"E{owner}" if owner >= 0 else "E-", ha="center", va="top", fontsize=6)
    fig.suptitle("SR-MAPPO Resource Utilization Heatmap", y=1.03)
    return _save_figure(fig, "07_resource_utilization_heatmap")


def _plot_tradeoff(load_results: List[Dict], target_results: List[Dict]) -> List[str]:
    points = []
    fig, ax = plt.subplots(figsize=(6.3, 4.8), constrained_layout=True)
    for item in load_results:
        x = item["aggregate"]["urllc_reliability"]
        y = item["aggregate"]["embb_rate"] / 1e6
        points.append((x, y))
        ax.scatter(x, y, color=METRIC_COLORS["throughput"], s=35)
        ax.annotate(f"L{int(item['load'])}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    for item in target_results:
        x = item["aggregate"]["urllc_reliability"]
        y = item["aggregate"]["embb_rate"] / 1e6
        points.append((x, y))
        ax.scatter(x, y, color=METRIC_COLORS["puncture"], marker="s", s=32)
        ax.annotate(f"t={item['target_error']:.0e}", (x, y), textcoords="offset points", xytext=(4, -8), fontsize=7)
    frontier = _pareto_frontier(points)
    if frontier:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], linestyle="--", color="#444444", linewidth=1.0)
    _set_axes_style(ax, "Admitted URLLC reliability", "Aggregate eMBB throughput (Mbps)", "Throughput-Reliability Tradeoff")
    return _save_figure(fig, "08_throughput_reliability_tradeoff")


def _plot_power_efficiency(episodes: List[Dict]) -> List[str]:
    x = np.asarray([episode["summary"]["total_power"] for episode in episodes], dtype=float) * 1e3
    y = np.asarray([episode["summary"]["embb_total_rate"] / 1e6 for episode in episodes], dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.scatter(x, y, color=METRIC_COLORS["throughput"], alpha=0.65, s=18)
    xs, ys = _linear_trend(x, y)
    ax.plot(xs, ys, color=METRIC_COLORS["puncture"], linewidth=1.4, linestyle="--")
    _set_power_axes_style(ax, "Total transmit power (mW)", "Aggregate eMBB throughput (Mbps)", "Power-Throughput Efficiency")
    return _save_figure(fig, "09_power_throughput_efficiency")


def _plot_overlay_opportunity_funnel(load_results: List[Dict]) -> List[str]:
    fig, ax = plt.subplots(figsize=(6.8, 4.3), constrained_layout=True)
    x = np.arange(len(load_results))
    labels = [str(int(item["load"])) for item in load_results]
    candidates = [item["overlay_candidate_mean"] for item in load_results]
    feasible = [item["overlay_feasible_mean"] for item in load_results]
    selected = [item["overlay_selected_mean"] for item in load_results]
    ax.bar(x - 0.25, candidates, width=0.25, color="#9ecae1", label="Candidate")
    ax.bar(x, feasible, width=0.25, color="#74c476", label="Feasible")
    ax.bar(x + 0.25, selected, width=0.25, color="#31a354", label="Selected")
    _set_axes_style(ax, "Average UE load per UAV", "Pairs / episode", "Overlay Opportunity Funnel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    return _save_figure(fig, "10_overlay_opportunity_funnel")


def _capture_topology_snapshot(load: float, seed: int = 12345) -> Dict[str, object]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, SRMAPPOConfig())
    env.reset(seed=seed)
    topology = getattr(env, "last_topology", None) or {}
    return {
        "num_uavs": sys_cfg.num_uavs,
        "num_embb": sys_cfg.num_embb_users,
        "num_urllc": sys_cfg.num_urllc_users,
        "user_positions": np.asarray(topology.get("user_positions", np.zeros((0, 2))), dtype=float),
        "uav_positions": np.asarray(topology.get("uav_positions", np.zeros((0, 2))), dtype=float),
        "serving_hints": np.asarray(topology.get("serving_hints", np.zeros((0,))), dtype=int),
        "area_width": float(sys_cfg.area_width),
        "area_height": float(sys_cfg.area_height),
    }


def _plot_uav_user_distribution(snapshot: Dict[str, object]) -> List[str]:
    users = snapshot["user_positions"]
    uavs = snapshot["uav_positions"]
    num_embb = int(snapshot["num_embb"])
    num_urllc = int(snapshot["num_urllc"])
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    if users.size > 0:
        embb = users[:num_embb]
        urllc = users[num_embb:num_embb + num_urllc]
        if embb.size > 0:
            ax.scatter(embb[:, 0], embb[:, 1], s=18, c="#4e79a7", label="eMBB UE", alpha=0.85, edgecolors="white", linewidths=0.3)
        if urllc.size > 0:
            ax.scatter(urllc[:, 0], urllc[:, 1], s=22, c="#e15759", label="URLLC UE", alpha=0.90, edgecolors="white", linewidths=0.3)
    if uavs.size > 0:
        ax.scatter(uavs[:, 0], uavs[:, 1], s=90, c="#000000", marker="^", label="UAV", edgecolors="white", linewidths=0.6)
    ax.set_xlim(0, snapshot["area_width"])
    ax.set_ylim(0, snapshot["area_height"])
    _set_axes_style(ax, "X (m)", "Y (m)", "UAV and UE Distribution")
    ax.legend(frameon=False, loc="upper right")
    return _save_figure(fig, "11_uav_ue_distribution")


def generate_publication_figures() -> Dict[str, object]:
    _apply_ieee_style()
    checkpoint = _select_checkpoint()
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

    load_results = []
    all_reliabilities = []
    all_channel_modes = []
    all_dual_feasible_points = []
    all_episode_points = []

    for load_idx, load in enumerate(LOAD_SWEEP):
        base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(load, base_sys, base_urllc, base_embb, base_algo, base_sim)
        env, cfg, model = _make_env_and_model(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, checkpoint)
        result = _evaluate_scenario(env, model, EPISODES_PER_LOAD, 7000 + 100 * load_idx)
        reliabilities = [value for episode in result["episodes"] for value in episode["scheduled_reliabilities"]]
        channel_modes = [item for episode in result["episodes"] for item in episode["selected_channel_modes"]]
        dual_points = [item for episode in result["episodes"] for item in episode["selected_dual_feasible"]]
        all_reliabilities.extend(reliabilities)
        all_channel_modes.extend(channel_modes)
        all_dual_feasible_points.extend(dual_points)
        all_episode_points.extend(result["episodes"])
        candidate_mean = _safe_mean([sum(row["overlay_candidate_count"] for row in episode["step_trace"]) for episode in result["episodes"]])
        feasible_mean = _safe_mean([sum(row["overlay_feasible_count"] for row in episode["step_trace"]) for episode in result["episodes"]])
        selected_mean = _safe_mean([float(episode["summary"]["overlay_count"]) for episode in result["episodes"]])
        load_results.append({
            "load": load,
            "aggregate": result["aggregate"],
            "representative": result["representative"],
            "overlay_candidate_mean": candidate_mean,
            "overlay_feasible_mean": feasible_mean,
            "overlay_selected_mean": selected_mean,
        })

    traffic_mix_results = []
    for mix_idx, (label, embb_share, urllc_share) in enumerate(TRAFFIC_MIXES):
        label, sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_ratio_scenario(label, embb_share, urllc_share)
        env, cfg, model = _make_env_and_model(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, checkpoint)
        result = _evaluate_scenario(env, model, EPISODES_PER_MIX, 9000 + 100 * mix_idx)
        traffic_mix_results.append({"label": label, "aggregate": result["aggregate"]})

    target_results = []
    for target_idx, target_error in enumerate(RELIABILITY_TARGETS):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_target_scenario(REPRESENTATIVE_LOAD, target_error)
        env, cfg, model = _make_env_and_model(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, checkpoint)
        result = _evaluate_scenario(env, model, EPISODES_PER_TARGET, 11000 + 100 * target_idx)
        violations = [float(value < (1.0 - target_error)) for episode in result["episodes"] for value in episode["scheduled_reliabilities"]]
        target_results.append({
            "target_error": target_error,
            "aggregate": result["aggregate"],
            "violation_probability": _safe_mean(violations, 0.0),
        })

    representative = next(item["representative"] for item in load_results if item["load"] == REPRESENTATIVE_LOAD)
    topology_snapshot = _capture_topology_snapshot(REPRESENTATIVE_LOAD)
    output_paths = []
    output_paths += _plot_traffic_mix_sensitivity(traffic_mix_results)
    output_paths += _plot_reliability_cdf(all_reliabilities, RELIABILITY_TARGETS[0])
    output_paths += _plot_violation_probability(target_results)
    output_paths += _plot_mode_selection_behavior(load_results, all_channel_modes)
    output_paths += _plot_damage_aware_behavior(all_dual_feasible_points)
    output_paths += _plot_temporal_evolution(representative)
    output_paths += _plot_resource_utilization_heatmap(representative)
    output_paths += _plot_tradeoff(load_results, target_results)
    output_paths += _plot_power_efficiency(all_episode_points)
    output_paths += _plot_overlay_opportunity_funnel(load_results)
    output_paths += _plot_uav_user_distribution(topology_snapshot)

    note_lines = [
        "# SR-MAPPO Publication Figure Notes",
        "",
        f"- Checkpoint: `{checkpoint.name}`",
        f"- Load sweep: {LOAD_SWEEP}",
        f"- Representative load: {REPRESENTATIVE_LOAD} UE/UAV",
        f"- Traffic-mix total load: {TRAFFIC_MIX_TOTAL_LOAD} UE/UAV",
        f"- Reliability target sweep: {RELIABILITY_TARGETS}",
        f"- Episodes per load/mix/target: {EPISODES_PER_LOAD}/{EPISODES_PER_MIX}/{EPISODES_PER_TARGET}",
        "- All figures are SR-MAPPO only; no Greedy baseline is plotted.",
        "- Resource heatmap uses the overlay-rich representative episode at 40 UE/UAV.",
        "",
        "## Figure Set",
        "- 01 Traffic mix sensitivity",
        "- 02 URLLC reliability CDF",
        "- 03 Reliability violation probability",
        "- 04 Mode selection behavior",
        "- 05 Damage-aware decision behavior",
        "- 06 Temporal evolution within slot",
        "- 07 Resource utilization heatmap",
        "- 08 Throughput-reliability tradeoff",
        "- 09 Power-throughput efficiency",
        "- 10 Overlay opportunity funnel",
        "- 11 UAV and UE distribution snapshot",
    ]
    (PUBLICATION_DIR / "PUBLICATION_FIGURE_NOTES.md").write_text("\n".join(note_lines), encoding="utf-8")

    payload = {
        "checkpoint": str(checkpoint),
        "loads": [
            {
                "load": item["load"],
                "aggregate": item["aggregate"],
                "overlay_candidate_mean": item["overlay_candidate_mean"],
                "overlay_feasible_mean": item["overlay_feasible_mean"],
                "overlay_selected_mean": item["overlay_selected_mean"],
            }
            for item in load_results
        ],
        "traffic_mix": traffic_mix_results,
        "target_sweep": target_results,
        "topology_snapshot": {
            "num_uavs": topology_snapshot["num_uavs"],
            "num_embb": topology_snapshot["num_embb"],
            "num_urllc": topology_snapshot["num_urllc"],
        },
        "output_paths": output_paths,
    }
    (PUBLICATION_DIR / "publication_figure_data.json").write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = generate_publication_figures()
    print(f"SR-MAPPO publication figures generated using {result['checkpoint']}")
