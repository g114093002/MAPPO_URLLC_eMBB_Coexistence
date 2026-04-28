from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np

from sr_mappo.compare import _build_main_like_configs, _configure_density_scenario
from sr_mappo.env import SRMAPPOPhaseAEnv
from sr_mappo.publication_figures import _apply_ieee_style, _json_ready, _linear_trend, _safe_mean
from sr_mappo.report import _build_model_for_env, _policy_actions, _select_checkpoint
from sr_mappo.types import CandidatePacket, MODE_KEEP, MODE_NAMES, MODE_OVERLAY, MODE_PUNCTURE


PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
PUNCTURE_DIR = RESULTS_DIR / "puncture_pressure_figures"

LOAD_SWEEP = [10.0, 20.0, 30.0, 40.0, 50.0]
REPRESENTATIVE_LOAD = 40.0
EPISODES_PER_LOAD = 10

MODE_COLORS = {
    MODE_KEEP: "#7f7f7f",
    MODE_OVERLAY: "#1b9e77",
    MODE_PUNCTURE: "#d95f02",
}
ACCENT_BLUE = "#1f77b4"
ACCENT_RED = "#d62728"


def _save_figure(fig: plt.Figure, stem: str) -> List[str]:
    PUNCTURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = PUNCTURE_DIR / f"{stem}.{ext}"
        try:
            fig.savefig(path, bbox_inches="tight")
            paths.append(str(path))
        except PermissionError:
            fallback = PUNCTURE_DIR / f"{stem}_latest.{ext}"
            fig.savefig(fallback, bbox_inches="tight")
            paths.append(str(fallback))
    plt.close(fig)
    return paths


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_plain_y_ticks(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _style_power_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    _style_axis(ax, xlabel, ylabel, title)
    _set_plain_y_ticks(ax)


def _flush_axis(ax: plt.Axes, xlim: Optional[Tuple[float, float]] = None, ylim: Optional[Tuple[float, float]] = None) -> None:
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0.0, y=0.02)


def _make_env_model_for_load(load: float, checkpoint: Path):
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    cfg, model = _build_model_for_env(env, checkpoint)
    return env, cfg, model


def _candidate_priority(candidate: CandidatePacket) -> Tuple[float, float, float]:
    return (
        float(candidate.best_utility),
        float(candidate.overlay_feasible),
        float(-candidate.puncture_loss),
    )


def _reference_candidate(env: SRMAPPOPhaseAEnv, observation, minislot: int, final_candidate: Optional[CandidatePacket]) -> Optional[CandidatePacket]:
    if final_candidate is not None:
        return final_candidate
    feasible = [c for c in observation.candidates if c.puncture_feasible or c.overlay_feasible]
    if not feasible:
        return None
    return max(feasible, key=_candidate_priority)


def _candidate_urgency(env: SRMAPPOPhaseAEnv, candidate: Optional[CandidatePacket], minislot: int) -> float:
    if candidate is None or candidate.packet_id >= env.packet_release_minislots.size:
        return 0.0
    release = int(env.packet_release_minislots[candidate.packet_id])
    age = max(minislot - release, 0)
    remaining = max(env.urllc_cfg.max_latency_minislots - age, 0)
    return float(1.0 - remaining / max(env.urllc_cfg.max_latency_minislots, 1))


def _convert_puncture_to_no_puncture(action, observation) -> object:
    if action.mode != MODE_PUNCTURE or action.packet_option <= 0:
        return action
    packet_idx = action.packet_option - 1
    if packet_idx < len(observation.candidates):
        candidate = observation.candidates[packet_idx]
        if candidate.overlay_feasible and observation.masks.packet_mask[MODE_OVERLAY, action.packet_option] > 0:
            return type(action)(mode=MODE_OVERLAY, packet_option=action.packet_option, power_delta=action.power_delta)
    return type(action)(mode=MODE_KEEP, packet_option=0, power_delta=0.0)


def _run_episode(
    env: SRMAPPOPhaseAEnv,
    model,
    seed: int,
    no_puncture: bool = False,
) -> Dict:
    observations, _ = env.reset(seed=seed)
    actor_hidden, critic_hidden = model.initial_state(
        batch_size=len(env.agent_ids),
        device=model.power_log_std.device,
    )

    decision_records: List[Dict] = []
    minislot_records: List[Dict] = []
    done = False

    while not done:
        current_obs = observations
        minislot, rb = env._current_cell()
        joint_actions, actor_hidden, critic_hidden = _policy_actions(
            env, model, current_obs, actor_hidden, critic_hidden
        )
        if no_puncture:
            joint_actions = {
                agent_id: _convert_puncture_to_no_puncture(action, current_obs[agent_id])
                for agent_id, action in joint_actions.items()
            }
        pre_shield = {
            agent_id: env.shield.sanitize_action(joint_actions[agent_id], current_obs[agent_id])
            for agent_id in env.agent_ids
        }
        resolved = env._enforce_joint_reliability(minislot, rb, current_obs, pre_shield)

        minislot_arrivals = int(np.sum(env.packet_release_minislots == minislot)) if env.packet_release_minislots.size > 0 else 0
        step_overlay = 0
        step_puncture = 0
        step_power = 0.0
        step_admitted = 0
        step_embb_proxy = 0.0

        for uav_idx, agent_id in enumerate(env.agent_ids):
            obs = current_obs[agent_id]
            final = resolved[agent_id]
            ref_candidate = _reference_candidate(env, obs, minislot, final.candidate)
            available_packets = len(env._available_packet_ids(minislot))
            feasible_overlay_count = sum(int(c.overlay_feasible) for c in obs.candidates)
            feasible_puncture_count = sum(int(c.puncture_feasible) for c in obs.candidates)
            deficit = max(0, available_packets - feasible_overlay_count)
            urgency = _candidate_urgency(env, ref_candidate, minislot)
            puncture_loss = float(ref_candidate.puncture_loss) if ref_candidate is not None else 0.0
            overlay_loss = float(ref_candidate.overlay_loss) if ref_candidate is not None else 0.0
            overlay_feasible = bool(ref_candidate.overlay_feasible) if ref_candidate is not None else False
            puncture_feasible = bool(ref_candidate.puncture_feasible) if ref_candidate is not None else False
            owner = int(env.owner_per_uav_rb[uav_idx, rb])
            base_rate = env._base_rate_for_cell(uav_idx, owner, rb)

            actual_power = 0.0
            if final.candidate is not None and final.action.mode != MODE_KEEP:
                actual_power = env._project_actual_power(
                    final.candidate.required_power_for_mode(final.action.mode),
                    final.action.power_delta,
                )
                step_power += float(actual_power)
                step_admitted += 1

            if final.action.mode == MODE_OVERLAY:
                step_overlay += 1
                chosen_loss = float(final.candidate.overlay_loss) if final.candidate is not None else 0.0
            elif final.action.mode == MODE_PUNCTURE:
                step_puncture += 1
                chosen_loss = float(final.candidate.puncture_loss) if final.candidate is not None else 0.0
            else:
                chosen_loss = 0.0
            step_embb_proxy += max(base_rate - chosen_loss, 0.0)

            decision_records.append(
                {
                    "load": float((env.sys_cfg.num_embb_users + env.sys_cfg.num_urllc_users) / max(env.sys_cfg.num_uavs, 1)),
                    "minislot": int(minislot),
                    "rb": int(rb),
                    "uav": int(uav_idx),
                    "available_packets": int(available_packets),
                    "overlay_feasible_count": int(feasible_overlay_count),
                    "puncture_feasible_count": int(feasible_puncture_count),
                    "resource_deficit": float(deficit),
                    "urgency": float(urgency),
                    "puncture_loss": float(puncture_loss),
                    "overlay_loss": float(overlay_loss),
                    "chosen_mode": int(final.action.mode),
                    "selected_overlay": int(final.action.mode == MODE_OVERLAY),
                    "selected_puncture": int(final.action.mode == MODE_PUNCTURE),
                    "selected_keep": int(final.action.mode == MODE_KEEP),
                    "overlay_feasible_exists": int(feasible_overlay_count > 0),
                    "puncture_feasible_exists": int(feasible_puncture_count > 0),
                    "chosen_channel_gain_db": float(10.0 * np.log10(max(ref_candidate.channel_gain, 1e-15))) if ref_candidate is not None else -150.0,
                    "collision_rewritten": int(final.collision_rewritten),
                    "joint_reliability_rewritten": int(final.joint_reliability_rewritten),
                }
            )

        minislot_records.append(
            {
                "minislot": int(minislot),
                "arrivals": minislot_arrivals,
                "admitted": int(step_admitted),
                "overlay": int(step_overlay),
                "puncture": int(step_puncture),
                "power": float(step_power),
                "embb_proxy_mbps": float(step_embb_proxy / 1e6),
            }
        )

        observations, _rewards, dones, _infos = env.step(joint_actions)
        done = all(dones.values())

    summary = env.summarize_episode()
    return {
        "summary": summary,
        "decision_records": decision_records,
        "minislot_records": minislot_records,
        "mode_grid": env.mode_grid.copy(),
        "owner_per_uav_rb": env.owner_per_uav_rb.copy(),
        "packet_grid": env.packet_grid.copy(),
        "packet_release_minislots": env.packet_release_minislots.copy(),
        "scheduled_reliabilities": env.scheduled_reliabilities[np.isfinite(env.scheduled_reliabilities)].astype(float).tolist(),
        "puncture_loss_sum": float(np.sum(env.selected_puncture_losses)) if env.selected_puncture_losses else 0.0,
        "overlay_loss_sum": float(np.sum(env.selected_overlay_losses)) if env.selected_overlay_losses else 0.0,
        "puncture_count": float(summary["puncture_count"]),
        "overlay_count": float(summary["overlay_count"]),
        "scheduled_packets": float(summary["scheduled_packets"]),
    }


def _evaluate_load(load: float, checkpoint: Path) -> Dict:
    env, cfg, model = _make_env_model_for_load(load, checkpoint)
    actual_episodes = []
    no_puncture_episodes = []
    for ep in range(EPISODES_PER_LOAD):
        seed = int(cfg.training.train_seed + 2000 + 100 * load + ep)
        actual_episodes.append(_run_episode(env, model, seed=seed, no_puncture=False))
        no_puncture_episodes.append(_run_episode(env, model, seed=seed, no_puncture=True))

    representative = max(
        actual_episodes,
        key=lambda item: (
            _safe_mean([row["arrivals"] for row in item["minislot_records"]]),
            item["summary"]["puncture_count"],
            item["summary"]["overlay_count"],
        ),
    )
    return {
        "load": load,
        "episodes": actual_episodes,
        "no_puncture_episodes": no_puncture_episodes,
        "representative": representative,
    }


def _bin_probabilities(records: List[Dict], key: str, num_bins: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([float(item[key]) for item in records], dtype=float)
    if values.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    lo, hi = float(np.min(values)), float(np.max(values))
    if np.isclose(lo, hi):
        bins = np.linspace(lo, hi + 1.0, num_bins + 1)
    else:
        bins = np.linspace(lo, hi, num_bins + 1)
    centers, puncture_prob, overlay_prob = [], [], []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (values >= left) & (values <= right if right == bins[-1] else values < right)
        subset = [item for item, keep in zip(records, mask) if keep]
        if not subset:
            continue
        centers.append((left + right) / 2.0)
        puncture_prob.append(np.mean([item["selected_puncture"] for item in subset]))
        overlay_prob.append(np.mean([item["selected_overlay"] for item in subset]))
    return np.asarray(centers), np.asarray(puncture_prob), np.asarray(overlay_prob)


def _plot_puncture_necessity_vs_decision(records: List[Dict]) -> List[str]:
    conflict_records = [item for item in records if item["overlay_feasible_exists"] or item["puncture_feasible_exists"]]
    x, puncture_prob, overlay_prob = _bin_probabilities(conflict_records, "resource_deficit", num_bins=7)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(x, puncture_prob, marker="o", color=MODE_COLORS[MODE_PUNCTURE], label="P(puncture)")
    ax.plot(x, overlay_prob, marker="s", color=MODE_COLORS[MODE_OVERLAY], label="P(overlay)")
    _style_axis(ax, "Minimum required URLLC resource deficit", "Decision probability", "Puncture Necessity vs Decision")
    if x.size:
        _flush_axis(ax, (float(np.min(x)), float(np.max(x))), (0.0, 1.0))
    ax.legend(frameon=False)
    return _save_figure(fig, "01_puncture_necessity_vs_decision")


def _plot_embb_damage_vs_decision(records: List[Dict]) -> List[str]:
    conflict_records = [item for item in records if item["puncture_feasible_exists"]]
    x, puncture_prob, overlay_prob = _bin_probabilities(conflict_records, "puncture_loss", num_bins=7)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    x_mbps = x / 1e6 if x.size else x
    ax.plot(x_mbps, puncture_prob, marker="o", color=MODE_COLORS[MODE_PUNCTURE], label="P(puncture)")
    ax.plot(x_mbps, overlay_prob, marker="s", color=MODE_COLORS[MODE_OVERLAY], label="P(overlay)")
    _style_axis(ax, "Expected eMBB loss if puncture (Mbps)", "Decision probability", "eMBB Damage vs Decision")
    if x_mbps.size:
        _flush_axis(ax, (float(np.min(x_mbps)), float(np.max(x_mbps))), (0.0, 1.0))
    ax.legend(frameon=False)
    return _save_figure(fig, "02_embb_damage_vs_decision")


def _plot_decision_boundary(records: List[Dict]) -> List[str]:
    filtered = [item for item in records if item["overlay_feasible_exists"] or item["puncture_feasible_exists"]]
    urgency_bins = np.linspace(0.0, 1.0, 7)
    damage_values = np.asarray([item["puncture_loss"] / 1e6 for item in filtered], dtype=float)
    damage_max = max(float(np.max(damage_values)) if damage_values.size else 1.0, 1.0)
    damage_bins = np.linspace(0.0, damage_max, 7)
    mode_map = np.full((len(damage_bins) - 1, len(urgency_bins) - 1), MODE_KEEP, dtype=int)
    for yi, (d0, d1) in enumerate(zip(damage_bins[:-1], damage_bins[1:])):
        for xi, (u0, u1) in enumerate(zip(urgency_bins[:-1], urgency_bins[1:])):
            subset = [
                item for item in filtered
                if (u0 <= item["urgency"] <= u1 if u1 == urgency_bins[-1] else u0 <= item["urgency"] < u1)
                and (d0 <= item["puncture_loss"] / 1e6 <= d1 if d1 == damage_bins[-1] else d0 <= item["puncture_loss"] / 1e6 < d1)
            ]
            if subset:
                counts = {mode: sum(int(item["chosen_mode"] == mode) for item in subset) for mode in [MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE]}
                mode_map[yi, xi] = max(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    cmap = ListedColormap([MODE_COLORS[MODE_KEEP], MODE_COLORS[MODE_OVERLAY], MODE_COLORS[MODE_PUNCTURE]])
    ax.imshow(mode_map, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(urgency_bins) - 1))
    ax.set_xticklabels([f"{v:.2f}" for v in urgency_bins[:-1]])
    ax.set_yticks(np.arange(len(damage_bins) - 1))
    ax.set_yticklabels([f"{v:.2f}" for v in damage_bins[:-1]])
    _style_axis(ax, "URLLC urgency (normalized)", "eMBB damage if puncture (Mbps)", "Decision Boundary Plot")
    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=MODE_NAMES[mode])
        for mode, color in MODE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    return _save_figure(fig, "03_decision_boundary")


def _plot_resource_conflict_timeline(representative: Dict) -> List[str]:
    rows = representative["minislot_records"]
    x = np.arange(len(rows))
    arrivals = np.asarray([row["arrivals"] for row in rows], dtype=float)
    admitted = np.asarray([row["admitted"] for row in rows], dtype=float)
    embb = np.asarray([row["embb_proxy_mbps"] for row in rows], dtype=float)
    puncture = np.asarray([row["puncture"] for row in rows], dtype=float)
    overlay = np.asarray([row["overlay"] for row in rows], dtype=float)

    spike_threshold = np.percentile(arrivals, 75) if arrivals.size else 0.0
    spike_slots = [idx for idx, value in enumerate(arrivals) if value >= spike_threshold and value > 0]

    fig, axes = plt.subplots(4, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=True)
    axes[0].bar(x, arrivals, color=ACCENT_BLUE, alpha=0.55, label="Arrivals")
    axes[0].plot(x, admitted, color=MODE_COLORS[MODE_OVERLAY], marker="o", label="Admitted")
    _style_axis(axes[0], "", "Packets", "URLLC Arrivals vs Admitted")
    axes[0].legend(frameon=False)

    axes[1].plot(x, embb, color=ACCENT_BLUE, marker="o")
    _style_axis(axes[1], "", "Mbps", "eMBB Throughput")

    axes[2].bar(x - 0.15, puncture, width=0.3, color=MODE_COLORS[MODE_PUNCTURE], label="Puncture")
    axes[2].bar(x + 0.15, overlay, width=0.3, color=MODE_COLORS[MODE_OVERLAY], label="Overlay")
    _style_axis(axes[2], "", "Count", "Mode Selection Count")
    axes[2].legend(frameon=False)

    axes[3].plot(x, np.asarray([row["power"] for row in rows], dtype=float) * 1e3, color=ACCENT_RED, marker="^")
    _style_power_axis(axes[3], "Minislot index", "mW", "Total Transmit Power")

    for ax in axes:
        for slot in spike_slots:
            ax.axvspan(slot - 0.5, slot + 0.5, color="#fdd0a2", alpha=0.18)
        _flush_axis(ax, (0.0, float(len(rows) - 1)))
    return _save_figure(fig, "04_resource_conflict_timeline")


def _plot_no_puncture_counterfactual(load_results: List[Dict]) -> List[str]:
    gaps = []
    colors = []
    for result in load_results:
        for actual, counter in zip(result["episodes"], result["no_puncture_episodes"]):
            gaps.append((actual["summary"]["embb_total_rate"] - counter["summary"]["embb_total_rate"]) / 1e6)
            colors.append(result["load"])
    gaps = np.asarray(gaps, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    axes[0].hist(gaps, bins=12, color=ACCENT_BLUE, alpha=0.75, edgecolor="white")
    axes[0].axvline(float(np.mean(gaps)) if gaps.size else 0.0, color=ACCENT_RED, linestyle="--", linewidth=1.4)
    _style_axis(axes[0], "Throughput gap vs no-puncture (Mbps)", "Episode count", '"What if No Puncture?"')
    if gaps.size:
        _flush_axis(axes[0], (float(np.min(gaps)), float(np.max(gaps))))

    scatter_x = np.arange(gaps.size)
    scatter = axes[1].scatter(scatter_x, gaps, c=np.asarray(colors, dtype=float), cmap="viridis", s=22, alpha=0.8)
    _style_axis(axes[1], "Episode index", "Throughput gap (Mbps)", "Gap by Episode and Load")
    _flush_axis(axes[1], (0.0, float(max(gaps.size - 1, 1))))
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label("UE load / UAV")
    return _save_figure(fig, "05_no_puncture_counterfactual")


def _plot_puncture_efficiency(load_results: List[Dict]) -> List[str]:
    puncture_counts = []
    packets_saved = []
    efficiencies = []
    loads = []
    for result in load_results:
        for actual, counter in zip(result["episodes"], result["no_puncture_episodes"]):
            punctures = float(actual["summary"]["puncture_count"])
            delta_packets = float(actual["summary"]["scheduled_packets"] - counter["summary"]["scheduled_packets"])
            puncture_counts.append(punctures)
            packets_saved.append(max(delta_packets, 0.0))
            efficiencies.append(max(delta_packets, 0.0) / punctures if punctures > 0 else 0.0)
            loads.append(result["load"])

    x = np.asarray(puncture_counts, dtype=float)
    y = np.asarray(packets_saved, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    scatter = axes[0].scatter(x, y, c=np.asarray(loads, dtype=float), cmap="plasma", s=24, alpha=0.8)
    trend_x, trend_y = _linear_trend(x, y)
    axes[0].plot(trend_x, trend_y, color=ACCENT_RED, linestyle="--", linewidth=1.2)
    _style_axis(axes[0], "# of punctures", "URLLC packets saved", "Puncture Efficiency")
    if x.size:
        _flush_axis(axes[0], (float(np.min(x)), float(np.max(x))), (0.0, float(np.max(y) * 1.05 if y.size else 1.0)))
    cbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.02)
    cbar.set_label("UE load / UAV")

    if x.size:
        lo, hi = float(np.min(x)), float(np.max(x))
        bins = np.linspace(lo, hi + (1.0 if np.isclose(lo, hi) else 0.0), 7)
        xb, eff_vals = [], []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (x >= left) & (x <= right if right == bins[-1] else x < right)
            if not np.any(mask):
                continue
            xb.append((left + right) / 2.0)
            eff_vals.append(float(np.mean(np.asarray(efficiencies)[mask])))
    else:
        xb, eff_vals = [], []
    axes[1].plot(xb, eff_vals, marker="o", color=ACCENT_BLUE)
    _style_axis(axes[1], "# of punctures (binned)", "Saved packets / puncture", "Puncture Efficiency Ratio")
    if xb:
        _flush_axis(axes[1], (float(np.min(xb)), float(np.max(xb))), (0.0, max(1.0, float(np.max(eff_vals) * 1.1))))
    return _save_figure(fig, "06_puncture_efficiency")


def _plot_embb_loss_attribution(load_results: List[Dict]) -> List[str]:
    labels = [str(int(item["load"])) for item in load_results]
    puncture_loss = []
    overlay_loss = []
    for item in load_results:
        puncture_loss.append(_safe_mean([episode["puncture_loss_sum"] for episode in item["episodes"]]) / 1e6)
        overlay_loss.append(_safe_mean([episode["overlay_loss_sum"] for episode in item["episodes"]]) / 1e6)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.bar(x, puncture_loss, color=MODE_COLORS[MODE_PUNCTURE], label="Puncture-induced loss")
    ax.bar(x, overlay_loss, bottom=puncture_loss, color=MODE_COLORS[MODE_OVERLAY], label="Overlay-induced loss")
    _style_axis(ax, "Average UE load per UAV", "Total eMBB loss (Mbps)", "eMBB Loss Attribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.legend(frameon=False)
    return _save_figure(fig, "07_embb_loss_attribution")


def _plot_overlay_missed_opportunities(load_results: List[Dict]) -> List[str]:
    labels = [str(int(item["load"])) for item in load_results]
    feasible = np.asarray([_safe_mean([sum(int(r["overlay_feasible_exists"]) for r in ep["decision_records"]) for ep in item["episodes"]]) for item in load_results], dtype=float)
    selected = np.asarray([_safe_mean([float(ep["summary"]["overlay_count"]) for ep in item["episodes"]]) for item in load_results], dtype=float)
    utilization = np.divide(selected, feasible, out=np.zeros_like(selected), where=feasible > 0)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    ax1.bar(x - 0.18, feasible, width=0.36, color="#9ecae1", label="Feasible overlay cases")
    ax1.bar(x + 0.18, selected, width=0.36, color=MODE_COLORS[MODE_OVERLAY], label="Selected overlay")
    _style_axis(ax1, "Average UE load per UAV", "Count / episode", "Overlay Missed Opportunities")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlim(-0.5, len(labels) - 0.5)
    ax2 = ax1.twinx()
    ax2.plot(x, utilization, marker="o", color=ACCENT_RED, linewidth=1.6, label="Utilization ratio")
    ax2.set_ylabel("Overlay utilization ratio")
    ax2.set_ylim(0.0, 1.0)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    return _save_figure(fig, "08_overlay_missed_opportunities")


def _plot_conflict_resolution_funnel(load_results: List[Dict]) -> List[str]:
    labels = [str(int(item["load"])) for item in load_results]
    candidate_steps = []
    feasible_overlay = []
    executed_overlay = []
    rewritten_keep = []
    for item in load_results:
        decisions = [record for episode in item["episodes"] for record in episode["decision_records"]]
        candidate_steps.append(float(sum(int(record["overlay_feasible_exists"] or record["puncture_feasible_exists"]) for record in decisions)))
        feasible_overlay.append(float(sum(int(record["overlay_feasible_exists"]) for record in decisions)))
        executed_overlay.append(float(sum(int(record["selected_overlay"]) for record in decisions)))
        rewritten_keep.append(float(sum(int(record["selected_keep"] and record["collision_rewritten"]) for record in decisions)))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.bar(x - 0.24, candidate_steps, width=0.24, color="#c6dbef", label="Conflict steps")
    ax.bar(x, feasible_overlay, width=0.24, color="#9ecae1", label="Feasible overlay")
    ax.bar(x + 0.24, executed_overlay, width=0.24, color=MODE_COLORS[MODE_OVERLAY], label="Executed overlay")
    _style_axis(ax, "Average UE load per UAV", "Count / episode", "Conflict Resolution Funnel")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.legend(frameon=False)
    return _save_figure(fig, "09_conflict_resolution_funnel")


def generate_puncture_pressure_figures() -> Dict[str, object]:
    _apply_ieee_style()
    checkpoint = _select_checkpoint()
    PUNCTURE_DIR.mkdir(parents=True, exist_ok=True)

    load_results = [_evaluate_load(load, checkpoint) for load in LOAD_SWEEP]
    all_records = [record for item in load_results for episode in item["episodes"] for record in episode["decision_records"]]
    representative = next(item["representative"] for item in load_results if item["load"] == REPRESENTATIVE_LOAD)

    output_paths = []
    output_paths += _plot_puncture_necessity_vs_decision(all_records)
    output_paths += _plot_embb_damage_vs_decision(all_records)
    output_paths += _plot_decision_boundary(all_records)
    output_paths += _plot_resource_conflict_timeline(representative)
    output_paths += _plot_no_puncture_counterfactual(load_results)
    output_paths += _plot_puncture_efficiency(load_results)
    output_paths += _plot_embb_loss_attribution(load_results)
    output_paths += _plot_overlay_missed_opportunities(load_results)
    output_paths += _plot_conflict_resolution_funnel(load_results)

    note_lines = [
        "# SR-MAPPO Puncture Pressure Figure Notes",
        "",
        f"- Checkpoint: `{checkpoint.name}`",
        f"- Load sweep: {LOAD_SWEEP}",
        f"- Episodes per load: {EPISODES_PER_LOAD}",
        f"- Representative load: {REPRESENTATIVE_LOAD} UE/UAV",
        "- Figures focus on executed SR-MAPPO behavior only.",
        "- No Greedy baseline is plotted in this folder.",
        "- Resource-deficit proxy = available URLLC packets - overlay-feasible candidates at the current decision cell.",
        "- No-puncture counterfactual reuses the same policy and same seed, but rewrites puncture to overlay-if-feasible, else KEEP.",
        "",
        "## Figure Set",
        "- 01 Puncture necessity vs decision",
        "- 02 eMBB damage vs decision",
        "- 03 Decision boundary plot",
        "- 04 Resource conflict timeline",
        "- 05 What-if no puncture counterfactual",
        "- 06 Puncture efficiency",
        "- 07 eMBB loss attribution",
        "- 08 Overlay missed opportunities",
        "- 09 Conflict resolution funnel",
    ]
    (PUNCTURE_DIR / "PUNCTURE_PRESSURE_NOTES.md").write_text("\n".join(note_lines), encoding="utf-8")

    summary = {
        "checkpoint": str(checkpoint),
        "loads": [
            {
                "load": item["load"],
                "mean_embb_rate": _safe_mean([episode["summary"]["embb_total_rate"] for episode in item["episodes"]]),
                "mean_admission": _safe_mean([episode["summary"]["urllc_admission_rate"] for episode in item["episodes"]], 1.0),
                "mean_reliability": _safe_mean([episode["summary"]["urllc_success_rate"] for episode in item["episodes"]], 1.0),
                "mean_puncture_ratio": _safe_mean([episode["summary"]["puncture_ratio"] for episode in item["episodes"]]),
                "mean_overlay_ratio": _safe_mean([episode["summary"]["overlay_ratio"] for episode in item["episodes"]]),
                "mean_throughput_gap_vs_no_puncture_mbps": _safe_mean([
                    (actual["summary"]["embb_total_rate"] - counter["summary"]["embb_total_rate"]) / 1e6
                    for actual, counter in zip(item["episodes"], item["no_puncture_episodes"])
                ]),
            }
            for item in load_results
        ],
        "output_paths": output_paths,
    }
    (PUNCTURE_DIR / "puncture_pressure_data.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = generate_puncture_pressure_figures()
    print(f"SR-MAPPO puncture-pressure figures generated using {result['checkpoint']}")
