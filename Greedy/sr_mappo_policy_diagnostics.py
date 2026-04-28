from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_mappo.compare import _build_main_like_configs, _configure_density_scenario
from sr_mappo.config import SRMAPPOConfig
from sr_mappo.env import SRMAPPOPhaseAEnv
from sr_mappo.networks import SRMAPPOActorCritic
from sr_mappo.types import MODE_KEEP, MODE_NAMES, MODE_OVERLAY, MODE_PUNCTURE, HybridAction


PACKAGE_DIR = ROOT / "sr_mappo"
RESULTS_DIR = PACKAGE_DIR / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOADS = [5.0, 10.0, 15.0, 20.0, 25.0]
EPISODES_PER_LOAD = 6
REPRESENTATIVE_LOAD = 25.0
MODE_LABELS = ["KEEP", "OVERLAY", "PUNCTURE"]
MODE_COLORS = {MODE_KEEP: "#6c757d", MODE_OVERLAY: "#2a9d8f", MODE_PUNCTURE: "#e76f51"}
ANCHOR_LAYOUTS = ["greedy", "banded_diversity", "overlay_oracle"]
ANCHOR_LAYOUT_LABELS = {
    "greedy": "Greedy anchors",
    "banded_diversity": "Banded/diversity anchors",
    "overlay_oracle": "Overlay-friendly oracle anchors",
}
OVERLAY_CAUSE_KEYS = [
    "cause_urllc_sinr_unachievable",
    "cause_embb_retention_below_threshold",
    "cause_required_power_exceeds_budget",
    "cause_rb_minislot_collision",
    "cause_packet_already_scheduled_elsewhere",
    "cause_cross_uav_interference_too_high",
    "cause_deadline_or_release_violation",
    "cause_other_structural_reason",
]
OVERLAY_CAUSE_LABELS = [
    "URLLC SINR not achievable",
    "eMBB retention too low",
    "Power exceeds budget",
    "RB/minislot collision",
    "Packet already scheduled",
    "Cross-UAV interference too high",
    "Deadline/release violation",
    "Other structural reason",
]
PRIMARY_CAUSE_PRIORITY = [
    "cause_required_power_exceeds_budget",
    "cause_cross_uav_interference_too_high",
    "cause_embb_retention_below_threshold",
    "cause_urllc_sinr_unachievable",
    "cause_rb_minislot_collision",
    "cause_packet_already_scheduled_elsewhere",
    "cause_deadline_or_release_violation",
    "cause_other_structural_reason",
]
SHIELD_CAUSE_KEYS = [
    "raw_mode_invalid",
    "raw_packet_invalid",
    "raw_mode_infeasible_for_packet",
    "raw_power_clipped",
    "packet_reused_collision",
    "post_shield_keep_from_fallback",
    "post_shield_keep_after_collision",
    "joint_reliability_rewrite",
]
SHIELD_CAUSE_LABELS = [
    "Raw mode invalid",
    "Raw packet invalid",
    "Raw mode infeasible for selected packet",
    "Raw power clipped",
    "Packet reused / collision",
    "Post-shield KEEP from fallback",
    "Post-shield KEEP after collision",
    "Joint reliability rewrite",
]


def _diag_log(message: str) -> None:
    timestamp = np.datetime64("now")
    print(f"[{timestamp}] [SR-MAPPO][DIAG] {message}", flush=True)


def _safe_mean(values, default=0.0):
    if not values:
        return float(default)
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(float(den)) <= 1e-12:
        return float(default)
    return float(num / den)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    denom = arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)


def _candidate_cause_flags(candidate) -> np.ndarray:
    return np.asarray([bool(getattr(candidate, key, False)) for key in OVERLAY_CAUSE_KEYS], dtype=float)


def _primary_cause_index(candidate) -> int:
    for key in PRIMARY_CAUSE_PRIORITY:
        if bool(getattr(candidate, key, False)):
            return OVERLAY_CAUSE_KEYS.index(key)
    return OVERLAY_CAUSE_KEYS.index("cause_other_structural_reason")


def _classify_shield_causes(raw: HybridAction, pre, post, obs) -> np.ndarray:
    causes = np.zeros(len(SHIELD_CAUSE_KEYS), dtype=float)
    mode_in_range = 0 <= int(raw.mode) < obs.masks.mode_mask.size
    mode_valid = mode_in_range and obs.masks.mode_mask[int(raw.mode)] > 0
    packet_in_range = 0 <= int(raw.packet_option) < obs.masks.packet_mask.shape[-1]
    packet_valid = (
        mode_in_range
        and packet_in_range
        and obs.masks.packet_mask[int(raw.mode), int(raw.packet_option)] > 0
    )

    if not mode_valid:
        causes[0] = 1.0
    elif int(raw.mode) != MODE_KEEP and (int(raw.packet_option) == 0 or not packet_valid):
        causes[1] = 1.0
    elif (
        mode_valid and int(raw.mode) != MODE_KEEP and packet_valid and int(raw.packet_option) > 0
        and int(raw.packet_option) - 1 < len(obs.candidates)
        and not obs.candidates[int(raw.packet_option) - 1].is_mode_feasible(int(raw.mode))
    ):
        causes[2] = 1.0

    if abs(float(raw.power_delta) - float(np.clip(raw.power_delta, -1.0, 1.0))) > 1e-8:
        causes[3] = 1.0
    if getattr(post, "collision_rewritten", False):
        causes[4] = 1.0
    if (
        getattr(pre, "used_greedy_fallback", False)
        and pre.candidate is None
        and pre.action.mode == MODE_KEEP
        and int(raw.mode) != MODE_KEEP
    ):
        causes[5] = 1.0
    if getattr(post, "collision_rewritten", False) and post.candidate is None and post.action.mode == MODE_KEEP:
        causes[6] = 1.0
    if getattr(post, "joint_reliability_rewritten", False):
        causes[7] = 1.0
    return causes


def _load_key(load: float) -> str:
    return f"{float(load):g}"


def _select_checkpoint() -> Path:
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(f"No SR-MAPPO checkpoint found in {CHECKPOINT_DIR}")


def _load_history() -> List[Dict]:
    checkpoint = CHECKPOINT_DIR / "sr_mappo_phase_a_final.pt"
    if not checkpoint.exists():
        return []
    payload = torch.load(checkpoint, map_location="cpu")
    return (payload.get("extra", {}) or {}).get("history", []) or []


def _embb_power_limit(env: SRMAPPOPhaseAEnv, embb_idx: int) -> float:
    power_limit_idx = min(embb_idx, len(env.embb_cfg.power_limits) - 1)
    return min(
        env.allocator._dbm_to_watts(env.embb_cfg.power_limits[power_limit_idx]),
        env.algo_cfg.power_upper_bound,
    )


def _build_anchor_state(env: SRMAPPOPhaseAEnv, layout: str) -> Dict:
    if layout == "greedy":
        return {
            "rb_allocation": env.embb_result["rb_allocation"].copy(),
            "alpha_e": env.embb_result["alpha_e"].copy(),
            "owner_per_uav_rb": env.embb_result["owner_per_uav_rb"].copy(),
            "best_uav_per_user": env.embb_result["best_uav_per_user"].copy(),
            "base_rb_rates": env.embb_result["base_rb_rates"].copy(),
            "user_tx_powers": env.embb_result["user_tx_powers"].copy(),
            "power_allocation": env.embb_result["power_allocation"].copy(),
            "rates": env.embb_result["rates"].copy(),
            "owner_per_rb": env.embb_result["owner_per_rb"].copy(),
        }

    num_embb = env.sys_cfg.num_embb_users
    num_uavs = env.sys_cfg.num_uavs
    num_rbs = env.sys_cfg.num_subcarriers
    num_urllc = env.sys_cfg.num_urllc_users
    associated = env.embb_selected_uavs.copy()
    channel_gains = env.channel_gains_mag_sq

    embb_rb_alloc = np.zeros((num_embb, num_rbs), dtype=int)
    alpha_e = np.zeros((num_embb, num_uavs, num_rbs), dtype=int)
    owner_per_uav_rb = np.full((num_uavs, num_rbs), -1, dtype=int)
    assigned_counts = np.zeros(num_embb, dtype=int)

    users_per_uav = {
        uav_idx: np.where(associated == uav_idx)[0]
        for uav_idx in range(num_uavs)
    }
    average_rbs_per_user = max(1.0, num_rbs / max(num_embb, 1))
    packet_sources = env.packet_sources if env.packet_sources.size > 0 else np.arange(num_urllc, dtype=int)

    for uav_idx in range(num_uavs):
        users = users_per_uav[uav_idx]
        if users.size == 0:
            continue

        avg_gains = {
            embb_idx: float(
                np.mean(channel_gains[num_urllc + embb_idx, uav_idx, :])
            )
            for embb_idx in users
        }
        sorted_users = sorted(users.tolist(), key=lambda idx: avg_gains[idx], reverse=True)

        for rb_idx in range(num_rbs):
            best_user = int(sorted_users[0])
            best_score = -np.inf
            for rank, embb_idx in enumerate(sorted_users):
                user_idx = num_urllc + embb_idx
                channel_gain_sq = channel_gains[user_idx, uav_idx, rb_idx]
                if channel_gain_sq <= 1e-15:
                    continue

                tentative_count = assigned_counts[embb_idx] + 1
                per_rb_power = _embb_power_limit(env, embb_idx) / max(tentative_count, 1)
                base_snir = per_rb_power * channel_gain_sq / max(env.sys_cfg.noise_power, 1e-15)
                base_rate = env.capacity_model.shannon_capacity(base_snir, env.sys_cfg.subcarrier_bw) / 1e6

                if layout == "banded_diversity":
                    score = base_rate / (1.0 + 0.75 * assigned_counts[embb_idx])
                    score += 0.05 * (len(sorted_users) - rank)
                elif layout == "overlay_oracle":
                    overlay_count = 0.0
                    overlay_margin = 0.0
                    embb_gain = channel_gain_sq
                    for source_user in packet_sources:
                        urllc_gain = channel_gains[int(source_user), uav_idx, rb_idx]
                        if urllc_gain <= 1e-15:
                            continue
                        gain_ratio = urllc_gain / max(embb_gain, 1e-12)
                        if gain_ratio < env.algo_cfg.min_noma_gain_ratio:
                            continue
                        local_noma_interference = per_rb_power * embb_gain
                        packet_bits = env._packet_bits_for_user(int(source_user))
                        max_power_w = min(
                            env.allocator._dbm_to_watts(
                                env.urllc_cfg.power_limits[min(int(source_user), len(env.urllc_cfg.power_limits) - 1)]
                            ),
                            env.algo_cfg.power_upper_bound,
                        )
                        overlay_power = env.allocator._bisection_search_urllc_power(
                            urllc_gain,
                            packet_bits,
                            env.urllc_cfg.target_error_probability,
                            max_power_w,
                            env.sys_cfg.channel_uses_per_minislot,
                            interference_power=local_noma_interference,
                        )
                        overlay_snir = overlay_power * urllc_gain / max(
                            env.sys_cfg.noise_power + local_noma_interference,
                            1e-15,
                        )
                        overlay_error = env.capacity_model.decoding_error_probability(
                            overlay_snir,
                            packet_bits,
                            env.sys_cfg.channel_uses_per_minislot,
                        )
                        post_sic_snir = per_rb_power * embb_gain / max(
                            env.sys_cfg.noise_power + env.algo_cfg.sic_residual_factor * overlay_power * urllc_gain,
                            1e-15,
                        )
                        min_post_sic = 10 ** (env.algo_cfg.embb_min_sic_snir_db / 10.0)
                        if overlay_error <= env.urllc_cfg.target_error_probability and post_sic_snir >= min_post_sic:
                            overlay_count += 1.0
                            overlay_margin += max(gain_ratio - env.algo_cfg.min_noma_gain_ratio, 0.0)
                    score = 4.0 * overlay_count + 0.25 * overlay_margin + 0.35 * base_rate
                    score -= 0.15 * assigned_counts[embb_idx]
                else:
                    raise ValueError(f"Unknown anchor layout: {layout}")

                if score > best_score:
                    best_score = score
                    best_user = int(embb_idx)

            embb_rb_alloc[best_user, rb_idx] = 1
            alpha_e[best_user, uav_idx, rb_idx] = 1
            owner_per_uav_rb[uav_idx, rb_idx] = best_user
            assigned_counts[best_user] += 1

    user_tx_powers = np.zeros(num_embb, dtype=float)
    for embb_idx in range(num_embb):
        quota = int(np.sum(embb_rb_alloc[embb_idx, :]))
        load_fraction = quota / max(num_rbs, 1)
        user_tx_powers[embb_idx] = _embb_power_limit(env, embb_idx) * load_fraction

    env.allocator.embb_owner_per_uav_rb = owner_per_uav_rb.copy()
    env.allocator.embb_selected_uavs = associated.copy()
    env.allocator.alpha_e_allocation = alpha_e.copy()
    env.allocator.rb_allocation = embb_rb_alloc.copy()
    env.allocator.embb_user_tx_power = user_tx_powers.copy()

    baseline = env.allocator._compute_embb_state(
        embb_rb_alloc,
        channel_gains,
        associated,
        user_tx_powers,
    )

    return {
        "rb_allocation": embb_rb_alloc,
        "alpha_e": alpha_e,
        "owner_per_uav_rb": owner_per_uav_rb,
        "best_uav_per_user": associated.copy(),
        "base_rb_rates": baseline["base_rb_rates"].copy(),
        "base_rb_rates_per_uav_rb": baseline["base_rb_rates_per_uav_rb"].copy(),
        "user_tx_powers": user_tx_powers.copy(),
        "power_allocation": baseline["power_allocation"].copy(),
        "rates": baseline["rates"].copy(),
        "owner_per_rb": baseline["owner_per_rb"].copy(),
    }


def _apply_anchor_layout(env: SRMAPPOPhaseAEnv, layout: str) -> None:
    state = _build_anchor_state(env, layout)
    env.embb_result = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in state.items()}
    env.owner_per_uav_rb = state["owner_per_uav_rb"].copy()
    env.embb_selected_uavs = state["best_uav_per_user"].copy()
    env.embb_base_rb_rates = state["base_rb_rates"].copy()
    env.embb_base_rb_rates_per_uav_rb = state["base_rb_rates_per_uav_rb"].copy()

    env.allocator.embb_owner_per_uav_rb = state["owner_per_uav_rb"].copy()
    env.allocator.embb_selected_uavs = state["best_uav_per_user"].copy()
    env.allocator.alpha_e_allocation = state["alpha_e"].copy()
    env.allocator.rb_allocation = state["rb_allocation"].copy()
    env.allocator.embb_base_rb_rates = state["base_rb_rates"].copy()
    env.allocator.embb_base_rb_rates_per_uav_rb = state["base_rb_rates_per_uav_rb"].copy()
    env.allocator.embb_user_tx_power = state["user_tx_powers"].copy()
    env.allocator.embb_power_allocation = state["power_allocation"].copy()
    env.allocator.embb_owner_per_rb = state["owner_per_rb"].copy()


def _policy_actions(env, model, observations, actor_hidden, critic_hidden):
    device = model.power_log_std.device
    local_obs = torch.from_numpy(np.stack([observations[a].local_obs for a in env.agent_ids]).astype(np.float32)).to(device)
    global_obs = torch.from_numpy(np.stack([observations[a].global_obs for a in env.agent_ids]).astype(np.float32)).to(device)
    mode_mask = torch.from_numpy(np.stack([observations[a].masks.mode_mask for a in env.agent_ids]).astype(np.float32)).to(device)
    packet_mask = torch.from_numpy(np.stack([observations[a].masks.packet_mask for a in env.agent_ids]).astype(np.float32)).to(device)
    embb_owner_mask = torch.from_numpy(np.stack([observations[a].masks.embb_owner_mask for a in env.agent_ids]).astype(np.float32)).to(device)
    output = model.act(
        local_obs=local_obs,
        global_obs=global_obs,
        mode_mask=mode_mask,
        packet_mask=packet_mask,
        embb_owner_mask=embb_owner_mask,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        deterministic=True,
    )
    joint = {}
    for idx, agent_id in enumerate(env.agent_ids):
        joint[agent_id] = HybridAction(
            mode=int(output.mode[idx].item()),
            packet_option=int(output.packet_option[idx].item()),
            power_delta=float(output.power_delta[idx].item()),
            embb_owner_option=int(output.embb_owner_option[idx].item()),
            embb_power_delta=float(output.embb_power_delta[idx].item()),
        )
    return joint, output, output.actor_hidden.detach(), output.critic_hidden.detach()


def _build_env_for_load(load: float) -> Tuple[SRMAPPOPhaseAEnv, SRMAPPOConfig]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )
    cfg = SRMAPPOConfig()
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)
    return env, cfg


def _collect_episode(
    env: SRMAPPOPhaseAEnv,
    model: SRMAPPOActorCritic,
    seed: int,
    collect_trace: bool = False,
    anchor_layout: str = "greedy",
) -> Dict:
    observations, _ = env.reset(seed=seed)
    if anchor_layout != "greedy":
        _apply_anchor_layout(env, anchor_layout)
        observations = env._build_observations()
    actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids), device=model.power_log_std.device)
    diag = {
        "conditional_policy_counts": np.zeros((2, 3), dtype=float),
        "conditional_final_counts": np.zeros((2, 3), dtype=float),
        "shield_confusion": np.zeros((3, 3), dtype=float),
        "raw_mode_counts": np.zeros(3, dtype=float),
        "shield_corrections_by_raw_mode": np.zeros(3, dtype=float),
        "overlay_utilization": [],
        "counterfactual_reward_diff": [],
        "puncture_loss_overlay_available": [],
        "puncture_loss_no_overlay": [],
        "mode_entropy": [],
        "packet_entropy": [],
        "step_candidate_pairs": [],
        "step_feasible_overlay_pairs": [],
        "step_selected_overlay": [],
        "step_selected_puncture": [],
        "step_selected_keep": [],
        "infeasible_overlay_total": 0.0,
        "infeasible_cause_counts": np.zeros(len(OVERLAY_CAUSE_KEYS), dtype=float),
        "primary_cause_counts": np.zeros(len(OVERLAY_CAUSE_KEYS), dtype=float),
        "per_uav_cause_counts": np.zeros((env.sys_cfg.num_uavs, len(OVERLAY_CAUSE_KEYS)), dtype=float),
        "per_uav_primary_cause_counts": np.zeros((env.sys_cfg.num_uavs, len(OVERLAY_CAUSE_KEYS)), dtype=float),
        "shield_cause_counts": np.zeros(len(SHIELD_CAUSE_KEYS), dtype=float),
        "shield_cause_counts_by_raw_mode": np.zeros((3, len(SHIELD_CAUSE_KEYS)), dtype=float),
        "shield_cause_counts_per_uav": np.zeros((env.sys_cfg.num_uavs, len(SHIELD_CAUSE_KEYS)), dtype=float),
        "advantage_by_condition_mode": [[[] for _ in range(3)] for _ in range(2)],
        "advantage_by_shield_mode": [[[] for _ in range(3)] for _ in range(2)],
        "entropy_by_condition": {
            "mode": [[], []],
            "packet": [[], []],
            "power": [[], []],
        },
        "overlay_gate_stats": {
            "total_candidates": 0,
            "overlay_feasible_total": 0,
            "base_rate_sum": 0.0,
            "post_sic_rate_sum": 0.0,
            "gain_ratio_sum": 0.0,
            "gain_ratio_min": None,
            "gain_ratio_max": None,
            "gain_ratio_ok": 0,
            "gain_ratio_fail": 0,
            "base_signal_db_sum": 0.0,
            "base_signal_db_min": None,
            "base_signal_db_max": None,
            "base_intercell_db_sum": 0.0,
            "base_intercell_db_min": None,
            "base_intercell_db_max": None,
            "base_snir_from_powers_db_sum": 0.0,
            "base_snir_from_powers_db_min": None,
            "base_snir_from_powers_db_max": None,
            "retention_sum": 0.0,
            "retention_min": None,
            "retention_max": None,
            "retention_fail": 0,
            "retained_ratio_sum": 0.0,
            "retained_ratio_min": None,
            "retained_ratio_max": None,
            "retained_ratio_fail": 0,
            "throughput_positive_fail": 0,
            "snir_fail": 0,
            "post_sic_snir_db_sum": 0.0,
            "post_sic_snir_db_min": None,
            "post_sic_snir_db_max": None,
            "post_sic_snir_db_fail": 0,
            "urllc_snir_db_sum": 0.0,
            "urllc_snir_db_min": None,
            "urllc_snir_db_max": None,
            "base_embb_snir_db_sum": 0.0,
            "base_embb_snir_db_min": None,
            "base_embb_snir_db_max": None,
            "urllc_error_pass": 0,
            "urllc_error_fail": 0,
            "post_sic_pass": 0,
            "post_sic_fail": 0,
        },
    }
    trace = []
    done = False
    corrected_action_count = 0

    while not done:
        current_obs = observations
        cell_index = env.current_cell_index
        minislot, rb = env._current_cell()
        joint_actions, output, actor_hidden, critic_hidden = _policy_actions(env, model, current_obs, actor_hidden, critic_hidden)
        pre = {aid: env.shield.sanitize_action(joint_actions[aid], current_obs[aid]) for aid in env.agent_ids}
        minislot, rb = env._current_cell()
        post = env._enforce_joint_reliability(minislot, rb, current_obs, pre)

        current_values = output.value.detach().cpu().numpy()
        mode_entropy_arr = output.mode_entropy.detach().cpu().numpy()
        packet_entropy_arr = output.packet_entropy.detach().cpu().numpy()
        total_entropy_arr = output.entropy.detach().cpu().numpy()
        diag["mode_entropy"].append(float(np.mean(mode_entropy_arr)))
        diag["packet_entropy"].append(float(np.mean(packet_entropy_arr)))

        step_candidate = 0
        step_feasible = 0
        step_overlay = 0
        step_puncture = 0
        step_keep = 0
        utility_gaps = []
        feasible_by_agent = {}
        corrected_by_agent = {}
        final_mode_by_agent = {}
        trace_agent_cells = []

        for uav_idx, aid in enumerate(env.agent_ids):
            obs = current_obs[aid]
            raw = joint_actions[aid]
            final = post[aid]
            for candidate in obs.candidates:
                diag["overlay_gate_stats"]["total_candidates"] += 1
                if candidate.overlay_feasible:
                    diag["overlay_gate_stats"]["overlay_feasible_total"] += 1
                post_sic_snir = float(getattr(candidate, "post_sic_snir", 0.0))
                post_sic_snir_db = 10.0 * float(np.log10(max(post_sic_snir, 1e-12)))
                diag["overlay_gate_stats"]["post_sic_snir_db_sum"] += post_sic_snir_db
                diag["overlay_gate_stats"]["post_sic_snir_db_min"] = post_sic_snir_db if diag["overlay_gate_stats"]["post_sic_snir_db_min"] is None else min(diag["overlay_gate_stats"]["post_sic_snir_db_min"], post_sic_snir_db)
                diag["overlay_gate_stats"]["post_sic_snir_db_max"] = post_sic_snir_db if diag["overlay_gate_stats"]["post_sic_snir_db_max"] is None else max(diag["overlay_gate_stats"]["post_sic_snir_db_max"], post_sic_snir_db)
                if post_sic_snir_db < float(env.algo_cfg.embb_min_sic_snir_db):
                    diag["overlay_gate_stats"]["post_sic_snir_db_fail"] += 1
                urllc_snir = float(getattr(candidate, "overlay_urllc_snir", 0.0))
                urllc_snir_db = 10.0 * float(np.log10(max(urllc_snir, 1e-12)))
                diag["overlay_gate_stats"]["urllc_snir_db_sum"] += urllc_snir_db
                diag["overlay_gate_stats"]["urllc_snir_db_min"] = urllc_snir_db if diag["overlay_gate_stats"]["urllc_snir_db_min"] is None else min(diag["overlay_gate_stats"]["urllc_snir_db_min"], urllc_snir_db)
                diag["overlay_gate_stats"]["urllc_snir_db_max"] = urllc_snir_db if diag["overlay_gate_stats"]["urllc_snir_db_max"] is None else max(diag["overlay_gate_stats"]["urllc_snir_db_max"], urllc_snir_db)
                base_embb_snir = float(getattr(candidate, "base_embb_snir", 0.0))
                base_embb_snir_db = 10.0 * float(np.log10(max(base_embb_snir, 1e-12)))
                diag["overlay_gate_stats"]["base_embb_snir_db_sum"] += base_embb_snir_db
                diag["overlay_gate_stats"]["base_embb_snir_db_min"] = base_embb_snir_db if diag["overlay_gate_stats"]["base_embb_snir_db_min"] is None else min(diag["overlay_gate_stats"]["base_embb_snir_db_min"], base_embb_snir_db)
                diag["overlay_gate_stats"]["base_embb_snir_db_max"] = base_embb_snir_db if diag["overlay_gate_stats"]["base_embb_snir_db_max"] is None else max(diag["overlay_gate_stats"]["base_embb_snir_db_max"], base_embb_snir_db)

                gain_ratio = 0.0
                embb_owners = env._overlay_owner_candidates_for_cell(uav_idx)
                if embb_owners:
                    urllc_gain = float(env.channel_gains_mag_sq[int(candidate.source_user), uav_idx, rb])
                    for embb_owner in embb_owners:
                        embb_user_idx = env.sys_cfg.num_urllc_users + int(embb_owner)
                        embb_gain = float(env.channel_gains_mag_sq[embb_user_idx, uav_idx, rb])
                        gain_ratio = max(gain_ratio, urllc_gain / max(embb_gain, 1e-12))
                diag["overlay_gate_stats"]["gain_ratio_sum"] += float(gain_ratio)
                diag["overlay_gate_stats"]["gain_ratio_min"] = gain_ratio if diag["overlay_gate_stats"]["gain_ratio_min"] is None else min(diag["overlay_gate_stats"]["gain_ratio_min"], gain_ratio)
                diag["overlay_gate_stats"]["gain_ratio_max"] = gain_ratio if diag["overlay_gate_stats"]["gain_ratio_max"] is None else max(diag["overlay_gate_stats"]["gain_ratio_max"], gain_ratio)
                if gain_ratio >= float(env.algo_cfg.min_noma_gain_ratio):
                    diag["overlay_gate_stats"]["gain_ratio_ok"] += 1
                else:
                    diag["overlay_gate_stats"]["gain_ratio_fail"] += 1

                base_rate = env.capacity_model.shannon_capacity(
                    base_embb_snir,
                    env.sys_cfg.subcarrier_bw,
                ) / max(env.sys_cfg.num_minislots, 1)
                post_sic_rate = env.capacity_model.shannon_capacity(
                    post_sic_snir,
                    env.sys_cfg.subcarrier_bw,
                ) / max(env.sys_cfg.num_minislots, 1)
                base_signal = float(getattr(candidate, "base_embb_signal_power", 0.0))
                base_intercell = float(getattr(candidate, "base_embb_intercell_power", 0.0))
                base_signal_db = 10.0 * float(np.log10(max(base_signal, 1e-15)))
                base_intercell_db = 10.0 * float(np.log10(max(base_intercell, 1e-15)))
                diag["overlay_gate_stats"]["base_signal_db_sum"] += base_signal_db
                diag["overlay_gate_stats"]["base_intercell_db_sum"] += base_intercell_db
                diag["overlay_gate_stats"]["base_signal_db_min"] = base_signal_db if diag["overlay_gate_stats"]["base_signal_db_min"] is None else min(diag["overlay_gate_stats"]["base_signal_db_min"], base_signal_db)
                diag["overlay_gate_stats"]["base_signal_db_max"] = base_signal_db if diag["overlay_gate_stats"]["base_signal_db_max"] is None else max(diag["overlay_gate_stats"]["base_signal_db_max"], base_signal_db)
                diag["overlay_gate_stats"]["base_intercell_db_min"] = base_intercell_db if diag["overlay_gate_stats"]["base_intercell_db_min"] is None else min(diag["overlay_gate_stats"]["base_intercell_db_min"], base_intercell_db)
                diag["overlay_gate_stats"]["base_intercell_db_max"] = base_intercell_db if diag["overlay_gate_stats"]["base_intercell_db_max"] is None else max(diag["overlay_gate_stats"]["base_intercell_db_max"], base_intercell_db)
                base_snir_from_powers = base_signal / max(env.sys_cfg.noise_power + base_intercell, 1e-15)
                base_snir_from_powers_db = 10.0 * float(np.log10(max(base_snir_from_powers, 1e-12)))
                diag["overlay_gate_stats"]["base_snir_from_powers_db_sum"] += base_snir_from_powers_db
                diag["overlay_gate_stats"]["base_snir_from_powers_db_min"] = base_snir_from_powers_db if diag["overlay_gate_stats"]["base_snir_from_powers_db_min"] is None else min(diag["overlay_gate_stats"]["base_snir_from_powers_db_min"], base_snir_from_powers_db)
                diag["overlay_gate_stats"]["base_snir_from_powers_db_max"] = base_snir_from_powers_db if diag["overlay_gate_stats"]["base_snir_from_powers_db_max"] is None else max(diag["overlay_gate_stats"]["base_snir_from_powers_db_max"], base_snir_from_powers_db)
                diag["overlay_gate_stats"]["base_rate_sum"] += float(base_rate)
                diag["overlay_gate_stats"]["post_sic_rate_sum"] += float(post_sic_rate)
                if base_rate > 0.0:
                    retained_ratio = float(post_sic_rate / base_rate)
                else:
                    retained_ratio = 0.0
                retention = float(min(
                    0.95,
                    max(env.algo_cfg.noma_retention_factor, retained_ratio),
                )) if base_rate > 0.0 else 0.0
                packet_bits = env._packet_bits_for_user(int(candidate.source_user))
                err_prob = env.capacity_model.decoding_error_probability(
                    urllc_snir,
                    packet_bits,
                    env.sys_cfg.channel_uses_per_minislot,
                )
                if err_prob <= env.urllc_cfg.target_error_probability:
                    diag["overlay_gate_stats"]["urllc_error_pass"] += 1
                else:
                    diag["overlay_gate_stats"]["urllc_error_fail"] += 1
                min_post_sic = 10 ** (float(env.algo_cfg.embb_min_sic_snir_db) / 10.0)
                if post_sic_snir >= min_post_sic:
                    diag["overlay_gate_stats"]["post_sic_pass"] += 1
                else:
                    diag["overlay_gate_stats"]["post_sic_fail"] += 1
                diag["overlay_gate_stats"]["retention_sum"] += retention
                diag["overlay_gate_stats"]["retained_ratio_sum"] += retained_ratio
                diag["overlay_gate_stats"]["retention_min"] = retention if diag["overlay_gate_stats"]["retention_min"] is None else min(diag["overlay_gate_stats"]["retention_min"], retention)
                diag["overlay_gate_stats"]["retention_max"] = retention if diag["overlay_gate_stats"]["retention_max"] is None else max(diag["overlay_gate_stats"]["retention_max"], retention)
                diag["overlay_gate_stats"]["retained_ratio_min"] = retained_ratio if diag["overlay_gate_stats"]["retained_ratio_min"] is None else min(diag["overlay_gate_stats"]["retained_ratio_min"], retained_ratio)
                diag["overlay_gate_stats"]["retained_ratio_max"] = retained_ratio if diag["overlay_gate_stats"]["retained_ratio_max"] is None else max(diag["overlay_gate_stats"]["retained_ratio_max"], retained_ratio)
                if retention < env.rl_cfg.env.min_overlay_retention:
                    diag["overlay_gate_stats"]["retention_fail"] += 1
                if retained_ratio < env.rl_cfg.env.min_overlay_retained_rate_ratio:
                    diag["overlay_gate_stats"]["retained_ratio_fail"] += 1
                puncture_retained_rate = max(base_rate - float(candidate.puncture_loss), 0.0)
                overlay_retained_rate = float(base_rate * retention)
                if overlay_retained_rate <= puncture_retained_rate:
                    diag["overlay_gate_stats"]["throughput_positive_fail"] += 1
                if bool(getattr(candidate, "cause_urllc_sinr_unachievable", False)):
                    diag["overlay_gate_stats"]["snir_fail"] += 1
                if not candidate.overlay_feasible:
                    flags = _candidate_cause_flags(candidate)
                    diag["infeasible_overlay_total"] += 1
                    diag["infeasible_cause_counts"] += flags
                    diag["per_uav_cause_counts"][uav_idx] += flags
                    primary_idx = _primary_cause_index(candidate)
                    diag["primary_cause_counts"][primary_idx] += 1
                    diag["per_uav_primary_cause_counts"][uav_idx, primary_idx] += 1
            feasible_overlay_count = int(sum(int(c.overlay_feasible) for c in obs.candidates))
            step_candidate += int(len(obs.candidates))
            condition = 0 if feasible_overlay_count > 0 else 1
            step_feasible += feasible_overlay_count
            feasible_by_agent[aid] = condition
            diag["conditional_policy_counts"][condition, raw.mode] += 1
            diag["conditional_final_counts"][condition, final.action.mode] += 1
            diag["shield_confusion"][raw.mode, final.action.mode] += 1
            diag["raw_mode_counts"][raw.mode] += 1
            corrected = (
                raw.mode != final.action.mode
                or raw.packet_option != final.action.packet_option
                or abs(raw.power_delta - final.action.power_delta) > 1e-8
            )
            corrected_by_agent[aid] = corrected
            final_mode_by_agent[aid] = int(final.action.mode)
            if corrected:
                diag["shield_corrections_by_raw_mode"][raw.mode] += 1
                corrected_action_count += 1
            shield_causes = _classify_shield_causes(raw, pre[aid], final, obs)
            diag["shield_cause_counts"] += shield_causes
            diag["shield_cause_counts_by_raw_mode"][raw.mode] += shield_causes
            diag["shield_cause_counts_per_uav"][uav_idx] += shield_causes

            if final.candidate is not None and final.action.mode == MODE_OVERLAY:
                step_overlay += 1
            elif final.candidate is not None and final.action.mode == MODE_PUNCTURE:
                step_puncture += 1
                puncture_loss = float(final.candidate.puncture_loss / 1e6)
                if feasible_overlay_count > 0:
                    diag["puncture_loss_overlay_available"].append(puncture_loss)
                else:
                    diag["puncture_loss_no_overlay"].append(puncture_loss)
            else:
                step_keep += 1

            dual = [c for c in obs.candidates if c.overlay_feasible and c.puncture_feasible]
            if dual:
                ref = max(dual, key=lambda c: max(c.overlay_utility, c.puncture_utility))
                overlay_reward, _, _ = env._counterfactual_local_reward(ref, MODE_OVERLAY, power_delta=0.0)
                puncture_reward, _, _ = env._counterfactual_local_reward(ref, MODE_PUNCTURE, power_delta=0.0)
                diag["counterfactual_reward_diff"].append(float(overlay_reward - puncture_reward))
            utility_gaps.append(float(final.utility - obs.greedy_reference_utility))
            if collect_trace:
                trace_agent_cells.append({
                    "uav_idx": int(uav_idx),
                    "raw_mode": int(raw.mode),
                    "final_mode": int(final.action.mode),
                    "packet_id": int(final.candidate.packet_id) if final.candidate is not None else -1,
                    "source_user": int(final.candidate.source_user) if final.candidate is not None else -1,
                    "candidate_count_cell": int(len(obs.candidates)),
                    "feasible_overlay_pairs_cell": int(feasible_overlay_count),
                    "corrected": bool(corrected),
                    "collision_rewritten": bool(getattr(final, "collision_rewritten", False)),
                    "used_greedy_fallback": bool(getattr(final, "used_greedy_fallback", False)),
                })

        if step_feasible > 0:
            diag["overlay_utilization"].append(_safe_div(step_overlay, step_feasible, default=0.0))
        diag["step_candidate_pairs"].append(step_candidate)
        diag["step_feasible_overlay_pairs"].append(step_feasible)
        diag["step_selected_overlay"].append(step_overlay)
        diag["step_selected_puncture"].append(step_puncture)
        diag["step_selected_keep"].append(step_keep)

        observations, rewards, dones, _infos = env.step(joint_actions)
        done = all(dones.values())
        team_reward = float(next(iter(rewards.values()))) if rewards else 0.0
        if not done:
            device = model.power_log_std.device
            next_local_obs = torch.from_numpy(np.stack([observations[a].local_obs for a in env.agent_ids]).astype(np.float32)).to(device)
            next_global_obs = torch.from_numpy(np.stack([observations[a].global_obs for a in env.agent_ids]).astype(np.float32)).to(device)
            next_mode_mask = torch.from_numpy(np.stack([observations[a].masks.mode_mask for a in env.agent_ids]).astype(np.float32)).to(device)
            next_packet_mask = torch.from_numpy(np.stack([observations[a].masks.packet_mask for a in env.agent_ids]).astype(np.float32)).to(device)
            with torch.no_grad():
                bootstrap = model.act(
                    local_obs=next_local_obs,
                    global_obs=next_global_obs,
                    mode_mask=next_mode_mask,
                    packet_mask=next_packet_mask,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                    deterministic=True,
                )
            next_values = bootstrap.value.detach().cpu().numpy()
        else:
            next_values = np.zeros_like(current_values)

        gamma = float(getattr(env.rl_cfg.training, "gamma", 0.99))
        for idx, aid in enumerate(env.agent_ids):
            td_advantage = team_reward + (0.0 if done else gamma * float(next_values[idx])) - float(current_values[idx])
            cond_idx = feasible_by_agent[aid]
            shield_idx = 1 if corrected_by_agent[aid] else 0
            mode_idx = final_mode_by_agent[aid]
            diag["advantage_by_condition_mode"][cond_idx][mode_idx].append(td_advantage)
            diag["advantage_by_shield_mode"][shield_idx][mode_idx].append(td_advantage)
            power_entropy = float(total_entropy_arr[idx] - mode_entropy_arr[idx] - packet_entropy_arr[idx])
            diag["entropy_by_condition"]["mode"][cond_idx].append(float(mode_entropy_arr[idx]))
            diag["entropy_by_condition"]["packet"][cond_idx].append(float(packet_entropy_arr[idx]))
            diag["entropy_by_condition"]["power"][cond_idx].append(power_entropy)

        if collect_trace:
            summary = env.summarize_episode()
            trace.append({
                "cell_index": int(cell_index),
                "minislot": int(minislot),
                "rb": int(rb),
                "arrivals": float(env.packet_arrivals_by_minislot[minislot] if env.packet_arrivals_by_minislot.size > minislot else 0.0),
                "scheduled_packets": float(summary["scheduled_packets"]),
                "feasible_overlay_pairs": float(step_feasible),
                "selected_overlay": float(step_overlay),
                "selected_puncture": float(step_puncture),
                "selected_keep": float(step_keep),
                "embb_rate": float(summary["embb_total_rate"]),
                "total_power": float(summary["total_power"]),
                "utility_gap": float(np.mean(utility_gaps)) if utility_gaps else 0.0,
                "agent_cells": trace_agent_cells,
            })

    episode_summary = env.summarize_episode()
    return {
        "diagnostics": diag,
        "trace": trace,
        "mode_grid": env.mode_grid.copy(),
        "packet_grid": env.packet_grid.copy(),
        "packet_sources": env.packet_sources.copy(),
        "owner_per_uav_rb": env.owner_per_uav_rb.copy(),
        "scheduled_uavs": env.scheduled_uavs.copy(),
        "anchor_layout": anchor_layout,
        "episode_summary": episode_summary,
        "empty_rb_fraction": float(np.mean(env.owner_per_uav_rb < 0)) if env.owner_per_uav_rb is not None else 1.0,
        "corrected_action_count": int(corrected_action_count),
    }


def _aggregate(diag_list: List[Dict]) -> Dict:
    policy = np.sum([d["conditional_policy_counts"] for d in diag_list], axis=0)
    final = np.sum([d["conditional_final_counts"] for d in diag_list], axis=0)
    confusion = np.sum([d["shield_confusion"] for d in diag_list], axis=0)
    raw_counts = np.sum([d["raw_mode_counts"] for d in diag_list], axis=0)
    corrections = np.sum([d["shield_corrections_by_raw_mode"] for d in diag_list], axis=0)
    infeasible_total = float(np.sum([d["infeasible_overlay_total"] for d in diag_list]))
    infeasible_cause_counts = np.sum([d["infeasible_cause_counts"] for d in diag_list], axis=0)
    primary_cause_counts = np.sum([d["primary_cause_counts"] for d in diag_list], axis=0)
    per_uav_cause_counts = np.sum([d["per_uav_cause_counts"] for d in diag_list], axis=0)
    per_uav_primary_cause_counts = np.sum([d["per_uav_primary_cause_counts"] for d in diag_list], axis=0)
    shield_cause_counts = np.sum([d["shield_cause_counts"] for d in diag_list], axis=0)
    shield_cause_counts_by_raw_mode = np.sum([d["shield_cause_counts_by_raw_mode"] for d in diag_list], axis=0)
    shield_cause_counts_per_uav = np.sum([d["shield_cause_counts_per_uav"] for d in diag_list], axis=0)
    overlay_utilization = [x for d in diag_list for x in d["overlay_utilization"]]
    candidate_pairs_per_minislot = [x for d in diag_list for x in d["step_candidate_pairs"]]
    feasible_pairs_per_minislot = [x for d in diag_list for x in d["step_feasible_overlay_pairs"]]
    selected_pairs_per_minislot = [x for d in diag_list for x in d["step_selected_overlay"]]
    cf_diff = [x for d in diag_list for x in d["counterfactual_reward_diff"]]
    punct_overlay = [x for d in diag_list for x in d["puncture_loss_overlay_available"]]
    punct_no = [x for d in diag_list for x in d["puncture_loss_no_overlay"]]
    advantage_by_condition_mode = [[[] for _ in range(3)] for _ in range(2)]
    advantage_by_shield_mode = [[[] for _ in range(3)] for _ in range(2)]
    entropy_by_condition = {"mode": [[], []], "packet": [[], []], "power": [[], []]}
    gate_stats = {
        "total_candidates": 0,
        "overlay_feasible_total": 0,
        "base_rate_sum": 0.0,
        "post_sic_rate_sum": 0.0,
        "gain_ratio_sum": 0.0,
        "gain_ratio_min": None,
        "gain_ratio_max": None,
        "gain_ratio_ok": 0,
        "gain_ratio_fail": 0,
        "base_signal_db_sum": 0.0,
        "base_signal_db_min": None,
        "base_signal_db_max": None,
        "base_intercell_db_sum": 0.0,
        "base_intercell_db_min": None,
        "base_intercell_db_max": None,
        "base_snir_from_powers_db_sum": 0.0,
        "base_snir_from_powers_db_min": None,
        "base_snir_from_powers_db_max": None,
        "retention_sum": 0.0,
        "retention_min": None,
        "retention_max": None,
        "retention_fail": 0,
        "retained_ratio_sum": 0.0,
        "retained_ratio_min": None,
        "retained_ratio_max": None,
        "retained_ratio_fail": 0,
        "throughput_positive_fail": 0,
        "snir_fail": 0,
        "post_sic_snir_db_sum": 0.0,
        "post_sic_snir_db_min": None,
        "post_sic_snir_db_max": None,
        "post_sic_snir_db_fail": 0,
        "urllc_snir_db_sum": 0.0,
        "urllc_snir_db_min": None,
        "urllc_snir_db_max": None,
        "base_embb_snir_db_sum": 0.0,
        "base_embb_snir_db_min": None,
        "base_embb_snir_db_max": None,
        "urllc_error_pass": 0,
        "urllc_error_fail": 0,
        "post_sic_pass": 0,
        "post_sic_fail": 0,
    }
    for d in diag_list:
        for cond_idx in range(2):
            for mode_idx in range(3):
                advantage_by_condition_mode[cond_idx][mode_idx].extend(d["advantage_by_condition_mode"][cond_idx][mode_idx])
                advantage_by_shield_mode[cond_idx][mode_idx].extend(d["advantage_by_shield_mode"][cond_idx][mode_idx])
        for head in ["mode", "packet", "power"]:
            entropy_by_condition[head][cond_idx].extend(d["entropy_by_condition"][head][cond_idx])
        stats = d.get("overlay_gate_stats", {})
        gate_stats["total_candidates"] += int(stats.get("total_candidates", 0))
        gate_stats["overlay_feasible_total"] += int(stats.get("overlay_feasible_total", 0))
        gate_stats["base_rate_sum"] += float(stats.get("base_rate_sum", 0.0))
        gate_stats["post_sic_rate_sum"] += float(stats.get("post_sic_rate_sum", 0.0))
        gate_stats["gain_ratio_sum"] += float(stats.get("gain_ratio_sum", 0.0))
        gate_stats["gain_ratio_ok"] += int(stats.get("gain_ratio_ok", 0))
        gate_stats["gain_ratio_fail"] += int(stats.get("gain_ratio_fail", 0))
        gate_stats["base_signal_db_sum"] += float(stats.get("base_signal_db_sum", 0.0))
        gate_stats["base_intercell_db_sum"] += float(stats.get("base_intercell_db_sum", 0.0))
        gate_stats["base_snir_from_powers_db_sum"] += float(stats.get("base_snir_from_powers_db_sum", 0.0))
        gate_stats["retention_sum"] += float(stats.get("retention_sum", 0.0))
        gate_stats["retention_fail"] += int(stats.get("retention_fail", 0))
        gate_stats["retained_ratio_sum"] += float(stats.get("retained_ratio_sum", 0.0))
        gate_stats["retained_ratio_fail"] += int(stats.get("retained_ratio_fail", 0))
        gate_stats["throughput_positive_fail"] += int(stats.get("throughput_positive_fail", 0))
        gate_stats["snir_fail"] += int(stats.get("snir_fail", 0))
        gate_stats["post_sic_snir_db_sum"] += float(stats.get("post_sic_snir_db_sum", 0.0))
        gate_stats["post_sic_snir_db_fail"] += int(stats.get("post_sic_snir_db_fail", 0))
        gate_stats["urllc_snir_db_sum"] += float(stats.get("urllc_snir_db_sum", 0.0))
        gate_stats["base_embb_snir_db_sum"] += float(stats.get("base_embb_snir_db_sum", 0.0))
        gate_stats["urllc_error_pass"] += int(stats.get("urllc_error_pass", 0))
        gate_stats["urllc_error_fail"] += int(stats.get("urllc_error_fail", 0))
        gate_stats["post_sic_pass"] += int(stats.get("post_sic_pass", 0))
        gate_stats["post_sic_fail"] += int(stats.get("post_sic_fail", 0))
        for key in [
            "retention_min",
            "retention_max",
            "retained_ratio_min",
            "retained_ratio_max",
            "post_sic_snir_db_min",
            "post_sic_snir_db_max",
            "urllc_snir_db_min",
            "urllc_snir_db_max",
            "base_embb_snir_db_min",
            "base_embb_snir_db_max",
            "gain_ratio_min",
            "gain_ratio_max",
            "base_signal_db_min",
            "base_signal_db_max",
            "base_intercell_db_min",
            "base_intercell_db_max",
            "base_snir_from_powers_db_min",
            "base_snir_from_powers_db_max",
        ]:
            val = stats.get(key, None)
            if val is None:
                continue
            if gate_stats[key] is None:
                gate_stats[key] = float(val)
            else:
                if "min" in key:
                    gate_stats[key] = min(float(val), gate_stats[key])
                else:
                    gate_stats[key] = max(float(val), gate_stats[key])

    empty_rb_fraction = _safe_mean([d.get("empty_rb_fraction", 0.0) for d in diag_list])
    return {
        "conditional_policy_probs": _normalize_rows(policy),
        "conditional_final_probs": _normalize_rows(final),
        "shield_confusion_probs": _normalize_rows(confusion),
        "correction_ratio_by_raw_mode": np.asarray([_safe_div(corrections[i], raw_counts[i], 0.0) for i in range(3)], dtype=float),
        "overlay_utilization_ratio": _safe_mean(overlay_utilization),
        "empty_rb_fraction": float(empty_rb_fraction),
        "avg_candidate_overlay_pairs": _safe_mean(candidate_pairs_per_minislot),
        "avg_feasible_overlay_pairs": _safe_mean(feasible_pairs_per_minislot),
        "avg_selected_overlay_pairs": _safe_mean(selected_pairs_per_minislot),
        "counterfactual_reward_diff": cf_diff,
        "counterfactual_reward_diff_mean": _safe_mean(cf_diff),
        "puncture_loss_overlay_available": punct_overlay,
        "puncture_loss_no_overlay": punct_no,
        "puncture_loss_overlay_available_mean": _safe_mean(punct_overlay),
        "puncture_loss_no_overlay_mean": _safe_mean(punct_no),
        "infeasible_overlay_total": infeasible_total,
        "infeasible_cause_counts": infeasible_cause_counts,
        "infeasible_cause_fraction": np.asarray([
            _safe_div(infeasible_cause_counts[idx], infeasible_total, 0.0) for idx in range(len(OVERLAY_CAUSE_KEYS))
        ], dtype=float),
        "primary_cause_counts": primary_cause_counts,
        "primary_cause_fraction": np.asarray([
            _safe_div(primary_cause_counts[idx], max(np.sum(primary_cause_counts), 1.0), 0.0) for idx in range(len(OVERLAY_CAUSE_KEYS))
        ], dtype=float),
        "per_uav_cause_counts": per_uav_cause_counts,
        "per_uav_primary_cause_counts": per_uav_primary_cause_counts,
        "per_uav_primary_cause_fraction": _normalize_rows(per_uav_primary_cause_counts),
        "shield_cause_counts": shield_cause_counts,
        "shield_cause_fraction": np.asarray([
            _safe_div(shield_cause_counts[idx], max(np.sum(shield_cause_counts), 1.0), 0.0) for idx in range(len(SHIELD_CAUSE_KEYS))
        ], dtype=float),
        "shield_cause_fraction_by_raw_mode": _normalize_rows(shield_cause_counts_by_raw_mode),
        "shield_cause_fraction_per_uav": _normalize_rows(shield_cause_counts_per_uav),
        "advantage_by_condition_mode": [
            [_summary_stats(advantage_by_condition_mode[cond_idx][mode_idx]) for mode_idx in range(3)]
            for cond_idx in range(2)
        ],
        "advantage_by_shield_mode": [
            [_summary_stats(advantage_by_shield_mode[cond_idx][mode_idx]) for mode_idx in range(3)]
            for cond_idx in range(2)
        ],
        "entropy_by_condition": {
            head: [_summary_stats(entropy_by_condition[head][cond_idx]) for cond_idx in range(2)]
            for head in ["mode", "packet", "power"]
        },
        "overlay_gate_stats": gate_stats,
    }


def _collect_anchor_episode(env: SRMAPPOPhaseAEnv, model: SRMAPPOActorCritic, seed: int, anchor_layout: str) -> Dict:
    observations, _ = env.reset(seed=seed)
    if anchor_layout != "greedy":
        _apply_anchor_layout(env, anchor_layout)
        observations = env._build_observations()

    actor_hidden, critic_hidden = model.initial_state(
        batch_size=len(env.agent_ids),
        device=model.power_log_std.device,
    )
    done = False
    feasible_pairs_per_minislot = []
    overlay_capable_rbs_per_uav = []

    while not done:
        minislot, rb = env._current_cell()
        if rb == 0:
            per_uav_capable = np.zeros(env.sys_cfg.num_uavs, dtype=float)
            feasible_pairs = 0.0
            for uav_idx in range(env.sys_cfg.num_uavs):
                for rb_scan in range(env.sys_cfg.num_subcarriers):
                    candidates = env._evaluate_candidates_for_cell(uav_idx, rb_scan, minislot)
                    feasible_count = float(sum(int(candidate.overlay_feasible) for candidate in candidates))
                    feasible_pairs += feasible_count
                    if feasible_count > 0:
                        per_uav_capable[uav_idx] += 1.0
            feasible_pairs_per_minislot.append(feasible_pairs)
            overlay_capable_rbs_per_uav.append(per_uav_capable)

        joint_actions, _output, actor_hidden, critic_hidden = _policy_actions(
            env, model, observations, actor_hidden, critic_hidden
        )
        observations, _rewards, dones, _infos = env.step(joint_actions)
        done = all(dones.values())

    summary = env.summarize_episode()
    if overlay_capable_rbs_per_uav:
        per_uav_avg = np.mean(np.stack(overlay_capable_rbs_per_uav, axis=0), axis=0)
    else:
        per_uav_avg = np.zeros(env.sys_cfg.num_uavs, dtype=float)

    return {
        "anchor_layout": anchor_layout,
        "embb_total_rate": float(summary["embb_total_rate"]),
        "urllc_admission_rate": float(summary["urllc_admission_rate"]),
        "urllc_success_rate": float(summary["urllc_success_rate"]),
        "overlay_ratio": float(summary["overlay_ratio"]),
        "puncture_ratio": float(summary["puncture_ratio"]),
        "avg_feasible_overlay_pairs": _safe_mean(feasible_pairs_per_minislot),
        "avg_overlay_capable_rbs_per_uav": float(np.mean(per_uav_avg)) if per_uav_avg.size > 0 else 0.0,
        "per_uav_overlay_capable_rbs": per_uav_avg,
    }


def _aggregate_anchor_layout(metrics_by_episode: List[Dict]) -> Dict:
    return {
        "embb_total_rate": _safe_mean([episode["embb_total_rate"] for episode in metrics_by_episode]),
        "urllc_admission_rate": _safe_mean([episode["urllc_admission_rate"] for episode in metrics_by_episode]),
        "urllc_success_rate": _safe_mean([episode["urllc_success_rate"] for episode in metrics_by_episode]),
        "overlay_ratio": _safe_mean([episode["overlay_ratio"] for episode in metrics_by_episode]),
        "puncture_ratio": _safe_mean([episode["puncture_ratio"] for episode in metrics_by_episode]),
        "avg_feasible_overlay_pairs": _safe_mean([episode["avg_feasible_overlay_pairs"] for episode in metrics_by_episode]),
        "avg_overlay_capable_rbs_per_uav": _safe_mean([episode["avg_overlay_capable_rbs_per_uav"] for episode in metrics_by_episode]),
        "per_uav_overlay_capable_rbs": np.mean(
            np.stack([episode["per_uav_overlay_capable_rbs"] for episode in metrics_by_episode], axis=0),
            axis=0,
        ) if metrics_by_episode else np.zeros(3, dtype=float),
    }


def _summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _collect_all(
    loads: List[float] | None = None,
    episodes_per_load: int = EPISODES_PER_LOAD,
    skip_anchor: bool = False,
    skip_trace: bool = False,
) -> Dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = _select_checkpoint()
    model = None
    loads = loads or LOADS
    metrics = {
        "loads": loads,
        "episodes_per_load": episodes_per_load,
        "diagnostics_by_load": {},
        "anchor_diagnostics_by_load": {},
    }
    representative = None
    representative_gallery = {}
    for load in loads:
        env, cfg = _build_env_for_load(load)
        if model is None:
            model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, cfg)
            payload = torch.load(checkpoint, map_location=cfg.training.device)
            model.load_state_dict(payload["model_state_dict"], strict=False)
            model.to(torch.device(cfg.training.device))
            model.eval()
        episodes = [_collect_episode(env, model, seed=3000 + int(load * 10) + ep, collect_trace=False) for ep in range(episodes_per_load)]
        metrics["diagnostics_by_load"][_load_key(load)] = _aggregate([episode["diagnostics"] for episode in episodes])
        if not skip_anchor:
            anchor_layout_stats = {}
            for layout in ANCHOR_LAYOUTS:
                anchor_episodes = [
                    _collect_anchor_episode(env, model, seed=9000 + int(load * 10) * 10 + ep, anchor_layout=layout)
                    for ep in range(episodes_per_load)
                ]
                anchor_layout_stats[layout] = _aggregate_anchor_layout(anchor_episodes)
            metrics["anchor_diagnostics_by_load"][_load_key(load)] = anchor_layout_stats
        if (not skip_trace) and abs(load - REPRESENTATIVE_LOAD) < 1e-9:
            trace_episodes = [
                _collect_episode(env, model, seed=20260329 + ep, collect_trace=True)
                for ep in range(max(4, episodes_per_load))
            ]
            representative = trace_episodes[0]
            representative["aggregated"] = metrics["diagnostics_by_load"][_load_key(load)]
            representative_gallery = {
                "puncture_dominant": max(
                    trace_episodes,
                    key=lambda episode: float(episode["episode_summary"]["puncture_count"]) - 0.2 * float(episode["episode_summary"]["overlay_count"]),
                ),
                "overlay_rich": max(
                    trace_episodes,
                    key=lambda episode: float(episode["episode_summary"]["overlay_count"]),
                ),
                "shield_changed": max(
                    trace_episodes,
                    key=lambda episode: float(episode["corrected_action_count"]),
                ),
                "opportunity_rich": max(
                    trace_episodes,
                    key=lambda episode: float(np.sum([entry["feasible_overlay_pairs"] for entry in episode["trace"]])),
                ),
            }
    metrics["checkpoint"] = str(checkpoint)
    metrics["history"] = _load_history()
    return {"metrics": metrics, "representative": representative, "representative_gallery": representative_gallery}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _pin_xaxis(ax):
    ax.set_xlim(min(LOADS), max(LOADS))
    ax.margins(x=0.0)


def plot_overlay_utilization(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    util = [metrics["diagnostics_by_load"][_load_key(load)]["overlay_utilization_ratio"] for load in LOADS]
    axes[0].plot(LOADS, util, marker="o", color=MODE_COLORS[MODE_OVERLAY])
    _style(axes[0], "Overlay feasibility utilization", "Average UE load per UAV", "selected overlay / feasible overlay")
    _pin_xaxis(axes[0])

    cf_gap = [metrics["diagnostics_by_load"][_load_key(load)]["counterfactual_reward_diff_mean"] for load in LOADS]
    axes[1].plot(LOADS, cf_gap, marker="s", color="#4c78a8")
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    _style(axes[1], "Counterfactual reward gap", "Average UE load per UAV", "overlay reward - puncture reward")
    _pin_xaxis(axes[1])

    path = RESULTS_DIR / "11_overlay_feasibility_utilization.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mode_conditionals(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, cond_idx, title in zip(axes, [0, 1], ["P(mode | feasible overlay exists)", "P(mode | no feasible overlay)"]):
        for mode_id, label in enumerate(MODE_LABELS):
            color = MODE_COLORS[mode_id]
            policy_series = [metrics["diagnostics_by_load"][_load_key(load)]["conditional_policy_probs"][cond_idx, mode_id] for load in LOADS]
            final_series = [metrics["diagnostics_by_load"][_load_key(load)]["conditional_final_probs"][cond_idx, mode_id] for load in LOADS]
            ax.plot(LOADS, policy_series, marker="o", linestyle="--", color=color, alpha=0.55, label=f"Policy {label}")
            ax.plot(LOADS, final_series, marker="s", linestyle="-", color=color, label=f"Shielded {label}")
        _style(ax, title, "Average UE load per UAV", "Probability")
        _pin_xaxis(ax)
    axes[0].legend(fontsize=8, ncol=2)
    path = RESULTS_DIR / "12_mode_decision_conditional_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_shield_analysis(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    rep = bundle["representative"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for mode_id, label in enumerate(MODE_LABELS):
        series = [metrics["diagnostics_by_load"][_load_key(load)]["correction_ratio_by_raw_mode"][mode_id] for load in LOADS]
        axes[0].plot(LOADS, series, marker="o", color=MODE_COLORS[mode_id], label=label)
    _style(axes[0], "Shield correction ratio by raw mode", "Average UE load per UAV", "Correction ratio")
    _pin_xaxis(axes[0])
    axes[0].legend()

    confusion = np.asarray(rep["aggregated"]["shield_confusion_probs"], dtype=float)
    im = axes[1].imshow(confusion, cmap="Blues", vmin=0.0, vmax=max(1e-9, np.max(confusion)))
    axes[1].set_title(f"Shield confusion matrix @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    axes[1].set_xlabel("Post-shield mode")
    axes[1].set_ylabel("Raw policy mode")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(MODE_LABELS)
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels(MODE_LABELS)
    for r in range(3):
        for c in range(3):
            axes[1].text(c, r, f"{confusion[r, c]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    path = RESULTS_DIR / "13_shield_intervention_analysis.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_advantage_entropy(bundle: Dict) -> Path:
    history = bundle["metrics"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    iterations = [record["iteration"] for record in history]
    keep = [record.get("rollout", {}).get("advantage_keep", np.nan) for record in history]
    overlay = [record.get("rollout", {}).get("advantage_overlay", np.nan) for record in history]
    puncture = [record.get("rollout", {}).get("advantage_puncture", np.nan) for record in history]
    axes[0].plot(iterations, keep, marker="o", color=MODE_COLORS[MODE_KEEP], label="KEEP")
    axes[0].plot(iterations, overlay, marker="s", color=MODE_COLORS[MODE_OVERLAY], label="OVERLAY")
    axes[0].plot(iterations, puncture, marker="^", color=MODE_COLORS[MODE_PUNCTURE], label="PUNCTURE")
    _style(axes[0], "Average PPO advantage by mode", "Training iteration", "Advantage")
    axes[0].legend()

    mode_entropy = [record.get("rollout", {}).get("mean_mode_entropy", np.nan) for record in history]
    packet_entropy = [record.get("rollout", {}).get("mean_packet_entropy", np.nan) for record in history]
    axes[1].plot(iterations, mode_entropy, marker="o", label="Mode entropy")
    axes[1].plot(iterations, packet_entropy, marker="s", label="Packet entropy")
    _style(axes[1], "Action entropy evolution", "Training iteration", "Entropy")
    axes[1].legend()

    path = RESULTS_DIR / "14_advantage_entropy_evolution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_counterfactual_and_misuse(bundle: Dict) -> Path:
    rep = bundle["representative"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    diff = np.asarray(rep["aggregated"]["counterfactual_reward_diff"], dtype=float)
    if diff.size > 0:
        axes[0].hist(diff, bins=min(20, max(8, diff.size // 4)), color="#4c78a8", alpha=0.85)
    axes[0].axvline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_title("Reward counterfactual distribution")
    axes[0].set_xlabel("overlay reward - puncture reward")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)

    misuse_yes = np.asarray(rep["aggregated"]["puncture_loss_overlay_available"], dtype=float)
    misuse_no = np.asarray(rep["aggregated"]["puncture_loss_no_overlay"], dtype=float)
    box_data, labels = [], []
    if misuse_yes.size > 0:
        box_data.append(misuse_yes)
        labels.append("Overlay alternative exists")
    if misuse_no.size > 0:
        box_data.append(misuse_no)
        labels.append("No overlay alternative")
    if box_data:
        axes[1].boxplot(box_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#ffb4a2"))
    axes[1].set_title("Puncture misuse analysis")
    axes[1].set_ylabel("Lost eMBB rate (Mbps/action)")
    axes[1].grid(True, axis="y", alpha=0.25)

    path = RESULTS_DIR / "15_counterfactual_and_puncture_misuse.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_single_slot_mode_map(bundle: Dict) -> Path:
    rep = bundle["representative"]
    owner = np.asarray(rep["owner_per_uav_rb"], dtype=int)
    mode_grid = np.asarray(rep["mode_grid"], dtype=int)
    packet_grid = np.asarray(rep["packet_grid"], dtype=int)
    packet_sources = np.asarray(rep["packet_sources"], dtype=int)
    _candidate_counts, _feasible_counts, corrected = _episode_cell_maps(rep)
    fig, axes = plt.subplots(1, owner.shape[0], figsize=(19.5, 6.4), constrained_layout=True)
    if owner.shape[0] == 1:
        axes = [axes]

    cmap = matplotlib.colors.ListedColormap(["#f5f5f5", "#9ec5fe", "#b7e4c7", "#ffb4a2"])
    labels = ["Idle", "KEEP / eMBB only", "Overlay", "Puncture"]
    last_im = None
    for uav_idx, ax in enumerate(axes):
        mode_map = np.zeros((owner.shape[1], mode_grid.shape[2]), dtype=int)
        for rb in range(owner.shape[1]):
            for ms in range(mode_grid.shape[2]):
                if mode_grid[uav_idx, rb, ms] == MODE_OVERLAY:
                    mode_map[rb, ms] = 2
                elif mode_grid[uav_idx, rb, ms] == MODE_PUNCTURE:
                    mode_map[rb, ms] = 3
                elif owner[uav_idx, rb] >= 0:
                    mode_map[rb, ms] = 1
        last_im = ax.imshow(mode_map, aspect="auto", cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"UAV {uav_idx + 1}", pad=12)
        ax.set_xlabel("Minislot")
        ax.set_ylabel("RB")
        ax.set_xticks(range(mode_grid.shape[2]))
        ax.set_yticks(range(owner.shape[1]))
        for rb in range(owner.shape[1]):
            embb_owner = owner[uav_idx, rb]
            for ms in range(mode_grid.shape[2]):
                packet_id = packet_grid[uav_idx, rb, ms]
                if embb_owner >= 0:
                    ax.text(
                        ms,
                        rb - 0.16,
                        f"E{embb_owner + 1}",
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
                    )
                if packet_id >= 0:
                    ax.text(
                        ms,
                        rb + 0.18,
                        f"U{int(packet_sources[packet_id]) + 1}",
                        ha="center",
                        va="center",
                        fontsize=6.0,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.80),
                    )
                if mode_grid[uav_idx, rb, ms] == MODE_OVERLAY:
                    ax.scatter(ms + 0.27, rb - 0.27, marker="o", s=30, c="#1b4332", edgecolors="white", linewidths=0.6)
                elif mode_grid[uav_idx, rb, ms] == MODE_PUNCTURE:
                    ax.scatter(ms + 0.27, rb - 0.27, marker="X", s=32, c="#9d0208", edgecolors="white", linewidths=0.6)
                if corrected[uav_idx, rb, ms]:
                    ax.scatter(ms - 0.28, rb - 0.28, marker="*", s=38, c="#000000", linewidths=0.4)

    legend_handles = [
        Patch(facecolor="#9ec5fe", edgecolor="none", label="KEEP / eMBB only"),
        Patch(facecolor="#b7e4c7", edgecolor="none", label="Overlay"),
        Patch(facecolor="#ffb4a2", edgecolor="none", label="Puncture"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b4332", markeredgecolor="white", label="Overlay pair marker", markersize=8),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#9d0208", markeredgecolor="white", label="Puncture pair marker", markersize=8),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#000000", markeredgecolor="#000000", label="Shield/correction applied", markersize=8),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)

    path = RESULTS_DIR / "16_single_slot_mode_pairing_map.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_overlay_infeasibility_fraction(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for idx, label in enumerate(OVERLAY_CAUSE_LABELS):
        series = [metrics["diagnostics_by_load"][_load_key(load)]["infeasible_cause_fraction"][idx] for load in LOADS]
        ax.plot(LOADS, series, marker="o", label=label)
    _style(ax, "Fraction of infeasible overlay pairs by cause", "Average UE load per UAV", "Fraction of infeasible overlay pairs")
    _pin_xaxis(ax)
    ax.legend(fontsize=8, ncol=2)
    path = RESULTS_DIR / "17_overlay_infeasibility_fraction_vs_load.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_overlay_infeasibility_stacked(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottom = np.zeros(len(LOADS), dtype=float)
    color_cycle = plt.cm.tab20(np.linspace(0, 1, len(OVERLAY_CAUSE_KEYS)))
    for idx, label in enumerate(OVERLAY_CAUSE_LABELS):
        values = np.asarray([metrics["diagnostics_by_load"][_load_key(load)]["primary_cause_fraction"][idx] for load in LOADS], dtype=float)
        ax.bar(LOADS, values, bottom=bottom, width=3.2, label=label, color=color_cycle[idx])
        bottom += values
    _style(ax, "Primary overlay infeasibility cause vs load", "Average UE load per UAV", "Fraction of infeasible overlay pairs")
    _pin_xaxis(ax)
    ax.legend(fontsize=8, ncol=2)
    path = RESULTS_DIR / "18_overlay_infeasibility_stacked_vs_load.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_per_uav_infeasibility_heatmap(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    loads = metrics["loads"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    axes = axes.flatten()
    vmax = 0.0
    matrices = []
    for load in loads:
        matrix = np.asarray(metrics["diagnostics_by_load"][_load_key(load)]["per_uav_primary_cause_fraction"], dtype=float)
        matrices.append(matrix)
        vmax = max(vmax, float(np.max(matrix)))
    for idx, ax in enumerate(axes):
        if idx >= len(loads):
            ax.axis("off")
            continue
        matrix = matrices[idx]
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(vmax, 1e-9))
        ax.set_title(f"Load {loads[idx]:g} UE/UAV")
        ax.set_xlabel("Primary infeasibility cause")
        ax.set_ylabel("UAV")
        ax.set_xticks(range(len(OVERLAY_CAUSE_LABELS)))
        ax.set_xticklabels(OVERLAY_CAUSE_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels([f"UAV {u+1}" for u in range(matrix.shape[0])])
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
    path = RESULTS_DIR / "19_per_uav_overlay_infeasibility_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_anchor_overlay_feasibility(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    for layout in ANCHOR_LAYOUTS:
        feasible = [
            metrics["anchor_diagnostics_by_load"][_load_key(load)][layout]["avg_feasible_overlay_pairs"]
            for load in LOADS
        ]
        capable = [
            metrics["anchor_diagnostics_by_load"][_load_key(load)][layout]["avg_overlay_capable_rbs_per_uav"]
            for load in LOADS
        ]
        axes[0].plot(LOADS, feasible, marker="o", label=ANCHOR_LAYOUT_LABELS[layout])
        axes[1].plot(LOADS, capable, marker="s", label=ANCHOR_LAYOUT_LABELS[layout])

    _style(axes[0], "Overlay-feasible pairs under different eMBB anchors", "Average UE load per UAV", "Avg feasible overlay pairs per minislot")
    _style(axes[1], "Overlay-capable RBs per UAV under different eMBB anchors", "Average UE load per UAV", "Avg overlay-capable RBs / UAV")
    _pin_xaxis(axes[0]); _pin_xaxis(axes[1])
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    path = RESULTS_DIR / "20_anchor_overlay_feasibility_vs_load.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_anchor_policy_performance(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    for layout in ANCHOR_LAYOUTS:
        throughput = [
            metrics["anchor_diagnostics_by_load"][_load_key(load)][layout]["embb_total_rate"] / 1e6
            for load in LOADS
        ]
        axes[0].plot(LOADS, throughput, marker="o", label=ANCHOR_LAYOUT_LABELS[layout])

    greedy_tp = np.asarray([
        metrics["anchor_diagnostics_by_load"][_load_key(load)]["greedy"]["embb_total_rate"]
        for load in LOADS
    ], dtype=float)
    for layout in ANCHOR_LAYOUTS:
        if layout == "greedy":
            continue
        gain = [
            (
                metrics["anchor_diagnostics_by_load"][_load_key(load)][layout]["embb_total_rate"] -
                metrics["anchor_diagnostics_by_load"][_load_key(load)]["greedy"]["embb_total_rate"]
            ) / 1e6
            for load in LOADS
        ]
        axes[1].plot(LOADS, gain, marker="s", label=f"{ANCHOR_LAYOUT_LABELS[layout]} - Greedy")

    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    _style(axes[0], "Same policy under different eMBB anchor layouts", "Average UE load per UAV", "Aggregate eMBB throughput (Mbps)")
    _style(axes[1], "Throughput gain over greedy anchors", "Average UE load per UAV", "Delta throughput (Mbps)")
    _pin_xaxis(axes[0]); _pin_xaxis(axes[1])
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    path = RESULTS_DIR / "21_anchor_policy_performance_vs_load.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_anchor_per_uav_capability(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    representative = metrics["anchor_diagnostics_by_load"][_load_key(REPRESENTATIVE_LOAD)]
    matrix = np.stack(
        [np.asarray(representative[layout]["per_uav_overlay_capable_rbs"], dtype=float) for layout in ANCHOR_LAYOUTS],
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(float(np.max(matrix)), 1e-9))
    ax.set_title(f"Per-UAV overlay-capable RBs @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    ax.set_xlabel("UAV")
    ax.set_ylabel("Anchor layout")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"UAV {idx + 1}" for idx in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([ANCHOR_LAYOUT_LABELS[layout] for layout in ANCHOR_LAYOUTS])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.05, label="Avg overlay-capable RBs")

    path = RESULTS_DIR / "22_anchor_per_uav_overlay_capability.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_shield_correction_breakdown(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    for idx, label in enumerate(SHIELD_CAUSE_LABELS):
        series = [
            metrics["diagnostics_by_load"][_load_key(load)]["shield_cause_fraction"][idx]
            for load in LOADS
        ]
        axes[0].plot(LOADS, series, marker="o", label=label)
    _style(axes[0], "Shield correction breakdown by error type", "Average UE load per UAV", "Fraction of corrections")
    _pin_xaxis(axes[0])
    axes[0].legend(fontsize=7, ncol=2)

    bottom = np.zeros(len(LOADS), dtype=float)
    color_cycle = plt.cm.Set3(np.linspace(0, 1, len(SHIELD_CAUSE_KEYS)))
    for idx, label in enumerate(SHIELD_CAUSE_LABELS):
        values = np.asarray([
            metrics["diagnostics_by_load"][_load_key(load)]["shield_cause_fraction"][idx]
            for load in LOADS
        ], dtype=float)
        axes[1].bar(LOADS, values, bottom=bottom, width=3.2, label=label, color=color_cycle[idx])
        bottom += values
    _style(axes[1], "Stacked correction causes vs load", "Average UE load per UAV", "Fraction of corrections")
    _pin_xaxis(axes[1])
    axes[1].legend(fontsize=7, ncol=2)

    path = RESULTS_DIR / "23_shield_correction_breakdown.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_shield_correction_conditioned(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    representative = metrics["diagnostics_by_load"][_load_key(REPRESENTATIVE_LOAD)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    matrix = np.asarray(representative["shield_cause_fraction_by_raw_mode"], dtype=float)
    im = axes[0].imshow(matrix, cmap="Purples", aspect="auto", vmin=0.0, vmax=max(float(np.max(matrix)), 1e-9))
    axes[0].set_title(f"Correction causes conditioned on raw mode @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    axes[0].set_xlabel("Correction cause")
    axes[0].set_ylabel("Raw mode")
    axes[0].set_xticks(range(len(SHIELD_CAUSE_LABELS)))
    axes[0].set_xticklabels(SHIELD_CAUSE_LABELS, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(MODE_LABELS)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            axes[0].text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    matrix_uav = np.asarray(representative["shield_cause_fraction_per_uav"], dtype=float)
    im2 = axes[1].imshow(matrix_uav, cmap="Oranges", aspect="auto", vmin=0.0, vmax=max(float(np.max(matrix_uav)), 1e-9))
    axes[1].set_title(f"Per-UAV correction cause heatmap @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    axes[1].set_xlabel("Correction cause")
    axes[1].set_ylabel("UAV")
    axes[1].set_xticks(range(len(SHIELD_CAUSE_LABELS)))
    axes[1].set_xticklabels(SHIELD_CAUSE_LABELS, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(matrix_uav.shape[0]))
    axes[1].set_yticklabels([f"UAV {idx + 1}" for idx in range(matrix_uav.shape[0])])
    for r in range(matrix_uav.shape[0]):
        for c in range(matrix_uav.shape[1]):
            axes[1].text(c, r, f"{matrix_uav[r, c]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    path = RESULTS_DIR / "24_shield_correction_conditioned.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_advantage_conditioned(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    x = np.arange(3)
    width = 0.35
    stats = metrics["diagnostics_by_load"][_load_key(REPRESENTATIVE_LOAD)]

    for cond_idx, (label, color, offset) in enumerate([
        ("Feasible overlay exists", "#2a9d8f", -width / 2),
        ("No feasible overlay", "#e76f51", width / 2),
    ]):
        medians = np.asarray([stats["advantage_by_condition_mode"][cond_idx][mode_idx]["median"] for mode_idx in range(3)], dtype=float)
        p25 = np.asarray([stats["advantage_by_condition_mode"][cond_idx][mode_idx]["p25"] for mode_idx in range(3)], dtype=float)
        p75 = np.asarray([stats["advantage_by_condition_mode"][cond_idx][mode_idx]["p75"] for mode_idx in range(3)], dtype=float)
        axes[0].bar(x + offset, medians, width=width, color=color, alpha=0.85, label=label)
        axes[0].errorbar(x + offset, medians, yerr=np.vstack([medians - p25, p75 - medians]), fmt="none", ecolor="black", capsize=4)
    axes[0].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODE_LABELS)
    axes[0].set_title(f"E[Advantage | mode, feasible overlay exists] @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    axes[0].set_ylabel("One-step TD advantage proxy")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    for shield_idx, (label, color, offset) in enumerate([
        ("Not shielded", "#457b9d", -width / 2),
        ("Shielded/corrected", "#e63946", width / 2),
    ]):
        medians = np.asarray([stats["advantage_by_shield_mode"][shield_idx][mode_idx]["median"] for mode_idx in range(3)], dtype=float)
        p25 = np.asarray([stats["advantage_by_shield_mode"][shield_idx][mode_idx]["p25"] for mode_idx in range(3)], dtype=float)
        p75 = np.asarray([stats["advantage_by_shield_mode"][shield_idx][mode_idx]["p75"] for mode_idx in range(3)], dtype=float)
        axes[1].bar(x + offset, medians, width=width, color=color, alpha=0.85, label=label)
        axes[1].errorbar(x + offset, medians, yerr=np.vstack([medians - p25, p75 - medians]), fmt="none", ecolor="black", capsize=4)
    axes[1].axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODE_LABELS)
    axes[1].set_title(f"E[Advantage | mode, shielded or not] @ {REPRESENTATIVE_LOAD:g} UE/UAV")
    axes[1].set_ylabel("One-step TD advantage proxy")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    path = RESULTS_DIR / "25_advantage_conditioned.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_entropy_conditioned(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    head_names = [("mode", "Mode entropy"), ("packet", "Packet entropy"), ("power", "Power entropy")]
    for ax, (head_key, title) in zip(axes, head_names):
        feasible = [
            metrics["diagnostics_by_load"][_load_key(load)]["entropy_by_condition"][head_key][0]["mean"]
            for load in LOADS
        ]
        infeasible = [
            metrics["diagnostics_by_load"][_load_key(load)]["entropy_by_condition"][head_key][1]["mean"]
            for load in LOADS
        ]
        ax.plot(LOADS, feasible, marker="o", color="#2a9d8f", label="Feasible overlay exists")
        ax.plot(LOADS, infeasible, marker="s", color="#e76f51", label="No feasible overlay")
        _style(ax, title, "Average UE load per UAV", "Entropy")
        _pin_xaxis(ax)
    axes[0].legend(fontsize=8)

    path = RESULTS_DIR / "26_entropy_conditioned.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _episode_cell_maps(episode: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    owner = np.asarray(episode["owner_per_uav_rb"], dtype=int)
    num_uavs, num_rbs = owner.shape
    num_minislots = np.asarray(episode["mode_grid"]).shape[2]
    candidate_counts = np.zeros((num_uavs, num_rbs, num_minislots), dtype=int)
    feasible_counts = np.zeros_like(candidate_counts)
    corrected = np.zeros_like(candidate_counts, dtype=bool)
    for entry in episode["trace"]:
        ms = int(entry["minislot"])
        rb = int(entry["rb"])
        for cell in entry["agent_cells"]:
            uav_idx = int(cell["uav_idx"])
            candidate_counts[uav_idx, rb, ms] = int(cell["candidate_count_cell"])
            feasible_counts[uav_idx, rb, ms] = int(cell["feasible_overlay_pairs_cell"])
            corrected[uav_idx, rb, ms] = bool(cell["corrected"])
    return candidate_counts, feasible_counts, corrected


def plot_representative_slot_gallery(bundle: Dict) -> Path:
    gallery = bundle["representative_gallery"]
    if not gallery:
        return RESULTS_DIR / "27_representative_slot_gallery.png"

    labels = [
        ("puncture_dominant", "Puncture-dominant slot"),
        ("overlay_rich", "Overlay-rich slot"),
        ("shield_changed", "Shield-changed slot"),
        ("opportunity_rich", "Opportunity-rich slot"),
    ]
    sample_episode = next(iter(gallery.values()))
    owner_shape = np.asarray(sample_episode["owner_per_uav_rb"]).shape
    fig, axes = plt.subplots(len(labels), owner_shape[0], figsize=(18, 4.5 * len(labels)), constrained_layout=True)
    if len(labels) == 1:
        axes = np.asarray([axes])

    cmap = matplotlib.colors.ListedColormap(["#f5f5f5", "#9ec5fe", "#b7e4c7", "#ffb4a2"])
    for row_idx, (key, title) in enumerate(labels):
        episode = gallery[key]
        owner = np.asarray(episode["owner_per_uav_rb"], dtype=int)
        mode_grid = np.asarray(episode["mode_grid"], dtype=int)
        packet_grid = np.asarray(episode["packet_grid"], dtype=int)
        packet_sources = np.asarray(episode["packet_sources"], dtype=int)
        _candidate_counts, _feasible_counts, corrected = _episode_cell_maps(episode)
        summary = episode["episode_summary"]
        row_title = (
            f"{title} | eMBB {summary['embb_total_rate']/1e6:.2f} Mbps | "
            f"overlay {int(summary['overlay_count'])} | puncture {int(summary['puncture_count'])} | "
            f"corrections {episode['corrected_action_count']}"
        )
        for uav_idx in range(owner.shape[0]):
            ax = axes[row_idx, uav_idx]
            mode_map = np.zeros((owner.shape[1], mode_grid.shape[2]), dtype=int)
            for rb in range(owner.shape[1]):
                for ms in range(mode_grid.shape[2]):
                    if mode_grid[uav_idx, rb, ms] == MODE_OVERLAY:
                        mode_map[rb, ms] = 2
                    elif mode_grid[uav_idx, rb, ms] == MODE_PUNCTURE:
                        mode_map[rb, ms] = 3
                    elif owner[uav_idx, rb] >= 0:
                        mode_map[rb, ms] = 1
            ax.imshow(mode_map, aspect="auto", cmap=cmap, vmin=0, vmax=3)
            if uav_idx == 0:
                ax.set_ylabel(f"{row_title}\nRB")
            ax.set_title(f"UAV {uav_idx + 1}")
            ax.set_xlabel("Minislot")
            ax.set_xticks(range(mode_grid.shape[2]))
            ax.set_yticks(range(owner.shape[1]))
            for rb in range(owner.shape[1]):
                embb_owner = owner[uav_idx, rb]
                for ms in range(mode_grid.shape[2]):
                    packet_id = packet_grid[uav_idx, rb, ms]
                    text_lines = []
                    if embb_owner >= 0:
                        text_lines.append(f"E{embb_owner + 1}")
                    if packet_id >= 0:
                        text_lines.append(f"U{int(packet_sources[packet_id]) + 1}")
                    if text_lines:
                        ax.text(ms, rb, "\n".join(text_lines), ha="center", va="center", fontsize=5.8)
                    if corrected[uav_idx, rb, ms]:
                        ax.scatter(ms + 0.28, rb - 0.28, marker="*", s=36, c="#000000")

    path = RESULTS_DIR / "27_representative_slot_gallery.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_opportunity_rich_slot(bundle: Dict) -> Path:
    episode = bundle["representative_gallery"].get("opportunity_rich")
    if episode is None:
        return RESULTS_DIR / "28_opportunity_rich_slot_overlay_map.png"
    owner = np.asarray(episode["owner_per_uav_rb"], dtype=int)
    mode_grid = np.asarray(episode["mode_grid"], dtype=int)
    candidate_counts, feasible_counts, corrected = _episode_cell_maps(episode)

    fig, axes = plt.subplots(1, owner.shape[0], figsize=(17, 5.6), constrained_layout=True)
    if owner.shape[0] == 1:
        axes = [axes]
    for uav_idx, ax in enumerate(axes):
        selected_overlay = (mode_grid[uav_idx] == MODE_OVERLAY).astype(float)
        heat = feasible_counts[uav_idx].astype(float)
        im = ax.imshow(heat, aspect="auto", cmap="YlGn", vmin=0.0, vmax=max(float(np.max(feasible_counts)), 1e-9))
        ax.set_title(f"UAV {uav_idx + 1}")
        ax.set_xlabel("Minislot")
        ax.set_ylabel("RB")
        ax.set_xticks(range(heat.shape[1]))
        ax.set_yticks(range(heat.shape[0]))
        for rb in range(heat.shape[0]):
            for ms in range(heat.shape[1]):
                ax.text(
                    ms, rb,
                    f"C{candidate_counts[uav_idx, rb, ms]}\nF{feasible_counts[uav_idx, rb, ms]}\nS{int(selected_overlay[rb, ms])}",
                    ha="center", va="center", fontsize=6.1,
                )
                if corrected[uav_idx, rb, ms]:
                    ax.scatter(ms + 0.28, rb - 0.28, marker="*", s=34, c="#000000")
    fig.colorbar(im, ax=axes, fraction=0.03, label="Feasible overlay pairs in cell")
    path = RESULTS_DIR / "28_opportunity_rich_slot_overlay_map.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_json(bundle: Dict) -> Path:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path = RESULTS_DIR / "sr_mappo_policy_diagnostics.json"
    path.write_text(json.dumps(convert(bundle), indent=2), encoding="utf-8")
    return path


def save_diagnostic_markdown(bundle: Dict) -> Path:
    metrics = bundle["metrics"]
    lines = [
        "# Latest SR-MAPPO Policy Diagnostics",
        "",
        "## Run Configuration",
        "",
        f"- Checkpoint: `{metrics['checkpoint']}`",
        f"- Loads: `{', '.join(str(int(load)) if float(load).is_integer() else str(load) for load in metrics['loads'])}` UE/UAV",
        f"- Episodes per load: `{metrics.get('episodes_per_load', EPISODES_PER_LOAD)}`",
        f"- Representative load: `{REPRESENTATIVE_LOAD}` UE/UAV",
        f"- Anchor layouts compared: `{', '.join(ANCHOR_LAYOUTS)}`",
        "",
        "## Key Readouts",
        "",
    ]
    rep_key = _load_key(REPRESENTATIVE_LOAD)
    rep = metrics["diagnostics_by_load"][rep_key]
    lines.extend([
        f"- Avg overlay candidate pairs per minislot: `{rep['avg_candidate_overlay_pairs']:.2f}`",
        f"- Avg overlay feasible pairs per minislot: `{rep['avg_feasible_overlay_pairs']:.2f}`",
        f"- Avg overlay selected per minislot: `{rep['avg_selected_overlay_pairs']:.2f}`",
        f"- Overlay utilization at representative load: `{rep['overlay_utilization_ratio']:.3f}`",
        f"- Empty RB fraction (no eMBB owner): `{rep.get('empty_rb_fraction', 0.0):.3f}`",
        f"- Mean counterfactual reward gap (overlay - puncture): `{rep['counterfactual_reward_diff_mean']:.3f}`",
        f"- Dominant overlay infeasibility cause at representative load: `{OVERLAY_CAUSE_LABELS[int(np.argmax(rep['primary_cause_fraction']))]}`",
        f"- Dominant shield correction cause at representative load: `{SHIELD_CAUSE_LABELS[int(np.argmax(rep['shield_cause_fraction']))]}`",
        "",
    ])
    gate = rep.get("overlay_gate_stats", {})
    if gate:
        total = max(float(gate.get("total_candidates", 0)), 1.0)
        lines.extend([
            "## Overlay Gate Diagnostics (Representative Load)",
            "",
            f"- Candidates analyzed: `{int(gate.get('total_candidates', 0))}`",
            f"- Overlay feasible (pre-gate): `{int(gate.get('overlay_feasible_total', 0))}`",
            f"- Gain ratio mean: `{gate.get('gain_ratio_sum', 0.0) / total:.3f}` (min `{gate.get('gain_ratio_min', 0.0):.3f}`, max `{gate.get('gain_ratio_max', 0.0):.3f}`)",
            f"- Gain ratio >= threshold: `{int(gate.get('gain_ratio_ok', 0))}` / fail `{int(gate.get('gain_ratio_fail', 0))}`",
            f"- Base signal power (dB) mean: `{gate.get('base_signal_db_sum', 0.0) / total:.2f}` (min `{gate.get('base_signal_db_min', 0.0):.2f}`, max `{gate.get('base_signal_db_max', 0.0):.2f}`)",
            f"- Base intercell power (dB) mean: `{gate.get('base_intercell_db_sum', 0.0) / total:.2f}` (min `{gate.get('base_intercell_db_min', 0.0):.2f}`, max `{gate.get('base_intercell_db_max', 0.0):.2f}`)",
            f"- Base SNIR from powers (dB) mean: `{gate.get('base_snir_from_powers_db_sum', 0.0) / total:.2f}` (min `{gate.get('base_snir_from_powers_db_min', 0.0):.2f}`, max `{gate.get('base_snir_from_powers_db_max', 0.0):.2f}`)",
            f"- Retention mean: `{gate.get('retention_sum', 0.0) / total:.3f}` (min `{gate.get('retention_min', 0.0):.3f}`, max `{gate.get('retention_max', 0.0):.3f}`)",
            f"- Retention fails: `{int(gate.get('retention_fail', 0))}`",
            f"- Retained ratio mean: `{gate.get('retained_ratio_sum', 0.0) / total:.3f}` (min `{gate.get('retained_ratio_min', 0.0):.3f}`, max `{gate.get('retained_ratio_max', 0.0):.3f}`)",
            f"- Retained ratio fails: `{int(gate.get('retained_ratio_fail', 0))}`",
            f"- Throughput-positive fails: `{int(gate.get('throughput_positive_fail', 0))}`",
            f"- URLLC SINR fail flags: `{int(gate.get('snir_fail', 0))}`",
            f"- Post-SIC SNIR (proxy, dB) mean: `{gate.get('post_sic_snir_db_sum', 0.0) / total:.2f}` (min `{gate.get('post_sic_snir_db_min', 0.0):.2f}`, max `{gate.get('post_sic_snir_db_max', 0.0):.2f}`)",
            f"- Post-SIC SNIR below threshold: `{int(gate.get('post_sic_snir_db_fail', 0))}`",
            f"- URLLC SNIR (proxy, dB) mean: `{gate.get('urllc_snir_db_sum', 0.0) / total:.2f}` (min `{gate.get('urllc_snir_db_min', 0.0):.2f}`, max `{gate.get('urllc_snir_db_max', 0.0):.2f}`)",
            f"- Base eMBB SNIR (dB) mean: `{gate.get('base_embb_snir_db_sum', 0.0) / total:.2f}` (min `{gate.get('base_embb_snir_db_min', 0.0):.2f}`, max `{gate.get('base_embb_snir_db_max', 0.0):.2f}`)",
            f"- URLLC error pass: `{int(gate.get('urllc_error_pass', 0))}` / fail `{int(gate.get('urllc_error_fail', 0))}`",
            f"- Post-SIC pass: `{int(gate.get('post_sic_pass', 0))}` / fail `{int(gate.get('post_sic_fail', 0))}`",
            "",
        ])
    lines.extend([
        "## Shield Correction Breakdown (Representative Load)",
        "",
    ])
    for idx, label in enumerate(SHIELD_CAUSE_LABELS):
        lines.append(f"- {label}: `{rep['shield_cause_fraction'][idx]:.3f}`")
    lines.extend([
        "",
        "## Generated Figures",
        "",
    ])
    for idx in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
        matches = sorted(RESULTS_DIR.glob(f"{idx:02d}_*.png"))
        for match in matches:
            lines.append(f"- `{match.name}`")
    lines.append("")
    path = RESULTS_DIR / "LATEST_POLICY_DIAGNOSTICS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_policy_diagnostics(
    loads: List[float] | None = None,
    episodes_per_load: int = EPISODES_PER_LOAD,
    skip_anchor: bool = False,
    skip_trace: bool = False,
) -> Dict:
    _diag_log("Starting policy diagnostics.")
    bundle = _collect_all(
        loads=loads,
        episodes_per_load=episodes_per_load,
        skip_anchor=skip_anchor,
        skip_trace=skip_trace,
    )
    _diag_log(f"Checkpoint selected: {bundle['metrics']['checkpoint']}")
    _diag_log(f"Loads: {bundle['metrics']['loads']}")
    _diag_log(f"Episodes per load: {episodes_per_load}")
    rep_key = _load_key(REPRESENTATIVE_LOAD)
    rep = bundle["metrics"]["diagnostics_by_load"][rep_key]
    _diag_log(
        "Overlay summary @ rep load: "
        f"candidate={rep['avg_candidate_overlay_pairs']:.2f}, "
        f"feasible={rep['avg_feasible_overlay_pairs']:.2f}, "
        f"selected={rep['avg_selected_overlay_pairs']:.2f}, "
        f"util={rep['overlay_utilization_ratio']:.3f}"
    )
    outputs = [
        plot_overlay_utilization(bundle),
        plot_mode_conditionals(bundle),
        plot_overlay_infeasibility_fraction(bundle),
        plot_overlay_infeasibility_stacked(bundle),
        plot_per_uav_infeasibility_heatmap(bundle),
        plot_shield_correction_breakdown(bundle),
        plot_shield_correction_conditioned(bundle),
    ]
    if bundle.get("representative"):
        outputs.extend([
            plot_shield_analysis(bundle),
            plot_advantage_entropy(bundle),
            plot_counterfactual_and_misuse(bundle),
            plot_single_slot_mode_map(bundle),
            plot_advantage_conditioned(bundle),
            plot_entropy_conditioned(bundle),
            plot_representative_slot_gallery(bundle),
            plot_opportunity_rich_slot(bundle),
        ])
    if bundle["metrics"].get("anchor_diagnostics_by_load"):
        outputs.extend([
            plot_anchor_overlay_feasibility(bundle),
            plot_anchor_policy_performance(bundle),
            plot_anchor_per_uav_capability(bundle),
        ])
    outputs.extend([
        save_json(bundle),
        save_diagnostic_markdown(bundle),
    ])
    _diag_log("Figures saved.")
    return {
        "checkpoint": bundle["metrics"]["checkpoint"],
        "output_paths": [str(path) for path in outputs[:-2]],
        "metrics_path": str(outputs[-2]),
        "markdown_path": str(outputs[-1]),
    }


def main():
    global LOADS, REPRESENTATIVE_LOAD
    parser = argparse.ArgumentParser(description="SR-MAPPO policy diagnostics")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation before diagnostics.")
    parser.add_argument("--episodes-per-load", type=int, default=EPISODES_PER_LOAD, help="Episodes per load.")
    parser.add_argument("--loads", type=str, default="", help="Comma-separated load list, e.g. 5,10,15.")
    parser.add_argument("--representative-load", type=float, default=REPRESENTATIVE_LOAD, help="Representative load.")
    parser.add_argument("--skip-anchor", action="store_true", help="Skip anchor layout diagnostics.")
    parser.add_argument("--skip-trace", action="store_true", help="Skip representative trace episodes.")
    args = parser.parse_args()
    if args.loads:
        LOADS = [float(item.strip()) for item in args.loads.split(",") if item.strip()]
    REPRESENTATIVE_LOAD = float(args.representative_load)

    if not args.no_report:
        try:
            from sr_mappo.report import generate_report
            generate_report()
        except Exception as exc:
            _diag_log(f"Warning: failed to refresh report figures before diagnostics: {exc}")
    result = generate_policy_diagnostics(
        loads=LOADS,
        episodes_per_load=int(args.episodes_per_load),
        skip_anchor=bool(args.skip_anchor),
        skip_trace=bool(args.skip_trace),
    )
    _diag_log("Generated policy diagnostics:")
    for path in result["output_paths"]:
        _diag_log(path)
    _diag_log(result["metrics_path"])
    _diag_log(result["markdown_path"])


if __name__ == "__main__":
    main()
