from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Dict, List

import numpy as np

from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .experiments import apply_experiment_preset
from .types import HybridAction
from capacity_models import CapacityModels
from config import AlgorithmConfig, SimulationConfig, SystemConfig, URLLCConfig, eMBBConfig


def _build_main_like_configs():
    sys_cfg = SystemConfig()
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    sim_cfg = SimulationConfig()
    return sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _configure_density_scenario(
    avg_users_per_uav: float,
    base_sys: SystemConfig,
    base_urllc: URLLCConfig,
    base_embb: eMBBConfig,
    base_algo: AlgorithmConfig,
    base_sim: SimulationConfig,
):
    sys_cfg = deepcopy(base_sys)
    urllc_cfg = deepcopy(base_urllc)
    embb_cfg = deepcopy(base_embb)
    algo_cfg = deepcopy(base_algo)
    sim_cfg = deepcopy(base_sim)

    total_users = int(round(float(avg_users_per_uav) * int(sys_cfg.num_uavs)))
    total_users = max(total_users, int(sys_cfg.num_uavs))
    urllc_ratio = float(getattr(sim_cfg, "urllc_user_ratio", 0.5) or 0.5)
    num_urllc = int(round(total_users * urllc_ratio))
    num_urllc = int(np.clip(num_urllc, 1, max(total_users - 1, 1)))
    num_embb = max(total_users - num_urllc, 1)
    sys_cfg.num_urllc_users = int(num_urllc)
    sys_cfg.num_embb_users = int(num_embb)
    return sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _probe_action_for_observation(obs, cfg: SRMAPPOConfig) -> HybridAction:
    planning_phase = bool(obs.metadata.get("planning_phase", 0.0))
    if planning_phase:
        owner_mask = np.asarray(obs.masks.embb_owner_mask, dtype=float)
        owner_option = 0
        valid_non_null = np.where(owner_mask[1:] > 0.5)[0]
        if valid_non_null.size > 0:
            owner_option = int(valid_non_null[0] + 1)
        return HybridAction(
            mode=0,
            packet_option=0,
            power_delta=0.0,
            embb_owner_option=owner_option,
            embb_power_delta=0.0,
        )

    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
    packet_option = 0
    mode = 0
    if mode_mask.size > 2 and mode_mask[2] > 0.5:
        mode = 2
        pkt_mask = np.asarray(obs.masks.packet_mask[2], dtype=float)
        valid_pkt = np.where(pkt_mask > 0.5)[0]
        if valid_pkt.size > 0:
            packet_option = int(valid_pkt[0])
    elif mode_mask.size > 1 and mode_mask[1] > 0.5:
        mode = 1
        pkt_mask = np.asarray(obs.masks.packet_mask[1], dtype=float)
        valid_pkt = np.where(pkt_mask > 0.5)[0]
        if valid_pkt.size > 0:
            packet_option = int(valid_pkt[0])
    return HybridAction(
        mode=int(mode),
        packet_option=int(packet_option),
        power_delta=0.0,
        embb_owner_option=0,
        embb_power_delta=0.0,
    )


def _count_owner_choices(mask: np.ndarray, include_null: bool = True) -> int:
    valid = int(np.count_nonzero(np.asarray(mask, dtype=float) > 0.5))
    if include_null:
        return valid
    return max(valid - 1, 0)


def run_audit(experiment: str, load: float, episodes: int, seed: int) -> Dict[str, object]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment)
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)

    phase0_owner_choice_counts: List[int] = []
    phase0_null_only_count = 0
    phase0_owner_active_count = 0
    phase0_steps = 0
    phasea_mode_feasible_counts: List[int] = []
    phasea_packet_choices_overlay: List[int] = []
    phasea_packet_choices_puncture: List[int] = []
    phasea_nonkeep_mode_count = 0
    phasea_steps = 0
    final_vs_snapshot_change_ratios: List[float] = []
    raw_vs_snapshot_change_ratios: List[float] = []
    final_vs_raw_diff_ratios: List[float] = []
    finalize_rewrite_flags = 0
    finalize_minrate_infeasible_flags = 0

    for ep in range(max(int(episodes), 1)):
        observations, _info = env.reset(seed=seed + ep)

        while not bool(getattr(env, "planning_done", True)):
            for agent_id in env.agent_ids:
                obs = observations[agent_id]
                activity = env.action_head_activity(obs)
                if activity["planning_phase"]:
                    phase0_steps += 1
                    owner_choices = _count_owner_choices(
                        np.asarray(obs.masks.embb_owner_mask, dtype=float),
                        include_null=bool(getattr(cfg.action, "include_null_embb_option", True)),
                    )
                    phase0_owner_choice_counts.append(float(owner_choices))
                    if owner_choices <= 1:
                        phase0_null_only_count += 1
                    if activity["owner_active"]:
                        phase0_owner_active_count += 1
            probe_actions = {
                agent_id: _probe_action_for_observation(observations[agent_id], cfg) for agent_id in env.agent_ids
            }
            observations, _rewards, _done, _infos = env.step(probe_actions, prebuilt_observations=observations)

        raw_map = getattr(env, "phase0_raw_owner_per_uav_rb", None)
        final_map = getattr(env, "owner_per_uav_rb", None)
        snapshot_map = getattr(env, "phase0_snapshot_owner_per_uav_rb", None)
        if raw_map is not None and final_map is not None and snapshot_map is not None:
            raw_arr = np.asarray(raw_map, dtype=int)
            final_arr = np.asarray(final_map, dtype=int)
            snap_arr = np.asarray(snapshot_map, dtype=int)
            denom = max(raw_arr.size, 1)
            raw_vs_snapshot_change_ratios.append(float(np.count_nonzero(raw_arr != snap_arr) / denom))
            final_vs_snapshot_change_ratios.append(float(np.count_nonzero(final_arr != snap_arr) / denom))
            final_vs_raw_diff_ratios.append(float(np.count_nonzero(final_arr != raw_arr) / denom))
            if np.any(final_arr != raw_arr):
                finalize_rewrite_flags += 1
        if bool(getattr(env, "phase0_minrate_infeasible_after_finalize", False)):
            finalize_minrate_infeasible_flags += 1

        while not bool(getattr(env, "episode_done", False)):
            for agent_id in env.agent_ids:
                obs = observations[agent_id]
                activity = env.action_head_activity(obs)
                if not activity["planning_phase"]:
                    phasea_steps += 1
                    mode_mask = np.asarray(obs.masks.mode_mask, dtype=float)
                    phasea_mode_feasible_counts.append(float(np.count_nonzero(mode_mask > 0.5)))
                    if mode_mask.size >= 3:
                        if mode_mask[1] > 0.5:
                            phasea_nonkeep_mode_count += 1
                            phasea_packet_choices_overlay.append(
                                float(np.count_nonzero(np.asarray(obs.masks.packet_mask[1], dtype=float) > 0.5))
                            )
                        if mode_mask[2] > 0.5:
                            phasea_nonkeep_mode_count += 1
                            phasea_packet_choices_puncture.append(
                                float(np.count_nonzero(np.asarray(obs.masks.packet_mask[2], dtype=float) > 0.5))
                            )
            probe_actions = {
                agent_id: _probe_action_for_observation(observations[agent_id], cfg) for agent_id in env.agent_ids
            }
            observations, _rewards, _done, _infos = env.step(probe_actions, prebuilt_observations=observations)

    result = {
        "experiment": experiment,
        "load": float(load),
        "episodes": int(episodes),
        "phase0": {
            "steps": int(phase0_steps),
            "owner_active_ratio": float(phase0_owner_active_count / max(phase0_steps, 1)),
            "mean_owner_choices_including_null": _mean(phase0_owner_choice_counts),
            "null_only_ratio": float(phase0_null_only_count / max(phase0_steps, 1)),
            "mean_raw_vs_snapshot_change_ratio": _mean(raw_vs_snapshot_change_ratios),
            "mean_final_vs_snapshot_change_ratio": _mean(final_vs_snapshot_change_ratios),
            "mean_final_vs_raw_diff_ratio": _mean(final_vs_raw_diff_ratios),
            "finalize_rewrite_episode_ratio": float(finalize_rewrite_flags / max(int(episodes), 1)),
            "minrate_infeasible_after_finalize_ratio": float(
                finalize_minrate_infeasible_flags / max(int(episodes), 1)
            ),
        },
        "phase_a": {
            "steps": int(phasea_steps),
            "mean_mode_choices": _mean(phasea_mode_feasible_counts),
            "mean_overlay_packet_choices": _mean(phasea_packet_choices_overlay),
            "mean_puncture_packet_choices": _mean(phasea_packet_choices_puncture),
            "nonkeep_mode_available_count": int(phasea_nonkeep_mode_count),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit effective action learnability for an SR-MAPPO preset.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--load", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()

    result = run_audit(args.experiment, args.load, args.episodes, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
