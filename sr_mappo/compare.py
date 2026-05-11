"""Compare the current Greedy baseline against SR-MAPPO on the same density sweep."""

from __future__ import annotations

import json
import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from . import _bootstrap  # noqa: F401
from .baseline_catalog import (
    baseline_label as _shared_baseline_label,
    baseline_metadata as _shared_baseline_metadata,
    baseline_narrative as _shared_baseline_narrative,
    normalize_baseline_mode as _shared_normalize_baseline_mode,
)
from .config import SRMAPPOConfig, cfg_from_dict
from .env import SRMAPPOPhaseAEnv
from .evaluate import rollout_episode
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .networks import SRMAPPOActorCritic

from config import AlgorithmConfig, SimulationConfig, SystemConfig, URLLCConfig, eMBBConfig
from simulation import create_simulation
from visualization import create_plotter

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'Greedy' / 'results'
DEFAULT_CHECKPOINTS = [
]


def _cleanup_compare_artifacts() -> List[str]:
    removed: List[str] = []
    for name in (
        'greedy_vs_sr_mappo_overview.png',
        'greedy_vs_sr_mappo_modes.png',
        'greedy_vs_sr_mappo_metrics.json',
        'greedy_vs_sr_mappo_manifest.json',
    ):
        path = RESULTS_DIR / name
        if path.exists():
            path.unlink()
            removed.append(name)
    return removed


def _write_compare_manifest(payload: Dict[str, object]) -> Path:
    path = RESULTS_DIR / 'greedy_vs_sr_mappo_manifest.json'
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    return path


def _build_main_like_configs() -> Tuple[SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig, SimulationConfig]:
    sys_cfg = SystemConfig()
    sys_cfg.num_subcarriers = 8
    sys_cfg.num_embb_users = 20
    sys_cfg.num_urllc_users = 8
    sys_cfg.shadowing_std = 6.0
    sys_cfg.los_probability = 0.8
    sys_cfg.refresh_derived_params()

    urllc_cfg = URLLCConfig()
    urllc_cfg.packet_lengths = [160, 180, 200]
    urllc_cfg.target_error_probability = 1e-3
    urllc_cfg.power_limits = [24] * sys_cfg.num_urllc_users

    embb_cfg = eMBBConfig()
    embb_cfg.power_limits = [23] * sys_cfg.num_embb_users

    algo_cfg = AlgorithmConfig()
    algo_cfg.power_upper_bound = 0.25

    sim_cfg = SimulationConfig()
    sim_cfg.verbose = False
    sim_cfg.urllc_arrival_prob = 0.45
    # Slot-based Poisson arrival rate (pkt/slot). We use load-scaled total-rate
    # semantics (fixed=False): lambda(load)=base_lambda*load/base_total_per_uav.
    # With base_total_per_uav~=10, base_lambda=25.6 maps load=25 to ~64 pkt/slot.
    sim_cfg.urllc_poisson_rate = 25.6
    sim_cfg.fixed_urllc_poisson_rate = False
    sim_cfg.urllc_user_ratio = 0.30
    sim_cfg.min_user_density = 1
    sim_cfg.max_user_density = 40
    sim_cfg.num_density_points = 6

    # Optional runtime override for report/compare sweeps.
    # Useful when we need a fixed per-user lambda debug run without editing presets.
    poisson_override = os.environ.get("SR_MAPPO_URLLC_POISSON_RATE_OVERRIDE", "").strip()
    if poisson_override:
        try:
            sim_cfg.urllc_poisson_rate = float(poisson_override)
            sim_cfg.fixed_urllc_poisson_rate = True
        except ValueError:
            pass
    return sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _resolve_checkpoint(checkpoint_path: str | None = None, cfg: SRMAPPOConfig | None = None) -> Tuple[Path, str]:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        return path, 'explicit'
    cfg = cfg or SRMAPPOConfig()
    run_name = str(cfg.training.run_name)
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    report_best_vs_throughput_feasible = checkpoint_dir / f'{run_name}_report_best_vs_throughput_feasible_oracle.pt'
    report_best_vs_throughput_only = checkpoint_dir / f'{run_name}_report_best_vs_throughput_only_greedy.pt'
    report_best_vs_channel_only = checkpoint_dir / f'{run_name}_report_best_vs_channel_only_greedy.pt'
    report_best_vs_original = checkpoint_dir / f'{run_name}_report_best_vs_original_greedy.pt'
    report_best_vs_matched = checkpoint_dir / f'{run_name}_report_best_vs_matched_greedy.pt'
    report_best_floor_throughput = checkpoint_dir / f'{run_name}_report_best_floor_throughput.pt'
    report_best_throughput = checkpoint_dir / f'{run_name}_report_best_throughput.pt'
    report_best_reward = checkpoint_dir / f'{run_name}_report_best_reward.pt'
    report_best_alias = checkpoint_dir / f'{run_name}_report_best.pt'
    best_vs_throughput_feasible = checkpoint_dir / f'{run_name}_best_vs_throughput_feasible_oracle.pt'
    best_vs_throughput_only = checkpoint_dir / f'{run_name}_best_vs_throughput_only_greedy.pt'
    best_vs_channel_only = checkpoint_dir / f'{run_name}_best_vs_channel_only_greedy.pt'
    best_vs_original = checkpoint_dir / f'{run_name}_best_vs_original_greedy.pt'
    best_vs_matched = checkpoint_dir / f'{run_name}_best_vs_matched_greedy.pt'
    best_floor_throughput = checkpoint_dir / f'{run_name}_best_floor_throughput.pt'
    best_throughput = checkpoint_dir / f'{run_name}_best_throughput.pt'
    best_reward = checkpoint_dir / f'{run_name}_best_reward.pt'
    best_alias = checkpoint_dir / f'{run_name}_best.pt'
    final_path = checkpoint_dir / f'{run_name}_final.pt'
    selection_mode = str(getattr(cfg.training, "selection_mode", "dual_metric") or "dual_metric").strip().lower()
    baseline_pref = str(getattr(cfg.training, "selection_baseline_mode", "original") or "original").strip().lower()
    selection_admission_floor_ratio = float(getattr(cfg.training, "selection_admission_floor_ratio_to_baseline", 0.0) or 0.0)
    has_loadwise_selection_constraints = bool(
        dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}) or {})
        or selection_admission_floor_ratio > 0.0
    )

    comparative_preferred = [
        (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
        (report_best_vs_original, 'report_best_vs_original_greedy'),
        (report_best_vs_matched, 'report_best_vs_matched_greedy'),
        (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
        (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
    ]
    comparative_best = [
        (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
        (best_vs_original, 'best_vs_original_greedy'),
        (best_vs_matched, 'best_vs_matched_greedy'),
        (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
        (best_vs_channel_only, 'best_vs_channel_only_greedy'),
    ]
    if baseline_pref in {"throughput_feasible", "throughput_feasible_oracle", "coexistence_oracle"}:
        comparative_preferred = [
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"matched", "matched_fixed_embb"}:
        comparative_preferred = [
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"throughput_biased", "throughput_biased_greedy"}:
        comparative_preferred = [
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"throughput_only", "throughput_only_greedy"}:
        comparative_preferred = [
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"channel_only", "channel_only_greedy"}:
        comparative_preferred = [
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
        ]
        comparative_best = [
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
        ]

    if has_loadwise_selection_constraints:
        preferred = [
            (report_best_floor_throughput, 'report_best_floor_throughput'),
            (best_floor_throughput, 'best_floor_throughput'),
            (report_best_throughput, 'report_best_throughput'),
            (best_throughput, 'best_throughput'),
            *comparative_preferred,
            (report_best_reward, 'report_best_reward'),
            *comparative_best,
            (best_reward, 'best_reward'),
            (report_best_alias, 'report_best'),
            (best_alias, 'best'),
        ]
        for path, reason in preferred:
            if path.exists():
                return path, reason

    if selection_mode == "throughput_only":
        preferred = [
            (report_best_throughput, 'report_best_throughput'),
            (best_throughput, 'best_throughput'),
            *comparative_preferred,
            *comparative_best,
            (report_best_reward, 'report_best_reward'),
            (best_reward, 'best_reward'),
        ]
    else:
        preferred = [
            *comparative_preferred,
            (report_best_throughput, 'report_best_throughput'),
            (report_best_reward, 'report_best_reward'),
            *comparative_best,
            (best_throughput, 'best_throughput'),
            (best_reward, 'best_reward'),
        ]
    for path, reason in preferred:
        if path.exists():
            return path, reason

    if report_best_alias.exists() and (
        report_best_throughput.exists()
        or report_best_reward.exists()
        or report_best_vs_throughput_feasible.exists()
        or report_best_vs_original.exists()
        or report_best_vs_matched.exists()
        or report_best_vs_channel_only.exists()
    ):
        return report_best_alias, 'report_best_alias'
    if best_alias.exists():
        return best_alias, 'best'
    if final_path.exists():
        return final_path, 'final'
    search_dir = PROJECT_ROOT / 'checkpoints'
    latest_any = sorted(search_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_any:
        return latest_any[0], 'latest_any'
    raise FileNotFoundError(
        f'No SR-MAPPO checkpoint found in {search_dir}'
    )


def _normalize_baseline_mode(mode: str | None) -> str:
    return _shared_normalize_baseline_mode(mode, default="original")


def _greedy_baseline_mode(cfg: SRMAPPOConfig) -> str:
    return _normalize_baseline_mode(getattr(cfg.training, "greedy_baseline_mode", "original"))


def _baseline_label(mode: str | None) -> str:
    return _shared_baseline_label(mode)


def _baseline_metadata(mode: str | None) -> Dict[str, object]:
    return _shared_baseline_metadata(mode)


def _baseline_narrative(mode: str | None) -> Dict[str, str]:
    return _shared_baseline_narrative(mode)


def _load_frozen_greedy_payload(cfg: SRMAPPOConfig) -> Dict:
    raw_path = str(getattr(cfg.training, "frozen_greedy_metrics_path", "") or "").strip()
    if not raw_path:
        raise FileNotFoundError("greedy_baseline_mode='frozen_json' requires training.frozen_greedy_metrics_path")
    payload_path = Path(raw_path).expanduser()
    if not payload_path.exists():
        raise FileNotFoundError(f"Frozen greedy metrics not found: {payload_path}")
    return json.loads(payload_path.read_text(encoding='utf-8'))


def _load_checkpoint_cfg(checkpoint_path: Path) -> SRMAPPOConfig:
    payload = torch.load(checkpoint_path, map_location='cpu')
    return cfg_from_dict(payload.get('cfg'))


def _configure_density_scenario(
    target_users_per_uav: float,
    base_sys_cfg: SystemConfig,
    base_urllc_cfg: URLLCConfig,
    base_embb_cfg: eMBBConfig,
    base_algo_cfg: AlgorithmConfig,
    base_sim_cfg: SimulationConfig,
) -> Tuple[SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig, SimulationConfig]:
    sys_cfg = deepcopy(base_sys_cfg)
    urllc_cfg = deepcopy(base_urllc_cfg)
    embb_cfg = deepcopy(base_embb_cfg)
    algo_cfg = deepcopy(base_algo_cfg)
    sim_cfg = deepcopy(base_sim_cfg)

    base_embb_per_uav = max(1, int(np.ceil(base_sys_cfg.num_embb_users / base_sys_cfg.num_uavs)))
    base_urllc_per_uav = max(1, int(np.ceil(base_sys_cfg.num_urllc_users / base_sys_cfg.num_uavs)))
    base_total_per_uav = base_embb_per_uav + base_urllc_per_uav
    scale = float(target_users_per_uav / max(base_total_per_uav, 1))

    total_users = max(1, int(round(base_total_per_uav * sys_cfg.num_uavs * scale)))
    urllc_ratio = float(getattr(sim_cfg, 'urllc_user_ratio', 0.0))
    urllc_ratio = float(np.clip(urllc_ratio, 0.0, 1.0))
    if urllc_ratio <= 0.0:
        # Explicitly support eMBB-only scenario (URLLC ratio = 0.0).
        sys_cfg.num_urllc_users = 0
        sys_cfg.num_embb_users = max(1, total_users)
    elif urllc_ratio >= 1.0:
        # Explicitly support URLLC-only scenario (URLLC ratio = 1.0).
        sys_cfg.num_urllc_users = max(1, total_users)
        sys_cfg.num_embb_users = 0
    else:
        sys_cfg.num_urllc_users = max(1, int(round(total_users * urllc_ratio)))
        sys_cfg.num_embb_users = max(1, total_users - sys_cfg.num_urllc_users)
    sys_cfg.refresh_derived_params()

    urllc_cfg.power_limits = [24] * sys_cfg.num_urllc_users
    embb_cfg.power_limits = [23] * sys_cfg.num_embb_users
    if not bool(getattr(sim_cfg, 'fixed_urllc_poisson_rate', False)):
        sim_cfg.urllc_poisson_rate = max(1e-6, base_sim_cfg.urllc_poisson_rate * scale)
    sim_cfg.verbose = False
    return sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg


def _aggregate_episode_summaries(summaries: List[Dict[str, float]]) -> Dict[str, float]:
    def _mean(key: str, default: float = 0.0) -> float:
        values = np.asarray([float(item.get(key, default)) for item in summaries], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float(default)

    return {
        'embb_rates': _mean('embb_total_rate'),
        'embb_user_rates': _mean('embb_user_rate_mean'),
        'embb_service_ratio': _mean('embb_service_ratio'),
        'urllc_admission': _mean('urllc_admission_rate'),
        'admitted_urllc_reliability': _mean('admitted_urllc_reliability', np.nan),
        'effective_urllc_success_over_arrivals': _mean('effective_urllc_success_over_arrivals', 0.0),
        'empty_admission_case': _mean('empty_admission_case', 0.0),
        'urllc_success': _mean('admitted_urllc_reliability', np.nan),
        'power_consumption': _mean('total_power'),
        'overlay_ratio': _mean('overlay_ratio'),
        'puncture_ratio': _mean('puncture_ratio'),
        'overlay_retention': _mean('avg_overlay_retention'),
        'puncture_embb_loss': _mean('avg_puncture_embb_loss'),
        'overlay_embb_loss': _mean('avg_overlay_embb_loss'),
        'jain_fairness': _mean('jain_fairness'),
        'embb_only_fraction': _mean('embb_only_fraction'),
        'overlay_fraction': _mean('overlay_fraction'),
        'puncture_fraction': _mean('puncture_fraction'),
        'idle_fraction': _mean('idle_fraction'),
        'minislot_utilization': _mean('minislot_utilization'),
        'offered_load_scale': _mean('active_packets'),
    }


def _summary_from_slot_result(result_slot: Dict, sys_cfg: SystemConfig) -> Dict[str, float]:
    metrics = result_slot['metrics']
    return {
        'embb_total_rate': float(metrics.get('embb_total_rate', 0.0)),
        'embb_user_rate_mean': float(metrics.get('embb_user_rate_mean', 0.0)),
        'embb_service_ratio': float(metrics.get('embb_served_users', 0.0) / max(sys_cfg.num_embb_users, 1)),
        'urllc_admission_rate': float(metrics.get('urllc_admission_rate', 1.0)),
        'admitted_urllc_reliability': float(
            metrics.get('admitted_urllc_reliability', metrics.get('urllc_success_rate', np.nan))
        ),
        'effective_urllc_success_over_arrivals': float(
            metrics.get('effective_urllc_success_over_arrivals', metrics.get('urllc_success_rate', 0.0))
        ),
        'empty_admission_case': float(metrics.get('empty_admission_case', 0.0)),
        'urllc_success_rate': float(
            metrics.get('admitted_urllc_reliability', metrics.get('urllc_success_rate', np.nan))
        ),
        'total_power': float(metrics.get('total_power', 0.0)),
        'overlay_ratio': float(metrics.get('overlay_ratio', 0.0)),
        'puncture_ratio': float(metrics.get('puncture_ratio', 0.0)),
        'avg_overlay_retention': float(metrics.get('avg_overlay_retention', 0.0)),
        'avg_puncture_embb_loss': float(metrics.get('avg_puncture_embb_loss', 0.0)),
        'avg_overlay_embb_loss': float(metrics.get('avg_overlay_embb_loss', 0.0)),
        'jain_fairness': float(metrics.get('jain_fairness', 0.0)),
        'embb_only_fraction': float(metrics.get('embb_only_fraction', 0.0)),
        'overlay_fraction': float(metrics.get('overlay_fraction', 0.0)),
        'puncture_fraction': float(metrics.get('puncture_fraction', 0.0)),
        'idle_fraction': float(metrics.get('idle_fraction', 0.0)),
        'minislot_utilization': float(metrics.get('minislot_utilization', 0.0)),
        'active_packets': float(metrics.get('active_urllc_users', 0.0)),
    }


def _build_model_for_env(env: SRMAPPOPhaseAEnv, rl_cfg: SRMAPPOConfig, checkpoint_path: Path) -> SRMAPPOActorCritic:
    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, rl_cfg)
    device = torch.device(rl_cfg.training.device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def run_sr_mappo_density_sweep(
    rl_cfg: SRMAPPOConfig,
    checkpoint_path: Path,
    densities: List[float],
    episodes_per_density: int = 10,
) -> Dict[str, List[float]]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    eval_cfg = deepcopy(_load_checkpoint_cfg(checkpoint_path))
    eval_cfg.env.include_greedy_reference_in_obs = False

    result = {
        'densities': [],
        'embb_rates': [],
        'embb_user_rates': [],
        'embb_service_ratio': [],
        'urllc_admission': [],
        'admitted_urllc_reliability': [],
        'effective_urllc_success_over_arrivals': [],
        'empty_admission_case': [],
        'urllc_success': [],
        'power_consumption': [],
        'overlay_ratio': [],
        'puncture_ratio': [],
        'overlay_retention': [],
        'puncture_embb_loss': [],
        'overlay_embb_loss': [],
        'jain_fairness': [],
        'embb_only_fraction': [],
        'overlay_fraction': [],
        'puncture_fraction': [],
        'idle_fraction': [],
        'minislot_utilization': [],
        'offered_load_scale': [],
    }

    model = None
    for density_idx, target_load in enumerate(densities):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            target_users_per_uav=float(target_load),
            base_sys_cfg=base_sys,
            base_urllc_cfg=base_urllc,
            base_embb_cfg=base_embb,
            base_algo_cfg=base_algo,
            base_sim_cfg=base_sim,
        )
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, eval_cfg)
        if model is None:
            model = _build_model_for_env(env, eval_cfg, checkpoint_path)

        summaries = []
        for episode_idx in range(episodes_per_density):
            seed = eval_cfg.training.train_seed + 5000 + density_idx * 100 + episode_idx
            summaries.append(rollout_episode(env, model=model, seed=seed, use_greedy=False))

        agg = _aggregate_episode_summaries(summaries)
        result['densities'].append(float(target_load))
        for key, value in agg.items():
            result[key].append(float(value))

        print(
            f"SR-MAPPO density {target_load:.2f} UE/UAV | "
            f"eMBB={agg['embb_rates']/1e6:.2f} Mbps, "
            f"admission={agg['urllc_admission']:.3f}, "
            f"power={agg['power_consumption']*1e3:.2f} mW"
        )

    return result


def run_env_greedy_density_sweep(
    rl_cfg: SRMAPPOConfig,
    densities: List[float],
    episodes_per_density: int = 10,
    greedy_policy: str = "reference",
) -> Dict[str, List[float]]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()

    result = {
        'densities': [],
        'embb_rates': [],
        'embb_user_rates': [],
        'embb_service_ratio': [],
        'urllc_admission': [],
        'admitted_urllc_reliability': [],
        'effective_urllc_success_over_arrivals': [],
        'empty_admission_case': [],
        'urllc_success': [],
        'power_consumption': [],
        'overlay_ratio': [],
        'puncture_ratio': [],
        'overlay_retention': [],
        'puncture_embb_loss': [],
        'overlay_embb_loss': [],
        'jain_fairness': [],
        'embb_only_fraction': [],
        'overlay_fraction': [],
        'puncture_fraction': [],
        'idle_fraction': [],
        'minislot_utilization': [],
        'offered_load_scale': [],
    }

    for density_idx, target_load in enumerate(densities):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            target_users_per_uav=float(target_load),
            base_sys_cfg=base_sys,
            base_urllc_cfg=base_urllc,
            base_embb_cfg=base_embb,
            base_algo_cfg=base_algo,
            base_sim_cfg=base_sim,
        )
        env_cfg = deepcopy(rl_cfg)
        env_cfg.env.include_greedy_reference_in_obs = False
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, env_cfg)
        summaries = []
        for episode_idx in range(episodes_per_density):
            seed = env_cfg.training.train_seed + 7000 + density_idx * 100 + episode_idx
            summaries.append(rollout_episode(env, model=None, seed=seed, use_greedy=True, greedy_policy=greedy_policy))

        agg = _aggregate_episode_summaries(summaries)
        result['densities'].append(float(target_load))
        for key, value in agg.items():
            result[key].append(float(value))

        print(
            f"Env-greedy density {target_load:.2f} UE/UAV | "
            f"eMBB={agg['embb_rates']/1e6:.2f} Mbps, "
            f"admission={agg['urllc_admission']:.3f}, "
            f"power={agg['power_consumption']*1e3:.2f} mW"
        )

    return result


def run_original_greedy_density_sweep(
    rl_cfg: SRMAPPOConfig,
    densities: List[float],
    episodes_per_density: int = 10,
    lite: bool = False,
    normal_mode: str = "",
) -> Dict[str, List[float]]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()

    result = {
        'densities': [], 'embb_rates': [], 'embb_user_rates': [], 'embb_service_ratio': [],
        'urllc_admission': [], 'admitted_urllc_reliability': [], 'effective_urllc_success_over_arrivals': [],
        'empty_admission_case': [], 'urllc_success': [], 'power_consumption': [], 'overlay_ratio': [],
        'puncture_ratio': [], 'overlay_retention': [], 'puncture_embb_loss': [], 'overlay_embb_loss': [],
        'jain_fairness': [], 'embb_only_fraction': [], 'overlay_fraction': [], 'puncture_fraction': [],
        'idle_fraction': [], 'minislot_utilization': [], 'offered_load_scale': [],
    }

    for density_idx, target_load in enumerate(densities):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            target_users_per_uav=float(target_load),
            base_sys_cfg=base_sys,
            base_urllc_cfg=base_urllc,
            base_embb_cfg=base_embb,
            base_algo_cfg=base_algo,
            base_sim_cfg=base_sim,
        )
        summaries = []
        for episode_idx in range(episodes_per_density):
            sim_local = deepcopy(sim_cfg)
            sim_local.random_seed = int(rl_cfg.training.train_seed + 7000 + density_idx * 100 + episode_idx)
            simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_local)
            if normal_mode == "v1":
                result_slot = simulation.run_single_allocation_normal_v1(slot_index=episode_idx)
            elif normal_mode == "v2":
                result_slot = simulation.run_single_allocation_normal_v2(slot_index=episode_idx)
            elif lite:
                result_slot = simulation.run_single_allocation_lite(slot_index=episode_idx)
            else:
                result_slot = simulation.run_single_allocation(slot_index=episode_idx)
            summaries.append(_summary_from_slot_result(result_slot, sys_cfg))
        agg = _aggregate_episode_summaries(summaries)
        result['densities'].append(float(target_load))
        for key, value in agg.items():
            result[key].append(float(value))

        print(
            f"{'Original-greedy-normal-v2' if normal_mode == 'v2' else ('Original-greedy-normal-v1' if normal_mode == 'v1' else ('Original-greedy-lite' if lite else 'Original-greedy'))} density {target_load:.2f} UE/UAV | "
            f"eMBB={agg['embb_rates']/1e6:.2f} Mbps, "
            f"admission={agg['urllc_admission']:.3f}, "
            f"power={agg['power_consumption']*1e3:.2f} mW"
        )

    return result


def run_upper_bound_density_sweep(
    rl_cfg: SRMAPPOConfig,
    densities: List[float],
    episodes_per_density: int = 10,
    mode: str = "embb_only_ceiling",
) -> Dict[str, List[float]]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()

    result = {
        'densities': [], 'embb_rates': [], 'embb_user_rates': [], 'embb_service_ratio': [],
        'urllc_admission': [], 'admitted_urllc_reliability': [], 'effective_urllc_success_over_arrivals': [],
        'empty_admission_case': [], 'urllc_success': [], 'power_consumption': [], 'overlay_ratio': [],
        'puncture_ratio': [], 'overlay_retention': [], 'puncture_embb_loss': [], 'overlay_embb_loss': [],
        'jain_fairness': [], 'embb_only_fraction': [], 'overlay_fraction': [], 'puncture_fraction': [],
        'idle_fraction': [], 'minislot_utilization': [], 'offered_load_scale': [],
    }

    for density_idx, target_load in enumerate(densities):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            target_users_per_uav=float(target_load),
            base_sys_cfg=base_sys,
            base_urllc_cfg=base_urllc,
            base_embb_cfg=base_embb,
            base_algo_cfg=base_algo,
            base_sim_cfg=base_sim,
        )
        summaries = []
        for episode_idx in range(episodes_per_density):
            sim_local = deepcopy(sim_cfg)
            sim_local.random_seed = int(rl_cfg.training.train_seed + 9000 + density_idx * 100 + episode_idx)
            simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_local)
            if mode == "throughput_feasible_oracle":
                result_slot = simulation.run_throughput_feasible_oracle(slot_index=episode_idx)
            else:
                result_slot = simulation.run_embb_only_ceiling(slot_index=episode_idx)
            summaries.append(_summary_from_slot_result(result_slot, sys_cfg))
        agg = _aggregate_episode_summaries(summaries)
        result['densities'].append(float(target_load))
        for key, value in agg.items():
            result[key].append(float(value))
    return result


def load_frozen_greedy_density_sweep(
    rl_cfg: SRMAPPOConfig,
    densities: List[float],
) -> Dict[str, List[float]]:
    payload = _load_frozen_greedy_payload(rl_cfg)
    metrics = payload.get('greedy_metrics', {})
    loads = [float(load) for load in metrics.get('loads', [])]
    if loads != [float(d) for d in densities]:
        raise ValueError(f"Frozen greedy loads mismatch: expected {densities}, got {loads}")
    return {
        'densities': loads,
        'embb_rates': list(metrics.get('embb_rate', [])),
        'embb_user_rates': list(metrics.get('embb_user_rate', [])),
        'embb_service_ratio': list(metrics.get('embb_service_ratio', [])),
        'urllc_admission': list(metrics.get('urllc_admission', [])),
        'admitted_urllc_reliability': list(
            metrics.get('admitted_urllc_reliability', metrics.get('urllc_reliability', []))
        ),
        'effective_urllc_success_over_arrivals': list(metrics.get('effective_urllc_success_over_arrivals', [])),
        'empty_admission_case': list(metrics.get('empty_admission_case', [])),
        'urllc_success': list(metrics.get('admitted_urllc_reliability', metrics.get('urllc_reliability', []))),
        'power_consumption': list(metrics.get('total_power', [])),
        'overlay_ratio': list(metrics.get('overlay_ratio', [])),
        'puncture_ratio': list(metrics.get('puncture_ratio', [])),
        'overlay_retention': list(metrics.get('avg_overlay_retention', [])),
        'puncture_embb_loss': list(metrics.get('avg_puncture_loss', [])),
        'overlay_embb_loss': list(metrics.get('avg_overlay_embb_loss', [0.0] * len(loads))),
        'jain_fairness': list(metrics.get('jain_fairness', [])),
        'embb_only_fraction': list(metrics.get('embb_only_fraction', [])),
        'overlay_fraction': list(metrics.get('overlay_fraction', [])),
        'puncture_fraction': list(metrics.get('puncture_fraction', [])),
        'idle_fraction': list(metrics.get('idle_fraction', [])),
        'minislot_utilization': list(metrics.get('minislot_utilization', [])),
        'offered_load_scale': list(metrics.get('active_packets', [])),
    }


def run_comparison(
    checkpoint_path: str | None = None,
    episodes_per_density: int = 10,
    experiment_line: str | None = None,
) -> Dict[str, object]:
    rl_cfg = apply_experiment_preset(SRMAPPOConfig(), experiment_line)
    checkpoint, checkpoint_reason = _resolve_checkpoint(checkpoint_path, rl_cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stale_removed = _cleanup_compare_artifacts()

    greedy_mode = _greedy_baseline_mode(rl_cfg)
    greedy_label = _baseline_label(greedy_mode)
    checkpoint_cfg = cfg_from_dict(torch.load(checkpoint, map_location='cpu').get('cfg'))
    rl_cfg.env.fixed_embb_baseline_policy = checkpoint_cfg.env.fixed_embb_baseline_policy
    print('Running Greedy density analysis for direct comparison...')
    if greedy_mode == "matched_fixed_embb":
        greedy_result = run_env_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            greedy_policy="reference",
        )
    elif greedy_mode == "throughput_feasible_oracle":
        greedy_result = run_env_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            greedy_policy="throughput_feasible",
        )
    elif greedy_mode == "throughput_only_greedy":
        greedy_result = run_env_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            greedy_policy="throughput_only",
        )
    elif greedy_mode == "throughput_biased_greedy":
        greedy_result = run_env_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            greedy_policy="throughput_biased",
        )
        greedy_label = "Throughput-biased Greedy"
    elif greedy_mode == "original_greedy_normal_v1":
        greedy_result = run_original_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            normal_mode="v1",
        )
    elif greedy_mode == "original_greedy_normal_v2":
        greedy_result = run_original_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            normal_mode="v2",
        )
    elif greedy_mode == "channel_only_greedy":
        greedy_result = run_env_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
            greedy_policy="channel_only",
        )
    elif greedy_mode == "frozen_json":
        greedy_result = load_frozen_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
        )
    else:
        greedy_result = run_original_greedy_density_sweep(
            rl_cfg=rl_cfg,
            densities=rl_cfg.training.eval_loads,
            episodes_per_density=episodes_per_density,
        )
    greedy_result.update(_baseline_metadata(greedy_mode))
    greedy_result.update(_baseline_narrative(greedy_mode))

    print(f'Loading SR-MAPPO checkpoint: {checkpoint} ({checkpoint_reason})')
    mappo_result = run_sr_mappo_density_sweep(
        rl_cfg=rl_cfg,
        checkpoint_path=checkpoint,
        densities=greedy_result['densities'],
        episodes_per_density=episodes_per_density,
    )
    embb_only_ceiling = run_upper_bound_density_sweep(
        rl_cfg=rl_cfg,
        densities=greedy_result['densities'],
        episodes_per_density=episodes_per_density,
        mode="embb_only_ceiling",
    )
    throughput_feasible_oracle = run_upper_bound_density_sweep(
        rl_cfg=rl_cfg,
        densities=greedy_result['densities'],
        episodes_per_density=episodes_per_density,
        mode="throughput_feasible_oracle",
    )
    throughput_feasible_oracle.update(_baseline_metadata("throughput_feasible_oracle"))
    throughput_feasible_oracle.update(_baseline_narrative("throughput_feasible_oracle"))
    baseline_catalog_payload = {
        mode: {
            **_baseline_metadata(mode),
            **_baseline_narrative(mode),
        }
        for mode in (
            "matched_fixed_embb",
            "throughput_feasible_oracle",
            "throughput_biased_greedy",
            "throughput_only_greedy",
        )
    }

    plotter = create_plotter(str(RESULTS_DIR))
    overview_path = RESULTS_DIR / 'greedy_vs_sr_mappo_overview.png'
    mode_path = RESULTS_DIR / 'greedy_vs_sr_mappo_modes.png'
    plotter.plot_greedy_vs_mappo_overview(
        greedy_result,
        mappo_result,
        mappo_label='MAPPO',
        greedy_label=greedy_label,
        save_path=str(overview_path),
    )
    plotter.plot_greedy_vs_mappo_mode_comparison(
        greedy_result,
        mappo_result,
        mappo_label='MAPPO',
        greedy_label=greedy_label,
        save_path=str(mode_path),
    )

    metrics_path = RESULTS_DIR / 'greedy_vs_sr_mappo_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as fp:
        json.dump(
            {
                'checkpoint': str(checkpoint),
                'checkpoint_selection_reason': checkpoint_reason,
                'experiment_line': rl_cfg.training.experiment_line,
                'episodes_per_density': episodes_per_density,
                'greedy_baseline_mode': greedy_mode,
                'comparison_baseline_key': greedy_mode,
                'comparison_baseline_label': greedy_label,
                'selected_baseline_key': greedy_mode,
                'selected_baseline_label': greedy_label,
                **_baseline_metadata(greedy_mode),
                **_baseline_narrative(greedy_mode),
                'baseline_catalog': baseline_catalog_payload,
                'phase': checkpoint_cfg.env.phase,
                'learn_embb_baseline': bool(checkpoint_cfg.env.learn_embb_baseline),
                'selection_admission_floor_ratio_to_baseline': float(getattr(checkpoint_cfg.training, 'selection_admission_floor_ratio_to_baseline', 0.0) or 0.0),
                'learn_phase0_embb_power': bool(getattr(checkpoint_cfg.env, 'learn_phase0_embb_power', True)),
                'allow_phase_a_embb_power_adjustment': bool(checkpoint_cfg.env.allow_phase_a_embb_power_adjustment),
                'enable_action_masking': bool(checkpoint_cfg.shield.enable_action_masking),
                'enable_feasibility_shield': bool(checkpoint_cfg.shield.enable_feasibility_shield),
                'apply_joint_reliability_rewrite': bool(checkpoint_cfg.shield.apply_joint_reliability_rewrite),
                'enable_greedy_fallback': bool(checkpoint_cfg.shield.enable_greedy_fallback),
                'greedy': greedy_result,
                'embb_only_ceiling': embb_only_ceiling,
                'throughput_feasible_oracle': throughput_feasible_oracle,
                'sr_mappo': mappo_result,
            },
            fp,
            indent=2,
            default=_json_default,
        )
    manifest_path = _write_compare_manifest(
        {
            'generated_at': datetime.now().isoformat(),
            'checkpoint': str(checkpoint),
            'checkpoint_selection_reason': checkpoint_reason,
            'experiment_line': rl_cfg.training.experiment_line,
            'greedy_baseline_mode': greedy_mode,
            'comparison_baseline_key': greedy_mode,
            'comparison_baseline_label': greedy_label,
            'selected_baseline_key': greedy_mode,
            'selected_baseline_label': greedy_label,
            **_baseline_metadata(greedy_mode),
            'phase': checkpoint_cfg.env.phase,
            'learn_embb_baseline': bool(checkpoint_cfg.env.learn_embb_baseline),
            'selection_admission_floor_ratio_to_baseline': float(getattr(checkpoint_cfg.training, 'selection_admission_floor_ratio_to_baseline', 0.0) or 0.0),
            'learn_phase0_embb_power': bool(getattr(checkpoint_cfg.env, 'learn_phase0_embb_power', True)),
            'allow_phase_a_embb_power_adjustment': bool(checkpoint_cfg.env.allow_phase_a_embb_power_adjustment),
            'enable_action_masking': bool(checkpoint_cfg.shield.enable_action_masking),
            'enable_feasibility_shield': bool(checkpoint_cfg.shield.enable_feasibility_shield),
            'apply_joint_reliability_rewrite': bool(checkpoint_cfg.shield.apply_joint_reliability_rewrite),
            'enable_greedy_fallback': bool(checkpoint_cfg.shield.enable_greedy_fallback),
            'episodes_per_density': int(episodes_per_density),
            'overview_plot': str(overview_path),
            'mode_plot': str(mode_path),
            'metrics_json': str(metrics_path),
            'stale_artifacts_removed': stale_removed,
        }
    )

    return {
        'checkpoint': str(checkpoint),
        'checkpoint_selection_reason': checkpoint_reason,
        'greedy_result': greedy_result,
        'embb_only_ceiling': embb_only_ceiling,
        'throughput_feasible_oracle': throughput_feasible_oracle,
        'sr_mappo_result': mappo_result,
        'overview_plot': str(overview_path),
        'mode_plot': str(mode_path),
        'metrics_json': str(metrics_path),
        'manifest_path': str(manifest_path),
    }


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Greedy baseline variants against SR-MAPPO")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override.")
    parser.add_argument("--episodes-per-density", type=int, default=10)
    args = parser.parse_args()

    outputs = run_comparison(
        checkpoint_path=args.checkpoint,
        episodes_per_density=int(args.episodes_per_density),
        experiment_line=args.experiment,
    )
    print('\nComparison complete.')
    print(f"Experiment line: {experiment_label(args.experiment)}")
    print(f"Overview plot: {outputs['overview_plot']}")
    print(f"Mode plot: {outputs['mode_plot']}")
    print(f"Metrics json: {outputs['metrics_json']}")


if __name__ == '__main__':
    main()
