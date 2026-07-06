from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from sr_mappo.config import SRMAPPOConfig, torch_load_checkpoint
from sr_mappo.evaluate import _evaluate_one_load
from sr_mappo.report import (
    CHECKPOINT_DIR,
    DEFAULT_EPISODES_PER_LOAD,
    DEFAULT_LOADS,
    RESULTS_DIR,
    _style,
    _top_axis_lambda,
    run_greedy_sweep,
    run_mappo_sweep,
)
from sr_mappo.trainer import build_default_components


def compatible_checkpoints() -> List[Path]:
    cfg = SRMAPPOConfig()
    _env, model, _trainer = build_default_components(cfg)
    candidates = sorted(CHECKPOINT_DIR.glob('sr_mappo_phase_a_iter*.pt'))
    for extra in [CHECKPOINT_DIR / 'sr_mappo_phase_a_best.pt', CHECKPOINT_DIR / 'sr_mappo_phase_a_final.pt']:
        if extra.exists() and extra not in candidates:
            candidates.append(extra)

    compatible = []
    for path in candidates:
        try:
            payload = torch_load_checkpoint(path, map_location=cfg.training.device)
            model.load_state_dict(payload['model_state_dict'])
            compatible.append(path)
        except Exception:
            continue
    compatible.sort(key=lambda p: p.name)
    return compatible


def screen_checkpoints(loads: List[float]) -> List[Dict]:
    cfg = SRMAPPOConfig()
    env, model, _trainer = build_default_components(cfg)
    screened = []
    for ckpt in compatible_checkpoints():
        payload = torch_load_checkpoint(ckpt, map_location=cfg.training.device)
        model.load_state_dict(payload['model_state_dict'])
        per_load = []
        for load_idx, load in enumerate(loads):
            per_load.append(_evaluate_one_load(env, model, cfg, float(load), cfg.training.train_seed + 9000 + 100 * load_idx))
        screened.append({'checkpoint': str(ckpt), 'per_load': per_load})
    return screened


def choose_expert_map(screened: List[Dict], loads: List[float]) -> Dict[float, str]:
    cfg = SRMAPPOConfig()
    expert_map: Dict[float, str] = {}
    for load_idx, load in enumerate(loads):
        feasible = []
        fallback = []
        for item in screened:
            summary = item['per_load'][load_idx]
            score = (
                float(summary['rate_ratio'])
                - 0.08 * max(float(summary['power_ratio']) - 1.0, 0.0)
                + 0.20 * float(summary['admission_gap'])
            )
            pair = (score, float(summary['rate_ratio']), item['checkpoint'])
            fallback.append(pair)
            if float(summary['admission_gap']) >= cfg.training.non_worse_admission_gap and float(summary['power_ratio']) <= cfg.training.non_worse_power_tolerance:
                feasible.append(pair)
        chosen = max(feasible or fallback, key=lambda item: (item[0], item[1]))
        expert_map[float(load)] = chosen[2]
    return expert_map


def run_expert_sweep(loads: List[float], episodes_per_load: int, expert_map: Dict[float, str]) -> Dict[str, List[float]]:
    aggregate = None
    for load in loads:
        metrics, _rep = run_mappo_sweep([float(load)], episodes_per_load, Path(expert_map[float(load)]))
        if aggregate is None:
            aggregate = {key: [] for key in metrics.keys()}
        for key, value in metrics.items():
            aggregate[key].append(value[0])
    return aggregate or {}


def plot_expert_core(greedy: Dict, expert: Dict) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    loads = greedy['loads']
    panels = [
        ('Aggregate eMBB throughput', 'embb_rate', 1e6, 'Mbps'),
        ('URLLC admission ratio', 'urllc_admission', 1.0, 'Ratio'),
        ('Admitted URLLC reliability', 'urllc_reliability', 1.0, 'Reliability'),
        ('eMBB served ratio', 'embb_service_ratio', 1.0, 'Ratio'),
        ('Per-user eMBB rate', 'embb_user_rate', 1e6, 'Mbps'),
        ('Total transmit power', 'total_power', 1e3, 'mW'),
    ]
    for ax, (title, key, scale, ylabel) in zip(axes.flat, panels):
        ax.plot(loads, np.asarray(greedy[key]) / scale, marker='o', label='Greedy')
        ax.plot(loads, np.asarray(expert[key]) / scale, marker='s', label='Load-Conditioned SR-MAPPO')
        _style(ax, title, 'Average UE load per UAV', ylabel)
        _top_axis_lambda(ax, loads)
        ax.legend(fontsize=8)
    path = RESULTS_DIR / '07_load_conditioned_expert_core.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_expert_modes(greedy: Dict, expert: Dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    loads = greedy['loads']

    axes[0, 0].plot(loads, greedy['overlay_ratio'], marker='o', label='Greedy overlay')
    axes[0, 0].plot(loads, greedy['puncture_ratio'], marker='o', linestyle='--', label='Greedy puncture')
    axes[0, 0].plot(loads, expert['overlay_ratio'], marker='s', label='Expert overlay')
    axes[0, 0].plot(loads, expert['puncture_ratio'], marker='s', linestyle='--', label='Expert puncture')
    _style(axes[0, 0], 'Mode selection ratio', 'Average UE load per UAV', 'Ratio')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(loads, np.asarray(greedy['avg_puncture_loss']) / 1e6, marker='o', label='Greedy')
    axes[0, 1].plot(loads, np.asarray(expert['avg_puncture_loss']) / 1e6, marker='s', label='Expert')
    _style(axes[0, 1], 'Average eMBB loss per puncture', 'Average UE load per UAV', 'Mbps/action')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(loads, greedy['avg_overlay_retention'], marker='o', label='Greedy')
    axes[1, 0].plot(loads, expert['avg_overlay_retention'], marker='s', label='Expert')
    _style(axes[1, 0], 'Average eMBB retention under overlay', 'Average UE load per UAV', 'Retention ratio')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(loads, greedy['jain_fairness'], marker='o', label='Greedy')
    axes[1, 1].plot(loads, expert['jain_fairness'], marker='s', label='Expert')
    _style(axes[1, 1], "Jain's fairness index", 'Average UE load per UAV', 'Fairness')
    axes[1, 1].legend(fontsize=8)

    path = RESULTS_DIR / '08_load_conditioned_expert_modes.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main(loads: List[float] | None = None, episodes_per_load: int = DEFAULT_EPISODES_PER_LOAD):
    loads = loads or DEFAULT_LOADS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    screened = screen_checkpoints(loads)
    expert_map = choose_expert_map(screened, loads)
    greedy, _ = run_greedy_sweep(loads, episodes_per_load)
    expert = run_expert_sweep(loads, episodes_per_load, expert_map)
    core = plot_expert_core(greedy, expert)
    modes = plot_expert_modes(greedy, expert)
    payload = {
        'loads': loads,
        'expert_map': expert_map,
        'greedy': greedy,
        'load_conditioned_sr_mappo': expert,
        'screened_checkpoints': screened,
    }
    metrics_path = RESULTS_DIR / 'load_conditioned_expert_metrics.json'
    metrics_path.write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), encoding='utf-8')
    print(core)
    print(modes)
    print(metrics_path)


if __name__ == '__main__':
    main()
