from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sr_mappo.config import SRMAPPOConfig
from sr_mappo.evaluate import evaluate_policy_only
from sr_mappo.expert_compare import plot_expert_core, plot_expert_modes
from sr_mappo.report import DEFAULT_EPISODES_PER_LOAD, DEFAULT_LOADS, RESULTS_DIR, run_greedy_sweep, run_mappo_sweep
from sr_mappo.run_docs import write_band_expert_markdown
from sr_mappo.trainer import run_training_loop

BANDS = {
    'low': [10.0, 15.0, 20.0],
    'mid': [20.0, 30.0, 40.0],
    'high': [30.0, 40.0, 50.0],
}


def build_band_cfg(band_name: str, loads: List[float]) -> SRMAPPOConfig:
    cfg = SRMAPPOConfig()
    cfg.training.run_name = f'sr_mappo_{band_name}_expert'
    cfg.training.curriculum_loads = list(loads)
    cfg.training.bc_loads = list(loads)
    cfg.training.eval_loads = list(loads)
    cfg.training.total_iterations = 24
    cfg.training.bc_episodes = 18
    cfg.training.bc_epochs = 6
    cfg.training.eval_every = 6
    cfg.training.checkpoint_every = 6
    cfg.training.eval_episodes_per_load = 4
    return cfg


def train_band_experts() -> tuple[Dict[str, Dict], Dict[str, SRMAPPOConfig]]:
    summaries: Dict[str, Dict] = {}
    cfgs: Dict[str, SRMAPPOConfig] = {}
    for band_name, loads in BANDS.items():
        cfg = build_band_cfg(band_name, loads)
        cfgs[band_name] = cfg
        summaries[band_name] = run_training_loop(cfg, evaluation_fn=evaluate_policy_only)
    return summaries, cfgs


def checkpoint_for_band(band_name: str) -> Path:
    base = Path('d:/URLLC_eMBB_Coexisting/checkpoints')
    candidates = [
        base / f'sr_mappo_{band_name}_expert_best.pt',
        base / f'sr_mappo_{band_name}_expert_final.pt',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'No checkpoint found for band {band_name}')


def band_for_load(load: float) -> str:
    if load <= 20.0:
        return 'low'
    if load <= 40.0:
        return 'mid'
    return 'high'


def run_banded_expert_sweep(loads: List[float], episodes_per_load: int) -> Dict[str, List[float]]:
    aggregate = None
    for load in loads:
        band = band_for_load(float(load))
        metrics, _rep = run_mappo_sweep([float(load)], episodes_per_load, checkpoint_for_band(band))
        if aggregate is None:
            aggregate = {key: [] for key in metrics.keys()}
        for key, value in metrics.items():
            aggregate[key].append(value[0])
    return aggregate or {}


def generate_banded_report(loads: List[float] | None = None, episodes_per_load: int = DEFAULT_EPISODES_PER_LOAD):
    loads = loads or DEFAULT_LOADS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    greedy, _ = run_greedy_sweep(loads, episodes_per_load)
    expert = run_banded_expert_sweep(loads, episodes_per_load)

    core_path = plot_expert_core(greedy, expert)
    modes_path = plot_expert_modes(greedy, expert)

    renamed_core = RESULTS_DIR / '09_banded_expert_core.png'
    renamed_modes = RESULTS_DIR / '10_banded_expert_modes.png'
    if renamed_core.exists():
        renamed_core.unlink()
    if renamed_modes.exists():
        renamed_modes.unlink()
    core_path.replace(renamed_core)
    modes_path.replace(renamed_modes)

    payload = {
        'loads': loads,
        'band_map': {str(load): band_for_load(float(load)) for load in loads},
        'checkpoints': {band: str(checkpoint_for_band(band)) for band in BANDS},
        'greedy': greedy,
        'banded_expert_sr_mappo': expert,
    }
    metrics_path = RESULTS_DIR / 'banded_expert_metrics.json'
    metrics_path.write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x), encoding='utf-8')
    return {
        'core': str(renamed_core),
        'modes': str(renamed_modes),
        'metrics': str(metrics_path),
    }


def main():
    train_summary, cfgs = train_band_experts()
    report = generate_banded_report()
    md_path = write_band_expert_markdown(train_summary, report_result=report, cfgs=cfgs)
    print('Band expert training complete')
    print(json.dumps({
        'trained_bands': list(train_summary.keys()),
        'artifacts': report,
        'run_markdown': str(md_path),
    }, indent=2))


if __name__ == '__main__':
    main()
