#!/usr/bin/env python
"""Launch pure_ppo_ff_v1_no_greedy_obs_v1 training"""
import sys
from pathlib import Path
from sr_mappo.config import SRMAPPOConfig
from sr_mappo.experiments import apply_experiment_preset
from sr_mappo.train import run_default_training

try:
    print("\n" + "="*80)
    print("LAUNCHING: pure_ppo_ff_v1_no_greedy_obs_v1")
    print("="*80)
    
    # Load config and adjust iterations
    cfg = apply_experiment_preset(SRMAPPOConfig(), 'pure_ppo_ff_v1_no_greedy_obs_v1')
    
    # Ensure minimum 1000 iterations as requested
    if cfg.training.total_iterations < 1000:
        print(f"Adjusting iterations: {cfg.training.total_iterations} → 1000")
        cfg.training.total_iterations = 1000
    else:
        print(f"Total iterations: {cfg.training.total_iterations}")
    
    print(f"Run name: {cfg.training.run_name}")
    print(f"Greedy obs removed: {not cfg.env.include_greedy_reference_in_obs}")
    print(f"Shield disabled: {not cfg.shield.enable_feasibility_shield}")
    print(f"Phase A power enabled: {cfg.env.allow_phase_a_embb_power_adjustment}")
    print("="*80 + "\n")
    
    # Run training WITH report generation
    result = run_default_training(
        with_report=False,  # We'll generate report separately after 1000 iters
        experiment='pure_ppo_ff_v1_no_greedy_obs_v1'
    )
    
    print("\n✅ Training completed!")
    print(f"Checkpoint directory: {result['training'].get('checkpoint_dir')}")
    print(f"Iterations in history: {len(result['training'].get('history', []))}")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
