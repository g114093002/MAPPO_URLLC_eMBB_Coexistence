#!/usr/bin/env python3
"""Minimal smoke test for modified environment."""

import sys
import numpy as np

try:
    import importlib
    import sr_mappo.env
    importlib.reload(sr_mappo.env)
    
    from sr_mappo.env import SRMAPPOPhaseAEnv
    from sr_mappo.config import SRMAPPOConfig
    from sr_mappo.experiments import apply_experiment_preset
    from sr_mappo.compare import _build_main_like_configs
    
    # Try loading a config using the correct method
    base_config = SRMAPPOConfig()
    config = apply_experiment_preset(base_config, 'pure_ppo_ff_v1')
    print("✓ Config loaded successfully")
    
    # Get the base simulation configs
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()
    print("✓ Base simulation configs created successfully")
    
    # Try creating environment
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, config)
    print("✓ Environment initialized successfully")
    
    # Test reset
    obs = env.reset()
    print(f"✓ Environment reset successful, obs keys: {obs.keys() if isinstance(obs, dict) else 'tuple'}")
    
    # Test summarize_episode which has our new metrics
    summary = env.summarize_episode()
    print(f"✓ Episode summary generated with {len(summary)} metrics")
    print(f"✓ Environment phase: {summary.get('phase', 'unknown')}")
    
    # Debug: print some keys
    sample_keys = list(summary.keys())[:10]
    print(f"Sample summary keys: {sample_keys}")
    
    # Test summarize_episode which has our new metrics
    summary = env.summarize_episode()
    print(f"✓ Episode summary generated with {len(summary)} metrics")
    print(f"✓ Summary type: {type(summary)}")
    print(f"✓ Environment phase: {summary.get('phase', 'unknown')}")
    
    # Debug: print some keys
    sample_keys = list(summary.keys())[:10]
    print(f"Sample summary keys: {sample_keys}")
    
    # Check for new metrics
    new_metrics = [
        'phase_a_embb_power_positive_clamped_to_zero_count',
        'phase_a_embb_power_positive_clamped_ratio',
        'phase_a_embb_power_negative_ratio',
        'phase_a_embb_power_negative_executed_ratio',
    ]
    
    # Check for new metrics
    new_metrics = [
        'phase_a_embb_power_positive_clamped_to_zero_count',
        'phase_a_embb_power_positive_clamped_ratio',
        'phase_a_embb_power_negative_ratio',
        'phase_a_embb_power_negative_executed_ratio',
    ]
    
    print(f"Checking for {len(new_metrics)} new metrics...")
    all_clamped_keys = [k for k in summary.keys() if 'clamped' in k]
    print(f"All keys containing 'clamped': {all_clamped_keys}")
    
    for metric in new_metrics:
        if metric in summary:
            print(f"✓ Found new metric: {metric} = {summary[metric]}")
        else:
            print(f"✗ Missing new metric: {metric}")
    
    print("\n✓✓ SMOKE TEST PASSED ✓✓")
    sys.exit(0)
    
except Exception as e:
    print(f"\n✗✗ SMOKE TEST FAILED ✗✗")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
