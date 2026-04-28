#!/usr/bin/env python3
"""Quick validation of diagnostic infrastructure setup."""

import sys

try:
    # Test imports
    from sr_mappo.config import SRMAPPOConfig
    from sr_mappo.experiments import (
        EXPERIMENT_PURE_PPO_FF_V1_NO_GREEDY_OBS_PLANNING_MULTIOBJ_V1,
        EXPERIMENT_ABLATION_PHASE0_FROZEN_GREEDY_PHASE_A_ONLY,
        EXPERIMENT_ABLATION_PHASE0_ONLY_FROZEN_PHASE_A,
        apply_experiment_preset,
    )
    from sr_mappo.report import DIAGNOSTIC_EPISODES_PER_LOAD
    
    print("✓ All imports successful")
    
    # Test experiment presets
    base_cfg = SRMAPPOConfig()
    
    # Test planning_multiobj_v1
    cfg1 = apply_experiment_preset(base_cfg, EXPERIMENT_PURE_PPO_FF_V1_NO_GREEDY_OBS_PLANNING_MULTIOBJ_V1)
    assert cfg1.reward.planning_embb_service_weight > 1e-6, "Service weight not set"
    assert cfg1.action.embb_power_delta_limit > 0.5, "Power delta limit not updated"
    assert cfg1.training.total_iterations >= 1500, "Training iterations not set"
    print(f"✓ planning_multiobj_v1: service={cfg1.reward.planning_embb_service_weight:.2f}, " 
          f"delta_limit={cfg1.action.embb_power_delta_limit:.2f}, iters={cfg1.training.total_iterations}")
    
    # Test ablation A: Phase-0 frozen
    cfg2 = apply_experiment_preset(base_cfg, EXPERIMENT_ABLATION_PHASE0_FROZEN_GREEDY_PHASE_A_ONLY)
    assert cfg2.env.learn_embb_baseline == False, "Phase-0 learning not disabled"
    assert cfg2.env.allow_phase_a_embb_power_adjustment == True, "Phase-A power not enabled"
    print(f"✓ ablation_phase0_frozen_phase_a_only: learn_embb={cfg2.env.learn_embb_baseline}, "
          f"phase_a_power={cfg2.env.allow_phase_a_embb_power_adjustment}")
    
    # Test ablation B: Phase-A frozen
    cfg3 = apply_experiment_preset(base_cfg, EXPERIMENT_ABLATION_PHASE0_ONLY_FROZEN_PHASE_A)
    assert cfg3.env.learn_embb_baseline == True, "Phase-0 learning not enabled"
    assert cfg3.env.allow_phase_a_embb_power_adjustment == False, "Phase-A power not disabled"
    print(f"✓ ablation_phase0_only_frozen_phase_a: learn_embb={cfg3.env.learn_embb_baseline}, "
          f"phase_a_power={cfg3.env.allow_phase_a_embb_power_adjustment}")
    
    # Test diagnostic constant
    assert DIAGNOSTIC_EPISODES_PER_LOAD == 30, f"Wrong DIAGNOSTIC_EPISODES_PER_LOAD: {DIAGNOSTIC_EPISODES_PER_LOAD}"
    print(f"✓ DIAGNOSTIC_EPISODES_PER_LOAD = {DIAGNOSTIC_EPISODES_PER_LOAD}")
    
    # Verify env.py changes by checking the code
    import sr_mappo.env as env_module
    env_code = env_module.__file__
    print(f"✓ env module location: {env_code}")
    
    # Check for modified init method
    import inspect
    reset_source = inspect.getsource(env_module.SRMAPPOPhaseAEnv._reset_episode_state)
    assert 'planning_owner_match_vs_greedy_count' in reset_source, "New counters not in _reset_episode_state"
    assert 'planning_reward_rate_component_sum' in reset_source, "Reward component trackers not in _reset_episode_state"
    print("✓ _reset_episode_state has all new diagnostic counters")
    
    # Check _step_embb_planning changes
    step_planning_source = inspect.getsource(env_module.SRMAPPOPhaseAEnv._step_embb_planning)
    assert 'planning_owner_match_vs_greedy_count' in step_planning_source, "Owner comparison logic missing"
    assert 'reward_components' in step_planning_source, "Reward component tracking missing"
    print("✓ _step_embb_planning has owner/reward diagnostics")
    
    # Check _sanitize_phase_a method changes
    sanitize_source = inspect.getsource(env_module.SRMAPPOPhaseAEnv._sanitize_phase_a_embb_power_actions)
    assert 'phase_a_embb_power_projection_count' in sanitize_source, "Phase-A projection tracking missing"
    assert 'phase_a_embb_power_delta_clipped_count' in sanitize_source, "Phase-A clipping tracking missing"
    print("✓ _sanitize_phase_a_embb_power_actions has bottleneck diagnostics")
    
    # Check summarize_episode changes
    summarize_source = inspect.getsource(env_module.SRMAPPOPhaseAEnv.summarize_episode)
    assert 'planning_owner_match_ratio_vs_greedy' in summarize_source, "Owner match ratio missing"
    assert 'planning_reward_.*_component_mean' in summarize_source or 'planning_reward_rate_component_mean' in summarize_source, "Reward component means missing"
    assert 'raw_executed_embb_power_gap_ratio' in summarize_source, "Power gap ratio missing"
    print("✓ summarize_episode exports all diagnostic metrics")
    
    print("\n✓✓✓ All diagnostic infrastructure validated successfully! ✓✓✓")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
