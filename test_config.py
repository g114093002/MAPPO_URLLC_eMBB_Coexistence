#!/usr/bin/env python
"""Quick config verification"""
from sr_mappo.config import SRMAPPOConfig
from sr_mappo.experiments import apply_experiment_preset

# 验证no_greedy_obs_v1配置
cfg = apply_experiment_preset(SRMAPPOConfig(), 'pure_ppo_ff_v1_no_greedy_obs_v1')

print("=" * 80)
print("CONFIGURATION VERIFICATION: pure_ppo_ff_v1_no_greedy_obs_v1")
print("=" * 80)
print(f"\n[1] Greedy reference removal:")
print(f"    include_greedy_reference_in_obs: {cfg.env.include_greedy_reference_in_obs}")
print(f"    use_greedy_terminal_reference: {cfg.reward.use_greedy_terminal_reference}")
print(f"    use_greedy_reference_bc: {cfg.training.use_greedy_reference_bc}")

print(f"\n[2] Shield and masking:")
print(f"    enable_action_masking: {cfg.shield.enable_action_masking}")
print(f"    enable_feasibility_shield: {cfg.shield.enable_feasibility_shield}")
print(f"    apply_joint_reliability_rewrite: {cfg.shield.apply_joint_reliability_rewrite}")
print(f"    enable_greedy_fallback: {cfg.shield.enable_greedy_fallback}")

print(f"\n[3] Phase A power configuration:")
print(f"    allow_phase_a_embb_power_adjustment: {cfg.env.allow_phase_a_embb_power_adjustment}")
print(f"    learn_phase0_embb_power: {cfg.env.learn_phase0_embb_power}")
print(f"    phase: {cfg.env.phase}")

print(f"\n[4] Training specifics:")
print(f"    run_name: {cfg.training.run_name}")
print(f"    experiment_line: {cfg.training.experiment_line}")
print(f"    total_iterations: {cfg.training.total_iterations}")
print(f"    eval_every: {cfg.training.eval_every}")

print(f"\n[5] Baseline mode:")
print(f"    greedy_baseline_mode: {cfg.training.greedy_baseline_mode}")
print(f"    selection_baseline_mode: {cfg.training.selection_baseline_mode}")

print("\n✅ Configuration verified. Proceeding with training...")
print("=" * 80)
