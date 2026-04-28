# SR-MAPPO URLLC Admission Collapse: Complete Technical Analysis

## Project Overview
- **Location**: `d:\URLLC_eMBB_Coexisting\`
- **Type**: Reinforcement Learning (SR-MAPPO) for spectrum sharing between URLLC and eMBB
- **Issue**: Policy performs like greedy baseline, URLLC admission collapses to ~0.3

## Root Causes Identified

### 1. CHECKPOINT SELECTION BIAS (CRITICAL)
**File**: sr_mappo/report.py lines 360-605
- Default preference: `best_throughput` (explicit bias)
- Evaluation IGNORES load-aware admission weighting (weight=3.0 at load=25.0)
- Comparative checkpoints (vs greedy) ranked BEFORE reward-based checkpoints

### 2. REWARD FUNCTION STRUCTURE (MIXED SIGNALS)
**File**: sr_mappo/config.py + sr_mappo/env.py
- Terminal (end-of-episode): strong admission penalty (12.0) when ratio < 0.78
- Step-wise (per-decision): weak allocation incentives (0.08—0.20 range)
- Gap: Policy can ignore admission during training, only penalized at episode end
- Load 25.0 has admission_weight=3.0 (non-uniform across loads)

### 3. ACTION SPACE CONFIGURATION
**File**: sr_mappo/trainer.py lines 157-182
- Phase A eMBB power actions may be disabled at runtime
- Check: `phase_a_embb_power_runtime_enabled()` calls config settings
- If False → output.embb_power_delta values clipped/ignored

### 4. ACTION MASKING & SHIELDING (ADMISSION BOTTLENECK)
**File**: sr_mappo/shield.py + sr_mappo/env.py
- When overlay/puncture become infeasible → mask=[0,0,0] for modes 1,2
- Policy forced to MODE_KEEP → no admission learning
- Fallback mechanism: shield.sanitize_action() returns KEEP on invalid masks

### 5. GREEDY BASELINE THROUGHPUT BIAS
**File**: sr_mappo/config.py
- Baseline: `myopic_throughput_greedy` (built-in throughput-first bias)
- RL policy implicitly trained to match this baseline behavior
- Shield & masking may amplify this through feasibility constraints

## Key Code Sections for Debugging

1. **Checkpoint evaluation**: sr_mappo/evaluate.py `evaluate_policy_only()` lines 1101-
2. **Reward computation**: sr_mappo/env.py `_compute_terminal_reward()` and `_compute_step_reward()`
3. **Action filtering**: sr_mappo/shield.py `sanitize_action()`
4. **Config structure**: sr_mappo/config.py `RewardConfig`, `EnvAdapterConfig`

## Checkpoint Details
- Best available: `sr_mappo_full_phase0_phasea_best.pt` (reward-based)
- Phase A variants: `*_phasea_*` files show phase A parameters evolution
- Smoke test versions: `checkpoints_v27_smoke/`, `checkpoints_v28_smoke/`

## Next Steps (if resuming)
1. Run detailed loss trajectory analysis to confirm training dynamics
2. Test reward schedule modifications (increase step-wise admission incentives)
3. Compare action distributions at load=25.0 between policy and greedy
4. Verify shield effectiveness during training vs inference
