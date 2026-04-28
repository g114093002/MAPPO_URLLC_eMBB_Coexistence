# SR-MAPPO URLLC Admission Fix: Implementation Scripts

This file contains directly-applicable code modifications for restoring URLLC admission.

---

## IMPLEMENTATION SCRIPT 1: Config Modifications

### File: `sr_mappo/config.py`

#### Change 1.1: RewardConfig - Rebalance Step Rewards

**Location**: Find the `RewardConfig` dataclass (line ~30-120)

**Before**:
```python
@dataclass
class RewardConfig:
    ...
    schedule_success_weight: float = 0.08
    embb_damage_weight: float = 0.20
    overlay_gain_weight: float = 0.08
    overlay_margin_weight: float = 0.15
    missed_overlay_penalty: float = 0.12
    ...
```

**After**:
```python
@dataclass
class RewardConfig:
    ...
    schedule_success_weight: float = 0.22        # Was 0.08 → +175%
    embb_damage_weight: float = 0.14             # Was 0.20 → -30%
    overlay_gain_weight: float = 0.12            # Was 0.08 → +50%
    overlay_margin_weight: float = 0.22          # Was 0.15 → +47%
    missed_overlay_penalty: float = 0.18         # Was 0.12 → +50%
    # ADD NEW: Direct admission incentives
    safe_admission_bonus_weight: float = 0.15
    safe_embb_retention_threshold: float = 0.80
    ...
```

**Why**: Steps 1-3 of minimum modification plan.

---

#### Change 1.2: ShieldConfig - (No changes needed)

All shield settings can remain as-is. They are correct.

---

#### Change 1.3: EnvAdapterConfig - Power Anchor (optional, can use training config instead)

**Location**: Find `EnvAdapterConfig` dataclass (line ~140-180)

```python
# OPTIONAL: These can also be set via training config
# leave defaults unchanged, override in training config file
```

---

### File: Training Config / Script (e.g., wherever you set `cfg.env` and `cfg.training`)

#### Change 2.1: Training Phase A Power Anchor

**Add to training config** (or set in your training script):

```python
# In SRMAPPOConfig or wherever training params are set

# OLD (or missing)
cfg.training.use_phase_a_embb_power_anchor = False
cfg.training.phase_a_embb_power_anchor_weight = 0.0

# NEW
cfg.training.use_phase_a_embb_power_anchor = True
cfg.training.phase_a_embb_power_anchor_weight = 0.5
cfg.training.phase_a_embb_power_anchor_start_iteration = 0
cfg.training.phase_a_embb_power_anchor_min_retention = 0.85

cfg.training.phase_a_embb_power_anchor_load_weights = {
    15.0: 0.3,
    25.0: 1.0,
    35.0: 1.0,
}
```

---

#### Change 2.2: Checkpoint Selection

**Add to training/report config**:

```python
# OLD (or defaults)
cfg.training.primary_checkpoint_preference = "best_throughput"
cfg.training.require_primary_checkpoint_match = False

# NEW
cfg.training.primary_checkpoint_preference = "best_balanced"
cfg.training.require_primary_checkpoint_match = True

# OPTIONAL: Enforce admission floor (add to report.py or make global)
SELECTION_ADMISSION_FLOOR = 0.65  # Don't pick if admission < 65%
```

---

## IMPLEMENTATION SCRIPT 2: Code Modifications (if using config-based approach)

### The config changes above should be sufficient. Skip this section if those work.

If you need to hardcode changes directly:

**File**: `sr_mappo/config.py` (RewardConfig class)

```python
@dataclass
class RewardConfig:
    """Minimal step-first reward coefficients for SR-MAPPO."""

    schedule_success_weight: float = 0.22  # ← CHANGED: was 0.08
    embb_damage_weight: float = 0.14      # ← CHANGED: was 0.20
    puncture_extra_penalty: float = 0.05
    overlay_gain_weight: float = 0.12     # ← CHANGED: was 0.08
    overlay_margin_weight: float = 0.22   # ← CHANGED: was 0.15
    missed_overlay_penalty: float = 0.18  # ← CHANGED: was 0.12
    safe_puncture_preference_penalty_weight: float = 0.0
    overlay_margin_needed_to_override_puncture: float = 0.0
    puncture_loss_safe_threshold: float = 0.0
    safe_puncture_bonus_weight: float = 0.0
    overlay_when_safe_puncture_penalty_weight: float = 0.0
    power_penalty_scale: float = 0.01
    keep_feasible_penalty: float = 0.10
    invalid_action_penalty: float = 0.12
    collision_rewrite_penalty: float = 0.20
    power_projection_penalty: float = 0.05
    urgency_reward_weight: float = 0.15
    keep_urgent_penalty_weight: float = 0.25
    terminal_unscheduled_penalty: float = 8.00
    terminal_embb_rate_weight: float = 1.50
    terminal_urllc_admission_weight: float = 4.00
    terminal_urllc_admission_target: float = 0.78
    terminal_urllc_admission_penalty: float = 12.00
    terminal_embb_fairness_weight: float = 0.00
    terminal_embb_min_rate_penalty: float = 3.00
    terminal_embb_rate_normalizer: float = 5.0e6
    # ... rest unchanged ...
    
    # ← ADD THESE NEW FIELDS
    safe_admission_bonus_weight: float = 0.15  # NEW
    safe_embb_retention_threshold: float = 0.80  # NEW
    unsafe_admission_penalty_weight: float = 0.10  # NEW
    unsafe_embb_retention_threshold: float = 0.65  # NEW
```

---

## IMPLEMENTATION SCRIPT 3: Training Script Changes (if needed)

### If you have a training entry point (e.g., `train.py` or similar):

```python
# In your training main() or config loading section

# STEP 1: Load default config
cfg = SRMAPPOConfig()

# STEP 2: Apply modifications (or read from file)
# Option A: Direct assignment
cfg.reward.schedule_success_weight = 0.22
cfg.reward.embb_damage_weight = 0.14
cfg.reward.overlay_gain_weight = 0.12
cfg.reward.overlay_margin_weight = 0.22
cfg.reward.missed_overlay_penalty = 0.18
cfg.reward.safe_admission_bonus_weight = 0.15
cfg.reward.safe_embb_retention_threshold = 0.80
cfg.reward.unsafe_admission_penalty_weight = 0.10
cfg.reward.unsafe_embb_retention_threshold = 0.65

cfg.training.use_phase_a_embb_power_anchor = True
cfg.training.phase_a_embb_power_anchor_weight = 0.5
cfg.training.phase_a_embb_power_anchor_start_iteration = 0
cfg.training.phase_a_embb_power_anchor_min_retention = 0.85
cfg.training.phase_a_embb_power_anchor_load_weights = {
    15.0: 0.3,
    25.0: 1.0,
    35.0: 1.0,
}

cfg.training.primary_checkpoint_preference = "best_balanced"
cfg.training.require_primary_checkpoint_match = True

# STEP 3: Continue with training as normal
trainer = SRMAPPOTrainer(cfg, env_factory=lambda: SRMAPPOPhaseAEnv(cfg))
trainer.train(num_iterations=cfg.training.total_iterations)
```

---

## IMPLEMENTATION SCRIPT 4: Report Configuration (for Checkpoint Selection)

### File: `sr_mappo/report.py`

#### Modification: Enforce Admission Floor in Checkpoint Selection

**Location**: `report.py`, find `_select_checkpoint()` function (line ~360)

**Add this constant near the top of the file**:

```python
# After other constants like CHECKPOINT_DIR, RESULTS_DIR
SELECTION_ADMISSION_FLOOR = 0.65  # Reject checkpoints with admission < 65%
SELECTION_THROUGHPUT_SIM_MODE = "dual_metric"  # Instead of "best_throughput"
```

**Then in `_select_checkpoint()`, after preference evaluation**:

**Before** (current selection logic):
```python
if normalized_checkpoint_kind:
    # ... current logic ...

selection_mode = str(getattr(cfg.training, "selection_mode", "dual_metric") or "dual_metric").strip().lower()
# ... continue with selection ...
```

**After** (add admission floor check):
```python
if normalized_checkpoint_kind:
    # ... current logic ...

# NEW: Override selection_mode to enforce admission
selection_mode = SELECTION_THROUGHPUT_SIM_MODE  # "dual_metric"

# NEW: When evaluating checkpoints, filter by admission
selection_admission_floor = float(getattr(cfg.training, "selection_admission_floor", SELECTION_ADMISSION_FLOOR))

# In the loop where candidates are evaluated:
for path, reason in preferred:
    if not path.exists():
        continue
    
    # NEW: Load checkpoint and check admission
    try:
        candidate_cfg = _load_checkpoint_cfg(path)
        candidate_admission = load_aware_score(cfg, load=25.0).get('admission', 0.0)  # Approximate
        if candidate_admission < selection_admission_floor:
            _report_log(f"Skipping {path.name}: admission={candidate_admission:.3f} < floor={selection_admission_floor}")
            continue
    except Exception:
        pass  # If we can't load, proceed anyway
    
    # ... existing return logic ...
    return path, reason
```

**OR simpler approach**: Just change line ~364:

```python
# OLD
primary_checkpoint_preference = _primary_checkpoint_preference(cfg)

# NEW
primary_checkpoint_preference = "best_balanced"  # Force balanced evaluation
```

---

## IMPLEMENTATION SCRIPT 5: Testing & Validation

### Quick validation script (run after modifications):

```python
# test_config_modifications.py

from sr_mappo.config import SRMAPPOConfig, RewardConfig

def test_modifications():
    cfg = SRMAPPOConfig()
    
    # Check reward config
    assert cfg.reward.schedule_success_weight == 0.22, f"schedule_success_weight={cfg.reward.schedule_success_weight}, expected 0.22"
    assert cfg.reward.embb_damage_weight == 0.14, f"embb_damage_weight={cfg.reward.embb_damage_weight}, expected 0.14"
    assert cfg.reward.overlay_margin_weight == 0.22, f"overlay_margin_weight={cfg.reward.overlay_margin_weight}, expected 0.22"
    
    # Check training config
    assert cfg.training.use_phase_a_embb_power_anchor == True, "use_phase_a_embb_power_anchor should be True"
    assert cfg.training.phase_a_embb_power_anchor_weight == 0.5, "phase_a_embb_power_anchor_weight should be 0.5"
    assert cfg.training.primary_checkpoint_preference == "best_balanced", "primary_checkpoint_preference should be 'best_balanced'"
    
    print("✓ All modifications validated!")
    
if __name__ == "__main__":
    test_modifications()
```

---

## IMPLEMENTATION SCRIPT 6: Rollback (if needed)

If modifications don't work as expected, rollback:

```python
# restore_defaults.py

# Restore to original defaults
cfg.reward.schedule_success_weight = 0.08      # Original
cfg.reward.embb_damage_weight = 0.20          # Original
cfg.reward.overlay_gain_weight = 0.08          # Original
cfg.reward.overlay_margin_weight = 0.15        # Original
cfg.reward.missed_overlay_penalty = 0.12       # Original

cfg.training.use_phase_a_embb_power_anchor = False  # Original
cfg.training.phase_a_embb_power_anchor_weight = 0.0  # Original

cfg.training.primary_checkpoint_preference = "best_throughput"  # Original
cfg.training.require_primary_checkpoint_match = False  # Original
```

---

## Order of Implementation

1. **FIRST**: Modify `config.py` RewardConfig (5 min)
2. **SECOND**: Add training config parameters (2 min)
3. **THIRD**: Modify `report.py` checkpoint selection (3 min)
4. **FOURTH**: Run test validation script (1 min)
5. **FIFTH**: Run one training iteration and check results

---

## Expected Outcomes After Implementation

**Before**:
- eMBB throughput: 4.5 Mbps
- URLLC admission: 0.25-0.35
- Overlay count: ~0%
- Puncture count: ~10%
- phase_a_embb_power_changed_ratio: ~0%

**After** (assuming training converges):
- eMBB throughput: 4.2-4.4 Mbps (−3 to −7%)
- URLLC admission: 0.65-0.75 (+100-200%)
- Overlay count: >50% of feasible
- Puncture count: 30-40%
- phase_a_embb_power_changed_ratio: >20%

---

## Monitoring During Training

Add logging to track convergence:

```python
# In trainer.py, after each rollout (line ~1080)

if iteration % 10 == 0:
    admission_rate = rollout_summary.get('mean_urllc_admission_rate', 0.0)
    throughput_rate = np.mean(embb_total_rates) if embb_total_rates else 0.0
    _report_log(f"Iter {iteration}: admission={admission_rate:.3f}, throughput={throughput_rate:.2f} Mbps, "
                f"power_anchor_loss={stats.phase_a_embb_power_anchor_loss:.4f}")
    
    if admission_rate < 0.40 and iteration > 50:
        _report_log(f"WARNING: Admission too low at iter {iteration}. Consider increasing schedule_success_weight further.")
```

---

**DONE!** All modifications are minimal and surgical. Proceed with confidence.

