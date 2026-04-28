# SR-MAPPO Checkpoint Selection & Configuration Analysis

## Key Finding: Software Architecture Biases Towards Throughput

The system has **multiple layers of default bias towards throughput-only optimization**. Below are the exact settings and mechanisms.

---

## 1. PRIMARY CHECKPOINT PREFERENCE SETTINGS

### Current Default Configuration
**Location**: [`config.py`](config.py#L286-L287)

```python
primary_checkpoint_preference: str = "best_throughput"  # DEFAULT
require_primary_checkpoint_match: bool = False          # DEFAULT
checkpoint_eval_scope: str = "representative_load"      # DEFAULT = Load 25.0
```

### Valid Preference Options
- **`"best_throughput"`** (DEFAULT) ← Prefers throughput-optimized checkpoint
- `"best_balanced"` - Optimizes balanced metric
- `"best_multiload_frontier"` - Pareto frontier across multiple loads
- `"best_multiload_tp_power"` - Throughput/power tradeoff frontier

### How It Works
**Location**: [`report.py:_primary_checkpoint_preference()`](report.py#L327-L333)

```python
def _primary_checkpoint_preference(cfg: Optional[SRMAPPOConfig] = None) -> str:
    cfg = cfg or SRMAPPOConfig()
    preference = str(
        getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") or "best_throughput"
    ).strip().lower()
    return preference if preference in {"best_throughput", "best_balanced", "best_multiload_frontier", "best_multiload_tp_power"} else "best_throughput"
```

**Default behavior**: If invalid/unset → defaults to `"best_throughput"`

---

## 2. CHECKPOINT SELECTION MECHANISM

### Selection Hierarchy
**Location**: [`report.py:_select_checkpoint()`](report.py#L351-650)

The selection logic uses **ordered preference lists** that vary based on configuration:

#### A. When `has_loadwise_selection_constraints` is TRUE
```python
preferred = [
    *multiload_preferred,          # Multiload frontier/tp_power if applicable
    *balanced_preferred,           # Best balanced if preferred
    report_best_floor_throughput,  # Prefers floor-constrained throughput
    best_throughput,
    *comparative_preferred,        # vs. each greedy baseline
    best_reward, best_alias        # Fallback values
]
```

#### B. When `selection_admission_floor > 0.0`
```python
preferred = [
    *multiload_preferred,
    *balanced_preferred,
    report_best_throughput,       # Throughput moves earlier
    best_floor_throughput,
    *comparative_preferred,
    best_reward
]
```

#### C. When `selection_mode == "throughput_only"`
```python
preferred = [
    *multiload_preferred,
    *balanced_preferred,
    report_best_throughput,       # Prioritizes throughput
    best_throughput,
    *comparative_preferred,
    *comparative_best,
    report_best_reward,
    best_reward
]
```

#### D. Default (`selection_mode == "dual_metric"`)
```python
preferred = [
    *multiload_preferred,
    *balanced_preferred,
    *comparative_preferred,       # Compares vs. baselines FIRST
    report_best_throughput,       # Then throughput
    report_best_reward,
    *comparative_best,
    best_throughput,
    best_reward
]
```

**Key Observation**: Even in default dual_metric mode, if comparative checkpoints aren't found, falls back to `best_throughput` before `best_reward`.

---

## 3. SELECTION MODE & ADMISSION FLOOR

### Configuration Options
**Location**: [`config.py:TrainingConfig`](config.py#L229-L233)

```python
selection_mode: str = "dual_metric"                              # DEFAULT
selection_admission_floor: float = 0.0                           # DEFAULT (no constraint)
selection_admission_floor_by_load: Dict[float, float] = {}       # Load-specific floors
selection_admission_floor_ratio_to_baseline: float = 0.0         # Ratio constraint
```

### Optional Load-Specific Constraints
```python
selection_power_ratio_ceiling_by_load: Dict[float, float] = {}
selection_throughput_ratio_floor_by_load: Dict[float, float] = {}
selection_puncture_ratio_ceiling: float = 1.0
selection_puncture_ratio_floor_by_load: Dict[float, float] = {}
selection_overlay_ratio_ceiling_by_load: Dict[float, float] = {}
selection_reliability_floor: float = 0.0
```

### Selection Floor Application
**Location**: [`report.py:3220-3230`](report.py#L3220-L3230)

```python
this_floor = float(
    selection_floor_for_load(
        load,
        getattr(cfg.training, 'selection_admission_floor_by_load', {}),
        fallback_floor=float(getattr(cfg.training, 'selection_admission_floor', 0.0) or 0.0),
    )
)
this_floor_pass = float(rl_adm >= this_floor - 1e-9)
```

**Impact**: When floor > 0, checkpoints failing floor constraint are filtered out BEFORE scoring.

---

## 4. URLLC ADMISSION WEIGHTING IN REWARD FUNCTION

### Default Weights
**Location**: [`config.py:RewardConfig`](config.py#L51-L53)

```python
terminal_urllc_admission_weight: float = 4.00      # Weight for positive contribution
terminal_urllc_admission_target: float = 0.78      # Target admission rate
terminal_urllc_admission_penalty: float = 12.00    # Penalty for missing target
```

### Reward Computation
**Location**: [`env.py:190-232`](env.py#L210-L232)

```python
# 1. Positive contribution (proportional to admission ratio)
terminal_urllc_admission = (
    load_reward_weights["terminal_urllc_admission_weight"] * admission_ratio
)
team_reward += terminal_urllc_admission

# 2. Target shortfall penalty (if below target)
admission_target = float(load_reward_weights["terminal_urllc_admission_target"])
admission_penalty = float(getattr(self.rl_cfg.reward, "terminal_urllc_admission_penalty", 0.0))
if admission_target > 0.0 and admission_penalty > 0.0:
    shortfall = max(admission_target - admission_ratio, 0.0)
    terminal_urllc_admission_shortfall = -admission_penalty * shortfall
    team_reward += terminal_urllc_admission_shortfall
```

### Load-Adaptive Weights
**Location**: [`load_aware.py:10-16`](load_aware.py#L10-L16)

URLLC admission weights **automatically adjust by load**:

| Load | Admission Weight | Target | Admission Threshold | Impact |
|------|-----------------|--------|-------------------|--------|
| 5.0  | 0.8  | 0.55 | Low urgency - throughput prioritized |
| 10.0 | 1.2  | 0.62 | Increasing URLLC importance |
| 15.0 | 1.8  | 0.68 | Moderate URLLC focus |
| 20.0 | 2.4  | 0.72 | URLLC becomes significant |
| 25.0 | 2.8  | 0.74 | **Highest URLLC weight at rep load** |

---

## 5. LOAD-AWARE CHECKPOINT SELECTION SCORING

### Selection Score Formula
**Location**: [`load_aware.py:49-56`](load_aware.py#L49-L56)

```python
def load_aware_selection_score(actual_load, throughput_excess, admission_gap, 
                               puncture_loss_gap, overlay_retention_gap, power_ratio):
    weights = _SELECTION_FORMULA[nearest_reference_load(actual_load)]
    power_ratio_excess = max(float(power_ratio) - 1.0, 0.0)
    return float(
        weights["throughput"] * float(throughput_excess)              # + bonus for higher throughput
        + weights["admission"] * float(admission_gap)                 # + bonus for higher admission
        - weights["puncture_loss"] * float(puncture_loss_gap)         # - penalty for higher loss
        + weights["overlay_retention"] * float(overlay_retention_gap) # + bonus for higher retention
        - weights["power"] * power_ratio_excess                       # - small power penalty
    )
```

### Per-Load Scoring Weights
**Location**: [`load_aware.py:17-22`](load_aware.py#L17-L22)

| Load | Throughput | Admission | Puncture Loss | Overlay | Power | Mix Weight |
|------|-----------|-----------|---------------|---------|-------|------------|
| 5.0  | 4.0 | 0.5 | -0.5 | 0.2 | -0.05 | 0.10 |
| 10.0 | 3.5 | 0.8 | -0.8 | 0.3 | -0.05 | 0.15 |
| 15.0 | 2.5 | 1.5 | -1.5 | 0.5 | -0.05 | 0.20 |
| 20.0 | 1.8 | 2.5 | -2.0 | 0.8 | -0.05 | 0.25 |
| 25.0 | 1.5 | 3.0 | -2.2 | 1.0 | -0.05 | 0.30 |

**Key Insight**: 
- At **Load 5.0** (lightest): Throughput weight = 4.0, Admission weight = 0.5 → **8x more weight on throughput**
- At **Load 25.0** (representative/heaviest): Throughput weight = 1.5, Admission weight = 3.0 → **2x more weight on admission**
- **System is optimized for load 25.0** (see `REPRESENTATIVE_LOAD = 25.0` in report.py:82)

---

## 6. PHASE A eMBB POWER ANCHOR SETTINGS

### Configuration
**Location**: [`config.py:TrainingConfig`](config.py#L293-L297)

```python
use_phase_a_embb_power_anchor: bool = False                     # DEFAULT: DISABLED
phase_a_embb_power_anchor_weight: float = 0.0                  # No weight when disabled
phase_a_embb_power_anchor_start_iteration: int = 0             # When to enable
phase_a_embb_power_anchor_load_weights: Dict[float, float] = {}  # Load-specific weights
phase_a_embb_power_anchor_min_retention: float = 0.0           # Minimum eMBB retention
phase_a_embb_power_anchor_positive_gap_only: bool = True       # Only anchor when beneficial
```

**Location**: [`env.py:EnvAdapterConfig`](config.py#L126)

```python
allow_phase_a_embb_power_adjustment: bool = False  # DEFAULT: DISABLED
phase_a_embb_power_delta_values: List[float] = [] # Power adjustment options
```

### Enablement Check
**Location**: [`trainer.py:166-172`](trainer.py#L166-L172)

```python
def phase_a_embb_power_anchor_enabled(cfg, iteration: int) -> bool:
    if not bool(getattr(cfg.training, "use_phase_a_embb_power_anchor", False)):
        return False  # ← Blocked by first default
    if not bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)):
        return False  # ← Blocked by second default
    start_iteration = max(int(getattr(cfg.training, "phase_a_embb_power_anchor_start_iteration", 0) or 0), 0)
    if start_iteration <= 0:
        return True
    return int(iteration) >= start_iteration
```

**Default State**: DISABLED (both config options default to False)

### Target Computation
**Location**: [`trainer.py:491-550`](trainer.py#L491-L550)

When enabled, computes power adjustment targets based on:
- Current load
- eMBB retention rates
- Utility gaps (overlay vs. puncture)
- Loss ratios

---

## 7. THROUGHPUT-ONLY TRAINING BIAS

### Where Throughput-Only Can Be Selected
1. **Via `selection_mode = "throughput_only"`** directly in config
2. **Via comparative checkpoints** if no specific preference is set
3. **Fall-through default** when primary preference not found

### Training Data Showing Different Experiment Configurations
**Location**: [`experiments.py:480-610`](experiments.py)

Different experiments override URLLC weights:

```python
# Low-priority URLLC
terminal_urllc_admission_weight = 1.25  (vs default 4.00)
terminal_urllc_admission_penalty = 3.00 (vs default 12.00)

# Minimal URLLC
terminal_urllc_admission_weight = 1.0   (vs default 4.00)
terminal_urllc_admission_penalty = 2.0  (vs default 12.00)

# High-priority URLLC
terminal_urllc_admission_weight = 2.25  (vs default 4.00)
terminal_urllc_admission_target = 0.70  (vs default 0.78)
```

---

## 8. CURRENT SYSTEM DEFAULTS SUMMARY

| Setting | Default Value | Bias |
|---------|--------------|------|
| `primary_checkpoint_preference` | `"best_throughput"` | ✓ Throughput |
| `checkpoint_eval_scope` | `"representative_load"` (25.0) | Load-adaptive, admission-focused at this load |
| `selection_mode` | `"dual_metric"` | Neutral, but comparative-first ordering |
| `selection_admission_floor` | 0.0 | No constraint |
| `require_primary_checkpoint_match` | False | Allows fallback away from preference |
| `use_phase_a_embb_power_anchor` | False | Disabled |
| `allow_phase_a_embb_power_adjustment` | False | Disabled |
| Terminal URLLC Admission Weight | 4.00 | ✓ Explicit URLLC weighting |
| Terminal URLLC Admission Target | 0.78 | ✓ Requires 78% admission |
| Terminal URLLC Admission Penalty | 12.00 | ✓ Heavy penalty for missing target |

---

## 9. EFFECTIVE SELECTION PATHS FOR DEFAULT CONFIG

### With Default Settings
1. Load representative_load = 25.0
2. primary_checkpoint_preference = "best_throughput"
3. selection_mode = "dual_metric"
4. No admission floor
5. No loadwise constraints

**Selection order**:
```
1. multiload checkpoints (if applicable)
2. balanced checkpoints (not with best_throughput preference)
3. comparative checkpoints (vs each greedy mode) ← FIRST substantive attempt
4. report_best_throughput ← Falls back here
5. report_best_reward
6. best_throughput
7. best_reward
8. report_best (alias)
9. best (alias)
10. final
11. latest_iter
12. latest_any
```

### With URLLC Emphasis
To prioritize URLLC admission:
```python
primary_checkpoint_preference = "best_balanced"           # OR "best_multiload_frontier"
require_primary_checkpoint_match = True                   # Require it
selection_admission_floor = 0.70                          # Enforce floor
selection_admission_floor_by_load = {25.0: 0.75}         # Load-specific
low_damage_admission_objective = True                     # Use different scoring
```

---

## 10. RECOMMENDATIONS

### To Enforce URLLC Prioritization:
1. Set `primary_checkpoint_preference = "best_balanced"` (not throughput)
2. Set `require_primary_checkpoint_match = True` (enforce it)
3. Set `selection_admission_floor = 0.70` (admission must be ≥70%)
4. Set `selection_admission_floor_by_load = {5.0: 0.60, 10.0: 0.65, 15.0: 0.70, 20.0: 0.75, 25.0: 0.78}`
5. Optionally set `low_damage_admission_objective = True` (changes scoring weights)

### To Use Multiload Pareto Frontier:
1. Train with `checkpoint_eval_scope = "all_loads"`
2. Set `primary_checkpoint_preference = "best_multiload_frontier"`
3. This requires checkpoints like `sr_mappo_best_multiload_frontier.pt` to exist

### To Debug Current Checkpoint Selection:
- Check `checkpoint_reason` in report outputs (e.g., "report_best_throughput" vs "best_balanced")
- Monitor `selection_floor_pass` vs `selection_floor_violation` metrics
- Review `loadwise_selection_score` for per-load contributions to overall score

---

## 11. REPRESENTATIVE LOAD SELECTION

**Location**: [`report.py:82`](report.py#L82)

```python
REPRESENTATIVE_LOAD = 25.0  # Load used for all report generation
```

This means:
- All checkpoints are selected based on performance at load 25.0
- At this load, admission weight is **3.0** (highest of all loads)
- This load is the "hardest" scenario in the default eval set

---

## File Reference Links
- Configuration: [`config.py`](config.py)
- Checkpoint Selection: [`report.py:327-650`](report.py#L327-L650)
- Load-Aware Scoring: [`load_aware.py`](load_aware.py)
- Reward Computation: [`env.py:190-250`](env.py#L190-L250)
- Training Control: [`trainer.py:166-180`](trainer.py#L166-L180)
- Experiments: [`experiments.py:480-610`](experiments.py#L480-L610)
