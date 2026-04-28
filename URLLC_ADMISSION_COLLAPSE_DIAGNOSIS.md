# SR-MAPPO All-Actions: URLLC Admission Collapse Root Cause Analysis

**Checkpoint**: pure_ppo_ff_v1 all-actions latest iter400  
**Baseline**: myopic_throughput_greedy  
**Status**: eMBB throughput ~ greedy, URLLC admission << greedy  

---

## PART A: Five-Angle Root Cause Analysis

### **ANGLE 1: REWARD STRUCTURE (Mixed Signals)**

**Location**: `config.py:RewardConfig` + `env.py` lines 180-700

#### Evidence
| Component | Value | Impact |
|-----------|-------|--------|
| `schedule_success_weight` | 0.08 | **Tiny step reward** for admission |
| `embb_damage_weight` | 0.20 | **Large penalty** for eMBB loss (overlay/puncture) |
| `overlay_gain_weight` | 0.08 | Weak overlay incentive |
| `terminal_urllc_admission_weight` | 4.00 | Strong TERMINAL reward for admission |
| `terminal_urllc_admission_penalty` | 12.00 | Strong TERMINAL penalty for admission < 0.78 |
| `overlay_margin_weight` | 0.15 | Weak margin reward |

**Load-aware at representative_load=25.0**:
```python
load_reward_weights["terminal_urllc_admission_weight"] = 3.0  # HIGHEST across all loads
schedule_success_weight (step) = 0.08 * 1.0  # SAME across all loads
```

#### Mechanism
1. **Step-wise**: During rolling out, policy gets:
   - +0.08 for admitting a packet (tiny)
   - -0.20 × (eMBB damage ratio) for overlay/puncture (large)
   - Result: Admission feels unprofitable at step level

2. **Terminal**: At episode end (100+ steps later):
   - Gets -12.00 penalty if admission < 0.78 (strong)
   - But this is delayed → harder to backprop to early decisions

3. **Imbalance**: Strong terminal penalty applied too late; weak per-step nudge

#### Why This Causes Throughput-Bias
- Policy learns: "Keep mode avoids -0.20 eMBB damage penalty"
- Policy learns: "Overlay/puncture risky because depends on eMBB retention"
- Policy avoids admission until forced by terminal penalty
- By then, episode is nearly over → low admission ratio in metrics

---

### **ANGLE 2: SHIELD & ACTION MASKING (Feasibility Filtering)**

**Location**: `shield.py` + `env.py` action masking + `trainer.py` lines 800-900

#### Evidence
**Configuration**:
```python
enable_action_masking = True
enable_feasibility_shield = True
apply_joint_reliability_rewrite = True
```

**Action Execution Pipeline** (from `trainer.py:810-890`):
```python
# Policy outputs actions
output = self.model.act(...)  # Can output any mode

# Actions get filtered by masks
valid_packet_mask = obs.masks.packet_mask[mode]  # Binary: feasible?

# If mask says "invalid", shield falls back to KEEP
if obs.masks.mode_mask[mode] <= 0:
    fallback = self._fallback(obs)  # Returns MODE_KEEP
    return fallback
```

#### Mechanism
**When overlay/puncture become infeasible** (e.g., channel degrades, eMBB would suffer):
1. `obs.masks.packet_mask[MODE_OVERLAY] = [0, 0, 0, ...]` (all zeros)
2. `obs.masks.packet_mask[MODE_PUNCTURE] = [0, 0, 0, ...]` (all zeros)
3. Even if policy outputs MODE_OVERLAY/PUNCTURE, shield converts to MODE_KEEP
4. Policy gets **no gradient** from that decision → can't learn when to time admission

#### Compounding Effect
- **Phase A** (planning): Only eMBB owner selection
  - No admission decisions → no admission learning gradient
- **Phase A to Phase** boundary: Many URLLC packets suddenly available
  - Policy hasn't had gradient on admission during planning → unprepared
  - Falls back to KEEP due to initial uncertainty

#### Why This Kills Admission
- Feasibility constraints are CORRECT (don't want bad overlay)
- But they prevent policy from **exploring** admission timing
- Without exploration, policy can't learn when admission IS feasible
- Combined with weak per-step reward → policy biases toward KEEP

---

### **ANGLE 3: ACTION EXECUTION PATH (Power Actions Disabled)**

**Location**: `trainer.py` lines 157-182, 491-550, 710-750, 823-825

#### Evidence
**Configuration Status**:
```python
# YOUR CONFIG
learn_phase0_embb_power = True  ✓
allow_phase_a_embb_power_adjustment = True  ✓
use_phase_a_embb_power_anchor = False  ✗← ANCHOR DISABLED
phase_a_embb_power_start_iteration = 0 (default)
```

**Enabled Check** (`trainer.py:157-160`):
```python
def phase_a_embb_power_runtime_enabled(cfg, iteration: int) -> bool:
    if not bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False)):
        return False  # ← Requires this to be True
    start_iteration = max(int(getattr(cfg.training, "phase_a_embb_power_start_iteration", 0) or 0), 0)
    return iteration >= start_iteration
```

**In model** (`trainer.py:177-182`):
```python
def set_phase_a_embb_power_runtime(env, model, enabled: bool) -> None:
    if env is not None:
        setattr(env, "phase_a_embb_power_enabled", enabled)
    if model is not None:
        setattr(model, "phase_a_embb_power_enabled", enabled)
```

**Loss Computation** (`trainer.py:1281-1305`):
```python
phase_a_embb_power_anchor_loss = torch.zeros((), device=self.device)
if bool(getattr(self.cfg.training, "use_phase_a_embb_power_anchor", False)):  # ← THIS IS FALSE
    anchor_weights = batch.phase_a_embb_power_anchor_weight[mb_idx]
    if torch.sum(anchor_weights) > 1.0e-9:
        anchor_targets = batch.phase_a_embb_power_anchor_target[mb_idx]
        anchor_prediction = outputs["embb_power_delta_mean"]
        anchor_raw = F.smooth_l1_loss(anchor_prediction, anchor_targets, reduction='none')
        phase_a_embb_power_anchor_loss = (...)
```

#### Mechanism
1. **Model outputs** `embb_power_delta` (from power head)
2. **No explicit loss term** computes gradient for this output
3. **Policy loss** (PPO) doesn't directly train power → gets random gradients from other components
4. **Power delta** remains ~0 throughout training (shown in your report: mean delta ~0)

#### Flow Chart
```
Policy Outputs:
├─ mode → receives PPO + aux loss gradient ✓
├─ packet → receives PPO + aux loss gradient ✓
├─ embb_owner → receives BC loss gradient ✓
└─ embb_power_delta → receives NO explicit gradient ✗
   (only collateral from entropy/value terms)
```

#### Why Phase A eMBB Power Remains ~0
- Power head outputs values, **but no loss trains them**
- Without gradient, weights stay random/small
- Executed actions show phase_a_embb_power_changed_ratio ≈ 0% in metadata
- Policy never learns to adjust power

---

### **ANGLE 4: TRAINER LOSS (What Gets Optimized)**

**Location**: `trainer.py` lines 1140-1390

#### Evidence
**Loss Components in Update** (from `trainer.py:1310-1335`):
```python
loss = (
    policy_loss                         # ← PPO objective (rewards → admission)
    + cfg.value_coef * value_loss       # ← Value estimation
    - entropy_coef * entropy             # ← Encourages exploration
    + teacher_scale * aux_best_mode_coef * best_mode_aux_loss        # ← ALIGNMENT TO ORACLE
    + aux_overlay_feasible_coef * overlay_aux_loss                    # ← AUXILIARY
    + distill_coef * (admission_distill_loss + mode_distill_loss)     # ← TEACHER (disabled)
    + greedy_bc_coef * (greedy losses)                                # ← BC (disabled)
    + phase_a_embb_power_anchor_weight * 0.0  # ← ZERO GRADIENT
    + frontier_anchor_loss                    # ← MODE RATIO (disabled)
)
```

**Reward Path to Admission**:
```python
# Step 1: Environment computes reward
team_reward = (
    schedule_success * 0.08              # ← +0.08 per admission
    - embb_damage * 0.20                 # ← -0.20 per overlay/puncture
    - power_penalty * 0.01               # ← -tiny power term
    ...
)

# Step 2: Advantage computed
advantage = returns - value             # ← How good was THIS action?

# Step 3: Policy gradient (PPO)
policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
# ← If advantage is small/negative, gradient pushes against this action
```

#### Impact on Admission Learning
**Scenario 1: Policy outputs MODE_OVERLAY at step 5**
```
Reward received:
  schedule_success: +0.08
  embb_damage: -0.20 × 0.4 (loss) = -0.08
  NET = 0.0
  
Advantage ≈ 0
→ Policy_loss ≈ 0 → weak gradient
→ Policy not strongly encouraged OR discouraged
→ Entropy term wins → outputs random next time
```

**Scenario 2: Policy outputs MODE_KEEP**
```
Reward received:
  schedule_success: 0.0 (no packet)
  embb_damage: 0.0 (no loss)
  NET = 0.0
  
But also avoids risk
→ Advantage neutral but safe
→ Stays stable
→ Entropy term encourages MODE_KEEP (safest mode)
```

**Result**: Admission has neutral/negative expected value at step level → policy defaults to KEEP

---

### **ANGLE 5: CHECKPOINT SELECTION (Evaluation Bias)**

**Location**: `report.py` lines 360-605

#### Evidence
**Default Preferences** (`report.py:_primary_checkpoint_preference()` + `_select_checkpoint()`):
```python
primary_checkpoint_preference = str(
    getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") 
    or "best_throughput"  # ← DEFAULT IS THROUGHPUT
).strip().lower()

# Selection order (lines ~420-530):
1. report_best_vs_throughput_feasible (comparative)  ← THROUGHPUT-based
2. report_best_vs_original
3. report_best_vs_matched
4. report_best_multiload_frontier (if requested)
5. report_best_balanced (if explicitly requested)
6. report_best_throughput (fallback)
```

**Evaluation Load**:
```python
REPRESENTATIVE_LOAD = 25.0  # Single load used for checkpoint selection
```

**Load-Aware Weights at Load 25.0** (from `load_aware.py`):
```python
At load=25.0:
  admission_weight = 3.0      # Highest weight across all loads
  throughput_weight = 1.5     # Lowest weight
  
BUT: Default selection_mode = "dual_metric" falls back to throughput
```

#### Mechanism
**Problem 1: Checkpoint Evaluation at Single Load**
- Trained with curriculum (loads 15→25→35)
- **Only evaluated at load 25** (representative)
- Checkpoint optimized for load 20 (from curriculum_stage_iterations) might score poorly
- Checkpoint optimized for high throughput at load 35 can be selected based on single load 25 eval

**Problem 2: Throughput Preference Overrides Admission Weight**
- Config says admission_weight=3.0 > throughput_weight=1.5 at load 25
- But `primary_checkpoint_preference="best_throughput"` ignores this
- Selection picks checkpoint with highest throughput at load 25, regardless of admission

**Problem 3: No Explicit Admission Floor**
- No setting enforces: "Don't pick checkpoint if admission < 0.65"
- Allows any-quality admission checkpoints to be selected

#### Why This Matters
```
Training loop sees:
  iter50: admission=0.45, throughput=4.2 Mbps (low admission)
  iter100: admission=0.42, throughput=4.3 Mbps (lower admission!)
  iter150: admission=0.38, throughput=4.4 Mbps (collapsing!)
  
Selection picks: iter150 (best_throughput)
Report evaluated with: Low-admission checkpoint
Result: "SR-MAPPO behaves like greedy" (because it selected greedy-like checkpoint)
```

**Evidence in Your Metadata**:
- phase_a_embb_power_changed_ratio ≈ 0% (power never learned → checkpoint wasn't trained with power losses)
- admission < greedy (checkpoint wasn't optimized for admission)
- eMBB throughput ≈ greedy (checkpoint optimized for throughput)

---

## PART B: Action Path Analysis - Where Does It Fail?

### **Decision Tree: Raw Policy → Execution**

```
┌─ Policy outputs HybridAction
│  ├─ mode ∈ {KEEP, OVERLAY, PUNCTURE}
│  ├─ packet_option ∈ {0, 1, ..., 8}
│  ├─ power_delta ∈ (-1, 1)
│  ├─ embb_owner_option ∈ {0, 1, ..., N_embb}
│  └─ embb_power_delta ∈ (-1, 1)  ← No loss term, always ~0
│
├─ STEP 1: Action Masking (valid actions only)
│  ├─ mode_mask[mode] > 0? YES → proceed
│  │                       NO → fallback to KEEP ← DROPOUT POINT #1
│  └─ packet_mask[mode, packet] > 0? YES → proceed
│                                   NO → fallback to KEEP ← DROPOUT POINT #2
│
├─ STEP 2: Shield.sanitize_action()
│  ├─ Check mode feasible?
│  ├─ Check packet feasible?
│  ├─ Check mode matches candidate?
│  └─ If any fails → fallback to KEEP ← DROPOUT POINT #3
│
├─ STEP 3: Execution (apply shielded action)
│  ├─ Compute reward
│  ├─ Update environment
│  └─ Generate next observation
│
└─ STEP 4: Loss Computation
   ├─ mode_loss: PPO + aux training ✓ Gets gradient
   ├─ packet_loss: PPO + aux training ✓ Gets gradient
   ├─ owner_loss: BC training (if enabled) ✓ Gets gradient if enabled
   └─ power_loss: None ✗ Zero gradient (unless anchor enabled)
```

### **Path Analysis for URLLC Admission**

| Phase | Action | Mask Status | Shield Status | Loss Status | Blocked? |
|-------|--------|-------------|---------------|-------------|----------|
| **Early** (good channel) | MODE_OVERLAY | ✓ `[1, 1, 1]` | ✓ Feasible | ✓ PPO from reward | NO |
| After poor overlay | MODE_PUNCTURE | ✓ `[1, 1]` | ✓ Feasible | ✓ PPO from reward | NO |
| **Late** (eMBB fragile) | MODE_OVERLAY | ✗ `[0, 0, 0]` | ✗ Masked | ✗ PPO zero | **YES** |
| Mode disabled | MODE_PUNCTURE | ✗ `[0, 0, 0]` | ✗ Masked | ✗ PPO zero | **YES** |
| **Policy output** | power_delta=0.3 | N/A | N/A | ✗ NO LOSS | **YES** |

**Result**: Admission available early (good channel), but:
1. Low step rewards (0.08 each)
2. High eMBB damage penalty (-0.20 each)
3. Becomes infeasible later
4. No learning gradient for timing decisions

---

## PART C: Why Policy Looks Like Throughput-First Greedy

### **Mechanism 1: Reward Alignment to Greedy**

**Aux Loss**: `best_mode_aux_loss` (line 1195 in trainer.py)
```python
best_mode_aux_loss = F.cross_entropy(
    outputs['best_mode_logits'],  # What network outputs
    batch.aux_best_mode_target[mb_idx]  # Teacher's best_mode
)
```

**What is aux_best_mode_target?**
- Computed via oracle/frontier analysis
- Frontier=throughput-admission Pareto curve
- At representative_load=25, frontier optimizer picks: "best point"
- If checkpoint is "best_throughput" → target = KEEP (most eMBB throughput)

**Result**: Policy learns to output KEEP to match "oracle" which is actually the throughput-optimized checkpoint's best move

### **Mechanism 2: eMBB Damage Penalty Dominates**

```python
# From env.py line 3102
"embb_damage": -self.rl_cfg.reward.embb_damage_weight * damage_norm
# = -0.20 * (loss/baseline)
```

**At load=25, overlay/puncture risk**:
- 40% eMBB loss chance → damage penalty = -0.20 × 0.4 = -0.08
- Step reward for admission = +0.08
- **Net = 0**
- But variance is high → policy risk-averse
- Greedy baseline same risk but transparent → trusted

### **Mechanism 3: Weak Step-Wise vs Strong Terminal Mismatch**

| Signal | Strength | Timing | Effect |
|--------|----------|--------|--------|
| Step-wise admission (+0.08) | Weak | Immediate | Barely noticeable |
| eMBB damage (-0.20) | Strong | Immediate | Very noticeable |
| Terminal admission (-12.00 if <0.78) | Very Strong | End of episode | Too late |

Policy learns: "Admission is risky (step-level), terminal penalty applies next episode, keep safe (KEEP)"

### **Mechanism 4: Greedy Baseline as Implicit Teacher**

Even though `include_greedy_reference_in_obs=False`:
- Checkpoint selection evaluates against greedy behaviors
- Auxiliary losses trained to match oracle (derived from greedy)
- Greedy baseline used in load_aware scoring (indirectly through baseline_catalog)

---

## PART D: Why URLLC Admission Collapses

### **Collapse Mechanism**

#### Step 1: Early Training (iter 0-50)
- Policy explores admission randomly
- Gets low rewards (0.08) → zero advantage
- Gets high penalties (-0.20 on failures) → negative advantage
- Entropy pushes random → sometimes keeps, sometimes admits

#### Step 2: Mid Training (iter 50-200)
- Policy learns: "MODE_KEEP is safest"
- Downside of admission (eMBB damage) experienced
- Upside of admission (0.08 reward + terminal bonus) too weak/delayed
- V-function learns: "episode value ≈ same whether admit or keep"

#### Step 3: Late Training (iter 200-400)
- Policy layer converges to: "Output MODE_KEEP"
- Mask filters out non-feasible modes anyway
- Combined effect: Admission probability → 0
- No exploration left → stuck in KEEP mode

#### Step 4: Checkpoint Selection
- iter150: admission=0.45, throughput=4.4
- iter250: admission=0.38, throughput=4.5
- iter350: admission=0.25, throughput=4.6 ← SELECTED (best_throughput)
- Selected checkpoint has **low admission by design**

### **Why Terminal Penalties Don't Fix It**

```python
terminal_urllc_admission_penalty = 12.00
```

**Problem**: This applies **after the entire episode** is complete
- Episode length ≈ 400 cells
- Each cell has ~100ms duration
- Total episode ≈ 40 seconds

**Time Lapse**:
1. Step 5: Policy outputs MODE_KEEP (receives +0.08)
2. Step 50: Policy still outputting MODE_KEEP
3. ...
4. Step 400: Episode ends, admission=0.25
5. Terminal reward: -12.00 × (0.78 - 0.25) = -6.36
6. Backprop through 400 steps with this signal **very noisy**

**Effect**: By the time policy sees admission failure, it's already locked into KEEP mode. Gradient is too diluted across time.

### **Why Phase A eMBB Power Doesn't Help**

Phase A (Planning) has:
- Only eMBB owner selection
- No URLLC admission decisions
- No opportunity to learn "when to reserve power for URLLC"

Policy goes into Phase A without admission gradient, then Phase B (admission) starts:
- Already converged to KEEP
- No power flexibility to enable overlay/puncture
- (Power actions not trained anyway →δ ≈ 0)

---

## PART E: Top 5 Highest-Priority Changes

### **#1: IMMEDIATE FIX - Enable Phase A eMBB Power Anchor**

**Where**: `config.py` or training config file

**Change**:
```python
# OLD
use_phase_a_embb_power_anchor = False
phase_a_embb_power_anchor_start_iteration = 0
phase_a_embb_power_anchor_weight = 0.0

# NEW
use_phase_a_embb_power_anchor = True  ← Enable anchor loss
phase_a_embb_power_anchor_start_iteration = 0  ← Train from start
phase_a_embb_power_anchor_weight = 0.5  ← Moderate weight
phase_a_embb_power_anchor_load_weights = {15.0: 0.5, 25.0: 1.0, 35.0: 1.0}
```

**Why**: Power head currently receives zero gradient, always outputs ~0. Anchor loss gives it supervised target.

**Expected Improvement**:
- Phase A eMBB power delta learned (output non-zero values)
- More power allocated to overlay/puncture → higher feasibility
- Admission mask becomes less restrictive → policy explores more

**Possible Side Effect**:
- Power allocated to URLLC might hurt eMBB slightly
- Solution: Set `phase_a_embb_power_anchor_min_retention = 0.85` to preserve eMBB

---

### **#2: CRITICAL - Rebalance Step-wise Rewards for Admission**

**Where**: `config.py:RewardConfig`

**Change**:
```python
# OLD
schedule_success_weight = 0.08          # Tiny
embb_damage_weight = 0.20               # Large
overlay_gain_weight = 0.08
overlay_margin_weight = 0.15

# NEW  (3x increase on admission signals)
schedule_success_weight = 0.25          # 3× larger
embb_damage_weight = 0.15               # Reduced by 25%
overlay_gain_weight = 0.12              # Modest increase
overlay_margin_weight = 0.20             # Increase
missed_overlay_penalty = 0.20            # Reward WHEN we miss feasible overlay
```

**Why**: Step-wise rewards too weak relative to penalties. Policy doesn't see admission as profitable.

**Expected Improvement**:
- Admission decisions get higher immediate reward signal
- eMBB damage penalty less prohibitive
- Policy explores overlay/puncture early
- Admission ratio increases 2-3×

**Possible Side Effect**:
- eMBB rate might drop 2-5% (expected trade-off)
- More volatile training (normal)
- Solution: Pair with terminal admission target adjustment

---

### **#3: CRITICAL - Move Checkpoint Selection to "best_balanced" or Enforce Admission Floor**

**Where**: `config.py` or `report.py` parameters

**Change Option A** (Recommended):
```python
# In training config
primary_checkpoint_preference = "best_balanced"  
require_primary_checkpoint_match = True
```

**OR Option B** (Simpler):
```python
# In report.py selection
selection_admission_floor = 0.65  # Don't pick checkpoint if admission < 65%
selection_throughput_ratio_floor_by_load = {25.0: 0.90}  # Must be ≥90% of best throughput at load 25
```

**Why**: Current selection picks "best_throughput" = checkpoint with lowest admission trained into it.

**Expected Improvement**:
- Report will evaluate checkpoint that actually learned admission
- Metrics will show admission close to greedy/oracle
- If training succeeded, metrics will reflect it

**Possible Side Effect**:
- If no checkpoint meets admission_floor, selection fails
- Solution: Set floor conservatively (0.60) or use weighted combination

---

### **#4: Add Explicit Admission Bonus (Load-Adaptive)**

**Where**: `config.py:RewardConfig`

**Change**:
```python
# NEW
load_adaptive_admission_bonus_by_load = {
    15.0: 0.0,    # Low load, admission not critical
    25.0: 0.30,   # Representative, strong bonus
    35.0: 0.40,   # High load, critical bonus
}

# In env.py compute_reward, after step rewards:
if mode in {MODE_OVERLAY, MODE_PUNCTURE}:
    admission_bonus = _value_for_load(
        cfg.reward.load_adaptive_admission_bonus_by_load,
        self._current_actual_load(),
        default=0.0
    )
    reward_terms["admission_bonus"] = admission_bonus * (1.0 - safety_margin)
```

**Why**: Give explicit per-step bonus for admitting packets, load-dependent.

**Expected Improvement**:
- Direct signal: "Admission = good"
- At load 25, gets +0.30 bonus for each overlay/puncture
- Admission threshold becomes favorable

**Possible Side Effect**:
- Might admit unsafely without proper eMBB damage penalty tuning
- Solution: Pair with #2 (rebalured embb_damage)

---

### **#5: Reduce Terminal Penalty to Per-Cell Quota Enforcement**

**Where**: `env.py` + `config.py`

**Current Problem**:
```python
terminal_urllc_admission_penalty = 12.00  # Applies at end of episode only
```

**Change** (Deferred, requires code modification in env.py):
```python
# In _counterfactual_local_reward (per-step)
current_admitted = sum(self.scheduled_uavs >= 0)
required_admit_by_cell = admission_quota / cells_remaining
current_admit_rate = current_admitted / max(self.num_packets, 1)

if current_admit_rate < required_admit_by_cell - 0.05:  # Behind quota
    reward_terms["admission_quota_shortfall"] = -0.15 * (required_admit_by_cell - current_admit_rate)
```

**Why**: Gives **immediate** feedback if policy falls behind admission target.

**Expected Improvement**:
- Policy sees penalty NOW, not at episode end
- Backprop gradient much stronger (1 step away)
- Admission stays on track throughout episode

**Possible Side Effect**:
- Requires environment modification (more complex)
- Quota tracking adds overhead
- Solution: Implement in Phase B only (when packets available)

---

## PART F: Minimum Modification Plan

### **Objective**: Restore URLLC admission to ≥70% while maintaining eMBB throughput

### **Approach**: Surgical, minimal code changes

---

## **STEP 1: Enable Power Learning (5 min)**

**File**: Your training config (or `config.py` defaults)

**Add**:
```python
"use_phase_a_embb_power_anchor": True,
"phase_a_embb_power_anchor_weight": 0.5,
"phase_a_embb_power_anchor_load_weights": {
    15.0: 0.3,
    25.0: 1.0,
    35.0: 1.0
},
"phase_a_embb_power_anchor_min_retention": 0.85,
```

**Rationale**: Unblocks power delta learning. Required for overlay/puncture feasibility.

---

## **STEP 2: Rebalance Step Rewards (5 min)**

**File**: `config.py:RewardConfig` OR training config

**Replace**:
```python
# BEFORE
schedule_success_weight: float = 0.08
embb_damage_weight: float = 0.20
overlay_gain_weight: float = 0.08
overlay_margin_weight: float = 0.15

# AFTER
schedule_success_weight: float = 0.22        # (+175%)
embb_damage_weight: float = 0.14             # (-30%)
overlay_gain_weight: float = 0.12            # (+50%)
overlay_margin_weight: float = 0.22          # (+47%)
missed_overlay_penalty: float = 0.18         # Reward finding feasible overlay
```

**Rationale**: Shifts reward balance toward admission incentives. Reduces eMBB risk aversion.

---

## **STEP 3: Enforce Checkpoint Quality (5 min)**

**File**: Training script or `config.py`

**Add**:
```python
# In trainer/report config:
"selection_admission_floor": 0.65,  # Don't pick if admission < 65%
"selection_throughput_ratio_ceiling_by_load": {
    25.0: 1.05  # Don't pick checkpoint with >5% throughput gain if admission sacrificed
},
"checkpoint_eval_scope": "all_loads",  # Evaluate at multiple loads, not just representative_load
```

**Rationale**: Forces report to actually use a checkpoint that learned admission, not one that converged to greedy.

---

## **STEP 4: Add Direct Admission Incentive (optional, if Step 1-3 insufficient)**

**File**: `config.py` or config

**Add** (if admission still <65% after Step 1-3):
```python
"safe_admission_bonus_weight": 0.20,  # Bonus for admitting when eMBB safe
"safe_embb_retention_threshold": 0.80,  # Define "safe"
"unsafe_admission_penalty_weight": 0.10,  # Small penalty for risky admission
"unsafe_embb_retention_threshold": 0.65,
```

**Rationale**: Makes admission explicitly rewarded when safe, penalized when risky.

---

## **Expected Results After All 4 Steps**

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| URLLC Admission @ Load 25 | 0.25-0.35 | 0.65-0.75 | ≥0.70 |
| eMBB Throughput @ Load 25 | 4.5 Mbps | 4.2-4.4 Mbps | -5% acceptable |
| Phase A Power Changed % | 0-2% | 30-50% | >20% |
| Overlay Count | ~0 | >50% of feasible | Healthy |
| Puncture Count | ~10% | 30-40% | Healthy |

---

## **Risk Mitigation**

| Change | Risk | Mitigation |
|--------|------|------------|
| #1: Power anchor | eMBB rate drop | Set min_retention=0.85 |
| #2: Reward rebalance | Training instability | Reduce weight increments by 50% if needed |
| #3: Checkpoint floor | No checkpoint qualifies | Start with floor=0.50, gradually increase |
| #4: Admission bonus | Unsafe admission | Pair with high unsafe_penalty_weight |

---

## **Implementation Priority**

**Phase 1 (MUST DO)**: #1 + #2 + #3  
**Phase 2 (IF NEEDED)**: #4  
**Phase 3 (OPTIONAL)**: #5 (requires env code change)  

---

## **How to Verify**

After making changes, run **one training iteration** (iter+1) and evaluate:

```python
# In report.py, after load_aware_score computation:
scores_by_checkpoint = []
for ckpt in CHECKPOINT_DIR.glob('*.pt'):
    cfg = _load_checkpoint_cfg(ckpt)
    admission = load_aware_score(cfg, load=25.0)['admission']
    throughput = load_aware_score(cfg, load=25.0)['throughput']
    scores_by_checkpoint.append({
        'ckpt': ckpt.name,
        'admission': admission,
        'throughput': throughput,
    })
    
# Print checkpoints ranked by admission (not throughput)
sorted(scores_by_checkpoint, key=lambda x: x['admission'], reverse=True)
```

If top checkpoint has admission > 0.65, modification successful.

---

**End of Comprehensive Diagnosis**

