# SR-MAPPO Framework

This document describes the current `Shielded Recurrent Action-Masked MAPPO (SR-MAPPO)` implementation under `d:/URLLC_eMBB_Coexisting/sr_mappo`.

For a report-oriented full MDP / Dec-POMDP write-up, see:
- [SR_MAPPO_MDP_REPORT.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_MDP_REPORT.md)
- [SR_MAPPO_REWARD_FUNCTION.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_REWARD_FUNCTION.md)

It is written for the current UAV uplink coexistence setting only:
- multi-UAV uplink system
- eMBB + URLLC coexistence
- only `eMBB-only`, `eMBB-URLLC overlay`, and `puncture` are allowed
- admitted URLLC packets must satisfy the hard reliability constraint

## 1. RL Formulation

The current learning problem is a cooperative **Dec-POMDP with CTDE**.

- Agents: one agent per UAV (default design).
- Actor: decentralized, shared-parameter recurrent actor
- Critic: centralized recurrent critic
- Time scale: one environment step corresponds to one `(RB, minislot)` decision cell across all UAVs
- Episode: one scheduling slot (S minislots)

This remains a **Phase-A** design with an **optional Phase-0 planning stage**:
- when `learn_embb_baseline = True`, SR-MAPPO selects eMBB RB owners and eMBB power scales at slot start
- Phase-A then learns URLLC coexistence on top of those **policy-selected** anchors
- puncturing targets the learned RB owner
- overlay/keep/puncture remain the only coexistence modes (no full joint eMBB scheduling yet)

Phase-0 is no longer observation-free:
- each UAV agent sees the **current planning RB**
- top eMBB owner candidates for that `(UAV, RB)` are exposed through the planning observation
- planning features now include current RB index, remaining planning RBs, current owner, candidate gain/rate summaries, assigned-RB ratio, and current eMBB power scale

Current default:
- `learn_embb_baseline = False`
- the environment starts from a **deterministic non-greedy eMBB baseline** (`fixed_embb_baseline_policy = deterministic_max_gain`)
- this isolates Phase-A coexistence control before re-enabling learned Phase-0 planning

Two clean experiment lines are now supported:
- `fair_fixed_baseline`
  - `learn_embb_baseline = False`
  - PPO and matched Greedy share the same fixed eMBB baseline
  - this isolates **coexistence policy quality**
- `full_joint_learning`
  - `learn_embb_baseline = True`
  - Phase-0 planning and Phase-A coexistence are both learned
  - this tests whether joint learning can improve **end-to-end throughput**

## 2. Current MDP / Dec-POMDP Elements

### State

There are two observation levels.

#### Actor local observation for `(UAV j, RB k)` agent

The actor sees the current local cell context plus a truncated candidate list.

Local context includes:
- slot progress
- normalized minislot index
- normalized RB index (agent-specific)
- normalized UAV index
- remaining cell ratio in the slot
- current eMBB owner on this RB
- baseline per-RB eMBB rate
- eMBB minimum-rate margin on this RB
- baseline per-RB eMBB power
- associated eMBB ratio in this UAV
- associated URLLC ratio in this UAV
- already scheduled packet ratio in this UAV
- overlay ratio in this UAV
- puncture ratio in this UAV
- rolling overlay success EMA
- rolling puncture loss EMA
- urgent backlog ratio (packets near deadline for this UAV)
- currently available packet ratio (this minislot)

Candidate packet features are extracted for the top-`M` candidates, where currently:
- `M = max_candidate_packets = 8`

For each candidate packet, the current implementation uses:
- validity flag
- source user id normalized index
- channel gain summary
- puncture feasibility
- overlay feasibility
- puncture required power
- overlay required power
- puncture reliability margin
- overlay reliability margin
- puncture eMBB loss
- overlay eMBB retention
- puncture utility proxy
- overlay utility proxy
- packet age (normalized)
- remaining deadline slack (normalized)
- packet release minislot (normalized)

**Association semantics (fixed):**
- each URLLC packet is assigned to exactly one serving UAV at creation time
- candidates are pre-bucketed by UAV
- the same packet never appears in multiple UAV candidate lists

Removed contention features (by design):
- naturally associated to this UAV
- feasible‑UAV count / overlay‑feasible count / puncture‑feasible count
- packet contention score across UAVs

#### Critic global observation

The centralized critic sees a compact global summary including:
- overall slot progress
- current minislot index
- current RB index
- unscheduled packet ratio
- scheduled packet ratio
- remaining cell ratio in the slot
- per-UAV summaries of associated load, scheduled load, overlay count, and puncture count
- per-UAV rolling overlay success (EMA) and puncture damage (EMA)
- per-UAV urgent backlog ratio

### Action

Each **UAV agent** outputs a **hybrid action** that applies to the **current (RB, minislot) cell** for that UAV in the step schedule:

`a_j = (mode, packet_option, power_delta, embb_owner_option, embb_power_delta)`

#### Discrete action head 1: `mode`

The current implementation defines **3 mode actions**:
- `0 = KEEP`
- `1 = NOMA`
- `2 = PUNCT`

#### Discrete action head 2: `packet_option`

The current implementation defines **9 packet-option actions**:
- `0 = null / no packet`
- `1..8 = candidate packet 1..8`

The packet head now uses a **mode-conditioned packet mask**:
- if `mode = KEEP`, only packet option `0` is valid
- if `mode = NOMA`, only **overlay-feasible** candidates are exposed
- if `mode = PUNCT`, only **puncture-feasible** candidates are exposed

This is meant to cut down `raw_mode_infeasible_for_packet` errors before the shield has to rewrite them.
The candidate subset is **association-aware but not contention-aware**:
- urgent packets are still preserved
- overlay-feasible packets are still preserved
- tie-breaks no longer use contention or cross-UAV association signals

When `learn_embb_baseline = True`, a separate **eMBB owner mask** is also applied in Phase-0:
- `embb_owner_option = 0` is always valid
- `embb_owner_option = 1..M` is valid only if the RB has that many candidate eMBB users

#### Continuous action head: `power_delta`

This is a **1-D continuous action** and is currently active.

- raw output is squashed by `tanh`
- final range is approximately `[-1, 1]`
- it scales around the feasibility-required power with `power_delta_limit = 0.45`
- by default, execution now uses **continuous power directly**
- optional discrete binning (`initial_power_bins`) still exists, but it is **disabled by default**

### Reward

The current reward is now a **minimal step-first shaped reward**.
It no longer directly reuses the allocator utility as the reward core.

#### Local step reward

For an executed `(mode, packet, power)` action:

```text
r_local
= schedule_success_weight
- embb_damage_weight * damage_norm
- power_penalty_scale * power_norm
- power_projection_penalty * projection_norm
+ overlay_gain_weight * overlay_gain_norm          (if OVERLAY)
+ overlay_margin_weight * positive_margin_norm    (if OVERLAY and overlay is preferred)
- puncture_extra_penalty                          (if PUNCTURE)
- missed_overlay_penalty * overlay_gap_norm       (if PUNCTURE while overlay is better)
- keep_feasible_penalty                           (if KEEP while a good candidate exists)
- invalid_action_penalty                          (if shield fallback occurs)
- collision_rewrite_penalty                       (if shield rewrites due to packet collision)
```

Current coefficients:

- `schedule_success_weight = 0.02`
- `embb_damage_weight = 0.20`
- `puncture_extra_penalty = 0.05`
- `overlay_gain_weight = 0.08`
- `overlay_margin_weight = 0.15`
- `missed_overlay_penalty = 0.12`
- `power_penalty_scale = 0.01`
- `urgency_reward_weight = 0.08`
- `keep_feasible_penalty = 0.02`
- `invalid_action_penalty = 0.12`
- `collision_rewrite_penalty = 0.20`
- `power_projection_penalty = 0.05`

Normalization used in the environment:

- `damage_norm = chosen_mode_loss / puncture_loss`
- `overlay_gain_norm = (puncture_loss - overlay_loss) / puncture_loss`
- `positive_margin_norm` is a bounded normalized utility gap between overlay and puncture
- `urgency` is derived from packet age / remaining deadline slack
- `power_norm = actual_power / Pmax`
- `projection_norm` measures how much the raw continuous power was clipped or pushed up to the feasible minimum

#### Terminal reward / penalty

The slot-level terminal terms are:

```text
+ terminal_embb_rate_weight * normalized_embb_rate
+ terminal_embb_fairness_weight * jain_fairness
+ terminal_urllc_admission_weight * admission_ratio
- terminal_urllc_admission_penalty * max(admission_target - admission_ratio, 0)
- terminal_unscheduled_penalty * unscheduled_ratio
- terminal_embb_min_rate_penalty * avg_rate_shortfall
```

`use_greedy_terminal_reference = False` by default, so the throughput term is **absolute throughput**.

Current coefficients:

- `terminal_embb_rate_weight = 3.00`
- `terminal_embb_fairness_weight = 0.00`
- `terminal_urllc_admission_weight = 2.00`
- `terminal_urllc_admission_target = 0.78`
- `terminal_urllc_admission_penalty = 6.00`
- `terminal_unscheduled_penalty = 2.00`
- `terminal_embb_min_rate_penalty = 1.50`

#### Interpretation

The current reward is meant to:

- keep local step shaping minimal (schedule success + eMBB damage)
- enforce an **admission floor** (0.78) at the terminal level
- enforce **eMBB minimum-rate compliance** at the terminal level
- optimize **absolute throughput** (no greedy-relative term)

## 3. Full Action Inventory

For one UAV on the current `(RB, minislot)` cell, the logical action branches are:
- `KEEP`
- `NOMA(packet_1, power_delta)`
- `NOMA(packet_2, power_delta)`
- `NOMA(packet_3, power_delta)`
- `NOMA(packet_4, power_delta)`
- `NOMA(packet_5, power_delta)`
- `NOMA(packet_6, power_delta)`
- `NOMA(packet_7, power_delta)`
- `NOMA(packet_8, power_delta)`
- `PUNCT(packet_1, power_delta)`
- `PUNCT(packet_2, power_delta)`
- `PUNCT(packet_3, power_delta)`
- `PUNCT(packet_4, power_delta)`
- `PUNCT(packet_5, power_delta)`
- `PUNCT(packet_6, power_delta)`
- `PUNCT(packet_7, power_delta)`
- `PUNCT(packet_8, power_delta)`

So the effective logical action family is:
- `1` keep branch
- up to `8` overlay branches
- up to `8` puncture branches
- each with a continuous power adjustment

In addition, when `learn_embb_baseline = True`, there is a **Phase-0 planning action** per `(UAV, RB)`:
- `embb_owner_option` (choose RB owner or leave empty)
- `embb_power_delta` (scale eMBB power for that UAV-RB during planning)

## 4. Which Actions Are Actually Used Right Now

### Already active in training and inference

These are truly active today:
- `KEEP`
- `NOMA`
- `PUNCT`
- continuous `power_delta`
- `embb_owner_option` (only when `learn_embb_baseline = True`)
- `embb_power_delta` (only when `learn_embb_baseline = True`, used per UAV-RB)
- action masking
- feasibility shield
- no cross-UAV packet collision rewrite (packets are uniquely assigned)
- association-aware candidate truncation (contention metadata disabled)

### Defined in config but not really activated as a runtime feature

These exist in config or code structure, but are **not** currently driving a separate runtime behavior:
- `continuous_power`
  - currently always assumed `True`
  - there is no working discrete/continuous runtime switch yet
- `max_packets_per_step_view`
  - conceptually overlaps with `action.max_candidate_packets`
  - no separate runtime pathway beyond the candidate truncation already in use
- `use_all_uavs_as_candidate_servers`
  - association is fixed, so packets are pre-bucketed by their serving UAV
  - the flag is not yet a meaningful ablation switch
- `enable_greedy_fallback`
  - the code path exists but the default is now `False`
  - the intended behavior is to repair infeasible actions with shield logic or drop to `KEEP`
- `force_overlay_when_better`
  - the code path exists in the shield, but the default is now `False`
  - this avoids hidden overlay overrides during normal policy learning

## 5. Execution-Layer Training Alignment

The PPO buffer stores the **executed action**.

Concretely, at each decision cell:
- the actor first samples a raw `(mode, packet_option, power_delta)`
- the environment optionally sanitizes invalid local actions through the feasibility shield
- the environment then performs the same **joint reliability / joint assignment rewrite** used during `env.step()`
- the trainer recomputes `old_log_prob` on this **executed** action tuple
- PPO updates are therefore aligned with the action that actually generated the reward and transition

This avoids the earlier execution-layer mismatch where reward and transitions were driven
by a rewritten action not matching the stored log-probability.

The current implementation still keeps shielding active, but policy optimization is now attached to the executed action semantics.

## 6. Hard Reliability Enforcement

URLLC reliability is treated as a **constraint**, not as a soft performance preference.

The current SR-MAPPO execution path enforces reliability mainly through:
- local feasibility masking / sanitization
- joint per-step reliability rewrite across UAVs on the current `(RB, minislot)`
- overlay-quality penalties (retention / ratio / margin are soft, not hard filters)

Overlay quality gates are now **soft** (penalty-based). Hard filtering only enforces
URLLC reliability feasibility. The default runtime now also applies a **joint reliability
search / rewrite** before execution, so the executed step is constrained by the URLLC target
rather than only being measured post hoc.

When a URLLC packet is finally admitted:
- the executed reliability from the joint-feasible action is stored at step time
- the episode summary reports admitted URLLC reliability from these executed reliabilities

So the reported `urllc_reliability` in `sr_mappo/results` is now tied to the executed joint-feasible schedule itself, rather than to a fixed placeholder or a separate post-hoc target.

## 7. Evaluation / Report Baseline Consistency

The SR-MAPPO evaluation baseline is now consistent across:
- `sr_mappo/evaluate.py`
- `sr_mappo/report.py`

The project now supports three Greedy baseline modes:
- `original`: run the original standalone simulator via `Greedy/simulation.py -> run_single_allocation()`
- `matched_fixed_embb`: build the same fixed eMBB baseline used by SR-MAPPO and then execute greedy URLLC coexistence decisions on top of that anchor
- `frozen_json`: load a precomputed Greedy payload from disk without rerunning the simulator

So the report / compare / evaluate path can either:
- preserve the original Greedy baseline as the main reference
- add a matched-baseline Greedy ablation for fair anchor alignment
- or freeze a Greedy payload so repeated reports do not keep rerunning the baseline

This fixes the earlier measurement-layer mismatch where:
- training evaluation compared against env-local greedy
- but the report compared against the standalone simulator greedy run

So the current report figures and training-time evaluation are now measured against the same baseline semantics.

The current reporting workflow now distinguishes:
- `sr_mappo_phase_a_best_reward.pt`: the checkpoint chosen during training-time evaluation by fixed-load mean team reward
- `sr_mappo_phase_a_best_throughput.pt`: the checkpoint chosen during training-time evaluation by fixed-load mean eMBB throughput
- `sr_mappo_phase_a_report_best_reward.pt`: the reward-best checkpoint after post-training rescreen
- `sr_mappo_phase_a_report_best_throughput.pt`: the throughput-best checkpoint after post-training rescreen
- `sr_mappo_phase_a_report_best.pt`: an alias to the throughput-best report checkpoint, used by the default report path

So the default report figures under `sr_mappo/results/` are generated from the throughput-oriented report-best checkpoint unless the user explicitly overrides the checkpoint.

## 8. Default Training Parameters

These are the current default values from `sr_mappo/config.py`.

### Training loop
- `total_iterations = 700`
- `rollout_horizon = 256`
- `ppo_epochs = 4`
- `minibatch_size = 256`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_ratio = 0.2`
- `learning_rate = 2e-4`
- `entropy_coef = 0.015`
- `value_coef = 0.5`
- `max_grad_norm = 10.0`
- `device = cpu`

### Behavior cloning warm start
- `bc_episodes = 0`
- `bc_epochs = 0`
- `bc_batch_size = 256`
- `bc_learning_rate = 1e-3`

### Auxiliary losses
- `aux_best_mode_coef = 0.00`
- `aux_overlay_feasible_coef = 0.12`
- `aux_best_packet_coef = 0.00`
- `teacher_guidance_decay_start_frac = 0.10`
- `teacher_guidance_decay_end_frac = 0.35`
- `teacher_guidance_final_scale = 0.00`

### Evaluation / checkpointing
- `eval_every = 50`
- `eval_episodes = 4`
- `checkpoint_every = 50`
- `eval_episodes_per_load = 6`
- `run_name = sr_mappo_phase_a`
- `keep_best_non_worse_than_greedy = False`

### Curriculum used now
- `curriculum_loads = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]`
- `bc_loads = [8..18]`
- `eval_loads = [5, 10, 15, 20, 25]`
- `curriculum_stage_iterations = 40`
- `hardest_load_sampling_bias = 0.70`
- `second_hardest_load_sampling_bias = 0.20`

### Environment rollout semantics used now
- `multi_rb_agents = False` (default; 1 UAV = 1 agent)
- `early_terminate_when_all_packets_scheduled = False`
- `learn_embb_baseline = False`
- `fixed_embb_baseline_policy = deterministic_max_gain`
- `enable_feasibility_shield = True`
- `use_all_uavs_as_candidate_servers = False`
- `force_overlay_when_better = False`
- `bootstrap_with_discrete_power = False`

This is intentional. The environment no longer shortens easy episodes once all URLLC packets happen to be scheduled early. Each slot is therefore exposed to the agent as a full `8 RB × 8 minislot = 64-step` control horizon, which makes training pressure closer to the final report-time difficulty.

### Current URLLC hard target used in SR-MAPPO experiments
- `target_error_probability = 1e-3`
- equivalently, admitted URLLC reliability must remain above roughly `0.999`

### Fixed traffic-mix assumptions used in report/compare
- URLLC user ratio capped at **30%** for evaluation
- URLLC Poisson arrival rate `λ` is **fixed** (not scaled with load)

## 9. Auto-Generated Artifacts After Training

Running:

```powershell
python -m sr_mappo.train
```

now does **two things automatically**:
1. train the current default SR-MAPPO model
2. immediately regenerate the SR-MAPPO report figures in `sr_mappo/results`

It also auto-writes:
- `sr_mappo/results/LATEST_TRAINING_RUN.md`

Running:

```powershell
python -m sr_mappo.band_experts
```

now also auto-writes:
- `sr_mappo/results/LATEST_BAND_EXPERT_RUN.md`

## 9. Current Limitations

The current implementation is still limited in several important ways:
- it is still Phase-A only
- eMBB baseline allocation is not learned yet
- mode learning is still too puncture-dominant
- overlay usage is still weaker than the greedy baseline in many load regimes
- some config flags are placeholders rather than real ablation switches

## 10. Current Bottom Line

The current SR-MAPPO implementation is already a real hybrid-action recurrent MAPPO framework with shielding and masking, but it is **not yet** a final strong replacement for the greedy baseline.

Right now it is best described as:
- a valid RL framework
- a valid experimental branch
- a promising but still not fully convincing method

## 11. Why The Current Defaults Were Increased

The newer defaults deliberately make training harder and more aligned with the final evaluation protocol:

- the base training scenario is now constructed from the same `_build_main_like_configs()` helper used by report/compare
- the curriculum spends much more time on the hardest loads instead of over-sampling easy slots
- the rollout horizon was expanded from `128` to `256` so PPO sees more same-slot and cross-cell coordination effects before each update
- early termination was disabled so the policy must experience full-slot dynamics instead of only short, easy episodes

The intent is not to make testing easier. The intent is to stop training from being artificially easier than the final report setting.


Candidate selection notes
- `packet_option`: top-k candidate URLLC packets seen by each UAV actor
  - current implementation uses `max_candidate_packets = 8`
  - candidate truncation is mode-aware rather than pure utility top-k
  - at least `min_overlay_candidate_slots = 1` overlay-feasible packet is preserved whenever available
  - the remaining slots are filled by best-utility packets
