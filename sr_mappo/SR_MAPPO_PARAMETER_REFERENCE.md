# SR-MAPPO Parameter Reference

## Which file is the "most complete" parameter file?

If you want the **most complete physical/system parameter file**, the main file is:

- [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)

That file contains:
- topology and time-frequency dimensions
- URLLC and eMBB power budgets
- reliability target
- NOMA / SIC thresholds
- channel-use calculation
- algorithm-side physical bounds

If you want the **SR-MAPPO-specific learning / action / reward / shield parameters**, the main file is:

- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)

So in practice:
- `Greedy/config.py` = physical model + communication constraints
- `sr_mappo/config.py` = RL behavior and training knobs

## Which file reflects the parameters of the actual SR-MAPPO experiments?

For the experiments and reports you have been looking at, the runtime scenario is assembled by:

- [compare.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/compare.py)
  Function: `_build_main_like_configs(...)`

That function overrides the raw defaults from `Greedy/config.py`, for example:
- `num_subcarriers = 8`
- `num_embb_users = 20`
- `num_urllc_users = 8`
- `urllc_cfg.target_error_probability = 1e-3`
- `urllc_cfg.power_limits = [24] * ...`
- `embb_cfg.power_limits = [23] * ...`
- `algo_cfg.power_upper_bound = 0.25`

So:
- if you want the **master parameter definitions**, read [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)
- if you want the **actual SR-MAPPO experiment profile you just ran**, read [compare.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/compare.py) together with [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)

## Power-related parameters

### 1. Per-user transmit-power budgets

These are in [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py):

- `URLLCConfig.power_limits`
  This is the per-URLLC-user max transmit power in dBm.

- `eMBBConfig.power_limits`
  This is the per-eMBB-user max transmit power in dBm.

### 2. Global algorithm power cap

Also in [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py):

- `AlgorithmConfig.power_upper_bound`
  This is the global max transmit power used by the allocator / environment after converting budgets into Watts.

This value directly affects whether overlay can be feasible, because both puncture and overlay required powers are clipped by it.

### 3. RL power action range

These are in [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py):

- `HybridActionConfig.continuous_power`
- `HybridActionConfig.power_delta_limit`
- `HybridActionConfig.initial_power_bins`

Current meaning:
- actor outputs `power_delta`
- requested power is:
  `required_power * (1 + power_delta_limit * power_delta)`
- then it is projected back into feasible range

### 4. Shield-side power projection

These are in [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py):

- `ShieldConfig.force_power_to_feasible_minimum`

And the actual projection code is in:

- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)
  Function: `_project_actual_power(...)`

Current logic:
- power is clipped to `[0, power_upper_bound]`
- if `force_power_to_feasible_minimum = True`, actual power is pushed back up to at least the required feasible power

That means the RL power head currently has limited freedom to reduce power below feasibility.

## Parameters that most strongly affect overlay feasibility

These are the main ones in [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py):

- `URLLCConfig.target_error_probability`
  Hard reliability target. Tighter target makes overlay harder.

- `AlgorithmConfig.min_noma_gain_ratio`
  Overlay is only considered if URLLC channel is at least this strong relative to the RB anchor eMBB.

- `AlgorithmConfig.embb_min_sic_snir_db`
  Even if URLLC can decode, overlay still fails if post-SIC eMBB SNIR is too low.

- `AlgorithmConfig.sic_residual_factor`
  Residual SIC interference. Larger residual makes overlay worse.

- `AlgorithmConfig.noma_retention_factor`
  Affects retained eMBB fraction under overlay.

- `AlgorithmConfig.power_upper_bound`
  If too low, overlay power becomes infeasible sooner.

## Parameters that affect whether SR-MAPPO can "see" overlay opportunities

These are in [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py):

- `HybridActionConfig.max_candidate_packets`
- `HybridActionConfig.min_overlay_candidate_slots`
- `HybridActionConfig.overlay_candidate_share`

These do not change physics, but they strongly affect whether the actor actually sees enough overlay-feasible candidates.

## Parameters that affect whether SR-MAPPO learns conservative puncture behavior

These are in [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py):

- `RewardConfig.schedule_success_weight`
- `RewardConfig.embb_damage_weight`
- `RewardConfig.puncture_extra_penalty`
- `RewardConfig.overlay_margin_weight`
- `RewardConfig.missed_overlay_penalty`
- `RewardConfig.terminal_embb_rate_weight`
- `RewardConfig.terminal_unscheduled_penalty`

And shield-side behavior:

- `ShieldConfig.enable_greedy_fallback`
- `ShieldConfig.resolve_packet_collisions`

## Where overlay is decided in code

Overlay feasibility is built in:

- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)
  Function: `_enumerate_candidates_for_cell(...)`

Important code-side overlay gates there are:
- URLLC error must be below `target_error_probability`
- post-SIC eMBB SNIR must exceed `embb_min_sic_snir_db`
- gain ratio must exceed `min_noma_gain_ratio`
- required power must stay within budget
- cross-UAV interference can kill an otherwise feasible overlay

Then SR-MAPPO only chooses among those candidates later.

## Bottom line

If you ask for only one "most complete" parameter file, use:

- [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)

because that is where the **communication model itself** is really defined.

If you ask for the SR-MAPPO behavior knobs on top of that, use:

- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)
