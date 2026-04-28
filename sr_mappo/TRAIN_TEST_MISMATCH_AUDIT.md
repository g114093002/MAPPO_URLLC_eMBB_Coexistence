# SR-MAPPO Train/Test Mismatch Audit

## Short answer

Based on the current codebase, the main problem does **not** appear to be:

- training on an easy environment
- then testing on a much harder environment

The current problem is much closer to:

- training on the **same scenario family**
- but with **too little effective experience**
- under a **very hard hybrid-action / joint-coordination problem**

So the issue is more:

- `insufficient training scale + difficult control problem`

than:

- `train/test scenario mismatch`

## 1. Training and report use the same base scenario profile

### Training environment

In [trainer.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/trainer.py), `build_default_components(...)` sets:

- `num_subcarriers = 8`
- `num_embb_users = 20`
- `num_urllc_users = 8`
- `shadowing_std = 6.0`
- `los_probability = 0.8`
- `urllc_cfg.packet_lengths = [160, 180, 200]`
- `urllc_cfg.target_error_probability = 1e-3`
- `urllc_cfg.power_limits = [24] * ...`
- `embb_cfg.power_limits = [23] * ...`
- `algo_cfg.power_upper_bound = 0.25`
- `sim_cfg.urllc_poisson_rate = 12`

### Report / evaluation scenario

In [compare.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/compare.py), `_build_main_like_configs(...)` uses the same values.

That means the base physical/system profile is aligned.

## 2. Load range seen in training already covers test

In [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py):

- `curriculum_loads = [10, 15, 20, 25, 30, 40, 50]`
- `eval_loads = [10, 20, 30, 40, 50]`

So training is **not** missing the report loads.

If anything:
- training sees a superset of report loads
- report is not harder in terms of average UE load per UAV

## 3. Why the current training is still not enough

### Training scale is still modest

Current training config:

- `total_iterations = 300`
- `rollout_horizon = 128`

That is:

- `300 x 128 = 38,400` environment steps

In this environment:

- `1 episode = 1 slot`
- `1 slot = 8 RB x 8 minislot = 64 steps`

So this is about:

- `38,400 / 64 = 600 slot-episodes`

This is not tiny, but for this problem it is still not large.

Why?

Because the agent is not solving a simple discrete scheduling task. It is solving:

- recurrent multi-agent control
- hybrid action (`mode + packet + continuous power`)
- hard URLLC feasibility
- packet collision coordination
- joint reliability coupling across UAVs

For that class of problem, `~600 slot-episodes` is still closer to:

- an early serious training run

than:

- a saturated training budget

## 4. The environment is hard even during training

The training environment is already structurally hard because:

1. `Phase-A` only controls coexistence, not the eMBB anchor layout.
   So the policy is optimizing on top of a fixed upstream structure.

2. Overlay opportunities are genuinely sparse.
   The diagnostics already showed feasible overlay ratios on the order of:
   - roughly `0.4% ~ 2%` or similarly low depending on the exact figure set

3. Shield still rewrites many actions.
   So policy autonomy is still limited.

4. Packet collision and joint reliability rewriting remain active bottlenecks.

This means the agent is **not** being trained on an artificially easy toy version.

## 5. The more realistic concern: effective experience per state regime is too low

Even if the nominal load range matches, the effective experience per difficult regime is not large.

Because training is spread across:

- 7 curriculum loads
- many random seeds
- many topology/channel realizations
- varying packet arrivals

So the policy may only see a modest number of:

- truly high-conflict states
- overlay-feasible-but-fragile states
- collision-heavy joint states

per load regime.

That is a much more plausible explanation for weak generalization than "test is harder than train."

## 6. What this means for your current interpretation

The current result should be interpreted as:

- the model is **not failing because report loads are outside training support**
- the model is **failing because the control problem is still too hard for the current data budget and action structure**

So the current diagnosis priority is:

1. increase useful training exposure
2. reduce action-space difficulty / improve coordination
3. only then worry about more subtle train-test mismatch stories

## 7. Practical next step

If you want to test this hypothesis directly, the clean experiment is:

1. keep the exact same train/test scenario family
2. increase training scale significantly
3. do not change the report loads

For example:

- `total_iterations = 800 ~ 1200`
- keep `rollout_horizon = 128` or raise to `256`
- increase `eval_episodes_per_load`

If SR-MAPPO improves under the same report loads, then the problem was training scale / coordination difficulty, not scenario mismatch.

## Bottom line

The code currently suggests:

- **No major train/test scenario mismatch**
- **Yes, the agent is still under-trained for how hard this environment is**

So the more accurate statement is:

> It is not that training is too easy and testing is too hard; it is that training and testing are drawn from the same family, but the policy still does not get enough effective experience to solve the full joint-coordination problem well.
