# V11 vs V10 Analysis (mix 5:5, total users 27~72, e20)

## Scope

- Main comparison target:
  - `v10`: `global_frontier_clean_mappo_vs_greedy_total27_72_e20_seed246813579_mix55_v10_iter540_preview_5_5_e20`
  - `v11`: `global_frontier_clean_mappo_vs_greedy_total27_72_e20_seed246813579_mix55_v11_phasea_power_effective_5_5_e20`
- Secondary reference:
  - `v10_no_urllc_powerhead_iter800`

Metrics source:
- [v10 main report](/D:/URLLC_eMBB_Coexisting/sr_mappo/results/global_frontier_clean_mappo_vs_greedy_total27_72_e20_seed246813579_mix55_v10_iter540_preview_5_5_e20/sr_mappo_report_metrics.json)
- [v10 no powerhead report](/D:/URLLC_eMBB_Coexisting/sr_mappo/results/global_frontier_clean_mappo_vs_greedy_total27_72_e20_seed246813579_mix55_v10_no_urllc_powerhead_iter800_5_5_e20/sr_mappo_report_metrics.json)
- [v11 report](/D:/URLLC_eMBB_Coexisting/sr_mappo/results/global_frontier_clean_mappo_vs_greedy_total27_72_e20_seed246813579_mix55_v11_phasea_power_effective_5_5_e20/sr_mappo_report_metrics.json)

Preset source:
- [v10/v11 experiment definitions](/D:/URLLC_eMBB_Coexisting/sr_mappo/experiments.py:4593)

## Executive Summary

v11 is better than the main v10 reference in the overall throughput/power/min-rate tradeoff, but it does not clearly improve URLLC admission. The main win is that the Phase-A eMBB power head is no longer fully dead; however, its actual effect is still small. The next step should be to keep the v11 throughput/power gains while recovering admission at medium-high and high loads.

## Main Outcome

Average across loads:

| Version | eMBB throughput (Mbps) | URLLC admission | Scheduled packets | Min-rate satisfaction ratio | Total power (mW) |
|---|---:|---:|---:|---:|---:|
| v10 iter540 preview | 77.640 | 0.7015 | 340.43 | 0.5347 | 4613.35 |
| v10 no URLLC powerhead iter800 | 77.953 | 0.7031 | 340.88 | 0.5416 | 4589.26 |
| v11 phasea power effective | 78.322 | 0.7011 | 339.33 | 0.5426 | 4576.20 |

Interpretation:

- Against the main v10 reference, v11 improves:
  - eMBB throughput: `+0.682 Mbps`
  - min-rate satisfaction ratio: `+0.0079`
  - total power: `-37.15 mW`
- But v11 does not improve:
  - URLLC admission: `-0.0003`
  - scheduled packets: `-1.09`

Relative to the stronger `v10_no_urllc_powerhead_iter800` reference:

- v11 still has the best throughput.
- v11 still has the best power.
- v11 only barely beats v10-no-powerhead on min-rate.
- v11 is slightly worse on admission and scheduled packets.

## Loadwise Differences (v11 - v10 main)

| Total users | eMBB users | Throughput gap (Mbps) | Admission gap | Scheduled packets gap | Min-rate ratio gap | Min-rate user-count gap | Power gap (mW) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 27 | 13 | +1.427 | +0.0031 | +1.20 | +0.0046 | +0.060 | -13.32 |
| 36 | 18 | +0.105 | +0.0184 | +6.65 | +0.0239 | +0.430 | -23.54 |
| 45 | 23 | +1.274 | -0.0061 | -2.55 | +0.0172 | +0.395 | -68.18 |
| 54 | 27 | -0.497 | -0.0080 | -4.70 | +0.0059 | +0.160 | -21.48 |
| 63 | 31 | +1.154 | -0.0017 | -1.45 | -0.0047 | -0.145 | -84.05 |
| 72 | 36 | +0.630 | -0.0076 | -5.70 | +0.0004 | +0.015 | -12.33 |

Takeaways:

- Clear wins:
  - `27 users`: throughput up, power down, no meaningful admission damage.
  - `36 users`: strongest admission gain in the whole sweep.
  - `45 users`: throughput and min-rate improve together, with large power reduction.
- Weak / problematic points:
  - `54 users`: throughput and admission both drop.
  - `63 users`: throughput improves, but min-rate slips slightly.
  - `72 users`: throughput still up, but admission and admitted packet count fall.

This pattern suggests v11 is stronger at low-mid load efficiency, but the high-load admission protection is not yet robust.

## What Actually Changed from v10 to v11

From [sr_mappo/experiments.py](/D:/URLLC_eMBB_Coexisting/sr_mappo/experiments.py:4593):

v10:

- `allow_phase_a_power_on_keep = False`
- `embb_power_delta_limit = 0.06`
- stronger anti-movement regularization
- `phase_a_power_target_write_ratio = 0.04`

v11:

- `allow_phase_a_power_on_keep = True`
- `embb_power_delta_limit = 0.12`
- earlier start: `phase_a_embb_power_start_iteration = 120` instead of `180`
- runtime relaxations:
  - `phase_a_embb_power_max_downscale_per_step = 0.12`
  - `phase_a_positive_boost_cap = 0.04`
  - `phase_a_power_guard_floor_margin = 0.01`
  - `phase_a_embb_power_scale_bound_relax = 1.35`
  - `phase_a_embb_power_scale_floor_relax = 1.50`
  - `phase_a_embb_power_scale_cap_relax = 1.15`
  - `phase_a_negative_only_embb_power_repair = True`
- softer regularization:
  - `phase_a_power_change_penalty_weight = 0.02` from `0.04`
  - `phase_a_power_smooth_delta_penalty_weight = 0.02` from `0.04`
  - `phase_a_power_delta_l2_penalty_weight = 0.04` from `0.08`
  - `phase_a_power_target_write_ratio = 0.10` from `0.04`

Design intent is clear: v11 tries to make Phase-A power actually move.

## Did Phase-A Power Really Become Effective?

Yes, but only partially.

In v10 main:

- `phase_a_embb_power_final_executed_mean_delta = 0.0`
- `phase_a_power_total_power_reduction_mean = 0.0`
- `phase_a_power_negative_executed_ratio = 0.0`

In v11:

- `phase_a_embb_power_final_executed_mean_delta` is nonzero at every load
- `phase_a_power_negative_executed_ratio` is nonzero and rises with load
- `phase_a_power_total_power_reduction_mean` is nonzero

However the magnitude is still small:

- Mean executed delta is only around `0.005`
- Negative executed ratio is only around `0.006`
- Intercell reduction is effectively zero in practice

Conclusion:

- v11 successfully turns the power head on.
- v11 does not yet make the power head strong enough to dominate behavior.

## Reliability of the Comparison

There is still a major report/eval caveat:

- In the v11 report, `pairing_fairness_audit.paired_all = false`
- `mismatched_episode_pairs = 120`
- mismatch keys are mainly:
  - `overlay_graph_hash`
  - `repair_sequence_hash`

Meaning:

- This is not a perfect same-scene paired comparison.
- Large directional conclusions are still useful.
- Small gaps should not be over-interpreted.

This should be fixed before using close-score differences to choose the next checkpoint.

## What Improved

- Throughput improved versus both v10 references.
- Power improved versus both v10 references.
- Min-rate ratio improved versus the main v10 reference.
- Phase-A eMBB power is no longer completely inactive.

## What Is Still Wrong

- Admission is not clearly improved.
- High-load admission remains fragile.
- `54` and `72` total users are still weak spots.
- Phase-A power changes are real but too weak.
- Comparison fairness is still compromised by mismatched overlay/repair trajectories.

## Recommended Next Step

### Priority 1: Fix evaluation pairing

Before more tuning, fix same-scenario pairing for report/eval so:

- `overlay_graph_hash` matches
- `repair_sequence_hash` matches

Without this, it will be hard to tell whether future `v12` gains are real or evaluation noise.

### Priority 2: Build v12 around high-load admission recovery

Target loads:

- `54 users`
- `63 users`
- `72 users`

Success criterion:

- Keep v11 throughput advantage
- Keep v11 power reduction
- Recover admission/scheduled packets to at least v10-no-powerhead level

### Priority 3: Make Phase-A matter more

The current Phase-A signal is too weak. Likely next knobs to try:

- Increase the effective write/use frequency without destabilizing:
  - `phase_a_power_target_write_ratio`
  - `phase_a_power_write_ratio_penalty_weight`
- Slightly relax movement suppression further:
  - `phase_a_power_change_penalty_weight`
  - `phase_a_power_smooth_delta_penalty_weight`
  - `phase_a_power_delta_l2_penalty_weight`
- Increase useful runtime headroom only if guards hold:
  - `embb_power_delta_limit`
  - `phase_a_embb_power_max_downscale_per_step`

### Priority 4: Do not chase throughput alone

v11 already shows that throughput can improve while admission weakens at high load. So v12 should be judged primarily on:

1. high-load admission
2. min-rate preservation
3. then throughput/power

## Suggested v12 Direction

Conservative plan:

- Keep the v11 Phase-A activation structure.
- Do not make the power head much more aggressive until pairing is fixed.
- Add more direct pressure on high-load admission retention.
- Evaluate again with `e20` first, then confirm with `e50`.

If forced to choose one axis only for the next edit, choose:

- `high-load admission recovery`, not `more throughput`, because throughput is already in a good place.

