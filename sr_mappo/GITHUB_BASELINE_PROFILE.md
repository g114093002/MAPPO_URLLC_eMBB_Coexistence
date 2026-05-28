## GitHub Baseline Profile

This project now distinguishes between two lines:

1. Reference curve
- Purpose: visual target / comparison only.
- Current reference: `bench_owner_topk_mix0573smooth_v3_e1`
- Note: this line includes additional mix-specific tuning and should not be used as the public default baseline.

2. GitHub baseline
- Purpose: clean, explainable baseline suitable for publication and reproduction.
- Principle: keep only global mechanisms and disable mix-specific experimental hacks by default.

### Included by default

- Global low-load / transition behavior
- Global stage-1 served-user-by-load control
- Global touch fraction behavior
- Standard greedy diagnostics and trace outputs

### Disabled by default

- `REQ05` mix-specific tuning
- `REQ07` mix-specific tuning
- Final-gate bypass / relaxation experiments
- Greedy continuity experiments
- One-off single-load boosts

### Clean baseline result files

- `3:7`: `bench_owner_topk_unfixedsubset_3_7_e10`
- `5:5`: `bench_owner_topk_mainline_5_5_e10`
- `7:3`: `bench_owner_topk_unfixedsubset_7_3_e10`

### Known limitations

- `3:7` has a candidate-feasibility dip at `load=18`
- `5:5` has a local irregularity around `load=21`

### Important default behavior

The following experiments are opt-in only and must be enabled explicitly through environment variables:

- `SR_MAPPO_OWNER_TOPK_REQ05_ENABLE`
- `SR_MAPPO_OWNER_TOPK_REQ07_ENABLE`
- `SR_MAPPO_GREEDY_CONTINUITY_ENABLE`
- `SR_MAPPO_GREEDY_DISABLE_FINAL_GATE`
- `SR_MAPPO_GREEDY_PUNCTURE_REL_MARGIN`

In particular, `SR_MAPPO_GREEDY_CONTINUITY_ENABLE` is now disabled by default so the GitHub baseline does not silently include continuity shaping.
