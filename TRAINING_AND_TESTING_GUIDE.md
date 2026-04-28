# SR-MAPPO Training, Testing, and Plotting Guide

This guide explains exactly how to run training, testing, and figure generation, where outputs go, and which scripts do what.

---

## 0) Where to Run

Always run commands from the project root:

```
d:\URLLC_eMBB_Coexisting
```

---

## 1) Full Pipeline (Train + Evaluate + Report)

This is the **one-shot** command that trains a model, evaluates it, and generates all report figures:

```bash
python -m sr_mappo.train
```

Two clean experiment lines are now supported:

Fair fixed-baseline coexistence:

```bash
python -m sr_mappo.train --experiment fair_fixed_baseline
```

Full joint learning (Phase-0 + Phase-A):

```bash
python -m sr_mappo.train --experiment full_joint_learning
```

What it does internally:

1. **Train** MAPPO (`trainer.run_training_loop`)
2. **Rescreen checkpoints** for report-best model (`evaluate.rescreen_checkpoints_for_report`)
3. **Generate report figures** (`report.generate_report`)
4. **Generate diagnostics** (policy diagnostics script if present)
5. **Write a markdown log** (`run_docs.write_training_run_markdown`)

Outputs:

- Checkpoints:  
  `d:\URLLC_eMBB_Coexisting\checkpoints\`
- Report figures + metrics JSON:  
  `d:\URLLC_eMBB_Coexisting\sr_mappo\results\`
- Training summary markdown:  
  `d:\URLLC_eMBB_Coexisting\sr_mappo\results\LATEST_TRAINING_RUN.md`

---

## 2) Only Generate Report Figures (No Training)

If you already have checkpoints and just want plots:

```bash
python -m sr_mappo.report
```

Or explicitly select an experiment line:

```bash
python -m sr_mappo.report --experiment fair_fixed_baseline
python -m sr_mappo.report --experiment full_joint_learning
```

What it does:

1. Loads checkpoints with this priority:
   `*_report_best_throughput.pt` -> `*_report_best_reward.pt` -> `*_report_best.pt` ->
   `*_best_throughput.pt` -> `*_best_reward.pt` -> `*_best.pt` -> `*_final.pt`
   -> latest `*_iter*.pt`
2. Runs a sweep over load levels.
3. Generates all figures into `sr_mappo/results/`.

Greedy baseline mode is controlled by:

- `sr_mappo/config.py -> TrainingConfig.greedy_baseline_mode`
- `original`
- `matched_fixed_embb`
- `frozen_json`

Main comparison figures `01~03` now plot three curves together:

- `Original Greedy`
- `Matched Greedy`
- `SR-MAPPO`

`greedy_baseline_mode` still controls which greedy variant is treated as the
selected baseline for timeslot-style figures and metadata.

If `greedy_baseline_mode = "frozen_json"`, set:

- `TrainingConfig.frozen_greedy_metrics_path`

Key files produced:

- `01_core_kpis_vs_load.png`
- `02_mode_diagnostics_vs_load.png`
- `03_fairness_and_uav_vs_load.png`
- `04_training_diagnostics.png`
- `05_slot_timeline_and_activity.png`
- `06_single_slot_mode_maps.png`
- `07_timeslot_kpis_comparison.png`
- `08_timeslot_power_mode_comparison.png`
- `09_timeslot_action_summary.png`
- `sr_mappo_report_metrics.json`

---

## 3) Publication-Style Figures (MAPPO-Only)

```bash
python -m sr_mappo.publication_figures
```

Outputs:

```
sr_mappo/results/publication_figures/
```

These are MAPPO-only, IEEE-style figures.

---

## 4) Puncture / Overlay Diagnostic Figures

```bash
python -m sr_mappo.puncture_pressure_figures
```

Outputs go to:

```
sr_mappo/results/puncture_pressure_figures/
```

These are the diagnostic plots about overlay feasibility, shield correction, advantage by mode, etc.

---

## 4.5) Freeze a Greedy Baseline JSON

To freeze a reusable greedy baseline payload:

```bash
python -m sr_mappo.freeze_greedy_baseline --mode original
```

or

```bash
python -m sr_mappo.freeze_greedy_baseline --mode matched_fixed_embb
```

You can also freeze against a specific experiment line:

```bash
python -m sr_mappo.freeze_greedy_baseline --experiment fair_fixed_baseline --mode matched_fixed_embb
python -m sr_mappo.freeze_greedy_baseline --experiment full_joint_learning --mode original
```

This writes a JSON payload under:

```bash
sr_mappo/results/
```

You can then point `TrainingConfig.frozen_greedy_metrics_path` to that file and set:

```bash
greedy_baseline_mode = "frozen_json"
```

---

## 5) Where to Change Training Length

File:

```
sr_mappo/config.py
```

Key knobs:

- `TrainingConfig.total_iterations`  
  Total training iterations.

- `TrainingConfig.rollout_horizon`  
  Steps collected per iteration.

- `TrainingConfig.eval_every`  
  How often evaluation runs.

- `TrainingConfig.eval_episodes_per_load`  
  Number of eval episodes per load.

---

## 6) Where Checkpoints Are Saved

By default:

```
d:\URLLC_eMBB_Coexisting\checkpoints\
```

Files:

- `sr_mappo_phase_a_iterXX.pt`
- `sr_mappo_phase_a_best.pt`
- `sr_mappo_phase_a_best_reward.pt`
- `sr_mappo_phase_a_best_throughput.pt`
- `sr_mappo_phase_a_final.pt`
- `sr_mappo_phase_a_report_best.pt`
- `sr_mappo_phase_a_report_best_reward.pt`
- `sr_mappo_phase_a_report_best_throughput.pt`

---

## 7) How the “Report-Best” Checkpoint Is Selected

Selection happens in:

```
sr_mappo/evaluate.py
```

Function:

```
rescreen_checkpoints_for_report()
```

It:

1. Screens all saved checkpoints
2. Rescores them on the fixed report load sweep
3. Saves two winners:
   `*_report_best_reward.pt` by mean team reward
   `*_report_best_throughput.pt` by mean eMBB throughput
4. Copies the throughput winner to `*_report_best.pt` for report-time default use

---

## 8) How to Run Only One Episode (Quick Smoke Test)

```bash
python -m sr_mappo.runner
```

This is just a sanity check that env + model can step.

---

## 9) Greedy Baseline Only (No RL)

To run the Greedy baseline simulation:

```bash
python Greedy/main.py
```

## 9.5) Compare Under One Experiment Line

```bash
python -m sr_mappo.compare --experiment fair_fixed_baseline
python -m sr_mappo.compare --experiment full_joint_learning
```

Outputs:

```
Greedy/results/
```

---

## 10) If Training Is Too Slow

Try reducing:

- `total_iterations`
- `rollout_horizon`
- `eval_episodes_per_load`

Then run:

```bash
python -m sr_mappo.train
```

---

## 11) Common Output Locations (Summary)

| Item | Location |
| --- | --- |
| Checkpoints | `d:\URLLC_eMBB_Coexisting\checkpoints\` |
| Report figures | `sr_mappo/results/` |
| Diagnostics | `sr_mappo/results/puncture_pressure_figures/` |
| Publication figures | `sr_mappo/results/publication_figures/` |
| Training log | `sr_mappo/results/LATEST_TRAINING_RUN.md` |

---

If you want this guide to also include **step-by-step CLI examples** (for different loads or different checkpoints), tell me your exact run style and I’ll add them.
