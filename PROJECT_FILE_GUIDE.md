# Project File Guide (What Each File Does)

This guide lists the purpose of each major file so you can edit the right place quickly.

---

## Greedy/ (Baseline Simulator)

- [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)  
  All system parameters: topology, time structure, bandwidth, noise, LoS/NLoS params, power limits, algorithm knobs.

- [channel_model.py](/d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py)  
  A2G channel model: topology generation (Gaussian clusters), LoS probability (a,b), path loss, shadowing, fading.

- [capacity_models.py](/d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py)  
  Shannon capacity + finite blocklength + decoding error probability.

- [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py)  
  Core scheduling logic:
  - eMBB RB allocation (greedy)
  - URLLC admission + overlay/puncture decision
  - SIC / interference handling
  - power allocation and refinements

- [simulation.py](/d:/URLLC_eMBB_Coexisting/Greedy/simulation.py)  
  Simulation loop over slots, aggregates metrics, calls allocator, collects results.

- [visualization.py](/d:/URLLC_eMBB_Coexisting/Greedy/visualization.py)  
  All plots for the Greedy simulator (throughput, power, timelines, heatmaps).

- [main.py](/d:/URLLC_eMBB_Coexisting/Greedy/main.py)  
  Entry point for Greedy baseline runs.

- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md)  
  How simulator variables map to the system model.

- [SYSTEM_MODEL_AND_PARAMETERS.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_AND_PARAMETERS.md)  
  Complete parameter + power + channel formulas.

---

## sr_mappo/ (RL + SR-MAPPO)

- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)  
  All RL configs: action space, reward weights, shield settings, network sizes, training hyperparams.

- [types.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/types.py)  
  Action/observation dataclasses + mode constants.

- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)  
  The SR‑MAPPO environment:
  - episode structure
  - candidate generation
  - URLLC coexistence decisions
  - eMBB baseline planning (new)
  - reward computation

- [shield.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/shield.py)  
  Action masking, feasibility correction, collision resolution.

- [networks.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/networks.py)  
  Actor‑critic network with masked hybrid actions.

- [buffer.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/buffer.py)  
  Rollout buffer for MAPPO (stores actions, masks, advantages).

- [trainer.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/trainer.py)  
  MAPPO training loop + PPO updates.

- [train.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/train.py)  
  End‑to‑end run: training + report generation.

- [evaluate.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/evaluate.py)  
  Evaluation utilities and scoring.

- [report.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/report.py)  
  Generates SR‑MAPPO report figures.

- [publication_figures.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/publication_figures.py)  
  Paper‑style plots (MAPPO only).

- [puncture_pressure_figures.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/puncture_pressure_figures.py)  
  Diagnostic plots for puncture/overlay behavior.

- [SR_MAPPO_ACTION_SPACE_MAPPING.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_ACTION_SPACE_MAPPING.md)  
  Action space explanation and system‑model mapping.

- [SR_MAPPO_REWARD_FUNCTION.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_REWARD_FUNCTION.md)  
  Full reward derivation and formulas.

- [SR_MAPPO_FRAMEWORK.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_FRAMEWORK.md)  
  High‑level MAPPO framework design.

---

## Where SIC and Power Logic Live

- **SIC / residual interference**  
  [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py)  
  Look in `_find_best_urllc_action()` and `_compute_embb_state()` where:
  `sic_residual_factor` and post‑SIC SNIR are used.

- **Power model (eMBB + URLLC)**  
  [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py)  
  - `_dbm_to_watts`, `_bisection_search_urllc_power`, `_get_embb_per_rb_power`

If you want this guide to include line‑level anchors for specific functions, tell me which functions you want indexed first and I’ll add them.
