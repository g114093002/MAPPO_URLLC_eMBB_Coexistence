# Multi-UAV URLLC-eMBB Coexistence Framework

## Overview

This project simulates uplink resource allocation for a multi-UAV network where eMBB and URLLC users coexist through:

- orthogonal eMBB RB assignment
- URLLC superposition (`NOMA`)
- URLLC puncturing

The implementation has been refactored to better match the system model in:

- [鄭昀曜_20260323_r5_system_model - 複製.pdf](d:/URLLC_eMBB_Coexisting/Greedy/%E9%84%AD%E6%98%80%E6%9B%9C_20260323_r5_system_model%20-%20%E8%A4%87%E8%A3%BD.pdf)

and the alignment notes are summarized in:

- [SYSTEM_MODEL_ALIGNMENT.md](d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md)

## Current Modeling Assumptions

- one user is associated with one UAV in a slot
- association is based on long-term large-scale A2G gain
- each UAV has its own RB pool
- eMBB allocation is stored as `alpha_e[q,j,k]`
- coexistence actions are stored as:
  - `rho_tensor[q,z,j,k,s]` for superposition
  - `varpi_tensor[q,z,j,k,s]` for puncturing
- admitted URLLC packets must satisfy the reliability target
- inter-cell interference is included in the current SINR evaluation

## Main Modules

- [config.py](d:/URLLC_eMBB_Coexisting/Greedy/config.py): system, service, and algorithm parameters
- [channel_model.py](d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py): topology generation and A2G fading model
- [capacity_models.py](d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py): Shannon and finite-blocklength formulas
- [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py): constrained greedy allocator
- [simulation.py](d:/URLLC_eMBB_Coexisting/Greedy/simulation.py): slot execution and density analysis
- [visualization.py](d:/URLLC_eMBB_Coexisting/Greedy/visualization.py): plots and reports
- [main.py](d:/URLLC_eMBB_Coexisting/Greedy/main.py): entry point

## Current Allocation Logic

The current allocator is no longer arrival-order driven. It now works in this order:

1. build static user-UAV association from large-scale gains
2. allocate eMBB on each UAV's RB pool
3. evaluate URLLC coexistence actions on `(j,k,s)` resources
4. admit URLLC only if:
   - URLLC reliability target is satisfied
   - eMBB minimum-rate constraint is not broken
5. choose `NOMA` or `PUNCT` per action
6. recompute eMBB rates and power with coexistence and inter-cell interference

## Key Metrics

- `eMBB Total Rate`: aggregate eMBB throughput
- `URLLC Admission Ratio`: admitted active URLLC arrivals / total active URLLC arrivals
- `Admitted URLLC Reliability`: reliability of admitted URLLC packets only
- `eMBB served ratio`: fraction of eMBB users with non-zero achieved rate
- `NOMA cell fraction`: fraction of `(j,k,s)` cells operating in superposition
- `Puncturing cell fraction`: fraction of `(j,k,s)` cells operating in puncturing
- `Joint Resource Pressure`: combined eMBB RB occupancy and URLLC minislot occupancy pressure

## Run

```bash
python main.py
```

Generated results are written to `results/`.

Typical files:

- `results/power_vs_density.png`
- `results/performance_timeline.png`
- `results/slot_timefreq_slot*.png`
- `results/spatial_grouping_slot*.png`

## Notes

- `slot_timefreq` plots now show a selected UAV view, because the internal model is per-UAV rather than one global RB pool
- admitted URLLC reliability and URLLC admission are intentionally separated
- some legacy outputs are still kept for backward compatibility with older plotting paths
