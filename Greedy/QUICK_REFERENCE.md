# Quick Reference

## Current Model Status

This project now follows the PDF system model more closely than the original greedy baseline:

- fixed user-UAV association based on large-scale A2G gains
- per-UAV RB allocation via `alpha_e[q,j,k]`
- coexistence actions stored as `rho_tensor[q,z,j,k,s]` and `varpi_tensor[q,z,j,k,s]`
- inter-cell interference included in the current SINR evaluation
- admitted URLLC reliability separated from URLLC admission ratio

## What The Main Metrics Mean

- `eMBB Total Rate`: aggregate eMBB throughput across the slot sequence
- `URLLC Admission Ratio`: fraction of active URLLC arrivals that are admitted
- `Admitted URLLC Reliability`: reliability of admitted URLLC packets only
- `RB Utilization`: eMBB per-UAV RB occupancy
- `Joint Resource Pressure`: eMBB RB occupancy plus URLLC minislot occupancy pressure
- `NOMA cell fraction`: fraction of `(j,k,s)` cells operating in superposition
- `Puncturing cell fraction`: fraction of `(j,k,s)` cells operating in puncturing

## Current Algorithm Summary

The allocator is no longer arrival-order driven. The current logic is:

1. Fix user-UAV association from long-term large-scale gains
2. Allocate eMBB on each UAV's RB pool
3. For each active URLLC user, evaluate feasible `(j,k,s)` actions
4. Choose `NOMA` or `PUNCT` only if:
   - URLLC reliability is satisfied
   - eMBB minimum-rate constraint is not violated
5. Recompute eMBB rates and power with coexistence and inter-cell interference

## Important Interpretation

- If `Admitted URLLC Reliability` is near `1.0` while `Admission Ratio` is low, it means the reliability constraint is being enforced by selective admission.
- High aggregate eMBB throughput does not imply fairness; it can still coexist with low served-user ratio.
- `slot_timefreq_slot*.png` now represents one selected UAV view of the per-UAV coexistence tensor, not a single global RB pool.

## Key Files

- [config.py](d:/URLLC_eMBB_Coexisting/Greedy/config.py): parameters and operating points
- [channel_model.py](d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py): topology, A2G path loss, fading
- [capacity_models.py](d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py): Shannon and finite-blocklength functions
- [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py): `alpha_e`, `rho`, `varpi`, constrained greedy logic
- [simulation.py](d:/URLLC_eMBB_Coexisting/Greedy/simulation.py): slot loop, density sweep, metric aggregation
- [visualization.py](d:/URLLC_eMBB_Coexisting/Greedy/visualization.py): plots and reporting
- [SYSTEM_MODEL_ALIGNMENT.md](d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md): PDF-to-code alignment notes

## Typical Commands

```bash
python main.py
```

Outputs are saved in:

- `results/power_vs_density.png`
- `results/performance_timeline.png`
- `results/slot_timefreq_slot*.png`
- `results/spatial_grouping_slot*.png`

## What Is Still Not Fully Final

- the interference model is now present, but still implemented in an engineering form rather than a fully symbolic paper-style tensor derivation
- the visualization layer still provides a selected-UAV view for readability
- some legacy helper outputs remain for backward compatibility with older plots
