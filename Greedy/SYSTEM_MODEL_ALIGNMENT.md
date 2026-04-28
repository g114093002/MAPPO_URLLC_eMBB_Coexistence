# System Model Alignment

This document maps the mathematical symbols in `鄭昀曜_20260323_r5_system_model - 複製.pdf` to the current codebase and highlights the gaps that still need refactoring.

## 1. Symbol Mapping

| PDF Symbol | Meaning in PDF | Current Code Representation | Status |
| --- | --- | --- | --- |
| `N` | UAV set | `SystemConfig.num_uavs` in [config.py](d:/URLLC_eMBB_Coexisting/Greedy/config.py) | Aligned |
| `E` | eMBB user set | `SystemConfig.num_embb_users` in [config.py](d:/URLLC_eMBB_Coexisting/Greedy/config.py) | Aligned |
| `U` | URLLC user set | `SystemConfig.num_urllc_users` in [config.py](d:/URLLC_eMBB_Coexisting/Greedy/config.py) | Aligned |
| `t` | time slot index | `slot_index` and `SystemConfig.num_slots` in [simulation.py](d:/URLLC_eMBB_Coexisting/Greedy/simulation.py) | Aligned |
| `s` | minislot index | `SystemConfig.num_minislots`, `urllc_timefreq_grid[:, s]` | Partially aligned |
| `k` | RB index | `SystemConfig.num_subcarriers`, RB columns in allocator matrices | Partially aligned |
| `\phi_{i,j}^t` | user-UAV association | `best_uav_per_user`, `allocation['user_association']` in [simulation.py](d:/URLLC_eMBB_Coexisting/Greedy/simulation.py) | Aligned |
| `d_{i,j}(t)` | 3D user-UAV distance | computed in [channel_model.py](d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py) | Aligned |
| `P_{i,j}^{LoS}(t)` | LoS probability | computed in [channel_model.py](d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py) | Aligned |
| `\beta_{i,j}(t)` | large-scale channel gain | `get_average_large_scale_gains()` in [channel_model.py](d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py) | Aligned |
| `h_{i,j}(t)` | channel coefficient | `channel_model.generate_channel_gains()` in [simulation.py](d:/URLLC_eMBB_Coexisting/Greedy/simulation.py) | Aligned |
| `p_q^t` | eMBB transmit power | `embb_user_tx_power` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `p_z^{s,t}` | URLLC transmit power | `urllc_power_allocation[z, j]` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `\alpha_{q,j,k}^E` | eMBB RB allocation at UAV `j` on RB `k` | `alpha_e[q, j, k]` and `embb_owner_per_uav_rb[j, k]` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `\rho_{q,z,k,j}^{s,t}` | superposition indicator | `rho_tensor[q, z, j, k, s]` and `rho_action_list` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `\varpi_{q,z,k,j}^{s,t}` | puncturing indicator | `varpi_tensor[q, z, j, k, s]` and `varpi_action_list` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `I_{j,k}^{s,t}` | aggregate inter-cell interference | not explicitly modeled; only local interference heuristics in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Not aligned |
| `\delta_{z,j,k}^{s,t}` | URLLC SINR | computed heuristically during URLLC candidate evaluation in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `\delta_{q,j,k}^{s,t}` | eMBB SINR after SIC / puncturing | approximated in `_compute_embb_state()` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `\Gamma_z^{th}` | URLLC SINR threshold from finite blocklength | implicitly enforced via `decoding_error_probability()` in [capacity_models.py](d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py) | Partially aligned |
| `R_{q,j}^{s,t}` | eMBB achievable rate | `embb_result['rates']` and `_compute_embb_state()` in [resource_allocator.py](d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) | Partially aligned |
| `R_{q,j,min}^{s,t}` | minimum eMBB rate constraint | no explicit constraint in code | Not aligned |

## 2. What Is Already Consistent With the PDF

- User-UAV association is now long-term and topology-driven rather than slot-by-slot fading-driven.
- UAV and UE positions are fixed within one simulation run.
- The A2G channel model uses 3D distance, elevation angle, probabilistic LoS/NLoS, and large-scale gain.
- URLLC reliability is treated as a hard constraint for admitted packets.
- Minislot-level coexistence is represented in the time-frequency grid.

## 3. Major Gaps That Still Need Refactoring

### 3.1 Per-UAV RB pool is missing

The PDF defines `\alpha_{q,j,k}^E`, meaning each UAV has its own RB pool.

Current code only uses:

- `rb_allocation[q, k]`
- `embb_owner_per_rb[k]`

This means the current allocator still behaves like there is one global RB grid, not `num_uavs` separate RB grids.

### 3.2 `\rho` and `\varpi` are not full decision tensors

The PDF defines:

- `\rho_{q,z,k,j}^{s,t}`
- `\varpi_{q,z,k,j}^{s,t}`

Current code only stores:

- `noma_decisions[s, k]`

So the current implementation loses:

- UAV dimension `j`
- explicit eMBB user `q`
- explicit URLLC user `z`
- slot-wise tensor semantics

### 3.3 Inter-cell interference `I_{j,k}^{s,t}` is not implemented

The PDF explicitly models interference from:

- eMBB users served by other UAVs
- URLLC users served by other UAVs

Current code only uses:

- same-RB local eMBB interference when testing NOMA
- noise plus optional SIC residual

This is the largest physical-model gap.

### 3.4 eMBB minimum-rate constraint is absent

The optimization problem includes:

- `R_{q,j}^{s,t} >= R_{q,j,min}^{s,t}`

Current code maximizes rate heuristically but does not explicitly enforce a minimum eMBB service constraint.

### 3.5 Algorithm objective does not match the PDF objective

The PDF objective is rate maximization under explicit constraints.

Current code still contains heuristic utility terms for:

- overload penalty
- optional NOMA preference
- PF-style eMBB weighting

These are algorithmic heuristics, not variables or constraints defined in the PDF.

## 4. Refactor Plan

### Stage A. Data structure alignment

Replace the current allocation objects with:

- `alpha_e[q, j, k]`
- `rho[q, z, j, k, s]`
- `varpi[q, z, j, k, s]`
- `p_embb[q]`
- `p_urllc[z, s]` or `p_urllc[z]` depending on the final interpretation

### Stage B. Per-UAV RB allocation

Refactor the eMBB baseline allocator so each UAV owns `K` RBs independently.

This means:

- RB indexing becomes local to each UAV
- `embb_owner_per_rb` must become `embb_owner_per_uav_rb[j, k]`

### Stage C. SINR model reconstruction

Implement explicit functions for:

- URLLC SINR under superposition
- URLLC SINR under puncturing
- eMBB SINR after SIC
- aggregate inter-cell interference `I_{j,k}^{s,t}`

### Stage D. Constraint-aware greedy design

Redesign the greedy scheduler to operate on candidate actions:

- assign `(q, j, k)` as eMBB baseline
- assign `(q, z, j, k, s)` as superposition
- assign `(q, z, j, k, s)` as puncturing

The scheduler should only accept actions that satisfy the PDF constraints, unless we explicitly document a heuristic relaxation.

### Stage E. Metric cleanup

Separate these metrics clearly:

- URLLC admission ratio
- admitted URLLC reliability
- URLLC end-to-end service ratio
- eMBB served-user ratio

## 5. Recommendation for the Next Coding Step

The next code change should not start from tuning heuristics.

The next correct step is:

1. replace global RB allocation with per-UAV RB allocation
2. change coexistence decisions from a 2D grid to a full per-UAV decision tensor
3. then implement the SINR/interference equations using those tensors

Only after those three are in place should the greedy logic itself be redesigned again.
