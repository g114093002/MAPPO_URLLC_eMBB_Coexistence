# Optimization-to-RL Coverage Map

This note links the system model optimization variables (as written in the PDF formulation) to what SR-MAPPO actually controls today. The goal is to make the mismatch explicit and easy to report.

## 1) Decision Variables in the Original Problem

The original formulation maximizes the **sum eMBB throughput** subject to URLLC reliability and resource constraints. Key decision variables include:

- **eMBB RB allocation**: which eMBB user owns RB `k` at UAV `j` and slot `t`.
- **Coexistence mode** on each (RB, minislot): superpose vs puncture.
- **Transmit powers** of eMBB and URLLC users.
- **SIC feasibility** constraints induced by interference and SINR thresholds.

## 2) What SR-MAPPO Controls Today

SR-MAPPO currently controls **only part** of the decision space:

- **Mode** per cell (keep / overlay / puncture).
- **URLLC packet selection** (which packet to schedule at the current cell).
- **URLLC power delta** (relative adjustment around required power).
- **(Phase-A planning) eMBB owner selection and eMBB power scaling** per UAV-RB, but this is limited and still not full RB allocation control in the global sense.

## 3) Mapping Table (Original Optimization → RL Coverage)

| Optimization Variable / Constraint | Symbol (PDF) | RL Coverage Today | Where Implemented | Consequence |
|---|---|---|---|---|
| eMBB RB allocation | `ρ_{q,j,k,t}` | **Partially** (Phase-A planning only, per UAV-RB owner option) | `SRMAPPOPhaseAEnv._step_embb_planning()` | RL cannot globally re-plan all eMBB allocations; throughput ceiling is fixed by anchor structure. |
| Superpose indicator | `q_{q,z,k,j}^{s,t}` | **Yes** (mode=OVERLAY) | `env.py` + `shield.py` | Only effective if overlay is feasible; many opportunities are blocked. |
| Puncture indicator | `q_{q,z,k,j}^{s,t}` | **Yes** (mode=PUNCTURE) | `env.py` + `shield.py` | Easy to execute, often dominates when overlay is infeasible. |
| URLLC power | `p_{z,t}` | **Yes** (power delta) | `env.py` | Only local control; global power budgets not optimized. |
| eMBB power | `p_{q,t}` | **Partial** (power scaling) | `env.py` | Global power allocation across RBs/users remains fixed by baseline. |
| URLLC reliability constraint | `ε_z ≤ ε_z^max` | **Hard enforced** by shield | `shield.py` + `_enforce_joint_reliability()` | RL is not actually choosing invalid actions; it is being rewritten. |
| eMBB min-rate constraint | `R_q ≥ R_q^min` | **Not explicit** | reward only | No explicit constraint; may be violated without strong penalty. |
| Inter-cell interference coupling | implicit | **Not controlled** | simulator/channel | RL cannot re-plan inter-cell structure, only react locally. |

## 4) Why Max-Throughput Does Not Fully Appear in RL Results

The RL agent **does not control the full optimization variables**. It only makes local coexistence decisions on top of a mostly fixed eMBB baseline. Thus the RL objective is a **proxy** of the original max-sum-rate problem, not the same problem.

Concretely:

- **eMBB anchor structure dominates the achievable sum-rate**, but RL cannot fully re-optimize it.
- **Overlay feasibility is rare**, so even if the policy prefers overlay, it has little impact.
- **Shield rewrites actions**, so the policy is not executing its own decisions.
- **Reward is localized**, while the original objective is global (sum-rate over all users and RBs).

## 5) Recommended Fixes (if goal is strict alignment)

1. **Expand RL action space** to include eMBB RB ownership/assignment for each UAV-RB.
2. **Replace hard shield** with feasibility penalty or action-space masking so the policy receives credit assignment.
3. **Increase terminal throughput weight** and simplify local shaping, to align with max-sum-rate.
4. **Add explicit eMBB min-rate/fairness constraints** if those are hard requirements.

This table can be copied directly into the report to justify why the current RL does not yet reflect the original max-throughput objective.
