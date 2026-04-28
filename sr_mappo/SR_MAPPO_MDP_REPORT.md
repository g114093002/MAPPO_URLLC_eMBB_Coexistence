# SR-MAPPO MDP for Report

This note is a report-ready description of the current **Shielded Recurrent Action-Masked MAPPO (SR-MAPPO)** formulation used in:

- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)
- [trainer.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/trainer.py)
- [networks.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/networks.py)
- [shield.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/shield.py)
- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)

It is written specifically for the current system model:

- multi-UAV uplink network
- eMBB and URLLC coexistence
- allowed coexistence modes:
  - eMBB-only
  - eMBB-URLLC overlay
  - puncturing
- URLLC admitted packets must satisfy a hard reliability constraint

## 1. Problem Type

The current learning problem is a **cooperative Dec-POMDP under CTDE**.

$$
\mathcal{G}=
\langle
\mathcal{N},
\mathcal{S},
\{\mathcal{O}_j\}_{j\in\mathcal{N}},
\{\mathcal{A}_j\}_{j\in\mathcal{N}},
\mathcal{P},
r,
\gamma
\rangle
$$

where:

- `N` agents correspond to UAVs
- each agent observes only local partial information
- all agents receive the same shared team reward
- training uses a centralized critic
- execution uses decentralized recurrent actors

## 2. Agents

Each UAV is one agent:

$$
\mathcal{N} = \{1,2,\dots,N\},
$$

where `N = num_uavs`.

In the current setup:

- `agent 1 = UAV 1`
- `agent 2 = UAV 2`
- `agent 3 = UAV 3`

The actor parameters are shared across UAVs.

## 3. Time Scale

### Episode

One episode corresponds to **one scheduling slot**.

$$
\text{1 episode} = \text{1 slot}
$$

### Step and Planning Phase

When `learn_embb_baseline = True`, the slot has two phases:

1. **Phase-0 eMBB planning**  
   One planning step corresponds to one RB across all UAVs:

$$
\text{1 planning step} = \text{RB } k
$$

There are `num_subcarriers` planning steps.

2. **Phase-A URLLC coexistence**  
   One environment step corresponds to one **joint decision cell**:

$$
\text{1 step} = (k,s)
$$

where:

- `k` = RB index
- `s` = minislot index

All UAVs act simultaneously on the same `(RB, minislot)` cell.

For the current setting:

- `8 RB`
- `8 minislots`

so the Phase-A part of a full episode contains:

$$
8 \times 8 = 64 \text{ steps}.
$$

## 4. State and Observation

The true environment state is global, but the actor sees only a partial observation.

### 4.1 Global state conceptually includes

- fixed user-UAV association
- current eMBB RB-ownership baseline on all UAVs and RBs
- current URLLC packet pool
- packet release times
- channel gains
- already scheduled packets and powers
- current coexistence mode grid
- remaining unscheduled packets

### 4.2 Local observation of UAV `j`

For each UAV agent `j`, the actor observes:

#### Local cell context

- normalized minislot index
- normalized RB index
- normalized UAV index
- current baseline eMBB owner on this RB
- baseline per-RB eMBB rate
- baseline per-RB eMBB power
- associated eMBB load ratio on this UAV
- associated URLLC load ratio on this UAV
- scheduled-packet ratio on this UAV
- overlay ratio on this UAV
- puncture ratio on this UAV
- remaining unscheduled-packet ratio

#### Candidate packet table

For the top-`M` candidate packets (`M = 8` currently), each candidate includes:

- source user index
- whether the packet is naturally associated with this UAV
- channel gain summary
- overlay feasibility
- puncture feasibility
- required power under overlay
- required power under puncture
- reliability margin under overlay
- reliability margin under puncture
- eMBB loss under overlay
- eMBB loss under puncture
- overlay utility proxy
- puncture utility proxy
- feasible-UAV count
- overlay-feasible UAV count
- puncture-feasible UAV count
- contention score

This means the actor is not observing raw packets only; it sees **physics-aware and conflict-aware candidate descriptors**.

In the current implementation, puncturing targets the **learned** eMBB RB owner.  
When `learn_embb_baseline = True`, the policy selects the RB owner during Phase-0, so Phase-A overlay and puncture operate on a **policy-selected** anchor rather than a fixed greedy anchor.

### 4.3 Critic global observation

The centralized critic uses a compact global feature vector containing:

- global slot progress
- current `(RB, minislot)` index
- global scheduled/unscheduled packet ratios
- per-UAV association load summaries
- per-UAV scheduling summaries
- per-UAV overlay and puncture statistics

## 5. Action Space

Each UAV agent outputs a hybrid action:

$$
a_j = (m_j, p_j, \delta_j, o_j, \delta^{(e)}_j)
$$

where:

- `m_j` = coexistence mode
- `p_j` = packet option
- `\delta_j` = URLLC power adjustment
- `o_j` = eMBB owner option (Phase-0)
- `\delta^{(e)}_j` = eMBB power adjustment (Phase-0 and Phase-A scaling)

### 5.1 Mode action

The mode set is:

$$
m_j \in \{\text{KEEP},\text{OVERLAY},\text{PUNCTURE}\}.
$$

Numerically:

- `0 = KEEP`
- `1 = OVERLAY`
- `2 = PUNCTURE`

### 5.2 Packet action

The packet-option set is:

$$
p_j \in \{0,1,\dots,M\},
$$

with:

- `0 = null / no packet`
- `1..M = candidate packet index`

### 5.3 URLLC power action

The actor also outputs:

$$
\delta_j \in [-1,1].
$$

This is converted into a power request around the minimum feasible required power.

### 5.4 eMBB owner action (Phase-0)

During the planning phase, each UAV selects an RB owner option:

- `o_j = 0` means "no eMBB owner" for this RB
- `o_j = 1..M` selects from the RB-specific candidate eMBB list on that UAV

The candidate list is built per `(UAV, RB)` from the UAV-associated eMBB users.

### 5.5 eMBB power action (Phase-0 and Phase-A scaling)

The policy also outputs a continuous **eMBB power delta**:

- raw output in `[-1, 1]`
- mapped to a scale factor:

$$
s_j = \mathrm{clip}(1 + \alpha \cdot \delta^{(e)}_j, s_{\min}, s_{\max})
$$

This scale factor is applied to the eMBB power budget for that UAV and is used in
both Phase-0 baseline computation and Phase-A coexistence feasibility.

## 6. Action Masking

The action space is partially restricted before execution.

### Mode mask

Invalid modes can be masked out if they are impossible for the current cell.

### Packet mask

The packet mask is **mode-conditioned**:

- if `mode = KEEP`, only packet option `0` is valid
- if `mode = OVERLAY`, only overlay-feasible packets remain visible
- if `mode = PUNCTURE`, only puncture-feasible packets remain visible

This is important because it reduces the mismatch:

- raw mode chosen by policy
- packet feasibility for that mode

## 7. Transition and Execution

The transition is not raw-policy execution only.

The actual execution pipeline is:

1. actor outputs raw hybrid actions for all UAVs
2. actions are sanitized by the shield
3. same-cell joint reliability and assignment are checked
4. collisions and infeasible combinations may be rewritten
5. executed actions update:
   - packet grid
   - mode grid
   - scheduled power
   - scheduled reliability
   - per-UAV counters

So the environment transition is:

$$
s_{t+1} \sim \mathcal{P}(s_{t+1}\mid s_t,\tilde{a}_t),
$$

where `\tilde{a}_t` is the **executed post-shield action**, not merely the raw sampled action.

## 8. Hard Reliability Constraint

This is a critical modeling point.

URLLC reliability is **not** a soft preference. It is a hard feasibility requirement.

That means:

- if a packet cannot be scheduled with the required reliability, it should not be admitted as a valid overlay/puncture action
- the shield and joint feasibility layer are responsible for enforcing this

The reward should therefore not be interpreted as ?rading reliability against throughput.??
Instead, the learning problem is:

- keep admitted URLLC packets reliability-feasible
- among feasible actions, maximize better coexistence behavior

## 9. Reward Function

The reward is a shared team reward:

$$
r_t^{team}
=
\sum_{j=1}^{N} r_{t,j}^{local} + r_t^{terminal}.
$$

### 9.1 Local step reward

For executed non-KEEP actions, the current local reward includes:

- schedule success reward
- eMBB damage penalty
- puncture penalty
- overlay margin reward
- missed-overlay penalty
- power penalty
- power projection penalty
- invalid-action penalty
- collision-rewrite penalty

### 9.2 Terminal reward

At episode end, the current terminal reward includes:

- unscheduled URLLC penalty
- **absolute** eMBB throughput (normalized)
- **absolute** eMBB fairness (Jain index)

More explicitly:

$$
r_T^{\mathrm{terminal}}
=
-\lambda_{\mathrm{uns}} \alpha_T^{\mathrm{unsched}}
+ \lambda_{\mathrm{rate}}
\frac{R_T^{\mathrm{SR}}}{R_{\mathrm{norm}}}
+ \lambda_{\mathrm{fair}} J_T^{\mathrm{SR}},
$$

where:

- $R_T^{\mathrm{SR}}$ = final aggregate eMBB throughput of SR-MAPPO
- $R_{\mathrm{norm}}$ = rate normalizer used in training
- $J_T^{\mathrm{SR}}$ = Jain's fairness index of the final SR-MAPPO eMBB rate vector

The Jain fairness index is:

$$
J_T
=
\frac{
\left(\sum_{q=1}^{Q} R_{T,q}\right)^2
}{
Q \sum_{q=1}^{Q} R_{T,q}^2 + \varepsilon
}.
$$

This terminal fairness term is placed at slot end because fairness is a population-level outcome, not a clean step-level signal.

## 10. Policy and Critic

### Actor

The actor is:

- shared across UAVs
- recurrent
- decentralized at execution time

Its outputs are:

- categorical distribution over modes
- categorical distribution over packet options
- categorical distribution over eMBB owner options (Phase-0)
- Gaussian-style continuous head for URLLC power delta
- Gaussian-style continuous head for eMBB power delta

### Critic

The critic is:

- centralized
- recurrent
- conditioned on global summary features

This is a standard CTDE structure:

- centralized training
- decentralized execution

## 11. Why This Is Still Phase-A

The current RL scope is still limited, even with Phase-0 planning.

SR-MAPPO does **not** yet control:

- user association across UAVs
- multi-RB joint eMBB scheduling beyond one owner per RB
- full cross-UAV coordination of eMBB anchors

It now learns:

- slot-level eMBB RB owners (Phase-0)
- eMBB power scaling (Phase-0 and Phase-A)
- whether to keep, overlay, or puncture
- which URLLC packet to use
- how to adjust URLLC power locally

Therefore, this is best described as:

**Phase-A coexistence with policy-selected eMBB RB anchors, but without full joint eMBB scheduling.**

## 12. Current Bottleneck Interpretation

From the current diagnostics, the main bottlenecks are no longer purely reward-direction errors.

The more likely bottlenecks are:

- scarcity of truly executable overlay opportunities
- packet contention across UAVs
- collision rewrites and joint reliability rewrites
- suboptimal eMBB anchor layouts chosen during Phase-0

So when describing the current method in a report, it is accurate to say:

> SR-MAPPO learns a structured coexistence policy on top of a policy-selected eMBB anchor layout, but its performance is still limited by joint packet-assignment conflicts and the scarcity of overlay-feasible opportunities.

## 13. One-Paragraph Report Version

If you want a short report paragraph, you can use this:

> The current SR-MAPPO scheduler is formulated as a cooperative Dec-POMDP with centralized training and decentralized execution. Each UAV acts as one agent, and one episode corresponds to one scheduling slot. When `learn_embb_baseline = True`, the slot begins with a Phase-0 planning stage where each UAV selects eMBB RB owners and an eMBB power scale. During the Phase-A coexistence stage, at each `(RB, minislot)` decision step, every UAV selects a hybrid action consisting of coexistence mode, candidate URLLC packet, and URLLC power adjustment. The actor observes local cell context and a truncated packet-candidate table with feasibility, damage, and contention descriptors, while the critic uses a compact global state summary. URLLC reliability is enforced as a hard constraint through action masking and shielded joint-feasibility execution. The reward combines immediate coexistence-quality signals with terminal slot-level objectives, including normalized absolute eMBB throughput and Jain fairness.

## 14. Code References

Main code locations:

- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)
- [trainer.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/trainer.py)
- [networks.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/networks.py)
- [shield.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/shield.py)
- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)
- [SR_MAPPO_REWARD_FUNCTION.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_REWARD_FUNCTION.md)

