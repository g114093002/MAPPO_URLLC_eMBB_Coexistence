# SR-MAPPO Reward Function

This file is the **latest reward definition** of the current SR-MAPPO implementation.

Code references:

- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)
- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)

It reflects the current design after:

- terminal KPI alignment using absolute throughput and fairness
- addition of eMBB fairness reward
- retention of hard URLLC reliability as a constraint rather than a soft reward target

## 1. Reward Design Goal

The reward is designed to encode the following behavior:

1. At each local `(RB, minislot)` decision, prefer a feasible action that:
   - serves URLLC when appropriate,
   - preserves more eMBB rate,
   - prefers overlay over puncture when overlay is better,
   - avoids invalid or collision-heavy actions,
   - avoids unnecessary power inflation.
2. At the end of the slot, prefer a policy that:
   - leaves fewer URLLC packets unscheduled,
   - achieves higher aggregate eMBB throughput than the original Greedy baseline,
   - improves eMBB fairness relative to the original Greedy baseline.

The reward **does not** trade URLLC reliability against throughput.  
URLLC reliability is enforced as a hard feasibility constraint by masking and shielded execution.

## 2. Shared Team Reward

At decision step $t$, all UAV agents receive the same shared reward:

$$
r_t^{\mathrm{team}}
=
\sum_{j=1}^{N} r_{t,j}^{\mathrm{local}}
+
r_t^{\mathrm{terminal}},
$$

where:

- $N$ is the number of UAV agents
- $r_{t,j}^{\mathrm{local}}$ is the executed local reward of UAV $j$
- $r_t^{\mathrm{terminal}}$ is nonzero only when the slot terminates

This is the reward that is actually stored in the rollout buffer and optimized by PPO.

## 3. Complete Symbol Table

### 3.1 Indices

- $t$: decision step inside one slot episode
- $j \in \{1,\dots,N\}$: UAV index
- $q \in \{1,\dots,Q\}$: eMBB user index

### 3.2 Executed action terms

- $m_{t,j} \in \{\mathrm{KEEP},\mathrm{OVERLAY},\mathrm{PUNCTURE}\}$: executed coexistence mode
- $P_{t,j}^{\mathrm{raw}}$: raw requested power before projection
- $P_{t,j}^{\mathrm{req}}$: minimum feasible required power for the chosen mode
- $P_{t,j}^{\mathrm{act}}$: final executed power after projection/shield
- $P_{\max}$: system power upper bound

### 3.3 Counterfactual eMBB damage and utility terms

- $R_{t,j}^{\mathrm{keep}}$: baseline eMBB rate of the current cell if the cell remains eMBB-only
- $L_{t,j}^{\mathrm{ov}}$: estimated eMBB rate loss if overlay is used
- $L_{t,j}^{\mathrm{pun}}$: estimated eMBB rate loss if puncture is used
- $L_{t,j}(m)$: estimated eMBB rate loss under the actually executed mode $m$
- $U_{t,j}^{\mathrm{ov}}$: overlay utility proxy
- $U_{t,j}^{\mathrm{pun}}$: puncture utility proxy
- $f_{\mathrm{rel}}(\cdot)$: reliability shaping function used inside the utility proxy
- $\Pi_{t,j}^{\mathrm{overload}}$: overload penalty used inside the utility proxy

### 3.4 Slot-level KPI terms

- $R_{t,q}^{\mathrm{SR}}$: final eMBB rate of user $q$ under SR-MAPPO in the current slot
- $R_t^{\mathrm{SR}}$: aggregate SR-MAPPO eMBB throughput
- $R_{\mathrm{norm}}$: throughput normalizer (default $10^6$ bps)
- $J_t^{\mathrm{SR}}$: Jain's fairness index of $\{R_{t,q}^{\mathrm{SR}}\}_{q=1}^{Q}$

### 3.5 URLLC completion terms

- $N_t^{\mathrm{active}}$: number of active URLLC packets in the slot
- $N_t^{\mathrm{unscheduled}}$: number of URLLC packets left unscheduled at slot end
- $\alpha_t^{\mathrm{unsched}}$: unscheduled-packet ratio

### 3.6 Numerical stabilizer

- $\varepsilon = 10^{-9}$: small constant used to avoid division by zero

This $\varepsilon$ is only a **numerical stabilizer**. It is **not** the URLLC decoding error target and it is **not** the packet error probability.

## 4. From Design Objective to Local Reward

The local reward is not written directly from raw throughput.  
Instead, it is built from normalized, interpretable shaping terms.

We decompose the executed local reward as:

$$
r_{t,j}^{\mathrm{local}}
=
r_{t,j}^{\mathrm{sched}}
+
r_{t,j}^{\mathrm{dmg}}
+
r_{t,j}^{\mathrm{mode}}
+
r_{t,j}^{\mathrm{pow}}
+
r_{t,j}^{\mathrm{safety}}.
$$

Each term is defined below.

## 5. Schedule-Success Term

If the selected packet is successfully scheduled, the agent receives a small positive shaping reward:

$$
r_{t,j}^{\mathrm{sched}} = \lambda_{\mathrm{sched}}.
$$

This term exists because the agent should not learn to avoid all actions just to reduce damage.

Current coefficient:

$$
\lambda_{\mathrm{sched}} = 0.01.
$$

## 6. eMBB Damage Penalty Derivation

The damage term is defined relative to the eMBB-only baseline of the current cell.
Let

$$
R_{t,j}^{\mathrm{keep}}
$$

denote the eMBB rate of the current $(\mathrm{RB},\mathrm{minislot})$ cell when no URLLC packet is inserted.
In the current implementation, this corresponds to the cell-level baseline eMBB rate `base_rate`.

The retained eMBB rate after coexistence can be written as:

$$
R_{t,j}^{\mathrm{ret}}(m)
=
R_{t,j}^{\mathrm{keep}} - L_{t,j}(m).
$$

So $L_{t,j}(m)$ is a **throughput loss term**, not the retained throughput itself.
Equivalently:

$$
L_{t,j}(m)
=
R_{t,j}^{\mathrm{keep}} - R_{t,j}^{\mathrm{ret}}(m).
$$

The first damage quantity is the actually executed eMBB loss:

$$
L_{t,j}(m) =
\begin{cases}
L_{t,j}^{\mathrm{ov}}, & m_{t,j} = \mathrm{OVERLAY}, \\
L_{t,j}^{\mathrm{pun}}, & m_{t,j} = \mathrm{PUNCTURE}, \\
0, & m_{t,j} = \mathrm{KEEP}.
\end{cases}
$$

Under the current SR-MAPPO semantics:

- `KEEP` means "leave the current cell as eMBB-only"
- it does **not** mean "continue the previous cell's overlay/puncture state"

Therefore:

$$
L_{t,j}(\mathrm{KEEP}) = 0.
$$

For the two coexistence modes:

$$
L_{t,j}^{\mathrm{pun}} = R_{t,j}^{\mathrm{keep}},
$$

because puncturing removes the local eMBB transmission in that cell, and

$$
L_{t,j}^{\mathrm{ov}}
=
R_{t,j}^{\mathrm{keep}}\left(1-\rho_{t,j}^{\mathrm{ov}}\right),
$$

where $\rho_{t,j}^{\mathrm{ov}} \in [0,1]$ is the overlay retention factor.

Since absolute losses vary by cell, we normalize them by the puncture-loss scale:

$$
d_{t,j}(m)
=
\mathrm{clip}
\left(
\frac{L_{t,j}(m)}
{\max(L_{t,j}^{\mathrm{pun}}, \varepsilon)},
0, 1.5
\right).
$$

This creates a dimensionless damage score:

- $d_{t,j}(m) \approx 0$ means negligible eMBB damage
- $d_{t,j}(m) \approx 1$ means damage comparable to puncture loss
- values above $1$ are clipped to avoid exploding reward scale

The upper clip value `1.5` is a **defensive normalization cap**, not a statement that overlay should normally be more damaging than puncture.
In the current implementation, the intended ordering is:

$$
0 \le L_{t,j}^{\mathrm{ov}} \le L_{t,j}^{\mathrm{pun}} = R_{t,j}^{\mathrm{keep}},
$$

so for normal overlay/puncture cases the normalized damage should usually lie in $[0,1]$:

$$
0 \le d_{t,j}(m) \le 1.
$$

The clip to `1.5` is kept only as a safety margin in case future variants introduce:

- alternative loss proxies,
- numerical mismatch between estimated and executed loss,
- or new modes whose local loss may exceed the current puncture-loss reference.

The damage penalty is then:

$$
r_{t,j}^{\mathrm{dmg}}
=
- \lambda_{\mathrm{dmg}} \, d_{t,j}(m).
$$

Current coefficient:

$$
\lambda_{\mathrm{dmg}} = 0.30.
$$

## 7. Utility Proxy Expansion

Before defining the overlay preference terms, we first expand the utility proxy $U$.

The symbols $U_{t,j}^{\mathrm{ov}}$ and $U_{t,j}^{\mathrm{pun}}$ do **not** come from the final PPO reward directly.
They come from the local action utility used by the coexistence allocator.

For a generic local action, the utility proxy is:

$$
U_{t,j}
=
\frac{\Delta R_{t,j}^{\mathrm{eMBB}}}{10^6}
+
w_{\mathrm{u}} \, f_{\mathrm{rel}}\!\left(\gamma_{t,j}^{\mathrm{URLLC}}\right)
-
w_{\mathrm{p}} P_{t,j}
-
\Pi_{t,j}^{\mathrm{overload}},
$$

where:

- $\Delta R_{t,j}^{\mathrm{eMBB}}$ is the local eMBB rate change in bps
- $\gamma_{t,j}^{\mathrm{URLLC}}$ is the achieved URLLC reliability of that local action
- $P_{t,j}$ is the required URLLC transmission power in watts
- $w_{\mathrm{u}}$ is the URLLC-utility weight
- $w_{\mathrm{p}}$ is the power-penalty weight

In the current implementation:

$$
w_{\mathrm{u}} = 8.0,
\qquad
w_{\mathrm{p}} = 0.05.
$$

### 7.1 Reliability shaping function

Let the reliability target be:

$$
\gamma_{\mathrm{tar}} = 1 - \varepsilon_{\mathrm{tar}},
$$

where $\varepsilon_{\mathrm{tar}}$ is the URLLC target error probability.
Define the normalized reliability margin:

$$
\eta_{t,j}
=
\frac{\gamma_{t,j}^{\mathrm{URLLC}} - \gamma_{\mathrm{tar}}}
{\max(\gamma_{\mathrm{tar}}, 10^{-12})}.
$$

Then the reliability shaping function is:

$$
f_{\mathrm{rel}}\!\left(\gamma_{t,j}^{\mathrm{URLLC}}\right)
=
\begin{cases}
1 + 0.1\,\eta_{t,j},
& \gamma_{t,j}^{\mathrm{URLLC}} \ge \gamma_{\mathrm{tar}},
\\[4pt]
-\left(1 + 4\,|\eta_{t,j}|\right),
& \gamma_{t,j}^{\mathrm{URLLC}} < \gamma_{\mathrm{tar}}.
\end{cases}
$$

So:

- if the action meets the reliability target, it receives a strong positive utility bonus
- if it violates the target, it receives a strong negative utility penalty
- the penalty below target is intentionally steeper than the reward above target

### 7.2 Overload penalty

The overload term is a scheduler-side load penalty, not a physical outage probability.

First define the packet budget:

$$
B_{\mathrm{pkt}}
=
\max\!\left(
1,
\left\lceil \alpha_{\mathrm{load}} K \right\rceil
\right),
$$

where:

- $K$ is the number of RBs
- $\alpha_{\mathrm{load}}$ is the admission-load limit

In the current implementation:

$$
\alpha_{\mathrm{load}} = 0.35.
$$

If $n_{t,j}^{\mathrm{sched}}$ packets have already been scheduled before evaluating the current candidate, then the overload ratio is:

$$
\chi_{t,j}^{\mathrm{overload}}
=
\frac{n_{t,j}^{\mathrm{sched}} + 1}{B_{\mathrm{pkt}}}.
$$

The raw overload penalty is:

$$
\Pi_{t,j}^{\mathrm{overload}}
=
w_{\mathrm{over}} \chi_{t,j}^{\mathrm{overload}},
$$

with

$$
w_{\mathrm{over}} = 3.0.
$$

If the same candidate is also reliability-infeasible, then the implementation strengthens this term:

$$
\Pi_{t,j}^{\mathrm{overload}}
\leftarrow
1.5\,\Pi_{t,j}^{\mathrm{overload}}.
$$

So $\Pi_{t,j}^{\mathrm{overload}}$ means:

- under light URLLC pressure, it is small
- as more URLLC packets are packed into the slot, it grows linearly
- if a candidate is both overload-prone and reliability-poor, it becomes harsher

## 8. Overlay Preference Derivation

The reward should prefer overlay over puncture **only when overlay is genuinely better**.
To express this, the reward uses two counterfactual comparisons.

### 8.1 Overlay gain from eMBB-loss comparison

We first compare the eMBB loss under puncture and overlay:

$$
g_{t,j}^{\mathrm{ov}}
=
\mathrm{clip}
\left(
\frac{L_{t,j}^{\mathrm{pun}} - L_{t,j}^{\mathrm{ov}}}
{\max(L_{t,j}^{\mathrm{pun}}, \varepsilon)},
0, 1.5
\right).
$$

Interpretation:

- if overlay causes much less eMBB loss than puncture, then $g_{t,j}^{\mathrm{ov}}$ is positive
- if overlay is not feasible or not better, the gain is near zero

### 8.2 Overlay margin from utility comparison

We also compare the utility proxies:

$$
\Delta u_{t,j}
=
\mathrm{clip}
\left(
\frac{U_{t,j}^{\mathrm{ov}} - U_{t,j}^{\mathrm{pun}}}
{|U_{t,j}^{\mathrm{ov}}| + |U_{t,j}^{\mathrm{pun}}| + \varepsilon},
-1, 1
\right).
$$

Interpretation:

- $\Delta u_{t,j} > 0$ means overlay is more attractive than puncture
- $\Delta u_{t,j} < 0$ means puncture is preferred by the utility proxy

### 8.3 Reward if the executed mode is overlay

If $m_{t,j} = \mathrm{OVERLAY}$, then:

$$
r_{t,j}^{\mathrm{mode}}
=
\lambda_{\mathrm{ov\text{-}gain}} g_{t,j}^{\mathrm{ov}}
+
\lambda_{\mathrm{ov\text{-}margin}} \max(\Delta u_{t,j}, 0).
$$

Current coefficients:

$$
\lambda_{\mathrm{ov\text{-}gain}} = 0.00,
\qquad
\lambda_{\mathrm{ov\text{-}margin}} = 0.70.
$$

So in the current implementation, the dominant overlay incentive is the positive utility margin, not the raw gain term.

## 9. Puncture Penalty Derivation

Puncture is always charged a fixed extra penalty:

$$
-\lambda_{\mathrm{punct}}.
$$

In addition, if overlay would have been the better choice, puncture is penalized again through a missed-opportunity term:

$$
-\lambda_{\mathrm{miss}}
\max\left(g_{t,j}^{\mathrm{ov}}, \Delta u_{t,j}\right)
\mathbf{1}\!\left[\Delta u_{t,j} > 0\right].
$$

Therefore, if $m_{t,j} = \mathrm{PUNCTURE}$:

$$
r_{t,j}^{\mathrm{mode}}
=
- \lambda_{\mathrm{punct}}
- \lambda_{\mathrm{miss}}
\max\left(g_{t,j}^{\mathrm{ov}}, \Delta u_{t,j}\right)
\mathbf{1}\!\left[\Delta u_{t,j} > 0\right].
$$

Current coefficients:

$$
\lambda_{\mathrm{punct}} = 0.35,
\qquad
\lambda_{\mathrm{miss}} = 0.75.
$$

This means:

- puncture is always discouraged
- puncture is discouraged even more when overlay would have been a better decision

## 10. Power Penalty Derivation

The power term has two parts.

### 10.1 Absolute power usage

Normalize the executed power:

$$
p_{t,j}
=
\frac{P_{t,j}^{\mathrm{act}}}{P_{\max}}.
$$

Then penalize it:

$$
- \lambda_{\mathrm{pow}} p_{t,j}.
$$

### 10.2 Power projection penalty

If the raw requested power is projected heavily by the shield, that means the action was numerically unrealistic.

Define the projection magnitude:

$$
\pi_{t,j}
=
\frac{|P_{t,j}^{\mathrm{act}} - P_{t,j}^{\mathrm{raw}}|}
{\max(P_{t,j}^{\mathrm{req}}, \varepsilon)}.
$$

Then penalize it:

$$
- \lambda_{\mathrm{proj}} \pi_{t,j}.
$$

So the full power term is:

$$
r_{t,j}^{\mathrm{pow}}
=
- \lambda_{\mathrm{pow}} p_{t,j}
- \lambda_{\mathrm{proj}} \pi_{t,j}.
$$

Current coefficients:

$$
\lambda_{\mathrm{pow}} = 0.03,
\qquad
\lambda_{\mathrm{proj}} = 0.08.
$$

## 11. Safety and Execution Penalties

These penalties exist because the final executed action may differ from the raw policy proposal.

### 11.1 Bad KEEP penalty

If the agent uses KEEP while a useful candidate exists:

$$
- \lambda_{\mathrm{keep}}.
$$

### 11.2 Invalid-action penalty

If the raw action must be rewritten because of invalid fallback or joint reliability rewrite:

$$
- \lambda_{\mathrm{invalid}}.
$$

### 11.3 Collision-rewrite penalty

If packet collision handling rewrites the action:

$$
- \lambda_{\mathrm{coll}}.
$$

Thus:

$$
r_{t,j}^{\mathrm{safety}}
=
- \lambda_{\mathrm{keep}}
\mathbf{1}\!\left[\text{KEEP while useful candidate exists}\right]
- \lambda_{\mathrm{invalid}}
\mathbf{1}\!\left[\text{invalid fallback or joint reliability rewrite}\right]
- \lambda_{\mathrm{coll}}
\mathbf{1}\!\left[\text{collision rewrite}\right].
$$

Current coefficients:

$$
\lambda_{\mathrm{keep}} = 0.15,
\qquad
\lambda_{\mathrm{invalid}} = 0.12,
\qquad
\lambda_{\mathrm{coll}} = 0.20.
$$

## 12. Slot-Level Terminal Reward Derivation

The terminal reward is only added when the slot ends:

$$
r_t^{\mathrm{terminal}}
=
r_t^{\mathrm{unsched}}
+
r_t^{\mathrm{rate}}
+
r_t^{\mathrm{fair}}.
$$

These are slot-level quantities, so they are kept at the terminal level instead of being forced into each step.

### 12.1 Unscheduled URLLC ratio

Define the fraction of URLLC packets that remain unscheduled:

$$
\alpha_t^{\mathrm{unsched}}
=
\frac{N_t^{\mathrm{unscheduled}}}
{\max(N_t^{\mathrm{active}}, 1)}.
$$

Then:

$$
r_t^{\mathrm{unsched}}
=
- \lambda_{\mathrm{uns}}
\alpha_t^{\mathrm{unsched}}.
$$

Current coefficient:

$$
\lambda_{\mathrm{uns}} = 1.50.
$$

### 12.2 Aggregate eMBB throughput

The slot-level aggregate eMBB throughput under SR-MAPPO is:

$$
R_t^{\mathrm{SR}}
=
\sum_{q=1}^{Q} R_{t,q}^{\mathrm{SR}}.
$$

We normalize the throughput by a fixed scale (default $10^6$ bps) to keep reward magnitude stable:

$$
\tilde{R}_t^{\mathrm{SR}}
=
\frac{R_t^{\mathrm{SR}}}{R_{\mathrm{norm}}}.
$$

Therefore:

$$
r_t^{\mathrm{rate}}
=
\lambda_{\mathrm{rate}} \, \tilde{R}_t^{\mathrm{SR}}.
$$

Current coefficient:

$$
\lambda_{\mathrm{rate}} = 0.80.
$$

### 12.3 Jain fairness derivation

This is the part you asked to make explicit.

Let the SR-MAPPO eMBB rate vector be:

$$
\mathbf{R}_t^{\mathrm{SR}}
=
\left[
R_{t,1}^{\mathrm{SR}},
R_{t,2}^{\mathrm{SR}},
\dots,
R_{t,Q}^{\mathrm{SR}}
\right].
$$

Then its Jain fairness index is:

$$
J_t^{\mathrm{SR}}
=
\frac{
\left(\sum_{q=1}^{Q} R_{t,q}^{\mathrm{SR}}\right)^2
}{
Q \sum_{q=1}^{Q} \left(R_{t,q}^{\mathrm{SR}}\right)^2 + \varepsilon
}.
$$

Interpretation:

- $J=1$ means perfectly even per-user eMBB rates
- smaller $J$ means stronger rate imbalance

Therefore:

$$
r_t^{\mathrm{fair}}
=
\lambda_{\mathrm{fair}} \, J_t^{\mathrm{SR}}.
$$

Current coefficient:

$$
\lambda_{\mathrm{fair}} = 0.10.
$$

So the symbol $J$ comes directly from **Jain's fairness index of the final eMBB rate vector**, not from an undefined heuristic score.

## 13. Final Reward Expression

Putting all parts together, the implemented shared reward is:

$$
r_t^{\mathrm{team}}
=
\sum_{j=1}^{N}
\Bigg[
\lambda_{\mathrm{sched}}
- \lambda_{\mathrm{dmg}} d_{t,j}(m)
- \lambda_{\mathrm{pow}} p_{t,j}
- \lambda_{\mathrm{proj}} \pi_{t,j}
$$
$$
\qquad
+ \mathbf{1}[m_{t,j}=\mathrm{OVERLAY}]
\left(
\lambda_{\mathrm{ov\text{-}gain}} g_{t,j}^{\mathrm{ov}}
+ \lambda_{\mathrm{ov\text{-}margin}} \max(\Delta u_{t,j},0)
\right)
$$
$$
\qquad
+ \mathbf{1}[m_{t,j}=\mathrm{PUNCTURE}]
\left(
- \lambda_{\mathrm{punct}}
- \lambda_{\mathrm{miss}}
\max(g_{t,j}^{\mathrm{ov}}, \Delta u_{t,j})
\mathbf{1}[\Delta u_{t,j}>0]
\right)
$$
$$
\qquad
- \lambda_{\mathrm{keep}}\mathbf{1}[\text{bad KEEP}]
- \lambda_{\mathrm{invalid}}\mathbf{1}[\text{invalid rewrite}]
- \lambda_{\mathrm{coll}}\mathbf{1}[\text{collision rewrite}]
\Bigg]
+
r_t^{\mathrm{terminal}},
$$

with

$$
r_t^{\mathrm{terminal}}
=
- \lambda_{\mathrm{uns}} \alpha_t^{\mathrm{unsched}}
+ \lambda_{\mathrm{rate}} \tilde{R}_t^{\mathrm{SR}}
+ \lambda_{\mathrm{fair}} J_t^{\mathrm{SR}}.
$$

## 14. Why This Form Is Used

This reward should be interpreted as:

- the **local terms** teach damage-aware coexistence behavior
- the **rate term** aligns training with absolute eMBB throughput
- the **fairness term** prevents the policy from improving aggregate throughput by sacrificing too many weak eMBB users
- the **unscheduled term** prevents the policy from ignoring URLLC service
- the **reliability requirement remains a hard constraint**, not a reward tradeoff term

## 15. Current Numerical Coefficients

For convenience, the current coefficients are:

$$
\lambda_{\mathrm{sched}} = 0.01,\quad
\lambda_{\mathrm{dmg}} = 0.30,\quad
\lambda_{\mathrm{ov\text{-}gain}} = 0.00,\quad
\lambda_{\mathrm{ov\text{-}margin}} = 0.70,
$$

$$
\lambda_{\mathrm{punct}} = 0.35,\quad
\lambda_{\mathrm{miss}} = 0.75,\quad
\lambda_{\mathrm{pow}} = 0.03,\quad
\lambda_{\mathrm{proj}} = 0.08,
$$

$$
\lambda_{\mathrm{keep}} = 0.15,\quad
\lambda_{\mathrm{invalid}} = 0.12,\quad
\lambda_{\mathrm{coll}} = 0.20,
$$

$$
\lambda_{\mathrm{uns}} = 1.50,\quad
\lambda_{\mathrm{rate}} = 0.80,\quad
\lambda_{\mathrm{fair}} = 0.10.
$$

## 16. Exact Implementation Mapping

Current code locations:

- [config.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/config.py)
  - `RewardConfig`
- [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py)
  - `step()`
  - `_counterfactual_local_reward()`
  - `_prepare_original_greedy_reference()`
  - `_compute_jain_fairness()`

Implementation mapping:

- $R_t^{\mathrm{SR}}$ comes from `summary["embb_total_rate"]`
- $J_t^{\mathrm{SR}}$ comes from `summary["jain_fairness"]`

So if you need to explain the reward in a report or oral presentation, every symbol in the formulas above now has a concrete origin in code.
