# System Model, Parameters, and Power/Channel Calculations

This document consolidates the **system-level parameters**, **power/energy calculations**, and **communication/channel models** used by the simulator. It is intended to be read directly as Markdown.

Primary code references:

- [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)
- [channel_model.py](/d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py)
- [capacity_models.py](/d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py)
- [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py)
- [simulation.py](/d:/URLLC_eMBB_Coexisting/Greedy/simulation.py)

---

## 1. System Topology and Time Structure

Defined in [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py), class `SystemConfig`.

Core parameters:

- `num_uavs = 3`
- `num_embb_users = 6`
- `num_urllc_users = 3`
- `num_subcarriers = 20` (RBs per UAV)
- `num_slots = 10`
- `num_minislots = 8` per slot
- `slot_duration = 1.0 ms`
- `minislot_duration = slot_duration / num_minislots`

Derived parameters:

- `subcarrier_bw = bandwidth / num_subcarriers`
- `channel_uses_per_minislot = round(subcarrier_bw * minislot_duration_s)`
- `noise_power` computed from thermal noise and noise figure

---

## 2. Bandwidth and Noise Model

From [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py), `SystemConfig`:

- `carrier_frequency = 2.0e9 Hz`
- `bandwidth = 10e6 Hz`
- `noise_figure = 5 dB`

Noise power:

$$
N_0 = -174 \text{ dBm/Hz}
$$

$$
P_{noise}(\text{dBm}) = N_0 + 10\log_{10}(B_{RB}) + NF
$$

$$
P_{noise}(\text{W}) = 10^{\frac{P_{noise}(\text{dBm}) - 30}{10}}
$$

Where:

- $B_{RB} = \text{subcarrier\_bw}$
- $NF = \text{noise\_figure}$

Implementation:

- `SystemConfig._calculate_noise_power()` in [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)

---

## 3. UE and UAV Spatial Distribution

From [channel_model.py](/d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py), `generate_topology()`:

- UAV positions are fixed by `uav_positions` and `uav_altitudes`.
- UE positions are **Gaussian clusters around UAVs**.
- Cluster spread is controlled by `user_cluster_spread`.
- UE coordinates are clipped to `area_width / area_height` with a boundary margin.

This is **not** a Poisson cluster process; it is deterministic UAV centers with Gaussian scatter.

Topology outputs:

- `user_positions`
- `uav_positions`
- `horizontal_distances`
- `distances` (3D including altitude)
- `serving_hints` (initial cluster association)

---

## 4. Air-to-Ground Channel Model

From [channel_model.py](/d:/URLLC_eMBB_Coexisting/Greedy/channel_model.py):

### 4.1 LoS Probability

$$
P_{\text{LoS}} = \frac{1}{1 + a \exp(-b(\theta - a))}
$$

Where:

- $\theta$ is the elevation angle in degrees
- `a2g_los_a = 9.61`, `a2g_los_b = 0.16`

### 4.2 Path Loss

Free-space path loss:

$$
PL_{\text{FS}}(d) = 20\log_{10}(d) + 20\log_{10}(f_c) + 20\log_{10}\left(\frac{4\pi}{c}\right)
$$

Add LoS or NLoS extra loss:

- `a2g_eta_los = 1.0 dB`
- `a2g_eta_nlos = 20.0 dB`

### 4.3 Shadowing

Log-normal shadowing:

- `shadowing_std = 4.0 dB`

### 4.4 Large-scale fading

$$
g_{\text{LS}} = 10^{-\frac{PL + \text{shadowing}}{10}}
$$

### 4.5 Small-scale fading

Per RB:

- Rayleigh or Rician (configurable by `csi_generation_method`)
- Rician uses `rician_k_factor = 5.0`

Final channel gain:

$$
h = \sqrt{g_{\text{LS}}} \cdot h_{\text{small}}
$$

Output:

$$
|h|^2
$$

---

## 5. Capacity and Reliability Models

From [capacity_models.py](/d:/URLLC_eMBB_Coexisting/Greedy/capacity_models.py):

### 5.1 eMBB Shannon Capacity

$$
C = B \log_2(1 + \text{SNIR})
$$

### 5.2 URLLC Finite Blocklength Approximation

For channel uses $n$ and error probability $\epsilon$:

$$
R \approx \log_2(1+\gamma) - \sqrt{\frac{V}{n}} Q^{-1}(\epsilon)
$$

Implementation uses:

- `decoding_error_probability(snir, packet_bits, channel_uses)`
- `finite_blocklength_capacity(...)`

Key URLLC parameters:

- `target_error_probability = 1e-5`
- `packet_lengths = [120,150,180]` bits

---

## 6. Power Model

### 6.1 Power limits

From [config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py):

- URLLC `power_limits = [26,26,26] dBm`
- eMBB `power_limits = [23,23,23,23,23,23] dBm`
- Global power clamp: `power_upper_bound = 1.0 W`

### 6.2 dBm to W conversion

$$
P(W) = 10^{\frac{P(\text{dBm}) - 30}{10}}
$$

Implemented in `ResourceAllocator._dbm_to_watts()`.

### 6.3 eMBB power allocation

From [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py):

- Each eMBB user has a max power from `power_limits`.
- Allocate per-RB power as:

$$
P_{RB} = \frac{P_{user}}{\text{assigned RBs}}
$$

- eMBB total per-user power:

$$
P_{user} = P_{\max} \cdot \text{load fraction}
$$

Where `load fraction = assigned_RBs / total_RBs`.

### 6.4 URLLC power allocation

For each URLLC packet:

1. Compute minimum power via bisection so that:

$$
P_e(\gamma) \le \epsilon_{\text{target}}
$$

2. Use finite blocklength error probability from `CapacityModels.decoding_error_probability`.

3. Respect `power_upper_bound` and per-user `power_limits`.

### 6.5 Total power

From [simulation.py](/d:/URLLC_eMBB_Coexisting/Greedy/simulation.py):

$$
P_{\text{total}} =
\frac{\sum P_{\text{URLLC}}}{\text{num\_minislots}}
+
\sum P_{\text{eMBB}}
$$

Both components are in Watts.

---

## 7. Scheduling and Coexistence Logic (Baseline)

From [resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py) and [simulation.py](/d:/URLLC_eMBB_Coexisting/Greedy/simulation.py):

Flow per slot:

1. Generate channel gains for all users and RBs.
2. Assign each user to one UAV using long-term large-scale gains.
3. Allocate eMBB RBs greedily per UAV.
4. Sample URLLC arrivals using Poisson rate.
5. For each URLLC packet, search best feasible action:
   - Puncture or Overlay
   - Minimum power for target reliability
   - Utility = eMBB loss + URLLC reward - power penalty - overload penalty
6. Recompute eMBB rates after URLLC coexistence.

Utility function (core idea):

$$
U = \Delta R_{\text{eMBB}} + w_u f(\text{reliability}) - w_p P - \Pi_{\text{overload}}
$$

Where:

- `urllc_utility_weight = 8.0`
- `power_penalty_weight = 0.05`
- `overload_penalty_weight = 3.0`
- `admission_load_limit = 0.35`

---

## 8. Interference and SIC

Interference model:

- Inter-cell interference from other UAVs on the same RB.
- URLLC overlay adds residual interference to eMBB:

$$
I_{\text{residual}} = \text{sic\_residual\_factor} \cdot P_{\text{URLLC}} \cdot |h|^2
$$

Key parameters:

- `sic_residual_factor = 0.02`
- `embb_min_sic_snir_db = 0.0`

---

## 9. Admission and Reliability Constraint

URLLC packets are admitted only if:

- `decoding_error_probability <= target_error_probability`
- optional overload penalty and utility threshold are satisfied

