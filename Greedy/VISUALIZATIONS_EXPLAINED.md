# Visualizations and Examples Explained

## Overview
This document explains what each visualization shows and what each example code demonstrates.

---

## Generated Visualizations

### 1. **power_vs_density.png** - Multi-Panel Performance Analysis
**What it shows:** How the system performs across different user density scenarios (1-10 users per UAV)

**Panels:**
- **Top-Left (Power Consumption):** Total system power usage increases with user density (more users = more power needed)
- **Top-Right (eMBB Throughput with Explanation):** Shows the INVERSE relationship - explained with a yellow annotation box
  - Low density (1 user/UAV) → 3 total eMBB users → lower aggregate throughput
  - High density (10 users/UAV) → 30 total eMBB users → higher aggregate throughput
  - **KEY INSIGHT:** Total system throughput depends on NUMBER of users!
- **Middle-Left (URLLC Reliability):** Success rate stays above 99% target across all density points
- **Middle-Right (Explanation Box):** Detailed breakdown of why eMBB is inverse
- **Bottom (Performance Summary):** Statistical ranges and averages across all density points

**Why This Matters:** Shows system scalability and resource allocation efficiency under varying loads

---

### 2. **allocation_timeline.png** - Instantaneous Resource Allocation (Time-Frequency Grid)
**What it shows:** How the 50 RBs are allocated between URLLC and eMBB users in a single time slot

**Format (Reference-style grid):**
- **Rows:** Individual radio bearers (RBs) numbered 0-49
- **Columns:** User assignment (which user/UAV gets this RB)
- **Colors:**
  - **Brown/Dark:** URLLC users (need guaranteed reliability)
  - **Blue/Light:** eMBB users (share remaining capacity)
  - **White/Empty:** Unallocated capacity

**Example Reading:**
- If RB 0-5 are brown, URLLC users have priority in these RBs
- If RB 6-49 are blue, eMBB users share these RBs
- Title shows user configuration: "用户配置: 6 eMBB + 3 URLLC 用户" (6 eMBB + 3 URLLC users)

**Why This Matters:** Visualizes the actual physical resource allocation pattern - confirms URLLC gets priority, eMBB uses remainder

---

### 3. **performance_timeline.png** - Performance Over Multiple Slots (Time-Series)
**What it shows:** How eMBB rate, URLLC success, and power consumption vary across 10 consecutive time slots

**Components:**
- **Y-axis (Left):** Rate in Mbps
- **Y-axis (Right):** Success rate (0-1) or Power (W)
- **X-axis:** Time slot number (0-9)
- **Colored lines:** Different metrics (rate, success, power)
- **Title shows:** "用户配置: 6 eMBB + 3 URLLC 用户" - which user configuration was simulated
- **Shaded bands:** Performance ranges showing variability

**Reading Example:**
- If eMBB rate fluctuates between 80-100 Mbps across slots, shows channel variability
- If URLLC success is consistently at 1.0, shows reliable guarantee
- If power varies 1.5-2.5W, shows adaptive power allocation

**Why This Matters:** Shows temporal behavior and variance - important for real-time systems

---

### 4. **allocation_heatmap.png** - Alternative Visualization
**Note:** This is an older format; allocation_timeline.png is the recommended version using time-frequency grid format

---

## Code Examples (examples.py)

### Example 1: Basic Simulation
**What it does:**
- Runs the complete simulation with DEFAULT system parameters
- Demonstrates the standard workflow

**Key Parameters:**
- 3 UAVs, 6 eMBB users, 3 URLLC users, 50 RBs, 10 slots

**Output:**
- Average eMBB rate, URLLC success rate, average power
- Resource utilization percentage

**Learning Purpose:**
- Understand baseline system performance
- See how to use `create_simulation()` function
- Learn expected output metrics

**Sample Output:**
```
eMBB Rate: 88.8591 Mbps
URLLC Success: 1.0000
Total Power: 2021.09 mW
RB Utilization: 100.00%
```

---

### Example 2: Custom Scenario - Heavy Load
**What it does:**
- Simulates a DIFFERENT network configuration
- Tests with 10 eMBB + 5 URLLC users (heavier load)

**Purpose:**
- Show how to pass custom parameters to `create_simulation()`
- Demonstrate system behavior under high load
- Verify algorithm scales to more users

**Key Parameters:**
- num_embb_users=10, num_urllc_users=5 (vs default 6 and 3)
- Same 50 RBs but more contention

**Learning Purpose:**
- Learn how to configure custom scenarios
- Understand performance degradation/scaling patterns
- See parameter passing syntax

**Sample Output:**
```
eMBB Users: 10, URLLC Users: 5
eMBB Rate: 201.5894 Mbps (higher total!)
URLLC Success: 0.9999 (still reliable)
```

---

### Example 3: Single Slot Detailed Inspection
**What it does:**
- Runs simulation for JUST ONE TIME SLOT
- Prints per-user allocation details (not just aggregates)

**Deep Details:**
- eMBB rate for EACH user individually
- URLLC reliability for EACH user
- Exact power allocation to each UAV
- Which users share which RBs

**Learning Purpose:**
- Understand the granular allocation decisions
- See how fairness is handled (some users get more RBs than others)
- Inspect algorithm internals

**Sample Output:**
```
Slot 0 Detailed Metrics:
eMBB Users:
  User 0: 18.2970 Mbps
  User 1: 22.5650 Mbps
  ...
URLLC Success:
  User 0: 1.0000
  User 1: 1.0000
  User 2: 1.0000
```

---

### Example 4: Capacity Model Calculations
**What it does:**
- Demonstrates the THEORETICAL capacity formulas
- NOT a full simulation - just math/formulas

**Formulas Shown:**
1. **Shannon Capacity:** C = B·log₂(1 + P/(N₀·B))
   - Classical formula for eMBB users
   - Shows rate vs SNIR (Signal-to-Noise Ratio)

2. **Finite Blocklength Capacity:** Modified Shannon for URLLC
   - Based on Polyanskiy-Poor-Verdú formula
   - Accounts for short packets (150 bits)
   - Shows achievable rate and error probability

**Learning Purpose:**
- Understand information-theoretic foundations
- Verify capacity functions are correctly implemented
- See how different packet lengths affect capacity

**Sample Output:**
```
Shannon (eMBB):
  SNIR: 10 dB
  Capacity: 3.4594 Mbps

Finite Blocklength (URLLC):
  Packet: 150 bits
  Uses: 14 channel uses
  Rate: 1.6638 bits/use
  Error Prob: 3.16e-17 (extremely low!)
```

---

### Example 5: Channel Model Generation
**What it does:**
- Generates random channel gains
- Shows statistics of the generated channels

**Channel Model:**
- 3GPP air-to-ground propagation
- Includes: path loss, shadowing, fading
- Returns gain matrix shape (5 users, 3 UAVs, 50 subcarriers)

**Learning Purpose:**
- Understand channel model implementation
- Learn how realistic wireless channels are generated
- See typical channel statistics

**Sample Output:**
```
Channel Shape: (5, 3, 50)
  - 5 users (3 URLLC + 2 eMBB in this example)
  - 3 UAVs
  - 50 subcarriers (RBs)

Statistics:
  Mean Gain (dB): -98.46 (very weak signal - typical for air-to-ground)
  Std Dev: 14.71 (high variability due to fading)
```

---

### Example 6: Resource Allocator Direct Usage
**What it does:**
- Calls the allocation functions directly
- Shows low-level algorithm steps

**Three Phases:**
1. **Phase 1 (URLLC):** Calculate minimum power needed for URLLC reliability
   - Output: URLLC power allocation + individual user reliability
2. **Phase 2 (eMBB):** Greedy algorithm for remaining capacity
   - Output: eMBB rate + RB allocation matrix
3. **Phase 3 (NOMA):** Decide NOMA vs Puncturing
   - Output: NOMA usage percentage

**Learning Purpose:**
- Understand the three-step algorithm structure
- Debug allocation decisions step-by-step
- Understand intermediate data structures

**Sample Output:**
```
Phase 1: URLLC Power Allocation
  URLLC Power Shape: (3, 3) - 3 users × 3 UAVs
  Average Reliability: 0.9999

Phase 2: eMBB Greedy Allocation
  Total eMBB Rate: 95.3248 Mbps
  RB Allocation Shape: (50, 3) - 50 RBs × 3 UAVs

Phase 3: NOMA vs Puncturing
  NOMA Usage: 106/200 (53.0%)
```

---

## Summary: When to Use Each Example

| Example | Use Case | Learn What |
|---------|----------|-----------|
| Ex 1 | Get started quickly | Basic workflow, typical performance |
| Ex 2 | Test different configurations | Custom parameters, scaling |
| Ex 3 | Debug fairness, imbalance issues | Per-user allocation details |
| Ex 4 | Verify math/formulas | Information theory foundations |
| Ex 5 | Understand channel model | Propagation, fading statistics |
| Ex 6 | Debug algorithm behavior | Three-phase resource allocation |

---

## Recommended Reading Order

1. **Start with:** Example 1 (Basic Simulation)
2. **Then view:** power_vs_density.png (understand system scalability)
3. **Then view:** performance_timeline.png (see temporal behavior)
4. **Deep dive:** Example 3 (Single Slot inspection)
5. **Theory check:** Example 4 (Capacity formulas)
6. **Advanced:** Example 6 (Algorithm internals)

---

## Interpreting the Main Results

**From examples.py output:**

```
[Example 1 Results]
eMBB Rate:     88.8591 Mbps  ← Good throughput for non-critical traffic
URLLC Success: 1.0000        ← Perfect reliability (exceeds 99% target)
Total Power:   2021.09 mW    ← Power efficiency of 49 Mbps/W

[Example 2 Results - Heavy Load]
eMBB Rate:     201.5894 Mbps ← Scales with more users (10 instead of 6)
URLLC Success: 0.9999        ← Still meets 99% requirement
```

**Conclusions:**
- ✅ Algorithm successfully prioritizes URLLC (always >99% success)
- ✅ Provides substantial eMBB throughput when capacity available
- ✅ Scales to handle heavy loads (10 eMBB users successfully)
- ✅ Power-efficient (high Mbps per Watt)

---

## Questions Answered

**Q: Why is eMBB throughput inverse to user density?**
A: When you have MORE users with the SAME RBs, total throughput increases because you're using RBs to serve more people, even if each person gets less. It's the aggregate effect that matters for total system capacity.

**Q: How can URLLC success be EXACTLY 1.0000?**
A: The algorithm allocates enough power to guarantee the target reliability. In these scenarios, channel conditions allow this without exceeding power budgets. In more constrained scenarios, you'd see values like 0.9999 or 0.99.

**Q: What does the time-frequency grid in allocation_timeline.png show?**
A: It shows which RB goes to which user. URLLC users (brown) get priority bandwidths for guaranteed delivery. eMBB users (blue) share the remainder. This is the physical "scheduling" of the radio resources.

---

## Advanced Topics

### Modifying Examples

To create your own test case, copy Example 2 and modify:

```python
def my_custom_scenario():
    # Change parameters here
    num_embb = 12
    num_urllc = 6
    num_rbs = 100
    
    simulator = create_simulation(
        num_embb_users=num_embb,
        num_urllc_users=num_urllc,
        num_subcarriers=num_rbs
    )
    
    result = simulator.run_full_simulation(num_slots=20)
    print(f"eMBB Rate: {result['embb_rates_avg']/1e6:.2f} Mbps")
```

### Understanding Algorithm Performance

- Higher eMBB rate = good resource utilization
- URLLC success ≥ 99% = meets reliability target
- Power < 5W = efficient for battery-powered UAVs
- RB Utilization = 100% = all spectrum is being used

---

## File Dependencies

```
examples.py (you run this)
├── config.py         → System parameters
├── simulation.py     → Main simulation engine
│   ├── channel_model.py        → Wireless propagation
│   ├── capacity_models.py      → Rate calculations
│   └── resource_allocator.py   → Allocation algorithm
└── visualization.py  → Plot generation
```

Each example uses the framework in different ways - study which modules each example imports and calls.
