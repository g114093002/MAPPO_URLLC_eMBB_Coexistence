# QUICK START GUIDE

## Installation & First Run - 5 Minutes

### Prerequisites
- Python 3.7+ (or 3.8, 3.9, 3.10, 3.11, 3.12)
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
cd d:\URLLC_eMBB_Coexisting\Greedy

# Install required packages
pip install -r requirements.txt
```

**Packages installed:**
- numpy (matrix operations)
- scipy (special functions)
- matplotlib (visualization)

### Step 2: Run Full Simulation

```bash
python main.py
```

**What happens:**
1. System prints configuration parameters
2. Runs 10 time slots of simulation
3. Prints per-slot allocation results
4. Performs user density sweep (5 different densities)
5. Generates plots in `./results/` directory
6. Prints final summary statistics

**Expected runtime:** 10-30 seconds

**Output includes:**
```
✓ Average eMBB Rate: ~95 Mbps
✓ Average URLLC Success: >99%
✓ Average Power: ~1.8 W
✓ 3 PNG plots in ./results/
```

### Step 3: View Results

Generated plots:
- `power_vs_density.png` - Performance vs user density
- `allocation_heatmap.png` - RB allocation visualization  
- `performance_timeline.png` - Metrics over time

### Step 4: Run Examples

```bash
python examples.py
```

Demonstrates 6 different usage patterns (with code examples).

---

## Configuration Quick Reference

All modifiable parameters are in `config.py`:

```python
# Number of users
sys_cfg.num_embb_users = 6       # Change this!
sys_cfg.num_urllc_users = 3      # Change this!

# Spectrum
sys_cfg.num_subcarriers = 50     # More RBs → higher rate
sys_cfg.bandwidth = 10e6         # Total bandwidth

# URLLC requirements
urllc_cfg.target_error_probability = 1e-5  # Stricter = more power

# Simulation
sim_cfg.num_slots = 10           # More slots → longer sim
sim_cfg.verbose = True           # Detailed output
```

---

## Three Common Scenarios

### Scenario A: Conservative (Low Load)
```python
sys_cfg.num_embb_users = 4
sys_cfg.num_urllc_users = 2
sys_cfg.num_subcarriers = 50
```

### Scenario B: Baseline (Medium Load)  ← DEFAULT
```python
sys_cfg.num_embb_users = 6
sys_cfg.num_urllc_users = 3
sys_cfg.num_subcarriers = 50
```

### Scenario C: Heavy Load
```python
sys_cfg.num_embb_users = 10
sys_cfg.num_urllc_users = 5
sys_cfg.num_subcarriers = 100
```

---

## What Each File Does

| File | What it does |
|------|-------------|
| **main.py** | Run full simulation (START HERE) |
| **examples.py** | 6 code examples showing how to use framework |
| **config.py** | All system parameters you might want to change |
| **channel_model.py** | Wireless channel generation |
| **capacity_models.py** | Rate calculations (Shannon, finite blocklength) |
| **resource_allocator.py** | The core algorithm (Bisection + Greedy) |
| **simulation.py** | Orchestrates the whole simulation |
| **visualization.py** | Creates plots and analyzes results |
| **README.md** | Full technical documentation |

---

## Typical Output (Console)

```
======================================================================
Multi-UAV URLLC-eMBB Coexistence Resource Allocation
Algorithm: Bisection Search + Greedy RB Allocation
======================================================================

System Configuration:
  UAVs: 3
  eMBB Users: 6
  URLLC Users: 3
  Total RBs: 50

Running Base Case Simulation (10 Time Slots)
...

============================================================
Slot 0
============================================================

Phase 1 - URLLC Power Allocation:
  URLLC Success Rate: 1.0000
  URLLC Power (W): 0.765803

Phase 2 - eMBB Greedy Allocation:
  Total eMBB Rate: 100.5646 Mbps
  eMBB Power (W): 1.197157
  RB Utilization: 100.00%

...

FINAL RESULTS
eMBB Rate:       95.41 Mbps
URLLC Success:   99.99%
Total Power:     1.83 W
```

---

## Simple Modification Example

### Task: Increase URLLC users and see impact

1. Open `config.py`
2. Find line: `self.num_urllc_users = 3`
3. Change to: `self.num_urllc_users = 5`
4. Save file
5. Run: `python main.py`
6. Compare results! (Lower eMBB rate expected)

---

## Troubleshooting

### Q: Module not found error?
**A:** Did you install dependencies?
```bash
pip install -r requirements.txt
```

### Q: Plots not showing?
**A:** They're saved to `./results/` directory as PNG files. Open them with an image viewer.

### Q: URLLC success rate < 99%?
**A:** System is congested. Try:
- Fewer eMBB users
- More RBs
- Looser URLLC requirement

### Q: Low eMBB rate?
**A:** Try:
- Increase RBs
- Fewer URLLC users (uses less power)
- Enable better channels (increase LoS probability)

---

## Code Example: Your First Modification

```python
#!/usr/bin/env python
"""Run simulation with custom parameters"""

from config import SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig, SimulationConfig
from simulation import create_simulation
from visualization import create_plotter, ResultsAnalyzer

# Custom configuration
sys_cfg = SystemConfig()
sys_cfg.num_embb_users = 8          # 8 eMBB users
sys_cfg.num_urllc_users = 4         # 4 URLLC users
sys_cfg.num_subcarriers = 64        # 64 RBs

urllc_cfg = URLLCConfig()
embb_cfg = eMBBConfig()
algo_cfg = AlgorithmConfig()

sim_cfg = SimulationConfig()
sim_cfg.num_slots = 5               # Shorter test
sim_cfg.verbose = True

# Run simulation
print("Running custom simulation...")
simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
results = simulation.run_full_simulation()

# Print results
print(f"\n=== RESULTS ===")
print(f"eMBB Rate: {results['avg_embb_rate']/1e6:.2f} Mbps")
print(f"URLLC Success: {results['avg_urllc_success']:.4f}")
print(f"Power: {results['avg_total_power']*1e3:.2f} mW")

# Save this as my_simulation.py and run: python my_simulation.py
```

---

## Learning Resources

### For Understanding the Algorithm
1. Read README.md section "Algorithm Description"
2. Look at `resource_allocator.py` Phase 1, 2, 3
3. Run examples.py example 6

### For Understanding Capacity
1. See `capacity_models.py` docstrings
2. Read README.md "Mathematical Foundations"
3. Run examples.py example 4

### For Understanding Simulation
1. Look at `simulation.py` main loop
2. Run `examples.py` example 1 and 3
3. Check console output format

---

## Performance Expectations

### With Default Configuration (3 UAVs, 6 eMBB, 3 URLLC, 50 RBs):

| Metric | Value |
|--------|-------|
| eMBB Rate | 80-110 Mbps |
| URLLC Success | >99.9% |
| Power Usage | 1.5-2.5 W |
| RB Utilization | 90-100% |
| Execution Time | 5-10 seconds |

### With Heavy Load (10 eMBB, 5 URLLC, 100 RBs):

| Metric | Value |
|--------|-------|
| eMBB Rate | 150-200 Mbps |
| URLLC Success | >99% |
| Power Usage | 3-5 W |
| RB Utilization | 85-95% |
| Execution Time | 15-30 seconds |

---

## Files Created During Run

- `./results/power_vs_density.png` - Plot 1
- `./results/allocation_heatmap.png` - Plot 2
- `./results/performance_timeline.png` - Plot 3
- `./results/channel_cdf.png` - Plot 4 (if called)
- `./results/` - Directory (auto-created)

---

## Next Steps After First Run

1. ✅ Run `python main.py` (try default)
2. ✅ Check `./results/` directory for plots
3. ✅ Run `python examples.py` to learn
4. ✅ Modify one parameter in `config.py`
5. ✅ Run simulation again and compare
6. ✅ Read README.md for technical details
7. ✅ Experiment with different scenarios

---

## Getting Help

1. **First check**: Run with `sim_cfg.verbose = True` to see details
2. **Check README.md**: "Troubleshooting" section
3. **Check examples.py**: Each example shows a use case
4. **Check docstrings**: Every function has documentation

```python
# View documentation for any function
from capacity_models import CapacityModels
help(CapacityModels.finite_blocklength_capacity)
```

---

## Quick Reference: Key Commands

```bash
# Run full simulation
python main.py

# Run examples
python examples.py

# Run custom Python script
python my_script.py

# Install packages
pip install -r requirements.txt

# Check Python version
python --version

# View file
type filename.py (Windows) or cat filename.py (Linux/Mac)
```

---

## Summary

✅ Framework installed and ready  
✅ Can run simulations  
✅ Can generate plots  
✅ Can modify parameters  
✅ Can use examples  
✅ Can extend code

**You're all set! Happy simulating! 🚀**

---

*For detailed technical documentation, see README.md*  
*For implementation overview, see IMPLEMENTATION_SUMMARY.md*
