# PROJECT INDEX & FILE GUIDE

**Multi-UAV URLLC-eMBB Coexistence Framework**  
Location: `d:\URLLC_eMBB_Coexisting\Greedy\`  
Status: ✅ COMPLETE & TESTED

---

## 📚 Documentation Files (Read These First!)

### 1. **QUICKSTART.md** - START HERE! ⭐
- **Size**: 8.55 KB
- **Purpose**: 5-minute quick start guide
- **Content**: 
  - Installation steps
  - First run instructions
  - Common scenarios
  - Troubleshooting
- **For**: Everyone, especially beginners
- **Read Time**: 5 minutes

### 2. **README.md** - Complete Reference
- **Size**: 12.14 KB
- **Purpose**: Full technical documentation
- **Content**:
  - System model overview
  - Algorithm description
  - Mathematical foundations
  - Configuration reference
  - References & citations
- **For**: Users wanting technical details
- **Read Time**: 15-20 minutes

### 3. **IMPLEMENTATION_SUMMARY.md** - This Project
- **Size**: 16.75 KB
- **Purpose**: What was implemented, how it works
- **Content**:
  - Module descriptions
  - Feature checklist
  - Test results
  - Performance characteristics
  - Next steps
- **For**: Understanding the codebase
- **Read Time**: 10-15 minutes

### 4. **requirements.txt** - Dependencies
- **Size**: 0.05 KB
- **Purpose**: Python package requirements
- **Content**: numpy, scipy, matplotlib
- **Use**: `pip install -r requirements.txt`

---

## 🐍 Python Modules (Main Code)

### 5. **config.py** - System Configuration ⚙️
- **Size**: 5.79 KB
- **Lines**: ~115
- **Purpose**: Centralized parameter management
- **Key Classes**:
  - `SystemConfig` - Network topology
  - `URLLCConfig` - URLLC requirements
  - `eMBBConfig` - eMBB targets
  - `NOMAConfig` - NOMA parameters
  - `AlgorithmConfig` - Algorithm tuning
  - `SimulationConfig` - Simulation control
- **When to Use**: Modify all system parameters here
- **Example**:
  ```python
  from config import SystemConfig
  sys_cfg = SystemConfig()
  sys_cfg.num_embb_users = 10
  ```

### 6. **channel_model.py** - Wireless Channel 📡
- **Size**: 5.45 KB
- **Lines**: ~135
- **Purpose**: Realistic air-to-ground channel modeling
- **Key Class**: `ChannelModel`
- **Key Methods**:
  - `generate_channel_gains()` - Create CSI
  - `get_large_scale_fading()` - Path loss + shadowing
  - `get_channel_magnitude_squared()` - Power gains
  - `update_channels()` - Temporal evolution
- **When to Use**: Channel generation, propagation modeling
- **Example**:
  ```python
  from channel_model import ChannelModel
  cm = ChannelModel(sys_cfg)
  gains = cm.generate_channel_gains(5, 3, 50)
  ```

### 7. **capacity_models.py** - Capacity Formulas 📊
- **Size**: 8.65 KB
- **Lines**: ~210
- **Purpose**: Information-theoretic capacity calculations
- **Key Class**: `CapacityModels`
- **Key Methods**:
  - `shannon_capacity()` - eMBB rate
  - `finite_blocklength_capacity()` - URLLC rate
  - `decoding_error_probability()` - Reliability
  - `min_power_for_reliability()` - Min power search
- **When to Use**: Rate and reliability calculations
- **Example**:
  ```python
  from capacity_models import CapacityModels
  cap = CapacityModels()
  rate = cap.shannon_capacity(10, 1e6)  # 10 linear SNIR, 1MHz BW
  ```

### 8. **resource_allocator.py** - Optimization Algorithm 🎯
- **Size**: 11.85 KB
- **Lines**: ~290
- **Purpose**: Core Bisection + Greedy algorithm
- **Key Class**: `ResourceAllocator`
- **Key Methods**:
  - `allocate_urllc_power()` - Phase 1: Bisection search
  - `allocate_embb_greedy()` - Phase 2: Greedy RB allocation
  - `decide_noma_puncturing()` - Phase 3: Mode selection
  - `_bisection_search_urllc_power()` - Helper: Bisection
- **When to Use**: Resource allocation
- **Example**:
  ```python
  from resource_allocator import ResourceAllocator
  alloc = ResourceAllocator(sys_cfg, urllc_cfg, embb_cfg, algo_cfg)
  power, reliability = alloc.allocate_urllc_power(channel_gains)
  ```

### 9. **simulation.py** - Simulation Engine 🔄
- **Size**: 10.11 KB
- **Lines**: ~220
- **Purpose**: Orchestrates complete simulation
- **Key Class**: `MultiUAVSimulation`
- **Key Methods**:
  - `run_single_allocation()` - Single time slot
  - `run_full_simulation()` - Multiple slots, aggregation
  - `run_user_density_analysis()` - Density sweep
  - `_aggregate_metrics()` - Statistics computation
- **When to Use**: Running simulations
- **Example**:
  ```python
  from simulation import create_simulation
  sim = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
  results = sim.run_full_simulation()
  ```

### 10. **visualization.py** - Plotting & Analysis 📈
- **Size**: 11.51 KB
- **Lines**: ~250
- **Purpose**: Results visualization and analysis
- **Key Classes**:
  - `SimulationPlotter` - Generate plots
  - `ResultsAnalyzer` - Statistical analysis
- **Key Methods**:
  - `plot_power_vs_density()` - 4-panel analysis
  - `plot_allocation_heatmap()` - RB allocation
  - `plot_performance_timeline()` - Time series
  - `plot_channel_cdf()` - Channel distribution
- **When to Use**: Visualizing results
- **Example**:
  ```python
  from visualization import create_plotter
  plotter = create_plotter('./results/')
  plotter.plot_power_vs_density(density_result)
  ```

### 11. **main.py** - Primary Entry Point 🚀
- **Size**: 5.60 KB
- **Lines**: ~120
- **Purpose**: Complete end-to-end simulation with plots
- **What it Does**:
  1. Setup configurations
  2. Print system parameters
  3. Run base case (10 slots)
  4. Run user density analysis
  5. Generate plots
  6. Print final summary
- **How to Run**: `python main.py`
- **Use Case**: Default, comprehensive simulation

### 12. **examples.py** - Usage Examples 📖
- **Size**: 8.48 KB
- **Lines**: ~280
- **Purpose**: 6 runnable examples demonstrating framework
- **Included Examples**:
  1. Basic simulation with defaults
  2. Custom heavy-load scenario
  3. Single slot detailed inspection
  4. Capacity calculation examples
  5. Channel model generation
  6. Direct allocator usage
- **How to Run**: `python examples.py` or run individual examples
- **Use Case**: Learning, code examples, starting template

---

## 📊 File Organization by Purpose

### Configuration & Setup
- `config.py` ← Modify parameters here
- `requirements.txt` ← Install dependencies

### Simulation Components
- `channel_model.py` ← Wireless channels
- `capacity_models.py` ← Rate calculations
- `resource_allocator.py` ← Core algorithm
- `simulation.py` ← Orchestration

### Running & Visualizing
- `main.py` ← Full simulation
- `examples.py` ← Code examples
- `visualization.py` ← Plots & analysis

### Documentation
- `QUICKSTART.md` ← Read first!
- `README.md` ← Technical reference
- `IMPLEMENTATION_SUMMARY.md` ← Overview
- This file (`PROJECT_INDEX.md`)

---

## 🔄 Quick Workflow

### For New Users
1. Read `QUICKSTART.md` (5 min) ⭐
2. Run `python main.py` (30 sec)
3. Check `./results/` for plots (2 min)
4. Run `python examples.py` (5 min)
5. Modify `config.py` and re-run (10 min)

### For Researchers
1. Read `README.md` mathematical section (15 min)
2. Study `resource_allocator.py` Phase 1-3 (20 min)
3. Run custom scenarios with examples.py (10 min)
4. Implement own algorithms (varies)

### For Developers
1. Review `IMPLEMENTATION_SUMMARY.md` (15 min)
2. Study module structure and dependencies
3. Modify/extend individual modules
4. Follow existing code patterns

---

## 🔗 Module Dependencies

```
main.py
  ├── config.py (all configs)
  ├── simulation.py
  │   ├── channel_model.py
  │   ├── capacity_models.py
  │   └── resource_allocator.py
  └── visualization.py

examples.py
  ├── config.py
  ├── simulation.py
  ├── channel_model.py
  ├── capacity_models.py
  ├── resource_allocator.py
  └── visualization.py
```

---

## 💾 File Statistics

| Category | Count | Total Size |
|----------|-------|-----------|
| Documentation | 3 | 37.4 KB |
| Python Modules | 8 | 77.4 KB |
| Metadata | 1 | 0.05 KB |
| **Total** | **12** | **~115 KB** |

---

## 🎯 Navigation by Task

### "How do I...?"

**...run a simulation?**
→ See `QUICKSTART.md` Step 2 or run `python main.py`

**...change the number of users?**
→ Edit `config.py` line 15-16

**...understand the algorithm?**
→ Read `README.md` "Algorithm Description" or study `resource_allocator.py`

**...visualize results?**
→ Check `./results/` directory after running simulation

**...use the framework in my own code?**
→ Look at `examples.py` for code templates

**...calculate channel capacity?**
→ Use `capacity_models.py` methods or see `examples.py` example 4

**...run a single time slot?**
→ Use `simulation.py` `run_single_allocation()` or see `examples.py` example 3

**...modify the greedy algorithm?**
→ Edit `resource_allocator.py` Phase 2 section

**...analyze power consumption?**
→ Generate plots with `visualization.py` or check console output

**...extend the framework?**
→ Review `IMPLEMENTATION_SUMMARY.md` "Next Steps"

---

## ⚙️ Common Edit Locations

### Increase eMBB capacity
1. `config.py` line 17: `num_subcarriers = 100` (more RBs)
2. `config.py` line 8: `num_embb_users = 8` (more users)

### Increase URLLC reliability
1. `config.py` line 47: `target_error_probability = 1e-6` (stricter)
2. `config.py` line 34: `max_latency_minislots = 4` (faster deadline)

### Change channel model
1. `simulation.py` line 58: `fading_type='rician'` (Rician instead)
2. `config.py` line 28: Modify pathloss parameters

### Modify bisection algorithm
1. `config.py` lines 97-100: Algorithm parameters
2. `resource_allocator.py` lines 85-95: Bisection implementation

### Add new plot
1. `visualization.py`: Add method to `SimulationPlotter` class
2. `main.py`: Call plotter method

---

## 📞 Reference Quick Links

| Topic | File | Location |
|-------|------|----------|
| System parameters | config.py | Lines 10-50 |
| URLLC settings | config.py | Lines 37-51 |
| eMBB settings | config.py | Lines 54-60 |
| Bisection search | resource_allocator.py | Lines 70-100 |
| Greedy algorithm | resource_allocator.py | Lines 105-160 |
| Shannon capacity | capacity_models.py | Lines 20-35 |
| Finite blocklength | capacity_models.py | Lines 48-85 |
| Main loop | simulation.py | Lines 50-105 |
| Full simulation | simulation.py | Lines 107-135 |
| Plotting functions | visualization.py | Lines 30+ |

---

## ✅ Verification Checklist

- [x] Documentation files complete
- [x] Python modules implemented
- [x] Examples provided
- [x] Simulation runs successfully
- [x] Plots generate correctly
- [x] All dependencies available
- [x] Code is well-documented
- [x] Performance is reasonable
- [x] Module organization is clear
- [x] File sizes are reasonable

---

## 🚀 You're Ready!

All files are in place and fully functional.

**Start with**: `QUICKSTART.md` → Run `python main.py` → Check `./results/`

**Questions?** → Check `README.md` troubleshooting section

---

*Last Updated: March 23, 2026*  
*Framework Version: 1.0 (Complete)*  
*Status: ✅ FULLY OPERATIONAL*
