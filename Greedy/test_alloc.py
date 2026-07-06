from config import SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig, SimulationConfig
from simulation import create_simulation
import numpy as np

scenarios = [
    {'num_embb': 2, 'num_urllc': 1, 'num_rbs': 4},
    {'num_embb': 4, 'num_urllc': 1, 'num_rbs': 4},
    {'num_embb': 8, 'num_urllc': 1, 'num_rbs': 4},
    {'num_embb': 16, 'num_urllc': 1, 'num_rbs': 4},
]

for s in scenarios:
    sys_cfg = SystemConfig()
    sys_cfg.num_embb_users = s['num_embb']
    sys_cfg.num_urllc_users = s['num_urllc']
    sys_cfg.num_subcarriers = s['num_rbs']
    # keep small slot/minislot
    sys_cfg.num_slots = 1
    sys_cfg.num_minislots = 7
    sys_cfg.refresh_derived_params()

    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    sim_cfg = SimulationConfig()
    sim_cfg.random_seed = 0
    sim_cfg.urllc_arrival_prob = 0.0  # disable URLLC for clear view

    sim = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    res = sim.run_single_allocation(slot_index=0)

    embb_rbs = res['allocation']['embb_rbs']
    print(f"\nScenario: eMBB={s['num_embb']}, RBs={s['num_rbs']}")
    counts = embb_rbs.sum(axis=1)
    for i, c in enumerate(counts):
        if c>0:
            print(f"  eMBB {i}: {int(c)} RB(s)")
    total_assigned = counts.sum()
    print(f"  Total assigned RBs: {int(total_assigned)} / {s['num_rbs']}")
