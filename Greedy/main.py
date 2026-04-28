"""
Main Entry Point - Run Multi-UAV URLLC-eMBB Simulation
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all components
from config import (SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig,
                    SimulationConfig)
from simulation import create_simulation
from visualization import create_plotter, ResultsAnalyzer


def main():
    """Main simulation runner"""
    
    print("\n" + "="*70)
    print("Multi-UAV URLLC-eMBB Coexistence Resource Allocation")
    print("Algorithm: Constrained Greedy with per-UAV RBs, NOMA/Puncturing, and Interference-Aware SINR")
    print("="*70)
    
    # ============= SETUP CONFIGURATIONS =============
    sys_cfg = SystemConfig()
    sys_cfg.num_subcarriers = 8
    sys_cfg.num_embb_users = 20
    sys_cfg.num_urllc_users = 8
    sys_cfg.refresh_derived_params()
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    sim_cfg = SimulationConfig()
    
    # Print system configuration
    print("\nSystem Configuration:")
    print(f"  UAVs: {sys_cfg.num_uavs}")
    print(f"  eMBB Users: {sys_cfg.num_embb_users}")
    print(f"  URLLC Users: {sys_cfg.num_urllc_users}")
    print(f"  Total RBs: {sys_cfg.num_subcarriers}")
    print(f"  Slots: {sys_cfg.num_slots}, Mini-slots: {sys_cfg.num_minislots}")
    print(f"  Carrier Freq: {sys_cfg.carrier_frequency/1e9:.1f} GHz")
    print(f"  Bandwidth: {sys_cfg.bandwidth/1e6:.0f} MHz")
    
    print("\nURLLC Configuration:")
    print(f"  Target Error Prob: {urllc_cfg.target_error_probability:.2e}")
    print(f"  Max Latency: {urllc_cfg.max_latency_minislots} mini-slots")
    print(f"  Packet Lengths: {urllc_cfg.packet_lengths} bits")
    
    print("\neMBB Configuration:")
    print(f"  Target Spectral Efficiency: {embb_cfg.target_spectral_efficiency:.2f} bps/Hz")
    
    print("\nAlgorithm Configuration:")
    print(f"  Bisection Max Iterations: {algo_cfg.bisection_max_iterations}")
    print(f"  Bisection Tolerance: {algo_cfg.bisection_tolerance}")
    
    # ============= CREATE SIMULATION =============
    sim_cfg.urllc_arrival_prob = 0.45
    sim_cfg.urllc_poisson_rate = 12
    sim_cfg.min_user_density = 1
    sim_cfg.max_user_density = 40
    sim_cfg.num_density_points = 6
    urllc_cfg.power_limits = [24] * sys_cfg.num_urllc_users
    embb_cfg.power_limits = [23] * sys_cfg.num_embb_users
    algo_cfg.power_upper_bound = 0.25
    sys_cfg.shadowing_std = 6.0
    sys_cfg.los_probability = 0.8
    urllc_cfg.packet_lengths = [160, 180, 200]
    urllc_cfg.target_error_probability = 1e-5

    print("\nRuntime Scenario:")
    print(f"  URLLC Arrival Model: Poisson(lambda={sim_cfg.urllc_poisson_rate:.2f} per slot)")
    print(f"  URLLC Packet Lengths: {urllc_cfg.packet_lengths} bits")
    print(f"  URLLC Target Error Prob: {urllc_cfg.target_error_probability:.2e}")
    print(f"  Power Upper Bound: {algo_cfg.power_upper_bound:.3f} W")

    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    
    # ============= RUN BASE SIMULATION =============
    print("\n" + "="*70)
    print("Running Base Case Simulation (10 Time Slots)")
    print("="*70)
    
    aggregated = simulation.run_full_simulation()
    base_allocation_summary = getattr(simulation, 'last_allocation', None)
    base_plot_slot_index = sys_cfg.num_slots - 1
    allocation_history = getattr(simulation, 'allocation_history', [])
    for idx, alloc in enumerate(allocation_history):
        urllc_grid = alloc.get('urllc_timefreq_grid')
        if urllc_grid is not None and np.any(urllc_grid >= 0):
            base_allocation_summary = alloc
            base_plot_slot_index = idx
    
    # ============= PRINT DETAILED REPORT =============
    ResultsAnalyzer.print_detailed_report(aggregated)
    
    # ============= RUN USER DENSITY ANALYSIS =============
    print("\n" + "="*70)
    print("Running User Density Analysis")
    print("="*70)
    
    density_result = simulation.run_user_density_analysis()
    
    # ============= VISUALIZATION =============
    print("\n" + "="*70)
    print("Generating Visualization")
    print("="*70)
    
    plotter = create_plotter()
    
    # 1. Power vs Density
    plotter.plot_power_vs_density(density_result)
    plotter.plot_mode_tradeoff_analysis(density_result)
    plotter.plot_fairness_load_analysis(density_result)
    plotter.plot_slot_mode_action_summary(aggregated)
    plotter.plot_per_uav_performance_decomposition(density_result)
    plotter.plot_overlay_feasibility_diagnostic(density_result)
    plotter.plot_retention_loss_distribution(density_result)
    plotter.plot_offered_load_curves(density_result)
    plotter.plot_resource_utilization_summary(density_result)
    plotter.plot_urllc_arrival_timeline(
        aggregated.get('all_active_urllc_users', []),
        scheduled_urllc_users=aggregated.get('all_scheduled_urllc_users', [])
    )
    plotter.plot_urllc_minislot_arrival_map(
        allocation_history,
        sys_cfg.num_minislots
    )
    
    # 2. Time-frequency heatmap for a single slot (use new slot-based visualization)
    # Use the simulation's recorded last-slot allocation (robust to other analyses)
    allocation_summary = base_allocation_summary
    if allocation_summary is None:
        allocation_summary = simulation.allocator.get_allocation_summary()

    if allocation_summary and allocation_summary.get('embb_rbs') is not None:
        topology = allocation_summary.get('topology')
        serving_uavs = allocation_summary.get('best_uav_per_user')
        coexistence_mode_per_uav = allocation_summary.get('coexistence_mode_per_uav')

        if topology is not None and serving_uavs is not None:
            plotter.plot_spatial_grouping(
                topology['user_positions'],
                topology['uav_positions'],
                serving_uavs,
                num_urllc=sys_cfg.num_urllc_users,
                slot_index=base_plot_slot_index
            )
        plotter.plot_per_uav_load_distribution(
            allocation_summary,
            num_uavs=sys_cfg.num_uavs,
            num_urllc=sys_cfg.num_urllc_users,
            num_embb=sys_cfg.num_embb_users,
            slot_index=base_plot_slot_index
        )

        # Produce time-frequency heatmap for the last slot (fills all minislots)
        plotter.plot_slot_timefreq_heatmap(
            allocation_summary['embb_rbs'],
            allocation_summary.get('urllc_timefreq_grid', None),
            num_urllc=sys_cfg.num_urllc_users,
            num_embb=sys_cfg.num_embb_users,
            num_minislots=sys_cfg.num_minislots,
            slot_index=base_plot_slot_index,
            embb_owner_per_rb=allocation_summary.get('embb_owner_per_rb'),
            noma_decisions=allocation_summary.get('noma_decisions'),
            embb_owner_per_uav_rb=allocation_summary.get('embb_owner_per_uav_rb'),
            coexistence_mode_per_uav=coexistence_mode_per_uav,
            coexistence_urllc_user_per_uav=allocation_summary.get('coexistence_urllc_user_per_uav'),
            plot_uav_index=0,
            embb_selected_uavs=allocation_summary.get('embb_selected_uavs'),
            urllc_selected_uavs=allocation_summary.get('urllc_selected_uavs')
        )

        # Additionally produce a compact users x RB heatmap for the same slot
        plotter.plot_single_slot_heatmap(
            allocation_summary['embb_rbs'],
            allocation_summary.get('urllc_rbs', None),
            num_urllc=sys_cfg.num_urllc_users,
            num_embb=sys_cfg.num_embb_users,
            slot_index=base_plot_slot_index
        )
    
    # 3. Performance timeline
    plotter.plot_performance_timeline(
        aggregated['all_embb_rates'],
        aggregated['all_urllc_success'],
        aggregated['all_power'],
        user_density=(sys_cfg.num_embb_users, sys_cfg.num_urllc_users),
        all_embb_power=aggregated.get('all_embb_power'),
        all_urllc_power=aggregated.get('all_urllc_power')
    )
    
    # ============= FINAL SUMMARY =============
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    print("\nKey Results:")
    print(f"  eMBB Total Rate: {aggregated['avg_embb_rate']/1e6:.4f} Mbps")
    print(f"  URLLC Admission Ratio: {aggregated['avg_urllc_admission']:.4f}")
    print(f"  Admitted URLLC Reliability: {aggregated['avg_urllc_success']:.4f} (Target: 0.99)")
    print(f"  Power Efficiency: {aggregated['avg_embb_rate']/(aggregated['avg_total_power']*1e6):.4f} Mbps/W")
    print(f"  RB Utilization: {aggregated['avg_rb_utilization']:.2%}")
    
    print("\nAll results and plots saved to: ./results/")
    
    return aggregated, density_result


def run_custom_scenario(num_embb=8, num_urllc=4, num_rbs=64):
    """Run simulation with custom parameters"""
    
    print(f"\nRunning Custom Scenario:")
    print(f"  eMBB: {num_embb}, URLLC: {num_urllc}, RBs: {num_rbs}")
    
    sys_cfg = SystemConfig()
    sys_cfg.num_embb_users = num_embb
    sys_cfg.num_urllc_users = num_urllc
    sys_cfg.num_subcarriers = num_rbs
    sys_cfg.refresh_derived_params()
    
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    sim_cfg = SimulationConfig()
    sim_cfg.verbose = False
    
    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    result = simulation.run_full_simulation()
    
    return result


if __name__ == '__main__':
    
    # Run main simulation
    aggregated, density_result = main()
    
    # Optional: Run additional custom scenarios
    print("\n" + "="*70)
    print("Custom Scenario Testing")
    print("="*70)
    
    scenarios = [
        (4, 2, 50),   # Conservative
        (6, 3, 50),   # Baseline
        (8, 4, 64),   # Heavy load
    ]
    
    scenario_results = []
    for num_embb, num_urllc, num_rbs in scenarios:
        result = run_custom_scenario(num_embb, num_urllc, num_rbs)
        scenario_results.append(result)
        print(f"  eMBB Rate: {result['avg_embb_rate']/1e6:.4f} Mbps, "
              f"Admission: {result['avg_urllc_admission']:.4f}, "
              f"Admitted Reliability: {result['avg_urllc_success']:.4f}")
    
    print("\n[OK] Results available in ./results/ directory")
