"""
Quick Start Example - How to Use the Multi-UAV Simulation Framework
"""

import numpy as np
from config import SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig, SimulationConfig
from simulation import create_simulation
from visualization import create_plotter, ResultsAnalyzer


def example_1_basic_simulation():
    """Example 1: Run basic simulation with default parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Simulation with Default Parameters")
    print("="*70)
    
    # Create simulation with default configs
    simulation = create_simulation()
    
    # Run simulation for 5 time slots
    simulation.sys_cfg.num_slots = 5
    
    results = simulation.run_full_simulation()
    
    print(f"\nResults:")
    print(f"  eMBB Rate: {results['avg_embb_rate']/1e6:.4f} Mbps")
    print(f"  URLLC Success: {results['avg_urllc_success']:.4f}")
    print(f"  Total Power: {results['avg_total_power']*1e3:.2f} mW")


def example_2_custom_scenario():
    """Example 2: Custom scenario with modified parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Scenario - Heavy Load Network")
    print("="*70)
    
    # Create custom configurations
    sys_cfg = SystemConfig()
    sys_cfg.num_embb_users = 10      # More eMBB users
    sys_cfg.num_urllc_users = 5      # More URLLC users
    sys_cfg.num_subcarriers = 100    # More RBs
    
    urllc_cfg = URLLCConfig()
    urllc_cfg.target_error_probability = 1e-4  # Less strict
    
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    
    sim_cfg = SimulationConfig()
    sim_cfg.verbose = False
    sim_cfg.num_slots = 3
    
    # Create and run simulation
    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    results = simulation.run_full_simulation()
    
    print(f"\nHeavy Load Results:")
    print(f"  eMBB Users: {sys_cfg.num_embb_users}, URLLC Users: {sys_cfg.num_urllc_users}")
    print(f"  eMBB Rate: {results['avg_embb_rate']/1e6:.4f} Mbps")
    print(f"  URLLC Success: {results['avg_urllc_success']:.4f}")


def example_3_single_slot():
    """Example 3: Run single time slot allocation and inspect results"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Slot Allocation with Detailed Inspection")
    print("="*70)
    
    sys_cfg = SystemConfig()
    sys_cfg.num_slots = 1
    
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    
    sim_cfg = SimulationConfig()
    sim_cfg.verbose = False
    
    simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
    
    # Run single allocation
    result = simulation.run_single_allocation(slot_index=0)
    
    metrics = result['metrics']
    
    print(f"\nSlot 0 Detailed Metrics:")
    print(f"  eMBB Total Rate: {metrics['embb_total_rate']/1e6:.4f} Mbps")
    print(f"  eMBB User Rates:")
    for i, rate in enumerate(metrics['embb_per_user_rate']):
        print(f"    User {i}: {rate/1e6:.4f} Mbps")
    
    print(f"\n  URLLC Success Rate: {metrics['urllc_success_rate']:.4f}")
    print(f"  URLLC Individual Success:")
    for i, success in enumerate(metrics['urllc_individual_success']):
        print(f"    User {i}: {success:.4f}")
    
    print(f"\n  Power Allocation:")
    print(f"    URLLC Total: {metrics['urllc_power']*1e3:.2f} mW")
    print(f"    eMBB Total: {metrics['embb_power']*1e3:.2f} mW")
    print(f"    Total System: {metrics['total_power']*1e3:.2f} mW")
    
    print(f"\n  RB Utilization: {metrics['rb_utilization']:.2%}")
    print(f"  NOMA Usage: {metrics['noma_ratio']:.1%}")


def example_4_capacity_calculations():
    """Example 4: Direct capacity model usage"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Capacity Model Calculations")
    print("="*70)
    
    from capacity_models import CapacityModels
    
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    
    cap_model = CapacityModels(urllc_cfg, embb_cfg)
    
    # Test Shannon capacity
    snir_db = 10  # dB
    bandwidth = 1e6  # 1 MHz
    
    shannon_rate = cap_model.shannon_capacity_bits(snir_db, bandwidth)
    
    print(f"\nShannon Capacity Calculation:")
    print(f"  SNIR: {snir_db} dB")
    print(f"  Bandwidth: {bandwidth/1e6:.0f} MHz")
    print(f"  Shannon Capacity: {shannon_rate/1e6:.4f} Mbps")
    
    # Test finite blocklength
    packet_bits = 150
    channel_uses = 14  # 1 OFDM symbol
    
    fb_rate = cap_model.finite_blocklength_capacity(
        10**(snir_db/10),  # Convert to linear
        packet_bits,
        channel_uses
    )
    
    error_prob = cap_model.decoding_error_probability(
        10**(snir_db/10),
        packet_bits,
        channel_uses
    )
    
    print(f"\nFinite Blocklength Calculation:")
    print(f"  Packet Length: {packet_bits} bits")
    print(f"  Channel Uses: {channel_uses}")
    print(f"  Achievable Rate: {fb_rate:.4f} bits/channel_use")
    print(f"  Error Probability: {error_prob:.4e}")


def example_5_channel_model():
    """Example 5: Channel model generation"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Channel Model Generation")
    print("="*70)
    
    from channel_model import ChannelModel
    import numpy as np
    
    sys_cfg = SystemConfig()
    channel_model = ChannelModel(sys_cfg)
    
    # Generate channels
    gains = channel_model.generate_channel_gains(
        num_users=5,
        num_uavs=3,
        num_subcarriers=50
    )
    
    mag_sq = channel_model.get_channel_magnitude_squared(gains)
    
    print(f"\nChannel Generation:")
    print(f"  Shape: {mag_sq.shape} (users, UAVs, subcarriers)")
    print(f"  Mean Power Gain: {np.mean(mag_sq):.6f}")
    print(f"  Min Power Gain: {np.min(mag_sq):.6f}")
    print(f"  Max Power Gain: {np.max(mag_sq):.6f}")
    
    gains_db = 10 * np.log10(mag_sq)
    print(f"\n  Mean Gain (dB): {np.mean(gains_db):.2f}")
    print(f"  Std Dev (dB): {np.std(gains_db):.2f}")


def example_6_resource_allocator():
    """Example 6: Direct resource allocator usage"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Resource Allocator Usage")
    print("="*70)
    
    from resource_allocator import create_allocator
    from channel_model import ChannelModel
    
    sys_cfg = SystemConfig()
    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    
    allocator = create_allocator(sys_cfg, urllc_cfg, embb_cfg, algo_cfg)
    channel_model = ChannelModel(sys_cfg)
    
    # Generate channels
    gains = channel_model.generate_channel_gains(
        sys_cfg.num_urllc_users + sys_cfg.num_embb_users,
        sys_cfg.num_uavs,
        sys_cfg.num_subcarriers
    )
    
    mag_sq = channel_model.get_channel_magnitude_squared(gains)
    
    # Phase 1: URLLC allocation
    print("\nPhase 1: URLLC Power Allocation")
    urllc_power, reliability = allocator.allocate_urllc_power(mag_sq)
    print(f"  URLLC Power Shape: {urllc_power.shape}")
    print(f"  Average Reliability: {np.mean(reliability):.4f}")
    
    # Phase 2: eMBB allocation
    print("\nPhase 2: eMBB Greedy Allocation")
    embb_result = allocator.allocate_embb_greedy(mag_sq)
    print(f"  Total eMBB Rate: {embb_result['total_rate']/1e6:.4f} Mbps")
    print(f"  RB Allocation Shape: {embb_result['rb_allocation'].shape}")
    
    # Phase 3: NOMA decision
    print("\nPhase 3: NOMA vs Puncturing")
    noma_decisions = allocator.decide_noma_puncturing(
        mag_sq,
        np.arange(sys_cfg.num_urllc_users),
        np.arange(sys_cfg.num_embb_users),
        urllc_power
    )
    noma_count = np.sum(noma_decisions == 'NOMA')
    print(f"  NOMA Usage: {noma_count}/{noma_decisions.size} ({noma_count/noma_decisions.size:.1%})")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Multi-UAV URLLC-eMBB Simulation Framework - Examples")
    print("="*70)
    
    # Run all examples
    example_1_basic_simulation()
    example_2_custom_scenario()
    example_3_single_slot()
    example_4_capacity_calculations()
    example_5_channel_model()
    example_6_resource_allocator()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Modify parameters in config.py for custom scenarios")
    print("  2. Run main.py for full simulation with visualization")
    print("  3. Check results/ directory for generated plots")
    print("  4. Refer to README.md for detailed documentation")
