"""
System Configuration and Parameters for Multi-UAV URLLC-eMBB Coexistence
Based on 3GPP Release 16/17 and fininte blocklength theory
"""

import numpy as np


class SystemConfig:
    """System-level parameters"""
    
    def __init__(self):
        # Network topology
        # 中文說明：下面是預設的系統參數（可直接在此檔案修改），
        #           主要用於快速啟動與測試。預設值代表"基準情境"（baseline）：
        #           - num_uavs = 3   : 3 個 UAV
        #           - num_embb_users = 6 : 每次模擬總共 6 個 eMBB 使用者（非每 UAV）
        #           - num_urllc_users = 3: 每次模擬總共 3 個 URLLC 使用者（非每 UAV）
        # 如果要執行 "heavy load" 或自訂情境，請參考 README 的 Custom Scenarios 範例
        # 或在外部程式中建立自訂的 SystemConfig 實例後再覆寫這些屬性。
        self.num_uavs = 3                          # Number of UAVs (N)
        self.num_embb_users = 6                    # eMBB users |E|
        self.num_urllc_users = 3                   # URLLC users |U|
        self.num_subcarriers = 12                  # Total RBs (K) per UAV (reduced)

        # Spatial topology for visualization / association
        self.area_width = 400.0                    # meters
        self.area_height = 400.0                   # meters
        self.uav_positions = np.array([
            [135.0, 75.0],
            [330.0, 235.0],
            [165.0, 320.0]
        ], dtype=float)
        self.uav_altitudes = np.array([120.0, 120.0, 120.0], dtype=float)
        self.user_cluster_spread = 35.0           # meters
        self.user_min_boundary_margin = 20.0      # meters
        self.random_seed = 42                     # Deterministic topology / shadowing seed
        
        # Time structure
        self.num_slots = 10                        # Simulation slots
        self.num_minislots = 8                     # Mini-slots per slot (S)
        self.slot_duration = 0.25                  # ms per slot
        self.minislot_duration = self.slot_duration / self.num_minislots  # ms
        
        # Channel parameters
        self.carrier_frequency = 2.0e9             # Hz (2.0 GHz band)
        self.bandwidth = 5.76e6                    # Hz (5.76 MHz)
        self.tx_antenna = 2                        # Tx antennas
        self.rx_antenna = 2                        # Rx antennas
        self.noise_figure = 5.0                    # dB
        self.subcarrier_spacing = 15e3            # Hz (assumed OFDM spacing)
        self.ofdm_symbols_per_slot = 14           # OFDM symbols in a 1 ms slot (15 kHz baseline)
        
        # Path loss model
        self.los_probability = 0.9                 # LoS probability in UAV scenario
        self.nlos_pathloss_exponent = 3.5
        self.los_pathloss_exponent = 2.0
        self.reference_distance = 1.0              # meters
        self.pathloss_at_1m = 20 * np.log10(4 * np.pi * self.carrier_frequency / 3e8)
        self.a2g_los_a = 9.61
        self.a2g_los_b = 0.16
        self.a2g_eta_los = 1.0                     # dB
        self.a2g_eta_nlos = 20.0                   # dB
        
        # Shadowing
        self.shadowing_std = 4.0                   # dB
        self.refresh_derived_params()

    def refresh_derived_params(self):
        """Recompute parameters derived from the current topology and timing."""
        self.minislot_duration = self.slot_duration / self.num_minislots  # ms
        self.subcarrier_bw = self.bandwidth / self.num_subcarriers        # Hz per RB
        self.noise_power = self._calculate_noise_power()
        minislot_duration_s = self.minislot_duration * 1e-3
        symbols_per_slot = max(
            1,
            int(round(self.ofdm_symbols_per_slot * (self.slot_duration / 1.0))),
        )
        symbols_per_minislot = max(
            1,
            int(round(symbols_per_slot / max(self.num_minislots, 1))),
        )
        subcarriers_per_rb = max(
            1,
            int(round(self.subcarrier_bw / max(self.subcarrier_spacing, 1.0))),
        )
        self.channel_uses_per_minislot = max(
            1,
            int(round(subcarriers_per_rb * symbols_per_minislot)),
        )
        
    def _calculate_noise_power(self):
        """Calculate noise power in watts"""
        noise_power_dbm = -174 + 10 * np.log10(self.subcarrier_bw) + self.noise_figure
        return 10 ** ((noise_power_dbm - 30) / 10)


class URLLCConfig:
    """URLLC-specific parameters"""
    
    def __init__(self):
        # Reliability requirements
        self.target_error_probability = 1e-4      # Error probability threshold (epsilon_z)
        self.packet_lengths = [120, 150, 180]     # bits (L_z) - 3 URLLC types
        
        # Latency constraints
        self.max_latency_slots = 1                 # Maximum allowed slots
        self.max_latency_minislots = 8             # Maximum allowed mini-slots
        
        # Finite blocklength parameters
        self.finite_blocklength_factor = 20        # Q-function approximation factor
        
        # Retransmission policy
        self.max_retransmissions = 2               # HARQ retransmissions
        
        # Power constraints
        self.power_limits = [30, 30, 30]           # dBm per URLLC user


class eMBBConfig:
    """eMBB-specific parameters"""
    
    def __init__(self):
        # Service requirements
        self.target_spectral_efficiency = 2.0     # bps/Hz
        self.min_rate_per_user_bps = 2e6          # Minimum per-user rate requirement (bps)
        # 每個 eMBB 使用者的資料量 (bits)
        self.buffer_sizes = [2000000]   # 2 Mbits
        
        # Power constraints
        self.power_limits = [23, 23, 23, 23, 23, 23]  # dBm per eMBB user (6 users)
        
        # Quality of Service
        self.outage_probability = 0.1              # 10% outage acceptable
        self.cqi_feedback_delay = 1                # slots


class NOMAConfig:
    """NOMA/Puncturing decision parameters"""
    
    def __init__(self):
        # SIC parameters
        self.sic_error_threshold = 0.05            # SIC failure probability threshold
        self.sic_efficiency_factor = 0.95          # SIC decoding efficiency
        
        # Mode selection
        self.prefer_noma = True                    # Default to NOMA if possible
        self.puncturing_backoff_ratio = 1.5        # Power increase factor for puncturing
        
        # Interference threshold
        self.max_tolerable_snir = 15.0             # dB


class AlgorithmConfig:
    """Algorithm parameters"""
    
    def __init__(self):
        # Bisection search
        self.bisection_max_iterations = 15         # Bisection search iterations
        self.bisection_tolerance = 1e-3            # Convergence tolerance (Watts)
        self.power_lower_bound = 1e-3              # Watts (minimum transmit power)
        self.power_upper_bound = 2.0               # Watts (maximum transmit power)
        
        # Greedy allocation
        self.csi_update_frequency = 1              # Update CSI every N slots
        self.allocation_granularity = 1            # RB allocation unit
        
        # Convergence
        self.max_iterations = 1000                 # Maximum algorithm iterations
        self.noma_retention_factor = 0.60          # eMBB rate fraction kept under NOMA overlay
        self.noma_snir_threshold = 0.1             # Very relaxed linear SNIR threshold for URLLC decode
        self.sic_power_gap_db = -5.0               # Allow weak power separation to still visualize SIC/NOMA
        self.sic_residual_factor = 0.001           # Residual interference after SIC (more optimistic SIC)
        self.urllc_utility_weight = 8.0            # Utility reward for a feasible URLLC packet
        self.power_penalty_weight = 0.05           # Utility penalty per watt
        self.overload_penalty_weight = 3.0         # Utility penalty under heavy URLLC load
        self.admission_load_limit = 0.35           # Max URLLC occupancy ratio before aggressive rejection
        self.force_urllc_immediate_service = True  # Schedule every arrived URLLC packet if reliability-feasible
        self.min_noma_gain_ratio = 1.05            # Require URLLC channel > eMBB channel (less strict)
        self.embb_min_sic_snir_db = 2.0            # Minimum post-SIC eMBB SNIR
        self.power_refine_step = 0.15              # Local eMBB power search step
        self.power_refine_iterations = 2           # Local refinement rounds


class SimulationConfig:
    """Simulation control parameters"""
    
    def __init__(self):
        # Operating points
        self.num_density_points = 5                # For user density analysis
        self.min_user_density = 1                  # Users per UAV
        self.max_user_density = 10                 # Users per UAV
        
        # Random seed
        self.random_seed = 42

        # URLLC arrival model
        # 中文說明：URLLC 為間歇性到達，這裡定義每個 URLLC 使用者在每個 time-slot 發生傳輸請求的機率。
        # 例如 0.1 表示每個 slot 有 10% 機率該 URLLC user 產生一個封包需要傳送。
        # 這會使得 allocation timeline 主要由 eMBB 填滿，而在少數 slot 出現 URLLC 的突發流量。
        self.urllc_arrival_prob = 0.1
        self.urllc_poisson_rate = 1.0             # Average URLLC arrivals per slot
        self.fixed_urllc_poisson_rate = False     # If True, do not scale lambda with load
        # URLLC user ratio (when scaling total users). 0.0 means follow original eMBB/URLLC counts.
        self.urllc_user_ratio = 0.0
        
        # Channel generation
        self.csi_generation_method = 'rayleigh'    # 'rayleigh' or 'rician'
        self.rician_k_factor = 5.0                 # K-factor for Rician
        
        # Output
        self.verbose = True
        self.plot_results = True
        self.save_figure_path = './results/'


# Default configuration bundle
DEFAULT_SYSTEM_CONFIG = SystemConfig()
DEFAULT_URLLC_CONFIG = URLLCConfig()
DEFAULT_EMBB_CONFIG = eMBBConfig()
DEFAULT_NOMA_CONFIG = NOMAConfig()
DEFAULT_ALGO_CONFIG = AlgorithmConfig()
DEFAULT_SIM_CONFIG = SimulationConfig()
