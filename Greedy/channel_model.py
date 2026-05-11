"""
Channel Model - Path Loss, Shadowing, and Fading
Implements realistic air-to-ground channel models for UAV communications
"""

import numpy as np
from config import SystemConfig


class ChannelModel:
    """Channel model for UAV communications"""
    
    def __init__(self, sys_config: SystemConfig):
        self.config = sys_config
        self.base_seed = int(getattr(self.config, 'random_seed', 42))
        self.rng = np.random.RandomState(self.base_seed)
        self.last_topology = None
        self._cached_topology = None
        self._cached_topology_shape = None
        self._nested_anchor_topology = None
        self._nested_anchor_shape = None

    def set_seed(self, seed: int):
        """Reset the internal RNG so channel/topology generation is repeatable."""
        self.base_seed = int(seed)
        self.rng = np.random.RandomState(self.base_seed)

    def reset_topology(self):
        """Clear cached user/UAV topology so a new one is generated on demand."""
        self._cached_topology = None
        self._cached_topology_shape = None
        self.last_topology = None
        self._nested_anchor_topology = None
        self._nested_anchor_shape = None

    def _build_topology(self, num_users: int, num_uavs: int):
        """Build one full topology sample for a specific user/UAV shape."""
        if (
            hasattr(self.config, 'uav_positions') and
            self.config.uav_positions is not None and
            len(self.config.uav_positions) >= num_uavs
        ):
            uav_positions = np.asarray(self.config.uav_positions[:num_uavs], dtype=float)
        else:
            x_coords = np.linspace(80.0, self.config.area_width - 80.0, num_uavs)
            y_coords = np.linspace(80.0, self.config.area_height - 80.0, num_uavs)
            uav_positions = np.column_stack((x_coords, y_coords[::-1]))

        user_positions = np.zeros((num_users, 2), dtype=float)
        serving_hints = np.zeros(num_users, dtype=int)

        def assign_group_positions(user_indices):
            if len(user_indices) == 0:
                return
            shuffled = np.asarray(user_indices, dtype=int).copy()
            self.rng.shuffle(shuffled)
            base_count = len(shuffled) // max(num_uavs, 1)
            remainder = len(shuffled) % max(num_uavs, 1)
            start = 0
            for uav_idx in range(num_uavs):
                quota = base_count + (1 if uav_idx < remainder else 0)
                if quota <= 0:
                    continue
                assigned_indices = shuffled[start:start + quota]
                start += quota
                center = uav_positions[uav_idx]
                for user_idx in assigned_indices:
                    offset = self.rng.normal(0.0, self.config.user_cluster_spread, size=2)
                    pos = center + offset
                    pos[0] = np.clip(
                        pos[0],
                        self.config.user_min_boundary_margin,
                        self.config.area_width - self.config.user_min_boundary_margin
                    )
                    pos[1] = np.clip(
                        pos[1],
                        self.config.user_min_boundary_margin,
                        self.config.area_height - self.config.user_min_boundary_margin
                    )
                    user_positions[user_idx] = pos
                    serving_hints[user_idx] = uav_idx

        configured_embb = int(getattr(self.config, 'num_embb_users', num_users))
        configured_urllc = int(getattr(self.config, 'num_urllc_users', 0))
        nested_max_users = int(getattr(self.config, 'nested_load_max_total_users', 0) or 0)
        if nested_max_users > 0 and int(num_users) == nested_max_users:
            configured_embb = int(getattr(self.config, 'nested_load_max_embb_users', configured_embb) or configured_embb)
            configured_urllc = int(getattr(self.config, 'nested_load_max_urllc_users', configured_urllc) or configured_urllc)
        if configured_embb + configured_urllc == num_users and configured_embb >= 0 and configured_urllc >= 0:
            embb_indices = np.arange(configured_embb, dtype=int)
            urllc_indices = np.arange(configured_embb, num_users, dtype=int)
            assign_group_positions(embb_indices)
            assign_group_positions(urllc_indices)
        else:
            assign_group_positions(np.arange(num_users, dtype=int))

        horizontal_distances = np.linalg.norm(
            user_positions[:, None, :] - uav_positions[None, :, :],
            axis=2
        )
        altitudes = np.asarray(
            getattr(self.config, 'uav_altitudes', np.full(num_uavs, 120.0)),
            dtype=float
        )[:num_uavs]
        distances = np.sqrt(horizontal_distances ** 2 + altitudes[None, :] ** 2)
        distances = np.clip(distances, self.config.reference_distance, None)

        return {
            'user_positions': user_positions,
            'uav_positions': uav_positions,
            'horizontal_distances': horizontal_distances,
            'uav_altitudes': altitudes,
            'distances': distances,
            'serving_hints': serving_hints
        }

    def generate_topology(self, num_users, num_uavs):
        """
        Generate a 2D topology that can be visualized directly.

        Users are placed around UAV-centered clusters so that the strongest-UAV
        association corresponds to a meaningful spatial grouping.
        """
        requested_shape = (num_users, num_uavs)
        if (
            self._cached_topology is not None and
            self._cached_topology_shape == requested_shape
        ):
            self.last_topology = self._cached_topology
            return self._cached_topology
        nested_enabled = bool(getattr(self.config, 'nested_load_from_max_users_enabled', False))
        nested_max_users = int(getattr(self.config, 'nested_load_max_total_users', 0) or 0)
        if nested_enabled and nested_max_users >= num_users:
            anchor_shape = (nested_max_users, num_uavs)
            if self._nested_anchor_topology is None or self._nested_anchor_shape != anchor_shape:
                self._nested_anchor_topology = self._build_topology(nested_max_users, num_uavs)
                self._nested_anchor_shape = anchor_shape
            anchor = self._nested_anchor_topology
            self._cached_topology = {
                'user_positions': np.asarray(anchor['user_positions'][:num_users], dtype=float).copy(),
                'uav_positions': np.asarray(anchor['uav_positions'], dtype=float).copy(),
                'horizontal_distances': np.asarray(anchor['horizontal_distances'][:num_users], dtype=float).copy(),
                'uav_altitudes': np.asarray(anchor['uav_altitudes'], dtype=float).copy(),
                'distances': np.asarray(anchor['distances'][:num_users], dtype=float).copy(),
                'serving_hints': np.asarray(anchor['serving_hints'][:num_users], dtype=int).copy(),
            }
        else:
            self._cached_topology = self._build_topology(num_users, num_uavs)
        self._cached_topology_shape = requested_shape
        self.last_topology = self._cached_topology
        return self._cached_topology

    def get_los_probability(self, horizontal_distance_m, altitude_m):
        """3GPP-style probabilistic LoS based on elevation angle."""
        horizontal_distance_m = max(horizontal_distance_m, self.config.reference_distance)
        elevation_angle_deg = np.degrees(np.arctan2(altitude_m, horizontal_distance_m))
        a_param = getattr(self.config, 'a2g_los_a', 9.61)
        b_param = getattr(self.config, 'a2g_los_b', 0.16)
        return 1.0 / (1.0 + a_param * np.exp(-b_param * (elevation_angle_deg - a_param)))

    def get_fspl_db(self, distance_m):
        """Free-space path loss in dB."""
        return (
            20 * np.log10(distance_m) +
            20 * np.log10(self.config.carrier_frequency) +
            20 * np.log10(4 * np.pi / 3e8)
        )

    def get_pathloss(self, distance_m, los_condition=True):
        """
        Calculate path loss using 3GPP air-to-ground model
        
        Args:
            distance_m: distance in meters
            los_condition: True for LoS, False for NLoS
            
        Returns:
            Path loss in dB
        """
        if distance_m < self.config.reference_distance:
            distance_m = self.config.reference_distance
            
        fspl_db = self.get_fspl_db(distance_m)
        if los_condition:
            return fspl_db + getattr(self.config, 'a2g_eta_los', 1.0)
        return fspl_db + getattr(self.config, 'a2g_eta_nlos', 20.0)
    
    def get_shadowing(self, num_samples, std_db=None):
        """
        Generate log-normal shadowing
        
        Args:
            num_samples: number of shadowing samples
            std_db: standard deviation in dB
            
        Returns:
            Shadowing in dB
        """
        if std_db is None:
            std_db = self.config.shadowing_std
            
        sigma_linear = std_db / np.sqrt(2)
        shadowing_db = self.rng.normal(0, std_db, num_samples)
        
        return shadowing_db
    
    def get_large_scale_fading(self, distance_m, los_condition=True):
        """
        Get large-scale fading (path loss + shadowing)
        
        Args:
            distance_m: distance in meters
            los_condition: LoS indicator
            
        Returns:
            Large-scale fading in linear scale
        """
        pathloss_db = self.get_pathloss(distance_m, los_condition)
        shadowing_db = self.get_shadowing(1)[0]
        
        fading_db = pathloss_db + shadowing_db
        fading_linear = 10 ** (-fading_db / 10)
        
        return fading_linear

    def get_average_large_scale_gains(self, num_users, num_uavs):
        """
        Compute deterministic long-term channel gains used for user association.

        This follows the average A2G path-loss model in the system document and
        intentionally excludes small-scale fading and fast slot-by-slot changes.
        """
        topology = self.generate_topology(num_users, num_uavs)
        distances = topology['distances']
        horizontal_distances = topology['horizontal_distances']
        altitudes = topology['uav_altitudes']

        avg_gains = np.zeros((num_users, num_uavs), dtype=float)
        for user_idx in range(num_users):
            for uav_idx in range(num_uavs):
                los_prob = self.get_los_probability(
                    horizontal_distances[user_idx, uav_idx],
                    altitudes[uav_idx]
                )
                pl_los = self.get_pathloss(distances[user_idx, uav_idx], los_condition=True)
                pl_nlos = self.get_pathloss(distances[user_idx, uav_idx], los_condition=False)
                avg_pathloss_db = los_prob * pl_los + (1.0 - los_prob) * pl_nlos
                avg_gains[user_idx, uav_idx] = 10 ** (-avg_pathloss_db / 10.0)

        return avg_gains

    def get_association_from_large_scale(self, num_users, num_uavs):
        """Return one serving UAV per user based on long-term channel quality."""
        avg_gains = self.get_average_large_scale_gains(num_users, num_uavs)
        return np.argmax(avg_gains, axis=1)
    
    def generate_channel_gains(self, num_users, num_uavs, num_subcarriers, 
                               fading_type='rayleigh', rician_k=5.0):
        """
        Generate CSI for user-UAV pairs across subcarriers
        
        Args:
            num_users: total users
            num_uavs: number of UAVs
            num_subcarriers: number of RBs
            fading_type: 'rayleigh' or 'rician'
            rician_k: K-factor for Rician fading
            
        Returns:
            Channel gains shape (num_users, num_uavs, num_subcarriers)
        """
        gains = np.zeros((num_users, num_uavs, num_subcarriers), dtype=complex)
        
        topology = self.generate_topology(num_users, num_uavs)
        distances = topology['distances']
        horizontal_distances = topology['horizontal_distances']
        altitudes = topology['uav_altitudes']

        for u in range(num_users):
            for b in range(num_uavs):
                distance = distances[u, b]
                horizontal_distance = horizontal_distances[u, b]
                los_prob = self.get_los_probability(horizontal_distance, altitudes[b])
                los = self.rng.uniform() < los_prob
                large_scale = self.get_large_scale_fading(distance, los)
                
                # Small-scale fading per subcarrier
                if fading_type == 'rayleigh':
                    h_real = np.random.randn(num_subcarriers)
                    h_imag = np.random.randn(num_subcarriers)
                elif fading_type == 'rician':
                    # Rician: Line of Sight component + Rayleigh
                    los_component = np.sqrt(rician_k / (rician_k + 1))
                    rayleigh_std = np.sqrt(1 / (2 * (rician_k + 1)))
                    h_real = los_component + rayleigh_std * np.random.randn(num_subcarriers)
                    h_imag = rayleigh_std * np.random.randn(num_subcarriers)
                else:
                    raise ValueError(f"Unknown fading type: {fading_type}")
                
                h = h_real + 1j * h_imag
                gains[u, b, :] = np.sqrt(large_scale) * h
        
        return gains
    
    def get_channel_magnitude_squared(self, gains):
        """
        Get |h|^2 (power gain)
        
        Args:
            gains: complex channel gains
            
        Returns:
            Power gains |h|^2
        """
        return np.abs(gains) ** 2
    
    def update_channels(self, current_gains, correlation=0.9):
        """
        Update channels with temporal correlation
        
        Args:
            current_gains: current channel gains
            correlation: temporal correlation (0.9 = slow fading)
            
        Returns:
            Updated channel gains
        """
        new_random = (1 - correlation) * np.random.randn(*current_gains.shape)
        new_imaginary = (1 - correlation) * np.random.randn(*current_gains.shape)
        
        updated = (correlation * current_gains + 
                   (1 - correlation) * (new_random + 1j * new_imaginary))
        
        return updated / np.sqrt(2)  # Normalize


def create_channel_model(sys_config):
    """Factory function to create channel model"""
    return ChannelModel(sys_config)
