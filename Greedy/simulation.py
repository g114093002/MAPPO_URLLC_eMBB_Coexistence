"""
Main Simulation Engine
"""

import numpy as np
from typing import Dict, List, Optional
from config import (SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig,
                    SimulationConfig)
from channel_model import ChannelModel
from capacity_models import CapacityModels
from resource_allocator import ResourceAllocator


class MultiUAVSimulation:
    """Main simulation engine for multi-UAV URLLC-eMBB coexistence."""

    def __init__(self, sys_cfg: SystemConfig, urllc_cfg: URLLCConfig,
                 embb_cfg: eMBBConfig, algo_cfg: AlgorithmConfig,
                 sim_cfg: SimulationConfig):
        self.sys_cfg = sys_cfg
        self.urllc_cfg = urllc_cfg
        self.embb_cfg = embb_cfg
        self.algo_cfg = algo_cfg
        self.sim_cfg = sim_cfg

        self.channel_model = ChannelModel(sys_cfg)
        self.capacity_model = CapacityModels(urllc_cfg, embb_cfg)
        self.allocator = ResourceAllocator(sys_cfg, urllc_cfg, embb_cfg, algo_cfg)

        np.random.seed(sim_cfg.random_seed)
        self.sys_cfg.random_seed = int(sim_cfg.random_seed)
        if hasattr(self.channel_model, 'set_seed'):
            self.channel_model.set_seed(sim_cfg.random_seed)
        self.last_allocation = None
        self.static_association = None

    def _prepare_static_association(self):
        """Prepare long-term user-UAV association based on large-scale gains."""
        total_users = self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users
        self.static_association = self.channel_model.get_association_from_large_scale(
            total_users,
            self.sys_cfg.num_uavs
        )
        return self.static_association

    def _generate_slot_context(self):
        """Sample one fixed slot snapshot for reuse across multiple solvers."""
        channel_gains = self.channel_model.generate_channel_gains(
            self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users,
            self.sys_cfg.num_uavs,
            self.sys_cfg.num_subcarriers,
            fading_type=self.sim_cfg.csi_generation_method,
            rician_k=self.sim_cfg.rician_k_factor
        )
        channel_gains_mag_sq = np.abs(channel_gains) ** 2
        if self.static_association is None:
            best_uav_per_user = self._prepare_static_association()
        else:
            best_uav_per_user = self.static_association.copy()

        num_urllc = self.sys_cfg.num_urllc_users
        if num_urllc > 0:
            poisson_rate = getattr(
                self.sim_cfg,
                'urllc_poisson_rate',
                max(self.sim_cfg.urllc_arrival_prob * num_urllc, 0.0)
            )
            arrival_count = int(np.random.poisson(poisson_rate))
            urllc_packet_sources = np.asarray([], dtype=int)
            if arrival_count > 0:
                urllc_packet_sources = np.random.choice(num_urllc, size=arrival_count, replace=True)
        else:
            urllc_packet_sources = np.asarray([], dtype=int)

        return {
            'channel_gains': channel_gains,
            'channel_gains_mag_sq': channel_gains_mag_sq,
            'best_uav_per_user': np.asarray(best_uav_per_user, dtype=int),
            'urllc_packet_sources': np.asarray(urllc_packet_sources, dtype=int),
        }

    def _initialize_empty_urllc_state(self, packet_sources):
        """Populate allocator buffers for an empty URLLC schedule."""
        num_packets = len(packet_sources)
        num_uavs = self.sys_cfg.num_uavs
        num_rbs = self.sys_cfg.num_subcarriers
        num_minislots = self.sys_cfg.num_minislots
        self.allocator.urllc_power_allocation = np.zeros((num_packets, num_uavs), dtype=float)
        self.allocator.urllc_rb_allocation = np.zeros((num_packets, num_rbs), dtype=int)
        self.allocator.urllc_selected_uavs = np.full(num_packets, -1, dtype=int)
        self.allocator.urllc_packet_sources = np.asarray(packet_sources, dtype=int)
        self.allocator.urllc_timefreq_grid = np.full((num_uavs, num_rbs, num_minislots), -1, dtype=int)
        self.allocator.noma_decisions = np.full((num_uavs, num_rbs, num_minislots), 'EMPTY', dtype='<U5')
        self.allocator.coexistence_mode_per_uav = np.full((num_uavs, num_rbs, num_minislots), 'EMPTY', dtype='<U5')
        self.allocator.coexistence_embb_user_per_uav = np.full((num_uavs, num_rbs, num_minislots), -1, dtype=int)
        self.allocator.coexistence_urllc_user_per_uav = np.full((num_uavs, num_rbs, num_minislots), -1, dtype=int)
        self.allocator.coexistence_urllc_packet_per_uav = np.full((num_uavs, num_rbs, num_minislots), -1, dtype=int)
        self.allocator.rho_tensor = np.zeros(
            (
                self.sys_cfg.num_embb_users,
                self.sys_cfg.num_urllc_users,
                num_uavs,
                num_rbs,
                num_minislots,
            ),
            dtype=np.uint8
        )
        self.allocator.varpi_tensor = np.zeros_like(self.allocator.rho_tensor)
        self.allocator.rho_action_list = []
        self.allocator.varpi_action_list = []

    def _package_allocation_result(
        self,
        slot_index,
        channel_gains,
        channel_gains_mag_sq,
        best_uav_per_user,
        embb_result,
        urllc_power,
        urllc_reliability,
        urllc_packet_sources,
        policy_name='original_greedy',
        admission_quota: Optional[int] = None,
    ):
        """Build the standard simulation result payload from current allocator state."""
        scheduled_urllc_mask = ~np.isnan(urllc_reliability)
        admitted_urllc_success_values = np.array([], dtype=float)
        if np.any(scheduled_urllc_mask):
            admitted_urllc_success_values = urllc_reliability[scheduled_urllc_mask]

        coexistence_mode = self.allocator.coexistence_mode_per_uav
        if coexistence_mode is not None:
            urllc_cell_fraction = float(np.mean(coexistence_mode != 'EMPTY'))
            noma_cell_fraction = float(np.mean(coexistence_mode == 'NOMA'))
            punct_cell_fraction = float(np.mean(coexistence_mode == 'PUNCT'))
        else:
            urllc_cell_fraction = 0.0
            noma_cell_fraction = 0.0
            punct_cell_fraction = 0.0

        alpha_e = embb_result.get('alpha_e')
        if alpha_e is not None:
            embb_rb_occupancy = np.any(alpha_e == 1, axis=0)
            embb_occupied_fraction = float(np.mean(embb_rb_occupancy))
        else:
            embb_occupied_fraction = float(np.mean(np.any(embb_result['rb_allocation'] == 1, axis=0)))

        metrics = {
            'embb_total_rate': embb_result['total_rate'],
            'embb_per_user_rate': embb_result['rates'],
            'urllc_success_rate': (
                float(np.mean(admitted_urllc_success_values))
                if admitted_urllc_success_values.size > 0 else float('nan')
            ),
            'urllc_admission_rate': (
                float(np.sum(scheduled_urllc_mask) / max(len(urllc_packet_sources), 1))
                if len(urllc_packet_sources) > 0 else float('nan')
            ),
            'urllc_individual_success': urllc_reliability,
            'total_power': float(np.sum(urllc_power) / max(self.sys_cfg.num_minislots, 1) + np.sum(embb_result['power_allocation'])),
            'urllc_power': float(np.sum(urllc_power) / max(self.sys_cfg.num_minislots, 1)),
            'embb_power': float(np.sum(embb_result['power_allocation'])),
            'rb_utilization': embb_occupied_fraction,
            'urllc_cell_occupancy': urllc_cell_fraction,
            'joint_resource_pressure': min(1.0, embb_occupied_fraction + urllc_cell_fraction),
            'noma_ratio': float(np.mean(self.allocator.noma_decisions == 'NOMA')) if self.allocator.noma_decisions is not None else 0.0,
            'noma_cell_fraction': noma_cell_fraction,
            'punct_cell_fraction': punct_cell_fraction,
            'active_urllc_users': int(len(urllc_packet_sources)),
            'scheduled_urllc_users': int(np.count_nonzero(scheduled_urllc_mask)),
            'urllc_constraint_violations': (
                int(np.count_nonzero(
                    admitted_urllc_success_values < (1.0 - self.urllc_cfg.target_error_probability)
                ))
                if admitted_urllc_success_values.size > 0 else 0
            ),
            'embb_served_users': int(np.count_nonzero(embb_result['rates'] > 0)),
            'embb_user_rate_mean': float(np.mean(embb_result['rates'])) if embb_result['rates'].size > 0 else 0.0
        }

        allocation = self.allocator.get_allocation_summary()
        explicit_association = np.zeros(
            (self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users, self.sys_cfg.num_uavs),
            dtype=int
        )
        explicit_association[np.arange(explicit_association.shape[0]), best_uav_per_user] = 1
        allocation['user_association'] = explicit_association
        allocation['best_uav_per_user'] = best_uav_per_user
        allocation['topology'] = getattr(self.channel_model, 'last_topology', None)
        allocation['slot_index'] = slot_index
        allocation['policy_name'] = policy_name
        allocation['admission_quota'] = None if admission_quota is None else int(admission_quota)
        self.last_allocation = allocation

        rho_actions = allocation.get('rho_action_list', []) or []
        varpi_actions = allocation.get('varpi_action_list', []) or []
        overlay_count = len(rho_actions)
        puncture_count = len(varpi_actions)
        total_mode_actions = overlay_count + puncture_count
        overlay_retention_values = [
            float(action.get('retained_fraction', 0.0))
            for action in rho_actions
            if action.get('q', -1) >= 0
        ]
        puncture_loss_values = [
            float(action.get('embb_loss_per_action', 0.0))
            for action in varpi_actions
            if action.get('q', -1) >= 0
        ]
        overlay_loss_values = [
            float(action.get('embb_loss_per_action', 0.0))
            for action in rho_actions
            if action.get('q', -1) >= 0
        ]
        overlay_diag = allocation.get('overlay_diagnostics', {}) or {}
        overlay_candidate_pairs = int(overlay_diag.get('candidate_pairs_total', 0))
        overlay_feasible_pairs = int(overlay_diag.get('feasible_overlay_pairs_total', 0))
        overlay_selected_pairs = int(overlay_diag.get('selected_overlay_pairs_total', overlay_count))

        embb_rates = np.asarray(embb_result['rates'], dtype=float)
        jain_fairness = self._compute_jain_fairness(embb_rates)
        cell_edge_served_ratio = self._compute_cell_edge_served_ratio(
            allocation.get('topology'),
            best_uav_per_user,
            embb_rates
        )
        per_uav_embb_assoc = np.bincount(
            best_uav_per_user[self.sys_cfg.num_urllc_users:],
            minlength=self.sys_cfg.num_uavs
        )
        per_uav_urllc_assoc = np.bincount(
            best_uav_per_user[:self.sys_cfg.num_urllc_users],
            minlength=self.sys_cfg.num_uavs
        )
        scheduled_urllc_uavs = allocation.get('urllc_selected_uavs')
        valid_scheduled_uavs = np.asarray([], dtype=int)
        if scheduled_urllc_uavs is not None:
            valid_scheduled_uavs = np.asarray(scheduled_urllc_uavs, dtype=int)
            valid_scheduled_uavs = valid_scheduled_uavs[valid_scheduled_uavs >= 0]
        per_uav_urllc_scheduled = np.bincount(
            valid_scheduled_uavs,
            minlength=self.sys_cfg.num_uavs
        ) if valid_scheduled_uavs.size > 0 else np.zeros(self.sys_cfg.num_uavs, dtype=int)
        embb_uavs = np.asarray(best_uav_per_user[self.sys_cfg.num_urllc_users:], dtype=int)
        per_uav_total_assoc = per_uav_embb_assoc + per_uav_urllc_assoc
        per_uav_total_load_std = float(np.std(per_uav_total_assoc))
        per_uav_urllc_sched_std = float(np.std(per_uav_urllc_scheduled))
        per_uav_embb_scheduled = np.bincount(
            embb_uavs,
            weights=(embb_rates > 0).astype(float),
            minlength=self.sys_cfg.num_uavs
        ).astype(int)
        per_uav_embb_throughput = np.bincount(
            embb_uavs,
            weights=embb_rates,
            minlength=self.sys_cfg.num_uavs
        ).astype(float)
        per_uav_overlay = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        per_uav_puncture = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        if rho_actions:
            per_uav_overlay = np.bincount(
                np.asarray([int(action['j']) for action in rho_actions], dtype=int),
                minlength=self.sys_cfg.num_uavs
            ).astype(int)
        if varpi_actions:
            per_uav_puncture = np.bincount(
                np.asarray([int(action['j']) for action in varpi_actions], dtype=int),
                minlength=self.sys_cfg.num_uavs
            ).astype(int)

        topology = allocation.get('topology')
        per_uav_avg_embb_distance = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_urllc_distance = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_embb_gain = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_urllc_gain = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        if topology is not None:
            distances = np.asarray(topology.get('horizontal_distances'), dtype=float)
            embb_distance_matrix = distances[self.sys_cfg.num_urllc_users:, :]
            urllc_distance_matrix = distances[:self.sys_cfg.num_urllc_users, :]
            for uav_idx in range(self.sys_cfg.num_uavs):
                embb_mask = embb_uavs == uav_idx
                urllc_mask = np.asarray(best_uav_per_user[:self.sys_cfg.num_urllc_users], dtype=int) == uav_idx
                if np.any(embb_mask):
                    per_uav_avg_embb_distance[uav_idx] = float(np.mean(embb_distance_matrix[embb_mask, uav_idx]))
                    embb_users_global = self.sys_cfg.num_urllc_users + np.where(embb_mask)[0]
                    per_uav_avg_embb_gain[uav_idx] = float(np.mean(channel_gains_mag_sq[embb_users_global, uav_idx, :]))
                if np.any(urllc_mask):
                    per_uav_avg_urllc_distance[uav_idx] = float(np.mean(urllc_distance_matrix[urllc_mask, uav_idx]))
                    urllc_users_global = np.where(urllc_mask)[0]
                    per_uav_avg_urllc_gain[uav_idx] = float(np.mean(channel_gains_mag_sq[urllc_users_global, uav_idx, :]))

        owner_per_uav_rb = allocation.get('embb_owner_per_uav_rb')
        coexistence_mode_per_uav = allocation.get('coexistence_mode_per_uav')
        embb_only_fraction = 0.0
        overlay_fraction = 0.0
        puncture_fraction = 0.0
        idle_fraction = 1.0
        minislot_utilization = 0.0
        if owner_per_uav_rb is not None:
            owner_per_uav_rb = np.asarray(owner_per_uav_rb, dtype=int)
            if coexistence_mode_per_uav is not None:
                coexistence_mode_per_uav = np.asarray(coexistence_mode_per_uav)
                owner_3d = np.repeat(owner_per_uav_rb[:, :, None], self.sys_cfg.num_minislots, axis=2)
                overlay_mask = coexistence_mode_per_uav == 'NOMA'
                puncture_mask = coexistence_mode_per_uav == 'PUNCT'
                empty_mask = coexistence_mode_per_uav == 'EMPTY'
                embb_only_mask = (owner_3d >= 0) & empty_mask
                idle_mask = (owner_3d < 0) & empty_mask
                total_cells = float(owner_3d.size)
                embb_only_fraction = float(np.sum(embb_only_mask) / total_cells)
                overlay_fraction = float(np.sum(overlay_mask) / total_cells)
                puncture_fraction = float(np.sum(puncture_mask) / total_cells)
                idle_fraction = float(np.sum(idle_mask) / total_cells)
                minislot_utilization = 1.0 - idle_fraction
            else:
                rb_occ = owner_per_uav_rb >= 0
                embb_only_fraction = float(np.mean(rb_occ))
                idle_fraction = 1.0 - embb_only_fraction
                minislot_utilization = embb_only_fraction

        return {
            'channel_gains': channel_gains,
            'allocation': allocation,
            'metrics': {
                **metrics,
                'overlay_count': overlay_count,
                'puncture_count': puncture_count,
                'mode_action_count': total_mode_actions,
                'overlay_ratio': float(overlay_count / max(total_mode_actions, 1)),
                'puncture_ratio': float(puncture_count / max(total_mode_actions, 1)),
                'avg_overlay_retention': float(np.mean(overlay_retention_values)) if overlay_retention_values else np.nan,
                'avg_puncture_embb_loss': float(np.mean(puncture_loss_values)) if puncture_loss_values else 0.0,
                'avg_overlay_embb_loss': float(np.mean(overlay_loss_values)) if overlay_loss_values else 0.0,
                'jain_fairness': jain_fairness,
                'cell_edge_served_ratio': cell_edge_served_ratio,
                'per_uav_total_load_std': per_uav_total_load_std,
                'per_uav_urllc_sched_std': per_uav_urllc_sched_std,
                'overlay_candidate_pairs': overlay_candidate_pairs,
                'overlay_feasible_pairs': overlay_feasible_pairs,
                'overlay_selected_pairs': overlay_selected_pairs,
                'overlay_retention_samples': overlay_retention_values,
                'puncture_loss_samples': puncture_loss_values,
                'overlay_loss_samples': overlay_loss_values,
                'per_uav_associated_embb': per_uav_embb_assoc.tolist(),
                'per_uav_associated_urllc': per_uav_urllc_assoc.tolist(),
                'per_uav_scheduled_embb': per_uav_embb_scheduled.tolist(),
                'per_uav_scheduled_urllc': per_uav_urllc_scheduled.tolist(),
                'per_uav_overlay_count': per_uav_overlay.tolist(),
                'per_uav_puncture_count': per_uav_puncture.tolist(),
                'per_uav_embb_throughput': per_uav_embb_throughput.tolist(),
                'per_uav_avg_embb_distance': per_uav_avg_embb_distance.tolist(),
                'per_uav_avg_urllc_distance': per_uav_avg_urllc_distance.tolist(),
                'per_uav_avg_embb_gain': per_uav_avg_embb_gain.tolist(),
                'per_uav_avg_urllc_gain': per_uav_avg_urllc_gain.tolist(),
                'embb_only_fraction': embb_only_fraction,
                'overlay_fraction': overlay_fraction,
                'puncture_fraction': puncture_fraction,
                'idle_fraction': idle_fraction,
                'minislot_utilization': minislot_utilization
            }
        }

    def run_single_allocation(
        self,
        slot_index=0,
        refine_embb_after_urllc: bool = True,
        policy_name: str = 'original_greedy',
        selection_policy: str = 'utility',
    ):
        """Run resource allocation for a single time slot."""
        if self.sim_cfg.verbose:
            print(f"\n{'='*60}")
            print(f"Slot {slot_index}")
            print(f"{'='*60}")

        channel_gains = self.channel_model.generate_channel_gains(
            self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users,
            self.sys_cfg.num_uavs,
            self.sys_cfg.num_subcarriers,
            fading_type=self.sim_cfg.csi_generation_method,
            rician_k=self.sim_cfg.rician_k_factor
        )
        channel_gains_mag_sq = np.abs(channel_gains) ** 2
        if self.static_association is None:
            best_uav_per_user = self._prepare_static_association()
        else:
            best_uav_per_user = self.static_association.copy()

        embb_result = self.allocator.allocate_embb_greedy(
            channel_gains_mag_sq,
            associated_uavs=best_uav_per_user
        )
        if self.sim_cfg.verbose:
            print("\nPhase 1 - eMBB Allocation:")
            print(f"  Total eMBB Rate: {embb_result['total_rate']/1e6:.4f} Mbps")
            print(f"  eMBB Power (W): {np.sum(embb_result['power_allocation']):.6f}")

        num_urllc = self.sys_cfg.num_urllc_users
        if num_urllc > 0:
            poisson_rate = getattr(
                self.sim_cfg,
                'urllc_poisson_rate',
                max(self.sim_cfg.urllc_arrival_prob * num_urllc, 0.0)
            )
            arrival_count = int(np.random.poisson(poisson_rate))
            urllc_packet_sources = np.asarray([], dtype=int)
            if arrival_count > 0:
                urllc_packet_sources = np.random.choice(num_urllc, size=arrival_count, replace=True)
        else:
            urllc_packet_sources = np.asarray([], dtype=int)

        urllc_power, urllc_reliability = self.allocator.allocate_urllc_power(
            channel_gains_mag_sq,
            packet_sources=urllc_packet_sources,
            embb_rb_alloc=embb_result['rb_allocation'],
            associated_uavs=best_uav_per_user[:self.sys_cfg.num_urllc_users],
            selection_policy=selection_policy,
        )
        urllc_timefreq_grid = self.allocator.create_urllc_timefreq_schedule()

        scheduled_urllc_mask = ~np.isnan(urllc_reliability)
        admitted_urllc_success_values = np.array([], dtype=float)
        if np.any(scheduled_urllc_mask):
            admitted_urllc_success_values = urllc_reliability[scheduled_urllc_mask]

        if self.sim_cfg.verbose:
            admitted_success = (
                float(np.mean(admitted_urllc_success_values))
                if admitted_urllc_success_values.size > 0 else float('nan')
            )
            admission_ratio = (
                float(np.sum(scheduled_urllc_mask) / max(len(urllc_packet_sources), 1))
                if len(urllc_packet_sources) > 0 else float('nan')
            )
            print("\nPhase 2 - URLLC Overlay:")
            print(f"  Active URLLC Packets: {len(urllc_packet_sources)} from {num_urllc} users")
            print(f"  URLLC Admission Ratio: {admission_ratio:.4f}")
            print(f"  Admitted URLLC Reliability: {admitted_success:.4f}")
            print(f"  URLLC Power (slot-avg W): {np.sum(urllc_power) / max(self.sys_cfg.num_minislots, 1):.6f}")

        noma_decisions = self.allocator.decide_noma_puncturing(
            channel_gains_mag_sq,
            np.arange(self.sys_cfg.num_urllc_users),
            np.arange(self.sys_cfg.num_embb_users),
            urllc_power,
            urllc_timefreq_grid=urllc_timefreq_grid,
            owner_per_rb=embb_result['owner_per_rb']
        )

        if refine_embb_after_urllc:
            embb_result = self.allocator.adjust_embb_after_urllc(
                embb_result['rb_allocation'],
                channel_gains_mag_sq,
                urllc_timefreq_grid=urllc_timefreq_grid,
                noma_decisions=noma_decisions,
                associated_uavs=best_uav_per_user
            )

        noma_count = int(np.sum(noma_decisions == 'NOMA'))
        punct_count = int(np.sum(noma_decisions == 'PUNCT'))
        if self.sim_cfg.verbose:
            print("\nPhase 3 - NOMA vs Puncturing:")
            print(f"  NOMA Minislot-RBs: {noma_count}")
            print(f"  Punctured Minislot-RBs: {punct_count}")
            print(f"  Final eMBB Rate: {embb_result['total_rate']/1e6:.4f} Mbps")

        alpha_e = embb_result.get('alpha_e')
        if alpha_e is not None:
            embb_rb_occupancy = np.any(alpha_e == 1, axis=0)
            embb_occupied_fraction = float(np.mean(embb_rb_occupancy))
        else:
            embb_occupied_fraction = float(np.mean(np.any(embb_result['rb_allocation'] == 1, axis=0)))

        coexistence_mode = self.allocator.coexistence_mode_per_uav
        if coexistence_mode is not None:
            urllc_cell_fraction = float(np.mean(coexistence_mode != 'EMPTY'))
            noma_cell_fraction = float(np.mean(coexistence_mode == 'NOMA'))
            punct_cell_fraction = float(np.mean(coexistence_mode == 'PUNCT'))
        else:
            urllc_cell_fraction = 0.0
            noma_cell_fraction = 0.0
            punct_cell_fraction = 0.0

        urllc_success_metric = (
            float(np.mean(admitted_urllc_success_values))
            if admitted_urllc_success_values.size > 0 else float('nan')
        )
        urllc_admission_metric = (
            float(np.sum(scheduled_urllc_mask) / max(len(urllc_packet_sources), 1))
            if len(urllc_packet_sources) > 0 else float('nan')
        )
        urllc_constraint_violation = (
            int(np.count_nonzero(
                admitted_urllc_success_values < (1.0 - self.urllc_cfg.target_error_probability)
            ))
            if admitted_urllc_success_values.size > 0 else 0
        )

        metrics = {
            'embb_total_rate': embb_result['total_rate'],
            'embb_per_user_rate': embb_result['rates'],
            'urllc_success_rate': urllc_success_metric,
            'urllc_admission_rate': urllc_admission_metric,
            'urllc_individual_success': urllc_reliability,
            'total_power': float(np.sum(urllc_power) / max(self.sys_cfg.num_minislots, 1) + np.sum(embb_result['power_allocation'])),
            'urllc_power': float(np.sum(urllc_power) / max(self.sys_cfg.num_minislots, 1)),
            'embb_power': float(np.sum(embb_result['power_allocation'])),
            'rb_utilization': embb_occupied_fraction,
            'urllc_cell_occupancy': urllc_cell_fraction,
            'joint_resource_pressure': min(1.0, embb_occupied_fraction + urllc_cell_fraction),
            'noma_ratio': float(noma_count / max(noma_count + punct_count, 1)),
            'noma_cell_fraction': noma_cell_fraction,
            'punct_cell_fraction': punct_cell_fraction,
            'active_urllc_users': int(len(urllc_packet_sources)),
            'scheduled_urllc_users': int(np.count_nonzero(scheduled_urllc_mask)),
            'urllc_constraint_violations': urllc_constraint_violation,
            'embb_served_users': int(np.count_nonzero(embb_result['rates'] > 0)),
            'embb_user_rate_mean': float(np.mean(embb_result['rates'])) if embb_result['rates'].size > 0 else 0.0
        }

        allocation = self.allocator.get_allocation_summary()
        explicit_association = np.zeros(
            (self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users, self.sys_cfg.num_uavs),
            dtype=int
        )
        explicit_association[np.arange(explicit_association.shape[0]), best_uav_per_user] = 1
        allocation['user_association'] = explicit_association
        allocation['best_uav_per_user'] = best_uav_per_user
        allocation['topology'] = getattr(self.channel_model, 'last_topology', None)
        allocation['slot_index'] = slot_index
        self.last_allocation = allocation

        rho_actions = allocation.get('rho_action_list', []) or []
        varpi_actions = allocation.get('varpi_action_list', []) or []
        overlay_count = len(rho_actions)
        puncture_count = len(varpi_actions)
        total_mode_actions = overlay_count + puncture_count
        overlay_retention_values = [
            float(action.get('retained_fraction', 0.0))
            for action in rho_actions
            if action.get('q', -1) >= 0
        ]
        puncture_loss_values = [
            float(action.get('embb_loss_per_action', 0.0))
            for action in varpi_actions
            if action.get('q', -1) >= 0
        ]
        overlay_loss_values = [
            float(action.get('embb_loss_per_action', 0.0))
            for action in rho_actions
            if action.get('q', -1) >= 0
        ]
        overlay_diag = allocation.get('overlay_diagnostics', {}) or {}
        overlay_candidate_pairs = int(overlay_diag.get('candidate_pairs_total', 0))
        overlay_feasible_pairs = int(overlay_diag.get('feasible_overlay_pairs_total', 0))
        overlay_selected_pairs = int(overlay_diag.get('selected_overlay_pairs_total', overlay_count))

        embb_rates = np.asarray(embb_result['rates'], dtype=float)
        jain_fairness = self._compute_jain_fairness(embb_rates)
        cell_edge_served_ratio = self._compute_cell_edge_served_ratio(
            allocation.get('topology'),
            best_uav_per_user,
            embb_rates
        )
        per_uav_embb_assoc = np.bincount(
            best_uav_per_user[self.sys_cfg.num_urllc_users:],
            minlength=self.sys_cfg.num_uavs
        )
        per_uav_urllc_assoc = np.bincount(
            best_uav_per_user[:self.sys_cfg.num_urllc_users],
            minlength=self.sys_cfg.num_uavs
        )
        scheduled_urllc_uavs = allocation.get('urllc_selected_uavs')
        valid_scheduled_uavs = np.asarray([], dtype=int)
        if scheduled_urllc_uavs is not None:
            valid_scheduled_uavs = np.asarray(scheduled_urllc_uavs, dtype=int)
            valid_scheduled_uavs = valid_scheduled_uavs[valid_scheduled_uavs >= 0]
        per_uav_urllc_scheduled = np.bincount(
            valid_scheduled_uavs,
            minlength=self.sys_cfg.num_uavs
        ) if valid_scheduled_uavs.size > 0 else np.zeros(self.sys_cfg.num_uavs, dtype=int)
        per_uav_total_assoc = per_uav_embb_assoc + per_uav_urllc_assoc
        per_uav_total_load_std = float(np.std(per_uav_total_assoc))
        per_uav_urllc_sched_std = float(np.std(per_uav_urllc_scheduled))
        per_uav_embb_scheduled = np.bincount(
            embb_uavs := np.asarray(best_uav_per_user[self.sys_cfg.num_urllc_users:], dtype=int),
            weights=(embb_rates > 0).astype(float),
            minlength=self.sys_cfg.num_uavs
        ).astype(int)
        per_uav_embb_throughput = np.bincount(
            embb_uavs,
            weights=embb_rates,
            minlength=self.sys_cfg.num_uavs
        ).astype(float)
        per_uav_overlay = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        per_uav_puncture = np.zeros(self.sys_cfg.num_uavs, dtype=int)
        if rho_actions:
            per_uav_overlay = np.bincount(
                np.asarray([int(action['j']) for action in rho_actions], dtype=int),
                minlength=self.sys_cfg.num_uavs
            ).astype(int)
        if varpi_actions:
            per_uav_puncture = np.bincount(
                np.asarray([int(action['j']) for action in varpi_actions], dtype=int),
                minlength=self.sys_cfg.num_uavs
            ).astype(int)

        topology = allocation.get('topology')
        per_uav_avg_embb_distance = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_urllc_distance = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_embb_gain = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        per_uav_avg_urllc_gain = np.full(self.sys_cfg.num_uavs, np.nan, dtype=float)
        if topology is not None:
            distances = np.asarray(topology.get('horizontal_distances'), dtype=float)
            embb_distance_matrix = distances[self.sys_cfg.num_urllc_users:, :]
            urllc_distance_matrix = distances[:self.sys_cfg.num_urllc_users, :]
            for uav_idx in range(self.sys_cfg.num_uavs):
                embb_mask = embb_uavs == uav_idx
                urllc_mask = np.asarray(best_uav_per_user[:self.sys_cfg.num_urllc_users], dtype=int) == uav_idx
                if np.any(embb_mask):
                    per_uav_avg_embb_distance[uav_idx] = float(np.mean(embb_distance_matrix[embb_mask, uav_idx]))
                    embb_users_global = self.sys_cfg.num_urllc_users + np.where(embb_mask)[0]
                    per_uav_avg_embb_gain[uav_idx] = float(np.mean(channel_gains_mag_sq[embb_users_global, uav_idx, :]))
                if np.any(urllc_mask):
                    per_uav_avg_urllc_distance[uav_idx] = float(np.mean(urllc_distance_matrix[urllc_mask, uav_idx]))
                    urllc_users_global = np.where(urllc_mask)[0]
                    per_uav_avg_urllc_gain[uav_idx] = float(np.mean(channel_gains_mag_sq[urllc_users_global, uav_idx, :]))

        owner_per_uav_rb = allocation.get('embb_owner_per_uav_rb')
        coexistence_mode_per_uav = allocation.get('coexistence_mode_per_uav')
        embb_only_fraction = 0.0
        overlay_fraction = 0.0
        puncture_fraction = 0.0
        idle_fraction = 1.0
        minislot_utilization = 0.0
        if owner_per_uav_rb is not None:
            owner_per_uav_rb = np.asarray(owner_per_uav_rb, dtype=int)
            if coexistence_mode_per_uav is not None:
                coexistence_mode_per_uav = np.asarray(coexistence_mode_per_uav)
                owner_3d = np.repeat(owner_per_uav_rb[:, :, None], self.sys_cfg.num_minislots, axis=2)
                overlay_mask = coexistence_mode_per_uav == 'NOMA'
                puncture_mask = coexistence_mode_per_uav == 'PUNCT'
                empty_mask = coexistence_mode_per_uav == 'EMPTY'
                embb_only_mask = (owner_3d >= 0) & empty_mask
                idle_mask = (owner_3d < 0) & empty_mask
                total_cells = float(owner_3d.size)
                embb_only_fraction = float(np.sum(embb_only_mask) / total_cells)
                overlay_fraction = float(np.sum(overlay_mask) / total_cells)
                puncture_fraction = float(np.sum(puncture_mask) / total_cells)
                idle_fraction = float(np.sum(idle_mask) / total_cells)
                minislot_utilization = 1.0 - idle_fraction
            else:
                rb_occ = owner_per_uav_rb >= 0
                embb_only_fraction = float(np.mean(rb_occ))
                idle_fraction = 1.0 - embb_only_fraction
                minislot_utilization = embb_only_fraction

        return {
            'channel_gains': channel_gains,
            'allocation': allocation,
            'metrics': {
                **metrics,
                'overlay_count': overlay_count,
                'puncture_count': puncture_count,
                'mode_action_count': total_mode_actions,
                'overlay_ratio': float(overlay_count / max(total_mode_actions, 1)),
                'puncture_ratio': float(puncture_count / max(total_mode_actions, 1)),
                'avg_overlay_retention': float(np.mean(overlay_retention_values)) if overlay_retention_values else np.nan,
                'avg_puncture_embb_loss': float(np.mean(puncture_loss_values)) if puncture_loss_values else 0.0,
                'avg_overlay_embb_loss': float(np.mean(overlay_loss_values)) if overlay_loss_values else 0.0,
                'jain_fairness': jain_fairness,
                'cell_edge_served_ratio': cell_edge_served_ratio,
                'per_uav_total_load_std': per_uav_total_load_std,
                'per_uav_urllc_sched_std': per_uav_urllc_sched_std,
                'overlay_candidate_pairs': overlay_candidate_pairs,
                'overlay_feasible_pairs': overlay_feasible_pairs,
                'overlay_selected_pairs': overlay_selected_pairs,
                'overlay_retention_samples': overlay_retention_values,
                'puncture_loss_samples': puncture_loss_values,
                'overlay_loss_samples': overlay_loss_values,
                'per_uav_associated_embb': per_uav_embb_assoc.tolist(),
                'per_uav_associated_urllc': per_uav_urllc_assoc.tolist(),
                'per_uav_scheduled_embb': per_uav_embb_scheduled.tolist(),
                'per_uav_scheduled_urllc': per_uav_urllc_scheduled.tolist(),
                'per_uav_overlay_count': per_uav_overlay.tolist(),
                'per_uav_puncture_count': per_uav_puncture.tolist(),
                'per_uav_embb_throughput': per_uav_embb_throughput.tolist(),
                'per_uav_avg_embb_distance': per_uav_avg_embb_distance.tolist(),
                'per_uav_avg_urllc_distance': per_uav_avg_urllc_distance.tolist(),
                'per_uav_avg_embb_gain': per_uav_avg_embb_gain.tolist(),
                'per_uav_avg_urllc_gain': per_uav_avg_urllc_gain.tolist(),
                'embb_only_fraction': embb_only_fraction,
                'overlay_fraction': overlay_fraction,
                'puncture_fraction': puncture_fraction,
                'idle_fraction': idle_fraction,
                'minislot_utilization': minislot_utilization
            }
        }

    def run_single_allocation_with_context(
        self,
        slot_context: Dict,
        slot_index=0,
        refine_embb_after_urllc: bool = True,
        policy_name: str = 'original_greedy',
        selection_policy: str = 'utility',
    ):
        """Run the original greedy pipeline on a fixed pre-sampled slot context."""
        channel_gains = slot_context['channel_gains']
        channel_gains_mag_sq = slot_context['channel_gains_mag_sq']
        best_uav_per_user = np.asarray(slot_context['best_uav_per_user'], dtype=int)
        urllc_packet_sources = np.asarray(slot_context.get('urllc_packet_sources', []), dtype=int)

        embb_result = self.allocator.allocate_embb_greedy(
            channel_gains_mag_sq,
            associated_uavs=best_uav_per_user
        )
        urllc_power, urllc_reliability = self.allocator.allocate_urllc_power(
            channel_gains_mag_sq,
            packet_sources=urllc_packet_sources,
            embb_rb_alloc=embb_result['rb_allocation'],
            associated_uavs=best_uav_per_user[:self.sys_cfg.num_urllc_users],
            selection_policy=selection_policy,
        )
        urllc_timefreq_grid = self.allocator.create_urllc_timefreq_schedule()
        noma_decisions = self.allocator.decide_noma_puncturing(
            channel_gains_mag_sq,
            np.arange(self.sys_cfg.num_urllc_users),
            np.arange(self.sys_cfg.num_embb_users),
            urllc_power,
            urllc_timefreq_grid=urllc_timefreq_grid,
            owner_per_rb=embb_result['owner_per_rb']
        )
        if refine_embb_after_urllc and np.any(~np.isnan(urllc_reliability)):
            embb_result = self.allocator.adjust_embb_after_urllc(
                embb_result['rb_allocation'],
                channel_gains_mag_sq,
                urllc_timefreq_grid=urllc_timefreq_grid,
                noma_decisions=noma_decisions,
                associated_uavs=best_uav_per_user
            )
        return self._package_allocation_result(
            slot_index,
            channel_gains,
            channel_gains_mag_sq,
            best_uav_per_user,
            embb_result,
            urllc_power,
            urllc_reliability,
            urllc_packet_sources,
            policy_name=policy_name,
        )

    def run_single_allocation_lite_with_context(self, slot_context: Dict, slot_index=0):
        """Run the lite greedy pipeline on a fixed pre-sampled slot context."""
        return self.run_single_allocation_with_context(
            slot_context,
            slot_index=slot_index,
            refine_embb_after_urllc=False,
            policy_name='original_greedy_normal_v1',
            selection_policy='utility',
        )

    def run_single_allocation_lite(self, slot_index=0):
        """Run the lite greedy pipeline without post-URLLC eMBB refinement."""
        return self.run_single_allocation(
            slot_index=slot_index,
            refine_embb_after_urllc=False,
            policy_name='original_greedy_normal_v1',
            selection_policy='utility',
        )

    def run_single_allocation_normal_v1_with_context(self, slot_context: Dict, slot_index=0):
        """Alias for the normal greedy v1 baseline."""
        return self.run_single_allocation_lite_with_context(slot_context, slot_index=slot_index)

    def run_single_allocation_normal_v1(self, slot_index=0):
        """Alias for the normal greedy v1 baseline."""
        return self.run_single_allocation_lite(slot_index=slot_index)

    def run_single_allocation_normal_v2_with_context(self, slot_context: Dict, slot_index=0):
        """Run a strictly local one-pass greedy baseline without retune/refine."""
        return self.run_single_allocation_with_context(
            slot_context,
            slot_index=slot_index,
            refine_embb_after_urllc=False,
            policy_name='original_greedy_normal_v2',
            selection_policy='normal_v2',
        )

    def run_single_allocation_normal_v2(self, slot_index=0):
        """Run a strictly local one-pass greedy baseline without retune/refine."""
        return self.run_single_allocation(
            slot_index=slot_index,
            refine_embb_after_urllc=False,
            policy_name='original_greedy_normal_v2',
            selection_policy='normal_v2',
        )

    def run_embb_only_ceiling(self, slot_index=0, slot_context: Optional[Dict] = None):
        """Run the eMBB-only ceiling baseline on a fixed slot context."""
        slot_context = slot_context or self._generate_slot_context()
        channel_gains = slot_context['channel_gains']
        channel_gains_mag_sq = slot_context['channel_gains_mag_sq']
        best_uav_per_user = np.asarray(slot_context['best_uav_per_user'], dtype=int)
        urllc_packet_sources = np.asarray(slot_context.get('urllc_packet_sources', []), dtype=int)

        embb_result = self.allocator.allocate_embb_greedy(
            channel_gains_mag_sq,
            associated_uavs=best_uav_per_user
        )
        self._initialize_empty_urllc_state(urllc_packet_sources)
        embb_result = self.allocator.adjust_embb_after_urllc(
            embb_result['rb_allocation'],
            channel_gains_mag_sq,
            urllc_timefreq_grid=self.allocator.urllc_timefreq_grid,
            noma_decisions=self.allocator.noma_decisions,
            associated_uavs=best_uav_per_user,
        )
        urllc_power = self.allocator.urllc_power_allocation
        urllc_reliability = np.full(len(urllc_packet_sources), np.nan, dtype=float)
        return self._package_allocation_result(
            slot_index,
            channel_gains,
            channel_gains_mag_sq,
            best_uav_per_user,
            embb_result,
            urllc_power,
            urllc_reliability,
            urllc_packet_sources,
            policy_name='embb_only_ceiling',
            admission_quota=0,
        )

    def run_throughput_feasible_oracle(
        self,
        slot_index=0,
        slot_context: Optional[Dict] = None,
        admission_quota: Optional[int] = None,
    ):
        """Run the throughput-first feasible oracle on a fixed slot context."""
        slot_context = slot_context or self._generate_slot_context()
        channel_gains = slot_context['channel_gains']
        channel_gains_mag_sq = slot_context['channel_gains_mag_sq']
        best_uav_per_user = np.asarray(slot_context['best_uav_per_user'], dtype=int)
        urllc_packet_sources = np.asarray(slot_context.get('urllc_packet_sources', []), dtype=int)

        embb_result = self.allocator.allocate_embb_greedy(
            channel_gains_mag_sq,
            associated_uavs=best_uav_per_user
        )
        if admission_quota is not None and int(admission_quota) <= 0:
            self._initialize_empty_urllc_state(urllc_packet_sources)
            urllc_power = self.allocator.urllc_power_allocation
            urllc_reliability = np.full(len(urllc_packet_sources), np.nan, dtype=float)
        else:
            urllc_power, urllc_reliability = self.allocator.allocate_urllc_power(
                channel_gains_mag_sq,
                packet_sources=urllc_packet_sources,
                embb_rb_alloc=embb_result['rb_allocation'],
                associated_uavs=best_uav_per_user[:self.sys_cfg.num_urllc_users],
                selection_policy='throughput_first',
                admission_quota=admission_quota,
            )
            urllc_timefreq_grid = self.allocator.create_urllc_timefreq_schedule()
            noma_decisions = self.allocator.decide_noma_puncturing(
                channel_gains_mag_sq,
                np.arange(self.sys_cfg.num_urllc_users),
                np.arange(self.sys_cfg.num_embb_users),
                urllc_power,
                urllc_timefreq_grid=urllc_timefreq_grid,
                owner_per_rb=embb_result['owner_per_rb']
            )
            if np.any(~np.isnan(urllc_reliability)):
                embb_result = self.allocator.adjust_embb_after_urllc(
                    embb_result['rb_allocation'],
                    channel_gains_mag_sq,
                    urllc_timefreq_grid=urllc_timefreq_grid,
                    noma_decisions=noma_decisions,
                    associated_uavs=best_uav_per_user
                )
        return self._package_allocation_result(
            slot_index,
            channel_gains,
            channel_gains_mag_sq,
            best_uav_per_user,
            embb_result,
            urllc_power,
            urllc_reliability,
            urllc_packet_sources,
            policy_name='throughput_feasible_oracle',
            admission_quota=admission_quota,
        )

    def run_throughput_admission_frontier(
        self,
        slot_index=0,
        slot_context: Optional[Dict] = None,
        quota_values: Optional[List[int]] = None,
    ):
        """Sweep admission quotas on one fixed slot context."""
        slot_context = slot_context or self._generate_slot_context()
        active_packets = int(len(slot_context.get('urllc_packet_sources', [])))
        if quota_values is None:
            quota_values = list(range(active_packets + 1))
        frontier = []
        for quota in quota_values:
            result = self.run_throughput_feasible_oracle(
                slot_index=slot_index,
                slot_context=slot_context,
                admission_quota=int(quota),
            )
            metrics = result['metrics']
            frontier.append({
                'quota': int(quota),
                'active_packets': active_packets,
                'embb_total_rate': float(metrics.get('embb_total_rate', 0.0)),
                'urllc_admission_rate': float(metrics.get('urllc_admission_rate', 0.0)),
                'urllc_success_rate': float(metrics.get('urllc_success_rate', np.nan)),
                'total_power': float(metrics.get('total_power', 0.0)),
                'overlay_ratio': float(metrics.get('overlay_ratio', 0.0)),
                'puncture_ratio': float(metrics.get('puncture_ratio', 0.0)),
                'scheduled_packets': int(metrics.get('scheduled_urllc_users', 0)),
            })
        return {
            'slot_index': int(slot_index),
            'active_packets': active_packets,
            'quota_values': [int(q) for q in quota_values],
            'frontier': frontier,
        }

    def run_full_simulation(self):
        """Run complete simulation across multiple slots."""
        print("\n" + "="*60)
        print("Starting Multi-UAV URLLC-eMBB Coexistence Simulation")
        print("="*60)

        self.channel_model.reset_topology()
        self.static_association = None
        all_metrics = []
        self.allocation_history = []
        for slot in range(self.sys_cfg.num_slots):
            result = self.run_single_allocation(slot)
            all_metrics.append(result['metrics'])
            self.allocation_history.append(result['allocation'])

        aggregated = self._aggregate_metrics(all_metrics)
        if self.sim_cfg.verbose:
            print("\n" + "="*60)
            print("Simulation Complete - Summary Statistics")
            print("="*60)
            self._print_summary(aggregated)

        return aggregated

    def run_user_density_analysis(self):
        """Analyze performance versus per-UAV user density."""
        print("\n" + "="*60)
        print("User Density Analysis")
        print("="*60)

        density_points = np.geomspace(
            self.sim_cfg.min_user_density,
            self.sim_cfg.max_user_density,
            self.sim_cfg.num_density_points
        )
        density_points = np.unique(np.round(density_points, 2))
        actual_users_per_uav = []

        embb_rates_per_density = []
        embb_user_rates_per_density = []
        urllc_success_per_density = []
        urllc_admission_per_density = []
        embb_service_per_density = []
        power_per_density = []
        offered_load_per_density = []
        rb_utilization_per_density = []
        overlay_ratio_per_density = []
        puncture_ratio_per_density = []
        overlay_retention_per_density = []
        puncture_loss_per_density = []
        overlay_loss_per_density = []
        jain_fairness_per_density = []
        cell_edge_served_per_density = []
        uav_load_std_per_density = []
        uav_urllc_sched_std_per_density = []
        overlay_candidate_pairs_per_density = []
        overlay_feasible_pairs_per_density = []
        overlay_selected_pairs_per_density = []
        representative_per_uav = []
        overlay_retention_distribution = []
        puncture_loss_distribution = []
        overlay_loss_distribution = []
        embb_only_fraction_per_density = []
        overlay_fraction_per_density = []
        puncture_fraction_per_density = []
        idle_fraction_per_density = []
        minislot_utilization_per_density = []

        base_embb_per_uav = max(1, int(np.ceil(self.sys_cfg.num_embb_users / self.sys_cfg.num_uavs)))
        base_urllc_per_uav = max(1, int(np.ceil(self.sys_cfg.num_urllc_users / self.sys_cfg.num_uavs)))
        original_embb = self.sys_cfg.num_embb_users
        original_urllc = self.sys_cfg.num_urllc_users
        original_poisson_rate = self.sim_cfg.urllc_poisson_rate

        for density in density_points:
            scale = density / self.sim_cfg.min_user_density
            self.sys_cfg.num_embb_users = max(1, int(round(base_embb_per_uav * self.sys_cfg.num_uavs * scale)))
            self.sys_cfg.num_urllc_users = max(1, int(round(base_urllc_per_uav * self.sys_cfg.num_uavs * scale)))
            self.sim_cfg.urllc_poisson_rate = max(1e-6, original_poisson_rate * scale)
            self.channel_model.reset_topology()
            self.static_association = None
            actual_users_per_uav.append(
                float((self.sys_cfg.num_embb_users + self.sys_cfg.num_urllc_users) / self.sys_cfg.num_uavs)
            )

            print(f"\nDensity: {density:.1f} users/UAV")
            print(f"  eMBB users: {self.sys_cfg.num_embb_users}, URLLC users: {self.sys_cfg.num_urllc_users}")

            all_metrics = []
            for slot in range(self.sys_cfg.num_slots):
                result = self.run_single_allocation(slot)
                all_metrics.append(result['metrics'])

            agg = self._aggregate_metrics(all_metrics)
            embb_rates_per_density.append(agg['avg_embb_rate'])
            embb_user_rates_per_density.append(agg['avg_embb_user_rate'])
            urllc_success_per_density.append(agg['avg_urllc_success'])
            urllc_admission_per_density.append(agg['avg_urllc_admission'])
            embb_service_per_density.append(agg['avg_embb_service_ratio'])
            power_per_density.append(agg['avg_total_power'])
            offered_load_per_density.append(agg['avg_offered_urllc_load'])
            rb_utilization_per_density.append(agg['avg_rb_utilization'])
            overlay_ratio_per_density.append(agg['avg_overlay_ratio'])
            puncture_ratio_per_density.append(agg['avg_puncture_ratio'])
            overlay_retention_per_density.append(agg['avg_overlay_retention'])
            puncture_loss_per_density.append(agg['avg_puncture_embb_loss'])
            overlay_loss_per_density.append(agg['avg_overlay_embb_loss'])
            jain_fairness_per_density.append(agg['avg_jain_fairness'])
            cell_edge_served_per_density.append(agg['avg_cell_edge_served_ratio'])
            uav_load_std_per_density.append(agg['avg_per_uav_total_load_std'])
            uav_urllc_sched_std_per_density.append(agg['avg_per_uav_urllc_sched_std'])
            overlay_candidate_pairs_per_density.append(agg['avg_overlay_candidate_pairs'])
            overlay_feasible_pairs_per_density.append(agg['avg_overlay_feasible_pairs'])
            overlay_selected_pairs_per_density.append(agg['avg_overlay_selected_pairs'])
            representative_per_uav.append({
                'density': float(density),
                'offered_load': float(agg['avg_offered_urllc_load']),
                'associated_embb': agg['avg_per_uav_associated_embb'],
                'associated_urllc': agg['avg_per_uav_associated_urllc'],
                'scheduled_embb': agg['avg_per_uav_scheduled_embb'],
                'scheduled_urllc': agg['avg_per_uav_scheduled_urllc'],
                'overlay_count': agg['avg_per_uav_overlay_count'],
                'puncture_count': agg['avg_per_uav_puncture_count'],
                'embb_throughput': agg['avg_per_uav_embb_throughput'],
                'avg_embb_distance': agg['avg_per_uav_avg_embb_distance'],
                'avg_urllc_distance': agg['avg_per_uav_avg_urllc_distance'],
                'avg_embb_gain': agg['avg_per_uav_avg_embb_gain'],
                'avg_urllc_gain': agg['avg_per_uav_avg_urllc_gain']
            })
            overlay_retention_distribution.append(agg['all_overlay_retention_samples'])
            puncture_loss_distribution.append(agg['all_puncture_loss_samples'])
            overlay_loss_distribution.append(agg['all_overlay_loss_samples'])
            embb_only_fraction_per_density.append(agg['avg_embb_only_fraction'])
            overlay_fraction_per_density.append(agg['avg_overlay_fraction'])
            puncture_fraction_per_density.append(agg['avg_puncture_fraction'])
            idle_fraction_per_density.append(agg['avg_idle_fraction'])
            minislot_utilization_per_density.append(agg['avg_minislot_utilization'])

            print(
                f"  eMBB Rate: {agg['avg_embb_rate']/1e6:.2f} Mbps, "
                f"Per-user: {agg['avg_embb_user_rate']/1e6:.2f} Mbps, "
                f"URLLC admission: {agg['avg_urllc_admission']:.2%}"
            )

        self.sys_cfg.num_embb_users = original_embb
        self.sys_cfg.num_urllc_users = original_urllc
        self.sim_cfg.urllc_poisson_rate = original_poisson_rate

        return {
            'densities': actual_users_per_uav,
            'density_scale': density_points,
            'offered_load_scale': offered_load_per_density,
            'offered_load_packets': offered_load_per_density,
            'embb_rates': embb_rates_per_density,
            'embb_user_rates': embb_user_rates_per_density,
            'urllc_success': urllc_success_per_density,
            'urllc_admission': urllc_admission_per_density,
            'embb_service_ratio': embb_service_per_density,
            'power_consumption': power_per_density,
            'rb_utilization': rb_utilization_per_density,
            'overlay_ratio': overlay_ratio_per_density,
            'puncture_ratio': puncture_ratio_per_density,
            'overlay_retention': overlay_retention_per_density,
            'puncture_embb_loss': puncture_loss_per_density,
            'overlay_embb_loss': overlay_loss_per_density,
            'jain_fairness': jain_fairness_per_density,
            'cell_edge_served_ratio': cell_edge_served_per_density,
            'per_uav_total_load_std': uav_load_std_per_density,
            'per_uav_urllc_sched_std': uav_urllc_sched_std_per_density,
            'overlay_candidate_pairs': overlay_candidate_pairs_per_density,
            'overlay_feasible_pairs': overlay_feasible_pairs_per_density,
            'overlay_selected_pairs': overlay_selected_pairs_per_density,
            'representative_per_uav': representative_per_uav,
            'overlay_retention_distribution': overlay_retention_distribution,
            'puncture_loss_distribution': puncture_loss_distribution,
            'overlay_loss_distribution': overlay_loss_distribution,
            'embb_only_fraction': embb_only_fraction_per_density,
            'overlay_fraction': overlay_fraction_per_density,
            'puncture_fraction': puncture_fraction_per_density,
            'idle_fraction': idle_fraction_per_density,
            'minislot_utilization': minislot_utilization_per_density
        }

    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across slots."""
        embb_rates = np.array([m['embb_total_rate'] for m in all_metrics], dtype=float)
        urllc_success = np.array([m['urllc_success_rate'] for m in all_metrics], dtype=float)
        total_power = np.array([m['total_power'] for m in all_metrics], dtype=float)
        embb_power = np.array([m['embb_power'] for m in all_metrics], dtype=float)
        urllc_power = np.array([m['urllc_power'] for m in all_metrics], dtype=float)
        rb_util = np.array([m['rb_utilization'] for m in all_metrics], dtype=float)
        urllc_cell_occupancy = np.array([m['urllc_cell_occupancy'] for m in all_metrics], dtype=float)
        joint_resource_pressure = np.array([m['joint_resource_pressure'] for m in all_metrics], dtype=float)
        noma_cell_fraction = np.array([m['noma_cell_fraction'] for m in all_metrics], dtype=float)
        punct_cell_fraction = np.array([m['punct_cell_fraction'] for m in all_metrics], dtype=float)
        embb_user_rate = np.array([m['embb_user_rate_mean'] for m in all_metrics], dtype=float)
        urllc_active = np.array([m['active_urllc_users'] for m in all_metrics], dtype=float)
        urllc_scheduled = np.array([m['scheduled_urllc_users'] for m in all_metrics], dtype=float)
        embb_served = np.array([m['embb_served_users'] for m in all_metrics], dtype=float)
        total_embb_users = max(self.sys_cfg.num_embb_users, 1)
        urllc_admission = np.array([m['urllc_admission_rate'] for m in all_metrics], dtype=float)
        urllc_constraint_violations = np.array([m['urllc_constraint_violations'] for m in all_metrics], dtype=float)
        overlay_ratio = np.array([m['overlay_ratio'] for m in all_metrics], dtype=float)
        puncture_ratio = np.array([m['puncture_ratio'] for m in all_metrics], dtype=float)
        overlay_retention = np.array([m['avg_overlay_retention'] for m in all_metrics], dtype=float)
        puncture_embb_loss = np.array([m['avg_puncture_embb_loss'] for m in all_metrics], dtype=float)
        overlay_embb_loss = np.array([m['avg_overlay_embb_loss'] for m in all_metrics], dtype=float)
        jain_fairness = np.array([m['jain_fairness'] for m in all_metrics], dtype=float)
        cell_edge_served_ratio = np.array([m['cell_edge_served_ratio'] for m in all_metrics], dtype=float)
        per_uav_total_load_std = np.array([m['per_uav_total_load_std'] for m in all_metrics], dtype=float)
        per_uav_urllc_sched_std = np.array([m['per_uav_urllc_sched_std'] for m in all_metrics], dtype=float)
        overlay_candidate_pairs = np.array([m['overlay_candidate_pairs'] for m in all_metrics], dtype=float)
        overlay_feasible_pairs = np.array([m['overlay_feasible_pairs'] for m in all_metrics], dtype=float)
        overlay_selected_pairs = np.array([m['overlay_selected_pairs'] for m in all_metrics], dtype=float)
        embb_only_fraction = np.array([m['embb_only_fraction'] for m in all_metrics], dtype=float)
        overlay_fraction = np.array([m['overlay_fraction'] for m in all_metrics], dtype=float)
        puncture_fraction = np.array([m['puncture_fraction'] for m in all_metrics], dtype=float)
        idle_fraction = np.array([m['idle_fraction'] for m in all_metrics], dtype=float)
        minislot_utilization = np.array([m['minislot_utilization'] for m in all_metrics], dtype=float)
        embb_service_ratio = embb_served / total_embb_users

        def _mean_per_uav(metric_name):
            values = [np.asarray(m[metric_name], dtype=float) for m in all_metrics if metric_name in m]
            if not values:
                return []
            return np.nanmean(np.vstack(values), axis=0).tolist()

        overlay_retention_samples = [
            float(value)
            for m in all_metrics
            for value in m.get('overlay_retention_samples', [])
        ]
        puncture_loss_samples = [
            float(value)
            for m in all_metrics
            for value in m.get('puncture_loss_samples', [])
        ]
        overlay_loss_samples = [
            float(value)
            for m in all_metrics
            for value in m.get('overlay_loss_samples', [])
        ]

        return {
            'avg_embb_rate': float(np.nanmean(embb_rates)),
            'std_embb_rate': float(np.nanstd(embb_rates)),
            'avg_embb_user_rate': float(np.nanmean(embb_user_rate)),
            'avg_urllc_success': float(np.nanmean(urllc_success)),
            'std_urllc_success': float(np.nanstd(urllc_success)),
            'avg_urllc_admission': float(np.nanmean(urllc_admission)),
            'avg_urllc_constraint_violations': float(np.nanmean(urllc_constraint_violations)),
            'avg_embb_service_ratio': float(np.nanmean(embb_service_ratio)),
            'avg_total_power': float(np.nanmean(total_power)),
            'std_total_power': float(np.nanstd(total_power)),
            'avg_rb_utilization': float(np.nanmean(rb_util)),
            'avg_urllc_cell_occupancy': float(np.nanmean(urllc_cell_occupancy)),
            'avg_joint_resource_pressure': float(np.nanmean(joint_resource_pressure)),
            'avg_noma_cell_fraction': float(np.nanmean(noma_cell_fraction)),
            'avg_punct_cell_fraction': float(np.nanmean(punct_cell_fraction)),
            'avg_overlay_ratio': float(np.nanmean(overlay_ratio)),
            'avg_puncture_ratio': float(np.nanmean(puncture_ratio)),
            'avg_overlay_retention': float(np.nanmean(overlay_retention)),
            'avg_puncture_embb_loss': float(np.nanmean(puncture_embb_loss)),
            'avg_overlay_embb_loss': float(np.nanmean(overlay_embb_loss)),
            'avg_jain_fairness': float(np.nanmean(jain_fairness)),
            'avg_cell_edge_served_ratio': float(np.nanmean(cell_edge_served_ratio)),
            'avg_per_uav_total_load_std': float(np.nanmean(per_uav_total_load_std)),
            'avg_per_uav_urllc_sched_std': float(np.nanmean(per_uav_urllc_sched_std)),
            'avg_overlay_candidate_pairs': float(np.nanmean(overlay_candidate_pairs)),
            'avg_overlay_feasible_pairs': float(np.nanmean(overlay_feasible_pairs)),
            'avg_overlay_selected_pairs': float(np.nanmean(overlay_selected_pairs)),
            'avg_offered_urllc_load': float(np.nanmean(urllc_active)),
            'avg_embb_only_fraction': float(np.nanmean(embb_only_fraction)),
            'avg_overlay_fraction': float(np.nanmean(overlay_fraction)),
            'avg_puncture_fraction': float(np.nanmean(puncture_fraction)),
            'avg_idle_fraction': float(np.nanmean(idle_fraction)),
            'avg_minislot_utilization': float(np.nanmean(minislot_utilization)),
            'all_embb_rates': embb_rates.tolist(),
            'all_urllc_success': urllc_success.tolist(),
            'all_power': total_power.tolist(),
            'all_embb_power': embb_power.tolist(),
            'all_urllc_power': urllc_power.tolist(),
            'all_active_urllc_users': urllc_active.tolist(),
            'all_scheduled_urllc_users': urllc_scheduled.tolist(),
            'all_overlay_counts': overlay_selected_pairs.tolist(),
            'all_puncture_counts': np.array([m['puncture_count'] for m in all_metrics], dtype=float).tolist(),
            'all_overlay_candidate_pairs': overlay_candidate_pairs.tolist(),
            'all_overlay_feasible_pairs': overlay_feasible_pairs.tolist(),
            'all_embb_only_fraction': embb_only_fraction.tolist(),
            'all_overlay_fraction': overlay_fraction.tolist(),
            'all_puncture_fraction': puncture_fraction.tolist(),
            'all_idle_fraction': idle_fraction.tolist(),
            'all_minislot_utilization': minislot_utilization.tolist(),
            'all_overlay_retention_samples': overlay_retention_samples,
            'all_puncture_loss_samples': puncture_loss_samples,
            'all_overlay_loss_samples': overlay_loss_samples,
            'avg_per_uav_associated_embb': _mean_per_uav('per_uav_associated_embb'),
            'avg_per_uav_associated_urllc': _mean_per_uav('per_uav_associated_urllc'),
            'avg_per_uav_scheduled_embb': _mean_per_uav('per_uav_scheduled_embb'),
            'avg_per_uav_scheduled_urllc': _mean_per_uav('per_uav_scheduled_urllc'),
            'avg_per_uav_overlay_count': _mean_per_uav('per_uav_overlay_count'),
            'avg_per_uav_puncture_count': _mean_per_uav('per_uav_puncture_count'),
            'avg_per_uav_embb_throughput': _mean_per_uav('per_uav_embb_throughput'),
            'avg_per_uav_avg_embb_distance': _mean_per_uav('per_uav_avg_embb_distance'),
            'avg_per_uav_avg_urllc_distance': _mean_per_uav('per_uav_avg_urllc_distance'),
            'avg_per_uav_avg_embb_gain': _mean_per_uav('per_uav_avg_embb_gain'),
            'avg_per_uav_avg_urllc_gain': _mean_per_uav('per_uav_avg_urllc_gain')
        }

    @staticmethod
    def _compute_jain_fairness(rates):
        """Compute Jain's fairness index across eMBB user rates."""
        rates = np.asarray(rates, dtype=float)
        if rates.size == 0:
            return float('nan')
        numerator = np.sum(rates) ** 2
        denominator = rates.size * np.sum(rates ** 2)
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    def _compute_cell_edge_served_ratio(self, topology, best_uav_per_user, embb_rates):
        """Compute served ratio among cell-edge eMBB users using large-scale distance."""
        if topology is None or embb_rates.size == 0:
            return float('nan')
        num_urllc = self.sys_cfg.num_urllc_users
        num_embb = self.sys_cfg.num_embb_users
        embb_uavs = np.asarray(best_uav_per_user[num_urllc:num_urllc + num_embb], dtype=int)
        distances = np.asarray(topology['horizontal_distances'][num_urllc:num_urllc + num_embb], dtype=float)
        if distances.size == 0:
            return float('nan')
        serving_distances = distances[np.arange(num_embb), embb_uavs]
        threshold = np.percentile(serving_distances, 75)
        edge_mask = serving_distances >= threshold
        if not np.any(edge_mask):
            return float('nan')
        return float(np.mean(np.asarray(embb_rates)[edge_mask] > 0))

    def _print_summary(self, agg_metrics: Dict):
        """Print summary statistics."""
        print("\neMBB Performance:")
        print(f"  Average Total Rate: {agg_metrics['avg_embb_rate']/1e6:.4f} +/- {agg_metrics['std_embb_rate']/1e6:.4f} Mbps")
        print("\nURLLC Performance:")
        print(f"  Average Admission Ratio: {agg_metrics['avg_urllc_admission']:.4f}")
        print(f"  Average Admitted Reliability: {agg_metrics['avg_urllc_success']:.4f} +/- {agg_metrics['std_urllc_success']:.4f}")
        print("\nPower Consumption:")
        print(f"  Average Total Power: {agg_metrics['avg_total_power']:.6f} +/- {agg_metrics['std_total_power']:.6f} W")
        print("\nResource Utilization:")
        print(f"  Average RB Utilization: {agg_metrics['avg_rb_utilization']:.2%}")


def create_simulation(sys_cfg=None, urllc_cfg=None, embb_cfg=None,
                      algo_cfg=None, sim_cfg=None):
    """Factory function to create simulation."""
    from config import (DEFAULT_SYSTEM_CONFIG, DEFAULT_URLLC_CONFIG,
                        DEFAULT_EMBB_CONFIG, DEFAULT_ALGO_CONFIG,
                        DEFAULT_SIM_CONFIG)

    sys_cfg = sys_cfg or DEFAULT_SYSTEM_CONFIG
    urllc_cfg = urllc_cfg or DEFAULT_URLLC_CONFIG
    embb_cfg = embb_cfg or DEFAULT_EMBB_CONFIG
    algo_cfg = algo_cfg or DEFAULT_ALGO_CONFIG
    sim_cfg = sim_cfg or DEFAULT_SIM_CONFIG

    return MultiUAVSimulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg)
