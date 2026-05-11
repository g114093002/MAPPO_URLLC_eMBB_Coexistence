"""
Resource Allocator - constrained eMBB baseline + utility-aware URLLC admission.

This file implements the core PHY/MAC logic used by both Greedy and SR-MAPPO:
- eMBB baseline RB ownership and power budgeting
- URLLC admission with overlay vs puncture decisions
- SIC feasibility checks and rate impact estimation
"""

import numpy as np
from config import SystemConfig, URLLCConfig, eMBBConfig, AlgorithmConfig
from capacity_models import CapacityModels


class ResourceAllocator:
    """Resource allocation engine."""

    def __init__(self, sys_cfg: SystemConfig, urllc_cfg: URLLCConfig,
                 embb_cfg: eMBBConfig, algo_cfg: AlgorithmConfig):
        # Configuration and reusable models (capacity, reliability, etc.)
        self.sys_cfg = sys_cfg
        self.urllc_cfg = urllc_cfg
        self.embb_cfg = embb_cfg
        self.algo_cfg = algo_cfg
        self.capacity_model = CapacityModels(urllc_cfg, embb_cfg)

        # Runtime state buffers (filled per slot / minislot)
        self.urllc_power_allocation = None
        self.embb_power_allocation = None
        self.rb_allocation = None
        self.noma_decisions = None
        self.urllc_rb_allocation = None
        self.urllc_timefreq_grid = None
        self.alpha_e_allocation = None
        self.embb_owner_per_rb = None
        self.embb_owner_per_uav_rb = None
        self.coexistence_mode_per_uav = None
        self.coexistence_embb_user_per_uav = None
        self.coexistence_urllc_user_per_uav = None
        self.coexistence_urllc_packet_per_uav = None
        self.rho_tensor = None
        self.varpi_tensor = None
        self.rho_action_list = []
        self.varpi_action_list = []
        self.overlay_diagnostics = {}
        self.urllc_selected_uavs = None
        self.urllc_packet_sources = None
        self.embb_selected_uavs = None
        self.user_association = None
        self.embb_user_tx_power = np.array([], dtype=float)
        self.embb_base_rb_rates = np.array([], dtype=float)
        self.embb_base_rb_rates_per_uav_rb = None

    def allocate_embb_greedy(self, channel_gains_mag_sq, urllc_power_per_uav=None,
                             remaining_power_budget=None, associated_uavs=None):
        """
        Rate-driven RB assignment for eMBB under per-UAV RB constraints.
        """
        # Reset URLLC-related state (new slot baseline)
        self.urllc_power_allocation = None
        self.urllc_rb_allocation = None
        self.urllc_selected_uavs = None
        self.urllc_packet_sources = None
        self.urllc_timefreq_grid = None
        self.noma_decisions = None
        self.coexistence_mode_per_uav = None
        self.coexistence_embb_user_per_uav = None
        self.coexistence_urllc_user_per_uav = None
        self.coexistence_urllc_packet_per_uav = None
        self.rho_tensor = None
        self.varpi_tensor = None
        self.rho_action_list = []
        self.varpi_action_list = []
        self.overlay_diagnostics = {
            'candidate_pairs_total': 0,
            'feasible_overlay_pairs_total': 0,
            'selected_overlay_pairs_total': 0,
            'candidate_pairs_per_uav': np.zeros(self.sys_cfg.num_uavs, dtype=int),
            'feasible_overlay_pairs_per_uav': np.zeros(self.sys_cfg.num_uavs, dtype=int),
            'selected_overlay_pairs_per_uav': np.zeros(self.sys_cfg.num_uavs, dtype=int)
        }

        # Pre-allocate baseline RB ownership tensors
        num_embb = self.sys_cfg.num_embb_users
        num_subcarriers = channel_gains_mag_sq.shape[2]
        embb_rb_alloc = np.zeros((num_embb, num_subcarriers), dtype=int)
        alpha_e = np.zeros((num_embb, self.sys_cfg.num_uavs, num_subcarriers), dtype=int)
        best_uav_per_user = np.zeros(num_embb, dtype=int)
        max_power_per_user = np.zeros(num_embb)

        # Decide which UAV should serve each eMBB user (static association)
        for embb_idx in range(num_embb):
            user_idx = self.sys_cfg.num_urllc_users + embb_idx
            power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
            max_power_per_user[embb_idx] = min(
                self._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                self.algo_cfg.power_upper_bound
            )
            if associated_uavs is None:
                avg_gain = np.mean(channel_gains_mag_sq[user_idx, :, :], axis=1)
                best_uav_per_user[embb_idx] = int(np.argmax(avg_gain))
            else:
                best_uav_per_user[embb_idx] = int(
                    associated_uavs[self.sys_cfg.num_urllc_users + embb_idx]
                )

        owner_per_uav_rb = np.full((self.sys_cfg.num_uavs, num_subcarriers), -1, dtype=int)
        users_per_uav = {
            uav_idx: np.where(best_uav_per_user == uav_idx)[0]
            for uav_idx in range(self.sys_cfg.num_uavs)
        }

        min_rate_target = float(getattr(self.embb_cfg, "min_rate_per_user_bps", 2.0e6) or 2.0e6)
        def _power_from_alloc(rb_alloc: np.ndarray) -> np.ndarray:
            tx = np.zeros(num_embb, dtype=float)
            for embb_idx in range(num_embb):
                quota = int(np.sum(rb_alloc[embb_idx, :]))
                load_fraction = quota / max(num_subcarriers, 1)
                tx[embb_idx] = float(max_power_per_user[embb_idx] * load_fraction)
            return tx

        def _evaluate_objective(rb_alloc: np.ndarray, phase: str) -> tuple[float, float]:
            tx_powers = _power_from_alloc(rb_alloc)
            state = self._compute_embb_state(
                rb_alloc,
                channel_gains_mag_sq,
                best_uav_per_user,
                tx_powers,
            )
            rates = np.asarray(state["rates"], dtype=float)
            minrate_progress = float(np.sum(np.minimum(rates, min_rate_target)))
            total_rate = float(np.sum(rates))
            if phase == "minrate":
                return (minrate_progress, total_rate)
            return (total_rate, minrate_progress)

        candidate_cells = [
            (uav_idx, rb_idx)
            for uav_idx in range(self.sys_cfg.num_uavs)
            for rb_idx in range(num_subcarriers)
            if users_per_uav[uav_idx].size > 0
        ]

        minrate_fully_satisfied = False
        resource_limited_infeasible = False
        for phase in ("minrate", "throughput"):
            if phase == "throughput" and not minrate_fully_satisfied:
                # User policy: throughput enhancement is allowed only after
                # all eMBB users satisfy the minimum-rate target.
                break
                current_obj = _evaluate_objective(embb_rb_alloc, phase=phase)
            unassigned = set(candidate_cells)
            while unassigned:
                rates_now = np.asarray(
                    self._compute_embb_state(
                        embb_rb_alloc,
                        channel_gains_mag_sq,
                        best_uav_per_user,
                        _power_from_alloc(embb_rb_alloc),
                    )["rates"],
                    dtype=float,
                )
                unmet_users = set(np.where(rates_now < (min_rate_target - 1.0))[0].tolist())
                best_move = None
                best_score = None
                best_alloc = None
                for uav_idx, rb_idx in list(unassigned):
                    if phase == "minrate":
                        candidate_users = np.asarray(
                            [u for u in users_per_uav[uav_idx] if int(u) in unmet_users],
                            dtype=int,
                        )
                        if candidate_users.size <= 0:
                            continue
                        # User-requested behavior: in min-rate phase, choose unmet users randomly
                        # instead of ranking by deficit-coverage efficiency.
                        np.random.shuffle(candidate_users)
                    else:
                        candidate_users = users_per_uav[uav_idx]
                    for embb_idx in candidate_users:
                        trial = embb_rb_alloc.copy()
                        trial[embb_idx, rb_idx] = 1
                        score = _evaluate_objective(trial, phase=phase)
                        if (best_score is None) or (score > best_score):
                            best_score = score
                            best_move = (uav_idx, rb_idx, int(embb_idx))
                            best_alloc = trial
                if best_move is None or best_alloc is None or best_score is None:
                    break
                # In throughput phase, only accept strict improvements.
                if phase == "throughput" and best_score <= current_obj:
                    break
                embb_rb_alloc = best_alloc
                uav_idx, rb_idx, embb_idx = best_move
                alpha_e[embb_idx, uav_idx, rb_idx] = 1
                owner_per_uav_rb[uav_idx, rb_idx] = embb_idx
                unassigned.discard((uav_idx, rb_idx))
                current_obj = best_score
            if phase == "minrate":
                rates_after_min = np.asarray(
                    self._compute_embb_state(
                        embb_rb_alloc,
                        channel_gains_mag_sq,
                        best_uav_per_user,
                        _power_from_alloc(embb_rb_alloc),
                    )["rates"],
                    dtype=float,
                )
                minrate_fully_satisfied = bool(np.all(rates_after_min >= min_rate_target - 1.0))
                if minrate_fully_satisfied:
                    continue
                # Resource exhausted with unresolved min-rate deficits.
                resource_limited_infeasible = True
                break

        # Convert per-user RB counts into per-user transmit power budgets
        embb_tx_powers = _power_from_alloc(embb_rb_alloc)

        # Refresh allocator state before recomputing eMBB rates. The rate
        # evaluation path reuses per-UAV owners and per-RB powers to estimate
        # cross-UAV interference, so stale state from a previous scenario can
        # otherwise leak in when the user count changes (for example under the
        # SR-MAPPO curriculum sweeps).
        self.embb_owner_per_uav_rb = owner_per_uav_rb
        self.embb_selected_uavs = best_uav_per_user
        self.alpha_e_allocation = alpha_e
        self.rb_allocation = embb_rb_alloc
        self.embb_user_tx_power = embb_tx_powers

        baseline = self._compute_embb_state(
            embb_rb_alloc,
            channel_gains_mag_sq,
            best_uav_per_user,
            embb_tx_powers
        )

        self.embb_base_rb_rates = baseline['base_rb_rates']
        self.embb_base_rb_rates_per_uav_rb = baseline['base_rb_rates_per_uav_rb']
        self.embb_owner_per_rb = baseline['owner_per_rb']
        self.embb_power_allocation = baseline['power_allocation']

        return {
            'rb_allocation': embb_rb_alloc,
            'alpha_e': alpha_e,
            'power_allocation': baseline['power_allocation'],
            'rates': baseline['rates'],
            'total_rate': np.sum(baseline['rates']),
            'minrate_fully_satisfied': bool(minrate_fully_satisfied),
            'resource_limited_infeasible': bool(resource_limited_infeasible),
            'owner_per_rb': baseline['owner_per_rb'],
            'owner_per_uav_rb': owner_per_uav_rb,
            'best_uav_per_user': best_uav_per_user,
            'base_rb_rates': baseline['base_rb_rates'],
            'base_rb_rates_per_uav_rb': baseline['base_rb_rates_per_uav_rb'],
            'user_tx_powers': embb_tx_powers
        }

    def allocate_urllc_power(self, channel_gains_mag_sq, minislot_index=0,
                             active_urllc_mask=None, packet_sources=None, embb_rb_alloc=None,
                             associated_uavs=None, selection_policy='utility',
                             admission_quota=None):
        """
        Utility-aware URLLC admission, mode selection, and minimum-power scheduling.
        """
        # Basic dimensions for this minislot
        num_urllc = self.sys_cfg.num_urllc_users
        num_uavs = channel_gains_mag_sq.shape[1]
        num_subcarriers = channel_gains_mag_sq.shape[2]
        num_minislots = self.sys_cfg.num_minislots
        channel_uses = self.sys_cfg.channel_uses_per_minislot

        # Build the packet list for this minislot (arrivals)
        if packet_sources is None:
            if active_urllc_mask is None:
                active_urllc_mask = np.ones(num_urllc, dtype=bool)
            packet_sources = np.where(active_urllc_mask)[0]
        packet_sources = np.asarray(packet_sources, dtype=int)
        num_packets = len(packet_sources)

        # If no eMBB baseline is provided, treat as empty
        if embb_rb_alloc is None:
            embb_rb_alloc = np.zeros((self.sys_cfg.num_embb_users, num_subcarriers), dtype=int)

        # Resolve eMBB owner per UAV/RB for interference estimation
        if self.embb_owner_per_uav_rb is None:
            owner_per_rb = np.full((self.sys_cfg.num_uavs, num_subcarriers), -1, dtype=int)
            if embb_rb_alloc.size > 0 and associated_uavs is not None:
                for embb_idx in range(self.sys_cfg.num_embb_users):
                    uav_idx = int(associated_uavs[self.sys_cfg.num_urllc_users + embb_idx])
                    assigned_rbs = np.where(embb_rb_alloc[embb_idx, :] == 1)[0]
                    owner_per_rb[uav_idx, assigned_rbs] = embb_idx
        else:
            owner_per_rb = self.embb_owner_per_uav_rb.copy()

        power_alloc = np.zeros((num_packets, num_uavs))
        urllc_rb_alloc = np.zeros((num_packets, num_subcarriers), dtype=int)
        selected_uavs = np.full(num_packets, -1, dtype=int)
        reliability_achieved = np.full(num_packets, np.nan)
        grid = np.full((num_uavs, num_subcarriers, num_minislots), -1, dtype=int)
        decisions = np.full((num_uavs, num_subcarriers, num_minislots), 'EMPTY', dtype='<U5')
        coexistence_mode = np.full((num_uavs, num_subcarriers, num_minislots), 'EMPTY', dtype='<U5')
        coexistence_embb = np.full((num_uavs, num_subcarriers, num_minislots), -1, dtype=int)
        coexistence_urllc = np.full((num_uavs, num_subcarriers, num_minislots), -1, dtype=int)
        coexistence_urllc_packet = np.full((num_uavs, num_subcarriers, num_minislots), -1, dtype=int)
        rho_tensor = np.zeros(
            (self.sys_cfg.num_embb_users, num_urllc, num_uavs, num_subcarriers, num_minislots),
            dtype=np.uint8
        )
        varpi_tensor = np.zeros_like(rho_tensor)
        rho_actions = []
        varpi_actions = []
        scheduled_count = 0
        packet_indices = list(range(num_packets))
        if selection_policy == 'normal_v2':
            priority_order = packet_indices
        elif associated_uavs is None:
            priority_order = packet_indices
        else:
            priority_order = sorted(
                packet_indices,
                key=lambda pkt_idx: float(np.max(
                    channel_gains_mag_sq[
                        packet_sources[pkt_idx],
                        int(associated_uavs[packet_sources[pkt_idx]]),
                        :
                    ]
                )),
                reverse=True
            )

        for packet_idx in priority_order:
            if admission_quota is not None and scheduled_count >= int(admission_quota):
                break
            action = self._find_best_urllc_action(
                packet_idx,
                packet_sources[packet_idx],
                channel_gains_mag_sq,
                associated_uavs,
                owner_per_rb,
                grid,
                scheduled_count,
                power_alloc,
                selected_uavs,
                packet_sources,
                selection_policy=selection_policy
            )
            if action is None:
                continue

            rb = action['rb']
            minislot = action['minislot']

            if (
                selection_policy == 'utility' and
                (not self.algo_cfg.force_urllc_immediate_service) and
                action['utility'] <= 0
            ):
                continue
            if action['error_prob'] > self.urllc_cfg.target_error_probability:
                continue

            uav_idx = action['uav']
            power_alloc[packet_idx, uav_idx] = action['power']
            urllc_rb_alloc[packet_idx, rb] = 1
            selected_uavs[packet_idx] = uav_idx
            reliability_achieved[packet_idx] = action['reliability']
            if grid[uav_idx, rb, minislot] >= 0:
                continue
            grid[uav_idx, rb, minislot] = packet_idx
            decisions[uav_idx, rb, minislot] = action['mode']
            coexistence_mode[uav_idx, rb, minislot] = action['mode']
            coexistence_embb[uav_idx, rb, minislot] = action['embb_owner']
            coexistence_urllc[uav_idx, rb, minislot] = packet_sources[packet_idx]
            coexistence_urllc_packet[uav_idx, rb, minislot] = packet_idx
            action_record = {
                'q': action['embb_owner'],
                'z': packet_sources[packet_idx],
                'packet_idx': packet_idx,
                'j': uav_idx,
                'k': rb,
                's': minislot,
                'mode': action['mode'],
                'base_rb_rate': float(action.get('base_rb_rate', 0.0)),
                'retained_fraction': float(action.get('retained_fraction', 0.0)),
                'embb_loss_per_action': float(action.get('embb_loss_per_action', 0.0))
            }
            if action['mode'] == 'NOMA':
                if action['embb_owner'] >= 0:
                    rho_tensor[action['embb_owner'], packet_sources[packet_idx], uav_idx, rb, minislot] = 1
                rho_actions.append(action_record)
                self.overlay_diagnostics['selected_overlay_pairs_total'] += 1
                self.overlay_diagnostics['selected_overlay_pairs_per_uav'][uav_idx] += 1
            else:
                if action['embb_owner'] >= 0:
                    varpi_tensor[action['embb_owner'], packet_sources[packet_idx], uav_idx, rb, minislot] = 1
                varpi_actions.append(action_record)
            scheduled_count += 1

        self.urllc_power_allocation = power_alloc
        self.urllc_rb_allocation = urllc_rb_alloc
        self.urllc_selected_uavs = selected_uavs
        self.urllc_packet_sources = packet_sources
        self.urllc_timefreq_grid = grid
        self.noma_decisions = decisions
        self.coexistence_mode_per_uav = coexistence_mode
        self.coexistence_embb_user_per_uav = coexistence_embb
        self.coexistence_urllc_user_per_uav = coexistence_urllc
        self.coexistence_urllc_packet_per_uav = coexistence_urllc_packet
        self.rho_tensor = rho_tensor
        self.varpi_tensor = varpi_tensor
        self.rho_action_list = rho_actions
        self.varpi_action_list = varpi_actions
        return power_alloc, reliability_achieved

    def create_urllc_timefreq_schedule(self, active_urllc_mask=None):
        """Return the already constructed URLLC time-frequency schedule."""
        # This is a simple accessor; allocation is done in allocate_urllc_power.
        if self.urllc_timefreq_grid is None:
            self.urllc_timefreq_grid = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
                -1,
                dtype=int
            )
        return self.urllc_timefreq_grid

    def decide_noma_puncturing(self, channel_gains_mag_sq, urllc_indices,
                               embb_indices, allocated_urllc_power,
                               urllc_timefreq_grid=None, owner_per_rb=None):
        """Return stored NOMA/puncturing decisions from the utility scheduler."""
        # Kept for API symmetry; decisions are computed inside allocate_urllc_power.
        if self.noma_decisions is None:
            self.noma_decisions = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
                'EMPTY',
                dtype='<U5'
            )
        if self.coexistence_mode_per_uav is None:
            self.coexistence_mode_per_uav = np.full(
                (self.sys_cfg.num_uavs, self.sys_cfg.num_subcarriers, self.sys_cfg.num_minislots),
                'EMPTY',
                dtype='<U5'
            )
        if self.rho_tensor is None:
            self.rho_tensor = np.zeros(
                (
                    self.sys_cfg.num_embb_users,
                    self.sys_cfg.num_urllc_users,
                    self.sys_cfg.num_uavs,
                    self.sys_cfg.num_subcarriers,
                    self.sys_cfg.num_minislots
                ),
                dtype=np.uint8
            )
        if self.varpi_tensor is None:
            self.varpi_tensor = np.zeros_like(self.rho_tensor)
        return self.noma_decisions

    def adjust_embb_after_urllc(self, embb_rb_alloc, channel_gains_mag_sq,
                                urllc_timefreq_grid=None, noma_decisions=None,
                                associated_uavs=None):
        """
        Recompute eMBB rates after URLLC overlay and refine transmit powers locally.
        """
        # Apply URLLC interference and (if overlay) SIC residuals, then re-evaluate rates.
        if associated_uavs is None:
            best_uav_per_user = np.zeros(self.sys_cfg.num_embb_users, dtype=int)
            for embb_idx in range(self.sys_cfg.num_embb_users):
                user_idx = self.sys_cfg.num_urllc_users + embb_idx
                avg_gain = np.mean(channel_gains_mag_sq[user_idx, :, :], axis=1)
                best_uav_per_user[embb_idx] = int(np.argmax(avg_gain))
        else:
            best_uav_per_user = np.array(
                associated_uavs[self.sys_cfg.num_urllc_users:self.sys_cfg.num_urllc_users + self.sys_cfg.num_embb_users],
                dtype=int
            )

        refined_powers = self._refine_embb_powers(
            embb_rb_alloc,
            channel_gains_mag_sq,
            best_uav_per_user,
            urllc_timefreq_grid,
            noma_decisions
        )

        final_state = self._compute_embb_state(
            embb_rb_alloc,
            channel_gains_mag_sq,
            best_uav_per_user,
            refined_powers,
            urllc_timefreq_grid=urllc_timefreq_grid,
            noma_decisions=noma_decisions,
            urllc_power_alloc=self.urllc_power_allocation
        )

        self.embb_power_allocation = final_state['power_allocation']
        self.rb_allocation = embb_rb_alloc
        self.embb_owner_per_rb = final_state['owner_per_rb']
        self.embb_selected_uavs = best_uav_per_user
        self.embb_user_tx_power = refined_powers

        return {
            'rb_allocation': embb_rb_alloc,
            'power_allocation': final_state['power_allocation'],
            'rates': final_state['rates'],
            'total_rate': np.sum(final_state['rates']),
            'owner_per_rb': final_state['owner_per_rb']
        }

    def _is_reliability_feasible_action(self, action):
        if action is None:
            return False
        return float(action.get('error_prob', np.inf)) <= float(self.urllc_cfg.target_error_probability)

    def _throughput_action_key(self, action):
        overlay_preference = 1.0 if action.get('mode') == 'NOMA' else 0.0
        return (
            -float(action.get('embb_loss_per_action', np.inf)),
            float(action.get('retained_fraction', 0.0)),
            -float(action.get('power', np.inf)),
            float(action.get('reliability', 0.0)),
            overlay_preference,
        )

    def _prefer_action(self, candidate, incumbent, selection_policy='utility'):
        if candidate is None:
            return incumbent
        if incumbent is None:
            return candidate
        if selection_policy == 'throughput_first':
            cand_ok = self._is_reliability_feasible_action(candidate)
            inc_ok = self._is_reliability_feasible_action(incumbent)
            if cand_ok != inc_ok:
                return candidate if cand_ok else incumbent
            if not cand_ok:
                return incumbent
            return candidate if self._throughput_action_key(candidate) > self._throughput_action_key(incumbent) else incumbent
        return candidate if float(candidate.get('utility', -np.inf)) > float(incumbent.get('utility', -np.inf)) else incumbent

    def _fixed_power_for_normal_greedy(self, max_power: float) -> float:
        """Use a single fixed URLLC power level for the normal-greedy baseline."""
        return float(max(max_power, 0.0))

    def _find_best_urllc_action(self, packet_idx, urllc_user_idx, channel_gains_mag_sq, associated_uavs,
                                owner_per_rb, grid, scheduled_count,
                                scheduled_urllc_power, scheduled_urllc_uavs, scheduled_packet_sources,
                                selection_policy='utility'):
        """Search the best utility-positive action for one URLLC packet."""
        # Evaluate puncture and overlay actions and keep the highest utility feasible option.
        packet_length = self.urllc_cfg.packet_lengths[urllc_user_idx % len(self.urllc_cfg.packet_lengths)]
        power_limit_idx = min(urllc_user_idx, len(self.urllc_cfg.power_limits) - 1)
        max_power_w = min(
            self._dbm_to_watts(self.urllc_cfg.power_limits[power_limit_idx]),
            self.algo_cfg.power_upper_bound
        )
        channel_uses = self.sys_cfg.channel_uses_per_minislot
        num_subcarriers = self.sys_cfg.num_subcarriers
        num_minislots = self.sys_cfg.num_minislots
        best_action = None

        normal_v2 = selection_policy == 'normal_v2'

        if associated_uavs is None:
            candidate_uavs = list(range(channel_gains_mag_sq.shape[1]))
        else:
            preferred_uav = int(associated_uavs[urllc_user_idx])
            if normal_v2:
                candidate_uavs = [preferred_uav]
            else:
                candidate_uavs = [preferred_uav]
                candidate_uavs.extend(
                    [u for u in range(channel_gains_mag_sq.shape[1]) if u != preferred_uav]
                )

        for uav_idx in candidate_uavs:
            for rb_idx in range(num_subcarriers):
                if np.all(grid[uav_idx, rb_idx, :] >= 0):
                    continue

                urllc_gain = channel_gains_mag_sq[urllc_user_idx, uav_idx, rb_idx]
                if urllc_gain <= 1e-12:
                    continue

                if np.ndim(owner_per_rb) == 2:
                    embb_owner = owner_per_rb[uav_idx, rb_idx]
                else:
                    embb_owner = owner_per_rb[rb_idx]
                embb_base_rate = 0.0
                embb_post_sic_ok = True
                embb_user_idx = None
                embb_interference = 0.0
                noma_allowed = embb_owner >= 0

                if embb_owner >= 0:
                    embb_user_idx = self.sys_cfg.num_urllc_users + embb_owner
                    embb_uav = self.embb_selected_uavs[embb_owner]
                    embb_gain = channel_gains_mag_sq[embb_user_idx, embb_uav, rb_idx]
                    embb_per_rb_power = self._get_embb_per_rb_power(embb_owner)
                    embb_interference = embb_per_rb_power * embb_gain
                    embb_snir = embb_interference / self.sys_cfg.noise_power
                    embb_base_rate = self.capacity_model.shannon_capacity(
                        embb_snir,
                        self.sys_cfg.subcarrier_bw
                    ) / num_minislots
                    gain_ratio = urllc_gain / max(embb_gain, 1e-12)
                    noma_allowed = gain_ratio >= self.algo_cfg.min_noma_gain_ratio
                else:
                    embb_base_rate = 0.0

                minislot_order = range(num_minislots) if normal_v2 else np.random.permutation(num_minislots)
                for minislot in minislot_order:
                    if grid[uav_idx, rb_idx, minislot] >= 0:
                        continue

                    packet_budget = max(1, int(np.ceil(self.algo_cfg.admission_load_limit * num_subcarriers)))
                    overload_ratio = (scheduled_count + 1) / packet_budget
                    overload_penalty = self.algo_cfg.overload_penalty_weight * overload_ratio

                    intercell_interference = self._compute_intercell_interference(
                        channel_gains_mag_sq,
                        serving_uav=uav_idx,
                        rb_idx=rb_idx,
                        minislot=minislot,
                        scheduled_urllc_power=scheduled_urllc_power,
                        scheduled_urllc_uavs=scheduled_urllc_uavs,
                        scheduled_packet_sources=scheduled_packet_sources,
                        local_mode='PUNCT',
                        local_embb_owner=embb_owner
                    )

                    if normal_v2:
                        punct_power = self._fixed_power_for_normal_greedy(max_power_w)
                    else:
                        punct_power = self._bisection_search_urllc_power(
                            urllc_gain,
                            packet_length,
                            self.urllc_cfg.target_error_probability,
                            max_power_w,
                            channel_uses,
                            interference_power=intercell_interference
                        )
                    punct_snir = punct_power * urllc_gain / (
                        self.sys_cfg.noise_power + intercell_interference
                    )
                    punct_error = self.capacity_model.decoding_error_probability(
                        punct_snir,
                        packet_length,
                        channel_uses
                    )
                    punct_reliability = 1 - punct_error
                    punct_utility = self._compute_action_utility(
                        embb_rate_delta=-1.6 * embb_base_rate,
                        urllc_reliability=punct_reliability,
                        power=punct_power,
                        overload_penalty=overload_penalty
                    )
                    punct_post_rate_ok = self._check_embb_rate_constraint(
                        embb_owner,
                        embb_base_rate,
                        surviving_fraction=0.0
                    )
                    if (
                        (not self.algo_cfg.force_urllc_immediate_service) and
                        embb_owner >= 0 and
                        not punct_post_rate_ok
                    ):
                        punct_utility = -np.inf

                    punct_action = {
                        'packet_idx': packet_idx,
                        'urllc_idx': urllc_user_idx,
                        'embb_owner': embb_owner,
                        'uav': uav_idx,
                        'rb': rb_idx,
                        'minislot': minislot,
                        'mode': 'PUNCT',
                        'power': punct_power,
                        'reliability': punct_reliability,
                        'error_prob': punct_error,
                        'utility': punct_utility,
                        'priority': punct_reliability / max(punct_power, 1e-12),
                        'base_rb_rate': embb_base_rate,
                        'retained_fraction': 0.0,
                        'embb_loss_per_action': embb_base_rate / max(num_minislots, 1)
                    }

                    best_action = self._prefer_action(
                        punct_action,
                        best_action,
                        selection_policy=selection_policy,
                    )

                    if not noma_allowed or embb_owner < 0:
                        continue

                    self.overlay_diagnostics['candidate_pairs_total'] += 1
                    self.overlay_diagnostics['candidate_pairs_per_uav'][uav_idx] += 1

                    noma_interference = intercell_interference + embb_interference
                    if normal_v2:
                        noma_power = self._fixed_power_for_normal_greedy(max_power_w)
                    else:
                        noma_power = self._bisection_search_urllc_power(
                            urllc_gain,
                            packet_length,
                            self.urllc_cfg.target_error_probability,
                            max_power_w,
                            channel_uses,
                            interference_power=noma_interference
                        )
                    noma_snir = noma_power * urllc_gain / (
                        self.sys_cfg.noise_power + noma_interference
                    )
                    noma_error = self.capacity_model.decoding_error_probability(
                        noma_snir,
                        packet_length,
                        channel_uses
                    )
                    noma_reliability = 1 - noma_error

                    embb_per_rb_power = self._get_embb_per_rb_power(embb_owner)
                    embb_gain = channel_gains_mag_sq[embb_user_idx, self.embb_selected_uavs[embb_owner], rb_idx]
                    embb_post_sic_snir = embb_per_rb_power * embb_gain / (
                        self.sys_cfg.noise_power +
                        intercell_interference +
                        self.algo_cfg.sic_residual_factor * noma_power * urllc_gain
                    )
                    min_embb_snir = 10 ** (self.algo_cfg.embb_min_sic_snir_db / 10)
                    embb_post_sic_ok = embb_post_sic_snir >= min_embb_snir
                    if not embb_post_sic_ok:
                        continue

                    if noma_reliability >= (1.0 - self.urllc_cfg.target_error_probability):
                        self.overlay_diagnostics['feasible_overlay_pairs_total'] += 1
                        self.overlay_diagnostics['feasible_overlay_pairs_per_uav'][uav_idx] += 1

                    retained_fraction = min(
                        0.95,
                        max(
                            self.algo_cfg.noma_retention_factor,
                            np.log2(1 + embb_post_sic_snir) / max(np.log2(1 + embb_per_rb_power * embb_gain / self.sys_cfg.noise_power), 1e-12)
                        )
                    )
                    noma_delta = -embb_base_rate * (1 - retained_fraction)
                    noma_post_rate_ok = self._check_embb_rate_constraint(
                        embb_owner,
                        embb_base_rate,
                        surviving_fraction=retained_fraction
                    )
                    if (not self.algo_cfg.force_urllc_immediate_service) and (not noma_post_rate_ok):
                        continue
                    noma_utility = self._compute_action_utility(
                        embb_rate_delta=noma_delta,
                        urllc_reliability=noma_reliability,
                        power=noma_power,
                        overload_penalty=overload_penalty
                    ) + 0.15 * embb_base_rate / 1e6
                    if (not normal_v2) and noma_reliability >= punct_reliability * 0.98 and noma_utility >= punct_utility - 0.35:
                        noma_utility += 0.5
                    noma_action = {
                        'packet_idx': packet_idx,
                        'urllc_idx': urllc_user_idx,
                        'embb_owner': embb_owner,
                        'uav': uav_idx,
                        'rb': rb_idx,
                        'minislot': minislot,
                        'mode': 'NOMA',
                        'power': noma_power,
                        'reliability': noma_reliability,
                        'error_prob': noma_error,
                        'utility': noma_utility,
                        'priority': (1.2 * noma_reliability) / max(noma_power, 1e-12),
                        'base_rb_rate': embb_base_rate,
                        'retained_fraction': retained_fraction,
                        'embb_loss_per_action': embb_base_rate * (1.0 - retained_fraction) / max(num_minislots, 1)
                    }
                    best_action = self._prefer_action(
                        noma_action,
                        best_action,
                        selection_policy=selection_policy,
                    )

        return best_action

    def _compute_action_utility(self, embb_rate_delta, urllc_reliability,
                                power, overload_penalty):
        """Marginal utility balancing eMBB loss, URLLC success, and power cost."""
        # Utility = URLLC success gain - eMBB loss - power cost - overload penalty.
        target = 1.0 - self.urllc_cfg.target_error_probability
        margin = (urllc_reliability - target) / max(target, 1e-12)
        if urllc_reliability >= target:
            urllc_term = self.algo_cfg.urllc_utility_weight * (1.0 + 0.1 * margin)
        else:
            urllc_term = -self.algo_cfg.urllc_utility_weight * (1.0 + 4.0 * abs(margin))

        embb_term = embb_rate_delta / 1e6
        power_term = self.algo_cfg.power_penalty_weight * power
        if overload_penalty > 0 and urllc_reliability < target:
            overload_penalty *= 1.5

        return embb_term + urllc_term - power_term - overload_penalty

    def _compute_intercell_interference(self, channel_gains_mag_sq, serving_uav, rb_idx, minislot,
                                        scheduled_urllc_power, scheduled_urllc_uavs, scheduled_packet_sources,
                                        local_mode='PUNCT', local_embb_owner=-1):
        """
        Aggregate inter-cell interference on a reused RB from all other UAVs.
        """
        # Sum eMBB and URLLC interference from other UAVs sharing this RB/minislot.
        total_interference = 0.0

        if self.embb_owner_per_uav_rb is None:
            return total_interference

        for other_uav in range(self.sys_cfg.num_uavs):
            if other_uav == serving_uav:
                continue

            embb_owner = self.embb_owner_per_uav_rb[other_uav, rb_idx]
            if embb_owner >= 0:
                other_mode = 'EMPTY'
                if self.coexistence_mode_per_uav is not None:
                    other_mode = self.coexistence_mode_per_uav[other_uav, rb_idx, minislot]
                embb_is_active = other_mode != 'PUNCT'
                if embb_is_active:
                    if embb_owner >= self.sys_cfg.num_embb_users:
                        continue
                    embb_user_idx = self.sys_cfg.num_urllc_users + embb_owner
                    embb_per_rb_power = self._get_embb_per_rb_power(embb_owner)
                    embb_cross_gain = channel_gains_mag_sq[embb_user_idx, serving_uav, rb_idx]
                    total_interference += embb_per_rb_power * embb_cross_gain

            if self.coexistence_urllc_packet_per_uav is not None:
                other_packet = self.coexistence_urllc_packet_per_uav[other_uav, rb_idx, minislot]
                if (
                    other_packet >= 0 and
                    scheduled_packet_sources is not None and
                    other_packet < len(scheduled_packet_sources) and
                    other_packet < scheduled_urllc_power.shape[0] and
                    scheduled_urllc_uavs[other_packet] == other_uav
                ):
                    other_urllc_user = scheduled_packet_sources[other_packet]
                    urllc_cross_gain = channel_gains_mag_sq[other_urllc_user, serving_uav, rb_idx]
                    total_interference += scheduled_urllc_power[other_packet, other_uav] * urllc_cross_gain

        return total_interference

    def _get_embb_min_rate_requirement(self):
        """Return the minimum required eMBB user rate in bps."""
        # Used to enforce per-user QoS constraints.
        if self.embb_cfg.min_rate_per_user_bps is not None:
            return float(self.embb_cfg.min_rate_per_user_bps)
        return float(self.embb_cfg.target_spectral_efficiency * self.sys_cfg.subcarrier_bw)

    def _estimate_embb_baseline_rate(self, embb_idx):
        """Estimate current baseline eMBB rate from stored RB rates."""
        # Sum the baseline RB rates allocated to this user.
        if self.rb_allocation is None or self.embb_base_rb_rates is None:
            return 0.0
        if self.embb_owner_per_uav_rb is not None and self.embb_base_rb_rates_per_uav_rb is not None:
            total = 0.0
            for uav_idx in range(self.embb_owner_per_uav_rb.shape[0]):
                assigned_rbs = np.where(self.embb_owner_per_uav_rb[uav_idx, :] == embb_idx)[0]
                if assigned_rbs.size == 0:
                    continue
                total += float(np.sum(self.embb_base_rb_rates_per_uav_rb[uav_idx, assigned_rbs]))
            return float(total)
        assigned_rbs = np.where(self.rb_allocation[embb_idx, :] == 1)[0]
        if assigned_rbs.size == 0:
            return 0.0
        return float(np.sum(self.embb_base_rb_rates[assigned_rbs]))

    def _check_embb_rate_constraint(self, embb_idx, affected_rb_rate, surviving_fraction):
        """Check whether a local coexistence action keeps the eMBB user above the minimum rate."""
        # Estimate post-action rate and compare against the min-rate requirement.
        if embb_idx < 0:
            return True
        current_rate = self._estimate_embb_baseline_rate(embb_idx)
        post_rate = current_rate - affected_rb_rate + affected_rb_rate * surviving_fraction
        return post_rate >= self._get_embb_min_rate_requirement()

    def _refine_embb_powers(self, embb_rb_alloc, channel_gains_mag_sq, best_uav_per_user,
                            urllc_timefreq_grid, noma_decisions):
        """Simple local search on eMBB transmit powers."""
        # Lightweight coordinate search to improve eMBB rate after URLLC actions.
        num_embb = self.sys_cfg.num_embb_users
        refined = self.embb_user_tx_power.copy()
        if refined.size != num_embb:
            refined = np.zeros(num_embb)

        step = self.algo_cfg.power_refine_step
        for _ in range(self.algo_cfg.power_refine_iterations):
            for embb_idx in range(num_embb):
                power_limit_idx = min(embb_idx, len(self.embb_cfg.power_limits) - 1)
                max_power = min(
                    self._dbm_to_watts(self.embb_cfg.power_limits[power_limit_idx]),
                    self.algo_cfg.power_upper_bound
                )
                candidates = [
                    max(0.0, refined[embb_idx] * (1.0 - step)),
                    refined[embb_idx],
                    min(max_power, refined[embb_idx] * (1.0 + step) if refined[embb_idx] > 0 else max_power * 0.3)
                ]
                best_power = refined[embb_idx]
                best_score = -np.inf
                for candidate_power in candidates:
                    trial = refined.copy()
                    trial[embb_idx] = candidate_power
                    trial_state = self._compute_embb_state(
                        embb_rb_alloc,
                        channel_gains_mag_sq,
                        best_uav_per_user,
                        trial,
                        urllc_timefreq_grid=urllc_timefreq_grid,
                        noma_decisions=noma_decisions,
                        urllc_power_alloc=self.urllc_power_allocation
                    )
                    score = (
                        trial_state['rates'][embb_idx] / 1e6
                        - self.algo_cfg.power_penalty_weight * candidate_power
                    )
                    if score > best_score:
                        best_score = score
                        best_power = candidate_power
                refined[embb_idx] = best_power
        return refined

    def _compute_embb_state(self, embb_rb_alloc, channel_gains_mag_sq, best_uav_per_user,
                            embb_tx_powers, urllc_timefreq_grid=None, noma_decisions=None,
                            urllc_power_alloc=None):
        """Compute eMBB rates and effective power for a given allocation state."""
        # Evaluates base rates, then applies survival fractions under URLLC coexistence.
        num_embb = self.sys_cfg.num_embb_users
        num_uavs = channel_gains_mag_sq.shape[1]
        num_subcarriers = channel_gains_mag_sq.shape[2]
        num_minislots = self.sys_cfg.num_minislots

        embb_power_alloc = np.zeros((num_embb, num_uavs))
        embb_rates = np.zeros(num_embb)
        owner_per_rb = np.full(num_subcarriers, -1, dtype=int)
        base_rb_rates = np.zeros(num_subcarriers)
        base_rb_rates_per_uav_rb = np.zeros((num_uavs, num_subcarriers))

        for embb_idx in range(num_embb):
            assigned_rbs = np.where(embb_rb_alloc[embb_idx, :] == 1)[0]
            if assigned_rbs.size == 0:
                continue
            best_uav = best_uav_per_user[embb_idx]
            user_idx = self.sys_cfg.num_urllc_users + embb_idx
            tx_power = embb_tx_powers[embb_idx]
            per_rb_power = tx_power / max(assigned_rbs.size, 1)

            total_survival = 0.0
            for rb in assigned_rbs:
                owner_per_rb[rb] = embb_idx
                ch_sq = channel_gains_mag_sq[user_idx, best_uav, rb]
                base_interference = self._compute_intercell_interference(
                    channel_gains_mag_sq,
                    serving_uav=best_uav,
                    rb_idx=rb,
                    minislot=0,
                    scheduled_urllc_power=urllc_power_alloc if urllc_power_alloc is not None else np.zeros((self.sys_cfg.num_urllc_users, self.sys_cfg.num_uavs)),
                    scheduled_urllc_uavs=self.urllc_selected_uavs if self.urllc_selected_uavs is not None else np.full(self.sys_cfg.num_urllc_users, -1, dtype=int),
                    scheduled_packet_sources=self.urllc_packet_sources,
                    local_mode='OMA',
                    local_embb_owner=embb_idx
                )
                base_snir = per_rb_power * ch_sq / (self.sys_cfg.noise_power + base_interference)
                base_rate = self.capacity_model.shannon_capacity(base_snir, self.sys_cfg.subcarrier_bw)
                base_rb_rates[rb] += base_rate
                base_rb_rates_per_uav_rb[best_uav, rb] = base_rate

                surviving_fraction = 1.0
                if urllc_timefreq_grid is not None:
                    if np.ndim(urllc_timefreq_grid) == 3:
                        local_grid = urllc_timefreq_grid[best_uav, rb, :]
                    else:
                        local_grid = urllc_timefreq_grid[rb, :]
                else:
                    local_grid = None

                if local_grid is not None and np.any(local_grid >= 0):
                    surviving_fraction = 0.0
                    for minislot in range(num_minislots):
                        urllc_idx = local_grid[minislot]
                        if urllc_idx < 0:
                            surviving_fraction += 1.0 / num_minislots
                            continue
                        mode = 'PUNCT'
                        if noma_decisions is not None:
                            if np.ndim(noma_decisions) == 3:
                                mode = noma_decisions[best_uav, rb, minislot]
                            else:
                                mode = noma_decisions[minislot, rb]
                        if mode == 'NOMA':
                            retained = self.algo_cfg.noma_retention_factor
                            if urllc_power_alloc is not None and self.urllc_selected_uavs is not None:
                                if (
                                    self.urllc_packet_sources is None or
                                    urllc_idx < 0 or
                                    urllc_idx >= len(self.urllc_packet_sources)
                                ):
                                    urllc_uav = -1
                                    urllc_user = -1
                                else:
                                    urllc_uav = int(self.urllc_selected_uavs[urllc_idx]) if urllc_idx < len(self.urllc_selected_uavs) else -1
                                    urllc_user = int(self.urllc_packet_sources[urllc_idx])
                                if (
                                    urllc_uav >= 0 and
                                    urllc_user >= 0 and
                                    urllc_user < self.sys_cfg.num_urllc_users and
                                    urllc_idx < urllc_power_alloc.shape[0]
                                ):
                                    urllc_power = urllc_power_alloc[urllc_idx, urllc_uav]
                                    urllc_gain = channel_gains_mag_sq[urllc_user, urllc_uav, rb]
                                    intercell_interference = self._compute_intercell_interference(
                                        channel_gains_mag_sq,
                                        serving_uav=best_uav,
                                        rb_idx=rb,
                                        minislot=minislot,
                                        scheduled_urllc_power=urllc_power_alloc,
                                        scheduled_urllc_uavs=self.urllc_selected_uavs,
                                        scheduled_packet_sources=self.urllc_packet_sources,
                                        local_mode='NOMA',
                                        local_embb_owner=embb_idx
                                    )
                                    post_sic_snir = per_rb_power * ch_sq / (
                                        self.sys_cfg.noise_power +
                                        intercell_interference +
                                        self.algo_cfg.sic_residual_factor * urllc_power * urllc_gain
                                    )
                                    retained = min(
                                        0.95,
                                        max(
                                            self.algo_cfg.noma_retention_factor,
                                            np.log2(1 + post_sic_snir) / max(np.log2(1 + base_snir), 1e-12)
                                        )
                                    )
                            surviving_fraction += retained / num_minislots
                        else:
                            surviving_fraction += 0.0

                embb_rates[embb_idx] += base_rate * surviving_fraction
                total_survival += surviving_fraction

            avg_survival = total_survival / assigned_rbs.size
            embb_power_alloc[embb_idx, best_uav] = tx_power * avg_survival

        return {
            'power_allocation': embb_power_alloc,
            'rates': embb_rates,
            'owner_per_rb': owner_per_rb,
            'base_rb_rates': base_rb_rates,
            'base_rb_rates_per_uav_rb': base_rb_rates_per_uav_rb,
        }

    def _bisection_search_urllc_power(self, channel_gain_sq, packet_bits,
                                      target_error_prob, max_power, channel_uses,
                                      interference_power=0.0):
        """Bisection search for minimum URLLC power under optional interference."""
        # Iteratively find the minimum power that meets the reliability target.
        lower_power = 0.0
        upper_power = max_power
        denom = self.sys_cfg.noise_power + max(interference_power, 0.0)

        for _ in range(self.algo_cfg.bisection_max_iterations):
            mid_power = (lower_power + upper_power) / 2
            snir = mid_power * channel_gain_sq / max(denom, 1e-15)
            error_prob = self.capacity_model.decoding_error_probability(
                snir, packet_bits, channel_uses
            )
            if error_prob > target_error_prob:
                lower_power = mid_power
            else:
                upper_power = mid_power
            if (upper_power - lower_power) < self.algo_cfg.bisection_tolerance:
                break

        return (lower_power + upper_power) / 2

    def _get_embb_per_rb_power(self, embb_idx):
        """Per-RB eMBB power for the currently assigned RB count."""
        # Power is split equally across the assigned RBs for the user.
        if self.rb_allocation is None or embb_idx < 0:
            return 0.0
        assigned = int(np.sum(self.rb_allocation[embb_idx, :]))
        if assigned == 0:
            return 0.0
        if self.embb_user_tx_power.size <= embb_idx:
            return 0.0
        return self.embb_user_tx_power[embb_idx] / assigned

    def _ensure_embb_state(self, num_embb):
        """Resize eMBB state vectors if user count changes."""
        # Prevents stale buffers when the number of users changes.
        if self.embb_user_tx_power.size != num_embb:
            self.embb_user_tx_power = np.zeros(num_embb, dtype=float)

    def _dbm_to_watts(self, power_dbm):
        """Convert dBm to Watts."""
        # Standard unit conversion helper.
        return 10 ** ((power_dbm - 30) / 10)

    def _watts_to_dbm(self, power_w):
        """Convert Watts to dBm."""
        # Standard unit conversion helper.
        return 30 + 10 * np.log10(power_w)

    def get_allocation_summary(self):
        """Return allocation results summary."""
        # Used by simulation/reporting to export the full allocation state.
        total_users = self.sys_cfg.num_embb_users + self.sys_cfg.num_urllc_users
        association = np.zeros((total_users, self.sys_cfg.num_uavs), dtype=int)
        if self.urllc_selected_uavs is not None and len(self.urllc_selected_uavs) == self.sys_cfg.num_urllc_users:
            for user_idx, uav_idx in enumerate(self.urllc_selected_uavs):
                if 0 <= uav_idx < self.sys_cfg.num_uavs:
                    association[user_idx, uav_idx] = 1
        if self.embb_selected_uavs is not None:
            for embb_idx, uav_idx in enumerate(self.embb_selected_uavs):
                if 0 <= uav_idx < self.sys_cfg.num_uavs:
                    association[self.sys_cfg.num_urllc_users + embb_idx, uav_idx] = 1
        self.user_association = association

        return {
            'urllc_power': self.urllc_power_allocation,
            'embb_power': self.embb_power_allocation,
            'embb_rbs': self.rb_allocation,
            'alpha_e': self.alpha_e_allocation,
            'urllc_rbs': self.urllc_rb_allocation,
            'urllc_timefreq_grid': self.urllc_timefreq_grid,
            'embb_owner_per_rb': self.embb_owner_per_rb,
            'embb_owner_per_uav_rb': self.embb_owner_per_uav_rb,
            'noma_decisions': self.noma_decisions,
            'coexistence_mode_per_uav': self.coexistence_mode_per_uav,
            'coexistence_embb_user_per_uav': self.coexistence_embb_user_per_uav,
            'coexistence_urllc_user_per_uav': self.coexistence_urllc_user_per_uav,
            'coexistence_urllc_packet_per_uav': self.coexistence_urllc_packet_per_uav,
            'rho_tensor': self.rho_tensor,
            'varpi_tensor': self.varpi_tensor,
            'rho_action_list': self.rho_action_list,
            'varpi_action_list': self.varpi_action_list,
            'overlay_diagnostics': self.overlay_diagnostics,
            'user_association': association,
            'embb_selected_uavs': self.embb_selected_uavs,
            'urllc_selected_uavs': self.urllc_selected_uavs,
            'urllc_packet_sources': self.urllc_packet_sources
        }


def create_allocator(sys_cfg, urllc_cfg, embb_cfg, algo_cfg):
    """Factory function."""
    return ResourceAllocator(sys_cfg, urllc_cfg, embb_cfg, algo_cfg)
