"""Action masking and feasibility shielding for SR-MAPPO."""

from typing import Dict, Iterable, Optional, Set

import numpy as np

from .config import SRMAPPOConfig
from .types import MODE_KEEP, MODE_OVERLAY, AgentObservation, HybridAction, ShieldedAction


class FeasibilityShield:
    """Mask invalid actions and fall back to safe greedy-like behavior."""

    def __init__(self, cfg: SRMAPPOConfig):
        self.cfg = cfg

    def sanitize_action(self, action: HybridAction, obs: AgentObservation) -> ShieldedAction:
        """Project one agent action back into the feasible region."""
        mode = int(action.mode)
        packet_option = int(action.packet_option)
        power_delta = float(np.clip(action.power_delta, -1.0, 1.0))
        embb_owner_option = int(action.embb_owner_option)
        embb_power_delta = float(np.clip(action.embb_power_delta, -1.0, 1.0))

        if mode < 0 or mode >= obs.masks.mode_mask.size:
            mode = MODE_KEEP
        if packet_option < 0 or packet_option >= obs.masks.packet_mask.shape[-1]:
            packet_option = 0

        if obs.masks.mode_mask[mode] <= 0:
            fallback = self._fallback(
                obs,
                embb_owner_option=embb_owner_option,
                embb_power_delta=embb_power_delta,
            )
            fallback.mask_invalid_fallback = True
            return fallback

        if mode == MODE_KEEP:
            return ShieldedAction(
                action=HybridAction(
                    mode=MODE_KEEP,
                    packet_option=0,
                    power_delta=0.0,
                    embb_owner_option=embb_owner_option,
                    embb_power_delta=embb_power_delta,
                ),
                candidate=None,
                utility=0.0,
            )

        valid_packet_mask = obs.masks.packet_mask[mode]
        if packet_option == 0 or valid_packet_mask[packet_option] <= 0:
            fallback = self._fallback(
                obs,
                embb_owner_option=embb_owner_option,
                embb_power_delta=embb_power_delta,
            )
            fallback.packet_invalid_fallback = True
            return fallback

        candidate = obs.candidates[packet_option - 1]
        mode_corrected = False
        if not candidate.is_mode_feasible(mode):
            if self.cfg.shield.allow_mode_correction and candidate.is_mode_feasible(candidate.best_mode):
                mode = candidate.best_mode
                mode_corrected = True
            else:
                fallback = self._fallback(
                    obs,
                    embb_owner_option=embb_owner_option,
                    embb_power_delta=embb_power_delta,
                )
                fallback.mode_corrected = True
                return fallback

        if (
            self.cfg.shield.force_overlay_when_better
            and mode != MODE_OVERLAY
            and candidate.overlay_feasible
        ):
            overlay_margin = float(candidate.overlay_utility - candidate.puncture_utility)
            if overlay_margin >= float(self.cfg.shield.force_overlay_utility_margin):
                mode = MODE_OVERLAY
                mode_corrected = True

        return ShieldedAction(
            action=HybridAction(
                mode=mode,
                packet_option=packet_option,
                power_delta=power_delta,
                embb_owner_option=embb_owner_option,
                embb_power_delta=embb_power_delta,
            ),
            candidate=candidate,
            utility=candidate.utility_for_mode(mode),
            mode_corrected=mode_corrected,
        )

    def resolve_collisions(
        self,
        shielded_actions: Dict[str, ShieldedAction],
        observations: Dict[str, AgentObservation],
    ) -> Dict[str, ShieldedAction]:
        """Ensure the same URLLC packet is not scheduled by two UAVs at once."""
        if not self.cfg.shield.resolve_packet_collisions:
            return shielded_actions

        winners_by_packet = {}
        losers = []
        for agent_id, shielded in shielded_actions.items():
            if shielded.candidate is None:
                continue
            packet_id = shielded.candidate.packet_id
            prev = winners_by_packet.get(packet_id)
            if prev is None or shielded.utility > prev[1].utility:
                if prev is not None:
                    losers.append(prev[0])
                winners_by_packet[packet_id] = (agent_id, shielded)
            else:
                losers.append(agent_id)

        if not losers:
            return shielded_actions

        reserved_packets: Set[int] = {
            pair[1].candidate.packet_id
            for pair in winners_by_packet.values()
            if pair[1].candidate is not None
        }
        resolved = dict(shielded_actions)
        for agent_id in losers:
            replacement = self._fallback(observations[agent_id], forbidden_packets=reserved_packets)
            replacement.collision_rewritten = True
            if replacement.candidate is not None:
                reserved_packets.add(replacement.candidate.packet_id)
            resolved[agent_id] = replacement

        return resolved

    def _fallback(
        self,
        obs: AgentObservation,
        forbidden_packets: Optional[Iterable[int]] = None,
        embb_owner_option: int = 0,
        embb_power_delta: float = 0.0,
    ) -> ShieldedAction:
        """Greedy fallback or keep action when no feasible action remains."""
        blocked = set(forbidden_packets or [])
        if self.cfg.shield.enable_greedy_fallback:
            for option_idx, candidate in enumerate(obs.candidates, start=1):
                if candidate.packet_id in blocked:
                    continue
                if candidate.is_mode_feasible(candidate.best_mode):
                    return ShieldedAction(
                        action=HybridAction(
                            mode=candidate.best_mode,
                            packet_option=option_idx,
                            power_delta=0.0,
                            embb_owner_option=embb_owner_option,
                            embb_power_delta=embb_power_delta,
                        ),
                        candidate=candidate,
                        utility=candidate.best_utility,
                        used_greedy_fallback=True,
                    )

        return ShieldedAction(
            action=HybridAction(
                mode=MODE_KEEP,
                packet_option=0,
                power_delta=0.0,
                embb_owner_option=embb_owner_option,
                embb_power_delta=embb_power_delta,
            ),
            candidate=None,
            utility=0.0,
            used_greedy_fallback=True,
        )
