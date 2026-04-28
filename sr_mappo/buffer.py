"""Rollout storage for shared-reward MAPPO training."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class RolloutBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    mode_mask: torch.Tensor
    packet_mask: torch.Tensor
    embb_owner_mask: torch.Tensor
    mode_actions: torch.Tensor
    packet_actions: torch.Tensor
    power_pre_tanh: torch.Tensor
    power_delta: torch.Tensor
    embb_owner_actions: torch.Tensor
    embb_power_pre_tanh: torch.Tensor
    embb_power_delta: torch.Tensor
    old_log_prob: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    actor_hidden: torch.Tensor
    critic_hidden: torch.Tensor
    aux_best_mode_target: torch.Tensor
    aux_overlay_feasible_target: torch.Tensor
    aux_best_packet_target: torch.Tensor
    teacher_admission_target: torch.Tensor
    teacher_mode_target: torch.Tensor
    teacher_admission_weight: torch.Tensor
    teacher_mode_weight: torch.Tensor
    greedy_bc_mode_target: torch.Tensor
    greedy_bc_packet_target: torch.Tensor
    greedy_bc_owner_target: torch.Tensor
    greedy_bc_mode_weight: torch.Tensor
    greedy_bc_packet_weight: torch.Tensor
    greedy_bc_owner_weight: torch.Tensor
    phase_a_embb_power_anchor_target: torch.Tensor
    phase_a_embb_power_anchor_weight: torch.Tensor
    phase_a_mask: torch.Tensor


class SharedRolloutBuffer:
    """Single-environment rollout buffer with per-agent storage."""

    def __init__(self, num_agents: int, local_obs_dim: int, global_obs_dim: int, hidden_dim: int):
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self) -> None:
        self.local_obs = []
        self.global_obs = []
        self.mode_mask = []
        self.packet_mask = []
        self.embb_owner_mask = []
        self.mode_actions = []
        self.packet_actions = []
        self.power_pre_tanh = []
        self.power_delta = []
        self.embb_owner_actions = []
        self.embb_power_pre_tanh = []
        self.embb_power_delta = []
        self.old_log_prob = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.actor_hidden = []
        self.critic_hidden = []
        self.aux_best_mode_target = []
        self.aux_overlay_feasible_target = []
        self.aux_best_packet_target = []
        self.teacher_admission_target = []
        self.teacher_mode_target = []
        self.teacher_admission_weight = []
        self.teacher_mode_weight = []
        self.greedy_bc_mode_target = []
        self.greedy_bc_packet_target = []
        self.greedy_bc_owner_target = []
        self.greedy_bc_mode_weight = []
        self.greedy_bc_packet_weight = []
        self.greedy_bc_owner_weight = []
        self.phase_a_embb_power_anchor_target = []
        self.phase_a_embb_power_anchor_weight = []
        self.phase_a_mask = []
        self.returns = None
        self.advantages = None

    def add_step(
        self,
        *,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        mode_mask: np.ndarray,
        packet_mask: np.ndarray,
        embb_owner_mask: np.ndarray,
        mode_actions: np.ndarray,
        packet_actions: np.ndarray,
        power_pre_tanh: np.ndarray,
        power_delta: np.ndarray,
        embb_owner_actions: np.ndarray,
        embb_power_pre_tanh: np.ndarray,
        embb_power_delta: np.ndarray,
        old_log_prob: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        actor_hidden: np.ndarray,
        critic_hidden: np.ndarray,
        aux_best_mode_target: np.ndarray,
        aux_overlay_feasible_target: np.ndarray,
        aux_best_packet_target: np.ndarray,
        teacher_admission_target: np.ndarray,
        teacher_mode_target: np.ndarray,
        teacher_admission_weight: np.ndarray,
        teacher_mode_weight: np.ndarray,
        greedy_bc_mode_target: np.ndarray,
        greedy_bc_packet_target: np.ndarray,
        greedy_bc_owner_target: np.ndarray,
        greedy_bc_mode_weight: np.ndarray,
        greedy_bc_packet_weight: np.ndarray,
        greedy_bc_owner_weight: np.ndarray,
        phase_a_embb_power_anchor_target: np.ndarray,
        phase_a_embb_power_anchor_weight: np.ndarray,
        phase_a_mask: np.ndarray,
    ) -> None:
        self.local_obs.append(np.asarray(local_obs, dtype=np.float32))
        self.global_obs.append(np.asarray(global_obs, dtype=np.float32))
        self.mode_mask.append(np.asarray(mode_mask, dtype=np.float32))
        self.packet_mask.append(np.asarray(packet_mask, dtype=np.float32))
        self.embb_owner_mask.append(np.asarray(embb_owner_mask, dtype=np.float32))
        self.mode_actions.append(np.asarray(mode_actions, dtype=np.int64))
        self.packet_actions.append(np.asarray(packet_actions, dtype=np.int64))
        self.power_pre_tanh.append(np.asarray(power_pre_tanh, dtype=np.float32).reshape(self.num_agents, 1))
        self.power_delta.append(np.asarray(power_delta, dtype=np.float32).reshape(self.num_agents, 1))
        self.embb_owner_actions.append(np.asarray(embb_owner_actions, dtype=np.int64))
        self.embb_power_pre_tanh.append(np.asarray(embb_power_pre_tanh, dtype=np.float32).reshape(self.num_agents, 1))
        self.embb_power_delta.append(np.asarray(embb_power_delta, dtype=np.float32).reshape(self.num_agents, 1))
        self.old_log_prob.append(np.asarray(old_log_prob, dtype=np.float32))
        self.values.append(np.asarray(values, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))
        self.actor_hidden.append(np.asarray(actor_hidden, dtype=np.float32))
        self.critic_hidden.append(np.asarray(critic_hidden, dtype=np.float32))
        self.aux_best_mode_target.append(np.asarray(aux_best_mode_target, dtype=np.int64))
        self.aux_overlay_feasible_target.append(np.asarray(aux_overlay_feasible_target, dtype=np.float32))
        self.aux_best_packet_target.append(np.asarray(aux_best_packet_target, dtype=np.int64))
        self.teacher_admission_target.append(np.asarray(teacher_admission_target, dtype=np.int64))
        self.teacher_mode_target.append(np.asarray(teacher_mode_target, dtype=np.int64))
        self.teacher_admission_weight.append(np.asarray(teacher_admission_weight, dtype=np.float32))
        self.teacher_mode_weight.append(np.asarray(teacher_mode_weight, dtype=np.float32))
        self.greedy_bc_mode_target.append(np.asarray(greedy_bc_mode_target, dtype=np.int64))
        self.greedy_bc_packet_target.append(np.asarray(greedy_bc_packet_target, dtype=np.int64))
        self.greedy_bc_owner_target.append(np.asarray(greedy_bc_owner_target, dtype=np.int64))
        self.greedy_bc_mode_weight.append(np.asarray(greedy_bc_mode_weight, dtype=np.float32))
        self.greedy_bc_packet_weight.append(np.asarray(greedy_bc_packet_weight, dtype=np.float32))
        self.greedy_bc_owner_weight.append(np.asarray(greedy_bc_owner_weight, dtype=np.float32))
        self.phase_a_embb_power_anchor_target.append(np.asarray(phase_a_embb_power_anchor_target, dtype=np.float32))
        self.phase_a_embb_power_anchor_weight.append(np.asarray(phase_a_embb_power_anchor_weight, dtype=np.float32))
        self.phase_a_mask.append(np.asarray(phase_a_mask, dtype=np.float32))

    @property
    def num_steps(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        last_values = np.asarray(last_values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        self.advantages = advantages
        self.returns = returns

    def as_torch(self, device: torch.device) -> RolloutBatch:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before converting the buffer.")

        local_obs = torch.from_numpy(np.asarray(self.local_obs, dtype=np.float32)).reshape(-1, self.local_obs_dim).to(device)
        global_obs = torch.from_numpy(np.asarray(self.global_obs, dtype=np.float32)).reshape(-1, self.global_obs_dim).to(device)
        mode_mask = torch.from_numpy(np.asarray(self.mode_mask, dtype=np.float32)).reshape(-1, np.asarray(self.mode_mask[0]).shape[-1]).to(device)
        packet_mask_shape = np.asarray(self.packet_mask[0]).shape[-2:]
        packet_mask = torch.from_numpy(np.asarray(self.packet_mask, dtype=np.float32)).reshape(-1, *packet_mask_shape).to(device)
        embb_owner_mask = torch.from_numpy(np.asarray(self.embb_owner_mask, dtype=np.float32)).reshape(-1, np.asarray(self.embb_owner_mask[0]).shape[-1]).to(device)
        mode_actions = torch.from_numpy(np.asarray(self.mode_actions, dtype=np.int64)).reshape(-1).to(device)
        packet_actions = torch.from_numpy(np.asarray(self.packet_actions, dtype=np.int64)).reshape(-1).to(device)
        power_pre_tanh = torch.from_numpy(np.asarray(self.power_pre_tanh, dtype=np.float32)).reshape(-1, 1).to(device)
        power_delta = torch.from_numpy(np.asarray(self.power_delta, dtype=np.float32)).reshape(-1, 1).to(device)
        embb_owner_actions = torch.from_numpy(np.asarray(self.embb_owner_actions, dtype=np.int64)).reshape(-1).to(device)
        embb_power_pre_tanh = torch.from_numpy(np.asarray(self.embb_power_pre_tanh, dtype=np.float32)).reshape(-1, 1).to(device)
        embb_power_delta = torch.from_numpy(np.asarray(self.embb_power_delta, dtype=np.float32)).reshape(-1, 1).to(device)
        old_log_prob = torch.from_numpy(np.asarray(self.old_log_prob, dtype=np.float32)).reshape(-1).to(device)
        values = torch.from_numpy(np.asarray(self.values, dtype=np.float32)).reshape(-1).to(device)
        returns = torch.from_numpy(np.asarray(self.returns, dtype=np.float32)).reshape(-1).to(device)
        advantages = torch.from_numpy(np.asarray(self.advantages, dtype=np.float32)).reshape(-1).to(device)
        actor_hidden = torch.from_numpy(np.asarray(self.actor_hidden, dtype=np.float32)).reshape(-1, self.hidden_dim).to(device)
        critic_hidden = torch.from_numpy(np.asarray(self.critic_hidden, dtype=np.float32)).reshape(-1, self.hidden_dim).to(device)
        aux_best_mode_target = torch.from_numpy(np.asarray(self.aux_best_mode_target, dtype=np.int64)).reshape(-1).to(device)
        aux_overlay_feasible_target = torch.from_numpy(np.asarray(self.aux_overlay_feasible_target, dtype=np.float32)).reshape(-1).to(device)
        aux_best_packet_target = torch.from_numpy(np.asarray(self.aux_best_packet_target, dtype=np.int64)).reshape(-1).to(device)
        teacher_admission_target = torch.from_numpy(np.asarray(self.teacher_admission_target, dtype=np.int64)).reshape(-1).to(device)
        teacher_mode_target = torch.from_numpy(np.asarray(self.teacher_mode_target, dtype=np.int64)).reshape(-1).to(device)
        teacher_admission_weight = torch.from_numpy(np.asarray(self.teacher_admission_weight, dtype=np.float32)).reshape(-1).to(device)
        teacher_mode_weight = torch.from_numpy(np.asarray(self.teacher_mode_weight, dtype=np.float32)).reshape(-1).to(device)
        greedy_bc_mode_target = torch.from_numpy(np.asarray(self.greedy_bc_mode_target, dtype=np.int64)).reshape(-1).to(device)
        greedy_bc_packet_target = torch.from_numpy(np.asarray(self.greedy_bc_packet_target, dtype=np.int64)).reshape(-1).to(device)
        greedy_bc_owner_target = torch.from_numpy(np.asarray(self.greedy_bc_owner_target, dtype=np.int64)).reshape(-1).to(device)
        greedy_bc_mode_weight = torch.from_numpy(np.asarray(self.greedy_bc_mode_weight, dtype=np.float32)).reshape(-1).to(device)
        greedy_bc_packet_weight = torch.from_numpy(np.asarray(self.greedy_bc_packet_weight, dtype=np.float32)).reshape(-1).to(device)
        greedy_bc_owner_weight = torch.from_numpy(np.asarray(self.greedy_bc_owner_weight, dtype=np.float32)).reshape(-1).to(device)
        phase_a_embb_power_anchor_target = torch.from_numpy(
            np.asarray(self.phase_a_embb_power_anchor_target, dtype=np.float32)
        ).reshape(-1).to(device)
        phase_a_embb_power_anchor_weight = torch.from_numpy(
            np.asarray(self.phase_a_embb_power_anchor_weight, dtype=np.float32)
        ).reshape(-1).to(device)
        phase_a_mask = torch.from_numpy(np.asarray(self.phase_a_mask, dtype=np.float32)).reshape(-1).to(device)

        return RolloutBatch(
            local_obs=local_obs,
            global_obs=global_obs,
            mode_mask=mode_mask,
            packet_mask=packet_mask,
            embb_owner_mask=embb_owner_mask,
            mode_actions=mode_actions,
            packet_actions=packet_actions,
            power_pre_tanh=power_pre_tanh,
            power_delta=power_delta,
            embb_owner_actions=embb_owner_actions,
            embb_power_pre_tanh=embb_power_pre_tanh,
            embb_power_delta=embb_power_delta,
            old_log_prob=old_log_prob,
            values=values,
            returns=returns,
            advantages=advantages,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            aux_best_mode_target=aux_best_mode_target,
            aux_overlay_feasible_target=aux_overlay_feasible_target,
            aux_best_packet_target=aux_best_packet_target,
            teacher_admission_target=teacher_admission_target,
            teacher_mode_target=teacher_mode_target,
            teacher_admission_weight=teacher_admission_weight,
            teacher_mode_weight=teacher_mode_weight,
            greedy_bc_mode_target=greedy_bc_mode_target,
            greedy_bc_packet_target=greedy_bc_packet_target,
            greedy_bc_owner_target=greedy_bc_owner_target,
            greedy_bc_mode_weight=greedy_bc_mode_weight,
            greedy_bc_packet_weight=greedy_bc_packet_weight,
            greedy_bc_owner_weight=greedy_bc_owner_weight,
            phase_a_embb_power_anchor_target=phase_a_embb_power_anchor_target,
            phase_a_embb_power_anchor_weight=phase_a_embb_power_anchor_weight,
            phase_a_mask=phase_a_mask,
        )

    def summary(self) -> Dict[str, float]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        return {
            "num_steps": float(self.num_steps),
            "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
            "terminal_fraction": float(np.mean(dones)) if dones.size > 0 else 0.0,
        }
