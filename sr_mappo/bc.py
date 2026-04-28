"""Behavior cloning warm start using greedy references from the SR-MAPPO environment."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .env import SRMAPPOPhaseAEnv


@dataclass
class BCDataset:
    local_obs: np.ndarray
    global_obs: np.ndarray
    mode_mask: np.ndarray
    packet_mask: np.ndarray
    target_mode: np.ndarray
    target_packet: np.ndarray
    target_power: np.ndarray


def collect_greedy_bc_dataset(
    env: SRMAPPOPhaseAEnv,
    episodes: int = 8,
    seed: int = 42,
    before_reset: Optional[Callable[[SRMAPPOPhaseAEnv, int], None]] = None,
    teacher_policy: str = "greedy_reference",
) -> BCDataset:
    local_obs: List[np.ndarray] = []
    global_obs: List[np.ndarray] = []
    mode_mask: List[np.ndarray] = []
    packet_mask: List[np.ndarray] = []
    target_mode: List[int] = []
    target_packet: List[int] = []
    target_power: List[float] = []

    for episode in range(episodes):
        if before_reset is not None:
            before_reset(env, episode)
        observations, _info = env.reset(seed=seed + episode)
        done = False
        while not done:
            teacher_actions = {}
            for agent_id in env.agent_ids:
                obs = observations[agent_id]
                teacher = env.bc_teacher_action(agent_id, obs, teacher_policy=teacher_policy)
                planning_phase = bool(obs.metadata.get("planning_phase", 0.0) > 0.5)
                if not planning_phase:
                    local_obs.append(obs.local_obs)
                    global_obs.append(obs.global_obs)
                    mode_mask.append(obs.masks.mode_mask)
                    packet_mask.append(obs.masks.packet_mask)
                    target_mode.append(int(teacher.mode))
                    target_packet.append(int(teacher.packet_option))
                    target_power.append(float(teacher.power_delta))
                teacher_actions[agent_id] = teacher
            observations, _rewards, dones, _infos = env.step(teacher_actions)
            done = all(dones.values())

    return BCDataset(
        local_obs=np.asarray(local_obs, dtype=np.float32),
        global_obs=np.asarray(global_obs, dtype=np.float32),
        mode_mask=np.asarray(mode_mask, dtype=np.float32),
        packet_mask=np.asarray(packet_mask, dtype=np.float32),
        target_mode=np.asarray(target_mode, dtype=np.int64),
        target_packet=np.asarray(target_packet, dtype=np.int64),
        target_power=np.asarray(target_power, dtype=np.float32).reshape(-1, 1),
    )


class GreedyWarmStartTrainer:
    """Supervised warm start for the shared SR-MAPPO actor."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(self, dataset: BCDataset, epochs: int = 5, batch_size: int = 128, learning_rate: float = 1e-3) -> Dict[str, float]:
        if dataset.local_obs.size == 0:
            return {
                "bc_samples": 0.0,
                "bc_epochs": 0.0,
                "bc_loss": 0.0,
            }
        tensors = TensorDataset(
            torch.from_numpy(dataset.local_obs),
            torch.from_numpy(dataset.global_obs),
            torch.from_numpy(dataset.mode_mask),
            torch.from_numpy(dataset.packet_mask),
            torch.from_numpy(dataset.target_mode),
            torch.from_numpy(dataset.target_packet),
            torch.from_numpy(dataset.target_power),
        )
        loader = DataLoader(tensors, batch_size=min(batch_size, len(tensors)), shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        loss_history = []
        for _epoch in range(epochs):
            for batch in loader:
                local_obs, global_obs, mode_mask, packet_mask, target_mode, target_packet, target_power = [b.to(self.device) for b in batch]
                actor_latent, _ = self.model.actor_encoder(local_obs, None)
                critic_latent, _ = self.model.critic_encoder(global_obs, None)
                mode_logits = self.model.compute_mode_logits(actor_latent, mode_mask)
                packet_logits = self.model.compute_packet_logits(actor_latent, target_mode, packet_mask)
                power_mean = self.model.compute_power_mean(actor_latent, target_mode, target_packet)
                _value = self.model.value_head(critic_latent)

                mode_loss = ce(mode_logits, target_mode)
                packet_loss = ce(packet_logits, target_packet)
                power_loss = mse(torch.tanh(power_mean), target_power)
                loss = mode_loss + packet_loss + 0.25 * power_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                loss_history.append(float(loss.item()))

        return {
            "bc_samples": float(len(tensors)),
            "bc_epochs": float(epochs),
            "bc_loss": float(np.mean(loss_history)) if loss_history else 0.0,
        }
