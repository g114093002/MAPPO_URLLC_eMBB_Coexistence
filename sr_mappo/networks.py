"""Recurrent actor-critic with masked hybrid actions and mode-first conditioning."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from .config import SRMAPPOConfig
from .types import MODE_KEEP


def _apply_mask_to_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return logits
    if mask.dtype != torch.bool:
        mask = mask > 0
    return logits.masked_fill(~mask, -1.0e9)


@dataclass
class PolicyStepOutput:
    """Network output for one multi-agent decision step."""

    mode: torch.Tensor
    packet_option: torch.Tensor
    embb_owner_option: torch.Tensor
    power_pre_tanh: torch.Tensor
    power_delta: torch.Tensor
    embb_power_pre_tanh: torch.Tensor
    embb_power_delta: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mode_entropy: torch.Tensor
    packet_entropy: torch.Tensor
    embb_owner_entropy: torch.Tensor
    actor_hidden: torch.Tensor
    critic_hidden: torch.Tensor
    value: torch.Tensor
    best_mode_logits: torch.Tensor
    overlay_feasible_logit: torch.Tensor
    embb_owner_logits: Optional[torch.Tensor] = None


class RecurrentEncoder(nn.Module):
    """Small MLP + GRU encoder used by both actor and critic."""

    def __init__(self, input_dim: int, hidden_dim: int, recurrent_dim: int, *, use_recurrent: bool = True):
        super().__init__()
        self.use_recurrent = bool(use_recurrent)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        if self.use_recurrent:
            self.gru = nn.GRU(hidden_dim, recurrent_dim, batch_first=True)
            self.proj = None
        else:
            # Feedforward fallback: keep the same output dimension as GRU would.
            self.gru = None
            self.proj = nn.Linear(hidden_dim, recurrent_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.mlp(x)
        if self.use_recurrent:
            assert self.gru is not None
            output, next_hidden = self.gru(features, hidden)
            return output[:, -1, :], next_hidden

        assert self.proj is not None
        latent = self.proj(features[:, -1, :])
        # Preserve the recurrent API: downstream code expects a (1, B, D) tensor.
        batch = int(latent.shape[0])
        next_hidden = torch.zeros(1, batch, latent.shape[-1], device=latent.device, dtype=latent.dtype)
        return latent, next_hidden


class SRMAPPOActorCritic(nn.Module):
    """Shared-actor, centralized-critic network for SR-MAPPO."""

    def __init__(self, local_obs_dim: int, global_obs_dim: int, cfg: SRMAPPOConfig):
        super().__init__()
        self.cfg = cfg
        action_cfg = cfg.action
        net_cfg = cfg.network

        self.packet_dim = action_cfg.max_candidate_packets + int(action_cfg.include_null_packet_option)
        owner_space = str(getattr(action_cfg, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if owner_space == "global_owner_id_no_null":
            self.embb_owner_dim = int(getattr(action_cfg, "global_embb_owner_dim", 0) or 0)
        else:
            self.embb_owner_dim = action_cfg.max_embb_candidates + int(action_cfg.include_null_embb_option)
        self.num_modes = action_cfg.num_mode_actions

        self.actor_encoder = RecurrentEncoder(
            input_dim=local_obs_dim,
            hidden_dim=net_cfg.local_encoder_dim,
            recurrent_dim=net_cfg.recurrent_hidden_dim,
            use_recurrent=bool(getattr(net_cfg, "use_recurrent", True)),
        )
        self.critic_encoder = RecurrentEncoder(
            input_dim=global_obs_dim,
            hidden_dim=net_cfg.global_encoder_dim,
            recurrent_dim=net_cfg.recurrent_hidden_dim,
            use_recurrent=bool(getattr(net_cfg, "use_recurrent", True)),
        )

        self.mode_head = nn.Linear(net_cfg.recurrent_hidden_dim, self.num_modes)
        self.packet_condition = nn.Sequential(
            nn.Linear(net_cfg.recurrent_hidden_dim + self.num_modes, net_cfg.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(net_cfg.actor_hidden_dim, net_cfg.actor_hidden_dim),
            nn.Tanh(),
        )
        self.packet_head = nn.Linear(net_cfg.actor_hidden_dim, self.packet_dim)

        self.power_condition = nn.Sequential(
            nn.Linear(net_cfg.recurrent_hidden_dim + self.num_modes + self.packet_dim, net_cfg.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(net_cfg.actor_hidden_dim, net_cfg.actor_hidden_dim),
            nn.Tanh(),
        )
        self.power_mean_head = nn.Linear(net_cfg.actor_hidden_dim, 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))

        self.embb_owner_head = nn.Linear(net_cfg.recurrent_hidden_dim, self.embb_owner_dim)
        self.embb_power_condition = nn.Sequential(
            nn.Linear(net_cfg.recurrent_hidden_dim + self.embb_owner_dim, net_cfg.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(net_cfg.actor_hidden_dim, net_cfg.actor_hidden_dim),
            nn.Tanh(),
        )
        self.embb_power_mean_head = nn.Linear(net_cfg.actor_hidden_dim, 1)
        self.embb_power_log_std = nn.Parameter(torch.full((1,), 0.35))
        self.embb_owner_non_null_logit_bias = 0.20
        self.embb_power_mean_gain = 1.35

        with torch.no_grad():
            if (
                owner_space != "global_owner_id_no_null"
                and bool(getattr(action_cfg, "include_null_embb_option", True))
                and self.embb_owner_head.bias is not None
                and self.embb_owner_dim > 1
            ):
                self.embb_owner_head.bias.zero_()
                self.embb_owner_head.bias[0] = -self.embb_owner_non_null_logit_bias
                self.embb_owner_head.bias[1:] = self.embb_owner_non_null_logit_bias

        self.best_mode_head = nn.Linear(net_cfg.recurrent_hidden_dim, self.num_modes)
        self.overlay_feasible_head = nn.Linear(net_cfg.recurrent_hidden_dim, 1)
        self.phase_a_embb_power_enabled = bool(getattr(cfg.env, "allow_phase_a_embb_power_adjustment", False))

        self.value_head = nn.Sequential(
            nn.Linear(net_cfg.recurrent_hidden_dim, net_cfg.critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(net_cfg.critic_hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        recurrent_dim = self.cfg.network.recurrent_hidden_dim
        actor_hidden = torch.zeros(1, batch_size, recurrent_dim, device=device)
        critic_hidden = torch.zeros(1, batch_size, recurrent_dim, device=device)
        return actor_hidden, critic_hidden

    def _mode_one_hot(self, mode_actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(mode_actions.long(), num_classes=self.num_modes).float()

    def _packet_one_hot(self, packet_actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(packet_actions.long(), num_classes=self.packet_dim).float()

    def _embb_owner_one_hot(self, embb_actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(embb_actions.long(), num_classes=self.embb_owner_dim).float()

    def _condition_packet_mask(self, packet_mask: Optional[torch.Tensor], mode_actions: torch.Tensor) -> Optional[torch.Tensor]:
        if packet_mask is None:
            return None
        if packet_mask.dim() == 3:
            if packet_mask.dtype != torch.bool:
                packet_mask = packet_mask > 0
            row_index = torch.arange(packet_mask.shape[0], device=packet_mask.device)
            conditioned = packet_mask[row_index, mode_actions.long()].clone()
        else:
            conditioned = packet_mask > 0 if packet_mask.dtype != torch.bool else packet_mask.clone()
            conditioned = conditioned.clone()

        for row_idx in range(conditioned.shape[0]):
            if int(mode_actions[row_idx].item()) == MODE_KEEP:
                conditioned[row_idx] = False
                conditioned[row_idx, 0] = True
            else:
                conditioned[row_idx, 0] = False
                if not bool(torch.any(conditioned[row_idx])):
                    conditioned[row_idx, 0] = True
        return conditioned

    def compute_mode_logits(self, actor_latent: torch.Tensor, mode_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _apply_mask_to_logits(self.mode_head(actor_latent), mode_mask)

    def compute_packet_logits(
        self,
        actor_latent: torch.Tensor,
        mode_actions: torch.Tensor,
        packet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode_one_hot = self._mode_one_hot(mode_actions)
        packet_latent = self.packet_condition(torch.cat([actor_latent, mode_one_hot], dim=-1))
        conditioned_mask = self._condition_packet_mask(packet_mask, mode_actions)
        return _apply_mask_to_logits(self.packet_head(packet_latent), conditioned_mask)

    def compute_power_mean(
        self,
        actor_latent: torch.Tensor,
        mode_actions: torch.Tensor,
        packet_actions: torch.Tensor,
    ) -> torch.Tensor:
        mode_one_hot = self._mode_one_hot(mode_actions)
        packet_one_hot = self._packet_one_hot(packet_actions)
        power_latent = self.power_condition(torch.cat([actor_latent, mode_one_hot, packet_one_hot], dim=-1))
        return self.power_mean_head(power_latent)

    def compute_embb_owner_logits(self, actor_latent: torch.Tensor, embb_owner_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.embb_owner_head(actor_latent)
        owner_space = str(getattr(self.cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
        if (
            owner_space != "global_owner_id_no_null"
            and bool(getattr(self.cfg.action, "include_null_embb_option", True))
            and embb_owner_mask is not None
            and self.embb_owner_dim > 1
        ):
            if embb_owner_mask.dtype != torch.bool:
                active_non_null = torch.sum(embb_owner_mask[..., 1:], dim=-1, keepdim=True) > 0.5
            else:
                active_non_null = torch.any(embb_owner_mask[..., 1:], dim=-1, keepdim=True)
            if torch.any(active_non_null):
                bias = torch.cat(
                    [
                        -self.embb_owner_non_null_logit_bias * active_non_null.float(),
                        self.embb_owner_non_null_logit_bias * active_non_null.float().expand(-1, self.embb_owner_dim - 1),
                    ],
                    dim=-1,
                )
                logits = logits + bias
        return _apply_mask_to_logits(logits, embb_owner_mask)

    def compute_embb_power_mean(
        self,
        actor_latent: torch.Tensor,
        embb_owner_actions: torch.Tensor,
    ) -> torch.Tensor:
        embb_one_hot = self._embb_owner_one_hot(embb_owner_actions)
        embb_latent = self.embb_power_condition(torch.cat([actor_latent, embb_one_hot], dim=-1))
        return self.embb_power_mean_gain * self.embb_power_mean_head(embb_latent)


    def compute_aux_predictions(self, actor_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.best_mode_head(actor_latent), self.overlay_feasible_head(actor_latent).squeeze(-1)

    def _embb_power_head_enabled(self) -> bool:
        return bool(getattr(self.cfg.env, "learn_embb_baseline", False)) or bool(
            getattr(self, "phase_a_embb_power_enabled", getattr(self.cfg.env, "allow_phase_a_embb_power_adjustment", False))
        )

    def _embb_activity_masks(
        self,
        embb_owner_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        owner_active = None
        if embb_owner_mask is not None:
            owner_space = str(getattr(self.cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
            if owner_space == "global_owner_id_no_null":
                owner_active = torch.sum(embb_owner_mask, dim=-1) > 0.5
            else:
                owner_active = torch.sum(embb_owner_mask[..., 1:], dim=-1) > 0.5
        if owner_active is None:
            if self._embb_power_head_enabled():
                power_active = None if embb_owner_mask is None else torch.ones(
                    embb_owner_mask.shape[0],
                    dtype=torch.bool,
                    device=embb_owner_mask.device,
                )
            else:
                power_active = None
        else:
            planning_power_active = owner_active & bool(getattr(self.cfg.env, "learn_phase0_embb_power", True))
            phase_a_power_active = (~owner_active) & bool(
                getattr(self, "phase_a_embb_power_enabled", getattr(self.cfg.env, "allow_phase_a_embb_power_adjustment", False))
            )
            power_active = planning_power_active | phase_a_power_active
        return owner_active, power_active

    def _combine_head_terms(
        self,
        *,
        mode_term: torch.Tensor,
        packet_term: torch.Tensor,
        embb_owner_term: torch.Tensor,
        embb_power_term: torch.Tensor,
        embb_owner_active: Optional[torch.Tensor],
        embb_power_active: Optional[torch.Tensor],
    ) -> torch.Tensor:
        total = mode_term + packet_term + embb_owner_term + embb_power_term
        if not bool(getattr(self.cfg.network, "normalize_joint_log_prob_by_active_heads", False)):
            return total
        active_heads = torch.full_like(mode_term, 2.0)
        if embb_owner_active is None:
            if self.cfg.env.learn_embb_baseline:
                active_heads = active_heads + 1.0
        else:
            active_heads = active_heads + embb_owner_active.to(dtype=mode_term.dtype)
        if embb_power_active is None:
            if self._embb_power_head_enabled():
                active_heads = active_heads + 1.0
        else:
            active_heads = active_heads + embb_power_active.to(dtype=mode_term.dtype)
        return total / torch.clamp(active_heads, min=1.0)

    def act(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        packet_mask: Optional[torch.Tensor] = None,
        embb_owner_mask: Optional[torch.Tensor] = None,
        actor_hidden: Optional[torch.Tensor] = None,
        critic_hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> PolicyStepOutput:
        actor_latent, next_actor_hidden = self.actor_encoder(local_obs, actor_hidden)
        critic_latent, next_critic_hidden = self.critic_encoder(global_obs, critic_hidden)

        mode_logits = self.compute_mode_logits(actor_latent, mode_mask)
        mode_dist = Categorical(logits=mode_logits)
        if deterministic:
            mode = torch.argmax(mode_logits, dim=-1)
        else:
            mode = mode_dist.sample()

        packet_logits = self.compute_packet_logits(actor_latent, mode, packet_mask)
        packet_dist = Categorical(logits=packet_logits)
        if deterministic:
            packet = torch.argmax(packet_logits, dim=-1)
        else:
            packet = packet_dist.sample()

        embb_power_head_enabled = self._embb_power_head_enabled()

        if self.cfg.env.learn_embb_baseline:
            embb_owner_active, embb_power_active = self._embb_activity_masks(embb_owner_mask)
            embb_owner_logits = self.compute_embb_owner_logits(actor_latent, embb_owner_mask)
            embb_owner_dist = Categorical(logits=embb_owner_logits)
            if deterministic:
                sampled_embb_owner = torch.argmax(embb_owner_logits, dim=-1)
            else:
                sampled_embb_owner = embb_owner_dist.sample()
            if embb_owner_active is None:
                embb_owner = sampled_embb_owner
            else:
                embb_owner = torch.where(
                    embb_owner_active,
                    sampled_embb_owner,
                    torch.zeros_like(sampled_embb_owner),
                )
        else:
            embb_owner = torch.zeros_like(mode)
            embb_owner_dist = None
            embb_owner_active = None
            embb_owner_logits = None
            embb_power_active = torch.ones_like(mode, dtype=torch.bool) if embb_power_head_enabled else None

        # URLLC tx power is solved inside the environment from the selected
        # mode/candidate pair, so the policy no longer learns or samples this head.
        # Keep zero-valued outputs here for compatibility with the existing
        # rollout/eval interfaces while removing the head from PPO training.
        power_pre_tanh = torch.zeros((mode.shape[0], 1), dtype=actor_latent.dtype, device=actor_latent.device)
        power_delta = torch.zeros_like(power_pre_tanh)

        if embb_power_head_enabled:
            embb_power_owner_context = embb_owner if self.cfg.env.learn_embb_baseline else torch.zeros_like(mode)
            embb_mean = self.compute_embb_power_mean(actor_latent, embb_power_owner_context)
            embb_log_std = torch.clamp(
                self.embb_power_log_std,
                min=self.cfg.network.min_power_log_std,
                max=self.cfg.network.max_power_log_std,
            )
            embb_std = torch.exp(embb_log_std).expand_as(embb_mean)
            embb_power_dist = Normal(embb_mean, embb_std)
            sampled_embb_power_pre_tanh = embb_mean if deterministic else embb_power_dist.rsample()
            sampled_embb_power_delta = torch.tanh(sampled_embb_power_pre_tanh)
            if self.cfg.env.learn_embb_baseline:
                sampled_embb_owner_log_prob = embb_owner_dist.log_prob(embb_owner)
                sampled_embb_owner_entropy = embb_owner_dist.entropy()
            else:
                sampled_embb_owner_log_prob = torch.zeros_like(mode, dtype=embb_mean.dtype)
                sampled_embb_owner_entropy = torch.zeros_like(mode, dtype=embb_mean.dtype)
            sampled_embb_power_log_prob = embb_power_dist.log_prob(sampled_embb_power_pre_tanh).sum(dim=-1)
            sampled_embb_power_entropy = embb_power_dist.entropy().sum(dim=-1)
            if not self.cfg.env.learn_embb_baseline:
                embb_owner_log_prob = sampled_embb_owner_log_prob
                embb_owner_entropy = sampled_embb_owner_entropy
            elif embb_owner_active is None:
                embb_owner_log_prob = sampled_embb_owner_log_prob
                embb_owner_entropy = sampled_embb_owner_entropy
            else:
                zero_like_owner_scalar = torch.zeros_like(sampled_embb_owner_log_prob)
                embb_owner_log_prob = torch.where(
                    embb_owner_active,
                    sampled_embb_owner_log_prob,
                    zero_like_owner_scalar,
                )
                embb_owner_entropy = torch.where(
                    embb_owner_active,
                    sampled_embb_owner_entropy,
                    zero_like_owner_scalar,
                )
            if embb_power_active is None:
                embb_power_pre_tanh = sampled_embb_power_pre_tanh
                embb_power_delta = sampled_embb_power_delta
                embb_power_log_prob = sampled_embb_power_log_prob
                embb_power_entropy = sampled_embb_power_entropy
            else:
                zero_like_power = torch.zeros_like(sampled_embb_power_pre_tanh)
                zero_like_power_scalar = torch.zeros_like(sampled_embb_power_log_prob)
                embb_power_pre_tanh = torch.where(
                    embb_power_active.unsqueeze(-1),
                    sampled_embb_power_pre_tanh,
                    zero_like_power,
                )
                embb_power_delta = torch.where(
                    embb_power_active.unsqueeze(-1),
                    sampled_embb_power_delta,
                    zero_like_power,
                )
                embb_power_log_prob = torch.where(
                    embb_power_active,
                    sampled_embb_power_log_prob,
                    zero_like_power_scalar,
                )
                embb_power_entropy = torch.where(
                    embb_power_active,
                    sampled_embb_power_entropy,
                    zero_like_power_scalar,
                )
            embb_log_prob = embb_owner_log_prob + embb_power_log_prob
        else:
            embb_mean = torch.zeros_like(power_pre_tanh)
            embb_power_pre_tanh = torch.zeros_like(power_pre_tanh)
            embb_power_delta = torch.zeros_like(power_delta)
            embb_owner_log_prob = torch.zeros_like(mode, dtype=power_pre_tanh.dtype)
            embb_log_prob = torch.zeros_like(mode, dtype=power_pre_tanh.dtype)
            embb_owner_entropy = torch.zeros_like(mode, dtype=power_pre_tanh.dtype)
            embb_power_entropy = torch.zeros_like(mode, dtype=power_pre_tanh.dtype)

        log_prob = self._combine_head_terms(
            mode_term=mode_dist.log_prob(mode),
            packet_term=packet_dist.log_prob(packet),
            embb_owner_term=embb_owner_log_prob,
            embb_power_term=embb_log_prob - embb_owner_log_prob,
            embb_owner_active=embb_owner_active,
            embb_power_active=embb_power_active,
        )
        mode_entropy = mode_dist.entropy()
        packet_entropy = packet_dist.entropy()
        entropy = self._combine_head_terms(
            mode_term=mode_entropy,
            packet_term=packet_entropy,
            embb_owner_term=embb_owner_entropy,
            embb_power_term=embb_power_entropy,
            embb_owner_active=embb_owner_active,
            embb_power_active=embb_power_active,
        )
        value = self.value_head(critic_latent).squeeze(-1)
        best_mode_logits, overlay_feasible_logit = self.compute_aux_predictions(actor_latent)

        return PolicyStepOutput(
            mode=mode,
            packet_option=packet,
            embb_owner_option=embb_owner,
            power_pre_tanh=power_pre_tanh,
            power_delta=power_delta.squeeze(-1),
            embb_power_pre_tanh=embb_power_pre_tanh,
            embb_power_delta=embb_power_delta.squeeze(-1),
            log_prob=log_prob,
            entropy=entropy,
            mode_entropy=mode_entropy,
            packet_entropy=packet_entropy,
            embb_owner_entropy=embb_owner_entropy,
            actor_hidden=next_actor_hidden,
            critic_hidden=next_critic_hidden,
            value=value,
            best_mode_logits=best_mode_logits,
            overlay_feasible_logit=overlay_feasible_logit,
            embb_owner_logits=embb_owner_logits,
        )

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        mode_actions: torch.Tensor,
        packet_actions: torch.Tensor,
        power_pre_tanh: torch.Tensor,
        embb_owner_actions: torch.Tensor,
        embb_power_pre_tanh: torch.Tensor,
        mode_mask: Optional[torch.Tensor] = None,
        packet_mask: Optional[torch.Tensor] = None,
        embb_owner_mask: Optional[torch.Tensor] = None,
        actor_hidden: Optional[torch.Tensor] = None,
        critic_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        actor_latent, next_actor_hidden = self.actor_encoder(local_obs, actor_hidden)
        critic_latent, next_critic_hidden = self.critic_encoder(global_obs, critic_hidden)

        mode_logits = self.compute_mode_logits(actor_latent, mode_mask)
        packet_logits = self.compute_packet_logits(actor_latent, mode_actions, packet_mask)
        mode_dist = Categorical(logits=mode_logits)
        packet_dist = Categorical(logits=packet_logits)

        power_pre_tanh = torch.zeros((mode_actions.shape[0], 1), dtype=actor_latent.dtype, device=actor_latent.device)

        embb_power_head_enabled = self._embb_power_head_enabled()

        if self.cfg.env.learn_embb_baseline:
            embb_owner_active, embb_power_active = self._embb_activity_masks(embb_owner_mask)
            embb_owner_logits = self.compute_embb_owner_logits(actor_latent, embb_owner_mask)
            embb_owner_dist = Categorical(logits=embb_owner_logits)
            sampled_embb_owner_log_prob = embb_owner_dist.log_prob(embb_owner_actions)
            sampled_embb_owner_entropy = embb_owner_dist.entropy()
            if embb_owner_active is None:
                embb_owner_log_prob = sampled_embb_owner_log_prob
                embb_owner_entropy = sampled_embb_owner_entropy
            else:
                zero_like_owner_scalar = torch.zeros_like(sampled_embb_owner_log_prob)
                embb_owner_log_prob = torch.where(
                    embb_owner_active,
                    sampled_embb_owner_log_prob,
                    zero_like_owner_scalar,
                )
                embb_owner_entropy = torch.where(
                    embb_owner_active,
                    sampled_embb_owner_entropy,
                    zero_like_owner_scalar,
                )
        else:
            embb_owner_log_prob = torch.zeros_like(mode_actions, dtype=actor_latent.dtype)
            embb_owner_entropy = torch.zeros_like(mode_actions, dtype=actor_latent.dtype)
            embb_power_active = torch.ones_like(mode_actions, dtype=torch.bool) if embb_power_head_enabled else None

        if embb_power_head_enabled:
            embb_power_owner_context = embb_owner_actions if self.cfg.env.learn_embb_baseline else torch.zeros_like(mode_actions)
            embb_mean = self.compute_embb_power_mean(actor_latent, embb_power_owner_context)
            embb_log_std = torch.clamp(
                self.embb_power_log_std,
                min=self.cfg.network.min_power_log_std,
                max=self.cfg.network.max_power_log_std,
            )
            embb_std = torch.exp(embb_log_std).expand_as(embb_mean)
            embb_power_dist = Normal(embb_mean, embb_std)
            sampled_embb_power_log_prob = embb_power_dist.log_prob(embb_power_pre_tanh).sum(dim=-1)
            sampled_embb_power_entropy = embb_power_dist.entropy().sum(dim=-1)
            if embb_power_active is None:
                embb_power_log_prob = sampled_embb_power_log_prob
                embb_power_entropy = sampled_embb_power_entropy
            else:
                zero_like_power_scalar = torch.zeros_like(sampled_embb_power_log_prob)
                embb_power_log_prob = torch.where(
                    embb_power_active,
                    sampled_embb_power_log_prob,
                    zero_like_power_scalar,
                )
                embb_power_entropy = torch.where(
                    embb_power_active,
                    sampled_embb_power_entropy,
                    zero_like_power_scalar,
                )
            embb_log_prob = embb_owner_log_prob + embb_power_log_prob
            embb_entropy = embb_owner_entropy + embb_power_entropy
        else:
            embb_mean = torch.zeros_like(power_pre_tanh)
            embb_log_prob = torch.zeros_like(mode_actions, dtype=actor_latent.dtype)
            embb_entropy = torch.zeros_like(mode_actions, dtype=actor_latent.dtype)

        log_prob = self._combine_head_terms(
            mode_term=mode_dist.log_prob(mode_actions),
            packet_term=packet_dist.log_prob(packet_actions),
            embb_owner_term=embb_owner_log_prob,
            embb_power_term=embb_log_prob - embb_owner_log_prob,
            embb_owner_active=embb_owner_active if self.cfg.env.learn_embb_baseline else None,
            embb_power_active=embb_power_active,
        )
        entropy = self._combine_head_terms(
            mode_term=mode_dist.entropy(),
            packet_term=packet_dist.entropy(),
            embb_owner_term=embb_owner_entropy,
            embb_power_term=embb_entropy - embb_owner_entropy,
            embb_owner_active=embb_owner_active if self.cfg.env.learn_embb_baseline else None,
            embb_power_active=embb_power_active,
        )
        value = self.value_head(critic_latent).squeeze(-1)
        best_mode_logits, overlay_feasible_logit = self.compute_aux_predictions(actor_latent)

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
            "actor_hidden": next_actor_hidden,
            "critic_hidden": next_critic_hidden,
            "actor_latent": actor_latent,
            "mode_logits": mode_logits,
            "best_mode_logits": best_mode_logits,
            "overlay_feasible_logit": overlay_feasible_logit,
            "embb_power_mean": embb_mean if embb_power_head_enabled else torch.zeros_like(power_pre_tanh),
            "embb_power_delta_mean": (
                torch.tanh(embb_mean).squeeze(-1)
                if embb_power_head_enabled
                else torch.zeros_like(mode_actions, dtype=actor_latent.dtype)
            ),
        }
