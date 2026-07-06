"""Independent PPO baseline built on the SR-MAPPO environment interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import _bootstrap  # noqa: F401
from .compare import _build_main_like_configs
from .env import SRMAPPOPhaseAEnv
from .networks import SRMAPPOActorCritic
from .trainer import configure_env_for_users_per_uav
from .types import HybridAction


@dataclass
class IPPOIterationStats:
    iteration: int
    mean_reward: float
    mean_episode_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    runtime_sec: float


class IPPOBaselineTrainer:
    """Simple decentralized PPO with one policy per agent and local critics."""

    def __init__(self, cfg, *, device: Optional[str] = None, component_overrides: Optional[Dict[str, Dict[str, object]]] = None):
        self.cfg = deepcopy(cfg)
        self.cfg.network.use_recurrent = False
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.component_overrides = dict(component_overrides or {})
        self.env = self._build_env(self.cfg)
        self.num_agents = len(self.env.agent_ids)
        self.models = [
            SRMAPPOActorCritic(self.env.local_obs_dim, self.env.local_obs_dim, self.cfg).to(self.device)
            for _ in range(self.num_agents)
        ]
        actor_lr = float(getattr(self.cfg.training, "actor_learning_rate", 3.0e-4) or 3.0e-4)
        critic_lr = float(getattr(self.cfg.training, "critic_learning_rate", actor_lr) or actor_lr)
        learning_rate = min(actor_lr, critic_lr)
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            for model in self.models
        ]
        self.training_curve: List[Dict[str, float]] = []

    def _build_env(self, cfg):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _build_main_like_configs()
        for name, target in (
            ("system", sys_cfg),
            ("urllc", urllc_cfg),
            ("embb", embb_cfg),
            ("algorithm", algo_cfg),
            ("simulation", sim_cfg),
        ):
            for key, value in dict(self.component_overrides.get(name, {}) or {}).items():
                if hasattr(target, key):
                    setattr(target, key, value)
        if hasattr(sys_cfg, "refresh_derived_params"):
            sys_cfg.refresh_derived_params()
        sim_cfg.verbose = False
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, cfg)
        env.admission_guard_training_jitter_enabled = False
        return env

    def configure_load(self, total_load: Optional[float], mix_ratio: Optional[float]) -> float:
        if mix_ratio is not None:
            self.env.sim_cfg.urllc_user_ratio = float(np.clip(mix_ratio, 0.0, 0.95))
        if total_load is None:
            return float((self.env.sys_cfg.num_embb_users + self.env.sys_cfg.num_urllc_users) / self.env.sys_cfg.num_uavs)
        return float(configure_env_for_users_per_uav(self.env, float(total_load)))

    def _model_action(self, model, obs) -> tuple[HybridAction, float, float]:
        local_obs = torch.from_numpy(np.asarray(obs.local_obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        local_value_obs = local_obs
        mode_mask = torch.from_numpy(np.asarray(obs.masks.mode_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
        packet_mask = torch.from_numpy(np.asarray(obs.masks.packet_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
        embb_owner_mask = torch.from_numpy(np.asarray(obs.masks.embb_owner_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
        actor_hidden, critic_hidden = model.initial_state(batch_size=1, device=self.device)
        output = model.act(
            local_obs=local_obs,
            global_obs=local_value_obs,
            mode_mask=mode_mask,
            packet_mask=packet_mask,
            embb_owner_mask=embb_owner_mask,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
            deterministic=False,
        )
        action = HybridAction(
            mode=int(output.mode[0].item()),
            packet_option=int(output.packet_option[0].item()),
            power_delta=0.0,
            embb_owner_option=int(output.embb_owner_option[0].item()),
            embb_power_delta=float(output.embb_power_delta[0].item()),
        )
        return action, float(output.log_prob[0].item()), float(output.value[0].item())

    def _planning_action(self, obs) -> HybridAction:
        baseline_policy = str(
            getattr(self.cfg.env, "fixed_embb_baseline_policy", "minrate_then_throughput") or "minrate_then_throughput"
        )
        return self.env._planning_owner_action_for_baseline(obs, baseline_policy)

    def collect_episode(self, seed: int, reward_scope: str = "global") -> Dict[str, object]:
        observations, _ = self.env.reset(seed=seed)
        done = False
        reward_scope = str(reward_scope or "global").strip().lower()
        trajectories: List[List[Dict[str, object]]] = [[] for _ in range(self.num_agents)]
        episode_reward_sum = np.zeros(self.num_agents, dtype=np.float32)

        while not done:
            planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in self.env.agent_ids)
            joint_actions: Dict[str, HybridAction] = {}
            step_cache: List[Dict[str, object]] = []
            for agent_idx, agent_id in enumerate(self.env.agent_ids):
                obs = observations[agent_id]
                if planning_phase:
                    action = self._planning_action(obs)
                    log_prob = 0.0
                    value = 0.0
                else:
                    action, log_prob, value = self._model_action(self.models[agent_idx], obs)
                joint_actions[agent_id] = action
                step_cache.append(
                    {
                        "local_obs": np.asarray(obs.local_obs, dtype=np.float32),
                        "mode_mask": np.asarray(obs.masks.mode_mask, dtype=np.float32),
                        "packet_mask": np.asarray(obs.masks.packet_mask, dtype=np.float32),
                        "embb_owner_mask": np.asarray(obs.masks.embb_owner_mask, dtype=np.float32),
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "planning_phase": planning_phase,
                    }
                )

            next_observations, rewards, dones, _infos = self.env.step(joint_actions)
            done = all(dones.values())
            reward_values = np.asarray([float(rewards[aid]) for aid in self.env.agent_ids], dtype=np.float32)
            if reward_scope == "global":
                reward_values[:] = float(np.mean(reward_values))

            for agent_idx in range(self.num_agents):
                entry = dict(step_cache[agent_idx])
                entry["reward"] = float(reward_values[agent_idx])
                entry["done"] = float(done)
                trajectories[agent_idx].append(entry)
                episode_reward_sum[agent_idx] += float(reward_values[agent_idx])

            observations = next_observations

        summary = self.env.summarize_episode()
        return {
            "trajectories": trajectories,
            "summary": summary,
            "episode_reward_sum": episode_reward_sum,
        }

    def _agent_tensors(self, trajectory: List[Dict[str, object]]) -> Optional[Dict[str, torch.Tensor]]:
        phase_a_samples = [item for item in trajectory if not bool(item.get("planning_phase", False))]
        if not phase_a_samples:
            return None

        local_obs = torch.from_numpy(np.stack([item["local_obs"] for item in phase_a_samples]).astype(np.float32)).to(self.device)
        mode_mask = torch.from_numpy(np.stack([item["mode_mask"] for item in phase_a_samples]).astype(np.float32)).to(self.device)
        packet_mask = torch.from_numpy(np.stack([item["packet_mask"] for item in phase_a_samples]).astype(np.float32)).to(self.device)
        embb_owner_mask = torch.from_numpy(np.stack([item["embb_owner_mask"] for item in phase_a_samples]).astype(np.float32)).to(self.device)
        mode_actions = torch.tensor([int(item["action"].mode) for item in phase_a_samples], dtype=torch.long, device=self.device)
        packet_actions = torch.tensor([int(item["action"].packet_option) for item in phase_a_samples], dtype=torch.long, device=self.device)
        embb_owner_actions = torch.tensor([int(item["action"].embb_owner_option) for item in phase_a_samples], dtype=torch.long, device=self.device)
        old_log_prob = torch.tensor([float(item["log_prob"]) for item in phase_a_samples], dtype=torch.float32, device=self.device)
        values = torch.tensor([float(item["value"]) for item in phase_a_samples], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([float(item["reward"]) for item in phase_a_samples], dtype=torch.float32, device=self.device)
        dones = torch.tensor([float(item["done"]) for item in phase_a_samples], dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        gamma = float(getattr(self.cfg.training, "gamma", 0.99) or 0.99)
        gae_lambda = float(getattr(self.cfg.training, "gae_lambda", 0.95) or 0.95)
        for t in reversed(range(len(phase_a_samples))):
            next_value = values[t + 1] if t + 1 < len(phase_a_samples) else torch.tensor(0.0, dtype=torch.float32, device=self.device)
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / max(float(advantages.std(unbiased=False).item()), 1.0e-8)

        zero_pre_tanh = torch.zeros((len(phase_a_samples), 1), dtype=torch.float32, device=self.device)
        return {
            "local_obs": local_obs,
            "global_obs": local_obs,
            "mode_mask": mode_mask,
            "packet_mask": packet_mask,
            "embb_owner_mask": embb_owner_mask,
            "mode_actions": mode_actions,
            "packet_actions": packet_actions,
            "embb_owner_actions": embb_owner_actions,
            "old_log_prob": old_log_prob,
            "values": values,
            "returns": returns,
            "advantages": advantages,
            "power_pre_tanh": zero_pre_tanh,
            "embb_power_pre_tanh": zero_pre_tanh,
        }

    def update(self, trajectories: List[List[Dict[str, object]]]) -> Dict[str, float]:
        ppo_epochs = int(getattr(self.cfg.training, "ppo_epochs", 4) or 4)
        clip_ratio = float(getattr(self.cfg.training, "clip_ratio", 0.2) or 0.2)
        value_coef = float(getattr(self.cfg.training, "value_coef", 0.5) or 0.5)
        entropy_coef = float(getattr(self.cfg.training, "entropy_coef", 0.01) or 0.01)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updated_agents = 0

        for agent_idx, trajectory in enumerate(trajectories):
            batch = self._agent_tensors(trajectory)
            if batch is None:
                continue
            model = self.models[agent_idx]
            optimizer = self.optimizers[agent_idx]

            for _ in range(ppo_epochs):
                outputs = model.evaluate_actions(
                    local_obs=batch["local_obs"],
                    global_obs=batch["global_obs"],
                    mode_actions=batch["mode_actions"],
                    packet_actions=batch["packet_actions"],
                    power_pre_tanh=batch["power_pre_tanh"],
                    embb_owner_actions=batch["embb_owner_actions"],
                    embb_power_pre_tanh=batch["embb_power_pre_tanh"],
                    mode_mask=batch["mode_mask"],
                    packet_mask=batch["packet_mask"],
                    embb_owner_mask=batch["embb_owner_mask"],
                    actor_hidden=None,
                    critic_hidden=None,
                )
                ratio = torch.exp(outputs["log_prob"] - batch["old_log_prob"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(outputs["value"], batch["returns"])
                entropy = outputs["entropy"].mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(getattr(self.cfg.training, "max_grad_norm", 0.5) or 0.5))
                optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                updated_agents += 1

        denom = max(updated_agents, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
        }

    def train(
        self,
        *,
        seed: int,
        total_load: Optional[float] = None,
        mix_ratio: Optional[float] = None,
        iterations: Optional[int] = None,
        reward_scope: str = "global",
    ) -> Dict[str, object]:
        self.configure_load(total_load, mix_ratio)
        train_iterations = int(iterations or getattr(self.cfg.training, "total_iterations", 50) or 50)
        self.training_curve = []
        last_summary: Optional[Dict[str, object]] = None

        for iteration in range(1, train_iterations + 1):
            iter_start = perf_counter()
            episode = self.collect_episode(seed=seed + iteration, reward_scope=reward_scope)
            update_stats = self.update(episode["trajectories"])
            mean_step_reward = float(np.mean([item["reward"] for traj in episode["trajectories"] for item in traj])) if episode["trajectories"] else 0.0
            mean_episode_reward = float(np.mean(np.asarray(episode["episode_reward_sum"], dtype=np.float32)))
            record = IPPOIterationStats(
                iteration=iteration,
                mean_reward=mean_step_reward,
                mean_episode_reward=mean_episode_reward,
                policy_loss=float(update_stats["policy_loss"]),
                value_loss=float(update_stats["value_loss"]),
                entropy=float(update_stats["entropy"]),
                runtime_sec=float(perf_counter() - iter_start),
            )
            self.training_curve.append(record.__dict__)
            last_summary = episode["summary"]

        return {
            "training_curve": list(self.training_curve),
            "summary": last_summary or {},
        }

    def evaluate(
        self,
        *,
        seed: int,
        total_load: Optional[float] = None,
        mix_ratio: Optional[float] = None,
        deterministic: bool = True,
    ) -> Dict[str, object]:
        self.configure_load(total_load, mix_ratio)
        observations, _ = self.env.reset(seed=seed)
        done = False
        runtime_start = perf_counter()
        avg_sinr_samples: List[float] = []

        while not done:
            planning_phase = all(bool(observations[aid].metadata.get("planning_phase", 0.0)) for aid in self.env.agent_ids)
            joint_actions: Dict[str, HybridAction] = {}
            for agent_idx, agent_id in enumerate(self.env.agent_ids):
                obs = observations[agent_id]
                if planning_phase:
                    joint_actions[agent_id] = self._planning_action(obs)
                    continue
                model = self.models[agent_idx]
                local_obs = torch.from_numpy(np.asarray(obs.local_obs, dtype=np.float32)).unsqueeze(0).to(self.device)
                mode_mask = torch.from_numpy(np.asarray(obs.masks.mode_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
                packet_mask = torch.from_numpy(np.asarray(obs.masks.packet_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
                embb_owner_mask = torch.from_numpy(np.asarray(obs.masks.embb_owner_mask, dtype=np.float32)).unsqueeze(0).to(self.device)
                actor_hidden, critic_hidden = model.initial_state(batch_size=1, device=self.device)
                output = model.act(
                    local_obs=local_obs,
                    global_obs=local_obs,
                    mode_mask=mode_mask,
                    packet_mask=packet_mask,
                    embb_owner_mask=embb_owner_mask,
                    actor_hidden=actor_hidden,
                    critic_hidden=critic_hidden,
                    deterministic=deterministic,
                )
                joint_actions[agent_id] = HybridAction(
                    mode=int(output.mode[0].item()),
                    packet_option=int(output.packet_option[0].item()),
                    power_delta=0.0,
                    embb_owner_option=int(output.embb_owner_option[0].item()),
                    embb_power_delta=float(output.embb_power_delta[0].item()),
                )

            if planning_phase:
                resolved = {
                    aid: self.env._raw_action_to_shielded_action(joint_actions[aid], observations[aid])
                    for aid in self.env.agent_ids
                }
            else:
                minislot, rb = self.env._current_cell()
                resolved = self.env._resolve_executed_actions(joint_actions, observations, minislot=minislot, rb=rb)
                for shielded in resolved.values():
                    candidate = shielded.candidate
                    if candidate is None:
                        continue
                    if int(shielded.action.mode) == 1:
                        avg_sinr_samples.append(float(candidate.overlay_urllc_snir))
                    elif int(shielded.action.mode) == 2:
                        avg_sinr_samples.append(float(candidate.puncture_urllc_snir))

            observations, _rewards, dones, _infos = self.env.step(joint_actions, prebuilt_observations=observations, pre_resolved_actions=resolved)
            done = all(dones.values())

        summary = self.env.summarize_episode()
        return {
            "summary": summary,
            "avg_urllc_sinr_linear": float(np.mean(avg_sinr_samples)) if avg_sinr_samples else 0.0,
            "runtime_sec": float(perf_counter() - runtime_start),
            "training_curve": list(self.training_curve),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "cfg": self.cfg,
            "state_dicts": [model.state_dict() for model in self.models],
            "training_curve": list(self.training_curve),
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path) -> None:
        # IPPO checkpoints in this repo serialize the full config object, so on
        # PyTorch 2.6+ we must opt out of the safer weights-only default when
        # loading trusted local training artifacts.
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        state_dicts = list(payload.get("state_dicts", []) or [])
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)
        self.training_curve = list(payload.get("training_curve", []) or [])
