"""Small smoke test for the SR-MAPPO package."""

import numpy as np
import torch

from config import AlgorithmConfig, SimulationConfig, SystemConfig, URLLCConfig, eMBBConfig

from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .networks import SRMAPPOActorCritic
from .types import HybridAction


def _policy_actions_from_output(agent_ids, output):
    actions = {}
    for idx, agent_id in enumerate(agent_ids):
        actions[agent_id] = HybridAction(
            mode=int(output.mode[idx].item()),
            packet_option=int(output.packet_option[idx].item()),
            power_delta=0.0,
            embb_owner_option=int(output.embb_owner_option[idx].item()),
            embb_power_delta=float(output.embb_power_delta[idx].item()),
        )
    return actions


def run_smoke_test():
    sys_cfg = SystemConfig()
    sys_cfg.num_subcarriers = 12
    sys_cfg.num_embb_users = 20
    sys_cfg.num_urllc_users = 8
    sys_cfg.refresh_derived_params()

    urllc_cfg = URLLCConfig()
    embb_cfg = eMBBConfig()
    algo_cfg = AlgorithmConfig()
    sim_cfg = SimulationConfig()
    sim_cfg.verbose = False
    sim_cfg.urllc_poisson_rate = 8
    sim_cfg.fixed_urllc_poisson_rate = True

    rl_cfg = SRMAPPOConfig()
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, rl_cfg)
    obs, info = env.reset(seed=sim_cfg.random_seed)

    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, rl_cfg)
    actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids))

    local_obs = torch.from_numpy(np.stack([obs[agent_id].local_obs for agent_id in env.agent_ids])).float()
    global_obs = torch.from_numpy(np.stack([obs[agent_id].global_obs for agent_id in env.agent_ids])).float()
    mode_mask = torch.from_numpy(np.stack([obs[agent_id].masks.mode_mask for agent_id in env.agent_ids])).float()
    packet_mask = torch.from_numpy(np.stack([obs[agent_id].masks.packet_mask for agent_id in env.agent_ids])).float()
    embb_owner_mask = torch.from_numpy(np.stack([obs[agent_id].masks.embb_owner_mask for agent_id in env.agent_ids])).float()
    output = model.act(
        local_obs=local_obs,
        global_obs=global_obs,
        mode_mask=mode_mask,
        packet_mask=packet_mask,
        embb_owner_mask=embb_owner_mask,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        deterministic=False,
    )
    actions = _policy_actions_from_output(env.agent_ids, output)
    next_obs, rewards, dones, infos = env.step(actions)

    print("SR-MAPPO smoke test")
    print(f"  Agents: {env.agent_ids}")
    print(f"  Reset info: {info}")
    print(f"  Local obs dim: {env.local_obs_dim}, Global obs dim: {env.global_obs_dim}")
    print(f"  Step rewards: {rewards}")
    print(f"  Step dones: {dones}")
    print(f"  Example info[uav_0]: {infos[env.agent_ids[0]]}")
    print(f"  Next obs keys: {list(next_obs.keys())}")


if __name__ == "__main__":
    run_smoke_test()
