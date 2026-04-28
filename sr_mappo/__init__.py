"""Shielded Recurrent Action-Masked MAPPO package."""

from . import _bootstrap  # noqa: F401

from .bc import GreedyWarmStartTrainer, collect_greedy_bc_dataset
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
from .evaluate import evaluate_against_greedy, rollout_episode
from .networks import SRMAPPOActorCritic
from .trainer import SRMAPPOTrainer, build_default_components, run_training_loop
from .types import (
    MODE_KEEP,
    MODE_OVERLAY,
    MODE_PUNCTURE,
    AgentObservation,
    CandidatePacket,
    HybridAction,
)

__all__ = [
    "SRMAPPOConfig",
    "SRMAPPOPhaseAEnv",
    "SRMAPPOActorCritic",
    "SRMAPPOTrainer",
    "GreedyWarmStartTrainer",
    "collect_greedy_bc_dataset",
    "build_default_components",
    "run_training_loop",
    "evaluate_against_greedy",
    "rollout_episode",
    "MODE_KEEP",
    "MODE_OVERLAY",
    "MODE_PUNCTURE",
    "HybridAction",
    "CandidatePacket",
    "AgentObservation",
]
