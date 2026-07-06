"""Shielded Recurrent Action-Masked MAPPO package."""

from . import _bootstrap  # noqa: F401
from .config import SRMAPPOConfig
from .env import SRMAPPOPhaseAEnv
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


def __getattr__(name):
    if name in {"GreedyWarmStartTrainer", "collect_greedy_bc_dataset"}:
        from .bc import GreedyWarmStartTrainer, collect_greedy_bc_dataset

        mapping = {
            "GreedyWarmStartTrainer": GreedyWarmStartTrainer,
            "collect_greedy_bc_dataset": collect_greedy_bc_dataset,
        }
        return mapping[name]
    if name in {"evaluate_against_greedy", "rollout_episode"}:
        from .evaluate import evaluate_against_greedy, rollout_episode

        mapping = {
            "evaluate_against_greedy": evaluate_against_greedy,
            "rollout_episode": rollout_episode,
        }
        return mapping[name]
    if name == "SRMAPPOActorCritic":
        from .networks import SRMAPPOActorCritic

        return SRMAPPOActorCritic
    if name in {"SRMAPPOTrainer", "build_default_components", "run_training_loop"}:
        from .trainer import SRMAPPOTrainer, build_default_components, run_training_loop

        mapping = {
            "SRMAPPOTrainer": SRMAPPOTrainer,
            "build_default_components": build_default_components,
            "run_training_loop": run_training_loop,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
