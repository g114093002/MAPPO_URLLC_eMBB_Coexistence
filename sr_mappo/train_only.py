"""Train SR-MAPPO without auto-running report and diagnostics."""

import argparse
import os
from pathlib import Path
from pprint import pprint

from .config import SRMAPPOConfig, torch_load_checkpoint
from .evaluate import evaluate_dual_selection
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .run_docs import write_training_run_markdown
from .trainer import run_training_loop

TRAIN_ONLY_DEFAULT_EVAL_EVERY = 100
TRAIN_ONLY_DEFAULT_EVAL_EPISODES_PER_LOAD = 1
TRAIN_ONLY_DEFAULT_PROGRESS_EVERY = 1
TRAIN_ONLY_DEFAULT_SIMPLIFIED_RB_SUMMARY_OBS = True


def run_train_only(
    experiment: str | None = None,
    resume_latest: bool = False,
    resume_path: str | None = None,
    additional_iterations: int | None = None,
    iterations: int | None = None,
    rollout_horizon: int | None = None,
    ppo_epochs: int | None = None,
    minibatch_size: int | None = None,
    eval_every: int | None = None,
    checkpoint_every: int | None = None,
    progress_every: int | None = None,
    eval_episodes_per_load: int | None = None,
    disable_eval: bool = False,
    disable_arrival_log: bool | None = None,
):
    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment)
    if disable_arrival_log is None:
        disable_arrival_log = True
    cfg.env.simplified_rb_summary_observation = bool(TRAIN_ONLY_DEFAULT_SIMPLIFIED_RB_SUMMARY_OBS)
    if disable_arrival_log:
        os.environ["SR_MAPPO_LOG_EFFECTIVE_LAMBDA"] = "0"
    if iterations is not None:
        cfg.training.total_iterations = int(max(iterations, 1))
    if rollout_horizon is not None:
        cfg.training.rollout_horizon = int(max(rollout_horizon, 1))
    if ppo_epochs is not None:
        cfg.training.ppo_epochs = int(max(ppo_epochs, 1))
    if minibatch_size is not None:
        cfg.training.minibatch_size = int(max(minibatch_size, 1))
    if eval_every is None:
        cfg.training.eval_every = int(TRAIN_ONLY_DEFAULT_EVAL_EVERY)
    else:
        cfg.training.eval_every = int(max(eval_every, 1))
    if checkpoint_every is not None:
        cfg.training.checkpoint_every = int(max(checkpoint_every, 1))
    if progress_every is None:
        cfg.training.progress_every = int(TRAIN_ONLY_DEFAULT_PROGRESS_EVERY)
    else:
        cfg.training.progress_every = int(max(progress_every, 1))
    if eval_episodes_per_load is None:
        cfg.training.eval_episodes_per_load = int(TRAIN_ONLY_DEFAULT_EVAL_EPISODES_PER_LOAD)
        cfg.training.checkpoint_eval_episodes_per_load = int(TRAIN_ONLY_DEFAULT_EVAL_EPISODES_PER_LOAD)
    else:
        cfg.training.eval_episodes_per_load = int(max(eval_episodes_per_load, 1))
        cfg.training.checkpoint_eval_episodes_per_load = int(max(eval_episodes_per_load, 1))
    if disable_eval:
        cfg.training.eval_every = max(int(cfg.training.total_iterations), 1) + 1
        cfg.training.light_eval_every = 0
        cfg.training.full_eval_every = 0
    if resume_latest and resume_path:
        raise ValueError("Use either resume_latest or resume_path, not both.")
    resume_target = None
    if resume_latest:
        checkpoint_dir = Path(cfg.training.checkpoint_dir)
        pattern = f"{cfg.training.run_name}_*.pt"
        candidates = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} for pattern {pattern}")
        resume_target = candidates[0]
    elif resume_path:
        resume_target = Path(resume_path).expanduser().resolve()
        if not resume_target.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_target}")
    if resume_target is not None and additional_iterations is not None:
        try:
            import torch
            ckpt = torch_load_checkpoint(resume_target, map_location='cpu')
            extra = ckpt.get('extra', {}) or {}
            resume_iter = int(extra.get('iteration', 0))
            cfg.training.total_iterations = int(resume_iter + additional_iterations)
        except Exception:
            pass
    result = run_training_loop(cfg, evaluation_fn=evaluate_dual_selection, resume_path=resume_target)
    markdown_path = write_training_run_markdown(cfg, training_result=result, report_result=None)

    print("SR-MAPPO train-only run")
    print(f"Experiment line: {experiment_label(cfg.training.experiment_line)}")
    print("BC stats:")
    pprint(result.get("bc", {}))
    print("Best evaluation:")
    pprint(result.get("best", {}))
    print(f"Checkpoint dir: {result.get('checkpoint_dir')}")
    print(f"Iterations collected: {len(result.get('history', []))}")
    print(f"Run markdown: {markdown_path}")
    return {
        "training": result,
        "run_markdown": str(markdown_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SR-MAPPO without report generation")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume training from the newest checkpoint for the selected experiment line.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Resume training from a specific checkpoint path.",
    )
    parser.add_argument(
        "--additional-iterations",
        type=int,
        default=None,
        help="If resuming, extend total episodes by this many iterations.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override total training iterations for this run.",
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=None,
        help="Override rollout horizon. Smaller values speed up each iteration almost linearly.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="Override PPO epochs. Fewer epochs reduce update cost.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="Override PPO minibatch size.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override training-time evaluation interval.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Override checkpoint interval.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Override how often training prints the main per-iteration summary line.",
    )
    parser.add_argument(
        "--eval-episodes-per-load",
        type=int,
        default=None,
        help="Override training-time evaluation episodes per load.",
    )
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Disable training-time evaluation for this run.",
    )
    parser.add_argument(
        "--disable-arrival-log",
        dest="disable_arrival_log",
        action="store_true",
        default=None,
        help="Disable per-reset URLLC arrival diagnostic logging for less console overhead.",
    )
    parser.add_argument(
        "--enable-arrival-log",
        dest="disable_arrival_log",
        action="store_false",
        help="Re-enable per-reset URLLC arrival diagnostic logging.",
    )
    args = parser.parse_args()
    run_train_only(
        experiment=args.experiment,
        resume_latest=args.resume_latest,
        resume_path=args.resume_path,
        additional_iterations=args.additional_iterations,
        iterations=args.iterations,
        rollout_horizon=args.rollout_horizon,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        eval_episodes_per_load=args.eval_episodes_per_load,
        disable_eval=args.disable_eval,
        disable_arrival_log=args.disable_arrival_log,
    )
