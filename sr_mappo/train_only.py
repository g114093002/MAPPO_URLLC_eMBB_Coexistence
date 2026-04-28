"""Train SR-MAPPO without auto-running report and diagnostics."""

import argparse
from pathlib import Path
from pprint import pprint

from .config import SRMAPPOConfig
from .evaluate import evaluate_dual_selection
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .run_docs import write_training_run_markdown
from .trainer import run_training_loop


def run_train_only(
    experiment: str | None = None,
    resume_latest: bool = False,
    resume_path: str | None = None,
    additional_iterations: int | None = None,
):
    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment)
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
            ckpt = torch.load(resume_target, map_location='cpu')
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
    args = parser.parse_args()
    run_train_only(
        experiment=args.experiment,
        resume_latest=args.resume_latest,
        resume_path=args.resume_path,
        additional_iterations=args.additional_iterations,
    )
