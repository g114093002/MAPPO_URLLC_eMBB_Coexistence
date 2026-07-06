"""Default training entrypoint for SR-MAPPO."""

import argparse
import importlib.util
from pathlib import Path
from pprint import pprint

from .config import SRMAPPOConfig, torch_load_checkpoint
from .evaluate import evaluate_dual_selection, rescreen_checkpoints_for_report
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .report import generate_report
from .run_docs import write_training_run_markdown
from .trainer import run_training_loop


def _run_policy_diagnostics():
    diagnostics_script = Path(__file__).resolve().parents[1] / 'Greedy' / 'sr_mappo_policy_diagnostics.py'
    if not diagnostics_script.exists():
        return None
    spec = importlib.util.spec_from_file_location('sr_mappo_policy_diagnostics', diagnostics_script)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'generate_policy_diagnostics'):
        return None
    try:
        return module.generate_policy_diagnostics()
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "note": "Training completed, but policy diagnostics failed.",
        }


def run_default_training(
    with_report: bool = False,
    resume_latest: bool = False,
    resume_path: str | None = None,
    additional_iterations: int | None = None,
    experiment: str | None = None,
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
        payload = SRMAPPOConfig()
        try:
            import torch
            ckpt = torch_load_checkpoint(resume_target, map_location='cpu')
            extra = ckpt.get('extra', {}) or {}
            resume_iter = int(extra.get('iteration', 0))
            cfg.training.total_iterations = int(resume_iter + additional_iterations)
        except Exception:
            pass
    result = run_training_loop(cfg, evaluation_fn=evaluate_dual_selection, resume_path=resume_target)
    rescreen_result = None
    report_result = None
    diagnostics_result = None
    markdown_path = None

    if with_report:
        rescreen_result = rescreen_checkpoints_for_report(cfg)
        report_result = generate_report(experiment_line=cfg.training.experiment_line)
        diagnostics_result = _run_policy_diagnostics()
        markdown_path = write_training_run_markdown(cfg, training_result=result, report_result=report_result)

    print("SR-MAPPO default training run")
    print(f"Experiment line: {experiment_label(cfg.training.experiment_line)}")
    print("BC stats:")
    pprint(result.get("bc", {}))
    print("Best evaluation:")
    pprint(result.get("best", {}))
    print(f"Checkpoint dir: {result.get('checkpoint_dir')}")
    if with_report:
        print("Report-best checkpoint selection:")
        pprint(rescreen_result)
        print(f"Episodes collected: {len(result.get('history', []))}")
        print("Generated report artifacts:")
        pprint(report_result)
        print("Generated policy diagnostics:")
        pprint(diagnostics_result)
        print(f"Run markdown: {markdown_path}")
    else:
        print(f"Episodes collected: {len(result.get('history', []))}")
    return {
        "training": result,
        "report": report_result,
        "rescreen": rescreen_result,
        "diagnostics": diagnostics_result,
        "run_markdown": str(markdown_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR-MAPPO training entrypoint")
    parser.add_argument(
        "--with-report",
        action="store_true",
        help="After training, run checkpoint rescreen, report generation, and diagnostics.",
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume training from the newest checkpoint in the checkpoint directory.",
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
        help="If resuming, extend total iterations by this many episodes.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    args = parser.parse_args()
    if args.only_train and args.with_report:
        raise ValueError("Use either --with-report or --only-train, not both.")
    run_default_training(
        with_report=bool(args.with_report) and not bool(args.only_train),
        resume_latest=args.resume_latest,
        resume_path=args.resume_path,
        additional_iterations=args.additional_iterations,
        experiment=args.experiment,
    )
