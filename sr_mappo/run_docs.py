"""Helpers for writing human-readable SR-MAPPO run documentation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional


PACKAGE_DIR = Path("d:/URLLC_eMBB_Coexisting/sr_mappo")
RESULTS_DIR = PACKAGE_DIR / "results"


def _fmt_float(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "True" if value else "False"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _artifact_lines(report_result: Optional[Dict]) -> str:
    if not report_result:
        return "- No report artifacts were generated in this run."
    lines = []
    for key, value in report_result.items():
        if isinstance(value, list):
            lines.append(f"- `{key}`:")
            for item in value:
                lines.append(f"  - `{item}`")
        else:
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def _training_summary_lines(training_result: Optional[Dict]) -> str:
    if not training_result:
        return "- No training result summary was provided."

    bc = training_result.get("bc") or {}
    best = training_result.get("best") or {}
    history = training_result.get("history") or []
    lines = [
        f"- Iterations completed: `{len(history)}`",
        f"- Checkpoint directory: `{training_result.get('checkpoint_dir', 'N/A')}`",
        f"- BC epochs actually run: `{bc.get('bc_epochs', 'N/A')}`",
        f"- BC dataset size: `{bc.get('bc_samples', 'N/A')}`",
    ]
    if best:
        lines.extend([
            f"- Best evaluation score: `{_fmt_float(best.get('policy_score'))}`",
            f"- Best evaluation eMBB throughput: `{_fmt_float(best.get('policy_mean_embb_rate', 0.0) / 1e6)} Mbps`",
            f"- Best evaluation URLLC scheduled ratio: `{_fmt_float(best.get('policy_mean_scheduled_ratio'))}`",
            f"- Best evaluation power: `{_fmt_float(best.get('policy_mean_power', 0.0) * 1e3)} mW`",
            f"- Non-worse-than-greedy gate: `{best.get('non_worse_than_greedy', 'N/A')}`",
        ])
    return "\n".join(lines)


def _config_sections(cfg) -> str:
    action = asdict(cfg.action)
    reward = asdict(cfg.reward)
    shield = asdict(cfg.shield)
    env = asdict(cfg.env)
    network = asdict(cfg.network)
    training = asdict(cfg.training)

    def section(title: str, values: Dict) -> str:
        body = "\n".join(f"- `{k}`: `{v}`" for k, v in values.items())
        return f"### {title}\n{body}"

    return "\n\n".join([
        section("Action Space Parameters", action),
        section("Reward Parameters", reward),
        section("Shield Parameters", shield),
        section("Environment Adapter Parameters", env),
        section("Network Parameters", network),
        section("Training Parameters", training),
    ])


def write_training_run_markdown(cfg, training_result: Optional[Dict] = None, report_result: Optional[Dict] = None, path: Optional[Path] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (RESULTS_DIR / "LATEST_TRAINING_RUN.md")
    content = f"""# SR-MAPPO Latest Training Run

This file is auto-generated after `python -m sr_mappo.train` finishes. It records the exact configuration that was used, the high-level training outcome, and the report artifacts that were generated immediately after training.

## Auto-plot Behavior

- The default training entrypoint now **always** runs report generation after training.
- This means every completed training run will also regenerate the latest KPI and diagnostic figures in `sr_mappo/results`.
- The plotting entry that is triggered automatically is `python -m sr_mappo.report` through the internal `generate_report()` function.

## Training Outcome Summary

## Run Identity

- `experiment_line`: `{cfg.training.experiment_line}`
- `run_name`: `{cfg.training.run_name}`
- `greedy_baseline_mode`: `{cfg.training.greedy_baseline_mode}`

{_training_summary_lines(training_result)}

## Generated Artifacts

{_artifact_lines(report_result)}

## Full Configuration Used In This Run

{_config_sections(cfg)}
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_band_expert_markdown(band_summaries: Dict[str, Dict], report_result: Optional[Dict] = None, cfgs: Optional[Dict[str, object]] = None, path: Optional[Path] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (RESULTS_DIR / "LATEST_BAND_EXPERT_RUN.md")
    lines = [
        "# SR-MAPPO Latest Band-Expert Run",
        "",
        "This file is auto-generated after `python -m sr_mappo.band_experts` finishes.",
        "It records the per-band training setup and the report artifacts regenerated at the end of the banded run.",
        "",
        "## Generated Artifacts",
        "",
        _artifact_lines(report_result),
        "",
        "## Per-Band Training Summaries",
        "",
    ]
    for band_name, summary in (band_summaries or {}).items():
        lines.append(f"### `{band_name}` band")
        lines.append(_training_summary_lines(summary))
        cfg = (cfgs or {}).get(band_name)
        if cfg is not None:
            lines.append("")
            lines.append(_config_sections(cfg))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
