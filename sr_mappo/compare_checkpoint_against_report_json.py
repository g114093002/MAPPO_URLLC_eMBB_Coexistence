from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_mappo.config import SRMAPPOConfig
from sr_mappo.experiments import EXPERIMENT_CHOICES, apply_experiment_preset
from sr_mappo.report import run_mappo_sweep


def _parse_loads(text: str) -> List[float]:
    loads: List[float] = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue
        loads.append(float(token))
    if not loads:
        raise ValueError("No loads provided.")
    return loads


def _series(payload: Dict, key: str) -> np.ndarray:
    return np.asarray(payload.get(key, []), dtype=float)


def _mbps(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1.0e6


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _label_for_section(section_name: str) -> str:
    return "reference.sr_mappo" if section_name == "sr_mappo" else f"reference.{section_name}"


def _plot_core_kpis(
    candidate_metrics: Dict,
    reference_metrics: Dict,
    reference_section_name: str,
    output_path: Path,
) -> None:
    candidate_loads = _series(candidate_metrics, "loads")
    reference_loads = _series(reference_metrics, "loads")
    ref_label = _label_for_section(reference_section_name)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    panels: List[Tuple[str, str, bool]] = [
        ("Aggregate eMBB throughput", "embb_rate", True),
        ("URLLC admission ratio", "urllc_admission", False),
        ("Admitted URLLC reliability", "admitted_urllc_reliability", False),
        ("eMBB served ratio", "embb_service_ratio", False),
        ("Per-user eMBB rate", "embb_user_rate", True),
        ("Total transmit power", "total_power", False),
    ]

    for ax, (title, key, use_mbps) in zip(axes.flat, panels):
        cand_y = _series(candidate_metrics, key)
        ref_y = _series(reference_metrics, key)
        if use_mbps:
            cand_y = _mbps(cand_y)
            ref_y = _mbps(ref_y)
            ylabel = "Mbps"
        else:
            ylabel = "Ratio" if "ratio" in key or "reliability" in key or "admission" in key else "Value"
            if key == "total_power":
                ylabel = "Power"
        ax.plot(candidate_loads, cand_y, marker="s", linewidth=2.0, label="candidate.mappo")
        ax.plot(reference_loads, ref_y, marker="o", linewidth=2.0, linestyle="--", label=ref_label)
        ax.set_title(title)
        ax.set_xlabel("Average UE load per UAV")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Core KPI comparison", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a checkpoint sweep against an existing report JSON.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset used to build the evaluation config.",
    )
    parser.add_argument(
        "--loads",
        type=str,
        default="9,12,15,18,21,24",
        help="Comma-separated loads. Default: 9,12,15,18,21,24",
    )
    parser.add_argument("--episodes-per-load", type=int, default=10, help="Episodes per load. Default: 10")
    parser.add_argument(
        "--reference-json",
        type=str,
        required=True,
        help="Existing report JSON, e.g. sr_mappo_report_metrics.json",
    )
    parser.add_argument(
        "--reference-section",
        type=str,
        default="sr_mappo",
        choices=["sr_mappo", "greedy"],
        help="Which top-level section in the reference JSON to compare against.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for the evaluated JSON and comparison plot.",
    )
    args = parser.parse_args()

    loads = _parse_loads(args.loads)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    reference_json = Path(args.reference_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = apply_experiment_preset(SRMAPPOConfig(), args.experiment)
    cfg.training.eval_loads = list(loads)
    cfg.training.coarse_eval_loads = list(loads)
    cfg.training.dense_eval_loads = list(loads)
    cfg.training.checkpoint_eval_loads = list(loads)

    candidate_metrics, _candidate_rep = run_mappo_sweep(
        loads,
        int(args.episodes_per_load),
        checkpoint_path,
        base_cfg=cfg,
    )

    reference_payload = json.loads(reference_json.read_text(encoding="utf-8"))
    if args.reference_section not in reference_payload:
        raise KeyError(f"Section '{args.reference_section}' not found in {reference_json}")
    reference_metrics = dict(reference_payload[args.reference_section])

    evaluated_payload = {
        "checkpoint": str(checkpoint_path),
        "experiment": str(args.experiment or ""),
        "loads": [float(x) for x in loads],
        "episodes_per_load": int(args.episodes_per_load),
        "candidate": candidate_metrics,
        "reference_json": str(reference_json),
        "reference_section": str(args.reference_section),
        "reference": reference_metrics,
    }
    evaluated_json = out_dir / "candidate_vs_reference_metrics.json"
    evaluated_json.write_text(json.dumps(evaluated_payload, indent=2, default=_json_default), encoding="utf-8")

    figure_path = out_dir / "candidate_vs_reference_core_kpis.png"
    _plot_core_kpis(candidate_metrics, reference_metrics, str(args.reference_section), figure_path)

    print(evaluated_json)
    print(figure_path)


if __name__ == "__main__":
    main()
