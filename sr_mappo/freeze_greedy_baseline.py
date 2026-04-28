from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import SRMAPPOConfig
from .experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from .report import (
    DEFAULT_EPISODES_PER_LOAD,
    DEFAULT_LOADS,
    RESULTS_DIR,
    TIMESLOT_SERIES_LOAD,
    TIMESLOT_SERIES_SLOTS,
    _load_checkpoint_cfg,
    _report_log,
    _select_checkpoint,
    run_greedy_timeslot_series,
    run_selected_greedy_sweep,
)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def freeze_greedy_baseline(
    mode: str = "original",
    output_path: str | None = None,
    loads: list[float] | None = None,
    episodes_per_load: int = DEFAULT_EPISODES_PER_LOAD,
    timeslot_load: float = TIMESLOT_SERIES_LOAD,
    timeslot_slots: int = TIMESLOT_SERIES_SLOTS,
    experiment_line: str | None = None,
) -> Path:
    cfg = apply_experiment_preset(SRMAPPOConfig(), experiment_line)
    cfg.training.greedy_baseline_mode = str(mode)
    if cfg.training.greedy_baseline_mode == "frozen_json":
        raise ValueError(
            "freeze_greedy_baseline cannot use mode='frozen_json'. Use "
            "'original', 'original_greedy_normal_v1', 'original_greedy_normal_v2', "
            "'myopic_throughput_greedy', 'matched_fixed_embb', 'throughput_only_greedy', or 'channel_only_greedy'."
        )

    loads = [float(load) for load in (loads or DEFAULT_LOADS)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint, checkpoint_reason = _select_checkpoint(cfg)
    cfg = apply_experiment_preset(_load_checkpoint_cfg(checkpoint), experiment_line)
    cfg.training.greedy_baseline_mode = str(mode)

    _report_log(f"[FREEZE] Selected checkpoint: {checkpoint.name} ({checkpoint_reason})")
    _report_log(f"[FREEZE] Experiment line: {experiment_label(cfg.training.experiment_line)}")
    _report_log(f"[FREEZE] Greedy baseline mode: {cfg.training.greedy_baseline_mode}")
    _report_log(f"[FREEZE] Loads: {loads}")
    _report_log(f"[FREEZE] Episodes per load: {episodes_per_load}")

    greedy_metrics, greedy_representative, _payload = run_selected_greedy_sweep(loads, episodes_per_load, cfg, checkpoint)
    slot_greedy, slot_meta = run_greedy_timeslot_series(timeslot_load, timeslot_slots, checkpoint, cfg=cfg)
    slot_meta_min = {
        "load": float(slot_meta.get("load", timeslot_load)),
        "num_slots": int(slot_meta.get("num_slots", timeslot_slots)),
        "greedy_baseline_mode": str(slot_meta.get("greedy_baseline_mode", cfg.training.greedy_baseline_mode)),
        "checkpoint": str(slot_meta.get("checkpoint", checkpoint)),
    }

    payload = {
        "greedy_baseline_mode": cfg.training.greedy_baseline_mode,
        "experiment_line": cfg.training.experiment_line,
        "checkpoint": str(checkpoint),
        "checkpoint_selection_reason": checkpoint_reason,
        "loads": loads,
        "episodes_per_load": int(episodes_per_load),
        "timeslot_series_load": float(timeslot_load),
        "timeslot_series_slots": int(timeslot_slots),
        "greedy_metrics": greedy_metrics,
        "greedy_representative": greedy_representative,
        "slot_greedy": slot_greedy,
        "slot_meta": slot_meta_min,
    }

    destination = Path(output_path) if output_path else RESULTS_DIR / f"frozen_greedy_{cfg.training.greedy_baseline_mode}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _report_log(f"[FREEZE] Frozen greedy baseline saved: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a greedy baseline payload for later SR-MAPPO reports.")
    parser.add_argument(
        "--mode",
        choices=[
            "original",
            "original_greedy_normal_v1",
            "original_greedy_normal_v2",
            "myopic_throughput_greedy",
            "matched_fixed_embb",
            "throughput_feasible_oracle",
            "throughput_biased_greedy",
            "throughput_only_greedy",
            "channel_only_greedy",
        ],
        default="original",
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--episodes-per-load", type=int, default=DEFAULT_EPISODES_PER_LOAD)
    parser.add_argument("--timeslot-load", type=float, default=TIMESLOT_SERIES_LOAD)
    parser.add_argument("--timeslot-slots", type=int, default=TIMESLOT_SERIES_SLOTS)
    parser.add_argument("--loads", type=float, nargs="*", default=None)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    args = parser.parse_args()

    freeze_greedy_baseline(
        mode=args.mode,
        output_path=args.output or None,
        loads=args.loads,
        episodes_per_load=int(args.episodes_per_load),
        timeslot_load=float(args.timeslot_load),
        timeslot_slots=int(args.timeslot_slots),
        experiment_line=args.experiment,
    )


if __name__ == "__main__":
    main()
