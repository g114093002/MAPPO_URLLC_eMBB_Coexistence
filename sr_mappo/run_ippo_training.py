from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SRMAPPOConfig
from .ippo_baseline import IPPOBaselineTrainer


def _jsonify(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if np is not None and isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone IPPO baseline.")
    parser.add_argument("--total-load", type=float, default=20.0)
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-scope", default="global", choices=["global", "local"])
    parser.add_argument("--out-dir", default="sr_mappo/checkpoints/ippo")
    parser.add_argument("--run-name", default="ippo_mix55_load20")
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--num-minislots", type=int, default=None)
    args = parser.parse_args()

    mix_map = {
        "7:3": 0.3,
        "5:5": 0.5,
        "3:7": 0.7,
        "10:0": 0.0,
    }
    if args.mix not in mix_map:
        raise ValueError(f"Unsupported mix={args.mix!r}. Allowed={sorted(mix_map)}")

    cfg = SRMAPPOConfig()
    overrides = {}
    if args.num_subcarriers is not None or args.num_minislots is not None:
        overrides["system"] = {}
        if args.num_subcarriers is not None:
            overrides["system"]["num_subcarriers"] = int(args.num_subcarriers)
        if args.num_minislots is not None:
            overrides["system"]["num_minislots"] = int(args.num_minislots)

    trainer = IPPOBaselineTrainer(cfg, component_overrides=overrides)
    result = trainer.train(
        seed=int(args.seed),
        total_load=float(args.total_load),
        mix_ratio=float(mix_map[args.mix]),
        iterations=int(args.iterations),
        reward_scope=str(args.reward_scope),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{args.run_name}.pt"
    metrics_path = out_dir / f"{args.run_name}_training_curve.json"
    trainer.save(checkpoint_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonify(
            {
                "run_name": str(args.run_name),
                "mix": str(args.mix),
                "total_load": float(args.total_load),
                "seed": int(args.seed),
                "iterations": int(args.iterations),
                "reward_scope": str(args.reward_scope),
                "training_curve": list(result.get("training_curve", []) or []),
                "final_summary": dict(result.get("summary", {}) or {}),
            }),
            handle,
            indent=2,
        )

    print(f"[IPPO] checkpoint: {checkpoint_path}", flush=True)
    print(f"[IPPO] training curve: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
