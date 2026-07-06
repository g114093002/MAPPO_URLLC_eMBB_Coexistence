from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_metrics_path(result_dir: Path) -> Path:
    path = result_dir / "unified_lambda_stress_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return path


def _extract_reward_curve(
    payload: Dict[str, object],
    *,
    mix: str,
) -> Tuple[str, List[float], List[float]]:
    raw_runs = payload.get("raw_runs", {})
    if not isinstance(raw_runs, dict) or mix not in raw_runs:
        raise ValueError(f"Mix {mix!r} not found in raw_runs")
    mix_payload = raw_runs[mix]
    if not isinstance(mix_payload, dict) or not mix_payload:
        raise ValueError(f"No policy payloads under mix {mix!r}")
    policy_name = next(iter(mix_payload.keys()))
    policy_payload = mix_payload[policy_name]
    if not isinstance(policy_payload, dict):
        raise ValueError(f"Malformed policy payload for mix {mix!r}")

    lambdas: List[float] = []
    rewards: List[float] = []
    for lambda_key in sorted(policy_payload.keys(), key=lambda value: float(str(value).split("_")[1])):
        runs = policy_payload[lambda_key]
        if not isinstance(runs, list) or not runs:
            continue
        lam = float(str(lambda_key).split("_")[1])
        vals = [float(run.get("episode_reward_mean", 0.0) or 0.0) for run in runs]
        lambdas.append(lam)
        rewards.append(float(np.mean(np.asarray(vals, dtype=float))))
    return policy_name, lambdas, rewards


def plot_reward_compare(
    result_dirs: List[Path],
    *,
    mix: str,
    out_dir: Path,
) -> Dict[str, Path]:
    series = []
    for result_dir in result_dirs:
        payload = _load_metrics(_infer_metrics_path(result_dir))
        policy_name, lambdas, rewards = _extract_reward_curve(payload, mix=mix)
        series.append(
            {
                "label": result_dir.name,
                "policy": policy_name,
                "lambdas": lambdas,
                "rewards": rewards,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for item in series:
        ax.plot(
            item["lambdas"],
            item["rewards"],
            marker="o",
            linewidth=2.0,
            label=f"{item['label']} ({item['policy']})",
        )

    ax.set_title(f"Unified Lambda-Stress Test Reward Compare ({mix})")
    ax.set_xlabel("Per-user lambda0")
    ax.set_ylabel("Mean episode reward")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()

    png_path = out_dir / "episode_reward_mean_vs_lambda_symlog.png"
    json_path = out_dir / "episode_reward_mean_vs_lambda_symlog.json"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps({"series": series, "mix": mix}, indent=2), encoding="utf-8")
    return {"png": png_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot unified lambda-stress test reward curves.")
    parser.add_argument("--result-dirs", nargs="+", required=True, help="Result directories containing unified_lambda_stress_metrics.json")
    parser.add_argument("--mix", required=True, help="Mix key, e.g. 5:5")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    outputs = plot_reward_compare(
        [Path(item) for item in args.result_dirs],
        mix=str(args.mix),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
