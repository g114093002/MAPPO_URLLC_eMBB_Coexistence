from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "sr_mappo" / "results"
DEFAULT_METRICS_PATH = DEFAULT_RESULTS_ROOT / "sr_mappo_report_metrics.json"


def _iter_checkpoint_paths(pattern_prefix: str, start: int, end: int, step: int) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for it in range(start, end + 1, step):
        ckpt = PROJECT_ROOT / "checkpoints" / f"{pattern_prefix}_iter{it}.pt"
        if ckpt.exists():
            pairs.append((it, ckpt))
    return pairs


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _fmt(x: float, nd: int = 6) -> str:
    if x != x:
        return "nan"
    return f"{x:.{nd}f}"


def run() -> int:
    parser = argparse.ArgumentParser(description="Scan SR-MAPPO checkpoints by repeatedly running sr_mappo.report and collecting summary CSV.")
    parser.add_argument("--experiment", required=True, help="Experiment preset name used by sr_mappo.report")
    parser.add_argument("--run-name-prefix", required=True, help="Checkpoint filename prefix without _iterXXXX.pt")
    parser.add_argument("--loads", default="10,20,25", help="Loads override passed via env (default: 10,20,25)")
    parser.add_argument("--episodes-per-load", type=int, default=2, help="Episodes per load override (default: 2)")
    parser.add_argument("--start-iter", type=int, default=100, help="Start iteration (default: 100)")
    parser.add_argument("--end-iter", type=int, default=2400, help="End iteration (default: 2400)")
    parser.add_argument("--step", type=int, default=100, help="Iteration step (default: 100)")
    parser.add_argument("--out-dir", default="", help="Output dir under sr_mappo/results (default: auto-generated)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (DEFAULT_RESULTS_ROOT / f"scan_{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _iter_checkpoint_paths(args.run_name_prefix, args.start_iter, args.end_iter, args.step)
    if not checkpoints:
        print("No checkpoints found for the given prefix/range.")
        return 2

    rows: list[dict[str, str]] = []
    for it, ckpt in checkpoints:
        print(f"[SCAN] iter={it} -> {ckpt.name}")
        env = os.environ.copy()
        env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(args.loads)
        env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(args.episodes_per_load))

        cmd = [
            sys.executable,
            "-m",
            "sr_mappo.report",
            "--experiment",
            args.experiment,
            "--fast",
            "--checkpoint-path",
            str(ckpt),
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        if proc.returncode != 0:
            rows.append(
                {
                    "iter": str(it),
                    "status": f"report_failed_rc_{proc.returncode}",
                    "mappo_adm_mean": "nan",
                    "greedy_adm_mean": "nan",
                    "mappo_embb_mean_mbps": "nan",
                    "greedy_embb_mean_mbps": "nan",
                }
            )
            continue

        if not DEFAULT_METRICS_PATH.exists():
            rows.append(
                {
                    "iter": str(it),
                    "status": "metrics_missing",
                    "mappo_adm_mean": "nan",
                    "greedy_adm_mean": "nan",
                    "mappo_embb_mean_mbps": "nan",
                    "greedy_embb_mean_mbps": "nan",
                }
            )
            continue

        iter_dir = out_dir / f"iter_{it}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        dst_metrics = iter_dir / "sr_mappo_report_metrics.json"
        shutil.copy2(DEFAULT_METRICS_PATH, dst_metrics)

        with dst_metrics.open("r", encoding="utf-8") as f:
            j = json.load(f)

        m_adm = [float(x) for x in j.get("sr_mappo", {}).get("urllc_admission", [])]
        g_adm = [float(x) for x in j.get("greedy", {}).get("urllc_admission", [])]
        m_embb = [float(x) / 1e6 for x in j.get("sr_mappo", {}).get("embb_rate", [])]
        g_embb = [float(x) / 1e6 for x in j.get("greedy", {}).get("embb_rate", [])]

        row = {
            "iter": str(it),
            "status": "ok",
            "mappo_adm_mean": _fmt(_mean(m_adm), 6),
            "greedy_adm_mean": _fmt(_mean(g_adm), 6),
            "mappo_embb_mean_mbps": _fmt(_mean(m_embb), 3),
            "greedy_embb_mean_mbps": _fmt(_mean(g_embb), 3),
        }
        rows.append(row)
        print(
            f"[SCAN] iter={it} adm(m/g)={row['mappo_adm_mean']}/{row['greedy_adm_mean']} "
            f"embb(m/g)={row['mappo_embb_mean_mbps']}/{row['greedy_embb_mean_mbps']} Mbps"
        )

    csv_path = out_dir / "scan_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iter",
                "status",
                "mappo_adm_mean",
                "greedy_adm_mean",
                "mappo_embb_mean_mbps",
                "greedy_embb_mean_mbps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SCAN] done. summary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
