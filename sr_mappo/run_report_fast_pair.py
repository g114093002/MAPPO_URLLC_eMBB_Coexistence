from __future__ import annotations

import argparse
from sr_mappo.report import generate_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast report: MAPPO vs selected baseline only")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--loads", default="5,10,15,20,25")
    ap.add_argument("--episodes-per-load", type=int, default=20)
    ap.add_argument("--greedy-only", action="store_true")
    args = ap.parse_args()

    loads = [float(x.strip()) for x in args.loads.split(",") if x.strip()]
    result = generate_report(
        loads=loads,
        episodes_per_load=int(args.episodes_per_load),
        fast=True,
        experiment_line=args.experiment,
        checkpoint_path=args.checkpoint_path,
        checkpoint_kind=None,
        greedy_only=bool(args.greedy_only),
    )
    print(result)


if __name__ == "__main__":
    main()
