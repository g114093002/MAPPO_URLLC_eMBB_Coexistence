from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def merge_obj(a: Any, b: Any) -> Any:
    if is_num(a) and is_num(b):
        return (a + b) / 2.0

    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for x, y in zip(a, b):
            if is_num(x) and is_num(y):
                out.append((x + y) / 2.0)
            else:
                out.append(x)
        return out

    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        out = {}
        for k in keys:
            if k in a and k in b:
                out[k] = merge_obj(a[k], b[k])
            elif k in a:
                out[k] = a[k]
            else:
                out[k] = b[k]
        return out

    return a


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge two greedy-grid result folders by averaging matching JSON metrics files."
    )
    p.add_argument("--part1", required=True, help="Folder path for first run")
    p.add_argument("--part2", required=True, help="Folder path for second run")
    p.add_argument("--out-dir", required=True, help="Output folder for merged JSON files")
    p.add_argument(
        "--pattern",
        default="share*_mix_*_sr_mappo_report_metrics.json",
        help="Glob pattern of metrics JSON files",
    )
    p.add_argument(
        "--episodes-total",
        type=int,
        default=50,
        help="Total episodes per load after merge (metadata only)",
    )
    args = p.parse_args()

    part1 = Path(args.part1)
    part2 = Path(args.part2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files1 = sorted(part1.glob(args.pattern))
    if not files1:
        raise SystemExit(f"No files found in part1 with pattern: {args.pattern}")

    merged_count = 0
    skipped_count = 0

    for f1 in files1:
        f2 = part2 / f1.name
        if not f2.exists():
            print(f"[skip] part2 missing: {f1.name}")
            skipped_count += 1
            continue

        d1 = json.loads(f1.read_text(encoding="utf-8"))
        d2 = json.loads(f2.read_text(encoding="utf-8"))

        merged = merge_obj(d1, d2)
        if isinstance(merged, dict):
            merged.setdefault("_merge_info", {})
            merged["_merge_info"]["source_part1"] = str(f1)
            merged["_merge_info"]["source_part2"] = str(f2)
            merged["_merge_info"]["method"] = "simple_mean"
            merged["_merge_info"]["episodes_per_load_total"] = args.episodes_total

        out_path = out_dir / f1.name
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] merged: {f1.name}")
        merged_count += 1

    print("\nDone")
    print(f"merged={merged_count}, skipped={skipped_count}")
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
