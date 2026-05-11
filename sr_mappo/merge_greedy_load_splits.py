from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def find_loads(obj: Dict[str, Any]) -> List[float]:
    for k in ("loads", "load_values", "x", "core_kpi_loads"):
        v = obj.get(k)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    rc = obj.get("report_config")
    if isinstance(rc, dict):
        v = rc.get("loads")
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    raise ValueError("Cannot find numeric loads list in JSON")


def align_list_by_load(lst: List[Any], src_loads: List[float], all_loads: List[float]) -> List[Any] | None:
    if len(lst) != len(src_loads):
        return None
    idx = {l: i for i, l in enumerate(src_loads)}
    out = []
    for l in all_loads:
        if l in idx:
            out.append(lst[idx[l]])
        else:
            out.append(None)
    return out


def merge_by_load(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    base_loads = find_loads(base)
    extra_loads = find_loads(extra)
    all_loads = sorted(set(base_loads) | set(extra_loads))

    def merge_node(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, vb in b.items():
                if k in out:
                    out[k] = merge_node(out[k], vb)
                else:
                    out[k] = vb
            return out

        if isinstance(a, list) and isinstance(b, list):
            aa = align_list_by_load(a, base_loads, all_loads)
            bb = align_list_by_load(b, extra_loads, all_loads)
            if aa is not None and bb is not None:
                out = []
                for xa, xb in zip(aa, bb):
                    out.append(xb if xb is not None else xa)
                return out
            return a

        return a

    merged = merge_node(base, extra)

    # write back common load keys
    for k in ("loads", "load_values", "x", "core_kpi_loads"):
        if k in merged and isinstance(merged[k], list):
            merged[k] = list(all_loads)
    if isinstance(merged.get("report_config"), dict) and isinstance(merged["report_config"].get("loads"), list):
        merged["report_config"]["loads"] = list(all_loads)

    merged.setdefault("_merge_info", {})
    merged["_merge_info"]["merged_loads"] = list(all_loads)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge same-share metrics JSONs across multiple load-split folders")
    ap.add_argument("--dirs", nargs="+", required=True, help="Input dirs in order (later dirs override overlapping loads)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    share_files = [
        "share05_mix_3_7_sr_mappo_report_metrics.json",
        "share10_mix_3_7_sr_mappo_report_metrics.json",
        "share15_mix_3_7_sr_mappo_report_metrics.json",
    ]

    for fname in share_files:
        merged = None
        used = []
        for d in dirs:
            p = d / fname
            if not p.exists():
                continue
            cur = load_json(p)
            merged = cur if merged is None else merge_by_load(merged, cur)
            used.append(str(p))

        if merged is None:
            print(f"[skip] no file found for {fname}")
            continue

        merged.setdefault("_merge_info", {})
        merged["_merge_info"]["source_files"] = used

        out_path = out_dir / fname
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
