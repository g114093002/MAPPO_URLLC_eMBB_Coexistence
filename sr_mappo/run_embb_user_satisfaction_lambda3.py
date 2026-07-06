from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .run_unified_lambda_stress import _jsonify, _parse_csv_floats, _parse_csv_strings
from .unified_policy_runner import MIX_PRESETS, run_policy


DISPLAY_NAMES = {
    "mappo": "MAPPO",
    "greedy": "Greedy",
    "pure_puncturing": "Pure puncturing",
    "pure_superposition": "Pure superposition",
    "naive_random": "Random scheduling",
}


def _build_policy_config(
    args: argparse.Namespace,
    *,
    policy: str,
    total_load: float,
    mix_ratio: float,
    lam: float,
    packet_bits: int | None,
    target_error_probability: float | None,
    channel_uses: int | None,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "total_load": float(total_load),
        "mix_ratio": float(mix_ratio),
        "simulation": {
            "fixed_urllc_poisson_rate": True,
            "urllc_poisson_rate": float(lam),
            "urllc_user_ratio": float(mix_ratio),
        },
        "env": {
            "urllc_poisson_rate_is_per_user": True,
            "urllc_poisson_rate_is_slot_level": False,
        },
    }
    urllc_cfg: Dict[str, object] = {}
    if target_error_probability is not None:
        urllc_cfg["target_error_probability"] = float(target_error_probability)
    if packet_bits is not None and int(packet_bits) > 0:
        urllc_cfg["packet_lengths"] = [int(packet_bits), int(packet_bits), int(packet_bits)]
    if urllc_cfg:
        cfg["urllc"] = urllc_cfg

    system_cfg: Dict[str, object] = {}
    if channel_uses is not None and int(channel_uses) > 0:
        system_cfg["channel_uses_per_minislot"] = int(channel_uses)
    if args.system_num_subcarriers is not None:
        system_cfg["num_subcarriers"] = int(args.system_num_subcarriers)
    if args.system_num_minislots is not None:
        system_cfg["num_minislots"] = int(args.system_num_minislots)
    if system_cfg:
        cfg["system"] = system_cfg
    if policy.startswith("mappo"):
        if not args.mappo_checkpoint_path:
            raise ValueError(f"Policy '{policy}' requires --mappo-checkpoint-path.")
        cfg["checkpoint_path"] = str(Path(args.mappo_checkpoint_path).expanduser())
    return cfg


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_cdf(out_path: Path, rows: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    methods = []
    for row in rows:
        method = str(row.get("method", ""))
        if method not in methods:
            methods.append(method)
    for method in methods:
        vals = [
            float(row["embb_min_rate_satisfaction_ratio"])
            for row in rows
            if str(row.get("method", "")) == method and np.isfinite(float(row.get("embb_min_rate_satisfaction_ratio", float("nan"))))
        ]
        if not vals:
            continue
        xs = np.sort(np.asarray(vals, dtype=float))
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(xs, ys, linewidth=2.2, label=DISPLAY_NAMES.get(method, method))
    ax.set_title("User-level eMBB min-rate satisfaction CDF at λ = 3")
    ax.set_xlabel("eMBB min-rate satisfaction ratio")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_rate_cdf(
    out_path: Path,
    rows: List[Dict[str, object]],
    *,
    field: str,
    title: str,
    xlabel: str,
    r_min_mbps: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    methods = []
    for row in rows:
        method = str(row.get("method", ""))
        if method not in methods:
            methods.append(method)
    for method in methods:
        vals = [
            float(row[field])
            for row in rows
            if str(row.get("method", "")) == method and np.isfinite(float(row.get(field, float("nan"))))
        ]
        if not vals:
            continue
        xs = np.sort(np.asarray(vals, dtype=float))
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(xs, ys, linewidth=2.2, label=DISPLAY_NAMES.get(method, method))
    ax.axvline(float(r_min_mbps), color="black", linestyle="--", linewidth=1.5, label=f"R_min = {float(r_min_mbps):g} Mbps")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_simple_cdf(
    out_path: Path,
    rows: List[Dict[str, object]],
    *,
    field: str,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    methods = []
    for row in rows:
        method = str(row.get("method", ""))
        if method not in methods:
            methods.append(method)
    for method in methods:
        vals = [
            float(row[field])
            for row in rows
            if str(row.get("method", "")) == method and np.isfinite(float(row.get(field, float("nan"))))
        ]
        if not vals:
            continue
        xs = np.sort(np.asarray(vals, dtype=float))
        ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
        ax.plot(xs, ys, linewidth=2.2, label=DISPLAY_NAMES.get(method, method))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate user-level eMBB min-rate satisfaction ratio CDF at lambda=3 under fixed finite-blocklength settings."
    )
    parser.add_argument("--policies", default="mappo,greedy")
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--load", type=float, default=24.0)
    parser.add_argument("--lambda-value", type=float, default=3.0)
    parser.add_argument("--packet-bits", type=int, default=0)
    parser.add_argument("--target-error-probability", type=float, default=-1.0)
    parser.add_argument("--channel-uses", type=int, default=0)
    parser.add_argument("--system-num-subcarriers", type=int, default=None)
    parser.add_argument("--system-num-minislots", type=int, default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--out-dir", default="sr_mappo/results/embb_user_satisfaction_lambda3")
    parser.add_argument("--mappo-checkpoint-path", default=None)
    args = parser.parse_args()

    mix_name = str(args.mix).strip()
    if mix_name not in MIX_PRESETS:
        raise ValueError(f"Unsupported mix={mix_name!r}. Allowed={sorted(MIX_PRESETS)}")

    policies = _parse_csv_strings(args.policies)
    seeds = [int(round(v)) for v in _parse_csv_floats(args.seeds)]
    mix_ratio = float(MIX_PRESETS[mix_name])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_user_rows: List[Dict[str, object]] = []
    raw_trace_preview: List[Dict[str, object]] = []
    summary_payload: Dict[str, object] = {
        "meta": {
            "mix": mix_name,
            "load": float(args.load),
            "lambda": float(args.lambda_value),
            "packet_bits": (None if int(args.packet_bits) <= 0 else int(args.packet_bits)),
            "target_error_probability": (None if float(args.target_error_probability) < 0.0 else float(args.target_error_probability)),
            "channel_uses": (None if int(args.channel_uses) <= 0 else int(args.channel_uses)),
            "system_num_subcarriers": args.system_num_subcarriers,
            "system_num_minislots": args.system_num_minislots,
            "time_granularity": "minislot-level",
            "denominator_definition": "active scheduled eMBB transmission minislot instants with embb_rate_mbps > 0 after local puncture deduction",
            "r_min_definition": "embb min rate threshold from config, reported in Mbps",
            "seeds": list(seeds),
            "policies": list(policies),
        },
        "methods": {},
    }

    for policy in policies:
        method_rows: List[Dict[str, object]] = []
        for seed in seeds:
            cfg = _build_policy_config(
                args,
                policy=policy,
                total_load=float(args.load),
                mix_ratio=float(mix_ratio),
                lam=float(args.lambda_value),
                packet_bits=(None if int(args.packet_bits) <= 0 else int(args.packet_bits)),
                target_error_probability=(None if float(args.target_error_probability) < 0.0 else float(args.target_error_probability)),
                channel_uses=(None if int(args.channel_uses) <= 0 else int(args.channel_uses)),
            )
            result = run_policy(policy, cfg, int(seed))
            trace = list(result.get("embb_user_minrate_trace", []) or [])
            raw_summary = dict(result.get("raw_summary", {}) or {})
            if trace and len(raw_trace_preview) < 20:
                raw_trace_preview.extend(trace[: max(0, 20 - len(raw_trace_preview))])
            grouped: Dict[tuple, List[Dict[str, object]]] = {}
            for item in trace:
                key = (
                    str(item.get("method", policy)),
                    int(item.get("seed", seed)),
                    int(item.get("episode", 1)),
                    int(item.get("embb_user_id", -1)),
                )
                grouped.setdefault(key, []).append(item)
            embb_user_count = int(round(float(raw_summary.get("embb_user_count", 0.0) or 0.0)))
            if embb_user_count <= 0:
                rates_arr = np.asarray(
                    raw_summary.get(
                        "embb_user_rates_after_puncture_deduction",
                        raw_summary.get("embb_user_rates", []),
                    )
                    or [],
                    dtype=float,
                )
                embb_user_count = int(rates_arr.size)
            total_minislots = (
                int(trace[0].get("episode_total_minislots", 0) or 0)
                if trace
                else int(args.system_num_minislots or 8)
            )
            r_min_mbps_default = float(
                (trace[0].get("r_min_mbps", 0.0) if trace else 0.0) or 0.0
            )
            method_name = str(getattr(next(iter(trace), {}), "get", lambda *_: policy)("method", policy)) if trace else str(policy)
            episode_id = int(getattr(next(iter(trace), {}), "get", lambda *_: 1)("episode", 1)) if trace else 1
            for embb_user_id in range(max(embb_user_count, 0)):
                items = grouped.get((str(policy), int(seed), int(episode_id), int(embb_user_id)), [])
                if not items:
                    # Some methods may never serve certain eMBB users. Keep them in the
                    # per-user table with zero service so service-ratio/all-time CDFs can
                    # reflect starvation instead of silently dropping those users.
                    items = grouped.get((method_name, int(seed), int(episode_id), int(embb_user_id)), [])
                rates = [float(item.get("embb_rate_mbps", 0.0) or 0.0) for item in items]
                sats = [int(item.get("is_satisfied", 0) or 0) for item in items]
                r_min_mbps = float(items[0].get("r_min_mbps", r_min_mbps_default) or r_min_mbps_default) if items else float(r_min_mbps_default)
                num_eval = int(len(rates))
                num_sat = int(sum(sats))
                ratio = float(num_sat / num_eval) if num_eval > 0 else float("nan")
                service_ratio = float(num_eval / total_minislots) if total_minislots > 0 else float("nan")
                all_time_avg_rate = float(sum(rates) / total_minislots) if total_minislots > 0 else float("nan")
                row = {
                    "method": str(policy),
                    "seed": int(seed),
                    "episode": int(episode_id),
                    "embb_user_id": int(embb_user_id),
                    "time_granularity": "minislot-level",
                    "r_min_mbps": float(r_min_mbps),
                    "total_time_instants": int(total_minislots),
                    "num_active_service_instants": int(num_eval),
                    "embb_service_ratio": float(service_ratio) if np.isfinite(service_ratio) else np.nan,
                    "all_time_average_embb_rate_mbps": float(all_time_avg_rate) if np.isfinite(all_time_avg_rate) else np.nan,
                    "num_satisfied_instants": int(num_sat),
                    "num_evaluated_instants": int(num_eval),
                    "embb_min_rate_satisfaction_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                    "mean_embb_rate_mbps": float(np.mean(np.asarray(rates, dtype=float))) if rates else np.nan,
                    "p5_embb_rate_mbps": _percentile(rates, 5),
                    "p10_embb_rate_mbps": _percentile(rates, 10),
                    "median_embb_rate_mbps": _percentile(rates, 50),
                    "min_embb_rate_mbps": float(np.min(np.asarray(rates, dtype=float))) if rates else np.nan,
                }
                method_rows.append(row)
                per_user_rows.append(row)

            print(
                f"[USER-SAT] policy={policy} seed={int(seed)} users={len(grouped)} "
                f"eval_instants={sum(len(v) for v in grouped.values())} "
                f"embb_minrate_episode_ratio={float(raw_summary.get('embb_min_rate_satisfaction_after_puncture_deduction', raw_summary.get('embb_min_rate_satisfaction_ratio', 0.0)) or 0.0):.4f}",
                flush=True,
            )

        valid_rows = [
            row for row in method_rows
            if int(row.get("num_evaluated_instants", 0) or 0) > 0
            and np.isfinite(float(row.get("embb_min_rate_satisfaction_ratio", np.nan)))
        ]
        sat_values = [float(row["embb_min_rate_satisfaction_ratio"]) for row in valid_rows]
        mean_rate_values = [float(row["mean_embb_rate_mbps"]) for row in valid_rows if np.isfinite(float(row["mean_embb_rate_mbps"]))]
        method_summary = {
            "method": str(policy),
            "num_users_in_cdf": int(len(valid_rows)),
            "mean_satisfaction_ratio": float(np.mean(np.asarray(sat_values, dtype=float))) if sat_values else np.nan,
            "std_satisfaction_ratio": float(np.std(np.asarray(sat_values, dtype=float))) if sat_values else np.nan,
            "p5_satisfaction_ratio": _percentile(sat_values, 5),
            "p10_satisfaction_ratio": _percentile(sat_values, 10),
            "p25_satisfaction_ratio": _percentile(sat_values, 25),
            "median_satisfaction_ratio": _percentile(sat_values, 50),
            "p75_satisfaction_ratio": _percentile(sat_values, 75),
            "fraction_users_below_0_95": float(np.mean(np.asarray(sat_values, dtype=float) < 0.95)) if sat_values else np.nan,
            "fraction_users_below_0_90": float(np.mean(np.asarray(sat_values, dtype=float) < 0.90)) if sat_values else np.nan,
            "fraction_users_below_0_80": float(np.mean(np.asarray(sat_values, dtype=float) < 0.80)) if sat_values else np.nan,
            "mean_embb_rate_over_users_mbps": float(np.mean(np.asarray(mean_rate_values, dtype=float))) if mean_rate_values else np.nan,
            "p5_embb_rate_over_users_mbps": _percentile(mean_rate_values, 5),
            "p10_embb_rate_over_users_mbps": _percentile(mean_rate_values, 10),
            "median_embb_rate_over_users_mbps": _percentile(mean_rate_values, 50),
        }
        summary_payload["methods"][policy] = method_summary

    per_user_csv = out_dir / "embb_user_satisfaction_lambda3.csv"
    summary_csv = out_dir / "embb_user_satisfaction_summary_lambda3.csv"
    figure_path = out_dir / "figX_embb_user_satisfaction_cdf_lambda3.png"
    average_rate_figure_path = out_dir / "figY_embb_user_average_rate_cdf_lambda3.png"
    p5_rate_figure_path = out_dir / "figZ_embb_user_p5_rate_cdf_lambda3.png"
    service_ratio_figure_path = out_dir / "figS_embb_user_service_ratio_cdf_lambda3.png"
    all_time_average_rate_figure_path = out_dir / "figT_embb_user_all_time_average_rate_cdf_lambda3.png"
    summary_json = out_dir / "summary.json"

    _write_csv(
        per_user_csv,
        [
            "method",
            "seed",
            "episode",
            "embb_user_id",
            "time_granularity",
            "r_min_mbps",
            "total_time_instants",
            "num_active_service_instants",
            "embb_service_ratio",
            "all_time_average_embb_rate_mbps",
            "num_satisfied_instants",
            "num_evaluated_instants",
            "embb_min_rate_satisfaction_ratio",
            "mean_embb_rate_mbps",
            "p5_embb_rate_mbps",
            "p10_embb_rate_mbps",
            "median_embb_rate_mbps",
            "min_embb_rate_mbps",
        ],
        per_user_rows,
    )
    _write_csv(
        summary_csv,
        [
            "method",
            "num_users_in_cdf",
            "mean_satisfaction_ratio",
            "std_satisfaction_ratio",
            "p5_satisfaction_ratio",
            "p10_satisfaction_ratio",
            "p25_satisfaction_ratio",
            "median_satisfaction_ratio",
            "p75_satisfaction_ratio",
            "fraction_users_below_0_95",
            "fraction_users_below_0_90",
            "fraction_users_below_0_80",
            "mean_embb_rate_over_users_mbps",
            "p5_embb_rate_over_users_mbps",
            "p10_embb_rate_over_users_mbps",
            "median_embb_rate_over_users_mbps",
        ],
        [summary_payload["methods"][policy] for policy in policies if policy in summary_payload["methods"]],
    )
    _plot_cdf(figure_path, per_user_rows)
    r_min_mbps = float(per_user_rows[0].get("r_min_mbps", 0.0) or 0.0) if per_user_rows else 0.0
    _plot_rate_cdf(
        average_rate_figure_path,
        per_user_rows,
        field="mean_embb_rate_mbps",
        title="Per-user average eMBB rate CDF at lambda = 3",
        xlabel="Per-user average eMBB rate (Mbps)",
        r_min_mbps=r_min_mbps,
    )
    _plot_rate_cdf(
        p5_rate_figure_path,
        per_user_rows,
        field="p5_embb_rate_mbps",
        title="Per-user 5th-percentile eMBB rate CDF at lambda = 3",
        xlabel="Per-user 5th-percentile eMBB rate (Mbps)",
        r_min_mbps=r_min_mbps,
    )
    _plot_simple_cdf(
        service_ratio_figure_path,
        per_user_rows,
        field="embb_service_ratio",
        title="Per-user eMBB service ratio CDF at lambda = 3",
        xlabel="Per-user eMBB service ratio",
    )
    _plot_simple_cdf(
        all_time_average_rate_figure_path,
        per_user_rows,
        field="all_time_average_embb_rate_mbps",
        title="Per-user all-time average eMBB rate CDF at lambda = 3",
        xlabel="Per-user all-time average eMBB rate (Mbps)",
    )

    summary_payload["raw_trace_preview"] = raw_trace_preview[:10]
    summary_json.write_text(json.dumps(_jsonify(summary_payload), indent=2), encoding="utf-8")

    print("\n[USER-SAT] Summary table", flush=True)
    for policy in policies:
        if policy not in summary_payload["methods"]:
            continue
        row = summary_payload["methods"][policy]
        print(
            f"{DISPLAY_NAMES.get(policy, policy):<20} "
            f"users={int(row['num_users_in_cdf'])} "
            f"p5={float(row['p5_satisfaction_ratio']):.4f} "
            f"p10={float(row['p10_satisfaction_ratio']):.4f} "
            f"median={float(row['median_satisfaction_ratio']):.4f} "
            f"frac<0.9={float(row['fraction_users_below_0_90']):.4f}",
            flush=True,
        )

    invalid_rows = [
        row for row in per_user_rows
        if int(row.get("num_evaluated_instants", 0) or 0) > 0
        and not (0.0 <= float(row.get("embb_min_rate_satisfaction_ratio", np.nan)) <= 1.0)
    ]
    zero_eval_rows = [row for row in per_user_rows if int(row.get("num_evaluated_instants", 0) or 0) <= 0]
    print(
        f"\n[USER-SAT] debug rows={len(per_user_rows)} invalid_ratio_rows={len(invalid_rows)} zero_eval_rows={len(zero_eval_rows)}",
        flush=True,
    )
    print(f"[USER-SAT] wrote {per_user_csv}", flush=True)
    print(f"[USER-SAT] wrote {summary_csv}", flush=True)
    print(f"[USER-SAT] wrote {figure_path}", flush=True)
    print(f"[USER-SAT] wrote {average_rate_figure_path}", flush=True)
    print(f"[USER-SAT] wrote {p5_rate_figure_path}", flush=True)
    print(f"[USER-SAT] wrote {service_ratio_figure_path}", flush=True)
    print(f"[USER-SAT] wrote {all_time_average_rate_figure_path}", flush=True)


if __name__ == "__main__":
    main()
