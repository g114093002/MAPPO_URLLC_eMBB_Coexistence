from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .unified_policy_runner import MIX_PRESETS, run_policy


DEFAULT_GREEDY_POLICY = "hard_feasible_throughput_greedy"


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_mix_ratio(raw: str) -> float:
    mix = str(raw or "").strip()
    if not mix:
        raise ValueError("Mix string must be non-empty.")
    if mix in MIX_PRESETS:
        return float(MIX_PRESETS[mix])
    if ":" not in mix:
        raise ValueError(f"Unsupported mix={mix!r}. Use a preset {sorted(MIX_PRESETS)} or a custom 'a:b' ratio.")
    left_raw, right_raw = mix.split(":", 1)
    try:
        left = float(left_raw.strip())
        right = float(right_raw.strip())
    except ValueError as exc:
        raise ValueError(f"Unsupported mix={mix!r}. Custom mixes must be numeric 'a:b' ratios.") from exc
    if left < 0.0 or right < 0.0:
        raise ValueError(f"Unsupported mix={mix!r}. Custom mixes must be non-negative.")
    total = float(left + right)
    if total <= 0.0:
        raise ValueError(f"Unsupported mix={mix!r}. At least one side of the mix must be positive.")
    return float(right / total)


def _parse_mix_weights(raw: str) -> Tuple[float, float]:
    mix = str(raw or "").strip()
    if not mix:
        raise ValueError("Mix string must be non-empty.")
    if mix in MIX_PRESETS:
        ratio = float(MIX_PRESETS[mix])
        return (float(1.0 - ratio), float(ratio))
    if ":" not in mix:
        raise ValueError(f"Unsupported mix={mix!r}. Use a preset {sorted(MIX_PRESETS)} or a custom 'a:b' ratio.")
    left_raw, right_raw = mix.split(":", 1)
    try:
        left = float(left_raw.strip())
        right = float(right_raw.strip())
    except ValueError as exc:
        raise ValueError(f"Unsupported mix={mix!r}. Custom mixes must be numeric 'a:b' ratios.") from exc
    if left < 0.0 or right < 0.0:
        raise ValueError(f"Unsupported mix={mix!r}. Custom mixes must be non-negative.")
    if (left + right) <= 0.0:
        raise ValueError(f"Unsupported mix={mix!r}. At least one side of the mix must be positive.")
    return (float(left), float(right))


def _parse_checkpoint_specs(paths_raw: str, labels_raw: Optional[str]) -> List[Tuple[str, str]]:
    paths = _parse_csv_strings(paths_raw)
    if not paths:
        return []
    labels = _parse_csv_strings(labels_raw or "")
    if labels and len(labels) != len(paths):
        raise ValueError("--mappo-labels count must match --mappo-checkpoint-paths count.")
    specs: List[Tuple[str, str]] = []
    for idx, checkpoint_path in enumerate(paths):
        label = labels[idx] if idx < len(labels) else Path(checkpoint_path).stem
        specs.append((label, checkpoint_path))
    return specs


def _urllc_throughput_mbps(metrics: Dict[str, object]) -> float:
    summary = dict(metrics.get("raw_summary", {}) or {})
    packet_bits = float(summary.get("urllc_packet_bits_mean", 0.0) or 0.0)
    slot_duration_s = float(summary.get("urllc_slot_duration_s", 0.0) or 0.0)
    scheduled_packets = float(metrics.get("scheduled_urllc_packets", 0.0) or 0.0)
    if packet_bits <= 0.0 or slot_duration_s <= 0.0:
        return 0.0
    return float((scheduled_packets * packet_bits) / max(slot_duration_s, 1.0e-9) / 1.0e6)


def _safe_array(values: object) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    return arr.astype(float, copy=False).ravel()


def _extract_embb_rate_vector(summary: Dict[str, object]) -> np.ndarray:
    rates = _safe_array(summary.get("embb_user_rates_after_puncture_deduction", None))
    if rates.size > 0:
        return rates
    rates = _safe_array(summary.get("embb_user_rates", None))
    if rates.size > 0:
        return rates
    return np.asarray([], dtype=float)


def _build_policy_config(
    *,
    total_load: float,
    mix_ratio: float,
    explicit_mix_weights: Optional[Tuple[float, float]] = None,
    activation_prob: float,
    policy: str,
    mappo_checkpoint_path: Optional[str] = None,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "total_load": float(total_load),
        "mix_ratio": float(mix_ratio),
        "simulation": {
            "fixed_urllc_poisson_rate": True,
            "urllc_poisson_rate": float(activation_prob),
            "urllc_user_ratio": float(mix_ratio),
        },
        "env": {
            "urllc_poisson_rate_is_per_user": True,
            "urllc_poisson_rate_is_slot_level": False,
            "urllc_arrival_mode": "bernoulli",
            "urllc_bernoulli_tx_prob": 0.5,
        },
    }
    if explicit_mix_weights is not None:
        cfg["explicit_mix_weights"] = [
            float(explicit_mix_weights[0]),
            float(explicit_mix_weights[1]),
        ]
    if policy == "mappo":
        if not mappo_checkpoint_path:
            raise ValueError("MAPPO evaluation requires a checkpoint path.")
        cfg["checkpoint_path"] = str(Path(mappo_checkpoint_path).expanduser())
    return cfg


def _episode_row(
    *,
    metrics: Dict[str, object],
    method: str,
    checkpoint_name: str,
    stage: str,
    mix_name: str,
    total_load: float,
    urllc_activation_prob: float,
    embb_min_rate_bps: float,
) -> Dict[str, object]:
    summary = dict(metrics.get("raw_summary", {}) or {})
    embb_rates_bps = _extract_embb_rate_vector(summary)
    embb_rates_mbps = embb_rates_bps / 1.0e6
    embb_min_rate_mbps = float(embb_min_rate_bps) / 1.0e6
    if embb_rates_mbps.size > 0 and embb_min_rate_mbps > 0.0:
        deficits_mbps = np.maximum(embb_min_rate_mbps - embb_rates_mbps, 0.0)
        normalized_deficits = np.maximum(
            1.0 - (embb_rates_mbps / max(embb_min_rate_mbps, 1.0e-12)),
            0.0,
        )
        satisfied_mask = embb_rates_mbps >= (embb_min_rate_mbps - 1.0e-12)
    else:
        deficits_mbps = np.zeros_like(embb_rates_mbps)
        normalized_deficits = np.zeros_like(embb_rates_mbps)
        satisfied_mask = embb_rates_mbps > 0.0

    embb_user_count = int(embb_rates_mbps.size or round(float(summary.get("embb_user_count", 0.0) or 0.0)))
    satisfied_users = int(np.count_nonzero(satisfied_mask)) if embb_rates_mbps.size > 0 else int(
        round(float(summary.get("embb_min_rate_satisfied_users_after_puncture_deduction", summary.get("embb_min_rate_satisfied_users", 0.0)) or 0.0))
    )
    violation_users = max(embb_user_count - satisfied_users, 0)

    total_power = float(summary.get("total_power", 0.0) or 0.0)
    embb_power = float(summary.get("embb_power", 0.0) or 0.0)
    urllc_power = float(summary.get("urllc_power", max(total_power - embb_power, 0.0)) or 0.0)
    overlay_action_count = float(summary.get("overlay_count", metrics.get("overlay_action_count", 0.0)) or 0.0)
    puncturing_action_count = float(summary.get("puncture_count", metrics.get("puncturing_action_count", 0.0)) or 0.0)
    coexist_action_total = overlay_action_count + puncturing_action_count

    return {
        "stage": str(stage),
        "mix": str(mix_name),
        "method": str(method),
        "checkpoint_name": str(checkpoint_name),
        "target_load": float(total_load),
        "urllc_activation_prob": float(urllc_activation_prob),
        "aggregate_embb_rate": float(metrics.get("total_embb_throughput", 0.0) or 0.0) / 1.0e6,
        "avg_embb_rate": float(metrics.get("average_embb_rate", 0.0) or 0.0) / 1.0e6,
        "min_embb_user_rate": float(np.min(embb_rates_mbps)) if embb_rates_mbps.size > 0 else 0.0,
        "p5_embb_user_rate": float(np.percentile(embb_rates_mbps, 5.0)) if embb_rates_mbps.size > 0 else 0.0,
        "p10_embb_user_rate": float(np.percentile(embb_rates_mbps, 10.0)) if embb_rates_mbps.size > 0 else 0.0,
        "embb_minrate_satisfied_users": float(satisfied_users),
        "embb_minrate_satisfaction_ratio": float(
            satisfied_users / max(embb_user_count, 1)
        ) if embb_user_count > 0 else float(summary.get("embb_min_rate_satisfaction_after_puncture_deduction", summary.get("embb_min_rate_satisfaction_ratio", 0.0)) or 0.0),
        "embb_minrate_violation_users": float(violation_users),
        "embb_minrate_violation_ratio": float(violation_users / max(embb_user_count, 1)) if embb_user_count > 0 else 0.0,
        "mean_embb_minrate_deficit": float(np.mean(deficits_mbps)) if deficits_mbps.size > 0 else 0.0,
        "max_embb_minrate_deficit": float(np.max(deficits_mbps)) if deficits_mbps.size > 0 else 0.0,
        "mean_normalized_embb_minrate_deficit": float(np.mean(normalized_deficits)) if normalized_deficits.size > 0 else 0.0,
        "max_normalized_embb_minrate_deficit": float(np.max(normalized_deficits)) if normalized_deficits.size > 0 else 0.0,
        "urllc_admission_ratio": float(metrics.get("urllc_admission_ratio", 0.0) or 0.0),
        "urllc_admitted_packets": float(metrics.get("scheduled_urllc_packets", 0.0) or 0.0),
        "urllc_total_packets": float(metrics.get("total_urllc_arrivals", 0.0) or 0.0),
        "urllc_throughput": float(_urllc_throughput_mbps(metrics)),
        "admitted_urllc_reliability": float(
            summary.get("admitted_urllc_reliability", summary.get("urllc_success_rate", np.nan))
        ),
        "total_power": float(total_power),
        "embb_power": float(embb_power),
        "urllc_power": float(urllc_power),
        "embb_power_share": float(embb_power / max(total_power, 1.0e-12)) if total_power > 0.0 else 0.0,
        "urllc_power_share": float(urllc_power / max(total_power, 1.0e-12)) if total_power > 0.0 else 0.0,
        "overlay_action_count": float(overlay_action_count),
        "overlay_action_ratio": float(overlay_action_count / max(coexist_action_total, 1.0e-12)) if coexist_action_total > 0.0 else 0.0,
        "puncturing_action_count": float(puncturing_action_count),
        "puncturing_action_ratio": float(puncturing_action_count / max(coexist_action_total, 1.0e-12)) if coexist_action_total > 0.0 else 0.0,
    }


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    first = rows[0]
    aggregated: Dict[str, object] = {
        "stage": str(first["stage"]),
        "mix": str(first["mix"]),
        "method": str(first["method"]),
        "checkpoint_name": str(first["checkpoint_name"]),
        "target_load": float(first["target_load"]),
        "urllc_activation_prob": float(first["urllc_activation_prob"]),
        "eval_episodes": int(len(rows)),
    }
    numeric_keys = [
        "aggregate_embb_rate",
        "avg_embb_rate",
        "min_embb_user_rate",
        "p5_embb_user_rate",
        "p10_embb_user_rate",
        "embb_minrate_satisfied_users",
        "embb_minrate_satisfaction_ratio",
        "embb_minrate_violation_users",
        "embb_minrate_violation_ratio",
        "mean_embb_minrate_deficit",
        "max_embb_minrate_deficit",
        "mean_normalized_embb_minrate_deficit",
        "max_normalized_embb_minrate_deficit",
        "urllc_admission_ratio",
        "urllc_admitted_packets",
        "urllc_total_packets",
        "urllc_throughput",
        "admitted_urllc_reliability",
        "total_power",
        "embb_power",
        "urllc_power",
        "embb_power_share",
        "urllc_power_share",
        "overlay_action_count",
        "overlay_action_ratio",
        "puncturing_action_count",
        "puncturing_action_ratio",
    ]
    for key in numeric_keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows if not np.isnan(float(row.get(key, 0.0) or 0.0))]
        aggregated[key] = float(np.mean(values)) if values else float("nan")
    return aggregated


def _evaluate_bucket(
    *,
    policy: str,
    method: str,
    checkpoint_name: str,
    stage: str,
    mix_name: str,
    mix_ratio: float,
    explicit_mix_weights: Optional[Tuple[float, float]],
    total_load: float,
    urllc_activation_prob: float,
    embb_min_rate_bps: float,
    seeds: Sequence[int],
    episodes_per_seed: int,
    mappo_checkpoint_path: Optional[str] = None,
    replace_minrate_failures: bool = False,
    accepted_minrate_ratio_threshold: float = 1.0,
    max_attempts_per_seed: int = 1000,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    episode_rows: List[Dict[str, object]] = []
    raw_runs: List[Dict[str, object]] = []
    for seed in seeds:
        accepted_episode_idx = 0
        attempt_idx = 0
        while accepted_episode_idx < int(max(episodes_per_seed, 1)):
            if attempt_idx >= int(max(max_attempts_per_seed, 1)):
                raise RuntimeError(
                    f"Exceeded max_attempts_per_seed={max_attempts_per_seed} while collecting "
                    f"{episodes_per_seed} accepted episodes for seed={seed}, "
                    f"threshold={accepted_minrate_ratio_threshold:.6f}."
                )
            derived_seed = int(seed) * 1000003 + int(attempt_idx)
            policy_cfg = _build_policy_config(
                total_load=float(total_load),
                mix_ratio=float(mix_ratio),
                explicit_mix_weights=explicit_mix_weights,
                activation_prob=float(urllc_activation_prob),
                policy=str(policy),
                mappo_checkpoint_path=mappo_checkpoint_path,
            )
            metrics = run_policy(policy, deepcopy(policy_cfg), int(derived_seed))
            episode_row = _episode_row(
                metrics=metrics,
                method=method,
                checkpoint_name=checkpoint_name,
                stage=stage,
                mix_name=mix_name,
                total_load=float(total_load),
                urllc_activation_prob=float(urllc_activation_prob),
                embb_min_rate_bps=float(embb_min_rate_bps),
            )
            accepted = True
            if replace_minrate_failures:
                accepted = bool(
                    float(episode_row.get("embb_minrate_satisfaction_ratio", 0.0) or 0.0)
                    >= float(accepted_minrate_ratio_threshold) - 1.0e-12
                )
            raw_runs.append(
                {
                    "seed": int(seed),
                    "episode_index_within_seed": int(accepted_episode_idx),
                    "attempt_index_within_seed": int(attempt_idx),
                    "derived_seed": int(derived_seed),
                    "accepted": bool(accepted),
                    "metrics": _jsonify(metrics),
                }
            )
            attempt_idx += 1
            if not accepted:
                continue
            episode_rows.append(episode_row)
            accepted_episode_idx += 1
    return _aggregate_rows(episode_rows), episode_rows, raw_runs


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "stage",
        "mix",
        "method",
        "checkpoint_name",
        "target_load",
        "urllc_activation_prob",
        "eval_episodes",
        "aggregate_embb_rate",
        "avg_embb_rate",
        "min_embb_user_rate",
        "p5_embb_user_rate",
        "p10_embb_user_rate",
        "embb_minrate_satisfied_users",
        "embb_minrate_satisfaction_ratio",
        "embb_minrate_violation_users",
        "embb_minrate_violation_ratio",
        "mean_embb_minrate_deficit",
        "max_embb_minrate_deficit",
        "mean_normalized_embb_minrate_deficit",
        "max_normalized_embb_minrate_deficit",
        "urllc_admission_ratio",
        "urllc_admitted_packets",
        "urllc_total_packets",
        "urllc_throughput",
        "admitted_urllc_reliability",
        "total_power",
        "embb_power",
        "urllc_power",
        "embb_power_share",
        "urllc_power_share",
        "overlay_action_count",
        "overlay_action_ratio",
        "puncturing_action_count",
        "puncturing_action_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _method_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("method", "")), str(row.get("checkpoint_name", "")))


def _plot_metric(
    *,
    rows: Sequence[Dict[str, object]],
    out_dir: Path,
    filename: str,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    coexist_rows = [row for row in rows if str(row.get("stage")) != "stage1_embb_only"]
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in coexist_rows:
        label = str(row.get("checkpoint_name") or row.get("method"))
        grouped.setdefault(label, []).append(dict(row))
    if not grouped:
        return
    plt.figure(figsize=(8.6, 5.2))
    for label, group_rows in sorted(grouped.items()):
        group_rows = sorted(group_rows, key=lambda item: float(item["urllc_activation_prob"]))
        xs = [float(item["urllc_activation_prob"]) for item in group_rows]
        ys = [float(item.get(metric_key, 0.0) or 0.0) for item in group_rows]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=label)
    plt.title(title)
    plt.xlabel("URLLC Activation Probability per Minislot")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


def _largest_activation_prob_at_threshold(rows: Sequence[Dict[str, object]], threshold: float) -> Optional[float]:
    feasible = [
        float(row["urllc_activation_prob"])
        for row in rows
        if float(row.get("embb_minrate_satisfaction_ratio", 0.0) or 0.0) >= float(threshold)
    ]
    return max(feasible) if feasible else None


def _format_activation_prob_list(values: Iterable[float]) -> str:
    arr = [float(v) for v in values]
    if not arr:
        return "none"
    return ", ".join(f"{value:g}" for value in arr)


def _build_report(
    *,
    summary_rows: Sequence[Dict[str, object]],
    greedy_method: str,
) -> str:
    lines: List[str] = []
    stage1_rows = [row for row in summary_rows if str(row.get("stage")) == "stage1_embb_only"]
    stage2_greedy_rows = sorted(
        [
            row for row in summary_rows
            if str(row.get("stage")) == "stage2_greedy_sweep" and str(row.get("method")) == greedy_method
        ],
        key=lambda row: float(row["urllc_activation_prob"]),
    )
    if stage1_rows:
        row = stage1_rows[0]
        lines.append("Stage 1: eMBB-only feasibility check")
        lines.append(
            f"- eMBB-only min-rate satisfaction ratio: {float(row['embb_minrate_satisfaction_ratio']):.4f}"
        )
        lines.append(
            f"- eMBB-only satisfied users: {float(row['embb_minrate_satisfied_users']):.2f}"
        )
        lines.append(
            f"- eMBB-only p5 / p10 user rate: {float(row['p5_embb_user_rate']):.4f} / {float(row['p10_embb_user_rate']):.4f} Mbps"
        )
        if float(row["embb_minrate_satisfaction_ratio"]) >= 0.95:
            lines.append("- Scenario verdict at activation_prob=0: feasible enough for hard eMBB min-rate analysis.")
        else:
            lines.append("- Scenario verdict at activation_prob=0: overloaded or structurally infeasible for hard eMBB min-rate analysis.")
        lines.append("")

    activation_prob_feasible_95 = _largest_activation_prob_at_threshold(stage2_greedy_rows, 0.95)
    activation_prob_feasible_90 = _largest_activation_prob_at_threshold(stage2_greedy_rows, 0.90)
    all_activation_probs = [float(row["urllc_activation_prob"]) for row in stage2_greedy_rows]
    feasible_regime = [p for p in all_activation_probs if activation_prob_feasible_95 is not None and p <= activation_prob_feasible_95]
    near_feasible_regime = [
        p for p in all_activation_probs
        if activation_prob_feasible_90 is not None and (activation_prob_feasible_95 is None or p > activation_prob_feasible_95) and p <= activation_prob_feasible_90
    ]
    overloaded_regime = [
        p for p in all_activation_probs
        if activation_prob_feasible_90 is None or p > activation_prob_feasible_90
    ]

    lines.append("Stage 2: greedy feasibility sweep")
    lines.append(f"- activation_prob_feasible_95: {activation_prob_feasible_95 if activation_prob_feasible_95 is not None else 'none'}")
    lines.append(f"- activation_prob_feasible_90: {activation_prob_feasible_90 if activation_prob_feasible_90 is not None else 'none'}")
    lines.append(f"- feasible regime: {_format_activation_prob_list(feasible_regime)}")
    lines.append(f"- near-feasible regime: {_format_activation_prob_list(near_feasible_regime)}")
    lines.append(f"- overloaded regime: {_format_activation_prob_list(overloaded_regime)}")
    lines.append("")

    stage3_methods = sorted(
        {
            str(row.get("method"))
            for row in summary_rows
            if str(row.get("stage")) == "stage3_checkpoint_compare" and str(row.get("method")) != greedy_method
        }
    )
    if stage3_methods:
        lines.append("Stage 3: MAPPO checkpoint comparison on coexistence activation probabilities")
        greedy_by_activation_prob = {
            float(row["urllc_activation_prob"]): row
            for row in summary_rows
            if str(row.get("method")) == greedy_method and str(row.get("stage")) in {"stage2_greedy_sweep", "stage3_checkpoint_compare"}
        }
        for method in stage3_methods:
            method_rows = sorted(
                [
                    row for row in summary_rows
                    if str(row.get("method")) == method and str(row.get("stage")) == "stage3_checkpoint_compare"
                ],
                key=lambda row: float(row["urllc_activation_prob"]),
            )
            feasible_rows = [
                row for row in method_rows
                if float(row["urllc_activation_prob"]) in feasible_regime
            ]
            keeps_qos = True
            improves_urllc = False
            detail_chunks: List[str] = []
            for row in feasible_rows:
                activation_prob = float(row["urllc_activation_prob"])
                greedy_row = greedy_by_activation_prob.get(activation_prob)
                if greedy_row is None:
                    continue
                sat = float(row["embb_minrate_satisfaction_ratio"])
                sat_g = float(greedy_row["embb_minrate_satisfaction_ratio"])
                adm = float(row["urllc_admission_ratio"])
                adm_g = float(greedy_row["urllc_admission_ratio"])
                tp = float(row["urllc_throughput"])
                tp_g = float(greedy_row["urllc_throughput"])
                if sat < 0.95 or sat < (sat_g - 0.02):
                    keeps_qos = False
                if (adm > adm_g + 1.0e-6) or (tp > tp_g + 1.0e-6):
                    improves_urllc = True
                detail_chunks.append(
                    f"activation_prob={activation_prob:g}: sat={sat:.4f} vs greedy {sat_g:.4f}, "
                    f"adm={adm:.4f} vs {adm_g:.4f}, tp={tp:.4f} vs {tp_g:.4f}"
                )
            lines.append(f"- {method}:")
            lines.append(
                f"  keeps eMBB min-rate QoS in feasible regime: {'yes' if keeps_qos and feasible_rows else 'no' if feasible_rows else 'n/a'}"
            )
            lines.append(
                f"  improves URLLC admission/throughput in feasible regime: {'yes' if improves_urllc and feasible_rows else 'no' if feasible_rows else 'n/a'}"
            )
            if detail_chunks:
                for chunk in detail_chunks:
                    lines.append(f"  {chunk}")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feasibility diagnostics for eMBB min-rate satisfaction before coexistence comparisons."
    )
    parser.add_argument("--mix", default="5:5")
    parser.add_argument("--target-load", type=float, default=24.0)
    parser.add_argument("--embb-min-rate-mbps", type=float, default=2.0)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--stage2-activation-probs", default="")
    parser.add_argument("--stage3-activation-probs", default="")
    parser.add_argument("--stage2-lambdas", default="")
    parser.add_argument("--stage3-lambdas", default="")
    parser.add_argument("--greedy-policy", default=DEFAULT_GREEDY_POLICY)
    parser.add_argument("--mappo-checkpoint-paths", default="")
    parser.add_argument("--mappo-labels", default="")
    parser.add_argument("--include-stage1-mappo", action="store_true")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--replace-minrate-failures", action="store_true")
    parser.add_argument("--accepted-minrate-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--max-attempts-per-seed", type=int, default=1000)
    parser.add_argument("--out-dir", default="sr_mappo/results/feasibility_diagnostic")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(value) for value in _parse_csv_floats(args.seeds)]
    episodes_per_seed = int(max(args.eval_episodes // max(len(seeds), 1), 1))
    while episodes_per_seed * max(len(seeds), 1) < int(args.eval_episodes):
        episodes_per_seed += 1
    stage2_activation_probs = _parse_csv_floats(args.stage2_activation_probs) if str(args.stage2_activation_probs).strip() else (
        _parse_csv_floats(args.stage2_lambdas) if str(args.stage2_lambdas).strip() else [0.5, 1.0, 1.5, 2.0, 3.0]
    )
    stage3_activation_probs = _parse_csv_floats(args.stage3_activation_probs) if str(args.stage3_activation_probs).strip() else (
        _parse_csv_floats(args.stage3_lambdas) if str(args.stage3_lambdas).strip() else list(stage2_activation_probs)
    )
    checkpoint_specs = _parse_checkpoint_specs(args.mappo_checkpoint_paths, args.mappo_labels)

    mix_ratio = float(_parse_mix_ratio(args.mix))
    explicit_mix_weights = _parse_mix_weights(args.mix)
    embb_min_rate_bps = float(args.embb_min_rate_mbps) * 1.0e6

    summary_rows: List[Dict[str, object]] = []
    per_episode_rows: List[Dict[str, object]] = []
    raw_runs: Dict[str, object] = {"buckets": []}

    stage1_summary, stage1_episode_rows, stage1_raw_runs = _evaluate_bucket(
        policy=str(args.greedy_policy),
        method=str(args.greedy_policy),
        checkpoint_name="embb_only_greedy",
        stage="stage1_embb_only",
        mix_name="embb_only",
        mix_ratio=0.0,
        explicit_mix_weights=(1.0, 0.0),
        total_load=float(args.target_load),
        urllc_activation_prob=0.0,
        embb_min_rate_bps=embb_min_rate_bps,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        replace_minrate_failures=bool(args.replace_minrate_failures),
        accepted_minrate_ratio_threshold=float(args.accepted_minrate_ratio_threshold),
        max_attempts_per_seed=int(args.max_attempts_per_seed),
    )
    summary_rows.append(stage1_summary)
    per_episode_rows.extend(stage1_episode_rows)
    raw_runs["buckets"].append(
        {
            "stage": "stage1_embb_only",
            "method": str(args.greedy_policy),
            "checkpoint_name": "embb_only_greedy",
            "raw_runs": stage1_raw_runs,
        }
    )

    if bool(args.include_stage1_mappo):
        for label, checkpoint_path in checkpoint_specs:
            stage1_mappo_summary, stage1_mappo_episode_rows, stage1_mappo_raw_runs = _evaluate_bucket(
                policy="mappo",
                method=str(label),
                checkpoint_name=str(label),
                stage="stage1_embb_only",
                mix_name="embb_only",
                mix_ratio=0.0,
                explicit_mix_weights=(1.0, 0.0),
                total_load=float(args.target_load),
                urllc_activation_prob=0.0,
                embb_min_rate_bps=embb_min_rate_bps,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                mappo_checkpoint_path=str(checkpoint_path),
                replace_minrate_failures=bool(args.replace_minrate_failures),
                accepted_minrate_ratio_threshold=float(args.accepted_minrate_ratio_threshold),
                max_attempts_per_seed=int(args.max_attempts_per_seed),
            )
            summary_rows.append(stage1_mappo_summary)
            per_episode_rows.extend(stage1_mappo_episode_rows)
            raw_runs["buckets"].append(
                {
                    "stage": "stage1_embb_only",
                    "method": str(label),
                    "checkpoint_name": str(label),
                    "checkpoint_path": str(checkpoint_path),
                    "raw_runs": stage1_mappo_raw_runs,
                }
            )

    if not bool(args.stage1_only):
        for activation_prob in stage2_activation_probs:
            summary, episode_rows, bucket_raw_runs = _evaluate_bucket(
                policy=str(args.greedy_policy),
                method=str(args.greedy_policy),
                checkpoint_name="greedy_baseline",
                stage="stage2_greedy_sweep",
                mix_name=str(args.mix),
                mix_ratio=mix_ratio,
                explicit_mix_weights=explicit_mix_weights,
                total_load=float(args.target_load),
                urllc_activation_prob=float(activation_prob),
                embb_min_rate_bps=embb_min_rate_bps,
                seeds=seeds,
                episodes_per_seed=episodes_per_seed,
                replace_minrate_failures=bool(args.replace_minrate_failures),
                accepted_minrate_ratio_threshold=float(args.accepted_minrate_ratio_threshold),
                max_attempts_per_seed=int(args.max_attempts_per_seed),
            )
            summary_rows.append(summary)
            per_episode_rows.extend(episode_rows)
            raw_runs["buckets"].append(
                {
                    "stage": "stage2_greedy_sweep",
                    "method": str(args.greedy_policy),
                    "checkpoint_name": "greedy_baseline",
                    "urllc_activation_prob": float(activation_prob),
                    "raw_runs": bucket_raw_runs,
                }
            )

        for label, checkpoint_path in checkpoint_specs:
            for activation_prob in stage3_activation_probs:
                summary, episode_rows, bucket_raw_runs = _evaluate_bucket(
                    policy="mappo",
                    method=str(label),
                    checkpoint_name=str(label),
                    stage="stage3_checkpoint_compare",
                    mix_name=str(args.mix),
                    mix_ratio=mix_ratio,
                    explicit_mix_weights=explicit_mix_weights,
                    total_load=float(args.target_load),
                    urllc_activation_prob=float(activation_prob),
                    embb_min_rate_bps=embb_min_rate_bps,
                    seeds=seeds,
                    episodes_per_seed=episodes_per_seed,
                    mappo_checkpoint_path=str(checkpoint_path),
                    replace_minrate_failures=bool(args.replace_minrate_failures),
                    accepted_minrate_ratio_threshold=float(args.accepted_minrate_ratio_threshold),
                    max_attempts_per_seed=int(args.max_attempts_per_seed),
                )
                summary_rows.append(summary)
                per_episode_rows.extend(episode_rows)
                raw_runs["buckets"].append(
                    {
                        "stage": "stage3_checkpoint_compare",
                        "method": str(label),
                        "checkpoint_name": str(label),
                        "checkpoint_path": str(checkpoint_path),
                        "urllc_activation_prob": float(activation_prob),
                        "raw_runs": bucket_raw_runs,
                    }
                )

    summary_rows = sorted(summary_rows, key=lambda row: (str(row["stage"]), _method_sort_key(row), float(row["urllc_activation_prob"])))

    _write_csv(out_dir / "feasibility_diagnostic_summary.csv", summary_rows)
    _write_csv(out_dir / "feasibility_diagnostic_episode_rows.csv", per_episode_rows)

    plot_specs = [
        ("embb_minrate_satisfaction_ratio_vs_activation_prob.png", "embb_minrate_satisfaction_ratio", "eMBB Min-Rate Satisfaction Ratio vs URLLC Activation Probability", "Ratio"),
        ("embb_minrate_satisfied_users_vs_activation_prob.png", "embb_minrate_satisfied_users", "eMBB Min-Rate Satisfied Users vs URLLC Activation Probability", "Users"),
        ("mean_embb_minrate_deficit_vs_activation_prob.png", "mean_embb_minrate_deficit", "Mean eMBB Min-Rate Deficit vs URLLC Activation Probability", "Mbps"),
        ("p5_p10_embb_user_rate_vs_activation_prob.png", "", "", ""),
        ("aggregate_embb_rate_vs_activation_prob.png", "aggregate_embb_rate", "Aggregate eMBB Rate vs URLLC Activation Probability", "Mbps"),
        ("urllc_admission_ratio_vs_activation_prob.png", "urllc_admission_ratio", "URLLC Admission Ratio vs URLLC Activation Probability", "Ratio"),
        ("urllc_admitted_packets_vs_activation_prob.png", "urllc_admitted_packets", "URLLC Admitted Packets vs URLLC Activation Probability", "Packets"),
        ("urllc_throughput_vs_activation_prob.png", "urllc_throughput", "URLLC Throughput vs URLLC Activation Probability", "Mbps"),
        ("total_power_vs_activation_prob.png", "total_power", "Total Power vs URLLC Activation Probability", "W"),
        ("embb_power_share_vs_activation_prob.png", "embb_power_share", "eMBB Power Share vs URLLC Activation Probability", "Share"),
        ("urllc_power_share_vs_activation_prob.png", "urllc_power_share", "URLLC Power Share vs URLLC Activation Probability", "Share"),
        ("overlay_puncturing_action_ratio_vs_activation_prob.png", "", "", ""),
    ]
    for filename, metric_key, title, ylabel in plot_specs:
        if metric_key:
            _plot_metric(
                rows=summary_rows,
                out_dir=out_dir,
                filename=filename,
                metric_key=metric_key,
                title=title,
                ylabel=ylabel,
            )

    coexist_rows = [row for row in summary_rows if str(row.get("stage")) != "stage1_embb_only"]
    if coexist_rows:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for row in coexist_rows:
            label = str(row.get("checkpoint_name") or row.get("method"))
            grouped.setdefault(label, []).append(dict(row))

        plt.figure(figsize=(8.6, 5.2))
        for label, group_rows in sorted(grouped.items()):
            group_rows = sorted(group_rows, key=lambda item: float(item["urllc_activation_prob"]))
            xs = [float(item["urllc_activation_prob"]) for item in group_rows]
            p5 = [float(item.get("p5_embb_user_rate", 0.0) or 0.0) for item in group_rows]
            p10 = [float(item.get("p10_embb_user_rate", 0.0) or 0.0) for item in group_rows]
            plt.plot(xs, p5, marker="o", linewidth=2.0, label=f"{label} p5")
            plt.plot(xs, p10, marker="s", linewidth=2.0, linestyle="--", label=f"{label} p10")
        plt.title("p5 / p10 eMBB User Rate vs URLLC Activation Probability")
        plt.xlabel("URLLC Activation Probability per Minislot")
        plt.ylabel("Mbps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "p5_p10_embb_user_rate_vs_activation_prob.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8.6, 5.2))
        for label, group_rows in sorted(grouped.items()):
            group_rows = sorted(group_rows, key=lambda item: float(item["urllc_activation_prob"]))
            xs = [float(item["urllc_activation_prob"]) for item in group_rows]
            overlay = [float(item.get("overlay_action_ratio", 0.0) or 0.0) for item in group_rows]
            puncture = [float(item.get("puncturing_action_ratio", 0.0) or 0.0) for item in group_rows]
            plt.plot(xs, overlay, marker="o", linewidth=2.0, label=f"{label} overlay")
            plt.plot(xs, puncture, marker="s", linewidth=2.0, linestyle="--", label=f"{label} puncture")
        plt.title("Overlay / Puncturing Action Ratio vs URLLC Activation Probability")
        plt.xlabel("URLLC Activation Probability per Minislot")
        plt.ylabel("Ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "overlay_puncturing_action_ratio_vs_activation_prob.png", dpi=220)
        plt.close()

    report_text = _build_report(summary_rows=summary_rows, greedy_method=str(args.greedy_policy))
    (out_dir / "feasibility_diagnostic_report.txt").write_text(report_text, encoding="utf-8")

    payload = {
        "meta": {
            "mix": str(args.mix),
            "target_load": float(args.target_load),
            "embb_min_rate_mbps": float(args.embb_min_rate_mbps),
            "requested_eval_episodes": int(args.eval_episodes),
            "actual_eval_episodes_per_bucket": int(episodes_per_seed * max(len(seeds), 1)),
            "seeds": [int(seed) for seed in seeds],
            "stage2_activation_probs": [float(p) for p in stage2_activation_probs],
            "stage3_activation_probs": [float(p) for p in stage3_activation_probs],
            "greedy_policy": str(args.greedy_policy),
            "replace_minrate_failures": bool(args.replace_minrate_failures),
            "accepted_minrate_ratio_threshold": float(args.accepted_minrate_ratio_threshold),
            "max_attempts_per_seed": int(args.max_attempts_per_seed),
            "mappo_checkpoints": [{"label": label, "path": path} for label, path in checkpoint_specs],
        },
        "summary_rows": _jsonify(summary_rows),
        "per_episode_rows": _jsonify(per_episode_rows),
        "raw_runs": _jsonify(raw_runs),
    }
    with (out_dir / "feasibility_diagnostic_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[FEASIBILITY] wrote summary CSV to {out_dir / 'feasibility_diagnostic_summary.csv'}", flush=True)
    print(f"[FEASIBILITY] wrote report to {out_dir / 'feasibility_diagnostic_report.txt'}", flush=True)
    print(f"[FEASIBILITY] wrote plots and raw metrics under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
