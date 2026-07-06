from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from .unified_policy_runner import run_policy


DEFAULT_MAPPO_CHECKPOINT = (
    Path("checkpoints")
    / "clean_mappo"
    / "sr_mappo_tp_full_mappo_v17zza_embb_qos_rebalance_v1_clean_best_eval.pt"
)

POLICY_LABELS = {
    "greedy": "Greedy",
    "mappo": "MAPPO (v17zza)",
}


def _parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(round(float(token))))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _scenario_label(urllc_users: int, packet_bits: int) -> str:
    return f"U{int(urllc_users)}-B{int(packet_bits)}"


def _apply_channel_setting_env(scene_id: str | None) -> Dict[str, str | None]:
    keys = (
        "SR_MAPPO_MOTHER_TOPOLOGY_FREEZE",
        "SR_MAPPO_MOTHER_TOPOLOGY_ID",
    )
    previous = {key: os.environ.get(key) for key in keys}
    if str(scene_id or "").strip():
        os.environ["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
        os.environ["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = str(scene_id).strip()
    else:
        for key in keys:
            os.environ.pop(key, None)
    return previous


def _build_scene_id(
    *,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    channel_setting_index: int,
    share_scene_across_packet_bits: bool,
    share_scene_across_urllc_users: bool = False,
    urllc_scene_anchor: int | None = None,
) -> str:
    urllc_scene_token = (
        f"urllc{int(urllc_scene_anchor if urllc_scene_anchor is not None else urllc_users)}"
        if bool(share_scene_across_urllc_users)
        else f"urllc{int(urllc_users)}"
    )
    base = f"embb{int(embb_users)}_{urllc_scene_token}"
    if not bool(share_scene_across_packet_bits):
        base = f"{base}_k{int(packet_bits)}"
    return f"{base}_scene{int(channel_setting_index)}"


def _derive_episode_seed(base_seed: int, episode_index: int) -> int:
    base = int(base_seed)
    episode = max(int(episode_index), 1)
    if episode == 1:
        return base
    return int(base + (episode - 1) * 1_000_003)


def _restore_env(previous: Dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


def _build_policy_config(
    *,
    policy: str,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    channel_uses: int | None,
    lambda_per_user: float | None,
    target_error_probability: float | None,
    mappo_checkpoint_path: str | None,
    geometry_profile: str | None,
    min_overlay_retention: float | None,
    nested_urllc_subset_from_max: bool = False,
    nested_max_urllc_users: int | None = None,
) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "system": {
            "num_embb_users": int(embb_users),
            "num_urllc_users": int(urllc_users),
        },
        "env": {
            "phase_a_protect_phase0_satisfied_embb_users": True,
        },
        "shield": {
            "apply_joint_minrate_rewrite": True,
        },
    }

    if channel_uses is not None and int(channel_uses) > 0:
        cfg["system"]["channel_uses_per_minislot"] = int(channel_uses)
    if str(geometry_profile or "").strip():
        cfg["env"]["geometry_profile"] = str(geometry_profile).strip()
    if min_overlay_retention is not None:
        cfg["env"]["min_overlay_retention"] = float(min_overlay_retention)

    urllc_cfg: Dict[str, object] = {
        "packet_lengths": [int(packet_bits), int(packet_bits), int(packet_bits)],
    }
    if target_error_probability is not None:
        urllc_cfg["target_error_probability"] = float(target_error_probability)
    cfg["urllc"] = urllc_cfg

    if lambda_per_user is not None:
        cfg["simulation"] = {
            "fixed_urllc_poisson_rate": True,
            "urllc_poisson_rate": float(lambda_per_user),
        }
        cfg["env"]["urllc_poisson_rate_is_per_user"] = True
        cfg["env"]["urllc_poisson_rate_is_slot_level"] = False

    if policy == "mappo":
        checkpoint = Path(str(mappo_checkpoint_path or DEFAULT_MAPPO_CHECKPOINT)).expanduser()
        cfg["checkpoint_path"] = str(checkpoint)

    if bool(nested_urllc_subset_from_max):
        max_urllc = int(max(nested_max_urllc_users or urllc_users, urllc_users))
        cfg["system"]["nested_load_from_max_users_enabled"] = True
        cfg["system"]["nested_load_max_total_users"] = int(embb_users + max_urllc)
        cfg["system"]["nested_load_max_embb_users"] = int(embb_users)
        cfg["system"]["nested_load_max_urllc_users"] = int(max_urllc)
        cfg["system"]["force_serving_hints_association"] = True
        cfg["env"]["nested_fixed_user_subset_across_episodes"] = True
        cfg["env"]["nested_fixed_user_subset_across_loads"] = True

    return cfg


def _mean_rows(rows: List[Dict[str, object]], metrics: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in metrics:
        out[key] = float(np.mean([float(row.get(key, 0.0) or 0.0) for row in rows])) if rows else 0.0
    return out


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                ordered.append(str(key))
    return ordered


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "None"):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_debug_trace(
    *,
    out_dir: Path,
    policy: str,
    seed: int,
    episode_index: int,
    episode_seed: int,
    embb_users: int,
    urllc_users: int,
    packet_bits: int,
    channel_setting_index: int,
    scene_id: str,
    result: Dict[str, object],
) -> None:
    traces_dir = out_dir / "debug_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        "policy": str(policy),
        "seed": int(seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "embb_users": int(embb_users),
        "urllc_users": int(urllc_users),
        "packet_bits": int(packet_bits),
        "channel_setting_index": int(channel_setting_index),
        "mother_topology_id": str(scene_id),
        "total_embb_throughput": float(result.get("total_embb_throughput", 0.0) or 0.0),
        "admitted_urllc_count": float(result.get("admitted_urllc_count", 0.0) or 0.0),
        "overlay_action_count": float(result.get("overlay_action_count", 0.0) or 0.0),
        "puncturing_action_count": float(result.get("puncturing_action_count", 0.0) or 0.0),
        "per_admission_embb_damage_samples": list(result.get("per_admission_embb_damage_samples", []) or []),
        "embb_user_minrate_trace": list(result.get("embb_user_minrate_trace", []) or []),
        "constraint_violation_log": dict(result.get("constraint_violation_log", {}) or {}),
        "raw_summary": dict(result.get("raw_summary", {}) or {}),
    }
    trace_name = (
        f"policy_{str(policy)}__embb_{int(embb_users)}__urllc_{int(urllc_users)}__"
        f"k_{int(packet_bits)}__seed_{int(seed)}__ep_{int(episode_index)}__"
        f"epseed_{int(episode_seed)}__scene_{int(channel_setting_index)}.json"
    )
    (traces_dir / trace_name).write_text(
        json.dumps(_json_safe(trace_payload), indent=2),
        encoding="utf-8",
    )


def _extract_embb_user_rates_after_mbps(result: Dict[str, object]) -> List[float]:
    raw_summary = dict(result.get("raw_summary", {}) or {})
    rates = raw_summary.get("embb_user_rates_after_puncture_deduction", raw_summary.get("embb_user_rates", []))
    arr = np.asarray(rates, dtype=float)
    if arr.size == 0:
        return []
    return [float(x / 1.0e6) for x in arr]


def _extract_embb_user_powers_after_watts(result: Dict[str, object]) -> List[float]:
    embb_result = dict(result.get("embb_result", {}) or {})
    powers = embb_result.get("user_tx_powers", result.get("embb_user_tx_powers", []))
    arr = np.asarray(powers, dtype=float)
    if arr.size == 0:
        return []
    return [float(x) for x in arr.tolist()]


def _extract_embb_jain_after(result: Dict[str, object], embb_user_rates_after_mbps: List[float]) -> float:
    raw_summary = dict(result.get("raw_summary", {}) or {})
    value = raw_summary.get("jain_fairness", None)
    if value not in (None, "", "None"):
        return float(value)
    arr = np.asarray(embb_user_rates_after_mbps, dtype=float)
    if arr.size == 0:
        return 0.0
    numer = float(np.sum(arr) ** 2)
    denom = float(arr.size * np.sum(np.square(arr)))
    return float(numer / denom) if denom > 0.0 else 0.0


def _extract_raw_summary_scalar(result: Dict[str, object], key: str, default: float = 0.0) -> float:
    raw_summary = dict(result.get("raw_summary", {}) or {})
    value = raw_summary.get(key, result.get(key, default))
    return _safe_float(value, default=default)


def _percentile(values: List[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, float(q)))


def _normalize_row_types(row: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    int_keys = {
        "seed",
        "episode_index",
        "episode_seed",
        "channel_setting_index",
        "embb_users",
        "urllc_users",
        "packet_bits",
        "channel_uses",
    }
    float_keys = {
        "lambda_per_user",
        "target_error_probability",
        "total_embb_throughput",
        "embb_sum_rate_after",
        "total_urllc_arrivals",
        "urllc_arrival_count",
        "average_power_consumption",
        "total_power",
        "embb_power",
        "urllc_power",
        "embb_served_user_count",
        "embb_blocked_user_count",
        "embb_blocked_count",
        "urllc_blocked_user_count",
        "urllc_blocked_count",
        "urllc_admitted_count",
        "embb_minimum_rate_violation_count",
        "overlay_action_count",
        "overlay_count",
        "puncturing_action_count",
        "puncture_count",
        "phase_a_total_decisions",
        "phase_a_rejected_min_rate_total",
        "phase_a_rejected_min_rate_per_decision",
        "keep_action_count",
        "keep_count",
        "urllc_admission_ratio",
        "admitted_urllc_count",
        "embb_jain_after",
        "embb_min_rate_after",
        "embb_5th_percentile_after",
        "embb_median_rate_after",
        "embb_rate_with_intercell_after_puncture_deduction",
        "no_intercell_rate_with_same_puncture_mask",
        "intercell_rate_loss_with_same_puncture_mask",
        "embb_rate_loss_due_to_intercell_ratio",
        "mean_intercell_interference_mw",
        "mean_intercell_interference_over_noise",
        "selected_action_intercell_cost_after_source_mask_mean",
        "selected_action_intercell_cost_after_source_mask_p95",
        "selected_action_intercell_cost_after_source_mask_over_noise_mean",
        "intercell_per_admitted_packet",
    }
    for key in int_keys:
        value = normalized.get(key, None)
        if value in (None, "", "None"):
            normalized[key] = None
        else:
            normalized[key] = int(round(float(value)))
    for key in float_keys:
        value = normalized.get(key, None)
        normalized[key] = 0.0 if value in (None, "", "None") else float(value)
    return normalized


def _run_key(row: Dict[str, object]) -> Tuple[int, int, int, str, int, int]:
    return (
        int(row["embb_users"]),
        int(row["urllc_users"]),
        int(row["packet_bits"]),
        str(row["policy"]),
        int(row["seed"]),
        int(row.get("episode_index", 1) or 1),
    )


def _make_progress_bar(*, total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=int(total), desc=desc, unit="run", dynamic_ncols=True)


def _refresh_outputs(
    *,
    out_dir: Path,
    per_run_rows: List[Dict[str, object]],
    grouped_runs: Dict[Tuple[int, int, int, str], List[Dict[str, object]]],
    metric_keys: List[str],
    embb_users_list: List[int],
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    seeds: List[int],
    args: argparse.Namespace,
    write_plots: bool,
    latest_completed_run: Dict[str, object] | None = None,
) -> None:
    if not per_run_rows:
        return

    per_run_csv = out_dir / "per_run_results.csv"
    _write_csv(per_run_csv, per_run_rows, _collect_fieldnames(per_run_rows))

    aggregated_rows: List[Dict[str, object]] = []
    grouped_means: Dict[Tuple[int, int, int, str], Dict[str, float]] = {}
    for key, rows in grouped_runs.items():
        if not rows:
            continue
        means = _mean_rows(
            rows,
            metric_keys + ["keep_action_count", "urllc_admission_ratio", "admitted_urllc_count", "total_urllc_arrivals"],
        )
        embb_users, urllc_users, packet_bits, policy = key
        grouped_means[key] = means
        aggregated_rows.append(
            {
                "embb_users": int(embb_users),
                "urllc_users": int(urllc_users),
                "packet_bits": int(packet_bits),
                "policy": str(policy),
                "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                "scenario_label": _scenario_label(int(urllc_users), int(packet_bits)),
                **means,
            }
        )

    aggregated_rows.sort(key=lambda row: (int(row["embb_users"]), int(row["urllc_users"]), int(row["packet_bits"]), str(row["policy"])))
    aggregated_csv = out_dir / "aggregated_results.csv"
    if aggregated_rows:
        _write_csv(aggregated_csv, aggregated_rows, _collect_fieldnames(aggregated_rows))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if write_plots and grouped_means:
        for embb_users in embb_users_list:
            embb_keys_present = [
                (embb_users, int(urllc_users), int(packet_bits), str(policy))
                for urllc_users in urllc_users_list
                for packet_bits in packet_bits_list
                for policy in policies
            ]
            if any(key not in grouped_means for key in embb_keys_present):
                continue
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_sum_rate.png",
                f"eMBB Sum Rate under Different Block Lengths (fixed eMBB={embb_users})",
                "eMBB sum rate (bps)",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "total_embb_throughput",
            )
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_total_power.png",
                f"Total Power under Different Block Lengths (fixed eMBB={embb_users})",
                "Total power",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "average_power_consumption",
            )
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_embb_blocked_users.png",
                f"eMBB Blocked Users under Different Block Lengths (fixed eMBB={embb_users})",
                "eMBB blocked users count",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "embb_blocked_user_count",
            )
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_urllc_blocked_users.png",
                f"URLLC Blocked Users under Different Block Lengths (fixed eMBB={embb_users})",
                "URLLC blocked users count",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "urllc_blocked_user_count",
            )
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_urllc_arrivals.png",
                f"URLLC Total Arrivals under Different Block Lengths (fixed eMBB={embb_users})",
                "URLLC total arrivals",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "total_urllc_arrivals",
            )
            _plot_metric_by_blocklength_panels(
                plots_dir / f"embb_{embb_users}_urllc_admitted_users.png",
                f"Admitted URLLC Packets under Different Block Lengths (fixed eMBB={embb_users})",
                "Admitted URLLC packets",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
                "admitted_urllc_count",
            )
            _plot_mode_ratio_lines_averaged_over_blocklength(
                plots_dir / f"embb_{embb_users}_mode_ratios.png",
                urllc_users_list,
                packet_bits_list,
                policies,
                grouped_means,
                embb_users,
            )

    total_expected_runs = (
        len(embb_users_list)
        * len(urllc_users_list)
        * len(packet_bits_list)
        * len(policies)
        * len(seeds)
        * max(int(args.episodes_per_channel_setting), 1)
    )
    metadata = {
        "policies": policies,
        "embb_users": embb_users_list,
        "urllc_users": urllc_users_list,
        "packet_bits": packet_bits_list,
        "stored_per_run_fields": [
            "embb_users",
            "urllc_users",
            "packet_size",
            "method",
            "seed",
            "episode_index",
            "episode_seed",
            "channel_setting_index",
            "mother_topology_id",
            "runtime_sec",
            "embb_sum_rate_after",
            "total_power",
            "embb_power",
            "urllc_power",
            "embb_blocked_count",
            "urllc_arrival_count",
            "urllc_admitted_count",
            "urllc_blocked_count",
            "embb_user_rates_after",
            "embb_user_powers_after",
            "embb_jain_after",
            "embb_min_rate_after",
            "embb_5th_percentile_after",
            "embb_median_rate_after",
            "embb_rate_with_intercell_after_puncture_deduction",
            "no_intercell_rate_with_same_puncture_mask",
            "intercell_rate_loss_with_same_puncture_mask",
            "embb_rate_loss_due_to_intercell_ratio",
            "mean_intercell_interference_mw",
            "mean_intercell_interference_over_noise",
            "selected_action_intercell_cost_after_source_mask_mean",
            "selected_action_intercell_cost_after_source_mask_p95",
            "selected_action_intercell_cost_after_source_mask_over_noise_mean",
            "intercell_per_admitted_packet",
            "phase0_actual_partial_minrate_user_count",
            "phase0_actual_reclaimed_rb_count",
            "phase0_actual_refill_rb_count",
            "phase0_actual_refill_gain_mbps",
            "phase0_actual_refill_intercell_delta_over_noise",
            "keep_count",
            "overlay_count",
            "puncture_count",
        ],
        "channel_uses": args.channel_uses,
        "lambda_per_user": args.lambda_per_user,
        "target_error_probability": args.target_error_probability,
        "episodes_per_channel_setting": int(args.episodes_per_channel_setting),
        "seed_semantics": "Each seed defines one frozen mother topology/channel setting (scene).",
        "episodes_per_channel_setting_semantics": "For each seed-defined scene, run this many independent episodes using derived episode seeds while keeping the scene frozen.",
        "seeds": seeds,
        "mappo_checkpoint_path": str(Path(args.mappo_checkpoint_path).expanduser()),
        "per_run_csv": str(per_run_csv),
        "aggregated_csv": str(aggregated_csv),
        "plots_dir": str(plots_dir),
        "completed_runs": len(per_run_rows),
        "total_expected_runs": int(total_expected_runs),
        "latest_completed_run": latest_completed_run,
    }
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _resolve_out_dir(args: argparse.Namespace, embb_users: int) -> Path:
    template = str(getattr(args, "out_dir_template", "") or "").strip()
    if not template:
        return Path(args.out_dir)
    return Path(template.format(embb=int(embb_users), embb_users=int(embb_users)))


def _plot_metric_bars(
    path: Path,
    title: str,
    ylabel: str,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    grouped_rows: Dict[Tuple[int, int, int, str], Dict[str, float]],
    embb_users: int,
    metric_key: str,
) -> None:
    x = np.arange(len(urllc_users_list), dtype=float)
    series_specs = [(int(bits), str(policy)) for bits in packet_bits_list for policy in policies]
    width = 0.8 / max(len(series_specs), 1)
    fig, ax = plt.subplots(figsize=(max(9.5, len(urllc_users_list) * 1.25), 5.8))
    colors = {"greedy": "#4C78A8", "mappo": "#E45756"}
    hatches = {int(packet_bits_list[0]): "", int(packet_bits_list[1]): "//", int(packet_bits_list[2]): "xx"} if len(packet_bits_list) >= 3 else {}

    for idx, (packet_bits, policy) in enumerate(series_specs):
        values = [
            float(grouped_rows[(embb_users, int(urllc_users), int(packet_bits), str(policy))][metric_key])
            for urllc_users in urllc_users_list
        ]
        offset = (idx - (len(series_specs) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=f"{POLICY_LABELS.get(policy, policy)} | B{int(packet_bits)}",
            color=colors.get(str(policy), None),
            hatch=hatches.get(int(packet_bits), ""),
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("URLLC users")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_metric_by_blocklength_panels(
    path: Path,
    title: str,
    ylabel: str,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    grouped_rows: Dict[Tuple[int, int, int, str], Dict[str, float]],
    embb_users: int,
    metric_key: str,
) -> None:
    fig, axes = plt.subplots(1, len(packet_bits_list), figsize=(max(12.0, len(packet_bits_list) * 4.2), 5.2), sharey=True)
    if len(packet_bits_list) == 1:
        axes = [axes]

    x = np.arange(len(urllc_users_list), dtype=float)
    width = 0.8 / max(len(policies), 1)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for panel_idx, (ax, packet_bits) in enumerate(zip(axes, packet_bits_list)):
        for policy_idx, policy in enumerate(policies):
            values = [
                float(grouped_rows[(embb_users, int(urllc_users), int(packet_bits), str(policy))][metric_key])
                for urllc_users in urllc_users_list
            ]
            offset = (policy_idx - (len(policies) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=POLICY_LABELS.get(policy, policy),
                edgecolor="black",
                linewidth=0.6,
            )

        panel_tag = panel_labels[panel_idx] if panel_idx < len(panel_labels) else f"({panel_idx + 1})"
        ax.set_title(f"{panel_tag} B{int(packet_bits)}")
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[: len(policies)],
        labels[: len(policies)],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(len(policies), 1),
        frameon=False,
    )
    fig.suptitle(title, y=1.05)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_mode_ratio_lines_averaged_over_blocklength(
    path: Path,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    grouped_rows: Dict[Tuple[int, int, int, str], Dict[str, float]],
    embb_users: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    x = np.asarray(urllc_users_list, dtype=float)
    mode_specs = [
        ("keep_action_count", "(a) KEEP ratio"),
        ("overlay_action_count", "(b) OVERLAY ratio"),
        ("puncturing_action_count", "(c) PUNCTURE ratio"),
    ]
    markers = ["o", "s", "^", "D", "v", "P"]

    for ax, (mode_key, title) in zip(axes, mode_specs):
        for urllc_users in urllc_users_list:
            pass
        for policy_idx, policy in enumerate(policies):
            ratios: List[float] = []
            for urllc_users in urllc_users_list:
                per_block_ratios: List[float] = []
                for packet_bits in packet_bits_list:
                    row = grouped_rows[(embb_users, int(urllc_users), int(packet_bits), str(policy))]
                    total = max(float(row.get("phase_a_total_decisions", 0.0) or 0.0), 1.0)
                    per_block_ratios.append(100.0 * float(row.get(mode_key, 0.0) or 0.0) / total)
                ratios.append(float(np.mean(np.asarray(per_block_ratios, dtype=float))))
            ax.plot(
                x,
                ratios,
                marker=markers[policy_idx % len(markers)],
                linewidth=2.0,
                markersize=6.0,
                label=POLICY_LABELS.get(policy, policy),
            )

        ax.set_title(title)
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Mode ratio (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[: len(policies)],
        labels[: len(policies)],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(len(policies), 1),
        frameon=False,
    )
    fig.suptitle(f"Mode Selection Ratios Averaged over Block Lengths (fixed eMBB={embb_users})", y=1.05)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare greedy and MAPPO under fixed eMBB users, URLLC user sweep, and finite blocklength packet sizes."
    )
    parser.add_argument("--policies", default="greedy,mappo")
    parser.add_argument("--embb-users", default="10,20,30")
    parser.add_argument("--urllc-users", default="10,20,30,40,50")
    parser.add_argument("--packet-bits", default="24")
    parser.add_argument("--channel-uses", type=int, default=None)
    parser.add_argument("--lambda-per-user", type=float, default=None)
    parser.add_argument("--target-error-probability", type=float, default=None)
    parser.add_argument("--geometry-profile", type=str, default=None)
    parser.add_argument("--min-overlay-retention", type=float, default=None)
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--episodes-per-channel-setting",
        type=int,
        default=10,
        help="For each seed-defined frozen channel/topology setting, run this many episodes with different episode seeds.",
    )
    parser.add_argument(
        "--mappo-checkpoint-path",
        default=str(DEFAULT_MAPPO_CHECKPOINT),
    )
    parser.add_argument("--out-dir", default="sr_mappo/results/fixed_user_blocklength_compare")
    parser.add_argument(
        "--out-dir-template",
        default=None,
        help="Optional per-eMBB output template, e.g. sr_mappo/results/fixed_compare_10seeds_embb{embb}_all5/add_seeds_52_81_zza_only",
    )
    parser.add_argument(
        "--save-every-run",
        action="store_true",
        help="Flush CSV/summary after every single (policy, seed) run instead of after each scenario.",
    )
    parser.add_argument(
        "--share-scene-across-packet-bits",
        action="store_true",
        help="Freeze the same mother topology/channel setting across different packet sizes for the same (eMBB, URLLC, scene index). Useful for clean payload-size ablations.",
    )
    parser.add_argument(
        "--share-scene-across-urllc-users",
        action="store_true",
        help="Freeze the same mother topology across URLLC-user counts and evaluate lower U as fixed nested subsets from the maximum-URLLC mother scene.",
    )
    args = parser.parse_args()

    policies = [token.strip() for token in str(args.policies).split(",") if token.strip()]
    embb_users_list = _parse_csv_ints(args.embb_users)
    urllc_users_list = _parse_csv_ints(args.urllc_users)
    packet_bits_list = _parse_csv_ints(args.packet_bits)
    seeds = _parse_csv_ints(args.seeds)
    max_urllc_users = max(urllc_users_list) if urllc_users_list else 0

    metric_keys = [
        "total_embb_throughput",
        "embb_sum_rate_after",
        "total_urllc_arrivals",
        "urllc_arrival_count",
        "average_power_consumption",
        "total_power",
        "embb_power",
        "urllc_power",
        "embb_blocked_user_count",
        "embb_blocked_count",
        "urllc_blocked_user_count",
        "urllc_blocked_count",
        "urllc_admitted_count",
        "embb_jain_after",
        "embb_min_rate_after",
        "embb_5th_percentile_after",
        "embb_median_rate_after",
        "embb_rate_with_intercell_after_puncture_deduction",
        "no_intercell_rate_with_same_puncture_mask",
        "intercell_rate_loss_with_same_puncture_mask",
        "embb_rate_loss_due_to_intercell_ratio",
        "mean_intercell_interference_mw",
        "mean_intercell_interference_over_noise",
        "selected_action_intercell_cost_after_source_mask_mean",
        "selected_action_intercell_cost_after_source_mask_p95",
        "selected_action_intercell_cost_after_source_mask_over_noise_mean",
        "intercell_per_admitted_packet",
        "overlay_action_count",
        "overlay_count",
        "puncturing_action_count",
        "puncture_count",
        "phase_a_total_decisions",
        "phase_a_rejected_min_rate_total",
        "phase_a_rejected_min_rate_per_decision",
    ]

    embb_users_by_out_dir: Dict[Path, List[int]] = {}
    for embb_users in embb_users_list:
        out_dir = _resolve_out_dir(args, int(embb_users))
        embb_users_by_out_dir.setdefault(out_dir, []).append(int(embb_users))

    for out_dir, embb_group in embb_users_by_out_dir.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        embb_group = list(dict.fromkeys(int(v) for v in embb_group))

        per_run_csv = out_dir / "per_run_results.csv"
        existing_rows = [_normalize_row_types(row) for row in _read_csv_rows(per_run_csv)]
        per_run_rows: List[Dict[str, object]] = list(existing_rows)
        grouped_runs: Dict[Tuple[int, int, int, str], List[Dict[str, object]]] = {}
        completed_run_keys = {_run_key(row) for row in existing_rows}

        for embb_users in embb_group:
            for urllc_users in urllc_users_list:
                for packet_bits in packet_bits_list:
                    for policy in policies:
                        key = (int(embb_users), int(urllc_users), int(packet_bits), str(policy))
                        grouped_runs[key] = []
        for row in existing_rows:
            key = (int(row["embb_users"]), int(row["urllc_users"]), int(row["packet_bits"]), str(row["policy"]))
            grouped_runs.setdefault(key, []).append(row)

        if existing_rows:
            print(
                f"[FIXED-USER-COMPARE] resuming from {len(existing_rows)} saved runs in {per_run_csv}",
                flush=True,
            )

        episode_count = max(int(args.episodes_per_channel_setting), 1)
        latest_completed_run: Dict[str, object] | None = per_run_rows[-1] if per_run_rows else None

        for embb_users in embb_group:
            existing_rows_for_embb = [
                row for row in existing_rows if int(row["embb_users"]) == int(embb_users)
            ]
            planned_run_keys = {
                (int(embb_users), int(urllc_users), int(packet_bits), str(policy), int(seed), int(episode_index))
                for urllc_users in urllc_users_list
                for packet_bits in packet_bits_list
                for policy in policies
                for seed in seeds
                for episode_index in range(1, episode_count + 1)
            }
            pending_run_count = sum(1 for run_key in planned_run_keys if run_key not in completed_run_keys)
            progress_bar = _make_progress_bar(
                total=pending_run_count,
                desc=f"eMBB={int(embb_users)}",
            )
            if progress_bar is None:
                print(
                    f"[FIXED-USER-COMPARE] eMBB={embb_users} pending_runs={pending_run_count} "
                    f"/ total_runs={len(planned_run_keys)}",
                    flush=True,
                )
            elif existing_rows_for_embb:
                progress_bar.set_description_str(f"eMBB={int(embb_users)} (resume)")

            try:
                for urllc_users in urllc_users_list:
                    for packet_bits in packet_bits_list:
                        latest_row_for_scenario: Dict[str, object] | None = None
                        for seed in seeds:
                            channel_setting_index = int(seed)
                            scene_id = _build_scene_id(
                                embb_users=int(embb_users),
                                urllc_users=int(urllc_users),
                                packet_bits=int(packet_bits),
                                channel_setting_index=int(channel_setting_index),
                                share_scene_across_packet_bits=bool(getattr(args, "share_scene_across_packet_bits", False)),
                                share_scene_across_urllc_users=bool(getattr(args, "share_scene_across_urllc_users", False)),
                                urllc_scene_anchor=int(max_urllc_users),
                            )
                            env_backup = _apply_channel_setting_env(scene_id)
                            try:
                                for episode_index in range(1, episode_count + 1):
                                    episode_seed = _derive_episode_seed(int(seed), int(episode_index))
                                    for policy in policies:
                                        key = (int(embb_users), int(urllc_users), int(packet_bits), str(policy))
                                        run_key = (
                                            int(embb_users),
                                            int(urllc_users),
                                            int(packet_bits),
                                            str(policy),
                                            int(seed),
                                            int(episode_index),
                                        )
                                        if run_key in completed_run_keys:
                                            print(
                                                f"[FIXED-USER-COMPARE] skip existing policy={policy} seed={seed} "
                                                f"ep={episode_index} eMBB={embb_users} URLLC={urllc_users} bits={packet_bits}",
                                                flush=True,
                                            )
                                            continue
                                        if progress_bar is not None:
                                            progress_bar.set_postfix_str(
                                                f"U={int(urllc_users)} B={int(packet_bits)} scene={int(channel_setting_index)} "
                                                f"seed={int(seed)} ep={int(episode_index)} policy={policy}"
                                            )
                                        cfg = _build_policy_config(
                                            policy=str(policy),
                                            embb_users=int(embb_users),
                                            urllc_users=int(urllc_users),
                                            packet_bits=int(packet_bits),
                                            channel_uses=args.channel_uses,
                                            lambda_per_user=args.lambda_per_user,
                                            target_error_probability=args.target_error_probability,
                                            mappo_checkpoint_path=args.mappo_checkpoint_path,
                                            geometry_profile=args.geometry_profile,
                                            min_overlay_retention=args.min_overlay_retention,
                                            nested_urllc_subset_from_max=bool(getattr(args, "share_scene_across_urllc_users", False)),
                                            nested_max_urllc_users=int(max_urllc_users),
                                        )
                                        result = run_policy(str(policy), deepcopy(cfg), int(episode_seed))
                                        phase_a_total_decisions = float(result.get("phase_a_total_decisions", 0.0) or 0.0)
                                        overlay_count = float(result.get("overlay_action_count", 0.0) or 0.0)
                                        puncture_count = float(result.get("puncturing_action_count", 0.0) or 0.0)
                                        keep_count = max(phase_a_total_decisions - overlay_count - puncture_count, 0.0)
                                        embb_user_rates_after_mbps = _extract_embb_user_rates_after_mbps(result)
                                        embb_user_powers_after_watts = _extract_embb_user_powers_after_watts(result)
                                        embb_jain_after = _extract_embb_jain_after(result, embb_user_rates_after_mbps)
                                        embb_min_rate_after = float(min(embb_user_rates_after_mbps)) if embb_user_rates_after_mbps else 0.0
                                        embb_5th_percentile_after = _percentile(embb_user_rates_after_mbps, 5.0)
                                        embb_median_rate_after = _percentile(embb_user_rates_after_mbps, 50.0)
                                        embb_rate_with_intercell_after_puncture_deduction = _extract_raw_summary_scalar(
                                            result,
                                            "embb_rate_with_intercell_after_puncture_deduction",
                                        )
                                        no_intercell_rate_with_same_puncture_mask = _extract_raw_summary_scalar(
                                            result,
                                            "no_intercell_rate_with_same_puncture_mask",
                                        )
                                        intercell_rate_loss_with_same_puncture_mask = _extract_raw_summary_scalar(
                                            result,
                                            "intercell_rate_loss_with_same_puncture_mask",
                                        )
                                        embb_rate_loss_due_to_intercell_ratio = _extract_raw_summary_scalar(
                                            result,
                                            "embb_rate_loss_due_to_intercell_ratio",
                                        )
                                        mean_intercell_interference_mw = _extract_raw_summary_scalar(
                                            result,
                                            "mean_intercell_interference_mw",
                                        )
                                        mean_intercell_interference_over_noise = _extract_raw_summary_scalar(
                                            result,
                                            "mean_intercell_interference_over_noise",
                                        )
                                        selected_action_intercell_cost_after_source_mask_mean = _extract_raw_summary_scalar(
                                            result,
                                            "selected_action_intercell_cost_after_source_mask_mean",
                                        )
                                        selected_action_intercell_cost_after_source_mask_p95 = _extract_raw_summary_scalar(
                                            result,
                                            "selected_action_intercell_cost_after_source_mask_p95",
                                        )
                                        selected_action_intercell_cost_after_source_mask_over_noise_mean = _extract_raw_summary_scalar(
                                            result,
                                            "selected_action_intercell_cost_after_source_mask_over_noise_mean",
                                        )
                                        intercell_per_admitted_packet = _extract_raw_summary_scalar(
                                            result,
                                            "intercell_per_admitted_packet",
                                        )
                                        phase0_actual_partial_minrate_user_count = _extract_raw_summary_scalar(
                                            result,
                                            "phase0_actual_partial_minrate_user_count",
                                        )
                                        phase0_actual_reclaimed_rb_count = _extract_raw_summary_scalar(
                                            result,
                                            "phase0_actual_reclaimed_rb_count",
                                        )
                                        phase0_actual_refill_rb_count = _extract_raw_summary_scalar(
                                            result,
                                            "phase0_actual_refill_rb_count",
                                        )
                                        phase0_actual_refill_gain_mbps = _extract_raw_summary_scalar(
                                            result,
                                            "phase0_actual_refill_gain_mbps",
                                        )
                                        phase0_actual_refill_intercell_delta_over_noise = _extract_raw_summary_scalar(
                                            result,
                                            "phase0_actual_refill_intercell_delta_over_noise",
                                        )
                                        phase_a_rejected_min_rate_total = _extract_raw_summary_scalar(
                                            result,
                                            "phase_a_rejected_min_rate_total",
                                        )
                                        phase_a_rejected_min_rate_per_decision = _extract_raw_summary_scalar(
                                            result,
                                            "phase_a_rejected_min_rate_per_decision",
                                        )
                                        total_embb_throughput = float(result.get("total_embb_throughput", 0.0) or 0.0)
                                        total_urllc_arrivals = float(result.get("total_urllc_arrivals", 0.0) or 0.0)
                                        average_power_consumption = float(result.get("average_power_consumption", 0.0) or 0.0)
                                        embb_power = float(result.get("embb_power", 0.0) or 0.0)
                                        urllc_power = float(result.get("urllc_power", 0.0) or 0.0)
                                        runtime_sec = float(result.get("runtime", 0.0) or 0.0)
                                        embb_blocked_user_count = float(result.get("embb_blocked_user_count", 0.0) or 0.0)
                                        urllc_blocked_user_count = float(result.get("urllc_blocked_user_count", 0.0) or 0.0)
                                        admitted_urllc_count = float(result.get("admitted_urllc_count", 0.0) or 0.0)
                                        row = {
                                        "method": str(policy),
                                        "policy": str(policy),
                                        "policy_label": POLICY_LABELS.get(str(policy), str(policy)),
                                        "seed": int(seed),
                                        "episode_index": int(episode_index),
                                        "episode_seed": int(episode_seed),
                                        "channel_setting_index": int(channel_setting_index),
                                        "mother_topology_id": str(scene_id),
                                        "embb_users": int(embb_users),
                                        "urllc_users": int(urllc_users),
                                        "packet_size": int(packet_bits),
                                        "packet_bits": int(packet_bits),
                                        "runtime_sec": runtime_sec,
                                        "channel_uses": (None if args.channel_uses is None else int(args.channel_uses)),
                                        "lambda_per_user": args.lambda_per_user,
                                        "target_error_probability": args.target_error_probability,
                                        "scenario_label": _scenario_label(int(urllc_users), int(packet_bits)),
                                        "total_embb_throughput": total_embb_throughput,
                                        "embb_sum_rate_after": total_embb_throughput,
                                        "total_urllc_arrivals": total_urllc_arrivals,
                                        "urllc_arrival_count": total_urllc_arrivals,
                                        "average_power_consumption": average_power_consumption,
                                        "total_power": average_power_consumption,
                                        "embb_power": embb_power,
                                        "urllc_power": urllc_power,
                                        "embb_served_user_count": float(result.get("embb_served_user_count", 0.0) or 0.0),
                                        "embb_blocked_user_count": embb_blocked_user_count,
                                        "embb_blocked_count": embb_blocked_user_count,
                                        "urllc_blocked_user_count": urllc_blocked_user_count,
                                        "urllc_blocked_count": urllc_blocked_user_count,
                                        "embb_minimum_rate_violation_count": float(result.get("embb_minimum_rate_violation_count", 0.0) or 0.0),
                                        "overlay_action_count": overlay_count,
                                        "overlay_count": overlay_count,
                                        "puncturing_action_count": puncture_count,
                                        "puncture_count": puncture_count,
                                        "phase_a_total_decisions": phase_a_total_decisions,
                                        "phase_a_rejected_min_rate_total": phase_a_rejected_min_rate_total,
                                        "phase_a_rejected_min_rate_per_decision": phase_a_rejected_min_rate_per_decision,
                                        "keep_action_count": keep_count,
                                        "keep_count": keep_count,
                                        "urllc_admission_ratio": float(result.get("urllc_admission_ratio", 0.0) or 0.0),
                                        "admitted_urllc_count": admitted_urllc_count,
                                        "urllc_admitted_count": admitted_urllc_count,
                                        "embb_user_rates_after": json.dumps(embb_user_rates_after_mbps),
                                        "embb_user_powers_after": json.dumps(embb_user_powers_after_watts),
                                        "embb_jain_after": embb_jain_after,
                                        "embb_min_rate_after": embb_min_rate_after,
                                        "embb_5th_percentile_after": embb_5th_percentile_after,
                                        "embb_median_rate_after": embb_median_rate_after,
                                        "embb_rate_with_intercell_after_puncture_deduction": embb_rate_with_intercell_after_puncture_deduction,
                                        "no_intercell_rate_with_same_puncture_mask": no_intercell_rate_with_same_puncture_mask,
                                        "intercell_rate_loss_with_same_puncture_mask": intercell_rate_loss_with_same_puncture_mask,
                                        "embb_rate_loss_due_to_intercell_ratio": embb_rate_loss_due_to_intercell_ratio,
                                        "mean_intercell_interference_mw": mean_intercell_interference_mw,
                                        "mean_intercell_interference_over_noise": mean_intercell_interference_over_noise,
                                        "selected_action_intercell_cost_after_source_mask_mean": selected_action_intercell_cost_after_source_mask_mean,
                                        "selected_action_intercell_cost_after_source_mask_p95": selected_action_intercell_cost_after_source_mask_p95,
                                        "selected_action_intercell_cost_after_source_mask_over_noise_mean": selected_action_intercell_cost_after_source_mask_over_noise_mean,
                                        "intercell_per_admitted_packet": intercell_per_admitted_packet,
                                        "phase0_actual_partial_minrate_user_count": phase0_actual_partial_minrate_user_count,
                                        "phase0_actual_reclaimed_rb_count": phase0_actual_reclaimed_rb_count,
                                        "phase0_actual_refill_rb_count": phase0_actual_refill_rb_count,
                                        "phase0_actual_refill_gain_mbps": phase0_actual_refill_gain_mbps,
                                        "phase0_actual_refill_intercell_delta_over_noise": phase0_actual_refill_intercell_delta_over_noise,
                                    }
                                        per_run_rows.append(row)
                                        grouped_runs[key].append(row)
                                        completed_run_keys.add(run_key)
                                        latest_row_for_scenario = row
                                        latest_completed_run = row
                                        _write_debug_trace(
                                            out_dir=out_dir,
                                            policy=str(policy),
                                            seed=int(seed),
                                            episode_index=int(episode_index),
                                            episode_seed=int(episode_seed),
                                            embb_users=int(embb_users),
                                            urllc_users=int(urllc_users),
                                            packet_bits=int(packet_bits),
                                            channel_setting_index=int(channel_setting_index),
                                            scene_id=str(scene_id),
                                            result=result,
                                        )
                                        _write_csv(per_run_csv, per_run_rows, _collect_fieldnames(per_run_rows))
                                        print(
                                            f"[FIXED-USER-COMPARE] policy={policy} seed={seed} ep={episode_index} eMBB={embb_users} "
                                            f"URLLC={urllc_users} bits={packet_bits} scene={int(channel_setting_index)} "
                                            f"embb_sumrate={row['total_embb_throughput']:.3e} power={row['average_power_consumption']:.3f}",
                                            flush=True,
                                        )
                                        if progress_bar is not None:
                                            progress_bar.update(1)
                                        if args.save_every_run:
                                            _refresh_outputs(
                                                out_dir=out_dir,
                                                per_run_rows=per_run_rows,
                                                grouped_runs=grouped_runs,
                                                metric_keys=metric_keys,
                                                embb_users_list=embb_group,
                                            urllc_users_list=urllc_users_list,
                                            packet_bits_list=packet_bits_list,
                                            policies=policies,
                                            seeds=seeds,
                                            args=args,
                                            write_plots=False,
                                            latest_completed_run=latest_completed_run,
                                        )
                            finally:
                                _restore_env(env_backup)
                        if latest_row_for_scenario is not None:
                            _refresh_outputs(
                                out_dir=out_dir,
                                per_run_rows=per_run_rows,
                                grouped_runs=grouped_runs,
                                metric_keys=metric_keys,
                                embb_users_list=embb_group,
                                urllc_users_list=urllc_users_list,
                                packet_bits_list=packet_bits_list,
                                policies=policies,
                                seeds=seeds,
                                args=args,
                                write_plots=False,
                                latest_completed_run=latest_completed_run,
                            )
            finally:
                if progress_bar is not None:
                    progress_bar.close()

        _refresh_outputs(
            out_dir=out_dir,
            per_run_rows=per_run_rows,
            grouped_runs=grouped_runs,
            metric_keys=metric_keys,
            embb_users_list=embb_group,
            urllc_users_list=urllc_users_list,
            packet_bits_list=packet_bits_list,
            policies=policies,
            seeds=seeds,
            args=args,
            write_plots=True,
            latest_completed_run=latest_completed_run,
        )
        print(f"[FIXED-USER-COMPARE] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
