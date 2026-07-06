from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_fixed_user_blocklength_compare import DEFAULT_MAPPO_CHECKPOINT, POLICY_LABELS, _build_policy_config, _parse_csv_ints
from .unified_policy_runner import run_policy


BOXPLOT_PACKET_BITS = 150
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_episode_row(row: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    int_keys = {"seed", "embb_users", "urllc_users", "packet_bits"}
    float_keys = {"jain_fairness", "total_embb_throughput"}
    for key in int_keys:
        value = normalized.get(key, None)
        normalized[key] = int(round(float(value))) if value not in (None, "", "None") else 0
    for key in float_keys:
        value = normalized.get(key, None)
        normalized[key] = float(value) if value not in (None, "", "None") else 0.0
    return normalized


def _normalize_user_row(row: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(row)
    int_keys = {"seed", "embb_users", "urllc_users", "packet_bits", "embb_user_id"}
    float_keys = {"embb_rate_bps", "embb_rate_mbps"}
    for key in int_keys:
        value = normalized.get(key, None)
        normalized[key] = int(round(float(value))) if value not in (None, "", "None") else 0
    for key in float_keys:
        value = normalized.get(key, None)
        normalized[key] = float(value) if value not in (None, "", "None") else 0.0
    return normalized


def _episode_key(row: Dict[str, object]) -> Tuple[int, int, int, str, int]:
    return (
        int(row["embb_users"]),
        int(row["urllc_users"]),
        int(row["packet_bits"]),
        str(row["policy"]),
        int(row["seed"]),
    )


def _extract_embb_user_rates_mbps(result: Dict[str, object]) -> np.ndarray:
    raw_summary = dict(result.get("raw_summary", {}) or {})
    rates = raw_summary.get("embb_user_rates_after_puncture_deduction", raw_summary.get("embb_user_rates", []))
    arr = np.asarray(rates, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr / 1.0e6


def _served_only_rates_mbps(rates_mbps: np.ndarray) -> np.ndarray:
    arr = np.asarray(rates_mbps, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return arr[arr > 1.0e-12]


def _extract_jain_fairness(result: Dict[str, object]) -> float:
    rates_mbps = _served_only_rates_mbps(_extract_embb_user_rates_mbps(result))
    if rates_mbps.size == 0:
        return 0.0
    numer = float(np.sum(rates_mbps) ** 2)
    denom = float(rates_mbps.size * np.sum(np.square(rates_mbps)))
    return float(numer / denom) if denom > 0.0 else 0.0


def _plot_boxplot(
    path: Path,
    *,
    embb_users: int,
    urllc_users_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
    user_rows: List[Dict[str, object]],
) -> None:
    filtered_rows = [row for row in user_rows if int(row["embb_users"]) == int(embb_users) and int(row["packet_bits"]) == BOXPLOT_PACKET_BITS]
    grouped: Dict[Tuple[int, str], List[float]] = {}
    for urllc_users in urllc_users_list:
        for policy in policies:
            grouped[(int(urllc_users), str(policy))] = [
                float(row["embb_rate_mbps"])
                for row in filtered_rows
                if int(row["urllc_users"]) == int(urllc_users) and str(row["policy"]) == str(policy)
            ]

    fig, ax = plt.subplots(figsize=(13.5, 5.8), constrained_layout=True)
    x = np.arange(len(urllc_users_list), dtype=float)
    width = 0.8 / max(len(policies), 1)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    legend_handles = []
    legend_labels = []
    for policy_idx, policy in enumerate(policies):
        positions = []
        data = []
        for group_idx, urllc_users in enumerate(urllc_users_list):
            positions.append(x[group_idx] + (policy_idx - (len(policies) - 1) / 2.0) * width)
            data.append(grouped[(int(urllc_users), str(policy))])
        color = default_colors[policy_idx % len(default_colors)] if default_colors else None
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.65, edgecolor="black", linewidth=0.8)
        for median in bp["medians"]:
            median.set(color="black", linewidth=1.3)
        for whisker in bp["whiskers"]:
            whisker.set(color="black", linewidth=0.8)
        for cap in bp["caps"]:
            cap.set(color="black", linewidth=0.8)
        for flier in bp["fliers"]:
            flier.set(marker="o", markersize=3.0, markerfacecolor=color, markeredgecolor="black", alpha=0.45)
        legend_handles.append(bp["boxes"][0])
        legend_labels.append(policy_label_map.get(str(policy), POLICY_LABELS.get(str(policy), str(policy))))

    ax.set_title(f"eMBB User-Rate Distribution (fixed eMBB={embb_users}, B{BOXPLOT_PACKET_BITS})")
    ax.set_xlabel("URLLC users")
    ax.set_ylabel("Per-user eMBB rate (Mbps)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(legend_handles, legend_labels, ncol=min(len(policies), 3), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_jain_fairness(
    path: Path,
    *,
    embb_users: int,
    urllc_users_list: List[int],
    packet_bits_list: List[int],
    policies: List[str],
    policy_label_map: Dict[str, str],
    episode_rows: List[Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, len(packet_bits_list), figsize=(max(12.0, len(packet_bits_list) * 4.2), 4.8), sharey=True)
    if len(packet_bits_list) == 1:
        axes = [axes]
    x = np.asarray(urllc_users_list, dtype=float)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    filtered_rows = [row for row in episode_rows if int(row["embb_users"]) == int(embb_users)]

    for panel_idx, (ax, packet_bits) in enumerate(zip(axes, packet_bits_list)):
        for policy_idx, policy in enumerate(policies):
            means = []
            for urllc_users in urllc_users_list:
                vals = [
                    float(row["jain_fairness"])
                    for row in filtered_rows
                    if int(row["packet_bits"]) == int(packet_bits)
                    and int(row["urllc_users"]) == int(urllc_users)
                    and str(row["policy"]) == str(policy)
                ]
                means.append(float(np.mean(np.asarray(vals, dtype=float))) if vals else 0.0)
            ax.plot(
                x,
                means,
                marker=MARKERS[policy_idx % len(MARKERS)],
                linewidth=2.0,
                markersize=6.0,
                color=default_colors[policy_idx % len(default_colors)] if default_colors else None,
                label=policy_label_map.get(str(policy), POLICY_LABELS.get(str(policy), str(policy))),
            )
        panel_tag = panel_labels[panel_idx] if panel_idx < len(panel_labels) else f"({panel_idx + 1})"
        ax.set_title(f"{panel_tag} B{int(packet_bits)}")
        ax.set_xlabel("URLLC users")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in urllc_users_list])
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.0, 1.02)

    axes[0].set_ylabel("Jain fairness index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(policies), 3), frameon=False)
    fig.suptitle(f"eMBB Jain Fairness under Different Block Lengths (fixed eMBB={embb_users})", y=1.08)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute eMBB user-rate distribution and Jain fairness figures for fixed-user comparisons.")
    parser.add_argument("--embb-users", default="10,20")
    parser.add_argument("--urllc-users", default="10,20,30,40,50")
    parser.add_argument("--packet-bits", default="24")
    parser.add_argument("--policies", default="greedy,mappo,pure_puncturing,pure_superposition,random_scheduler")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--channel-uses", type=int, default=None)
    parser.add_argument("--lambda-per-user", type=float, default=None)
    parser.add_argument("--target-error-probability", type=float, default=None)
    parser.add_argument("--mappo-checkpoint-path", default=str(DEFAULT_MAPPO_CHECKPOINT))
    parser.add_argument("--out-dir", default="sr_mappo/results/fixed_compare_embb_rate_fairness")
    args = parser.parse_args()

    embb_users_list = _parse_csv_ints(args.embb_users)
    urllc_users_list = _parse_csv_ints(args.urllc_users)
    packet_bits_list = _parse_csv_ints(args.packet_bits)
    seeds = _parse_csv_ints(args.seeds)
    policies = [token.strip() for token in str(args.policies).split(",") if token.strip()]
    policy_label_map = {policy: POLICY_LABELS.get(policy, policy) for policy in policies}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = out_dir / "episode_level_metrics.csv"
    user_csv = out_dir / "embb_user_rate_samples.csv"

    episode_rows = [_normalize_episode_row(row) for row in _read_csv_rows(episode_csv)]
    user_rows = [_normalize_user_row(row) for row in _read_csv_rows(user_csv)]
    completed = {_episode_key(row) for row in episode_rows}

    for embb_users in embb_users_list:
        for urllc_users in urllc_users_list:
            for packet_bits in packet_bits_list:
                for policy in policies:
                    for seed in seeds:
                        run_key = (int(embb_users), int(urllc_users), int(packet_bits), str(policy), int(seed))
                        if run_key in completed:
                            print(
                                f"[EMBB-RATE-FAIRNESS] skip existing policy={policy} seed={seed} "
                                f"eMBB={embb_users} URLLC={urllc_users} bits={packet_bits}",
                                flush=True,
                            )
                            continue
                        cfg = _build_policy_config(
                            policy=str(policy),
                            embb_users=int(embb_users),
                            urllc_users=int(urllc_users),
                            packet_bits=int(packet_bits),
                            channel_uses=args.channel_uses,
                            lambda_per_user=args.lambda_per_user,
                            target_error_probability=args.target_error_probability,
                            mappo_checkpoint_path=args.mappo_checkpoint_path,
                            geometry_profile=None,
                        )
                        result = run_policy(str(policy), deepcopy(cfg), int(seed))
                        rates_mbps = _served_only_rates_mbps(_extract_embb_user_rates_mbps(result))
                        jain = _extract_jain_fairness(result)
                        ep_row = {
                            "policy": str(policy),
                            "policy_label": policy_label_map.get(str(policy), str(policy)),
                            "seed": int(seed),
                            "embb_users": int(embb_users),
                            "urllc_users": int(urllc_users),
                            "packet_bits": int(packet_bits),
                            "jain_fairness": float(jain),
                            "total_embb_throughput": float(result.get("total_embb_throughput", 0.0) or 0.0),
                        }
                        episode_rows.append(ep_row)
                        for embb_user_id, rate_mbps in enumerate(rates_mbps):
                            user_rows.append(
                                {
                                    "policy": str(policy),
                                    "policy_label": policy_label_map.get(str(policy), str(policy)),
                                    "seed": int(seed),
                                    "embb_users": int(embb_users),
                                    "urllc_users": int(urllc_users),
                                    "packet_bits": int(packet_bits),
                                    "embb_user_id": int(embb_user_id),
                                    "embb_rate_bps": float(rate_mbps * 1.0e6),
                                    "embb_rate_mbps": float(rate_mbps),
                                }
                            )
                        completed.add(run_key)
                        _write_csv(episode_csv, episode_rows, list(episode_rows[0].keys()))
                        _write_csv(user_csv, user_rows, list(user_rows[0].keys()))
                        print(
                            f"[EMBB-RATE-FAIRNESS] policy={policy} seed={seed} eMBB={embb_users} "
                            f"URLLC={urllc_users} bits={packet_bits} jain={jain:.4f}",
                            flush=True,
                        )

    for embb_users in embb_users_list:
        _plot_boxplot(
            plots_dir / f"embb_{embb_users}_user_rate_distribution_B{BOXPLOT_PACKET_BITS}.png",
            embb_users=int(embb_users),
            urllc_users_list=urllc_users_list,
            policies=policies,
            policy_label_map=policy_label_map,
            user_rows=user_rows,
        )
        _plot_jain_fairness(
            plots_dir / f"embb_{embb_users}_jain_fairness_by_blocklength.png",
            embb_users=int(embb_users),
            urllc_users_list=urllc_users_list,
            packet_bits_list=packet_bits_list,
            policies=policies,
            policy_label_map=policy_label_map,
            episode_rows=episode_rows,
        )

    summary = {
        "embb_users": embb_users_list,
        "urllc_users": urllc_users_list,
        "packet_bits": packet_bits_list,
        "policies": policies,
        "seeds": seeds,
        "boxplot_packet_bits": int(BOXPLOT_PACKET_BITS),
        "episode_csv": str(episode_csv),
        "user_csv": str(user_csv),
        "plots_dir": str(plots_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[EMBB-RATE-FAIRNESS] wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
