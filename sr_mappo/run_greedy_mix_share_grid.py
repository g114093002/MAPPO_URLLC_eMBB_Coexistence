from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_NUM_UAVS = 3


def _format_share_label_from_pct(share_pct: int) -> str:
    return "greedy" if int(share_pct) == 0 else f"share {int(share_pct)}%"


def _format_share_label_from_key(share_key: str) -> str:
    try:
        share_pct = int(str(share_key).replace("share", ""))
    except Exception:
        share_pct = 0
    return _format_share_label_from_pct(share_pct)


def _format_mix_label(mix_label: str) -> str:
    return f"eMBB:URLLC={str(mix_label).split('(', 1)[0].strip()}"


def _to_system_loads(loads, num_uavs: int):
    return np.asarray(loads, dtype=float) * max(int(num_uavs), 1)


def _primary_system_loads(data: dict) -> np.ndarray:
    loads = np.asarray(data.get("loads", []), dtype=float)
    return _to_system_loads(loads, int(data.get("num_uavs", DEFAULT_NUM_UAVS)))


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"
MIX_BASE_EXPERIMENT = {
    "10:0": "phase0_joint_full_power_service_interference_repair_v8_greedy_mix100_debug",
    "7:3": "phase0_joint_full_power_service_interference_repair_v8_greedy_mix73_debug",
    "5:5": "phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug",
    "3:7": "phase0_joint_full_power_service_interference_repair_v8_greedy_mix37_debug",
}


def _run_report(
    experiment: str,
    ratio: float,
    share: float,
    episodes_per_load: int,
    loads_override: str | None = None,
    seed_base: int | None = None,
    urllc_poisson_rate_override: float | None = None,
    share_ref_pre_mbps_by_load: str | None = None,
) -> None:
    env = os.environ.copy()
    env["SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE"] = f"{ratio:.6f}"
    env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(episodes_per_load))
    if loads_override:
        env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(loads_override)
    if seed_base is not None:
        env["SR_MAPPO_REPORT_SEED_BASE"] = str(int(seed_base))
    if urllc_poisson_rate_override is not None:
        env["SR_MAPPO_URLLC_POISSON_RATE_OVERRIDE"] = str(float(urllc_poisson_rate_override))
    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        experiment,
        "--fast",
        "--greedy-only",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def _build_report_cmd_env(
    experiment: str,
    ratio: float,
    share: float,
    episodes_per_load: int,
    loads_override: str | None = None,
    seed_base: int | None = None,
    urllc_poisson_rate_override: float | None = None,
    share_ref_pre_mbps_by_load: str | None = None,
):
    env = os.environ.copy()
    env["SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE"] = f"{ratio:.6f}"
    env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(episodes_per_load))
    if loads_override:
        env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(loads_override)
    if seed_base is not None:
        env["SR_MAPPO_REPORT_SEED_BASE"] = str(int(seed_base))
    if urllc_poisson_rate_override is not None:
        env["SR_MAPPO_URLLC_POISSON_RATE_OVERRIDE"] = str(float(urllc_poisson_rate_override))
    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        experiment,
        "--fast",
        "--greedy-only",
    ]
    return cmd, env


def _build_share_reference_env_value(loads: np.ndarray, pre_mbps: np.ndarray) -> str:
    pairs = []
    for l, v in zip(loads.tolist(), pre_mbps.tolist()):
        pairs.append(f"{float(l):g}:{float(v):.12g}")
    return ",".join(pairs)


def _load_greedy_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    g = j["greedy"]
    admitted_packets = np.asarray(g.get("scheduled_packets", []), dtype=float)
    avg_embb_loss = np.asarray(g.get("greedy_avg_embb_loss", []), dtype=float)
    per_admit_embb_loss = np.divide(
        avg_embb_loss,
        np.maximum(admitted_packets, 1.0),
        out=np.zeros_like(avg_embb_loss, dtype=float),
        where=np.maximum(admitted_packets, 1.0) > 0,
    )
    return {
        "loads": np.asarray(j["loads"], dtype=float),
        "embb_rate_mbps": np.asarray(g["embb_rate"], dtype=float) / 1e6,
        "embb_rate_pre_urllc_admission_mbps": np.asarray(
            g.get("embb_rate_pre_urllc_admission", g.get("embb_rate", [])),
            dtype=float,
        ) / 1e6,
        "urllc_admission": np.asarray(g["urllc_admission"], dtype=float),
        "urllc_tp_mbps": np.asarray(g["urllc_throughput_mbps_slot_est"], dtype=float),
        "urllc_admitted_packets": admitted_packets,
        "budget_used_ratio": np.asarray(g.get("greedy_urllc_budget_used_ratio", []), dtype=float),
        "avg_embb_loss": avg_embb_loss,
        "embb_loss_per_admit": per_admit_embb_loss,
        "embb_service_ratio": np.asarray(g["embb_service_ratio"], dtype=float),
        "embb_min_rate": np.asarray(g["embb_min_rate_satisfaction_ratio"], dtype=float),
        "total_power_mw": np.asarray(g["total_power"], dtype=float) * 1e3,
        "embb_user_count": np.asarray(g.get("embb_user_count", []), dtype=float),
        "urllc_user_count": np.asarray(g.get("urllc_user_count", []), dtype=float),
        "greedy_episode_arrivals_samples": g.get("greedy_episode_arrivals_samples", []),
        "greedy_episode_admitted_samples": g.get("greedy_episode_admitted_samples", []),
        "greedy_episode_budget_used_ratio_samples": g.get("greedy_episode_budget_used_ratio_samples", []),
        "num_uavs": int(j.get("num_uavs", DEFAULT_NUM_UAVS) or DEFAULT_NUM_UAVS),
    }


def _load_full_metrics_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    return {
        "loads": j.get("loads", []),
        "greedy": j.get("greedy", {}),
        "experiment_line": j.get("experiment_line", ""),
        "urllc_poisson_rate": j.get("urllc_poisson_rate", None),
        "episodes_per_load": j.get("episodes_per_load", None),
        "report_run_seed_base": j.get("report_run_seed_base", None),
    }


def _try_load_cached_baseline(
    out_dir: Path,
    mix_key: str,
    loads_expected: list[float],
    episodes_per_load: int,
    seed_base: int | None,
) -> dict | None:
    baseline_path = out_dir / f"embb_only_mix_{mix_key}_sr_mappo_report_metrics.json"
    if not baseline_path.exists():
        return None
    try:
        with baseline_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        loads_found = [float(x) for x in payload.get("loads", [])]
        if len(loads_found) != len(loads_expected):
            return None
        if any(abs(a - b) > 1e-9 for a, b in zip(loads_found, loads_expected)):
            return None
        if int(payload.get("episodes_per_load", -1)) != int(episodes_per_load):
            return None
        found_seed = payload.get("report_run_seed_base", None)
        if seed_base is not None and found_seed is not None and int(found_seed) != int(seed_base):
            return None
        return _load_greedy_metrics(baseline_path)
    except Exception:
        return None


def _try_reuse_cached_share_metrics(
    out_dir: Path,
    mix_key: str,
    share_key: str,
    loads_expected: list[float] | None,
    episodes_per_load: int,
    seed_base: int | None,
    urllc_poisson_rate_override: float | None,
) -> Path | None:
    metrics_path = out_dir / f"{share_key}_mix_{mix_key}_sr_mappo_report_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if loads_expected is not None:
            loads_found = [float(x) for x in payload.get("loads", [])]
            if len(loads_found) != len(loads_expected):
                return None
            if any(abs(a - b) > 1e-9 for a, b in zip(loads_found, loads_expected)):
                return None
        if int(payload.get("episodes_per_load", -1)) != int(episodes_per_load):
            return None
        found_seed = payload.get("report_run_seed_base", None)
        if seed_base is not None and found_seed is not None and int(found_seed) != int(seed_base):
            return None
        if urllc_poisson_rate_override is not None:
            found_lambda = payload.get("urllc_poisson_rate", None)
            if found_lambda is None:
                return None
            if abs(float(found_lambda) - float(urllc_poisson_rate_override)) > 1e-9:
                return None
        return metrics_path
    except Exception:
        return None


def _plot_grid(data: dict, out_path: Path) -> None:
    color_map = {
        "10:0": "#1f77b4",
        "7:3": "#ff7f0e",
        "5:5": "#2ca02c",
        "3:7": "#d62728",
    }
    share_keys = sorted(list(data.keys()))
    mixes_present = sorted({m for sk in share_keys for m in data.get(sk, {}).keys()})
    if not share_keys or not mixes_present:
        return
    # If only one mix is present, overlay all shares in the same panel for direct share-effect comparison.
    if len(mixes_present) == 1:
        mix_label = mixes_present[0]
        metrics = [
            ("embb_rate_mbps", "Aggregate eMBB throughput", "Mbps"),
            ("urllc_admission", "URLLC admission ratio", "Ratio"),
            ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
            ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(26, 6.4), constrained_layout=True)
        legend_handles = None
        legend_labels = None
        share_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for c, (k, title, ylab) in enumerate(metrics):
            ax = axes[c]
            for idx, share_key in enumerate(share_keys):
                if mix_label not in data.get(share_key, {}):
                    continue
                d = data[share_key][mix_label]
                share_pct = int(share_key.replace("share", ""))
                ax.plot(
                    _primary_system_loads(d),
                    d[k],
                    marker="o",
                    linewidth=2,
                    color=share_colors[idx % len(share_colors)],
                    label=_format_share_label_from_pct(share_pct),
                )
            ax.set_title(title)
            ax.set_xlabel("Total system load")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.35)
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        fig.suptitle(f"Greedy Mix ({mix_label}) Share Comparison", fontsize=16)
        if legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.03),
                ncol=min(len(legend_labels), 4),
                frameon=False,
                fontsize=10,
            )
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return
    metrics = [
        ("embb_rate_pre_urllc_admission_mbps", "Aggregate eMBB throughput (pre-URLLC admission)", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
    ]
    fig, axes = plt.subplots(len(share_keys), 3, figsize=(21, max(5.6 * len(share_keys), 6.8)), constrained_layout=True)
    if len(share_keys) == 1:
        axes = np.asarray([axes])
    legend_handles = None
    legend_labels = None
    for r, share_key in enumerate(share_keys):
        share_pct = int(share_key.replace("share", ""))
        for c, (k, title, ylab) in enumerate(metrics):
            ax = axes[r][c]
            for mix in mixes_present:
                d = data[share_key][mix]
                if k == "urllc_admission" and mix == "10:0":
                    continue
                ax.plot(
                    _primary_system_loads(d),
                    d[k],
                    marker="o",
                    linewidth=2,
                    color=color_map[mix],
                    label=_format_mix_label(mix),
                )
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"Share={share_pct}%\n{ylab}")
            else:
                ax.set_ylabel(ylab)
            ax.set_xlabel("Total system load")
            ax.grid(True, alpha=0.35)
            if r == 0 and c == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle(f"Greedy Mix ({'/'.join(mixes_present)}) x Share Comparison", fontsize=16)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_per_mix_share_comparison(
    data: dict,
    mix_label: str,
    out_path: Path,
    share_keys: list[str],
    embb_only_baseline: dict | None = None,
) -> None:
    share_labels = []
    for share_key in share_keys:
        if not share_key.startswith("share"):
            continue
        share_num = share_key.replace("share", "")
        try:
            share_pct = int(share_num)
        except ValueError:
            continue
        share_labels.append((share_key, _format_share_label_from_pct(share_pct)))
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_map = {label: color_cycle[idx % len(color_cycle)] for idx, (_, label) in enumerate(share_labels)}
    metrics = [
        ("embb_rate_pre_urllc_admission_mbps", "Aggregate eMBB throughput (pre-URLLC admission)", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
        ("budget_used_ratio", "Share-budget used ratio", "Ratio"),
        ("avg_embb_loss", "Average eMBB loss", "bps"),
        ("embb_loss_per_admit", "eMBB loss per admitted packet", "bps/packet"),
        ("embb_service_ratio", "eMBB service ratio", "Ratio"),
        ("embb_min_rate", "eMBB min-rate satisfaction", "Ratio"),
        ("total_power_mw", "Total transmit power", "mW"),
    ]

    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 5.4 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    legend_handles = None
    legend_labels = None
    for ax, (k, title, ylab) in zip(axes, metrics):
        baseline_loads = None
        baseline_pre = None
        if k == "embb_rate_pre_urllc_admission_mbps" and embb_only_baseline is not None:
            baseline_loads = _primary_system_loads(embb_only_baseline)
            baseline_pre = np.asarray(
                embb_only_baseline.get(
                    "embb_rate_pre_urllc_admission_mbps",
                    embb_only_baseline.get("embb_rate_mbps", []),
                ),
                dtype=float,
            )
        for share_key, share_text in share_labels:
            if share_key not in data or mix_label not in data.get(share_key, {}):
                continue
            d = data[share_key][mix_label]
            if k == "embb_rate_pre_urllc_admission_mbps" and baseline_loads is not None and baseline_pre is not None and baseline_loads.size:
                try:
                    share_ratio = float(share_text.replace("%", "")) / 100.0
                except Exception:
                    share_ratio = 0.0
                # Plot fixed-discount semantics directly against the share0 baseline.
                y = baseline_pre * (1.0 - share_ratio)
                x = baseline_loads
            else:
                x = _primary_system_loads(d)
                y = d[k]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=color_map[share_text],
                label=share_text,
            )
        if k == "embb_rate_pre_urllc_admission_mbps" and embb_only_baseline is not None:
            b_loads = np.asarray(embb_only_baseline.get("loads", []), dtype=float)
            b_embb = np.asarray(
                embb_only_baseline.get(
                    "embb_rate_pre_urllc_admission_mbps",
                    embb_only_baseline.get("embb_rate_mbps", []),
                ),
                dtype=float,
            )
            if b_loads.size and b_embb.size:
                ax.plot(
                    _to_system_loads(b_loads, int(embb_only_baseline.get("num_uavs", DEFAULT_NUM_UAVS))),
                    b_embb,
                    linestyle="--",
                    linewidth=2.2,
                    color="#111111",
                    label="pre-URLLC eMBB baseline (same mix)",
                )
        ax.set_title(title)
        ax.set_xlabel("Total system load")
        ax.set_ylabel(ylab)
        if share_labels:
            first_share_key = share_labels[0][0]
            if first_share_key in data and mix_label in data[first_share_key]:
                ax.set_xticks(_primary_system_loads(data[first_share_key][mix_label]))
        ax.grid(True, alpha=0.35)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Greedy Share Comparison under eMBB:URLLC = {mix_label}", fontsize=15)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_per_share_mix_comparison(data: dict, share_key: str, out_path: Path) -> None:
    mix_order = [m for m in ["10:0", "7:3", "5:5", "3:7"] if m in data.get(share_key, {})]
    if not mix_order:
        return
    color_map = {
        "10:0": "#1f77b4",
        "7:3": "#ff7f0e",
        "5:5": "#2ca02c",
        "3:7": "#d62728",
    }
    metrics = [
        ("embb_rate_mbps", "Aggregate eMBB throughput", "Mbps"),
        ("urllc_admission", "URLLC admission ratio", "Ratio"),
        ("urllc_tp_mbps", "URLLC throughput (slot est.)", "Mbps"),
        ("urllc_admitted_packets", "URLLC admitted packets", "Packets"),
        ("budget_used_ratio", "Share-budget used ratio", "Ratio"),
        ("avg_embb_loss", "Average eMBB loss", "bps"),
        ("embb_loss_per_admit", "eMBB loss per admitted packet", "bps/packet"),
        ("embb_service_ratio", "eMBB service ratio", "Ratio"),
        ("embb_min_rate", "eMBB min-rate satisfaction", "Ratio"),
        ("total_power_mw", "Total transmit power", "mW"),
    ]

    share_pct = int(share_key.replace("share", ""))
    ncols = 3
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 5.4 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    legend_handles = None
    legend_labels = None
    for ax, (k, title, ylab) in zip(axes, metrics):
        for mix_label in mix_order:
            d = data[share_key][mix_label]
            if k == "urllc_admission" and mix_label == "10:0":
                continue
            ax.plot(
                _primary_system_loads(d),
                d[k],
                marker="o",
                linewidth=2,
                color=color_map[mix_label],
                label=_format_mix_label(mix_label),
            )
        ax.set_title(title)
        ax.set_xlabel("Total system load")
        ax.set_ylabel(ylab)
        if mix_order:
            ax.set_xticks(_primary_system_loads(data[share_key][mix_order[0]]))
        ax.grid(True, alpha=0.35)
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Greedy Mix Comparison under Share={share_pct}%", fontsize=15)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_labels), 4),
            frameon=False,
            fontsize=10,
        )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_share_budget_hit_cdf_for_mix(data: dict, mix_label: str, out_path: Path) -> None:
    """CDF-like view: for each share, how many arrived packets are seen before share cap is exhausted.

    We approximate "cap reached at packet k" by:
      - if budget_used_ratio < 0.999 in an episode: no cap hit -> k = arrivals (right-tail point)
      - else: k = admitted (admissions consumed until cap prevented further admits)
    """
    # Share mode is disabled in current branch; support both legacy multi-share
    # payloads and no-share (share00) payloads.
    share_labels = []
    for key in sorted(data.keys()):
        if not key.startswith("share"):
            continue
        try:
            pct = int(key.replace("share", ""))
            share_labels.append((key, _format_share_label_from_pct(pct)))
        except ValueError:
            continue
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_map = {label: color_cycle[idx % len(color_cycle)] for idx, (_k, label) in enumerate(share_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plotted_any = False
    for share_key, share_text in share_labels:
        if share_key not in data or mix_label not in data[share_key]:
            continue
        d = data[share_key][mix_label]
        arrivals_by_load = d.get("greedy_episode_arrivals_samples", [])
        admitted_by_load = d.get("greedy_episode_admitted_samples", [])
        budget_used_by_load = d.get("greedy_episode_budget_used_ratio_samples", [])
        pkt_until_cap: list[float] = []
        for a_list, m_list, b_list in zip(arrivals_by_load, admitted_by_load, budget_used_by_load):
            for a, m, b in zip(a_list, m_list, b_list):
                a = float(a)
                m = float(m)
                b = float(b)
                if a <= 0.0:
                    continue
                if b >= 0.999:
                    pkt_until_cap.append(max(0.0, min(m, a)))
                else:
                    pkt_until_cap.append(a)
        if not pkt_until_cap:
            continue
        x = np.sort(np.asarray(pkt_until_cap, dtype=float))
        y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
        ax.plot(x, y, linewidth=2, color=color_map[share_text], label=share_text)
        plotted_any = True

    ax.set_title(f"Packet Index to Share-Cap Exhaustion CDF (mix {mix_label})")
    ax.set_xlabel("Arrivals processed before cap exhaustion (episode proxy)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.35)
    if plotted_any:
        ax.legend(fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "No share-cap data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run greedy mix-share grid and plot comparisons.")
    parser.add_argument(
        "--experiment",
        default="phase0_joint_full_power_service_interference_repair_v8_greedy_share20_debug",
        help="Base greedy experiment preset to reuse while overriding mix/share at runtime.",
    )
    parser.add_argument("--episodes-per-load", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "mix_share_grid"),
        help="Directory for archived metrics and comparison plots.",
    )
    parser.add_argument(
        "--mixes",
        default="10:0,7:3,5:5,3:7",
        help="Comma-separated mix list from {10:0,7:3,5:5,3:7}.",
    )
    parser.add_argument(
        "--shares",
        default="10,20,30",
        help="Comma-separated share percentages. Example: 10,20,30",
    )
    parser.add_argument(
        "--loads",
        default="",
        help="Comma-separated loads override. Example: 5,10,15,20,25,30,40,50,60,70,80",
    )
    parser.add_argument(
        "--urllc-poisson-rate",
        type=float,
        default=None,
        help="Override URLLC Poisson rate (pkts/slot) for this run.",
    )
    parser.add_argument(
        "--paired-seed",
        action="store_true",
        help="Use the same report seed base across all runs in this grid for scenario pairing (default enabled).",
    )
    parser.add_argument(
        "--unpaired-seed",
        action="store_true",
        help="Disable paired seed and allow independent random seeds across runs.",
    )
    parser.add_argument(
        "--paired-seed-base",
        type=int,
        default=None,
        help="Explicit fixed seed base for full reproducible pairing across all runs.",
    )
    parser.add_argument(
        "--parallel-shares",
        dest="parallel_shares",
        action="store_true",
        help="Run shares of the same mix in parallel subprocesses to reduce wall-clock time.",
    )
    parser.add_argument(
        "--sequential-shares",
        dest="parallel_shares",
        action="store_false",
        help="Force sequential share execution (debug only).",
    )
    parser.set_defaults(parallel_shares=True)
    parser.add_argument(
        "--respect-mix-preset",
        dest="respect_mix_preset",
        action="store_true",
        help="Use the canonical mix-specific experiment preset as base (recommended).",
    )
    parser.add_argument(
        "--no-respect-mix-preset",
        dest="respect_mix_preset",
        action="store_false",
        help="Force using --experiment for all mixes (legacy behavior).",
    )
    parser.set_defaults(respect_mix_preset=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_pool = {
        "10:0": 0.0,
        "7:3": 0.3,
        "5:5": 0.5,
        "3:7": 0.7,
    }
    mix_list = [m.strip() for m in str(args.mixes).split(",") if m.strip()]
    invalid_mixes = [m for m in mix_list if m not in mix_pool]
    if invalid_mixes:
        raise ValueError(f"Unsupported mixes: {invalid_mixes}. Allowed: {list(mix_pool.keys())}")
    mixes = {m: mix_pool[m] for m in mix_list}

    # Share dimension is fully disabled. Keep CLI arg for backward compatibility
    # but force a single no-share run.
    shares = [0.0]
    loads_override = str(args.loads or "").strip()
    loads_expected = None
    if loads_override:
        try:
            loads_expected = [float(x.strip()) for x in loads_override.split(",") if x.strip()]
        except Exception:
            loads_expected = None

    paired_enabled = (not bool(args.unpaired_seed))
    if args.paired_seed_base is not None:
        paired_seed_base = int(args.paired_seed_base)
    else:
        if paired_enabled:
            paired_seed_base = int(secrets.randbelow(1_000_000_000))
        else:
            paired_seed_base = None
    if paired_seed_base is not None:
        print(f"[GRID] paired seed base = {paired_seed_base}", flush=True)

    all_data: dict = {}
    embb_only_baseline_by_mix: dict[str, dict | None] = {}
    detailed_payload: dict = {"meta": {}, "grid": {}}
    detailed_payload["meta"] = {
        "base_experiment": args.experiment,
        "episodes_per_load": int(args.episodes_per_load),
        "mixes": list(mixes.keys()),
        "shares": [float(s) for s in shares],
        "urllc_poisson_rate_override": (
            float(args.urllc_poisson_rate) if args.urllc_poisson_rate is not None else None
        ),
        "respect_mix_preset": bool(args.respect_mix_preset),
        "paired_seed_enabled": bool(paired_seed_base is not None),
        "paired_seed_base": int(paired_seed_base) if paired_seed_base is not None else None,
    }
    for share in shares:
        share_key = f"share{int(round(share * 100)):02d}"
        all_data[share_key] = {}
        detailed_payload["grid"][share_key] = {}

    for mix_label, ratio in mixes.items():
        run_experiment = MIX_BASE_EXPERIMENT.get(mix_label, args.experiment) if args.respect_mix_preset else args.experiment
        print(f"[GRID] mix={mix_label} using experiment preset: {run_experiment}", flush=True)
        mix_key = mix_label.replace(":", "_")
        # Share mode is removed; run only one share00 greedy report per mix.
        embb_only_baseline_by_mix[mix_label] = None

        if args.parallel_shares:
            procs = []
            run_dirs = []
            for share in shares:
                share_key = f"share{int(round(share * 100)):02d}"
                cached_metrics = _try_reuse_cached_share_metrics(
                    out_dir=out_dir,
                    mix_key=mix_key,
                    share_key=share_key,
                    loads_expected=loads_expected,
                    episodes_per_load=int(args.episodes_per_load),
                    seed_base=paired_seed_base,
                    urllc_poisson_rate_override=args.urllc_poisson_rate,
                )
                if cached_metrics is not None:
                    print(
                        f"[GRID] reuse cached share metrics: {cached_metrics}",
                        flush=True,
                    )
                    core_candidate = out_dir / f"{share_key}_mix_{mix_key}_core_kpi_debug.png"
                    if not core_candidate.exists():
                        print(
                            f"[GRID] cached metrics found but missing {core_candidate.name}; plotting later from metrics.",
                            flush=True,
                        )
                    continue
                run_dir = out_dir / f"_run_{mix_key}_{share_key}_{int(time.time()*1000)}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd, env = _build_report_cmd_env(
                    run_experiment,
                    ratio=ratio,
                    share=share,
                    episodes_per_load=args.episodes_per_load,
                    loads_override=loads_override,
                    seed_base=paired_seed_base,
                    urllc_poisson_rate_override=args.urllc_poisson_rate,
                )
                env["SR_MAPPO_RESULTS_DIR_OVERRIDE"] = str(run_dir)
                print(
                    f"[GRID] running(parallel) mix={mix_label}, share={share:.2f}, seed_base={paired_seed_base} ...",
                    flush=True,
                )
                procs.append((share_key, subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)))
                run_dirs.append((share_key, run_dir))
            for run_tag, p in procs:
                rc = p.wait()
                if rc != 0:
                    raise RuntimeError(f"parallel run failed: mix={mix_label}, tag={run_tag}, rc={rc}")
            for run_tag, run_dir in run_dirs:
                metrics_src = run_dir / "sr_mappo_report_metrics.json"
                core_src = run_dir / "core_kpi_debug.png"
                share = float(run_tag.replace("share", "")) / 100.0
                share_key = f"share{int(round(share * 100)):02d}"
                metrics_dst = out_dir / f"{share_key}_mix_{mix_key}_sr_mappo_report_metrics.json"
                core_dst = out_dir / f"{share_key}_mix_{mix_key}_core_kpi_debug.png"
                shutil.copy2(metrics_src, metrics_dst)
                if core_src.exists():
                    shutil.copy2(core_src, core_dst)
        else:
            for share in shares:
                share_key = f"share{int(round(share * 100)):02d}"
                cached_metrics = _try_reuse_cached_share_metrics(
                    out_dir=out_dir,
                    mix_key=mix_key,
                    share_key=share_key,
                    loads_expected=loads_expected,
                    episodes_per_load=int(args.episodes_per_load),
                    seed_base=paired_seed_base,
                    urllc_poisson_rate_override=args.urllc_poisson_rate,
                )
                if cached_metrics is not None:
                    print(
                        f"[GRID] reuse cached share metrics: {cached_metrics}",
                        flush=True,
                    )
                    continue
                print(
                    f"[GRID] running mix={mix_label}, share={share:.2f}, seed_base={paired_seed_base} ...",
                    flush=True,
                )
                _run_report(
                    run_experiment,
                    ratio=ratio,
                    share=share,
                    episodes_per_load=args.episodes_per_load,
                    loads_override=loads_override,
                    seed_base=paired_seed_base,
                    urllc_poisson_rate_override=args.urllc_poisson_rate,
                )
                metrics_dst = out_dir / f"{share_key}_mix_{mix_key}_sr_mappo_report_metrics.json"
                core_dst = out_dir / f"{share_key}_mix_{mix_key}_core_kpi_debug.png"
                shutil.copy2(RESULTS_DIR / "sr_mappo_report_metrics.json", metrics_dst)
                core_src = RESULTS_DIR / "core_kpi_debug.png"
                if core_src.exists():
                    shutil.copy2(core_src, core_dst)

        for share in shares:
            share_key = f"share{int(round(share * 100)):02d}"
            metrics_dst = out_dir / f"{share_key}_mix_{mix_key}_sr_mappo_report_metrics.json"
            d = _load_greedy_metrics(metrics_dst)
            detailed_payload["grid"][share_key][mix_label] = _load_full_metrics_payload(metrics_dst)
            detailed_payload["grid"][share_key][mix_label]["greedy_episode_arrivals_samples"] = d.get(
                "greedy_episode_arrivals_samples", []
            )
            detailed_payload["grid"][share_key][mix_label]["greedy_episode_admitted_samples"] = d.get(
                "greedy_episode_admitted_samples", []
            )
            detailed_payload["grid"][share_key][mix_label]["greedy_episode_budget_used_ratio_samples"] = d.get(
                "greedy_episode_budget_used_ratio_samples", []
            )
            embb_users = d["embb_user_count"]
            urllc_users = d["urllc_user_count"]
            if embb_users.size and urllc_users.size:
                eu = float(np.mean(embb_users))
                uu = float(np.mean(urllc_users))
                total = eu + uu
                realized_urllc_ratio = (uu / total) if total > 0 else 0.0
                print(
                    f"[GRID][CHECK] mix={mix_label} share={share:.2f} target_urllc_ratio={ratio:.3f} "
                    f"realized_mean(embb/urllc)={eu:.2f}/{uu:.2f} "
                    f"realized_urllc_ratio={realized_urllc_ratio:.3f}",
                    flush=True,
                )
                if abs(realized_urllc_ratio - ratio) > 0.08:
                    raise RuntimeError(
                        f"ratio mismatch: mix={mix_label}, target={ratio:.3f}, realized={realized_urllc_ratio:.3f}"
                    )
            all_data[share_key][mix_label] = d

    summary_plot = out_dir / "mix_share_grid_comparison.png"
    _plot_grid(all_data, summary_plot)

    # Additional 3 figures requested: fixed mix (7:3/5:5/3:7), compare share 10/20/30.
    per_mix_targets = [m for m in ["7:3", "5:5", "3:7"] if m in mixes]
    share_keys_ordered = [f"share{int(round(s * 100)):02d}" for s in shares]
    for mix_label in per_mix_targets:
        mix_key = mix_label.replace(":", "_")
        out_plot = out_dir / f"mix_{mix_key}_share_10_20_30_comparison.png"
        _plot_per_mix_share_comparison(
            all_data,
            mix_label,
            out_plot,
            share_keys_ordered,
            embb_only_baseline=embb_only_baseline_by_mix.get(mix_label),
        )
        cdf_plot = out_dir / f"mix_{mix_key}_share_cap_packet_cdf.png"
        _plot_share_budget_hit_cdf_for_mix(all_data, mix_label, cdf_plot)

    # Additional 3 figures: fixed share (10/20/30), compare all mix ratios.
    for share in shares:
        share_key = f"share{int(round(share * 100)):02d}"
        mix_label_suffix = "_".join([m.replace(":", "_") for m in mixes.keys()])
        share_plot = out_dir / f"{share_key}_mix_{mix_label_suffix}_comparison.png"
        _plot_per_share_mix_comparison(all_data, share_key, share_plot)

    detailed_json = out_dir / "mix_share_grid_detailed_metrics.json"
    with detailed_json.open("w", encoding="utf-8") as f:
        json.dump(detailed_payload, f, indent=2)

    print(f"[GRID] done. summary plot: {summary_plot}", flush=True)


if __name__ == "__main__":
    main()
