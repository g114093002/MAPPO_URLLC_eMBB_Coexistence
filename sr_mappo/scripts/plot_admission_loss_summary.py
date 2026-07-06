from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sr_mappo.run_fixed_user_blocklength_compare import _build_policy_config
from sr_mappo.unified_policy_runner import run_policy


POLICY_LABELS = {
    "greedy": "Greedy",
    "pure_puncturing": "Greedy(pure puncturing)",
    "pure_superposition": "Greedy(pure superposition)",
}


def _losses_mbps(result: Dict[str, object]) -> np.ndarray:
    samples = list(result.get("per_admission_embb_damage_samples", []) or [])
    return np.asarray(
        [float(sample.get("embb_rate_loss_due_to_action", 0.0) or 0.0) / 1.0e6 for sample in samples],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot URLLC admission and eMBB per-packet loss summary.")
    parser.add_argument("--embb-users", type=int, required=True)
    parser.add_argument("--urllc-users", type=int, required=True)
    parser.add_argument("--packet-bits", default="24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-prefix", type=str, default=None)
    parser.add_argument("--min-overlay-retention", type=float, default=0.90)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    packet_bits = [int(token.strip()) for token in str(args.packet_bits).split(",") if token.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = ["greedy", "pure_puncturing", "pure_superposition"]
    by_policy: Dict[str, Dict[int, Dict[str, object]]] = {policy: {} for policy in policies}

    for bits in packet_bits:
        scene_id = None
        if str(args.scene_prefix or "").strip():
            scene_id = f"{str(args.scene_prefix).strip()}_k{int(bits)}_scene1"
        if scene_id:
            import os

            os.environ["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
            os.environ["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = scene_id
        for policy in policies:
            cfg = _build_policy_config(
                policy=policy,
                embb_users=int(args.embb_users),
                urllc_users=int(args.urllc_users),
                packet_bits=int(bits),
                channel_uses=None,
                lambda_per_user=None,
                target_error_probability=None,
                mappo_checkpoint_path=None,
                geometry_profile=None,
                min_overlay_retention=float(args.min_overlay_retention),
            )
            result = run_policy(policy, cfg, int(args.seed))
            losses = _losses_mbps(result)
            admitted = float(len(losses))
            arrivals = float(result.get("total_urllc_arrivals", 0.0) or 0.0)
            by_policy[policy][int(bits)] = {
                "losses": losses,
                "admitted": admitted,
                "arrivals": arrivals,
                "admission_ratio": (admitted / arrivals) if arrivals > 0.0 else 0.0,
                "avg_loss": float(np.mean(losses)) if losses.size > 0 else 0.0,
                "total_loss": float(np.sum(losses)) if losses.size > 0 else 0.0,
            }

    x = np.asarray(packet_bits, dtype=int)

    plt.figure(figsize=(7.2, 4.8))
    for policy in policies:
        y = [100.0 * float(by_policy[policy][bits]["admission_ratio"]) for bits in packet_bits]
        plt.plot(x, y, marker="o", linewidth=2.0, label=POLICY_LABELS[policy])
    plt.xlabel("Payload size (bits)")
    plt.ylabel("URLLC admission ratio (%)")
    plt.title(f"Fig. 1  URLLC Admission Ratio (eMBB={args.embb_users}, URLLC={args.urllc_users})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_urllc_admission_ratio.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, len(packet_bits), figsize=(5.2 * len(packet_bits), 4.6), sharey=True)
    if len(packet_bits) == 1:
        axes = [axes]
    for ax, bits in zip(axes, packet_bits):
        for policy in policies:
            losses = np.sort(np.asarray(by_policy[policy][bits]["losses"], dtype=float))
            if losses.size == 0:
                continue
            y = np.arange(1, losses.size + 1, dtype=float) / float(losses.size)
            ax.plot(losses, y, linewidth=2.0, label=POLICY_LABELS[policy])
        ax.set_title(f"k={bits}")
        ax.set_xlabel("eMBB loss per admitted URLLC packet (Mbps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("CDF")
    axes[-1].legend()
    fig.suptitle(f"Fig. 2  CDF of Per-packet eMBB Loss (eMBB={args.embb_users}, URLLC={args.urllc_users})")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_cdf_per_packet_embb_loss.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7.2, 4.8))
    for policy in policies:
        y = [float(by_policy[policy][bits]["avg_loss"]) for bits in packet_bits]
        plt.plot(x, y, marker="o", linewidth=2.0, label=POLICY_LABELS[policy])
    plt.xlabel("Payload size (bits)")
    plt.ylabel("Average eMBB loss per admitted packet (Mbps)")
    plt.title(f"Fig. 3  Average eMBB Loss per Admitted Packet (eMBB={args.embb_users}, URLLC={args.urllc_users})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_avg_embb_loss_per_admitted_packet.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
