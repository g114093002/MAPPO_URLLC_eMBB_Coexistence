"""
Visualization and Analysis Tools
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Times New Roman',
    'Georgia',
    'Microsoft JhengHei',
    'DejaVu Serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False


class SimulationPlotter:
    """Generate plots for simulation results."""

    def __init__(self, output_dir='./results/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _pin_xaxis(ax, x_values):
        """Make the plotted x-range tightly touch the y-axis without extra left/right padding."""
        x_values = np.asarray(x_values, dtype=float)
        if x_values.size == 0:
            return
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        if np.isclose(x_min, x_max):
            pad = max(1.0, abs(x_min) * 0.05)
            ax.set_xlim(x_min - pad, x_max + pad)
        else:
            ax.set_xlim(x_min, x_max)
        ax.margins(x=0)

    def plot_power_vs_density(self, density_analysis_result, save_path=None):
        """Plot stress indicators versus user density."""
        densities = np.asarray(density_analysis_result['densities'])
        power = np.asarray(density_analysis_result['power_consumption'])
        embb_rates = np.asarray(density_analysis_result['embb_rates'])
        embb_user_rates = np.asarray(density_analysis_result.get('embb_user_rates', embb_rates))
        urllc_success = np.asarray(density_analysis_result['urllc_success'])
        urllc_admission = np.asarray(density_analysis_result.get('urllc_admission', np.ones_like(densities)))
        embb_service_ratio = np.asarray(density_analysis_result.get('embb_service_ratio', np.ones_like(densities)))

        fig = plt.figure(figsize=(14, 11))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(densities, power * 1e3, color='#355C7D', marker='o', linewidth=2.5)
        ax1.fill_between(densities, 0, power * 1e3, color='#355C7D', alpha=0.18)
        ax1.set_title('Total Tx Power vs User Density', fontweight='bold')
        ax1.set_xlabel('Average UE Load per UAV')
        ax1.set_ylabel('Total Power (mW)')
        ax1.grid(alpha=0.25)
        self._pin_xaxis(ax1, densities)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(densities, embb_rates / 1e6, color='#6C9A8B', marker='s', linewidth=2.5)
        ax2.fill_between(densities, 0, embb_rates / 1e6, color='#6C9A8B', alpha=0.18)
        ax2.set_title('Aggregate eMBB Throughput', fontweight='bold')
        ax2.set_xlabel('Average UE Load per UAV')
        ax2.set_ylabel('eMBB Rate (Mbps)')
        ax2.grid(alpha=0.25)
        self._pin_xaxis(ax2, densities)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(densities, embb_user_rates / 1e6, color='#4E79A7', marker='D', linewidth=2.4)
        ax3.fill_between(densities, 0, embb_user_rates / 1e6, color='#4E79A7', alpha=0.16)
        ax3.set_title('Per-user eMBB Rate Collapse', fontweight='bold')
        ax3.set_xlabel('Average UE Load per UAV')
        ax3.set_ylabel('Mean User Rate (Mbps)')
        ax3.grid(alpha=0.25)
        self._pin_xaxis(ax3, densities)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(densities, embb_service_ratio, color='#F28E2B', marker='o', linewidth=2.4, label='eMBB served ratio')
        ax4.plot(densities, urllc_admission, color='#B07AA1', marker='^', linewidth=2.4, label='URLLC admission ratio')
        ax4.axhline(1.0, color='#777777', linestyle='--', linewidth=1.0)
        ax4.set_title('Admission / Service Stress', fontweight='bold')
        ax4.set_xlabel('Average UE Load per UAV')
        ax4.set_ylabel('Ratio')
        ax4.set_ylim(0.0, 1.05)
        ax4.grid(alpha=0.25)
        ax4.legend(frameon=False, loc='lower left')
        self._pin_xaxis(ax4, densities)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(densities, urllc_success, color='#C06C84', marker='^', linewidth=2.5)
        ax5.axhline(0.99, color='#7A1F2B', linestyle='--', linewidth=1.5, label='99% target')
        ax5.set_title('Admitted URLLC Reliability', fontweight='bold')
        ax5.set_xlabel('Average UE Load per UAV')
        ax5.set_ylabel('Reliability')
        ax5.set_ylim(0.0, 1.02)
        ax5.grid(alpha=0.25)
        ax5.legend()
        self._pin_xaxis(ax5, densities)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        note = (
            "Stress Interpretation\n\n"
            "1. Aggregate throughput can stay high even when the system is already overloaded.\n"
            "2. Watch the per-user eMBB rate and served ratio: these usually collapse first.\n"
            "3. URLLC admission ratio below 1 means arrivals are starting to exceed what one slot can carry.\n"
            "4. Power saturation with shrinking service ratios is the clearest sign the system is near breakdown."
        )
        ax6.text(0.04, 0.95, note, va='top', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F6E7CB', alpha=0.9))
        summary = (
            f"Aggregate rate range: {np.min(embb_rates)/1e6:.2f} to {np.max(embb_rates)/1e6:.2f} Mbps\n"
            f"Per-user rate range: {np.min(embb_user_rates)/1e6:.2f} to {np.max(embb_user_rates)/1e6:.2f} Mbps\n"
            f"Power range: {np.min(power)*1e3:.2f} to {np.max(power)*1e3:.2f} mW\n"
            f"URLLC admission range: {np.nanmin(urllc_admission):.2%} to {np.nanmax(urllc_admission):.2%}"
        )
        ax6.text(0.04, 0.33, summary, va='top', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#E7F0FD', alpha=0.9))

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'power_vs_density.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_performance_timeline(self, all_embb_rates, all_urllc_success,
                                  all_power, user_density=None, save_path=None,
                                  all_embb_power=None, all_urllc_power=None):
        """Plot slot-level performance timeline."""
        time_slots = np.arange(len(all_embb_rates))
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        density_str = ""
        if user_density:
            density_str = f" ({user_density[0]} eMBB, {user_density[1]} URLLC)"
        fig.suptitle(f'Slot-Level Performance Timeline{density_str}', fontweight='bold')

        axes[0].plot(time_slots, np.asarray(all_embb_rates) / 1e6, color='#4E79A7', marker='o', linewidth=2)
        axes[0].set_ylabel('eMBB Rate (Mbps)')
        axes[0].grid(alpha=0.25)
        axes[0].set_xlim(0, max(len(time_slots) - 1, 0))
        axes[0].margins(x=0)

        axes[1].plot(time_slots, all_urllc_success, color='#E15759', marker='s', linewidth=2)
        axes[1].axhline(0.99, color='#9C2F2F', linestyle='--', linewidth=1.5)
        axes[1].set_ylabel('Admitted URLLC Reliability')
        axes[1].set_ylim(0.0, 1.02)
        axes[1].grid(alpha=0.25)
        axes[1].margins(x=0)

        axes[2].plot(
            time_slots, np.asarray(all_power) * 1e3,
            color='#2E8B57', marker='^', linewidth=2.2, label='Total Tx power'
        )
        if all_embb_power is not None:
            axes[2].plot(
                time_slots, np.asarray(all_embb_power) * 1e3,
                color='#4E79A7', marker='o', linewidth=1.8, label='eMBB Tx power'
            )
        if all_urllc_power is not None:
            axes[2].plot(
                time_slots, np.asarray(all_urllc_power) * 1e3,
                color='#9C6B30', marker='s', linewidth=1.8, label='URLLC Tx power'
            )
        axes[2].set_xlabel('Time Slot')
        axes[2].set_ylabel('Tx Power (mW)')
        axes[2].grid(alpha=0.25)
        axes[2].set_xlim(0, max(len(time_slots) - 1, 0))
        axes[2].margins(x=0)
        axes[2].set_xticks(time_slots)
        axes[2].legend(frameon=False, loc='upper right')

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'performance_timeline.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_mode_tradeoff_analysis(self, density_analysis_result, save_path=None):
        """Plot overlay/puncture usage and their eMBB impact versus density."""
        densities = np.asarray(density_analysis_result['densities'])
        overlay_ratio = np.asarray(density_analysis_result.get('overlay_ratio', np.zeros_like(densities)))
        puncture_ratio = np.asarray(density_analysis_result.get('puncture_ratio', np.zeros_like(densities)))
        overlay_retention = np.asarray(density_analysis_result.get('overlay_retention', np.full_like(densities, np.nan, dtype=float)))
        puncture_loss = np.asarray(density_analysis_result.get('puncture_embb_loss', np.zeros_like(densities)))
        overlay_loss = np.asarray(density_analysis_result.get('overlay_embb_loss', np.zeros_like(densities)))

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        axes[0, 0].plot(densities, overlay_ratio, marker='o', linewidth=2.2, color='#4E79A7', label='Overlay ratio')
        axes[0, 0].plot(densities, puncture_ratio, marker='s', linewidth=2.2, color='#E15759', label='Puncture ratio')
        axes[0, 0].set_title('Mode Selection Ratio vs User Density', fontweight='bold')
        axes[0, 0].set_xlabel('Average UE Load per UAV')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].grid(alpha=0.25)
        axes[0, 0].legend(frameon=False)
        self._pin_xaxis(axes[0, 0], densities)

        axes[0, 1].plot(densities, overlay_retention, marker='D', linewidth=2.2, color='#2E8B57')
        axes[0, 1].axhline(1.0, linestyle='--', linewidth=1.0, color='#777777')
        axes[0, 1].set_title('Average eMBB Retention Under Overlay', fontweight='bold')
        axes[0, 1].set_xlabel('Average UE Load per UAV')
        axes[0, 1].set_ylabel('Retention Fraction')
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[0, 1], densities)

        axes[1, 0].plot(densities, puncture_loss / 1e6, marker='^', linewidth=2.2, color='#9C6B30')
        axes[1, 0].set_title('Average eMBB Loss Per Puncture Action', fontweight='bold')
        axes[1, 0].set_xlabel('Average UE Load per UAV')
        axes[1, 0].set_ylabel('Loss (Mbps)')
        axes[1, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 0], densities)

        axes[1, 1].plot(densities, overlay_loss / 1e6, marker='o', linewidth=2.2, color='#B07AA1')
        axes[1, 1].set_title('Average eMBB Loss Per Overlay Action', fontweight='bold')
        axes[1, 1].set_xlabel('Average UE Load per UAV')
        axes[1, 1].set_ylabel('Loss (Mbps)')
        axes[1, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 1], densities)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'mode_tradeoff_analysis.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_fairness_load_analysis(self, density_analysis_result, save_path=None):
        """Plot fairness, cell-edge service, and UAV load imbalance versus density."""
        densities = np.asarray(density_analysis_result['densities'])
        jain_fairness = np.asarray(density_analysis_result.get('jain_fairness', np.full_like(densities, np.nan, dtype=float)))
        cell_edge_served = np.asarray(density_analysis_result.get('cell_edge_served_ratio', np.full_like(densities, np.nan, dtype=float)))
        uav_load_std = np.asarray(density_analysis_result.get('per_uav_total_load_std', np.zeros_like(densities)))
        uav_urllc_sched_std = np.asarray(density_analysis_result.get('per_uav_urllc_sched_std', np.zeros_like(densities)))

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        axes[0, 0].plot(densities, jain_fairness, marker='o', linewidth=2.2, color='#4E79A7')
        axes[0, 0].set_title("Jain's Fairness Index vs User Density", fontweight='bold')
        axes[0, 0].set_xlabel('Average UE Load per UAV')
        axes[0, 0].set_ylabel('Fairness')
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[0, 0], densities)

        axes[0, 1].plot(densities, cell_edge_served, marker='D', linewidth=2.2, color='#F28E2B')
        axes[0, 1].set_title('Cell-edge eMBB Served Ratio', fontweight='bold')
        axes[0, 1].set_xlabel('Average UE Load per UAV')
        axes[0, 1].set_ylabel('Served Ratio')
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[0, 1], densities)

        axes[1, 0].plot(densities, uav_load_std, marker='s', linewidth=2.2, color='#59A14F')
        axes[1, 0].set_title('Per-UAV Associated Load Imbalance', fontweight='bold')
        axes[1, 0].set_xlabel('Average UE Load per UAV')
        axes[1, 0].set_ylabel('Std. Dev. of Associated Users')
        axes[1, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 0], densities)

        axes[1, 1].plot(densities, uav_urllc_sched_std, marker='^', linewidth=2.2, color='#B07AA1')
        axes[1, 1].set_title('Per-UAV Scheduled URLLC Imbalance', fontweight='bold')
        axes[1, 1].set_xlabel('Average UE Load per UAV')
        axes[1, 1].set_ylabel('Std. Dev. of Scheduled URLLC')
        axes[1, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 1], densities)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'fairness_load_analysis.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_slot_mode_action_summary(self, aggregated_metrics, save_path=None):
        """Plot slot-level arrivals, admissions, coexistence actions, and eMBB throughput."""
        arrivals = np.asarray(aggregated_metrics.get('all_active_urllc_users', []), dtype=float)
        admitted = np.asarray(aggregated_metrics.get('all_scheduled_urllc_users', []), dtype=float)
        overlay = np.asarray(aggregated_metrics.get('all_overlay_counts', []), dtype=float)
        puncture = np.asarray(aggregated_metrics.get('all_puncture_counts', []), dtype=float)
        embb_rate = np.asarray(aggregated_metrics.get('all_embb_rates', []), dtype=float) / 1e6
        slots = np.arange(len(arrivals))

        fig, ax1 = plt.subplots(figsize=(13.5, 5.6))
        width = 0.18
        ax1.bar(slots - 1.5 * width, arrivals, width=width, color='#C97C5D', alpha=0.8, label='URLLC arrivals')
        ax1.bar(slots - 0.5 * width, admitted, width=width, color='#5B8E7D', alpha=0.85, label='Admitted URLLC')
        ax1.bar(slots + 0.5 * width, overlay, width=width, color='#4E79A7', alpha=0.85, label='Overlay count')
        ax1.bar(slots + 1.5 * width, puncture, width=width, color='#9C6B30', alpha=0.85, label='Puncture count')
        ax1.set_xlabel('Time Slot')
        ax1.set_ylabel('Packet / Action Count')
        ax1.set_xticks(slots)
        ax1.grid(axis='y', alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(slots, embb_rate, color='#7A1F2B', marker='o', linewidth=2.4, label='Slot eMBB throughput')
        ax2.set_ylabel('eMBB Throughput (Mbps)')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc='upper right', ncol=3)
        ax1.set_title('Per-Slot Mode / Action Summary', fontweight='bold')

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'slot_mode_action_summary.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_per_uav_performance_decomposition(self, density_analysis_result, save_path=None):
        """Plot representative per-UAV decomposition for low/mid/high load points."""
        reps = density_analysis_result.get('representative_per_uav', [])
        if not reps:
            return

        selected_idx = sorted(set([0, len(reps) // 2, len(reps) - 1]))
        selected = [reps[idx] for idx in selected_idx]
        fig, axes = plt.subplots(len(selected), 2, figsize=(14, 4.8 * len(selected)))
        if len(selected) == 1:
            axes = np.array([axes])

        for row, rep in enumerate(selected):
            num_uavs = len(rep.get('associated_embb', []))
            x = np.arange(num_uavs)

            ax_left = axes[row, 0]
            assoc_embb = np.asarray(rep.get('associated_embb', []), dtype=float)
            assoc_urllc = np.asarray(rep.get('associated_urllc', []), dtype=float)
            sched_embb = np.asarray(rep.get('scheduled_embb', []), dtype=float)
            sched_urllc = np.asarray(rep.get('scheduled_urllc', []), dtype=float)
            width = 0.34
            ax_left.bar(x - width / 2, assoc_embb, width=width, color='#8FB9E3', label='Assoc eMBB')
            ax_left.bar(x - width / 2, assoc_urllc, width=width, bottom=assoc_embb, color='#E8A5A5', label='Assoc URLLC')
            ax_left.bar(x + width / 2, sched_embb, width=width, color='#4E79A7', label='Sched eMBB')
            ax_left.bar(x + width / 2, sched_urllc, width=width, bottom=sched_embb, color='#9C6B30', label='Admitted URLLC')
            ax_left.set_title(
                f"Density {rep['density']:.2f} users/UAV\nOffered load {rep['offered_load']:.2f} packets/slot",
                fontweight='bold'
            )
            ax_left.set_ylabel('User / Packet Count')
            ax_left.set_xticks(x)
            ax_left.set_xticklabels([f'UAV {i + 1}' for i in x])
            ax_left.grid(axis='y', alpha=0.25)
            if row == 0:
                ax_left.legend(frameon=False, ncol=2, fontsize=9)

            ax_right = axes[row, 1]
            overlay = np.asarray(rep.get('overlay_count', []), dtype=float)
            puncture = np.asarray(rep.get('puncture_count', []), dtype=float)
            throughput = np.asarray(rep.get('embb_throughput', []), dtype=float) / 1e6
            avg_dist = np.asarray(rep.get('avg_embb_distance', []), dtype=float)
            ax_right.bar(x - 0.25, overlay, width=0.24, color='#4E79A7', label='Overlay')
            ax_right.bar(x, puncture, width=0.24, color='#9C6B30', label='Puncture')
            ax_right.bar(x + 0.25, throughput, width=0.24, color='#59A14F', label='eMBB throughput (Mbps)')
            ax_right.set_ylabel('Action Count / Throughput')
            ax_right.set_xticks(x)
            ax_right.set_xticklabels([f'UAV {i + 1}' for i in x])
            ax_right.grid(axis='y', alpha=0.25)
            ax_dist = ax_right.twinx()
            ax_dist.plot(x, avg_dist, color='#7A1F2B', marker='D', linewidth=2.0, label='Avg eMBB distance')
            ax_dist.set_ylabel('Avg Distance (m)')
            if row == 0:
                h1, l1 = ax_right.get_legend_handles_labels()
                h2, l2 = ax_dist.get_legend_handles_labels()
                ax_right.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc='upper right')

        fig.suptitle('Per-UAV Performance Decomposition', fontweight='bold', y=0.995)
        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'per_uav_performance_decomposition.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_overlay_feasibility_diagnostic(self, density_analysis_result, save_path=None):
        """Plot overlay candidate, feasible, and selected counts versus density."""
        densities = np.asarray(density_analysis_result['densities'], dtype=float)
        candidate = np.asarray(density_analysis_result.get('overlay_candidate_pairs', np.zeros_like(densities)), dtype=float)
        feasible = np.asarray(density_analysis_result.get('overlay_feasible_pairs', np.zeros_like(densities)), dtype=float)
        selected = np.asarray(density_analysis_result.get('overlay_selected_pairs', np.zeros_like(densities)), dtype=float)

        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        ax.plot(densities, candidate, marker='o', linewidth=2.3, color='#B07AA1', label='Candidate overlay pairs')
        ax.plot(densities, feasible, marker='s', linewidth=2.3, color='#59A14F', label='Feasible overlay pairs')
        ax.plot(densities, selected, marker='D', linewidth=2.3, color='#4E79A7', label='Selected overlay pairs')
        ax.set_title('Overlay Feasibility / Selection Diagnostic', fontweight='bold')
        ax.set_xlabel('Average UE Load per UAV')
        ax.set_ylabel('Average Count Per Slot')
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        self._pin_xaxis(ax, densities)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'overlay_feasibility_diagnostic.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_retention_loss_distribution(self, density_analysis_result, save_path=None):
        """Plot retention/loss distributions for representative densities."""
        reps = density_analysis_result.get('representative_per_uav', [])
        if not reps:
            return
        selected_idx = sorted(set([0, len(reps) // 2, len(reps) - 1]))
        selected_labels = [f"{density_analysis_result['densities'][idx]:.2f}" for idx in selected_idx]
        overlay_retention = density_analysis_result.get('overlay_retention_distribution', [])
        puncture_loss = density_analysis_result.get('puncture_loss_distribution', [])

        retention_data = []
        puncture_data = []
        for idx in selected_idx:
            retention_values = np.asarray(overlay_retention[idx], dtype=float) if idx < len(overlay_retention) else np.asarray([])
            puncture_values = np.asarray(puncture_loss[idx], dtype=float) / 1e6 if idx < len(puncture_loss) else np.asarray([])
            retention_data.append(retention_values if retention_values.size > 0 else np.asarray([np.nan]))
            puncture_data.append(puncture_values if puncture_values.size > 0 else np.asarray([np.nan]))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        axes[0].boxplot(retention_data, tick_labels=selected_labels, patch_artist=True,
                        boxprops=dict(facecolor='#BFD7EA', alpha=0.8),
                        medianprops=dict(color='#1F4E79', linewidth=2))
        axes[0].set_title('eMBB Retention Under Overlay', fontweight='bold')
        axes[0].set_xlabel('Average UE Load per UAV')
        axes[0].set_ylabel('Retention Fraction')
        axes[0].grid(axis='y', alpha=0.25)

        axes[1].boxplot(puncture_data, tick_labels=selected_labels, patch_artist=True,
                        boxprops=dict(facecolor='#E8C39E', alpha=0.85),
                        medianprops=dict(color='#7A3E00', linewidth=2))
        axes[1].set_title('eMBB Loss Under Puncturing', fontweight='bold')
        axes[1].set_xlabel('Average UE Load per UAV')
        axes[1].set_ylabel('Loss (Mbps)')
        axes[1].grid(axis='y', alpha=0.25)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'retention_loss_distribution.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_offered_load_curves(self, density_analysis_result, save_path=None):
        """Plot key curves against actual offered URLLC load per slot."""
        offered_load = np.asarray(density_analysis_result.get('offered_load_scale', []), dtype=float)
        admission = np.asarray(density_analysis_result.get('urllc_admission', []), dtype=float)
        overlay = np.asarray(density_analysis_result.get('overlay_ratio', []), dtype=float)
        puncture = np.asarray(density_analysis_result.get('puncture_ratio', []), dtype=float)
        embb_rate = np.asarray(density_analysis_result.get('embb_rates', []), dtype=float) / 1e6
        fairness = np.asarray(density_analysis_result.get('jain_fairness', []), dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0, 0].plot(offered_load, admission, marker='o', linewidth=2.2, color='#5B8E7D')
        axes[0, 0].set_title('URLLC Admission vs Offered Load', fontweight='bold')
        axes[0, 0].set_xlabel('Average URLLC Offered Load (packets/slot)')
        axes[0, 0].set_ylabel('Admission Ratio')
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[0, 0], offered_load)

        axes[0, 1].plot(offered_load, overlay, marker='D', linewidth=2.2, color='#4E79A7', label='Overlay')
        axes[0, 1].plot(offered_load, puncture, marker='s', linewidth=2.2, color='#9C6B30', label='Puncture')
        axes[0, 1].set_title('Mode Ratio vs Offered Load', fontweight='bold')
        axes[0, 1].set_xlabel('Average URLLC Offered Load (packets/slot)')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].grid(alpha=0.25)
        axes[0, 1].legend(frameon=False)
        self._pin_xaxis(axes[0, 1], offered_load)

        axes[1, 0].plot(offered_load, embb_rate, marker='^', linewidth=2.2, color='#59A14F')
        axes[1, 0].set_title('eMBB Throughput vs Offered Load', fontweight='bold')
        axes[1, 0].set_xlabel('Average URLLC Offered Load (packets/slot)')
        axes[1, 0].set_ylabel('Aggregate eMBB Throughput (Mbps)')
        axes[1, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 0], offered_load)

        axes[1, 1].plot(offered_load, fairness, marker='o', linewidth=2.2, color='#B07AA1')
        axes[1, 1].set_title("Fairness vs Offered Load", fontweight='bold')
        axes[1, 1].set_xlabel('Average URLLC Offered Load (packets/slot)')
        axes[1, 1].set_ylabel("Jain's Fairness Index")
        axes[1, 1].set_ylim(0.0, 1.05)
        axes[1, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 1], offered_load)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'offered_load_curves.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_resource_utilization_summary(self, density_analysis_result, save_path=None):
        """Plot resource-utilization fractions versus user density."""
        densities = np.asarray(density_analysis_result['densities'], dtype=float)
        embb_only = np.asarray(density_analysis_result.get('embb_only_fraction', []), dtype=float)
        overlay = np.asarray(density_analysis_result.get('overlay_fraction', []), dtype=float)
        puncture = np.asarray(density_analysis_result.get('puncture_fraction', []), dtype=float)
        idle = np.asarray(density_analysis_result.get('idle_fraction', []), dtype=float)
        minislot_util = np.asarray(density_analysis_result.get('minislot_utilization', []), dtype=float)
        rb_util = np.asarray(density_analysis_result.get('rb_utilization', []), dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0, 0].stackplot(
            densities, embb_only, overlay, puncture, idle,
            labels=['eMBB only', 'Overlay', 'Puncture', 'Idle'],
            colors=['#BFD7EA', '#4E79A7', '#9C6B30', '#E6E6E6'],
            alpha=0.9
        )
        axes[0, 0].set_title('Resource Cell Composition', fontweight='bold')
        axes[0, 0].set_xlabel('Average UE Load per UAV')
        axes[0, 0].set_ylabel('Fraction of UAV-RB-minislot Cells')
        axes[0, 0].set_ylim(0.0, 1.0)
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].legend(frameon=False, loc='upper right')
        self._pin_xaxis(axes[0, 0], densities)

        axes[0, 1].plot(densities, minislot_util, marker='o', linewidth=2.2, color='#5B8E7D')
        axes[0, 1].set_title('Mini-slot Utilization', fontweight='bold')
        axes[0, 1].set_xlabel('Average UE Load per UAV')
        axes[0, 1].set_ylabel('Utilization Ratio')
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].grid(alpha=0.25)
        self._pin_xaxis(axes[0, 1], densities)

        axes[1, 0].plot(densities, 1.0 - idle, marker='D', linewidth=2.2, color='#4E79A7')
        axes[1, 0].set_title('Non-idle Resource Fraction', fontweight='bold')
        axes[1, 0].set_xlabel('Average UE Load per UAV')
        axes[1, 0].set_ylabel('Fraction')
        axes[1, 0].set_ylim(0.0, 1.05)
        axes[1, 0].grid(alpha=0.25)
        self._pin_xaxis(axes[1, 0], densities)

        axes[1, 1].plot(densities, idle, marker='s', linewidth=2.2, color='#7F7F7F', label='Idle fraction')
        axes[1, 1].plot(densities, rb_util, marker='^', linewidth=2.2, color='#F28E2B', label='RB utilization')
        axes[1, 1].set_title('Idle Resources and RB Utilization', fontweight='bold')
        axes[1, 1].set_xlabel('Average UE Load per UAV')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].set_ylim(0.0, 1.05)
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(frameon=False)
        self._pin_xaxis(axes[1, 1], densities)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'resource_utilization_summary.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_urllc_arrival_timeline(self, active_urllc_users, scheduled_urllc_users=None,
                                    slot_index_start=0, save_path=None):
        """Plot URLLC arrivals and scheduled packets over time slots."""
        active_urllc_users = np.asarray(active_urllc_users, dtype=float)
        time_slots = np.arange(slot_index_start, slot_index_start + len(active_urllc_users))
        fig, ax = plt.subplots(figsize=(12, 4.8))

        ax.bar(time_slots, active_urllc_users, color='#C97C5D', alpha=0.72, width=0.75, label='URLLC packet arrivals')
        if scheduled_urllc_users is not None:
            scheduled_urllc_users = np.asarray(scheduled_urllc_users, dtype=float)
            ax.plot(
                time_slots, scheduled_urllc_users,
                color='#5B8E7D', marker='o', linewidth=2.2, label='URLLC packets scheduled'
            )

        ax.set_title('URLLC Packet Arrivals Per Time Slot', fontweight='bold')
        ax.set_xlabel('Time Slot')
        ax.set_ylabel('Number of URLLC Packets')
        ax.set_xticks(time_slots)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(frameon=False)

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'urllc_arrival_timeline.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_urllc_minislot_arrival_map(self, allocation_history, num_minislots, save_path=None):
        """Plot scheduled URLLC packet appearances for each slot and minislot."""
        num_slots = len(allocation_history)
        minislot_counts = np.zeros((num_slots, num_minislots), dtype=int)

        for slot_idx, allocation in enumerate(allocation_history):
            coexistence_urllc = allocation.get('coexistence_urllc_user_per_uav')
            if coexistence_urllc is None:
                continue
            for minislot in range(min(num_minislots, coexistence_urllc.shape[2])):
                minislot_counts[slot_idx, minislot] = int(np.count_nonzero(coexistence_urllc[:, :, minislot] >= 0))

        fig, ax = plt.subplots(figsize=(12, 5.2))
        im = ax.imshow(minislot_counts, aspect='auto', cmap='YlOrBr', interpolation='nearest')
        ax.set_title('URLLC Scheduled Packet Activity by Mini-slot', fontweight='bold')
        ax.set_xlabel('Mini-slot')
        ax.set_ylabel('Time Slot')
        ax.set_xticks(np.arange(num_minislots))
        ax.set_xticklabels([str(i + 1) for i in range(num_minislots)])
        ax.set_yticks(np.arange(num_slots))
        ax.set_yticklabels([str(i) for i in range(num_slots)])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Scheduled URLLC packet count')

        for slot_idx in range(num_slots):
            for minislot in range(num_minislots):
                ax.text(
                    minislot, slot_idx, str(minislot_counts[slot_idx, minislot]),
                    ha='center', va='center', fontsize=8,
                    color='black' if minislot_counts[slot_idx, minislot] < np.max(minislot_counts) * 0.55 else 'white'
                )

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'urllc_minislot_arrival_map.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_per_uav_load_distribution(self, allocation_summary, num_uavs, num_urllc, num_embb,
                                       slot_index=None, save_path=None):
        """Plot per-UAV associated and scheduled load as stacked bars."""
        embb_selected = np.asarray(allocation_summary.get('embb_selected_uavs', []), dtype=int)
        urllc_selected = np.asarray(allocation_summary.get('urllc_selected_uavs', []), dtype=int)
        best_uav_per_user = np.asarray(allocation_summary.get('best_uav_per_user', []), dtype=int)

        embb_assoc = np.zeros(num_uavs, dtype=int)
        urllc_assoc = np.zeros(num_uavs, dtype=int)
        urllc_sched = np.zeros(num_uavs, dtype=int)

        if best_uav_per_user.size >= num_urllc + num_embb:
            urllc_assoc = np.bincount(best_uav_per_user[:num_urllc], minlength=num_uavs)
            embb_assoc = np.bincount(best_uav_per_user[num_urllc:num_urllc + num_embb], minlength=num_uavs)
        elif embb_selected.size > 0:
            embb_assoc = np.bincount(embb_selected, minlength=num_uavs)

        valid_urllc_sched = urllc_selected[urllc_selected >= 0] if urllc_selected.size > 0 else np.asarray([], dtype=int)
        if valid_urllc_sched.size > 0:
            urllc_sched = np.bincount(valid_urllc_sched, minlength=num_uavs)

        x = np.arange(num_uavs)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        axes[0].bar(x, embb_assoc, color='#8FB9E3', label='Associated eMBB')
        axes[0].bar(x, urllc_assoc, bottom=embb_assoc, color='#E8A5A5', label='Associated URLLC')
        axes[0].set_title('Per-UAV Associated User Load', fontweight='bold')
        axes[0].set_xlabel('UAV')
        axes[0].set_ylabel('User Count')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'UAV {idx + 1}' for idx in x])
        axes[0].grid(axis='y', alpha=0.25)
        axes[0].legend(frameon=False)

        axes[1].bar(x, urllc_sched, color='#8B4513', alpha=0.82, label='Scheduled URLLC packets')
        axes[1].set_title('Per-UAV Scheduled URLLC Packet Load', fontweight='bold')
        axes[1].set_xlabel('UAV')
        axes[1].set_ylabel('Scheduled Packet Count')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'UAV {idx + 1}' for idx in x])
        axes[1].grid(axis='y', alpha=0.25)
        axes[1].legend(frameon=False)

        title = 'Per-UAV Load Distribution'
        if slot_index is not None:
            title += f' (Slot {slot_index})'
        fig.suptitle(title, fontweight='bold')

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, f'per_uav_load_distribution_slot{slot_index if slot_index is not None else 0}.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_single_slot_heatmap(self, embb_rbs, urllc_rbs, num_urllc, num_embb,
                                 slot_index=0, save_path=None):
        """Plot a users-by-RB occupancy heatmap for a single slot."""
        num_rbs = embb_rbs.shape[1]
        total_users = num_urllc + num_embb
        matrix = np.zeros((total_users, num_rbs), dtype=int)

        if urllc_rbs is not None:
            for u in range(min(urllc_rbs.shape[0], num_urllc)):
                matrix[u, np.where(urllc_rbs[u, :] == 1)[0]] = u + 1

        for e in range(min(embb_rbs.shape[0], num_embb)):
            matrix[num_urllc + e, np.where(embb_rbs[e, :] == 1)[0]] = num_urllc + e + 1

        colors = ['#FFFFFF']
        colors.extend(plt.get_cmap('Oranges')(np.linspace(0.45, 0.85, max(num_urllc, 1))))
        colors.extend(plt.get_cmap('Pastel1')(np.linspace(0.1, 0.9, max(num_embb, 1))))
        cmap = ListedColormap(colors[: np.max(matrix) + 1 if np.max(matrix) > 0 else 1])

        fig, ax = plt.subplots(figsize=(12, max(4, total_users * 0.35)))
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, origin='lower')
        ax.set_title(f'Single-Slot User/RB Occupancy (Slot {slot_index})', fontweight='bold')
        ax.set_xlabel('Resource Block')
        ax.set_ylabel('Users')
        ax.set_yticks(np.arange(total_users))
        ax.set_yticklabels(
            [f'URLLC {u + 1}' for u in range(num_urllc)] +
            [f'eMBB {e + 1}' for e in range(num_embb)]
        )
        plt.colorbar(im, ax=ax, label='User ID')

        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, f'single_slot_heatmap_slot{slot_index}.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_slot_timefreq_heatmap(self, embb_rbs, urllc_timefreq_grid, num_urllc, num_embb,
                                   num_minislots=7, slot_index=0, save_path=None,
                                   embb_owner_per_rb=None, noma_decisions=None,
                                   embb_owner_per_uav_rb=None, coexistence_mode_per_uav=None,
                                   coexistence_urllc_user_per_uav=None, plot_uav_index=0,
                                   embb_selected_uavs=None, urllc_selected_uavs=None):
        """
        Plot a slot time-frequency grid in the style of eMBB background bands
        with URLLC minislot puncturing overlays.
        """
        num_rbs = embb_rbs.shape[1]
        num_embb_display = max(num_embb, 1)
        embb_palette = [
            '#BFD7EA', '#F7C6C7', '#D5ECC2', '#F3D1A7', '#D9C2F0',
            '#B8E0D2', '#F4B8A8', '#C7D3F5', '#E3F2B3', '#F6CCE0'
        ]
        urllc_color = '#8B4513'

        def draw_uav_panel(ax, uav_index):
            if embb_owner_per_uav_rb is not None:
                band_owner = np.asarray(embb_owner_per_uav_rb[uav_index], dtype=int).copy()
            else:
                if embb_owner_per_rb is None:
                    local_owner = np.argmax(embb_rbs, axis=0)
                else:
                    local_owner = embb_owner_per_rb
                band_owner = np.asarray(local_owner, dtype=int).copy()

            band_owner[band_owner < 0] = 0
            if band_owner.size < num_rbs:
                pad = np.full(num_rbs - band_owner.size, band_owner[-1] if band_owner.size > 0 else 0, dtype=int)
                band_owner = np.concatenate([band_owner, pad])
            band_owner = np.mod(band_owner[:num_rbs], num_embb_display)

            background = np.zeros((num_rbs, num_minislots, 3))
            for rb in range(num_rbs):
                rgb = matplotlib.colors.to_rgb(embb_palette[band_owner[rb] % len(embb_palette)])
                background[rb, :, :] = rgb
            ax.imshow(background, aspect='auto', origin='lower', interpolation='none')

            for x in np.arange(-0.5, num_minislots, 1):
                ax.axvline(x, color='white', linewidth=0.8, alpha=0.9)
            for y in np.arange(-0.5, num_rbs, 1):
                ax.axhline(y, color='white', linewidth=0.8, alpha=0.9)

            if coexistence_mode_per_uav is not None and coexistence_urllc_user_per_uav is not None:
                local_mode = coexistence_mode_per_uav[uav_index]
                local_urllc = coexistence_urllc_user_per_uav[uav_index]
                for rb in range(min(num_rbs, local_mode.shape[0])):
                    for minislot in range(min(num_minislots, local_mode.shape[1])):
                        decision = local_mode[rb, minislot]
                        urllc_id = local_urllc[rb, minislot]
                        if urllc_id >= 0 and decision != 'EMPTY':
                            if decision == 'NOMA':
                                overlay = plt.Rectangle(
                                    (minislot - 0.32, rb - 0.32), 0.64, 0.64,
                                    facecolor=urllc_color, edgecolor='#F8E7D0',
                                    linewidth=1.1, alpha=0.88, hatch='////'
                                )
                                ax.add_patch(overlay)
                            else:
                                puncture = plt.Rectangle(
                                    (minislot - 0.48, rb - 0.48), 0.96, 0.96,
                                    facecolor=urllc_color, edgecolor='#F8E7D0',
                                    linewidth=1.0, alpha=0.92, hatch='....'
                                )
                                ax.add_patch(puncture)

            embb_served = 0
            if embb_selected_uavs is not None:
                embb_served = int(np.count_nonzero(np.asarray(embb_selected_uavs) == uav_index))
            urllc_served = 0
            if urllc_selected_uavs is not None:
                urllc_served = int(np.count_nonzero(np.asarray(urllc_selected_uavs) == uav_index))

            ax.set_title(
                f'UAV {uav_index + 1}\neMBB served: {embb_served}, URLLC packets: {urllc_served}',
                fontweight='bold',
                fontsize=11,
                pad=8
            )
            ax.set_xlabel('Mini-slot index')
            ax.set_xticks(np.arange(num_minislots))
            ax.set_xticklabels([str(i + 1) for i in range(num_minislots)], fontsize=9)
            ax.set_yticks(np.arange(num_rbs))
            ax.set_yticklabels([str(i + 1) for i in range(num_rbs)], fontsize=9)
            ax.set_xlim(-0.5, num_minislots - 0.5)
            ax.set_ylim(-0.5, num_rbs - 0.5)

        if embb_owner_per_uav_rb is not None and coexistence_mode_per_uav is not None:
            num_uavs = embb_owner_per_uav_rb.shape[0]
            fig, axes = plt.subplots(1, num_uavs, figsize=(5.2 * num_uavs, 6.2), sharey=True)
            if num_uavs == 1:
                axes = [axes]
            for uav_index, ax in enumerate(axes):
                draw_uav_panel(ax, uav_index)
                if uav_index == 0:
                    ax.set_ylabel('Frequency')
            fig.suptitle(f'Slot Time-Frequency Allocation Across All UAVs (Slot {slot_index})', fontweight='bold', y=0.98)
        else:
            fig, ax = plt.subplots(figsize=(11.5, 6.2))
            axes = [ax]
            draw_uav_panel(ax, plot_uav_index)
            ax.set_ylabel('Frequency')
            fig.suptitle(f'Slot Time-Frequency Allocation (Slot {slot_index})', fontweight='bold', y=0.97)

        legend_handles = []
        for idx in range(min(num_embb_display, 10)):
            legend_handles.append(
                Patch(facecolor=embb_palette[idx % len(embb_palette)], edgecolor='none', label=f'eMBB user {idx + 1}')
            )
        legend_handles.append(Patch(facecolor=urllc_color, edgecolor='#F8E7D0', hatch='....', label='URLLC puncture'))
        has_noma = False
        if coexistence_mode_per_uav is not None:
            has_noma = bool(np.any(coexistence_mode_per_uav == 'NOMA'))
        elif noma_decisions is not None:
            has_noma = bool(np.any(noma_decisions == 'NOMA'))
        if has_noma:
            legend_handles.append(Patch(facecolor=urllc_color, edgecolor='#F8E7D0', hatch='////', label='NOMA/RSMA overlay'))
        fig.legend(handles=legend_handles, frameon=False, loc='center left', bbox_to_anchor=(0.995, 0.5), fontsize=10)

        plt.subplots_adjust(left=0.06, right=0.86, bottom=0.12, top=0.84, wspace=0.15)
        save_path = save_path or os.path.join(self.output_dir, f'slot_timefreq_slot{slot_index}.png')
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_spatial_grouping(self, user_positions, uav_positions, serving_uavs,
                              num_urllc=0, slot_index=None, save_path=None):
        """Plot 2D UAV-user grouping in an X-Y plane."""
        user_positions = np.asarray(user_positions, dtype=float)
        uav_positions = np.asarray(uav_positions, dtype=float)
        serving_uavs = np.asarray(serving_uavs, dtype=int)

        fig, ax = plt.subplots(figsize=(9.2, 7.2))
        cluster_colors = list(plt.get_cmap('tab10').colors)

        num_embb = user_positions.shape[0] - num_urllc
        embb_positions = user_positions[:num_embb]
        embb_serving = serving_uavs[:num_embb]
        urllc_positions = user_positions[num_embb:]
        urllc_serving = serving_uavs[num_embb:]

        embb_legend_added = set()
        urllc_legend_added = set()

        for uav_idx in range(len(uav_positions)):
            embb_mask = embb_serving == uav_idx
            if np.any(embb_mask):
                ax.scatter(
                    embb_positions[embb_mask, 0],
                    embb_positions[embb_mask, 1],
                    s=42,
                    color=cluster_colors[uav_idx % len(cluster_colors)],
                    edgecolors='white',
                    linewidths=0.8,
                    marker='o',
                    alpha=0.95,
                    label=f'eMBB cluster {uav_idx + 1}' if uav_idx not in embb_legend_added else None
                )
                embb_legend_added.add(uav_idx)

        if num_urllc > 0 and urllc_positions.size > 0:
            for uav_idx in range(len(uav_positions)):
                urllc_mask = urllc_serving == uav_idx
                if np.any(urllc_mask):
                    ax.scatter(
                        urllc_positions[urllc_mask, 0],
                        urllc_positions[urllc_mask, 1],
                        s=78,
                        marker='D',
                        color=cluster_colors[uav_idx % len(cluster_colors)],
                        edgecolors='black',
                        linewidths=1.0,
                        alpha=0.95,
                        label=f'URLLC cluster {uav_idx + 1}' if uav_idx not in urllc_legend_added else None
                    )
                    urllc_legend_added.add(uav_idx)

        ax.scatter(
            uav_positions[:, 0],
            uav_positions[:, 1],
            s=85,
            marker='s',
            facecolors='white',
            edgecolors='black',
            linewidths=2.0,
            label='UAV location',
            zorder=5
        )

        for idx, (x_coord, y_coord) in enumerate(uav_positions):
            ax.text(x_coord + 6, y_coord + 6, f'UAV {idx + 1}', fontsize=10, fontweight='bold')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        title = 'UAV-User Spatial Association'
        if slot_index is not None:
            title += f' (Slot {slot_index})'
        ax.set_title(title, fontweight='bold')
        ax.text(
            0.02, 0.98,
            'Circle = eMBB user\nDiamond = URLLC user\nThis figure shows association, not per-slot scheduling.',
            transform=ax.transAxes,
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#777777')
        )
        ax.grid(alpha=0.25)
        ax.set_xlim(0, max(400, np.max(user_positions[:, 0]) + 30))
        ax.set_ylim(0, max(400, np.max(user_positions[:, 1]) + 30))

        handles, labels = ax.get_legend_handles_labels()
        dedup_handles = []
        dedup_labels = []
        for handle, label in zip(handles, labels):
            if label not in dedup_labels:
                dedup_handles.append(handle)
                dedup_labels.append(label)
        ax.legend(dedup_handles, dedup_labels, frameon=True, facecolor='white',
                  edgecolor='#444444', loc='lower right', fontsize=11)

        plt.tight_layout()
        if save_path is None:
            if slot_index is None:
                save_path = os.path.join(self.output_dir, 'default_spatial_grouping.png')
            else:
                save_path = os.path.join(self.output_dir, f'spatial_grouping_slot{slot_index}.png')
        plt.savefig(save_path, dpi=220, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()


    def plot_greedy_vs_mappo_overview(self, greedy_result, mappo_result, mappo_label='SR-MAPPO', greedy_label='Greedy', save_path=None):
        """Plot a direct Greedy vs SR-MAPPO comparison on the main system-level metrics."""
        densities = np.asarray(greedy_result.get('densities', mappo_result.get('densities', [])), dtype=float)
        if densities.size == 0:
            raise ValueError('Comparison plotting requires density points.')

        def _series(result, key, default=np.nan):
            values = np.asarray(result.get(key, []), dtype=float)
            if values.size == 0:
                return np.full_like(densities, default, dtype=float)
            if values.size == densities.size:
                return values
            source_x = np.asarray(result.get('densities', np.arange(values.size)), dtype=float)
            if source_x.size == values.size and values.size > 1:
                return np.interp(densities, source_x, values)
            if values.size == 1:
                return np.full_like(densities, float(values.item()), dtype=float)
            padded = np.full_like(densities, default, dtype=float)
            padded[:min(values.size, padded.size)] = values[:min(values.size, padded.size)]
            return padded

        greedy_color = '#355C7D'
        mappo_color = '#C06C84'

        panels = [
            ('embb_rates', 'Aggregate eMBB Throughput', 'Rate (Mbps)', 1e-6),
            ('embb_user_rates', 'Per-user eMBB Rate', 'Rate (Mbps)', 1e-6),
            ('embb_service_ratio', 'eMBB Served Ratio', 'Ratio', 1.0),
            ('urllc_admission', 'URLLC Admission Ratio', 'Ratio', 1.0),
            ('urllc_success', 'Admitted URLLC Reliability', 'Reliability', 1.0),
            ('power_consumption', 'Total Tx Power', 'Power (mW)', 1e3),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        axes = axes.flatten()
        for ax, (key, title, ylabel, scale) in zip(axes, panels):
            greedy_values = _series(greedy_result, key) * scale
            mappo_values = _series(mappo_result, key) * scale
            ax.plot(densities, greedy_values, color=greedy_color, marker='o', linewidth=2.4, label=greedy_label)
            ax.plot(densities, mappo_values, color=mappo_color, marker='s', linewidth=2.4, label=mappo_label)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Average UE Load per UAV')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if 'Ratio' in ylabel or 'Reliability' in ylabel:
                ax.set_ylim(0.0, 1.05)
            self._pin_xaxis(ax, densities)

        axes[0].legend(frameon=False, loc='best')
        fig.suptitle(f'{greedy_label} vs {mappo_label}: System-Level Comparison', fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'greedy_vs_sr_mappo_overview.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()

    def plot_greedy_vs_mappo_mode_comparison(self, greedy_result, mappo_result, mappo_label='SR-MAPPO', greedy_label='Greedy', save_path=None):
        """Plot a direct Greedy vs SR-MAPPO comparison on coexistence-mode and efficiency metrics."""
        densities = np.asarray(greedy_result.get('densities', mappo_result.get('densities', [])), dtype=float)
        if densities.size == 0:
            raise ValueError('Comparison plotting requires density points.')

        def _series(result, key, default=np.nan):
            values = np.asarray(result.get(key, []), dtype=float)
            if values.size == 0:
                return np.full_like(densities, default, dtype=float)
            if values.size == densities.size:
                return values
            source_x = np.asarray(result.get('densities', np.arange(values.size)), dtype=float)
            if source_x.size == values.size and values.size > 1:
                return np.interp(densities, source_x, values)
            if values.size == 1:
                return np.full_like(densities, float(values.item()), dtype=float)
            padded = np.full_like(densities, default, dtype=float)
            padded[:min(values.size, padded.size)] = values[:min(values.size, padded.size)]
            return padded

        greedy_color = '#355C7D'
        mappo_color = '#C06C84'

        panels = [
            ('overlay_ratio', 'Overlay Ratio', 'Ratio', 1.0),
            ('puncture_ratio', 'Puncture Ratio', 'Ratio', 1.0),
            ('overlay_retention', 'Average Overlay Retention', 'Retention Fraction', 1.0),
            ('puncture_embb_loss', 'Average eMBB Loss per Puncture', 'Loss (Mbps)', 1e-6),
            ('jain_fairness', "Jain's Fairness Index", 'Fairness', 1.0),
            ('minislot_utilization', 'Mini-slot Utilization', 'Utilization Ratio', 1.0),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        axes = axes.flatten()
        for ax, (key, title, ylabel, scale) in zip(axes, panels):
            greedy_values = _series(greedy_result, key) * scale
            mappo_values = _series(mappo_result, key) * scale
            ax.plot(densities, greedy_values, color=greedy_color, marker='o', linewidth=2.4, label=greedy_label)
            ax.plot(densities, mappo_values, color=mappo_color, marker='s', linewidth=2.4, label=mappo_label)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Average UE Load per UAV')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if 'Ratio' in ylabel or 'Retention' in ylabel or 'Fairness' in ylabel:
                ax.set_ylim(0.0, 1.05)
            self._pin_xaxis(ax, densities)

        axes[0].legend(frameon=False, loc='best')
        fig.suptitle(f'{greedy_label} vs {mappo_label}: Mode and Fairness Comparison', fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = save_path or os.path.join(self.output_dir, 'greedy_vs_sr_mappo_modes.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()


class ResultsAnalyzer:
    """Analyze and summarize simulation results."""

    @staticmethod
    def compute_statistics(aggregated_metrics):
        """Compute basic statistics from list-like entries."""
        stats = {}
        for key, values in aggregated_metrics.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                arr = np.asarray(values, dtype=float)
                stats[f'{key}_mean'] = float(np.nanmean(arr))
                stats[f'{key}_std'] = float(np.nanstd(arr))
                stats[f'{key}_min'] = float(np.nanmin(arr))
                stats[f'{key}_max'] = float(np.nanmax(arr))
                stats[f'{key}_median'] = float(np.nanmedian(arr))
        return stats

    @staticmethod
    def print_detailed_report(aggregated_metrics):
        """Print detailed performance report."""
        print("\n" + "=" * 70)
        print("DETAILED PERFORMANCE REPORT")
        print("=" * 70)

        print("\neMBB Performance Metrics:")
        print(f"  Average Rate: {aggregated_metrics['avg_embb_rate']/1e6:.4f} Mbps")
        print(f"  Std Dev: {aggregated_metrics['std_embb_rate']/1e6:.4f} Mbps")
        print(f"  Min: {np.min(aggregated_metrics['all_embb_rates'])/1e6:.4f} Mbps")
        print(f"  Max: {np.max(aggregated_metrics['all_embb_rates'])/1e6:.4f} Mbps")

        print("\nURLLC Reliability Metrics:")
        print(f"  Average Admission Ratio: {aggregated_metrics['avg_urllc_admission']:.4f}")
        print(f"  Average Admitted Reliability: {aggregated_metrics['avg_urllc_success']:.4f}")
        print(f"  Std Dev: {aggregated_metrics['std_urllc_success']:.4f}")
        print(f"  Min: {np.nanmin(aggregated_metrics['all_urllc_success']):.4f}")
        print(f"  Max: {np.nanmax(aggregated_metrics['all_urllc_success']):.4f}")
        print("  Target: 0.99 (99%)")

        print("\nPower Consumption Metrics:")
        print(f"  Average: {aggregated_metrics['avg_total_power']:.6f} W ({aggregated_metrics['avg_total_power']*1e3:.3f} mW)")
        print(f"  Std Dev: {aggregated_metrics['std_total_power']:.6f} W")
        print(f"  Min: {np.min(aggregated_metrics['all_power']):.6f} W")
        print(f"  Max: {np.max(aggregated_metrics['all_power']):.6f} W")

        print("\nResource Utilization:")
        print(f"  RB Utilization: {aggregated_metrics['avg_rb_utilization']:.2%}")


def create_plotter(output_dir='./results/'):
    """Factory function."""
    return SimulationPlotter(output_dir)
