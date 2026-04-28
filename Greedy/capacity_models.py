"""
Capacity Models - Shannon and finite blocklength
Implements information-theoretic formulas for eMBB and URLLC
"""

import numpy as np
from scipy.special import erfc, erfcinv
from config import URLLCConfig, eMBBConfig


class CapacityModels:
    """Shannon and finite blocklength capacity formulas"""

    def __init__(self, urllc_cfg: URLLCConfig = None, embb_cfg: eMBBConfig = None):
        self.urllc_cfg = urllc_cfg or URLLCConfig()
        self.embb_cfg = embb_cfg or eMBBConfig()

    def shannon_capacity(self, snir_linear, bandwidth_hz):
        """Shannon capacity in bits/s."""
        if snir_linear <= 0:
            return 0.0
        return bandwidth_hz * np.log2(1 + snir_linear)

    def shannon_capacity_bits(self, snir_db, bandwidth_hz):
        """Shannon capacity with SNIR in dB."""
        snir_linear = 10 ** (snir_db / 10)
        return self.shannon_capacity(snir_linear, bandwidth_hz)

    def finite_blocklength_capacity(self, snir_linear, packet_bits, channel_uses,
                                    target_error_prob=None):
        """
        Finite blocklength achievable rate in bits/channel use.
        """
        if target_error_prob is None:
            target_error_prob = self.urllc_cfg.target_error_probability

        if snir_linear <= 0:
            return 0.0

        channel_uses = max(1, int(channel_uses))
        c_shannon = np.log2(1 + snir_linear)
        dispersion = (1 - 1 / ((1 + snir_linear) ** 2)) * (np.log2(np.e) ** 2)
        dispersion = max(dispersion, 1e-12)
        q_inv = self._q_inverse(target_error_prob)
        penalty = np.sqrt(dispersion / channel_uses) * q_inv

        return max(0.0, c_shannon - penalty)

    def finite_blocklength_capacity_db(self, snir_db, packet_bits, channel_uses,
                                       target_error_prob=None):
        """SNIR in dB version."""
        snir_linear = 10 ** (snir_db / 10)
        return self.finite_blocklength_capacity(
            snir_linear, packet_bits, channel_uses, target_error_prob
        )

    def decoding_error_probability(self, snir_linear, packet_bits, channel_uses):
        """
        Normal-approximation decoding error probability for finite blocklength.
        """
        if snir_linear <= 0:
            return 1.0

        channel_uses = max(1, int(channel_uses))
        rate = packet_bits / channel_uses
        capacity = np.log2(1 + snir_linear)

        if rate >= capacity:
            return 1.0

        dispersion = (1 - 1 / ((1 + snir_linear) ** 2)) * (np.log2(np.e) ** 2)
        dispersion = max(dispersion, 1e-12)
        argument = (capacity - rate) / np.sqrt(dispersion / channel_uses)
        error_prob = 0.5 * erfc(argument / np.sqrt(2))

        return float(np.clip(error_prob, 0.0, 1.0))

    def _q_inverse(self, prob):
        """Inverse Q-function."""
        if prob <= 0:
            return np.inf
        if prob >= 1:
            return 0.0
        return np.sqrt(2) * erfcinv(2 * prob)

    def min_power_for_reliability(self, target_error_prob, packet_bits,
                                  channel_uses, noise_power_linear):
        """
        Find minimum SNIR required to satisfy the target error probability.
        """
        lower_snir = 1e-6
        upper_snir = 1e4
        tolerance = 1e-4

        for _ in range(40):
            mid_snir = (lower_snir + upper_snir) / 2
            error_prob = self.decoding_error_probability(
                mid_snir, packet_bits, channel_uses
            )

            if error_prob > target_error_prob:
                lower_snir = mid_snir
            else:
                upper_snir = mid_snir

            if (upper_snir - lower_snir) / max(mid_snir, 1e-12) < tolerance:
                break

        return (lower_snir + upper_snir) / 2

    def spectral_efficiency_embb(self, snir_db):
        """Spectral efficiency for eMBB (bps/Hz)."""
        snir_linear = 10 ** (snir_db / 10)
        return np.log2(1 + snir_linear)

    def spectral_efficiency_urllc(self, snir_db, packet_bits, slot_duration_ms):
        """Effective URLLC spectral efficiency (bps/Hz)."""
        max_rate = packet_bits / (slot_duration_ms * 1e-3)
        snir_linear = 10 ** (snir_db / 10)
        shannon_rate = 1e6 * np.log2(1 + snir_linear)
        return min(shannon_rate, max_rate) / 1e6

    def calculate_required_rb_count(self, target_bits, snir_db, bandwidth_hz,
                                    duration_ms):
        """Calculate minimum RBs needed for the target payload."""
        snir_linear = 10 ** (snir_db / 10)
        capacity_per_hz_per_s = np.log2(1 + snir_linear)
        total_capacity_bits = capacity_per_hz_per_s * bandwidth_hz * duration_ms / 1000
        rb_count = np.ceil(target_bits / max(total_capacity_bits, 1e-12))
        return int(rb_count)


def create_capacity_model(urllc_cfg=None, embb_cfg=None):
    """Factory function."""
    return CapacityModels(urllc_cfg, embb_cfg)
