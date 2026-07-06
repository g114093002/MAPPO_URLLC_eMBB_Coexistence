import unittest

import torch

from sr_mappo.config import SRMAPPOConfig
from sr_mappo.networks import SRMAPPOActorCritic


class PhaseAwareLogProbTest(unittest.TestCase):
    def test_inactive_heads_do_not_affect_aggregated_log_prob(self) -> None:
        torch.manual_seed(0)

        cfg = SRMAPPOConfig()
        cfg.env.learn_embb_baseline = True
        cfg.env.learn_phase0_embb_power = True
        cfg.env.allow_phase_a_embb_power_adjustment = False
        cfg.action.max_candidate_packets = 8
        cfg.action.max_embb_candidates = 12
        cfg.action.include_null_packet_option = True
        cfg.action.include_null_embb_option = True

        model = SRMAPPOActorCritic(local_obs_dim=7, global_obs_dim=9, cfg=cfg)
        model.eval()

        batch_size = 2
        local_obs = torch.randn(batch_size, 7)
        global_obs = torch.randn(batch_size, 9)

        mode_mask = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        packet_mask = torch.zeros(batch_size, 3, 9, dtype=torch.float32)
        packet_mask[0, :, 0] = 1.0
        packet_mask[1, 0, 0] = 1.0
        packet_mask[1, 1, 1] = 1.0
        packet_mask[1, 2, 2] = 1.0

        embb_owner_mask = torch.zeros(batch_size, 13, dtype=torch.float32)
        embb_owner_mask[0, 1] = 1.0
        embb_owner_mask[0, 2] = 1.0
        embb_owner_mask[1, 0] = 1.0

        base = model.evaluate_actions(
            local_obs=local_obs,
            global_obs=global_obs,
            mode_actions=torch.tensor([0, 1], dtype=torch.long),
            packet_actions=torch.tensor([0, 1], dtype=torch.long),
            power_pre_tanh=torch.zeros(batch_size, 1),
            embb_owner_actions=torch.tensor([1, 0], dtype=torch.long),
            embb_power_pre_tanh=torch.zeros(batch_size, 1),
            mode_mask=mode_mask,
            packet_mask=packet_mask,
            embb_owner_mask=embb_owner_mask,
        )

        phase0_changed = model.evaluate_actions(
            local_obs=local_obs,
            global_obs=global_obs,
            mode_actions=torch.tensor([2, 1], dtype=torch.long),
            packet_actions=torch.tensor([8, 1], dtype=torch.long),
            power_pre_tanh=torch.zeros(batch_size, 1),
            embb_owner_actions=torch.tensor([1, 0], dtype=torch.long),
            embb_power_pre_tanh=torch.zeros(batch_size, 1),
            mode_mask=mode_mask,
            packet_mask=packet_mask,
            embb_owner_mask=embb_owner_mask,
        )

        phasea_changed = model.evaluate_actions(
            local_obs=local_obs,
            global_obs=global_obs,
            mode_actions=torch.tensor([0, 1], dtype=torch.long),
            packet_actions=torch.tensor([0, 1], dtype=torch.long),
            power_pre_tanh=torch.zeros(batch_size, 1),
            embb_owner_actions=torch.tensor([1, 5], dtype=torch.long),
            embb_power_pre_tanh=torch.zeros(batch_size, 1),
            mode_mask=mode_mask,
            packet_mask=packet_mask,
            embb_owner_mask=embb_owner_mask,
        )

        self.assertTrue(torch.allclose(base["log_prob"][0], phase0_changed["log_prob"][0], atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(base["log_prob"][1], phasea_changed["log_prob"][1], atol=1.0e-6, rtol=1.0e-6))
        self.assertAlmostEqual(float(base["mode_log_prob_term"][0].item()), 0.0, places=7)
        self.assertAlmostEqual(float(base["packet_log_prob_term"][0].item()), 0.0, places=7)
        self.assertAlmostEqual(float(base["embb_owner_log_prob_term"][1].item()), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
