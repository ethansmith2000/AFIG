"""Tests for the history-robust and head-position AR variants."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progressive_tokenizer import (  # noqa: E402
    AutoregressiveFlowConfig,
    AutoregressiveRectifiedFlow,
)


def tiny_config(**overrides) -> AutoregressiveFlowConfig:
    base = dict(
        sequence_length=6,
        token_dim=4,
        width=32,
        trunk_depth=2,
        head_depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    base.update(overrides)
    return AutoregressiveFlowConfig(**base)


class TestARVariants(unittest.TestCase):
    def test_parameter_counts_match_recovered_lost_run_specs(self):
        # Recovered W&B configs at 64x16, width 512: plain 76,595,728;
        # history-robust 76,859,408; headpos 77,152,784.
        dims = dict(sequence_length=64, token_dim=16, width=512)
        plain = AutoregressiveRectifiedFlow(AutoregressiveFlowConfig(**dims))
        robust = AutoregressiveRectifiedFlow(
            AutoregressiveFlowConfig(**dims, history_reliability_conditioning=True)
        )
        headpos = AutoregressiveRectifiedFlow(
            AutoregressiveFlowConfig(**dims, head_position_conditioning=True)
        )
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        self.assertEqual(count(plain), 76_595_728)
        self.assertEqual(count(robust), 76_859_408)
        self.assertEqual(count(headpos), 77_152_784)

    def test_zero_history_noise_matches_clean_trunk(self):
        torch.manual_seed(21)
        model = AutoregressiveRectifiedFlow(
            tiny_config(history_reliability_conditioning=True)
        ).eval()
        clean = torch.randn(2, 6, 4)
        baseline = model.trunk(clean)
        zero_sigma = model.trunk(clean, torch.zeros(2, 6))
        torch.testing.assert_close(baseline, zero_sigma, rtol=0, atol=0)
        noisy = model.trunk(clean, torch.full((2, 6), 0.5))
        self.assertFalse(torch.equal(baseline, noisy))

    def test_history_noise_does_not_leak_into_targets(self):
        torch.manual_seed(22)
        model = AutoregressiveRectifiedFlow(
            tiny_config(history_reliability_conditioning=True)
        ).eval()
        clean = torch.randn(2, 6, 4)
        time = torch.rand(2, 6)
        noise = torch.randn_like(clean)
        torch.manual_seed(7)
        with_noise = model(
            clean, time=time, noise=noise, history_noise_sigma=torch.zeros(2, 6)
        )
        torch.manual_seed(7)
        without = model(clean, time=time, noise=noise)
        torch.testing.assert_close(with_noise["loss"], without["loss"])

    def test_head_position_conditioning_forward_and_generate(self):
        torch.manual_seed(23)
        model = AutoregressiveRectifiedFlow(
            tiny_config(head_position_conditioning=True)
        ).eval()
        clean = torch.randn(2, 6, 4)
        output = model(clean)
        self.assertTrue(torch.isfinite(output["loss"]))
        samples = model.generate(2, steps=2)
        self.assertEqual(samples.shape, (2, 6, 4))
        self.assertTrue(torch.isfinite(samples).all())
        # slot identity must change the head's behavior. The zero-initialized
        # AdaLN gates make the untrained head condition-independent, so give
        # every head parameter a nonzero value first.
        with torch.no_grad():
            for parameter in model.head.parameters():
                torch.nn.init.normal_(parameter, std=0.02)
        condition = model.trunk(clean)[:, 0]
        noisy = torch.randn(2, 4)
        time = torch.zeros(2)
        slot0 = model.head.predict_velocity(
            noisy, time, condition, torch.zeros(2, dtype=torch.long)
        )
        slot3 = model.head.predict_velocity(
            noisy, time, condition, torch.full((2,), 3, dtype=torch.long)
        )
        self.assertFalse(torch.equal(slot0, slot3))

    def test_plain_config_is_backward_compatible(self):
        model = AutoregressiveRectifiedFlow(tiny_config())
        self.assertIsNone(model.trunk.reliability)
        self.assertIsNone(model.head.slot_embedding)
        clean = torch.randn(2, 6, 4)
        output = model(clean)
        self.assertTrue(torch.isfinite(output["loss"]))


if __name__ == "__main__":
    unittest.main()
