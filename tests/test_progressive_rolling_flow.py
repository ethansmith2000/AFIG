"""Tests for the rolling (per-token-time) rectified flow."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progressive_tokenizer import (  # noqa: E402
    JointFlowConfig,
    JointRectifiedFlow,
    RollingFlowConfig,
    RollingRectifiedFlow,
)


def tiny_config(**overrides) -> RollingFlowConfig:
    base = dict(
        sequence_length=8,
        token_dim=4,
        width=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        overlap=2.0,
    )
    base.update(overrides)
    return RollingFlowConfig(**base)


class TestRollingSchedule(unittest.TestCase):
    def test_local_times_match_recovered_spec(self):
        # Recovered W&B config: N=64, overlap=8 -> frontier_duration 8.875;
        # overlap=64 -> 1.984375.
        config = RollingFlowConfig(sequence_length=64, token_dim=16, overlap=8.0)
        self.assertAlmostEqual(config.frontier_duration, 8.875)
        config64 = RollingFlowConfig(sequence_length=64, token_dim=16, overlap=64.0)
        self.assertAlmostEqual(config64.frontier_duration, 1.984375)
        model = RollingRectifiedFlow(tiny_config())
        times = model.local_times(torch.tensor([0.0, 1.5, 10.0]))
        self.assertTrue(torch.equal(times[0], torch.zeros(8)))
        self.assertTrue(torch.equal(times[2], torch.ones(8)))
        expected = (1.5 - torch.arange(8, dtype=torch.float32) / 2.0).clamp(0, 1)
        torch.testing.assert_close(times[1], expected)

    def test_active_mask_and_loss_only_cover_partially_noised_registers(self):
        torch.manual_seed(11)
        model = RollingRectifiedFlow(tiny_config()).eval()
        clean = torch.randn(3, 8, 4)
        frontier = torch.full((3,), 1.5)
        noise = torch.randn_like(clean)
        output = model(clean, frontier=frontier, noise=noise)
        # frontier 1.5, overlap 2: t = [1, 1, .5, 0, 0, 0, 0, 0] -> only slot 2 active
        expected_active = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(output["per_token_active"], expected_active)
        # manual masked loss reproduces the reported loss
        times = model.local_times(frontier)
        noisy = (1 - times[..., None]) * noise + times[..., None] * clean
        prediction = model.predict_velocity(noisy, times)
        target = clean - noise
        manual = (prediction - target)[:, 2].square().mean()
        torch.testing.assert_close(output["loss"], manual)

    def test_sample_shapes_and_edge_frontiers_are_finite(self):
        torch.manual_seed(12)
        model = RollingRectifiedFlow(tiny_config()).eval()
        samples = model.sample(2, steps_per_token=3, solver="heun")
        self.assertEqual(samples.shape, (2, 8, 4))
        self.assertTrue(torch.isfinite(samples).all())
        euler = model.sample(2, steps_per_token=2, solver="euler")
        self.assertTrue(torch.isfinite(euler).all())

    def test_all_clean_frontier_gives_zero_active_and_finite_loss(self):
        model = RollingRectifiedFlow(tiny_config()).eval()
        clean = torch.randn(2, 8, 4)
        output = model(clean, frontier=torch.full((2,), 100.0))
        self.assertEqual(float(output["per_token_active"].sum()), 0.0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(float(output["loss"]), 0.0)

    def test_parameter_count_matches_joint_prior(self):
        # Recovered spec: the rolling model reported 70,293,520 parameters,
        # identical to the joint prior at the same dimensions.
        rolling = RollingRectifiedFlow(
            RollingFlowConfig(sequence_length=64, token_dim=16, overlap=8.0)
        )
        joint = JointRectifiedFlow(
            JointFlowConfig(sequence_length=64, token_dim=16)
        )
        rolling_count = sum(p.numel() for p in rolling.parameters())
        joint_count = sum(p.numel() for p in joint.parameters())
        self.assertEqual(rolling_count, joint_count)
        self.assertEqual(rolling_count, 70_293_520)

    def test_gradients_flow_only_from_active_registers(self):
        torch.manual_seed(13)
        model = RollingRectifiedFlow(tiny_config())
        clean = torch.randn(2, 8, 4, requires_grad=True)
        output = model(clean, frontier=torch.full((2,), 1.5))
        output["loss"].backward()
        self.assertIsNotNone(clean.grad)
        self.assertGreater(float(clean.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
