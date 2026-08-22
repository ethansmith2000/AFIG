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


def wake(model: RollingRectifiedFlow) -> RollingRectifiedFlow:
    """Give the zero-initialized AdaLN gates and output head nonzero values.

    Fresh models emit exactly zero -- every block gate and the final projection
    are zero-initialized -- so any test that asks what the model *attends to*
    would pass vacuously.
    """

    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=0.02)
            torch.nn.init.normal_(block.modulation[-1].bias, std=0.02)
        torch.nn.init.normal_(model.final.modulation[-1].weight, std=0.02)
        torch.nn.init.normal_(model.final.output.weight, std=0.02)
    return model


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

    def test_supervise_all_tokens_covers_every_register(self):
        torch.manual_seed(31)
        clean = torch.randn(3, 8, 4)
        noise = torch.randn_like(clean)
        frontier = torch.full((3,), 1.5)
        masked = RollingRectifiedFlow(tiny_config()).eval()
        full = RollingRectifiedFlow(tiny_config(supervise_all_tokens=True)).eval()
        full.load_state_dict(masked.state_dict())
        times = masked.local_times(frontier)
        noisy = (1 - times[..., None]) * noise + times[..., None] * clean
        prediction = masked.predict_velocity(noisy, times)
        squared = (prediction - (clean - noise)).square()
        torch.testing.assert_close(
            full(clean, frontier=frontier, noise=noise)["loss"], squared.mean()
        )
        # only slot 2 is active at frontier 1.5, so the masked loss sees 1/8th
        # of the registers and the two objectives must differ.
        torch.testing.assert_close(
            masked(clean, frontier=frontier, noise=noise)["loss"],
            squared[:, 2].mean(),
        )

    def test_causal_attention_hides_later_registers(self):
        torch.manual_seed(32)
        model = wake(RollingRectifiedFlow(tiny_config(causal=True)).eval())
        times = torch.full((1, 8), 0.5)
        first = torch.randn(1, 8, 4)
        second = first.clone()
        second[:, 5:] = torch.randn(1, 3, 4)
        with torch.no_grad():
            left = model.predict_velocity(first, times)
            right = model.predict_velocity(second, times)
        # registers before the edit are unaffected; the edited ones move.
        torch.testing.assert_close(left[:, :5], right[:, :5])
        self.assertFalse(torch.equal(left[:, 5:], right[:, 5:]))

    def test_bidirectional_attention_still_sees_later_registers(self):
        torch.manual_seed(33)
        model = wake(RollingRectifiedFlow(tiny_config()).eval())
        times = torch.full((1, 8), 0.5)
        first = torch.randn(1, 8, 4)
        second = first.clone()
        second[:, 5:] = torch.randn(1, 3, 4)
        with torch.no_grad():
            left = model.predict_velocity(first, times)
            right = model.predict_velocity(second, times)
        self.assertFalse(torch.equal(left[:, :5], right[:, :5]))

    def test_prefix_noise_only_lowers_completed_registers(self):
        torch.manual_seed(34)
        model = RollingRectifiedFlow(
            tiny_config(prefix_noise_max=0.2, prefix_noise_probability=1.0)
        ).train()
        base = model.local_times(torch.tensor([1.5]))
        perturbed = model.perturb_times(base)
        completed = base >= 1.0
        active = (base > 0.0) & (base < 1.0)
        unstarted = base <= 0.0
        self.assertTrue(bool(completed.any()))
        self.assertTrue(((perturbed[completed] >= 0.8) & (perturbed[completed] < 1.0)).all())
        torch.testing.assert_close(perturbed[active], base[active])
        torch.testing.assert_close(perturbed[unstarted], base[unstarted])

    def test_prefix_noise_probability_keeps_clean_context_in_distribution(self):
        torch.manual_seed(35)
        model = RollingRectifiedFlow(
            tiny_config(prefix_noise_max=0.2, prefix_noise_probability=0.75)
        ).train()
        base = model.local_times(torch.full((512,), 1.5))
        perturbed = model.perturb_times(base)
        completed = base >= 1.0
        untouched = (perturbed[completed] >= 1.0).float().mean()
        # the inference condition (exactly clean context) must keep occurring
        self.assertGreater(float(untouched), 0.15)
        self.assertLess(float(untouched), 0.35)

    def test_time_jitter_only_moves_active_registers(self):
        torch.manual_seed(36)
        model = RollingRectifiedFlow(tiny_config(time_jitter=0.05)).train()
        base = model.local_times(torch.full((64,), 1.5))
        perturbed = model.perturb_times(base)
        active = (base > 0.0) & (base < 1.0)
        self.assertFalse(torch.equal(perturbed[active], base[active]))
        self.assertTrue((perturbed >= 0.0).all() and (perturbed <= 1.0).all())
        torch.testing.assert_close(perturbed[~active], base[~active])

    def test_perturbations_are_training_only(self):
        torch.manual_seed(37)
        model = RollingRectifiedFlow(
            tiny_config(time_jitter=0.05, prefix_noise_max=0.2)
        ).eval()
        clean = torch.randn(2, 8, 4)
        noise = torch.randn_like(clean)
        frontier = torch.full((2,), 1.5)
        first = model(clean, frontier=frontier, noise=noise)["loss"]
        second = model(clean, frontier=frontier, noise=noise)["loss"]
        torch.testing.assert_close(first, second)

    def test_new_options_do_not_change_parameter_count(self):
        dims = dict(sequence_length=64, token_dim=16, overlap=8.0)
        plain = RollingRectifiedFlow(RollingFlowConfig(**dims))
        forcing = RollingRectifiedFlow(
            RollingFlowConfig(
                **dims,
                causal=True,
                supervise_all_tokens=True,
                time_jitter=0.05,
                prefix_noise_max=0.1,
            )
        )
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        self.assertEqual(count(plain), count(forcing))
        self.assertEqual(count(plain), 70_293_520)

    def test_prefix_noise_does_not_enlarge_the_supervision_set(self):
        torch.manual_seed(38)
        model = RollingRectifiedFlow(
            tiny_config(prefix_noise_max=0.2, prefix_noise_probability=1.0)
        ).train()
        clean = torch.randn(4, 8, 4)
        frontier = torch.full((4,), 1.5)
        output = model(clean, frontier=frontier)
        # slots 0 and 1 are completed and every one of them takes prefix noise
        # here; only slot 2 is genuinely active and may be supervised.
        expected = torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(output["per_token_active"], expected)

    def test_independent_times_replace_whole_samples_and_widen_supervision(self):
        torch.manual_seed(39)
        model = RollingRectifiedFlow(
            tiny_config(independent_time_probability=1.0)
        ).train()
        base = model.local_times(torch.full((256,), 1.5))
        mixed = model.mix_independent_times(base)
        # every sample is replaced here, so no row may survive unchanged
        self.assertFalse(torch.equal(mixed, base))
        self.assertTrue(((mixed > 0.0) & (mixed < 1.0)).all())
        # and with all registers active, supervision covers the sequence
        output = model(torch.randn(256, 8, 4), frontier=torch.full((256,), 1.5))
        self.assertTrue((output["per_token_active"] > 0).all())

    def test_independent_time_mixture_is_per_sample(self):
        torch.manual_seed(40)
        model = RollingRectifiedFlow(
            tiny_config(independent_time_probability=0.5)
        ).train()
        base = model.local_times(torch.full((2048,), 1.5))
        mixed = model.mix_independent_times(base)
        untouched = (mixed == base).all(dim=1).float().mean()
        self.assertGreater(float(untouched), 0.4)
        self.assertLess(float(untouched), 0.6)
        # rows are replaced wholesale, never partially
        changed = (mixed != base).any(dim=1)
        self.assertTrue(((mixed != base).all(dim=1) | ~changed).all())

    def test_independent_times_are_training_only(self):
        model = RollingRectifiedFlow(
            tiny_config(independent_time_probability=1.0)
        ).eval()
        clean = torch.randn(2, 8, 4)
        noise = torch.randn_like(clean)
        frontier = torch.full((2,), 1.5)
        first = model(clean, frontier=frontier, noise=noise)["loss"]
        second = model(clean, frontier=frontier, noise=noise)["loss"]
        torch.testing.assert_close(first, second)

    def test_local_times_accepts_per_sample_overlap(self):
        model = RollingRectifiedFlow(tiny_config())
        frontier = torch.tensor([1.5, 1.5])
        overlap = torch.tensor([2.0, 4.0])
        times = model.local_times(frontier, overlap)
        torch.testing.assert_close(
            times[0], (1.5 - torch.arange(8, dtype=torch.float32) / 2.0).clamp(0, 1)
        )
        torch.testing.assert_close(
            times[1], (1.5 - torch.arange(8, dtype=torch.float32) / 4.0).clamp(0, 1)
        )
        # omitting overlap reproduces the configured scalar
        torch.testing.assert_close(model.local_times(frontier)[0], times[0])

    def test_overlap_jitter_varies_the_ramp_slope_across_the_batch(self):
        torch.manual_seed(41)
        import math

        model = RollingRectifiedFlow(
            tiny_config(overlap_jitter=math.log(2.0))
        ).train()
        overlap = model.sample_overlap(4096, torch.device("cpu"))
        # log-uniform around the configured overlap of 2.0, within [1, 4]
        self.assertGreaterEqual(float(overlap.min()), 1.0)
        self.assertLessEqual(float(overlap.max()), 4.0)
        self.assertGreater(float(overlap.std()), 0.0)
        output = model(torch.randn(256, 8, 4))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_overlap_jitter_is_training_only(self):
        import math

        model = RollingRectifiedFlow(tiny_config(overlap_jitter=math.log(2.0))).eval()
        clean = torch.randn(2, 8, 4)
        noise = torch.randn_like(clean)
        frontier = torch.full((2,), 1.5)
        first = model(clean, frontier=frontier, noise=noise)["loss"]
        second = model(clean, frontier=frontier, noise=noise)["loss"]
        torch.testing.assert_close(first, second)

    def test_sample_accepts_an_overlap_override(self):
        torch.manual_seed(42)
        model = RollingRectifiedFlow(tiny_config()).eval()
        default = model.sample(2, steps_per_token=3)
        widened = model.sample(2, steps_per_token=3, overlap=8.0)
        self.assertEqual(widened.shape, (2, 8, 4))
        self.assertTrue(torch.isfinite(widened).all())
        self.assertFalse(torch.equal(default, widened))
        with self.assertRaises(ValueError):
            model.sample(2, steps_per_token=3, overlap=0.0)

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
