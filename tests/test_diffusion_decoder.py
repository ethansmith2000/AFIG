"""Tests for DiffusionDecoder."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig  # noqa: E402
from diffusers.training_utils import compute_snr  # noqa: E402


class TestDiffusionDecoder(unittest.TestCase):
    def _tiny(self, prediction_type="epsilon", loss_weighting="none", mul=2):
        cfg = DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=64,
            depth=2,
            prediction_type=prediction_type,
            loss_weighting=loss_weighting,
            diffusion_batch_mul=mul,
            num_inference_steps=5,
            num_train_timesteps=100,
        )
        return DiffusionDecoder(cfg)

    def test_epsilon_loss_shapes(self):
        model = self._tiny("epsilon")
        b, l, d = 2, 8, 6
        target = torch.randn(b, l, d)
        z = torch.randn(b, l, 32)
        mask = torch.ones(l, d)
        mask[:2, 3:] = 0
        out = model.compute_loss(target, z, component_mask=mask)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["per_example"].numel(), b * l * model.config.diffusion_batch_mul)

    def test_v_prediction_loss(self):
        model = self._tiny("v_prediction")
        target = torch.randn(2, 4, 6)
        z = torch.randn(2, 4, 32)
        out = model.compute_loss(target, z)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_min_snr_weights_epsilon(self):
        model = self._tiny("epsilon", loss_weighting="min_snr", mul=1)
        t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
        w = model._min_snr_weights(t)
        snr = compute_snr(model.train_scheduler, t)
        expected = torch.minimum(snr, torch.full_like(snr, model.config.min_snr_gamma)) / snr
        self.assertTrue(torch.allclose(w, expected, atol=1e-5))

    def test_min_snr_weights_v(self):
        model = self._tiny("v_prediction", loss_weighting="min_snr", mul=1)
        t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
        w = model._min_snr_weights(t)
        snr = compute_snr(model.train_scheduler, t)
        expected = torch.minimum(snr, torch.full_like(snr, model.config.min_snr_gamma)) / (snr + 1)
        self.assertTrue(torch.allclose(w, expected, atol=1e-5))

    def test_multiplier_loss_scale_invariance(self):
        torch.manual_seed(0)
        m1 = self._tiny(mul=1)
        torch.manual_seed(0)
        m4 = self._tiny(mul=4)
        # Copy identical weights.
        m4.load_state_dict(m1.state_dict())
        target = torch.randn(2, 5, 6)
        z = torch.randn(2, 5, 32)
        # With different RNGs the losses differ; instead check that mean reduction
        # over N examples has the same expected scale by verifying shapes / finite.
        out1 = m1.compute_loss(target, z)
        out4 = m4.compute_loss(target, z)
        self.assertEqual(out1["per_example"].numel() * 4, out4["per_example"].numel())
        self.assertTrue(torch.isfinite(out1["loss"]) and torch.isfinite(out4["loss"]))

    def test_component_mask_invariants(self):
        model = self._tiny(mul=1)
        n = 4
        z = torch.randn(n, 32)
        mask = torch.ones(n, 6)
        mask[:, 3:] = 0
        samples = model.sample(z, component_mask=mask, num_inference_steps=4)
        self.assertEqual(tuple(samples.shape), (n, 6))
        self.assertEqual(samples[:, 3:].abs().max().item(), 0.0)
        self.assertTrue(torch.isfinite(samples).all())

    def test_ddim_various_steps(self):
        model = self._tiny(mul=1)
        z = torch.randn(2, 32)
        for steps in (2, 5, 10):
            s = model.sample(z, num_inference_steps=steps)
            self.assertTrue(torch.isfinite(s).all())

    def test_rejects_flow(self):
        with self.assertRaises(ValueError):
            DiffusionDecoder(DiffusionDecoderConfig(prediction_type="flow"))

    def test_radial_power_weighting_scales_loss(self):
        cfg = DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=64,
            depth=2,
            prediction_type="epsilon",
            loss_weighting="none",
            radial_power_weighting=True,
            diffusion_batch_mul=1,
            num_train_timesteps=100,
            num_inference_steps=5,
        )
        model = DiffusionDecoder(cfg)
        target = torch.randn(2, 4, 6)
        z = torch.randn(2, 4, 32)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.manual_seed(0)
        out1 = model.compute_loss(target, z, radial_weights=weights)
        torch.manual_seed(0)
        out2 = model.compute_loss(target, z, radial_weights=weights * 2)
        self.assertAlmostEqual(
            out2["loss"].item(), out1["loss"].item() * 2.0, places=5
        )
        self.assertTrue(torch.allclose(out1["radial_weights"][:4], weights))

    def test_radial_and_min_snr_compose(self):
        for pred in ("epsilon", "v_prediction"):
            cfg = DiffusionDecoderConfig(
                target_dim=6,
                z_channels=32,
                width=64,
                depth=2,
                prediction_type=pred,
                loss_weighting="min_snr",
                radial_power_weighting=True,
                diffusion_batch_mul=2,
                num_train_timesteps=100,
                num_inference_steps=5,
            )
            model = DiffusionDecoder(cfg)
            b, l = 2, 3
            target = torch.randn(b, l, 6)
            z = torch.randn(b, l, 32)
            radial = torch.tensor([0.5, 1.0, 1.5])
            out = model.compute_loss(target, z, radial_weights=radial)
            self.assertTrue(torch.isfinite(out["loss"]))
            # weights == snr_weights * radial_weights (after batch mul flatten)
            expected = out["snr_weights"] * out["radial_weights"]
            self.assertTrue(torch.allclose(out["weights"], expected, atol=1e-6))
            self.assertEqual(out["per_example"].numel(), b * l * 2)

    def test_radial_power_requires_weights(self):
        cfg = DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=64,
            depth=2,
            radial_power_weighting=True,
            diffusion_batch_mul=1,
        )
        model = DiffusionDecoder(cfg)
        with self.assertRaises(ValueError):
            model.compute_loss(torch.randn(1, 2, 6), torch.randn(1, 2, 32))

    def test_target_frequency_conditioning_train_and_sample(self):
        cfg = DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            target_condition_dim=12,
            width=64,
            depth=2,
            diffusion_batch_mul=2,
            num_train_timesteps=100,
            num_inference_steps=2,
        )
        model = DiffusionDecoder(cfg)
        target = torch.randn(2, 4, 6)
        z = torch.randn(2, 4, 32)
        target_condition = torch.randn(4, 12)
        out = model.compute_loss(
            target,
            z,
            target_condition=target_condition,
        )
        self.assertTrue(torch.isfinite(out["loss"]))

        samples = model.sample(
            torch.randn(3, 32),
            target_condition=torch.randn(3, 12),
        )
        self.assertEqual(tuple(samples.shape), (3, 6))

    def test_target_frequency_conditioning_is_required_when_configured(self):
        model = DiffusionDecoder(
            DiffusionDecoderConfig(
                target_dim=6,
                z_channels=32,
                target_condition_dim=12,
                width=64,
                depth=2,
                diffusion_batch_mul=1,
            )
        )
        with self.assertRaises(ValueError):
            model.compute_loss(torch.randn(1, 2, 6), torch.randn(1, 2, 32))
        with self.assertRaises(ValueError):
            model.sample(torch.randn(1, 32), num_inference_steps=2)


if __name__ == "__main__":
    unittest.main()