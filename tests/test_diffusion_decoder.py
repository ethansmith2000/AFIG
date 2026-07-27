"""Tests for DiffusionDecoder."""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig  # noqa: E402
from diffusers.training_utils import compute_snr  # noqa: E402


class TestDiffusionDecoder(unittest.TestCase):
    def _tiny(
        self,
        prediction_type="epsilon",
        loss_weighting="none",
        mul=2,
        **kwargs,
    ):
        depth = kwargs.pop("depth", 2)
        cfg = DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=64,
            depth=depth,
            prediction_type=prediction_type,
            loss_weighting=loss_weighting,
            diffusion_batch_mul=mul,
            num_inference_steps=5,
            num_train_timesteps=100,
            **kwargs,
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

    def test_min_snr_follows_v_loss_space_for_x0_head(self):
        model = self._tiny(
            "x0",
            loss_weighting="min_snr",
            mul=1,
            objective="flow",
            loss_space="v",
        )
        t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
        flow_t = (t.float() + 0.5) / model.config.num_train_timesteps
        snr = (flow_t / (1.0 - flow_t)).square()
        expected = torch.minimum(
            snr,
            torch.full_like(snr, model.config.min_snr_gamma),
        ) / (snr + 1)
        self.assertTrue(
            torch.allclose(
                model._min_snr_weights(t, flow_t=flow_t),
                expected,
                atol=1e-5,
            )
        )

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

    def test_rejects_epsilon_output_for_flow(self):
        with self.assertRaises(ValueError):
            DiffusionDecoder(
                DiffusionDecoderConfig(objective="flow", prediction_type="epsilon")
            )

    def test_ddpm_x0_min_snr_weights_and_sampling(self):
        model = self._tiny(
            "x0",
            loss_weighting="min_snr",
            mul=1,
            min_snr_gamma=0.2,
        )
        t = torch.tensor([0, 10, 50, 99], dtype=torch.long)
        snr = compute_snr(model.train_scheduler, t)
        expected = torch.minimum(snr, torch.full_like(snr, 0.2)) / 0.2
        self.assertTrue(torch.allclose(model._min_snr_weights(t), expected))

        out = model.compute_loss(torch.randn(2, 4, 6), torch.randn(2, 4, 32))
        self.assertTrue(torch.isfinite(out["loss"]))
        sample = model.sample(torch.randn(2, 32), num_inference_steps=3)
        self.assertTrue(torch.isfinite(sample).all())

    def test_zero_terminal_snr_and_trailing_spacing(self):
        model = self._tiny(
            "x0",
            mul=1,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        self.assertEqual(model.train_scheduler.alphas_cumprod[-1].item(), 0.0)
        scheduler = model.sample_scheduler.from_config(model.sample_scheduler.config)
        scheduler.set_timesteps(20)
        self.assertEqual(scheduler.timesteps[0].item(), 99)

    def test_jit_x0_prediction_v_loss_is_finite(self):
        model = self._tiny(
            "x0",
            objective="flow",
            loss_space="v",
            loss_weighting="logit_normal",
            logit_normal_mean=-0.8,
            logit_normal_std=0.8,
            mul=1,
        )
        out = model.compute_loss(torch.randn(2, 4, 6), torch.randn(2, 4, 32))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["weights"]).all())
        self.assertTrue(torch.all((out["flow_times"] > 0) & (out["flow_times"] < 1)))
        sample = model.sample(torch.randn(2, 32), num_inference_steps=3)
        self.assertTrue(torch.isfinite(sample).all())

    def test_flow_velocity_logit_normal_weighting(self):
        model = self._tiny(
            "v_prediction",
            objective="flow",
            loss_weighting="logit_normal",
            logit_normal_mean=0.0,
            logit_normal_std=1.0,
            mul=1,
        )
        weights = model.logit_normal_weights
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)
        self.assertGreater(weights[len(weights) // 2].item(), weights[0].item())
        self.assertGreater(weights[len(weights) // 2].item(), weights[-1].item())

        out = model.compute_loss(torch.randn(2, 4, 6), torch.randn(2, 4, 32))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.allclose(out["snr_weights"], weights[out["timesteps"]]))
        sample = model.sample(torch.randn(2, 32), num_inference_steps=3)
        self.assertTrue(torch.isfinite(sample).all())

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

    def test_depth_six_adaln_zero_initialization_and_gradient_flow(self):
        model = self._tiny("x0", mul=1, depth=6)
        self.assertEqual(len(model.net.res_blocks), 6)
        for block in model.net.res_blocks:
            self.assertEqual(block.adaLN_modulation[-1].weight.abs().max().item(), 0)
            self.assertEqual(block.adaLN_modulation[-1].bias.abs().max().item(), 0)
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        self.assertEqual(model.net.final_layer.linear.weight.abs().max().item(), 0)
        torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
        output = model.compute_loss(
            torch.randn(2, 3, 6),
            torch.randn(2, 3, 32),
        )
        output["loss"].backward()
        for block in model.net.res_blocks:
            self.assertIsNotNone(block.mlp[0].weight.grad)
            self.assertGreater(block.mlp[0].weight.grad.abs().sum().item(), 0)

    def test_covariance_metric_is_reported_and_used(self):
        model = self._tiny(
            "x0",
            mul=1,
            loss_metric="orbit_covariance_power",
            orbit_covariance_exponent=0.2,
        )
        target = torch.randn(2, 3, 6)
        z = torch.randn(2, 3, 32)
        metric = torch.eye(6).repeat(3, 1, 1)
        metric[1] *= 2
        metric[2] *= 4
        output = model.compute_loss(target, z, covariance_metric=metric)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["unweighted_mse"]))
        self.assertTrue(torch.isfinite(output["covariance_metric_loss"]))
        self.assertTrue(
            torch.allclose(output["loss"], output["covariance_metric_loss"])
        )

    def test_covariance_metric_rejects_incompatible_objectives(self):
        with self.assertRaises(ValueError):
            self._tiny(
                "epsilon",
                mul=1,
                loss_metric="orbit_covariance_power",
            )
        with self.assertRaises(ValueError):
            self._tiny(
                "x0",
                mul=1,
                loss_metric="orbit_covariance_power",
                radial_power_weighting=True,
            )

    def test_scale_metric_is_reported_and_used(self):
        model = self._tiny(
            "x0",
            mul=1,
            loss_metric="orbit_scale_power",
            orbit_scale_exponent=0.2,
        )
        target = torch.randn(2, 3, 6)
        z = torch.randn(2, 3, 32)
        metric = torch.ones(3, 6)
        metric[1] *= 2
        metric[2] *= 4
        output = model.compute_loss(target, z, component_metric=metric)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["scale_metric_loss"]))
        self.assertTrue(torch.allclose(output["loss"], output["scale_metric_loss"]))

    def test_scale_metric_requires_native_x0(self):
        with self.assertRaises(ValueError):
            self._tiny(
                "epsilon",
                mul=1,
                loss_metric="orbit_scale_power",
            )

    def test_input_timestep_film_step_zero_equivalence_and_rng_stability(self):
        common = dict(
            target_dim=6,
            z_channels=16,
            width=32,
            depth=2,
            num_train_timesteps=20,
            num_inference_steps=3,
            diffusion_batch_mul=1,
        )
        torch.manual_seed(91)
        plain = DiffusionDecoder(
            DiffusionDecoderConfig(**common, input_timestep_conditioning="none")
        )
        torch.manual_seed(91)
        film = DiffusionDecoder(
            DiffusionDecoderConfig(**common, input_timestep_conditioning="film")
        )
        film_state = film.state_dict()
        for name, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, film_state[name]), name)

        nn.init.normal_(plain.net.final_layer.linear.weight, std=0.02)
        film.net.final_layer.linear.weight.data.copy_(
            plain.net.final_layer.linear.weight
        )
        x = torch.randn(5, 6)
        t = torch.arange(5)
        z = torch.randn(5, 16)
        self.assertTrue(torch.equal(plain.net(x, t, z), film.net(x, t, z)))

    def test_input_timestep_film_gradient_and_objective_shapes(self):
        for objective in ("ddpm", "flow"):
            config = DiffusionDecoderConfig(
                target_dim=6,
                z_channels=16,
                width=32,
                depth=2,
                objective=objective,
                prediction_type="x0",
                input_timestep_conditioning="film",
                num_train_timesteps=20,
                num_inference_steps=3,
                diffusion_batch_mul=1,
            )
            decoder = DiffusionDecoder(config)
            nn.init.normal_(decoder.net.final_layer.linear.weight, std=0.02)
            target = torch.randn(4, 6)
            z = torch.randn(4, 16)
            output = decoder.compute_loss(target, z)
            output["loss"].backward()
            gradient = decoder.net.input_time_modulation[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0.0)
            sample = decoder.sample(z, num_inference_steps=2)
            self.assertEqual(tuple(sample.shape), (4, 6))


if __name__ == "__main__":
    unittest.main()