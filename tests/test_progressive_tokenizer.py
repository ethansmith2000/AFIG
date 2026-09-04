"""Focused tests for the whole-image progressive tokenizer."""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progressive_tokenizer import ProgressiveTokenizer, TokenizerConfig  # noqa: E402
from progressive_tokenizer.model import Rotary2D  # noqa: E402
from progressive_tokenizer.training import (  # noqa: E402
    gaussian_lowpass_pyramid_fft,
    lpips_reconstruction_loss,
    marginal_kurtosis_penalty,
    optimizer_parameter_groups,
    radial_log_power_reconstruction_loss,
    slot_variance_balance_penalty,
)


def tiny_config() -> TokenizerConfig:
    return TokenizerConfig(
        image_size=8,
        patch_size=4,
        num_latents=4,
        latent_dim=8,
        width=32,
        num_heads=4,
        encoder_depth=1,
        pool_depth=1,
        decoder_depth=1,
        mlp_ratio=2.0,
    )


class TestProgressiveTokenizer(unittest.TestCase):
    def test_gaussian_pyramid_preserves_dc_and_has_exact_endpoint(self):
        constant = torch.full((2, 3, 8, 8), 0.25)
        levels = gaussian_lowpass_pyramid_fft(constant, (4.0, 2.0, 0.0))
        self.assertEqual(levels.shape, (2, 3, 3, 8, 8))
        torch.testing.assert_close(levels[:, 0], constant)
        torch.testing.assert_close(levels[:, 1], constant)
        self.assertTrue(torch.equal(levels[:, -1], constant.float()))

    def test_gaussian_pyramid_dog_residuals_telescope(self):
        torch.manual_seed(0)
        images = torch.randn(2, 3, 8, 8)
        levels = gaussian_lowpass_pyramid_fft(images, (4.0, 2.0, 1.0, 0.0))
        residuals = torch.cat(
            (levels[:, :1], levels[:, 1:] - levels[:, :-1]), dim=1
        )
        torch.testing.assert_close(residuals.sum(dim=1), images)
        checkerboard = torch.tensor(
            [[(-1.0) ** (row + column) for column in range(8)] for row in range(8)]
        ).view(1, 1, 8, 8)
        filtered = gaussian_lowpass_pyramid_fft(checkerboard, (2.0, 1.0, 0.0))
        rms = filtered.square().mean(dim=(0, 2, 3, 4)).sqrt()
        self.assertLess(float(rms[0]), float(rms[1]))
        self.assertLess(float(rms[1]), float(rms[2]))

    def test_grouped_hierarchy_decodes_share_noisy_latents_and_train(self):
        torch.manual_seed(1)
        model = ProgressiveTokenizer(tiny_config())
        images = torch.randn(4, 3, 8, 8)
        output = model(
            images,
            noise_mode="add",
            noise_scales=torch.full((4, 4), 0.05),
            hierarchy_prefix_lengths=torch.tensor([2, 4]),
            hierarchy_previous_prefix_lengths=torch.tensor([1, 2]),
        )
        self.assertEqual(output["hierarchy_reconstruction"].shape, (2, 3, 8, 8))
        self.assertEqual(
            output["hierarchy_previous_reconstruction"].shape, (2, 3, 8, 8)
        )
        loss = (
            F.mse_loss(output["reconstruction"], images)
            + output["hierarchy_reconstruction"].square().mean()
            + output["hierarchy_previous_reconstruction"].square().mean()
        )
        loss.backward()
        self.assertGreater(float(model.pool_queries.grad.abs().sum()), 0.0)

    def test_radial_log_power_loss_is_distinct_finite_and_differentiable(self):
        torch.manual_seed(1)
        target = torch.randn(2, 3, 8, 8)
        reconstruction = (target + 0.1 * torch.randn_like(target)).requires_grad_()
        identical = radial_log_power_reconstruction_loss(target, target)
        loss = radial_log_power_reconstruction_loss(target, reconstruction)
        self.assertLess(float(identical), 1e-7)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(reconstruction.grad).all())
        self.assertGreater(float(reconstruction.grad.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            radial_log_power_reconstruction_loss(
                target, reconstruction.detach(), relative_floor=0.0
            )

    def test_lpips_adapter_preserves_reconstruction_gradients_for_tiny_images(self):
        class SquaredFeatureDistance(torch.nn.Module):
            def forward(self, first, second, normalize=False):
                self.normalize = normalize
                self.shape = first.shape
                return (first - second).square().mean(dim=(1, 2, 3), keepdim=True)

        model = SquaredFeatureDistance()
        target = torch.randn(2, 3, 8, 8)
        reconstruction = torch.randn_like(target, requires_grad=True)
        loss = lpips_reconstruction_loss(model, target, reconstruction)
        loss.backward()
        self.assertEqual(model.shape, (2, 3, 32, 32))
        self.assertFalse(model.normalize)
        self.assertTrue(torch.isfinite(reconstruction.grad).all())

    def test_slot_variance_balance_is_scale_invariant_and_has_gradients(self):
        torch.manual_seed(2)
        base = torch.randn(32, 1, 3)
        scales = torch.tensor([0.5, 1.0, 2.0, 4.0]).view(1, 4, 1)
        latents = (base.repeat(1, 4, 1) * scales).requires_grad_()
        penalty = slot_variance_balance_penalty(latents)
        balanced = slot_variance_balance_penalty(latents / scales)
        globally_scaled = slot_variance_balance_penalty(latents * 7.0)
        self.assertGreater(float(penalty.detach()), 0.1)
        self.assertLess(float(balanced.detach()), 1e-10)
        torch.testing.assert_close(penalty, globally_scaled)
        penalty.backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(float(latents.grad.abs().sum()), 0.0)

    def test_marginal_kurtosis_penalty_is_finite_and_has_gradients(self):
        torch.manual_seed(3)
        latents = torch.randn(64, 4, 3).requires_grad_()
        penalty = marginal_kurtosis_penalty(latents)
        self.assertTrue(torch.isfinite(penalty))
        penalty.backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(float(latents.grad.abs().sum()), 0.0)

    def test_shapes_determinism_and_full_prefix_equivalence(self):
        torch.manual_seed(3)
        model = ProgressiveTokenizer(tiny_config()).eval()
        images = torch.randn(2, 3, 8, 8)
        first = model(images)
        second = model(images)
        self.assertEqual(first["latents"].shape, (2, 4, 8))
        self.assertEqual(first["reconstruction"].shape, images.shape)
        self.assertTrue(torch.equal(first["latents"], second["latents"]))
        self.assertTrue(
            torch.equal(first["reconstruction"], second["reconstruction"])
        )
        explicit_full = model.decode(first["latents"], prefix_lengths=4)
        self.assertTrue(torch.equal(first["reconstruction"], explicit_full))

    def test_masked_tail_cannot_affect_prefix_reconstruction(self):
        torch.manual_seed(4)
        model = ProgressiveTokenizer(tiny_config()).eval()
        latents = torch.randn(2, 4, 8)
        perturbed = latents.clone()
        perturbed[:, 2:] = torch.randn_like(perturbed[:, 2:]) * 1000
        baseline = model.decode(latents, prefix_lengths=2)
        changed = model.decode(perturbed, prefix_lengths=2)
        torch.testing.assert_close(baseline, changed, rtol=0, atol=0)

    def test_per_example_prefix_mask_and_validation(self):
        model = ProgressiveTokenizer(tiny_config()).eval()
        latents = torch.randn(2, 4, 8)
        output = model.decode(latents, prefix_lengths=torch.tensor([1, 3]))
        self.assertEqual(output.shape, (2, 3, 8, 8))
        with self.assertRaises(ValueError):
            model.decode(latents, prefix_lengths=0)
        with self.assertRaises(ValueError):
            model.decode(latents, prefix_lengths=torch.tensor([1, 5]))

    def test_gradients_reach_pooling_and_decoder_cross_attention(self):
        torch.manual_seed(5)
        model = ProgressiveTokenizer(tiny_config())
        images = torch.randn(2, 3, 8, 8)
        output = model(images, prefix_lengths=torch.tensor([2, 3]))
        F.mse_loss(output["reconstruction"], images).backward()
        self.assertIsNotNone(model.pool_queries.grad)
        self.assertGreater(float(model.pool_queries.grad.abs().sum()), 0)
        cross_weight = model.decoder_blocks[0].cross_attention.kv.weight
        self.assertIsNotNone(cross_weight.grad)
        self.assertGreater(float(cross_weight.grad.abs().sum()), 0)
        self.assertIsNotNone(model.patch_embed.weight.grad)

    def test_weight_decay_excludes_identity_and_qk_norm_parameters(self):
        model = ProgressiveTokenizer(tiny_config())
        groups = optimizer_parameter_groups(model, weight_decay=0.05)
        decayed = {id(parameter) for parameter in groups[0]["params"]}
        protected = {id(parameter) for parameter in groups[1]["params"]}
        self.assertIn(id(model.patch_embed.weight), decayed)
        self.assertIn(id(model.encoder_blocks[0].attention.qkv.weight), decayed)
        self.assertIn(id(model.encoder_position), protected)
        self.assertIn(id(model.pool_queries), protected)
        self.assertIn(
            id(model.encoder_blocks[0].attention.query_norm.weight), protected
        )
        self.assertTrue(decayed.isdisjoint(protected))
        self.assertEqual(
            len(decayed) + len(protected),
            sum(1 for parameter in model.parameters() if parameter.requires_grad),
        )

    def test_cross_only_pool_exports_attended_values_and_trains(self):
        config = tiny_config()
        config = TokenizerConfig(
            **{
                **config.fingerprint(),
                "pool_type": "cross_only",
                "pool_depth": 1,
            }
        )
        model = ProgressiveTokenizer(config)
        images = torch.randn(2, 3, 8, 8)
        output = model(images, prefix_lengths=torch.tensor([2, 3]))
        self.assertEqual(output["latents"].shape, (2, 4, 8))
        F.mse_loss(output["reconstruction"], images).backward()
        self.assertIsNotNone(model.pool_attention)
        self.assertIsNotNone(model.pool_attention.kv.weight.grad)
        self.assertGreater(float(model.pool_attention.kv.weight.grad.abs().sum()), 0)

    def test_register_tokens_join_patch_sequence_and_train(self):
        config = TokenizerConfig(
            **{
                **tiny_config().fingerprint(),
                "pool_type": "register_tokens",
                "pool_depth": 1,
            }
        )
        model = ProgressiveTokenizer(config)
        images = torch.randn(2, 3, 8, 8)
        output = model(images, prefix_lengths=torch.tensor([2, 3]))
        self.assertEqual(output["latents"].shape, (2, 4, 8))
        F.mse_loss(output["reconstruction"], images).backward()
        self.assertIsNone(model.pool_attention)
        self.assertIsNotNone(model.register_joint_block)
        self.assertIsNotNone(model.register_adapter)
        self.assertGreater(
            float(model.register_joint_block.attention.qkv.weight.grad.abs().sum()),
            0,
        )
        self.assertGreater(float(model.register_adapter.input.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.pool_queries.grad.abs().sum()), 0)

    def test_stage_a_pooling_arms_are_parameter_exact(self):
        base = tiny_config().fingerprint()
        cross = ProgressiveTokenizer(
            TokenizerConfig(
                **{
                    **base,
                    "encoder_depth": 2,
                    "pool_type": "cross_only",
                    "pool_depth": 1,
                }
            )
        )
        residual = ProgressiveTokenizer(
            TokenizerConfig(
                **{
                    **base,
                    "encoder_depth": 1,
                    "pool_type": "residual",
                    "pool_depth": 1,
                }
            )
        )
        registers = ProgressiveTokenizer(
            TokenizerConfig(
                **{
                    **base,
                    "encoder_depth": 1,
                    "pool_type": "register_tokens",
                    "pool_depth": 1,
                }
            )
        )
        counts = {
            sum(parameter.numel() for parameter in model.parameters())
            for model in (cross, residual, registers)
        }
        self.assertEqual(len(counts), 1)

    def test_encoder_patch_size_is_decoupled_from_decoder_grid(self):
        config = TokenizerConfig(
            **{
                **tiny_config().fingerprint(),
                "encoder_patch_size": 2,
                "encoder_stem": "patch",
            }
        )
        model = ProgressiveTokenizer(config).eval()
        images = torch.randn(2, 3, 8, 8)
        output = model(images)
        self.assertEqual(model.encoder_position.shape[1], 16)
        self.assertEqual(model.output_position.shape[1], 4)
        self.assertEqual(output["latents"].shape, (2, 4, 8))
        self.assertEqual(output["reconstruction"].shape, images.shape)

    def test_fine_conv_stem_returns_historical_transformer_grid(self):
        config = TokenizerConfig(
            **{
                **tiny_config().fingerprint(),
                "encoder_patch_size": 2,
                "encoder_stem": "fine_conv",
            }
        )
        model = ProgressiveTokenizer(config)
        images = torch.randn(2, 3, 8, 8)
        output = model(images)
        self.assertEqual(model.encoder_position.shape[1], 4)
        self.assertEqual(model.output_position.shape[1], 4)
        F.mse_loss(output["reconstruction"], images).backward()
        first_conv = model.patch_embed[0]
        depthwise_conv = model.patch_embed[2]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertIsNotNone(depthwise_conv.weight.grad)

    def test_variational_sampling_and_deterministic_eval_encode(self):
        torch.manual_seed(6)
        config = TokenizerConfig(**{**tiny_config().fingerprint(), "variational": True})
        model = ProgressiveTokenizer(config)
        images = torch.randn(2, 3, 8, 8)
        model.train()
        first = model(images)
        second = model(images)
        self.assertEqual(first["latents"].shape, (2, 4, 8))
        self.assertEqual(first["mean"].shape, (2, 4, 8))
        self.assertEqual(first["log_variance"].shape, (2, 4, 8))
        self.assertFalse(torch.equal(first["latents"], second["latents"]))
        self.assertTrue(torch.equal(first["mean"], second["mean"]))
        model.eval()
        with torch.no_grad():
            encoded = model.encode(images)
            evaluated = model(images)
        self.assertTrue(torch.equal(encoded, evaluated["latents"]))
        self.assertTrue(torch.equal(encoded, evaluated["mean"]))

    def test_soft_log_variance_bound_stays_differentiable(self):
        config = TokenizerConfig(
            **{**tiny_config().fingerprint(), "variational": True}
        )
        model = ProgressiveTokenizer(config)
        raw = torch.tensor([-20.0, -8.0, 0.0, 20.0], requires_grad=True)
        bounded = model._bound_log_variance(raw)
        self.assertTrue(torch.all(bounded >= config.log_variance_floor))
        self.assertTrue(torch.all(bounded <= -config.log_variance_floor))
        bounded.sum().backward()
        self.assertTrue(torch.all(raw.grad > 0))

        hard_config = TokenizerConfig(
            **{
                **config.fingerprint(),
                "hard_log_variance_clamp": True,
            }
        )
        hard_model = ProgressiveTokenizer(hard_config)
        hard_raw = torch.tensor([-20.0], requires_grad=True)
        hard_model._bound_log_variance(hard_raw).sum().backward()
        self.assertEqual(float(hard_raw.grad), 0.0)

    def test_log_variance_floor_must_be_negative(self):
        with self.assertRaises(ValueError):
            TokenizerConfig(
                **{**tiny_config().fingerprint(), "log_variance_floor": 0.0}
            )

    def test_latent_noise_modes_zero_scale_identity_and_validation(self):
        torch.manual_seed(7)
        model = ProgressiveTokenizer(tiny_config()).eval()
        images = torch.randn(2, 3, 8, 8)
        clean = model(images)
        mix_clean = model(
            images, noise_mode="mix", noise_scales=torch.ones(2, 4)
        )
        torch.testing.assert_close(
            clean["reconstruction"], mix_clean["reconstruction"], rtol=0, atol=0
        )
        add_clean = model(
            images, noise_mode="add", noise_scales=torch.zeros(2, 4)
        )
        torch.testing.assert_close(
            clean["reconstruction"], add_clean["reconstruction"], rtol=0, atol=0
        )
        noisy = model(
            images,
            noise_mode="mix",
            noise_scales=torch.zeros(2, 4),
            include_full_reconstruction=True,
        )
        self.assertFalse(
            torch.equal(clean["reconstruction"], noisy["reconstruction"])
        )
        torch.testing.assert_close(
            clean["reconstruction"], noisy["full_reconstruction"], rtol=0, atol=0
        )
        self.assertTrue(torch.equal(noisy["latents"], clean["latents"]))
        with self.assertRaises(ValueError):
            model(images, noise_mode="mix", noise_scales=torch.ones(2, 3))
        with self.assertRaises(ValueError):
            model(images, noise_mode="mix")

    def test_rms_qk_scales_start_at_one(self):
        model = ProgressiveTokenizer(tiny_config())
        attention = model.encoder_blocks[0].attention
        torch.testing.assert_close(
            attention.query_norm.weight, torch.ones_like(attention.query_norm.weight)
        )
        torch.testing.assert_close(
            attention.key_norm.weight, torch.ones_like(attention.key_norm.weight)
        )
        self.assertIsNone(attention.logit_scale)


class TestRotaryPrecision(unittest.TestCase):
    def test_tables_are_float32_and_bf16_use_is_finite(self):
        rope = Rotary2D(grid_size=2, head_dim=8)
        self.assertEqual(rope.cos.dtype, torch.float32)
        self.assertEqual(rope.sin.dtype, torch.float32)
        values = torch.randn(2, 4, 4, 8, dtype=torch.bfloat16)
        rotated = rope.rotate(values)
        self.assertEqual(rotated.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(rotated).all())


if __name__ == "__main__":
    unittest.main()
