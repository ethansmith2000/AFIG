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
from progressive_tokenizer.training import optimizer_parameter_groups  # noqa: E402


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

    def test_weight_decay_excludes_identity_and_temperature_parameters(self):
        model = ProgressiveTokenizer(tiny_config())
        groups = optimizer_parameter_groups(model, weight_decay=0.05)
        decayed = {id(parameter) for parameter in groups[0]["params"]}
        protected = {id(parameter) for parameter in groups[1]["params"]}
        self.assertIn(id(model.patch_embed.weight), decayed)
        self.assertIn(id(model.encoder_blocks[0].attention.qkv.weight), decayed)
        self.assertIn(id(model.encoder_position), protected)
        self.assertIn(id(model.pool_queries), protected)
        self.assertIn(
            id(model.encoder_blocks[0].attention.logit_scale), protected
        )
        self.assertTrue(decayed.isdisjoint(protected))
        self.assertEqual(
            len(decayed) + len(protected),
            sum(1 for parameter in model.parameters() if parameter.requires_grad),
        )


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

