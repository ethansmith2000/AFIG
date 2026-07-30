"""Tests for the frozen target-12 latent AFIG path."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoencoder_models import AutoencoderConfig, CausalFrequencyAutoencoder  # noqa: E402
from causal_transformer import CausalTransformerConfig  # noqa: E402
from diffusion_decoder import DiffusionDecoderConfig  # noqa: E402
from frequency import FrequencyCodec, FrequencyCodecConfig  # noqa: E402
from latent_autoencoder_interface import (  # noqa: E402
    FrozenLatentAutoencoder,
    POSITION_FEATURE_SCHEMA,
)
from model_latent_continuous import (  # noqa: E402
    LatentContinuousConfig,
    LatentContinuousModel,
)
from model_joint_latent_diffusion import (  # noqa: E402
    JointLatentDiffusionConfig,
    JointLatentDiffusionModel,
)
from train_latent_continuous import (  # noqa: E402
    load_latent_checkpoint,
    save_latent_checkpoint,
)


def _tiny_model(
    metadata_dim: int = 11,
    latent_loss_weighting: str = "unweighted",
    loss_component_weights: torch.Tensor | None = None,
) -> LatentContinuousModel:
    return LatentContinuousModel(
        LatentContinuousConfig(
            metadata_dim=metadata_dim,
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=2,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
            ),
            diffusion=DiffusionDecoderConfig(
                target_dim=64,
                z_channels=32,
                target_condition_dim=metadata_dim,
                condition_fusion="concat_mlp",
                width=32,
                depth=2,
                num_train_timesteps=20,
                num_inference_steps=1,
                diffusion_batch_mul=1,
                loss_metric=(
                    "normalized"
                    if latent_loss_weighting == "unweighted"
                    else "component_weighted"
                ),
            ),
            latent_loss_weighting=latent_loss_weighting,
        ),
        loss_component_weights=loss_component_weights,
    )


def _write_fake_interface(directory: str) -> tuple[str, str]:
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            ordering="radial",
            normalization="orbit_standardize",
            value_transform="identity",
        )
    )

    class Loader:
        def __iter__(self):
            generator = torch.Generator().manual_seed(7)
            for _ in range(2):
                yield torch.rand(2, 3, 32, 32, generator=generator)

    codec.fit_from_loader(Loader())
    config = AutoencoderConfig(
        mode="causal_ring",
        latent_dim=64,
        model_width=32,
        perceiver_width=32,
        perceiver_heads=4,
        ring_transformer_layers=1,
        pooler="perceiver_sector",
        target_tokens_per_latent=12,
        max_ring_latents=8,
    )
    metadata = codec.position_metadata()
    metadata["empirical_scale"] = codec.orbit_scale_for_policy(
        codec.effective_scale_policy()
    ).mean(dim=-1)
    autoencoder = CausalFrequencyAutoencoder(config, metadata, codec.component_mask)
    if autoencoder.exported_token_count != 53:
        raise AssertionError("Fixture no longer produces the target-12 53-token layout")
    checkpoint = os.path.join(directory, "ae.pt")
    torch.save(
        {
            "version": 1,
            "global_step": 1,
            "config": config.fingerprint(),
            "model": autoencoder.state_dict(),
            "codec": codec.export_state(),
        },
        checkpoint,
    )
    interface = os.path.join(directory, "latent_interface.pt")
    torch.save(
        {
            "version": 1,
            "checkpoint": os.path.abspath(checkpoint),
            "latent_mean": torch.randn(53, 64) * 0.1,
            "latent_std": torch.rand(53, 64) + 0.5,
            "probe_validation_mse": 0.9,
            "zero_baseline_mse": 1.0,
        },
        interface,
    )
    return checkpoint, interface


class TestLatentContinuous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.checkpoint, cls.interface = _write_fake_interface(cls.temporary.name)
        cls.adapter = FrozenLatentAutoencoder(cls.checkpoint, cls.interface)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_normalization_shape_metadata_and_frozen_decode(self):
        adapter = self.adapter
        raw = torch.randn(2, 53, 64)
        self.assertTrue(
            torch.allclose(adapter.denormalize(adapter.normalize(raw)), raw, atol=1e-6)
        )
        self.assertEqual(adapter.position_features.shape, (53, len(POSITION_FEATURE_SCHEMA)))
        self.assertTrue(torch.isfinite(adapter.position_features).all())
        self.assertEqual(adapter.position_features[0, 0].item(), 0.0)
        self.assertEqual(adapter.position_features[-1, 0].item(), 1.0)
        images = adapter.decode_latents(torch.zeros(1, 53, 64))
        self.assertEqual(images.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(images).all())
        self.assertFalse(any(parameter.requires_grad for parameter in adapter.autoencoder.parameters()))

    def test_target_aligned_shift_and_context_dropout(self):
        model = _tiny_model()
        tokens = torch.randn(2, 53, 64)
        metadata = self.adapter.position_features
        features = model.shifted_features(tokens, metadata)
        self.assertTrue(torch.equal(features[:, 0, :64], torch.zeros_like(tokens[:, 0])))
        self.assertTrue(torch.equal(features[:, 1:, :64], tokens[:, :-1]))
        self.assertTrue(torch.equal(features[0, :, 64:-1], metadata))
        self.assertTrue(torch.all(features[:, 0, -1] == 1))
        self.assertTrue(torch.all(features[:, 1:, -1] == 0))
        hidden = torch.randn(2, 53, 32)
        replaced, dropped = model.apply_context_dropout(
            hidden, torch.ones(2, 53, dtype=torch.bool)
        )
        expected = model.null_context.view(1, 1, -1).expand_as(hidden)
        self.assertTrue(torch.equal(replaced, expected))
        self.assertTrue(dropped.all())

    def test_component_weighted_loss_has_gradients(self):
        weights = torch.linspace(0.1, 2.0, 53 * 64).reshape(53, 64)
        model = _tiny_model(
            latent_loss_weighting="raw_variance",
            loss_component_weights=weights,
        )
        tokens = torch.randn(2, 53, 64)
        output = model(tokens, self.adapter.position_features)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertAlmostEqual(
            float(model.loss_component_weights.mean()), 1.0, places=6
        )

    def test_joint_diffusion_is_bidirectional_and_has_gradients(self):
        config = JointLatentDiffusionConfig(
            metadata_dim=self.adapter.position_features.shape[-1],
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=2,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
            ),
            num_train_timesteps=20,
            num_inference_steps=2,
        )
        model = JointLatentDiffusionModel(config).eval()
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
        noisy = torch.randn(1, 53, 64)
        changed = noisy.clone()
        changed[:, -1] += 1.0
        flow_time = torch.full((1,), 0.5)
        prediction = model.predict_velocity(
            noisy, flow_time, self.adapter.position_features
        )
        changed_prediction = model.predict_velocity(
            changed, flow_time, self.adapter.position_features
        )
        self.assertFalse(
            torch.allclose(prediction[:, 0], changed_prediction[:, 0])
        )
        model.train()
        output = model(torch.randn(2, 53, 64), self.adapter.position_features)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIsNotNone(model.input_projection.weight.grad)

    def test_joint_diffusion_sampling_is_deterministic(self):
        model = JointLatentDiffusionModel(
            JointLatentDiffusionConfig(
                metadata_dim=self.adapter.position_features.shape[-1],
                transformer=CausalTransformerConfig(
                    width=32,
                    num_layers=1,
                    num_heads=4,
                    ff_mult=2,
                    max_seq_len=53,
                ),
                num_train_timesteps=20,
                num_inference_steps=2,
            )
        ).eval()
        first = model.generate_latents(
            1,
            self.adapter.position_features,
            generator=torch.Generator().manual_seed(4),
        )
        second = model.generate_latents(
            1,
            self.adapter.position_features,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (1, 53, 64))

    def test_causality_and_kv_cache_parity(self):
        model = _tiny_model().eval()
        tokens = torch.randn(1, 53, 64)
        metadata = self.adapter.position_features
        with torch.no_grad():
            full, _ = model.forward_backbone(model.shifted_inputs(tokens, metadata), metadata)
            changed = tokens.clone()
            changed[:, 20:] = torch.randn_like(changed[:, 20:])
            changed_full, _ = model.forward_backbone(
                model.shifted_inputs(changed, metadata), metadata
            )
            self.assertTrue(torch.allclose(full[:, :21], changed_full[:, :21], atol=1e-5))

            cached, caches = model.init_cache(1, metadata)
            self.assertTrue(torch.allclose(cached, full[:, 0], atol=1e-5))
            for target_index in range(1, 10):
                cached, caches = model.forward_step(
                    tokens[:, target_index - 1],
                    target_index,
                    metadata[target_index],
                    caches,
                )
                self.assertTrue(
                    torch.allclose(cached, full[:, target_index], atol=1e-5, rtol=1e-5)
                )

    def test_cfg_scale_one_equivalence(self):
        model = _tiny_model().eval()
        condition = torch.randn(3, 32)
        metadata = self.adapter.position_features[:1].expand(3, -1)
        generator_a = torch.Generator().manual_seed(91)
        generator_b = torch.Generator().manual_seed(91)
        plain = model.diffusion.sample(
            condition,
            target_condition=metadata,
            num_inference_steps=1,
            generator=generator_a,
        )
        guided = model.diffusion.sample(
            condition,
            target_condition=metadata,
            unconditional_z=model.null_context[None].expand_as(condition),
            cfg_scale=1.0,
            num_inference_steps=1,
            generator=generator_b,
        )
        self.assertTrue(torch.equal(plain, guided))
        torch.nn.init.normal_(model.diffusion.net.final_layer.linear.weight, std=0.02)
        noisy = torch.randn(3, 64)
        timesteps = torch.full((3,), 10, dtype=torch.long)
        matched = model.diffusion._guided_prediction(
            noisy,
            timesteps,
            condition,
            metadata,
            model.null_context[None].expand_as(condition),
            cfg_scale=2.0,
            cfg_norm_match=True,
        )
        conditional = model.diffusion.net(
            noisy, timesteps, condition, target_condition=metadata
        )
        self.assertTrue(
            torch.allclose(
                matched.float().norm(dim=-1),
                conditional.float().norm(dim=-1),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        generated = model.generate_latents(
            1,
            self.adapter.position_features,
            cfg_scale=1.0,
            num_inference_steps=1,
            generator=torch.Generator().manual_seed(19),
        )
        self.assertEqual(generated.shape, (1, 53, 64))
        decoded = self.adapter.decode_latents(generated)
        self.assertEqual(decoded.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(decoded).all())

    def test_checkpoint_contract_and_model_roundtrip(self):
        model = _tiny_model()
        path = os.path.join(self.temporary.name, "latent.pt")
        save_latent_checkpoint(path, model, self.adapter, 12)
        loaded, step = load_latent_checkpoint(path, self.adapter)
        self.assertEqual(step, 12)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
        contract = self.adapter.checkpoint_contract()
        contract["sequence_length"] = 54
        with self.assertRaises(ValueError):
            self.adapter.assert_contract_compatible(contract)

    def test_forward_gradients(self):
        model = _tiny_model()
        output = model(
            torch.randn(2, 53, 64),
            self.adapter.position_features,
            context_dropout_mask=torch.zeros(2, 53, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.layers[0].attn.qkv.weight.grad)
        self.assertIsNotNone(model.diffusion.net.input_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
