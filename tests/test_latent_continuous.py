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
from model_ring_latent_continuous import (  # noqa: E402
    RingLatentContinuousConfig,
    RingLatentContinuousModel,
)
from model_joint_latent_diffusion import (  # noqa: E402
    JointLatentDiffusionConfig,
    JointLatentDiffusionModel,
)
from train_latent_continuous import (  # noqa: E402
    load_latent_checkpoint,
    save_latent_checkpoint,
)
from train_joint_latent_diffusion import build_optimizer_parameters  # noqa: E402
from train_ring_latent_continuous import (  # noqa: E402
    load_checkpoint as load_ring_checkpoint,
    save_checkpoint as save_ring_checkpoint,
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


def _tiny_ring_model(adapter: FrozenLatentAutoencoder) -> RingLatentContinuousModel:
    return RingLatentContinuousModel(
        adapter.autoencoder.layout.latent_parent,
        RingLatentContinuousConfig(
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=2,
                num_heads=4,
                ff_mult=2,
                max_seq_len=23,
                qk_norm=True,
            ),
            diffusion=DiffusionDecoderConfig(
                target_dim=4 * 64,
                z_channels=32,
                target_condition_dim=0,
                condition_fusion="add",
                width=32,
                depth=2,
                objective="flow",
                prediction_type="v_prediction",
                component_reduction="fixed_dim",
                num_train_timesteps=20,
                num_inference_steps=1,
                diffusion_batch_mul=1,
            ),
        ),
    )


def _write_fake_interface(
    directory: str,
    *,
    target_tokens_per_latent: int = 12,
    max_ring_latents: int = 8,
    latent_dim: int = 64,
) -> tuple[str, str]:
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
        latent_dim=latent_dim,
        model_width=32,
        perceiver_width=32,
        perceiver_heads=4,
        ring_transformer_layers=1,
        pooler="perceiver_sector",
        target_tokens_per_latent=target_tokens_per_latent,
        max_ring_latents=max_ring_latents,
    )
    metadata = codec.position_metadata()
    metadata["empirical_scale"] = codec.orbit_scale_for_policy(
        codec.effective_scale_policy()
    ).mean(dim=-1)
    autoencoder = CausalFrequencyAutoencoder(config, metadata, codec.component_mask)
    if target_tokens_per_latent == 12 and autoencoder.exported_token_count != 53:
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
            "latent_mean": torch.randn(autoencoder.exported_token_count, latent_dim) * 0.1,
            "latent_std": torch.rand(autoencoder.exported_token_count, latent_dim) + 0.5,
            "probe_validation_mse": 0.9,
            "zero_baseline_mse": 1.0,
        },
        interface,
    )
    return checkpoint, interface


class TestLatentContinuous(unittest.TestCase):
    def test_latent_normalization_scopes(self):
        from fit_autoencoder_latent_interface import _normalization_stats

        generator = torch.Generator().manual_seed(17)
        values = torch.randn(64, 5, 3, generator=generator)
        values = values * torch.tensor([0.5, 1.0, 2.0])[None, None]
        values = values + torch.arange(5).float()[None, :, None]

        position_mean, position_std = _normalization_stats(
            values, "position_channel"
        )
        self.assertEqual(tuple(position_mean.shape), (5, 3))
        self.assertTrue(
            torch.allclose(
                ((values - position_mean) / position_std).mean(dim=0),
                torch.zeros(5, 3),
                atol=1e-5,
            )
        )

        channel_mean, channel_std = _normalization_stats(values, "channel")
        self.assertTrue(torch.equal(channel_mean[0], channel_mean[-1]))
        channel_normalized = (values - channel_mean) / channel_std
        self.assertTrue(
            torch.allclose(
                channel_normalized.mean(dim=(0, 1)), torch.zeros(3), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                channel_normalized.std(dim=(0, 1)), torch.ones(3), atol=1e-5
            )
        )

        tensor_mean, tensor_std = _normalization_stats(values, "tensor")
        self.assertEqual(torch.unique(tensor_mean).numel(), 1)
        self.assertEqual(torch.unique(tensor_std).numel(), 1)
        tensor_normalized = (values - tensor_mean) / tensor_std
        self.assertAlmostEqual(float(tensor_normalized.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(tensor_normalized.std()), 1.0, places=5)

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

    def test_dynamic_target4_interface(self):
        directory = os.path.join(self.temporary.name, "target4")
        os.makedirs(directory)
        checkpoint, interface = _write_fake_interface(
            directory,
            target_tokens_per_latent=4,
            max_ring_latents=16,
            latent_dim=16,
        )
        adapter = FrozenLatentAutoencoder(checkpoint, interface)
        self.assertEqual(adapter.sequence_length, 134)
        self.assertEqual(adapter.token_dim, 16)
        self.assertEqual(adapter.position_features.shape, (134, 11))
        latents = adapter.encode_images(torch.rand(1, 3, 32, 32))
        self.assertEqual(latents.shape, (1, 134, 16))
        self.assertEqual(adapter.decode_latents(latents).shape, (1, 3, 32, 32))

    def test_dynamic_token_grouping_model(self):
        parent = torch.tensor([0, 0, 1, 2, 2, 2])
        model = RingLatentContinuousModel(
            parent,
            RingLatentContinuousConfig(
                latent_sequence_length=6,
                latent_dim=8,
                ring_sequence_length=6,
                max_ring_latents=1,
                grouping="token",
                transformer=CausalTransformerConfig(
                    width=32,
                    num_layers=1,
                    num_heads=4,
                    ff_mult=2,
                    max_seq_len=6,
                    qk_norm=True,
                ),
                diffusion=DiffusionDecoderConfig(
                    target_dim=8,
                    z_channels=32,
                    target_condition_dim=0,
                    condition_fusion="add",
                    width=32,
                    depth=2,
                    objective="flow",
                    prediction_type="v_prediction",
                    component_reduction="fixed_dim",
                    num_train_timesteps=20,
                    num_inference_steps=1,
                    diffusion_batch_mul=1,
                ),
            ),
        )
        latents = torch.randn(2, 6, 8)
        self.assertTrue(torch.equal(model.unpack_rings(model.pack_rings(latents)), latents))
        output = model(latents)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        generated = model.generate_latents(
            1, num_inference_steps=1, generator=torch.Generator().manual_seed(5)
        )
        self.assertEqual(generated.shape, (1, 6, 8))
        path = os.path.join(self.temporary.name, "dynamic_token_latent.pt")
        save_ring_checkpoint(path, model, self.adapter, 23)
        # This adapter has a different latent layout, so exercise the exact
        # state-dict buffer copy directly; contract validation is covered by the
        # matched-layout checkpoint test below.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reloaded = RingLatentContinuousModel(parent, model.config)
        reloaded.load_state_dict(payload["model"])
        self.assertTrue(
            torch.equal(reloaded.ring_component_mask, model.ring_component_mask)
        )

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

    def test_ring_pack_roundtrip_mask_and_shift(self):
        model = _tiny_ring_model(self.adapter)
        latents = torch.randn(2, 53, 64)
        rings = model.pack_rings(latents)
        self.assertEqual(rings.shape, (2, 23, 256))
        self.assertTrue(torch.equal(model.unpack_rings(rings), latents))
        self.assertEqual(model.ring_counts.tolist(), [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 2, 1, 1, 1])
        self.assertTrue(torch.all(rings[~model.ring_component_mask[None].expand(2, -1, -1)] == 0))
        shifted = model.shifted_inputs_from_rings(rings)
        expected_zero = torch.zeros_like(rings)
        expected_zero[:, 1:] = rings[:, :-1]
        bos = torch.zeros(2, 23, 1)
        bos[:, 0] = 1
        projected = model.input_projection(torch.cat([expected_zero, bos], dim=-1))
        projected = projected + model.target_slot.weight[None]
        self.assertTrue(torch.allclose(shifted, projected))

    def test_ring_model_causality_cache_gradients_and_sampling(self):
        model = _tiny_ring_model(self.adapter).eval()
        latents = torch.randn(1, 53, 64)
        rings = model.pack_rings(latents)
        with torch.no_grad():
            full, _ = model.forward_backbone(model.shifted_inputs_from_rings(rings))
            changed = rings.clone()
            changed[:, 8:] = torch.randn_like(changed[:, 8:])
            changed_full, _ = model.forward_backbone(
                model.shifted_inputs_from_rings(changed)
            )
            self.assertTrue(torch.allclose(full[:, :9], changed_full[:, :9], atol=1e-5))

            cached, caches = model.init_cache(1)
            self.assertTrue(torch.allclose(cached, full[:, 0], atol=1e-5, rtol=1e-5))
            for target_ring in range(1, 8):
                cached, caches = model.forward_step(
                    rings[:, target_ring - 1], target_ring, caches
                )
                self.assertTrue(
                    torch.allclose(
                        cached, full[:, target_ring], atol=1e-5, rtol=1e-5
                    )
                )

        self.assertEqual(model.rope_cos.dtype, torch.float32)
        self.assertIsNotNone(model.layers[0].attn.q_norm)
        first = model.generate_latents(
            1, num_inference_steps=1, generator=torch.Generator().manual_seed(17)
        )
        second = model.generate_latents(
            1, num_inference_steps=1, generator=torch.Generator().manual_seed(17)
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (1, 53, 64))

        model.train()
        output = model(torch.randn(2, 53, 64))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertIsNotNone(model.target_slot.weight.grad)
        self.assertIsNotNone(model.layers[0].attn.q_norm.weight.grad)
        self.assertIsNotNone(model.diffusion.net.input_proj.weight.grad)

    def test_ring_checkpoint_contract_and_roundtrip(self):
        model = _tiny_ring_model(self.adapter)
        path = os.path.join(self.temporary.name, "ring_latent.pt")
        save_ring_checkpoint(path, model, self.adapter, 19)
        loaded, step = load_ring_checkpoint(path, self.adapter)
        self.assertEqual(step, 19)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))

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

    def test_joint_diffusion_supports_32d_latents(self):
        config = JointLatentDiffusionConfig(
            sequence_length=53,
            token_dim=32,
            metadata_dim=self.adapter.position_features.shape[-1],
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=1,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
            ),
            num_train_timesteps=20,
            num_inference_steps=1,
        )
        model = JointLatentDiffusionModel(config)
        output = model(
            torch.randn(2, 53, 32), self.adapter.position_features
        )
        output["loss"].backward()
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIsNotNone(model.input_projection.weight.grad)
        self.assertEqual(model.final_layer.linear.out_features, 32)
        generated = model.eval().generate_latents(
            1,
            self.adapter.position_features,
            generator=torch.Generator().manual_seed(5),
        )
        self.assertEqual(generated.shape, (1, 53, 32))

    def test_modern_joint_blocks_are_affine_free_qknorm_adaln_zero(self):
        config = JointLatentDiffusionConfig(
            metadata_dim=11,
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=1,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
                qk_norm=True,
            ),
            num_train_timesteps=20,
            num_inference_steps=2,
            position_embedding_film=True,
            rope="radius_angle",
            block_conditioning="adaln_zero",
        )
        model = JointLatentDiffusionModel(config)
        block = model.layers[0]
        self.assertFalse(block.attn.norm.elementwise_affine)
        self.assertFalse(block.ff.norm.elementwise_affine)
        self.assertIsInstance(block.attn.q_norm, torch.nn.LayerNorm)
        self.assertFalse(block.attn.q_norm.elementwise_affine)
        self.assertIsNotNone(block.adaln)

        states = torch.randn(2, 53, 32)
        condition = torch.randn_like(states)
        rope = model._rope_tables(self.adapter.position_features)
        transformed, _ = block(states, condition, rope=rope)
        self.assertTrue(torch.equal(states, transformed))

        output = model(torch.randn(2, 53, 64), self.adapter.position_features)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        self.assertIsNotNone(model.final_layer.linear.weight.grad)

    def test_joint_matrix_only_weight_decay_partition(self):
        config = JointLatentDiffusionConfig(
            metadata_dim=11,
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=1,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
                qk_norm=True,
            ),
            position_embedding_film=True,
            rope="radius_angle",
            block_conditioning="adaln_zero",
        )
        model = JointLatentDiffusionModel(config)
        groups, report = build_optimizer_parameters(model, 0.02, "matrix_only")
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        grouped_names = [
            parameter_names[id(parameter)]
            for group in groups
            for parameter in group["params"]
        ]

        self.assertEqual(len(grouped_names), len(set(grouped_names)))
        self.assertEqual(set(grouped_names), set(parameter_names.values()))
        self.assertEqual(groups[0]["weight_decay"], 0.02)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertIn("position_embedding_film", report["no_decay_names"])
        self.assertIn("layers.0.adaln.net.1.bias", report["no_decay_names"])
        self.assertIn("layers.0.adaln.net.1.weight", report["decay_names"])
        self.assertTrue(
            all(
                not name.endswith(".bias")
                for name in report["decay_names"]
            )
        )

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
