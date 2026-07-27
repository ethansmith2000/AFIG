"""Tests for ContinuousFFTDecoder."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_decoder import DiffusionDecoderConfig  # noqa: E402
from frequency import FrequencyCodec, FrequencyCodecConfig  # noqa: E402
from model_continuous import (  # noqa: E402
    ContinuousFFTDecoder,
    ContinuousModelConfig,
    CorruptionConfig,
    FrequencyConditioningConfig,
    GenerationConfig,
    HistoryFeatureConfig,
    PolarHistoryConfig,
    TransformerConfig,
)
import torch.nn as nn  # noqa: E402


def _tiny_cfg(**kwargs) -> ContinuousModelConfig:
    return ContinuousModelConfig(
        codec=FrequencyCodecConfig(),
        transformer=TransformerConfig(
            width=64,
            num_layers=2,
            num_heads=4,
            ff_mult=2,
            max_seq_len=515,
        ),
        diffusion=DiffusionDecoderConfig(
            target_dim=6,
            z_channels=64,
            width=64,
            depth=2,
            num_train_timesteps=50,
            num_inference_steps=2,
            diffusion_batch_mul=1,
        ),
        **kwargs,
    )


def _fitted_codec():
    codec = FrequencyCodec(FrequencyCodecConfig())

    class L:
        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            for _ in range(4):
                yield torch.rand(4, 3, 32, 32, generator=g)

    codec.fit_from_loader(L())
    return codec


class TestContinuousModel(unittest.TestCase):
    def test_teacher_forcing_alignment_and_grad(self):
        codec = _fitted_codec()
        model = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        tokens = torch.randn(2, codec.seq_len, 6)
        tokens = tokens * codec.component_mask[None, :, :]
        out = model(tokens, corrupt=False)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.token_proj.weight.grad)
        self.assertIsNotNone(model.diffusion.net.input_proj.weight.grad)
        # Grad reaches an early transformer layer.
        self.assertIsNotNone(model.layers[0].attn.qkv.weight.grad)

    def test_physical_phase_auxiliary_is_finite_and_differentiable(self):
        codec_config = FrequencyCodecConfig(
            normalization="orbit_standardize",
            value_transform="identity",
        )
        codec = FrequencyCodec(codec_config)

        class Loader:
            def __iter__(self):
                generator = torch.Generator().manual_seed(41)
                for _ in range(6):
                    yield torch.rand(4, 3, 32, 32, generator=generator)

        codec.fit_from_loader(Loader())
        config = _tiny_cfg()
        config.codec = codec_config
        diff = config.diffusion.fingerprint()
        diff.update(
            prediction_type="x0",
            loss_space="native",
            phase_aux_weight=0.05,
            phase_aux_gate=0.1,
        )
        config.diffusion = DiffusionDecoderConfig(**diff)
        model = ContinuousFFTDecoder(config, codec=codec)
        tokens = codec.encode(torch.rand(1, 3, 32, 32))
        out = model(tokens, corrupt=False)
        self.assertTrue(torch.isfinite(out["phase_aux_loss"]))
        self.assertAlmostEqual(
            out["loss"].item(),
            out["base_loss"].item() + 0.05 * out["phase_aux_loss"].item(),
            places=5,
        )
        out["loss"].backward()
        self.assertGreater(
            model.diffusion.net.final_layer.linear.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_causal_invariance(self):
        codec = _fitted_codec()
        model = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        model.eval()
        tokens = torch.randn(1, codec.seq_len, 6) * codec.component_mask[None, :, :]
        with torch.no_grad():
            hist = tokens[:, :-1, :]
            x = model.embed_tokens(hist, include_bos=True)
            h_full, _ = model.forward_backbone(x)
            # Mutate a future token beyond position 10; conditions up to 10 must match.
            tokens2 = tokens.clone()
            tokens2[:, 50:, :] = torch.randn_like(tokens2[:, 50:, :])
            tokens2 = tokens2 * codec.component_mask[None, :, :]
            hist2 = tokens2[:, :-1, :]
            x2 = model.embed_tokens(hist2, include_bos=True)
            h2, _ = model.forward_backbone(x2)
            self.assertTrue(torch.allclose(h_full[:, :11], h2[:, :11], atol=1e-5))

    def test_kv_cache_matches_full_prefix(self):
        codec = _fitted_codec()
        model = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        model.eval()
        b = 1
        device = torch.device("cpu")
        # Build a random prefix of length 8.
        prefix = torch.randn(b, 8, 6) * codec.component_mask[None, :8, :]
        with torch.no_grad():
            # Full prefix path: BOS + prefix -> last hidden
            x = model.embed_tokens(prefix, include_bos=True)
            h_full, _ = model.forward_backbone(x)
            z_full = h_full[:, -1, :]

            # Cached path
            z, caches = model.init_cache(b, device, model.token_proj.weight.dtype)
            for i in range(8):
                z, caches = model.forward_step(prefix[:, i, :], position=i, kv_caches=caches)
            self.assertTrue(torch.allclose(z_full, z, atol=1e-4, rtol=1e-4))

    def test_generate_shape(self):
        codec = _fitted_codec()
        # Use tiny seq by monkeypatching would be hard; instead run a few steps manually.
        model = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        model.eval()
        # Smoke only first 3 AR steps via cache API.
        z, caches = model.init_cache(2, torch.device("cpu"), model.token_proj.weight.dtype)
        mask = codec.component_mask
        toks = []
        for i in range(3):
            s = model.diffusion.sample(z, component_mask=mask[i], num_inference_steps=2)
            toks.append(s)
            z, caches = model.forward_step(s, position=i, kv_caches=caches)
        self.assertEqual(torch.stack(toks, dim=1).shape, (2, 3, 6))

    def test_cfg_rejected(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(generation=GenerationConfig(cfg_enabled=True))
        model = ContinuousFFTDecoder(cfg, codec=codec)
        tokens = torch.randn(1, codec.seq_len, 6) * codec.component_mask[None, :, :]
        with self.assertRaises(NotImplementedError):
            model(tokens, corrupt=False)

    def test_grouping_rejected(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(generation=GenerationConfig(grouping="chunk"))
        model = ContinuousFFTDecoder(cfg, codec=codec)
        tokens = torch.randn(1, codec.seq_len, 6) * codec.component_mask[None, :, :]
        with self.assertRaises(NotImplementedError):
            model(tokens, corrupt=False)

    def test_gaussian_corruption(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            corruption=CorruptionConfig(
                history_corruption="gaussian",
                history_corruption_prob=1.0,
                history_noise_min=0.05,
                history_noise_max=0.05,
            )
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        tokens = torch.zeros(2, codec.seq_len, 6)
        corrupted, strength = model.corrupt_history(tokens)
        self.assertTrue(torch.all(strength == 0.05))
        self.assertGreater(corrupted.abs().sum().item(), 0.0)
        self.assertEqual(corrupted[:, codec.is_self_conjugate, 3:].abs().max().item(), 0.0)

    def test_gaussian_corruption_ramp(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            corruption=CorruptionConfig(
                history_corruption="gaussian",
                history_corruption_prob=1.0,
                history_noise_min=0.05,
                history_noise_max=0.05,
                history_noise_ramp_fraction=0.2,
            )
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        tokens = torch.zeros(2, codec.seq_len, 6)
        clean, initial = model.corrupt_history(tokens, training_progress=0.0)
        self.assertEqual(initial.abs().max().item(), 0.0)
        self.assertEqual(clean.abs().max().item(), 0.0)
        _, halfway = model.corrupt_history(tokens, training_progress=0.1)
        self.assertTrue(torch.allclose(halfway, torch.full_like(halfway, 0.025)))
        _, complete = model.corrupt_history(tokens, training_progress=0.2)
        self.assertTrue(torch.allclose(complete, torch.full_like(complete, 0.05)))

        encoded = codec.encode(torch.rand(1, 3, 32, 32))
        output = model(encoded, corrupt=True, training_progress=0.2)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_learned_orbit_output_gain_initialization_and_gradient(self):
        codec_config = FrequencyCodecConfig(normalization="orbit_standardize")
        codec = FrequencyCodec(codec_config)

        class Loader:
            def __iter__(self):
                generator = torch.Generator().manual_seed(17)
                for _ in range(8):
                    yield torch.rand(8, 3, 32, 32, generator=generator)

        codec.fit_from_loader(Loader())
        cfg = _tiny_cfg()
        cfg.codec = codec_config
        diffusion = cfg.diffusion.fingerprint()
        diffusion["prediction_type"] = "x0"
        diffusion["learned_output_gain"] = True
        cfg.diffusion = DiffusionDecoderConfig(**diffusion)
        model = ContinuousFFTDecoder(cfg, codec=codec)
        self.assertEqual(tuple(model.output_log_gain.shape), (codec.seq_len, 3))
        self.assertEqual(model.output_log_gain.abs().max().item(), 0.0)

        nn.init.normal_(model.diffusion.net.final_layer.linear.weight, std=0.02)
        tokens = codec.encode(torch.rand(2, 3, 32, 32))
        output = model(tokens, corrupt=False)
        output["loss"].backward()
        self.assertIsNotNone(model.output_log_gain.grad)
        self.assertGreater(model.output_log_gain.grad.abs().sum().item(), 0.0)

    def test_polar_disabled_matches_baseline_modules(self):
        codec = _fitted_codec()
        model = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        self.assertIsNone(model.polar_proj)

    def test_polar_proj_grad_and_kv_cache(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(polar_history=PolarHistoryConfig(enabled=True))
        # Enable radial power weighting too.
        diff = cfg.diffusion.fingerprint()
        diff["radial_power_weighting"] = True
        cfg.diffusion = DiffusionDecoderConfig(**diff)
        model = ContinuousFFTDecoder(cfg, codec=codec)
        self.assertIsNotNone(model.polar_proj)
        # Break zero-inits so polar residual and AdaLN condition path participate.
        nn.init.normal_(model.polar_proj.weight, std=0.02)
        nn.init.zeros_(model.polar_proj.bias)
        nn.init.normal_(model.diffusion.net.final_layer.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(model.diffusion.net.final_layer.linear.weight, std=0.02)
        for block in model.diffusion.net.res_blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        tokens = torch.randn(2, codec.seq_len, 6) * codec.component_mask[None, :, :]
        out = model(tokens, corrupt=False)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.polar_proj.weight.grad)
        self.assertGreater(model.polar_proj.weight.grad.abs().sum().item(), 0.0)

        model.eval()
        prefix = torch.randn(1, 8, 6) * codec.component_mask[None, :8, :]
        with torch.no_grad():
            x = model.embed_tokens(prefix, include_bos=True)
            h_full, _ = model.forward_backbone(x)
            z_full = h_full[:, -1, :]
            z, caches = model.init_cache(1, torch.device("cpu"), model.token_proj.weight.dtype)
            for i in range(8):
                z, caches = model.forward_step(prefix[:, i, :], position=i, kv_caches=caches)
            self.assertTrue(torch.allclose(z_full, z, atol=1e-4, rtol=1e-4))

    def test_polar_causal_invariance(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(polar_history=PolarHistoryConfig(enabled=True))
        model = ContinuousFFTDecoder(cfg, codec=codec)
        model.eval()
        tokens = torch.randn(1, codec.seq_len, 6) * codec.component_mask[None, :, :]
        with torch.no_grad():
            hist = tokens[:, :-1, :]
            x = model.embed_tokens(hist, include_bos=True)
            h_full, _ = model.forward_backbone(x)
            tokens2 = tokens.clone()
            tokens2[:, 50:, :] = torch.randn_like(tokens2[:, 50:, :])
            tokens2 = tokens2 * codec.component_mask[None, :, :]
            x2 = model.embed_tokens(tokens2[:, :-1, :], include_bos=True)
            h2, _ = model.forward_backbone(x2)
            self.assertTrue(torch.allclose(h_full[:, :11], h2[:, :11], atol=1e-5))

    def test_functional_frequency_conditioning_shapes_and_zero_init(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            frequency_conditioning=FrequencyConditioningConfig(
                enabled=True,
                num_frequencies=4,
                max_frequency=8.0,
            )
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        self.assertTrue(model.pos_embed.functional)
        self.assertEqual(model.config.diffusion.target_condition_dim, 64)

        positions = torch.tensor([0, 1, 17])
        condition = model.target_position_condition(
            positions,
            batch_size=2,
            dtype=model.token_proj.weight.dtype,
        )
        self.assertEqual(tuple(condition.shape), (2, 3, 64))
        self.assertFalse(torch.allclose(condition[:, 0], condition[:, 1]))

        for layer in model.layers:
            for film in (layer.attn.position_film, layer.ff.position_film):
                self.assertIsNotNone(film)
                self.assertEqual(film.net[-1].weight.abs().max().item(), 0.0)
                self.assertEqual(film.net[-1].bias.abs().max().item(), 0.0)

    def test_frequency_conditioning_routes_are_independently_optional(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            frequency_conditioning=FrequencyConditioningConfig(
                enabled=True,
                input_addition=False,
                rms_normalize=True,
                transformer_film=False,
                diffusion_target_conditioning=False,
            )
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        self.assertEqual(model.config.diffusion.target_condition_dim, 0)
        for layer in model.layers:
            self.assertIsNone(layer.attn.position_film)
            self.assertIsNone(layer.ff.position_film)

        tokens = torch.randn(2, 3, 6) * codec.component_mask[None, :3, :]
        embedded = model.embed_tokens(tokens, include_bos=True)
        expected_tokens = model.token_proj(tokens)
        self.assertTrue(torch.allclose(embedded[:, 1:], expected_tokens))
        self.assertTrue(
            torch.allclose(embedded[:, :1], model.bos.expand(2, -1, -1))
        )

        condition = model.target_position_condition(
            torch.tensor([0, 1, 2]),
            batch_size=1,
            dtype=model.token_proj.weight.dtype,
        )
        rms = condition.square().mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))
        full_tokens = torch.randn(1, codec.seq_len, 6)
        full_tokens = full_tokens * codec.component_mask[None]
        predicted = model.predict_x0_diagnostics(
            full_tokens,
            torch.zeros(1, codec.seq_len, dtype=torch.long),
            torch.zeros_like(full_tokens),
        )
        self.assertEqual(tuple(predicted.shape), tuple(full_tokens.shape))

    def test_frequency_conditioning_reaches_diffusion_target_adaln(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            frequency_conditioning=FrequencyConditioningConfig(enabled=True)
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        captured = []
        nn.init.normal_(
            model.diffusion.net.final_layer.adaLN_modulation[-1].weight,
            std=0.02,
        )
        nn.init.normal_(model.diffusion.net.final_layer.linear.weight, std=0.02)
        for block in model.diffusion.net.res_blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)

        def capture_condition(_module, inputs):
            captured.append(inputs[0].detach())

        handle = model.diffusion.net.target_condition_embed.register_forward_pre_hook(
            capture_condition
        )
        tokens = torch.randn(1, codec.seq_len, 6) * codec.component_mask[None, :, :]
        out = model(tokens, corrupt=False)
        handle.remove()
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(tuple(captured[0].shape), (codec.seq_len, 64))
        self.assertFalse(torch.allclose(captured[0][0], captured[0][1]))
        out["loss"].backward()
        target_grad = model.diffusion.net.target_condition_embed.weight.grad
        self.assertIsNotNone(target_grad)
        self.assertGreater(target_grad.abs().sum().item(), 0.0)
        for layer in model.layers:
            for film in (layer.attn.position_film, layer.ff.position_film):
                self.assertIsNotNone(film.net[-1].weight.grad)
                self.assertGreater(film.net[-1].weight.grad.abs().sum().item(), 0.0)

    def test_frequency_conditioning_causality_and_cache_parity(self):
        codec = _fitted_codec()
        cfg = _tiny_cfg(
            frequency_conditioning=FrequencyConditioningConfig(enabled=True)
        )
        model = ContinuousFFTDecoder(cfg, codec=codec)
        model.eval()
        prefix = torch.randn(1, 8, 6) * codec.component_mask[None, :8, :]
        with torch.no_grad():
            x = model.embed_tokens(prefix, include_bos=True)
            h_full, _ = model.forward_backbone(x)

            z, caches = model.init_cache(
                1,
                torch.device("cpu"),
                model.token_proj.weight.dtype,
            )
            for i in range(8):
                z, caches = model.forward_step(
                    prefix[:, i, :],
                    position=i,
                    kv_caches=caches,
                )
            self.assertTrue(
                torch.allclose(h_full[:, -1], z, atol=1e-4, rtol=1e-4)
            )

            mutated = prefix.clone()
            mutated[:, 5:, :] = torch.randn_like(mutated[:, 5:, :])
            x_mutated = model.embed_tokens(mutated, include_bos=True)
            h_mutated, _ = model.forward_backbone(x_mutated)
            self.assertTrue(
                torch.allclose(h_full[:, :6], h_mutated[:, :6], atol=1e-5)
            )

    def test_sincos_backbone_table_exact_bos_trainable_and_independent(self):
        codec = _fitted_codec()
        position = FrequencyConditioningConfig(
            enabled=True,
            transformer_film=False,
            backbone_position_mode="sincos_table",
            input_scale_init=0.1,
        )
        model = ContinuousFFTDecoder(
            _tiny_cfg(frequency_conditioning=position),
            codec=codec,
        )
        coordinates = torch.stack([codec.ky_signed, codec.kx_signed], dim=-1)
        expected = model.backbone_pos_embed._sincos_table(
            coordinates,
            width=model.width,
            max_seq_len=model.config.transformer.max_seq_len,
        )
        self.assertTrue(
            torch.equal(model.backbone_pos_embed.seq_embed.weight, expected)
        )
        empty = torch.empty(2, 0, 6)
        embedded = model.embed_tokens(empty, include_bos=True)
        self.assertTrue(torch.equal(embedded, model.bos.expand(2, -1, -1)))

        tokens = torch.randn(1, 4, 6) * codec.component_mask[None, :4]
        model.embed_tokens(tokens, include_bos=True).sum().backward()
        self.assertGreater(
            model.backbone_pos_embed.seq_embed.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(model.input_position_scale.grad.abs().item(), 0.0)

        torch.manual_seed(123)
        random_model = ContinuousFFTDecoder(
            _tiny_cfg(
                frequency_conditioning=FrequencyConditioningConfig(
                    enabled=True,
                    transformer_film=False,
                    backbone_position_mode="random_table",
                )
            ),
            codec=codec,
        )
        torch.manual_seed(123)
        sincos_model = ContinuousFFTDecoder(
            _tiny_cfg(frequency_conditioning=position),
            codec=codec,
        )
        self.assertTrue(
            torch.equal(
                random_model.pos_embed.seq_embed.weight,
                sincos_model.pos_embed.seq_embed.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                random_model.layers[0].attn.qkv.weight,
                sincos_model.layers[0].attn.qkv.weight,
            )
        )

    def test_sincos_backbone_cache_parity_and_causality(self):
        codec = _fitted_codec()
        model = ContinuousFFTDecoder(
            _tiny_cfg(
                frequency_conditioning=FrequencyConditioningConfig(
                    enabled=True,
                    transformer_film=False,
                    backbone_position_mode="sincos_table",
                )
            ),
            codec=codec,
        )
        model.eval()
        prefix = torch.randn(1, 8, 6) * codec.component_mask[None, :8]
        with torch.no_grad():
            full, _ = model.forward_backbone(
                model.embed_tokens(prefix, include_bos=True)
            )
            cached, caches = model.init_cache(
                1, torch.device("cpu"), model.token_proj.weight.dtype
            )
            for position in range(prefix.shape[1]):
                cached, caches = model.forward_step(
                    prefix[:, position],
                    position,
                    caches,
                )
            self.assertTrue(
                torch.allclose(full[:, -1], cached, atol=1e-4, rtol=1e-4)
            )

    def test_phase_preserving_history_teacher_and_cache_alignment(self):
        config = FrequencyCodecConfig(
            normalization="orbit_standardize",
            centering="all",
        )
        codec = FrequencyCodec(config)

        class Loader:
            def __iter__(self):
                generator = torch.Generator().manual_seed(19)
                for _ in range(8):
                    yield torch.rand(8, 3, 32, 32, generator=generator)

        codec.fit_from_loader(Loader())
        cfg = _tiny_cfg(
            history_features=HistoryFeatureConfig(
                cartesian_mode="phase_preserving"
            )
        )
        cfg.codec = config
        model = ContinuousFFTDecoder(cfg, codec=codec)
        model.eval()
        prefix = codec.encode(torch.rand(1, 3, 32, 32))[:, :8]
        with torch.no_grad():
            features = model._history_cartesian_features(
                prefix,
                torch.arange(prefix.shape[1]),
            )
            expected = codec.phase_preserving_history_features(
                prefix,
                torch.arange(prefix.shape[1]),
            )
            self.assertTrue(torch.equal(features, expected))
            full, _ = model.forward_backbone(
                model.embed_tokens(prefix, include_bos=True)
            )
            cached, caches = model.init_cache(
                1, torch.device("cpu"), model.token_proj.weight.dtype
            )
            for position in range(prefix.shape[1]):
                cached, caches = model.forward_step(
                    prefix[:, position],
                    position,
                    caches,
                )
            self.assertTrue(
                torch.allclose(full[:, -1], cached, atol=1e-4, rtol=1e-4)
            )

    def test_optional_polar_module_does_not_change_shared_initialization(self):
        codec = _fitted_codec()
        torch.manual_seed(71)
        plain = ContinuousFFTDecoder(_tiny_cfg(), codec=codec)
        torch.manual_seed(71)
        polar = ContinuousFFTDecoder(
            _tiny_cfg(polar_history=PolarHistoryConfig(enabled=True)),
            codec=codec,
        )
        for name in (
            "token_proj.weight",
            "bos",
            "layers.0.attn.qkv.weight",
            "diffusion.net.input_proj.weight",
        ):
            self.assertTrue(
                torch.equal(plain.state_dict()[name], polar.state_dict()[name]),
                name,
            )

    def test_cached_and_uncached_generation_match_new_feature_routes(self):
        codec_config = FrequencyCodecConfig(normalization="orbit_standardize")
        codec = FrequencyCodec(codec_config)

        class Loader:
            def __iter__(self):
                generator = torch.Generator().manual_seed(41)
                for _ in range(4):
                    yield torch.rand(4, 3, 32, 32, generator=generator)

        codec.fit_from_loader(Loader())
        cfg = _tiny_cfg(
            polar_history=PolarHistoryConfig(enabled=True),
            history_features=HistoryFeatureConfig(
                cartesian_mode="policy",
                mean_policy="pooled_ordinary",
                scale_policy="uncentered_rms",
            ),
            frequency_conditioning=FrequencyConditioningConfig(
                enabled=True,
                transformer_film=False,
                backbone_position_mode="sincos_table",
            ),
        )
        cfg.codec = codec_config
        model = ContinuousFFTDecoder(cfg, codec=codec)
        model.eval()
        cached_generator = torch.Generator().manual_seed(333)
        uncached_generator = torch.Generator().manual_seed(333)
        with torch.no_grad():
            cached = model.generate(
                batch_size=1,
                generator=cached_generator,
                num_inference_steps=2,
                return_tokens=True,
                max_tokens=6,
            )["tokens"][:, :6]
            uncached = model.generate_uncached_prefix(
                batch_size=1,
                generator=uncached_generator,
                num_inference_steps=2,
                max_tokens=6,
            )
        self.assertTrue(torch.allclose(cached, uncached, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()