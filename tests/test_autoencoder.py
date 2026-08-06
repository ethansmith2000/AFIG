"""Tests for compressed frequency and spatial AFIG autoencoders."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoencoder_models import (  # noqa: E402
    AutoencoderConfig,
    CausalFrequencyAutoencoder,
    CausalTCN,
    GroupLayout,
    ImageAutoencoderAdapter,
    LatentCausalProbe,
    LatentFourierNormalizer,
    RealLatentFFT,
    SpatialAutoencoder,
)
from frequency import FrequencyCodec, FrequencyCodecConfig  # noqa: E402


def _fitted_codec(size: int = 8) -> FrequencyCodec:
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            height=size,
            width=size,
            normalization="orbit_standardize",
        )
    )

    class Loader:
        def __iter__(self):
            generator = torch.Generator().manual_seed(17)
            for _ in range(4):
                yield torch.rand(4, 3, size, size, generator=generator)

    codec.fit_from_loader(Loader())
    return codec


class TestGrouping(unittest.TestCase):
    def test_fixed_chunk_layout_covers_every_position(self):
        radius = torch.arange(34)
        layout = GroupLayout(
            seq_len=34,
            mode="causal_k",
            group_size=4,
            radius_bin=radius,
            target_tokens_per_latent=8,
            max_ring_latents=4,
            pooler="perceiver_full",
        )
        self.assertEqual(layout.num_latents, 9)
        self.assertEqual(layout.num_parents, 9)
        self.assertTrue(torch.equal(torch.unique(layout.token_parent), torch.arange(9)))
        covered = []
        for group in range(layout.num_latents):
            covered.extend(
                layout.gather_indices[group][layout.gather_mask[group]].tolist()
            )
        self.assertEqual(covered, list(range(34)))

    def test_radial_layout_adapts_latents_and_balances_sectors(self):
        radius = torch.tensor([0] + [1] * 3 + [2] * 9 + [3] * 20)
        layout = GroupLayout(
            seq_len=len(radius),
            mode="causal_ring",
            group_size=4,
            radius_bin=radius,
            target_tokens_per_latent=5,
            max_ring_latents=4,
            pooler="perceiver_sector",
        )
        self.assertEqual(layout.num_parents, 4)
        self.assertTrue(torch.equal(layout.parent_counts, torch.tensor([1, 1, 2, 4])))
        for parent in range(layout.num_parents):
            members = []
            latent_ids = torch.nonzero(
                layout.latent_parent == parent, as_tuple=False
            ).flatten()
            for latent in latent_ids:
                members.extend(
                    layout.gather_indices[latent][layout.gather_mask[latent]].tolist()
                )
            expected = torch.nonzero(
                layout.token_parent == parent, as_tuple=False
            ).flatten().tolist()
            self.assertEqual(sorted(members), expected)

    def test_full_ring_queries_share_membership(self):
        radius = torch.tensor([0] + [1] * 9)
        layout = GroupLayout(
            seq_len=10,
            mode="causal_ring",
            group_size=4,
            radius_bin=radius,
            target_tokens_per_latent=4,
            max_ring_latents=4,
            pooler="perceiver_full",
        )
        ring_latents = torch.nonzero(layout.latent_parent == 1).flatten()
        first = layout.gather_indices[ring_latents[0]][
            layout.gather_mask[ring_latents[0]]
        ]
        for latent in ring_latents[1:]:
            current = layout.gather_indices[latent][layout.gather_mask[latent]]
            self.assertTrue(torch.equal(first, current))


class TestCausalFrequencyAutoencoder(unittest.TestCase):
    def _config(self, mode: str, variational: bool = False) -> AutoencoderConfig:
        return AutoencoderConfig(
            mode=mode,
            variational=variational,
            latent_dim=12,
            model_width=24,
            depth=3,
            kernel_size=3,
            group_size=4,
            pooler="perceiver_sector" if mode == "causal_ring" else "perceiver_full",
            target_tokens_per_latent=4,
            max_ring_latents=3,
        )

    def test_shapes_masks_gradients_and_variational_path(self):
        codec = _fitted_codec()
        for mode in ("causal_k", "causal_ring"):
            config = self._config(mode, variational=True)
            model = CausalFrequencyAutoencoder(
                config, codec.position_metadata(), codec.component_mask
            )
            tokens = codec.encode(torch.rand(2, 3, 8, 8))
            output = model(tokens, sample_posterior=True)
            self.assertEqual(output["reconstruction"].shape, tokens.shape)
            self.assertEqual(output["latents"].shape[-1], config.latent_dim)
            self.assertEqual(
                output["latents"].shape[1], model.exported_token_count
            )
            exported = model.export_latents(tokens)
            self.assertEqual(exported["latents"].shape, output["latents"].shape)
            self.assertEqual(exported["latent_parent"].numel(), model.exported_token_count)
            self.assertTrue(torch.isfinite(output["kl_per_dim"]).all())
            self.assertEqual(
                output["reconstruction"][:, codec.is_self_conjugate, 3:]
                .abs()
                .max()
                .item(),
                0.0,
            )
            output["reconstruction"].square().mean().backward()
            if mode == "causal_ring":
                self.assertIsNotNone(model.ring_encoder.input.weight.grad)
                self.assertIsNotNone(model.ring_encoder.output[-1].weight.grad)
            else:
                self.assertIsNotNone(model.token_proj.weight.grad)
                self.assertIsNotNone(model.pool.out[-1].weight.grad)

    def test_deterministic_path_reports_zero_kl(self):
        codec = _fitted_codec()
        config = self._config("causal_k", variational=False)
        model = CausalFrequencyAutoencoder(
            config, codec.position_metadata(), codec.component_mask
        )
        output = model(codec.encode(torch.rand(2, 3, 8, 8)))
        self.assertEqual(output["kl_per_dim"].abs().max().item(), 0.0)

    def test_symmetric_group_conditioning_has_gradients(self):
        codec = _fitted_codec()
        metadata = codec.position_metadata()
        metadata["empirical_scale"] = codec.orbit_std.mean(dim=-1)
        config = AutoencoderConfig(
            mode="causal_ring",
            latent_dim=16,
            model_width=24,
            depth=3,
            pooler="perceiver_sector",
            target_tokens_per_latent=4,
            max_ring_latents=3,
            group_conditioning="film_low_rank",
            conditioning_rank=4,
        )
        model = CausalFrequencyAutoencoder(
            config, metadata, codec.component_mask
        )
        output = model(codec.encode(torch.rand(2, 3, 8, 8)))
        output["reconstruction"].square().mean().backward()
        conditioned = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ("condition" in name or "film" in name) and parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in conditioned))
        self.assertEqual(model.token_condition.shape[-1], 10)

    def test_flat_mlp_pooler(self):
        codec = _fitted_codec()
        config = AutoencoderConfig(
            mode="causal_k",
            latent_dim=8,
            model_width=16,
            depth=2,
            group_size=4,
            pooler="flat_mlp",
        )
        model = CausalFrequencyAutoencoder(
            config, codec.position_metadata(), codec.component_mask
        )
        output = model(codec.encode(torch.rand(1, 3, 8, 8)))
        self.assertEqual(output["latents"].shape, (1, 9, 8))

    def test_flat_mlp_is_rejected_for_sequential_ring_codec(self):
        codec = _fitted_codec()
        config = AutoencoderConfig(
            mode="causal_ring",
            latent_dim=32,
            model_width=24,
            depth=3,
            pooler="flat_mlp",
            target_tokens_per_latent=4,
            max_ring_latents=3,
            group_conditioning="film_low_rank",
        )
        metadata = codec.position_metadata()
        metadata["empirical_scale"] = codec.orbit_std.mean(dim=-1)
        with self.assertRaisesRegex(ValueError, "sector-local Perceiver"):
            CausalFrequencyAutoencoder(config, metadata, codec.component_mask)

    def test_encoder_future_perturbation_invariance(self):
        codec = _fitted_codec()
        model = CausalFrequencyAutoencoder(
            self._config("causal_k"),
            codec.position_metadata(),
            codec.component_mask,
        ).eval()
        tokens = codec.encode(torch.rand(1, 3, 8, 8))
        changed = tokens.clone()
        changed[:, 12:] = torch.randn_like(changed[:, 12:])
        with torch.no_grad():
            original, _ = model.encode(tokens)
            perturbed, _ = model.encode(changed)
        completed = (
            model.layout.gather_indices.masked_fill(
                ~model.layout.gather_mask, -1
            ).max(dim=1).values
            < 12
        )
        self.assertTrue(
            torch.allclose(original[:, completed], perturbed[:, completed], atol=1e-6)
        )

    def test_decoder_future_latent_invariance(self):
        codec = _fitted_codec()
        model = CausalFrequencyAutoencoder(
            self._config("causal_ring"),
            codec.position_metadata(),
            codec.component_mask,
        ).eval()
        latents = torch.randn(1, model.exported_token_count, model.config.latent_dim)
        changed = latents.clone()
        cutoff_parent = max(model.layout.num_parents // 2, 1)
        changed[:, model.layout.latent_parent >= cutoff_parent] += 10
        with torch.no_grad():
            original = model.decode(latents)
            perturbed = model.decode(changed)
        earlier = model.layout.token_parent < cutoff_parent
        self.assertTrue(
            torch.allclose(original[:, earlier], perturbed[:, earlier], atol=1e-6)
        )

    def test_ring_sector_level_encoder_and_decoder_causality(self):
        codec = _fitted_codec()
        model = CausalFrequencyAutoencoder(
            self._config("causal_ring"),
            codec.position_metadata(),
            codec.component_mask,
        ).eval()
        cutoff = max(model.exported_token_count // 2, 1)
        tokens = codec.encode(torch.rand(1, 3, 8, 8))
        changed_tokens = tokens.clone()
        changed_tokens[:, model.layout.token_latent >= cutoff] += 5.0
        with torch.no_grad():
            original_latents, _ = model.encode(tokens)
            changed_latents, _ = model.encode(changed_tokens)
        self.assertTrue(
            torch.allclose(
                original_latents[:, :cutoff],
                changed_latents[:, :cutoff],
                atol=1e-6,
            )
        )

        latents = torch.randn(
            1, model.exported_token_count, model.config.latent_dim
        )
        changed_latents = latents.clone()
        changed_latents[:, cutoff:] += 5.0
        with torch.no_grad():
            original_tokens = model.decode(latents)
            changed_tokens = model.decode(changed_latents)
        earlier_tokens = model.layout.token_latent < cutoff
        self.assertTrue(
            torch.allclose(
                original_tokens[:, earlier_tokens],
                changed_tokens[:, earlier_tokens],
                atol=1e-6,
            )
        )

    def test_ring_block_masks_are_bidirectional_within_ring(self):
        codec = _fitted_codec()
        config = AutoencoderConfig(
            **{
                **self._config("causal_ring").fingerprint(),
                "ring_block_causal": True,
            }
        )
        model = CausalFrequencyAutoencoder(
            config, codec.position_metadata(), codec.component_mask
        )
        encoder_mask = model.ring_encoder.block_causal_mask
        decoder_mask = model.ring_decoder.latent_causal_mask
        coordinate_mask = model.ring_decoder.coordinate_causal_mask

        ring = int(torch.nonzero(model.layout.parent_counts > 1)[0])
        ring_tokens = torch.nonzero(model.layout.token_parent == ring).flatten()
        ring_latents = torch.nonzero(model.layout.latent_parent == ring).flatten()
        self.assertTrue(encoder_mask[ring_tokens[:, None], ring_tokens[None, :]].all())
        self.assertTrue(decoder_mask[ring_latents[:, None], ring_latents[None, :]].all())
        self.assertTrue(coordinate_mask[ring_tokens[:, None], ring_latents[None, :]].all())

        if ring + 1 < model.layout.num_parents:
            later_tokens = torch.nonzero(
                model.layout.token_parent == ring + 1
            ).flatten()
            later_latents = torch.nonzero(
                model.layout.latent_parent == ring + 1
            ).flatten()
            self.assertFalse(
                encoder_mask[ring_tokens[:, None], later_tokens[None, :]].any()
            )
            self.assertTrue(
                encoder_mask[later_tokens[:, None], ring_tokens[None, :]].all()
            )
            self.assertFalse(
                decoder_mask[ring_latents[:, None], later_latents[None, :]].any()
            )
            self.assertFalse(
                coordinate_mask[ring_tokens[:, None], later_latents[None, :]].any()
            )

    def test_tcn_full_streaming_parity(self):
        torch.manual_seed(7)
        model = CausalTCN(width=16, depth=4, kernel_size=3).eval()
        values = torch.randn(2, 19, 16)
        with torch.no_grad():
            full = model(values)
            streaming = model.forward_streaming(values)
        self.assertTrue(torch.allclose(full, streaming, atol=1e-6, rtol=1e-6))

    def test_automatic_depth_covers_full_sequence(self):
        codec = _fitted_codec()
        config = AutoencoderConfig(
            mode="causal_k",
            depth=0,
            model_width=16,
            latent_dim=8,
            group_size=4,
            pooler="perceiver_full",
        )
        model = CausalFrequencyAutoencoder(
            config, codec.position_metadata(), codec.component_mask
        )
        self.assertGreaterEqual(model.encoder.receptive_field, codec.seq_len)


class TestSpatialAutoencoder(unittest.TestCase):
    def test_latent_fft_roundtrip_arbitrary_channels(self):
        bridge = RealLatentFFT(8, 8)
        maps = torch.randn(2, 5, 8, 8)
        tokens = bridge.encode(maps)
        reconstructed = bridge.decode(tokens)
        self.assertEqual(tokens.shape, (2, 34, 10))
        self.assertTrue(torch.allclose(reconstructed, maps, atol=2e-5, rtol=2e-5))

    def test_spatial_vae_shapes_and_gradients(self):
        config = AutoencoderConfig(
            mode="spatial_downsample",
            variational=True,
            spatial_resolution=32,
            spatial_downsample=4,
            spatial_latent_channels=6,
            spatial_base_channels=16,
        )
        model = SpatialAutoencoder(config)
        images = torch.rand(2, 3, 32, 32)
        output = model(images, sample_posterior=True)
        self.assertEqual(output["reconstruction"].shape, images.shape)
        self.assertEqual(output["latents"].shape, (2, 6, 8, 8))
        self.assertEqual(output["latent_tokens"].shape, (2, 34, 12))
        exported = model.export_latents(images)
        self.assertEqual(exported["latent_tokens"].shape, (2, 34, 12))
        output["reconstruction"].square().mean().backward()
        self.assertIsNotNone(model.encoder[0].weight.grad)

    def test_deterministic_spatial_path_reports_zero_kl(self):
        config = AutoencoderConfig(
            mode="spatial_downsample",
            variational=False,
            spatial_resolution=8,
            spatial_downsample=2,
            spatial_latent_channels=4,
            spatial_base_channels=8,
        )
        output = SpatialAutoencoder(config)(torch.rand(1, 3, 8, 8))
        self.assertEqual(output["kl_per_dim"].abs().max().item(), 0.0)

    def test_latent_fourier_normalizer_roundtrip_and_pairing(self):
        bridge = RealLatentFFT(8, 8)
        tokens = bridge.encode(torch.randn(32, 3, 8, 8))
        normalizer = LatentFourierNormalizer(bridge, channels=3)
        normalizer.fit(tokens)
        normalized = normalizer.normalize(tokens)
        reconstructed = normalizer.denormalize(normalized)
        self.assertTrue(torch.allclose(reconstructed, tokens, atol=2e-5, rtol=2e-5))
        self.assertFalse(bool(normalizer.center_ordinary))
        self.assertEqual(
            normalizer.mean[~bridge.is_self_conjugate].abs().max().item(), 0.0
        )

    def test_existing_image_autoencoder_adapter(self):
        class Posterior:
            def __init__(self, mean):
                self.mean = mean
                self.logvar = torch.zeros_like(mean)

            def mode(self):
                return self.mean

            def sample(self):
                return self.mean + 0.01 * torch.randn_like(self.mean)

        class Encoded:
            def __init__(self, latent):
                self.latent_dist = Posterior(latent)

        class Decoded:
            def __init__(self, sample):
                self.sample = sample

        class TinyImageAE(torch.nn.Module):
            class Config:
                scaling_factor = 0.5

            config = Config()

            def encode(self, images):
                return Encoded(torch.nn.functional.avg_pool2d(images, 2))

            def decode(self, latents):
                return Decoded(
                    torch.nn.functional.interpolate(
                        latents, scale_factor=2, mode="nearest"
                    )
                )

        adapter = ImageAutoencoderAdapter(
            TinyImageAE(), latent_height=4, latent_width=4
        )
        output = adapter(torch.rand(2, 3, 8, 8))
        self.assertEqual(output["latents"].shape, (2, 3, 4, 4))
        self.assertEqual(output["latent_tokens"].shape, (2, 10, 6))
        self.assertEqual(output["reconstruction"].shape, (2, 3, 8, 8))

    def test_causal_probe_shapes_and_gradients(self):
        probe = LatentCausalProbe(latent_dim=12, width=16)
        latents = torch.randn(3, 7, 12)
        loss = probe.loss(latents)
        loss.backward()
        self.assertEqual(probe(latents).shape, latents.shape)
        self.assertIsNotNone(probe.output.weight.grad)


class TestAutoencoderTrainerSmoke(unittest.TestCase):
    def test_spectral_losses_are_independent_and_differentiable(self):
        from train_autoencoder import _spectral_loss_terms

        target = torch.rand(2, 3, 8, 8)
        reconstruction = (target + 0.05 * torch.randn_like(target)).requires_grad_()
        losses = _spectral_loss_terms(target, reconstruction, phase_gate=0.1)
        self.assertGreater(losses["log_amplitude"].item(), 0.0)
        self.assertGreater(losses["phase"].item(), 0.0)
        self.assertGreater(losses["radial_log_power"].item(), 0.0)
        sum(losses.values()).backward()
        self.assertIsNotNone(reconstruction.grad)

    def test_one_step_cpu_smoke_all_modes(self):
        from accelerate.state import AcceleratorState
        from train_autoencoder import main, parse_args

        for mode in ("causal_k", "causal_ring", "spatial_downsample"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                argv = [
                    "--smoke",
                    "--mode",
                    mode,
                    "--output_dir",
                    directory,
                    "--report_to",
                    "none",
                ]
                if mode == "causal_k":
                    argv.append("--save_final_checkpoint")
                args = parse_args(argv)
                try:
                    main(args)
                finally:
                    AcceleratorState._reset_state(reset_partial_state=True)
                self.assertTrue(
                    os.path.isfile(os.path.join(directory, "reconstruction_1.png"))
                )
                if mode == "causal_k":
                    checkpoint = torch.load(
                        os.path.join(directory, "checkpoint_1.pt"),
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertEqual(checkpoint["global_step"], 1)
                    self.assertIn("codec", checkpoint)
                    self.assertEqual(checkpoint["config"]["mode"], "causal_k")


if __name__ == "__main__":
    unittest.main()
