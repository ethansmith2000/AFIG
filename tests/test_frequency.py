"""Tests for the canonical FrequencyCodec."""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frequency import (  # noqa: E402
    FrequencyCodec,
    FrequencyCodecConfig,
    build_orbit_table,
)


class TestOrbitTable(unittest.TestCase):
    def test_orbit_count_and_self_conjugate(self):
        table = build_orbit_table(32, 32, ordering="radial")
        self.assertEqual(int(table["seq_len"]), 514)
        self.assertEqual(int(table["is_self_conjugate"].sum()), 4)
        coords = list(zip(table["ky"].tolist(), table["kx"].tolist()))
        self.assertEqual(len(coords), len(set(coords)))
        multiplicity = table["conjugate_multiplicity"]
        self.assertTrue(torch.all(multiplicity[table["is_self_conjugate"]] == 1))
        self.assertTrue(torch.all(multiplicity[~table["is_self_conjugate"]] == 2))
        self.assertEqual(int(multiplicity.sum().item()), 32 * 32)

    def test_full_coverage(self):
        table = build_orbit_table(32, 32, ordering="radial")
        covered = set()
        for ky, kx, is_self in zip(
            table["ky"].tolist(),
            table["kx"].tolist(),
            table["is_self_conjugate"].tolist(),
        ):
            covered.add((ky, kx))
            if not is_self:
                covered.add(((-ky) % 32, (-kx) % 32))
        self.assertEqual(len(covered), 32 * 32)

    def test_radial_ordering_monotonic_radius(self):
        table = build_orbit_table(32, 32, ordering="radial")
        radii = table["radius"].tolist()
        self.assertEqual(radii, sorted(radii))

    def test_square_spiral_deterministic(self):
        a = build_orbit_table(32, 32, ordering="square_spiral")
        b = build_orbit_table(32, 32, ordering="square_spiral")
        self.assertTrue(torch.equal(a["ky"], b["ky"]))
        self.assertTrue(torch.equal(a["kx"], b["kx"]))


class TestFrequencyCodec(unittest.TestCase):
    def _synth_loader(self, n_batches=8, batch_size=4, seed=0):
        g = torch.Generator().manual_seed(seed)

        class _Loader:
            def __iter__(inner_self):
                for _ in range(n_batches):
                    yield torch.rand(batch_size, 3, 32, 32, generator=g)

        return _Loader()

    def test_raw_roundtrip(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        img = torch.rand(3, 3, 32, 32)
        rec = codec.decode_raw(codec.encode_raw(img))
        self.assertLess((img - rec).abs().max().item(), 1e-5)

    def test_hermitian_by_construction(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        tokens = torch.randn(2, codec.seq_len, 6)
        tokens[:, codec.is_self_conjugate, 3:] = 0
        spectrum = codec.tokens_to_spectrum(tokens)
        self.assertLess(codec.hermitian_violation(spectrum).item(), 1e-6)

    def test_identity_fit_roundtrip_and_whitening(self):
        codec = FrequencyCodec(FrequencyCodecConfig(value_transform="identity"))
        codec.fit_from_loader(self._synth_loader())
        img = torch.rand(4, 3, 32, 32)
        tokens = codec.encode(img)
        self.assertEqual(tuple(tokens.shape), (4, 514, 6))
        # Inactive imag components are zero.
        self.assertEqual(tokens[:, codec.is_self_conjugate, 3:].abs().max().item(), 0.0)
        rec = codec.decode(tokens)
        self.assertLess((img - rec).abs().max().item(), 1e-4)
        # Rough unit variance after whitening (allowing sampling noise).
        active = tokens[..., :3]  # always active
        self.assertTrue(0.5 < active.std().item() < 1.5)

    def test_orbit_zca_roundtrip_and_unit_covariance(self):
        codec = FrequencyCodec(
            FrequencyCodecConfig(
                value_transform="identity",
                normalization="orbit_whiten",
            )
        )
        batches = list(self._synth_loader(n_batches=32, batch_size=8))
        codec.fit_from_loader(batches)
        images = torch.cat(batches, dim=0)
        tokens = codec.encode(images)
        reconstruction = codec.decode(tokens)
        self.assertLess((images - reconstruction).abs().max().item(), 2e-4)
        for position in (1, 10, 100):
            values = tokens[:, position]
            covariance = torch.cov(values.T)
            self.assertTrue(
                torch.allclose(
                    covariance,
                    torch.eye(6),
                    atol=0.35,
                    rtol=0.35,
                )
            )

    def test_orbit_statistics_separate_axis_and_off_axis(self):
        generator = torch.Generator().manual_seed(4)

        class StripeLoader:
            def __iter__(self):
                for _ in range(16):
                    rows = torch.randn(8, 3, 32, 1, generator=generator)
                    cols = torch.randn(8, 3, 1, 32, generator=generator)
                    noise = 0.01 * torch.randn(8, 3, 32, 32, generator=generator)
                    yield torch.sigmoid(rows + cols + noise)

        codec = FrequencyCodec(
            FrequencyCodecConfig(normalization="orbit_whiten")
        )
        codec.fit_from_loader(StripeLoader())
        axis = ((codec.ky_signed == 0) & (codec.kx_signed == 5)).nonzero()[0, 0]
        off_axis = ((codec.ky_signed == 3) & (codec.kx_signed == 4)).nonzero()[0, 0]
        axis_power = torch.diagonal(codec.orbit_cov[axis]).sum()
        off_axis_power = torch.diagonal(codec.orbit_cov[off_axis]).sum()
        self.assertGreater(axis_power.item(), off_axis_power.item() * 5)

    def test_covariance_power_metric_alpha_identities(self):
        codec = FrequencyCodec(
            FrequencyCodecConfig(normalization="orbit_whiten")
        )
        codec.fit_from_loader(self._synth_loader(n_batches=16, batch_size=8))
        alpha_zero = codec.orbit_covariance_power_metric(0.0)
        active = codec.component_mask.bool()
        diagonal = torch.diagonal(alpha_zero, dim1=-2, dim2=-1)
        normalized = diagonal[active] / codec.conjugate_multiplicity[:, None].expand_as(
            diagonal
        )[active]
        self.assertLess(normalized.std().item(), 1e-5)
        self.assertEqual(
            (alpha_zero - torch.diag_embed(diagonal)).abs().max().item() < 1e-6,
            True,
        )

        alpha_one = codec.orbit_covariance_power_metric(1.0)
        expected = codec.orbit_cov * codec.conjugate_multiplicity[:, None, None]
        scale = codec.seq_len / torch.diagonal(
            expected, dim1=-2, dim2=-1
        ).sum()
        self.assertTrue(
            torch.allclose(alpha_one, expected * scale, atol=1e-6, rtol=1e-4)
        )

    def test_asinh_fit_roundtrip(self):
        codec = FrequencyCodec(FrequencyCodecConfig(value_transform="asinh"))
        codec.fit_from_loader(self._synth_loader(n_batches=12))
        img = torch.rand(2, 3, 32, 32)
        rec = codec.decode(codec.encode(img))
        self.assertLess((img - rec).abs().max().item(), 1e-4)

    def test_export_load_roundtrip(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        payload = codec.export_state()
        codec2 = FrequencyCodec(FrequencyCodecConfig())
        codec2.load_exported(payload)
        img = torch.rand(2, 3, 32, 32)
        t1 = codec.encode(img)
        t2 = codec2.encode(img)
        self.assertTrue(torch.allclose(t1, t2, atol=1e-6))

    def test_rejects_unfitted(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        with self.assertRaises(RuntimeError):
            codec.encode(torch.rand(1, 3, 32, 32))

    def test_rejects_incompatible_meta(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        payload = codec.export_state()
        payload["config"]["height"] = 64
        codec2 = FrequencyCodec(FrequencyCodecConfig())
        with self.assertRaises(ValueError):
            codec2.load_exported(payload)

    def test_standardize_mode(self):
        codec = FrequencyCodec(
            FrequencyCodecConfig(normalization="radial_standardize")
        )
        codec.fit_from_loader(self._synth_loader())
        img = torch.rand(2, 3, 32, 32)
        rec = codec.decode(codec.encode(img))
        self.assertLess((img - rec).abs().max().item(), 1e-4)

    def test_radial_loss_weights_normalized(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        w = codec.radial_loss_weights()
        self.assertEqual(tuple(w.shape), (codec.seq_len,))
        self.assertTrue(torch.all(w > 0))
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        powers = codec.bin_expected_centered_power()
        self.assertTrue(torch.all(powers > 0))
        full_power = codec.radial_loss_weights(exponent=1.0)
        self.assertLess(w.max().item(), full_power.max().item())
        flat = codec.radial_loss_weights(exponent=0.0)
        self.assertTrue(torch.allclose(flat, torch.ones_like(flat)))
        with self.assertRaises(ValueError):
            codec.radial_loss_weights(exponent=1.1)

    def test_polar_features_known_values(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        # Build a normalized token that denormalizes to a known complex value
        # by going through encode of a synthetic image, then overriding raw.
        # Simpler: craft denorm-space via normalize of a crafted raw token.
        raw = torch.zeros(1, codec.seq_len, 6)
        # Channel 0: re=3, im=4 → amp=5, phase=atan2(4,3)
        raw[0, 10, 0] = 3.0
        raw[0, 10, 3] = 4.0
        # Respect self-conjugate imag mask.
        raw[:, codec.is_self_conjugate, 3:] = 0
        tokens = codec.normalize(codec.apply_value_transform(raw))
        feats = codec.polar_history_features(tokens[:, 10:11, :], positions=torch.tensor([10]))
        self.assertEqual(tuple(feats.shape), (1, 1, 9))
        self.assertTrue(torch.isfinite(feats).all())
        scale = codec.channel_amplitude_scale()[codec.radius_bin[10], 0]
        a = 5.0 / scale.item()
        self.assertAlmostEqual(feats[0, 0, 0].item(), math.log1p(a), places=4)
        g = a / (1.0 + a)
        cos_t = 3.0 / 5.0
        sin_t = 4.0 / 5.0
        self.assertAlmostEqual(feats[0, 0, 1].item(), g * cos_t, places=4)
        self.assertAlmostEqual(feats[0, 0, 2].item(), g * sin_t, places=4)

    def test_polar_zero_amplitude_finite(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        tokens = torch.zeros(2, 4, 6)
        # Zero in normalized space → near-mean after denorm; still must be finite.
        feats = codec.polar_history_features(tokens)
        self.assertEqual(tuple(feats.shape), (2, 4, 9))
        self.assertTrue(torch.isfinite(feats).all())

    def test_polar_self_conjugate_imag_gated(self):
        codec = FrequencyCodec(FrequencyCodecConfig())
        codec.fit_from_loader(self._synth_loader())
        idx = int(codec.is_self_conjugate.nonzero(as_tuple=False)[0].item())
        tokens = torch.randn(1, 1, 6)
        tokens[..., 3:] = 1.0  # would-be imag; must be masked out for self-conjugate
        feats = codec.polar_history_features(tokens, positions=torch.tensor([idx]))
        # For self-conjugate, imag is forced 0 → sin phase terms for all channels ≈ 0
        # when amp from real only; sin features are indices 2,5,8.
        self.assertTrue(feats[0, 0, [2, 5, 8]].abs().max().item() < 1e-5)


if __name__ == "__main__":
    unittest.main()
