"""Deterministic tests for objective-independent spectral diagnostics."""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frequency import FrequencyCodec, FrequencyCodecConfig  # noqa: E402
from spectral_diagnostics import (  # noqa: E402
    compute_normalization_phase_distortion,
    compute_perturbation_diagnostics,
    compute_spectral_diagnostics,
)


def _fitted_codec(value_transform: str = "identity") -> FrequencyCodec:
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            height=8,
            width=8,
            normalization="orbit_standardize",
            value_transform=value_transform,
        )
    )
    codec.orbit_mean.zero_()
    codec.orbit_std.fill_(1.0)
    codec.orbit_asinh_scale.fill_(2.0)
    codec.is_fitted.fill_(True)
    return codec


def _normalized(codec: FrequencyCodec, physical: torch.Tensor) -> torch.Tensor:
    physical = physical.clone()
    physical[..., 3:] *= (~codec.is_self_conjugate).to(physical.dtype)[None, :, None]
    return codec.normalize(codec.apply_value_transform(physical))


class TestSpectralDiagnostics(unittest.TestCase):
    def test_exact_predictions_identity_and_asinh(self):
        for transform in ("identity", "asinh"):
            codec = _fitted_codec(transform)
            generator = torch.Generator().manual_seed(11)
            physical = torch.randn(2, codec.seq_len, 6, generator=generator)
            tokens = _normalized(codec, physical)
            metrics = compute_spectral_diagnostics(
                tokens,
                tokens.clone(),
                codec,
                timesteps=torch.tensor([3, 7]),
                phase_amplitude_gate=0.0,
            )

            for key in (
                "normalized_active_mse",
                "physical_complex_nrmse",
                "log_amplitude_mae",
                "log_amplitude_bias",
                "phase_circular_error",
                "radial_power_relative_error",
                "timestep/3/physical_complex_nrmse",
                "timestep/7/normalized_active_mse",
            ):
                self.assertEqual(metrics[key].item(), 0.0, key)
            self.assertAlmostEqual(metrics["phase_coherence"].item(), 1.0, places=6)
            self.assertIn("radius/0/radial_power_relative_error", metrics)
            self.assertTrue(all(value.ndim == 0 for value in metrics.values()))

    def test_controlled_phase_rotation(self):
        codec = _fitted_codec()
        angle = math.pi / 3.0
        target_physical = torch.zeros(2, codec.seq_len, 6)
        predicted_physical = torch.zeros_like(target_physical)
        target_physical[..., :3] = 2.0
        predicted_physical[..., :3] = 2.0
        non_self = ~codec.is_self_conjugate
        predicted_physical[:, non_self, :3] = 2.0 * math.cos(angle)
        predicted_physical[:, non_self, 3:] = 2.0 * math.sin(angle)

        metrics = compute_spectral_diagnostics(
            _normalized(codec, predicted_physical),
            _normalized(codec, target_physical),
            codec,
            phase_amplitude_gate=0.1,
        )

        self.assertAlmostEqual(
            metrics["phase_circular_error"].item(), 1.0 - math.cos(angle), places=6
        )
        self.assertAlmostEqual(metrics["phase_coherence"].item(), 1.0, places=6)
        self.assertLess(metrics["log_amplitude_mae"].item(), 1e-6)
        self.assertLess(abs(metrics["log_amplitude_bias"].item()), 1e-6)

    def test_self_conjugate_coordinates_are_excluded(self):
        codec = _fitted_codec()
        target = torch.ones(1, codec.seq_len, 6)
        target[..., 3:] = 0.0
        predicted = target.clone()
        predicted[:, codec.is_self_conjugate, :3] = -1.0
        predicted[:, codec.is_self_conjugate, 3:] = 1000.0

        metrics = compute_spectral_diagnostics(
            predicted, target, codec, phase_amplitude_gate=0.0
        )

        self.assertEqual(metrics["phase_circular_error"].item(), 0.0)
        self.assertAlmostEqual(metrics["phase_coherence"].item(), 1.0, places=6)
        active_error = (predicted[..., :3] - target[..., :3]).square().sum()
        expected_mse = active_error / codec.component_mask.sum()
        self.assertAlmostEqual(
            metrics["normalized_active_mse"].item(), expected_mse.item(), places=6
        )

    def test_amplitude_bias_and_radial_power(self):
        codec = _fitted_codec()
        target_physical = torch.zeros(1, codec.seq_len, 6)
        target_physical[..., :3] = 1.0
        predicted_physical = 2.0 * target_physical
        metrics = compute_spectral_diagnostics(
            _normalized(codec, predicted_physical),
            _normalized(codec, target_physical),
            codec,
            phase_amplitude_gate=0.0,
        )

        self.assertAlmostEqual(metrics["log_amplitude_bias"].item(), math.log(2), places=6)
        self.assertAlmostEqual(metrics["log_amplitude_mae"].item(), math.log(2), places=6)
        self.assertAlmostEqual(metrics["radial_power_relative_error"].item(), 3.0, places=6)
        for radius in torch.unique(codec.radius_bin).tolist():
            self.assertAlmostEqual(
                metrics[f"radius/{radius}/radial_power_relative_error"].item(),
                3.0,
                places=6,
            )


class TestPhaseDistortion(unittest.TestCase):
    def test_asinh_physical_mean_and_controlled_distortion(self):
        codec = _fitted_codec("asinh")
        # The transformed-space mean inverts to physical mu = 1 + 0i.
        codec.orbit_mean[..., :3] = torch.asinh(torch.tensor(0.5))
        codec.orbit_mean[..., 3:] = 0.0

        physical = torch.zeros(2, codec.seq_len, 6)
        physical[..., 3:] = 1.0  # z = i for all non-self coordinates
        physical[:, codec.is_self_conjugate, :3] = 1e6
        metrics = compute_normalization_phase_distortion(
            physical,
            codec,
            timesteps=torch.tensor([2, 2]),
            amplitude_gate=0.1,
            dominance_c=1.0,
        )

        expected = 1.0 - math.cos(math.pi / 4.0)
        self.assertAlmostEqual(
            metrics["phase_distortion_circular_error"].item(), expected, places=6
        )
        self.assertAlmostEqual(
            metrics["phase_distortion_resultant_length"].item(), 1.0, places=6
        )
        self.assertAlmostEqual(metrics["mu_over_z_q50"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["mu_over_z_q90"].item(), 1.0, places=6)
        self.assertAlmostEqual(
            metrics["mean_dominance_fraction_c1"].item(), 1.0, places=6
        )
        self.assertAlmostEqual(
            metrics["timestep/2/phase_distortion_circular_error"].item(),
            expected,
            places=6,
        )


class TestPerturbationDiagnostics(unittest.TestCase):
    def test_rms_norm_ratios_and_timesteps(self):
        codec = _fitted_codec()
        mask = codec.component_mask[None]
        tokens = mask.expand(2, -1, -1).clone()
        noise = 2.0 * tokens
        perturbation = 0.5 * tokens
        metrics = compute_perturbation_diagnostics(
            tokens,
            noise,
            perturbation,
            codec,
            timesteps=torch.tensor([0, 1]),
            quantiles=(0.0, 0.5, 1.0),
        )

        self.assertAlmostEqual(metrics["token_rms"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["noise_rms"].item(), 2.0, places=6)
        self.assertAlmostEqual(metrics["perturbation_rms"].item(), 0.5, places=6)
        for quantile in ("q0", "q50", "q100"):
            self.assertAlmostEqual(
                metrics[f"perturbation_to_token_norm_{quantile}"].item(),
                0.5,
                places=6,
            )
        self.assertAlmostEqual(metrics["timestep/0/token_rms"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["timestep/1/noise_rms"].item(), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
