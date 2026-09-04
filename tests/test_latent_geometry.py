from __future__ import annotations

import pytest
import torch

from progressive_tokenizer.latent_geometry import (
    axis_mode_energy,
    descending_eigh,
    first_sustained_below,
    fit_axis_geometry,
    kronecker_approximation,
    snr1_crossing,
    summarize_ordered_energy,
    swap_axis_band,
)
from scripts.analyze_known_clean_denoising import _comparison


def test_snr1_crossing_matches_signal_to_noise_odds() -> None:
    power = torch.tensor([9.0, 1.0, 0.25])
    crossing = snr1_crossing(power)
    torch.testing.assert_close(crossing, torch.tensor([0.25, 0.5, 2.0 / 3.0]))
    snr = crossing.square() * power / (1.0 - crossing).square()
    torch.testing.assert_close(snr, torch.ones_like(snr))


def test_fit_axis_geometry_recovers_shapes_and_sorted_power() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(2000, 4, 3, generator=generator)
    values[:, :, 0] *= 3.0
    values[:, 0] *= 2.0
    fitted = fit_axis_geometry(values)
    assert fitted["element_mean"].shape == (4, 3)
    assert fitted["flattened_eigenvectors"].shape == (12, 12)
    assert fitted["sequence_eigenvectors"].shape == (4, 4)
    assert fitted["channel_eigenvectors"].shape == (3, 3)
    for key in ("flattened_eigenvalues", "sequence_eigenvalues", "channel_eigenvalues"):
        eigenvalues = fitted[key]
        assert bool((eigenvalues[:-1] >= eigenvalues[1:]).all())
    assert int(fitted["token_order"][0]) == 0


@pytest.mark.parametrize("axis", ["flattened", "sequence", "channel", "per_token"])
def test_axis_mode_energy_has_one_value_per_named_mode(axis: str) -> None:
    values = torch.randn(5, 4, 3)
    fitted = fit_axis_geometry(values)
    centered = values - fitted["element_mean"]
    basis = None if axis == "per_token" else fitted[f"{axis}_eigenvectors"]
    order = fitted["token_order"] if axis == "per_token" else None
    energy = axis_mode_energy(centered, axis, basis=basis, token_order=order)
    expected = {"flattened": 12, "sequence": 4, "channel": 3, "per_token": 4}[axis]
    assert energy.shape == (5, expected)


@pytest.mark.parametrize("axis", ["flattened", "sequence", "channel", "per_token"])
def test_swapping_complete_axis_reproduces_permuted_centered_values(axis: str) -> None:
    values = torch.randn(6, 4, 3)
    fitted = fit_axis_geometry(values)
    centered = values - fitted["element_mean"]
    permutation = torch.roll(torch.arange(len(values)), 1)
    if axis == "flattened":
        indices = torch.arange(12)
    elif axis == "sequence":
        indices = torch.arange(4)
    elif axis == "channel":
        indices = torch.arange(3)
    else:
        indices = torch.arange(4)
    basis = None if axis == "per_token" else fitted[f"{axis}_eigenvectors"]
    changed = swap_axis_band(
        centered, axis, indices, permutation, basis=basis
    )
    torch.testing.assert_close(changed, centered[permutation], atol=2e-5, rtol=2e-5)


def test_spectrum_summary_uses_per_sample_band_means() -> None:
    energies = torch.tensor(
        [[9.0, 8.0, 2.0, 1.0], [7.0, 6.0, 3.0, 2.0]]
    )
    summary = summarize_ordered_energy(
        energies, torch.tensor([8.0, 7.0, 2.5, 1.5]), [0, 2, 4]
    )
    assert summary["adjacent_band_descending_probability"] == [1.0]
    assert summary["bands"][0]["population_power_per_mode"] == pytest.approx(7.5)
    assert len(summary["population_power"]) == 4


def test_kronecker_diagnostic_is_exact_for_separable_covariance() -> None:
    sequence = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    channel = torch.tensor([[1.5, 0.1], [0.1, 0.5]])
    full = 3.0 * torch.kron(sequence, channel)
    result = kronecker_approximation(full, sequence, channel)
    assert result["best_scale"] == pytest.approx(3.0)
    assert result["relative_frobenius_residual"] == pytest.approx(0.0, abs=1e-7)
    assert result["squared_covariance_cosine"] == pytest.approx(1.0)


def test_descending_eigh_and_sustained_threshold_validation() -> None:
    values, vectors = descending_eigh(torch.diag(torch.tensor([1.0, 3.0, 2.0])))
    torch.testing.assert_close(values, torch.tensor([3.0, 2.0, 1.0]))
    torch.testing.assert_close(vectors.square().sum(dim=0), torch.ones(3))
    assert first_sustained_below([0.8, 0.2, 0.3, 0.1], [0.1, 0.2, 0.3, 0.4], 0.25) == 0.4
    assert first_sustained_below([0.8, 0.4], [0.1, 0.2], 0.25) is None


def test_complex_comparison_preserves_fft_phase_and_imaginary_energy() -> None:
    clean = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j])
    identical = _comparison(clean, clean)
    assert identical["relative_mse"] == pytest.approx(0.0)
    assert identical["correlation"] == pytest.approx(1.0)
    changed = _comparison(torch.tensor([1.0 + 0.0j, 3.0 + 0.0j]), clean)
    assert changed["relative_mse"] > 0.0
