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
from progressive_tokenizer.whitening import (
    apply_zca,
    covariance_diagnostics,
    invert_linear,
    invert_zca,
    power_whitening_gains,
    project_linear,
    regularized_whitening_gains,
    tempered_token_profile,
    zca_power_gains,
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


def test_regularized_whitening_caps_gain_and_preserves_mean_power() -> None:
    fitted = regularized_whitening_gains(torch.tensor([16.0, 4.0, 0.01]), 4.0)
    gains = fitted["gains"]
    transformed = fitted["transformed_power"]
    assert isinstance(gains, torch.Tensor)
    assert isinstance(transformed, torch.Tensor)
    assert fitted["relative_gain_range"] == pytest.approx(4.0)
    assert float(transformed.mean()) == pytest.approx(1.0)
    assert float(transformed[-1]) < 1.0


def test_power_whitening_interpolates_log_spectrum() -> None:
    power = torch.tensor([16.0, 4.0, 1.0])
    identity = power_whitening_gains(power, 0.0)
    full = power_whitening_gains(power, 1.0)
    half = power_whitening_gains(power, 0.5)
    torch.testing.assert_close(
        full["transformed_power"], torch.ones(3), atol=1e-6, rtol=1e-6
    )
    assert identity["relative_gain_range"] == pytest.approx(1.0)
    assert full["relative_gain_range"] == pytest.approx(4.0)
    assert half["relative_gain_range"] == pytest.approx(2.0)
    expected = power.sqrt() / power.sqrt().mean()
    torch.testing.assert_close(half["transformed_power"], expected)


def test_power_whitening_rejects_zero_power_for_full_rank_transform() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        power_whitening_gains(torch.tensor([1.0, 0.0]), 1.0)


def test_zca_power_whitening_has_exact_identity_anchor() -> None:
    power = torch.tensor([16.0, 4.0, 1.0])
    identity = zca_power_gains(power, 0.0)
    full = zca_power_gains(power, 1.0)
    assert torch.equal(identity["gains"], torch.ones_like(power))
    torch.testing.assert_close(
        full["transformed_power"], power.mean().expand_as(power)
    )
    assert full["relative_gain_range"] == pytest.approx(4.0)


def test_zca_round_trip_and_zero_exponent_preserve_native_axes() -> None:
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(7, 3, 2, generator=generator)
    mean = torch.randn(3, 2, generator=generator)
    basis, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator))
    power = torch.tensor([9.0, 5.0, 3.0, 2.0, 1.0, 0.25])
    identity = zca_power_gains(power, 0.0)["gains"]
    gains = zca_power_gains(power, 0.5)["gains"]
    assert isinstance(identity, torch.Tensor)
    assert isinstance(gains, torch.Tensor)
    torch.testing.assert_close(
        apply_zca(values, mean, basis, identity), values, atol=1e-6, rtol=1e-6
    )
    transformed = apply_zca(values, mean, basis, gains)
    restored = invert_zca(transformed, mean, basis, gains)
    torch.testing.assert_close(restored, values, atol=2e-5, rtol=2e-5)


def test_linear_whitening_round_trip_is_exact() -> None:
    values = torch.randn(7, 4, 3)
    mean = torch.randn(4, 3)
    basis, _ = torch.linalg.qr(torch.randn(12, 12))
    gains = torch.linspace(0.25, 2.0, 12)
    projected = project_linear(values, mean, basis, gains)
    restored = invert_linear(projected, mean, basis, gains)
    torch.testing.assert_close(restored, values, atol=2e-5, rtol=2e-5)


def test_covariance_diagnostics_detects_diagonal_whitening() -> None:
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(20000, 4, generator=generator)
    values[:, 0] *= 4.0
    before = covariance_diagnostics(values)
    values[:, 0] /= 4.0
    after = covariance_diagnostics(values)
    assert after["effective_rank"] > before["effective_rank"]
    assert after["diagonal_variance_std"] < before["diagonal_variance_std"]


def test_tempered_profile_has_common_control_and_bounded_floored_range() -> None:
    power = torch.tensor([64.0, 4.0, 0.0001])
    common = tempered_token_profile(power, 8.0, 0.0)
    assert common["snr1_crossings"] == pytest.approx([0.5, 0.5, 0.5])
    assert common["signal_metric_loss_weights"] == pytest.approx([1.0, 1.0, 1.0])
    softened = tempered_token_profile(power, 8.0, 0.5)
    assert softened["ranges"]["rational_odds"] == pytest.approx(8.0**0.5)
    assert softened["ranges"]["signal_metric_loss"] == pytest.approx(8.0)
    assert softened["ranges"]["flow_target_energy_loss"] < 3.0
