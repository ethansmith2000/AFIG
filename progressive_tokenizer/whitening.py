"""Regularized, exactly invertible whitening transforms for latent caches."""

from __future__ import annotations

import math

import torch


def regularized_whitening_gains(
    power: torch.Tensor, relative_gain_cap: float
) -> dict[str, torch.Tensor | float]:
    """Return diagonal whitening gains with a bounded relative range.

    The weakest fitted powers are floored at ``max(power) / cap**2``. A final
    scalar makes the mean transformed training power one without changing the
    relative gain range.
    """

    if power.ndim != 1 or power.numel() == 0:
        raise ValueError("power must be a nonempty vector")
    if not math.isfinite(relative_gain_cap) or relative_gain_cap < 1.0:
        raise ValueError("relative_gain_cap must be finite and at least one")
    values = power.double()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("power must be finite and nonnegative")
    maximum = values.max()
    if float(maximum) <= 0.0:
        raise ValueError("at least one power must be positive")
    floor = maximum / relative_gain_cap**2
    effective = values.clamp_min(floor)
    gains = effective.rsqrt()
    transformed_power = values * gains.square()
    global_scale = transformed_power.mean().clamp_min(1e-30).rsqrt()
    gains = gains * global_scale
    transformed_power = values * gains.square()
    return {
        "gains": gains.to(dtype=power.dtype),
        "effective_power": effective.to(dtype=power.dtype),
        "transformed_power": transformed_power.to(dtype=power.dtype),
        "power_floor": float(floor),
        "global_scale": float(global_scale),
        "relative_gain_range": float(gains.max() / gains.min()),
    }


def power_whitening_gains(
    power: torch.Tensor, exponent: float
) -> dict[str, torch.Tensor | float]:
    """Smoothly interpolate from rotation-only to complete whitening.

    ``exponent=0`` applies only one global scale, while ``exponent=1`` uses
    ordinary inverse-standard-deviation whitening. Intermediate values linearly
    compress the covariance spectrum in log space. The final scalar makes mean
    transformed training power one.
    """

    if power.ndim != 1 or power.numel() == 0:
        raise ValueError("power must be a nonempty vector")
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("exponent must lie in [0,1]")
    values = power.double()
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("power must be finite and strictly positive")
    gains = values.pow(-exponent / 2.0)
    transformed_power = values * gains.square()
    global_scale = transformed_power.mean().clamp_min(1e-30).rsqrt()
    gains = gains * global_scale
    transformed_power = values * gains.square()
    return {
        "gains": gains.to(dtype=power.dtype),
        "transformed_power": transformed_power.to(dtype=power.dtype),
        "global_scale": float(global_scale),
        "relative_gain_range": float(gains.max() / gains.min()),
        "exponent": float(exponent),
    }


def zca_power_gains(
    power: torch.Tensor, exponent: float
) -> dict[str, torch.Tensor | float]:
    """Return identity-anchored gains for symmetric/ZCA power whitening.

    The gains are ``(power / mean(power))**(-exponent/2)``. Exponent zero is
    therefore exactly the identity, exponent one makes every fitted
    eigendirection have the original mean power, and intermediate exponents
    smoothly compress the log spectrum.
    """

    if power.ndim != 1 or power.numel() == 0:
        raise ValueError("power must be a nonempty vector")
    if not math.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("exponent must lie in [0,1]")
    values = power.double()
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("power must be finite and strictly positive")
    reference = values.mean()
    gains = (values / reference).pow(-exponent / 2.0)
    transformed_power = values * gains.square()
    return {
        "gains": gains.to(dtype=power.dtype),
        "transformed_power": transformed_power.to(dtype=power.dtype),
        "reference_power": float(reference),
        "relative_gain_range": float(gains.max() / gains.min()),
        "exponent": float(exponent),
    }


def zca_matrix(basis: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Construct ``basis @ diag(gains) @ basis.T`` after validation."""

    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis must be square")
    if gains.shape != (basis.shape[1],):
        raise ValueError("gains must match the basis rank")
    if not bool(torch.isfinite(basis).all()) or not bool(torch.isfinite(gains).all()):
        raise ValueError("basis and gains must be finite")
    if bool((gains <= 0).any()):
        raise ValueError("gains must be strictly positive")
    return (basis * gains[None, :]) @ basis.T


def apply_zca(
    values: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    """Apply a symmetric power transform while retaining the original axes."""

    if values.ndim != 3 or mean.shape != values.shape[1:]:
        raise ValueError("values and mean must have shapes [N,T,D] and [T,D]")
    dimensions = values.shape[1] * values.shape[2]
    if basis.shape != (dimensions, dimensions) or gains.shape != (dimensions,):
        raise ValueError("basis and gains do not match the flattened latent")
    matrix = zca_matrix(
        basis.to(values.device, dtype=torch.float32),
        gains.to(values.device, dtype=torch.float32),
    )
    centered = values.float() - mean.to(values.device, dtype=torch.float32)
    transformed = centered.flatten(1) @ matrix
    return transformed.reshape_as(values) + mean.to(values.device, dtype=torch.float32)


def invert_zca(
    values: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    """Invert :func:`apply_zca` in floating-point arithmetic."""

    return apply_zca(values, mean, basis, gains.reciprocal())


def zca_inverse_affine(
    mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize a ZCA inverse as ``values @ inverse_basis.T + offset``.

    ``mean`` is expressed in the same physical gauge as the values stored in a
    cache. The returned tensors therefore plug directly into the repository's
    ``linear_inverse`` latent-transform payload.
    """

    dimensions = mean.numel()
    if basis.shape != (dimensions, dimensions) or gains.shape != (dimensions,):
        raise ValueError("mean, basis, and gains have incompatible shapes")
    inverse_matrix = zca_matrix(basis, gains.reciprocal())
    flat_mean = mean.flatten().to(inverse_matrix.device, dtype=inverse_matrix.dtype)
    offset = flat_mean - flat_mean @ inverse_matrix
    return offset, inverse_matrix.T


def project_linear(
    values: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    """Center, rotate, and scale ``[N,T,D]`` values into flat coefficients."""

    if values.ndim != 3 or mean.shape != values.shape[1:]:
        raise ValueError("values and mean must have shapes [N,T,D] and [T,D]")
    dimensions = values.shape[1] * values.shape[2]
    if basis.shape != (dimensions, dimensions) or gains.shape != (dimensions,):
        raise ValueError("basis and gains do not match the flattened latent")
    centered = values.float() - mean.to(values.device, dtype=torch.float32)
    coefficients = centered.flatten(1) @ basis.to(values.device, dtype=torch.float32)
    return coefficients * gains.to(values.device, dtype=torch.float32)


def invert_linear(
    coefficients: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
) -> torch.Tensor:
    """Exactly invert :func:`project_linear` in floating-point arithmetic."""

    if coefficients.ndim != 2 or gains.shape != (coefficients.shape[1],):
        raise ValueError("coefficients and gains have incompatible shapes")
    dimensions = coefficients.shape[1]
    if basis.shape != (dimensions, dimensions) or mean.numel() != dimensions:
        raise ValueError("basis or mean does not match the coefficient count")
    unscaled = coefficients.float() / gains.to(
        coefficients.device, dtype=torch.float32
    )
    centered = unscaled @ basis.to(coefficients.device, dtype=torch.float32).T
    return centered.reshape(coefficients.shape[0], *mean.shape) + mean.to(
        coefficients.device, dtype=torch.float32
    )


def covariance_diagnostics(coefficients: torch.Tensor) -> dict[str, object]:
    """Summarize the complete covariance of held-out flat coefficients."""

    if coefficients.ndim != 2 or coefficients.shape[0] < 2:
        raise ValueError("coefficients must have shape [N,M] with N >= 2")
    # Consumer GPUs have very slow float64 eigensolvers; float32 is amply
    # accurate for this diagnostic after the explicit gain floor.
    analysis_dtype = torch.float64 if coefficients.device.type == "cpu" else torch.float32
    values = coefficients.to(dtype=analysis_dtype)
    centered = values - values.mean(dim=0)
    covariance = centered.T @ centered / values.shape[0]
    diagonal = covariance.diagonal().clamp_min(0.0)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0).flip(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    off_diagonal = covariance - torch.diag(diagonal)
    total_norm = covariance.square().sum().sqrt().clamp_min(1e-30)
    quantile_levels = torch.tensor(
        [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
        device=diagonal.device,
        dtype=diagonal.dtype,
    )
    quantiles = torch.quantile(diagonal, quantile_levels)
    return {
        "sample_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "effective_rank": float(effective_rank),
        "stable_rank": float(eigenvalues.sum() / eigenvalues.max().clamp_min(1e-30)),
        "off_diagonal_frobenius_fraction": float(
            off_diagonal.square().sum().sqrt() / total_norm
        ),
        "diagonal_variance_mean": float(diagonal.mean()),
        "diagonal_variance_std": float(diagonal.std(unbiased=False)),
        "diagonal_variance_quantiles": {
            label: float(value)
            for label, value in zip(
                ("minimum", "p05", "p25", "median", "p75", "p95", "maximum"),
                quantiles,
            )
        },
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def tempered_token_profile(
    token_power: torch.Tensor,
    relative_power_cap: float,
    beta: float,
) -> dict[str, object]:
    """Derive a softened rational clock and two mean-one loss profiles."""

    if token_power.ndim != 1 or token_power.numel() == 0:
        raise ValueError("token_power must be a nonempty vector")
    if not math.isfinite(relative_power_cap) or relative_power_cap < 1.0:
        raise ValueError("relative_power_cap must be finite and at least one")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and nonnegative")
    power = token_power.double()
    if not bool(torch.isfinite(power).all()) or bool((power < 0).any()):
        raise ValueError("token_power must be finite and nonnegative")
    maximum = power.max()
    if float(maximum) <= 0.0:
        raise ValueError("at least one token power must be positive")
    floor = maximum / relative_power_cap**2
    floored = power.clamp_min(floor)
    normalized = floored / floored.mean()
    effective_signal = normalized.pow(beta)
    odds = normalized.pow(beta / 2.0)
    crossings = 1.0 / (1.0 + odds)
    signal_weights = effective_signal / effective_signal.mean()
    target_weights = (1.0 + effective_signal)
    target_weights = target_weights / target_weights.mean()

    def _range(values: torch.Tensor) -> float:
        return float(values.max() / values.min().clamp_min(1e-30))

    return {
        "beta": float(beta),
        "power_floor": float(floor),
        "normalized_floored_power": [float(value) for value in normalized],
        "rational_odds": [float(value) for value in odds],
        "snr1_crossings": [float(value) for value in crossings],
        "signal_metric_loss_weights": [float(value) for value in signal_weights],
        "flow_target_energy_loss_weights": [float(value) for value in target_weights],
        "ranges": {
            "rational_odds": _range(odds),
            "snr1_crossing_min": float(crossings.min()),
            "snr1_crossing_max": float(crossings.max()),
            "signal_metric_loss": _range(signal_weights),
            "flow_target_energy_loss": _range(target_weights),
        },
    }
