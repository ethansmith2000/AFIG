"""Population geometry helpers for matrix-shaped continuous latents."""

from __future__ import annotations

import math
from typing import Optional

import torch


def snr1_crossing(power: torch.Tensor) -> torch.Tensor:
    """Return ``t`` where ``t^2 power / (1-t)^2 == 1``."""

    return 1.0 / (1.0 + power.clamp_min(0.0).sqrt())


def descending_eigh(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition ordered from greatest to least power."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    return eigenvalues[order].clamp_min(0.0), eigenvectors[:, order]


def fit_axis_geometry(values: torch.Tensor) -> dict[str, object]:
    """Fit channel, sequence, flattened, and token-power population views.

    ``values`` must already be in the prior's tensor-wide standardized gauge.
    An elementwise population mean is removed before every covariance.
    Sequence and channel covariance eigenvalues are defined per complementary
    coordinate, so isotropic unit noise has variance one in every eigenmode.
    """

    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("values must have shape [N,T,D] with N >= 2")
    count, tokens, channels = values.shape
    element_mean = values.mean(dim=0)
    centered = values - element_mean
    flat = centered.flatten(1)
    flat_covariance = flat.T @ flat / count
    sequence_covariance = torch.einsum("ntd,nsd->ts", centered, centered) / (
        count * channels
    )
    channel_covariance = torch.einsum("ntd,nte->de", centered, centered) / (
        count * tokens
    )
    flat_values, flat_vectors = descending_eigh(flat_covariance)
    sequence_values, sequence_vectors = descending_eigh(sequence_covariance)
    channel_values, channel_vectors = descending_eigh(channel_covariance)
    token_power = centered.square().mean(dim=(0, 2))
    token_order = token_power.argsort(descending=True)
    return {
        "element_mean": element_mean,
        "flattened_covariance": flat_covariance,
        "flattened_eigenvalues": flat_values,
        "flattened_eigenvectors": flat_vectors,
        "sequence_covariance": sequence_covariance,
        "sequence_eigenvalues": sequence_values,
        "sequence_eigenvectors": sequence_vectors,
        "channel_covariance": channel_covariance,
        "channel_eigenvalues": channel_values,
        "channel_eigenvectors": channel_vectors,
        "token_power": token_power,
        "token_order": token_order,
    }


def axis_coefficients(
    centered: torch.Tensor,
    axis: str,
    *,
    basis: Optional[torch.Tensor] = None,
    token_order: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a centered latent into one axis view.

    Flattened coefficients have shape ``[N,TD]``; sequence coefficients
    ``[N,T,D]``; channel coefficients ``[N,T,D]`` with the last dimension
    interpreted as channel-mode rank; and per-token coefficients retain
    ``[N,T,D]`` with optional population-power token ordering.
    """

    if centered.ndim != 3:
        raise ValueError("centered must have shape [N,T,D]")
    if axis == "flattened":
        if basis is None:
            raise ValueError("flattened energy requires a basis")
        return centered.flatten(1) @ basis
    if axis == "sequence":
        if basis is None:
            raise ValueError("sequence energy requires a basis")
        return torch.einsum("ntd,tk->nkd", centered, basis)
    if axis == "channel":
        if basis is None:
            raise ValueError("channel energy requires a basis")
        return torch.einsum("ntd,dk->ntk", centered, basis)
    if axis == "per_token":
        return centered if token_order is None else centered[:, token_order]
    raise ValueError(f"unsupported axis: {axis}")


def axis_mode_energy(
    centered: torch.Tensor,
    axis: str,
    *,
    basis: Optional[torch.Tensor] = None,
    token_order: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-example energy for modes in one population view."""

    coefficients = axis_coefficients(
        centered, axis, basis=basis, token_order=token_order
    )
    if axis == "flattened":
        return coefficients.square()
    if axis == "sequence":
        return coefficients.square().mean(dim=2)
    if axis == "channel":
        return coefficients.square().mean(dim=1)
    if axis == "per_token":
        return coefficients.square().mean(dim=2)
    raise ValueError(f"unsupported axis: {axis}")


def swap_axis_band(
    centered: torch.Tensor,
    axis: str,
    indices: torch.Tensor,
    permutation: torch.Tensor,
    *,
    basis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replace one band with coefficients from other examples in the batch."""

    if centered.ndim != 3:
        raise ValueError("centered must have shape [N,T,D]")
    if permutation.ndim != 1 or permutation.numel() != centered.shape[0]:
        raise ValueError("permutation must have one entry per example")
    indices = indices.to(device=centered.device, dtype=torch.long)
    permutation = permutation.to(device=centered.device, dtype=torch.long)
    changed = centered.clone()
    if axis == "per_token":
        changed[:, indices] = centered[permutation][:, indices]
        return changed
    if basis is None:
        raise ValueError(f"{axis} swap requires a basis")
    selected = basis[:, indices]
    if axis == "flattened":
        flat = centered.flatten(1)
        coefficients = flat @ selected
        delta = (coefficients[permutation] - coefficients) @ selected.T
        return (flat + delta).reshape_as(centered)
    if axis == "channel":
        coefficients = torch.einsum("ntd,dk->ntk", centered, selected)
        delta = coefficients[permutation] - coefficients
        return centered + torch.einsum("ntk,dk->ntd", delta, selected)
    if axis == "sequence":
        coefficients = torch.einsum("ntd,tk->nkd", centered, selected)
        delta = coefficients[permutation] - coefficients
        return centered + torch.einsum("nkd,tk->ntd", delta, selected)
    raise ValueError(f"unsupported axis: {axis}")


def effective_rank(power: torch.Tensor) -> float:
    probabilities = power.double().clamp_min(0.0)
    probabilities = probabilities / probabilities.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def stable_rank(power: torch.Tensor) -> float:
    power = power.double().clamp_min(0.0)
    return float(power.sum() / power.max().clamp_min(1e-30))


def quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95],
        device=values.device,
        dtype=values.dtype,
    )
    measured = torch.quantile(values, probabilities)
    return {
        label: float(value)
        for label, value in zip(("p05", "p25", "p50", "p75", "p95"), measured)
    }


def _rank_correlation_quantiles(energies: torch.Tensor) -> dict[str, float]:
    mode_count = energies.shape[1]
    if mode_count < 2:
        return {key: 1.0 for key in ("p05", "p25", "p50", "p75", "p95")}
    sample_ranks = torch.argsort(torch.argsort(-energies, dim=1), dim=1).float()
    population_ranks = torch.arange(
        mode_count, device=energies.device, dtype=sample_ranks.dtype
    )
    population_ranks = population_ranks - population_ranks.mean()
    sample_ranks = sample_ranks - sample_ranks.mean(dim=1, keepdim=True)
    correlation = (sample_ranks * population_ranks).sum(dim=1) / (
        sample_ranks.square().sum(dim=1).sqrt()
        * population_ranks.square().sum().sqrt()
    ).clamp_min(1e-30)
    return quantiles(correlation)


def summarize_ordered_energy(
    energies: torch.Tensor,
    population_power: torch.Tensor,
    edges: list[int],
) -> dict[str, object]:
    """Summarize a population-ordered spectrum and individual-sample order."""

    if energies.ndim != 2 or population_power.ndim != 1:
        raise ValueError("energies must be [N,M] and population_power [M]")
    if energies.shape[1] != population_power.numel():
        raise ValueError("energy and population dimensions disagree")
    if edges[0] != 0 or edges[-1] != population_power.numel():
        raise ValueError("edges must span the complete mode dimension")
    if any(left >= right for left, right in zip(edges[:-1], edges[1:])):
        raise ValueError("edges must be strictly increasing")
    band_energy = torch.stack(
        [energies[:, lo:hi].mean(dim=1) for lo, hi in zip(edges[:-1], edges[1:])],
        dim=1,
    )
    band_power = torch.stack(
        [population_power[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])]
    )
    adjacent = [
        float((band_energy[:, index] > band_energy[:, index + 1]).float().mean())
        for index in range(band_energy.shape[1] - 1)
    ]
    bands = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bands.append(
            {
                "range_zero_based_half_open": [lo, hi],
                "population_power_per_mode": float(band_power[index]),
                "population_snr1_t": float(snr1_crossing(band_power[index])),
                "sample_power_quantiles": quantiles(band_energy[:, index]),
                "sample_snr1_t_quantiles": quantiles(
                    snr1_crossing(band_energy[:, index])
                ),
            }
        )
    power = population_power.double().clamp_min(0.0)
    cumulative = power.cumsum(0) / power.sum().clamp_min(1e-30)
    return {
        "mode_count": int(power.numel()),
        "population_power": [float(value) for value in power],
        "population_snr1_t": [float(value) for value in snr1_crossing(power)],
        "cumulative_power_share": [float(value) for value in cumulative],
        "effective_rank": effective_rank(power),
        "stable_rank": stable_rank(power),
        "largest_to_median_power_ratio": float(
            power[0] / power.median().clamp_min(1e-30)
        ),
        "largest_to_smallest_power_ratio": float(
            power[0] / power[-1].clamp_min(1e-30)
        ),
        "sample_mode_rank_correlation": _rank_correlation_quantiles(energies),
        "adjacent_band_descending_probability": adjacent,
        "bands": bands,
    }


def kronecker_approximation(
    full: torch.Tensor, sequence: torch.Tensor, channel: torch.Tensor
) -> dict[str, float]:
    """Best scalar fit of ``sequence kron channel`` to full covariance."""

    approximation = torch.kron(sequence, channel)
    if approximation.shape != full.shape:
        raise ValueError("Kronecker factors do not match full covariance")
    full64 = full.double()
    approximation64 = approximation.double()
    scale = (full64 * approximation64).sum() / approximation64.square().sum().clamp_min(1e-30)
    fitted = scale * approximation64
    cosine = (full64 * fitted).sum() / (
        full64.square().sum().sqrt() * fitted.square().sum().sqrt()
    ).clamp_min(1e-30)
    residual = (full64 - fitted).square().sum().sqrt() / full64.square().sum().sqrt().clamp_min(1e-30)
    return {
        "best_scale": float(scale),
        "covariance_cosine": float(cosine),
        "squared_covariance_cosine": float(cosine.square()),
        "relative_frobenius_residual": float(residual),
    }


def first_sustained_below(
    values: list[float], times: list[float], threshold: float
) -> Optional[float]:
    if len(values) != len(times) or not values:
        raise ValueError("values and times must have equal nonzero length")
    first: Optional[float] = None
    suffix = True
    for index in range(len(values) - 1, -1, -1):
        suffix = suffix and math.isfinite(values[index]) and values[index] <= threshold
        if suffix:
            first = times[index]
    return first
