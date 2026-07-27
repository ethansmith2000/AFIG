"""Objective-independent diagnostics for continuous Fourier tokens.

All public helpers return flat dictionaries whose values are scalar tensors, so
their outputs can be passed directly to common experiment loggers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch

from frequency import FrequencyCodec


ScalarMetrics = Dict[str, torch.Tensor]


def _validate_tokens(tokens: torch.Tensor, codec: FrequencyCodec, name: str) -> None:
    expected = (codec.seq_len, 6)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(
            f"{name} must have shape [B,{codec.seq_len},6], got {tuple(tokens.shape)}"
        )
    if tokens.device != codec.component_mask.device:
        raise ValueError(f"{name} and codec must be on the same device")


def _validate_pair(
    predicted: torch.Tensor, target: torch.Tensor, codec: FrequencyCodec
) -> None:
    _validate_tokens(predicted, codec, "predicted")
    _validate_tokens(target, codec, "target")
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target shapes differ: {predicted.shape} vs {target.shape}"
        )


def _validate_timesteps(
    timesteps: Optional[torch.Tensor], batch_size: int, device: torch.device
) -> Optional[torch.Tensor]:
    if timesteps is None:
        return None
    timesteps = torch.as_tensor(timesteps, device=device)
    if timesteps.ndim != 1 or timesteps.numel() != batch_size:
        raise ValueError(f"timesteps must have shape [{batch_size}]")
    return timesteps


def _complex(tokens: torch.Tensor) -> torch.Tensor:
    return torch.complex(tokens[..., :3].float(), tokens[..., 3:].float())


def _weighted_mean(
    values: torch.Tensor, weights: torch.Tensor, eps: float
) -> torch.Tensor:
    expanded = weights.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(eps)


def _weighted_resultant(
    phase_delta: torch.Tensor, weights: torch.Tensor, eps: float
) -> torch.Tensor:
    expanded = weights.expand_as(phase_delta)
    denominator = expanded.sum()
    if not bool(denominator > 0):
        return phase_delta.new_tensor(float("nan"))
    cosine = (torch.cos(phase_delta) * expanded).sum() / denominator.clamp_min(eps)
    sine = (torch.sin(phase_delta) * expanded).sum() / denominator.clamp_min(eps)
    return torch.sqrt(cosine.square() + sine.square())


def _quantile_label(quantile: float) -> str:
    return f"q{100.0 * quantile:g}"


def _timestep_label(value: torch.Tensor) -> str:
    number = value.item()
    return f"{number:g}" if isinstance(number, float) else str(number)


@torch.no_grad()
def normalized_to_physical(
    normalized_tokens: torch.Tensor, codec: FrequencyCodec
) -> torch.Tensor:
    """Invert codec normalization and value transforms into raw Fourier tokens."""
    _validate_tokens(normalized_tokens, codec, "normalized_tokens")
    transformed = codec.denormalize(normalized_tokens)
    physical = codec.invert_value_transform(transformed)
    physical = physical.clone()
    non_self = (~codec.is_self_conjugate).to(dtype=physical.dtype)
    physical[..., 3:] *= non_self[None, :, None]
    return physical


def _spectral_summary(
    predicted_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    predicted_physical: torch.Tensor,
    target_physical: torch.Tensor,
    codec: FrequencyCodec,
    phase_amplitude_gate: float,
    eps: float,
    orbit_selection: Optional[torch.Tensor] = None,
) -> ScalarMetrics:
    if orbit_selection is None:
        orbit_selection = torch.ones(
            codec.seq_len, dtype=torch.bool, device=predicted_normalized.device
        )

    component_mask = codec.component_mask[orbit_selection].to(
        device=predicted_normalized.device, dtype=predicted_normalized.dtype
    )
    normalized_error = (
        predicted_normalized[:, orbit_selection]
        - target_normalized[:, orbit_selection]
    )
    normalized_mse = (
        normalized_error.square() * component_mask[None]
    ).sum() / (
        component_mask.sum() * predicted_normalized.shape[0]
    ).clamp_min(eps)

    predicted_complex = _complex(predicted_physical[:, orbit_selection])
    target_complex = _complex(target_physical[:, orbit_selection])
    multiplicity = codec.conjugate_multiplicity[orbit_selection].to(
        device=predicted_complex.device, dtype=predicted_complex.real.dtype
    )[None, :, None]
    squared_error = (predicted_complex - target_complex).abs().square()
    target_power = target_complex.abs().square()
    physical_nrmse = torch.sqrt(
        (squared_error * multiplicity).sum()
        / (target_power * multiplicity).sum().clamp_min(eps)
    )

    predicted_amplitude = predicted_complex.abs()
    target_amplitude = target_complex.abs()
    log_amplitude_error = torch.log(predicted_amplitude.clamp_min(eps)) - torch.log(
        target_amplitude.clamp_min(eps)
    )
    log_amplitude_bias = _weighted_mean(log_amplitude_error, multiplicity, eps)
    log_amplitude_mae = _weighted_mean(log_amplitude_error.abs(), multiplicity, eps)

    selected_non_self = (~codec.is_self_conjugate[orbit_selection]).to(
        device=predicted_complex.device
    )
    phase_valid = (
        selected_non_self[None, :, None]
        & (target_amplitude > phase_amplitude_gate)
    )
    phase_weights = phase_valid.to(dtype=predicted_complex.real.dtype)
    phase_delta = torch.angle(predicted_complex * target_complex.conj())
    if bool(phase_valid.any()):
        phase_circular_error = _weighted_mean(
            1.0 - torch.cos(phase_delta), phase_weights, eps
        )
        phase_coherence = _weighted_resultant(phase_delta, phase_weights, eps)
    else:
        phase_circular_error = predicted_complex.real.new_tensor(float("nan"))
        phase_coherence = predicted_complex.real.new_tensor(float("nan"))
    phase_valid_fraction = phase_weights.sum() / (
        selected_non_self.sum() * predicted_complex.shape[0] * predicted_complex.shape[2]
    ).clamp_min(1)

    predicted_radial_power = (predicted_amplitude.square() * multiplicity).sum()
    target_radial_power = (target_amplitude.square() * multiplicity).sum()
    radial_power_relative_error = (
        predicted_radial_power - target_radial_power
    ).abs() / target_radial_power.clamp_min(eps)

    return {
        "normalized_active_mse": normalized_mse,
        "physical_complex_nrmse": physical_nrmse,
        "log_amplitude_mae": log_amplitude_mae,
        "log_amplitude_bias": log_amplitude_bias,
        "phase_circular_error": phase_circular_error,
        "phase_coherence": phase_coherence,
        "phase_valid_fraction": phase_valid_fraction,
        "radial_power_relative_error": radial_power_relative_error,
    }


@torch.no_grad()
def compute_spectral_diagnostics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    codec: FrequencyCodec,
    timesteps: Optional[torch.Tensor] = None,
    *,
    phase_amplitude_gate: float = 1e-3,
    eps: float = 1e-8,
) -> ScalarMetrics:
    """Compare predicted and target normalized x0 tokens.

    Aggregate metrics are accompanied by ``radius/<bin>/...`` metrics. If
    ``timesteps`` is supplied, aggregate summaries for each observed timestep
    are emitted under ``timestep/<value>/...``.
    """
    _validate_pair(predicted, target, codec)
    if phase_amplitude_gate < 0:
        raise ValueError("phase_amplitude_gate must be non-negative")
    timesteps = _validate_timesteps(timesteps, predicted.shape[0], predicted.device)

    predicted_physical = normalized_to_physical(predicted, codec)
    target_physical = normalized_to_physical(target, codec)
    metrics = _spectral_summary(
        predicted,
        target,
        predicted_physical,
        target_physical,
        codec,
        phase_amplitude_gate,
        eps,
    )

    radial_errors = []
    for radius in torch.unique(codec.radius_bin, sorted=True):
        selection = codec.radius_bin == radius
        radius_metrics = _spectral_summary(
            predicted,
            target,
            predicted_physical,
            target_physical,
            codec,
            phase_amplitude_gate,
            eps,
            orbit_selection=selection,
        )
        prefix = f"radius/{int(radius.item())}"
        for name, value in radius_metrics.items():
            metrics[f"{prefix}/{name}"] = value
        radial_errors.append(radius_metrics["radial_power_relative_error"])
    if radial_errors:
        metrics["radial_power_relative_error"] = torch.stack(radial_errors).mean()

    if timesteps is not None:
        for timestep in torch.unique(timesteps, sorted=True):
            batch_selection = timesteps == timestep
            timestep_metrics = _spectral_summary(
                predicted[batch_selection],
                target[batch_selection],
                predicted_physical[batch_selection],
                target_physical[batch_selection],
                codec,
                phase_amplitude_gate,
                eps,
            )
            prefix = f"timestep/{_timestep_label(timestep)}"
            for name, value in timestep_metrics.items():
                metrics[f"{prefix}/{name}"] = value
    return metrics


def _codec_physical_mean(codec: FrequencyCodec) -> torch.Tensor:
    transformed_mean = (
        codec.orbit_mean
        if codec.uses_orbit_statistics
        else codec.bin_mean[codec.radius_bin]
    )
    physical_mean = codec.invert_value_transform(transformed_mean.unsqueeze(0))[0]
    physical_mean = physical_mean.clone()
    physical_mean[..., 3:] *= (~codec.is_self_conjugate).to(
        dtype=physical_mean.dtype
    )[:, None]
    return physical_mean


def _phase_distortion_summary(
    physical_tokens: torch.Tensor,
    physical_mean: torch.Tensor,
    codec: FrequencyCodec,
    amplitude_gate: float,
    dominance_c: float,
    quantiles: Sequence[float],
    eps: float,
) -> ScalarMetrics:
    values = _complex(physical_tokens)
    means = _complex(physical_mean)[None]
    non_self = (~codec.is_self_conjugate)[None, :, None].expand_as(values)
    amplitude = values.abs()
    mean_amplitude = means.abs().expand_as(amplitude)

    ratios = (mean_amplitude / (amplitude + eps))[non_self]
    metrics: ScalarMetrics = {}
    for quantile in quantiles:
        metrics[f"mu_over_z_{_quantile_label(quantile)}"] = torch.quantile(
            ratios.float(), quantile
        )
    metrics[f"mean_dominance_fraction_c{dominance_c:g}"] = (
        (amplitude <= dominance_c * mean_amplitude)[non_self].float().mean()
    )

    centered = values - means
    phase_delta = torch.angle(centered * values.conj())
    valid = non_self & (amplitude > amplitude_gate)
    weights = valid.to(dtype=values.real.dtype)
    if bool(valid.any()):
        metrics["phase_distortion_circular_error"] = _weighted_mean(
            1.0 - torch.cos(phase_delta), weights, eps
        )
        metrics["phase_distortion_resultant_length"] = _weighted_resultant(
            phase_delta, weights, eps
        )
    else:
        metrics["phase_distortion_circular_error"] = values.real.new_tensor(float("nan"))
        metrics["phase_distortion_resultant_length"] = values.real.new_tensor(float("nan"))
    metrics["phase_distortion_valid_fraction"] = weights.sum() / non_self.sum().clamp_min(
        1
    )
    return metrics


@torch.no_grad()
def compute_normalization_phase_distortion(
    physical_tokens: torch.Tensor,
    codec: FrequencyCodec,
    timesteps: Optional[torch.Tensor] = None,
    *,
    amplitude_gate: float = 1e-3,
    dominance_c: float = 1.0,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99),
    eps: float = 1e-8,
) -> ScalarMetrics:
    """Measure phase changes caused by subtracting the codec mean in raw space."""
    _validate_tokens(physical_tokens, codec, "physical_tokens")
    if amplitude_gate < 0 or dominance_c < 0:
        raise ValueError("amplitude_gate and dominance_c must be non-negative")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    timesteps = _validate_timesteps(
        timesteps, physical_tokens.shape[0], physical_tokens.device
    )
    physical_mean = _codec_physical_mean(codec).to(
        device=physical_tokens.device, dtype=physical_tokens.dtype
    )
    metrics = _phase_distortion_summary(
        physical_tokens,
        physical_mean,
        codec,
        amplitude_gate,
        dominance_c,
        quantiles,
        eps,
    )
    if timesteps is not None:
        for timestep in torch.unique(timesteps, sorted=True):
            selected = timesteps == timestep
            subset = _phase_distortion_summary(
                physical_tokens[selected],
                physical_mean,
                codec,
                amplitude_gate,
                dominance_c,
                quantiles,
                eps,
            )
            prefix = f"timestep/{_timestep_label(timestep)}"
            for name, value in subset.items():
                metrics[f"{prefix}/{name}"] = value
    return metrics


def _perturbation_summary(
    tokens: torch.Tensor,
    noise: torch.Tensor,
    perturbation: torch.Tensor,
    mask: torch.Tensor,
    quantiles: Iterable[float],
    eps: float,
) -> ScalarMetrics:
    expanded_mask = mask.expand_as(tokens)
    count = expanded_mask.sum().clamp_min(1)
    metrics = {
        "token_rms": torch.sqrt((tokens.square() * expanded_mask).sum() / count),
        "noise_rms": torch.sqrt((noise.square() * expanded_mask).sum() / count),
        "perturbation_rms": torch.sqrt(
            (perturbation.square() * expanded_mask).sum() / count
        ),
    }
    dimensions = tuple(range(1, tokens.ndim))
    token_norm = torch.sqrt((tokens.square() * expanded_mask).sum(dim=dimensions))
    perturbation_norm = torch.sqrt(
        (perturbation.square() * expanded_mask).sum(dim=dimensions)
    )
    norm_ratio = perturbation_norm / (token_norm + eps)
    for quantile in quantiles:
        metrics[
            f"perturbation_to_token_norm_{_quantile_label(quantile)}"
        ] = torch.quantile(norm_ratio.float(), quantile)
    return metrics


@torch.no_grad()
def compute_perturbation_diagnostics(
    tokens: torch.Tensor,
    noise: torch.Tensor,
    perturbation: torch.Tensor,
    codec: Optional[FrequencyCodec] = None,
    timesteps: Optional[torch.Tensor] = None,
    *,
    quantiles: Sequence[float] = (0.5, 0.9, 0.99),
    eps: float = 1e-8,
) -> ScalarMetrics:
    """Report token/noise RMS and per-example perturbation/token norm ratios."""
    if tokens.ndim != 3 or tokens.shape[-1] != 6:
        raise ValueError(f"tokens must have shape [B,L,6], got {tuple(tokens.shape)}")
    if noise.shape != tokens.shape or perturbation.shape != tokens.shape:
        raise ValueError("noise and perturbation must have the same shape as tokens")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    if codec is None:
        mask = torch.ones(
            1, tokens.shape[1], tokens.shape[2], device=tokens.device, dtype=tokens.dtype
        )
    else:
        _validate_tokens(tokens, codec, "tokens")
        mask = codec.component_mask.to(device=tokens.device, dtype=tokens.dtype)[None]
    timesteps = _validate_timesteps(timesteps, tokens.shape[0], tokens.device)
    metrics = _perturbation_summary(tokens, noise, perturbation, mask, quantiles, eps)
    if timesteps is not None:
        for timestep in torch.unique(timesteps, sorted=True):
            selected = timesteps == timestep
            subset = _perturbation_summary(
                tokens[selected],
                noise[selected],
                perturbation[selected],
                mask,
                quantiles,
                eps,
            )
            prefix = f"timestep/{_timestep_label(timestep)}"
            for name, value in subset.items():
                metrics[f"{prefix}/{name}"] = value
    return metrics
