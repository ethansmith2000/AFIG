"""Training utilities for the progressive tokenizer."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def optimizer_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    """Apply decay only to Linear/Conv weights, never identities or norms."""

    decay_ids: set[int] = set()
    matrix_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    for module in model.modules():
        if isinstance(module, matrix_modules) and module.weight.requires_grad:
            decay_ids.add(id(module.weight))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if id(parameter) in decay_ids else no_decay).append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("optimizer grouping unexpectedly produced an empty group")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def pixel_psnr(normalized_mse: float) -> float:
    """PSNR for pixels represented in [-1, 1]."""

    if normalized_mse <= 0:
        return float("inf")
    return 10.0 * math.log10(4.0 / normalized_mse)


def marginal_kurtosis_penalty(latents: torch.Tensor) -> torch.Tensor:
    """Match each flattened latent coordinate's batch kurtosis to Gaussian 3."""

    coordinates = latents.float().reshape(latents.shape[0], -1)
    centered = coordinates - coordinates.mean(dim=0)
    second = centered.square().mean(dim=0)
    fourth = centered.pow(4).mean(dim=0)
    kurtosis = fourth / (second.square() + 1e-8)
    return (kurtosis - 3.0).square().mean()


def slot_variance_balance_penalty(latents: torch.Tensor) -> torch.Tensor:
    """Equalize sample-varying power across slots without fixing global scale."""

    centered = latents.float() - latents.float().mean(dim=0, keepdim=True)
    slot_power = centered.square().mean(dim=(0, 2))
    relative_power = slot_power / slot_power.mean().clamp_min(1e-8)
    return (relative_power - 1.0).square().mean()


def radial_log_power_reconstruction_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    relative_floor: float = 1e-3,
) -> torch.Tensor:
    """Match radial spectra with a signal-relative floor for low-SNR bands."""

    if target.ndim != 4 or reconstruction.shape != target.shape:
        raise ValueError("target and reconstruction must share [batch, channel, H, W]")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    target_fft = torch.fft.fft2(target.float(), norm="ortho")
    reconstruction_fft = torch.fft.fft2(reconstruction.float(), norm="ortho")
    reference_power = (
        target_fft.abs().square().mean(dim=(-2, -1)) * relative_floor + 1e-8
    )
    height, width = target.shape[-2:]
    vertical = torch.fft.fftfreq(height, device=target.device) * height
    horizontal = torch.fft.fftfreq(width, device=target.device) * width
    grid_vertical, grid_horizontal = torch.meshgrid(
        vertical, horizontal, indexing="ij"
    )
    radius = torch.floor(
        torch.sqrt(grid_vertical.square() + grid_horizontal.square())
    ).long()
    terms = []
    for band in torch.unique(radius):
        select = radius == band
        target_power = target_fft[..., select].abs().square().mean(dim=-1)
        reconstruction_power = (
            reconstruction_fft[..., select].abs().square().mean(dim=-1)
        )
        terms.append(
            (
                torch.log(reconstruction_power + reference_power)
                - torch.log(target_power + reference_power)
            ).abs()
        )
    return torch.stack(terms, dim=-1).mean()


def lpips_reconstruction_loss(
    perceptual_model: nn.Module,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Evaluate frozen LPIPS on [-1, 1] images, upsampling tiny smokes safely."""

    if target.ndim != 4 or reconstruction.shape != target.shape:
        raise ValueError("target and reconstruction must share [batch, channel, H, W]")
    target = target.float()
    reconstruction = reconstruction.float()
    if min(target.shape[-2:]) < 32:
        target = F.interpolate(target, size=(32, 32), mode="bilinear", align_corners=False)
        reconstruction = F.interpolate(
            reconstruction, size=(32, 32), mode="bilinear", align_corners=False
        )
    return perceptual_model(reconstruction, target, normalize=False).mean()


def gaussian_lowpass_pyramid_fft(
    images: torch.Tensor,
    sigmas: Iterable[float],
) -> torch.Tensor:
    """Return periodic Gaussian low-passes as [B, levels, C, H, W].

    ``sigma`` is measured in pixels. A zero-sigma final level is returned as
    the input exactly, which makes adjacent differences a telescoping DoG
    decomposition with no numerical residual at the endpoint.
    """

    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channel, H, W]")
    sigma_values = tuple(float(value) for value in sigmas)
    if not sigma_values:
        raise ValueError("at least one Gaussian sigma is required")
    if any(not math.isfinite(value) or value < 0 for value in sigma_values):
        raise ValueError("Gaussian sigmas must be finite and non-negative")

    values = images.float()
    height, width = values.shape[-2:]
    vertical = torch.fft.fftfreq(height, device=values.device)
    horizontal = torch.fft.rfftfreq(width, device=values.device)
    radius_squared = vertical[:, None].square() + horizontal[None, :].square()
    spectrum = torch.fft.rfft2(values, norm="ortho")
    levels = []
    for sigma in sigma_values:
        if sigma == 0:
            levels.append(values)
            continue
        response = torch.exp(
            -2.0 * math.pi**2 * sigma**2 * radius_squared
        )
        levels.append(
            torch.fft.irfft2(
                spectrum * response,
                s=(height, width),
                norm="ortho",
            )
        )
    return torch.stack(levels, dim=1)


class LatentMomentAccumulator:
    """Streaming global and covariance diagnostics for clean latent tokens."""

    def __init__(self, latent_dim: int, num_latents: int):
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.count = 0
        self.sum = torch.zeros(latent_dim, dtype=torch.float64)
        self.square_sum = torch.zeros(latent_dim, dtype=torch.float64)
        self.cross_sum = torch.zeros(latent_dim, latent_dim, dtype=torch.float64)
        self.slot_square_sum = torch.zeros(num_latents, dtype=torch.float64)
        self.image_count = 0
        self.peak_ratio_sum = 0.0
        self.token_count = 0

    @torch.no_grad()
    def update(self, latents: torch.Tensor) -> None:
        values = latents.detach().float().cpu()
        batch, tokens, width = values.shape
        if tokens != self.num_latents or width != self.latent_dim:
            raise ValueError("latent shape does not match accumulator")
        flat = values.reshape(-1, width).double()
        self.count += flat.shape[0]
        self.sum += flat.sum(dim=0)
        self.square_sum += flat.square().sum(dim=0)
        self.cross_sum += flat.T @ flat
        self.slot_square_sum += values.double().square().mean(dim=-1).sum(dim=0)
        self.image_count += batch
        rms = values.square().mean(dim=-1).sqrt().clamp_min(1e-12)
        peak = values.abs().amax(dim=-1)
        self.peak_ratio_sum += float((peak / rms).sum())
        self.token_count += batch * tokens

    def compute(self) -> dict:
        if self.count == 0 or self.image_count == 0:
            raise RuntimeError("no latent values accumulated")
        mean = self.sum / self.count
        second = self.square_sum / self.count
        variance = (second - mean.square()).clamp_min(0.0)
        covariance = self.cross_sum / self.count - torch.outer(mean, mean)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        total = float(eigenvalues.sum())
        if total > 0:
            probabilities = eigenvalues / total
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
            effective_rank = float(entropy.exp())
        else:
            effective_rank = 0.0
        slot_rms = (self.slot_square_sum / self.image_count).sqrt()
        return {
            "global_mean": float(mean.mean()),
            "global_std": float(
                (self.square_sum.sum() / (self.count * self.latent_dim)
                 - mean.mean().square()).clamp_min(0.0).sqrt()
            ),
            "coordinate_std_min": float(variance.sqrt().min()),
            "coordinate_std_median": float(variance.sqrt().median()),
            "coordinate_std_max": float(variance.sqrt().max()),
            "covariance_effective_rank": effective_rank,
            "mean_peak_to_rms": self.peak_ratio_sum / self.token_count,
            "slot_rms": slot_rms.tolist(),
        }


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)
