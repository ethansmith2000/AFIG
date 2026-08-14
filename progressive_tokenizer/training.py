"""Training utilities for the progressive tokenizer."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


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
