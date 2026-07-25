"""Checkpoint-free streaming evaluation for CIFAR-10 AFIG runs."""

from __future__ import annotations

import os
import fcntl
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


@dataclass
class StreamingMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.total = torch.zeros(self.dimension, dtype=torch.float64)
        self.cross = torch.zeros(self.dimension, self.dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64)
        self.count += values.shape[0]
        self.total += values.sum(dim=0)
        self.cross += values.T @ values

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise ValueError("At least two feature vectors are required.")
        mean = self.total / self.count
        covariance = (self.cross - self.count * torch.outer(mean, mean)) / (
            self.count - 1
        )
        return mean.float(), covariance.float()


class InceptionFeatures:
    """torch-fidelity's canonical FID Inception feature extractor."""

    def __init__(self, device: torch.device):
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
        )

        self.model = FeatureExtractorInceptionV3(
            name="inception-v3-compat",
            features_list=["2048"],
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        uint8 = images.clamp(0, 1).mul(255).round().to(torch.uint8)
        return self.model(uint8.to(self.device))[0].float()


def _fid(
    real_mean: torch.Tensor,
    real_covariance: torch.Tensor,
    generated_mean: torch.Tensor,
    generated_covariance: torch.Tensor,
) -> float:
    from scipy import linalg

    mu1 = real_mean.double().numpy()
    mu2 = generated_mean.double().numpy()
    sigma1 = real_covariance.double().numpy()
    sigma2 = generated_covariance.double().numpy()
    covariance_mean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covariance_mean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = mu1 - mu2
    return float(
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covariance_mean)
    )


def _kid(
    real: torch.Tensor,
    generated: torch.Tensor,
    subsets: int = 50,
    subset_size: int = 1000,
) -> float:
    real = real.float().cpu()
    generated = generated.float().cpu()
    subset_size = min(subset_size, real.shape[0], generated.shape[0])
    if subset_size < 2:
        return float("nan")
    generator = torch.Generator().manual_seed(123)
    estimates = []
    dimension = real.shape[1]
    for _ in range(subsets):
        x = real[torch.randperm(real.shape[0], generator=generator)[:subset_size]]
        y = generated[
            torch.randperm(generated.shape[0], generator=generator)[:subset_size]
        ]
        kernel_x = (x @ x.T / dimension + 1.0).pow(3)
        kernel_y = (y @ y.T / dimension + 1.0).pow(3)
        kernel_xy = (x @ y.T / dimension + 1.0).pow(3)
        diagonal_x = torch.diagonal(kernel_x).sum()
        diagonal_y = torch.diagonal(kernel_y).sum()
        m = float(subset_size)
        estimate = (
            (kernel_x.sum() - diagonal_x) / (m * (m - 1.0))
            + (kernel_y.sum() - diagonal_y) / (m * (m - 1.0))
            - 2.0 * kernel_xy.mean()
        )
        estimates.append(estimate)
    return torch.stack(estimates).mean().item()


def _radial_power(images: torch.Tensor, codec) -> torch.Tensor:
    spectrum = torch.fft.fft2(images.float(), norm="ortho")
    power = spectrum.real.square() + spectrum.imag.square()
    representatives = power[:, :, codec.ky, codec.kx].mean(dim=(0, 1))
    bins = codec.radius_bin
    radial = torch.zeros(codec.num_bins, device=images.device)
    counts = torch.zeros(codec.num_bins, device=images.device)
    radial.scatter_add_(0, bins, representatives)
    counts.scatter_add_(0, bins, torch.ones_like(representatives))
    return radial / counts.clamp_min(1.0)


def _orbit_moments(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = tokens.double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = torch.einsum("bli,blj->lij", centered, centered) / max(
        values.shape[0] - 1, 1
    )
    return mean.float(), covariance.float()


def _codec_reference_fingerprint(codec) -> Dict[str, Any]:
    return {
        key: getattr(codec.config, key)
        for key in ("height", "width", "ordering", "fft_norm")
    }


@torch.no_grad()
def build_reference_cache(
    loader: Iterable,
    codec,
    extractor: InceptionFeatures,
    path: str,
    max_samples: int = 50_000,
    kid_samples: int = 5_000,
) -> Dict[str, Any]:
    moments = StreamingMoments(2048)
    kid_features = []
    image_sum = torch.zeros(3, dtype=torch.float64)
    image_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    radial_total = torch.zeros(codec.num_bins, dtype=torch.float64)
    orbit_sum = torch.zeros(codec.seq_len, 6, dtype=torch.float64)
    orbit_cross = torch.zeros(codec.seq_len, 6, 6, dtype=torch.float64)
    seen = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images[: max_samples - seen].to(extractor.device)
        if images.shape[0] == 0:
            break
        features = extractor(images)
        moments.update(features)
        if sum(item.shape[0] for item in kid_features) < kid_samples:
            remaining = kid_samples - sum(item.shape[0] for item in kid_features)
            kid_features.append(features[:remaining].cpu())
        image_sum += images.double().sum(dim=(0, 2, 3)).cpu()
        image_sum_sq += images.double().square().sum(dim=(0, 2, 3)).cpu()
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
        radial_total += _radial_power(images, codec).double().cpu() * images.shape[0]
        raw = codec.encode_raw(images).double()
        orbit_sum += raw.sum(dim=0).cpu()
        orbit_cross += torch.einsum("bli,blj->lij", raw, raw).cpu()
        seen += images.shape[0]
        if seen >= max_samples:
            break
    mean, covariance = moments.compute()
    orbit_mean = orbit_sum / seen
    orbit_covariance = (
        orbit_cross - seen * torch.einsum("li,lj->lij", orbit_mean, orbit_mean)
    ) / max(seen - 1, 1)
    payload = {
        "version": 1,
        "samples": seen,
        "codec_config": _codec_reference_fingerprint(codec),
        "feature_mean": mean,
        "feature_covariance": covariance,
        "kid_features": torch.cat(kid_features, dim=0),
        "channel_mean": (image_sum / pixel_count).float(),
        "channel_std": (
            image_sum_sq / pixel_count - (image_sum / pixel_count).square()
        )
        .clamp_min(0)
        .sqrt()
        .float(),
        "radial_power": (radial_total / seen).float(),
        "orbit_mean": orbit_mean.float(),
        "orbit_covariance": orbit_covariance.float(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return payload


@torch.no_grad()
def evaluate_live(
    model,
    reference_loader: Iterable,
    num_samples: int,
    batch_size: int,
    reference_cache_path: str,
    output_dir: str,
    num_inference_steps: int,
    reference_samples: int = 50_000,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    codec = model.codec
    extractor = InceptionFeatures(device)
    os.makedirs(os.path.dirname(reference_cache_path) or ".", exist_ok=True)
    with open(f"{reference_cache_path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(reference_cache_path):
            reference = torch.load(reference_cache_path, map_location="cpu")
            if reference.get("codec_config") != _codec_reference_fingerprint(codec):
                raise ValueError("Reference cache codec configuration is incompatible.")
        else:
            reference = build_reference_cache(
                reference_loader,
                codec,
                extractor,
                reference_cache_path,
                max_samples=reference_samples,
            )

    feature_moments = StreamingMoments(2048)
    generated_features = []
    image_sum = torch.zeros(3, dtype=torch.float64)
    image_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    clip_count = 0
    value_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    gradient_total = 0.0
    radial_total = torch.zeros(codec.num_bins, dtype=torch.float64)
    token_batches = []
    generated = 0
    model.eval()
    while generated < num_samples:
        current = min(batch_size, num_samples - generated)
        output = model.generate(
            batch_size=current,
            num_inference_steps=num_inference_steps,
            return_tokens=True,
            progress=False,
        )
        images = output["images"].float()
        tokens = output["tokens"].float()
        features = extractor(images)
        feature_moments.update(features)
        generated_features.append(features.cpu())
        image_sum += images.double().sum(dim=(0, 2, 3)).cpu()
        image_sum_sq += images.double().square().sum(dim=(0, 2, 3)).cpu()
        pixel_count += current * images.shape[2] * images.shape[3]
        clip_count += ((images < 0) | (images > 1)).sum().item()
        value_count += images.numel()
        minimum = min(minimum, images.min().item())
        maximum = max(maximum, images.max().item())
        vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
        horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
        gradient_total += (
            0.5 * (vertical.square().mean() + horizontal.square().mean())
        ).item() * current
        radial_total += _radial_power(images, codec).double().cpu() * current
        token_batches.append(tokens.cpu())
        generated += current

    generated_mean, generated_covariance = feature_moments.compute()
    generated_feature_tensor = torch.cat(generated_features, dim=0)
    all_tokens = torch.cat(token_batches, dim=0)
    generated_raw = codec.invert_value_transform(
        codec.denormalize(all_tokens.to(device))
    ).cpu()
    orbit_mean, orbit_covariance = _orbit_moments(generated_raw)
    reference_orbit_mean = reference["orbit_mean"]
    reference_orbit_covariance = reference["orbit_covariance"]
    radial = radial_total / generated
    reference_radial = reference["radial_power"].double()
    radial_error = (
        (radial - reference_radial).abs()
        / reference_radial.abs().clamp_min(1e-8)
    ).mean()
    mean_error = (orbit_mean - reference_orbit_mean).square().mean().sqrt()
    covariance_error = (
        (orbit_covariance - reference_orbit_covariance).square().mean().sqrt()
    )

    residual = generated_raw - reference_orbit_mean.unsqueeze(0)
    residual_profile = {
        "version": 1,
        "samples": generated,
        "mean": residual.mean(dim=0),
        "rms": residual.square().mean(dim=0).sqrt(),
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(residual_profile, os.path.join(output_dir, "rollout_residual_profile.pt"))

    channel_mean = image_sum / pixel_count
    channel_std = (
        image_sum_sq / pixel_count - channel_mean.square()
    ).clamp_min(0).sqrt()
    metrics = {
        "final/fid_5k": _fid(
            reference["feature_mean"],
            reference["feature_covariance"],
            generated_mean,
            generated_covariance,
        ),
        "final/kid_5k": _kid(reference["kid_features"], generated_feature_tensor),
        "final/clipping_fraction": clip_count / max(value_count, 1),
        "final/unclipped_min": minimum,
        "final/unclipped_max": maximum,
        "final/image_gradient_energy": gradient_total / generated,
        "final/radial_power_relative_error": radial_error.item(),
        "final/orbit_mean_rmse": mean_error.item(),
        "final/orbit_covariance_rmse": covariance_error.item(),
    }
    for channel, name in enumerate(("r", "g", "b")):
        metrics[f"final/channel_mean_{name}"] = channel_mean[channel].item()
        metrics[f"final/channel_std_{name}"] = channel_std[channel].item()
    model.train()
    return metrics
