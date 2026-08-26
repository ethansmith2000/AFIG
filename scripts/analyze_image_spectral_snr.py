#!/usr/bin/env python3
"""Measure CIFAR frequency/color-mode SNR crossings under the pixel-flow path.

Images are standardized exactly like the matched pixel prior, then transformed
with an orthonormal 2-D FFT.  At every frequency we estimate the complex 3x3
RGB cross-spectral covariance.  Its eigenvalues are signal variances along
frequency-specific color eigendirections; isotropic pixel noise has variance 1
along each unit direction and remains unit variance under the orthonormal FFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision


BANDS = (
    ("r1-2", 0.5, 2.5),
    ("r3-4", 2.5, 4.5),
    ("r5-6", 4.5, 6.5),
    ("r7-8", 6.5, 8.5),
    ("r9-12", 8.5, 12.5),
    ("r13-16", 12.5, 16.5),
)


def _crossing(variance: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + variance.clamp_min(0.0).sqrt())


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95], dtype=values.dtype
    )
    result = torch.quantile(values, probabilities)
    return {
        label: float(value)
        for label, value in zip(("p05", "p25", "p50", "p75", "p95"), result)
    }


def _batches(images: torch.Tensor, batch_size: int):
    for start in range(0, images.shape[0], batch_size):
        yield images[start : start + batch_size]


def _standardized_fft(
    images: torch.Tensor, mean: float, scale: float
) -> torch.Tensor:
    values = images.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    values = (values - mean) / scale
    return torch.fft.fft2(values, norm="ortho")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument(
        "--pixel_cache_metadata",
        default="/workspace/AFIG/pixel_runs/v8-cifar10-patches-original-flip.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    metadata = json.loads(Path(args.pixel_cache_metadata).read_text())
    global_mean = float(metadata["global_mean"])
    global_scale = float(metadata["global_std"])
    if not global_scale > 0:
        raise ValueError("pixel cache scale must be positive")

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=False
    )
    images = torch.from_numpy(dataset.data)
    count, height, width, channels = images.shape
    if (height, width, channels) != (32, 32, 3):
        raise ValueError(f"unexpected CIFAR shape: {tuple(images.shape)}")

    # First pass: population Fourier mean, including frequency-specific color
    # means.  Removing it ensures SNR measures sample-dependent content.
    fourier_sum = torch.zeros(channels, height, width, dtype=torch.complex128)
    for batch in _batches(images, args.batch_size):
        fourier_sum += _standardized_fft(batch, global_mean, global_scale).sum(
            dim=0, dtype=torch.complex128
        )
    fourier_mean = fourier_sum / count

    # Second pass: exact per-frequency RGB cross-spectral covariance and
    # per-sample radial energy for schedule-consistency measurements.
    covariance_sum = torch.zeros(
        height, width, channels, channels, dtype=torch.complex128
    )
    frequency = torch.fft.fftfreq(height) * height
    fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
    radius = (fy.square() + fx.square()).sqrt()
    masks = [(radius > lo) & (radius <= hi) for _, lo, hi in BANDS]
    sample_band_energy = []
    for batch in _batches(images, args.batch_size):
        centered = (
            _standardized_fft(batch, global_mean, global_scale)
            - fourier_mean[None]
        )
        covariance_sum += torch.einsum(
            "bchw,bdhw->hwcd", centered, centered.conj()
        ).to(torch.complex128)
        # Average over RGB directions and FFT bins.  Unit pixel noise has
        # expected energy 1 under exactly this normalization.
        power_per_direction = centered.abs().square().mean(dim=1)
        sample_band_energy.append(
            torch.stack(
                [power_per_direction[:, mask].mean(dim=1) for mask in masks],
                dim=1,
            ).float()
        )
    sample_band_energy = torch.cat(sample_band_energy)

    covariance = covariance_sum / count
    eigenvalues = torch.linalg.eigvalsh(covariance).real.clamp_min(0.0)
    # eigvalsh is ascending: mode 1 is strongest, mode 3 weakest.
    eigenvalues = eigenvalues.flip(-1)
    average_direction_variance = eigenvalues.mean(dim=-1)

    radial_bands = []
    for band_index, ((label, low, high), mask) in enumerate(zip(BANDS, masks)):
        mode_variance = eigenvalues[mask].mean(dim=0)
        average_variance = average_direction_variance[mask].mean()
        per_sample_energy = sample_band_energy[:, band_index]
        radial_bands.append(
            {
                "label": label,
                "radius": [low, high],
                "fft_bins": int(mask.sum()),
                "population_variance_per_rgb_direction": float(average_variance),
                "population_snr1_t": float(_crossing(average_variance)),
                "color_eigenvalues_descending": [
                    float(value) for value in mode_variance
                ],
                "color_mode_snr1_t": [
                    float(value) for value in _crossing(mode_variance)
                ],
                "top_color_mode_variance_share": float(
                    mode_variance[0] / mode_variance.sum().clamp_min(1e-30)
                ),
                "per_sample_variance_quantiles": _quantiles(per_sample_energy),
                "per_sample_snr1_t_quantiles": _quantiles(
                    _crossing(per_sample_energy)
                ),
            }
        )

    adjacent_order = [
        float(
            (sample_band_energy[:, index] > sample_band_energy[:, index + 1])
            .float()
            .mean()
        )
        for index in range(len(BANDS) - 1)
    ]
    log_band_energy = sample_band_energy.clamp_min(1e-12).log10()
    band_correlation = torch.corrcoef(log_band_energy.T)

    # Preserve exact 32x32 population maps for later radial/oriented figures.
    exact_frequency_maps = {
        "radius": radius.tolist(),
        "average_rgb_direction_variance": average_direction_variance.tolist(),
        "color_eigenvalues_descending": eigenvalues.tolist(),
        "average_rgb_direction_snr1_t": _crossing(
            average_direction_variance
        ).tolist(),
        "color_mode_snr1_t": _crossing(eigenvalues).tolist(),
    }
    result = {
        "dataset": "CIFAR-10 train original images",
        "examples": count,
        "normalization": {
            "type": "matched_pixel_prior_tensor_wide_population",
            "mean": global_mean,
            "scale": global_scale,
        },
        "definition": {
            "path": "z_t = (1-t) eps + t z",
            "fft": "orthonormal 2-D FFT",
            "noise_variance_per_unit_rgb_frequency_direction": 1.0,
            "signal": "sample-dependent Fourier coefficient after population mean removal",
            "snr": "t^2 * directional_variance / (1-t)^2",
        },
        "radial_bands": radial_bands,
        "per_sample_schedule_consistency": {
            "adjacent_descending_probability": adjacent_order,
            "log_energy_correlation": band_correlation.tolist(),
        },
        "exact_frequency_maps": exact_frequency_maps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
