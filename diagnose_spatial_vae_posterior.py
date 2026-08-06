"""Paired posterior-mean and posterior-sample audit for a spatial VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from train_spatial_latent_hartley_ar import load_spatial_ae


def scalar_shape(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().flatten()
    centered = flat - flat.mean()
    variance = centered.square().mean().clamp_min(1e-12)
    levels = torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float32)
    quantiles = torch.quantile(flat.abs(), levels)
    return {
        "mean": float(flat.mean()),
        "std": float(centered.square().mean().sqrt()),
        "rms": float(flat.square().mean().sqrt()),
        "abs_p50": float(quantiles[0]),
        "abs_p90": float(quantiles[1]),
        "abs_p99": float(quantiles[2]),
        "abs_p99_9": float(quantiles[3]),
        "abs_max": float(flat.abs().max()),
        "skew": float(centered.pow(3).mean() / variance.pow(1.5)),
        "excess_kurtosis": float(centered.pow(4).mean() / variance.square() - 3.0),
    }


def coordinate_dependence(values: torch.Tensor) -> dict[str, float]:
    matrix = values.float().flatten(1)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0).sqrt().clamp_min(1e-6)
    standardized = centered / scale
    correlation = standardized.T @ standardized / matrix.shape[0]
    offdiag = correlation - torch.diag_embed(correlation.diagonal())
    covariance = centered.T @ centered / matrix.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).float().clamp_min(1e-12)
    return {
        "coordinate_count": int(matrix.shape[1]),
        "correlation_offdiag_rms": float(offdiag.square().mean().sqrt()),
        "covariance_condition": float(eigenvalues.max() / eigenvalues.min()),
        "covariance_eigen_p01": float(torch.quantile(eigenvalues, 0.01)),
        "covariance_eigen_p50": float(torch.quantile(eigenvalues, 0.50)),
        "covariance_eigen_p99": float(torch.quantile(eigenvalues, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    autoencoder = load_spatial_ae(args.checkpoint, device)
    if not autoencoder.config.variational:
        raise ValueError("Posterior audit requires a variational checkpoint")
    dataset = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    count = min(args.images, len(dataset))
    means, logvars, samples = [], [], []
    mean_squared_error = 0.0
    sample_squared_error = 0.0
    decode_junction_squared_error = 0.0
    pixel_count = 0
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            stop = min(start + args.batch_size, count)
            images = torch.stack([dataset[index][0] for index in range(start, stop)]).to(
                device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                mean, logvar = autoencoder.encode(images)
                sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
                mean_decoded = autoencoder.decode(mean)
                sample_decoded = autoencoder.decode(sample)
            means.append(mean.float().cpu())
            logvars.append(logvar.float().cpu())
            samples.append(sample.float().cpu())
            mean_squared_error += float(
                (mean_decoded.float() - images.float()).square().sum()
            )
            sample_squared_error += float(
                (sample_decoded.float() - images.float()).square().sum()
            )
            decode_junction_squared_error += float(
                (sample_decoded.float() - mean_decoded.float()).square().sum()
            )
            pixel_count += images.numel()

    mean = torch.cat(means)
    logvar = torch.cat(logvars)
    sample = torch.cat(samples)
    posterior_variance = logvar.exp()
    mean_matrix = mean.flatten(1)
    mean_coordinate_variance = mean_matrix.var(dim=0, unbiased=False).mean()
    expected_posterior_variance = posterior_variance.mean()
    expected_aggregate_variance = mean_coordinate_variance + expected_posterior_variance
    posterior_std = torch.exp(0.5 * logvar)
    std_levels = torch.tensor([0.5, 0.9, 0.99, 0.999])
    std_quantiles = torch.quantile(posterior_std.flatten(), std_levels)
    mean_mse = mean_squared_error / pixel_count
    sample_mse = sample_squared_error / pixel_count
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "images": count,
        "latent_shape": list(mean.shape[1:]),
        "mean": scalar_shape(mean),
        "sample": scalar_shape(sample),
        "posterior_std": {
            "rms": float(posterior_variance.mean().sqrt()),
            "p50": float(std_quantiles[0]),
            "p90": float(std_quantiles[1]),
            "p99": float(std_quantiles[2]),
            "p99_9": float(std_quantiles[3]),
            "max": float(posterior_std.max()),
        },
        "variance_decomposition": {
            "mean_coordinate_variance": float(mean_coordinate_variance),
            "expected_posterior_variance": float(expected_posterior_variance),
            "expected_aggregate_variance": float(expected_aggregate_variance),
            "posterior_fraction": float(
                expected_posterior_variance / expected_aggregate_variance.clamp_min(1e-12)
            ),
            "paired_sample_noise_rms": float((sample - mean).square().mean().sqrt()),
        },
        "mean_dependence": coordinate_dependence(mean),
        "sample_dependence": coordinate_dependence(sample),
        "decode": {
            "posterior_mean_mse": mean_mse,
            "posterior_mean_psnr": float(-10.0 * torch.log10(torch.tensor(mean_mse))),
            "posterior_sample_mse": sample_mse,
            "posterior_sample_psnr": float(-10.0 * torch.log10(torch.tensor(sample_mse))),
            "mean_vs_sample_mse": decode_junction_squared_error / pixel_count,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
