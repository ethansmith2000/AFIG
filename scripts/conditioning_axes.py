#!/usr/bin/env python3
"""Measure the two properties that make a wide SNR spectrum *useful*.

A wide crossing spread is nearly automatic for any anisotropic data and settles
nothing. What matters for diffusability is:

  Axis A -- consistency: does an individual sample resolve in the population's
  order? If per-sample spectra deviate wildly from the population spectrum, the
  schedule is only right on average and carries no per-sample information.

  Axis B -- conditioning gain: do the directions that clear the noise floor
  early actually help predict the ones that clear it later? In the population
  eigenbasis the directions are uncorrelated *by construction*, so the entire
  linear conditioning gain is exactly zero and any real value is higher-order.
  Energy (squared-magnitude) correlation is therefore a clean probe: it is
  exactly 0 for a Gaussian of any covariance, and strongly positive for natural
  images, whose scale-mixture structure is what makes coarse-to-fine
  conditioning pay.

Reports both for a latent cache and, as the reference, for CIFAR-10 pixels.
"""

from __future__ import annotations

import argparse
import json

import torch


def eigenbasis(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.double()
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / values.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return values @ eigenvectors[:, order], eigenvalues[order].clamp_min(1e-12)


def axis_a(rotated: torch.Tensor, eigenvalues: torch.Tensor) -> dict:
    """Per-sample resolving order vs the population order."""

    energy = rotated.square()
    population = eigenvalues
    # per-sample Spearman of direction energies against the population spectrum
    sample = energy[: min(2000, energy.shape[0])]
    rank_population = torch.argsort(torch.argsort(population, descending=True)).double()
    correlations = []
    for row in sample:
        rank_row = torch.argsort(torch.argsort(row, descending=True)).double()
        a = rank_row - rank_row.mean()
        b = rank_population - rank_population.mean()
        correlations.append(float((a @ b) / (a.norm() * b.norm())))
    correlations = torch.tensor(correlations)
    # dispersion of per-sample energy about the population value, in log units
    log_ratio = (energy.clamp_min(1e-12) / population[None, :]).log()
    return {
        "per_sample_order_correlation_mean": float(correlations.mean()),
        "per_sample_order_correlation_std": float(correlations.std()),
        "log_energy_dispersion_median": float(log_ratio.std(dim=0).median()),
    }


def axis_b(rotated: torch.Tensor, bands: list[tuple[int, int]]) -> dict:
    """Higher-order conditioning gain: energy correlation in the eigenbasis.

    Zero for any Gaussian, because eigen-directions are uncorrelated and a
    Gaussian's squared magnitudes inherit no dependence from zero covariance.
    """

    energy = rotated.square()
    energy = energy - energy.mean(dim=0, keepdim=True)
    energy = energy / energy.std(dim=0, keepdim=True).clamp_min(1e-12)
    samples = energy.shape[0]
    correlation = energy.T @ energy / samples
    dimension = correlation.shape[0]
    off = ~torch.eye(dimension, dtype=torch.bool)
    noise_floor = (1.0 / samples) ** 0.5  # finite-sample null scale

    band_gain = {}
    for a, b in bands:
        for c, d in bands:
            if (a, b) >= (c, d):
                continue
            block = correlation[a:b, c:d]
            band_gain[f"{a}-{b} vs {c}-{d}"] = round(float(block.mean()), 4)

    return {
        "energy_correlation_offdiagonal_mean": float(correlation[off].mean()),
        "energy_correlation_offdiagonal_absmean": float(correlation[off].abs().mean()),
        "finite_sample_null_scale": noise_floor,
        "signal_to_null_ratio": float(correlation[off].abs().mean() / noise_floor),
        "band_energy_correlation": band_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--cifar_pixels", action="store_true")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.cifar_pixels:
        import torchvision
        import torchvision.transforms as transforms

        dataset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=False,
            transform=transforms.ToTensor(),
        )
        flat = torch.stack(
            [dataset[i][0] for i in range(min(args.max_samples, len(dataset)))]
        ).reshape(-1, 3 * 32 * 32)
        label = "cifar10-pixels"
    else:
        payload = torch.load(args.cache, map_location="cpu", weights_only=False)
        latents = payload["train_latents"][: args.max_samples].float()
        flat = latents.reshape(latents.shape[0], -1)
        label = args.cache

    rotated, eigenvalues = eigenbasis(flat)
    dimension = rotated.shape[1]
    bands = [(0, 8), (8, 32), (32, 128), (128, dimension // 2), (dimension // 2, dimension)]
    result = {
        "source": label,
        "samples": rotated.shape[0],
        "dimensions": dimension,
        "crossing_spread": float(
            (1 / (1 + (eigenvalues / eigenvalues.mean()).sqrt())).max()
            - (1 / (1 + (eigenvalues / eigenvalues.mean()).sqrt())).min()
        ),
        "axis_a_consistency": axis_a(rotated, eigenvalues),
        "axis_b_conditioning": axis_b(rotated, bands),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        open(args.output, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
