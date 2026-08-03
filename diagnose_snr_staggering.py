"""How much natural coarse-to-fine ordering does each representation provide?

Under isotropic diffusion noise, an eigendirection with variance lambda crosses
SNR = 1 at t* = 1 / (1 + sqrt(lambda)) (rectified flow, x_t = t*x + (1-t)*eps).
A wide spread of t* across directions means the representation resolves coarse
structure early and detail late *by itself* -- the natural schedule that makes
pixel-space image diffusion work.  All directions piling up at one t* means the
model must decide everything simultaneously.

The subtlety this measures: per-dimension normalization sets the covariance
*diagonal* to 1 but does not flatten the eigenvalue spectrum, and SNR staggering
is governed by eigenvalues rather than the coordinate basis.  For raw FFT the two
coincide (frequencies are near-decorrelated, so frequency axes are approximately
eigendirections and per-frequency whitening genuinely flattens the spectrum).
For AE latents it may not.

Compares three spaces:
  a) raw FFT tokens, pre-whitening
  b) raw AE latents, pre-normalization
  c) normalized AE latents, i.e. what the generative models actually see
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms

from latent_autoencoder_interface import FrozenLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def spectrum_stats(matrix: torch.Tensor, label: str) -> Dict[str, object]:
    """Eigen-spectrum of the covariance and the induced t* distribution.

    t* is reported two ways: unweighted across directions, and weighted by each
    direction's share of total variance (directions carrying no variance cannot
    influence a sample, so the weighted view is the meaningful one).
    """
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(1e-30)

    t_star = 1.0 / (1.0 + eigenvalues.sqrt())
    order = torch.argsort(t_star)
    sorted_t = t_star[order]
    weights = eigenvalues[order] / total
    cumulative = torch.cumsum(weights, dim=0)

    def weighted_quantile(q: float) -> float:
        target = torch.tensor(q, dtype=cumulative.dtype, device=cumulative.device)
        index = int(torch.searchsorted(cumulative, target))
        return float(sorted_t[min(index, sorted_t.numel() - 1)])

    positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-10]
    return {
        "label": label,
        "dims": int(eigenvalues.numel()),
        "eigenvalue_max": float(eigenvalues.max()),
        "eigenvalue_min_positive": float(positive.min()) if positive.numel() else 0.0,
        "eigenvalue_dynamic_range": float(
            eigenvalues.max() / positive.min().clamp_min(1e-30)
        )
        if positive.numel()
        else 0.0,
        "participation_ratio": float(total**2 / (eigenvalues**2).sum().clamp_min(1e-30)),
        "diagonal_dynamic_range": float(
            covariance.diagonal().max() / covariance.diagonal().clamp_min(1e-30).min()
        ),
        "t_star_variance_weighted_p05": weighted_quantile(0.05),
        "t_star_variance_weighted_p25": weighted_quantile(0.25),
        "t_star_variance_weighted_p50": weighted_quantile(0.50),
        "t_star_variance_weighted_p75": weighted_quantile(0.75),
        "t_star_variance_weighted_p95": weighted_quantile(0.95),
        "t_star_spread_p05_p95": weighted_quantile(0.95) - weighted_quantile(0.05),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)
    codec = interface.codec
    autoencoder = interface.autoencoder

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    count = min(args.num_images, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    raw_fft: List[torch.Tensor] = []
    raw_latents: List[torch.Tensor] = []
    normalized: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device).float()
        raw_fft.append(codec.encode_raw(images))
        tokens = codec.encode(images)
        latents = autoencoder.export_latents(tokens, sample_posterior=False)["latents"]
        raw_latents.append(latents)
        normalized.append(interface.normalize(latents))

    mask = codec.component_mask.to(device).bool().reshape(-1)
    fft_matrix = torch.cat(raw_fft, dim=0).reshape(count, -1)[:, mask].double()
    latent_matrix = torch.cat(raw_latents, dim=0).reshape(count, -1).double()
    normalized_matrix = torch.cat(normalized, dim=0).reshape(count, -1).double()
    del raw_fft, raw_latents, normalized
    print(f"raw FFT {tuple(fft_matrix.shape)}, raw latents {tuple(latent_matrix.shape)}")

    results = [
        spectrum_stats(fft_matrix, "raw_fft_prewhitening"),
        spectrum_stats(latent_matrix, "raw_latents_prenormalization"),
        spectrum_stats(normalized_matrix, "normalized_latents_current"),
    ]

    path = os.path.join(args.output_dir, "staggering_report.json")
    with open(path, "w") as handle:
        json.dump({"num_images": count, "spaces": results}, handle, indent=2)

    print(f"\n{'space':<32} {'dims':>5} {'eig range':>11} {'diag range':>11} {'PR':>7}")
    for row in results:
        print(
            f"{row['label']:<32} {row['dims']:>5d} {row['eigenvalue_dynamic_range']:>11.3e}"
            f" {row['diagonal_dynamic_range']:>11.3e} {row['participation_ratio']:>7.1f}"
        )
    print(f"\n{'space':<32} {'t*p05':>7} {'t*p25':>7} {'t*p50':>7} {'t*p75':>7} {'t*p95':>7} {'spread':>8}")
    for row in results:
        print(
            f"{row['label']:<32} {row['t_star_variance_weighted_p05']:>7.3f}"
            f" {row['t_star_variance_weighted_p25']:>7.3f}"
            f" {row['t_star_variance_weighted_p50']:>7.3f}"
            f" {row['t_star_variance_weighted_p75']:>7.3f}"
            f" {row['t_star_variance_weighted_p95']:>7.3f}"
            f" {row['t_star_spread_p05_p95']:>8.3f}"
        )
    print("\n(t* = time at which a direction crosses SNR=1; wider spread = more")
    print(" natural coarse-to-fine ordering. Variance-weighted percentiles.)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
