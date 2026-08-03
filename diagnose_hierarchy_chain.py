"""Where in the pipeline does the natural spectral hierarchy actually die?

Corrects two problems with diagnose_snr_staggering.py:

1. It omitted the *whitened* FFT tokens -- the codec output, which is what the AE
   actually consumes.  Without that row one cannot tell whether the hierarchy was
   destroyed by the codec whitening or by the autoencoder.
2. It measured "implicit loss weighting" in the coordinate basis (covariance
   diagonal) and "SNR staggering" in the eigenbasis (eigenvalues), then compared
   them.  Those are the same phenomenon -- variance heterogeneity -- viewed in two
   different bases, so the comparison was confounded.  Here both bases are
   reported for every space.

Under isotropic noise a direction with variance lambda crosses SNR=1 at
t* = 1/(1+sqrt(lambda)).  Reported as variance-weighted percentiles, which are
robust; raw min/max eigenvalue ratios are dominated by numerically-tiny
directions and are NOT reported as headline numbers.

Key structural point this is designed to expose: for natural images the FFT basis
is approximately the eigenbasis, so per-frequency whitening flattens the
*eigenspectrum* itself.  Per-pixel normalization would not, because pixels all
have similar variance yet are strongly correlated.
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


def weighted_percentiles(
    variances: torch.Tensor, quantiles: List[float]
) -> Dict[str, float]:
    """Variance-weighted percentiles of t* = 1/(1+sqrt(var)) over directions."""
    variances = variances.double().clamp_min(0.0)
    total = variances.sum().clamp_min(1e-30)
    t_star = 1.0 / (1.0 + variances.sqrt())
    order = torch.argsort(t_star)
    sorted_t = t_star[order]
    cumulative = torch.cumsum(variances[order] / total, dim=0)
    out: Dict[str, float] = {}
    for q in quantiles:
        target = torch.tensor(q, dtype=cumulative.dtype, device=cumulative.device)
        index = int(torch.searchsorted(cumulative, target))
        out[f"p{int(q * 100):02d}"] = float(sorted_t[min(index, sorted_t.numel() - 1)])
    return out


def analyze(matrix: torch.Tensor, label: str) -> Dict[str, object]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    diagonal = covariance.diagonal().clamp_min(0.0)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0.0)

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    coordinate = weighted_percentiles(diagonal, quantiles)
    eigen = weighted_percentiles(eigenvalues, quantiles)

    def ratio(values: torch.Tensor) -> float:
        values = values.double()
        total = values.sum().clamp_min(1e-30)
        order = torch.argsort(values, descending=True)
        cumulative = torch.cumsum(values[order] / total, dim=0)
        # robust spread: 95th vs 5th percentile of the variance-carrying directions
        high = float(values[order][max(int(0.05 * values.numel()) - 1, 0)])
        low = float(values[order][min(int(0.95 * values.numel()), values.numel() - 1)])
        return high / max(low, 1e-30)

    return {
        "label": label,
        "dims": int(diagonal.numel()),
        "coordinate_p05_p95_variance_ratio": ratio(diagonal),
        "eigen_p05_p95_variance_ratio": ratio(eigenvalues),
        "participation_ratio": float(
            eigenvalues.sum() ** 2 / (eigenvalues**2).sum().clamp_min(1e-30)
        ),
        "coordinate_t_star": coordinate,
        "eigen_t_star": eigen,
        "coordinate_t_star_spread": coordinate["p95"] - coordinate["p05"],
        "eigen_t_star_spread": eigen["p95"] - eigen["p05"],
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

    pixels: List[torch.Tensor] = []
    raw_fft: List[torch.Tensor] = []
    whitened_fft: List[torch.Tensor] = []
    raw_latents: List[torch.Tensor] = []
    normalized: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device).float()
        pixels.append(images.reshape(images.shape[0], -1))
        raw_fft.append(codec.encode_raw(images))
        tokens = codec.encode(images)
        whitened_fft.append(tokens)
        latents = autoencoder.export_latents(tokens, sample_posterior=False)["latents"]
        raw_latents.append(latents)
        normalized.append(interface.normalize(latents))

    mask = codec.component_mask.to(device).bool().reshape(-1)
    spaces = [
        (torch.cat(pixels, dim=0).double(), "0_pixels"),
        (torch.cat(raw_fft, dim=0).reshape(count, -1)[:, mask].double(), "1_fft_raw"),
        (
            torch.cat(whitened_fft, dim=0).reshape(count, -1)[:, mask].double(),
            "2_fft_whitened_AE_input",
        ),
        (torch.cat(raw_latents, dim=0).reshape(count, -1).double(), "3_latents_raw"),
        (
            torch.cat(normalized, dim=0).reshape(count, -1).double(),
            "4_latents_normalized",
        ),
    ]
    del pixels, raw_fft, whitened_fft, raw_latents, normalized

    results = []
    for matrix, label in spaces:
        print(f"analyzing {label} {tuple(matrix.shape)} ...")
        results.append(analyze(matrix, label))
        del matrix

    path = os.path.join(args.output_dir, "hierarchy_chain_report.json")
    with open(path, "w") as handle:
        json.dump({"num_images": count, "spaces": results}, handle, indent=2)

    print(f"\n{'space':<26} {'dims':>5} {'coord var ratio':>16} {'eigen var ratio':>16} {'PR':>8}")
    for row in results:
        print(
            f"{row['label']:<26} {row['dims']:>5d}"
            f" {row['coordinate_p05_p95_variance_ratio']:>16.3e}"
            f" {row['eigen_p05_p95_variance_ratio']:>16.3e}"
            f" {row['participation_ratio']:>8.1f}"
        )
    print(f"\n{'space':<26} {'coord t* spread':>16} {'eigen t* spread':>16}")
    for row in results:
        print(
            f"{row['label']:<26} {row['coordinate_t_star_spread']:>16.3f}"
            f" {row['eigen_t_star_spread']:>16.3f}"
        )
    print("\n(both ratios are 5th-vs-95th percentile of variance across directions,")
    print(" robust to the numerically-tiny eigenvalues that make min/max meaningless)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
