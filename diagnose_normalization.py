"""Normalization and tail geometry, for raw FFT tokens and for AE latents.

Three questions:

1. How heavy-tailed are the per-orbit standardized FFT tokens?  The codec uses
   normalization=orbit_standardize with value_transform=identity, so each of the
   ~3084 (orbit, component) slots is centered and divided by its own dataset
   standard deviation, and no compressive transform is applied.  Natural-image
   Fourier coefficients are famously heavy-tailed, and frequency.py ships an
   unused asinh transform, so it is worth knowing what the tails look like.

2. Would a single global mean/std work for the *latents*, the way Stable
   Diffusion uses one tensor-wide scalar?  This depends on how much the raw
   (pre-normalization) latent scale varies across positions.

3. If the raw latent scale does vary, does it vary in a way that tracks
   perceptual importance?  If it does, global scalar normalization would preserve
   the natural weighting automatically -- no explicit weight tensor needed -- and
   the current per-position standardization is actively destroying it.
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
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--damage_weights", default=None)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tail_stats(values: torch.Tensor) -> Dict[str, float]:
    """Kurtosis and tail mass of a flat sample, assumed already standardized."""
    centered = values - values.mean()
    std = centered.std().clamp_min(1e-12)
    z = centered / std
    return {
        "kurtosis": float((z**4).mean()),
        "frac_abs_gt_4": float((z.abs() > 4).float().mean()),
        "frac_abs_gt_8": float((z.abs() > 8).float().mean()),
        "max_abs_z": float(z.abs().max()),
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

    token_chunks: List[torch.Tensor] = []
    raw_latent_chunks: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device)
        tokens = codec.encode(images.float())
        token_chunks.append(tokens)
        raw_latent_chunks.append(
            autoencoder.export_latents(tokens, sample_posterior=False)["latents"]
        )
    tokens = torch.cat(token_chunks, dim=0)
    raw_latents = torch.cat(raw_latent_chunks, dim=0)
    del token_chunks, raw_latent_chunks
    print(f"whitened FFT tokens: {tuple(tokens.shape)}")
    print(f"raw AE latents:      {tuple(raw_latents.shape)}")

    mask = codec.component_mask.to(device).bool()
    report: Dict[str, object] = {"num_images": int(tokens.shape[0])}

    # --- 1. FFT token tails, overall and by radius --------------------------
    active = tokens[:, mask]
    report["fft_tokens_overall"] = tail_stats(active.reshape(-1))

    radius_bin = codec.radius_bin.to(device)
    per_radius: List[Dict[str, float]] = []
    for radius in range(int(radius_bin.max()) + 1):
        selector = radius_bin == radius
        if not bool(selector.any()):
            continue
        sub = tokens[:, selector][:, mask[selector]]
        stats = tail_stats(sub.reshape(-1))
        stats["radius"] = radius
        stats["orbits"] = int(selector.sum())
        per_radius.append(stats)
    report["fft_tokens_by_radius"] = per_radius

    # --- 2. Raw latent scale structure --------------------------------------
    latent_std = raw_latents.std(dim=0)
    latent_mean = raw_latents.mean(dim=0)
    position_std = latent_std.mean(dim=-1)
    global_std = float(raw_latents.std())
    report["latent_scale"] = {
        "global_std": global_std,
        "position_std_min": float(position_std.min()),
        "position_std_max": float(position_std.max()),
        "position_std_ratio": float(position_std.max() / position_std.min().clamp_min(1e-12)),
        "per_dim_std_min": float(latent_std.min()),
        "per_dim_std_max": float(latent_std.max()),
        "per_dim_std_ratio": float(latent_std.max() / latent_std.min().clamp_min(1e-12)),
        "global_mean_abs": float(latent_mean.abs().mean()),
        "position_std": [float(x) for x in position_std],
    }
    report["latent_tails"] = tail_stats(
        ((raw_latents - latent_mean) / latent_std.clamp_min(1e-12)).reshape(-1)
    )

    # --- 3. Does natural latent scale track perceptual importance? ----------
    if args.damage_weights and os.path.exists(args.damage_weights):
        payload = torch.load(args.damage_weights, map_location="cpu", weights_only=False)
        damage = payload["measured_damage"].float()
        log_scale = position_std.cpu().log10()
        log_damage = damage.clamp_min(1e-30).log10()

        def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
            a = a - a.mean()
            b = b - b.mean()
            return float(
                (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False)).clamp_min(1e-12)
            )

        def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
            return pearson(a.argsort().argsort().float(), b.argsort().argsort().float())

        report["scale_vs_perceptual_importance"] = {
            "log_pearson": pearson(log_scale, log_damage),
            "spearman": spearman(position_std.cpu(), damage),
            "note": (
                "high correlation would mean a single global std preserves the natural"
                " perceptual weighting, and per-position standardization destroys it"
            ),
        }

    path = os.path.join(args.output_dir, "normalization_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    overall = report["fft_tokens_overall"]
    print("\n=== Whitened FFT tokens (orbit_standardize, value_transform=identity) ===")
    print(f"  overall kurtosis {overall['kurtosis']:.2f}  (Gaussian = 3)")
    print(f"  P(|z|>4) {overall['frac_abs_gt_4']:.3e}   P(|z|>8) {overall['frac_abs_gt_8']:.3e}"
          f"   max|z| {overall['max_abs_z']:.1f}")
    print("\n  radius  orbits  kurtosis   P(|z|>4)   P(|z|>8)   max|z|")
    for row in per_radius:
        print(f"  {row['radius']:>6d} {row['orbits']:>7d} {row['kurtosis']:>9.2f}"
              f" {row['frac_abs_gt_4']:>10.2e} {row['frac_abs_gt_8']:>10.2e} {row['max_abs_z']:>8.1f}")

    scale = report["latent_scale"]
    print("\n=== Raw AE latent scale (pre-normalization) ===")
    print(f"  global std {scale['global_std']:.4f}")
    print(f"  per-position std range {scale['position_std_min']:.4f} .. {scale['position_std_max']:.4f}"
          f"   ratio {scale['position_std_ratio']:.2f}x")
    print(f"  per-dim std range      {scale['per_dim_std_min']:.4f} .. {scale['per_dim_std_max']:.4f}"
          f"   ratio {scale['per_dim_std_ratio']:.2f}x")
    print(f"  latent kurtosis {report['latent_tails']['kurtosis']:.2f}  (Gaussian = 3)")
    if "scale_vs_perceptual_importance" in report:
        correlation = report["scale_vs_perceptual_importance"]
        print("\n=== Natural latent scale vs measured perceptual importance ===")
        print(f"  log10 Pearson {correlation['log_pearson']:+.4f}   Spearman {correlation['spearman']:+.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
