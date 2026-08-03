"""Decoder robustness diagnostic for latent AFIG.

Central question: how accurate must a generated latent be for the frozen decoder
to produce a coherent image?  Both the AR and joint-diffusion generative models
reach a per-token normalized error of roughly 0.1-0.3 MSE, yet their decoded
samples are texture-like.  If injecting comparable Gaussian error into *real*
latents also destroys the image, the bottleneck is the representation/decoder,
not the generative model.

Outputs a JSON report plus image grids under the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from latent_autoencoder_interface import FrozenLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=256)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def load_images(data_root: str, count: int) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=False, download=False, transform=transform
    )
    images = torch.stack([dataset[i][0] for i in range(count)], dim=0)
    return images


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    images = load_images(args.data_root, args.num_images).to(device)
    latents = interface.encode_images(images)
    baseline = interface.decode_latents(latents)

    report: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "num_images": int(images.shape[0]),
        "latent_shape": list(latents.shape[1:]),
        "reconstruction_psnr": psnr(baseline, images),
        "latent_rms": float(latents.pow(2).mean().sqrt()),
    }

    grid_n = args.grid_images
    save_image(images[:grid_n], os.path.join(args.output_dir, "00_real.png"), nrow=grid_n)
    save_image(
        baseline[:grid_n], os.path.join(args.output_dir, "01_reconstruction.png"), nrow=grid_n
    )

    # --- 1. Global isotropic noise sweep -------------------------------------
    # sigma is in units of the normalized latent std, so injected per-token MSE
    # equals sigma**2 -- directly comparable to the generative models' losses.
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.5, 0.7, 1.0]
    global_sweep: List[Dict[str, float]] = []
    grids = [images[:grid_n], baseline[:grid_n]]
    for sigma in sigmas:
        noise = torch.randn_like(latents) * sigma
        decoded = interface.decode_latents(latents + noise)
        global_sweep.append(
            {
                "sigma": sigma,
                "injected_latent_mse": sigma**2,
                "psnr_vs_real": psnr(decoded, images),
                "psnr_vs_reconstruction": psnr(decoded, baseline),
            }
        )
        if sigma > 0:
            grids.append(decoded[:grid_n])
    report["global_noise_sweep"] = global_sweep
    save_image(
        torch.cat(grids, dim=0),
        os.path.join(args.output_dir, "02_global_noise_sweep.png"),
        nrow=grid_n,
    )

    # --- 2. Prefix vs suffix corruption --------------------------------------
    # Where does the perceptual damage live?  Corrupt only early (low-frequency)
    # positions, then only late (high-frequency) positions.
    seq_len = latents.shape[1]
    split_points = [4, 8, 16, 24, 32, 40, seq_len]
    prefix_suffix: List[Dict[str, float]] = []
    for sigma in (0.2, 0.35, 0.5):
        for split in split_points:
            prefix_noise = torch.zeros_like(latents)
            prefix_noise[:, :split] = torch.randn_like(latents[:, :split]) * sigma
            suffix_noise = torch.zeros_like(latents)
            suffix_noise[:, split:] = torch.randn_like(latents[:, split:]) * sigma
            prefix_suffix.append(
                {
                    "sigma": sigma,
                    "split": split,
                    "psnr_corrupt_prefix_only": psnr(
                        interface.decode_latents(latents + prefix_noise), baseline
                    ),
                    "psnr_corrupt_suffix_only": psnr(
                        interface.decode_latents(latents + suffix_noise), baseline
                    ),
                }
            )
    report["prefix_suffix_corruption"] = prefix_suffix

    # --- 3. Per-position sensitivity -----------------------------------------
    # Corrupt exactly one position at a fixed sigma and measure the damage.
    per_position: List[Dict[str, float]] = []
    for position in range(seq_len):
        noise = torch.zeros_like(latents)
        noise[:, position] = torch.randn_like(latents[:, position]) * 0.5
        decoded = interface.decode_latents(latents + noise)
        per_position.append(
            {
                "position": position,
                "psnr_vs_reconstruction": psnr(decoded, baseline),
                "pixel_mse_delta": float(
                    torch.mean((decoded - baseline) ** 2)
                ),
            }
        )
    report["per_position_sensitivity_sigma0.5"] = per_position

    # --- 4. Prior-sample decode ----------------------------------------------
    # What does a latent drawn from the model's assumed prior N(0, I) decode to?
    # If the true latent distribution were close to the prior, these would look
    # like plausible images; if they look like our generative samples, the
    # generative model is nearly irrelevant to the failure.
    prior = torch.randn_like(latents)
    prior_decoded = interface.decode_latents(prior)
    save_image(
        prior_decoded[: grid_n * 2],
        os.path.join(args.output_dir, "03_prior_samples.png"),
        nrow=grid_n,
    )
    report["prior_decode"] = {
        "pixel_mean": float(prior_decoded.mean()),
        "pixel_std": float(prior_decoded.std()),
    }

    # --- 5. Real-latent distribution shape -----------------------------------
    # Normalization forces per-dim unit variance, but the *joint* structure can
    # still be far from isotropic Gaussian.  Report per-position norms and the
    # PCA spectrum: a fast-decaying spectrum means the real latents occupy a
    # thin subspace that isotropic noise (and a weak generator) will leave.
    flat = latents.reshape(latents.shape[0], -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    energy = singular**2
    energy = energy / energy.sum()
    cumulative = torch.cumsum(energy, dim=0)
    total_dims = int(flat.shape[1])
    report["latent_geometry"] = {
        "total_dims": total_dims,
        "num_samples": int(flat.shape[0]),
        "position_norm_mean": [
            float(latents[:, p].pow(2).sum(dim=-1).sqrt().mean()) for p in range(seq_len)
        ],
        "pca_energy_top1": float(energy[0]),
        "pca_energy_top10": float(cumulative[min(9, total_dims - 1)]),
        "pca_energy_top50": float(cumulative[min(49, total_dims - 1)]),
        "pca_dims_for_90pct": int((cumulative < 0.9).sum().item() + 1),
        "pca_dims_for_99pct": int((cumulative < 0.99).sum().item() + 1),
    }

    path = os.path.join(args.output_dir, "report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"reconstruction PSNR: {report['reconstruction_psnr']:.2f} dB")
    print("\nglobal noise sweep (sigma -> PSNR vs clean reconstruction):")
    for row in global_sweep:
        print(
            f"  sigma={row['sigma']:.2f} (latent MSE {row['injected_latent_mse']:.3f})"
            f"  {row['psnr_vs_reconstruction']:.2f} dB"
        )
    geometry = report["latent_geometry"]
    print(
        f"\nPCA: {geometry['pca_dims_for_90pct']}/{geometry['total_dims']} dims for 90% energy, "
        f"{geometry['pca_dims_for_99pct']} for 99%"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
