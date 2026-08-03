"""Capstone: does a Gaussian fit of the latent distribution decode to mush?

The trained joint model beats a linear/Gaussian predictor by only ~0.05-0.17 MSE
per position, i.e. it has learned a slightly-better-than-Gaussian model of the
latent distribution.  If sampling an exact Gaussian fit N(mu, Sigma) of the real
latents also decodes to texture mush, then the model's samples are mush for the
simple reason that it never got meaningfully past second order -- and all the
perceptual structure lives in the non-Gaussian part it has barely touched.

Also reports how much the autoencoder actually compresses, since a latent that is
no smaller than the image cannot have discarded the incompressible detail that
makes the distribution hard to model.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=40000)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    seq_len, token_dim = LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM
    dims = seq_len * token_dim

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    count = min(args.num_images, len(train_set))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_set, range(count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )
    chunks = []
    for images, _ in loader:
        latents = interface.encode_images(images.to(device))
        chunks.append(latents.reshape(latents.shape[0], -1))
    real = torch.cat(chunks, dim=0).double()
    print(f"real latents: {tuple(real.shape)}")

    mean = real.mean(dim=0)
    centered = real - mean
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)

    grid_n = args.grid_images
    total = grid_n * 2
    noise = torch.randn(total, dims, dtype=torch.float64, device=device)
    gaussian = mean[None, :] + noise @ (vectors * eigenvalues.sqrt()[None, :]).T
    gaussian_latents = gaussian.float().reshape(total, seq_len, token_dim)
    decoded_gaussian = interface.decode_latents(gaussian_latents)

    prior_latents = torch.randn(total, seq_len, token_dim, device=device)
    decoded_prior = interface.decode_latents(prior_latents)

    test_set = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transform
    )
    real_images = torch.stack([test_set[i][0] for i in range(grid_n)], dim=0).to(device)
    decoded_real = interface.decode_latents(interface.encode_images(real_images))

    save_image(
        torch.cat(
            [decoded_real, decoded_gaussian[:grid_n], decoded_prior[:grid_n]], dim=0
        ),
        os.path.join(args.output_dir, "gaussian_vs_prior.png"),
        nrow=grid_n,
    )

    # Effective dimensionality of the latent distribution.
    energy = eigenvalues / eigenvalues.sum()
    cumulative = torch.cumsum(energy, dim=0)
    participation = float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())
    descending = torch.flip(cumulative, dims=[0])
    image_dims = 3 * 32 * 32

    report = {
        "num_real_samples": int(real.shape[0]),
        "latent_dims": dims,
        "image_dims": image_dims,
        "compression_ratio_image_over_latent": image_dims / dims,
        "participation_ratio": participation,
        "dims_for_90pct_variance": int((torch.cumsum(torch.flip(energy, dims=[0]), 0) < 0.9).sum().item() + 1),
        "dims_for_99pct_variance": int((torch.cumsum(torch.flip(energy, dims=[0]), 0) < 0.99).sum().item() + 1),
        "grid_row_order": ["real_reconstruction", "gaussian_fit_samples", "prior_samples"],
    }
    path = os.path.join(args.output_dir, "gaussian_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nlatent dims {dims} vs image dims {image_dims}"
          f"  -> compression {report['compression_ratio_image_over_latent']:.2f}x")
    print(f"participation ratio: {participation:.1f} effective dims")
    print(
        f"variance: {report['dims_for_90pct_variance']} dims for 90%,"
        f" {report['dims_for_99pct_variance']} for 99%"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
