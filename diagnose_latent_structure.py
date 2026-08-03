"""Calibrate what the joint latent diffusion model actually learned.

The normalized latents have unit variance per dimension, so an N(0, I) model --
i.e. one that has learned *nothing* -- attains a known rectified-flow velocity
MSE (2 - pi/2 ~= 1.571).  Any improvement must come from correlation structure.
This script splits that structure in two:

  * within-token  : correlations among the 64 dims of a single latent position.
                    Enough for locally plausible texture; carries no global
                    image layout.
  * cross-token   : correlations between different positions/rings.  This is
                    what makes an image globally coherent.

For Gaussian data with covariance Sigma the Bayes-optimal RF velocity MSE is
available in closed form, so we can compute the MSE a model would reach if it
captured only within-token structure (block-diagonal Sigma) versus all
second-order structure (full Sigma), and compare both to the achieved loss.

Also compares the covariance of *generated* latents against real ones.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import torch
import torchvision
from torchvision import transforms

from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--joint_checkpoint", default=None)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=40000)
    parser.add_argument("--num_generated", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--time_grid", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def gaussian_flow_mse(
    eigenvalues: torch.Tensor, time_grid: int, dims: int
) -> float:
    """Bayes-optimal rectified-flow velocity MSE for N(0, Sigma) data.

    With x_t = t*x + (1-t)*eps and target v = x - eps, the optimal predictor is
    linear, so in Sigma's eigenbasis the per-dimension residual is

        (lambda + 1) - (t*lambda - (1-t))^2 / (t^2*lambda + (1-t)^2)

    averaged over eigenvalues and over t ~ Uniform(0, 1).
    """
    lam = eigenvalues.double().clamp_min(0.0)
    times = (torch.arange(time_grid, dtype=torch.float64, device=lam.device) + 0.5) / time_grid
    t = times[:, None]
    numerator = (t * lam[None, :] - (1.0 - t)) ** 2
    denominator = t**2 * lam[None, :] + (1.0 - t) ** 2
    residual = (lam[None, :] + 1.0) - numerator / denominator.clamp_min(1e-12)
    # Eigenvalues of a rank-deficient estimate are padded with zeros, which
    # correspond to directions the data never occupies; they still contribute.
    return float(residual.sum(dim=1).mean() / dims)


def block_diagonal(covariance: torch.Tensor, seq_len: int, token_dim: int) -> torch.Tensor:
    mask = torch.zeros_like(covariance)
    for position in range(seq_len):
        start = position * token_dim
        stop = start + token_dim
        mask[start:stop, start:stop] = 1.0
    return covariance * mask


def correlation_split(
    covariance: torch.Tensor, seq_len: int, token_dim: int
) -> Dict[str, float]:
    """Split off-diagonal correlation energy into within-token and cross-token."""
    std = covariance.diagonal().clamp_min(1e-12).sqrt()
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    within_mask = torch.zeros_like(correlation)
    for position in range(seq_len):
        start = position * token_dim
        stop = start + token_dim
        within_mask[start:stop, start:stop] = 1.0
    within = (off_diagonal * within_mask).pow(2).sum()
    cross = (off_diagonal * (1.0 - within_mask)).pow(2).sum()
    within_pairs = seq_len * token_dim * (token_dim - 1)
    cross_pairs = (seq_len * token_dim) ** 2 - seq_len * token_dim * token_dim
    return {
        "within_token_energy": float(within),
        "cross_token_energy": float(cross),
        "within_token_rms_correlation": float((within / max(within_pairs, 1)).sqrt()),
        "cross_token_rms_correlation": float((cross / max(cross_pairs, 1)).sqrt()),
        "cross_fraction_of_offdiag_energy": float(cross / (within + cross).clamp_min(1e-12)),
    }


@torch.no_grad()
def encode_dataset(
    interface: FrozenLatentAutoencoder,
    data_root: str,
    count: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=True, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    chunks = []
    for images, _ in loader:
        latents = interface.encode_images(images.to(device))
        chunks.append(latents.reshape(latents.shape[0], -1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def generate_latents(
    joint_checkpoint: str,
    interface: FrozenLatentAutoencoder,
    count: int,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    from train_joint_latent_diffusion import load_checkpoint

    model, step = load_checkpoint(joint_checkpoint, interface)
    model = model.to(device).eval()
    print(f"loaded joint checkpoint at step {step}")
    generator = torch.Generator(device=device).manual_seed(1234)
    chunks = []
    remaining = count
    while remaining > 0:
        take = min(batch_size, remaining)
        latents = model.generate_latents(
            take, interface.position_features, generator=generator
        )
        chunks.append(latents.reshape(take, -1).float())
        remaining -= take
    return torch.cat(chunks, dim=0)


def covariance_of(matrix: torch.Tensor) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return (centered.T @ centered) / max(centered.shape[0] - 1, 1)


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

    print(f"encoding {args.num_images} images ...")
    real = encode_dataset(
        interface, args.data_root, args.num_images, args.batch_size, device
    ).float()
    print(f"real latents: {tuple(real.shape)}")

    covariance = covariance_of(real)
    report: Dict[str, object] = {
        "num_real_samples": int(real.shape[0]),
        "dims": dims,
        "real_per_dim_variance_mean": float(covariance.diagonal().mean()),
        "achieved_joint_flow_mse": 0.9083032011985779,
    }

    identity_eigenvalues = torch.ones(dims, dtype=torch.float64, device=device)
    full_eigenvalues = torch.linalg.eigvalsh(covariance.double())
    block_eigenvalues = torch.linalg.eigvalsh(
        block_diagonal(covariance, seq_len, token_dim).double()
    )

    floors = {
        "no_structure_identity": gaussian_flow_mse(
            identity_eigenvalues, args.time_grid, dims
        ),
        "within_token_only_block_diagonal": gaussian_flow_mse(
            block_eigenvalues, args.time_grid, dims
        ),
        "all_second_order_full_covariance": gaussian_flow_mse(
            full_eigenvalues, args.time_grid, dims
        ),
    }
    report["gaussian_flow_mse_floors"] = floors
    report["real_correlation_split"] = correlation_split(covariance, seq_len, token_dim)

    achieved = report["achieved_joint_flow_mse"]
    span = floors["no_structure_identity"] - floors["all_second_order_full_covariance"]
    report["achieved_fraction_of_available_gain"] = float(
        (floors["no_structure_identity"] - achieved) / max(span, 1e-9)
    )
    within_span = (
        floors["no_structure_identity"] - floors["within_token_only_block_diagonal"]
    )
    report["within_token_share_of_total_gain"] = float(within_span / max(span, 1e-9))

    # Generated-latent comparison, sample-count matched so estimator noise in the
    # off-diagonal energy affects both sides equally.
    if args.joint_checkpoint:
        print("generating latents ...")
        generated = generate_latents(
            args.joint_checkpoint, interface, args.num_generated, 512, device
        )
        print(f"generated latents: {tuple(generated.shape)}")
        matched_real = real[: generated.shape[0]]
        generated_covariance = covariance_of(generated)
        report["generated"] = {
            "num_samples": int(generated.shape[0]),
            "latent_rms": float(generated.pow(2).mean().sqrt()),
            "per_dim_variance_mean": float(generated_covariance.diagonal().mean()),
            "correlation_split": correlation_split(
                generated_covariance, seq_len, token_dim
            ),
            "correlation_split_real_same_sample_count": correlation_split(
                covariance_of(matched_real), seq_len, token_dim
            ),
        }

    path = os.path.join(args.output_dir, "structure_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Rectified-flow velocity MSE reference points ===")
    print(f"  no structure         N(0,I)          : {floors['no_structure_identity']:.4f}")
    print(
        f"  within-token only    block-diag Sigma: {floors['within_token_only_block_diagonal']:.4f}"
    )
    print(
        f"  all second order     full Sigma      : {floors['all_second_order_full_covariance']:.4f}"
    )
    print(f"  ACHIEVED by joint model              : {achieved:.4f}")
    print(
        f"\n  -> captured {report['achieved_fraction_of_available_gain'] * 100:.1f}% of the"
        " second-order gain"
    )
    print(
        f"  -> within-token structure alone accounts for "
        f"{report['within_token_share_of_total_gain'] * 100:.1f}% of that gain"
    )
    split = report["real_correlation_split"]
    print("\n=== Real latent correlation structure ===")
    print(f"  within-token RMS correlation: {split['within_token_rms_correlation']:.4f}")
    print(f"  cross-token  RMS correlation: {split['cross_token_rms_correlation']:.4f}")
    if args.joint_checkpoint:
        generated_split = report["generated"]["correlation_split"]
        matched_split = report["generated"]["correlation_split_real_same_sample_count"]
        print("\n=== Generated vs real (same sample count) ===")
        print(
            f"  within-token RMS correlation: generated {generated_split['within_token_rms_correlation']:.4f}"
            f"  vs real {matched_split['within_token_rms_correlation']:.4f}"
        )
        print(
            f"  cross-token  RMS correlation: generated {generated_split['cross_token_rms_correlation']:.4f}"
            f"  vs real {matched_split['cross_token_rms_correlation']:.4f}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
