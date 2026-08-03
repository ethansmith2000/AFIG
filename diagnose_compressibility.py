"""How much redundancy is actually left in the latent, and of what kind?

Calling the high-frequency latent content "incompressible" is too strong: the
cross-token energy coupling measured at 0.399 is exactly a correlation between
frequency bands.  This script separates two different kinds of redundancy.

  * value redundancy  -- can a token's actual values be predicted from the other
    tokens?  Whitening exists precisely to remove linear redundancy of this kind,
    so this should be small.
  * envelope redundancy -- can a token's *log energy* be predicted from the other
    tokens' log energies?  This is the classic natural-image statistic (magnitudes
    correlate strongly across scale even when coefficients are decorrelated), and
    should be large.

If the split comes out that way, the correct statement is: the latent is
compressible in a higher-order/envelope sense, but the fine detail that dominates
the per-dimension MSE budget is close to irreducible -- and perceptually you only
need the envelope, not the detail.  That is exactly why generated suffixes are
usable while the MSE objective cannot tell the difference.

Both quantities use closed-form multivariate linear regression via the precision
matrix: for a block j, the residual covariance given all other dimensions is
(Lambda_jj)^{-1} where Lambda = Sigma^{-1}.  Linear R^2 is a lower bound on true
predictability.
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
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def block_conditional_r2(
    covariance: torch.Tensor, num_blocks: int, block_size: int, ridge: float
) -> List[float]:
    """Linear R^2 for each block given all other dimensions.

    Residual covariance of block j given the rest is (Lambda_jj)^{-1} with
    Lambda = Sigma^{-1}.  R^2 = 1 - tr(residual) / tr(Sigma_jj).
    """
    dims = covariance.shape[0]
    regularized = covariance + ridge * torch.eye(
        dims, dtype=covariance.dtype, device=covariance.device
    )
    precision = torch.linalg.inv(regularized)
    scores: List[float] = []
    for index in range(num_blocks):
        start = index * block_size
        stop = start + block_size
        residual = torch.linalg.inv(precision[start:stop, start:stop])
        total = torch.diagonal(covariance[start:stop, start:stop]).sum()
        scores.append(float(1.0 - torch.diagonal(residual).sum() / total.clamp_min(1e-12)))
    return scores


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    seq_len, token_dim = LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

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
    chunks = []
    for images, _ in loader:
        chunks.append(interface.encode_images(images.to(device)))
    latents = torch.cat(chunks, dim=0)
    print(f"real latents: {tuple(latents.shape)}")

    # --- value redundancy -------------------------------------------------
    flat = latents.reshape(latents.shape[0], -1).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    value_covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    value_r2 = block_conditional_r2(value_covariance, seq_len, token_dim, args.ridge)

    # --- envelope redundancy ----------------------------------------------
    energy = latents.pow(2).mean(dim=-1).clamp_min(1e-8).log().double()
    energy_centered = energy - energy.mean(dim=0, keepdim=True)
    energy_covariance = (energy_centered.T @ energy_centered) / (
        energy_centered.shape[0] - 1
    )
    energy_r2 = block_conditional_r2(energy_covariance, seq_len, 1, args.ridge)

    prefix = slice(0, 16)
    suffix = slice(16, seq_len)
    report: Dict[str, object] = {
        "num_samples": int(latents.shape[0]),
        "value_r2_per_token": value_r2,
        "energy_r2_per_token": energy_r2,
        "value_r2_mean": sum(value_r2) / len(value_r2),
        "energy_r2_mean": sum(energy_r2) / len(energy_r2),
        "value_r2_mean_prefix16": sum(value_r2[prefix]) / 16,
        "value_r2_mean_suffix": sum(value_r2[suffix]) / (seq_len - 16),
        "energy_r2_mean_prefix16": sum(energy_r2[prefix]) / 16,
        "energy_r2_mean_suffix": sum(energy_r2[suffix]) / (seq_len - 16),
    }

    # Irreducible share of the total unit-variance MSE budget, under a linear
    # predictor: how much of each token's variance survives conditioning.
    report["irreducible_value_variance_fraction"] = 1.0 - report["value_r2_mean"]

    path = os.path.join(args.output_dir, "compressibility_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Linear predictability of each token from ALL other tokens ===")
    print("(R^2; linear is a lower bound on true predictability)")
    print(f"  token VALUES   mean R^2 {report['value_r2_mean']:.4f}"
          f"   prefix[0:16] {report['value_r2_mean_prefix16']:.4f}"
          f"   suffix[16:] {report['value_r2_mean_suffix']:.4f}")
    print(f"  token ENERGY   mean R^2 {report['energy_r2_mean']:.4f}"
          f"   prefix[0:16] {report['energy_r2_mean_prefix16']:.4f}"
          f"   suffix[16:] {report['energy_r2_mean_suffix']:.4f}")
    print("\n pos   value R^2   energy R^2")
    for index in range(seq_len):
        print(f"  {index:>2d}   {value_r2[index]:>9.4f}   {energy_r2[index]:>10.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
