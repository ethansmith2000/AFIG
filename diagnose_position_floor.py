"""Is the low-frequency prefix underfit, or already at its achievable floor?

Reweighting or a coarse-to-fine cascade only helps if the model is leaving
achievable accuracy on the table at the perceptually critical prefix positions.
This computes, per position, the Bayes-optimal rectified-flow velocity MSE for a
Gaussian model with the real latents' full covariance, and compares it to the
per-position loss the trained joint model actually reached.

The Gaussian optimum is a *linear* predictor, so a well-fit neural model should
beat it everywhere.  Positions where the model is *worse* than the linear
optimum are being starved by the uniformly-weighted loss.

In Sigma's eigenbasis the residual covariance is R_t = U D_t U^T with

    D_t = (Lambda + 1) - (t*Lambda - (1-t))^2 / (t^2*Lambda + (1-t)^2)

so the per-dimension residual is (U**2) @ D_t.  That is linear in D_t, so we can
average D_t over t first and take a single matrix-vector product.
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
    parser.add_argument("--joint_metrics", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--time_grid", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
        latents = interface.encode_images(images.to(device))
        chunks.append(latents.reshape(latents.shape[0], -1))
    real = torch.cat(chunks, dim=0).double()
    print(f"real latents: {tuple(real.shape)}")

    centered = real - real.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)

    times = (
        torch.arange(args.time_grid, dtype=torch.float64, device=device) + 0.5
    ) / args.time_grid
    t = times[:, None]
    lam = eigenvalues[None, :]
    d_t = (lam + 1.0) - (t * lam - (1.0 - t)) ** 2 / (
        t**2 * lam + (1.0 - t) ** 2
    ).clamp_min(1e-12)
    d_mean = d_t.mean(dim=0)
    per_dim_floor = (vectors**2) @ d_mean
    position_floor = per_dim_floor.reshape(seq_len, token_dim).mean(dim=-1)

    # The trained model's final per-position losses.
    achieved: Dict[int, float] = {}
    with open(args.joint_metrics) as handle:
        for line in handle:
            record = json.loads(line)
            for key, value in record.items():
                if key.startswith("position_loss/"):
                    achieved[int(key.split("/")[1])] = float(value)

    rows: List[Dict[str, float]] = []
    for position in range(seq_len):
        floor = float(position_floor[position])
        model_loss = achieved.get(position, float("nan"))
        rows.append(
            {
                "position": position,
                "gaussian_linear_floor": floor,
                "model_loss": model_loss,
                "model_minus_floor": model_loss - floor,
                "model_beats_linear": bool(model_loss < floor),
            }
        )

    report = {
        "num_samples": int(real.shape[0]),
        "aggregate_gaussian_floor": float(position_floor.mean()),
        "aggregate_model_loss": float(
            sum(achieved.values()) / max(len(achieved), 1)
        ),
        "per_position": rows,
    }
    path = os.path.join(args.output_dir, "position_floor_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(
        f"\naggregate: gaussian linear floor {report['aggregate_gaussian_floor']:.4f}"
        f"  vs model {report['aggregate_model_loss']:.4f}"
    )
    print("\n pos   linear-floor   model    model-floor   verdict")
    for row in rows:
        verdict = "beats linear" if row["model_beats_linear"] else "WORSE than linear"
        print(
            f"  {row['position']:>2d}   {row['gaussian_linear_floor']:>10.4f}"
            f"  {row['model_loss']:>7.4f}   {row['model_minus_floor']:>+9.4f}   {verdict}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
