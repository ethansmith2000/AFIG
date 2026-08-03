"""Empirical per-position, per-timestep SNR analysis for the joint latent model.

Because the latents are standardized per position to unit variance, the *nominal*
rectified-flow SNR at time t is t^2 / (1-t)^2 for every position identically -- the
schedule allocates equal resolution effort everywhere.  Perceptual need is not
uniform at all (~1000:1 concentrated in the first few positions), so the schedule
and the normalization interact.

This resolves the velocity MSE by (position, t) rather than averaging over t, for
both the trained model on held-out data and the Gaussian-optimal floor, and then
asks where in t the *perceptually weighted* deficit actually lives.  That is the
quantity a schedule change should target.

The Gaussian floor is computed in the train covariance eigenbasis, where the
per-dimension residual at time t is (U**2) @ D_t with

    D_t = (Lambda + 1) - (t*Lambda - (1-t))^2 / (t^2*Lambda + (1-t)^2)
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
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument("--damage_weights", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_eval", type=int, default=4096)
    parser.add_argument("--covariance_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_times", type=int, default=21)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def encode(
    interface: FrozenLatentAutoencoder,
    data_root: str,
    train: bool,
    count: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=train, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return torch.cat([interface.encode_images(x.to(device)) for x, _ in loader], dim=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    seq_len, token_dim = LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    test = encode(interface, args.data_root, False, args.num_eval, args.batch_size, device)
    train = encode(
        interface, args.data_root, True, args.covariance_images, args.batch_size, device
    )
    flat = train.reshape(train.shape[0], -1).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    squared_vectors = vectors**2
    del train, flat, centered, covariance

    from train_joint_latent_diffusion import load_checkpoint

    model, step = load_checkpoint(args.joint_checkpoint, interface)
    model = model.to(device).eval()
    metadata = interface.position_features
    print(f"joint step {step}; eval on {tuple(test.shape)}")

    weights = torch.load(args.damage_weights, map_location="cpu", weights_only=False)[
        "measured_damage"
    ].double().to(device)
    weights = weights / weights.sum()

    times = torch.linspace(0.05, 0.95, args.num_times, dtype=torch.float64)
    rows: List[Dict[str, object]] = []
    for time_value in times:
        t = float(time_value)
        generator = torch.Generator(device=device).manual_seed(4242)
        model_accumulator = torch.zeros(seq_len, dtype=torch.float64, device=device)
        seen = 0
        for start in range(0, test.shape[0], args.batch_size):
            batch = test[start : start + args.batch_size]
            noise = torch.randn(
                batch.shape, device=device, dtype=batch.dtype, generator=generator
            )
            noisy = t * batch + (1.0 - t) * noise
            target = batch - noise
            flow_time = torch.full((batch.shape[0],), t, device=device)
            prediction = model.predict_velocity(noisy, flow_time, metadata)
            squared = (prediction.float() - target.float()).square().mean(dim=-1)
            model_accumulator += squared.sum(dim=0).double()
            seen += batch.shape[0]
        model_position = model_accumulator / seen

        d_t = (eigenvalues + 1.0) - (t * eigenvalues - (1.0 - t)) ** 2 / (
            t**2 * eigenvalues + (1.0 - t) ** 2
        ).clamp_min(1e-12)
        floor_position = (squared_vectors @ d_t).reshape(seq_len, token_dim).mean(dim=-1)

        no_structure = 2.0 - (2.0 * t - 1.0) ** 2 / (t**2 + (1.0 - t) ** 2)
        deficit = model_position - floor_position
        rows.append(
            {
                "t": t,
                "nominal_snr": t**2 / max((1.0 - t) ** 2, 1e-12),
                "no_structure_mse": no_structure,
                "model_mse_mean": float(model_position.mean()),
                "floor_mse_mean": float(floor_position.mean()),
                "model_mse_prefix4": float(model_position[:4].mean()),
                "floor_mse_prefix4": float(floor_position[:4].mean()),
                "model_mse_suffix16": float(model_position[16:].mean()),
                "floor_mse_suffix16": float(floor_position[16:].mean()),
                "weighted_model_mse": float((weights * model_position).sum()),
                "weighted_floor_mse": float((weights * floor_position).sum()),
                "weighted_deficit": float((weights * deficit).sum()),
                "unweighted_deficit": float(deficit.mean()),
                "model_mse_per_position": [float(x) for x in model_position],
                "floor_mse_per_position": [float(x) for x in floor_position],
            }
        )
        print(
            f"  t={t:.2f}  model {float(model_position.mean()):.4f}"
            f"  floor {float(floor_position.mean()):.4f}"
            f"  prefix4 model {float(model_position[:4].mean()):.4f}"
            f"  weighted deficit {float((weights * deficit).sum()):+.4f}"
        )

    weighted_total = sum(r["weighted_deficit"] for r in rows)
    unweighted_total = sum(r["unweighted_deficit"] for r in rows)
    report = {
        "joint_step": step,
        "num_eval": int(test.shape[0]),
        "rows": rows,
        "weighted_deficit_share_by_t": [
            r["weighted_deficit"] / weighted_total if weighted_total else 0.0 for r in rows
        ],
        "unweighted_deficit_share_by_t": [
            r["unweighted_deficit"] / unweighted_total if unweighted_total else 0.0
            for r in rows
        ],
    }
    path = os.path.join(args.output_dir, "snr_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n    t   nominalSNR   model   floor   noStruct |  w.model  w.floor  w.deficit  w.share")
    for row, share in zip(rows, report["weighted_deficit_share_by_t"]):
        print(
            f"  {row['t']:.2f} {row['nominal_snr']:>11.3f} {row['model_mse_mean']:>7.4f}"
            f" {row['floor_mse_mean']:>7.4f} {row['no_structure_mse']:>9.4f} |"
            f" {row['weighted_model_mse']:>8.4f} {row['weighted_floor_mse']:>8.4f}"
            f" {row['weighted_deficit']:>10.4f} {share * 100:>7.1f}%"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
