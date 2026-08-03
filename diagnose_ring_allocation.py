"""How does the autoencoder allocate latent capacity across rings, and should it?

The layout rule in GroupLayout is

    latents_per_ring = min(max_ring_latents, max(1, ceil(ring_size / target_tokens_per_latent)))

so latents per ring grows with the number of Fourier coefficients in the ring and
saturates at max_ring_latents.  Since ring size grows with radius, low-frequency
rings get the fewest latents.  Perceptual importance runs the other way, so this
may be allocating capacity backwards.

This script tabulates, per ring: coefficient count, latents assigned, latent
dimensions, the resulting compression ratio, the population spectral amplitude,
and the measured perceptual damage from corrupting that ring's latents.

It also emits a *spectral amplitude* loss weight tensor.  The codec whitens by
dividing each coefficient by its population standard deviation, so weighting the
squared error in whitened space by the population variance restores an
approximately pixel-space L2 objective -- which is the weighting proposed as an
alternative to the flat mean over 53 positions.
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
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights_output", default=None)
    parser.add_argument("--num_images", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)
    layout = interface.autoencoder.layout
    codec = interface.codec

    latent_parent = layout.latent_parent.cpu()
    members_per_latent = layout.gather_mask.sum(dim=1).cpu()
    num_rings = int(layout.num_parents)

    # Population spectral scale per coefficient, pooled onto latents.  With
    # pooler=perceiver_sector the sectors partition the ring, so summing member
    # counts per ring recovers the ring's coefficient count.
    scale = codec.orbit_scale_for_policy(codec.effective_scale_policy()).mean(dim=-1)
    scale = scale.detach().cpu().float()
    indices = layout.gather_indices.cpu()
    mask = layout.gather_mask.cpu()
    gathered = scale[indices] * mask.float()
    counts = mask.sum(dim=1).clamp_min(1).float()
    latent_scale = gathered.sum(dim=1) / counts
    latent_power = latent_scale**2

    # Measured perceptual damage: corrupt one latent at a time and decode.
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transform
    )
    images = torch.stack(
        [dataset[i][0] for i in range(args.num_images)], dim=0
    ).to(device)
    latents = interface.encode_images(images)
    baseline = interface.decode_latents(latents)

    damage = torch.zeros(LATENT_SEQUENCE_LENGTH)
    for position in range(LATENT_SEQUENCE_LENGTH):
        noise = torch.zeros_like(latents)
        noise[:, position] = torch.randn_like(latents[:, position]) * args.sigma
        decoded = interface.decode_latents(latents + noise)
        damage[position] = float(((decoded - baseline) ** 2).mean())

    rings: List[Dict[str, float]] = []
    for ring in range(num_rings):
        selector = latent_parent == ring
        latent_count = int(selector.sum())
        coefficient_count = int(members_per_latent[selector].sum())
        latent_dims = latent_count * LATENT_TOKEN_DIM
        ring_damage = float(damage[selector].sum())
        rings.append(
            {
                "ring": ring,
                "coefficients": coefficient_count,
                "latents": latent_count,
                "latent_dims": latent_dims,
                "coefficients_per_latent_dim": coefficient_count / max(latent_dims, 1),
                "population_power": float(latent_power[selector].mean()),
                "damage_mse": ring_damage,
                "damage_per_latent_dim": ring_damage / max(latent_dims, 1),
            }
        )

    total_damage = sum(row["damage_mse"] for row in rings)
    total_dims = sum(row["latent_dims"] for row in rings)
    for row in rings:
        row["damage_share"] = row["damage_mse"] / max(total_damage, 1e-12)
        row["dim_share"] = row["latent_dims"] / max(total_dims, 1)

    # Spectral-amplitude loss weights, mean-normalized so the overall loss scale
    # is unchanged relative to the flat objective.
    spectral_weights = latent_power / latent_power.mean().clamp_min(1e-12)
    measured_weights = damage / damage.mean().clamp_min(1e-12)

    report = {
        "num_rings": num_rings,
        "sigma": args.sigma,
        "rings": rings,
        "spectral_weight_per_position": [float(x) for x in spectral_weights],
        "measured_damage_weight_per_position": [float(x) for x in measured_weights],
        "spectral_weight_max_over_min": float(
            spectral_weights.max() / spectral_weights.min().clamp_min(1e-30)
        ),
        "measured_weight_max_over_min": float(
            measured_weights.max() / measured_weights.min().clamp_min(1e-30)
        ),
    }
    path = os.path.join(args.output_dir, "ring_allocation_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    if args.weights_output:
        torch.save(
            {
                "version": 1,
                "ae_checkpoint": os.path.abspath(args.checkpoint),
                "latent_interface": os.path.abspath(args.latent_interface),
                "spectral_amplitude": spectral_weights.clone(),
                "measured_damage": measured_weights.clone(),
                "note": "per-position weights, mean-normalized; expand to [53,64] by broadcast",
            },
            args.weights_output,
        )
        print(f"wrote weights to {args.weights_output}")

    print(f"\n{'ring':>5} {'coef':>6} {'lat':>4} {'dims':>5} {'coef/dim':>9}"
          f" {'pop.power':>11} {'damage%':>8} {'dim%':>6} {'dmg/dim':>10}")
    for row in rings:
        print(
            f"{row['ring']:>5d} {row['coefficients']:>6d} {row['latents']:>4d}"
            f" {row['latent_dims']:>5d} {row['coefficients_per_latent_dim']:>9.3f}"
            f" {row['population_power']:>11.3e} {row['damage_share'] * 100:>7.2f}%"
            f" {row['dim_share'] * 100:>5.1f}% {row['damage_per_latent_dim']:>10.3e}"
        )
    print(
        f"\nspectral weight dynamic range: {report['spectral_weight_max_over_min']:.3e}"
        f"   measured-damage weight range: {report['measured_weight_max_over_min']:.3e}"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
