"""Localize the coherence failure: is it cross-token dependency?

The joint diffusion model matches the real latents' second-order structure
almost exactly, yet its samples are texture-like.  So either (a) matching the
latent distribution to second order is not sufficient for coherent images, or
(b) the coherence information lives in higher-order dependencies the model is
missing.  Both are the same statement about *cross-token* dependency, and this
script tests it nonparametrically.

Key test -- independent per-position shuffling of real latents.  Shuffling each
position independently across the batch exactly preserves every position's
marginal distribution and all within-token structure, while destroying all
dependency between positions.  If shuffled real latents decode to the same
texture mush as our generative samples, then cross-token dependency is precisely
what is missing, and it is invisible to the second-order statistics the model
already matches.

Also measures cross-token *magnitude* coupling (correlation of per-token
energy), the dependency natural images carry across frequency bands and which
linear correlation of whitened coefficients cannot see.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

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
    parser.add_argument("--joint_checkpoint", default=None)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=4096)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def energy_coupling(latents: torch.Tensor) -> Dict[str, float]:
    """Cross-token coupling of per-token energy (a higher-order dependency).

    latents: [N, seq, dim].  Returns statistics of the correlation matrix of
    per-token log energy across positions.
    """
    energy = latents.pow(2).mean(dim=-1).clamp_min(1e-8).log()
    centered = energy - energy.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False).clamp_min(1e-8)
    normalized = centered / std
    correlation = (normalized.T @ normalized) / normalized.shape[0]
    seq_len = latents.shape[1]
    off_diagonal = correlation - torch.diag(torch.diagonal(correlation))
    pairs = seq_len * (seq_len - 1)
    return {
        "mean_offdiag_energy_correlation": float(off_diagonal.sum() / pairs),
        "rms_offdiag_energy_correlation": float(
            (off_diagonal.pow(2).sum() / pairs).sqrt()
        ),
        "max_offdiag_energy_correlation": float(off_diagonal.max()),
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
        data_root, train=False, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    chunks = [interface.encode_images(images.to(device)) for images, _ in loader]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def generate_latents(
    joint_checkpoint: str,
    interface: FrozenLatentAutoencoder,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    from train_joint_latent_diffusion import load_checkpoint

    model, step = load_checkpoint(joint_checkpoint, interface)
    model = model.to(device).eval()
    print(f"loaded joint checkpoint at step {step}")
    generator = torch.Generator(device=device).manual_seed(1234)
    chunks = []
    remaining = count
    while remaining > 0:
        take = min(512, remaining)
        chunks.append(
            model.generate_latents(
                take, interface.position_features, generator=generator
            ).float()
        )
        remaining -= take
    return torch.cat(chunks, dim=0)


def shuffle_positions(
    latents: torch.Tensor, start: int, generator: torch.Generator
) -> torch.Tensor:
    """Independently permute each position at index >= start across the batch."""
    out = latents.clone()
    count = latents.shape[0]
    for position in range(start, latents.shape[1]):
        permutation = torch.randperm(count, device=latents.device, generator=generator)
        out[:, position] = latents[permutation, position]
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    real = encode_dataset(
        interface, args.data_root, args.num_images, args.batch_size, device
    )
    print(f"real latents: {tuple(real.shape)}")
    grid_n = args.grid_images
    report: Dict[str, object] = {"num_samples": int(real.shape[0])}

    report["real_energy_coupling"] = energy_coupling(real)

    # Full shuffle: perfect marginals and within-token structure, zero
    # cross-token dependency.
    fully_shuffled = shuffle_positions(real, 0, generator)
    report["shuffled_energy_coupling"] = energy_coupling(fully_shuffled)

    rows = [
        interface.decode_latents(real[:grid_n]),
        interface.decode_latents(fully_shuffled[:grid_n]),
    ]
    labels = ["real_reconstruction", "all_positions_shuffled"]

    # Partial shuffles: keep a coherent low-frequency prefix, destroy the rest.
    partial: Dict[str, object] = {}
    for start in (1, 2, 4, 8, 16, 32):
        shuffled = shuffle_positions(real, start, generator)
        partial[f"keep_prefix_{start}"] = energy_coupling(shuffled)
        rows.append(interface.decode_latents(shuffled[:grid_n]))
        labels.append(f"shuffled_from_{start}")
    report["partial_shuffle_energy_coupling"] = partial

    if args.joint_checkpoint:
        generated = generate_latents(
            args.joint_checkpoint, interface, args.num_images, device
        )
        print(f"generated latents: {tuple(generated.shape)}")
        report["generated_energy_coupling"] = energy_coupling(generated)
        rows.append(interface.decode_latents(generated[:grid_n]))
        labels.append("joint_diffusion_samples")

    save_image(
        torch.cat(rows, dim=0),
        os.path.join(args.output_dir, "shuffle_comparison.png"),
        nrow=grid_n,
    )
    report["grid_row_order"] = labels

    path = os.path.join(args.output_dir, "cross_token_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Cross-token energy (magnitude) coupling ===")
    print(
        "  real latents      : mean "
        f"{report['real_energy_coupling']['mean_offdiag_energy_correlation']:.4f}"
        f"  rms {report['real_energy_coupling']['rms_offdiag_energy_correlation']:.4f}"
    )
    if args.joint_checkpoint:
        print(
            "  joint samples     : mean "
            f"{report['generated_energy_coupling']['mean_offdiag_energy_correlation']:.4f}"
            f"  rms {report['generated_energy_coupling']['rms_offdiag_energy_correlation']:.4f}"
        )
    print(
        "  shuffled (zero)   : mean "
        f"{report['shuffled_energy_coupling']['mean_offdiag_energy_correlation']:.4f}"
        f"  rms {report['shuffled_energy_coupling']['rms_offdiag_energy_correlation']:.4f}"
    )
    print(f"\ngrid rows: {labels}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
