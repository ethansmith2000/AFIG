#!/usr/bin/env python3
"""Probe a raw compact-FFT checkpoint with forward-consistent low-frequency clamps.

This is a zero-training conditional diagnostic, not an unconditional generator
evaluation.  Known Fourier coordinates follow the exact linear training bridge
from one fixed Gaussian base sample to the held-out target and are overwritten
after every Heun update.  The unknown coordinates remain model generated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from control_pixel_diffusion import (
    PatchDiffusion,
    build_compact_isometric_codec,
    compact_isometric_fft_to_tokens,
    compact_isometric_orbit_mask,
    compact_isometric_tokens_to_images,
    orbit_order_permutation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="latent_continuous_runs/fft_compact_isometric_spiral_control/final.pt",
    )
    parser.add_argument(
        "--history",
        default="latent_continuous_runs/fft_compact_isometric_spiral_control/history.json",
    )
    parser.add_argument(
        "--output_dir",
        default="diagnostics/compact_fft_low_frequency_clamp",
    )
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--dataset_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--cutoffs", default="0,2,4,8")
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--token_dim", type=int, default=48)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _bridge_value(
    target: torch.Tensor, base_noise: torch.Tensor, time: float
) -> torch.Tensor:
    return time * target + (1.0 - time) * base_noise


@torch.inference_mode()
def sample_with_clamp(
    model: PatchDiffusion,
    target: torch.Tensor,
    base_noise: torch.Tensor,
    mask: torch.Tensor | None,
    steps: int,
) -> torch.Tensor:
    """Integrate the learned flow while enforcing a known-coordinate bridge."""
    x = base_noise.clone()
    batch = x.shape[0]
    device = x.device
    expanded_mask = None if mask is None else mask[None].to(device=device)

    def enforce(values: torch.Tensor, time: float) -> torch.Tensor:
        if expanded_mask is None:
            return values
        known = _bridge_value(target, base_noise, time)
        return torch.where(expanded_mask, known, values)

    x = enforce(x, 0.0)
    dt = 1.0 / steps
    for index in range(steps):
        time_value = index / steps
        next_value = (index + 1) / steps
        time = torch.full((batch,), time_value, device=device)
        x = enforce(x, time_value)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            velocity = model.velocity(x, time).float()
        proposal = enforce(x + dt * velocity, next_value)
        if index + 1 < steps:
            next_time = torch.full((batch,), next_value, device=device)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                next_velocity = model.velocity(proposal, next_time).float()
            x = enforce(x + 0.5 * dt * (velocity + next_velocity), next_value)
        else:
            x = proposal
    return enforce(x, 1.0)


def main() -> None:
    args = parse_args()
    if args.inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    history = json.loads(Path(args.history).read_text())
    mean = float(history["mean"])
    std = float(history["std"])
    model_args = SimpleNamespace(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        flow_path="linear",
    )
    total_values = 3 * args.image_size * args.image_size
    if total_values % args.token_dim:
        raise ValueError("token_dim must divide the image scalar count")
    model = PatchDiffusion(
        total_values // args.token_dim, args.token_dim, model_args
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    dataset = torchvision.datasets.CIFAR10(
        args.data_root,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    end = args.dataset_offset + args.count
    if args.dataset_offset < 0 or end > len(dataset):
        raise ValueError("requested held-out dataset range is invalid")
    images = torch.stack(
        [dataset[index][0] for index in range(args.dataset_offset, end)]
    ).to(device)

    codec = build_compact_isometric_codec(args.image_size, device)
    permutation = orbit_order_permutation(codec, "square_spiral")
    target = compact_isometric_fft_to_tokens(
        codec,
        (images - mean) / std,
        permutation,
        token_dim=args.token_dim,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    base_noise = torch.randn(
        target.shape, device=device, dtype=target.dtype, generator=generator
    )

    cutoffs = sorted({int(value) for value in args.cutoffs.split(",") if value})
    if not cutoffs or cutoffs[0] < 0:
        raise ValueError("cutoffs must contain non-negative integer radius bins")

    rows = [images.float()]
    row_labels = ["heldout_reference"]
    unconditional = sample_with_clamp(
        model, target, base_noise, None, args.inference_steps
    )

    def decode(tokens: torch.Tensor) -> torch.Tensor:
        return (
            compact_isometric_tokens_to_images(codec, tokens, permutation) * std
            + mean
        ).float().clamp(0.0, 1.0)

    unconditional_images = decode(unconditional)
    rows.append(unconditional_images)
    row_labels.append("unconditional_same_noise")
    save_image(
        unconditional_images,
        output_dir / "unconditional.png",
        nrow=min(args.count, 8),
    )

    results = []
    target_energy = target.square().sum().clamp_min(1e-12)
    for cutoff in cutoffs:
        selected_orbits = codec.radius_bin <= cutoff
        mask = compact_isometric_orbit_mask(
            codec, selected_orbits, permutation, token_dim=args.token_dim
        )
        lowpass_tokens = torch.where(mask[None], target, torch.zeros_like(target))
        lowpass_images = decode(lowpass_tokens)
        completion = sample_with_clamp(
            model, target, base_noise, mask, args.inference_steps
        )
        completion_images = decode(completion)
        rows.extend([lowpass_images, completion_images])
        row_labels.extend(
            [f"radius_{cutoff}_known_only", f"radius_{cutoff}_completion"]
        )
        save_image(
            completion_images,
            output_dir / f"completion_radius_{cutoff}.png",
            nrow=min(args.count, 8),
        )
        selected_energy = target[:, mask].square().sum() / target_energy
        unknown = ~mask
        unknown_target = target[:, unknown]
        unknown_completion = completion[:, unknown]
        unknown_unconditional = unconditional[:, unknown]
        unknown_zero_mse = unknown_target.square().mean()
        unknown_completion_mse = (unknown_completion - unknown_target).square().mean()
        unknown_unconditional_mse = (
            unknown_unconditional - unknown_target
        ).square().mean()
        completion_pixel_mse = (completion_images - images).square().mean()
        lowpass_pixel_mse = (lowpass_images - images).square().mean()
        results.append(
            {
                "radius_cutoff": cutoff,
                "selected_orbits": int(selected_orbits.sum().item()),
                "selected_scalars": int(mask.sum().item()),
                "selected_scalar_fraction": float(mask.float().mean().item()),
                "heldout_target_energy_fraction": float(selected_energy.item()),
                "unknown_zero_fill_mse": float(unknown_zero_mse.item()),
                "unknown_completion_mse": float(unknown_completion_mse.item()),
                "unknown_unconditional_mse": float(unknown_unconditional_mse.item()),
                "unknown_mse_fractional_improvement_over_zero": float(
                    (1.0 - unknown_completion_mse / unknown_zero_mse.clamp_min(1e-12)).item()
                ),
                "unknown_mse_fractional_improvement_over_unconditional": float(
                    (
                        1.0
                        - unknown_completion_mse
                        / unknown_unconditional_mse.clamp_min(1e-12)
                    ).item()
                ),
                "lowpass_pixel_mse": float(lowpass_pixel_mse.item()),
                "completion_pixel_mse": float(completion_pixel_mse.item()),
            }
        )

    full_mask = torch.ones_like(target[0], dtype=torch.bool)
    full = sample_with_clamp(
        model, target, base_noise, full_mask, args.inference_steps
    )
    full_images = decode(full)
    rows.append(full_images)
    row_labels.append("full_spectrum_clamp")
    full_error = float((full - target).abs().max().item())
    save_image(
        torch.cat(rows, dim=0),
        output_dir / "comparison_rows.png",
        nrow=args.count,
    )
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "heldout_split": "cifar10_test",
        "dataset_offset": args.dataset_offset,
        "count": args.count,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "normalization": {"mean": mean, "std": std},
        "row_labels": row_labels,
        "cutoffs": results,
        "full_clamp_max_token_error": full_error,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
