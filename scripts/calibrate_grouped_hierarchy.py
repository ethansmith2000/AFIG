#!/usr/bin/env python3
"""Calibrate grouped pyramid losses by decoder-output gradient ratio."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progressive_tokenizer import ProgressiveTokenizer, TokenizerConfig
from progressive_tokenizer.training import gaussian_lowpass_pyramid_fft


def parse_csv(raw: str, conversion):
    return tuple(conversion(value.strip()) for value in raw.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--group_sizes", default="11,11,11,11,10,10")
    parser.add_argument("--blur_sigmas", default="8,4,2,1,0.5,0")
    parser.add_argument("--target_gradient_ratio", type=float, default=0.25)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def combined_norm(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hierarchy calibration")
    if args.examples <= 0 or args.batch_size <= 0:
        raise ValueError("examples and batch_size must be positive")
    if args.target_gradient_ratio <= 0:
        raise ValueError("target_gradient_ratio must be positive")

    group_sizes = parse_csv(args.group_sizes, int)
    blur_sigmas = parse_csv(args.blur_sigmas, float)
    if len(group_sizes) != len(blur_sigmas):
        raise ValueError("group and sigma counts must match")
    if any(value <= 0 for value in group_sizes):
        raise ValueError("group sizes must be positive")
    if blur_sigmas[-1] != 0 or any(
        first <= second for first, second in zip(blur_sigmas, blur_sigmas[1:])
    ):
        raise ValueError("blur sigmas must strictly decrease to zero")
    prefix_ends = []
    running = 0
    for size in group_sizes:
        running += size
        prefix_ends.append(running)

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TokenizerConfig(**payload["model_config"])
    if running != config.num_latents:
        raise ValueError("group sizes must sum to checkpoint num_latents")
    tokenizer = ProgressiveTokenizer(config).to(device).eval()
    tokenizer.load_state_dict(payload["model"])
    tokenizer.requires_grad_(False)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    records: dict[str, list[float]] = {
        "pixel_mse": [],
        "cumulative_loss": [],
        "innovation_loss": [],
        "cumulative_gradient_ratio": [],
        "innovation_gradient_ratio": [],
    }
    consumed = 0
    prefix_lookup = torch.tensor(prefix_ends, device=device, dtype=torch.long)
    for images, _ in loader:
        if consumed >= args.examples:
            break
        images = images[: args.examples - consumed].to(device, non_blocking=True)
        batch = images.shape[0]
        group_indices = (
            torch.arange(consumed, consumed + batch, device=device)
            % len(group_sizes)
        )
        current_prefix = prefix_lookup[group_indices]
        previous_indices = (group_indices - 1).clamp_min(0)
        previous_prefix = prefix_lookup[previous_indices]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latents = tokenizer.encode(images)
            full_output = tokenizer.decode(latents)
            current_output = tokenizer.decode(latents, current_prefix)
            previous_output = tokenizer.decode(latents, previous_prefix)
        full_output = full_output.float().detach().requires_grad_(True)
        current_output = current_output.float().detach().requires_grad_(True)
        previous_output = previous_output.float().detach().requires_grad_(True)
        targets = gaussian_lowpass_pyramid_fft(images, blur_sigmas)
        batch_indices = torch.arange(batch, device=device)
        current_target = targets[batch_indices, group_indices]
        previous_target = targets[batch_indices, previous_indices]
        has_previous = (group_indices > 0).view(-1, 1, 1, 1)
        previous_target = torch.where(
            has_previous, previous_target, torch.zeros_like(previous_target)
        )
        masked_previous_output = torch.where(
            has_previous, previous_output, torch.zeros_like(previous_output)
        )

        pixel = F.mse_loss(full_output, images.float())
        cumulative = F.mse_loss(current_output, current_target)
        innovation = F.mse_loss(
            current_output - masked_previous_output,
            current_target - previous_target,
        )
        pixel_gradient = torch.autograd.grad(
            pixel, full_output, retain_graph=True
        )[0]
        cumulative_gradient = torch.autograd.grad(
            cumulative, current_output, retain_graph=True
        )[0]
        innovation_gradients = torch.autograd.grad(
            innovation, (current_output, previous_output)
        )
        pixel_norm = pixel_gradient.norm().clamp_min(1e-12)
        records["pixel_mse"].append(float(pixel.detach()))
        records["cumulative_loss"].append(float(cumulative.detach()))
        records["innovation_loss"].append(float(innovation.detach()))
        records["cumulative_gradient_ratio"].append(
            float((cumulative_gradient.norm() / pixel_norm).detach())
        )
        records["innovation_gradient_ratio"].append(
            float((combined_norm(innovation_gradients) / pixel_norm).detach())
        )
        consumed += batch

    summary = {name: summarize(values) for name, values in records.items()}
    weights = {
        "cumulative": (
            args.target_gradient_ratio
            / summary["cumulative_gradient_ratio"]["median"]
        ),
        "innovation": (
            args.target_gradient_ratio
            / summary["innovation_gradient_ratio"]["median"]
        ),
    }
    pixel_median = summary["pixel_mse"]["median"]
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": consumed,
        "batch_size": args.batch_size,
        "group_sizes": group_sizes,
        "prefix_ends": prefix_ends,
        "blur_sigmas_pixels": blur_sigmas,
        "target_weighted_output_gradient_ratio": args.target_gradient_ratio,
        "raw": summary,
        "recommended_weights": weights,
        "estimated_weighted_loss_ratio": {
            "cumulative_to_pixel_mse": (
                weights["cumulative"]
                * summary["cumulative_loss"]["median"]
                / pixel_median
            ),
            "innovation_to_pixel_mse": (
                weights["innovation"]
                * summary["innovation_loss"]["median"]
                / pixel_median
            ),
        },
        "gradient_gauge": (
            "Euclidean norm across detached decoder outputs; innovation combines "
            "current- and previous-prefix gradients"
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
