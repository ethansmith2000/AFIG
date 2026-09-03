#!/usr/bin/env python3
"""Calibrate decoder auxiliaries by output-gradient ratio on a frozen tokenizer."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progressive_tokenizer import ProgressiveTokenizer, TokenizerConfig
from progressive_tokenizer.training import (
    lpips_reconstruction_loss,
    radial_log_power_reconstruction_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_gradient_ratio", type=float, default=0.1)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoder-objective calibration")
    if args.examples <= 0 or args.batch_size <= 0:
        raise ValueError("examples and batch_size must be positive")
    if args.target_gradient_ratio <= 0:
        raise ValueError("target_gradient_ratio must be positive")

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TokenizerConfig(**payload["model_config"])
    tokenizer = ProgressiveTokenizer(config).to(device).eval()
    tokenizer.load_state_dict(payload["model"])
    tokenizer.requires_grad_(False)
    perceptual = lpips.LPIPS(net="alex", verbose=True).to(device).eval()
    perceptual.requires_grad_(False)

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
        "radial_log_power": [],
        "lpips_alex": [],
        "radial_gradient_ratio": [],
        "lpips_gradient_ratio": [],
    }
    consumed = 0
    for images, _ in loader:
        if consumed >= args.examples:
            break
        images = images[: args.examples - consumed].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            reconstruction = tokenizer(images)["reconstruction"]
        reconstruction = reconstruction.float().detach().requires_grad_(True)
        target = images.float()
        pixel = F.mse_loss(reconstruction, target)
        radial = radial_log_power_reconstruction_loss(target, reconstruction)
        perceptual_loss = lpips_reconstruction_loss(
            perceptual, target, reconstruction
        )
        pixel_gradient = torch.autograd.grad(
            pixel, reconstruction, retain_graph=True
        )[0]
        radial_gradient = torch.autograd.grad(
            radial, reconstruction, retain_graph=True
        )[0]
        perceptual_gradient = torch.autograd.grad(
            perceptual_loss, reconstruction
        )[0]
        pixel_norm = pixel_gradient.norm().clamp_min(1e-12)
        records["pixel_mse"].append(float(pixel.detach()))
        records["radial_log_power"].append(float(radial.detach()))
        records["lpips_alex"].append(float(perceptual_loss.detach()))
        records["radial_gradient_ratio"].append(
            float((radial_gradient.norm() / pixel_norm).detach())
        )
        records["lpips_gradient_ratio"].append(
            float((perceptual_gradient.norm() / pixel_norm).detach())
        )
        consumed += images.shape[0]

    summary = {name: summarize(values) for name, values in records.items()}
    radial_weight = (
        args.target_gradient_ratio / summary["radial_gradient_ratio"]["median"]
    )
    perceptual_weight = (
        args.target_gradient_ratio / summary["lpips_gradient_ratio"]["median"]
    )
    pixel_median = summary["pixel_mse"]["median"]
    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": consumed,
        "batch_size": args.batch_size,
        "target_weighted_gradient_ratio": args.target_gradient_ratio,
        "raw": summary,
        "recommended_weights": {
            "radial_log_power": radial_weight,
            "lpips_alex": perceptual_weight,
        },
        "estimated_weighted_loss_ratio": {
            "radial_log_power_to_pixel_mse": (
                radial_weight
                * summary["radial_log_power"]["median"]
                / pixel_median
            ),
            "lpips_alex_to_pixel_mse": (
                perceptual_weight * summary["lpips_alex"]["median"] / pixel_median
            ),
        },
        "lpips": {"package_version": "0.1.4", "network": "alex", "version": "0.1"},
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
