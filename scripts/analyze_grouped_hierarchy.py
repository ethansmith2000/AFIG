#!/usr/bin/env python3
"""Measure and visualize cumulative/DoG alignment at grouped token prefixes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progressive_tokenizer import ProgressiveTokenizer, TokenizerConfig
from progressive_tokenizer.training import gaussian_lowpass_pyramid_fft, pixel_psnr


def parse_csv(raw: str, conversion):
    return tuple(conversion(value.strip()) for value in raw.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--preview_examples", type=int, default=8)
    parser.add_argument("--group_sizes", default="11,11,11,11,10,10")
    parser.add_argument("--blur_sigmas", default="8,4,2,1,0.5,0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hierarchy analysis")
    if args.examples <= 0 or args.batch_size <= 0 or args.preview_examples <= 0:
        raise ValueError("example and batch sizes must be positive")
    group_sizes = parse_csv(args.group_sizes, int)
    blur_sigmas = parse_csv(args.blur_sigmas, float)
    if len(group_sizes) != len(blur_sigmas):
        raise ValueError("group and sigma counts must match")
    prefix_ends = []
    running = 0
    for size in group_sizes:
        if size <= 0:
            raise ValueError("group sizes must be positive")
        running += size
        prefix_ends.append(running)

    device = torch.device("cuda")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TokenizerConfig(**payload["model_config"])
    if running != config.num_latents:
        raise ValueError("group sizes must sum to checkpoint num_latents")
    model = ProgressiveTokenizer(config).to(device).eval()
    model.load_state_dict(payload["model"])

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

    levels = len(group_sizes)
    cumulative_sse = [0.0] * levels
    full_image_sse = [0.0] * levels
    innovation_sse = [0.0] * levels
    innovation_dot = [0.0] * levels
    innovation_prediction_energy = [0.0] * levels
    innovation_target_energy = [0.0] * levels
    element_count = 0
    consumed = 0
    preview = None
    with torch.no_grad():
        for images, _ in loader:
            if consumed >= args.examples:
                break
            images = images[: args.examples - consumed].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents = model.encode(images)
                reconstructions = torch.stack(
                    [model.decode(latents, prefix) for prefix in prefix_ends],
                    dim=1,
                ).float()
            targets = gaussian_lowpass_pyramid_fft(images, blur_sigmas)
            zero = torch.zeros_like(images)
            previous_reconstruction = zero
            previous_target = zero
            for index in range(levels):
                reconstruction = reconstructions[:, index]
                target = targets[:, index]
                prediction_delta = reconstruction - previous_reconstruction
                target_delta = target - previous_target
                cumulative_sse[index] += float((reconstruction - target).square().sum())
                full_image_sse[index] += float((reconstruction - images).square().sum())
                innovation_sse[index] += float(
                    (prediction_delta - target_delta).square().sum()
                )
                innovation_dot[index] += float((prediction_delta * target_delta).sum())
                innovation_prediction_energy[index] += float(prediction_delta.square().sum())
                innovation_target_energy[index] += float(target_delta.square().sum())
                previous_reconstruction = reconstruction
                previous_target = target
            if preview is None:
                count = min(args.preview_examples, images.shape[0])
                preview = torch.cat(
                    [images[:count]]
                    + [targets[:count, index] for index in range(levels)]
                    + [reconstructions[:count, index] for index in range(levels)],
                    dim=0,
                ).cpu()
            element_count += images.numel()
            consumed += images.shape[0]

    groups = []
    for index, (size, prefix, sigma) in enumerate(
        zip(group_sizes, prefix_ends, blur_sigmas)
    ):
        cumulative_mse = cumulative_sse[index] / element_count
        full_image_mse = full_image_sse[index] / element_count
        innovation_mse = innovation_sse[index] / element_count
        denominator = math.sqrt(
            innovation_prediction_energy[index] * innovation_target_energy[index]
        )
        groups.append(
            {
                "group": index + 1,
                "size": size,
                "prefix": prefix,
                "blur_sigma_pixels": sigma,
                "cumulative_target_mse": cumulative_mse,
                "cumulative_target_psnr_db": pixel_psnr(cumulative_mse),
                "full_image_mse": full_image_mse,
                "full_image_psnr_db": pixel_psnr(full_image_mse),
                "innovation_target_mse": innovation_mse,
                "innovation_cosine": (
                    innovation_dot[index] / denominator if denominator > 0 else 0.0
                ),
                "prediction_increment_rms": math.sqrt(
                    innovation_prediction_energy[index] / element_count
                ),
                "target_increment_rms": math.sqrt(
                    innovation_target_energy[index] / element_count
                ),
            }
        )

    output = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": consumed,
        "group_sizes": group_sizes,
        "prefix_ends": prefix_ends,
        "blur_sigmas_pixels": blur_sigmas,
        "groups": groups,
        "mean_cumulative_target_mse": sum(
            group["cumulative_target_mse"] for group in groups
        ) / levels,
        "mean_innovation_target_mse": sum(
            group["innovation_target_mse"] for group in groups
        ) / levels,
        "mean_innovation_cosine": sum(
            group["innovation_cosine"] for group in groups
        ) / levels,
        "visual_layout": (
            "rows are source, six Gaussian targets coarse-to-full, and six "
            "prefix reconstructions; columns are fixed CIFAR test examples"
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    assert preview is not None
    image_path = path.with_name("prefix_contact_sheet.png")
    save_image(
        preview.add(1).div(2).clamp(0, 1),
        image_path,
        nrow=min(args.preview_examples, consumed),
    )
    print(json.dumps({**output, "visual": str(image_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
