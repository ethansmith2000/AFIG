#!/usr/bin/env python3
"""Evaluate oracle C4 scaffolds and compact-FFT residual completions together."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from control_pixel_diffusion import build_compact_isometric_codec
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid, _radial_power
from train_scaffold_fft_residual import (
    ScaffoldResidualDenoiser,
    deterministic_scaffold,
    fft_state_to_images,
    model_args_from_checkpoint,
    sample_residual_fft,
)
from train_spatial_latent_hartley_ar import load_spatial_ae
from control_pixel_diffusion import patchify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--reference_cache",
        default="continuous_runs/cifar10_inception_reference_radial.pt",
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--preview_images", type=int, default=64)
    parser.add_argument(
        "--condition_mode",
        choices=("aligned", "shuffled", "zero"),
        default="aligned",
        help=(
            "Condition the residual sampler on its aligned scaffold, a within-batch "
            "permutation, or a zero normalized context. The sampled residual is "
            "always added to the original aligned scaffold."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ImageAccumulator:
    def __init__(self, radial_bins: int):
        self.moments = StreamingMoments(2048)
        self.features: list[torch.Tensor] = []
        self.channel_sum = torch.zeros(3, dtype=torch.float64)
        self.channel_sum_sq = torch.zeros(3, dtype=torch.float64)
        self.radial_total = torch.zeros(radial_bins, dtype=torch.float64)
        self.gradient_total = 0.0
        self.pixel_count = 0
        self.image_count = 0
        self.clipping_count = 0
        self.value_count = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, images: torch.Tensor, features: torch.Tensor, radial_codec) -> None:
        count, _, height, width = images.shape
        self.moments.update(features)
        self.features.append(features.cpu())
        self.channel_sum += images.double().sum(dim=(0, 2, 3)).cpu()
        self.channel_sum_sq += images.double().square().sum(dim=(0, 2, 3)).cpu()
        self.pixel_count += count * height * width
        self.image_count += count
        self.clipping_count += int(((images < 0.0) | (images > 1.0)).sum())
        self.value_count += images.numel()
        self.minimum = min(self.minimum, float(images.min()))
        self.maximum = max(self.maximum, float(images.max()))
        vertical = images[:, :, 1:] - images[:, :, :-1]
        horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
        self.gradient_total += float(
            0.5 * (vertical.square().mean() + horizontal.square().mean())
        ) * count
        self.radial_total += _radial_power(images, radial_codec).double().cpu() * count

    def compute(self, reference: dict, name: str) -> dict[str, object]:
        mean, covariance = self.moments.compute()
        channel_mean = self.channel_sum / self.pixel_count
        channel_std = (
            self.channel_sum_sq / self.pixel_count - channel_mean.square()
        ).clamp_min(0.0).sqrt()
        radial = self.radial_total / self.image_count
        reference_radial = reference["radial_power"].double()
        return {
            "name": name,
            "fid_5k": _fid(
                reference["feature_mean"],
                reference["feature_covariance"],
                mean,
                covariance,
            ),
            "kid_5k": _kid(reference["kid_features"], torch.cat(self.features)),
            "clipping_fraction": self.clipping_count / max(self.value_count, 1),
            "unclipped_min": self.minimum,
            "unclipped_max": self.maximum,
            "image_gradient_energy": self.gradient_total / self.image_count,
            "radial_power_relative_error": float(
                (
                    (radial - reference_radial).abs()
                    / reference_radial.clamp_min(1e-8)
                )
                .mean()
                .item()
            ),
            "channel_mean": [float(value) for value in channel_mean],
            "channel_std": [float(value) for value in channel_std],
        }


def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if args.batch_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("batch_size and inference steps must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = payload["args"]
    image_size = int(saved["image_size"])
    patch = int(saved["patch"])
    token_dim = int(saved["compact_token_dim"])
    local_tokens = (image_size // patch) ** 2
    patch_dim = 3 * patch**2
    model = ScaffoldResidualDenoiser(
        local_tokens, patch_dim, model_args_from_checkpoint(saved)
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    autoencoder = load_spatial_ae(payload["ae_checkpoint"], device)

    codec = build_compact_isometric_codec(image_size, device)
    layout_orbit = payload["compact_layout_orbit"].to(device)
    layout_component = payload["compact_layout_component"].to(device)
    normalization = payload["normalization"]
    scaffold_mean = float(normalization["scaffold_mean"])
    scaffold_std = float(normalization["scaffold_std"])
    residual_mean = float(normalization["residual_mean"])
    residual_std = float(normalization["residual_std"])

    test_set = datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transforms.ToTensor()
    )
    sample_count = min(args.num_samples, len(test_set))
    loader = DataLoader(
        Subset(test_set, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)
    radial_codec = build_compact_isometric_codec(image_size, device)
    scaffold_accumulator = ImageAccumulator(radial_codec.num_bins)
    completion_accumulator = ImageAccumulator(radial_codec.num_bins)
    scaffold_squared_error = 0.0
    completion_squared_error = 0.0
    paired_value_count = 0
    previews: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(args.seed)

    generated = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            scaffold = deterministic_scaffold(autoencoder, images)
            scaffold_patches = patchify(
                (scaffold - scaffold_mean) / scaffold_std, patch
            )
            if args.condition_mode == "shuffled":
                condition_patches = scaffold_patches.roll(1, dims=0)
            elif args.condition_mode == "zero":
                condition_patches = torch.zeros_like(scaffold_patches)
            else:
                condition_patches = scaffold_patches
            sampled_fft = sample_residual_fft(
                model,
                codec,
                condition_patches,
                layout_orbit=layout_orbit,
                layout_component=layout_component,
                patch=patch,
                image_size=image_size,
                token_dim=token_dim,
                steps=args.num_inference_steps,
                generator=generator,
            )
        normalized_residual = fft_state_to_images(
            codec, sampled_fft, layout_orbit, layout_component
        )
        completion = scaffold + normalized_residual * residual_std + residual_mean
        scaffold_features = extractor(scaffold)
        completion_features = extractor(completion)
        scaffold_accumulator.update(scaffold, scaffold_features, radial_codec)
        completion_accumulator.update(completion, completion_features, radial_codec)

        scaffold_squared_error += float((images - scaffold).double().square().sum())
        completion_squared_error += float((images - completion).double().square().sum())
        paired_value_count += images.numel()
        preview_count = sum(item.shape[0] for item in previews)
        if preview_count < args.preview_images:
            keep = min(args.preview_images - preview_count, images.shape[0])
            previews.append(
                torch.stack(
                    [images[:keep], scaffold[:keep], completion[:keep]], dim=1
                )
                .clamp(0, 1)
                .cpu()
            )
        generated += images.shape[0]
        print(f"generated={generated}/{sample_count}", flush=True)

    scaffold_mse = scaffold_squared_error / paired_value_count
    completion_mse = completion_squared_error / paired_value_count
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(payload["step"]),
        "samples": generated,
        "seed": args.seed,
        "inference_steps": args.num_inference_steps,
        "condition_mode": args.condition_mode,
        "normalization": normalization,
        "paired_scaffold_psnr": -10.0 * math.log10(max(scaffold_mse, 1e-12)),
        "paired_completion_psnr": -10.0 * math.log10(max(completion_mse, 1e-12)),
        "scaffold": scaffold_accumulator.compute(reference, "deterministic_c4_scaffold"),
        "completion": completion_accumulator.compute(
            reference, f"sampled_fft_residual_completion_{args.condition_mode}"
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if previews:
        # Each example occupies one row: reference, scaffold, sampled completion.
        preview = torch.cat(previews).reshape(-1, 3, image_size, image_size)
        save_image(preview, output_dir / "paired_samples.png", nrow=3)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
