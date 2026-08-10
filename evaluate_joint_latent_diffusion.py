#!/usr/bin/env python3
"""Evaluate joint latent diffusion with the shared CIFAR-10 FID/KID protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import build_compact_isometric_codec
from latent_autoencoder_interface import FrozenLatentAutoencoder
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid, _radial_power
from train_joint_latent_diffusion import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--reference_cache",
        default="continuous_runs/cifar10_inception_reference_radial.pt",
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--preview_images", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if args.batch_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("batch_size and num_inference_steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    adapter = FrozenLatentAutoencoder(
        args.ae_checkpoint,
        args.latent_interface,
        sample_posterior=False,
    ).to(device)
    model, checkpoint_step = load_checkpoint(args.checkpoint, adapter)
    model = model.to(device).eval()
    reference = torch.load(
        args.reference_cache, map_location="cpu", weights_only=False
    )
    extractor = InceptionFeatures(device)
    radial_codec = build_compact_isometric_codec(32, device)

    # Loading the model and Inception can consume RNG. A dedicated generator
    # gives every checkpoint exactly the same sequence of base samples.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    moments = StreamingMoments(2048)
    generated_features = []
    previews = []
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    clipping_count = 0
    value_count = 0
    radial_total = torch.zeros(radial_codec.num_bins, dtype=torch.float64)
    gradient_total = 0.0
    latent_square_total = 0.0
    latent_value_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    generated = 0

    while generated < args.num_samples:
        current = min(args.batch_size, args.num_samples - generated)
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            latents = model.generate_latents(
                current,
                adapter.position_features,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )
            images = adapter.decode_latents(latents).float()

        features = extractor(images)
        moments.update(features)
        generated_features.append(features.cpu())
        preview_count = sum(item.shape[0] for item in previews)
        if preview_count < args.preview_images:
            previews.append(
                images[: args.preview_images - preview_count].clamp(0.0, 1.0).cpu()
            )

        channel_sum += images.double().sum(dim=(0, 2, 3)).cpu()
        channel_sum_sq += images.double().square().sum(dim=(0, 2, 3)).cpu()
        pixel_count += current * 32 * 32
        clipping_count += int(((images < 0.0) | (images > 1.0)).sum().item())
        value_count += images.numel()
        minimum = min(minimum, float(images.min()))
        maximum = max(maximum, float(images.max()))
        vertical = images[:, :, 1:] - images[:, :, :-1]
        horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
        gradient_total += float(
            0.5 * (vertical.square().mean() + horizontal.square().mean())
        ) * current
        radial_total += _radial_power(images, radial_codec).double().cpu() * current
        latent_square_total += float(latents.float().square().sum())
        latent_value_count += latents.numel()
        generated += current
        print(f"generated={generated}/{args.num_samples}", flush=True)

    generated_mean, generated_covariance = moments.compute()
    feature_tensor = torch.cat(generated_features)
    channel_mean = channel_sum / pixel_count
    channel_std = (
        channel_sum_sq / pixel_count - channel_mean.square()
    ).clamp_min(0.0).sqrt()
    radial = radial_total / generated
    reference_radial = reference["radial_power"].double()
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "samples": generated,
        "seed": args.seed,
        "inference_steps": args.num_inference_steps,
        "fid_5k": _fid(
            reference["feature_mean"],
            reference["feature_covariance"],
            generated_mean,
            generated_covariance,
        ),
        "kid_5k": _kid(reference["kid_features"], feature_tensor),
        "latent_rms": math.sqrt(latent_square_total / latent_value_count),
        "clipping_fraction": clipping_count / max(value_count, 1),
        "unclipped_min": minimum,
        "unclipped_max": maximum,
        "image_gradient_energy": gradient_total / generated,
        "radial_power_relative_error": float(
            ((radial - reference_radial).abs() / reference_radial.clamp_min(1e-8))
            .mean()
            .item()
        ),
        "channel_mean": [float(value) for value in channel_mean],
        "channel_std": [float(value) for value in channel_std],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if previews:
        preview = torch.cat(previews)
        save_image(
            preview,
            output_dir / "samples.png",
            nrow=min(8, max(1, int(math.sqrt(preview.shape[0])))),
        )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
