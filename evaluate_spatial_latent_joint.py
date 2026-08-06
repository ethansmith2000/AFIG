#!/usr/bin/env python3
"""Evaluate a joint spatial-AE latent checkpoint with the shared CIFAR protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import PatchDiffusion, build_compact_isometric_codec
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid, _radial_power
from train_spatial_latent_hartley_ar import (
    load_spatial_ae,
    tokens_to_latent_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = payload["args"]
    ae_checkpoint = payload.get("ae_checkpoint", saved["ae_checkpoint"])
    autoencoder = load_spatial_ae(ae_checkpoint, device)
    autoencoder.eval()
    ae_config = autoencoder.config
    latent_size = ae_config.spatial_resolution // ae_config.spatial_downsample
    patch = int(saved["latent_patch"])
    if latent_size % patch:
        raise ValueError("latent map size is incompatible with saved latent_patch")
    token_count = (latent_size // patch) ** 2
    token_dim = ae_config.spatial_latent_channels * patch**2
    model_args = SimpleNamespace(
        width=int(saved["width"]),
        num_layers=int(saved["num_layers"]),
        num_heads=int(saved["num_heads"]),
        ff_mult=int(saved["ff_mult"]),
        flow_path=str(saved.get("flow_path", "linear")),
    )
    model = PatchDiffusion(token_count, token_dim, model_args).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    channel_mean = payload["channel_mean"].to(device)
    channel_std = payload["channel_std"].to(device)
    latent_basis = str(saved["latent_basis"])

    reference = torch.load(
        args.reference_cache, map_location="cpu", weights_only=False
    )
    extractor = InceptionFeatures(device)
    radial_codec = build_compact_isometric_codec(
        ae_config.spatial_resolution, device
    )

    # Reset after model/Inception construction so all arms use the same base
    # samples regardless of loading-side RNG consumption.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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
    minimum = float("inf")
    maximum = float("-inf")
    generated = 0

    while generated < args.num_samples:
        current = min(args.batch_size, args.num_samples - generated)
        with torch.no_grad(), torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            values = model.sample(current, args.num_inference_steps, device)
            maps = tokens_to_latent_maps(
                values.float(),
                channel_mean,
                channel_std,
                patch,
                latent_size,
                basis=latent_basis,
            )
            images = autoencoder.decode(
                maps.to(next(autoencoder.parameters()).dtype)
            ).float()

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
        pixel_count += current * ae_config.spatial_resolution**2
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
        generated += current
        print(f"generated={generated}/{args.num_samples}", flush=True)

    generated_mean, generated_covariance = moments.compute()
    feature_tensor = torch.cat(generated_features)
    channel_mean_image = channel_sum / pixel_count
    channel_std_image = (
        channel_sum_sq / pixel_count - channel_mean_image.square()
    ).clamp_min(0.0).sqrt()
    radial = radial_total / generated
    reference_radial = reference["radial_power"].double()
    metrics = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": int(payload["step"]),
        "latent_basis": latent_basis,
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
        "clipping_fraction": clipping_count / max(value_count, 1),
        "unclipped_min": minimum,
        "unclipped_max": maximum,
        "image_gradient_energy": gradient_total / generated,
        "radial_power_relative_error": float(
            ((radial - reference_radial).abs() / reference_radial.clamp_min(1e-8))
            .mean()
            .item()
        ),
        "channel_mean": [float(value) for value in channel_mean_image],
        "channel_std": [float(value) for value in channel_std_image],
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
