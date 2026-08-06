#!/usr/bin/env python3
"""Evaluate saved direct-control checkpoints with a shared CIFAR FID/KID protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import (
    PatchDiffusion,
    build_compact_isometric_codec,
    compact_isometric_tokens_to_images,
    compact_scalar_tokens_to_images,
    full_idctify,
    full_ihartleyify,
    orbit_order_permutation,
    patch_idctify,
    patch_grid_idctify,
    unpatchify,
)
from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid, _radial_power


REPRESENTATIONS = (
    "pixels",
    "patch_dct",
    "patch_grid_dct",
    "full_dct",
    "full_hartley",
    "fft_compact_isometric_spiral",
    "fft_compact_isometric_gridlocal",
    "fft_compact_isometric_scale",
)

COMPACT_REPRESENTATIONS = (
    "fft_compact_isometric_spiral",
    "fft_compact_isometric_gridlocal",
    "fft_compact_isometric_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
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
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _layout(args: argparse.Namespace) -> tuple[int, int]:
    if args.representation in COMPACT_REPRESENTATIONS:
        total = 3 * args.image_size * args.image_size
        if total % 48:
            raise ValueError("compact control requires 48 to divide image scalars")
        return total // 48, 48
    tokens = (args.image_size // args.patch) ** 2
    return tokens, 3 * args.patch * args.patch


def _decode(
    values: torch.Tensor,
    args: argparse.Namespace,
    mean: float,
    std: float,
    codec,
    permutation: torch.Tensor | None,
    layout_orbit: torch.Tensor | None,
    layout_component: torch.Tensor | None,
) -> torch.Tensor:
    if args.representation == "pixels":
        normalized = unpatchify(values, args.patch, args.image_size)
    elif args.representation == "patch_dct":
        normalized = patch_idctify(values, args.patch, args.image_size)
    elif args.representation == "patch_grid_dct":
        normalized = patch_grid_idctify(values, args.patch, args.image_size)
    elif args.representation == "full_dct":
        normalized = full_idctify(values, args.patch, args.image_size)
    elif args.representation == "full_hartley":
        normalized = full_ihartleyify(values, args.patch, args.image_size)
    else:
        if codec is None:
            raise RuntimeError("compact FFT decoder was not initialized")
        if layout_orbit is not None and layout_component is not None:
            normalized = compact_scalar_tokens_to_images(
                codec, values, layout_orbit, layout_component
            )
        else:
            if permutation is None:
                raise RuntimeError("legacy compact FFT permutation is missing")
            normalized = compact_isometric_tokens_to_images(codec, values, permutation)
    return normalized * std + mean


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

    checkpoint_dir = Path(args.checkpoint_dir)
    history = json.loads((checkpoint_dir / "history.json").read_text())
    mean = float(history["mean"])
    std = float(history["std"])
    tokens, dim = _layout(args)
    model_args = SimpleNamespace(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        flow_path="linear",
    )
    model = PatchDiffusion(tokens, dim, model_args).to(device)
    payload = torch.load(checkpoint_dir / "final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    # One radial codec is used only to reproduce the cached reference's radial
    # summary convention. Compact decoding uses the same object when applicable.
    radial_codec = build_compact_isometric_codec(args.image_size, device)
    codec = radial_codec if args.representation in COMPACT_REPRESENTATIONS else None
    permutation = None
    layout_orbit = None
    layout_component = None
    if codec is not None:
        saved_permutation = payload.get("compact_orbit_permutation")
        permutation = (
            saved_permutation.to(device)
            if saved_permutation is not None
            else orbit_order_permutation(codec, "square_spiral")
        )
        if args.representation != "fft_compact_isometric_spiral":
            saved_orbit = payload.get("compact_layout_orbit")
            saved_component = payload.get("compact_layout_component")
            if saved_orbit is None or saved_component is None:
                raise RuntimeError(
                    "corrected compact checkpoint is missing its frozen scalar layout"
                )
            layout_orbit = saved_orbit.to(device)
            layout_component = saved_component.to(device)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)

    # Model construction and Inception loading may consume RNG; reset immediately
    # before the first base sample so every representation sees the same seed.
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
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            values = model.sample(current, args.num_inference_steps, device)
        images = _decode(
            values.float(),
            args,
            mean,
            std,
            codec,
            permutation,
            layout_orbit,
            layout_component,
        ).float()
        features = extractor(images)
        moments.update(features)
        generated_features.append(features.cpu())
        if sum(item.shape[0] for item in previews) < args.preview_images:
            remaining = args.preview_images - sum(item.shape[0] for item in previews)
            previews.append(images[:remaining].clamp(0.0, 1.0).cpu())

        channel_sum += images.double().sum(dim=(0, 2, 3)).cpu()
        channel_sum_sq += images.double().square().sum(dim=(0, 2, 3)).cpu()
        pixel_count += current * args.image_size * args.image_size
        clipping_count += int(((images < 0.0) | (images > 1.0)).sum().item())
        value_count += images.numel()
        minimum = min(minimum, float(images.min().item()))
        maximum = max(maximum, float(images.max().item()))
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
    channel_mean = channel_sum / pixel_count
    channel_std = (
        channel_sum_sq / pixel_count - channel_mean.square()
    ).clamp_min(0.0).sqrt()
    radial = radial_total / generated
    reference_radial = reference["radial_power"].double()
    metrics = {
        "representation": args.representation,
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
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
