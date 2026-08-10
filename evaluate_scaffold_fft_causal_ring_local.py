#!/usr/bin/env python3
"""Evaluate causal-ring FFT residual generation with aligned local computation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from control_pixel_diffusion import build_compact_isometric_codec, patchify
from evaluate_scaffold_fft_residual import ImageAccumulator
from live_evaluation import InceptionFeatures
from scaffold_fft_causal_ring_local import (
    CausalRingLocalDenoiser,
    model_args_from_joint_checkpoint,
    sample_causal_ring_fft,
)
from train_scaffold_fft_residual import deterministic_scaffold, fft_state_to_images
from train_spatial_latent_hartley_ar import load_spatial_ae


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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--preview_images", type=int, default=64)
    parser.add_argument(
        "--condition_mode", choices=("aligned", "shuffled", "zero"), default="aligned"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 2 or args.batch_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("num_samples >= 2 and positive batch/solver steps are required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "scaffold_fft_causal_ring_local":
        raise ValueError("checkpoint is not a causal-ring local-compute model")
    saved = payload["joint_model_args"]
    image_size = int(saved["image_size"])
    patch = int(saved["patch"])
    token_dim = int(saved["compact_token_dim"])
    local_tokens = (image_size // patch) ** 2
    patch_dim = 3 * patch**2
    scalar_ring = payload["scalar_ring"].to(device)
    ring_count = int(scalar_ring.max()) + 1
    model = CausalRingLocalDenoiser(
        tokens=local_tokens,
        patch_dim=patch_dim,
        ring_count=ring_count,
        args=model_args_from_joint_checkpoint(saved),
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
    scaffold_accumulator = ImageAccumulator(codec.num_bins)
    completion_accumulator = ImageAccumulator(codec.num_bins)
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
            sampled_fft = sample_causal_ring_fft(
                model,
                codec,
                condition_patches,
                scalar_ring,
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
        scaffold_accumulator.update(scaffold, extractor(scaffold), codec)
        completion_accumulator.update(completion, extractor(completion), codec)
        scaffold_squared_error += float((images - scaffold).double().square().sum())
        completion_squared_error += float((images - completion).double().square().sum())
        paired_value_count += images.numel()
        preview_count = sum(item.shape[0] for item in previews)
        if preview_count < args.preview_images:
            keep = min(args.preview_images - preview_count, images.shape[0])
            previews.append(
                torch.stack([images[:keep], scaffold[:keep], completion[:keep]], dim=1)
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
        "inference_steps_per_ring": args.num_inference_steps,
        "ring_count": ring_count,
        "condition_mode": args.condition_mode,
        "normalization": normalization,
        "paired_scaffold_psnr": -10.0 * math.log10(max(scaffold_mse, 1e-12)),
        "paired_completion_psnr": -10.0 * math.log10(max(completion_mse, 1e-12)),
        "scaffold": scaffold_accumulator.compute(reference, "deterministic_c4_scaffold"),
        "completion": completion_accumulator.compute(
            reference, f"causal_ring_local_completion_{args.condition_mode}"
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if previews:
        preview = torch.cat(previews).reshape(-1, 3, image_size, image_size)
        save_image(preview, output_dir / "paired_samples.png", nrow=3)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
