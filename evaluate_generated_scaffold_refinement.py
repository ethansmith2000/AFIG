#!/usr/bin/env python3
"""Apply the oracle-trained FFT residual refiner to generated local-C4 scaffolds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import PatchDiffusion, build_compact_isometric_codec, patchify
from evaluate_scaffold_fft_residual import ImageAccumulator
from live_evaluation import InceptionFeatures
from train_hartley_ar import HartleyTileAR
from train_scaffold_fft_residual import (
    ScaffoldResidualDenoiser,
    fft_state_to_images,
    model_args_from_checkpoint,
    sample_residual_fft,
)
from train_spatial_latent_hartley_ar import (
    load_spatial_ae,
    tokens_to_latent_maps,
    ungroup_radial_hartley_tiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold_checkpoint", required=True)
    parser.add_argument(
        "--scaffold_model",
        choices=("joint", "ar"),
        default="joint",
        help="Architecture stored in scaffold_checkpoint.",
    )
    parser.add_argument("--refiner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--reference_cache",
        default="continuous_runs/cifar10_inception_reference_radial.pt",
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scaffold_inference_steps", type=int, default=50)
    parser.add_argument("--refiner_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--preview_images", type=int, default=64)
    parser.add_argument(
        "--condition_mode",
        choices=("aligned", "shuffled", "zero"),
        default="aligned",
        help=(
            "Condition on aligned, within-batch shuffled, or zero normalized "
            "scaffold patches while always adding the residual to the original "
            "generated scaffold."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 2 or args.batch_size <= 0:
        raise ValueError("num_samples must be >=2 and batch_size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    scaffold_payload = torch.load(
        args.scaffold_checkpoint, map_location="cpu", weights_only=False
    )
    scaffold_saved = scaffold_payload["args"]
    autoencoder = load_spatial_ae(scaffold_payload["ae_checkpoint"], device)
    ae_config = autoencoder.config
    latent_size = ae_config.spatial_resolution // ae_config.spatial_downsample
    latent_patch = int(scaffold_saved["latent_patch"])
    physical_grid = latent_size // latent_patch
    latent_basis = str(scaffold_saved["latent_basis"])
    tiles_per_token = int(scaffold_saved.get("tiles_per_token", 1))
    if args.scaffold_model == "joint":
        scaffold_tokens = physical_grid**2
        scaffold_dim = ae_config.spatial_latent_channels * latent_patch**2
        scaffold_model_args = SimpleNamespace(
            width=int(scaffold_saved["width"]),
            num_layers=int(scaffold_saved["num_layers"]),
            num_heads=int(scaffold_saved["num_heads"]),
            ff_mult=int(scaffold_saved["ff_mult"]),
            flow_path=str(scaffold_saved.get("flow_path", "linear")),
        )
        scaffold_model = PatchDiffusion(
            scaffold_tokens, scaffold_dim, scaffold_model_args
        ).to(device)
        model_token_order = None
    else:
        token_order = str(scaffold_saved.get("token_order", "auto"))
        if token_order == "auto":
            token_order = "radial" if latent_basis == "hartley" else "raster"
        if latent_basis in ("block_dct", "compact_fft"):
            fixed_dim = int(
                scaffold_saved.get(
                    "block_dct_token_dim"
                    if latent_basis == "block_dct"
                    else "compact_fft_token_dim",
                    16,
                )
            )
            group_count = (
                ae_config.spatial_latent_channels * latent_size**2 // fixed_dim
            )
            scaffold_dim = fixed_dim
        else:
            group_count = physical_grid**2 // tiles_per_token
            scaffold_dim = (
                ae_config.spatial_latent_channels
                * latent_patch**2
                * tiles_per_token
            )
        model_grid = math.isqrt(group_count)
        if model_grid**2 != group_count:
            raise ValueError("AR scaffold token count must be a perfect square")
        model_token_order = "raster" if tiles_per_token > 1 else token_order
        scaffold_model = HartleyTileAR(
            width=int(scaffold_saved["width"]),
            num_layers=int(scaffold_saved["layers"]),
            num_heads=int(scaffold_saved["heads"]),
            ff_mult=int(scaffold_saved["ff_mult"]),
            diff_width=int(scaffold_saved["diff_width"]),
            diff_depth=int(scaffold_saved["diff_depth"]),
            inference_steps=int(scaffold_saved["inference_steps"]),
            grid=model_grid,
            token_dim=scaffold_dim,
            token_order=model_token_order,
            rope_mode=str(scaffold_saved.get("rope_mode", "frequency_2d")),
        ).to(device)
    scaffold_model.load_state_dict(scaffold_payload["model"], strict=True)
    scaffold_model.eval()
    channel_mean = scaffold_payload["channel_mean"].to(device)
    channel_std = scaffold_payload["channel_std"].to(device)
    block_dct_token_dim = int(scaffold_saved.get("block_dct_token_dim", 16))
    compact_fft_token_dim = int(scaffold_saved.get("compact_fft_token_dim", 16))
    dct_support = int(scaffold_saved.get("dct_support", 2))

    refiner_payload = torch.load(
        args.refiner_checkpoint, map_location="cpu", weights_only=False
    )
    refiner_saved = refiner_payload["args"]
    refiner_ae = Path(refiner_payload["ae_checkpoint"]).resolve()
    scaffold_ae = Path(scaffold_payload["ae_checkpoint"]).resolve()
    if refiner_ae != scaffold_ae:
        raise ValueError(
            f"scaffold and refiner use different AEs: {scaffold_ae} vs {refiner_ae}"
        )
    image_size = int(refiner_saved["image_size"])
    patch = int(refiner_saved["patch"])
    token_dim = int(refiner_saved["compact_token_dim"])
    local_tokens = (image_size // patch) ** 2
    patch_dim = 3 * patch**2
    refiner = ScaffoldResidualDenoiser(
        local_tokens, patch_dim, model_args_from_checkpoint(refiner_saved)
    ).to(device)
    refiner.load_state_dict(refiner_payload["model"], strict=True)
    refiner.eval()
    normalization = refiner_payload["normalization"]
    scaffold_mean = float(normalization["scaffold_mean"])
    scaffold_std = float(normalization["scaffold_std"])
    residual_mean = float(normalization["residual_mean"])
    residual_std = float(normalization["residual_std"])
    codec = build_compact_isometric_codec(image_size, device)
    layout_orbit = refiner_payload["compact_layout_orbit"].to(device)
    layout_component = refiner_payload["compact_layout_component"].to(device)

    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)
    radial_codec = build_compact_isometric_codec(image_size, device)
    scaffold_accumulator = ImageAccumulator(radial_codec.num_bins)
    completion_accumulator = ImageAccumulator(radial_codec.num_bins)
    previews: list[torch.Tensor] = []
    refiner_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    scaffold_generator = torch.Generator(device=device).manual_seed(args.seed)

    # Reset only after model and extractor construction. PatchDiffusion.sample
    # uses the global generator; the residual sampler has an independent stream.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generated = 0
    while generated < args.num_samples:
        current = min(args.batch_size, args.num_samples - generated)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if args.scaffold_model == "joint":
                raster_tokens = scaffold_model.sample(
                    current, args.scaffold_inference_steps, device
                ).float()
            else:
                ordered_tokens = scaffold_model.generate(
                    current,
                    args.scaffold_inference_steps,
                    scaffold_generator,
                )
                if tiles_per_token > 1:
                    raster_tokens = ungroup_radial_hartley_tiles(
                        ordered_tokens.float(), physical_grid, tiles_per_token
                    )
                else:
                    raster_tokens = scaffold_model.restore_raster(
                        ordered_tokens.float()
                    )
            latent_maps = tokens_to_latent_maps(
                raster_tokens,
                channel_mean,
                channel_std,
                latent_patch,
                latent_size,
                basis=latent_basis,
                dct_support=dct_support,
                block_dct_token_dim=block_dct_token_dim,
                compact_fft_token_dim=compact_fft_token_dim,
            )
            scaffold = autoencoder.decode(
                latent_maps.to(next(autoencoder.parameters()).dtype)
            ).float()
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
                refiner,
                codec,
                condition_patches,
                layout_orbit=layout_orbit,
                layout_component=layout_component,
                patch=patch,
                image_size=image_size,
                token_dim=token_dim,
                steps=args.refiner_inference_steps,
                generator=refiner_generator,
            )
        normalized_residual = fft_state_to_images(
            codec, sampled_fft, layout_orbit, layout_component
        )
        completion = scaffold + normalized_residual * residual_std + residual_mean
        scaffold_accumulator.update(scaffold, extractor(scaffold), radial_codec)
        completion_accumulator.update(completion, extractor(completion), radial_codec)

        preview_count = sum(item.shape[0] for item in previews)
        if preview_count < args.preview_images:
            keep = min(args.preview_images - preview_count, current)
            previews.append(
                torch.stack([scaffold[:keep], completion[:keep]], dim=1)
                .clamp(0, 1)
                .cpu()
            )
        generated += current
        print(f"generated={generated}/{args.num_samples}", flush=True)

    metrics = {
        "scaffold_checkpoint": str(Path(args.scaffold_checkpoint).resolve()),
        "scaffold_checkpoint_step": int(scaffold_payload["step"]),
        "scaffold_model": args.scaffold_model,
        "refiner_checkpoint": str(Path(args.refiner_checkpoint).resolve()),
        "refiner_checkpoint_step": int(refiner_payload["step"]),
        "samples": generated,
        "seed": args.seed,
        "scaffold_inference_steps": args.scaffold_inference_steps,
        "refiner_inference_steps": args.refiner_inference_steps,
        "condition_mode": args.condition_mode,
        "scaffold": scaffold_accumulator.compute(reference, "generated_local_c4_scaffold"),
        "completion": completion_accumulator.compute(
            reference,
            f"generated_scaffold_plus_fft_residual_{args.condition_mode}",
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if previews:
        preview = torch.cat(previews).reshape(-1, 3, image_size, image_size)
        save_image(preview, output_dir / "paired_samples.png", nrow=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
