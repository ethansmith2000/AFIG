#!/usr/bin/env python3
"""Train scaffold-conditioned causal radial-ring compact-FFT residual generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from control_pixel_diffusion import patchify
from scaffold_fft_ring import ScaffoldFFTRingConfig, ScaffoldFFTRingModel
from train_scaffold_fft_residual import (
    deterministic_scaffold,
    fft_state_to_images,
    images_to_fft_state,
    make_compact_layout,
    psnr,
)
from train_spatial_latent_hartley_ar import load_spatial_ae


DEFAULT_INIT = (
    "latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000/"
    "checkpoint_30000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--compact_token_dim", type=int, default=48)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--scaffold_layers", type=int, default=4)
    parser.add_argument("--ring_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diffusion_width", type=int, default=768)
    parser.add_argument("--diffusion_depth", type=int, default=6)
    parser.add_argument("--diffusion_batch_mul", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--validation_images", type=int, default=16)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def initialize_scaffold_interface(
    model: ScaffoldFFTRingModel, source_state: dict[str, torch.Tensor]
) -> list[str]:
    """Reuse only the proven scaffold input adapter and absolute patch slots."""
    copied = []
    mappings = {
        "scaffold_projection.weight": "scaffold_projection.weight",
        "scaffold_projection.bias": "scaffold_projection.bias",
        "position": "scaffold_position",
    }
    target_state = model.state_dict()
    with torch.no_grad():
        for source_name, target_name in mappings.items():
            if source_name not in source_state:
                continue
            source = source_state[source_name]
            if target_state[target_name].shape != source.shape:
                continue
            target_state[target_name].copy_(source)
            copied.append(f"{source_name}->{target_name}")
    return copied


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.width = 64
        args.scaffold_layers = 1
        args.ring_layers = 1
        args.num_heads = 4
        args.ff_mult = 2
        args.diffusion_width = 64
        args.diffusion_depth = 2
        args.warmup = 1
        args.validation_images = 2
        args.preview_steps = 1
        args.checkpoint_steps = 1
        args.inference_steps = 2

    if args.image_size != 32 or args.patch != 4 or args.compact_token_dim != 48:
        raise ValueError("this controlled gate requires the matched 32x32, 4x4, 64x48 setup")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    init_payload = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    normalization = init_payload["normalization"]
    required_stats = ("scaffold_mean", "scaffold_std", "residual_mean", "residual_std")
    if any(name not in normalization for name in required_stats):
        raise ValueError("initialization checkpoint lacks scaffold/residual scalar statistics")
    autoencoder = load_spatial_ae(init_payload["ae_checkpoint"], device)
    if autoencoder.config.variational:
        raise ValueError("the oracle scaffold gate requires a deterministic AE")

    codec, layout_orbit, layout_component = make_compact_layout(args.image_size, device)
    saved_orbit = init_payload["compact_layout_orbit"].to(device)
    saved_component = init_payload["compact_layout_component"].to(device)
    if not torch.equal(layout_orbit, saved_orbit) or not torch.equal(
        layout_component, saved_component
    ):
        raise RuntimeError("current compact FFT layout differs from the Stage-C checkpoint")
    scalar_ring = codec.radius_bin[layout_orbit]
    ring_counts = torch.bincount(scalar_ring, minlength=codec.num_bins)

    local_tokens = (args.image_size // args.patch) ** 2
    patch_dim = 3 * args.patch**2
    config = ScaffoldFFTRingConfig(
        local_tokens=local_tokens,
        patch_dim=patch_dim,
        ring_count=codec.num_bins,
        max_ring_dim=int(ring_counts.max()),
        width=args.width,
        scaffold_layers=args.scaffold_layers,
        ring_layers=args.ring_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        diffusion_width=args.diffusion_width,
        diffusion_depth=args.diffusion_depth,
        diffusion_batch_mul=args.diffusion_batch_mul,
        num_inference_steps=args.inference_steps,
    )
    model = ScaffoldFFTRingModel(scalar_ring, config).to(device)
    copied = initialize_scaffold_interface(model, init_payload["model"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "scaffold FFT rings: "
        f"rings={config.ring_count}, dimensions={ring_counts.tolist()}, "
        f"max={config.max_ring_dim}, params={parameter_count / 1e6:.2f}M"
    )
    print("normalization=" + json.dumps(normalization, sort_keys=True))
    print("initialized=" + json.dumps(copied))

    train_set = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    test_set = datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transforms.ToTensor()
    )
    validation_images = torch.stack(
        [test_set[index][0] for index in range(args.validation_images)]
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        validation_scaffold = deterministic_scaffold(autoencoder, validation_images)

    scaffold_mean = float(normalization["scaffold_mean"])
    scaffold_std = float(normalization["scaffold_std"])
    residual_mean = float(normalization["residual_mean"])
    residual_std = float(normalization["residual_std"])
    validation_scaffold_patches = patchify(
        (validation_scaffold - scaffold_mean) / scaffold_std, args.patch
    )
    save_image(
        torch.cat([validation_images, validation_scaffold], dim=0).clamp(0, 1),
        output_dir / "scaffold_baseline.png",
        nrow=args.validation_images,
    )
    print(f"validation scaffold PSNR={psnr(validation_images, validation_scaffold):.3f}dB")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    history: list[dict[str, float | int]] = []
    global_step = 0
    progress = tqdm(total=args.steps, desc="scaffold-fft-rings")
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_scale(global_step)
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scaffold = deterministic_scaffold(autoencoder, images)
            residual = (images.float() - scaffold - residual_mean) / residual_std
            scaffold_patches = patchify(
                (scaffold - scaffold_mean) / scaffold_std, args.patch
            )
            target_fft = images_to_fft_state(
                codec,
                residual,
                layout_orbit,
                layout_component,
                args.compact_token_dim,
            )
            model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(target_fft, scaffold_patches)
                loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0 or global_step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=optimizer.param_groups[0]["lr"],
                )
            if global_step % 100 == 0 or global_step == args.steps:
                history.append(
                    {
                        "step": global_step,
                        "loss": float(loss.detach()),
                        "unweighted_mse": float(output["unweighted_mse"]),
                        "grad_norm": float(grad_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            if args.preview_steps > 0 and global_step % args.preview_steps == 0:
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    sampled_flat = model.generate_fft(
                        validation_scaffold_patches,
                        num_inference_steps=args.inference_steps,
                        generator=generator,
                    )
                sampled_normalized = fft_state_to_images(
                    codec,
                    sampled_flat,
                    layout_orbit,
                    layout_component,
                )
                completion = validation_scaffold + sampled_normalized * residual_std + residual_mean
                completion_psnr = psnr(validation_images, completion.clamp(0, 1))
                save_image(
                    torch.cat(
                        [validation_images, validation_scaffold, completion], dim=0
                    ).clamp(0, 1),
                    output_dir / f"preview_{global_step:07d}.png",
                    nrow=args.validation_images,
                )
                print(f"PREVIEW step={global_step} completion_psnr={completion_psnr:.3f}dB")

            if args.checkpoint_steps > 0 and global_step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "kind": "scaffold_fft_ring_residual",
                        "step": global_step,
                        "model": model.state_dict(),
                        "model_config": config.fingerprint(),
                        "args": vars(args),
                        "ae_checkpoint": str(Path(init_payload["ae_checkpoint"]).resolve()),
                        "initialization_checkpoint": str(Path(args.init_checkpoint).resolve()),
                        "initialized_parameters": copied,
                        "normalization": normalization,
                        "compact_layout_orbit": layout_orbit.cpu(),
                        "compact_layout_component": layout_component.cpu(),
                        "scalar_ring": scalar_ring.cpu(),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    with (output_dir / "history.json").open("w") as handle:
        json.dump(
            {
                "model_config": config.fingerprint(),
                "normalization": normalization,
                "parameter_count": parameter_count,
                "ring_counts": ring_counts.tolist(),
                "initialized_parameters": copied,
                "history": history,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print("done")


if __name__ == "__main__":
    main()

