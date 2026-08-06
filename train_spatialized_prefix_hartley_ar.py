"""Train global-Hartley AR with a local inverse-transformed prefix trunk."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from spatialized_prefix_ar import SpatializedPrefixHartleyAR
from train_spatial_latent_hartley_ar import (
    encode_images,
    fit_channel_stats,
    latent_maps_to_tokens,
    load_spatial_ae,
    tokens_to_latent_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diff_width", type=int, default=768)
    parser.add_argument("--diff_depth", type=int, default=3)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--diagnostic_steps", type=int, default=250)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--validation_images", type=int, default=16)
    parser.add_argument("--latent_patch", type=int, default=2)
    parser.add_argument("--stats_images", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.width = 64
        args.layers = 1
        args.heads = 4
        args.ff_mult = 2
        args.diff_width = 64
        args.diff_depth = 1
        args.inference_steps = 2
        args.preview_steps = 1
        args.diagnostic_steps = 1
        args.checkpoint_steps = 0
        args.validation_images = 2
        args.stats_images = 8

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_spatial_ae(args.ae_checkpoint, device)
    config = autoencoder.config
    latent_size = config.spatial_resolution // config.spatial_downsample
    if latent_size % args.latent_patch:
        raise ValueError("latent map size must be divisible by latent_patch")

    plain = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    channel_mean, channel_std = fit_channel_stats(
        autoencoder, plain, args.stats_images, device
    )
    train_set = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )
    loader = torch.utils.data.DataLoader(
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
        validation_maps = encode_images(autoencoder, validation_images)
    validation_raster = latent_maps_to_tokens(
        validation_maps,
        channel_mean,
        channel_std,
        args.latent_patch,
        basis="hartley",
    )

    model = SpatializedPrefixHartleyAR(
        width=args.width,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        diff_width=args.diff_width,
        diff_depth=args.diff_depth,
        inference_steps=args.inference_steps,
        latent_size=latent_size,
        patch=args.latent_patch,
        channels=config.spatial_latent_channels,
    ).to(device)
    validation_tokens = model.order_tokens(validation_raster)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"spatialized-prefix Hartley AR: {model.seq_len} x {model.token_dim}; "
        f"local-state={model.seq_len} x {model.token_dim}; "
        f"latent={config.spatial_latent_channels}x{latent_size}x{latent_size}; "
        f"params={parameter_count / 1e6:.1f}M"
    )
    print(
        "latent channel mean="
        + ",".join(f"{value:.4f}" for value in channel_mean.flatten().tolist())
    )
    print(
        "latent channel std="
        + ",".join(f"{value:.4f}" for value in channel_std.flatten().tolist())
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        ae_reconstruction = autoencoder.decode(
            validation_maps.to(next(autoencoder.parameters()).dtype)
        )
    save_image(
        torch.cat([validation_images, ae_reconstruction.float()], dim=0).clamp(0, 1),
        output_dir / "ae_reconstruction.png",
        nrow=args.validation_images,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def schedule(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history = []
    progress = tqdm(total=args.steps, desc="spatialized-prefix-hartley-ar")
    global_step = 0
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                maps = encode_images(autoencoder, images)
                raster = latent_maps_to_tokens(
                    maps,
                    channel_mean,
                    channel_std,
                    args.latent_patch,
                    basis="hartley",
                )
                tokens = model.order_tokens(raster)
            model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(tokens)
                loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0 or global_step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=scheduler.get_last_lr()[0],
                )
            record: Dict[str, float] = {
                "step": global_step,
                "loss": float(loss.detach()),
            }
            if args.diagnostic_steps and global_step % args.diagnostic_steps == 0:
                model.eval()
                cpu_state = torch.random.get_rng_state()
                cuda_state = (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                )
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    clean = model(validation_tokens)["loss"]
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                shuffled_history = validation_tokens.roll(1, 0)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    shuffled = model(
                        validation_tokens, history_override=shuffled_history
                    )["loss"]
                record.update(
                    clean=float(clean),
                    shuffled=float(shuffled),
                    gap=float(shuffled - clean),
                )
                print(
                    f"DIAGNOSTIC step={global_step} clean={float(clean):.6f} "
                    f"shuffled={float(shuffled):.6f} gap={float(shuffled-clean):.6f}"
                )
            history.append(record)
            if args.preview_steps and global_step % args.preview_steps == 0:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    ordered = model.generate(
                        2 if args.smoke else 16,
                        args.inference_steps,
                        generator,
                    )
                    raster = model.restore_raster(ordered.float())
                    maps = tokens_to_latent_maps(
                        raster,
                        channel_mean,
                        channel_std,
                        args.latent_patch,
                        latent_size,
                        basis="hartley",
                    )
                    decoded = autoencoder.decode(
                        maps.to(next(autoencoder.parameters()).dtype)
                    )
                save_image(
                    decoded.float().clamp(0, 1),
                    output_dir / f"samples_{global_step}.png",
                    nrow=2 if args.smoke else 4,
                )
            if args.checkpoint_steps and global_step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                        "channel_mean": channel_mean.cpu(),
                        "channel_std": channel_std.cpu(),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                "target_basis": "hartley",
                "trunk_basis": "spatialized_prefix",
                "token_order": "radial",
                "channel_mean": channel_mean.flatten().cpu().tolist(),
                "channel_std": channel_std.flatten().cpu().tolist(),
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
