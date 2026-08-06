"""Joint rectified flow over selectable token bases of a spatial AE map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from control_pixel_diffusion import PatchDiffusion
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--latent_patch", type=int, default=2)
    parser.add_argument("--stats_images", type=int, default=4096)
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument(
        "--latent_basis",
        choices=[
            "hartley",
            "spatial",
            "patch_dct",
            "patch_dct_freq_major",
            "full_dct_tiles",
        ],
        default="hartley",
        help=(
            "Global Hartley/DCT tiles, local raster patches, or an orthonormal "
            "DCT inside each local patch of the same normalized map. The "
            "frequency-major option is an exact regrouping of the local-DCT "
            "scalars matching the passing 16-step AR interface."
        ),
    )
    parser.add_argument(
        "--flow_path",
        choices=["linear", "trig_vp"],
        default="linear",
        help=(
            "Base-to-data stochastic interpolant. trig_vp preserves unit "
            "variance when both endpoints are isotropic Gaussian."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.width = 64
        args.num_layers = 1
        args.num_heads = 4
        args.ff_mult = 2
        args.inference_steps = 2
        args.preview_steps = 1
        args.checkpoint_steps = 0
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
        autoencoder,
        plain,
        args.stats_images,
        device,
        sample_posterior=args.sample_posterior,
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
    tokens = (latent_size // args.latent_patch) ** 2
    token_dim = config.spatial_latent_channels * args.latent_patch**2
    model = PatchDiffusion(tokens, token_dim, args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"joint latent {args.latent_basis}: {tokens} x {token_dim}; "
        f"latent={config.spatial_latent_channels}x{latent_size}x{latent_size}; "
        f"params={parameter_count / 1e6:.1f}M; flow_path={args.flow_path}; "
        f"sample_posterior={args.sample_posterior}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_multiplier(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 1.0 - 0.75 * min(progress, 1.0)

    history = []
    progress = tqdm(total=args.steps, desc="latent-hartley-joint")
    step = 0
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_multiplier(step)
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                maps = encode_images(
                    autoencoder,
                    images,
                    sample_posterior=args.sample_posterior,
                )
                x = latent_maps_to_tokens(
                    maps,
                    channel_mean,
                    channel_std,
                    args.latent_patch,
                    basis=args.latent_basis,
                )
            model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = model.loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            progress.update(1)
            if step % 25 == 0 or step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=optimizer.param_groups[0]["lr"],
                )
            if step % 250 == 0:
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                print(f"DIAGNOSTIC step={step} loss={float(loss):.6f}")
            if args.preview_steps and step % args.preview_steps == 0:
                model.eval()
                cpu_state = torch.random.get_rng_state()
                cuda_state = (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                )
                torch.manual_seed(12345)
                if device.type == "cuda":
                    torch.cuda.manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    samples = model.sample(
                        2 if args.smoke else 16,
                        args.inference_steps,
                        device,
                    )
                    maps = tokens_to_latent_maps(
                        samples.float(),
                        channel_mean,
                        channel_std,
                        args.latent_patch,
                        latent_size,
                        basis=args.latent_basis,
                    )
                    decoded = autoencoder.decode(
                        maps.to(next(autoencoder.parameters()).dtype)
                    )
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                save_image(
                    decoded.float().clamp(0, 1),
                    output_dir / f"samples_{step}.png",
                    nrow=2 if args.smoke else 4,
                )
            if args.checkpoint_steps and step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                        "channel_mean": channel_mean.cpu(),
                        "channel_std": channel_std.cpu(),
                    },
                    output_dir / f"checkpoint_{step}.pt",
                )
    progress.close()
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                "sample_posterior": bool(args.sample_posterior),
                "flow_path": args.flow_path,
                "latent_basis": args.latent_basis,
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
