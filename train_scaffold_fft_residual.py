"""Conditional residual flow with compact-FFT state and local patch computation.

The deterministic C4 autoencoder supplies an oracle coarse scaffold for each
training image.  The stochastic state, Gaussian base, flow target, and Heun
updates remain in an exact orthonormal compact FFT coordinate system.  At every
velocity evaluation the state is transformed back to local pixel patches, where
the transformer is conditioned on aligned scaffold patches; its local velocity
is then transformed back to the compact FFT state.

This is the first Stage-C gate: can a local-compute Fourier residual model
preserve or improve a correct real-image scaffold before generated-scaffold
exposure is introduced?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from causal_transformer import CausalTransformerBlock
from control_pixel_diffusion import (
    build_compact_isometric_codec,
    compact_active_scalar_layout,
    compact_scalar_fft_to_tokens,
    compact_scalar_tokens_to_images,
    flow_interpolate_and_velocity,
    orbit_order_permutation,
    patchify,
    unpatchify,
)
from diffusion_decoder import FinalLayer, TimestepEmbedder
from train_spatial_latent_hartley_ar import encode_images, load_spatial_ae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ae_checkpoint",
        default=(
            "autoencoder_runs/"
            "ae-spatial-perceptual-c4-deterministic-noise01-10k/"
            "checkpoint_10000.pt"
        ),
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--compact_token_dim", type=int, default=48)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--stats_images", type=int, default=4096)
    parser.add_argument("--validation_images", type=int, default=16)
    parser.add_argument("--preview_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_steps", type=int, default=5000)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class ScaffoldResidualDenoiser(nn.Module):
    """Bidirectional local-patch transformer conditioned on a C4 scaffold."""

    def __init__(self, tokens: int, patch_dim: int, args: Any):
        super().__init__()
        width = args.width
        self.tokens = tokens
        self.patch_dim = patch_dim
        self.residual_projection = nn.Linear(patch_dim, width)
        self.scaffold_projection = nn.Linear(patch_dim, width)
        self.position = nn.Parameter(torch.zeros(tokens, width))
        self.time_embed = TimestepEmbedder(width)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=args.num_heads,
                    ff_mult=args.ff_mult,
                    dropout=0.0,
                    conditional_film=True,
                    causal=False,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.final_layer = FinalLayer(width, patch_dim)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def velocity_local(
        self,
        noisy_residual_patches: torch.Tensor,
        scaffold_patches: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.tokens, self.patch_dim)
        if noisy_residual_patches.shape[1:] != expected:
            raise ValueError(
                "noisy residual patches have shape "
                f"{tuple(noisy_residual_patches.shape[1:])}, expected {expected}"
            )
        if scaffold_patches.shape != noisy_residual_patches.shape:
            raise ValueError("scaffold and noisy residual patch shapes must match")
        hidden = (
            self.residual_projection(noisy_residual_patches)
            + self.scaffold_projection(scaffold_patches)
            + self.position.to(noisy_residual_patches.dtype)
        )
        condition = self.time_embed(flow_time * 999.0).unsqueeze(1).expand_as(hidden)
        for layer in self.layers:
            hidden, _ = layer(hidden, condition=condition)
        return self.final_layer(hidden, condition)


def make_compact_layout(image_size: int, device: torch.device):
    """Build the corrected grid-local active-scalar compact FFT layout."""
    codec = build_compact_isometric_codec(image_size, device)
    permutation = orbit_order_permutation(codec, "square_spiral")
    layout_orbit, layout_component = compact_active_scalar_layout(codec, permutation)
    return codec, layout_orbit, layout_component


def images_to_fft_state(
    codec,
    images: torch.Tensor,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    token_dim: int,
) -> torch.Tensor:
    return compact_scalar_fft_to_tokens(
        codec,
        images.float(),
        layout_orbit,
        layout_component,
        token_dim=token_dim,
    )


def fft_state_to_images(
    codec,
    state: torch.Tensor,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
) -> torch.Tensor:
    return compact_scalar_tokens_to_images(
        codec,
        state.float(),
        layout_orbit,
        layout_component,
    )


def dual_domain_velocity(
    model: ScaffoldResidualDenoiser,
    codec,
    noisy_fft_state: torch.Tensor,
    scaffold_patches: torch.Tensor,
    flow_time: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
) -> torch.Tensor:
    """Evaluate a local-patch velocity while retaining an FFT solver state."""
    noisy_images = fft_state_to_images(
        codec, noisy_fft_state, layout_orbit, layout_component
    )
    local_velocity = model.velocity_local(
        patchify(noisy_images, patch), scaffold_patches, flow_time
    )
    velocity_images = unpatchify(local_velocity, patch, image_size)
    return images_to_fft_state(
        codec,
        velocity_images,
        layout_orbit,
        layout_component,
        token_dim,
    )


def dual_domain_flow_loss(
    model: ScaffoldResidualDenoiser,
    codec,
    target_fft_state: torch.Tensor,
    scaffold_patches: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
) -> torch.Tensor:
    batch = target_fft_state.shape[0]
    flow_time = torch.rand(batch, device=target_fft_state.device)
    noise = torch.randn_like(target_fft_state)
    noisy, target_velocity = flow_interpolate_and_velocity(
        target_fft_state, noise, flow_time, "linear"
    )
    predicted_velocity = dual_domain_velocity(
        model,
        codec,
        noisy,
        scaffold_patches,
        flow_time,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=patch,
        image_size=image_size,
        token_dim=token_dim,
    )
    return (predicted_velocity - target_velocity).square().mean()


@torch.no_grad()
def sample_residual_fft(
    model: ScaffoldResidualDenoiser,
    codec,
    scaffold_patches: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Heun solve from an isotropic Gaussian compact-FFT base."""
    count = scaffold_patches.shape[0]
    state = torch.randn(
        count,
        3 * image_size * image_size // token_dim,
        token_dim,
        device=scaffold_patches.device,
        dtype=torch.float32,
        generator=generator,
    )
    dt = 1.0 / steps
    for index in range(steps):
        time = torch.full(
            (count,), index / steps, device=state.device, dtype=torch.float32
        )
        velocity = dual_domain_velocity(
            model,
            codec,
            state,
            scaffold_patches,
            time,
            layout_orbit=layout_orbit,
            layout_component=layout_component,
            patch=patch,
            image_size=image_size,
            token_dim=token_dim,
        )
        proposal = state + dt * velocity
        if index + 1 < steps:
            next_time = torch.full(
                (count,),
                (index + 1) / steps,
                device=state.device,
                dtype=torch.float32,
            )
            next_velocity = dual_domain_velocity(
                model,
                codec,
                proposal,
                scaffold_patches,
                next_time,
                layout_orbit=layout_orbit,
                layout_component=layout_component,
                patch=patch,
                image_size=image_size,
                token_dim=token_dim,
            )
            state = state + 0.5 * dt * (velocity + next_velocity)
        else:
            state = proposal
    return state


@torch.no_grad()
def deterministic_scaffold(autoencoder, images: torch.Tensor) -> torch.Tensor:
    maps = encode_images(autoencoder, images, sample_posterior=False)
    parameter_dtype = next(autoencoder.parameters()).dtype
    return autoencoder.decode(maps.to(parameter_dtype)).float()


@torch.no_grad()
def fit_scaffold_residual_stats(
    autoencoder,
    dataset,
    count: int,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Fit scalar train-population moments without frequency whitening."""
    count = min(count, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=min(batch_size, count),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    sums = {"scaffold": 0.0, "scaffold_sq": 0.0, "residual": 0.0, "residual_sq": 0.0}
    scalar_count = 0
    for images, _ in tqdm(loader, desc="fit-scaffold-stats", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            scaffold = deterministic_scaffold(autoencoder, images)
        residual = images.float() - scaffold
        sums["scaffold"] += float(scaffold.double().sum())
        sums["scaffold_sq"] += float(scaffold.double().square().sum())
        sums["residual"] += float(residual.double().sum())
        sums["residual_sq"] += float(residual.double().square().sum())
        scalar_count += scaffold.numel()
    scaffold_mean = sums["scaffold"] / scalar_count
    residual_mean = sums["residual"] / scalar_count
    scaffold_var = sums["scaffold_sq"] / scalar_count - scaffold_mean**2
    residual_var = sums["residual_sq"] / scalar_count - residual_mean**2
    return {
        "images": count,
        "scalar_count": scalar_count,
        "scaffold_mean": scaffold_mean,
        "scaffold_std": max(scaffold_var, 0.0) ** 0.5,
        "residual_mean": residual_mean,
        "residual_std": max(residual_var, 0.0) ** 0.5,
    }


def psnr(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = (reference - prediction).square().mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def model_args_from_checkpoint(arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        width=arguments["width"],
        num_layers=arguments["num_layers"],
        num_heads=arguments["num_heads"],
        ff_mult=arguments["ff_mult"],
    )


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
        args.warmup = 1
        args.stats_images = 8
        args.validation_images = 2
        args.preview_steps = 1
        args.checkpoint_steps = 1
        args.inference_steps = 2

    if args.image_size != 32:
        raise ValueError("the C4 oracle checkpoint currently requires 32x32 images")
    if args.image_size % args.patch:
        raise ValueError("patch must divide image_size")
    total_values = 3 * args.image_size * args.image_size
    if total_values % args.compact_token_dim:
        raise ValueError("compact_token_dim must divide the image scalar count")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    autoencoder = load_spatial_ae(args.ae_checkpoint, device)
    if autoencoder.config.variational:
        raise ValueError("the oracle scaffold gate requires a deterministic AE")

    plain_train = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    stats = fit_scaffold_residual_stats(
        autoencoder,
        plain_train,
        args.stats_images,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if stats["scaffold_std"] < 1e-6 or stats["residual_std"] < 1e-6:
        raise RuntimeError(f"degenerate fitted statistics: {stats}")
    print("normalization=" + json.dumps(stats, sort_keys=True))

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

    codec, layout_orbit, layout_component = make_compact_layout(
        args.image_size, device
    )
    local_tokens = (args.image_size // args.patch) ** 2
    patch_dim = 3 * args.patch**2
    fft_tokens = total_values // args.compact_token_dim
    if fft_tokens != local_tokens or args.compact_token_dim != patch_dim:
        raise ValueError(
            "this matched gate requires the FFT and local views to both be "
            "64x48; use image_size=32, patch=4, compact_token_dim=48"
        )
    model = ScaffoldResidualDenoiser(local_tokens, patch_dim, args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"oracle scaffold FFT residual: state={fft_tokens}x{args.compact_token_dim}; "
        f"compute={local_tokens}x{patch_dim}; params={parameter_count / 1e6:.2f}M"
    )

    scaffold_mean = stats["scaffold_mean"]
    scaffold_std = stats["scaffold_std"]
    residual_mean = stats["residual_mean"]
    residual_std = stats["residual_std"]

    validation_scaffold_patches = patchify(
        (validation_scaffold - scaffold_mean) / scaffold_std, args.patch
    )
    save_image(
        torch.cat([validation_images, validation_scaffold], dim=0).clamp(0, 1),
        output_dir / "scaffold_baseline.png",
        nrow=args.validation_images,
    )
    print(
        f"validation scaffold PSNR={psnr(validation_images, validation_scaffold):.3f}dB"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 1.0 - 0.75 * min(progress, 1.0)

    history: list[dict[str, float | int]] = []
    global_step = 0
    progress = tqdm(total=args.steps, desc="scaffold-fft-residual")
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_at(global_step)
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
                loss = dual_domain_flow_loss(
                    model,
                    codec,
                    target_fft,
                    scaffold_patches,
                    layout_orbit=layout_orbit,
                    layout_component=layout_component,
                    patch=args.patch,
                    image_size=args.image_size,
                    token_dim=args.compact_token_dim,
                )
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
                        "grad_norm": float(grad_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            should_preview = (
                args.preview_steps > 0 and global_step % args.preview_steps == 0
            )
            if should_preview:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    sampled_fft = sample_residual_fft(
                        model,
                        codec,
                        validation_scaffold_patches,
                        layout_orbit=layout_orbit,
                        layout_component=layout_component,
                        patch=args.patch,
                        image_size=args.image_size,
                        token_dim=args.compact_token_dim,
                        steps=args.inference_steps,
                        generator=generator,
                    )
                sampled_normalized = fft_state_to_images(
                    codec, sampled_fft, layout_orbit, layout_component
                )
                completion = (
                    validation_scaffold
                    + sampled_normalized * residual_std
                    + residual_mean
                )
                completion_psnr = psnr(
                    validation_images, completion.clamp(0, 1)
                )
                save_image(
                    torch.cat(
                        [validation_images, validation_scaffold, completion], dim=0
                    ).clamp(0, 1),
                    output_dir / f"preview_{global_step:07d}.png",
                    nrow=args.validation_images,
                )
                print(
                    f"PREVIEW step={global_step} completion_psnr="
                    f"{completion_psnr:.3f}dB"
                )

            should_checkpoint = (
                args.checkpoint_steps > 0
                and global_step % args.checkpoint_steps == 0
            )
            if should_checkpoint:
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                        "normalization": stats,
                        "compact_layout_orbit": layout_orbit.cpu(),
                        "compact_layout_component": layout_component.cpu(),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    with (output_dir / "history.json").open("w") as handle:
        json.dump(
            {
                "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                "normalization": stats,
                "parameter_count": parameter_count,
                "state": "corrected grid-local compact isometric FFT residual",
                "computation": "local 4x4 residual patches plus aligned C4 scaffold",
                "history": history,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print("done")


if __name__ == "__main__":
    main()
