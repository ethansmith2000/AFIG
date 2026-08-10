#!/usr/bin/env python3
"""Sample a saved joint latent-diffusion checkpoint with an explicit fresh seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from latent_autoencoder_interface import FrozenLatentAutoencoder
from train_joint_latent_diffusion import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights",
        choices=["auto", "raw", "ema"],
        default="raw",
        help="raw is the safe default; auto uses EMA when present and raw weights "
        "for legacy checkpoints",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("num_images must be positive")
    device = torch.device(args.device)
    adapter = FrozenLatentAutoencoder(
        args.ae_checkpoint,
        args.latent_interface,
        sample_posterior=False,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    has_ema = payload.get("ema") is not None
    if args.weights == "ema" and not has_ema:
        raise ValueError("--weights ema requested but checkpoint has no EMA")
    use_ema = args.weights == "ema" or (args.weights == "auto" and has_ema)
    del payload
    model, step = load_checkpoint(
        args.checkpoint, adapter, use_ema_weights=use_ema
    )
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = model.generate_latents(
        args.num_images,
        adapter.position_features,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )
    images = adapter.decode_latents(latents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        images,
        output,
        nrow=max(int(args.num_images**0.5), 1),
    )
    print(
        f"wrote={output} checkpoint_step={step} seed={args.seed} "
        f"weights={'ema' if use_ema else 'raw'} "
        f"images={args.num_images} latent_rms="
        f"{latents.float().square().mean().sqrt().item():.6f}"
    )


if __name__ == "__main__":
    main()
