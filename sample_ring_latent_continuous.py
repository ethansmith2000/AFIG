#!/usr/bin/env python3
"""Sample a trained joint-within-ring latent generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from latent_autoencoder_interface import FrozenLatentAutoencoder
from train_ring_latent_continuous import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_norm_match", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=54321)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = FrozenLatentAutoencoder(
        args.ae_checkpoint, args.latent_interface
    ).to(device)
    model, _ = load_checkpoint(args.checkpoint, adapter)
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = model.generate_latents(
        args.num_images,
        cfg_scale=args.cfg_scale,
        cfg_norm_match=args.cfg_norm_match,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        generator=generator,
    )
    images = adapter.decode_latents(latents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        images.float().cpu(),
        output,
        nrow=max(1, int(args.num_images**0.5)),
    )
    print(output)


if __name__ == "__main__":
    main()
