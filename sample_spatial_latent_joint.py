"""Decode a fresh-seed sample grid from a trained joint spatial-latent flow."""

from __future__ import annotations

import argparse
from argparse import Namespace

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import PatchDiffusion
from train_spatial_latent_hartley_ar import (
    load_spatial_ae,
    tokens_to_latent_maps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=54321)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = Namespace(**payload["args"])
    autoencoder = load_spatial_ae(payload["ae_checkpoint"], device)
    config = autoencoder.config
    latent_size = config.spatial_resolution // config.spatial_downsample
    tokens = (latent_size // train_args.latent_patch) ** 2
    token_dim = config.spatial_latent_channels * train_args.latent_patch**2
    model = PatchDiffusion(tokens, token_dim, train_args).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    channel_mean = payload["channel_mean"].to(device)
    channel_std = payload["channel_std"].to(device)
    latent_basis = getattr(train_args, "latent_basis", "hartley")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        sampled_tokens = model.sample(
            args.count, train_args.inference_steps, device
        )
        maps = tokens_to_latent_maps(
            sampled_tokens.float(),
            channel_mean,
            channel_std,
            train_args.latent_patch,
            latent_size,
            basis=latent_basis,
        )
        decoded = autoencoder.decode(
            maps.to(next(autoencoder.parameters()).dtype)
        )
    save_image(decoded.float().clamp(0, 1), args.output, nrow=4)


if __name__ == "__main__":
    main()
