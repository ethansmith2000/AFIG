"""Decode a fresh-seed grid from a spatialized-prefix Hartley AR checkpoint."""

from __future__ import annotations

import argparse
from argparse import Namespace

import torch
from torchvision.utils import save_image

from spatialized_prefix_ar import SpatializedPrefixHartleyAR
from train_spatial_latent_hartley_ar import load_spatial_ae, tokens_to_latent_maps


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
    model = SpatializedPrefixHartleyAR(
        width=train_args.width,
        num_layers=train_args.layers,
        num_heads=train_args.heads,
        ff_mult=train_args.ff_mult,
        diff_width=train_args.diff_width,
        diff_depth=train_args.diff_depth,
        inference_steps=train_args.inference_steps,
        latent_size=latent_size,
        patch=train_args.latent_patch,
        channels=config.spatial_latent_channels,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    channel_mean = payload["channel_mean"].to(device)
    channel_std = payload["channel_std"].to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        ordered = model.generate(
            args.count,
            train_args.inference_steps,
            generator,
        )
        raster = model.restore_raster(ordered.float())
        maps = tokens_to_latent_maps(
            raster,
            channel_mean,
            channel_std,
            train_args.latent_patch,
            latent_size,
            basis="hartley",
        )
        decoded = autoencoder.decode(
            maps.to(next(autoencoder.parameters()).dtype)
        )
    save_image(decoded.float().clamp(0, 1), args.output, nrow=4)


if __name__ == "__main__":
    main()
