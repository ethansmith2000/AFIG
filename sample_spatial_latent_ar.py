"""Decode a fresh-seed sample grid from a trained AR spatial-latent model."""

from __future__ import annotations

import argparse
import math
from argparse import Namespace

import torch
from torchvision.utils import save_image

from train_hartley_ar import HartleyTileAR
from train_spatial_latent_hartley_ar import (
    load_spatial_ae,
    tokens_to_latent_maps,
    ungroup_radial_hartley_tiles,
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
    token_order = getattr(train_args, "token_order", "auto")
    latent_basis = getattr(train_args, "latent_basis", "hartley")
    if token_order == "auto":
        token_order = "radial" if latent_basis == "hartley" else "raster"
    tiles_per_token = getattr(train_args, "tiles_per_token", 1)
    physical_grid = latent_size // train_args.latent_patch
    block_dct_token_dim = getattr(train_args, "block_dct_token_dim", 16)
    compact_fft_token_dim = getattr(train_args, "compact_fft_token_dim", 16)
    if latent_basis in ("block_dct", "compact_fft"):
        fixed_token_dim = (
            block_dct_token_dim
            if latent_basis == "block_dct"
            else compact_fft_token_dim
        )
        group_count = (
            config.spatial_latent_channels * latent_size**2 // fixed_token_dim
        )
        model_token_dim = fixed_token_dim
    else:
        group_count = physical_grid**2 // tiles_per_token
        model_token_dim = (
            config.spatial_latent_channels
            * train_args.latent_patch**2
            * tiles_per_token
        )
    model_grid = math.isqrt(group_count)
    if model_grid**2 != group_count:
        raise ValueError("grouped AR token count must be a perfect square")
    model_token_order = "raster" if tiles_per_token > 1 else token_order
    model = HartleyTileAR(
        width=train_args.width,
        num_layers=train_args.layers,
        num_heads=train_args.heads,
        ff_mult=train_args.ff_mult,
        diff_width=train_args.diff_width,
        diff_depth=train_args.diff_depth,
        inference_steps=train_args.inference_steps,
        grid=model_grid,
        token_dim=model_token_dim,
        token_order=model_token_order,
        rope_mode=getattr(train_args, "rope_mode", "frequency_2d"),
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
        if tiles_per_token > 1:
            raster = ungroup_radial_hartley_tiles(
                ordered.float(), physical_grid, tiles_per_token
            )
        else:
            raster = model.restore_raster(ordered.float())
        maps = tokens_to_latent_maps(
            raster,
            channel_mean,
            channel_std,
            train_args.latent_patch,
            latent_size,
            basis=latent_basis,
            dct_support=getattr(train_args, "dct_support", 2),
            block_dct_token_dim=block_dct_token_dim,
            compact_fft_token_dim=compact_fft_token_dim,
        )
        decoded = autoencoder.decode(
            maps.to(next(autoencoder.parameters()).dtype)
        )
    save_image(decoded.float().clamp(0, 1), args.output, nrow=4)


if __name__ == "__main__":
    main()
