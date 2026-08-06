"""Decode the generator's standardized Gaussian base through a spatial AE."""

from __future__ import annotations

import argparse

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_spatial_latent_hartley_ar import fit_channel_stats, load_spatial_ae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--stats_images", type=int, default=4096)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_spatial_ae(args.checkpoint, device)
    dataset = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    channel_mean, channel_std = fit_channel_stats(
        model, dataset, args.stats_images, device, sample_posterior=False
    )
    config = model.config
    latent_size = config.spatial_resolution // config.spatial_downsample
    generator = torch.Generator(device=device).manual_seed(args.seed)
    normalized = torch.randn(
        args.count,
        config.spatial_latent_channels,
        latent_size,
        latent_size,
        device=device,
        generator=generator,
    )
    latents = normalized * channel_std + channel_mean
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        decoded = model.decode(latents.to(next(model.parameters()).dtype))
    save_image(decoded.float().clamp(0, 1), args.output, nrow=4)


if __name__ == "__main__":
    main()
