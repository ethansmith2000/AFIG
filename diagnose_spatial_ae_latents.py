"""Distribution diagnostics for a spatial AE and its Hartley-token bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from train_spatial_latent_hartley_ar import (
    encode_images,
    latent_maps_to_tokens,
    load_spatial_ae,
)


def quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor([0.5, 0.9, 0.99, 0.999], device=values.device)
    result = torch.quantile(values.float(), levels)
    return {
        "p50": float(result[0]),
        "p90": float(result[1]),
        "p99": float(result[2]),
        "p99_9": float(result[3]),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patch", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_spatial_ae(args.checkpoint, device)
    dataset = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    chunks = []
    count = min(args.images, len(dataset))
    with torch.no_grad():
        for start in range(0, count, args.batch_size):
            images = torch.stack(
                [dataset[index][0] for index in range(start, min(count, start + args.batch_size))]
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                chunks.append(encode_images(autoencoder, images).cpu())
    maps = torch.cat(chunks).float()
    channel_mean = maps.mean(dim=(0, 2, 3), keepdim=True)
    channel_std = maps.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    normalized = (maps - channel_mean) / channel_std
    tokens = latent_maps_to_tokens(normalized, torch.zeros_like(channel_mean), torch.ones_like(channel_std), args.patch)
    flat = normalized.flatten()
    flat_centered = flat - flat.mean()
    variance = flat_centered.square().mean().clamp_min(1e-8)
    coordinate_mean = normalized.mean(dim=0)
    coordinate_std = normalized.std(dim=0)
    tile_rms = tokens.square().mean(dim=(0, 2)).sqrt()
    dc = tokens[:, 0].flatten()
    non_dc = tokens[:, 1:].flatten()
    report = {
        "images": count,
        "latent_shape": list(maps.shape[1:]),
        "latent_scalars": int(maps[0].numel()),
        "channel_mean": channel_mean.flatten().tolist(),
        "channel_std": channel_std.flatten().tolist(),
        "normalized_abs": quantiles(flat.abs()),
        "normalized_skew": float(flat_centered.pow(3).mean() / variance.pow(1.5)),
        "normalized_excess_kurtosis": float(flat_centered.pow(4).mean() / variance.square() - 3.0),
        "coordinate_mean_abs": quantiles(coordinate_mean.abs().flatten()),
        "coordinate_std": quantiles(coordinate_std.flatten()),
        "hartley_abs": quantiles(tokens.abs().flatten()),
        "hartley_dc_abs": quantiles(dc.abs()),
        "hartley_non_dc_abs": quantiles(non_dc.abs()),
        "hartley_tile_rms": tile_rms.tolist(),
        "hartley_tile_rms_ratio": float(tile_rms.max() / tile_rms.min().clamp_min(1e-8)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
