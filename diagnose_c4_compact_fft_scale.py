"""Compare raw-pixel and C4-latent compact-FFT scale hierarchies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torchvision import datasets, transforms

from frequency import build_orbit_table
from train_spatial_latent_hartley_ar import encode_images, load_spatial_ae


def spectral_summary(maps: torch.Tensor) -> dict:
    batch, channels, height, width = maps.shape
    table = build_orbit_table(height, width, ordering="radial")
    ky, kx = table["ky"], table["kx"]
    is_self = table["is_self_conjugate"]
    radius_bin = table["radius_bin"]
    spectrum = torch.fft.fft2(maps.float(), norm="ortho")
    coeffs = spectrum[:, :, ky, kx].permute(0, 2, 1)
    amplitude = coeffs.abs()

    active = [coeffs[:, is_self].real.reshape(-1)]
    ordinary = coeffs[:, ~is_self]
    active.extend(
        [
            (math.sqrt(2.0) * ordinary.real).reshape(-1),
            (math.sqrt(2.0) * ordinary.imag).reshape(-1),
        ]
    )
    active_abs = torch.cat(active).abs()
    amplitude_flat = amplitude.reshape(-1)
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999], device=maps.device)

    radial = []
    for value in radius_bin.unique(sorted=True):
        selected = radius_bin == value
        radial.append(
            {
                "radius_bin": int(value),
                "orbits": int(selected.sum()),
                "amplitude_rms": float(amplitude[:, selected].square().mean().sqrt()),
                "amplitude_median": float(amplitude[:, selected].median()),
            }
        )
    dc_rms = radial[0]["amplitude_rms"]
    highest_rms = radial[-1]["amplitude_rms"]
    return {
        "shape": [batch, channels, height, width],
        "active_abs_quantiles": torch.quantile(active_abs, quantiles).tolist(),
        "active_abs_max": float(active_abs.max()),
        "amplitude_quantiles": torch.quantile(amplitude_flat, quantiles).tolist(),
        "amplitude_max": float(amplitude_flat.max()),
        "dc_to_highest_bin_rms": dc_rms / highest_rms,
        "max_to_min_radial_rms": max(x["amplitude_rms"] for x in radial)
        / min(x["amplitude_rms"] for x in radial),
        "radial": radial,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    images = torch.stack([dataset[index][0] for index in range(args.count)])
    pixel_mean = images.mean()
    pixel_std = images.std()
    raw_summary = spectral_summary((images - pixel_mean) / pixel_std)

    autoencoder = load_spatial_ae(args.ae_checkpoint, device)
    chunks = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = images[start : start + args.batch_size].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                chunks.append(encode_images(autoencoder, batch).cpu())
    latents = torch.cat(chunks)
    latent_mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    latent_std = latents.std(dim=(0, 2, 3), keepdim=True)
    latent_summary = spectral_summary((latents - latent_mean) / latent_std)

    payload = {
        "count": args.count,
        "pixel_mean": float(pixel_mean),
        "pixel_std": float(pixel_std),
        "latent_channel_mean": latent_mean.flatten().tolist(),
        "latent_channel_std": latent_std.flatten().tolist(),
        "raw_pixels": raw_summary,
        "c4_latents": latent_summary,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    for name, summary in (("raw", raw_summary), ("c4", latent_summary)):
        print(
            f"{name}: dc/highest={summary['dc_to_highest_bin_rms']:.3f} "
            f"max/min radial={summary['max_to_min_radial_rms']:.3f} "
            f"active |x| q50/q90/q99/q999/max="
            + "/".join(f"{x:.4f}" for x in summary["active_abs_quantiles"])
            + f"/{summary['active_abs_max']:.4f}"
        )


if __name__ == "__main__":
    main()
