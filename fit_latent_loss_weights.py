#!/usr/bin/env python3
"""Fit raw-variance and decoder-sensitivity latent loss weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from latent_autoencoder_interface import FrozenLatentAutoencoder
from train_autoencoder import make_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--probes_per_batch", type=int, default=4)
    return parser.parse_args()


def _normalize(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    floor = weights.median().clamp_min(1e-12) * 1e-6
    weights = weights.clamp_min(floor)
    return weights / weights.mean().clamp_min(1e-12)


def _summary(weights: torch.Tensor) -> dict[str, float]:
    flat = weights.flatten().float()
    quantiles = torch.quantile(
        flat, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    )
    return {
        "min": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "max": float(quantiles[4]),
        "effective_fraction": float(
            flat.sum().square().div(flat.square().sum() * flat.numel()).item()
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = FrozenLatentAutoencoder(
        args.ae_checkpoint,
        args.latent_interface,
        sample_posterior=False,
    ).to(device)
    if adapter.sample_posterior:
        raise ValueError("Loss weights must be fitted from posterior means")

    dataset = make_dataset(
        SimpleNamespace(
            dataset=args.dataset,
            data_root=args.data_root,
            resolution=32,
            smoke=False,
            seed=0,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    sensitivity = torch.zeros(
        adapter.latent_mean.shape, dtype=torch.float64, device="cpu"
    )
    examples = 0
    generator = torch.Generator(device=device).manual_seed(1729)
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            latents = adapter.encode_images(images)
        latents = latents.detach().requires_grad_(True)
        decoded = adapter.decode_latents_with_grad(latents)
        pixel_count = decoded[0].numel()
        for probe_index in range(args.probes_per_batch):
            signs = torch.randint(
                0,
                2,
                decoded.shape,
                device=device,
                generator=generator,
                dtype=torch.int8,
            ).to(decoded.dtype)
            signs = (2.0 * signs - 1.0) / math.sqrt(pixel_count)
            gradient = torch.autograd.grad(
                (decoded * signs).sum(),
                latents,
                retain_graph=probe_index + 1 < args.probes_per_batch,
            )[0]
            sensitivity += gradient.detach().double().square().sum(dim=0).cpu()
        examples += images.shape[0] * args.probes_per_batch

    if examples == 0:
        raise RuntimeError("No examples were processed")
    raw_variance = _normalize(adapter.latent_std.detach().cpu().square())
    decoder_sensitivity = _normalize((sensitivity / examples).float())
    result = {
        "version": 1,
        "ae_checkpoint": adapter.checkpoint_path,
        "latent_interface": adapter.latent_interface_path,
        "sample_posterior": False,
        "num_examples_times_probes": examples,
        "raw_variance": raw_variance,
        "decoder_sensitivity": decoder_sensitivity,
        "summary": {
            "raw_variance": _summary(raw_variance),
            "decoder_sensitivity": _summary(decoder_sensitivity),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
