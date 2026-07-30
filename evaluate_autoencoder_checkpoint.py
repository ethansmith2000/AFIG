#!/usr/bin/env python3
"""Evaluate trained frequency autoencoders on the official CIFAR-10 test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fit_autoencoder_latent_interface import _load_model
from train_autoencoder import reconstruction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_posterior", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: str,
    loader: DataLoader,
    device: torch.device,
    sample_posterior: bool = False,
) -> dict[str, float]:
    model, codec, _ = _load_model(checkpoint, device)
    totals: dict[str, float] = {}
    example_count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            tokens = codec.encode(images)
            output = model(tokens, sample_posterior=sample_posterior)
            reconstruction = codec.decode(output["reconstruction"])
        batch_metrics = reconstruction_metrics(images, reconstruction)
        mask = codec.component_mask[None].to(output["reconstruction"].dtype)
        token_mse = (
            (output["reconstruction"].float() - tokens.float()).square() * mask
        ).sum() / (mask.sum() * images.shape[0]).clamp_min(1.0)
        batch_metrics["reconstruction/whitened_token_mse"] = token_mse
        batch_size = images.shape[0]
        example_count += batch_size
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * batch_size
    return {
        "test/examples": float(example_count),
        **{key: value / example_count for key, value in totals.items()},
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(
        args.data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    for checkpoint in args.checkpoints:
        metrics = evaluate_checkpoint(
            checkpoint, loader, device, sample_posterior=args.sample_posterior
        )
        filename = (
            "cifar10_test_metrics_sampled.json"
            if args.sample_posterior
            else "cifar10_test_metrics.json"
        )
        output = os.path.join(os.path.dirname(checkpoint), filename)
        Path(output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"checkpoint": checkpoint, **metrics}, indent=2))


if __name__ == "__main__":
    main()
