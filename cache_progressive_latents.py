#!/usr/bin/env python3
"""Encode fixed CIFAR splits with a frozen progressive tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include_horizontal_flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a deterministic horizontal flip of every training image.",
    )
    return parser.parse_args()


def make_dataset(root: str, train: bool, *, horizontal_flip: bool = False):
    operations = []
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=1.0))
    operations.extend(
        [transforms.ToTensor(), transforms.Lambda(lambda image: image.mul(2.0).sub(1.0))]
    )
    transform = transforms.Compose(operations)
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=transform
    )


@torch.no_grad()
def encode_split(model, dataset, args, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    latent_batches = []
    label_batches = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            latents = model.encode(images)
        latent_batches.append(latents.float().cpu())
        label_batches.append(labels.cpu())
    return torch.cat(latent_batches), torch.cat(label_batches)


def latent_statistics(latents: torch.Tensor) -> dict:
    values = latents.float()
    flat = values.flatten()
    coordinate_values = values.reshape(-1, values.shape[-1])
    coordinate_mean = coordinate_values.mean(dim=0)
    coordinate_std = coordinate_values.std(dim=0, unbiased=False)
    slot_mean = values.mean(dim=(0, 2))
    slot_std = values.flatten(0, 0).std(dim=(0, 2), unbiased=False)
    centered = coordinate_values - coordinate_mean
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    return {
        "global_mean": flat.mean(),
        "global_std": flat.std(unbiased=False),
        "global_min": flat.min(),
        "global_max": flat.max(),
        "coordinate_mean": coordinate_mean,
        "coordinate_std": coordinate_std,
        "slot_mean": slot_mean,
        "slot_std": slot_std,
        "coordinate_covariance_effective_rank": effective_rank.float(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, checkpoint = load_tokenizer_checkpoint(args.tokenizer_checkpoint)
    model = model.to(device).eval()
    train_dataset = make_dataset(args.data_root, True)
    train_views = ["original"]
    if args.include_horizontal_flip:
        train_dataset = ConcatDataset(
            [
                train_dataset,
                make_dataset(args.data_root, True, horizontal_flip=True),
            ]
        )
        train_views.append("horizontal_flip")
    train, train_labels = encode_split(model, train_dataset, args, device)
    test, test_labels = encode_split(
        model, make_dataset(args.data_root, False), args, device
    )
    statistics = latent_statistics(train)
    payload = {
        "version": 1,
        "tokenizer_checkpoint": str(Path(args.tokenizer_checkpoint).resolve()),
        "tokenizer_step": int(checkpoint.get("step", -1)),
        "model_config": checkpoint["model_config"],
        "train_views": train_views,
        "train_latents": train.half(),
        "train_labels": train_labels,
        "test_latents": test.half(),
        "test_labels": test_labels,
        "statistics": statistics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    printable = {
        "output": str(output),
        "tokenizer_step": payload["tokenizer_step"],
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_views": train_views,
        "global_mean": float(statistics["global_mean"]),
        "global_std": float(statistics["global_std"]),
        "global_min": float(statistics["global_min"]),
        "global_max": float(statistics["global_max"]),
        "coordinate_std_min": float(statistics["coordinate_std"].min()),
        "coordinate_std_median": float(statistics["coordinate_std"].median()),
        "coordinate_std_max": float(statistics["coordinate_std"].max()),
        "slot_std_min": float(statistics["slot_std"].min()),
        "slot_std_max": float(statistics["slot_std"].max()),
        "coordinate_effective_rank": float(
            statistics["coordinate_covariance_effective_rank"]
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(printable, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(printable, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
