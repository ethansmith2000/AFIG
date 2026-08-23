#!/usr/bin/env python3
"""Cache CIFAR-10 as reversible 4x4 pixel-patch tokens for the joint RF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from cache_progressive_latents import latent_statistics
from progressive_tokenizer.representations import PIXEL_PATCHES, patchify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--include_horizontal_flip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def make_dataset(root: str, train: bool, *, horizontal_flip: bool = False):
    operations = []
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=1.0))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )
    return torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transforms.Compose(operations),
    )


@torch.no_grad()
def encode(dataset, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    values = []
    labels = []
    for images, batch_labels in loader:
        values.append(patchify(images, args.patch_size).half())
        labels.append(batch_labels)
    return torch.cat(values), torch.cat(labels)


def main() -> None:
    args = parse_args()
    if args.image_size != 32:
        raise ValueError("the CIFAR-10 control currently requires image_size=32")
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
    train, train_labels = encode(train_dataset, args)
    test, test_labels = encode(make_dataset(args.data_root, False), args)
    statistics = latent_statistics(train)
    representation_config = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "in_channels": 3,
        "value_range": "[-1,1]",
        "layout": "raster_nonoverlapping",
    }
    payload = {
        "version": 1,
        "representation_type": PIXEL_PATCHES,
        "representation_config": representation_config,
        "train_views": train_views,
        "train_latents": train,
        "train_labels": train_labels,
        "test_latents": test,
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
        "representation_type": PIXEL_PATCHES,
        "representation_config": representation_config,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_views": train_views,
        "global_mean": float(statistics["global_mean"]),
        "global_std": float(statistics["global_std"]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(printable, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(printable, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
