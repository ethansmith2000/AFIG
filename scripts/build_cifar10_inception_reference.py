#!/usr/bin/env python3
"""Build the shared CIFAR-10 test-set Inception feature reference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_evaluation import InceptionFeatures, StreamingMoments


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument(
        "--output", default="/workspace/AFIG/data/cifar10_test_inception.pt"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if int(payload.get("samples", -1)) != 10_000:
            raise ValueError(f"incompatible existing reference cache: {output}")
        print(json.dumps({"reference": str(output), "samples": 10_000, "cached": True}))
        return

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    extractor = InceptionFeatures(device)
    moments = StreamingMoments(2048)
    kid_batches = []
    seen = 0
    for images, _ in loader:
        features = extractor(images.to(device, non_blocking=True))
        moments.update(features)
        if seen < 5_000:
            kid_batches.append(features[: 5_000 - seen].cpu())
        seen += images.shape[0]
    mean, covariance = moments.compute()
    payload = {
        "version": 1,
        "samples": seen,
        "feature_mean": mean,
        "feature_covariance": covariance,
        "kid_features": torch.cat(kid_batches)[:5_000],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(json.dumps({"reference": str(output), "samples": seen, "cached": False}))


if __name__ == "__main__":
    main()
