"""Does the AR model's conditional advantage survive on held-out images?

The AR training diagnostics report conditional x0 MSE ~0.128 against a null-context
~0.298, which has been read throughout this project as "the AR model learns real
conditional structure". Those numbers were logged on training batches.

A linear readout from the frozen trunk's hidden state to its target scores
train R^2 = 0.66 but test R^2 = 0.005. With 434k samples and 769 predictors,
in-sample optimism should be ~0.002, so that gap cannot be regression overfitting
-- the trunk's hidden-to-target relationship differs between train and test
images. This script checks the same thing with the model's own nonlinear head.

Paired protocol: identical timesteps and identical noise for train and test, and
for conditional and null within each split, so differences are not sampling noise.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms

from latent_autoencoder_interface import FrozenLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def split_losses(
    interface: FrozenLatentAutoencoder,
    model,
    data_root: str,
    train: bool,
    count: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, float]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=train, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    metadata = interface.position_features
    conditional_total = 0.0
    null_total = 0.0
    seen = 0
    for images, _ in loader:
        latents = interface.encode_images(images.to(device))
        batch = latents.shape[0]
        for repeat in range(repeats):
            # Same seed for both splits and both context settings.
            torch.manual_seed(1234 + repeat)
            keep = torch.zeros(batch, latents.shape[1], dtype=torch.bool, device=device)
            torch.manual_seed(1234 + repeat)
            conditional = model(latents, metadata, context_dropout_mask=keep)
            torch.manual_seed(1234 + repeat)
            drop = torch.ones_like(keep)
            null = model(latents, metadata, context_dropout_mask=drop)
            key = (
                "normalized_target_mse"
                if "normalized_target_mse" in conditional
                else "loss"
            )
            conditional_total += float(conditional[key]) * batch
            null_total += float(null[key]) * batch
        seen += batch * repeats
    return {
        "conditional_mse": conditional_total / seen,
        "null_mse": null_total / seen,
        "conditional_null_gap": (null_total - conditional_total) / seen,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)
    from train_latent_continuous import load_latent_checkpoint

    model, step = load_latent_checkpoint(args.ar_checkpoint, interface)
    model = model.to(device).eval()
    print(f"AR checkpoint step {step}")

    train = split_losses(
        interface, model, args.data_root, True, args.num_images,
        args.batch_size, args.repeats, device,
    )
    test = split_losses(
        interface, model, args.data_root, False, args.num_images,
        args.batch_size, args.repeats, device,
    )

    report = {"ar_step": step, "train": train, "test": test}
    with open(os.path.join(args.output_dir, "ar_generalization.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\n{'split':<8} {'conditional':>12} {'null':>10} {'gap':>10}")
    for name, values in (("train", train), ("test", test)):
        print(
            f"{name:<8} {values['conditional_mse']:>12.4f} {values['null_mse']:>10.4f}"
            f" {values['conditional_null_gap']:>10.4f}"
        )
    retained = (
        test["conditional_null_gap"] / train["conditional_null_gap"]
        if train["conditional_null_gap"]
        else float("nan")
    )
    print(f"\nfraction of the conditional advantage retained on held-out data: {retained:.3f}")
    if retained < 0.5:
        print("-> the AR conditional advantage is largely a training-set effect")
    else:
        print("-> the AR conditional advantage generalizes")


if __name__ == "__main__":
    main()
