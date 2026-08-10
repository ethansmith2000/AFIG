#!/usr/bin/env python3
"""Evaluate a progressive-token joint-flow checkpoint on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer import (
    AutoregressiveFlowConfig,
    AutoregressiveRectifiedFlow,
    JointFlowConfig,
    JointRectifiedFlow,
)
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument(
        "--reference_cache", default="/workspace/AFIG/data/cifar10_test_inception.pt"
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def reference_features(args, extractor: InceptionFeatures) -> dict:
    path = Path(args.reference_cache)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    moments = StreamingMoments(2048)
    kid_batches = []
    seen = 0
    for images, _ in loader:
        features = extractor(images.to(extractor.device, non_blocking=True))
        moments.update(features)
        if seen < 5000:
            kid_batches.append(features[: 5000 - seen].cpu())
        seen += images.shape[0]
    mean, covariance = moments.compute()
    payload = {
        "version": 1,
        "samples": seen,
        "feature_mean": mean,
        "feature_covariance": covariance,
        "kid_features": torch.cat(kid_batches)[:5000],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num_samples must be at least two")
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = payload.get("model_type")
    if model_type == "progressive_joint_rectified_flow":
        model = JointRectifiedFlow(JointFlowConfig(**payload["model_config"]))
        sample_method = "sample"
    elif model_type == "progressive_autoregressive_rectified_flow":
        model = AutoregressiveRectifiedFlow(
            AutoregressiveFlowConfig(**payload["model_config"])
        )
        sample_method = "generate"
    else:
        raise ValueError("not a supported progressive-token flow checkpoint")
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    mean = payload["normalization"]["mean"].float().to(device)
    scale = payload["normalization"]["scale"].float().to(device)
    tokenizer, tokenizer_payload = load_tokenizer_checkpoint(
        payload["tokenizer_checkpoint"]
    )
    if int(tokenizer_payload.get("step", -1)) != int(payload["tokenizer_step"]):
        raise ValueError("prior and tokenizer checkpoint steps do not match")
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    extractor = InceptionFeatures(device)
    reference = reference_features(args, extractor)

    generated_moments = StreamingMoments(2048)
    generated_features = []
    preview = []
    generated = 0
    latent_sum = 0.0
    latent_square_sum = 0.0
    latent_count = 0
    image_minimum = float("inf")
    image_maximum = float("-inf")
    clipped = 0
    image_count = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    while generated < args.num_samples:
        current = min(args.batch_size, args.num_samples - generated)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if sample_method == "sample":
                standardized = model.sample(
                    current,
                    steps=args.sample_steps,
                    solver="heun",
                    generator=generator,
                )
            else:
                standardized = model.generate(
                    current,
                    steps=args.sample_steps,
                    generator=generator,
                )
            raw_latents = standardized.float() * scale + mean
            decoded = tokenizer.decode(raw_latents).float()
        images = decoded.add(1.0).div(2.0)
        features = extractor(images)
        generated_moments.update(features)
        generated_features.append(features.cpu())
        if len(preview) < 64:
            preview.append(images[: 64 - sum(item.shape[0] for item in preview)].cpu())
        latent_sum += float(standardized.double().sum())
        latent_square_sum += float(standardized.double().square().sum())
        latent_count += standardized.numel()
        image_minimum = min(image_minimum, float(decoded.min()))
        image_maximum = max(image_maximum, float(decoded.max()))
        clipped += int(((decoded < -1) | (decoded > 1)).sum())
        image_count += decoded.numel()
        generated += current
        print(json.dumps({"generated": generated}), flush=True)

    generated_mean, generated_covariance = generated_moments.compute()
    features = torch.cat(generated_features)
    latent_mean = latent_sum / latent_count
    latent_variance = latent_square_sum / latent_count - latent_mean**2
    metrics = {
        "checkpoint_step": int(payload["step"]),
        "model_type": model_type,
        "num_samples": generated,
        "sample_steps": args.sample_steps,
        "fid": _fid(
            reference["feature_mean"],
            reference["feature_covariance"],
            generated_mean,
            generated_covariance,
        ),
        "kid": _kid(reference["kid_features"], features),
        "standardized_latent_mean": latent_mean,
        "standardized_latent_std": max(latent_variance, 0.0) ** 0.5,
        "decoded_min": image_minimum,
        "decoded_max": image_maximum,
        "decoded_clipping_fraction": clipped / image_count,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    save_image(
        torch.cat(preview).clamp(0, 1),
        output / "samples.png",
        nrow=8,
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
