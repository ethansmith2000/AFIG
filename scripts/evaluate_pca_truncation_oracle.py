#!/usr/bin/env python3
"""Evaluate reconstruction after PCA truncation of one frozen latent cache."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid  # noqa: E402
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint  # noqa: E402


def cifar_test(root: str):
    return torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
            ]
        ),
    )


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--basis_output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument(
        "--reference_cache",
        default="/workspace/AFIG/data/cifar10_test_inception.pt",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[128, 256, 512, 768, 1024, 1536, 2048, 3072],
    )
    parser.add_argument("--pca_examples", type=int, default=25000)
    parser.add_argument("--num_examples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cache_path = Path(args.cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("token_scale") is not None:
        raise ValueError("PCA oracle requires an unscaled physical latent cache")
    train = cache["train_latents"]
    test = cache["test_latents"]
    if train.ndim != 3 or test.ndim != 3 or train.shape[1:] != test.shape[1:]:
        raise ValueError("cache train/test latents must share shape [N,L,D]")
    physical_shape = tuple(int(value) for value in train.shape[1:])
    scalar_count = physical_shape[0] * physical_shape[1]
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] <= 0 or ranks[-1] > scalar_count:
        raise ValueError(f"ranks must lie in [1, {scalar_count}]")
    if args.pca_examples <= 1 or args.pca_examples > train.shape[0]:
        raise ValueError("invalid pca_examples")
    if args.num_examples <= 1 or args.num_examples > test.shape[0]:
        raise ValueError("invalid num_examples")

    generator = torch.Generator().manual_seed(args.seed)
    fit_indices = torch.randperm(train.shape[0], generator=generator)[
        : args.pca_examples
    ]
    fit = train[fit_indices].float().reshape(args.pca_examples, scalar_count)
    mean = fit.mean(dim=0)
    centered = fit - mean
    covariance = (centered.T @ centered).double() / args.pca_examples
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0).float()
    eigenvectors = eigenvectors.flip(1).float()
    total_variance = float(eigenvalues.sum())

    basis_output = Path(args.basis_output)
    basis_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "cache": str(cache_path.resolve()),
            "seed": args.seed,
            "fit_examples": args.pca_examples,
            "physical_shape": physical_shape,
            "mean": mean,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        },
        basis_output,
    )
    del fit, centered, covariance
    gc.collect()

    tokenizer, tokenizer_payload = load_tokenizer_checkpoint(
        cache["tokenizer_checkpoint"]
    )
    if int(tokenizer_payload.get("step", -1)) != int(cache["tokenizer_step"]):
        raise ValueError("cache and tokenizer checkpoint steps differ")
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    extractor = InceptionFeatures(device)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    dataset = cifar_test(args.data_root)
    target_images = torch.stack(
        [dataset[index][0] for index in range(args.num_examples)]
    )
    test_flat = test[: args.num_examples].float().reshape(args.num_examples, scalar_count)
    mean_device = mean.to(device)
    basis_device = eigenvectors.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "cache": str(cache_path.resolve()),
        "basis": str(basis_output.resolve()),
        "tokenizer_step": int(cache["tokenizer_step"]),
        "physical_shape": list(physical_shape),
        "pca_examples": args.pca_examples,
        "evaluation_examples": args.num_examples,
        "seed": args.seed,
        "ranks": {},
    }
    previews = [target_images[:8].add(1).div(2)]

    for rank in ranks:
        basis = basis_device[:, :rank]
        moments = StreamingMoments(2048)
        kid_features = []
        normalized_squared_error = 0.0
        scalar_total = 0
        preview = None
        for start in range(0, args.num_examples, args.batch_size):
            stop = min(start + args.batch_size, args.num_examples)
            clean = test_flat[start:stop].to(device)
            centered_batch = clean - mean_device
            coefficients = centered_batch @ basis
            reconstructed_flat = coefficients @ basis.T + mean_device
            reconstructed_latents = reconstructed_flat.reshape(
                stop - start, *physical_shape
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                decoded = tokenizer.decode(reconstructed_latents).float()
            targets = target_images[start:stop].to(device)
            normalized_squared_error += float(
                (decoded - targets).square().sum().cpu()
            )
            scalar_total += targets.numel()
            display = decoded.add(1).div(2).clamp(0, 1)
            features = extractor(display)
            moments.update(features)
            kid_features.append(features.cpu())
            if preview is None:
                preview = display[:8].cpu()

        feature_mean, feature_covariance = moments.compute()
        normalized_mse = normalized_squared_error / scalar_total
        pixel_mse = normalized_mse / 4.0
        metrics = {
            "retained_variance": float(eigenvalues[:rank].sum())
            / max(total_variance, 1e-30),
            "normalized_mse": normalized_mse,
            "pixel_mse": pixel_mse,
            "psnr_db": -10.0 * math.log10(max(pixel_mse, 1e-30)),
            "reconstruction_fid": _fid(
                reference["feature_mean"],
                reference["feature_covariance"],
                feature_mean,
                feature_covariance,
            ),
            "reconstruction_kid": _kid(
                reference["kid_features"], torch.cat(kid_features)
            ),
        }
        result["ranks"][str(rank)] = metrics
        previews.append(preview)
        atomic_json(result, output_dir / "metrics.json")
        print(json.dumps({"rank": rank, **metrics}, sort_keys=True), flush=True)

    save_image(torch.cat(previews), output_dir / "reconstructions.png", nrow=8)
    print(json.dumps({"complete": str(output_dir / 'metrics.json')}), flush=True)


if __name__ == "__main__":
    main()
