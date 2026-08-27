#!/usr/bin/env python3
"""Project a frozen tokenizer cache into a retained PCA coefficient cache.

The output remains a tokenizer-latent representation, but carries an explicit
``pca_inverse`` transform. Prior samples are inverse-projected into the original
physical latent shape before the unchanged tokenizer decoder is called.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=1536)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project(
    latents: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    sequence_length: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    rank = basis.shape[1]
    token_dim = rank // sequence_length
    output = torch.empty(
        latents.shape[0], sequence_length, token_dim, dtype=torch.float16
    )
    for start in range(0, latents.shape[0], chunk_size):
        stop = min(start + chunk_size, latents.shape[0])
        flat = latents[start:stop].float().flatten(1).to(device)
        coefficients = (flat - mean) @ basis
        output[start:stop] = coefficients.reshape(
            stop - start, sequence_length, token_dim
        ).half().cpu()
        print(json.dumps({"projected": stop, "total": latents.shape[0]}), flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.rank <= 0 or args.sequence_length <= 0 or args.chunk_size <= 0:
        raise ValueError("rank, sequence_length, and chunk_size must be positive")
    if args.rank % args.sequence_length:
        raise ValueError("rank must be divisible by sequence_length")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    cache_path = Path(args.cache).resolve()
    basis_path = Path(args.basis).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    pca = torch.load(basis_path, map_location="cpu", weights_only=False)
    if cache.get("token_scale") is not None or cache.get("latent_transform") is not None:
        raise ValueError("source cache must be an unscaled physical latent cache")
    physical_shape = tuple(int(value) for value in cache["train_latents"].shape[1:])
    if physical_shape != tuple(int(value) for value in pca["physical_shape"]):
        raise ValueError("PCA basis and cache physical shapes differ")
    scalar_count = math.prod(physical_shape)
    if args.rank > scalar_count:
        raise ValueError("rank exceeds the physical latent scalar count")
    if Path(pca["cache"]).resolve() != cache_path:
        raise ValueError("PCA basis was fitted from a different latent cache")

    mean_cpu = pca["mean"].float()
    basis_cpu = pca["eigenvectors"][:, : args.rank].float().contiguous()
    if tuple(basis_cpu.shape) != (scalar_count, args.rank):
        raise ValueError("invalid retained PCA basis shape")
    mean = mean_cpu.to(device)
    basis = basis_cpu.to(device)
    train = project(
        cache["train_latents"],
        mean,
        basis,
        args.sequence_length,
        args.chunk_size,
        device,
    )
    test = project(
        cache["test_latents"],
        mean,
        basis,
        args.sequence_length,
        args.chunk_size,
        device,
    )

    train_float = train.float()
    eigenvalues = pca["eigenvalues"][: args.rank].double().clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    )
    output_payload = dict(cache)
    output_payload["train_latents"] = train
    output_payload["test_latents"] = test
    output_payload["statistics"] = {
        "global_mean": train_float.mean(),
        "global_std": train_float.std(),
        "global_min": train_float.min(),
        "global_max": train_float.max(),
        "coordinate_mean": train_float.mean(dim=(0, 1)),
        "coordinate_std": train_float.std(dim=(0, 1)),
        "slot_mean": train_float.mean(dim=(0, 2)),
        "slot_std": train_float.std(dim=(0, 2)),
        "coordinate_covariance_effective_rank": effective_rank.float(),
    }
    output_payload["latent_transform"] = {
        "type": "pca_inverse",
        "physical_shape": list(physical_shape),
        "mean": mean_cpu,
        "basis": basis_cpu,
        "source": str(basis_path),
    }
    output_payload["pca_config"] = {
        "basis": str(basis_path),
        "rank": args.rank,
        "prior_shape": [args.sequence_length, args.rank // args.sequence_length],
        "physical_shape": list(physical_shape),
        "retained_variance": float(
            pca["eigenvalues"][: args.rank].sum() / pca["eigenvalues"].sum()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "complete": str(output.resolve()),
                "rank": args.rank,
                "prior_shape": list(train.shape[1:]),
                "physical_shape": list(physical_shape),
                "retained_variance": output_payload["pca_config"]["retained_variance"],
                "global_mean": float(output_payload["statistics"]["global_mean"]),
                "global_std": float(output_payload["statistics"]["global_std"]),
                "effective_rank": float(effective_rank),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
