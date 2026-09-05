#!/usr/bin/env python3
"""Build an invertible factorized power-whitened float16 prior cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from progressive_tokenizer.representations import invert_latent_transform
from progressive_tokenizer.whitening import power_whitening_gains
from scripts.build_whitened_prior_cache import _project_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or not 0 <= args.gamma <= 1:
        raise ValueError("require positive chunk_size and gamma in [0,1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_path = Path(args.cache).resolve()
    transform_path = Path(args.transform).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    transform = torch.load(transform_path, map_location="cpu", weights_only=False)
    if cache.get("latent_transform") is not None or cache.get("token_scale") is not None:
        raise ValueError("source cache must be untransformed")
    if Path(transform["source_cache"]).resolve() != cache_path:
        raise ValueError("transform and source cache disagree")

    source_mean = transform["source_normalization_mean"].float().to(device)
    source_scale = transform["source_normalization_scale"].float().to(device)
    element_mean = transform["standardized_element_mean"].float().to(device)
    basis = transform["basis"].float().to(device)
    power = transform["coordinate_power"].float().to(device)
    fitted = power_whitening_gains(power, args.gamma)
    gains = fitted["gains"]
    assert isinstance(gains, torch.Tensor)

    train = _project_split(
        cache["train_latents"], source_mean, source_scale, element_mean,
        basis, gains, args.chunk_size, device
    )
    test = _project_split(
        cache["test_latents"], source_mean, source_scale, element_mean,
        basis, gains, args.chunk_size, device
    )
    physical_shape = tuple(int(value) for value in train.shape[1:])
    inverse_basis = (basis / gains[None, :] * source_scale).cpu().float()
    physical_mean = (element_mean * source_scale + source_mean).cpu().float().flatten()
    train_float = train.float()
    payload = dict(cache)
    payload["train_latents"] = train
    payload["test_latents"] = test
    payload["statistics"] = {
        "global_mean": train_float.mean(),
        "global_std": train_float.std(),
        "global_min": train_float.min(),
        "global_max": train_float.max(),
        "coordinate_mean": train_float.mean(dim=(0, 1)),
        "coordinate_std": train_float.std(dim=(0, 1)),
        "slot_mean": train_float.mean(dim=(0, 2)),
        "slot_std": train_float.std(dim=(0, 2)),
    }
    payload["latent_transform"] = {
        "type": "linear_inverse",
        "physical_shape": list(physical_shape),
        "mean": physical_mean,
        "basis": inverse_basis,
        "source": str(transform_path),
    }
    payload["whitening_config"] = {
        "type": "factorized_power",
        "gamma": float(args.gamma),
        "relative_gain_range": float(fitted["relative_gain_range"]),
        "formula": "gain proportional to coordinate_power^(-gamma/2)",
        "source_cache": str(cache_path),
        "transform": str(transform_path),
    }
    restored = invert_latent_transform(train[:16].float(), payload)
    source = cache["train_latents"][:16].float()
    relative_error = float(
        (restored - source).double().square().mean().sqrt()
        / source.double().square().mean().sqrt().clamp_min(1e-30)
    )
    if not math.isfinite(relative_error) or relative_error > 0.002:
        raise RuntimeError(f"cache inversion failed: relative RMS {relative_error}")
    payload["whitening_config"]["cache_roundtrip_relative_rms"] = relative_error

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "complete": str(output.resolve()),
                "gamma": args.gamma,
                "relative_gain_range": fitted["relative_gain_range"],
                "global_mean": float(payload["statistics"]["global_mean"]),
                "global_std": float(payload["statistics"]["global_std"]),
                "roundtrip_relative_rms": relative_error,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
