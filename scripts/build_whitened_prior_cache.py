#!/usr/bin/env python3
"""Build a float16 prior cache from a frozen invertible whitening transform."""

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
from progressive_tokenizer.whitening import project_linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def _project_split(
    values: torch.Tensor,
    source_mean: torch.Tensor,
    source_scale: torch.Tensor,
    element_mean: torch.Tensor,
    basis: torch.Tensor,
    gains: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty_like(values, dtype=torch.float16, device="cpu")
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        raw = values[start:stop].to(device).float()
        standardized = (raw - source_mean) / source_scale
        coefficients = project_linear(
            standardized, element_mean, basis, gains
        ).reshape(stop - start, *values.shape[1:])
        output[start:stop] = coefficients.half().cpu()
        print(json.dumps({"projected": stop, "total": len(values)}), flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_path = Path(args.cache).resolve()
    transform_path = Path(args.transform).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    transform = torch.load(transform_path, map_location="cpu", weights_only=False)
    if cache.get("latent_transform") is not None or cache.get("token_scale") is not None:
        raise ValueError("source cache must be an untransformed physical latent cache")
    if transform.get("type") != "regularized_linear_whitening":
        raise ValueError("unsupported transform asset")
    if Path(transform["source_cache"]).resolve() != cache_path:
        raise ValueError("transform was fitted from a different source cache")
    physical_shape = tuple(int(value) for value in cache["train_latents"].shape[1:])
    if physical_shape != tuple(int(value) for value in transform["physical_shape"]):
        raise ValueError("cache and transform physical shapes differ")

    source_mean = transform["source_normalization_mean"].float().to(device)
    source_scale = transform["source_normalization_scale"].float().to(device)
    element_mean = transform["standardized_element_mean"].float().to(device)
    basis = transform["basis"].float().to(device)
    gains = transform["gains"].float().to(device)
    dimensions = physical_shape[0] * physical_shape[1]
    if basis.shape != (dimensions, dimensions) or gains.shape != (dimensions,):
        raise ValueError("invalid basis or gain shape")

    train = _project_split(
        cache["train_latents"],
        source_mean,
        source_scale,
        element_mean,
        basis,
        gains,
        args.chunk_size,
        device,
    )
    test = _project_split(
        cache["test_latents"],
        source_mean,
        source_scale,
        element_mean,
        basis,
        gains,
        args.chunk_size,
        device,
    )

    # y = ((x - scalar_mean)/scalar_scale - element_mean) @ basis * gains
    # x = y @ (scalar_scale * basis / gains).T
    #     + scalar_scale * element_mean + scalar_mean
    inverse_basis = (
        basis / gains[None, :] * source_scale
    ).detach().cpu().float()
    physical_mean = (
        element_mean * source_scale + source_mean
    ).detach().cpu().float().flatten()
    output_payload = dict(cache)
    output_payload["train_latents"] = train
    output_payload["test_latents"] = test
    train_float = train.float()
    selected = dict(transform["selection"])
    output_payload["statistics"] = {
        "global_mean": train_float.mean(),
        "global_std": train_float.std(),
        "global_min": train_float.min(),
        "global_max": train_float.max(),
        "coordinate_mean": train_float.mean(dim=(0, 1)),
        "coordinate_std": train_float.std(dim=(0, 1)),
        "slot_mean": train_float.mean(dim=(0, 2)),
        "slot_std": train_float.std(dim=(0, 2)),
    }
    output_payload["latent_transform"] = {
        "type": "linear_inverse",
        "physical_shape": list(physical_shape),
        "mean": physical_mean,
        "basis": inverse_basis,
        "source": str(transform_path),
    }
    output_payload["whitening_config"] = {
        "type": "factorized_sequence_channel",
        "relative_gain_cap": float(selected["gain_cap"]),
        "beta": float(selected["beta"]),
        "snr1_crossings": list(selected["snr1_crossings"]),
        "flow_target_energy_loss_weights": list(selected["loss_weights"]),
        "source_cache": str(cache_path),
        "transform": str(transform_path),
    }

    restored = invert_latent_transform(train[:16].float(), output_payload)
    source = cache["train_latents"][:16].float()
    relative_error = float(
        (restored - source).double().square().mean().sqrt()
        / source.double().square().mean().sqrt().clamp_min(1e-30)
    )
    if not math.isfinite(relative_error) or relative_error > 0.002:
        raise RuntimeError(f"cache inversion failed: relative RMS {relative_error}")
    output_payload["whitening_config"]["cache_roundtrip_relative_rms"] = relative_error

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(output_payload, temporary)
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "complete": str(output.resolve()),
                "shape": list(train.shape),
                "global_mean": float(output_payload["statistics"]["global_mean"]),
                "global_std": float(output_payload["statistics"]["global_std"]),
                "roundtrip_relative_rms": relative_error,
                "selection": selected,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
