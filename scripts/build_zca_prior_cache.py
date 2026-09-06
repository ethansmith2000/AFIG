#!/usr/bin/env python3
"""Build an invertible rotate-back/ZCA float16 prior cache."""

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
from progressive_tokenizer.whitening import (
    zca_inverse_affine,
    zca_matrix,
    zca_power_gains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        choices=("channel", "sequence", "axial", "flattened"),
        default="axial",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _zca_spec(
    geometry: dict[str, object],
    variant: str,
    gamma: float,
    tokens: int,
    channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    sequence_basis = geometry["sequence_eigenvectors"]
    sequence_power = geometry["sequence_eigenvalues"]
    channel_basis = geometry["channel_eigenvectors"]
    channel_power = geometry["channel_eigenvalues"]
    flattened_basis = geometry["flattened_eigenvectors"]
    flattened_power = geometry["flattened_eigenvalues"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            sequence_basis,
            sequence_power,
            channel_basis,
            channel_power,
            flattened_basis,
            flattened_power,
        )
    ):
        raise ValueError("geometry asset is missing required eigenspaces")
    sequence_basis = sequence_basis.float().to(device).contiguous()
    sequence_power = sequence_power.float().to(device)
    channel_basis = channel_basis.float().to(device).contiguous()
    channel_power = channel_power.float().to(device)
    flattened_basis = flattened_basis.float().to(device).contiguous()
    flattened_power = flattened_power.float().to(device)
    identity_tokens = torch.eye(tokens, device=device, dtype=sequence_basis.dtype)
    identity_channels = torch.eye(channels, device=device, dtype=channel_basis.dtype)
    if variant == "channel":
        fitted = zca_power_gains(channel_power, gamma)
        axis_gains = fitted["gains"]
        assert isinstance(axis_gains, torch.Tensor)
        basis = torch.kron(identity_tokens, channel_basis)
        gains = axis_gains.repeat(tokens)
        metadata = {
            "channel_relative_gain_range": float(fitted["relative_gain_range"]),
            "relative_gain_range": float(fitted["relative_gain_range"]),
        }
    elif variant == "sequence":
        fitted = zca_power_gains(sequence_power, gamma)
        axis_gains = fitted["gains"]
        assert isinstance(axis_gains, torch.Tensor)
        basis = torch.kron(sequence_basis, identity_channels)
        gains = axis_gains.repeat_interleave(channels)
        metadata = {
            "sequence_relative_gain_range": float(fitted["relative_gain_range"]),
            "relative_gain_range": float(fitted["relative_gain_range"]),
        }
    elif variant == "axial":
        sequence_fit = zca_power_gains(sequence_power, gamma)
        channel_fit = zca_power_gains(channel_power, gamma)
        sequence_gains = sequence_fit["gains"]
        channel_gains = channel_fit["gains"]
        assert isinstance(sequence_gains, torch.Tensor)
        assert isinstance(channel_gains, torch.Tensor)
        basis = torch.kron(sequence_basis, channel_basis)
        gains = (sequence_gains[:, None] * channel_gains[None, :]).flatten()
        metadata = {
            "sequence_relative_gain_range": float(sequence_fit["relative_gain_range"]),
            "channel_relative_gain_range": float(channel_fit["relative_gain_range"]),
            "relative_gain_range": float(gains.max() / gains.min()),
        }
    elif variant == "flattened":
        fitted = zca_power_gains(flattened_power, gamma)
        gains = fitted["gains"]
        assert isinstance(gains, torch.Tensor)
        basis = flattened_basis
        metadata = {
            "flattened_relative_gain_range": float(fitted["relative_gain_range"]),
            "relative_gain_range": float(fitted["relative_gain_range"]),
        }
    else:
        raise ValueError(f"unknown ZCA variant: {variant}")
    return basis, gains, metadata


@torch.no_grad()
def _transform_split(
    values: torch.Tensor,
    physical_mean: torch.Tensor,
    matrix: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty_like(values, dtype=torch.float16, device="cpu")
    flat_mean = physical_mean.flatten()
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        raw = values[start:stop].to(device).float()
        transformed = (raw - physical_mean).flatten(1) @ matrix + flat_mean
        output[start:stop] = transformed.reshape_as(raw).half().cpu()
        print(json.dumps({"transformed": stop, "total": len(values)}), flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or not 0 <= args.gamma <= 1:
        raise ValueError("require positive chunk_size and gamma in [0,1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    cache_path = Path(args.cache).resolve()
    geometry_path = Path(args.geometry).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=False)
    if cache.get("latent_transform") is not None or cache.get("token_scale") is not None:
        raise ValueError("source cache must be an untransformed physical latent cache")
    if geometry.get("type") != "zca_axis_geometry" or int(geometry.get("version", -1)) != 1:
        raise ValueError("unsupported ZCA geometry asset")
    if Path(str(geometry["source_cache"])).resolve() != cache_path:
        raise ValueError("geometry was fitted from a different source cache")
    physical_shape = tuple(int(value) for value in cache["train_latents"].shape[1:])
    if physical_shape != tuple(int(value) for value in geometry["physical_shape"]):
        raise ValueError("cache and geometry physical shapes differ")

    dimensions = physical_shape[0] * physical_shape[1]
    basis, gains, gain_metadata = _zca_spec(
        geometry,
        args.variant,
        args.gamma,
        physical_shape[0],
        physical_shape[1],
        device,
    )
    if basis.shape != (dimensions, dimensions) or gains.shape != (dimensions,):
        raise ValueError("expanded ZCA transform has the wrong shape")
    matrix = zca_matrix(basis, gains)
    source_mean = geometry["source_normalization_mean"]
    source_scale = geometry["source_normalization_scale"]
    element_mean = geometry["standardized_element_mean"]
    if not all(isinstance(value, torch.Tensor) for value in (source_mean, source_scale, element_mean)):
        raise ValueError("geometry asset is missing normalization tensors")
    physical_mean = (
        element_mean.float().to(device) * source_scale.float().to(device)
        + source_mean.float().to(device)
    )

    train = _transform_split(
        cache["train_latents"], physical_mean, matrix, args.chunk_size, device
    )
    test = _transform_split(
        cache["test_latents"], physical_mean, matrix, args.chunk_size, device
    )
    inverse_offset, inverse_basis = zca_inverse_affine(physical_mean, basis, gains)

    payload = dict(cache)
    payload["train_latents"] = train
    payload["test_latents"] = test
    train_float = train.float()
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
        "mean": inverse_offset.detach().cpu().float(),
        "basis": inverse_basis.detach().cpu().float(),
        "source": str(geometry_path),
    }
    payload["whitening_config"] = {
        "type": f"{args.variant}_zca_power",
        "variant": args.variant,
        "gamma": float(args.gamma),
        **gain_metadata,
        "formula": "rotate-back ZCA with selected-axis gain proportional to normalized eigenvalue^(-gamma/2)",
        "source_cache": str(cache_path),
        "geometry": str(geometry_path),
        "clean_token_magnitude_rescaling": False,
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
                "variant": args.variant,
                "gamma": args.gamma,
                **gain_metadata,
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
