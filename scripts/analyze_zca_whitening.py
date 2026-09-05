#!/usr/bin/env python3
"""Audit rotate-back/ZCA whitening variants before matched-prior training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from PIL import Image

from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.latent_geometry import effective_rank, fit_axis_geometry
from progressive_tokenizer.whitening import (
    apply_zca,
    covariance_diagnostics,
    invert_zca,
    zca_matrix,
    zca_power_gains,
)
from scripts.analyze_generation_trajectory import PLOT_COLORS, draw_line_chart
from scripts.analyze_regularized_whitening import _decode, _quantiles, _relative_rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prior_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fit_samples", type=int, default=25000)
    parser.add_argument("--eval_samples", type=int, default=10000)
    parser.add_argument("--roundtrip_samples", type=int, default=1024)
    parser.add_argument("--decode_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_gammas(value: str) -> list[float]:
    try:
        gammas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("gammas must be comma-separated floats") from error
    if not gammas or any(not math.isfinite(g) or not 0 <= g <= 1 for g in gammas):
        raise ValueError("gammas must lie in [0,1]")
    return gammas


def _reorthogonalize(basis: torch.Tensor) -> torch.Tensor:
    orthogonal, _ = torch.linalg.qr(basis)
    signs = torch.sign((orthogonal * basis).sum(dim=0)).clamp(min=-1, max=1)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return orthogonal * signs


def _expanded_spec(
    name: str,
    geometry: dict[str, object],
    tokens: int,
    channels: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    def summarize(fitted: dict[str, torch.Tensor | float]) -> dict[str, object]:
        fitted_gains = fitted["gains"]
        transformed_power = fitted["transformed_power"]
        assert isinstance(fitted_gains, torch.Tensor)
        assert isinstance(transformed_power, torch.Tensor)
        return {
            "reference_power": fitted["reference_power"],
            "relative_gain_range": fitted["relative_gain_range"],
            "gain_quantiles": _quantiles(fitted_gains),
            "transformed_power_quantiles": _quantiles(transformed_power),
        }

    sequence_basis = geometry["sequence_eigenvectors"]
    sequence_power = geometry["sequence_eigenvalues"]
    channel_basis = geometry["channel_eigenvectors"]
    channel_power = geometry["channel_eigenvalues"]
    flattened_basis = geometry["flattened_eigenvectors"]
    flattened_power = geometry["flattened_eigenvalues"]
    assert all(
        isinstance(value, torch.Tensor)
        for value in (
            sequence_basis,
            sequence_power,
            channel_basis,
            channel_power,
            flattened_basis,
            flattened_power,
        )
    )
    device = sequence_basis.device
    dtype = sequence_basis.dtype
    if name == "channel":
        fitted = zca_power_gains(channel_power, gamma)
        axis_gains = fitted["gains"]
        assert isinstance(axis_gains, torch.Tensor)
        basis = torch.kron(torch.eye(tokens, device=device, dtype=dtype), channel_basis)
        gains = axis_gains.repeat(tokens)
        detail = {"channel": summarize(fitted)}
    elif name == "sequence":
        fitted = zca_power_gains(sequence_power, gamma)
        axis_gains = fitted["gains"]
        assert isinstance(axis_gains, torch.Tensor)
        basis = torch.kron(sequence_basis, torch.eye(channels, device=device, dtype=dtype))
        gains = axis_gains.repeat_interleave(channels)
        detail = {"sequence": summarize(fitted)}
    elif name == "axial":
        sequence_fit = zca_power_gains(sequence_power, gamma)
        channel_fit = zca_power_gains(channel_power, gamma)
        sequence_gains = sequence_fit["gains"]
        channel_gains = channel_fit["gains"]
        assert isinstance(sequence_gains, torch.Tensor)
        assert isinstance(channel_gains, torch.Tensor)
        basis = torch.kron(sequence_basis, channel_basis)
        gains = (sequence_gains[:, None] * channel_gains[None, :]).flatten()
        detail = {
            "sequence": summarize(sequence_fit),
            "channel": summarize(channel_fit),
        }
    elif name == "flattened":
        fitted = zca_power_gains(flattened_power, gamma)
        gains = fitted["gains"]
        assert isinstance(gains, torch.Tensor)
        basis = flattened_basis
        detail = {"flattened": summarize(fitted)}
    else:
        raise ValueError(f"unknown ZCA variant: {name}")
    return basis, gains, detail


def _axis_ranks(values: torch.Tensor) -> dict[str, object]:
    centered = values - values.mean(dim=0)
    count, tokens, channels = centered.shape
    sequence_covariance = torch.einsum("ntd,nsd->ts", centered, centered) / (
        count * channels
    )
    channel_covariance = torch.einsum("ntd,nte->de", centered, centered) / (
        count * tokens
    )
    sequence_power = torch.linalg.eigvalsh(sequence_covariance).clamp_min(0)
    channel_power = torch.linalg.eigvalsh(channel_covariance).clamp_min(0)
    token_power = centered.square().mean(dim=(0, 2))
    return {
        "sequence_effective_rank": effective_rank(sequence_power),
        "channel_effective_rank": effective_rank(channel_power),
        "token_effective_rank": effective_rank(token_power),
        "token_power_ratio": float(token_power.max() / token_power.min().clamp_min(1e-30)),
        "token_power": [float(value) for value in token_power],
    }


def _matrix_geometry(matrix: torch.Tensor, tokens: int, channels: int) -> dict[str, object]:
    squared = matrix.square()
    blocks = squared.reshape(tokens, channels, tokens, channels)
    denominator = blocks.sum(dim=(0, 1, 3)).clamp_min(1e-30)
    self_energy = torch.stack(
        [blocks[index, :, index, :].sum() for index in range(tokens)]
    ) / denominator
    diagonal_fraction = float(matrix.diagonal().square().sum() / squared.sum().clamp_min(1e-30))
    return {
        "diagonal_coordinate_energy_fraction": diagonal_fraction,
        "matching_native_token_energy_fraction": {
            "minimum": float(self_energy.min()),
            "median": float(self_energy.median()),
            "mean": float(self_energy.mean()),
            "maximum": float(self_energy.max()),
        },
        "frobenius_displacement_from_identity": float(
            (matrix - torch.eye(matrix.shape[0], device=matrix.device)).square().mean().sqrt()
        ),
        "symmetry_max_abs_error": float((matrix - matrix.T).abs().max()),
    }


def _plots(result: dict[str, object], output: Path) -> None:
    variants = result["variants"]
    assert isinstance(variants, dict)
    rank_series = []
    token_series = []
    for index, (name, record) in enumerate(variants.items()):
        gammas = record["gammas"]
        xs = [float(key) for key in gammas]
        ranks = [float(gammas[key]["heldout_covariance"]["effective_rank"]) for key in gammas]
        token_self = [
            float(gammas[key]["matrix_geometry"]["matching_native_token_energy_fraction"]["mean"])
            for key in gammas
        ]
        rank_series.append((name, xs, ranks, PLOT_COLORS[index]))
        token_series.append((name, xs, token_self, PLOT_COLORS[index]))
    canvas = Image.new("RGB", (1100, 650), "#FAFAF8")
    draw_line_chart(
        canvas,
        (0, 0, 550, 650),
        rank_series,
        title="ZCA covariance conditioning",
        y_label="rank",
    )
    draw_line_chart(
        canvas,
        (550, 0, 1100, 650),
        token_series,
        title="Native-token self attribution",
        y_label="fraction",
    )
    canvas.save(output / "zca_rank_and_token_attribution.png", optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    gammas = _parse_gammas(args.gammas)
    if min(
        args.fit_samples,
        args.eval_samples,
        args.roundtrip_samples,
        args.decode_samples,
        args.batch_size,
    ) <= 0:
        raise ValueError("sample counts and batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache).resolve()
    checkpoint_path = Path(args.prior_checkpoint).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_mean = checkpoint["normalization"]["mean"].float().to(device)
    source_scale = checkpoint["normalization"]["scale"].float().to(device)

    fit_count = min(args.fit_samples, len(cache["train_latents"]))
    eval_count = min(args.eval_samples, len(cache["test_latents"]))
    fit_raw = cache["train_latents"][:fit_count].to(device).float()
    fit = (fit_raw - source_mean) / source_scale
    geometry = fit_axis_geometry(fit)
    for axis in ("sequence", "channel", "flattened"):
        key = f"{axis}_eigenvectors"
        basis = geometry[key]
        assert isinstance(basis, torch.Tensor)
        geometry[key] = _reorthogonalize(basis).contiguous()
    element_mean = geometry["element_mean"]
    assert isinstance(element_mean, torch.Tensor)
    del fit_raw, fit

    eval_raw = cache["test_latents"][:eval_count].to(device).float()
    standardized = (eval_raw - source_mean) / source_scale
    centered = standardized - element_mean
    baseline_covariance = covariance_diagnostics(centered.flatten(1))
    baseline_axes = _axis_ranks(standardized)
    tokenizer, _ = load_tokenizer_checkpoint(Path(cache["tokenizer_checkpoint"]))
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    decode_count = min(args.decode_samples, eval_count)
    original_images = _decode(eval_raw[:decode_count].cpu(), tokenizer, args.batch_size, device)
    roundtrip_count = min(args.roundtrip_samples, eval_count)
    tokens, channels = standardized.shape[1:]

    result: dict[str, object] = {
        "definition": {
            "transform": "mean + (x - mean) @ U @ diag(gain) @ U.T",
            "gain": "(eigenvalue / mean_eigenvalue)^(-gamma/2)",
            "identity": "gamma zero bypasses matrix multiplication and is exactly the native standardized representation",
        },
        "source": {
            "cache": str(cache_path),
            "prior_checkpoint": str(checkpoint_path),
            "fit_samples": fit_count,
            "eval_samples": eval_count,
            "shape": [tokens, channels],
        },
        "baseline": {
            "heldout_covariance": baseline_covariance,
            "axis_ranks": baseline_axes,
        },
        "variants": {},
    }
    identity_errors = []
    fp32_errors = []
    fp16_errors = []
    pixel_errors = []
    variants = result["variants"]
    assert isinstance(variants, dict)
    for variant_name in ("channel", "sequence", "axial", "flattened"):
        variant: dict[str, object] = {"gammas": {}}
        variants[variant_name] = variant
        gamma_records = variant["gammas"]
        assert isinstance(gamma_records, dict)
        for gamma in gammas:
            basis, gains, gain_detail = _expanded_spec(
                variant_name, geometry, tokens, channels, gamma
            )
            if gamma == 0:
                transformed = standardized
                matrix = torch.eye(tokens * channels, device=device)
                restored_float = standardized[:roundtrip_count]
            else:
                transformed = apply_zca(
                    standardized, element_mean, basis, gains
                )
                matrix = zca_matrix(basis, gains)
                restored_float = invert_zca(
                    transformed[:roundtrip_count], element_mean, basis, gains
                )
            identity_error = (
                float((transformed - standardized).abs().max()) if gamma == 0 else None
            )
            if identity_error is not None:
                identity_errors.append(identity_error)
            fp32_error = _relative_rms(
                restored_float,
                standardized[:roundtrip_count],
            )
            transformed_raw = transformed[:roundtrip_count] * source_scale + source_mean
            stored = transformed_raw.half().float()
            stored_standardized = (stored - source_mean) / source_scale
            restored_half = invert_zca(
                stored_standardized, element_mean, basis, gains
            ) if gamma != 0 else stored_standardized
            restored_raw = restored_half * source_scale + source_mean
            fp16_error = _relative_rms(
                restored_raw,
                eval_raw[:roundtrip_count],
            )
            decoded = _decode(
                restored_raw[:decode_count].cpu(), tokenizer, args.batch_size, device
            )
            pixel_error = float(
                (decoded - original_images).double().square().mean().sqrt()
            )
            fp32_errors.append(fp32_error)
            fp16_errors.append(fp16_error)
            pixel_errors.append(pixel_error)
            transformed_centered = transformed - transformed.mean(dim=0)
            native_correlation = float(
                (transformed_centered * (standardized - standardized.mean(dim=0))).sum()
                / (
                    transformed_centered.square().sum().sqrt()
                    * (standardized - standardized.mean(dim=0)).square().sum().sqrt()
                ).clamp_min(1e-30)
            )
            record = {
                "gamma": gamma,
                "gain_detail": gain_detail,
                "relative_gain_range": float(gains.max() / gains.min()),
                "heldout_covariance": covariance_diagnostics(transformed_centered.flatten(1)),
                "axis_ranks": _axis_ranks(transformed),
                "matrix_geometry": _matrix_geometry(matrix, tokens, channels),
                "native_centered_correlation": native_correlation,
                "relative_displacement_rms": _relative_rms(
                    transformed, standardized
                ),
                "identity_max_abs_error": identity_error,
                "float32_roundtrip_relative_rms": fp32_error,
                "float16_cache_roundtrip_relative_rms": fp16_error,
                "decoded_pixel_delta_rms": pixel_error,
            }
            gamma_records[f"{gamma:g}"] = record
            print(
                json.dumps(
                    {
                        "variant": variant_name,
                        "gamma": gamma,
                        "rank": record["heldout_covariance"]["effective_rank"],
                        "token_self": record["matrix_geometry"]["matching_native_token_energy_fraction"]["mean"],
                        "fp16": fp16_error,
                        "pixel": pixel_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    variant_gates = {}
    for variant_name, variant in variants.items():
        records = list(variant["gammas"].values())
        identity_error = max(
            float(record["identity_max_abs_error"])
            for record in records
            if record["identity_max_abs_error"] is not None
        )
        maximum_float32 = max(
            float(record["float32_roundtrip_relative_rms"]) for record in records
        )
        maximum_float16 = max(
            float(record["float16_cache_roundtrip_relative_rms"]) for record in records
        )
        maximum_pixel = max(float(record["decoded_pixel_delta_rms"]) for record in records)
        variant_gates[variant_name] = {
            "identity_max_abs_error": identity_error,
            "maximum_float32_roundtrip_relative_rms": maximum_float32,
            "maximum_float16_cache_roundtrip_relative_rms": maximum_float16,
            "maximum_decoded_pixel_delta_rms": maximum_pixel,
            "healthy": (
                identity_error <= 1e-6
                and maximum_float32 <= 1e-5
                and maximum_float16 <= 0.002
                and maximum_pixel <= 0.002
            ),
        }
    selected_variant = (
        "axial"
        if variant_gates["axial"]["healthy"]
        else "channel" if variant_gates["channel"]["healthy"] else None
    )
    result["gates"] = {
        "identity_max_abs_error_limit": 1e-6,
        "float32_roundtrip_relative_rms_limit": 1e-5,
        "float16_cache_roundtrip_relative_rms_limit": 0.002,
        "decoded_pixel_delta_relative_rms_limit": 0.002,
        "observed_identity_max_abs_error": max(identity_errors),
        "observed_float32_roundtrip_relative_rms": max(fp32_errors),
        "observed_float16_cache_roundtrip_relative_rms": max(fp16_errors),
        "observed_decoded_pixel_delta_rms": max(pixel_errors),
        "variants": variant_gates,
        "selected_training_variant": selected_variant,
        "training_authorized": selected_variant is not None,
    }
    asset = {
        "version": 1,
        "type": "zca_axis_geometry",
        "source_cache": str(cache_path),
        "prior_checkpoint": str(checkpoint_path),
        "physical_shape": [tokens, channels],
        "source_normalization_mean": source_mean.detach().cpu(),
        "source_normalization_scale": source_scale.detach().cpu(),
        "standardized_element_mean": element_mean.detach().cpu(),
        "fit_samples": fit_count,
        "sequence_eigenvectors": geometry["sequence_eigenvectors"].detach().cpu(),
        "sequence_eigenvalues": geometry["sequence_eigenvalues"].detach().cpu(),
        "channel_eigenvectors": geometry["channel_eigenvectors"].detach().cpu(),
        "channel_eigenvalues": geometry["channel_eigenvalues"].detach().cpu(),
        "flattened_eigenvectors": geometry["flattened_eigenvectors"].detach().cpu(),
        "flattened_eigenvalues": geometry["flattened_eigenvalues"].detach().cpu(),
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(asset, output / "zca_geometry.pt")
    _plots(result, output)
    print(json.dumps({"complete": str(metrics_path), **result["gates"]}), flush=True)


if __name__ == "__main__":
    main()
