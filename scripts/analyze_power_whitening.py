#!/usr/bin/env python3
"""Analyze smooth power whitening and weak-tail stability before training."""

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
from progressive_tokenizer.latent_geometry import descending_eigh
from progressive_tokenizer.whitening import (
    covariance_diagnostics,
    invert_linear,
    power_whitening_gains,
    project_linear,
)
from scripts.analyze_generation_trajectory import PLOT_COLORS, draw_line_chart
from scripts.analyze_regularized_whitening import _decode, _quantiles, _relative_rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fit_samples", type=int, default=25000)
    parser.add_argument("--eval_samples", type=int, default=10000)
    parser.add_argument("--roundtrip_samples", type=int, default=1024)
    parser.add_argument("--decode_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gammas", default="0,0.125,0.25,0.5,0.75,1")
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


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    return float(
        (left_rank * right_rank).sum()
        / (
            left_rank.square().sum().sqrt()
            * right_rank.square().sum().sqrt()
        ).clamp_min(1e-30)
    )


def _fixed_basis_split_stability(
    coefficients: torch.Tensor,
) -> dict[str, object]:
    middle = coefficients.shape[0] // 2
    powers = [
        coefficients[:middle].square().mean(dim=0),
        coefficients[middle:].square().mean(dim=0),
    ]
    token_powers = [power.reshape(64, 16).mean(dim=1) for power in powers]
    log_ratio = (powers[0].clamp_min(1e-30) / powers[1].clamp_min(1e-30)).log()
    token_log_ratio = (
        token_powers[0].clamp_min(1e-30)
        / token_powers[1].clamp_min(1e-30)
    ).log()
    return {
        "coordinate_power_spearman": _rank_correlation(*powers),
        "token_power_spearman": _rank_correlation(*token_powers),
        "coordinate_absolute_log_ratio_quantiles": _quantiles(log_ratio.abs()),
        "token_absolute_log_ratio_quantiles": _quantiles(token_log_ratio.abs()),
        "half_token_power": [
            [float(value) for value in token_power] for token_power in token_powers
        ],
    }


def _sequence_subspace_stability(
    standardized: torch.Tensor,
) -> dict[str, object]:
    middle = standardized.shape[0] // 2
    bases = []
    values = []
    for half in (standardized[:middle], standardized[middle:]):
        centered = half - half.mean(dim=0)
        covariance = torch.einsum("ntd,nsd->ts", centered, centered) / (
            centered.shape[0] * centered.shape[2]
        )
        eigenvalues, eigenvectors = descending_eigh(covariance)
        values.append(eigenvalues)
        bases.append(eigenvectors)
    edges = [0, 2, 4, 8, 16, 32, 63, 64]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        overlap = bases[0][:, lo:hi].T @ bases[1][:, lo:hi]
        bands.append(
            {
                "range_zero_based_half_open": [lo, hi],
                "mean_squared_principal_overlap": float(
                    overlap.square().sum() / (hi - lo)
                ),
                "half_1_mean_power": float(values[0][lo:hi].mean()),
                "half_2_mean_power": float(values[1][lo:hi].mean()),
            }
        )
    return {
        "bands": bands,
        "eigenvalue_spearman": _rank_correlation(values[0], values[1]),
    }


def _source_quantization_proxy(
    stored: torch.Tensor,
    source_scale: torch.Tensor,
    basis: torch.Tensor,
    signal_power: torch.Tensor,
) -> dict[str, object]:
    if stored.dtype != torch.float16:
        raise ValueError("quantization proxy expects the source float16 cache")
    positive_infinity = torch.full_like(stored, float("inf"))
    steps = (torch.nextafter(stored, positive_infinity) - stored).float()
    input_variance = steps.square().mean(dim=0) / 12.0
    input_variance = input_variance / source_scale.float().cpu().square()
    mode_variance = input_variance.flatten().to(basis.device) @ basis.square()
    snr = signal_power / mode_variance.clamp_min(1e-30)
    token_signal = signal_power.reshape(64, 16).mean(dim=1)
    token_noise = mode_variance.reshape(64, 16).mean(dim=1)
    token_snr = token_signal / token_noise.clamp_min(1e-30)
    return {
        "definition": "fitted mode power divided by propagated mean local float16 rounding-step variance (step^2/12)",
        "coordinate_snr_quantiles": _quantiles(snr),
        "token_snr_quantiles": _quantiles(token_snr),
        "coordinate_fraction_below_10": float((snr < 10).float().mean()),
        "coordinate_fraction_below_100": float((snr < 100).float().mean()),
        "coordinate_fraction_below_1000": float((snr < 1000).float().mean()),
        "weakest_coordinate_snr": float(snr.min()),
        "weakest_token_snr": float(token_snr.min()),
    }


def _plots(result: dict[str, object], output: Path) -> None:
    gammas = result["gammas"]
    assert isinstance(gammas, dict)
    x = [float(key) for key in gammas]
    ranks = [float(gammas[key]["heldout_covariance"]["effective_rank"]) for key in gammas]
    gains = [float(gammas[key]["relative_gain_range"]) for key in gammas]
    canvas = Image.new("RGB", (1000, 600), "#FAFAF8")
    draw_line_chart(
        canvas,
        (0, 0, 500, 600),
        [("effective rank", x, ranks, PLOT_COLORS[0])],
        title="Power whitening versus covariance rank",
        y_label="rank",
    )
    draw_line_chart(
        canvas,
        (500, 0, 1000, 600),
        [("gain range", x, gains, PLOT_COLORS[1])],
        title="Intentional weak-direction amplification",
        y_label="gain",
        log_y=True,
    )
    canvas.save(output / "gamma_rank_and_gain.png", optimize=True)

    spectrum = Image.new("RGB", (1000, 620), "#FAFAF8")
    series = []
    for index, key in enumerate(gammas):
        if float(key) not in {0.0, 0.25, 0.5, 1.0}:
            continue
        values = gammas[key]["heldout_covariance"]["eigenvalues"]
        relative = [
            max(float(value) / max(float(values[0]), 1e-30), 1e-12)
            for value in values
        ]
        series.append(
            (f"gamma={key}", list(range(1, len(values) + 1)), relative, PLOT_COLORS[index])
        )
    draw_line_chart(
        spectrum,
        (0, 0, 1000, 620),
        series,
        title="Held-out covariance under smooth power whitening",
        y_label="relative eig.",
        log_y=True,
    )
    spectrum.save(output / "gamma_covariance_spectra.png", optimize=True)


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
    transform_path = Path(args.transform).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    transform = torch.load(transform_path, map_location="cpu", weights_only=False)
    if Path(transform["source_cache"]).resolve() != cache_path:
        raise ValueError("transform and source cache disagree")
    basis = transform["basis"].float().to(device)
    power = transform["coordinate_power"].float().to(device)
    element_mean = transform["standardized_element_mean"].float().to(device)
    source_mean = transform["source_normalization_mean"].float().to(device)
    source_scale = transform["source_normalization_scale"].float().to(device)

    fit_count = min(args.fit_samples, len(cache["train_latents"]))
    raw_fit = cache["train_latents"][:fit_count]
    standardized_fit = (raw_fit.to(device).float() - source_mean) / source_scale
    fixed_coefficients = project_linear(
        standardized_fit, element_mean, basis, torch.ones_like(power)
    )
    split_stability = _fixed_basis_split_stability(fixed_coefficients)
    sequence_stability = _sequence_subspace_stability(standardized_fit)
    quantization = _source_quantization_proxy(
        raw_fit.cpu(), source_scale.cpu(), basis, power
    )
    del standardized_fit, fixed_coefficients

    eval_count = min(args.eval_samples, len(cache["test_latents"]))
    raw_test = cache["test_latents"][:eval_count].to(device).float()
    standardized_test = (raw_test - source_mean) / source_scale
    base_coefficients = project_linear(
        standardized_test, element_mean, basis, torch.ones_like(power)
    )
    tokenizer, _ = load_tokenizer_checkpoint(Path(cache["tokenizer_checkpoint"]))
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    decode_count = min(args.decode_samples, eval_count)
    original_images = _decode(
        cache["test_latents"][:decode_count].float(), tokenizer, args.batch_size, device
    )
    roundtrip_count = min(args.roundtrip_samples, eval_count)

    gamma_results: dict[str, object] = {}
    for gamma in gammas:
        fitted = power_whitening_gains(power, gamma)
        gains = fitted["gains"]
        assert isinstance(gains, torch.Tensor)
        transformed = base_coefficients * gains
        diagnostics = covariance_diagnostics(transformed)
        source_subset = standardized_test[:roundtrip_count]
        projected = project_linear(source_subset, element_mean, basis, gains)
        restored_float = invert_linear(projected, element_mean, basis, gains)
        restored_half = invert_linear(
            projected.half().float(), element_mean, basis, gains
        )
        recovered_raw = (
            restored_half[:decode_count] * source_scale + source_mean
        ).cpu()
        recovered_images = _decode(
            recovered_raw, tokenizer, args.batch_size, device
        )
        pixel_delta = recovered_images.double() - original_images.double()
        metrics = {
            "relative_gain_range": float(fitted["relative_gain_range"]),
            "gain_quantiles": _quantiles(gains),
            "training_transformed_power_quantiles": _quantiles(
                fitted["transformed_power"]  # type: ignore[arg-type]
            ),
            "heldout_covariance": diagnostics,
            "float32_relative_roundtrip_rms": _relative_rms(
                restored_float, source_subset
            ),
            "float16_relative_roundtrip_rms": _relative_rms(
                restored_half, source_subset
            ),
            "decoded_float16_pixel_delta_rms": float(
                pixel_delta.square().mean().sqrt()
            ),
            "decoded_float16_pixel_delta_max_abs": float(pixel_delta.abs().max()),
        }
        metrics["numeric_health"] = bool(
            metrics["float32_relative_roundtrip_rms"] <= 1e-5
            and metrics["float16_relative_roundtrip_rms"] <= 0.002
            and metrics["decoded_float16_pixel_delta_rms"] <= 0.002
        )
        gamma_results[f"{gamma:g}"] = metrics
        print(
            json.dumps(
                {
                    "gamma": gamma,
                    "gain_range": metrics["relative_gain_range"],
                    "effective_rank": diagnostics["effective_rank"],
                    "numeric_health": metrics["numeric_health"],
                }
            ),
            flush=True,
        )

    result: dict[str, object] = {
        "status": "complete",
        "source_cache": str(cache_path),
        "transform": str(transform_path),
        "fit_samples": fit_count,
        "eval_samples": eval_count,
        "fixed_basis_split_stability": split_stability,
        "sequence_subspace_split_stability": sequence_stability,
        "source_float16_quantization_proxy": quantization,
        "gammas": gamma_results,
        "training_gamma_values": [0.0, 0.25, 0.5, 1.0],
        "training_authorized": all(
            gamma_results[f"{gamma:g}"]["numeric_health"]  # type: ignore[index]
            for gamma in (0.0, 0.25, 0.5, 1.0)
        ),
    }
    result_path = output / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plots(result, output)
    print(json.dumps({"complete": str(result_path.resolve())}), flush=True)


if __name__ == "__main__":
    main()
