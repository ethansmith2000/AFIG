#!/usr/bin/env python3
"""Measure v27 recovery from controlled noise against known clean latents."""

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

from live_evaluation import InceptionFeatures
from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.latent_geometry import (
    axis_coefficients,
    first_sustained_below,
    snr1_crossing,
)
from scripts.analyze_generation_trajectory import (
    PLOT_COLORS,
    contact_sheet,
    decode_standardized,
    draw_line_chart,
    radial_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_samples", type=int, default=2048)
    parser.add_argument("--decoded_samples", type=int, default=256)
    parser.add_argument("--preview_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--times", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_inception", action="store_true")
    return parser.parse_args()


def parse_times(value: str) -> list[float]:
    times = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not times or times != sorted(times) or any(time <= 0 or time >= 1 for time in times):
        raise ValueError("times must be increasing and lie strictly inside (0,1)")
    return times


def _basis_for_axis(payload: dict, axis: str, device: torch.device) -> torch.Tensor | None:
    if axis == "per_token":
        return None
    return payload[f"{axis}_eigenvectors"].to(device)


def _ordered_power(payload: dict, axis: str, device: torch.device) -> torch.Tensor:
    if axis == "per_token":
        power = payload["token_power"].to(device)
        order = payload["token_order"].to(device)
        return power[order]
    return payload[f"{axis}_eigenvalues"].to(device)


def _select_band(values: torch.Tensor, axis: str, lo: int, hi: int) -> torch.Tensor:
    if axis in {"flattened", "sequence", "per_token"}:
        return values[:, lo:hi]
    if axis == "channel":
        return values[:, :, lo:hi]
    raise ValueError(f"unsupported axis: {axis}")


def _comparison(predicted: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    dtype = torch.complex128 if predicted.is_complex() or clean.is_complex() else torch.float64
    predicted = predicted.to(dtype).flatten()
    clean = clean.to(dtype).flatten()
    error = (predicted - clean).abs().square().sum()
    clean_power = clean.abs().square().sum().clamp_min(1e-30)
    predicted_power = predicted.abs().square().sum()
    correlation = (predicted.conj() * clean).sum().real / (
        predicted_power.sqrt() * clean_power.sqrt()
    ).clamp_min(1e-30)
    return {
        "relative_mse": float(error / clean_power),
        "correlation": float(correlation),
        "power_ratio": float(predicted_power / clean_power),
    }


def _decode_batches(
    standardized: torch.Tensor,
    cache: dict,
    tokenizer,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(standardized), batch_size):
        values = standardized[start : start + batch_size].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            images = decode_standardized(values, cache, tokenizer, mean, scale)
        output.append(images.float().cpu())
    return torch.cat(output)


def _feature_batches(
    images: torch.Tensor,
    extractor: InceptionFeatures,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    output = []
    for start in range(0, len(images), batch_size):
        values = (images[start : start + batch_size].to(device) + 1.0) / 2.0
        output.append(extractor(values).cpu())
    return torch.cat(output)


def _decoded_metrics(
    predicted: torch.Tensor,
    clean: torch.Tensor,
    predicted_features: torch.Tensor | None,
    clean_features: torch.Tensor | None,
    feature_mean: torch.Tensor | None,
) -> dict[str, object]:
    difference = predicted.float() - clean.float()
    mse = float(difference.double().square().mean())
    masks = radial_masks(predicted.shape[-1], torch.device("cpu"))
    predicted_fft = torch.fft.fft2(predicted.float(), norm="ortho")
    clean_fft = torch.fft.fft2(clean.float(), norm="ortho")
    radial = {}
    for label, mask in masks.items():
        radial[label] = _comparison(predicted_fft[:, :, mask], clean_fft[:, :, mask])
    feature = None
    if predicted_features is not None and clean_features is not None:
        center = 0.0 if feature_mean is None else feature_mean
        feature = _comparison(predicted_features - center, clean_features - center)
    return {
        "mse": mse,
        "psnr_db": None if mse <= 1e-30 else 10.0 * math.log10(4.0 / mse),
        "rgb_mean_rmse": float(
            difference.double().mean(dim=(2, 3)).square().mean().sqrt()
        ),
        "inception": feature,
        "radial_frequency": radial,
    }


def _plot_axis_curves(result: dict, output: Path) -> None:
    axes = ["channel", "sequence", "flattened", "per_token"]
    canvas = Image.new("RGB", (1100, 800), "#FAFAF8")
    boxes = ((0, 0, 550, 400), (550, 0, 1100, 400), (0, 400, 550, 800), (550, 400, 1100, 800))
    for axis, box in zip(axes, boxes):
        series = []
        for index, band in enumerate(result["axes"][axis]["bands"]):
            values = [snapshot["relative_mse"] for snapshot in band["snapshots"]]
            series.append((band["label"], result["times"], values, PLOT_COLORS[index % len(PLOT_COLORS)]))
        draw_line_chart(
            canvas,
            box,
            series,
            title=f"{axis}: known-clean recovery",
            y_label="relative MSE",
            log_y=True,
        )
    canvas.save(output, optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    times = parse_times(args.times)
    if min(args.eval_samples, args.decoded_samples, args.preview_samples, args.batch_size) <= 0:
        raise ValueError("sample and batch counts must be positive")
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["model_config"])
    config_values.setdefault("qk_norm", "l2_temperature")
    model = JointRectifiedFlow(JointFlowConfig(**config_values))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval().requires_grad_(False)
    if model.config.time_parameterization != "global":
        raise ValueError("known-clean audit is frozen to the selected common-time prior")

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    basis_payload = torch.load(args.basis, map_location="cpu", weights_only=False)
    mean = checkpoint["normalization"]["mean"].float().to(device)
    scale = checkpoint["normalization"]["scale"].float().to(device)
    clean = cache["test_latents"][: args.eval_samples].to(device).float()
    clean = (clean - mean) / scale
    element_mean = basis_payload["element_mean"].to(device)
    clean_centered = clean - element_mean
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(clean.shape, device=device, generator=generator)

    tokenizer, _ = load_tokenizer_checkpoint(cache["tokenizer_checkpoint"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    decoded_count = min(args.decoded_samples, len(clean))
    clean_images = _decode_batches(
        clean[:decoded_count], cache, tokenizer, mean, scale, args.batch_size, device
    )
    extractor = None if args.skip_inception else InceptionFeatures(device)
    clean_features = None
    feature_mean = None
    if extractor is not None:
        clean_features = _feature_batches(clean_images, extractor, args.batch_size, device)
        reference = torch.load(
            "/workspace/AFIG/data/cifar10_test_inception.pt",
            map_location="cpu",
            weights_only=False,
        )
        feature_mean = reference["feature_mean"].float()

    axes = {}
    for axis, edges in basis_payload["axis_band_edges"].items():
        power = _ordered_power(basis_payload, axis, device)
        bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            band_power = power[lo:hi].mean()
            bands.append(
                {
                    "label": f"rank{lo + 1}-{hi}",
                    "range_zero_based_half_open": [lo, hi],
                    "population_power_per_mode": float(band_power),
                    "population_snr1_t": float(snr1_crossing(band_power)),
                    "snapshots": [],
                }
            )
        axes[axis] = {"bands": bands}

    decoded_snapshots = []
    preview_snapshots = []
    for time in times:
        predicted_chunks = []
        for start in range(0, len(clean), args.batch_size):
            clean_batch = clean[start : start + args.batch_size]
            noise_batch = noise[start : start + args.batch_size]
            noisy = (1.0 - time) * noise_batch + time * clean_batch
            time_values = torch.full((len(clean_batch),), time, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                velocity = model.predict_velocity(noisy, time_values)
            predicted_chunks.append((noisy + (1.0 - time) * velocity.float()))
        predicted = torch.cat(predicted_chunks)
        predicted_centered = predicted - element_mean
        for axis, axis_result in axes.items():
            basis = _basis_for_axis(basis_payload, axis, device)
            token_order = basis_payload["token_order"].to(device) if axis == "per_token" else None
            predicted_coefficients = axis_coefficients(
                predicted_centered, axis, basis=basis, token_order=token_order
            )
            clean_coefficients = axis_coefficients(
                clean_centered, axis, basis=basis, token_order=token_order
            )
            for band in axis_result["bands"]:
                lo, hi = band["range_zero_based_half_open"]
                metrics = _comparison(
                    _select_band(predicted_coefficients, axis, lo, hi),
                    _select_band(clean_coefficients, axis, lo, hi),
                )
                band["snapshots"].append({"time": time, **metrics})

        predicted_images = _decode_batches(
            predicted[:decoded_count], cache, tokenizer, mean, scale, args.batch_size, device
        )
        predicted_features = None
        if extractor is not None:
            predicted_features = _feature_batches(
                predicted_images, extractor, args.batch_size, device
            )
        decoded_snapshots.append(
            {
                "time": time,
                **_decoded_metrics(
                    predicted_images,
                    clean_images,
                    predicted_features,
                    clean_features,
                    feature_mean,
                ),
            }
        )
        preview_snapshots.append(predicted_images[: args.preview_samples])
        print(json.dumps({"completed_time": time}), flush=True)

    for axis_result in axes.values():
        for band in axis_result["bands"]:
            band["empirical_relative_mse_025_settling_t"] = first_sustained_below(
                [snapshot["relative_mse"] for snapshot in band["snapshots"]],
                times,
                0.25,
            )

    result = {
        "definition": {
            "data": "held-out real v27 latents with known clean targets",
            "path": "z_t=(1-t) noise + t clean with one fixed noise realization per example across times",
            "predicted_clean": "z_t + (1-t) v_theta(z_t,t)",
            "axis_error": "population-centered predicted-clean relative MSE against clean coefficients",
            "settling": "first measured time from which relative MSE remains <=0.25",
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "basis": str(Path(args.basis).resolve()),
        "eval_samples": len(clean),
        "decoded_samples": decoded_count,
        "seed": args.seed,
        "times": times,
        "axes": axes,
        "decoded": decoded_snapshots,
    }
    result_path = output / "known_clean_denoising.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _plot_axis_curves(result, output / "known_clean_axis_recovery.png")
    previews = {"v27 known-clean": preview_snapshots + [clean_images[: args.preview_samples]]}
    contact_sheet(previews, times + [1.0], output / "known_clean_contact_sheet.png")
    print(json.dumps({"complete": str(result_path.resolve())}), flush=True)


if __name__ == "__main__":
    main()
