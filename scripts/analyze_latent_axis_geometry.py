#!/usr/bin/env python3
"""Audit latent spectra, sample consistency, separability, and decoded roles."""

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
from PIL import Image, ImageDraw

from live_evaluation import InceptionFeatures
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.latent_geometry import (
    axis_mode_energy,
    fit_axis_geometry,
    kronecker_approximation,
    summarize_ordered_energy,
    swap_axis_band,
)
from scripts.analyze_generation_trajectory import (
    PLOT_COLORS,
    decode_standardized,
    draw_line_chart,
    plot_font,
    radial_masks,
    tensor_to_image,
)


AXIS_EDGES = {
    "channel": [0, 1, 2, 4, 8, 16],
    "sequence": [0, 2, 4, 8, 16, 32, 64],
    "flattened": [0, 8, 32, 128, 512, 1024],
    "per_token": [0, 8, 16, 24, 32, 40, 48, 56, 64],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prior_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--basis_output", default=None)
    parser.add_argument("--fit_samples", type=int, default=25000)
    parser.add_argument("--eval_samples", type=int, default=10000)
    parser.add_argument("--role_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--preview_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_inception", action="store_true")
    return parser.parse_args()


def _basis_for_axis(geometry: dict[str, object], axis: str) -> torch.Tensor | None:
    if axis == "per_token":
        return None
    return geometry[f"{axis}_eigenvectors"]  # type: ignore[return-value]


def _power_for_axis(geometry: dict[str, object], axis: str) -> torch.Tensor:
    if axis == "per_token":
        power = geometry["token_power"]
        order = geometry["token_order"]
        assert isinstance(power, torch.Tensor) and isinstance(order, torch.Tensor)
        return power[order]
    value = geometry[f"{axis}_eigenvalues"]
    assert isinstance(value, torch.Tensor)
    return value


def _axis_indices(geometry: dict[str, object], axis: str, lo: int, hi: int) -> torch.Tensor:
    device = _power_for_axis(geometry, axis).device
    if axis == "per_token":
        order = geometry["token_order"]
        assert isinstance(order, torch.Tensor)
        return order[lo:hi]
    return torch.arange(lo, hi, device=device)


def _decode_batches(
    standardized: torch.Tensor,
    cache: dict,
    tokenizer,
    mean: torch.Tensor,
    scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    images = []
    device_mean = mean.to(device)
    device_scale = scale.to(device)
    for start in range(0, standardized.shape[0], batch_size):
        values = standardized[start : start + batch_size].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            decoded = decode_standardized(
                values, cache, tokenizer, device_mean, device_scale
            )
        images.append(decoded.float().cpu())
    return torch.cat(images)


def _feature_batches(
    images: torch.Tensor,
    extractor: InceptionFeatures,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features = []
    for start in range(0, images.shape[0], batch_size):
        batch = (images[start : start + batch_size].to(device) + 1.0) / 2.0
        features.append(extractor(batch).cpu())
    return torch.cat(features)


def _role_metrics(
    baseline: torch.Tensor,
    changed: torch.Tensor,
    baseline_features: torch.Tensor | None,
    changed_features: torch.Tensor | None,
) -> dict[str, object]:
    delta = changed.float() - baseline.float()
    spectrum = torch.fft.fft2(delta, norm="ortho")
    power = spectrum.abs().square()
    masks = radial_masks(delta.shape[-1], torch.device("cpu"))
    band_power = {
        label: float(power[:, :, mask].double().sum())
        for label, mask in masks.items()
    }
    total = sum(band_power.values())
    feature_rms = None
    if baseline_features is not None and changed_features is not None:
        feature_rms = float(
            (changed_features.float() - baseline_features.float())
            .square()
            .mean()
            .sqrt()
        )
    return {
        "pixel_delta_rms": float(delta.double().square().mean().sqrt()),
        "rgb_mean_delta_rms": float(
            delta.double().mean(dim=(2, 3)).square().mean().sqrt()
        ),
        "inception_feature_delta_rms": feature_rms,
        "radial_fft_power_fraction": {
            label: value / max(total, 1e-30) for label, value in band_power.items()
        },
    }


def _spectrum_plots(result: dict, output: Path) -> None:
    axes = ["channel", "sequence", "flattened", "per_token"]
    canvas = Image.new("RGB", (1100, 800), "#FAFAF8")
    boxes = ((0, 0, 550, 400), (550, 0, 1100, 400), (0, 400, 550, 800), (550, 400, 1100, 800))
    for axis, box in zip(axes, boxes):
        power = result["axes"][axis]["population_power"]
        relative = [max(value / max(power[0], 1e-30), 1e-12) for value in power]
        draw_line_chart(
            canvas,
            box,
            [(axis, list(range(1, len(power) + 1)), relative, PLOT_COLORS[0])],
            title=f"{axis} population spectrum",
            y_label="power / strongest",
            log_y=True,
        )
    canvas.save(output, optimize=True)

    crossing_canvas = Image.new("RGB", (1100, 800), "#FAFAF8")
    for axis, box in zip(axes, boxes):
        crossing = result["axes"][axis]["population_snr1_t"]
        draw_line_chart(
            crossing_canvas,
            box,
            [(axis, list(range(1, len(crossing) + 1)), crossing, PLOT_COLORS[1])],
            title=f"{axis} magnitude-derived clock",
            y_label="analytic SNR=1 time",
        )
    crossing_canvas.save(output.with_name("axis_snr1_crossings.png"), optimize=True)


def _role_heatmap(role_results: dict[str, dict], output: Path) -> None:
    radial = list(radial_masks(32, torch.device("cpu")))
    rows = [(axis, band, values) for axis, bands in role_results.items() for band, values in bands.items()]
    cell_w, cell_h = 86, 24
    left, top = 245, 42
    canvas = Image.new("RGB", (left + cell_w * len(radial) + 18, top + cell_h * len(rows) + 25), "#FAFAF8")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Decoded change: radial FFT power fraction", font=plot_font(15, bold=True), fill="#202326")
    for column, label in enumerate(radial):
        draw.text((left + column * cell_w + 8, 24), label, font=plot_font(9), fill="#55595D")
    for row, (axis, band, values) in enumerate(rows):
        y = top + row * cell_h
        draw.text((8, y + 5), f"{axis}: {band}", font=plot_font(9), fill="#34373A")
        fractions = values["radial_fft_power_fraction"]
        maximum = max(fractions.values())
        for column, label in enumerate(radial):
            fraction = fractions[label]
            intensity = 0.0 if maximum <= 0 else math.sqrt(fraction / maximum)
            color = (
                round(245 - 170 * intensity),
                round(247 - 105 * intensity),
                round(244 - 35 * intensity),
            )
            x = left + column * cell_w
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color)
            draw.text((x + 8, y + 5), f"{100*fraction:.1f}%", font=plot_font(8), fill="#202326")
    canvas.save(output, optimize=True)


def _role_contact_sheet(
    baseline: torch.Tensor,
    previews: dict[str, list[tuple[str, torch.Tensor]]],
    output: Path,
) -> None:
    examples = baseline.shape[0]
    max_bands = max(len(items) for items in previews.values())
    cell, gap, left, top = 64, 3, 145, 34
    columns = max_bands + 1
    rows = len(previews) * examples
    canvas = Image.new("RGB", (left + columns * (cell + gap), top + rows * (cell + gap)), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    row = 0
    for axis, items in previews.items():
        for example in range(examples):
            y = top + row * (cell + gap)
            draw.text((7, y + 25), f"{axis} #{example}", font=plot_font(10), fill="#303336")
            canvas.paste(tensor_to_image(baseline[example], 2), (left, y))
            draw.rectangle((left, y, left + 27, y + 10), fill="#F4F1E8")
            draw.text((left + 2, y), "base", font=plot_font(7), fill="#303336")
            for column, (label, images) in enumerate(items, start=1):
                x = left + column * (cell + gap)
                canvas.paste(tensor_to_image(images[example], 2), (x, y))
                short = label.replace("power-rank", "p").replace("rank", "r")
                draw.rectangle((x, y, x + min(62, 5 + 4 * len(short)), y + 10), fill="#F4F1E8")
                draw.text((x + 2, y), short, font=plot_font(7), fill="#303336")
            row += 1
    canvas.save(output, optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(args.fit_samples, args.eval_samples, args.role_samples, args.batch_size) <= 0:
        raise ValueError("sample and batch counts must be positive")
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    mean = checkpoint["normalization"]["mean"].float()
    scale = checkpoint["normalization"]["scale"].float()
    train = cache["train_latents"][: args.fit_samples].to(device).float()
    train = (train - mean.to(device)) / scale.to(device)
    geometry = fit_axis_geometry(train)
    del train

    test = cache["test_latents"][: args.eval_samples].to(device).float()
    test = (test - mean.to(device)) / scale.to(device)
    element_mean = geometry["element_mean"]
    assert isinstance(element_mean, torch.Tensor)
    centered = test - element_mean
    axes: dict[str, dict] = {}
    for axis, edges in AXIS_EDGES.items():
        basis = _basis_for_axis(geometry, axis)
        token_order = geometry["token_order"] if axis == "per_token" else None
        assert token_order is None or isinstance(token_order, torch.Tensor)
        energies = axis_mode_energy(
            centered, axis, basis=basis, token_order=token_order
        )
        axes[axis] = summarize_ordered_energy(
            energies, _power_for_axis(geometry, axis), edges
        )
        if axis == "per_token":
            axes[axis]["population_rank_to_native_token"] = [
                int(value) for value in geometry["token_order"]  # type: ignore[index]
            ]

    factorization = kronecker_approximation(
        geometry["flattened_covariance"],  # type: ignore[arg-type]
        geometry["sequence_covariance"],  # type: ignore[arg-type]
        geometry["channel_covariance"],  # type: ignore[arg-type]
    )
    result = {
        "definition": {
            "centering": "elementwise training-population mean after checkpoint tensor-wide normalization",
            "sequence_covariance": "E[Z Z^T] / channel_count",
            "channel_covariance": "E[Z^T Z] / token_count",
            "flattened_covariance": "E[vec(Z) vec(Z)^T]",
            "per_token": "mean centered feature power per native token",
            "sample_consistency": "mode power is aggregated over the complementary matrix axis before per-sample comparisons",
        },
        "cache": str(cache_path.resolve()),
        "prior_checkpoint": str(Path(args.prior_checkpoint).resolve()),
        "fit_samples": min(args.fit_samples, len(cache["train_latents"])),
        "eval_samples": len(test),
        "axis_band_edges": AXIS_EDGES,
        "axes": axes,
        "factorized_covariance": factorization,
        "decoded_role_swaps": {},
    }

    basis_payload = {
        "element_mean": element_mean.detach().cpu(),
        "normalization_mean": mean.cpu(),
        "normalization_scale": scale.cpu(),
        "axis_band_edges": AXIS_EDGES,
        "token_order": geometry["token_order"].detach().cpu(),  # type: ignore[union-attr]
    }
    for axis in ("channel", "sequence", "flattened"):
        basis_payload[f"{axis}_eigenvalues"] = geometry[f"{axis}_eigenvalues"].detach().cpu()  # type: ignore[union-attr]
        basis_payload[f"{axis}_eigenvectors"] = geometry[f"{axis}_eigenvectors"].detach().cpu()  # type: ignore[union-attr]
    basis_payload["token_power"] = geometry["token_power"].detach().cpu()  # type: ignore[union-attr]
    if args.basis_output:
        basis_path = Path(args.basis_output)
        basis_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(basis_payload, basis_path)

    role_count = min(args.role_samples, len(test))
    role_values = test[:role_count]
    role_centered = centered[:role_count]
    tokenizer_path = Path(cache["tokenizer_checkpoint"])
    tokenizer, _ = load_tokenizer_checkpoint(tokenizer_path)
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    baseline = _decode_batches(
        role_values, cache, tokenizer, mean, scale, args.batch_size, device
    )
    extractor = None if args.skip_inception else InceptionFeatures(device)
    baseline_features = None
    if extractor is not None:
        baseline_features = _feature_batches(
            baseline, extractor, args.batch_size, device
        )
    permutation = torch.roll(torch.arange(role_count, device=device), 1)
    preview_count = min(args.preview_samples, role_count)
    previews: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for axis, edges in AXIS_EDGES.items():
        previews[axis] = []
        axis_results = {}
        basis = _basis_for_axis(geometry, axis)
        for lo, hi in zip(edges[:-1], edges[1:]):
            indices = _axis_indices(geometry, axis, lo, hi)
            changed_centered = swap_axis_band(
                role_centered,
                axis,
                indices,
                permutation,
                basis=basis,
            )
            changed_values = changed_centered + element_mean
            changed = _decode_batches(
                changed_values, cache, tokenizer, mean, scale, args.batch_size, device
            )
            changed_features = None
            if extractor is not None:
                changed_features = _feature_batches(
                    changed, extractor, args.batch_size, device
                )
            label = f"rank{lo + 1}-{hi}"
            if axis == "per_token":
                native = [int(value) for value in indices]
                label = f"power-rank{lo + 1}-{hi}"
            else:
                native = None
            axis_results[label] = {
                "mode_range_zero_based_half_open": [lo, hi],
                "native_token_indices": native,
                **_role_metrics(
                    baseline, changed, baseline_features, changed_features
                ),
            }
            previews[axis].append((label, changed[:preview_count]))
            print(json.dumps({"decoded_role": axis, "range": [lo, hi]}), flush=True)
        result["decoded_role_swaps"][axis] = axis_results

    result_path = output / "geometry.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _spectrum_plots(result, output / "axis_spectra.png")
    _role_heatmap(result["decoded_role_swaps"], output / "decoded_role_frequency.png")
    _role_contact_sheet(
        baseline[:preview_count], previews, output / "decoded_role_contact_sheet.png"
    )
    print(json.dumps({"complete": str(result_path.resolve())}), flush=True)


if __name__ == "__main__":
    main()
