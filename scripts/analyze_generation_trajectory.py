#!/usr/bin/env python3
"""Decode and measure what forms during a joint-flow sampling trajectory.

The noisy ODE state is useful for debugging but is not itself a clean latent.
The primary visualization therefore decodes the velocity field's local clean
endpoint estimate at selected solver steps. Aggregate measurements compare each
estimate with its own final generated sample in image, Fourier, latent-token,
and population-PCA spaces. This measures emergence/stabilization; it does not
by itself prove that an early component causally helps predict a later one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from live_evaluation import InceptionFeatures
from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.representations import (
    TOKENIZER_LATENTS,
    decode_representation,
    invert_latent_transform,
    representation_type,
)


RADIAL_BANDS = (
    ("r0-2", 0.0, 2.5),
    ("r3-4", 2.5, 4.5),
    ("r5-6", 4.5, 6.5),
    ("r7-8", 6.5, 8.5),
    ("r9-12", 8.5, 12.5),
    ("r13-16", 12.5, 16.5),
    ("r17+", 16.5, float("inf")),
)


def parse_named_paths(values: list[str], argument: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument} entries must have LABEL=PATH form")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label or label in result:
            raise ValueError(f"{argument} labels must be non-empty and unique")
        result[label] = Path(raw_path)
    return result


def radial_masks(size: int, device: torch.device) -> dict[str, torch.Tensor]:
    frequency = torch.fft.fftfreq(size, device=device) * size
    vertical, horizontal = torch.meshgrid(frequency, frequency, indexing="ij")
    radius = (vertical.square() + horizontal.square()).sqrt()
    masks = {
        label: (radius >= low) & (radius < high)
        for label, low, high in RADIAL_BANDS
    }
    coverage = torch.stack(list(masks.values())).sum(dim=0)
    if not bool((coverage == 1).all()):
        raise RuntimeError("radial bands must partition every FFT bin exactly once")
    return masks


def first_sustained_time(
    values: list[float], times: list[float], *, threshold: float, below: bool
) -> Optional[float]:
    if len(values) != len(times) or not values:
        raise ValueError("values and times must have equal nonzero length")
    good = [value <= threshold if below else value >= threshold for value in values]
    suffix_good = True
    first: Optional[float] = None
    for index in range(len(good) - 1, -1, -1):
        suffix_good = suffix_good and good[index]
        if suffix_good:
            first = times[index]
    return first


def physical_latent_layout(values: torch.Tensor, payload: dict) -> torch.Tensor:
    layout = payload.get("latent_layout")
    if not layout:
        return values
    if layout.get("type") != "consecutive_blocks":
        raise ValueError("unsupported latent layout")
    length = int(layout["physical_sequence_length"])
    width = int(layout["physical_token_dim"])
    if values.shape[1] * values.shape[2] != length * width:
        raise ValueError("sampled latent layout does not match tokenizer layout")
    return values.reshape(values.shape[0], length, width)


def decode_standardized(
    values: torch.Tensor,
    payload: dict,
    tokenizer,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = physical_latent_layout(values.float() * scale + mean, payload)
    token_scale = payload.get("token_scale")
    if token_scale is not None:
        raw = raw / token_scale.float().to(raw.device)[None, :, None]
    raw = invert_latent_transform(raw, payload)
    return decode_representation(raw, payload, tokenizer=tokenizer).float()


def fit_population_pca(
    cache_path: Path,
    mean: torch.Tensor,
    scale: torch.Tensor,
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    values = payload["train_latents"][:samples].float()
    flat = ((values - mean.cpu()) / scale.cpu()).flatten(1)
    coordinate_mean = flat.mean(dim=0)
    centered = flat - coordinate_mean
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    return coordinate_mean, eigenvalues[order].clamp_min(0), eigenvectors[:, order]


def pca_bands(dimensions: int) -> list[tuple[str, int, int]]:
    edges = [0, 8, 32, 128, 512, dimensions]
    edges = sorted(set(min(max(edge, 0), dimensions) for edge in edges))
    return [
        (f"pc{start + 1}-{end}", start, end)
        for start, end in zip(edges[:-1], edges[1:])
        if end > start
    ]


def empty_accumulator(tokens: int, radial: list[str], pca: list[str]) -> dict:
    return {
        "pixel_error": 0.0,
        "pixel_count": 0,
        "rgb_mean_error": 0.0,
        "rgb_mean_count": 0,
        "feature": {
            "error": 0.0,
            "predicted_power": 0.0,
            "final_power": 0.0,
            "cross": 0.0,
        },
        "latent_error": 0.0,
        "latent_signal": 0.0,
        "token_error": torch.zeros(tokens, dtype=torch.float64),
        "token_signal": torch.zeros(tokens, dtype=torch.float64),
        "radial": {
            label: {
                "error": 0.0,
                "predicted_power": 0.0,
                "final_power": 0.0,
                "cross": 0.0,
            }
            for label in radial
        },
        "pca": {
            label: {
                "error": 0.0,
                "predicted_power": 0.0,
                "final_power": 0.0,
                "cross": 0.0,
            }
            for label in pca
        },
    }


def accumulate_vector_comparison(store: dict, predicted: torch.Tensor, final: torch.Tensor) -> None:
    difference = predicted - final
    store["error"] += float(difference.double().square().sum())
    store["predicted_power"] += float(predicted.double().square().sum())
    store["final_power"] += float(final.double().square().sum())
    store["cross"] += float((predicted.double() * final.double()).sum())


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 1e-30:
        return None
    return numerator / denominator


def finalize_vector_comparison(store: dict) -> dict[str, Optional[float]]:
    denominator = math.sqrt(store["predicted_power"] * store["final_power"])
    return {
        "relative_error": safe_ratio(store["error"], store["final_power"]),
        "power_ratio": safe_ratio(store["predicted_power"], store["final_power"]),
        "correlation": safe_ratio(store["cross"], denominator),
    }


def tensor_to_image(values: torch.Tensor, upscale: int) -> Image.Image:
    array = (
        values.detach()
        .float()
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    return image.resize((image.width * upscale, image.height * upscale), Image.Resampling.NEAREST)


def contact_sheet(
    previews: dict[str, list[torch.Tensor]],
    times: list[float],
    output: Path,
    *,
    upscale: int = 2,
) -> None:
    labels = list(previews)
    examples = next(iter(previews.values()))[0].shape[0]
    cell = 32 * upscale
    gap = 3
    left = 150
    top = 35
    width = left + len(times) * (cell + gap) + gap
    height = top + len(labels) * examples * (cell + gap) + gap
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    regular = ImageFont.truetype(str(font_path), 12) if font_path.exists() else ImageFont.load_default()
    bold = ImageFont.truetype(str(bold_path), 12) if bold_path.exists() else regular
    for column, time in enumerate(times):
        draw.text((left + column * (cell + gap) + 14, 10), f"t={time:.1f}", font=bold, fill=(30, 33, 36))
    row = 0
    for label in labels:
        for example in range(examples):
            y = top + row * (cell + gap)
            draw.text((8, y + cell // 2 - 7), f"{label}  seed#{example}", font=regular, fill=(30, 33, 36))
            for column, snapshot in enumerate(previews[label]):
                image = tensor_to_image(snapshot[example], upscale)
                x = left + column * (cell + gap)
                canvas.paste(image, (x, y))
            row += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


PLOT_COLORS = (
    "#2A5FA0",
    "#C2692D",
    "#3D8C68",
    "#8A5FA0",
    "#B89B2F",
    "#497C86",
    "#A24D63",
)


def plot_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_line_chart(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    series: list[tuple[str, list[float], list[Optional[float]], str]],
    *,
    title: str,
    y_label: str,
    log_y: bool = False,
) -> None:
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    chart_left, chart_top = left + 62, top + 35
    chart_right, chart_bottom = right - 16, bottom - 48
    transformed = []
    finite_values = []
    for label, xs, ys, color in series:
        converted = []
        for value in ys:
            if value is None or not math.isfinite(value):
                converted.append(None)
            else:
                changed = math.log10(max(value, 1e-8)) if log_y else value
                converted.append(changed)
                finite_values.append(changed)
        transformed.append((label, xs, converted, color))
    if not finite_values:
        finite_values = [0.0, 1.0]
    y_min, y_max = min(finite_values), max(finite_values)
    if y_max <= y_min:
        y_min -= 0.5
        y_max += 0.5
    padding = 0.06 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    x_values = [value for _, xs, _, _ in transformed for value in xs]
    x_min, x_max = min(x_values), max(x_values)

    draw.text((left + 8, top + 7), title, font=plot_font(15, bold=True), fill="#202326")
    draw.text((left + 8, chart_top + 4), y_label, font=plot_font(10), fill="#55595D")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = chart_bottom - fraction * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill="#DDDEDC", width=1)
        value = y_min + fraction * (y_max - y_min)
        label = f"10^{value:.1f}" if log_y else f"{value:.2f}"
        draw.text((left + 7, y - 6), label, font=plot_font(9), fill="#6B6F73")
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="#AEB1AE", width=1)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = chart_left + fraction * (chart_right - chart_left)
        draw.text((x - 8, chart_bottom + 8), f"{x_min + fraction * (x_max - x_min):.2g}", font=plot_font(9), fill="#6B6F73")
    legend_x, legend_y = chart_left + 7, chart_top + 6
    for index, (label, xs, ys, color) in enumerate(transformed):
        points = []
        for x_value, y_value in zip(xs, ys):
            if y_value is None:
                if len(points) >= 2:
                    draw.line(points, fill=color, width=3)
                points = []
                continue
            x = chart_left + (x_value - x_min) / max(x_max - x_min, 1e-12) * (chart_right - chart_left)
            y = chart_bottom - (y_value - y_min) / (y_max - y_min) * (chart_bottom - chart_top)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.line((legend_x, legend_y + index * 14 + 5, legend_x + 17, legend_y + index * 14 + 5), fill=color, width=3)
        draw.text((legend_x + 22, legend_y + index * 14), label, font=plot_font(9), fill="#34373A")


def plot_summary(result: dict, output: Path) -> None:
    colors = {"v27": "#2A5FA0", "v34-common": "#3D8C68", "v34-soft25": "#C2692D"}
    canvas = Image.new("RGB", (1100, 800), "#FAFAF8")
    panels = [[], [], [], []]
    for label, run in result["runs"].items():
        color = colors.get(label, PLOT_COLORS[len(panels[0]) % len(PLOT_COLORS)])
        snapshots = run["snapshots"]
        times = [item["base_time"] for item in snapshots]
        panels[0].append((label, times, [item["image"]["inception_centered_correlation_to_final"] for item in snapshots], color))
        panels[1].append((label, times, [item["image"]["psnr_to_final_db"] for item in snapshots], color))
        panels[2].append((label, times, [max(item["latent"]["relative_error"], 1e-8) for item in snapshots], color))
        panels[3].append((label, times, [item["latent"]["token_fraction_relative_error_below_025"] for item in snapshots], color))
    boxes = ((0, 0, 550, 400), (550, 0, 1100, 400), (0, 400, 550, 800), (550, 400, 1100, 800))
    settings = (
        ("Inception-feature agreement", "centered correlation to final", False),
        ("Image agreement", "PSNR to final (dB)", False),
        ("Latent endpoint error", "relative MSE", True),
        ("Token stabilization", "fraction below 0.25", False),
    )
    for box, series, (title, y_label, log_y) in zip(boxes, panels, settings):
        draw_line_chart(canvas, box, series, title=title, y_label=y_label, log_y=log_y)
    canvas.save(output, optimize=True)


def plot_banded(result: dict, output: Path, field: str) -> None:
    runs = result["runs"]
    panel_width, height = 500, 430
    canvas = Image.new("RGB", (panel_width * len(runs), height), "#FAFAF8")
    for run_index, (label, run) in enumerate(runs.items()):
        snapshots = run["snapshots"]
        times = [item["base_time"] for item in snapshots]
        bands = snapshots[0][field]
        series = []
        for band_index, band in enumerate(bands):
            metric = "correlation" if field == "radial_frequency" else "relative_error"
            values = [item[field][band][metric] for item in snapshots]
            series.append((band, times, values, PLOT_COLORS[band_index % len(PLOT_COLORS)]))
        draw_line_chart(
            canvas,
            (run_index * panel_width, 0, (run_index + 1) * panel_width, height),
            series,
            title=label,
            y_label="correlation to final" if field == "radial_frequency" else "relative MSE to final",
            log_y=field == "pca_bands",
        )
    canvas.save(output, optimize=True)


def heat_color(value: float) -> tuple[int, int, int]:
    value = min(max((value + 2.0) / 3.0, 0.0), 1.0)
    anchors = ((34, 48, 92), (38, 122, 137), (94, 181, 108), (239, 221, 65))
    position = value * (len(anchors) - 1)
    index = min(int(position), len(anchors) - 2)
    fraction = position - index
    return tuple(round(anchors[index][channel] * (1 - fraction) + anchors[index + 1][channel] * fraction) for channel in range(3))


def plot_token_heatmaps(result: dict, output: Path) -> None:
    runs = result["runs"]
    panel_width, height = 420, 590
    canvas = Image.new("RGB", (panel_width * len(runs), height), "#FAFAF8")
    draw = ImageDraw.Draw(canvas)
    for run_index, (label, run) in enumerate(runs.items()):
        snapshots = run["snapshots"]
        times = [item["base_time"] for item in snapshots]
        matrix = np.asarray([item["latent"]["token_relative_error"] for item in snapshots]).T
        values = np.log10(np.maximum(matrix, 1e-5))
        pixels = np.empty((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                pixels[values.shape[0] - 1 - row, column] = heat_color(float(values[row, column]))
        heatmap = Image.fromarray(pixels, mode="RGB").resize((330, 500), Image.Resampling.NEAREST)
        left = run_index * panel_width + 65
        top = 42
        canvas.paste(heatmap, (left, top))
        draw.rectangle((left, top, left + 330, top + 500), outline="#AEB1AE", width=1)
        draw.text((run_index * panel_width + 12, 10), label, font=plot_font(15, bold=True), fill="#202326")
        draw.text((run_index * panel_width + 8, top + 235), "token index", font=plot_font(10), fill="#55595D")
        draw.text((left + 110, top + 520), "base sampling time", font=plot_font(10), fill="#55595D")
        for column, time in enumerate(times):
            if column % 2 == 0 or column == len(times) - 1:
                x = left + column / max(len(times) - 1, 1) * 330
                draw.text((x - 8, top + 503), f"{time:.1f}", font=plot_font(8), fill="#6B6F73")
    canvas.save(output, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH; repeat for each run")
    parser.add_argument("--cache", action="append", required=True, help="LABEL=PATH matching every checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--preview_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--snapshot_steps", default="0,5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--pca_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_inception", action="store_true")
    parser.add_argument(
        "--reference_cache",
        default="/workspace/AFIG/data/cifar10_test_inception.pt",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoints = parse_named_paths(args.checkpoint, "checkpoint")
    caches = parse_named_paths(args.cache, "cache")
    if checkpoints.keys() != caches.keys():
        raise ValueError("checkpoint and cache labels must match in the same order")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.preview_samples <= 0:
        raise ValueError("sample and batch counts must be positive")
    if args.preview_samples > args.batch_size or args.preview_samples > args.num_samples:
        raise ValueError("preview_samples cannot exceed batch_size or num_samples")
    snapshot_steps = tuple(int(value) for value in args.snapshot_steps.split(","))
    if tuple(sorted(snapshot_steps)) != snapshot_steps:
        raise ValueError("snapshot_steps must be ascending")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    extractor = None if args.skip_inception else InceptionFeatures(device)
    inception_mean = None
    if extractor is not None:
        reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
        inception_mean = reference["feature_mean"].float().to(device)
    result = {
        "definition": {
            "sampler": f"{args.steps}-step Heun, base time 0=noise and 1=data",
            "predicted_clean": "z_hat = z_state + (1-path_time) * v_theta(z_state,path_time), applied per token for rational paths",
            "comparison": "every intermediate estimate is compared with its own final generated endpoint",
            "radial_metric": "complex Fourier correlation and relative error preserve phase as well as power",
            "conditioning_limit": "early stabilization is necessary but not sufficient for useful conditioning; causal utility requires a context intervention",
            "paired_noise": "each run resets the same RNG seed and therefore starts from the same standardized Gaussian tensor; different learned coordinate systems do not imply paired semantic samples",
        },
        "settings": {
            "num_samples": args.num_samples,
            "preview_samples": args.preview_samples,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "snapshot_steps": list(snapshot_steps),
            "snapshot_times": [step / args.steps for step in snapshot_steps],
            "pca_samples": args.pca_samples,
            "seed": args.seed,
            "inception_features": not args.skip_inception,
            "inception_centering": None if args.skip_inception else str(Path(args.reference_cache).resolve()),
        },
        "runs": {},
    }
    predicted_previews: dict[str, list[torch.Tensor]] = {}
    state_previews: dict[str, list[torch.Tensor]] = {}
    pca_cache: dict[tuple[str, float, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for label, checkpoint_path in checkpoints.items():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("model_type") != "progressive_joint_rectified_flow":
            raise ValueError(f"{label}: trajectory analysis requires a joint-flow checkpoint")
        if representation_type(payload) != TOKENIZER_LATENTS:
            raise ValueError(f"{label}: only tokenizer latents are supported")
        config_values = dict(payload["model_config"])
        config_values.setdefault("qk_norm", "l2_temperature")
        model = JointRectifiedFlow(JointFlowConfig(**config_values))
        model.load_state_dict(payload["model"])
        model = model.to(device).eval().requires_grad_(False)
        tokenizer, tokenizer_payload = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
        if int(tokenizer_payload.get("step", -1)) != int(payload["tokenizer_step"]):
            raise ValueError(f"{label}: prior and tokenizer checkpoint steps differ")
        tokenizer = tokenizer.to(device).eval().requires_grad_(False)
        mean = payload["normalization"]["mean"].float().to(device)
        scale = payload["normalization"]["scale"].float().to(device)
        pca_key = (str(caches[label].resolve()), float(mean), float(scale))
        if pca_key not in pca_cache:
            pca_cache[pca_key] = fit_population_pca(caches[label], mean, scale, args.pca_samples)
        coordinate_mean_cpu, eigenvalues, basis_cpu = pca_cache[pca_key]
        coordinate_mean = coordinate_mean_cpu.to(device)
        basis = basis_cpu.to(device)
        bands = pca_bands(basis.shape[1])
        radial = radial_masks(32, device)
        accumulators = {
            step: empty_accumulator(config_values["sequence_length"], list(radial), [name for name, _, _ in bands])
            for step in snapshot_steps
        }
        predicted_previews[label] = []
        state_previews[label] = []
        generator = torch.Generator(device=device).manual_seed(args.seed)
        generated = 0
        while generated < args.num_samples:
            current = min(args.batch_size, args.num_samples - generated)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                final, trajectory = model.sample_trajectory(
                    current,
                    steps=args.steps,
                    solver="heun",
                    generator=generator,
                    snapshot_steps=snapshot_steps,
                )
                final_image = decode_standardized(final, payload, tokenizer, mean, scale)
            final_flat = final.float().flatten(1)
            final_centered = final_flat - coordinate_mean
            final_pca = final_centered @ basis
            final_fft = torch.fft.fft2(final_image.float(), norm="ortho")
            final_features = None
            if extractor is not None:
                assert inception_mean is not None
                final_features = extractor(final_image.add(1.0).div(2.0)) - inception_mean

            for snapshot in trajectory:
                step = int(snapshot["step"])
                predicted = snapshot["predicted_clean"].float()
                state = snapshot["state"].float()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    predicted_image = decode_standardized(predicted, payload, tokenizer, mean, scale)
                store = accumulators[step]
                image_difference = predicted_image - final_image
                store["pixel_error"] += float(image_difference.double().square().sum())
                store["pixel_count"] += image_difference.numel()
                predicted_rgb = predicted_image.mean(dim=(-2, -1))
                final_rgb = final_image.mean(dim=(-2, -1))
                store["rgb_mean_error"] += float((predicted_rgb - final_rgb).double().square().sum())
                store["rgb_mean_count"] += predicted_rgb.numel()
                if extractor is not None and final_features is not None:
                    assert inception_mean is not None
                    predicted_features = extractor(predicted_image.add(1.0).div(2.0)) - inception_mean
                    accumulate_vector_comparison(
                        store["feature"], predicted_features, final_features
                    )

                predicted_flat = predicted.flatten(1)
                latent_difference = predicted_flat - final_flat
                final_content = final_flat - coordinate_mean
                store["latent_error"] += float(latent_difference.double().square().sum())
                store["latent_signal"] += float(final_content.double().square().sum())
                token_difference = (predicted - final).double().square().sum(dim=(0, 2)).cpu()
                token_content = (final - coordinate_mean.reshape(1, *final.shape[1:])).double().square().sum(dim=(0, 2)).cpu()
                store["token_error"] += token_difference
                store["token_signal"] += token_content

                predicted_pca = (predicted_flat - coordinate_mean) @ basis
                for band_name, start, end in bands:
                    accumulate_vector_comparison(store["pca"][band_name], predicted_pca[:, start:end], final_pca[:, start:end])

                predicted_fft = torch.fft.fft2(predicted_image.float(), norm="ortho")
                for band_name, mask in radial.items():
                    predicted_band = torch.view_as_real(predicted_fft[:, :, mask]).flatten(1)
                    final_band = torch.view_as_real(final_fft[:, :, mask]).flatten(1)
                    accumulate_vector_comparison(store["radial"][band_name], predicted_band, final_band)

                if generated == 0:
                    predicted_previews[label].append(predicted_image[: args.preview_samples].cpu())
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        state_image = decode_standardized(state[: args.preview_samples], payload, tokenizer, mean, scale)
                    state_previews[label].append(state_image.cpu())
            generated += current
            print(json.dumps({"run": label, "generated": generated}), flush=True)

        snapshots = []
        for step in snapshot_steps:
            store = accumulators[step]
            pixel_mse = store["pixel_error"] / store["pixel_count"]
            token_relative = torch.where(
                store["token_signal"] > 1e-30,
                store["token_error"] / store["token_signal"],
                torch.full_like(store["token_signal"], float("nan")),
            )
            base_time = step / args.steps
            path_time = model.path_time(torch.tensor([base_time], device=device)).flatten().float().cpu()
            feature_metrics = (
                {"relative_error": None, "power_ratio": None, "correlation": None}
                if extractor is None
                else finalize_vector_comparison(store["feature"])
            )
            snapshots.append(
                {
                    "step": step,
                    "base_time": base_time,
                    "path_time": {
                        "minimum": float(path_time.min()),
                        "median": float(path_time.median()),
                        "maximum": float(path_time.max()),
                    },
                    "image": {
                        "mse_to_final": pixel_mse,
                        "psnr_to_final_db": None if pixel_mse <= 1e-30 else 10.0 * math.log10(4.0 / pixel_mse),
                        "rgb_mean_rmse_to_final": math.sqrt(store["rgb_mean_error"] / store["rgb_mean_count"]),
                        "inception_centered_correlation_to_final": feature_metrics["correlation"],
                        "inception_relative_error_to_final": feature_metrics["relative_error"],
                        "inception_power_ratio_to_final": feature_metrics["power_ratio"],
                    },
                    "radial_frequency": {
                        name: finalize_vector_comparison(values)
                        for name, values in store["radial"].items()
                    },
                    "latent": {
                        "relative_error": store["latent_error"] / max(store["latent_signal"], 1e-30),
                        "token_relative_error": [float(value) for value in token_relative],
                        "token_fraction_relative_error_below_025": float((token_relative <= 0.25).double().mean()),
                    },
                    "pca_bands": {
                        name: finalize_vector_comparison(values)
                        for name, values in store["pca"].items()
                    },
                }
            )
        times = [item["base_time"] for item in snapshots]
        token_matrix = np.asarray([item["latent"]["token_relative_error"] for item in snapshots])
        token_settling = [
            first_sustained_time(token_matrix[:, index].tolist(), times, threshold=0.25, below=True)
            for index in range(token_matrix.shape[1])
        ]
        valid_settling = np.asarray([time if time is not None else 1.1 for time in token_settling])
        index_correlation = None
        if valid_settling.std() > 0:
            index_correlation = float(np.corrcoef(np.arange(len(valid_settling)), valid_settling)[0, 1])
        run_result = {
            "checkpoint": str(checkpoint_path.resolve()),
            "cache": str(caches[label].resolve()),
            "checkpoint_step": int(payload["step"]),
            "model_config": config_values,
            "pca_eigenvalue_share": {
                "top8": float(eigenvalues[:8].sum() / eigenvalues.sum()),
                "top32": float(eigenvalues[:32].sum() / eigenvalues.sum()),
                "top128": float(eigenvalues[:128].sum() / eigenvalues.sum()),
            },
            "snapshots": snapshots,
            "settling": {
                "inception_centered_correlation_090": first_sustained_time(
                    [item["image"]["inception_centered_correlation_to_final"] if item["image"]["inception_centered_correlation_to_final"] is not None else -1.0 for item in snapshots],
                    times,
                    threshold=0.9,
                    below=False,
                ),
                "radial_correlation_090": {
                    band: first_sustained_time(
                        [item["radial_frequency"][band]["correlation"] or -1.0 for item in snapshots],
                        times,
                        threshold=0.9,
                        below=False,
                    )
                    for band in radial
                },
                "pca_relative_error_025": {
                    band: first_sustained_time(
                        [item["pca_bands"][band]["relative_error"] or 0.0 for item in snapshots],
                        times,
                        threshold=0.25,
                        below=True,
                    )
                    for band, _, _ in bands
                },
                "token_relative_error_025": token_settling,
                "token_index_vs_settling_time_correlation": index_correlation,
            },
        }
        result["runs"][label] = run_result
        del model, tokenizer, basis
        if device.type == "cuda":
            torch.cuda.empty_cache()

    times = result["settings"]["snapshot_times"]
    contact_sheet(predicted_previews, times, output / "predicted_clean_contact_sheet.png")
    contact_sheet(state_previews, times, output / "noisy_state_contact_sheet.png")
    plot_summary(result, output / "trajectory_summary.png")
    plot_banded(result, output / "frequency_emergence.png", "radial_frequency")
    plot_banded(result, output / "pca_emergence.png", "pca_bands")
    plot_token_heatmaps(result, output / "token_emergence.png")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": str(output.resolve())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
