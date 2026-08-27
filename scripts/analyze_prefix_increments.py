#!/usr/bin/env python3
"""Measure what each newly available latent token changes in decoded image space.

Figure contract
---------------
Question: Do successive latent tokens add progressively finer, useful image
information, or do they mainly rewrite content introduced by earlier tokens?
Evidence: every one-token prefix increment on the first N held-out CIFAR-10
examples for the frozen progressive and unordered 64x16 tokenizers. No examples
are selected by outcome. Surface: reproducible static PNGs plus machine-readable
JSON for the autoencoder-program report. Charts: token-by-frequency heatmaps and
ordered token traces. Palette: one blue root for progressive, one orange root for
unordered, neutral references, and line style/position in addition to color.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


MODEL_SPECS = (
    (
        "progressive",
        "Progressive prefix-trained",
        "tokenizer_runs/v5-vae-kl1e4-s1/latents_final_original_flip.pt",
        "#2A5FA0",
    ),
    (
        "unordered",
        "Unordered full-only",
        "tokenizer_runs/v8-unordered-vae-s1/latents_final_original_flip.pt",
        "#C2692D",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="reports/2026-08-26_autoencoder_program/prefix_increment_audit",
    )
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--num_examples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--orientation_bins", type=int, default=8)
    parser.add_argument("--contact_examples", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cifar_targets(root: str, count: int) -> torch.Tensor:
    dataset = torchvision.datasets.CIFAR10(
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
    return torch.stack([dataset[index][0] for index in range(count)])


def frequency_maps(size: int, orientation_bins: int, device: torch.device) -> dict:
    coordinates = torch.arange(size, device=device) - size // 2
    vertical, horizontal = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = (horizontal.square() + vertical.square()).float().sqrt()
    radial_index = radius.round().long()
    angle = torch.remainder(torch.atan2(vertical.float(), horizontal.float()), math.pi)
    orientation_index = torch.floor(angle / math.pi * orientation_bins).long()
    orientation_index.clamp_(max=orientation_bins - 1)
    return {
        "radius": radius,
        "radial_index": radial_index,
        "radial_bins": int(radial_index.max()) + 1,
        "orientation_index": orientation_index,
    }


def bin_power(power: torch.Tensor, index: torch.Tensor, bins: int) -> torch.Tensor:
    """Average a [B,C,H,W] power map into bins, preserving sample dimension."""

    sample_power = power.mean(dim=1).flatten(1)
    flat_index = index.flatten()
    output = torch.zeros(sample_power.shape[0], bins, device=power.device)
    output.scatter_add_(1, flat_index[None].expand(sample_power.shape[0], -1), sample_power)
    counts = torch.bincount(flat_index, minlength=bins).clamp_min(1)
    return output / counts[None]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(values: np.ndarray) -> float:
    indices = np.arange(len(values), dtype=np.float64)
    ranks = rankdata(values)
    if ranks.std() == 0:
        return 0.0
    return float(np.corrcoef(indices, ranks)[0, 1])


@torch.no_grad()
def analyze_model(
    key: str,
    label: str,
    cache_path: Path,
    color: str,
    targets: torch.Tensor,
    args: argparse.Namespace,
    maps: dict,
) -> tuple[dict, torch.Tensor]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    latents = cache["test_latents"][: args.num_examples]
    if tuple(latents.shape[1:]) != (64, 16):
        raise ValueError(f"{cache_path} must contain [N,64,16] test latents")
    tokenizer, tokenizer_payload = load_tokenizer_checkpoint(
        cache["tokenizer_checkpoint"]
    )
    if int(tokenizer_payload.get("step", -1)) != int(cache["tokenizer_step"]):
        raise ValueError("cache and tokenizer checkpoint steps differ")
    device = torch.device(args.device)
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    sequence_length = latents.shape[1]
    radial_bins = int(maps["radial_bins"])
    orientation_bins = args.orientation_bins

    mse_sum = torch.zeros(sequence_length + 1, dtype=torch.float64)
    increment_square_sum = torch.zeros(sequence_length, dtype=torch.float64)
    increment_count = 0
    improvement_count = torch.zeros(sequence_length, dtype=torch.float64)
    residual_alignment_sum = torch.zeros(sequence_length, dtype=torch.float64)
    radial_sum = torch.zeros(sequence_length, radial_bins, dtype=torch.float64)
    orientation_sum = torch.zeros(
        sequence_length, orientation_bins, dtype=torch.float64
    )
    examples_seen = 0
    path_ratio_sum = 0.0
    contact_deltas = None

    for start in range(0, args.num_examples, args.batch_size):
        stop = min(start + args.batch_size, args.num_examples)
        clean = latents[start:stop].float().to(device)
        target = targets[start:stop].to(device)
        batch = clean.shape[0]
        masked = torch.zeros_like(clean)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            previous = tokenizer.decode(masked).float()
        previous_error = (previous - target).square().flatten(1).mean(dim=1)
        mse_sum[0] += previous_error.double().sum().cpu()
        path_length = torch.zeros(batch, device=device)
        initial = previous
        batch_contacts = []

        for token_index in range(sequence_length):
            masked[:, token_index] = clean[:, token_index]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                current = tokenizer.decode(masked).float()
            current_error = (current - target).square().flatten(1).mean(dim=1)
            delta = current - previous
            delta_flat = delta.flatten(1)
            residual_flat = (target - previous).flatten(1)
            delta_norm = delta_flat.norm(dim=1)
            residual_norm = residual_flat.norm(dim=1)
            alignment = (delta_flat * residual_flat).sum(dim=1) / (
                delta_norm * residual_norm
            ).clamp_min(1e-12)

            spectrum = torch.fft.fftshift(
                torch.fft.fft2(delta, norm="ortho"), dim=(-2, -1)
            ).abs().square()
            radial = bin_power(
                spectrum, maps["radial_index"], radial_bins
            ).mean(dim=0)
            non_dc = spectrum.clone()
            non_dc[..., target.shape[-2] // 2, target.shape[-1] // 2] = 0
            oriented = bin_power(
                non_dc, maps["orientation_index"], orientation_bins
            ).mean(dim=0)

            mse_sum[token_index + 1] += current_error.double().sum().cpu()
            increment_square_sum[token_index] += delta.double().square().sum().cpu()
            improvement_count[token_index] += (current_error < previous_error).double().sum().cpu()
            residual_alignment_sum[token_index] += alignment.double().sum().cpu()
            radial_sum[token_index] += radial.double().cpu() * batch
            orientation_sum[token_index] += oriented.double().cpu() * batch
            increment_count += delta.numel()
            path_length += delta_norm
            if start == 0 and len(batch_contacts) < sequence_length:
                batch_contacts.append(delta[: args.contact_examples].cpu())
            previous = current
            previous_error = current_error

        displacement = (previous - initial).flatten(1).norm(dim=1).clamp_min(1e-12)
        path_ratio_sum += float((path_length / displacement).sum().cpu())
        examples_seen += batch
        if start == 0:
            contact_deltas = torch.stack(batch_contacts, dim=0)

    assert contact_deltas is not None
    mse = (mse_sum / examples_seen).numpy()
    pixel_mse = mse / 4.0
    psnr = -10.0 * np.log10(np.maximum(pixel_mse, 1e-30))
    increment_rms = np.sqrt(
        increment_square_sum.numpy()
        / (examples_seen * targets.shape[1] * targets.shape[2] * targets.shape[3])
    )
    mse_gain = mse[:-1] - mse[1:]
    radial = (radial_sum / examples_seen).numpy()
    oriented = (orientation_sum / examples_seen).numpy()
    radial_total = radial.sum(axis=1).clip(min=1e-30)
    spectral_centroid = (
        radial * np.arange(radial_bins, dtype=np.float64)[None]
    ).sum(axis=1) / radial_total
    radial_fraction = radial / radial_total[:, None]
    orientation_fraction = oriented / oriented.sum(axis=1, keepdims=True).clip(min=1e-30)
    total_gain = max(mse[0] - mse[-1], 1e-30)
    cumulative_gain = (mse[0] - mse) / total_gain
    prefix_50 = int(np.argmax(cumulative_gain >= 0.5))
    prefix_90 = int(np.argmax(cumulative_gain >= 0.9))
    quartile_centroids = []
    quartile_power = []
    radial_coordinates = np.arange(radial_bins, dtype=np.float64)
    for start in range(0, sequence_length, 16):
        aggregate = radial[start : start + 16].sum(axis=0)
        power = float(aggregate.sum())
        quartile_power.append(power)
        quartile_centroids.append(
            float((aggregate * radial_coordinates).sum() / max(power, 1e-30))
        )

    result = {
        "key": key,
        "label": label,
        "color": color,
        "cache": str(cache_path.resolve()),
        "tokenizer_step": int(cache["tokenizer_step"]),
        "examples": examples_seen,
        "mse_by_prefix_normalized_minus1_1": mse.tolist(),
        "psnr_db_by_prefix": psnr.tolist(),
        "increment_rms_normalized_minus1_1": increment_rms.tolist(),
        "mse_gain_by_token": mse_gain.tolist(),
        "fraction_examples_improved_by_token": (
            improvement_count / examples_seen
        ).tolist(),
        "mean_residual_alignment_by_token": (
            residual_alignment_sum / examples_seen
        ).tolist(),
        "radial_power_by_token": radial.tolist(),
        "radial_power_fraction_by_token": radial_fraction.tolist(),
        "orientation_power_by_token": oriented.tolist(),
        "orientation_power_fraction_by_token": orientation_fraction.tolist(),
        "spectral_centroid_by_token": spectral_centroid.tolist(),
        "summary": {
            "full_psnr_db": float(psnr[-1]),
            "tokens_for_50pct_error_reduction": prefix_50,
            "tokens_for_90pct_error_reduction": prefix_90,
            "fraction_token_steps_with_positive_population_gain": float(
                (mse_gain > 0).mean()
            ),
            "mean_fraction_examples_improved": float(
                (improvement_count / examples_seen).mean()
            ),
            "spectral_centroid_spearman_vs_token_index": spearman(
                spectral_centroid
            ),
            "energy_weighted_spectral_centroid_by_token_quartile": quartile_centroids,
            "increment_power_by_token_quartile": quartile_power,
            "fraction_adjacent_centroids_ascending": float(
                (np.diff(spectral_centroid) > 0).mean()
            ),
            "mean_decoder_path_length_ratio": path_ratio_sum / examples_seen,
        },
    }
    return result, contact_deltas


def overview_figure(results: list[dict], output: Path) -> None:
    blue_map = LinearSegmentedColormap.from_list(
        "blue_open", ["#F8FAFC", "#A9C4E2", "#2A5FA0", "#17365C"]
    )
    orange_map = LinearSegmentedColormap.from_list(
        "orange_open", ["#FCFAF7", "#E8C2A7", "#C2692D", "#6F3514"]
    )
    cmaps = [blue_map, orange_map]
    figure, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    figure.patch.set_facecolor("#FAFAF8")
    tokens = np.arange(1, 65)
    for row, (result, cmap) in enumerate(zip(results, cmaps)):
        color = result["color"]
        radial = np.asarray(result["radial_power_fraction_by_token"])
        oriented = np.asarray(result["orientation_power_fraction_by_token"])
        increment_rms = np.asarray(result["increment_rms_normalized_minus1_1"])
        gain = np.asarray(result["mse_gain_by_token"])
        improved = np.asarray(result["fraction_examples_improved_by_token"])
        mse = np.asarray(result["mse_by_prefix_normalized_minus1_1"])
        cumulative = (mse[0] - mse) / max(mse[0] - mse[-1], 1e-30)

        image = axes[row, 0].imshow(
            radial.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=np.quantile(radial, 0.99),
            extent=(0.5, 64.5, -0.5, radial.shape[1] - 0.5),
        )
        axes[row, 0].set_title("Radial composition of each increment")
        axes[row, 0].set_ylabel(f"{result['label']}\nFFT radius (pixels⁻¹ bin)")
        figure.colorbar(image, ax=axes[row, 0], label="fraction of increment power")

        image = axes[row, 1].imshow(
            oriented.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=np.quantile(oriented, 0.99),
            extent=(0.5, 64.5, 0, 180),
        )
        axes[row, 1].set_title("Orientation composition (DC excluded)")
        axes[row, 1].set_ylabel("orientation (degrees, modulo 180°)")
        figure.colorbar(image, ax=axes[row, 1], label="fraction of increment power")

        axes[row, 2].plot(tokens, increment_rms, color=color, lw=2, label="increment RMS")
        axes[row, 2].set_yscale("log")
        axes[row, 2].set_ylabel("increment RMS (log scale)")
        axes[row, 2].set_title("Magnitude and population error gain")
        secondary = axes[row, 2].twinx()
        secondary.axhline(0, color="#55595E", lw=1, ls="--")
        secondary.plot(tokens, gain, color="#32373D", lw=1.3, ls="--", label="MSE gain")
        secondary.set_ylabel("MSE(k−1) − MSE(k)")

        axes[row, 3].plot(
            np.arange(65), cumulative, color=color, lw=2.2, label="cumulative error reduction"
        )
        axes[row, 3].plot(
            tokens, improved, color="#32373D", lw=1.5, ls="--", label="samples improved"
        )
        axes[row, 3].axhline(0.5, color="#A2A5A8", lw=1, ls=":")
        axes[row, 3].axhline(0.9, color="#A2A5A8", lw=1, ls=":")
        axes[row, 3].set_ylim(-0.04, 1.04)
        axes[row, 3].set_title("Usefulness of additional tokens")
        axes[row, 3].set_ylabel("fraction")
        axes[row, 3].legend(frameon=False, loc="lower right", fontsize=8)

        for column in range(4):
            axes[row, column].set_xlabel("newly available token index")
            axes[row, column].grid(False)
            axes[row, column].spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "One-token decoder increments across all 64 latent slots",
        fontsize=18,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def contact_figure(results: list[dict], contacts: list[torch.Tensor], output: Path) -> None:
    examples = contacts[0].shape[1]
    figure, axes = plt.subplots(
        len(results) * examples,
        1,
        figsize=(18, 4.1 * len(results) * examples),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for model_index, (result, deltas) in enumerate(zip(results, contacts)):
        display_limit = float(torch.quantile(deltas.abs().float(), 0.995))
        display_limit = max(display_limit, 1e-8)
        for example_index in range(examples):
            row = model_index * examples + example_index
            tiles = []
            for token_index in range(64):
                delta = deltas[token_index, example_index].float()
                signed = (0.5 + delta / (2.0 * display_limit)).clamp(0, 1)
                tiles.append(signed.permute(1, 2, 0).numpy())
            sheet_rows = [np.concatenate(tiles[start : start + 8], axis=1) for start in range(0, 64, 8)]
            sheet = np.concatenate(sheet_rows, axis=0)
            axes[row].imshow(sheet, interpolation="nearest")
            axes[row].set_title(
                f"{result['label']} — held-out example {example_index}; Δ1…Δ64 left-to-right, top-to-bottom; shared ±{display_limit:.3f} scale"
            )
            axes[row].set_xticks([])
            axes[row].set_yticks([])
            for boundary in range(1, 8):
                axes[row].axvline(boundary * 32 - 0.5, color="white", lw=0.35, alpha=0.7)
                axes[row].axhline(boundary * 32 - 0.5, color="white", lw=0.35, alpha=0.7)
    figure.suptitle(
        "Signed image-space change from adding exactly one latent token",
        fontsize=18,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, facecolor="#FAFAF8")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0 or args.num_examples > 10_000:
        raise ValueError("num_examples must lie in [1, 10000]")
    if args.batch_size <= 0 or args.orientation_bins <= 1:
        raise ValueError("batch_size must be positive and orientation_bins > 1")
    if args.contact_examples <= 0 or args.contact_examples > args.batch_size:
        raise ValueError("contact_examples must lie in [1, batch_size]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = cifar_targets(args.data_root, args.num_examples)
    maps = frequency_maps(targets.shape[-1], args.orientation_bins, device)
    results = []
    contacts = []
    for key, label, cache, color in MODEL_SPECS:
        result, contact = analyze_model(
            key, label, Path(cache), color, targets, args, maps
        )
        results.append(result)
        contacts.append(contact)
        print(json.dumps({key: result["summary"]}, sort_keys=True), flush=True)

    payload = {
        "version": 1,
        "question": (
            "Do successive latent tokens add progressively finer useful information, "
            "or mostly rewrite earlier decoder content?"
        ),
        "dataset": "CIFAR-10 test, first examples in fixed dataset order",
        "num_examples": args.num_examples,
        "orientation_bins": args.orientation_bins,
        "radial_bin_definition": "rounded radius on centered 32x32 orthonormal FFT",
        "models": {result["key"]: result for result in results},
    }
    atomic_json(payload, output / "metrics.json")
    overview_figure(results, output / "prefix_increment_overview.png")
    contact_figure(results, contacts, output / "prefix_increment_contact_sheet.png")
    print(json.dumps({"complete": str(output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
