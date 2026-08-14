#!/usr/bin/env python3
"""Audit sequence/feature geometry and spectral prefix reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--geometry_examples", type=int, default=50000)
    parser.add_argument("--spectral_examples", type=int, default=2048)
    parser.add_argument("--decode_batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def effective_rank(covariance: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def off_diagonal_summary(correlation: torch.Tensor) -> dict[str, float]:
    count = correlation.shape[0]
    mask = ~torch.eye(count, dtype=torch.bool, device=correlation.device)
    values = correlation[mask]
    return {
        "mean": float(values.mean()),
        "mean_abs": float(values.abs().mean()),
        "p95_abs": float(values.abs().quantile(0.95)),
        "max_abs": float(values.abs().max()),
    }


def distance_profile(matrix: torch.Tensor) -> list[float]:
    return [
        float(torch.diagonal(matrix, offset=distance).mean())
        for distance in range(1, matrix.shape[0])
    ]


def residual_fraction(
    covariance: torch.Tensor,
    target: slice,
    context: slice,
) -> float:
    target_cov = covariance[target, target]
    context_cov = covariance[context, context]
    cross_cov = covariance[context, target]
    scale = torch.diagonal(context_cov).mean().clamp_min(1e-8)
    ridge = 1e-5 * scale
    solution = torch.linalg.solve(
        context_cov
        + ridge
        * torch.eye(
            context_cov.shape[0],
            device=context_cov.device,
            dtype=context_cov.dtype,
        ),
        cross_cov,
    )
    residual = target_cov - cross_cov.T @ solution
    return float(
        torch.diagonal(residual).sum().clamp_min(0)
        / torch.diagonal(target_cov).sum().clamp_min(1e-12)
    )


@torch.no_grad()
def geometry_metrics(latents: torch.Tensor, device: torch.device) -> dict:
    values = latents[:].float().to(device)
    _, tokens, width = values.shape
    centered = values - values.mean(dim=0, keepdim=True)

    # Axis-specific covariance: token positions are variables after averaging
    # corresponding feature coordinates; feature coordinates are variables after
    # averaging token positions. Both avoid treating T*D as one undifferentiated axis.
    sequence_covariance = torch.einsum("ntd,nsd->ts", centered, centered)
    sequence_covariance /= centered.shape[0] * width
    feature_covariance = torch.einsum("ntd,nte->de", centered, centered)
    feature_covariance /= centered.shape[0] * tokens

    sequence_scale = torch.diagonal(sequence_covariance).clamp_min(1e-12).sqrt()
    sequence_correlation = sequence_covariance / (
        sequence_scale[:, None] * sequence_scale[None, :]
    )
    feature_scale = torch.diagonal(feature_covariance).clamp_min(1e-12).sqrt()
    feature_correlation = feature_covariance / (
        feature_scale[:, None] * feature_scale[None, :]
    )

    # The full block covariance is used only to distinguish local token coupling
    # from distributed coupling and to measure conditional linear predictability.
    flattened = centered.reshape(centered.shape[0], tokens * width)
    block_covariance = flattened.T @ flattened / flattened.shape[0]
    coupling = torch.eye(tokens, device=device)
    for first in range(tokens):
        first_slice = slice(first * width, (first + 1) * width)
        first_norm = torch.linalg.matrix_norm(
            block_covariance[first_slice, first_slice], ord="fro"
        )
        for second in range(first + 1, tokens):
            second_slice = slice(second * width, (second + 1) * width)
            second_norm = torch.linalg.matrix_norm(
                block_covariance[second_slice, second_slice], ord="fro"
            )
            cross_norm = torch.linalg.matrix_norm(
                block_covariance[first_slice, second_slice], ord="fro"
            )
            value = cross_norm / (first_norm * second_norm).sqrt().clamp_min(1e-12)
            coupling[first, second] = coupling[second, first] = value

    slot_effective_ranks = []
    for index in range(tokens):
        slot = centered[:, index]
        slot_covariance = slot.T @ slot / slot.shape[0]
        slot_effective_ranks.append(effective_rank(slot_covariance))

    previous_only = [1.0]
    full_prefix = [1.0]
    for index in range(1, tokens):
        target = slice(index * width, (index + 1) * width)
        previous = slice((index - 1) * width, index * width)
        prefix = slice(0, index * width)
        previous_only.append(residual_fraction(block_covariance, target, previous))
        full_prefix.append(residual_fraction(block_covariance, target, prefix))

    strongest_pairs = []
    for first in range(tokens):
        for second in range(first + 1, tokens):
            strongest_pairs.append(
                (float(coupling[first, second]), first + 1, second + 1)
            )
    strongest_pairs.sort(reverse=True)

    return {
        "shape": list(values.shape),
        "sequence_axis": {
            "effective_rank": effective_rank(sequence_covariance),
            "rank_denominator": tokens,
            "correlation_off_diagonal": off_diagonal_summary(sequence_correlation),
            "signed_correlation_by_distance": distance_profile(sequence_correlation),
            "block_coupling_by_distance": distance_profile(coupling),
            "strongest_block_coupling_pairs": [
                {"first": first, "second": second, "coupling": value}
                for value, first, second in strongest_pairs[:12]
            ],
            "linear_residual_fraction_previous_token": previous_only,
            "linear_residual_fraction_full_prefix": full_prefix,
        },
        "feature_axis": {
            "effective_rank": effective_rank(feature_covariance),
            "rank_denominator": width,
            "correlation_off_diagonal": off_diagonal_summary(feature_correlation),
            "slot_effective_rank": slot_effective_ranks,
        },
    }


def frequency_groups(height: int, width: int, device: torch.device):
    vertical = torch.fft.fftfreq(height, device=device) * height
    horizontal = torch.fft.rfftfreq(width, device=device) * width
    radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
    masks = {
        "radius_0_3": radius < 4,
        "radius_4_7": (radius >= 4) & (radius < 8),
        "radius_8_11": (radius >= 8) & (radius < 12),
        "radius_12_plus": radius >= 12,
    }
    multiplicity = torch.ones_like(radius)
    if width % 2 == 0:
        multiplicity[:, 1:-1] = 2
    else:
        multiplicity[:, 1:] = 2
    return masks, multiplicity


@torch.no_grad()
def spectral_prefix_metrics(
    tokenizer,
    latents: torch.Tensor,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    tokens = latents.shape[1]
    masks, multiplicity = frequency_groups(images.shape[-2], images.shape[-1], device)
    errors = {name: [] for name in masks}
    pixel_mse = []
    for prefix in range(1, tokens + 1):
        band_sums = {name: 0.0 for name in masks}
        band_counts = {name: 0.0 for name in masks}
        square_error = 0.0
        pixel_count = 0
        for start in range(0, latents.shape[0], batch_size):
            stop = min(start + batch_size, latents.shape[0])
            target = images[start:stop].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = tokenizer.decode(
                    latents[start:stop].to(device), prefix_lengths=prefix
                ).float()
            difference = prediction - target
            square_error += float(difference.square().sum())
            pixel_count += difference.numel()
            spectrum = torch.fft.rfft2(difference, norm="ortho").abs().square()
            for name, mask in masks.items():
                weights = (mask * multiplicity).to(spectrum.dtype)
                band_sums[name] += float((spectrum * weights[None, None]).sum())
                band_counts[name] += float(weights.sum() * spectrum.shape[0] * spectrum.shape[1])
        pixel_mse.append(square_error / pixel_count)
        for name in masks:
            errors[name].append(band_sums[name] / band_counts[name])

    recoverable_fraction = {}
    half_recovery_prefix = {}
    for name, values in errors.items():
        first, final = values[0], values[-1]
        denominator = max(first - final, 1e-12)
        fractions = [(first - value) / denominator for value in values]
        recoverable_fraction[name] = fractions
        half_recovery_prefix[name] = next(
            (index + 1 for index, value in enumerate(fractions) if value >= 0.5),
            None,
        )

    return {
        "examples": int(latents.shape[0]),
        "pixel_mse_by_prefix": pixel_mse,
        "fourier_error_by_prefix": errors,
        "recoverable_error_fraction_by_prefix": recoverable_fraction,
        "half_recovery_prefix": half_recovery_prefix,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    geometry_count = min(args.geometry_examples, cache["train_latents"].shape[0])
    geometry = geometry_metrics(cache["train_latents"][:geometry_count], device)

    tokenizer, _ = load_tokenizer_checkpoint(cache["tokenizer_checkpoint"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda image: image.mul(2).sub(1)),
            ]
        ),
    )
    spectral_count = min(args.spectral_examples, cache["test_latents"].shape[0])
    images = torch.stack([dataset[index][0] for index in range(spectral_count)])
    spectral = spectral_prefix_metrics(
        tokenizer,
        cache["test_latents"][:spectral_count].float(),
        images,
        args.decode_batch_size,
        device,
    )
    payload = {
        "latent_cache": str(Path(args.latent_cache).resolve()),
        "geometry": geometry,
        "spectral_prefix": spectral,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
