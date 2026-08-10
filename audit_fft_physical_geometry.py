"""Audit physical compact-FFT geometry before decoder parameterization sweeps.

The audit deliberately uses ``global_standardize``: one train-population pixel
mean/std followed by the exact active-only isometric Hermitian packing.  It
therefore measures the coordinates seen by a geometry-safe direct Fourier model
without per-frequency centering or whitening.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from frequency import FrequencyCodec, FrequencyCodecConfig
from train_continuous import make_dataloader


QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec_stats_path",
        default="autoencoder_runs/codec_stats_32_global_standardize.pt",
    )
    parser.add_argument(
        "--output_path",
        default="diagnostics/fft_physical_geometry/stats.json",
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    stride = max(math.ceil(values.numel() / 4_000_000), 1)
    values = values[::stride]
    probs = torch.tensor(QUANTILES, dtype=torch.float32)
    result = torch.quantile(values, probs)
    return {
        f"p{100.0 * probability:g}": float(value)
        for probability, value in zip(QUANTILES, result)
    }


def moments(values: torch.Tensor) -> dict[str, float]:
    values = values.double().reshape(-1)
    mean = values.mean()
    centered = values - mean
    variance = centered.square().mean()
    std = variance.sqrt().clamp_min(1e-12)
    standardized = centered / std
    return {
        "mean": float(mean),
        "std": float(std),
        "skew": float(standardized.pow(3).mean()),
        "excess_kurtosis": float(standardized.pow(4).mean() - 3.0),
    }


def build_loader(args: argparse.Namespace):
    loader_args = SimpleNamespace(
        smoke=False,
        synthetic_data=False,
        dataset=args.dataset,
        data_root=args.data_root,
        spectral_panel_size=16,
        train_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
    )
    _, loader = make_dataloader(loader_args)
    return loader


@torch.no_grad()
def collect_tokens(
    loader,
    codec: FrequencyCodec,
    max_examples: int,
) -> torch.Tensor:
    chunks = []
    seen = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        take = min(int(images.shape[0]), max_examples - seen)
        if take <= 0:
            break
        chunks.append(codec.encode(images[:take]).cpu())
        seen += take
        if seen >= max_examples:
            break
    if not chunks:
        raise RuntimeError("No examples were collected")
    return torch.cat(chunks, dim=0)


def ring_report(
    amplitude: torch.Tensor,
    power: torch.Tensor,
    codec: FrequencyCodec,
) -> list[dict[str, Any]]:
    reports = []
    total_energy = power.sum().double().clamp_min(1e-12)
    total_values = power.numel()
    cumulative_energy = 0.0
    for radius in range(codec.max_radius_bin + 1):
        selected = codec.radius_bin == radius
        if not bool(selected.any()):
            continue
        ring_amplitude = amplitude[:, selected, :]
        ring_power = power[:, selected, :]
        energy_fraction = float(ring_power.sum().double() / total_energy)
        cumulative_energy += energy_fraction
        # Per-coordinate RMS is the physical signal scale relevant to a unit
        # Gaussian flow bridge.  SNR(t)=1 for x_t=t*x0+(1-t)*eps at t=1/(1+s).
        active_coordinates = (
            codec.component_mask[selected].sum().item() * amplitude.shape[0]
        )
        coordinate_rms = math.sqrt(
            float(ring_power.sum()) / max(float(active_coordinates), 1.0)
        )
        reports.append(
            {
                "radius_bin": radius,
                "orbits": int(selected.sum()),
                "coordinate_fraction": float(active_coordinates / (amplitude.shape[0] * 3072)),
                "energy_fraction": energy_fraction,
                "cumulative_energy_fraction": cumulative_energy,
                "active_coordinate_rms": coordinate_rms,
                "flow_t_at_snr_one": 1.0 / (1.0 + coordinate_rms),
                "amplitude_quantiles": quantiles(ring_amplitude),
            }
        )
    if total_values <= 0:
        raise RuntimeError("No power values were available")
    return reports


def epsilon_report(amplitude: torch.Tensor) -> list[dict[str, Any]]:
    positive = amplitude[amplitude > 0].float()
    reference = torch.quantile(
        positive,
        torch.tensor([0.001, 0.01, 0.05, 0.1, 0.5]),
    )
    candidates = sorted(
        {
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            *(float(value) for value in reference),
        }
    )
    # Higher-moment/quantile transforms are repeated for every epsilon.  A
    # deterministic uniform stride keeps this audit bounded without changing
    # the full-population knee fractions below.
    stride = max(math.ceil(positive.numel() / 2_000_000), 1)
    shape_values = positive[::stride]
    reports = []
    for epsilon in candidates:
        coordinate = torch.log(shape_values + epsilon)
        reports.append(
            {
                "epsilon": epsilon,
                "fraction_below_knee": float((positive < epsilon).float().mean()),
                "shape_sample_count": int(shape_values.numel()),
                "coordinate_moments": moments(coordinate),
                "coordinate_quantiles": quantiles(coordinate),
            }
        )
    return reports


def main() -> None:
    args = parse_args()
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            normalization="global_standardize",
            value_transform="identity",
            coordinate_packing="isometric",
        )
    )
    payload = torch.load(args.codec_stats_path, map_location="cpu", weights_only=False)
    codec.load_exported(payload)
    tokens = collect_tokens(build_loader(args), codec, args.max_examples)

    real = tokens[..., :3]
    imag = tokens[..., 3:]
    amplitude = torch.sqrt(real.square() + imag.square())
    power = amplitude.square()
    active = codec.component_mask.bool()[None].expand(tokens.shape[0], -1, -1)
    active_values = tokens[active]

    channel_reports = []
    for channel in range(3):
        values = amplitude[..., channel]
        channel_reports.append(
            {
                "channel": channel,
                "amplitude_quantiles": quantiles(values),
                "power_quantiles": quantiles(values.square()),
                "log_amplitude_moments_eps_1e-6": moments(
                    torch.log(values.clamp_min(1e-6))
                ),
            }
        )

    ordinary = ~codec.is_self_conjugate
    phase = torch.atan2(imag[:, ordinary], real[:, ordinary])
    phase_resultant = torch.sqrt(
        torch.cos(phase).mean(dim=(0, 1)).square()
        + torch.sin(phase).mean(dim=(0, 1)).square()
    )

    report: dict[str, Any] = {
        "version": 1,
        "examples": int(tokens.shape[0]),
        "coordinate_system": (
            "one population pixel mean/std, orthonormal FFT, active-only "
            "isometric Hermitian real/imag packing"
        ),
        "global_pixel_mean": float(codec.global_pixel_mean),
        "global_pixel_std": float(codec.global_scale),
        "active_coordinate_moments": moments(active_values),
        "active_coordinate_quantiles": quantiles(active_values),
        "amplitude_quantiles": quantiles(amplitude),
        "power_quantiles": quantiles(power),
        "channel_reports": channel_reports,
        "ordinary_phase_resultant_length_by_rgb": [
            float(value) for value in phase_resultant
        ],
        "epsilon_candidates": epsilon_report(amplitude),
        "rings": ring_report(amplitude, power, codec),
        "loss_geometry": {
            "total_active_coordinates": 3072,
            "cartesian_mse_phase_local_metric": "da^2 + a^2*dtheta^2",
            "ring_energy_fraction_interpretation": (
                "Expected share of squared physical Cartesian signal energy; "
                "also the natural local phase-error weighting by ring."
            ),
        },
    }

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
