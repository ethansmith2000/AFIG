"""Screen amplitude coordinates before launching factorized-polar training arms.

The decoder may use a different amplitude coordinate from the Transformer
history.  This audit compares candidate decoder transforms on the exact compact
FFT of globally standardized CIFAR-10 pixels.  It measures distribution shape,
Gaussian-base support mismatch, and the radial allocation induced by both flat
coordinate loss and the inverse transform Jacobian.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F

from audit_fft_physical_geometry import collect_tokens, moments, quantiles
from frequency import FrequencyCodec, FrequencyCodecConfig
from train_continuous import make_dataloader


@dataclass(frozen=True)
class Transform:
    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]
    derivative: Callable[[torch.Tensor], torch.Tensor]
    lower_bound: float
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec_stats_path",
        default="autoencoder_runs/codec_stats_32_global_standardize.pt",
    )
    parser.add_argument(
        "--output_path",
        default="diagnostics/amplitude_transform_screen/report.json",
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--max_examples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


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


def make_transforms() -> list[Transform]:
    transforms = [
        Transform(
            name="log_eps_0p003",
            forward=lambda a: torch.log(a + 0.003),
            inverse=lambda y: torch.exp(y) - 0.003,
            derivative=lambda a: 1.0 / (a + 0.003),
            lower_bound=math.log(0.003),
            description="current ordinary log with epsilon floor",
        )
    ]
    for tau, label in ((1.0, "1"), (0.2, "0p2")):
        transforms.append(
            Transform(
                name=f"log1p_tau_{label}",
                forward=lambda a, tau=tau: torch.log1p(a / tau),
                inverse=lambda y, tau=tau: tau * torch.expm1(y),
                derivative=lambda a, tau=tau: 1.0 / (tau + a),
                lower_bound=0.0,
                description=f"linear below amplitude {tau:g}, logarithmic above",
            )
        )
    # Inverse-softplus is logarithmic for weak amplitudes and linear for strong
    # amplitudes. It therefore keeps small values readable without placing the
    # strong tail behind an exponential inverse. The epsilon gives exact zeros
    # a finite coordinate while leaving the Gaussian support mismatch tiny.
    epsilon = 0.003
    for tau, label in (
        (0.1, "0p1"),
        (0.2, "0p2"),
        (0.5, "0p5"),
        (1.0, "1"),
        (2.0, "2"),
        (5.0, "5"),
    ):
        transforms.append(
            Transform(
                name=f"inverse_softplus_tau_{label}_eps_0p003",
                forward=lambda a, tau=tau: (
                    (a + epsilon) / tau
                    + torch.log(-torch.expm1(-(a + epsilon) / tau))
                ),
                inverse=lambda y, tau=tau: tau * F.softplus(y) - epsilon,
                derivative=lambda a, tau=tau: 1.0
                / (
                    tau
                    * (1.0 - torch.exp(-(a + epsilon) / tau)).clamp_min(1e-12)
                ),
                lower_bound=math.log(math.expm1(epsilon / tau)),
                description=(
                    f"inverse-softplus amplitude with transition {tau:g}; "
                    "logarithmic below the transition and linear above"
                ),
            )
        )
    transforms.extend(
        [
            Transform(
                name="log1p_squared_tau_1",
                forward=lambda a: torch.log1p(a).square(),
                inverse=lambda y: torch.expm1(y.clamp_min(0.0).sqrt()),
                derivative=lambda a: 2.0 * torch.log1p(a) / (1.0 + a),
                lower_bound=0.0,
                description="squared log1p; suppresses the derivative at zero",
            ),
            Transform(
                name="raw",
                forward=lambda a: a,
                inverse=lambda y: y,
                derivative=lambda a: torch.ones_like(a),
                lower_bound=0.0,
                description="uncompressed physical amplitude",
            ),
        ]
    )
    # A Box--Cox-style continuum from strong compression to the raw coordinate.
    # After global affine standardization, (a**p - 1) / p is identical to a**p,
    # so the simpler expression is sufficient here.
    for exponent, label in (
        (1.0 / 3.0, "1over3"),
        (0.5, "1over2"),
        (2.0 / 3.0, "2over3"),
        (0.8, "0p8"),
    ):
        transforms.append(
            Transform(
                name=f"power_{label}",
                forward=lambda a, exponent=exponent: a.pow(exponent),
                inverse=lambda y, exponent=exponent: y.clamp_min(0.0).pow(
                    1.0 / exponent
                ),
                derivative=lambda a, exponent=exponent: exponent
                * a.clamp_min(1e-12).pow(exponent - 1.0),
                lower_bound=0.0,
                description=(
                    f"Box-Cox/power amplitude with exponent {exponent:g}; "
                    "preserves progressively more physical dynamic range as p approaches 1"
                ),
            )
        )
    return transforms


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def finite_moments(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    report = moments(finite)
    report["finite_fraction"] = float(finite.numel() / values.numel())
    return report


def radial_shares(
    amplitude: torch.Tensor,
    standardized: torch.Tensor,
    inverse_jacobian_sq: torch.Tensor,
    coordinate_metric_sq: torch.Tensor,
    codec: FrequencyCodec,
) -> list[dict[str, float | int]]:
    physical_energy = amplitude.square()
    totals = {
        "physical": physical_energy.sum().double().clamp_min(1e-30),
        "x0_zero": standardized.square().sum().double().clamp_min(1e-30),
        # For a zero velocity prediction and independent unit Gaussian source,
        # E[(u-z)^2 | u] = u^2 + 1.
        "v_zero": (standardized.square() + 1.0).sum().double().clamp_min(1e-30),
        # Unit coordinate error mapped into physical amplitude error.
        "inverse_jacobian": inverse_jacobian_sq.sum().double().clamp_min(1e-30),
        # Unit physical amplitude error measured by flat coordinate MSE.
        "coordinate_metric": coordinate_metric_sq.sum().double().clamp_min(1e-30),
    }
    reports = []
    for radius in range(codec.max_radius_bin + 1):
        selected = codec.radius_bin == radius
        if not bool(selected.any()):
            continue
        count = int(amplitude[:, selected, :].numel())
        reports.append(
            {
                "radius_bin": radius,
                "orbits": int(selected.sum()),
                "scalar_amplitude_fraction": float(count / amplitude.numel()),
                "physical_energy_share": float(
                    physical_energy[:, selected, :].sum().double() / totals["physical"]
                ),
                "zero_x0_target_share": float(
                    standardized[:, selected, :].square().sum().double()
                    / totals["x0_zero"]
                ),
                "zero_velocity_target_share": float(
                    (standardized[:, selected, :].square() + 1.0).sum().double()
                    / totals["v_zero"]
                ),
                "inverse_jacobian_sq_share": float(
                    inverse_jacobian_sq[:, selected, :].sum().double()
                    / totals["inverse_jacobian"]
                ),
                "coordinate_metric_sq_share": float(
                    coordinate_metric_sq[:, selected, :].sum().double()
                    / totals["coordinate_metric"]
                ),
                "standardized_mean": float(standardized[:, selected, :].mean()),
                "standardized_std": float(
                    standardized[:, selected, :].double().std(unbiased=False)
                ),
            }
        )
    return reports


def cumulative(reports: list[dict[str, float | int]], key: str, radius: int) -> float:
    return float(
        sum(float(row[key]) for row in reports if int(row["radius_bin"]) <= radius)
    )


@torch.no_grad()
def transform_report(
    transform: Transform,
    amplitude: torch.Tensor,
    codec: FrequencyCodec,
) -> dict:
    coordinate = transform.forward(amplitude.float())
    mean = coordinate.double().mean()
    std = coordinate.double().std(unbiased=False).clamp_min(1e-12)
    standardized = ((coordinate.double() - mean) / std).float()
    standardized_lower_bound = (transform.lower_bound - float(mean)) / float(std)

    derivative = transform.derivative(amplitude.float()) / float(std)
    coordinate_metric_sq = derivative.square()
    inverse_jacobian_sq = derivative.clamp_min(1e-30).reciprocal().square()
    radial = radial_shares(
        amplitude,
        standardized,
        inverse_jacobian_sq,
        coordinate_metric_sq,
        codec,
    )

    base = torch.linspace(-4.5, 4.5, 20001)
    unstandardized_base = base * float(std) + float(mean)
    base_amplitude = transform.inverse(unstandardized_base)
    valid_base = unstandardized_base >= transform.lower_bound

    return {
        "name": transform.name,
        "description": transform.description,
        "mean": float(mean),
        "std": float(std),
        "standardized_lower_bound": standardized_lower_bound,
        "gaussian_base_below_support_probability": normal_cdf(
            standardized_lower_bound
        ),
        "standardized_moments": moments(standardized),
        "standardized_quantiles": quantiles(standardized),
        "standardized_tail_fractions": {
            "abs_gt_3": float((standardized.abs() > 3.0).float().mean()),
            "abs_gt_4": float((standardized.abs() > 4.0).float().mean()),
            "abs_gt_8": float((standardized.abs() > 8.0).float().mean()),
        },
        "coordinate_metric_sq_moments": finite_moments(coordinate_metric_sq),
        "coordinate_metric_sq_quantiles": quantiles(
            coordinate_metric_sq.clamp_max(1e30)
        ),
        "inverse_jacobian_sq_moments": finite_moments(inverse_jacobian_sq),
        "inverse_jacobian_sq_quantiles": quantiles(
            inverse_jacobian_sq.clamp_max(1e30)
        ),
        "base_grid_valid_fraction": float(valid_base.float().mean()),
        "base_grid_amplitude_quantiles_after_clamp": quantiles(
            base_amplitude.clamp_min(0.0)
        ),
        "radial": radial,
        "cumulative_radius_0_2": {
            key: cumulative(radial, key, 2)
            for key in (
                "scalar_amplitude_fraction",
                "physical_energy_share",
                "zero_x0_target_share",
                "zero_velocity_target_share",
                "inverse_jacobian_sq_share",
                "coordinate_metric_sq_share",
            )
        },
        "cumulative_radius_0_5": {
            key: cumulative(radial, key, 5)
            for key in (
                "scalar_amplitude_fraction",
                "physical_energy_share",
                "zero_x0_target_share",
                "zero_velocity_target_share",
                "inverse_jacobian_sq_share",
                "coordinate_metric_sq_share",
            )
        },
    }


def main() -> None:
    args = parse_args()
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            normalization="global_standardize",
            value_transform="identity",
            coordinate_packing="isometric",
        )
    )
    codec.load_exported(
        torch.load(args.codec_stats_path, map_location="cpu", weights_only=False)
    )
    tokens = collect_tokens(build_loader(args), codec, args.max_examples)
    amplitude = torch.sqrt(tokens[..., :3].square() + tokens[..., 3:].square())

    report = {
        "version": 1,
        "examples": int(tokens.shape[0]),
        "coordinate_system": (
            "one population pixel mean/std, orthonormal FFT, active-only "
            "isometric compact Hermitian packing"
        ),
        "amplitude_moments": moments(amplitude),
        "amplitude_quantiles": quantiles(amplitude),
        "transforms": [
            transform_report(transform, amplitude, codec)
            for transform in make_transforms()
        ],
        "interpretation": {
            "coordinate_metric_sq": (
                "Local weight placed on a unit physical amplitude error by flat "
                "MSE in the standardized transform coordinate: (f'(a)/std(f))^2."
            ),
            "inverse_jacobian_sq": (
                "Physical amplitude error caused by a unit standardized-coordinate "
                "error: (std(f)/f'(a))^2. This is the radial factor entering a "
                "Cartesian endpoint loss before its residual term."
            ),
            "support_probability": (
                "Probability that a standard-Gaussian decoder state lies below "
                "the exact inverse transform's physical lower bound."
            ),
        },
    }

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
