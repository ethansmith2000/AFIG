#!/usr/bin/env python3
"""Audit regularized factorized and flattened whitening before prior training."""

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
from progressive_tokenizer.latent_geometry import fit_axis_geometry
from progressive_tokenizer.whitening import (
    covariance_diagnostics,
    invert_linear,
    project_linear,
    regularized_whitening_gains,
    tempered_token_profile,
)
from scripts.analyze_generation_trajectory import PLOT_COLORS, draw_line_chart


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
    parser.add_argument("--gain_caps", default="4,8,16,32")
    parser.add_argument("--betas", default="0,0.125,0.25,0.5")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _csv_floats(value: str, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated floats") from error
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise ValueError(f"{name} must contain finite values")
    return parsed


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor(
        [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
        device=values.device,
        dtype=values.dtype,
    )
    measured = torch.quantile(values, levels)
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "p05", "p25", "median", "p75", "p95", "maximum"),
            measured,
        )
    }


def _relative_rms(changed: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (changed.double() - reference.double()).square().mean().sqrt()
    denominator = reference.double().square().mean().sqrt().clamp_min(1e-30)
    return float(numerator / denominator)


@torch.no_grad()
def _decode(
    values: torch.Tensor,
    tokenizer,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    images = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            images.append(tokenizer.decode(batch).float().cpu())
    return torch.cat(images)


def _health(
    metrics: dict[str, object], baseline: dict[str, object], cap: float
) -> dict[str, object]:
    scalar_metrics = (
        float(metrics["relative_gain_range"]),
        float(metrics["float32_relative_roundtrip_rms"]),
        float(metrics["float16_relative_roundtrip_rms"]),
        float(metrics["decoded_float16_pixel_delta_rms"]),
        float(metrics["heldout_covariance"]["effective_rank"]),  # type: ignore[index]
        float(metrics["heldout_covariance"]["off_diagonal_frobenius_fraction"]),  # type: ignore[index]
    )
    checks = {
        "finite": all(math.isfinite(value) for value in scalar_metrics),
        "cap_at_most_16": cap <= 16.0,
        "relative_gain_at_most_16": float(metrics["relative_gain_range"]) <= 16.0001,
        "float32_roundtrip": float(metrics["float32_relative_roundtrip_rms"]) <= 1e-5,
        "float16_roundtrip": float(metrics["float16_relative_roundtrip_rms"]) <= 0.002,
        "decoded_float16_roundtrip": float(metrics["decoded_float16_pixel_delta_rms"]) <= 0.002,
        "effective_rank_increases": float(metrics["heldout_covariance"]["effective_rank"]) > float(baseline["effective_rank"]),  # type: ignore[index]
        "off_diagonal_fraction_decreases": float(metrics["heldout_covariance"]["off_diagonal_frobenius_fraction"]) < float(baseline["off_diagonal_frobenius_fraction"]),  # type: ignore[index]
    }
    return {"pass": all(checks.values()), "checks": checks}


def _schedule_pass(profile: dict[str, object]) -> bool:
    ranges = profile["ranges"]
    assert isinstance(ranges, dict)
    return (
        float(ranges["rational_odds"]) <= 4.0001
        and float(ranges["signal_metric_loss"]) <= 16.0001
        and float(ranges["flow_target_energy_loss"]) <= 3.0001
    )


def _select_cap(
    candidate: dict[str, object], baseline_rank: float
) -> str | None:
    caps = candidate["caps"]
    assert isinstance(caps, dict)
    healthy = [
        key
        for key, value in caps.items()
        if float(key) <= 16.0 and bool(value["health"]["pass"])
    ]
    if not healthy:
        return None
    reference_key = min(healthy, key=lambda key: abs(float(key) - 16.0))
    reference_rank = float(caps[reference_key]["heldout_covariance"]["effective_rank"])
    required = baseline_rank + 0.95 * max(reference_rank - baseline_rank, 0.0)
    eligible = [
        key
        for key in healthy
        if float(caps[key]["heldout_covariance"]["effective_rank"]) >= required
    ]
    return min(eligible or healthy, key=float)


def _plots(result: dict[str, object], output: Path) -> None:
    candidates = result["candidates"]
    assert isinstance(candidates, dict)
    cap_series = []
    for index, (name, candidate) in enumerate(candidates.items()):
        caps = candidate["caps"]
        x = [float(key) for key in caps]
        y = [float(caps[key]["heldout_covariance"]["effective_rank"]) for key in caps]
        cap_series.append((name, x, y, PLOT_COLORS[index]))
    baseline_rank = float(result["baseline"]["effective_rank"])  # type: ignore[index]
    cap_values = sorted({value for _, values, _, _ in cap_series for value in values})
    cap_series.append(
        ("untransformed", cap_values, [baseline_rank] * len(cap_values), PLOT_COLORS[3])
    )
    canvas = Image.new("RGB", (900, 560), "#FAFAF8")
    draw_line_chart(
        canvas,
        (0, 0, 900, 560),
        cap_series,
        title="Held-out covariance effective rank",
        y_label="effective rank / 1024",
    )
    canvas.save(output / "whitening_effective_rank.png", optimize=True)

    selected = result["selection"]
    assert isinstance(selected, dict)
    selected_candidate = str(selected["candidate"])
    selected_cap = str(selected["gain_cap"])
    profiles = candidates[selected_candidate]["caps"][selected_cap]["schedule_profiles"]
    schedule_series = []
    for index, (beta, profile) in enumerate(profiles.items()):
        crossings = profile["snr1_crossings"]
        schedule_series.append(
            (f"beta={beta}", list(range(1, len(crossings) + 1)), crossings, PLOT_COLORS[index])
        )
    schedule = Image.new("RGB", (1000, 600), "#FAFAF8")
    draw_line_chart(
        schedule,
        (0, 0, 1000, 600),
        schedule_series,
        title=f"{selected_candidate} cap {selected_cap}: explicit SNR=1 clocks",
        y_label="base integration time",
    )
    schedule.save(output / "schedule_profiles.png", optimize=True)

    spectrum_series = []
    baseline_values = result["baseline"]["eigenvalues"]  # type: ignore[index]
    spectrum_series.append(
        (
            "untransformed",
            list(range(1, len(baseline_values) + 1)),
            [max(float(v) / max(float(baseline_values[0]), 1e-30), 1e-12) for v in baseline_values],
            PLOT_COLORS[3],
        )
    )
    for index, name in enumerate(candidates):
        candidate = candidates[name]
        key = str(candidate["selected_cap"])
        values = candidate["caps"][key]["heldout_covariance"]["eigenvalues"]
        spectrum_series.append(
            (
                name,
                list(range(1, len(values) + 1)),
                [max(float(v) / max(float(values[0]), 1e-30), 1e-12) for v in values],
                PLOT_COLORS[index],
            )
        )
    spectrum = Image.new("RGB", (1000, 600), "#FAFAF8")
    draw_line_chart(
        spectrum,
        (0, 0, 1000, 600),
        spectrum_series,
        title="Held-out covariance spectra after selected regularization",
        y_label="eigenvalue / strongest",
        log_y=True,
    )
    spectrum.save(output / "heldout_covariance_spectra.png", optimize=True)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(
        args.fit_samples,
        args.eval_samples,
        args.roundtrip_samples,
        args.decode_samples,
        args.batch_size,
    ) <= 0:
        raise ValueError("sample counts and batch size must be positive")
    gain_caps = _csv_floats(args.gain_caps, "gain_caps")
    betas = _csv_floats(args.betas, "betas")
    if any(cap < 1.0 for cap in gain_caps) or any(beta < 0.0 for beta in betas):
        raise ValueError("gain caps must be at least one and betas nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        # The audit has an explicit 1e-5 inverse gate. TF32 is appropriate for
        # model training but not for testing a nominally exact linear inverse.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)
    checkpoint_path = Path(args.prior_checkpoint)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_mean = checkpoint["normalization"]["mean"].float().to(device)
    source_scale = checkpoint["normalization"]["scale"].float().to(device)

    fit_count = min(args.fit_samples, len(cache["train_latents"]))
    eval_count = min(args.eval_samples, len(cache["test_latents"]))
    train = cache["train_latents"][:fit_count].to(device).float()
    train = (train - source_mean) / source_scale
    geometry = fit_axis_geometry(train)
    element_mean = geometry["element_mean"]
    assert isinstance(element_mean, torch.Tensor)
    del train

    test = cache["test_latents"][:eval_count].to(device).float()
    test = (test - source_mean) / source_scale
    centered_test = test - element_mean
    baseline = covariance_diagnostics(centered_test.flatten(1))
    baseline_rank = float(baseline["effective_rank"])
    print(json.dumps({"baseline_effective_rank": baseline_rank}), flush=True)

    full_covariance = geometry["flattened_covariance"]
    sequence_basis = geometry["sequence_eigenvectors"]
    channel_basis = geometry["channel_eigenvectors"]
    flattened_basis = geometry["flattened_eigenvectors"]
    flattened_power = geometry["flattened_eigenvalues"]
    assert all(
        isinstance(value, torch.Tensor)
        for value in (
            full_covariance,
            sequence_basis,
            channel_basis,
            flattened_basis,
            flattened_power,
        )
    )
    factorized_basis = torch.kron(sequence_basis, channel_basis)
    factorized_power = (
        factorized_basis * (full_covariance @ factorized_basis)
    ).sum(dim=0).clamp_min(0.0)

    tokenizer, _ = load_tokenizer_checkpoint(Path(cache["tokenizer_checkpoint"]))
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    decode_count = min(args.decode_samples, eval_count)
    original_raw = cache["test_latents"][:decode_count].float()
    original_images = _decode(original_raw, tokenizer, args.batch_size, device)
    roundtrip_count = min(args.roundtrip_samples, eval_count)

    candidate_definitions = {
        "factorized": (factorized_basis, factorized_power),
        "flattened": (flattened_basis, flattened_power),
    }
    candidates: dict[str, dict[str, object]] = {}
    exact_assets: dict[str, dict[str, torch.Tensor]] = {}
    for name, (basis, power) in candidate_definitions.items():
        base_coefficients = centered_test.flatten(1) @ basis
        token_power = power.reshape(64, 16).mean(dim=1)
        candidate: dict[str, object] = {
            "definition": (
                "sequence-rank tokens with channel-rank features"
                if name == "factorized"
                else "consecutive flattened-PCA ranks packed 16 per token"
            ),
            "basis_orthogonality_max_abs_error": float(
                (basis.T @ basis - torch.eye(1024, device=device)).abs().max()
            ),
            "prewhitening_coordinate_power_quantiles": _quantiles(power),
            "prewhitening_token_power": [float(value) for value in token_power],
            "prewhitening_token_power_ratio": float(
                token_power.max() / token_power.min().clamp_min(1e-30)
            ),
            "caps": {},
        }
        exact_assets[name] = {
            "basis": basis.detach().cpu().float(),
            "coordinate_power": power.detach().cpu().float(),
            "token_power": token_power.detach().cpu().float(),
        }
        for cap in gain_caps:
            fitted = regularized_whitening_gains(power, cap)
            gains = fitted["gains"]
            assert isinstance(gains, torch.Tensor)
            transformed = base_coefficients * gains
            diagnostics = covariance_diagnostics(transformed)
            source_subset = test[:roundtrip_count]
            projected = project_linear(source_subset, element_mean, basis, gains)
            restored_float = invert_linear(projected, element_mean, basis, gains)
            restored_half = invert_linear(
                projected.half().float(), element_mean, basis, gains
            )
            recovered_raw = (
                restored_half[:decode_count] * source_scale + source_mean
            ).float().cpu()
            recovered_images = _decode(
                recovered_raw, tokenizer, args.batch_size, device
            )
            pixel_delta = recovered_images.double() - original_images.double()
            schedule_profiles = {
                f"{beta:g}": tempered_token_profile(token_power, cap, beta)
                for beta in betas
            }
            metrics: dict[str, object] = {
                "relative_gain_range": float(fitted["relative_gain_range"]),
                "gain_quantiles": _quantiles(gains),
                "power_floor": float(fitted["power_floor"]),
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
                "schedule_profiles": schedule_profiles,
            }
            metrics["health"] = _health(metrics, baseline, cap)
            candidate["caps"][f"{cap:g}"] = metrics  # type: ignore[index]
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "cap": cap,
                        "effective_rank": diagnostics["effective_rank"],
                        "off_diagonal": diagnostics[
                            "off_diagonal_frobenius_fraction"
                        ],
                        "health": metrics["health"],
                    }
                ),
                flush=True,
            )
        selected_cap = _select_cap(candidate, baseline_rank)
        candidate["selected_cap"] = selected_cap
        candidates[name] = candidate

    selected_candidate = None
    if candidates["factorized"]["selected_cap"] is not None:
        selected_candidate = "factorized"
    elif candidates["flattened"]["selected_cap"] is not None:
        selected_candidate = "flattened"
    if selected_candidate is None:
        failed_result = {
            "status": "complete_no_healthy_candidate",
            "source_cache": str(cache_path.resolve()),
            "prior_checkpoint": str(checkpoint_path.resolve()),
            "fit_samples": fit_count,
            "eval_samples": eval_count,
            "gain_caps": gain_caps,
            "betas": betas,
            "baseline": baseline,
            "candidates": candidates,
            "selection": None,
        }
        failed_path = output / "metrics.json"
        failed_path.write_text(
            json.dumps(failed_result, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(
            f"no whitening candidate passed the frozen health gates; diagnostics at {failed_path}"
        )
    selected_cap = str(candidates[selected_candidate]["selected_cap"])
    selected_metrics = candidates[selected_candidate]["caps"][selected_cap]  # type: ignore[index]
    schedule_profiles = selected_metrics["schedule_profiles"]
    selected_betas = [
        float(beta)
        for beta, profile in schedule_profiles.items()
        if _schedule_pass(profile)
    ]
    if not selected_betas:
        raise RuntimeError("no tempered schedule profile passed the frozen bounds")
    selected_beta = max(selected_betas)
    selected_profile = schedule_profiles[f"{selected_beta:g}"]
    selection = {
        "candidate": selected_candidate,
        "gain_cap": float(selected_cap),
        "beta": selected_beta,
        "loss_family": "flow_target_energy",
        "token_group_sizes": [1] * 64,
        "snr1_crossings": selected_profile["snr1_crossings"],
        "rational_odds": selected_profile["rational_odds"],
        "loss_weights": selected_profile["flow_target_energy_loss_weights"],
        "schedule_ranges": selected_profile["ranges"],
        "training_authorized_by_numeric_gates": True,
    }
    result: dict[str, object] = {
        "status": "complete",
        "source_cache": str(cache_path.resolve()),
        "prior_checkpoint": str(checkpoint_path.resolve()),
        "fit_samples": fit_count,
        "eval_samples": eval_count,
        "gain_caps": gain_caps,
        "betas": betas,
        "baseline": baseline,
        "candidates": candidates,
        "selection": selection,
    }
    result_path = output / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    selected_asset = exact_assets[selected_candidate]
    selected_power = selected_asset["coordinate_power"].to(device)
    selected_gain_fit = regularized_whitening_gains(
        selected_power, float(selected_cap)
    )
    selected_gains = selected_gain_fit["gains"]
    assert isinstance(selected_gains, torch.Tensor)
    torch.save(
        {
            "version": 1,
            "type": "regularized_linear_whitening",
            "source_cache": str(cache_path.resolve()),
            "physical_shape": [64, 16],
            "source_normalization_mean": source_mean.detach().cpu(),
            "source_normalization_scale": source_scale.detach().cpu(),
            "standardized_element_mean": element_mean.detach().cpu(),
            "basis": selected_asset["basis"],
            "gains": selected_gains.detach().cpu().float(),
            "coordinate_power": selected_asset["coordinate_power"],
            "token_power": selected_asset["token_power"],
            "selection": selection,
        },
        output / "selected_transform.pt",
    )
    _plots(result, output)
    print(json.dumps({"complete": str(result_path.resolve()), **selection}), flush=True)


if __name__ == "__main__":
    main()
