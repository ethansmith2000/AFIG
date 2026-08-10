"""Audit conditional calibration of a factorized Fourier token decoder.

Pointwise x0 reconstruction metrics are useful optimization diagnostics, but they
do not test whether a stochastic decoder has the right conditional spread.  This
script gives every absolute frequency its true causal prefix, draws several
independent decoder samples, and reports a proper multivariate energy score plus
spread/error and endpoint-support diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from factorized_polar_decoder import polar_to_cartesian
from frequency import FrequencyCodec
from model_continuous import ContinuousFFTDecoder
from train_continuous import build_model_config


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[ContinuousFFTDecoder, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = Namespace(**payload["args"])
    config = build_model_config(saved_args)
    codec = FrequencyCodec(config.codec)
    codec.load_exported(payload["codec"])
    model = ContinuousFFTDecoder(config, codec=codec)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    if model.factorized_decoder is None:
        raise ValueError("checkpoint does not use decoder_geometry=factorized_polar")
    if model.coefficients_per_token != 1:
        raise ValueError("this diagnostic currently expects one coefficient per AR step")
    return model, int(payload.get("global_step", payload.get("step", -1)))


def _endpoint_lower_bound(model: ContinuousFFTDecoder) -> float:
    config = model.config.factorized_polar
    if config.amplitude_transform == "log_eps":
        return math.log(float(config.log_epsilon))
    if config.amplitude_transform == "inverse_softplus":
        return math.log(
            math.expm1(float(config.log_epsilon) / float(config.amplitude_transform_parameter))
        )
    if config.amplitude_transform in {"log1p", "power", "raw"}:
        return 0.0
    raise ValueError(f"unsupported amplitude transform: {config.amplitude_transform}")


@torch.no_grad()
def conditional_samples(
    model: ContinuousFFTDecoder,
    true_tokens: torch.Tensor,
    *,
    draws: int,
    steps: int,
    position_chunk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw tokens independently at every position under its true causal prefix."""
    batch, length, _ = true_tokens.shape
    hidden, _ = model.forward_backbone(
        model.embed_tokens(true_tokens[:, :-1], include_bos=True),
        use_cache=False,
    )
    decoder = model.factorized_decoder
    assert decoder is not None
    generator = torch.Generator(device=true_tokens.device).manual_seed(seed)
    raw_chunks = []
    coordinate_chunks = []

    for start in range(0, length, position_chunk):
        stop = min(start + position_chunk, length)
        positions = torch.arange(start, stop, device=true_tokens.device)
        count = stop - start
        z = hidden[:, start:stop]
        slot = model.prediction_slot_condition(
            positions, batch_size=batch, dtype=hidden.dtype
        )
        z = z.unsqueeze(0).expand(draws, -1, -1, -1).reshape(
            draws * batch * count, -1
        )
        slot = slot.unsqueeze(0).expand(draws, -1, -1, -1).reshape(
            draws * batch * count, -1
        )
        repeated_positions = positions[None, None].expand(
            draws, batch, count
        ).reshape(-1)
        is_self = model.codec.is_self_conjugate[positions][None, None].expand(
            draws, batch, count
        ).reshape(-1)
        amplitude_coordinate, phase = decoder.sample_coordinates(
            z=z,
            slot_condition=slot,
            generator=generator,
            steps=steps,
            is_self_conjugate=is_self,
            positions=repeated_positions,
        )
        scale = model.factorized_amplitude_scale(positions)[None, None].expand(
            draws, batch, count, 3
        ).reshape(-1, 3)
        raw = polar_to_cartesian(
            amplitude_coordinate.float(),
            phase.float(),
            scale,
            is_self,
            model.config.factorized_polar.log_epsilon,
            (
                decoder.amplitude_coordinate_mean
                if model.config.factorized_polar.amplitude_standardization != "none"
                else None
            ),
            (
                decoder.amplitude_coordinate_std
                if model.config.factorized_polar.amplitude_standardization != "none"
                else None
            ),
            model.config.factorized_polar.amplitude_transform,
            model.config.factorized_polar.amplitude_transform_parameter,
        )
        raw_chunks.append(raw.reshape(draws, batch, count, 6).cpu())
        coordinate_chunks.append(
            amplitude_coordinate.float().reshape(draws, batch, count, 3).cpu()
        )

    return torch.cat(raw_chunks, dim=2), torch.cat(coordinate_chunks, dim=2)


def _pair_indices(draws: int) -> Iterable[tuple[int, int]]:
    return combinations(range(draws), 2)


def calibration_summary(
    samples: torch.Tensor,
    target: torch.Tensor,
    component_mask: torch.Tensor,
    selection: torch.Tensor,
) -> Dict[str, float]:
    """Return physical-space conditional accuracy and calibration summaries."""
    predicted = samples[:, :, selection].double()
    truth = target[:, selection].double()
    mask = component_mask[selection].double()
    error_sq = ((predicted - truth.unsqueeze(0)).square() * mask).sum(-1)
    error_distance = error_sq.sqrt()

    pair_sq_parts = []
    pair_distance_parts = []
    for left, right in _pair_indices(predicted.shape[0]):
        pair_sq = ((predicted[left] - predicted[right]).square() * mask).sum(-1)
        pair_sq_parts.append(pair_sq)
        pair_distance_parts.append(pair_sq.sqrt())
    pair_sq = torch.stack(pair_sq_parts) if pair_sq_parts else torch.zeros_like(
        error_sq[:1]
    )
    pair_distance = (
        torch.stack(pair_distance_parts)
        if pair_distance_parts
        else torch.zeros_like(error_distance[:1])
    )

    target_energy = (truth.square() * mask).sum(-1).mean().clamp_min(1e-16)
    mean_error_sq = error_sq.mean()
    mean_pair_sq = pair_sq.mean()
    energy_score = error_distance.mean() - 0.5 * pair_distance.mean()
    ensemble_mean = predicted.mean(0)
    ensemble_mean_error = (
        ((ensemble_mean - truth).square() * mask).sum(-1).mean()
    )

    target_amplitude = torch.sqrt(
        truth[..., :3].square() + truth[..., 3:].square()
    )
    predicted_amplitude = torch.sqrt(
        predicted[..., :3].square() + predicted[..., 3:].square()
    )
    amplitude_denominator = target_amplitude.mean().clamp_min(1e-16)

    return {
        "positions": int(selection.sum()),
        "target_rms": float(target_energy.sqrt()),
        "sample_to_target_nrmse": float((mean_error_sq / target_energy).sqrt()),
        "ensemble_mean_to_target_nrmse": float(
            (ensemble_mean_error / target_energy).sqrt()
        ),
        "pairwise_spread_nrmse": float((mean_pair_sq / target_energy).sqrt()),
        "spread_to_error_sq_ratio": float(
            mean_pair_sq / mean_error_sq.clamp_min(1e-16)
        ),
        "energy_score": float(energy_score),
        "normalized_energy_score": float(energy_score / target_energy.sqrt()),
        "sample_power_to_target_ratio": float(
            (predicted.square() * mask).sum(-1).mean() / target_energy
        ),
        "mean_amplitude_ratio": float(predicted_amplitude.mean() / amplitude_denominator),
        "near_zero_amplitude_fraction": float(
            (predicted_amplitude <= 1e-8).double().mean()
        ),
    }


def support_summary(
    model: ContinuousFFTDecoder,
    standardized_coordinate: torch.Tensor,
    selection: torch.Tensor,
) -> Dict[str, float]:
    decoder = model.factorized_decoder
    assert decoder is not None
    coordinate = standardized_coordinate[:, :, selection].double()
    if model.config.factorized_polar.amplitude_standardization != "none":
        coordinate = (
            coordinate * decoder.amplitude_coordinate_std.detach().cpu().double()
            + decoder.amplitude_coordinate_mean.detach().cpu().double()
        )
    lower = _endpoint_lower_bound(model)
    invalid = coordinate < lower
    quantiles = torch.quantile(
        coordinate.reshape(-1),
        torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], dtype=torch.double),
    )
    return {
        "endpoint_lower_bound": lower,
        "below_endpoint_support_fraction": float(invalid.double().mean()),
        "coordinate_p0p1": float(quantiles[0]),
        "coordinate_p1": float(quantiles[1]),
        "coordinate_p50": float(quantiles[2]),
        "coordinate_p99": float(quantiles[3]),
        "coordinate_p99p9": float(quantiles[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--position_chunk", type=int, default=16)
    parser.add_argument(
        "--max_positions",
        type=int,
        default=None,
        help="Optional causal-prefix length for inexpensive smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.draws < 2:
        raise ValueError("draws must be at least two for a spread estimate")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_step = load_model(args.checkpoint, device)
    dataset = CIFAR10(
        root=str(args.data_root), train=False, transform=ToTensor(), download=False
    )
    images = torch.stack([dataset[index][0] for index in range(args.num_images)]).to(
        device
    )
    true_tokens = model.codec.encode(images)
    if args.max_positions is not None:
        if not 0 < args.max_positions <= model.codec.seq_len_int:
            raise ValueError("max_positions must be in [1, codec sequence length]")
        true_tokens = true_tokens[:, : args.max_positions]
    length = true_tokens.shape[1]
    positions = torch.arange(length, device=device)
    target = model.factorized_cartesian_target(true_tokens, positions).cpu()
    inference_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with inference_context:
        samples, coordinates = conditional_samples(
            model,
            true_tokens,
            draws=args.draws,
            steps=args.steps,
            position_chunk=args.position_chunk,
            seed=args.seed,
        )
    component_mask = model.codec.component_mask[:length].cpu()
    radius = model.codec.radius_bin[:length].cpu()

    selections: Dict[str, torch.Tensor] = {
        "all": torch.ones(length, dtype=torch.bool),
    }
    for value in torch.unique(radius, sorted=True):
        selections[f"radius/{int(value)}"] = radius == value
    sequence_decile = torch.arange(length) * 10 // length
    for decile in range(10):
        selection = sequence_decile == decile
        if bool(selection.any()):
            selections[f"ar_decile/{decile}"] = selection

    summaries = {}
    for name, selection in selections.items():
        summary = calibration_summary(samples, target, component_mask, selection)
        summary.update(support_summary(model, coordinates, selection))
        summaries[name] = summary

    report = {
        "version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint_step,
        "amplitude_transform": model.config.factorized_polar.amplitude_transform,
        "amplitude_transform_parameter": float(
            model.config.factorized_polar.amplitude_transform_parameter
        ),
        "log_epsilon": float(model.config.factorized_polar.log_epsilon),
        "images": args.num_images,
        "conditional_draws": args.draws,
        "solver_steps": args.steps,
        "positions": length,
        "seed": args.seed,
        "definition": (
            "Each absolute frequency is sampled repeatedly with its held-out true "
            "causal prefix. Energy score is E||X-y|| - 0.5 E||X-X'|| in the "
            "decoder's physical Cartesian coordinate. For a calibrated conditional "
            "sampler, pairwise squared spread / sample-target squared error tends to 1."
        ),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summaries"]["all"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
