"""Measure phase-head dependence on completed amplitude by frequency.

Unlike the original pooled sensitivity audit, this keeps the learned target
slot fixed and shuffles only completed amplitude between held-out images at the
same absolute frequency.  The Transformer history, phase state, and flow time
remain aligned unless their named control is being measured.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from evaluate_continuous_checkpoint import load_model
from factorized_polar_decoder import wrap_angle
from train_continuous import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args()


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    return ((left * right).mean() / denominator.clamp_min(1e-12)).item()


def comparison(
    baseline: torch.Tensor,
    variant: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float | int]:
    baseline = baseline[selected].float()
    variant = variant[selected].float()
    target = target[selected].float()
    baseline_rms = baseline.square().mean().sqrt().item()
    delta_rms = (baseline - variant).square().mean().sqrt().item()
    baseline_mse = (baseline - target).square().mean().item()
    variant_mse = (variant - target).square().mean().item()
    return {
        "tokens": int(selected.sum()),
        "baseline_output_rms": baseline_rms,
        "output_delta_rms": delta_rms,
        "delta_over_baseline_rms": delta_rms / max(baseline_rms, 1e-12),
        "output_correlation": correlation(baseline, variant),
        "baseline_phase_velocity_mse": baseline_mse,
        "variant_phase_velocity_mse": variant_mse,
        "phase_velocity_mse_ratio": variant_mse / max(baseline_mse, 1e-12),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least two for within-position shuffling")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model, saved_args, checkpoint_step = load_model(args.checkpoint, device)
    if model.factorized_decoder is None:
        raise ValueError("checkpoint does not use a factorized polar decoder")
    if model.config.factorized_polar.phase_process != "geodesic_flow":
        raise ValueError("this diagnostic currently implements geodesic phase flow")

    saved_args.smoke = False
    saved_args.synthetic_data = False
    saved_args.train_batch_size = args.batch_size
    saved_args.dataloader_num_workers = 0
    dataset, _ = make_dataloader(saved_args)
    if len(dataset) < args.batch_size:
        raise ValueError("dataset is smaller than the requested held-out batch")
    # Training excludes the deterministic tail panel. Use its final examples.
    images = torch.stack(
        [dataset[index][0] for index in range(len(dataset) - args.batch_size, len(dataset))]
    ).to(device)

    use_autocast = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        tokens = model.codec.encode(images)
        history = tokens[:, :-1]
        hidden, _ = model.forward_backbone(
            model.embed_tokens(history, include_bos=True), use_cache=False
        )
        length = tokens.shape[1]
        positions = torch.arange(length, device=device)
        slot = model.prediction_slot_condition(
            positions, batch_size=args.batch_size, dtype=hidden.dtype
        )
        raw = model.factorized_cartesian_target(tokens, positions)
        amplitude, target_phase, _ = model.factorized_decoder.target_coordinates(
            raw, model.factorized_amplitude_scale(positions)[None]
        )

        timestep = torch.randint(
            0,
            model.factorized_decoder.num_train_timesteps,
            (args.batch_size, length),
            generator=generator,
            device=device,
        )
        t = (timestep.float() + 0.5) / model.factorized_decoder.num_train_timesteps
        base_phase = (
            torch.rand(
                args.batch_size,
                length,
                3,
                generator=generator,
                device=device,
            )
            * (2.0 * math.pi)
            - math.pi
        )
        angular_target = wrap_angle(target_phase.float() - base_phase)
        noisy_phase = wrap_angle(base_phase + t[..., None] * angular_target)
        phase_state = torch.cat(
            [torch.cos(noisy_phase), torch.sin(noisy_phase)], dim=-1
        )

        batch_permutation = torch.stack(
            [
                torch.randperm(args.batch_size, generator=generator, device=device)
                for _ in range(length)
            ],
            dim=1,
        )
        position_index = positions[None].expand(args.batch_size, -1)
        shuffled_amplitude = amplitude[batch_permutation, position_index]
        shuffled_hidden = hidden[batch_permutation, position_index]

        def phase_output(
            current_hidden: torch.Tensor,
            current_amplitude: torch.Tensor,
        ) -> torch.Tensor:
            return model.factorized_decoder.phase_net(
                phase_state.reshape(-1, 6).to(hidden.dtype),
                (t * (model.factorized_decoder.num_train_timesteps - 1)).reshape(-1),
                current_hidden.reshape(-1, hidden.shape[-1]),
                target_condition=torch.cat(
                    [slot, current_amplitude.to(slot.dtype)], dim=-1
                ).reshape(-1, slot.shape[-1] + 3),
            ).reshape(args.batch_size, length, 3).float()

        baseline = phase_output(hidden, amplitude)
        variants = {
            "amplitude_same_frequency_shuffled": phase_output(
                hidden, shuffled_amplitude
            ),
            "amplitude_global_mean": phase_output(
                hidden, torch.zeros_like(amplitude)
            ),
            "trunk_same_frequency_shuffled": phase_output(
                shuffled_hidden, amplitude
            ),
        }

    radius = model.codec.radius_bin.to(device)
    all_selected = torch.ones(
        args.batch_size, length, device=device, dtype=torch.bool
    )
    report: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "held_out_examples": args.batch_size,
        "definition": (
            "Completed amplitude is shuffled only between held-out images at the "
            "same absolute frequency; learned slot, h, phase state, and time stay fixed."
        ),
        "overall": {
            name: comparison(baseline, value, angular_target, all_selected)
            for name, value in variants.items()
        },
        "by_radius": [],
        "by_position_decile": [],
        "by_phase_time_decile": [],
    }
    for ring in range(int(model.codec.max_radius_bin) + 1):
        selected = (radius == ring)[None].expand(args.batch_size, -1)
        if not bool(selected.any()):
            continue
        report["by_radius"].append(
            {
                "radius": ring,
                **{
                    name: comparison(baseline, value, angular_target, selected)
                    for name, value in variants.items()
                },
            }
        )
    for decile in range(10):
        start = length * decile // 10
        end = length * (decile + 1) // 10
        selected = torch.zeros_like(all_selected)
        selected[:, start:end] = True
        report["by_position_decile"].append(
            {
                "decile": decile,
                "position_start": start,
                "position_end_exclusive": end,
                **{
                    name: comparison(baseline, value, angular_target, selected)
                    for name, value in variants.items()
                },
            }
        )
        time_selected = (
            torch.div(
                timestep * 10,
                model.factorized_decoder.num_train_timesteps,
                rounding_mode="floor",
            ).clamp(0, 9)
            == decile
        )
        report["by_phase_time_decile"].append(
            {
                "decile": decile,
                **{
                    name: comparison(
                        baseline, value, angular_target, time_selected
                    )
                    for name, value in variants.items()
                },
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
