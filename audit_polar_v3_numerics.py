"""Numerical audit for the physical polar-v3 direct-FFT objective."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from evaluate_continuous_checkpoint import load_model
from train_continuous import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--time_bins", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--component_reduction_override",
        choices=("active_mean", "fixed_dim"),
        default=None,
        help="Reproduce a historical objective reduction after implementation fixes.",
    )
    return parser.parse_args()


def tensor_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0],
            device=flat.device,
        ),
    ).cpu()
    names = ("p0", "p0.1", "p1", "p10", "p50", "p90", "p99", "p99.9", "p100")
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "rms": flat.square().mean().sqrt().item(),
        **{name: value.item() for name, value in zip(names, quantiles)},
    }


class ActivationMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.abs_max = 0.0

    def update(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        self.count += value.numel()
        self.total += value.sum().item()
        self.total_sq += value.square().sum().item()
        self.abs_max = max(self.abs_max, value.abs().max().item())

    def report(self) -> dict[str, float]:
        mean = self.total / max(self.count, 1)
        second = self.total_sq / max(self.count, 1)
        return {
            "mean": mean,
            "std": math.sqrt(max(second - mean * mean, 0.0)),
            "rms": math.sqrt(max(second, 0.0)),
            "abs_max": self.abs_max,
            "count": self.count,
        }


def grouped_report(
    group: torch.Tensor,
    values: dict[str, torch.Tensor],
    groups: int,
) -> list[dict[str, float]]:
    result = []
    for index in range(groups):
        selected = group == index
        if not bool(selected.any()):
            continue
        row: dict[str, float] = {"index": index, "count": int(selected.sum())}
        for name, value in values.items():
            row[name] = value[selected].float().mean().item()
        result.append(row)
    return result


def flat_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    return (
        (left * right).mean()
        / (left.square().mean().sqrt() * right.square().mean().sqrt()).clamp_min(1e-12)
    ).item()


def head_sensitivity_report(
    module: torch.nn.Module,
    captured: dict[str, torch.Tensor | None],
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    x = captured["x"]
    timestep = captured["timestep"]
    condition = captured["condition"]
    target_condition = captured["target_condition"]
    assert x is not None and timestep is not None and condition is not None
    generator = torch.Generator(device=device).manual_seed(seed)
    permutation = torch.randperm(x.shape[0], generator=generator, device=device)

    def evaluate(
        current_x: torch.Tensor,
        current_timestep: torch.Tensor,
        current_condition: torch.Tensor,
        current_target_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16
        ):
            return module(
                current_x,
                current_timestep,
                current_condition,
                target_condition=current_target_condition,
            ).float()

    baseline = evaluate(x, timestep, condition, target_condition)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=False
    ):
        fp32_compute = module(
            x.float(),
            timestep.float(),
            condition.float(),
            target_condition=(
                None if target_condition is None else target_condition.float()
            ),
        ).float()
    variants = {
        "fp32_compute": fp32_compute,
        "zero_state": evaluate(torch.zeros_like(x), timestep, condition, target_condition),
        "shuffled_state": evaluate(x[permutation], timestep, condition, target_condition),
        "shuffled_time": evaluate(x, timestep[permutation], condition, target_condition),
        "shuffled_trunk_condition": evaluate(
            x, timestep, condition[permutation], target_condition
        ),
    }
    if target_condition is not None:
        variants["shuffled_target_condition"] = evaluate(
            x, timestep, condition, target_condition[permutation]
        )
    baseline_rms = baseline.square().mean().sqrt().item()
    result: dict[str, Any] = {
        "examples": int(x.shape[0]),
        "baseline_output_rms": baseline_rms,
        "perturbations": {},
    }
    for name, value in variants.items():
        delta_rms = (baseline - value).square().mean().sqrt().item()
        result["perturbations"][name] = {
            "output_delta_rms": delta_rms,
            "delta_over_baseline_rms": delta_rms / max(baseline_rms, 1e-12),
            "output_correlation": flat_correlation(baseline, value),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, saved_args, checkpoint_step = load_model(args.checkpoint, device)
    if model.factorized_decoder is None:
        raise ValueError("checkpoint does not use the factorized polar decoder")
    if args.component_reduction_override is not None:
        model.factorized_decoder.base = replace(
            model.factorized_decoder.base,
            component_reduction=args.component_reduction_override,
        )
    model.eval()

    saved_args.smoke = False
    saved_args.synthetic_data = False
    saved_args.train_batch_size = args.batch_size
    saved_args.dataloader_num_workers = min(int(saved_args.dataloader_num_workers), 4)
    _, loader = make_dataloader(saved_args)

    activation_moments: dict[str, ActivationMoments] = defaultdict(ActivationMoments)
    captured_head_calls: dict[str, dict[str, torch.Tensor | None]] = {}
    handles = []

    def hook(name: str):
        def record(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            activation_moments[name].update(output)

        return record

    def capture_head_call(name: str):
        def record(_module, inputs, kwargs, _output):
            if name in captured_head_calls:
                return
            limit = min(int(inputs[0].shape[0]), 4096)
            captured_head_calls[name] = {
                "x": inputs[0][:limit].detach(),
                "timestep": inputs[1][:limit].detach(),
                "condition": inputs[2][:limit].detach(),
                "target_condition": (
                    None
                    if kwargs.get("target_condition") is None
                    else kwargs["target_condition"][:limit].detach()
                ),
            }

        return record

    modules = {
        "trunk/token_projection": model.token_proj,
        "trunk/final_norm": model.final_norm,
        "amplitude/input_projection": model.factorized_decoder.amplitude_net.input_proj,
        "amplitude/time_embedding": model.factorized_decoder.amplitude_net.time_embed,
        "amplitude/trunk_condition_embedding": (
            model.factorized_decoder.amplitude_net.cond_embed
        ),
        "amplitude/target_condition_embedding": (
            model.factorized_decoder.amplitude_net.target_condition_embed
        ),
        "amplitude/final_output": model.factorized_decoder.amplitude_net.final_layer,
        "phase/input_projection": model.factorized_decoder.phase_net.input_proj,
        "phase/time_embedding": model.factorized_decoder.phase_net.time_embed,
        "phase/trunk_condition_embedding": model.factorized_decoder.phase_net.cond_embed,
        "phase/target_condition_embedding": (
            model.factorized_decoder.phase_net.target_condition_embed
        ),
        "phase/final_output": model.factorized_decoder.phase_net.final_layer,
    }
    for index, block in enumerate(model.factorized_decoder.amplitude_net.res_blocks):
        modules[f"amplitude/resblock_{index}"] = block
    for index, block in enumerate(model.factorized_decoder.phase_net.res_blocks):
        modules[f"phase/resblock_{index}"] = block
    if model.factorized_decoder.sign_net is not None:
        modules["sign/logits"] = model.factorized_decoder.sign_net
    for name, module in modules.items():
        if module is None:
            continue
        handles.append(module.register_forward_hook(hook(name)))
    for name, module in (
        ("amplitude", model.factorized_decoder.amplitude_net),
        ("phase", model.factorized_decoder.phase_net),
    ):
        handles.append(
            module.register_forward_hook(capture_head_call(name), with_kwargs=True)
        )

    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    seen = 0
    positions = torch.arange(model.codec.seq_len_int, device=device)
    amplitude_scale = model.factorized_amplitude_scale(positions)[None]
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            remaining = args.examples - seen
            if remaining <= 0:
                break
            images = images[:remaining].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                tokens = model.codec.encode(images)
                output = model(tokens, corrupt=False)
            batch_size = images.shape[0]
            seen += batch_size
            raw = model.factorized_cartesian_target(tokens, positions)
            log_amplitude, phase, relative_amplitude = (
                model.factorized_decoder.target_coordinates(raw, amplitude_scale)
            )
            radius = model.codec.radius_bin[None].expand(batch_size, -1).reshape(-1)
            for key in (
                "amplitude_timesteps",
                "phase_timesteps",
                "amplitude_per_example",
                "phase_per_example",
                "sign_per_example",
                "cartesian_per_example",
                "cartesian_raw_per_example",
                "target_energy_per_example",
                "target_energy_sum_per_example",
                "active_component_count_per_example",
            ):
                collected[key].append(output[key].detach().float().cpu())
            collected["radius"].append(radius.cpu())
            collected["log_amplitude"].append(log_amplitude.detach().float().cpu())
            collected["phase"].append(phase.detach().float().cpu())
            collected["amplitude"].append(relative_amplitude.detach().float().cpu())

    for handle in handles:
        handle.remove()
    head_sensitivity = {
        name: head_sensitivity_report(
            (
                model.factorized_decoder.amplitude_net
                if name == "amplitude"
                else model.factorized_decoder.phase_net
            ),
            captured,
            device=device,
            seed=args.seed + index + 1,
        )
        for index, (name, captured) in enumerate(captured_head_calls.items())
    }
    data = {key: torch.cat(values) for key, values in collected.items()}
    radius = data["radius"].long()
    max_radius = int(model.codec.max_radius_bin)
    config = model.config.factorized_polar
    weights = {
        "amplitude": float(config.amplitude_loss_weight),
        "phase": float(config.phase_loss_weight),
        "sign": float(config.phase_loss_weight),
        "cartesian": float(config.cartesian_loss_weight),
    }
    component_values = {
        "amplitude": data["amplitude_per_example"],
        "phase": data["phase_per_example"],
        "sign": data["sign_per_example"],
        "cartesian": data["cartesian_per_example"],
    }
    weighted = {
        name: values * weights[name] for name, values in component_values.items()
    }
    total_weighted = sum(value.sum() for value in weighted.values()).clamp_min(1e-12)
    objective_share = {
        name: (value.sum() / total_weighted).item() for name, value in weighted.items()
    }

    ring_rows: list[dict[str, Any]] = []
    coordinate_radius = radius[:, None].expand(-1, 3).reshape(-1)
    flattened_log_amplitude = data["log_amplitude"].reshape(-1)
    for ring in range(max_radius + 1):
        selected = radius == ring
        coordinate_selected = coordinate_radius == ring
        if not bool(selected.any()):
            continue
        row: dict[str, Any] = {
            "radius_bin": ring,
            "coefficient_examples": int(selected.sum()),
            "coefficient_fraction": selected.float().mean().item(),
            "target_energy_mean": data["target_energy_per_example"][selected].mean().item(),
            "target_energy_share": (
                data["target_energy_per_example"][selected].sum()
                / data["target_energy_per_example"].sum().clamp_min(1e-12)
            ).item(),
            "exact_active_coordinate_energy_share": (
                data["target_energy_sum_per_example"][selected].sum()
                / data["target_energy_sum_per_example"].sum().clamp_min(1e-12)
            ).item(),
            "active_component_count_mean": (
                data["active_component_count_per_example"][selected].mean().item()
            ),
        }
        ring_total = sum(value[selected].sum() for value in weighted.values())
        row["total_objective_share"] = (ring_total / total_weighted).item()
        for name in component_values:
            component_total = weighted[name].sum().clamp_min(1e-12)
            row[f"{name}_mean"] = component_values[name][selected].mean().item()
            row[f"{name}_objective_share"] = (
                weighted[name][selected].sum() / component_total
            ).item()
        coordinate = flattened_log_amplitude[coordinate_selected]
        coordinate_mean = coordinate.mean()
        coordinate_std = coordinate.std(unbiased=False)
        coordinate_rms = coordinate.square().mean().sqrt()
        row.update(
            {
                "standardized_log_amplitude_mean": coordinate_mean.item(),
                "standardized_log_amplitude_std": coordinate_std.item(),
                "standardized_log_amplitude_rms": coordinate_rms.item(),
                "amplitude_flow_t_at_second_moment_snr_one": 1.0
                / (1.0 + coordinate_rms.item()),
                "amplitude_flow_t_at_centered_snr_one": 1.0
                / (1.0 + coordinate_std.item()),
            }
        )
        ring_rows.append(row)

    bins = int(args.time_bins)
    amp_bin = torch.div(
        data["amplitude_timesteps"].long() * bins,
        model.factorized_decoder.num_train_timesteps,
        rounding_mode="floor",
    ).clamp(0, bins - 1)
    phase_bin = torch.div(
        data["phase_timesteps"].long() * bins,
        model.factorized_decoder.num_train_timesteps,
        rounding_mode="floor",
    ).clamp(0, bins - 1)
    amplitude_time = grouped_report(
        amp_bin,
        {
            "amplitude_loss": data["amplitude_per_example"],
            "cartesian_loss_conditioned_on_amplitude_time": data["cartesian_per_example"],
        },
        bins,
    )
    phase_time = grouped_report(
        phase_bin,
        {
            "phase_loss": data["phase_per_example"],
            "cartesian_loss_conditioned_on_phase_time": data["cartesian_per_example"],
        },
        bins,
    )
    radius_time_rows = []
    for ring in range(max_radius + 1):
        selected = radius == ring
        if not bool(selected.any()):
            continue
        radius_time_rows.append(
            {
                "radius_bin": ring,
                "amplitude_time": grouped_report(
                    amp_bin[selected],
                    {
                        "amplitude_loss": data["amplitude_per_example"][selected],
                        "cartesian_loss": data["cartesian_per_example"][selected],
                    },
                    bins,
                ),
                "phase_time": grouped_report(
                    phase_bin[selected],
                    {
                        "phase_loss": data["phase_per_example"][selected],
                        "cartesian_loss": data["cartesian_per_example"][selected],
                    },
                    bins,
                ),
            }
        )

    phase = data["phase"]
    report = {
        "version": 3,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "examples": seen,
        "objective_weights": weights,
        "component_reduction": model.factorized_decoder.base.component_reduction,
        "objective_component_share": objective_share,
        "coordinate_distributions": {
            "standardized_log_amplitude": tensor_summary(data["log_amplitude"]),
            "physical_amplitude": tensor_summary(data["amplitude"]),
            "phase_cosine": tensor_summary(torch.cos(phase)),
            "phase_sine": tensor_summary(torch.sin(phase)),
            "phase_resultant_length_rgb": [
                math.hypot(
                    torch.cos(phase[..., channel]).mean().item(),
                    torch.sin(phase[..., channel]).mean().item(),
                )
                for channel in range(3)
            ],
        },
        "activation_moments": {
            name: moments.report() for name, moments in activation_moments.items()
        },
        "head_output_sensitivity": head_sensitivity,
        "loss_by_amplitude_time_bin": amplitude_time,
        "loss_by_phase_time_bin": phase_time,
        "loss_by_radius_and_time": radius_time_rows,
        "loss_and_coordinates_by_radius": ring_rows,
        "notes": {
            "phase_snr": (
                "Geodesic phase flow starts from the uniform Haar measure on S1. "
                "Its marginal angle remains uniform, so ordinary Euclidean marginal "
                "SNR is not defined; phase time must be audited by conditional angular "
                "error and Cartesian endpoint loss."
            ),
            "independent_times": (
                "Amplitude and phase draw independent flow times. Aggregate polar loss "
                "must not be assigned to the amplitude timestep alone."
            ),
            "energy_reductions": (
                "target_energy_share follows the factorized objective's active-component "
                "mean. exact_active_coordinate_energy_share instead sums every independent "
                "Cartesian coordinate once; disagreement identifies the extra weight that "
                "active_mean gives self-conjugate three-component coefficients."
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
