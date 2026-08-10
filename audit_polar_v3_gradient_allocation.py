"""Audit polar-v3 head-output gradient allocation by radius and flow time."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--time_bins", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--component_reduction",
        choices=("active_mean", "fixed_dim"),
        default="active_mean",
    )
    return parser.parse_args()


def allocation(values: torch.Tensor, groups: torch.Tensor) -> list[dict[str, Any]]:
    total = values.sum().clamp_min(1e-30)
    rows = []
    for group in range(int(groups.max().item()) + 1):
        selected = groups == group
        if bool(selected.any()):
            rows.append(
                {
                    "index": group,
                    "count": int(selected.sum().item()),
                    "mean_gradient_energy": values[selected].mean().item(),
                    "gradient_energy_share": (values[selected].sum() / total).item(),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, saved_args, checkpoint_step = load_model(args.checkpoint, device)
    if model.factorized_decoder is None:
        raise ValueError("checkpoint does not use the factorized polar decoder")
    model.factorized_decoder.base = replace(
        model.factorized_decoder.base,
        component_reduction=args.component_reduction,
    )
    model.eval()

    saved_args.smoke = False
    saved_args.synthetic_data = False
    saved_args.train_batch_size = args.batch_size
    saved_args.dataloader_num_workers = min(int(saved_args.dataloader_num_workers), 4)
    _, loader = make_dataloader(saved_args)
    batch = next(iter(loader))
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    images = images[: args.batch_size].to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        tokens = model.codec.encode(images)

    original_config = model.factorized_decoder.config
    component_weights = {
        "amplitude_native": (1.0, 0.0, 0.0),
        "phase_native": (0.0, 1.0, 0.0),
        "cartesian_endpoint": (0.0, 0.0, 1.0),
        "configured_total": (
            float(original_config.amplitude_loss_weight),
            float(original_config.phase_loss_weight),
            float(original_config.cartesian_loss_weight),
        ),
    }
    report_components: dict[str, Any] = {}
    for component_index, (name, weights) in enumerate(component_weights.items()):
        amplitude_weight, phase_weight, cartesian_weight = weights
        model.factorized_decoder.config = replace(
            original_config,
            amplitude_loss_weight=amplitude_weight,
            phase_loss_weight=phase_weight,
            cartesian_loss_weight=cartesian_weight,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture(key: str):
            def record(_module, _inputs, output):
                captured[key] = output

            return record

        handles = [
            model.factorized_decoder.amplitude_net.register_forward_hook(
                capture("amplitude_output")
            ),
            model.factorized_decoder.phase_net.register_forward_hook(
                capture("phase_output")
            ),
            model.final_norm.register_forward_hook(capture("trunk_output")),
        ]
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(tokens, corrupt=False)
        for handle in handles:
            handle.remove()
        targets = [
            captured["amplitude_output"],
            captured["phase_output"],
            captured["trunk_output"],
        ]
        gradients = torch.autograd.grad(
            output["loss"], targets, allow_unused=True, retain_graph=False
        )
        radius = output["radius_bin"].detach().long()
        amplitude_time = output["amplitude_timesteps"].detach().long()
        phase_time = output["phase_timesteps"].detach().long()
        time_scale = args.time_bins / model.factorized_decoder.num_train_timesteps
        amplitude_bin = (amplitude_time.float() * time_scale).long().clamp(
            0, args.time_bins - 1
        )
        phase_bin = (phase_time.float() * time_scale).long().clamp(
            0, args.time_bins - 1
        )
        component_report: dict[str, Any] = {
            "weights": {
                "amplitude": amplitude_weight,
                "phase_and_sign": phase_weight,
                "cartesian": cartesian_weight,
            },
            "loss": output["loss"].detach().float().item(),
        }
        for target_name, gradient in zip(
            ("amplitude_output", "phase_output", "trunk_output"), gradients
        ):
            if gradient is None:
                component_report[target_name] = None
                continue
            energy = gradient.detach().float().square().sum(-1).reshape(-1)
            target_report: dict[str, Any] = {
                "total_gradient_energy": energy.sum().item(),
                "gradient_rms": gradient.detach().float().square().mean().sqrt().item(),
                "by_radius": allocation(energy, radius),
            }
            if target_name == "amplitude_output":
                target_report["by_time"] = allocation(energy, amplitude_bin)
            elif target_name == "phase_output":
                target_report["by_time"] = allocation(energy, phase_bin)
            component_report[target_name] = target_report
        report_components[name] = component_report

    model.factorized_decoder.config = original_config
    report = {
        "version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "batch_size": int(images.shape[0]),
        "component_reduction": args.component_reduction,
        "notes": (
            "Shares are squared gradients with respect to the head outputs, not "
            "parameter-gradient shares. All component passes reuse identical data, "
            "flow times, and base noise. phase_native includes the sign weight, but "
            "the sign head has no gradient to the continuous phase output."
        ),
        "components": report_components,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
