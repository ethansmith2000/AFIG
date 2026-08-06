"""Oracle gate for p(phase field | complete true amplitude field).

This intentionally is not an unconditional generator.  It removes amplitude
uncertainty and causal phase rollout, then asks whether a joint intrinsic phase
model can sample a coherent image when given the complete held-out amplitude
spectrum.  A visual pass justifies a later amplitude-first generator; a failure
closes that expensive composition before it is built.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torchvision.utils import save_image

from causal_transformer import CausalTransformerBlock, build_rope_tables
from diffusion_decoder import FinalLayer, TimestepEmbedder
from factorized_polar_decoder import wrap_angle
from frequency import FrequencyCodec
from train_continuous import build_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--qk_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--phase_gate", type=float, default=0.1)
    parser.add_argument("--cartesian_loss_weight", type=float, default=0.1)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--checkpointing_steps", type=int, default=2500)
    parser.add_argument("--num_validation_images", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument(
        "--save_final_checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def pad_and_group(values: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack [B,L,C] into [B,ceil(L/S),S*C] with zero orbit padding."""
    batch, length, channels = values.shape
    groups = math.ceil(length / group_size)
    padded = values.new_zeros(batch, groups * group_size, channels)
    padded[:, :length] = values
    return padded.reshape(batch, groups, group_size * channels)


def ungroup_and_trim(
    values: torch.Tensor,
    group_size: int,
    length: int,
    channels: int,
) -> torch.Tensor:
    return values.reshape(values.shape[0], -1, channels)[:, :length]


def grouped_frequency_coordinates(codec: FrequencyCodec, group_size: int) -> torch.Tensor:
    coordinates = torch.stack([codec.ky_signed, codec.kx_signed], dim=-1).float()
    groups = math.ceil(codec.seq_len_int / group_size)
    padded = coordinates.new_zeros(groups * group_size, 2)
    active = coordinates.new_zeros(groups * group_size, 1)
    padded[: codec.seq_len_int] = coordinates
    active[: codec.seq_len_int] = 1.0
    padded = padded.reshape(groups, group_size, 2)
    active = active.reshape(groups, group_size, 1)
    return (padded * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


class JointPhaseOracle(nn.Module):
    """Bidirectional Riemannian flow over phases conditioned on all amplitudes."""

    def __init__(
        self,
        *,
        sequence_length: int,
        group_size: int,
        group_coordinates: torch.Tensor,
        width: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        qk_norm: bool,
        rope_base: float,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if group_coordinates.shape != (math.ceil(sequence_length / group_size), 2):
            raise ValueError("group_coordinates has the wrong shape")
        self.sequence_length = int(sequence_length)
        self.group_size = int(group_size)
        self.groups = math.ceil(sequence_length / group_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.phase_projection = nn.Linear(group_size * 6, width)
        self.amplitude_projection = nn.Linear(group_size * 3, width)
        self.amplitude_condition = nn.Sequential(
            nn.Linear(group_size * 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.position = nn.Parameter(torch.zeros(self.groups, width))
        nn.init.normal_(self.position, std=0.02)
        self.time_embed = TimestepEmbedder(width)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=0.0,
                    conditional_film=True,
                    causal=False,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(width, group_size * 3)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        head_dim = width // num_heads
        rope_cos, rope_sin = build_rope_tables(
            group_coordinates.float(), head_dim, base=float(rope_base)
        )
        self.register_buffer("rope_cos", rope_cos, persistent=True)
        self.register_buffer("rope_sin", rope_sin, persistent=True)

    def velocity(
        self,
        phase: torch.Tensor,
        standardized_log_amplitude: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        if phase.shape != standardized_log_amplitude.shape:
            raise ValueError("phase and amplitude coordinates must have identical shape")
        if phase.shape[1:] != (self.sequence_length, 3):
            raise ValueError(
                f"Expected [B,{self.sequence_length},3], got {tuple(phase.shape)}"
            )
        phasor = torch.cat([torch.cos(phase.float()), torch.sin(phase.float())], dim=-1)
        phase_group = pad_and_group(phasor, self.group_size)
        amplitude_group = pad_and_group(
            standardized_log_amplitude.float(), self.group_size
        )
        hidden = (
            self.phase_projection(phase_group.to(self.phase_projection.weight.dtype))
            + self.amplitude_projection(
                amplitude_group.to(self.amplitude_projection.weight.dtype)
            )
            + self.position[None].to(self.phase_projection.weight.dtype)
        )
        condition = (
            self.time_embed(flow_time.float() * 999.0)[:, None]
            + self.amplitude_condition(
                amplitude_group.to(self.amplitude_projection.weight.dtype)
            )
        )
        rope = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda state, cond, block=layer: block(
                        state, condition=cond, rope=rope
                    )[0],
                    hidden,
                    condition,
                    use_reentrant=False,
                )
            else:
                hidden, _ = layer(hidden, condition=condition, rope=rope)
        grouped = self.final_layer(hidden, condition)
        return ungroup_and_trim(
            grouped, self.group_size, self.sequence_length, channels=3
        ).float()

    def loss(
        self,
        target_phase: torch.Tensor,
        standardized_log_amplitude: torch.Tensor,
        relative_amplitude: torch.Tensor,
        physical_amplitude: torch.Tensor,
        is_self_conjugate: torch.Tensor,
        phase_gate: float,
        cartesian_loss_weight: float,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch = target_phase.shape[0]
        device = target_phase.device
        flow_time = torch.rand(batch, device=device, generator=generator)
        base_phase = (
            torch.rand(
                target_phase.shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * (2.0 * math.pi)
            - math.pi
        )
        target_velocity = wrap_angle(target_phase.float() - base_phase)
        time_view = flow_time[:, None, None]
        noisy_phase = wrap_angle(base_phase + time_view * target_velocity)
        predicted_velocity = self.velocity(
            noisy_phase, standardized_log_amplitude, flow_time
        )
        gate = relative_amplitude.float().square() / (
            relative_amplitude.float().square() + float(phase_gate) ** 2
        )
        angular_error = predicted_velocity - target_velocity
        phase_per_image = (gate * angular_error.square()).sum(dim=(1, 2)) / gate.sum(
            dim=(1, 2)
        ).clamp_min(1e-6)

        predicted_endpoint = wrap_angle(
            noisy_phase + (1.0 - time_view) * predicted_velocity
        )
        predicted_real = physical_amplitude.float() * torch.cos(predicted_endpoint)
        predicted_imag = physical_amplitude.float() * torch.sin(predicted_endpoint)
        predicted_imag = predicted_imag * (
            ~is_self_conjugate
        )[None, :, None].to(predicted_imag.dtype)
        target_real = physical_amplitude.float() * torch.cos(target_phase.float())
        target_imag = physical_amplitude.float() * torch.sin(target_phase.float())
        target_imag = target_imag * (
            ~is_self_conjugate
        )[None, :, None].to(target_imag.dtype)
        cartesian_error = torch.cat(
            [predicted_real - target_real, predicted_imag - target_imag], dim=-1
        )
        component_mask = torch.ones_like(cartesian_error)
        component_mask[:, is_self_conjugate, 3:] = 0.0
        cartesian_per_image = (
            (cartesian_error.square() * component_mask).sum(dim=(1, 2))
            / component_mask.sum(dim=(1, 2)).clamp_min(1.0)
        )
        target_cartesian = torch.cat([target_real, target_imag], dim=-1)
        global_energy = (
            (target_cartesian.square() * component_mask).sum()
            / component_mask.sum().clamp_min(1.0)
        ).detach().clamp_min(1e-8)
        cartesian_per_image = cartesian_per_image / global_energy
        total = phase_per_image + float(cartesian_loss_weight) * cartesian_per_image
        return {
            "loss": total.mean(),
            "phase_loss": phase_per_image.mean().detach(),
            "cartesian_loss": cartesian_per_image.mean().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        standardized_log_amplitude: torch.Tensor,
        *,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        phase = (
            torch.rand(
                standardized_log_amplitude.shape,
                device=standardized_log_amplitude.device,
                dtype=torch.float32,
                generator=generator,
            )
            * (2.0 * math.pi)
            - math.pi
        )
        dt = 1.0 / float(steps)
        for index in range(steps):
            t = torch.full(
                (phase.shape[0],),
                index / float(steps),
                device=phase.device,
                dtype=torch.float32,
            )
            velocity = self.velocity(phase, standardized_log_amplitude, t)
            proposed = wrap_angle(phase + dt * velocity)
            if index + 1 < steps:
                next_t = torch.full_like(t, (index + 1) / float(steps))
                next_velocity = self.velocity(
                    proposed, standardized_log_amplitude, next_t
                )
                phase = wrap_angle(phase + 0.5 * dt * (velocity + next_velocity))
            else:
                phase = proposed
        return phase


def load_factorized_interface(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[FrequencyCodec, torch.Tensor, torch.Tensor, float, Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = Namespace(**payload["args"])
    config = build_model_config(saved_args)
    if not config.factorized_polar.enabled:
        raise ValueError("source checkpoint must use factorized-polar decoding")
    codec = FrequencyCodec(config.codec)
    codec.load_exported(payload["codec"])
    state = payload["model"]
    mean = state["factorized_decoder.amplitude_coordinate_mean"].float().reshape(3)
    std = state["factorized_decoder.amplitude_coordinate_std"].float().reshape(3)
    metadata: Dict[str, object] = {
        "source_checkpoint": str(checkpoint_path),
        "codec_config": config.codec.fingerprint(),
        "factorized_polar_config": config.factorized_polar.fingerprint(),
    }
    del payload, state
    gc.collect()
    return codec.to(device).eval(), mean.to(device), std.to(device), float(
        config.factorized_polar.log_epsilon
    ), metadata


def amplitude_scale(codec: FrequencyCodec) -> torch.Tensor:
    positions = torch.arange(codec.seq_len_int, device=codec.ky.device)
    if codec.uses_orbit_statistics:
        rms = codec.orbit_uncentered_rms()[positions, :3]
        imag_active = codec.component_mask[positions, 3:]
        return rms * torch.sqrt(1.0 + imag_active)
    return codec.channel_amplitude_scale()[codec.radius_bin[positions]]


@torch.no_grad()
def polar_coordinates(
    codec: FrequencyCodec,
    images: torch.Tensor,
    coordinate_mean: torch.Tensor,
    coordinate_std: torch.Tensor,
    log_epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = codec.encode_raw(images)
    real, imag = raw[..., :3], raw[..., 3:]
    physical_amplitude = torch.sqrt(real.float().square() + imag.float().square())
    scale = amplitude_scale(codec)[None].to(physical_amplitude)
    relative = physical_amplitude / scale.clamp_min(1e-8)
    standardized_log = (
        torch.log(relative + float(log_epsilon))
        - coordinate_mean[None, None].to(relative)
    ) / coordinate_std[None, None].to(relative)
    phase = torch.atan2(imag.float(), real.float())
    return standardized_log, phase, relative, physical_amplitude


@torch.no_grad()
def decode_with_phase(
    codec: FrequencyCodec,
    physical_amplitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    real = physical_amplitude.float() * torch.cos(phase.float())
    imag = physical_amplitude.float() * torch.sin(phase.float())
    imag = imag * (~codec.is_self_conjugate)[None, :, None].to(imag.dtype)
    return codec.decode_raw(torch.cat([real, imag], dim=-1))


def save_checkpoint(
    path: Path,
    model: JointPhaseOracle,
    args: argparse.Namespace,
    step: int,
    metadata: Dict[str, object],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "step": step,
            "interface": metadata,
        },
        path,
    )


@torch.no_grad()
def save_preview_grid(
    *,
    model: JointPhaseOracle,
    codec: FrequencyCodec,
    validation: torch.Tensor,
    coordinate_mean: torch.Tensor,
    coordinate_std: torch.Tensor,
    log_epsilon: float,
    inference_steps: int,
    sample_seed: int,
    output_path: Path,
) -> None:
    device = validation.device
    val_u, val_phase, _, val_amp = polar_coordinates(
        codec,
        validation,
        coordinate_mean,
        coordinate_std,
        log_epsilon,
    )
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        sampled_phase = model.sample(
            val_u,
            steps=inference_steps,
            generator=generator,
        )
    random_generator = torch.Generator(device=device).manual_seed(314159)
    random_phase = (
        torch.rand(
            val_phase.shape,
            device=device,
            generator=random_generator,
        )
        * (2.0 * math.pi)
        - math.pi
    )
    true_decode = decode_with_phase(codec, val_amp, val_phase)
    random_decode = decode_with_phase(codec, val_amp, random_phase)
    sampled_decode = decode_with_phase(codec, val_amp, sampled_phase)
    save_image(
        torch.cat([validation, true_decode, random_decode, sampled_decode]).clamp(0, 1),
        output_path,
        nrow=validation.shape[0],
    )


def main() -> None:
    args = parse_args()
    if args.group_size <= 0 or args.phase_gate <= 0:
        raise ValueError("group_size and phase_gate must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    codec, coordinate_mean, coordinate_std, log_epsilon, metadata = (
        load_factorized_interface(args.source_checkpoint, device)
    )
    coordinates = grouped_frequency_coordinates(codec, args.group_size).to(device)
    model = JointPhaseOracle(
        sequence_length=codec.seq_len_int,
        group_size=args.group_size,
        group_coordinates=coordinates,
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        qk_norm=args.qk_norm,
        rope_base=args.rope_base,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        str(args.data_root), train=True, download=False, transform=train_transform
    )
    validation_dataset = torchvision.datasets.CIFAR10(
        str(args.data_root), train=False, download=False, transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
        pin_memory=True,
    )
    validation = torch.stack(
        [validation_dataset[index][0] for index in range(args.num_validation_images)]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def learning_rate_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            args.max_train_steps - args.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    history = []
    step = 0
    autocast_enabled = device.type == "cuda"
    while step < args.max_train_steps:
        for images, _ in loader:
            if step >= args.max_train_steps:
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                amp_u, phase, relative, physical_amp = polar_coordinates(
                    codec,
                    images,
                    coordinate_mean,
                    coordinate_std,
                    log_epsilon,
                )
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * learning_rate_scale(step)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                losses = model.loss(
                    phase,
                    amp_u,
                    relative,
                    physical_amp,
                    codec.is_self_conjugate,
                    args.phase_gate,
                    args.cartesian_loss_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.logging_steps == 0:
                record = {
                    "step": step,
                    "loss": float(losses["loss"].detach()),
                    "phase_loss": float(losses["phase_loss"]),
                    "cartesian_loss": float(losses["cartesian_loss"]),
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(record)
                print(json.dumps(record), flush=True)

            if args.checkpointing_steps and step % args.checkpointing_steps == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_{step}.pt",
                    model,
                    args,
                    step,
                    metadata,
                )
            if args.preview_steps and step % args.preview_steps == 0:
                model.eval()
                save_preview_grid(
                    model=model,
                    codec=codec,
                    validation=validation,
                    coordinate_mean=coordinate_mean,
                    coordinate_std=coordinate_std,
                    log_epsilon=log_epsilon,
                    inference_steps=args.num_inference_steps,
                    sample_seed=args.seed + 100000,
                    output_path=args.output_dir / f"samples_{step}.png",
                )
                model.train()

    model.eval()
    save_preview_grid(
        model=model,
        codec=codec,
        validation=validation,
        coordinate_mean=coordinate_mean,
        coordinate_std=coordinate_std,
        log_epsilon=log_epsilon,
        inference_steps=args.num_inference_steps,
        sample_seed=54321,
        output_path=args.output_dir / f"samples_{step}_fresh_54321.png",
    )
    if args.save_final_checkpoint:
        save_checkpoint(
            args.output_dir / "checkpoint_final.pt", model, args, step, metadata
        )
    (args.output_dir / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "source_checkpoint": str(args.source_checkpoint),
                "amplitude_coordinate_mean": coordinate_mean.cpu().tolist(),
                "amplitude_coordinate_std": coordinate_std.cpu().tolist(),
                "log_epsilon": log_epsilon,
                "rows": ["reference", "true_phase", "uniform_phase", "sampled_phase"],
            },
            indent=2,
        )
        + "\n"
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
