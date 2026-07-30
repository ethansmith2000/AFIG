#!/usr/bin/env python3
"""Train the minimal continuous AR model on frozen target-12 AE latents."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from causal_transformer import CausalTransformerConfig
from diffusion_decoder import DiffusionDecoderConfig
from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_latent_continuous import (
    LATENT_SEQUENCE_LENGTH,
    LatentContinuousConfig,
    LatentContinuousModel,
    latent_config_from_dict,
)
from train_autoencoder import make_dataset, reconstruction_metrics
from train_continuous import ModelEMA


def log_metrics(
    accelerator: Accelerator,
    output_dir: str,
    metrics: Dict[str, float],
    step: int,
) -> None:
    accelerator.log(metrics, step=step)
    if accelerator.is_main_process:
        record = {"step": int(step), **metrics}
        with open(
            os.path.join(output_dir, "metrics.jsonl"), "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output_dir", default="latent_continuous_runs/default")
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_end_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diffusion_width", type=int, default=512)
    parser.add_argument("--diffusion_depth", type=int, default=3)
    parser.add_argument("--diffusion_batch_mul", type=int, default=4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--objective", choices=["ddpm", "flow"], default="ddpm")
    parser.add_argument(
        "--prediction_type",
        choices=["epsilon", "v_prediction", "x0"],
        default="x0",
    )
    parser.add_argument("--context_dropout_probability", type=float, default=0.1)
    parser.add_argument("--transformer_metadata_film", action="store_true")
    parser.add_argument("--cfg_norm_match", action="store_true")
    parser.add_argument(
        "--latent_loss_weighting",
        choices=["unweighted", "raw_variance", "decoder_sensitivity"],
        default="unweighted",
    )
    parser.add_argument("--latent_loss_weights", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--diagnostic_steps", type=int, default=250)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--preview_images", type=int, default=8)
    parser.add_argument("--checkpointing_steps", type=int, default=2500)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--tracker_project_name", default="afig-latent-continuous")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def build_model_config(args: argparse.Namespace, metadata_dim: int) -> LatentContinuousConfig:
    transformer = CausalTransformerConfig(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        max_seq_len=LATENT_SEQUENCE_LENGTH,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    diffusion = DiffusionDecoderConfig(
        target_dim=64,
        z_channels=args.width,
        target_condition_dim=metadata_dim,
        condition_fusion="concat_mlp",
        width=args.diffusion_width,
        depth=args.diffusion_depth,
        objective=args.objective,
        diffusion_batch_mul=args.diffusion_batch_mul,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        prediction_type=args.prediction_type,
        radial_power_weighting=False,
        learned_output_gain=False,
        phase_aux_weight=0.0,
        loss_metric=(
            "normalized"
            if args.latent_loss_weighting == "unweighted"
            else "component_weighted"
        ),
    )
    return LatentContinuousConfig(
        metadata_dim=metadata_dim,
        transformer=transformer,
        diffusion=diffusion,
        transformer_metadata_film=args.transformer_metadata_film,
        context_dropout_probability=args.context_dropout_probability,
        latent_loss_weighting=args.latent_loss_weighting,
    )


def build_lr_scheduler(args: argparse.Namespace, optimizer):
    if args.lr_scheduler != "linear_floor":
        return get_scheduler(
            args.lr_scheduler,
            optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    if not 0.0 <= args.lr_end_ratio <= 1.0:
        raise ValueError("lr_end_ratio must be in [0,1]")
    warmup = max(int(args.lr_warmup_steps), 1)
    decay_steps = max(int(args.max_train_steps) - warmup, 1)

    def multiplier(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max((step - warmup) / decay_steps, 0.0), 1.0)
        return 1.0 - progress * (1.0 - args.lr_end_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_latent_checkpoint(
    path: str,
    model: LatentContinuousModel,
    adapter: FrozenLatentAutoencoder,
    global_step: int,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
) -> None:
    payload = {
        "version": 1,
        "kind": "latent_afig_continuous",
        "global_step": int(global_step),
        "model_config": model.config.fingerprint(),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "latent_contract": adapter.checkpoint_contract(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_latent_checkpoint(
    path: str,
    adapter: FrozenLatentAutoencoder,
    model: Optional[LatentContinuousModel] = None,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
) -> tuple[LatentContinuousModel, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("kind") != "latent_afig_continuous":
        raise ValueError("Not a compatible latent AFIG checkpoint")
    adapter.assert_contract_compatible(payload["latent_contract"])
    saved_config = latent_config_from_dict(payload["model_config"])
    if model is None:
        model = LatentContinuousModel(
            saved_config,
            loss_component_weights=payload["model"].get(
                "loss_component_weights"
            ),
        )
    elif model.config.fingerprint() != saved_config.fingerprint():
        raise ValueError("Latent model configuration does not match checkpoint")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return model, int(payload["global_step"])


@torch.no_grad()
def condition_diagnostics(
    model: LatentContinuousModel,
    latents: torch.Tensor,
    metadata: torch.Tensor,
) -> Dict[str, float]:
    inputs = model.shifted_inputs(latents, metadata)
    hidden, _ = model.forward_backbone(inputs, metadata)
    null = model.null_context.to(hidden.dtype).view(1, 1, -1).expand_as(hidden)
    batch, length, _ = latents.shape
    timesteps = torch.full(
        (batch, length),
        model.config.diffusion.num_train_timesteps // 2,
        device=latents.device,
        dtype=torch.long,
    )
    generator = torch.Generator(device=latents.device).manual_seed(1729)
    noise = torch.randn(
        latents.shape,
        device=latents.device,
        dtype=latents.dtype,
        generator=generator,
    )
    conditional = model.diffusion.predict_x0_deterministic(
        latents, hidden, timesteps, noise, target_condition=metadata
    )
    unconditional = model.diffusion.predict_x0_deterministic(
        latents, null, timesteps, noise, target_condition=metadata
    )
    conditional_mse = (conditional.float() - latents.float()).square().mean()
    null_mse = (unconditional.float() - latents.float()).square().mean()
    return {
        "diagnostic/normalized_target_mse": float(conditional_mse.item()),
        "diagnostic/null_x0_mse": float(null_mse.item()),
        "diagnostic/conditional_null_gap": float(
            (conditional.float() - unconditional.float()).square().mean().sqrt().item()
        ),
        "baseline/zero_mse": float(latents.float().square().mean().item()),
    }


@torch.no_grad()
def generated_spectrum_metrics(
    images: torch.Tensor, prefix: str
) -> Dict[str, float]:
    images = images.float()
    spectrum = torch.fft.fft2(images, norm="ortho")
    power = spectrum.abs().square().mean(dim=(0, 1))
    height, width = power.shape
    ky = torch.fft.fftfreq(height, device=images.device) * height
    kx = torch.fft.fftfreq(width, device=images.device) * width
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square())
    logs = {
        f"{prefix}/pixel_mean": float(images.mean().item()),
        f"{prefix}/pixel_std": float(images.std().item()),
        f"{prefix}/log_amplitude_mean": float(
            torch.log(spectrum.abs().clamp_min(1e-6)).mean().item()
        ),
    }
    radial_values = []
    for radial_bin in range(int(radius.max().item()) + 1):
        selected = (radius >= radial_bin) & (radius < radial_bin + 1)
        value = power[selected].mean().clamp_min(1e-12)
        radial_values.append(value)
        logs[f"{prefix}/radial_log_power_{radial_bin:02d}"] = float(
            value.log().item()
        )
    split = max(len(radial_values) // 2, 1)
    logs[f"{prefix}/high_low_power_ratio"] = float(
        torch.stack(radial_values[split:]).mean().div(
            torch.stack(radial_values[:split]).mean().clamp_min(1e-12)
        ).item()
    )
    return logs


@torch.no_grad()
def generate_previews(
    model: LatentContinuousModel,
    adapter: FrozenLatentAutoencoder,
    output_dir: str,
    global_step: int,
    batch_size: int,
    inference_steps: int,
    cfg_norm_match: bool,
) -> tuple[Dict[str, float], Dict[str, str]]:
    started = time.perf_counter()
    logs: Dict[str, float] = {}
    image_paths: Dict[str, str] = {}
    for scale in (1.0, 1.5, 2.0):
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(12345)
        latents = model.generate_latents(
            batch_size,
            adapter.position_features,
            cfg_scale=scale,
            cfg_norm_match=cfg_norm_match,
            num_inference_steps=inference_steps,
            generator=generator,
        )
        images = adapter.decode_latents(latents)
        path = os.path.join(output_dir, f"preview_{global_step:07d}_cfg{scale:.1f}.png")
        torchvision.utils.save_image(images.float().cpu(), path, nrow=max(1, int(math.sqrt(batch_size))))
        image_paths[f"preview/images_cfg_{scale:.1f}"] = path
        logs[f"preview/cfg_{scale:.1f}_latent_rms"] = float(
            latents.float().square().mean().sqrt().item()
        )
        logs.update(generated_spectrum_metrics(images, f"preview/cfg_{scale:.1f}"))
    logs["timing/generation_ms_per_image"] = (
        1000.0 * (time.perf_counter() - started) / (batch_size * 3)
    )
    return logs, image_paths


def log_preview_images(
    accelerator: Accelerator,
    image_paths: Dict[str, str],
    step: int,
) -> None:
    """Log saved preview grids when a W&B tracker is active."""
    if not accelerator.is_main_process:
        return
    try:
        tracker = accelerator.get_tracker("wandb", unwrap=True)
    except (KeyError, ValueError):
        return
    import wandb

    tracker.log(
        {name: wandb.Image(path) for name, path in image_paths.items()},
        step=step,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.max_train_steps = min(args.max_train_steps, 2)
        args.train_batch_size = min(args.train_batch_size, 2)
        args.dataloader_num_workers = 0
        args.width = 64
        args.num_layers = 2
        args.num_heads = 4
        args.ff_mult = 2
        args.diffusion_width = 64
        args.diffusion_depth = 2
        args.diffusion_batch_mul = 1
        args.num_train_timesteps = 20
        args.num_inference_steps = 1
        args.preview_images = 1
        args.logging_steps = 1
        args.diagnostic_steps = 1
        args.preview_steps = 0
        args.checkpointing_steps = 0
        args.mixed_precision = "no"
        args.report_to = "none"

    project = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    adapter = FrozenLatentAutoencoder(args.ae_checkpoint, args.latent_interface)
    config = build_model_config(args, adapter.position_features.shape[-1])
    loss_component_weights = None
    if args.latent_loss_weighting != "unweighted":
        if args.latent_loss_weights is None:
            raise ValueError(
                "Weighted latent losses require --latent_loss_weights"
            )
        weight_payload = torch.load(
            args.latent_loss_weights, map_location="cpu", weights_only=False
        )
        loss_component_weights = weight_payload[args.latent_loss_weighting]
    model = LatentContinuousModel(
        config, loss_component_weights=loss_component_weights
    )
    dataset = make_dataset(
        SimpleNamespace(
            dataset="synthetic" if args.smoke else args.dataset,
            data_root=args.data_root,
            resolution=32,
            smoke=args.smoke,
            seed=args.seed,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(args, optimizer)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    adapter.to(accelerator.device)
    metadata = adapter.position_features
    ema = ModelEMA(accelerator.unwrap_model(model), args.ema_decay) if args.use_ema else None
    global_step = 0
    if args.resume_from_checkpoint:
        _, global_step = load_latent_checkpoint(
            args.resume_from_checkpoint,
            adapter,
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            ema,
        )
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}} if args.run_name else None,
        )

    iterator = iter(loader)
    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="latent AFIG",
    )
    window_started = time.perf_counter()
    while global_step < args.max_train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch[0].to(accelerator.device, non_blocking=True)
        with torch.no_grad():
            latents = adapter.encode_images(images)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(latents, metadata)
            loss = output["loss"]
        accelerator.backward(loss)
        gradient_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1
        if ema is not None:
            ema.update(accelerator.unwrap_model(model))
        progress.update(1)

        if global_step % args.logging_steps == 0:
            elapsed = max(time.perf_counter() - window_started, 1e-6)
            per_example = output["normalized_per_example"]
            multiplier = args.diffusion_batch_mul
            per_position = per_example.view(multiplier, latents.shape[0], LATENT_SEQUENCE_LENGTH).mean((0, 1))
            ring_index = metadata[:, 1]
            boundary = torch.ones_like(ring_index, dtype=torch.bool)
            boundary[1:] = ring_index[1:] != ring_index[:-1]
            logs = {
                "train/loss": float(loss.detach().item()),
                "train/diffusion_objective_mse": float(output["unweighted_mse"].item()),
                "train/context_drop_fraction": float(output["context_drop_fraction"].item()),
                "train/context_null_gap": float(output["context_null_gap"].item()),
                "train/latent_rms": float(latents.float().square().mean().sqrt().item()),
                "train/ring_boundary_loss": float(per_position[boundary].mean().item()),
                "train/non_boundary_loss": float(per_position[~boundary].mean().item()),
                "train/grad_norm": float(gradient_norm),
                "train/learning_rate": float(scheduler.get_last_lr()[0]),
                "timing/steps_per_second": args.logging_steps / elapsed,
            }
            if args.prediction_type == "x0":
                logs["train/normalized_target_mse"] = float(
                    output["unweighted_mse"].item()
                )
            for position in range(LATENT_SEQUENCE_LENGTH):
                logs[f"position_loss/{position:02d}"] = float(per_position[position].item())
            log_metrics(accelerator, args.output_dir, logs, global_step)
            window_started = time.perf_counter()

        if args.diagnostic_steps > 0 and global_step % args.diagnostic_steps == 0:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            logs = condition_diagnostics(unwrapped, latents[: min(8, len(latents))], metadata)
            logs["baseline/causal_probe_mse"] = adapter.probe_validation_mse
            log_metrics(accelerator, args.output_dir, logs, global_step)
            unwrapped.train()

        if (
            accelerator.is_main_process
            and args.preview_steps > 0
            and global_step % args.preview_steps == 0
        ):
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            logs, image_paths = generate_previews(
                unwrapped,
                adapter,
                args.output_dir,
                global_step,
                args.preview_images,
                args.num_inference_steps,
                args.cfg_norm_match,
            )
            with torch.no_grad():
                reconstructed = adapter.decode_latents(latents[: args.preview_images])
                metrics = reconstruction_metrics(images[: len(reconstructed)], reconstructed)
            logs.update({f"ae_reconstruction/{key}": float(value.item()) for key, value in metrics.items()})
            log_metrics(accelerator, args.output_dir, logs, global_step)
            log_preview_images(accelerator, image_paths, global_step)
            unwrapped.train()

        if (
            accelerator.is_main_process
            and args.checkpointing_steps > 0
            and global_step % args.checkpointing_steps == 0
        ):
            save_latent_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{global_step}.pt"),
                accelerator.unwrap_model(model),
                adapter,
                global_step,
                optimizer,
                scheduler,
                ema,
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_latent_checkpoint(
            os.path.join(args.output_dir, "checkpoint_final.pt"),
            accelerator.unwrap_model(model),
            adapter,
            global_step,
            optimizer,
            scheduler,
            ema,
        )
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "global_step": global_step,
                    "model_config": config.fingerprint(),
                    "latent_contract": {
                        "layout_fingerprint": adapter.layout_hash,
                        "sequence_length": 53,
                        "token_dim": 64,
                    },
                },
                handle,
                indent=2,
            )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
