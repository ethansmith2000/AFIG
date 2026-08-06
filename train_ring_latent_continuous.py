#!/usr/bin/env python3
"""Train a causal-between-rings, joint-within-ring latent generator."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from causal_transformer import CausalTransformerConfig
from diffusion_decoder import DiffusionDecoderConfig
from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_ring_latent_continuous import (
    MAX_RING_LATENTS,
    RING_SEQUENCE_LENGTH,
    RingLatentContinuousConfig,
    RingLatentContinuousModel,
    ring_latent_config_from_dict,
)
from train_autoencoder import make_dataset, reconstruction_metrics
from train_continuous import ModelEMA
from train_latent_continuous import (
    build_lr_scheduler,
    generated_spectrum_metrics,
    log_metrics,
    log_preview_images,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--output_dir", default="latent_continuous_runs/ring-block")
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_end_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diffusion_width", type=int, default=768)
    parser.add_argument("--diffusion_depth", type=int, default=6)
    parser.add_argument("--diffusion_batch_mul", type=int, default=2)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--flow_solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--context_dropout_probability", type=float, default=0.1)
    parser.add_argument("--cfg_norm_match", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument(
        "--generation_grouping", choices=["ring", "token"], default="ring"
    )
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--diagnostic_steps", type=int, default=250)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--preview_images", type=int, default=16)
    parser.add_argument("--checkpointing_steps", type=int, default=2500)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--tracker_project_name", default="afig-ring-latent-continuous")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def build_model_config(
    args: argparse.Namespace,
    latent_sequence_length: int = 53,
    latent_dim: int = 64,
    group_sequence_length: int = RING_SEQUENCE_LENGTH,
    max_group_latents: int = MAX_RING_LATENTS,
) -> RingLatentContinuousConfig:
    transformer = CausalTransformerConfig(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        max_seq_len=group_sequence_length,
        gradient_checkpointing=args.gradient_checkpointing,
        qk_norm=True,
    )
    diffusion = DiffusionDecoderConfig(
        target_dim=max_group_latents * latent_dim,
        z_channels=args.width,
        target_condition_dim=0,
        condition_fusion="add",
        width=args.diffusion_width,
        depth=args.diffusion_depth,
        objective="flow",
        prediction_type="v_prediction",
        flow_solver=args.flow_solver,
        diffusion_batch_mul=args.diffusion_batch_mul,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        component_reduction="fixed_dim",
        loss_metric="normalized",
    )
    return RingLatentContinuousConfig(
        latent_sequence_length=latent_sequence_length,
        latent_dim=latent_dim,
        ring_sequence_length=group_sequence_length,
        max_ring_latents=max_group_latents,
        grouping=args.generation_grouping,
        transformer=transformer,
        diffusion=diffusion,
        context_dropout_probability=args.context_dropout_probability,
        rope_base=args.rope_base,
    )


def save_checkpoint(
    path: str,
    model: RingLatentContinuousModel,
    adapter: FrozenLatentAutoencoder,
    global_step: int,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
) -> None:
    payload = {
        "version": 1,
        "kind": "ring_latent_afig_continuous",
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


def load_checkpoint(
    path: str,
    adapter: FrozenLatentAutoencoder,
    model: Optional[RingLatentContinuousModel] = None,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
) -> tuple[RingLatentContinuousModel, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("kind") != "ring_latent_afig_continuous":
        raise ValueError("Not a compatible ring-latent AFIG checkpoint")
    adapter.assert_contract_compatible(payload["latent_contract"])
    saved_config = ring_latent_config_from_dict(payload["model_config"])
    latent_parent = adapter.autoencoder.layout.latent_parent
    if model is None:
        model = RingLatentContinuousModel(latent_parent, saved_config)
    elif model.config.fingerprint() != saved_config.fingerprint():
        raise ValueError("Ring latent model configuration does not match checkpoint")
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
    model: RingLatentContinuousModel, latents: torch.Tensor
) -> Dict[str, float]:
    rings = model.pack_rings(latents)
    hidden, _ = model.forward_backbone(model.shifted_inputs_from_rings(rings))
    null = model.null_context.to(hidden.dtype).view(1, 1, -1).expand_as(hidden)
    batch, length, _ = rings.shape
    timesteps = torch.full(
        (batch, length),
        model.config.diffusion.num_train_timesteps // 2,
        device=rings.device,
        dtype=torch.long,
    )
    generator = torch.Generator(device=rings.device).manual_seed(1729)
    noise = torch.randn(rings.shape, device=rings.device, dtype=rings.dtype, generator=generator)
    conditional = model.diffusion.predict_x0_deterministic(
        rings,
        hidden,
        timesteps,
        noise,
        component_mask=model.ring_component_mask,
    )
    unconditional = model.diffusion.predict_x0_deterministic(
        rings,
        null,
        timesteps,
        noise,
        component_mask=model.ring_component_mask,
    )
    mask = model.ring_component_mask[None].float()
    denominator = mask.sum() * batch

    def masked_mse(value: torch.Tensor, target: torch.Tensor) -> float:
        return float((((value.float() - target.float()).square() * mask).sum() / denominator).item())

    return {
        "diagnostic/normalized_target_mse": masked_mse(conditional, rings),
        "diagnostic/null_x0_mse": masked_mse(unconditional, rings),
        "diagnostic/conditional_null_gap": masked_mse(conditional, unconditional) ** 0.5,
        "baseline/zero_mse": float((rings.float().square() * mask).sum().div(denominator).item()),
    }


@torch.no_grad()
def generate_previews(
    model: RingLatentContinuousModel,
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
            cfg_scale=scale,
            cfg_norm_match=cfg_norm_match,
            num_inference_steps=inference_steps,
            generator=generator,
        )
        images = adapter.decode_latents(latents)
        path = os.path.join(output_dir, f"preview_{global_step:07d}_cfg{scale:.1f}.png")
        torchvision.utils.save_image(
            images.float().cpu(), path, nrow=max(1, int(math.sqrt(batch_size)))
        )
        image_paths[f"preview/images_cfg_{scale:.1f}"] = path
        logs[f"preview/cfg_{scale:.1f}_latent_rms"] = float(
            latents.float().square().mean().sqrt().item()
        )
        logs.update(generated_spectrum_metrics(images, f"preview/cfg_{scale:.1f}"))
    logs["timing/generation_ms_per_image"] = (
        1000.0 * (time.perf_counter() - started) / (batch_size * 3)
    )
    return logs, image_paths


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

    project = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    adapter = FrozenLatentAutoencoder(args.ae_checkpoint, args.latent_interface)
    # The generator factorization needs only the causal-ring parent layout.  It
    # is intentionally compatible with both the higher-fidelity legacy codec
    # and the new bidirectional-within-ring codec so their generator arms can be
    # compared without changing the 23-step model.
    source_parent = adapter.autoencoder.layout.latent_parent
    if args.generation_grouping == "ring":
        group_sequence_length = int(source_parent.max().item()) + 1
        max_group_latents = int(torch.bincount(source_parent).max().item())
    else:
        group_sequence_length = adapter.sequence_length
        max_group_latents = 1
    config = build_model_config(
        args,
        latent_sequence_length=adapter.sequence_length,
        latent_dim=adapter.token_dim,
        group_sequence_length=group_sequence_length,
        max_group_latents=max_group_latents,
    )
    model = RingLatentContinuousModel(
        source_parent,
        config,
    )
    dataset = make_dataset(
        SimpleNamespace(
            dataset="synthetic" if args.smoke else args.dataset,
            data_root=args.data_root,
            resolution=32,
            smoke=args.smoke,
            seed=args.seed,
            augment_brightness=0.0,
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
        fused=torch.cuda.is_available(),
    )
    scheduler = build_lr_scheduler(args, optimizer)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    adapter.to(accelerator.device)
    ema = ModelEMA(accelerator.unwrap_model(model), args.ema_decay) if args.use_ema else None
    global_step = 0
    if args.resume_from_checkpoint:
        _, global_step = load_checkpoint(
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
        desc="ring latent AFIG",
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
            output = model(latents)
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
            unwrapped = accelerator.unwrap_model(model)
            per_ring = output["normalized_per_example"].view(
                args.diffusion_batch_mul,
                latents.shape[0],
                unwrapped.config.ring_sequence_length,
            ).mean((0, 1))
            active_mse = per_ring * unwrapped.ring_dim / (
                unwrapped.ring_counts.to(per_ring.dtype) * unwrapped.config.latent_dim
            )
            logs = {
                "train/loss": float(loss.detach().item()),
                "train/diffusion_objective_mse": float(output["unweighted_mse"].item()),
                "train/context_drop_fraction": float(output["context_drop_fraction"].item()),
                "train/context_null_gap": float(output["context_null_gap"].item()),
                "train/latent_rms": float(latents.float().square().mean().sqrt().item()),
                "train/grad_norm": float(gradient_norm),
                "train/learning_rate": float(scheduler.get_last_lr()[0]),
                "timing/steps_per_second": args.logging_steps / elapsed,
            }
            for group in range(unwrapped.config.ring_sequence_length):
                logs[f"group_loss/{group:03d}"] = float(active_mse[group].item())
            log_metrics(accelerator, args.output_dir, logs, global_step)
            window_started = time.perf_counter()

        if args.diagnostic_steps > 0 and global_step % args.diagnostic_steps == 0:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            logs = condition_diagnostics(unwrapped, latents[: min(8, len(latents))])
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
            logs.update(
                {f"ae_reconstruction/{key}": float(value.item()) for key, value in metrics.items()}
            )
            log_metrics(accelerator, args.output_dir, logs, global_step)
            log_preview_images(accelerator, image_paths, global_step)
            unwrapped.train()

        if (
            accelerator.is_main_process
            and args.checkpointing_steps > 0
            and global_step % args.checkpointing_steps == 0
        ):
            save_checkpoint(
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
        save_checkpoint(
            os.path.join(args.output_dir, "checkpoint_final.pt"),
            accelerator.unwrap_model(model),
            adapter,
            global_step,
            optimizer,
            scheduler,
            ema,
        )
        Path(os.path.join(args.output_dir, "summary.json")).write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "model_config": config.fingerprint(),
                    "latent_contract": {
                        "layout_fingerprint": adapter.layout_hash,
                        "sequence_length": adapter.sequence_length,
                        "token_dim": adapter.token_dim,
                        "grouping": config.grouping,
                        "group_sequence_length": config.ring_sequence_length,
                        "max_group_latents": config.max_ring_latents,
                    },
                },
                indent=2,
            )
            + "\n"
        )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
