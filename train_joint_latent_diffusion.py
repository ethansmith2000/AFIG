#!/usr/bin/env python3
"""Train full-sequence rectified flow on frozen target-12 VAE latents."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from causal_transformer import CausalTransformerConfig
from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_joint_latent_diffusion import (
    JointLatentDiffusionConfig,
    JointLatentDiffusionModel,
    joint_config_from_dict,
)
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM
from train_autoencoder import make_dataset, reconstruction_metrics
from train_latent_continuous import (
    build_lr_scheduler,
    generated_spectrum_metrics,
    log_metrics,
    log_preview_images,
)
from train_continuous import ModelEMA


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument(
        "--output_dir", default="latent_continuous_runs/joint-default"
    )
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--lr_scheduler", default="linear_floor")
    parser.add_argument("--lr_warmup_steps", type=int, default=2000)
    parser.add_argument("--lr_end_ratio", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument(
        "--weight_decay_mode",
        choices=["all", "matrix_only"],
        default="all",
        help="matrix_only decays projection matrices, excluding biases, vectors, "
        "normalization parameters, and learned absolute-position tables",
    )
    parser.add_argument(
        "--augment_brightness",
        type=float,
        default=0.0,
        help="torchvision ColorJitter brightness factor; 0 disables. "
        "Acts mostly on the DC term, i.e. latent position 0.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument(
        "--qk_norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--block_conditioning",
        choices=["legacy_film", "adaln_zero"],
        default="legacy_film",
    )
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--flow_solver", choices=["euler", "heun"], default="heun")
    parser.add_argument(
        "--rope",
        choices=["none", "sequence", "radius_angle"],
        default="none",
        help="Rotary embeddings on q/k. radius_angle uses each latent's pooled "
        "polar frequency coordinates instead of a sequence index.",
    )
    parser.add_argument("--position_embedding_input", action="store_true")
    parser.add_argument("--position_embedding_film", action="store_true")
    parser.add_argument(
        "--timestep_weighting",
        choices=["uniform", "snr_interpolate"],
        default="uniform",
        help="uniform matches prior runs; snr_interpolate blends toward a "
        "high-noise-weighted objective",
    )
    parser.add_argument(
        "--timestep_weighting_alpha",
        type=float,
        default=0.5,
        help="0 = uniform, 1 = full inverse-available-gain weighting",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--preview_images", type=int, default=8)
    parser.add_argument("--checkpointing_steps", type=int, default=7500)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument(
        "--tracker_project_name", default="afig-latent-continuous"
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def build_model_config(
    args: argparse.Namespace,
    metadata_dim: int,
    sequence_length: int = LATENT_SEQUENCE_LENGTH,
    token_dim: int = LATENT_TOKEN_DIM,
) -> JointLatentDiffusionConfig:
    return JointLatentDiffusionConfig(
        sequence_length=sequence_length,
        token_dim=token_dim,
        metadata_dim=metadata_dim,
        transformer=CausalTransformerConfig(
            width=args.width,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            max_seq_len=sequence_length,
            gradient_checkpointing=args.gradient_checkpointing,
            qk_norm=args.qk_norm,
        ),
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        flow_solver=args.flow_solver,
        position_embedding_input=args.position_embedding_input,
        position_embedding_film=args.position_embedding_film,
        rope=args.rope,
        block_conditioning=args.block_conditioning,
        timestep_sampling=(
            "snr_interpolate" if args.timestep_weighting == "snr_interpolate" else "uniform"
        ),
        timestep_sampling_alpha=(
            args.timestep_weighting_alpha
            if args.timestep_weighting == "snr_interpolate"
            else 0.0
        ),
    )


def save_checkpoint(
    path: str,
    model: JointLatentDiffusionModel,
    adapter: FrozenLatentAutoencoder,
    global_step: int,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "model_type": "joint_latent_diffusion",
        "global_step": global_step,
        "model_config": model.config.fingerprint(),
        "model": model.state_dict(),
        "latent_contract": adapter.checkpoint_contract(),
        "ema": None if ema is None else ema.state_dict(),
        "optimizer_config": optimizer_config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    adapter: FrozenLatentAutoencoder,
    model: Optional[JointLatentDiffusionModel] = None,
    optimizer=None,
    scheduler=None,
    ema: Optional[ModelEMA] = None,
    use_ema_weights: bool = False,
) -> tuple[JointLatentDiffusionModel, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "joint_latent_diffusion":
        raise ValueError("Not a joint latent diffusion checkpoint")
    adapter.assert_contract_compatible(payload["latent_contract"])
    config = joint_config_from_dict(payload["model_config"])
    if model is None:
        model = JointLatentDiffusionModel(config)
    elif model.config.fingerprint() != config.fingerprint():
        raise ValueError("Joint model configuration does not match checkpoint")
    model_state = payload["model"]
    if use_ema_weights:
        if payload.get("ema") is None:
            raise ValueError("Checkpoint does not contain EMA weights")
        model_state = dict(model_state)
        model_state.update(payload["ema"])
    model.load_state_dict(model_state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return model, int(payload["global_step"])


@contextmanager
def ema_weights(model: JointLatentDiffusionModel, ema: Optional[ModelEMA]):
    """Temporarily evaluate with EMA weights while retaining resumable raw weights."""

    if ema is None:
        yield
        return
    state = model.state_dict()
    backup = {
        key: value.detach().clone()
        for key, value in state.items()
        if key in ema.shadow
    }
    ema.copy_to(model)
    try:
        yield
    finally:
        current = model.state_dict()
        for key, value in backup.items():
            current[key].copy_(value)


def build_optimizer_parameters(
    model: JointLatentDiffusionModel,
    weight_decay: float,
    mode: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Partition AdamW decay without weakening learned identity parameters."""

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if mode == "all":
        return [
            {
                "params": [parameter for _, parameter in named],
                "weight_decay": weight_decay,
            }
        ], {
            "mode": mode,
            "decay_parameter_count": sum(parameter.numel() for _, parameter in named),
            "no_decay_parameter_count": 0,
        }
    if mode != "matrix_only":
        raise ValueError(f"Unknown weight_decay_mode: {mode}")

    absolute_position_names = {
        "position_embedding_input",
        "position_embedding_film",
    }
    decay = []
    no_decay = []
    decay_names = []
    no_decay_names = []
    for name, parameter in named:
        if parameter.ndim < 2 or name in absolute_position_names:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if len(decay) + len(no_decay) != len(named):
        raise RuntimeError("Optimizer parameter partition is incomplete")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], {
        "mode": mode,
        "decay_parameter_count": sum(parameter.numel() for parameter in decay),
        "no_decay_parameter_count": sum(parameter.numel() for parameter in no_decay),
        "decay_tensor_count": len(decay),
        "no_decay_tensor_count": len(no_decay),
        "decay_names": decay_names,
        "no_decay_names": no_decay_names,
    }


@torch.no_grad()
def generate_preview(
    model: JointLatentDiffusionModel,
    adapter: FrozenLatentAutoencoder,
    output_dir: str,
    step: int,
    count: int,
    inference_steps: int,
    variant: str = "joint",
) -> tuple[Dict[str, float], Dict[str, str]]:
    generator = torch.Generator(device=adapter.latent_mean.device).manual_seed(
        10000 + step
    )
    started = time.perf_counter()
    latents = model.generate_latents(
        count,
        adapter.position_features,
        num_inference_steps=inference_steps,
        generator=generator,
    )
    images = adapter.decode_latents(latents)
    elapsed = time.perf_counter() - started
    path = os.path.join(output_dir, f"preview_{step:07d}_{variant}.png")
    torchvision.utils.save_image(images, path, nrow=max(int(count**0.5), 1))
    metric_prefix = f"preview/{variant}"
    logs = generated_spectrum_metrics(images, metric_prefix)
    logs[f"{metric_prefix}_latent_rms"] = float(
        latents.float().square().mean().sqrt().item()
    )
    logs[f"timing/{variant}_generation_ms_per_image"] = 1000.0 * elapsed / count
    return logs, {metric_prefix: path}


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
        args.num_train_timesteps = 20
        args.num_inference_steps = 1
        args.preview_images = 1
        args.logging_steps = 1
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

    adapter = FrozenLatentAutoencoder(
        args.ae_checkpoint,
        args.latent_interface,
        sample_posterior=False,
    )
    config = build_model_config(
        args,
        adapter.position_features.shape[-1],
        sequence_length=adapter.sequence_length,
        token_dim=adapter.token_dim,
    )
    model = JointLatentDiffusionModel(config)
    dataset = make_dataset(
        SimpleNamespace(
            dataset="synthetic" if args.smoke else args.dataset,
            data_root=args.data_root,
            resolution=32,
            smoke=args.smoke,
            seed=args.seed,
            augment_brightness=args.augment_brightness,
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
    optimizer_parameters, decay_partition = build_optimizer_parameters(
        model, args.weight_decay, args.weight_decay_mode
    )
    optimizer_config = {
        "name": "AdamW",
        "learning_rate": args.learning_rate,
        "betas": [args.adam_beta1, args.adam_beta2],
        "epsilon": 1e-8,
        "weight_decay": args.weight_decay,
        "weight_decay_partition": decay_partition,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_end_ratio": args.lr_end_ratio,
        "max_grad_norm": args.max_grad_norm,
    }
    if accelerator.is_main_process:
        print(
            "optimizer: "
            f"AdamW lr={args.learning_rate:g} "
            f"betas=({args.adam_beta1:g},{args.adam_beta2:g}) "
            f"weight_decay={args.weight_decay:g} mode={args.weight_decay_mode} "
            f"decay/no_decay params={decay_partition['decay_parameter_count']}/"
            f"{decay_partition['no_decay_parameter_count']} "
            f"scheduler={args.lr_scheduler} warmup={args.lr_warmup_steps}"
        )
    # Fused AdamW requires all params on CUDA, so fall back on CPU/smoke runs.
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )
    scheduler = build_lr_scheduler(args, optimizer)
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    adapter.to(accelerator.device)
    metadata = adapter.position_features
    ema = (
        ModelEMA(accelerator.unwrap_model(model), args.ema_decay)
        if args.use_ema
        else None
    )
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
            init_kwargs={"wandb": {"name": args.run_name}}
            if args.run_name
            else None,
        )

    iterator = iter(loader)
    progress = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="joint latent diffusion",
    )
    window_started = time.perf_counter()
    while global_step < args.max_train_steps:
        try:
            images, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(accelerator.device, non_blocking=True)
        with torch.no_grad():
            latents = adapter.encode_images(images)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = model(latents, metadata)
            loss = output["loss"]
        accelerator.backward(loss)
        gradient_norm = accelerator.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        global_step += 1
        if ema is not None:
            ema.update(accelerator.unwrap_model(model))
        progress.update(1)

        if global_step % args.logging_steps == 0:
            elapsed = max(time.perf_counter() - window_started, 1e-6)
            per_position = output["per_position"].mean(dim=0)
            logs = {
                "train/loss": float(loss.detach().item()),
                "train/flow_mse": float(output["unweighted_mse"].item()),
                "train/prediction_rms": float(output["prediction_rms"].item()),
                "train/target_rms": float(output["target_rms"].item()),
                "train/latent_rms": float(
                    latents.float().square().mean().sqrt().item()
                ),
                "train/grad_norm": float(gradient_norm),
                "train/learning_rate": float(scheduler.get_last_lr()[0]),
                "timing/steps_per_second": args.logging_steps / elapsed,
            }
            for position in range(config.sequence_length):
                logs[f"position_loss/{position:02d}"] = float(
                    per_position[position].item()
                )
            log_metrics(accelerator, args.output_dir, logs, global_step)
            window_started = time.perf_counter()

        if (
            accelerator.is_main_process
            and args.preview_steps > 0
            and global_step % args.preview_steps == 0
        ):
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            if ema is None:
                logs, image_paths = generate_preview(
                    unwrapped,
                    adapter,
                    args.output_dir,
                    global_step,
                    args.preview_images,
                    args.num_inference_steps,
                )
            else:
                logs, image_paths = generate_preview(
                    unwrapped,
                    adapter,
                    args.output_dir,
                    global_step,
                    args.preview_images,
                    args.num_inference_steps,
                    variant="joint_raw",
                )
                with ema_weights(unwrapped, ema):
                    ema_logs, ema_paths = generate_preview(
                        unwrapped,
                        adapter,
                        args.output_dir,
                        global_step,
                        args.preview_images,
                        args.num_inference_steps,
                        variant="joint_ema",
                    )
                logs.update(ema_logs)
                image_paths.update(ema_paths)
            logs["preview/uses_ema"] = float(ema is not None)
            with torch.no_grad():
                reconstruction = adapter.decode_latents(
                    latents[: args.preview_images]
                )
                metrics = reconstruction_metrics(
                    images[: len(reconstruction)], reconstruction
                )
            logs.update(
                {
                    f"ae_reconstruction/{key}": float(value.item())
                    for key, value in metrics.items()
                }
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
                optimizer_config,
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
            optimizer_config,
        )
        Path(args.output_dir, "summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "model_type": "joint_latent_diffusion",
                    "model_config": config.fingerprint(),
                    "ema_decay": args.ema_decay if ema is not None else None,
                    "optimizer_config": optimizer_config,
                },
                indent=2,
            )
            + "\n"
        )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
