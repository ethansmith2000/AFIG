#!/usr/bin/env python3
"""Train full-sequence rectified flow on frozen target-12 VAE latents."""

from __future__ import annotations

import argparse
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
from model_latent_continuous import LATENT_SEQUENCE_LENGTH
from train_autoencoder import make_dataset, reconstruction_metrics
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
    parser.add_argument("--lr_scheduler", default="linear_floor")
    parser.add_argument("--lr_warmup_steps", type=int, default=2000)
    parser.add_argument("--lr_end_ratio", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=0.1)
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
    args: argparse.Namespace, metadata_dim: int
) -> JointLatentDiffusionConfig:
    return JointLatentDiffusionConfig(
        metadata_dim=metadata_dim,
        transformer=CausalTransformerConfig(
            width=args.width,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            max_seq_len=LATENT_SEQUENCE_LENGTH,
            gradient_checkpointing=args.gradient_checkpointing,
        ),
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        flow_solver=args.flow_solver,
        position_embedding_input=args.position_embedding_input,
        position_embedding_film=args.position_embedding_film,
        rope=args.rope,
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
) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "model_type": "joint_latent_diffusion",
        "global_step": global_step,
        "model_config": model.config.fingerprint(),
        "model": model.state_dict(),
        "latent_contract": adapter.checkpoint_contract(),
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
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return model, int(payload["global_step"])


@torch.no_grad()
def generate_preview(
    model: JointLatentDiffusionModel,
    adapter: FrozenLatentAutoencoder,
    output_dir: str,
    step: int,
    count: int,
    inference_steps: int,
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
    path = os.path.join(output_dir, f"preview_{step:07d}_joint.png")
    torchvision.utils.save_image(images, path, nrow=max(int(count**0.5), 1))
    logs = generated_spectrum_metrics(images, "preview/joint")
    logs["preview/joint_latent_rms"] = float(
        latents.float().square().mean().sqrt().item()
    )
    logs["timing/generation_ms_per_image"] = 1000.0 * elapsed / count
    return logs, {"preview/joint": path}


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
    config = build_model_config(args, adapter.position_features.shape[-1])
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
    # Fused AdamW requires all params on CUDA, so fall back on CPU/smoke runs.
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
    metadata = adapter.position_features
    global_step = 0
    if args.resume_from_checkpoint:
        _, global_step = load_checkpoint(
            args.resume_from_checkpoint,
            adapter,
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
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
            for position in range(LATENT_SEQUENCE_LENGTH):
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
            logs, image_paths = generate_preview(
                unwrapped,
                adapter,
                args.output_dir,
                global_step,
                args.preview_images,
                args.num_inference_steps,
            )
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
        )
        Path(args.output_dir, "summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "model_type": "joint_latent_diffusion",
                    "model_config": config.fingerprint(),
                },
                indent=2,
            )
            + "\n"
        )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
