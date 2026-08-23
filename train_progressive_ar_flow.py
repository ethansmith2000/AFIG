#!/usr/bin/env python3
"""Train an autoregressive flow prior over progressive image tokens."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from progressive_tokenizer import (
    AutoregressiveFlowConfig,
    AutoregressiveRectifiedFlow,
)
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.training import count_parameters, optimizer_parameter_groups
from progressive_tokenizer.tracking import WandbTracker
from train_progressive_joint_flow import (
    atomic_save,
    prune_numbered_checkpoints,
    autocast_context,
    normalize,
    seed_everything,
    set_learning_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_cache", required=True)
    parser.add_argument("--output_dir", default="prior_runs/ar-flow-s1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--trunk_depth", type=int, default=12)
    parser.add_argument("--head_depth", type=int, default=6)
    parser.add_argument(
        "--block_size",
        type=int,
        default=1,
        help="Concatenate this many consecutive latent registers per AR decision.",
    )
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument(
        "--qk_norm", choices=["rms", "l2_temperature"], default="rms"
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the training forward with mode=default and fullgraph=True.",
    )
    parser.add_argument("--report_to", choices=["none", "wandb"], default="wandb")
    parser.add_argument(
        "--tracker_project_name", default="afig-progressive-tokenizer"
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_group", default="ar-prior")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.995)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "bf16"], default="bf16")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--preview_every", type=int, default=1000)
    parser.add_argument("--preview_images", type=int, default=16)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=2500)
    parser.add_argument(
        "--keep_numbered_checkpoints",
        type=int,
        default=0,
        help="How many step-numbered checkpoints to retain. 0 (default) keeps "
        "only checkpoint_latest.pt for resume plus the final checkpoint; "
        "raise it only when intermediate steps will actually be evaluated.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--history_noise_max", type=float, default=0.0)
    parser.add_argument("--history_noise_min", type=float, default=0.0)
    parser.add_argument("--history_noise_probability", type=float, default=0.75)
    parser.add_argument("--history_noise_ramp_steps", type=int, default=4000)
    parser.add_argument("--history_noise_reference", type=float, default=0.1)
    parser.add_argument(
        "--history_reliability_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--head_position_conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block_size must be positive")
    if args.history_noise_max < args.history_noise_min or args.history_noise_min < 0:
        parser.error("history noise range must satisfy 0 <= min <= max")
    if not 0.0 <= args.history_noise_probability <= 1.0:
        parser.error("--history_noise_probability must lie in [0, 1]")
    return args


def block_latents(latents: torch.Tensor, block_size: int) -> torch.Tensor:
    """Group consecutive [T,D] registers into [T/block_size, block_size*D]."""

    if latents.ndim != 3:
        raise ValueError("latents must have shape [N,T,D]")
    if latents.shape[1] % block_size:
        raise ValueError("latent sequence length must be divisible by block_size")
    if block_size == 1:
        return latents
    return latents.reshape(
        latents.shape[0],
        latents.shape[1] // block_size,
        latents.shape[2] * block_size,
    )


def unblock_latents(
    latents: torch.Tensor,
    *,
    sequence_length: int,
    token_dim: int,
) -> torch.Tensor:
    """Restore grouped AR tokens to the tokenizer's physical register layout."""

    if latents.numel() != latents.shape[0] * sequence_length * token_dim:
        raise ValueError("blocked latents do not match tokenizer latent dimensions")
    return latents.reshape(latents.shape[0], sequence_length, token_dim)


@torch.no_grad()
def evaluate(
    model,
    loader,
    mean,
    scale,
    device,
    args,
) -> dict:
    model.eval()
    loss_total = 0.0
    token_total = torch.zeros(model.config.sequence_length, device=device)
    examples = 0
    generator = torch.Generator(device=device).manual_seed(20260810)
    for index, (latents,) in enumerate(loader):
        if index >= args.eval_batches:
            break
        clean = normalize(latents.to(device, non_blocking=True), mean, scale)
        time_values = torch.rand(
            clean.shape[:-1], device=device, generator=generator
        )
        noise = torch.randn(
            clean.shape,
            device=device,
            dtype=clean.dtype,
            generator=generator,
        )
        with autocast_context(args, device):
            output = model(clean, time=time_values, noise=noise)
        loss_total += float(output["loss"]) * clean.shape[0]
        token_total += output["per_token_mse"] * clean.shape[0]
        examples += clean.shape[0]
    model.train()
    per_token = token_total / max(examples, 1)
    return {
        "examples": examples,
        "teacher_forced_flow_mse": loss_total / max(examples, 1),
        "per_token_mse": per_token.cpu().tolist(),
        "per_token_min": float(per_token.min()),
        "per_token_max": float(per_token.max()),
    }


@torch.no_grad()
def save_preview(
    model,
    tokenizer,
    mean,
    scale,
    output,
    step,
    args,
    device,
    token_scale=None,
) -> dict:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(54321)
    with autocast_context(args, device):
        standardized = model.generate(
            args.preview_images,
            steps=args.sample_steps,
            generator=generator,
        )
        raw = standardized.float() * scale + mean
        physical_raw = unblock_latents(
            raw,
            sequence_length=tokenizer.config.num_latents,
            token_dim=tokenizer.config.latent_dim,
        )
        # applied after unblock_latents: token_scale indexes physical
        # registers, matching the evaluator's post-layout divide
        if token_scale is not None:
            physical_raw = physical_raw / token_scale
        images = tokenizer.decode(physical_raw)
    save_image(
        images.float().add(1).div(2).clamp(0, 1),
        output,
        nrow=max(int(math.sqrt(args.preview_images)), 1),
    )
    model.train()
    return {
        "standardized_mean": float(standardized.float().mean()),
        "standardized_std": float(standardized.float().std(unbiased=False)),
        "standardized_min": float(standardized.float().min()),
        "standardized_max": float(standardized.float().max()),
        "raw_rms": float(raw.square().mean().sqrt()),
        "image_min": float(images.float().min()),
        "image_max": float(images.float().max()),
    }


def checkpoint_payload(model, optimizer, step, cache, mean, scale) -> dict:
    return {
        "version": 1,
        "model_type": "progressive_autoregressive_rectified_flow",
        "model_config": model.config.fingerprint(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "normalization": {"mean": mean.cpu(), "scale": scale.cpu()},
        "tokenizer_checkpoint": cache["tokenizer_checkpoint"],
        "tokenizer_step": cache["tokenizer_step"],
        # without this a magnitude-rescaled cache decodes un-inverted at eval
        "token_scale": cache.get("token_scale"),
        "latent_layout": {
            "type": "consecutive_blocks",
            "block_size": model.config.token_dim // cache["model_config"]["latent_dim"],
            "physical_sequence_length": cache["model_config"]["num_latents"],
            "physical_token_dim": cache["model_config"]["latent_dim"],
        },
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("autoregressive flow training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    physical_train_latents = cache["train_latents"]
    physical_test_latents = cache["test_latents"]
    if physical_train_latents.shape[1] % args.block_size:
        raise ValueError("tokenizer sequence length must be divisible by block_size")
    train_latents = block_latents(physical_train_latents, args.block_size)
    test_latents = block_latents(physical_test_latents, args.block_size)
    sequence_length, token_dim = train_latents.shape[1:]
    global_mean = cache["statistics"]["global_mean"].float().to(device)
    global_scale = cache["statistics"]["global_std"].float().to(device)

    train_loader = DataLoader(
        TensorDataset(train_latents),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        TensorDataset(test_latents),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    config = AutoregressiveFlowConfig(
        sequence_length=sequence_length,
        token_dim=token_dim,
        width=args.width,
        trunk_depth=args.trunk_depth,
        head_depth=args.head_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qk_norm=args.qk_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        history_reliability_conditioning=args.history_reliability_conditioning,
        history_noise_reference=args.history_noise_reference,
        head_position_conditioning=args.head_position_conditioning,
    )
    model = AutoregressiveRectifiedFlow(config).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
    )
    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["model_config"] != config.fingerprint():
            raise ValueError("resume model configuration mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
    if args.compile:
        model.compile(mode="default", fullgraph=True)
    tokenizer, tokenizer_payload = load_tokenizer_checkpoint(
        cache["tokenizer_checkpoint"]
    )
    if int(tokenizer_payload.get("step", -1)) != int(cache["tokenizer_step"]):
        raise ValueError("latent cache and tokenizer checkpoint step differ")
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    preview_token_scale = cache.get("token_scale")
    if preview_token_scale is not None:
        preview_token_scale = preview_token_scale.float().to(device)[None, :, None]
    config_payload = {
        "model": config.fingerprint(),
        "training": vars(args),
        "normalization": {
            "type": "tensor_wide_population",
            "mean": float(global_mean),
            "scale": float(global_scale),
        },
        "parameters": count_parameters(model.parameters()),
        "tokenizer_step": cache["tokenizer_step"],
        "latent_cache": {
            "train_examples": int(train_latents.shape[0]),
            "test_examples": int(test_latents.shape[0]),
            "train_views": cache.get("train_views", ["original"]),
        },
        "alignment": (
            f"BOS, b_1, ..., b_{sequence_length - 1} -> "
            f"predict b_1, ..., b_{sequence_length}; "
            f"each block contains {args.block_size} consecutive physical registers"
        ),
        "latent_layout": {
            "type": "consecutive_blocks",
            "block_size": args.block_size,
            "physical_sequence_length": int(physical_train_latents.shape[1]),
            "physical_token_dim": int(physical_train_latents.shape[2]),
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(config_payload, sort_keys=True), flush=True)
    tracker = WandbTracker(
        enabled=args.report_to == "wandb",
        output_dir=output_dir,
        project=args.tracker_project_name,
        name=args.run_name or output_dir.name,
        group=args.run_group,
        config=config_payload,
    )

    iterator = iter(train_loader)
    rolling = {"loss": 0.0, "prediction_rms": 0.0, "target_rms": 0.0}
    rolling_count = 0
    window_start = time.monotonic()
    history = output_dir / "history.jsonl"
    for step in range(start_step, args.max_train_steps):
        try:
            (latents,) = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            (latents,) = next(iterator)
        clean = normalize(latents.to(device, non_blocking=True), global_mean, global_scale)
        learning_rate = set_learning_rate(
            optimizer, step, args.learning_rate, args.warmup_steps
        )
        optimizer.zero_grad(set_to_none=True)
        history_sigma = None
        if args.history_noise_max > 0:
            ramp = min(1.0, (step + 1) / max(1, args.history_noise_ramp_steps))
            mask = (
                torch.rand(clean.shape[:2], device=device)
                < args.history_noise_probability
            ).float()
            magnitude = args.history_noise_min + (
                args.history_noise_max - args.history_noise_min
            ) * torch.rand(clean.shape[:2], device=device)
            history_sigma = mask * magnitude * ramp
        with autocast_context(args, device):
            output = model(clean, history_noise_sigma=history_sigma)
        output["loss"].backward()
        gradient_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed_step = step + 1
        for key in rolling:
            rolling[key] += float(
                output[key].detach() if key == "loss" else output[key]
            )
        rolling_count += 1
        if completed_step == 1 or completed_step % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = max(time.monotonic() - window_start, 1e-9)
            record = {
                "step": completed_step,
                **{key: value / rolling_count for key, value in rolling.items()},
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "steps_per_second": rolling_count / elapsed,
            }
            with history.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            tracker.log(record, step=completed_step, prefix="train")
            rolling = {key: 0.0 for key in rolling}
            rolling_count = 0
            window_start = time.monotonic()
        if args.eval_every > 0 and completed_step % args.eval_every == 0:
            metrics = evaluate(
                model, test_loader, global_mean, global_scale, device, args
            )
            metrics["step"] = completed_step
            (output_dir / "metrics_latest.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({"evaluation": metrics}, sort_keys=True), flush=True)
            tracker.log(metrics, step=completed_step, prefix="eval")
        if args.preview_every > 0 and completed_step % args.preview_every == 0:
            preview_path = output_dir / f"samples_{completed_step:06d}.png"
            metrics = save_preview(
                model,
                tokenizer,
                global_mean,
                global_scale,
                preview_path,
                completed_step,
                args,
                device,
                preview_token_scale,
            )
            print(
                json.dumps({"preview": {"step": completed_step, **metrics}}, sort_keys=True),
                flush=True,
            )
            tracker.log(metrics, step=completed_step, prefix="preview")
            tracker.log_image(
                preview_path, step=completed_step, key="preview/samples"
            )
        if args.checkpoint_every > 0 and completed_step % args.checkpoint_every == 0:
            latest = output_dir / "checkpoint_latest.pt"
            atomic_save(
                checkpoint_payload(
                    model, optimizer, completed_step, cache, global_mean, global_scale
                ),
                latest,
            )
            if args.keep_numbered_checkpoints > 0:
                numbered = output_dir / f"checkpoint_{completed_step:06d}.pt"
                if numbered.exists():
                    numbered.unlink()
                os.link(latest, numbered)
                prune_numbered_checkpoints(
                    output_dir, args.keep_numbered_checkpoints
                )

    final = checkpoint_payload(
        model, optimizer, args.max_train_steps, cache, global_mean, global_scale
    )
    final.pop("optimizer")
    atomic_save(final, output_dir / "checkpoint_final.pt")
    print(json.dumps({"complete": args.max_train_steps}), flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
