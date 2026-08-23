#!/usr/bin/env python3
"""Train joint rectified flow on frozen progressive-token sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint
from progressive_tokenizer.training import count_parameters, optimizer_parameter_groups
from progressive_tokenizer.tracking import WandbTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_cache", required=True)
    parser.add_argument("--output_dir", default="prior_runs/joint-flow-s1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
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
    parser.add_argument("--run_group", default="joint-prior")
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
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prune_numbered_checkpoints(output_dir: Path, keep: int) -> None:
    """Keep only the `keep` most recent numbered checkpoints.

    A 60k run at the default 2500-step cadence otherwise leaves 24 numbered
    checkpoints of roughly 840 MB each -- 20 GB per run, on a shared box.
    checkpoint_latest.pt (for resume) and checkpoint_final.pt are never touched.
    """

    numbered = sorted(output_dir.glob("checkpoint_[0-9]*.pt"))
    for stale in numbered[: max(0, len(numbered) - keep)]:
        stale.unlink()


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_learning_rate(
    optimizer: torch.optim.Optimizer, step: int, base: float, warmup: int
) -> float:
    learning_rate = base * min(1.0, (step + 1) / max(warmup, 1))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def autocast_context(args: argparse.Namespace, device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=args.mixed_precision == "bf16" and device.type == "cuda",
    )


def normalize(
    latents: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return (latents.float() - mean) / scale


@torch.no_grad()
def evaluate(
    model: JointRectifiedFlow,
    loader: DataLoader,
    mean: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
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
        time_values = torch.rand(clean.shape[0], device=device, generator=generator)
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
        "flow_mse": loss_total / max(examples, 1),
        "per_token_mse": per_token.cpu().tolist(),
        "per_token_min": float(per_token.min()),
        "per_token_max": float(per_token.max()),
    }


@torch.no_grad()
def save_preview(
    model: JointRectifiedFlow,
    tokenizer,
    mean: torch.Tensor,
    scale: torch.Tensor,
    output: Path,
    step: int,
    args: argparse.Namespace,
    device: torch.device,
    token_scale: Optional[torch.Tensor] = None,
) -> dict:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(10_000 + step)
    with autocast_context(args, device):
        standardized = model.sample(
            args.preview_images,
            steps=args.sample_steps,
            solver="heun",
            generator=generator,
        )
        raw = standardized.float() * scale + mean
        # a magnitude-rescaled cache must be inverted before decoding, exactly
        # as the evaluator does -- otherwise previews of a healthy run look
        # broken (register 0 up to 5.6x hot, register 63 down to 0.18x)
        if token_scale is not None:
            raw = raw / token_scale
        images = tokenizer.decode(raw)
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


def checkpoint_payload(
    model: JointRectifiedFlow,
    optimizer: torch.optim.Optimizer,
    step: int,
    cache: dict,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> dict:
    return {
        "version": 1,
        "model_type": "progressive_joint_rectified_flow",
        "model_config": model.config.fingerprint(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "normalization": {"mean": mean.cpu(), "scale": scale.cpu()},
        "tokenizer_checkpoint": cache["tokenizer_checkpoint"],
        "tokenizer_step": cache["tokenizer_step"],
        # a rescaled cache carries the per-register magnitude profile; the
        # evaluator must divide it out before handing latents to the decoder
        "token_scale": cache.get("token_scale"),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("joint flow training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    if cache.get("version") != 1:
        raise ValueError("unsupported latent-cache version")
    train_latents = cache["train_latents"]
    test_latents = cache["test_latents"]
    if train_latents.ndim != 3:
        raise ValueError("cached train latents must be [N,L,D]")
    sequence_length, token_dim = train_latents.shape[1:]
    global_mean = cache["statistics"]["global_mean"].float().to(device)
    global_scale = cache["statistics"]["global_std"].float().to(device)
    if not bool(torch.isfinite(global_scale)) or float(global_scale) <= 0:
        raise ValueError("invalid global latent scale")

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
    config = JointFlowConfig(
        sequence_length=sequence_length,
        token_dim=token_dim,
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qk_norm=args.qk_norm,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = JointRectifiedFlow(config).to(device)
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
        with autocast_context(args, device):
            output = model(clean)
        output["loss"].backward()
        gradient_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed_step = step + 1
        rolling["loss"] += float(output["loss"].detach())
        rolling["prediction_rms"] += float(output["prediction_rms"])
        rolling["target_rms"] += float(output["target_rms"])
        rolling_count += 1

        if completed_step == 1 or completed_step % args.log_every == 0:
            torch.cuda.synchronize()
            elapsed = max(time.monotonic() - window_start, 1e-9)
            record = {
                "step": completed_step,
                "loss": rolling["loss"] / rolling_count,
                "prediction_rms": rolling["prediction_rms"] / rolling_count,
                "target_rms": rolling["target_rms"] / rolling_count,
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
                model,
                test_loader,
                global_mean,
                global_scale,
                device,
                args,
            )
            metrics["step"] = completed_step
            (output_dir / "metrics_latest.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({"evaluation": metrics}, sort_keys=True), flush=True)
            tracker.log(metrics, step=completed_step, prefix="eval")

        if args.preview_every > 0 and completed_step % args.preview_every == 0:
            preview_path = output_dir / f"samples_{completed_step:06d}.png"
            preview_metrics = save_preview(
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
                json.dumps(
                    {"preview": {"step": completed_step, **preview_metrics}},
                    sort_keys=True,
                ),
                flush=True,
            )
            tracker.log(preview_metrics, step=completed_step, prefix="preview")
            tracker.log_image(
                preview_path, step=completed_step, key="preview/samples"
            )

        if args.checkpoint_every > 0 and completed_step % args.checkpoint_every == 0:
            latest = output_dir / "checkpoint_latest.pt"
            atomic_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    completed_step,
                    cache,
                    global_mean,
                    global_scale,
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

    final_payload = checkpoint_payload(
        model,
        optimizer,
        args.max_train_steps,
        cache,
        global_mean,
        global_scale,
    )
    final_payload.pop("optimizer")
    atomic_save(final_payload, output_dir / "checkpoint_final.pt")
    print(json.dumps({"complete": args.max_train_steps}), flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
