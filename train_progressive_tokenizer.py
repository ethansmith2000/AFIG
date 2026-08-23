#!/usr/bin/env python3
"""Train the deterministic whole-image continuous-token autoencoder."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from progressive_tokenizer import ProgressiveTokenizer, TokenizerConfig
from progressive_tokenizer.training import (
    LatentMomentAccumulator,
    count_parameters,
    optimizer_parameter_groups,
    pixel_psnr,
)
from progressive_tokenizer.tracking import WandbTracker


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="tokenizer_runs/n32-d64-full-s1")
    parser.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--objective", choices=["full", "progressive"], default="full")
    parser.add_argument("--prefix_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--variational",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encoder emits a posterior mean and log-variance; training samples.",
    )
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument(
        "--hard_log_variance_clamp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the historical hard [-8, 8] posterior log-variance clamp. "
            "This is retained for exact controls against checkpoints trained "
            "before the differentiable bound was introduced."
        ),
    )
    parser.add_argument(
        "--energy_reg_weight",
        type=float,
        default=0.0,
        help="Weight on the per-coordinate kurtosis->3 energy-consistency penalty.",
    )
    parser.add_argument(
        "--latent_shaping",
        choices=["none", "frontier", "ramp"],
        default="none",
        help="Replace the random-prefix loss with a latent-noise reconstruction: "
        "'frontier' mixes toward noise along a sampled crescendo frontier, "
        "'ramp' adds a static ascending per-token noise floor.",
    )
    parser.add_argument("--shaping_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--frontier_overlap",
        type=float,
        default=8.0,
        help="Rolling-schedule overlap: token i's data time is "
        "clamp(frontier - i/overlap, 0, 1).",
    )
    parser.add_argument("--ramp_sigma_max", type=float, default=1.0)
    parser.add_argument("--ramp_power", type=float, default=1.0)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--encoder_depth", type=int, default=8)
    parser.add_argument("--pool_depth", type=int, default=2)
    parser.add_argument(
        "--pool_type", choices=["residual", "cross_only"], default="residual"
    )
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument(
        "--qk_norm", choices=["rms", "l2_temperature"], default="rms"
    )
    parser.add_argument(
        "--cross_attention_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.995)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--run_group", default="tokenizer")

    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_examples", type=int, default=2048)
    parser.add_argument("--checkpoint_every", type=int, default=2500)
    parser.add_argument(
        "--discard_resume_state",
        action="store_true",
        help="Delete checkpoint_latest.pt on completion. Off by default: it "
        "carries the optimizer state a completed run needs to be extended.",
    )
    parser.add_argument(
        "--keep_numbered_checkpoints",
        action="store_true",
        help="Retain every periodic optimizer checkpoint in addition to the latest one.",
    )
    parser.add_argument("--preview_examples", type=int, default=16)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init_from",
        default=None,
        help="Load model weights but start a fresh optimizer and step count.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.prefix_loss_weight < 0:
        parser.error("--prefix_loss_weight must be non-negative")
    if args.kl_weight < 0 or args.shaping_loss_weight < 0 or args.energy_reg_weight < 0:
        parser.error("loss weights must be non-negative")
    if args.kl_weight > 0 and not args.variational:
        parser.error("--kl_weight requires --variational")
    if args.latent_shaping != "none" and args.objective != "progressive":
        parser.error(
            "--latent_shaping replaces the prefix term of the progressive objective"
        )
    if args.frontier_overlap <= 0:
        parser.error("--frontier_overlap must be positive")
    if args.ramp_sigma_max < 0 or args.ramp_power <= 0:
        parser.error("--ramp_sigma_max must be >= 0 and --ramp_power > 0")
    if args.resume and args.init_from:
        parser.error("--resume and --init_from are mutually exclusive")
    if args.smoke:
        args.dataset = "synthetic"
        args.image_size = 8
        args.patch_size = 4
        args.num_latents = 4
        args.latent_dim = 8
        args.width = 64
        args.num_heads = 4
        args.encoder_depth = 1
        args.pool_depth = 1
        args.decoder_depth = 1
        args.mlp_ratio = 2.0
        args.train_batch_size = 4
        args.eval_batch_size = 4
        args.num_workers = 0
        args.max_train_steps = 2
        args.warmup_steps = 1
        args.log_every = 1
        args.eval_every = 1
        args.eval_examples = 8
        args.checkpoint_every = 0
        args.preview_examples = 4
        args.mixed_precision = "no"
        args.compile = False
        args.report_to = "none"
    return args


def make_model_config(args: argparse.Namespace) -> TokenizerConfig:
    return TokenizerConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        width=args.width,
        num_heads=args.num_heads,
        encoder_depth=args.encoder_depth,
        pool_depth=args.pool_depth,
        pool_type=args.pool_type,
        decoder_depth=args.decoder_depth,
        mlp_ratio=args.mlp_ratio,
        qk_norm=args.qk_norm,
        cross_attention_bias=args.cross_attention_bias,
        variational=args.variational,
        hard_log_variance_clamp=args.hard_log_variance_clamp,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _synthetic_dataset(size: int, resolution: int, seed: int) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    # Smooth structured samples make the smoke loss more informative than white noise.
    low_resolution = max(2, resolution // 4)
    values = torch.rand(
        size, 3, low_resolution, low_resolution, generator=generator
    )
    images = F.interpolate(values, size=(resolution, resolution), mode="bilinear")
    images = images.mul(2.0).sub(1.0)
    return TensorDataset(images, torch.zeros(size, dtype=torch.long))


def make_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset == "synthetic":
        return (
            _synthetic_dataset(
                max(64, 2 * args.train_batch_size), args.image_size, args.seed
            ),
            _synthetic_dataset(
                max(16, args.eval_batch_size), args.image_size, args.seed + 1
            ),
        )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.mul(2.0).sub(1.0)),
        ]
    )
    train = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )
    test = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    return train, test


def make_loaders(
    args: argparse.Namespace, train: Dataset, test: Dataset
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = args.mixed_precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    base_lr: float,
    warmup_steps: int,
) -> float:
    multiplier = min(1.0, (step + 1) / max(1, warmup_steps))
    learning_rate = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_preview(
    model: ProgressiveTokenizer,
    images: torch.Tensor,
    path: Path,
    args: argparse.Namespace,
    prefixes: Optional[Sequence[int]] = None,
) -> None:
    model.eval()
    with torch.no_grad(), autocast_context(args, images.device):
        latents = model.encode(images)
        requested = list(prefixes) if prefixes is not None else [args.num_latents]
        reconstructions = [
            model.decode(latents, prefix_lengths=prefix) for prefix in requested
        ]
    count = min(args.preview_examples, images.shape[0])
    panel = torch.cat(
        [images[:count]] + [reconstruction[:count] for reconstruction in reconstructions],
        dim=0,
    )
    save_image(panel.float().add(1).div(2).clamp(0, 1), path, nrow=count)


@torch.no_grad()
def evaluate(
    model: ProgressiveTokenizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    max_examples: Optional[int],
    prefixes: Sequence[int],
    collect_latent_stats: bool,
) -> dict:
    model.eval()
    prefix_sse = {int(prefix): 0.0 for prefix in prefixes}
    element_count = 0
    example_count = 0
    moments = (
        LatentMomentAccumulator(args.latent_dim, args.num_latents)
        if collect_latent_stats
        else None
    )
    for images, _ in loader:
        if max_examples is not None and example_count >= max_examples:
            break
        if max_examples is not None:
            images = images[: max_examples - example_count]
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            latents = model.encode(images)
            reconstructions = {
                prefix: model.decode(latents, prefix_lengths=prefix)
                for prefix in prefixes
            }
        if moments is not None:
            moments.update(latents)
        for prefix, reconstruction in reconstructions.items():
            prefix_sse[prefix] += float(
                (reconstruction.float() - images.float()).square().sum()
            )
        element_count += images.numel()
        example_count += images.shape[0]
    metrics = {
        "examples": example_count,
        "prefix": {},
    }
    for prefix in prefixes:
        mse = prefix_sse[prefix] / element_count
        metrics["prefix"][str(prefix)] = {
            "mse_normalized": mse,
            "mse_pixel": mse / 4.0,
            "psnr_db": pixel_psnr(mse),
        }
    if moments is not None:
        metrics["latent"] = moments.compute()
    return metrics


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = make_model_config(args)
    config_payload = {"model": model_config.fingerprint(), "training": vars(args)}
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    )

    if not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("A CUDA device is required for non-smoke training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    train_dataset, test_dataset = make_datasets(args)
    train_loader, test_loader = make_loaders(
        args, train_dataset, test_dataset
    )
    model = ProgressiveTokenizer(model_config).to(device)
    groups = optimizer_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
    )
    start_step = 0
    if args.init_from:
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if payload["model_config"] != model_config.fingerprint():
            raise ValueError("initial checkpoint model configuration does not match")
        model.load_state_dict(payload["model"])
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload["model_config"] != model_config.fingerprint():
            raise ValueError("resume checkpoint model configuration does not match")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
    if args.compile:
        model.compile(mode="default", fullgraph=True)

    parameter_count = count_parameters(model.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": parameter_count,
                "model": model_config.fingerprint(),
                "objective": args.objective,
                "train_examples": len(train_dataset),
                "test_examples": len(test_dataset),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tracker = WandbTracker(
        enabled=args.report_to == "wandb",
        output_dir=output_dir,
        project=args.tracker_project_name,
        name=args.run_name or output_dir.name,
        group=args.run_group,
        config=config_payload,
    )

    fixed_images = next(iter(test_loader))[0][: args.preview_examples].to(device)
    history_path = output_dir / "history.jsonl"
    train_iterator = iter(train_loader)
    rolling_loss = 0.0
    rolling_full = 0.0
    rolling_prefix = 0.0
    rolling_shaping = 0.0
    rolling_kl = 0.0
    rolling_kurtosis = 0.0
    rolling_energy_cv = 0.0
    rolling_count = 0
    rolling_start = time.monotonic()

    for step in range(start_step, args.max_train_steps):
        (images, _), train_iterator = _next_batch(train_iterator, train_loader)
        images = images.to(device, non_blocking=True)
        learning_rate = set_learning_rate(
            optimizer, step, args.learning_rate, args.warmup_steps
        )
        optimizer.zero_grad(set_to_none=True)
        prefix_lengths = None
        noise_mode = None
        noise_scales = None
        if args.objective == "progressive" and args.latent_shaping == "none":
            prefix_lengths = torch.randint(
                1,
                args.num_latents,
                (images.shape[0],),
                device=device,
            )
        elif args.latent_shaping == "frontier":
            noise_mode = "mix"
            duration = (args.num_latents - 1) / args.frontier_overlap + 1.0
            frontier = torch.rand(images.shape[0], 1, device=device) * duration
            index_time = (
                torch.arange(args.num_latents, device=device, dtype=torch.float32)
                / args.frontier_overlap
            )
            noise_scales = (frontier - index_time[None, :]).clamp(0.0, 1.0)
        elif args.latent_shaping == "ramp":
            noise_mode = "add"
            positions = torch.arange(
                args.num_latents, device=device, dtype=torch.float32
            )
            sigmas = args.ramp_sigma_max * (
                positions / max(1, args.num_latents - 1)
            ) ** args.ramp_power
            noise_scales = sigmas[None, :].expand(images.shape[0], -1)
        with autocast_context(args, device):
            output = model(
                images,
                prefix_lengths,
                include_full_reconstruction=args.objective == "progressive",
                noise_mode=noise_mode,
                noise_scales=noise_scales,
            )
            latents = output["latents"]
            full_reconstruction = output.get(
                "full_reconstruction", output["reconstruction"]
            )
            full_loss = F.mse_loss(full_reconstruction, images)
            prefix_loss = full_loss.new_zeros(())
            shaping_loss = full_loss.new_zeros(())
            prefix_mean = float(args.num_latents)
            if prefix_lengths is not None:
                prefix_loss = F.mse_loss(output["reconstruction"], images)
                prefix_mean = float(prefix_lengths.float().mean())
            elif noise_mode is not None:
                shaping_loss = F.mse_loss(output["reconstruction"], images)
            loss = (
                full_loss
                + args.prefix_loss_weight * prefix_loss
                + args.shaping_loss_weight * shaping_loss
            )
        kl_per_dim = loss.new_zeros(())
        if args.variational:
            posterior_mean = output["mean"].float()
            log_variance = output["log_variance"].float()
            kl_per_dim = 0.5 * (
                posterior_mean.square() + log_variance.exp() - 1.0 - log_variance
            ).mean()
            loss = loss + args.kl_weight * kl_per_dim
        kurtosis_penalty = loss.new_zeros(())
        token_energy_cv = loss.new_zeros(())
        if args.energy_reg_weight > 0:
            coordinates = latents.float().reshape(latents.shape[0], -1)
            centered = coordinates - coordinates.mean(dim=0)
            second = centered.square().mean(dim=0)
            fourth = centered.pow(4).mean(dim=0)
            kurtosis = fourth / (second.square() + 1e-8)
            kurtosis_penalty = (kurtosis - 3.0).square().mean()
            token_energy = centered.reshape(latents.shape).square().mean(dim=-1)
            token_energy_cv = (
                token_energy.var(dim=0)
                / (token_energy.mean(dim=0).square() + 1e-8)
            ).mean()
            loss = loss + args.energy_reg_weight * kurtosis_penalty
        loss.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        completed_step = step + 1

        rolling_loss += float(loss.detach())
        rolling_full += float(full_loss.detach())
        rolling_prefix += float(prefix_loss.detach())
        rolling_shaping += float(shaping_loss.detach())
        rolling_kl += float(kl_per_dim.detach())
        rolling_kurtosis += float(kurtosis_penalty.detach())
        rolling_energy_cv += float(token_energy_cv.detach())
        rolling_count += 1
        if completed_step % args.log_every == 0 or completed_step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.monotonic() - rolling_start, 1e-9)
            record = {
                "step": completed_step,
                "loss": rolling_loss / rolling_count,
                "full_loss": rolling_full / rolling_count,
                "prefix_loss": rolling_prefix / rolling_count,
                "shaping_loss": rolling_shaping / rolling_count,
                "kl_per_dim": rolling_kl / rolling_count,
                "kurtosis_penalty": rolling_kurtosis / rolling_count,
                "token_energy_cv": rolling_energy_cv / rolling_count,
                "prefix_mean": prefix_mean,
                "psnr_db": pixel_psnr(rolling_full / rolling_count),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "images_per_second": rolling_count * images.shape[0] / elapsed,
            }
            with history_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            tracker.log(record, step=completed_step, prefix="train")
            rolling_loss = rolling_full = rolling_prefix = 0.0
            rolling_shaping = rolling_kl = 0.0
            rolling_kurtosis = rolling_energy_cv = 0.0
            rolling_count = 0
            rolling_start = time.monotonic()

        if args.eval_every > 0 and completed_step % args.eval_every == 0:
            evaluation_prefixes = (
                tuple(
                    sorted(
                        set(
                            prefix
                            for prefix in (1, 2, 4, 8, 16, 32, args.num_latents)
                            if prefix <= args.num_latents
                        )
                    )
                )
                if args.objective == "progressive"
                else (args.num_latents,)
            )
            metrics = evaluate(
                model,
                test_loader,
                device,
                args,
                max_examples=args.eval_examples,
                prefixes=evaluation_prefixes,
                collect_latent_stats=False,
            )
            metrics["step"] = completed_step
            (output_dir / "metrics_latest.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n"
            )
            preview_path = output_dir / f"reconstruction_{completed_step:06d}.png"
            save_preview(
                model,
                fixed_images,
                preview_path,
                args,
                prefixes=evaluation_prefixes,
            )
            print(json.dumps({"evaluation": metrics}, sort_keys=True), flush=True)
            tracker.log(metrics, step=completed_step, prefix="eval")
            tracker.log_image(
                preview_path,
                step=completed_step,
                key="eval/reconstruction",
            )
            model.train()

        if args.checkpoint_every > 0 and completed_step % args.checkpoint_every == 0:
            latest_checkpoint = output_dir / "checkpoint_latest.pt"
            atomic_torch_save(
                {
                    "model_config": model_config.fingerprint(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": completed_step,
                },
                latest_checkpoint,
            )
            if args.keep_numbered_checkpoints:
                numbered_checkpoint = output_dir / f"checkpoint_{completed_step:06d}.pt"
                if numbered_checkpoint.exists():
                    numbered_checkpoint.unlink()
                os.link(latest_checkpoint, numbered_checkpoint)

    prefixes = sorted(
        set(
            prefix
            for prefix in (1, 2, 4, 8, 16, 32, args.num_latents)
            if prefix <= args.num_latents
        )
    )
    final_metrics = evaluate(
        model,
        test_loader,
        device,
        args,
        max_examples=None,
        prefixes=prefixes,
        collect_latent_stats=True,
    )
    final_metrics.update(
        {
            "step": args.max_train_steps,
            "parameters": parameter_count,
            "objective": args.objective,
        }
    )
    (output_dir / "metrics_final.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n"
    )
    save_preview(
        model,
        fixed_images,
        output_dir / "reconstruction_final.png",
        args,
        prefixes=prefixes,
    )
    atomic_torch_save(
        {
            "model_config": model_config.fingerprint(),
            "model": model.state_dict(),
            "step": args.max_train_steps,
            "metrics": final_metrics,
        },
        output_dir / "checkpoint_final.pt",
    )
    # checkpoint_latest.pt is the only optimizer-bearing checkpoint, so
    # deleting it makes a completed run impossible to extend or resume. This
    # project has already lost weights once; keep it unless asked otherwise.
    latest = output_dir / "checkpoint_latest.pt"
    if args.discard_resume_state and latest.exists():
        latest.unlink()
    print(json.dumps({"final": final_metrics}, sort_keys=True), flush=True)
    tracker.log(final_metrics, step=args.max_train_steps, prefix="eval/final")
    tracker.log_image(
        output_dir / "reconstruction_final.png",
        step=args.max_train_steps,
        key="eval/final_reconstruction",
    )
    tracker.finish()


if __name__ == "__main__":
    main()
