#!/usr/bin/env python3
"""Standalone trainer for compressed AFIG autoencoder/VAE representations."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator, DataLoaderConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from autoencoder_models import (
    AutoencoderConfig,
    CausalFrequencyAutoencoder,
    SpatialAutoencoder,
)
from frequency import FrequencyCodec, FrequencyCodecConfig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="causal_k",
        choices=["causal_k", "causal_ring", "spatial_downsample"],
    )
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=["auto", "cifar10", "huggingface_cifar", "imagefolder", "synthetic"],
    )
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--output_dir", default="autoencoder_runs/default")
    parser.add_argument("--codec_stats_path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--model_width", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--perceiver_width", type=int, default=256)
    parser.add_argument("--perceiver_heads", type=int, default=4)
    parser.add_argument("--ring_transformer_layers", type=int, default=2)
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Causal TCN depth; 0 chooses full-sequence receptive-field coverage.",
    )
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument(
        "--pooler",
        choices=["flat_mlp", "perceiver_full", "perceiver_sector"],
        default="perceiver_sector",
    )
    parser.add_argument("--target_tokens_per_latent", type=int, default=16)
    parser.add_argument("--max_ring_latents", type=int, default=4)
    parser.add_argument(
        "--group_conditioning",
        choices=["none", "film", "low_rank", "film_low_rank"],
        default="film_low_rank",
    )
    parser.add_argument("--conditioning_rank", type=int, default=16)
    parser.add_argument("--spatial_downsample", type=int, default=4)
    parser.add_argument("--spatial_latent_channels", type=int, default=8)
    parser.add_argument("--spatial_base_channels", type=int, default=64)
    parser.add_argument("--variational", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--kl_free_bits", type=float, default=0.0)
    parser.add_argument("--token_loss_weight", type=float, default=0.01)
    parser.add_argument("--image_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--fourier_loss_weight",
        type=float,
        default=0.0,
        help="Deprecated raw complex FFT MSE weight; duplicates pixel MSE under ortho FFT.",
    )
    parser.add_argument(
        "--reconstruction_loss",
        choices=["mse", "charbonnier"],
        default="mse",
    )
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--log_amplitude_weight", type=float, default=0.0)
    parser.add_argument("--phase_loss_weight", type=float, default=0.0)
    parser.add_argument("--phase_loss_gate", type=float, default=0.1)
    parser.add_argument("--radial_log_power_weight", type=float, default=0.0)
    parser.add_argument("--loss_gradient_diagnostic_steps", type=int, default=1000)
    parser.add_argument("--latent_noise_std", type=float, default=0.0)
    parser.add_argument("--latent_ring_dropout", type=float, default=0.0)
    parser.add_argument("--latent_high_frequency_dropout", type=float, default=0.0)
    parser.add_argument("--latent_moment_weight", type=float, default=0.0)
    parser.add_argument("--prefix_fractions", default="0.25,0.5,0.75")
    parser.add_argument("--eval_panel_size", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--preview_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=0)
    parser.add_argument("--save_final_checkpoint", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--tracker_project_name", default="afig-autoencoder")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--run_group", default="afig-autoencoder-gates")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.dataset = "synthetic"
        args.resolution = 32
        args.train_batch_size = 2
        args.dataloader_num_workers = 0
        args.max_train_steps = 1
        args.model_width = 32
        args.latent_dim = 16
        args.perceiver_width = 32
        args.perceiver_heads = 2
        args.ring_transformer_layers = 1
        args.depth = 2
        args.eval_panel_size = 4
        args.eval_steps = 1
        args.preview_steps = 1
        args.logging_steps = 1
        args.mixed_precision = "no"
    if args.resolution <= 0 or args.resolution % 2:
        parser.error("--resolution must be a positive even integer")
    if args.mode == "spatial_downsample" and args.resolution % args.spatial_downsample:
        parser.error("--resolution must be divisible by --spatial_downsample")
    if args.kl_weight > 0 and not args.variational:
        parser.error("--kl_weight > 0 requires --variational")
    for name in (
        "token_loss_weight",
        "image_loss_weight",
        "fourier_loss_weight",
        "log_amplitude_weight",
        "phase_loss_weight",
        "radial_log_power_weight",
        "latent_noise_std",
        "latent_ring_dropout",
        "latent_high_frequency_dropout",
        "latent_moment_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    return args


def build_model_config(args: argparse.Namespace) -> AutoencoderConfig:
    pooler = args.pooler
    if args.mode == "causal_k" and pooler == "perceiver_sector":
        # A fixed chunk has one exported latent; sector/full are equivalent.
        pooler = "perceiver_full"
    sequence_length = args.resolution * args.resolution // 2 + 2
    auto_depth = math.ceil(
        math.log2((sequence_length - 1) / max(args.kernel_size - 1, 1) + 1)
    )
    depth = args.depth if args.depth > 0 else auto_depth
    return AutoencoderConfig(
        mode=args.mode,
        variational=args.variational,
        latent_dim=args.latent_dim,
        model_width=args.model_width,
        perceiver_width=args.perceiver_width,
        perceiver_heads=args.perceiver_heads,
        ring_transformer_layers=args.ring_transformer_layers,
        depth=depth,
        kernel_size=args.kernel_size,
        group_size=args.group_size,
        pooler=pooler,
        target_tokens_per_latent=args.target_tokens_per_latent,
        max_ring_latents=args.max_ring_latents,
        group_conditioning=args.group_conditioning,
        conditioning_rank=args.conditioning_rank,
        spatial_resolution=args.resolution,
        spatial_downsample=args.spatial_downsample,
        spatial_latent_channels=args.spatial_latent_channels,
        spatial_base_channels=args.spatial_base_channels,
    )


def _synthetic_dataset(size: int, resolution: int, seed: int) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(size, 3, resolution, resolution, generator=generator)
    labels = torch.zeros(size, dtype=torch.long)
    return TensorDataset(images, labels)


def _hf_cifar_dataset(transform) -> Optional[Dataset]:
    try:
        from train_continuous import _dataset_from_hf_arrow, _hf_cifar_paths

        for path in _hf_cifar_paths():
            try:
                return _dataset_from_hf_arrow(path, transform=transform)
            except Exception:
                continue
    except Exception:
        return None
    return None


def make_dataset(args: argparse.Namespace) -> Dataset:
    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    if args.dataset == "synthetic":
        return _synthetic_dataset(64 if args.smoke else 4096, args.resolution, args.seed)
    if args.dataset == "imagefolder":
        if not os.path.isdir(args.data_root):
            raise FileNotFoundError(f"ImageFolder root does not exist: {args.data_root}")
        return torchvision.datasets.ImageFolder(args.data_root, transform=transform)

    batches = os.path.join(args.data_root, "cifar-10-batches-py")
    if args.dataset in ("auto", "cifar10") and os.path.isdir(batches):
        return torchvision.datasets.CIFAR10(
            args.data_root, train=True, download=False, transform=transform
        )
    if args.dataset in ("auto", "huggingface_cifar"):
        dataset = _hf_cifar_dataset(transform)
        if dataset is not None:
            return dataset
    if args.dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            args.data_root, train=True, download=True, transform=transform
        )
    if args.dataset == "auto":
        try:
            return torchvision.datasets.CIFAR10(
                args.data_root, train=True, download=True, transform=transform
            )
        except Exception:
            return _synthetic_dataset(4096, args.resolution, args.seed)
    raise RuntimeError(f"Could not resolve dataset={args.dataset}")


def make_loaders(
    args: argparse.Namespace,
) -> tuple[Dataset, DataLoader, torch.Tensor]:
    dataset = make_dataset(args)
    panel_count = min(args.eval_panel_size, max(len(dataset) // 10, 1))
    train_count = len(dataset) - panel_count
    if train_count < args.train_batch_size:
        raise ValueError("Dataset is too small for the requested train batch size")
    train_dataset = Subset(dataset, range(train_count))
    panel_images = []
    for index in range(train_count, len(dataset)):
        item = dataset[index]
        panel_images.append(item[0] if isinstance(item, (tuple, list)) else item)
    loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.dataloader_num_workers > 0,
    )
    return dataset, loader, torch.stack(panel_images)


def fit_or_load_codec(
    args: argparse.Namespace,
    accelerator: Accelerator,
    train_loader: DataLoader,
) -> FrequencyCodec:
    config = FrequencyCodecConfig(
        height=args.resolution,
        width=args.resolution,
        normalization="orbit_standardize",
        value_transform="identity",
    )
    codec = FrequencyCodec(config)
    stats_path = args.codec_stats_path or os.path.join(
        args.output_dir, f"codec_stats_{args.resolution}.pt"
    )
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        # Separate reconstruction gates commonly share one codec export. Hold an
        # OS-level lock across fit + atomic replacement so concurrent jobs never
        # observe a partial file or redundantly fit the dataset.
        with open(f"{stats_path}.lock", "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(stats_path):
                codec.load_exported(torch.load(stats_path, map_location="cpu"))
            else:
                fit_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=args.train_batch_size,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
                )
                codec.fit_from_loader(
                    fit_loader,
                    max_batches=4 if args.smoke else None,
                    device=torch.device("cpu"),
                )
                temporary_path = f"{stats_path}.tmp.{os.getpid()}"
                torch.save(codec.export_state(), temporary_path)
                os.replace(temporary_path, stats_path)
    accelerator.wait_for_everyone()
    if not bool(codec.is_fitted.item()):
        codec.load_exported(torch.load(stats_path, map_location="cpu"))
    return codec


def build_model(
    config: AutoencoderConfig,
    codec: Optional[FrequencyCodec],
) -> torch.nn.Module:
    if config.mode == "spatial_downsample":
        return SpatialAutoencoder(config)
    if codec is None:
        raise ValueError("Frequency autoencoders require a fitted codec")
    metadata = codec.position_metadata()
    metadata["empirical_scale"] = codec.orbit_scale_for_policy(
        codec.effective_scale_policy()
    ).mean(dim=-1)
    return CausalFrequencyAutoencoder(
        config,
        metadata,
        codec.component_mask,
    )


def _kl_losses(
    kl_per_dim: torch.Tensor,
    mode: str,
    free_bits: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "spatial_downsample":
        per_channel = kl_per_dim.mean(dim=(0, 2, 3))
    else:
        per_channel = kl_per_dim.mean(dim=(0, 1))
    raw = per_channel.mean()
    effective = (per_channel - free_bits).clamp_min(0.0).mean()
    return raw, effective


def _radial_power_error(
    target_fft: torch.Tensor,
    predicted_fft: torch.Tensor,
) -> torch.Tensor:
    height, width = target_fft.shape[-2:]
    ky = torch.fft.fftfreq(height, device=target_fft.device) * height
    kx = torch.fft.fftfreq(width, device=target_fft.device) * width
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
    bins = torch.floor(torch.sqrt(grid_y.square() + grid_x.square())).long()
    target_power = target_fft.abs().square()
    predicted_power = predicted_fft.abs().square()
    errors = []
    for radius in torch.unique(bins):
        select = bins == radius
        target = target_power[..., select].mean()
        predicted = predicted_power[..., select].mean()
        errors.append((predicted - target).abs() / target.clamp_min(1e-8))
    return torch.stack(errors).mean()


def _image_reconstruction_loss(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    difference = reconstruction.float() - target.float()
    if args.reconstruction_loss == "charbonnier":
        return (
            torch.sqrt(difference.square() + args.charbonnier_eps**2)
            - args.charbonnier_eps
        ).mean()
    return difference.square().mean()


def _spectral_loss_terms(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    phase_gate: float,
) -> Dict[str, torch.Tensor]:
    target_fft = torch.fft.fft2(target.float(), norm="ortho")
    predicted_fft = torch.fft.fft2(reconstruction.float(), norm="ortho")
    target_amp = target_fft.abs()
    predicted_amp = predicted_fft.abs()
    log_amplitude = (
        torch.log(predicted_amp.clamp_min(1e-6))
        - torch.log(target_amp.clamp_min(1e-6))
    ).abs().mean()
    reference = target_amp.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    gate = target_amp.square() / (
        target_amp.square() + (phase_gate * reference).square()
    )
    phase_delta = torch.angle(predicted_fft * target_fft.conj())
    phase = (gate * (1.0 - torch.cos(phase_delta))).sum() / gate.sum().clamp_min(1.0)

    height, width = target.shape[-2:]
    ky = torch.fft.fftfreq(height, device=target.device) * height
    kx = torch.fft.fftfreq(width, device=target.device) * width
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
    bins = torch.floor(torch.sqrt(grid_y.square() + grid_x.square())).long()
    radial_terms = []
    for radius in torch.unique(bins):
        select = bins == radius
        target_power = target_fft[..., select].abs().square().mean(dim=-1)
        predicted_power = predicted_fft[..., select].abs().square().mean(dim=-1)
        radial_terms.append(
            (
                torch.log(predicted_power.clamp_min(1e-8))
                - torch.log(target_power.clamp_min(1e-8))
            ).abs().mean()
        )
    return {
        "log_amplitude": log_amplitude,
        "phase": phase,
        "radial_log_power": torch.stack(radial_terms).mean(),
        "raw_complex_mse": (predicted_fft - target_fft).abs().square().mean(),
    }


def _corrupt_latents(
    model: torch.nn.Module,
    latents: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    corrupted = latents
    if args.latent_noise_std > 0:
        dims = tuple(range(1, latents.ndim))
        rms = latents.float().square().mean(dim=dims, keepdim=True).sqrt()
        corrupted = corrupted + torch.randn_like(corrupted) * (
            args.latent_noise_std * rms.to(corrupted.dtype)
        )
    if latents.ndim == 3 and (
        args.latent_ring_dropout > 0 or args.latent_high_frequency_dropout > 0
    ):
        keep = torch.ones(
            latents.shape[0], latents.shape[1], 1, device=latents.device, dtype=latents.dtype
        )
        if args.latent_ring_dropout > 0:
            keep = keep * (
                torch.rand_like(keep) >= args.latent_ring_dropout
            ).to(keep.dtype)
        if args.latent_high_frequency_dropout > 0:
            high = torch.arange(latents.shape[1], device=latents.device)
            high = high >= latents.shape[1] // 2
            random_drop = (
                torch.rand(latents.shape[0], latents.shape[1], 1, device=latents.device)
                < args.latent_high_frequency_dropout
            )
            keep = keep * (~(random_drop & high[None, :, None])).to(keep.dtype)
        corrupted = corrupted * keep
    elif latents.ndim == 4 and args.latent_high_frequency_dropout > 0:
        tokens = model.latent_fft.encode(corrupted)
        high = torch.arange(tokens.shape[1], device=tokens.device) >= tokens.shape[1] // 2
        drop = (
            torch.rand(tokens.shape[0], tokens.shape[1], 1, device=tokens.device)
            < args.latent_high_frequency_dropout
        )
        tokens = tokens * (~(drop & high[None, :, None])).to(tokens.dtype)
        corrupted = model.latent_fft.decode(tokens).to(latents.dtype)
    return corrupted


@torch.no_grad()
def reconstruction_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    target = target.float()
    reconstruction = reconstruction.float()
    pixel_mse = (reconstruction - target).square().mean()
    target_fft = torch.fft.fft2(target, norm="ortho")
    predicted_fft = torch.fft.fft2(reconstruction, norm="ortho")
    difference = predicted_fft - target_fft
    nrmse = (
        difference.abs().square().sum()
        / target_fft.abs().square().sum().clamp_min(1e-8)
    ).sqrt()
    target_amp = target_fft.abs()
    predicted_amp = predicted_fft.abs()
    log_amp = (
        torch.log(predicted_amp.clamp_min(1e-6))
        - torch.log(target_amp.clamp_min(1e-6))
    ).abs().mean()
    phase_delta = torch.angle(predicted_fft * target_fft.conj())
    phase_valid = target_amp > target_amp.mean() * 0.05
    phase_error = (1.0 - torch.cos(phase_delta))[phase_valid].mean()
    return {
        "reconstruction/pixel_mse": pixel_mse,
        "reconstruction/psnr": -10.0 * torch.log10(pixel_mse.clamp_min(1e-12)),
        "reconstruction/physical_fourier_nrmse": nrmse,
        "reconstruction/log_amplitude_mae": log_amp,
        "reconstruction/phase_circular_error": phase_error,
        "reconstruction/radial_power_relative_error": _radial_power_error(
            target_fft, predicted_fft
        ),
    }


def compute_batch_loss(
    model: torch.nn.Module,
    codec: Optional[FrequencyCodec],
    images: torch.Tensor,
    args: argparse.Namespace,
    *,
    sample_posterior: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    if args.mode == "spatial_downsample":
        output = model(images, sample_posterior=sample_posterior)
        token_loss = images.new_zeros(())
    else:
        if codec is None:
            raise ValueError("Frequency loss requires codec")
        with torch.no_grad():
            tokens = codec.encode(images)
        output = model(tokens, sample_posterior=sample_posterior)
        reconstructed_tokens = output["reconstruction"]
        mask = codec.component_mask[None].to(reconstructed_tokens.dtype)
        token_loss = (
            (reconstructed_tokens.float() - tokens.float()).square() * mask
        ).sum() / (mask.sum() * tokens.shape[0]).clamp_min(1.0)
    latents = output["latents"]
    if args.latent_moment_weight > 0:
        if latents.ndim == 3:
            latent_samples = latents.float().reshape(-1, latents.shape[-1])
        else:
            latent_samples = latents.float().permute(0, 2, 3, 1).reshape(
                -1, latents.shape[1]
            )
        latent_mean = latent_samples.mean(dim=0)
        latent_variance = latent_samples.var(dim=0, unbiased=False)
        latent_moment_loss = latent_mean.square().mean() + (
            latent_variance - 1.0
        ).square().mean()
    else:
        latent_moment_loss = latents.new_zeros(())
    if model.training and (
        args.latent_noise_std > 0
        or args.latent_ring_dropout > 0
        or args.latent_high_frequency_dropout > 0
    ):
        latents = _corrupt_latents(model, latents, args)
    if args.mode == "spatial_downsample":
        reconstruction = model.decode(latents)
    else:
        assert codec is not None
        reconstruction = codec.decode(model.decode(latents))

    image_loss = _image_reconstruction_loss(images, reconstruction, args)
    spectral_enabled = (
        args.fourier_loss_weight > 0
        or args.log_amplitude_weight > 0
        or args.phase_loss_weight > 0
        or args.radial_log_power_weight > 0
    )
    if spectral_enabled:
        spectral = _spectral_loss_terms(
            images, reconstruction, args.phase_loss_gate
        )
    else:
        zero = image_loss.new_zeros(())
        spectral = {
            "log_amplitude": zero,
            "phase": zero,
            "radial_log_power": zero,
            "raw_complex_mse": zero,
        }
    fourier_loss = spectral["raw_complex_mse"]
    raw_kl, effective_kl = _kl_losses(
        output["kl_per_dim"], args.mode, args.kl_free_bits
    )
    loss = (
        args.token_loss_weight * token_loss
        + args.image_loss_weight * image_loss
        + args.fourier_loss_weight * fourier_loss
        + args.log_amplitude_weight * spectral["log_amplitude"]
        + args.phase_loss_weight * spectral["phase"]
        + args.radial_log_power_weight * spectral["radial_log_power"]
        + args.kl_weight * effective_kl
        + args.latent_moment_weight * latent_moment_loss
    )
    logs = {
        "loss": loss.detach(),
        "loss/token": token_loss.detach(),
        "loss/image": image_loss.detach(),
        "loss/fourier": fourier_loss.detach(),
        "loss/log_amplitude": spectral["log_amplitude"].detach(),
        "loss/phase": spectral["phase"].detach(),
        "loss/radial_log_power": spectral["radial_log_power"].detach(),
        "loss/kl_raw": raw_kl.detach(),
        "loss/kl_effective": effective_kl.detach(),
        "loss/latent_moment": latent_moment_loss.detach(),
        "latent/mean": output["mean"].detach().float().mean(),
        "latent/std": output["latents"].detach().float().std(),
        "latent/rms": output["latents"].detach().float().square().mean().sqrt(),
    }
    return loss, logs, reconstruction


def loss_gradient_ratios(
    images: torch.Tensor,
    reconstruction: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    if (
        args.log_amplitude_weight == 0
        and args.phase_loss_weight == 0
        and args.radial_log_power_weight == 0
    ):
        return {
            "gradient_ratio/log_amplitude": 0.0,
            "gradient_ratio/phase": 0.0,
            "gradient_ratio/radial_log_power": 0.0,
        }
    base = _image_reconstruction_loss(images, reconstruction, args)
    spectral = _spectral_loss_terms(images, reconstruction, args.phase_loss_gate)
    base_grad = torch.autograd.grad(
        base, reconstruction, retain_graph=True, allow_unused=False
    )[0]
    base_norm = base_grad.float().norm().clamp_min(1e-12)
    ratios: Dict[str, float] = {}
    weighted_terms = {
        "log_amplitude": args.log_amplitude_weight * spectral["log_amplitude"],
        "phase": args.phase_loss_weight * spectral["phase"],
        "radial_log_power": args.radial_log_power_weight
        * spectral["radial_log_power"],
    }
    for name, term in weighted_terms.items():
        if float(term.detach().abs().item()) == 0.0:
            ratios[f"gradient_ratio/{name}"] = 0.0
            continue
        gradient = torch.autograd.grad(
            term, reconstruction, retain_graph=True, allow_unused=False
        )[0]
        ratios[f"gradient_ratio/{name}"] = float(
            (gradient.float().norm() / base_norm).item()
        )
    return ratios


def _decode_latents_to_images(
    model: torch.nn.Module,
    codec: Optional[FrequencyCodec],
    latents: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    reconstructed = model.decode(latents)
    return codec.decode(reconstructed) if mode != "spatial_downsample" else reconstructed


@torch.no_grad()
def latent_diagnostics(
    model: torch.nn.Module,
    codec: Optional[FrequencyCodec],
    images: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    if args.mode == "spatial_downsample":
        output = model(images, sample_posterior=False)
    else:
        assert codec is not None
        output = model(codec.encode(images), sample_posterior=False)
    latents = output["latents"].float()
    flattened = (
        latents.permute(0, 2, 3, 1).reshape(-1, latents.shape[1])
        if latents.ndim == 4
        else latents.reshape(-1, latents.shape[-1])
    )
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    diagonal = covariance.diag().clamp_min(1e-8)
    correlation = covariance / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    off_diagonal = correlation - torch.diag_embed(correlation.diag())
    logs: Dict[str, torch.Tensor] = {
        "latent/covariance_condition": torch.linalg.cond(
            covariance + 1e-5 * torch.eye(covariance.shape[0], device=covariance.device)
        ),
        "latent/correlation_offdiag_rms": off_diagonal.square().mean().sqrt(),
    }

    if latents.ndim == 4:
        tokens = model.latent_fft.encode(latents)
        roundtrip = model.latent_fft.decode(tokens)
        logs["latent/hermitian_roundtrip_mse"] = (roundtrip - latents).square().mean()
        prefix_source = tokens
        decode_prefix = lambda value: model.decode(model.latent_fft.decode(value).to(latents.dtype))
    else:
        prefix_source = latents
        decode_prefix = lambda value: _decode_latents_to_images(
            model, codec, value.to(latents.dtype), args.mode
        )

    fractions = [
        float(item) for item in str(args.prefix_fractions).split(",") if item.strip()
    ]
    for fraction in fractions:
        count = max(1, min(prefix_source.shape[1], math.ceil(prefix_source.shape[1] * fraction)))
        prefix = prefix_source.clone()
        prefix[:, count:] = 0
        reconstruction = decode_prefix(prefix)
        logs[f"latent_prefix/pixel_mse_{fraction:.2f}"] = (
            reconstruction.float() - images.float()
        ).square().mean()
        logs[f"latent_prefix/psnr_{fraction:.2f}"] = -10.0 * torch.log10(
            logs[f"latent_prefix/pixel_mse_{fraction:.2f}"].clamp_min(1e-12)
        )

    rms = latents.square().mean().sqrt()
    noisy = latents + 0.1 * rms * torch.randn_like(latents)
    noisy_reconstruction = _decode_latents_to_images(model, codec, noisy, args.mode)
    logs["latent_perturb/noise_0.1_pixel_mse"] = (
        noisy_reconstruction.float() - images.float()
    ).square().mean()

    clean = output["reconstruction"]
    if args.mode != "spatial_downsample":
        clean = codec.decode(clean)
    residual = clean.float() - images.float()
    edge_target_x = images[..., 1:] - images[..., :-1]
    edge_recon_x = clean[..., 1:] - clean[..., :-1]
    edge_target_y = images[..., 1:, :] - images[..., :-1, :]
    edge_recon_y = clean[..., 1:, :] - clean[..., :-1, :]
    logs["artifact/edge_mse"] = 0.5 * (
        (edge_target_x - edge_recon_x).square().mean()
        + (edge_target_y - edge_recon_y).square().mean()
    )
    parity_means = [
        residual[..., y::2, x::2].mean() for y in range(2) for x in range(2)
    ]
    logs["artifact/checkerboard_residual_std"] = torch.stack(parity_means).std()
    boundary = torch.cat(
        [
            residual[..., 0, :].flatten(),
            residual[..., -1, :].flatten(),
            residual[..., :, 0].flatten(),
            residual[..., :, -1].flatten(),
        ]
    )
    logs["artifact/boundary_mse"] = boundary.square().mean()
    return logs


def compression_logs(
    model: torch.nn.Module,
    args: argparse.Namespace,
    codec: Optional[FrequencyCodec],
) -> Dict[str, float]:
    if args.mode == "spatial_downsample":
        latent_scalars = (
            args.spatial_latent_channels
            * (args.resolution // args.spatial_downsample) ** 2
        )
        source_scalars = 3 * args.resolution**2
        exported = model.exported_token_count
        source_tokens = args.resolution * args.resolution // 2 + 2
    else:
        assert codec is not None
        latent_scalars = model.exported_token_count * args.latent_dim
        source_scalars = int(codec.component_mask.sum().item())
        exported = model.exported_token_count
        source_tokens = codec.seq_len
    logs = {
        "compression/exported_tokens": float(exported),
        "compression/source_to_latent_token_ratio": float(
            source_tokens / exported
        ),
        "compression/source_to_latent_scalar_ratio": source_scalars
        / max(latent_scalars, 1),
    }
    if args.mode != "spatial_downsample":
        if args.mode == "causal_ring":
            logs["compression/causal_receptive_field"] = float(codec.seq_len)
            logs["compression/causal_depth"] = float(
                args.ring_transformer_layers
            )
        else:
            logs["compression/causal_receptive_field"] = float(
                model.encoder.receptive_field
            )
            logs["compression/causal_depth"] = float(model.effective_depth)
    return logs


def save_checkpoint(
    path: str,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    config: AutoencoderConfig,
    global_step: int,
    codec: Optional[FrequencyCodec],
) -> None:
    payload = {
        "version": 1,
        "global_step": global_step,
        "args": vars(args),
        "config": config.fingerprint(),
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "codec": codec.export_state() if codec is not None else None,
    }
    torch.save(payload, path)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        dataloader_config=DataLoaderConfiguration(
            non_blocking=torch.cuda.is_available() and not args.smoke
        ),
    )
    dataset, train_loader, panel = make_loaders(args)
    del dataset
    codec = (
        None
        if args.mode == "spatial_downsample"
        else fit_or_load_codec(args, accelerator, train_loader)
    )
    config = build_model_config(args)
    model = build_model(config, codec)
    if codec is not None:
        codec.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=min(args.lr_warmup_steps, args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    global_step = 0
    if args.resume_from_checkpoint:
        payload = torch.load(args.resume_from_checkpoint, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        global_step = int(payload["global_step"])

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.report_to != "none":
            run_name = args.run_name or (
                f"ae-{args.mode}-z{args.latent_dim}-s{args.seed}-n{args.max_train_steps}"
            )
            accelerator.init_trackers(
                args.tracker_project_name,
                config={**vars(args), **{f"model/{k}": v for k, v in config.fingerprint().items()}},
                init_kwargs={
                    "wandb": {
                        "name": run_name,
                        "group": args.run_group,
                    }
                }
                if args.report_to == "wandb"
                else None,
            )
    panel = panel.to(accelerator.device)
    compression = compression_logs(accelerator.unwrap_model(model), args, codec)
    iterator = iter(train_loader)
    window_started = time.perf_counter()
    window_step = global_step
    model.train()
    while global_step < args.max_train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        images = batch[0].to(accelerator.device)
        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss, batch_logs, reconstruction = compute_batch_loss(
                model, codec, images, args, sample_posterior=args.variational
            )
        gradient_logs: Dict[str, float] = {}
        if (
            args.loss_gradient_diagnostic_steps > 0
            and (global_step + 1) % args.loss_gradient_diagnostic_steps == 0
        ):
            gradient_logs = loss_gradient_ratios(images, reconstruction, args)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        should_log = global_step % args.logging_steps == 0 or global_step == args.max_train_steps
        should_eval = global_step % args.eval_steps == 0 or global_step == args.max_train_steps
        should_preview = (
            args.preview_steps > 0
            and (global_step % args.preview_steps == 0 or global_step == args.max_train_steps)
        )
        logs: Dict[str, float] = {}
        if should_log:
            elapsed = max(time.perf_counter() - window_started, 1e-8)
            steps = max(global_step - window_step, 1)
            logs.update({key: float(value.item()) for key, value in batch_logs.items()})
            logs.update(gradient_logs)
            logs.update(compression)
            logs["grad_norm"] = float(
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
            )
            logs["lr"] = scheduler.get_last_lr()[0]
            logs["performance/steps_per_sec"] = steps / elapsed
            logs["performance/examples_per_sec"] = (
                steps * args.train_batch_size / elapsed
            )
            window_started = time.perf_counter()
            window_step = global_step

        if (should_eval or should_preview) and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.eval()
            with torch.no_grad(), accelerator.autocast():
                _, eval_logs, reconstruction = compute_batch_loss(
                    unwrapped,
                    codec,
                    panel,
                    args,
                    sample_posterior=False,
                )
            if should_eval:
                logs.update(
                    {f"eval/{key}": float(value.item()) for key, value in eval_logs.items()}
                )
                logs.update(
                    {
                        key: float(value.item())
                        for key, value in latent_diagnostics(
                            unwrapped, codec, panel, args
                        ).items()
                    }
                )
                logs.update(
                    {
                        key: float(value.item())
                        for key, value in reconstruction_metrics(panel, reconstruction).items()
                    }
                )
            if should_preview:
                count = min(8, panel.shape[0])
                grid = torch.cat([panel[:count], reconstruction[:count]], dim=0)
                save_image(
                    grid.clamp(0, 1),
                    os.path.join(args.output_dir, f"reconstruction_{global_step}.png"),
                    nrow=count,
                )
            unwrapped.train()

        if logs and args.report_to != "none":
            accelerator.log(logs, step=global_step)
        if (
            args.checkpointing_steps > 0
            and global_step % args.checkpointing_steps == 0
            and accelerator.is_main_process
        ):
            save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{global_step}.pt"),
                accelerator,
                model,
                optimizer,
                scheduler,
                args,
                config,
                global_step,
                codec,
            )

    if args.save_final_checkpoint and accelerator.is_main_process:
        save_checkpoint(
            os.path.join(args.output_dir, f"checkpoint_{global_step}.pt"),
            accelerator,
            model,
            optimizer,
            scheduler,
            args,
            config,
            global_step,
            codec,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
