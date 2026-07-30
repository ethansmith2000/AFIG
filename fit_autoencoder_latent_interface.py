#!/usr/bin/env python3
"""Fit exported-latent statistics and a next-token causal probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from autoencoder_models import (
    AutoencoderConfig,
    CausalFrequencyAutoencoder,
    LatentCausalProbe,
)
from frequency import FrequencyCodec, FrequencyCodecConfig
from train_autoencoder import make_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=64)
    parser.add_argument("--probe_steps", type=int, default=500)
    parser.add_argument("--probe_width", type=int, default=128)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[CausalFrequencyAutoencoder, FrequencyCodec, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AutoencoderConfig(**payload["config"])
    if config.mode not in ("causal_k", "causal_ring"):
        raise ValueError("This fitter currently expects a custom frequency codec")
    codec_payload = payload["codec"]
    codec = FrequencyCodec(FrequencyCodecConfig(**codec_payload["config"]))
    codec.load_exported(codec_payload)
    metadata = codec.position_metadata()
    metadata["empirical_scale"] = codec.orbit_scale_for_policy(
        codec.effective_scale_policy()
    ).mean(dim=-1)
    model = CausalFrequencyAutoencoder(config, metadata, codec.component_mask)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = {
        key
        for key in incompatible.missing_keys
        if key.endswith("token_latent")
    }
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible checkpoint: "
            f"missing={sorted(disallowed_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return model.to(device).eval(), codec.to(device), payload


@torch.no_grad()
def _collect_latents(
    model: CausalFrequencyAutoencoder,
    codec: FrequencyCodec,
    loader: DataLoader,
    num_batches: int,
    device: torch.device,
    sample_posterior: bool = False,
) -> torch.Tensor:
    values = []
    for index, batch in enumerate(loader):
        if index >= num_batches:
            break
        images = batch[0].to(device)
        tokens = codec.encode(images)
        values.append(
            model.export_latents(
                tokens, sample_posterior=sample_posterior
            )["latents"].float().cpu()
        )
    if not values:
        raise RuntimeError("No latent batches were collected")
    return torch.cat(values)


def _fit_probe(
    train: torch.Tensor,
    validation: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, LatentCausalProbe, float, float, float]:
    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-6)
    normalized_train = (train - mean) / std
    normalized_validation = (validation - mean) / std
    probe = LatentCausalProbe(
        latent_dim=train.shape[-1], width=args.probe_width
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)
    generator = torch.Generator(device=device).manual_seed(0)
    for _ in range(args.probe_steps):
        indices = torch.randint(
            normalized_train.shape[0],
            (min(args.batch_size, normalized_train.shape[0]),),
            generator=generator,
            device=device,
        )
        loss = probe.loss(normalized_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probe_mse = probe.loss(normalized_validation)
        zero_mse = normalized_validation.square().mean()
        improvement = 1.0 - probe_mse / zero_mse.clamp_min(1e-8)
    return (
        mean,
        std,
        probe,
        float(probe_mse.item()),
        float(zero_mse.item()),
        float(improvement.item()),
    )


def _ring_summaries(
    latents: torch.Tensor,
    latent_parent: torch.Tensor,
    num_parents: int,
) -> torch.Tensor:
    summaries = latents.new_zeros(latents.shape[0], num_parents, latents.shape[-1])
    summaries.index_add_(1, latent_parent, latents)
    counts = torch.bincount(latent_parent, minlength=num_parents).to(latents.dtype)
    return summaries / counts[None, :, None].clamp_min(1.0)


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, codec, payload = _load_model(args.checkpoint, device)
    resolution = codec.config.height
    dataset = make_dataset(
        SimpleNamespace(
            dataset=args.dataset,
            data_root=args.data_root,
            resolution=resolution,
            smoke=False,
            seed=0,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    torch.manual_seed(0)
    latents = _collect_latents(
        model,
        codec,
        loader,
        args.num_batches,
        device,
        sample_posterior=args.sample_posterior,
    )
    split = max(int(latents.shape[0] * 0.9), 1)
    train = latents[:split].to(device)
    validation = latents[split:].to(device)
    if validation.numel() == 0:
        validation = train[-1:]
    mean, std, probe, probe_mse, zero_mse, improvement = _fit_probe(
        train, validation, args, device
    )

    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "latent_interface.pt"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "version": 1,
        "checkpoint": os.path.abspath(args.checkpoint),
        "global_step": payload["global_step"],
        "config": payload["config"],
        "latent_mean": mean.cpu(),
        "latent_std": std.cpu(),
        "sample_posterior": bool(args.sample_posterior),
        "probe": probe.state_dict(),
        "probe_width": args.probe_width,
        "probe_validation_mse": probe_mse,
        "zero_baseline_mse": zero_mse,
        "probe_fractional_improvement": improvement,
    }
    if model.config.mode == "causal_ring":
        latent_parent = model.layout.latent_parent.to(device)
        ring_train = _ring_summaries(
            train, latent_parent, model.layout.num_parents
        )
        ring_validation = _ring_summaries(
            validation, latent_parent, model.layout.num_parents
        )
        (
            ring_mean,
            ring_std,
            ring_probe,
            ring_probe_mse,
            ring_zero_mse,
            ring_improvement,
        ) = _fit_probe(ring_train, ring_validation, args, device)
        result.update(
            {
                "ring_latent_mean": ring_mean.cpu(),
                "ring_latent_std": ring_std.cpu(),
                "ring_probe": ring_probe.state_dict(),
                "ring_probe_validation_mse": ring_probe_mse,
                "ring_zero_baseline_mse": ring_zero_mse,
                "ring_probe_fractional_improvement": ring_improvement,
            }
        )
    torch.save(result, output_path)
    print(
        json.dumps(
            {
                "output": output_path,
                "probe_validation_mse": result["probe_validation_mse"],
                "zero_baseline_mse": result["zero_baseline_mse"],
                "probe_fractional_improvement": result[
                    "probe_fractional_improvement"
                ],
                "ring_probe_fractional_improvement": result.get(
                    "ring_probe_fractional_improvement"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
