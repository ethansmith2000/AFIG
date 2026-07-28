#!/usr/bin/env python3
"""Evaluate an existing image autoencoder through the latent-FFT interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from autoencoder_models import ImageAutoencoderAdapter, LatentFourierNormalizer
from train_autoencoder import make_dataset, reconstruction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subfolder", default=None)
    parser.add_argument("--dataset", default="huggingface_cifar")
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--sample_posterior", action="store_true")
    parser.add_argument("--output_dir", default="autoencoder_runs/external_image_ae")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--prefix_fractions", default="0.25,0.5,0.75")
    parser.add_argument("--save_latent_stats", action="store_true")
    return parser.parse_args()


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(row[key] for row in rows if key in row)
        / sum(key in row for row in rows)
        for key in keys
    }


@torch.no_grad()
def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]
    dataset_args = SimpleNamespace(
        dataset=args.dataset,
        data_root=args.data_root,
        resolution=args.resolution,
        smoke=False,
        seed=0,
    )
    dataset = make_dataset(dataset_args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    from diffusers import AutoencoderKL

    kwargs = {}
    if args.subfolder:
        kwargs["subfolder"] = args.subfolder
    model = AutoencoderKL.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    model.to(device).eval()
    downsample = 2 ** (len(model.config.block_out_channels) - 1)
    latent_height = args.resolution // downsample
    latent_width = args.resolution // downsample
    adapter = ImageAutoencoderAdapter(
        model,
        latent_height=latent_height,
        latent_width=latent_width,
    ).to(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float]] = []
    token_batches = []
    first_preview = None
    fractions = [float(item) for item in args.prefix_fractions.split(",") if item]
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        images = batch[0].to(device=device, dtype=dtype)
        output = adapter(images, sample_posterior=args.sample_posterior)
        reconstruction = output["reconstruction"].float().clamp(0, 1)
        metrics = {
            key: float(value.item())
            for key, value in reconstruction_metrics(images, reconstruction).items()
        }
        tokens = output["latent_tokens"].float()
        for fraction in fractions:
            count = max(1, min(tokens.shape[1], int(tokens.shape[1] * fraction + 0.999)))
            prefix = tokens.clone()
            prefix[:, count:] = 0
            prefix_reconstruction = adapter.decode(
                adapter.latent_fft.decode(prefix).to(dtype)
            ).float()
            mse = (prefix_reconstruction - images.float()).square().mean()
            metrics[f"latent_prefix/pixel_mse_{fraction:.2f}"] = float(mse.item())
            metrics[f"latent_prefix/psnr_{fraction:.2f}"] = float(
                (-10.0 * torch.log10(mse.clamp_min(1e-12))).item()
            )
        metric_rows.append(metrics)
        token_batches.append(tokens.cpu())
        if first_preview is None:
            count = min(8, images.shape[0])
            first_preview = torch.cat(
                [images[:count].float(), reconstruction[:count]], dim=0
            )

    if not token_batches:
        raise RuntimeError("No image-autoencoder evaluation batches were produced")
    all_tokens = torch.cat(token_batches)
    channels = all_tokens.shape[-1] // 2
    normalizer = LatentFourierNormalizer(adapter.latent_fft, channels).cpu()
    normalizer.fit(all_tokens)
    normalized = normalizer.normalize(all_tokens)
    summary = _mean_metrics(metric_rows)
    summary.update(
        {
            "interface/input_resolution": args.resolution,
            "interface/downsample": downsample,
            "interface/latent_height": latent_height,
            "interface/latent_width": latent_width,
            "interface/latent_channels": channels,
            "interface/exported_tokens": adapter.latent_fft.seq_len,
            "interface/source_to_latent_token_ratio": (
                args.resolution * args.resolution // 2 + 2
            )
            / adapter.latent_fft.seq_len,
            "latent/normalized_mean": float(normalized.mean().item()),
            "latent/normalized_rms": float(normalized.square().mean().sqrt().item()),
        }
    )
    with open(
        os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    if first_preview is not None:
        save_image(
            first_preview.clamp(0, 1),
            os.path.join(args.output_dir, "reconstruction.png"),
            nrow=first_preview.shape[0] // 2,
        )
    if args.save_latent_stats:
        torch.save(
            {
                "model": args.model,
                "subfolder": args.subfolder,
                "resolution": args.resolution,
                "downsample": downsample,
                "normalizer": normalizer.state_dict(),
            },
            os.path.join(args.output_dir, "latent_fft_stats.pt"),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
