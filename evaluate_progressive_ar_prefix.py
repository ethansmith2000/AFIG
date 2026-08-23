#!/usr/bin/env python3
"""Evaluate an AR suffix rollout after replacing a prefix with real latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer import AutoregressiveFlowConfig, AutoregressiveRectifiedFlow
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix_length", type=int, required=True)
    parser.add_argument(
        "--reference_cache", default="/workspace/AFIG/data/cifar10_test_inception.pt"
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()

def reject_rescaled_cache(payload: dict, path: str) -> None:
    """Refuse a magnitude-rescaled cache rather than decode it un-inverted.

    These scripts decode cached latents directly. A cache carrying
    `token_scale` holds registers scaled by up to 5.6x / 0.18x, so decoding it
    raw yields a wrongly inflated reconstruction floor -- which would make a
    prior-vs-oracle gap look artificially small.
    """

    if isinstance(payload, dict) and payload.get("token_scale") is not None:
        raise ValueError(
            f"{path} carries token_scale (profile "
            f"{payload.get('token_scale_config')}); this script is not "
            "rescale-aware. Point it at the unscaled cache."
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reject_rescaled_cache(payload, args.latent_cache)
    config = AutoregressiveFlowConfig(**payload["model_config"])
    if not 0 <= args.prefix_length <= config.sequence_length:
        raise ValueError("prefix_length must be between zero and sequence length")
    model = AutoregressiveRectifiedFlow(config)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    mean = payload["normalization"]["mean"].float().to(device)
    scale = payload["normalization"]["scale"].float().to(device)
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    if Path(cache["tokenizer_checkpoint"]).resolve() != Path(
        payload["tokenizer_checkpoint"]
    ).resolve():
        raise ValueError("checkpoint and cache use different tokenizers")
    test = cache["test_latents"][: args.num_samples].float()
    if test.shape[0] != args.num_samples:
        raise ValueError("latent cache has fewer examples than requested")
    standardized_test = (test - mean.cpu()) / scale.cpu()
    tokenizer, _ = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)

    moments = StreamingMoments(2048)
    generated_features = []
    generated = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    while generated < args.num_samples:
        stop = min(generated + args.batch_size, args.num_samples)
        real = standardized_test[generated:stop].to(device)
        tokens = torch.zeros_like(real)
        tokens[:, : args.prefix_length] = real[:, : args.prefix_length]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for index in range(args.prefix_length, config.sequence_length):
                condition = model.trunk(tokens)[:, index]
                tokens[:, index] = model.head.sample(
                    condition, steps=args.sample_steps, generator=generator
                )
            decoded = tokenizer.decode(tokens.float() * scale + mean).float()
        features = extractor(decoded.add(1).div(2))
        moments.update(features)
        generated_features.append(features.cpu())
        generated = stop
        print(json.dumps({"generated": generated}), flush=True)

    generated_mean, generated_covariance = moments.compute()
    metrics = {
        "checkpoint_step": int(payload["step"]),
        "prefix_length": args.prefix_length,
        "generated_suffix_length": config.sequence_length - args.prefix_length,
        "num_samples": args.num_samples,
        "sample_steps": args.sample_steps,
        "fid": _fid(
            reference["feature_mean"],
            reference["feature_covariance"],
            generated_mean,
            generated_covariance,
        ),
        "kid": _kid(reference["kid_features"], torch.cat(generated_features)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
