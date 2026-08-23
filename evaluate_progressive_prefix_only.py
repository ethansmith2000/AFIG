#!/usr/bin/env python3
"""Evaluate reconstructions made from exact latent prefixes without an AR suffix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from live_evaluation import InceptionFeatures, StreamingMoments, _fid, _kid
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix_lengths", type=int, nargs="+", required=True)
    parser.add_argument(
        "--reference_cache", default="/workspace/AFIG/data/cifar10_test_inception.pt"
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
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
    cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
    reject_rescaled_cache(cache, args.latent_cache)
    latents = cache["test_latents"][: args.num_samples].float()
    tokenizer, _ = load_tokenizer_checkpoint(cache["tokenizer_checkpoint"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    sequence_length = tokenizer.config.num_latents
    for prefix in args.prefix_lengths:
        if not 1 <= prefix <= sequence_length:
            raise ValueError("prefix lengths must be within the tokenizer sequence")
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)
    results = {}
    for prefix in args.prefix_lengths:
        moments = StreamingMoments(2048)
        feature_batches = []
        for start in range(0, args.num_samples, args.batch_size):
            values = latents[start : start + args.batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                decoded = tokenizer.decode(values, prefix_lengths=prefix).float()
            features = extractor(decoded.add(1).div(2))
            moments.update(features)
            feature_batches.append(features.cpu())
        generated_mean, generated_covariance = moments.compute()
        metrics = {
            "prefix_length": prefix,
            "num_samples": args.num_samples,
            "fid": _fid(
                reference["feature_mean"],
                reference["feature_covariance"],
                generated_mean,
                generated_covariance,
            ),
            "kid": _kid(reference["kid_features"], torch.cat(feature_batches)),
        }
        results[str(prefix)] = metrics
        print(json.dumps(metrics, sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
