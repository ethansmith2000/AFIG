#!/usr/bin/env python3
"""Decoder sensitivity: reconstruction FID under injected latent noise.

Decodes test-set latents perturbed by isotropic noise at several sigmas (in
tensor-wide-std units) and reports the degradation curve — the direct probe of
"brittle latent directions": high clean PSNR with a steep curve means the code
packs information where a prior cannot reliably land.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# this script lives in scripts/ but imports the project root modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from live_evaluation import InceptionFeatures, StreamingMoments, _fid  # noqa: E402
from progressive_tokenizer.checkpoints import load_tokenizer_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference_cache", default="/workspace/AFIG/data/cifar10_test_inception.pt"
    )
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--num_examples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    tokenizer, _ = load_tokenizer_checkpoint(payload["tokenizer_checkpoint"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    latents = payload["test_latents"].float()[: args.num_examples]
    scale = float(payload["statistics"]["global_std"])
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    extractor = InceptionFeatures(device)

    results = {"cache": str(Path(args.cache).resolve()), "sigmas": {}}
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for sigma in args.sigmas:
        moments = StreamingMoments(2048)
        for start in range(0, latents.shape[0], args.batch_size):
            clean = latents[start : start + args.batch_size].to(device)
            noise = torch.randn(
                clean.shape, device=device, generator=generator
            ) * (sigma * scale)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                decoded = tokenizer.decode(clean + noise).float()
            moments.update(extractor(decoded.add(1).div(2).clamp(0, 1)))
        mean, covariance = moments.compute()
        fid = _fid(
            reference["feature_mean"],
            reference["feature_covariance"],
            mean,
            covariance,
        )
        results["sigmas"][str(sigma)] = {"reconstruction_fid": fid}
        print(json.dumps({"sigma": sigma, "fid": fid}), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
