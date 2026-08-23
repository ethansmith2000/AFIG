#!/usr/bin/env python3
"""Impose a per-register magnitude profile on a cached latent sequence.

The tokenizer's nested objective makes registers *semantically* ordered, but on
the v5 vae cache register 0 carries only 1.59x the energy of register 63, and
under rectified flow every register crosses the noise floor within 10.4% of the
schedule -- they resolve simultaneously. Natural images have a ~10^3 low/high
frequency energy ratio, spreading crossings over ~94% of the schedule.

Scaling register i by a_i imposes the missing spectrum. Because the forward
noise is isotropic, this is exactly a per-register noise schedule: it changes
resolution order, the effective per-register loss weight, and how discretisation
error is allocated. It is a no-op only for an infinite-capacity, infinite-step
model, so it separates "ordering helps" from "ordering is a reparameterisation"
at zero autoencoder cost.

The profile is stored as `token_scale` in the output cache so the evaluator can
divide it out before decoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def build_profile(
    sequence_length: int, kind: str, alpha: float
) -> torch.Tensor:
    index = torch.arange(sequence_length, dtype=torch.float64)
    if kind == "power":
        scale = (index + 1.0).pow(-alpha)
    elif kind == "exponential":
        scale = torch.exp(-alpha * index / max(sequence_length - 1, 1))
    elif kind == "flat":
        scale = torch.ones(sequence_length, dtype=torch.float64)
    else:
        raise ValueError(f"unknown profile {kind}")
    # preserve total energy so the prior sees the same overall scale and only
    # the *relative* register magnitudes change
    scale = scale / scale.square().mean().sqrt()
    return scale.float()


def crossing_times(latents: torch.Tensor) -> torch.Tensor:
    energy = latents.square().mean(dim=(0, 2)).double()
    lam = energy / energy.mean()
    return 1.0 / (1.0 + lam.sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile", choices=["power", "exponential", "flat"], default="power"
    )
    parser.add_argument("--alpha", type=float, default=0.83)
    args = parser.parse_args()

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    if payload.get("token_scale") is not None:
        raise ValueError("cache already carries a token_scale; rescale the original")
    train = payload["train_latents"]
    sequence_length = train.shape[1]
    scale = build_profile(sequence_length, args.profile, args.alpha)

    before = crossing_times(train.float())
    energy_before = train.float().square().mean(dim=(0, 2))
    payload["train_latents"] = (train.float() * scale[None, :, None]).to(train.dtype)
    payload["test_latents"] = (
        payload["test_latents"].float() * scale[None, :, None]
    ).to(payload["test_latents"].dtype)
    after = crossing_times(payload["train_latents"].float())
    energy_after = payload["train_latents"].float().square().mean(dim=(0, 2))

    payload["token_scale"] = scale
    payload["token_scale_config"] = {"profile": args.profile, "alpha": args.alpha}
    # the cached statistics described the unscaled tensor; the trainer reads
    # global_mean/global_std from here to standardise, so recompute rather than
    # drop them
    scaled = payload["train_latents"].float()
    statistics = dict(payload["statistics"])
    statistics["global_mean"] = scaled.mean()
    statistics["global_std"] = scaled.std()
    statistics["global_min"] = scaled.min()
    statistics["global_max"] = scaled.max()
    statistics["coordinate_mean"] = scaled.mean(dim=(0, 1))
    statistics["coordinate_std"] = scaled.std(dim=(0, 1))
    statistics["slot_mean"] = scaled.mean(dim=(0, 2))
    statistics["slot_std"] = scaled.std(dim=(0, 2))
    payload["statistics"] = statistics
    torch.save(payload, args.output)

    print(
        json.dumps(
            {
                "output": args.output,
                "profile": args.profile,
                "alpha": args.alpha,
                "scale_first": float(scale[0]),
                "scale_last": float(scale[-1]),
                # the profile multiplier is NOT the resulting data ratio: it
                # multiplies the ratio the cache already had (1.59 on the v5
                # vae cache). Report both -- the earlier runs were labelled
                # 8x/64x/996x when the data ratios were 12.8/102/1588.
                "profile_multiplier": float((scale[0] / scale[-1]).square()),
                "energy_ratio_before": float(energy_before[0] / energy_before[-1]),
                "energy_ratio_after": float(energy_after[0] / energy_after[-1]),
                "crossing_spread_before": float(before.max() - before.min()),
                "crossing_spread_after": float(after.max() - after.min()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
