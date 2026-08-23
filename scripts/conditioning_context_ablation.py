#!/usr/bin/env python3
"""Measure context actually used by a trained joint rectified-flow prior.

For a fixed noisy input and late-coordinate velocity target, compare held-out
MSE when early context belongs to the same example versus when it is shuffled
between examples.  Shuffling preserves the context marginal while destroying
sample-specific dependence, so the paired MSE increase measures context used by
this trained denoiser.  It is deliberately named a model-context ablation, not
an intrinsic or causal "conditioning gain" of the representation.

Two notions of early/late are reported:

* population covariance eigen-directions selected by their SNR at each time;
* literal token prefixes, which tests alignment with the progressive axis.

The covariance basis is fit on training data and all MSEs are evaluated on the
disjoint test cache.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow


def parse_csv(values: str, cast) -> list:
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pca_samples", type=int, default=20000)
    parser.add_argument("--eval_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--times", default="0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--early_snr", type=float, default=4.0)
    parser.add_argument("--late_snr", type=float, default=0.25)
    parser.add_argument("--max_band_dims", type=int, default=256)
    parser.add_argument("--token_prefixes", default="8,16,32")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_eigenbasis(
    values: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = values[:max_samples].float().reshape(min(max_samples, len(values)), -1)
    flat = (flat - mean.cpu()) / scale.cpu()
    coordinate_mean = flat.mean(dim=0)
    centered = flat - coordinate_mean
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    return (
        coordinate_mean,
        eigenvalues[order].clamp_min(0),
        eigenvectors[:, order],
    )


def capped_indices(indices: torch.Tensor, maximum: int) -> torch.Tensor:
    if maximum <= 0 or indices.numel() <= maximum:
        return indices
    return indices[:maximum]


def shuffled_eigen_context(
    noisy: torch.Tensor,
    time: float,
    coordinate_mean: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-shuffled and mean-ablated versions of selected PCs."""

    flat = noisy.flatten(1)
    centered = flat - time * coordinate_mean
    coefficients = centered @ basis
    permutation = torch.roll(torch.arange(flat.shape[0], device=flat.device), 1)
    shuffled = centered + (coefficients[permutation] - coefficients) @ basis.T
    ablated = centered - coefficients @ basis.T
    offset = time * coordinate_mean
    return (
        (shuffled + offset).reshape_as(noisy),
        (ablated + offset).reshape_as(noisy),
    )


def shuffled_token_context(
    noisy: torch.Tensor,
    prefix: int,
    expected_noisy_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.roll(torch.arange(noisy.shape[0], device=noisy.device), 1)
    shuffled = noisy.clone()
    shuffled[:, :prefix] = noisy[permutation, :prefix]
    ablated = noisy.clone()
    ablated[:, :prefix] = expected_noisy_mean[:, :prefix]
    return shuffled, ablated


def accumulate(store: dict, name: str, errors: torch.Tensor) -> None:
    entry = store.setdefault(name, {"sse": 0.0, "count": 0})
    entry["sse"] += float(errors.double().square().sum())
    entry["count"] += errors.numel()


def finalize(store: dict) -> dict:
    mse = {key: value["sse"] / max(value["count"], 1) for key, value in store.items()}
    true = mse["true"]
    return {
        "mse_true": true,
        "mse_shuffled": mse["shuffled"],
        "mse_ablated": mse["ablated"],
        "relative_gain_vs_shuffled": (mse["shuffled"] - true) / max(mse["shuffled"], 1e-30),
        "relative_gain_vs_ablated": (mse["ablated"] - true) / max(mse["ablated"], 1e-30),
        "scalar_targets": store["true"]["count"],
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.early_snr <= args.late_snr or args.late_snr < 0:
        raise ValueError("require early_snr > late_snr >= 0")
    if args.eval_samples < 2 or args.pca_samples < 2:
        raise ValueError("pca_samples and eval_samples must be at least two")
    times = parse_csv(args.times, float)
    prefixes = parse_csv(args.token_prefixes, int)
    if not times or any(time <= 0 or time >= 1 for time in times):
        raise ValueError("times must lie strictly between zero and one")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "progressive_joint_rectified_flow":
        raise ValueError("context ablation requires a joint-flow checkpoint")
    config_values = dict(checkpoint["model_config"])
    config_values.setdefault("qk_norm", "l2_temperature")
    model = JointRectifiedFlow(JointFlowConfig(**config_values))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval().requires_grad_(False)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train = cache["train_latents"]
    test = cache["test_latents"][: args.eval_samples]
    expected = (model.config.sequence_length, model.config.token_dim)
    if tuple(train.shape[1:]) != expected or tuple(test.shape[1:]) != expected:
        raise ValueError("cache layout does not match the prior checkpoint")
    mean = checkpoint["normalization"]["mean"].float()
    scale = checkpoint["normalization"]["scale"].float()
    coordinate_mean, eigenvalues, eigenvectors = fit_eigenbasis(
        train, mean, scale, args.pca_samples
    )
    coordinate_mean = coordinate_mean.to(device)
    eigenvectors = eigenvectors.to(device)
    eigenvalues = eigenvalues.to(device)

    loader = DataLoader(
        TensorDataset(test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "pca_samples": min(args.pca_samples, len(train)),
        "eval_samples": len(test),
        "early_snr": args.early_snr,
        "late_snr": args.late_snr,
        "max_band_dims": args.max_band_dims,
        "interpretation": (
            "Held-out context used by this trained denoiser; not an intrinsic "
            "conditioning-gain estimate of the representation."
        ),
        "eigen_context": {},
        "token_prefix_context": {},
    }

    for time in times:
        snr = (time / (1.0 - time)) ** 2 * eigenvalues
        early_candidates = torch.where(snr >= args.early_snr)[0]
        late_candidates = torch.where(snr <= args.late_snr)[0]
        early = capped_indices(early_candidates, args.max_band_dims)
        # Start at the unresolved directions nearest their crossing, rather
        # than letting a huge near-null tail dominate the target average.
        late = capped_indices(late_candidates, args.max_band_dims)
        time_key = f"{time:g}"
        if early.numel() == 0 or late.numel() == 0:
            result["eigen_context"][time_key] = {
                "skipped": True,
                "early_candidates": int(early_candidates.numel()),
                "late_candidates": int(late_candidates.numel()),
            }
            continue
        early_basis = eigenvectors[:, early]
        late_basis = eigenvectors[:, late]
        eigen_store: dict = {}
        token_stores = {prefix: {} for prefix in prefixes if 0 < prefix < expected[0]}
        noise_generator = torch.Generator(device=device).manual_seed(
            args.seed + round(10_000 * time)
        )
        for (raw_clean,) in loader:
            clean = (raw_clean.to(device).float() - mean.to(device)) / scale.to(device)
            noise = torch.randn(
                clean.shape,
                device=device,
                dtype=clean.dtype,
                generator=noise_generator,
            )
            noisy = (1.0 - time) * noise + time * clean
            target = clean - noise
            time_values = torch.full((clean.shape[0],), time, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                true_prediction = model.predict_velocity(noisy, time_values)
            true_error = (true_prediction.float() - target).flatten(1) @ late_basis
            accumulate(eigen_store, "true", true_error)

            shuffled, ablated = shuffled_eigen_context(
                noisy, time, coordinate_mean, early_basis
            )
            for name, changed in (("shuffled", shuffled), ("ablated", ablated)):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model.predict_velocity(changed, time_values)
                error = (prediction.float() - target).flatten(1) @ late_basis
                accumulate(eigen_store, name, error)

            for prefix, store in token_stores.items():
                accumulate(store, "true", (true_prediction.float() - target)[:, prefix:])
                expected_noisy_mean = time * coordinate_mean.reshape(1, *expected)
                token_shuffled, token_ablated = shuffled_token_context(
                    noisy, prefix, expected_noisy_mean
                )
                for name, changed in (("shuffled", token_shuffled), ("ablated", token_ablated)):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        prediction = model.predict_velocity(changed, time_values)
                    accumulate(store, name, (prediction.float() - target)[:, prefix:])

        result["eigen_context"][time_key] = {
            "early_candidates": int(early_candidates.numel()),
            "late_candidates": int(late_candidates.numel()),
            "early_dimensions_used": int(early.numel()),
            "late_dimensions_scored": int(late.numel()),
            "early_index_range": [int(early[0]), int(early[-1])],
            "late_index_range": [int(late[0]), int(late[-1])],
            **finalize(eigen_store),
        }
        for prefix, store in token_stores.items():
            key = f"t={time:g},prefix={prefix}"
            result["token_prefix_context"][key] = {
                "prefix": prefix,
                "tail_tokens_scored": expected[0] - prefix,
                **finalize(store),
            }
        print(json.dumps({"completed_time": time}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
