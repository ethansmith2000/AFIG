#!/usr/bin/env python3
"""Measure when latent tokens cross SNR=1 on the joint-flow path.

The prior trains on a tensor-wide scalar-standardized latent ``z`` and the
rectified-flow path ``z_t = (1-t) eps + t z``.  For token ``i`` we define its
content variance per feature as

    v_i = E[||z_i - E[z_i]||^2] / d.

With unit-variance isotropic noise, its aggregate token SNR and population
crossing time are

    SNR_i(t) = t^2 v_i / (1-t)^2,
    t_i* = 1 / (1 + sqrt(v_i)).

Token-specific means are excluded because they are fixed dataset parameters,
not image content.  Within-token eigenspectra are reported as a guard against
an aggregate token SNR hiding one active direction and many dead features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TIMESTEPS = (0.2, 0.35, 0.5, 0.65, 0.8)


def _crossing(variance: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + variance.clamp_min(0.0).sqrt())


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95], dtype=values.dtype
    )
    result = torch.quantile(values, probabilities)
    return {
        label: float(value)
        for label, value in zip(("p05", "p25", "p50", "p75", "p95"), result)
    }


def _prefix_blocks(length: int) -> list[tuple[int, int]]:
    edges = [0, 1]
    while edges[-1] < length:
        edges.append(min(length, edges[-1] * 2))
    return list(zip(edges[:-1], edges[1:]))


def analyze_cache(path: Path, max_examples: int) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload["train_latents"][:max_examples].float()
    if values.ndim != 3:
        raise ValueError(f"{path}: train_latents must have shape [N,L,D]")

    statistics = payload["statistics"]
    global_mean = torch.as_tensor(statistics["global_mean"]).float()
    global_scale = torch.as_tensor(statistics["global_std"]).float()
    if not bool(torch.isfinite(global_scale)) or float(global_scale) <= 0:
        raise ValueError(f"{path}: invalid global standard deviation")
    values = (values - global_mean) / global_scale

    # Remove a separate mean vector at every token.  This makes the signal
    # energy describe sample-dependent content rather than a learned constant.
    token_mean = values.mean(dim=0)
    centered = values - token_mean
    sample_token_variance = centered.square().mean(dim=2)
    token_variance = sample_token_variance.mean(dim=0)
    token_crossing = _crossing(token_variance)
    sample_token_crossing = _crossing(sample_token_variance)

    per_token = []
    for index in range(values.shape[1]):
        rows = centered[:, index].double()
        covariance = rows.T @ rows / rows.shape[0]
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        )
        per_token.append(
            {
                "index": index,
                "content_variance_per_feature": float(token_variance[index]),
                "content_rms": float(token_variance[index].sqrt()),
                "mean_vector_rms": float(token_mean[index].square().mean().sqrt()),
                "snr1_population_t": float(token_crossing[index]),
                "snr1_per_sample_t": _quantiles(sample_token_crossing[:, index]),
                "within_token_effective_rank": float(effective_rank),
                "within_token_top1_variance_share": float(probabilities[-1]),
            }
        )

    population_profile = token_variance.clamp_min(1e-12).log()
    sample_profiles = sample_token_variance.clamp_min(1e-12).log()
    population_profile = population_profile - population_profile.mean()
    sample_profiles = sample_profiles - sample_profiles.mean(dim=1, keepdim=True)
    profile_cosine = torch.nn.functional.cosine_similarity(
        sample_profiles, population_profile.expand_as(sample_profiles), dim=1
    )

    adjacent_order = [
        float(
            (
                sample_token_variance[:, index]
                > sample_token_variance[:, index + 1]
            )
            .float()
            .mean()
        )
        for index in range(values.shape[1] - 1)
    ]
    all_forward_order = []
    for left in range(values.shape[1] - 1):
        comparisons = (
            sample_token_variance[:, left, None]
            > sample_token_variance[:, left + 1 :]
        )
        all_forward_order.append(comparisons.float().mean())
    forward_order_mean = torch.stack(all_forward_order).mean()

    blocks = []
    for start, end in _prefix_blocks(values.shape[1]):
        block_variance = sample_token_variance[:, start:end].mean(dim=1)
        blocks.append(
            {
                "token_range": [start, end],
                "population_variance_per_feature": float(block_variance.mean()),
                "snr1_population_t": float(_crossing(block_variance.mean())),
                "snr1_per_sample_t": _quantiles(_crossing(block_variance)),
            }
        )

    timestep_summary = []
    for timestep in TIMESTEPS:
        snr = timestep**2 * token_variance / (1.0 - timestep) ** 2
        timestep_summary.append(
            {
                "t": timestep,
                "tokens_above_snr1": int((snr >= 1.0).sum()),
                "fraction_tokens_above_snr1": float((snr >= 1.0).float().mean()),
                "token_snr_quantiles": _quantiles(snr),
            }
        )

    return {
        "cache": str(path.resolve()),
        "examples": int(values.shape[0]),
        "shape": list(values.shape[1:]),
        "normalization": {
            "type": "tensor_wide_population",
            "mean": float(global_mean),
            "scale": float(global_scale),
        },
        "definition": {
            "path": "z_t = (1-t) eps + t z",
            "noise_variance_per_feature": 1.0,
            "signal": "sample-dependent token content after token-specific mean removal",
            "snr": "t^2 * token_content_variance / (1-t)^2",
        },
        "token_profile": {
            "content_variance_quantiles": _quantiles(token_variance),
            "snr1_population_t_quantiles": _quantiles(token_crossing),
            "sample_to_population_log_energy_cosine": _quantiles(profile_cosine),
            "adjacent_descending_probability_mean": float(
                torch.tensor(adjacent_order).mean()
            ),
            "all_forward_pairs_descending_probability_mean": float(
                forward_order_mean
            ),
            "adjacent_descending_probability": adjacent_order,
        },
        "prefix_blocks": blocks,
        "timestep_summary": timestep_summary,
        "per_token": per_token,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Latent cache path; repeat to compare representations.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples <= 0:
        raise ValueError("max_examples must be positive")
    result = {
        "analyses": [
            analyze_cache(Path(cache), args.max_examples) for cache in args.cache
        ]
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
