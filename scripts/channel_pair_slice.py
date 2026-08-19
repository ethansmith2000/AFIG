#!/usr/bin/env python3
"""Measure the channel-axis fourth-moment pair slice E[z_i^2 z_j^2].

The flattened fourth-moment tensor is [N,N,N,N] and intractable at N=1024, but
only two slices carry the structure we care about: the diagonal E[z_i^4]
(kurtosis, already targeted by the energycv arm) and the pair slice
E[z_i^2 z_j^2] -- the energy covariance. For a Gaussian with covariance L the
pair slice is exactly L_ii L_jj + 2 L_ij^2, so the excess over that prediction
isolates genuinely higher-order channel structure.

Restricted to the within-token channel axis (16 channels -> a 16x16 slice),
which the v5 scorecard identifies as the axis no second-order intervention
moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=20000)
    return parser.parse_args()


def eigen_kurtosis(values: torch.Tensor) -> dict:
    """Kurtosis in the channel eigenbasis.

    A Gaussian stays Gaussian under every rotation, so if the raw coordinates
    look Gaussian but the eigen-rotated ones do not, the joint carries
    higher-order structure that the raw basis happens to mix away.
    """

    values = values.double()
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / values.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    rotated = values @ eigenvectors[:, order]
    variance = rotated.square().mean(dim=0).clamp_min(1e-12)
    kurtosis = rotated.square().square().mean(dim=0) / variance.square()
    return {
        "eigen_kurtosis_by_rank": [round(float(k), 4) for k in kurtosis],
        "eigen_kurtosis_mean": float(kurtosis.mean()),
        "eigen_kurtosis_max": float(kurtosis.max()),
        "eigenvalue_ratio_top_to_bottom": float(
            eigenvalues[order][0] / eigenvalues[order][-1].clamp_min(1e-12)
        ),
    }


def pair_slice_excess(values: torch.Tensor) -> dict:
    """values: [samples, channels]. Centered internally."""

    values = values.double()
    values = values - values.mean(dim=0, keepdim=True)
    samples = values.shape[0]
    covariance = values.T @ values / samples
    energy = values.square()
    observed = energy.T @ energy / samples
    diagonal = covariance.diagonal()
    gaussian = diagonal[:, None] * diagonal[None, :] + 2.0 * covariance.square()
    # Normalize by the Gaussian prediction so the ratio is scale free -- the
    # whole point is that per-channel rescaling is gauge and must not show up.
    ratio = observed / gaussian.clamp_min(1e-12)
    off = ~torch.eye(values.shape[1], dtype=torch.bool)
    return {
        "kurtosis_mean": float(ratio.diagonal().mean() * 3.0),
        "pair_ratio_diagonal_mean": float(ratio.diagonal().mean()),
        "pair_ratio_offdiagonal_mean": float(ratio[off].mean()),
        "pair_ratio_offdiagonal_max": float(ratio[off].max()),
        "pair_ratio_offdiagonal_std": float(ratio[off].std()),
        "offdiagonal_excess_rms": float((ratio[off] - 1.0).square().mean().sqrt()),
    }


def main() -> None:
    args = parse_args()
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    latents = payload["train_latents"] if isinstance(payload, dict) else payload
    if latents.dim() != 3:
        raise ValueError(f"expected [samples, tokens, channels], got {tuple(latents.shape)}")
    latents = latents[: args.max_samples].float()
    samples, tokens, channels = latents.shape

    flat = latents.reshape(-1, channels)
    pooled = pair_slice_excess(flat)
    pooled_eigen = eigen_kurtosis(flat)
    per_token = [pair_slice_excess(latents[:, index]) for index in range(tokens)]
    per_token_eigen = [eigen_kurtosis(latents[:, index]) for index in range(tokens)]

    def collect(key: str) -> list:
        return [round(entry[key], 4) for entry in per_token]

    result = {
        "cache": args.cache,
        "samples": samples,
        "tokens": tokens,
        "channels": channels,
        "pooled_channel_axis": pooled,
        "pooled_channel_eigenbasis": pooled_eigen,
        "per_token_eigen_kurtosis_mean": [
            round(entry["eigen_kurtosis_mean"], 4) for entry in per_token_eigen
        ],
        "per_token_eigen_kurtosis_max_mean": sum(
            entry["eigen_kurtosis_max"] for entry in per_token_eigen
        )
        / tokens,
        "per_token_offdiagonal_excess_rms": collect("offdiagonal_excess_rms"),
        "per_token_kurtosis_mean": collect("kurtosis_mean"),
        "per_token_summary": {
            "offdiagonal_excess_rms_mean": sum(
                entry["offdiagonal_excess_rms"] for entry in per_token
            )
            / tokens,
            "kurtosis_mean_mean": sum(entry["kurtosis_mean"] for entry in per_token)
            / tokens,
            "eigen_kurtosis_mean_mean": sum(
                entry["eigen_kurtosis_mean"] for entry in per_token_eigen
            )
            / tokens,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        k: v
        for k, v in result.items()
        if not k.startswith("per_token_o")
        and not k.startswith("per_token_k")
        and not k.startswith("per_token_e")
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
