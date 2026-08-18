#!/usr/bin/env python3
"""Axis-A schedule-consistency scorecard for a progressive latent cache.

Statistics in the flattened population eigenbasis, banded by eigenvalue rank:
per-sample band-energy ordering consistency, per-direction activity CV, and
cross-band log-energy correlation — each compared against a Gaussian surrogate
with the same eigenvalue spectrum (the "image-like" reference: a Gaussian is
perfectly schedule-consistent up to chi-square sampling noise).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def band_stats(projections: torch.Tensor, bands: list[tuple[int, int]]) -> dict:
    energies = torch.stack(
        [projections[:, lo:hi].square().mean(dim=1) for lo, hi in bands], dim=1
    )
    logp = energies.clamp_min(1e-12).log10()
    order = [
        float((energies[:, k] > energies[:, k + 1]).float().mean())
        for k in range(len(bands) - 1)
    ]
    corr = torch.corrcoef(logp.T)
    per_direction_energy = projections.square()
    activity_cv2 = per_direction_energy.var(dim=0) / (
        per_direction_energy.mean(dim=0).square() + 1e-12
    )
    band_cv = [float(activity_cv2[lo:hi].median()) for lo, hi in bands]
    return {
        "adjacent_order_consistency": order,
        "band_log_energy_corr": [[round(float(v), 4) for v in row] for row in corr],
        "activity_cv2_median_by_band": band_cv,
        "activity_cv2_overall_median": float(activity_cv2.median()),
        "activity_cv2_frac_above_4": float((activity_cv2 > 4.0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_examples", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    latents = payload["train_latents"].float()
    flat = latents.reshape(latents.shape[0], -1)
    generator = torch.Generator().manual_seed(args.seed)
    if flat.shape[0] > args.max_examples:
        index = torch.randperm(flat.shape[0], generator=generator)[: args.max_examples]
        flat = flat[index]
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = (centered.T @ centered).double() / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # descending eigenvalue order
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)
    projections = centered.double() @ eigenvectors

    dim = flat.shape[1]
    edges = [0, 8, 32, 128, 512, dim]
    bands = list(zip(edges[:-1], edges[1:]))

    surrogate = torch.randn(
        flat.shape[0], dim, generator=generator, dtype=torch.float64
    ) * eigenvalues.sqrt()[None, :]

    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)

    # Per-tensor-axis consistency. The full-covariance surrogate reproduces all
    # second-order structure, so any consistency difference against it is
    # higher-order (the quantity of interest).
    tokens, channels = latents.shape[1], latents.shape[2]
    surrogate_original = (surrogate * 1.0) @ eigenvectors.T

    def axis_stats(values_flat: torch.Tensor) -> dict:
        values = values_flat.reshape(-1, tokens, channels)
        token_energy = values.square().mean(dim=2)
        token_cv2 = token_energy.var(dim=0) / (
            token_energy.mean(dim=0).square() + 1e-12
        )
        log_energy = token_energy.clamp_min(1e-12).log10()
        profile = log_energy.mean(dim=0)
        centered_log = log_energy - profile
        profile_correlation = torch.nn.functional.cosine_similarity(
            log_energy - log_energy.mean(dim=1, keepdim=True),
            (profile - profile.mean()).expand_as(log_energy),
            dim=1,
        )
        adjacent = [
            float(torch.corrcoef(torch.stack((log_energy[:, i], log_energy[:, i + 1])))[0, 1])
            for i in range(tokens - 1)
        ]
        # channel axis: pooled 16-D feature eigenbasis over (sample, token) rows
        rows = values.reshape(-1, channels).double()
        rows = rows - rows.mean(dim=0)
        feature_cov = rows.T @ rows / rows.shape[0]
        feature_values, feature_vectors = torch.linalg.eigh(feature_cov)
        feature_values = feature_values.flip(0).clamp_min(0)
        feature_vectors = feature_vectors.flip(1)
        feature_proj = rows @ feature_vectors
        feature_energy = feature_proj.square()
        feature_cv2 = feature_energy.var(dim=0) / (
            feature_energy.mean(dim=0).square() + 1e-12
        )
        channel_order = [
            float((feature_energy[:, k] > feature_energy[:, k + 1]).float().mean())
            for k in range(channels - 1)
        ]
        return {
            "token_energy_cv2_median": float(token_cv2.median()),
            "token_energy_cv2_max": float(token_cv2.max()),
            "token_profile_correlation_mean": float(profile_correlation.mean()),
            "token_adjacent_log_energy_corr_mean": sum(adjacent) / len(adjacent),
            "channel_eigen_order_consistency_mean": sum(channel_order)
            / len(channel_order),
            "channel_activity_cv2_median": float(feature_cv2.median()),
        }

    result = {
        "cache": str(Path(args.cache).resolve()),
        "examples": int(flat.shape[0]),
        "bands_by_rank": [list(b) for b in bands],
        "effective_rank": float(
            torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
        ),
        "eigen_share_top_8_32_128": [
            float(probabilities[:k].sum()) for k in (8, 32, 128)
        ],
        "latent": band_stats(projections, bands),
        "gaussian_surrogate": band_stats(surrogate, bands),
        "latent_axes": axis_stats(centered.double()),
        "gaussian_surrogate_axes": axis_stats(surrogate_original),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
