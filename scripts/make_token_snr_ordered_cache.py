#!/usr/bin/env python3
"""Reorder latent slots by content RMS while preserving exact decode semantics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from progressive_tokenizer.representations import invert_latent_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("latent_transform") is not None:
        raise ValueError("source cache already carries a latent transform")
    train = payload["train_latents"]
    test = payload["test_latents"]
    if train.ndim != 3 or test.shape[1:] != train.shape[1:]:
        raise ValueError("latent caches must have compatible [N,L,D] tensors")

    values = train.float()
    token_mean = values.mean(dim=0)
    content_variance = (values - token_mean).square().mean(dim=(0, 2))
    content_rms = content_variance.sqrt()
    permutation = torch.argsort(content_rms, descending=True)

    result = dict(payload)
    result["train_latents"] = train[:, permutation].contiguous()
    result["test_latents"] = test[:, permutation].contiguous()
    statistics = dict(payload["statistics"])
    for key in ("slot_mean", "slot_std"):
        if key in statistics:
            statistics[key] = statistics[key][permutation].contiguous()
    result["statistics"] = statistics
    result["latent_transform"] = {
        "type": "token_permutation_inverse",
        "permutation": permutation.tolist(),
        "source": str(source_path.resolve()),
        "ordering": "descending_content_rms",
    }
    result["token_ordering"] = {
        "definition": "sqrt(mean_feature Var_example[z[token,feature]]))",
        "permutation_prior_to_physical": permutation.tolist(),
        "content_rms_physical": content_rms.tolist(),
        "content_rms_prior_order": content_rms[permutation].tolist(),
    }

    restored = invert_latent_transform(result["test_latents"][:2], result)
    torch.testing.assert_close(restored, test[:2])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(result, temporary)
    os.replace(temporary, output_path)
    print(
        {
            "output": str(output_path),
            "shape": list(train.shape[1:]),
            "permutation": permutation.tolist(),
            "content_rms_range": [float(content_rms.min()), float(content_rms.max())],
        }
    )


if __name__ == "__main__":
    main()
