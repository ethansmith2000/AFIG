"""Reversible data representations consumed by the joint-flow harness."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


TOKENIZER_LATENTS = "tokenizer_latents"
PIXEL_PATCHES = "cifar10_pixel_patches"


def representation_type(payload: Mapping[str, Any]) -> str:
    """Return the explicit representation type, preserving legacy caches."""

    return str(payload.get("representation_type", TOKENIZER_LATENTS))


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pack non-overlapping raster patches as ``[B, L, C*p*p]`` tokens."""

    if images.ndim != 4:
        raise ValueError("images must have shape [B,C,H,W]")
    batch, channels, height, width = images.shape
    if patch_size <= 0 or height != width or height % patch_size:
        raise ValueError("images must be square and divisible by patch_size")
    grid = height // patch_size
    patches = images.reshape(
        batch, channels, grid, patch_size, grid, patch_size
    )
    return patches.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, grid * grid, channels * patch_size * patch_size
    )


def unpatchify(tokens: torch.Tensor, config: Mapping[str, Any]) -> torch.Tensor:
    """Invert :func:`patchify` from a serialized representation config."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,L,D]")
    image_size = int(config["image_size"])
    patch_size = int(config["patch_size"])
    channels = int(config.get("in_channels", 3))
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
        raise ValueError("invalid pixel-patch representation config")
    grid = image_size // patch_size
    expected = (grid * grid, channels * patch_size * patch_size)
    if tuple(tokens.shape[1:]) != expected:
        raise ValueError(
            f"pixel tokens must have shape [B,{expected[0]},{expected[1]}]"
        )
    images = tokens.reshape(
        tokens.shape[0], grid, grid, channels, patch_size, patch_size
    )
    return images.permute(0, 3, 1, 4, 2, 5).reshape(
        tokens.shape[0], channels, image_size, image_size
    )


def decode_representation(
    tokens: torch.Tensor,
    payload: Mapping[str, Any],
    *,
    tokenizer: Optional[Any] = None,
) -> torch.Tensor:
    """Decode raw flow outputs to normalized ``[-1,1]`` image tensors."""

    kind = representation_type(payload)
    if kind == TOKENIZER_LATENTS:
        if tokenizer is None:
            raise ValueError("tokenizer_latents require a tokenizer decoder")
        return tokenizer.decode(tokens)
    if kind == PIXEL_PATCHES:
        return unpatchify(tokens, payload["representation_config"])
    raise ValueError(f"unsupported representation type: {kind}")
