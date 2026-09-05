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


def invert_latent_transform(
    tokens: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    """Invert an optional prior-space transform before representation decoding.

    This is separate from ``latent_layout`` and ``token_scale``. Callers first
    restore the physical token layout, then undo any per-token scale, and only
    then invoke this transform.
    """

    transform = payload.get("latent_transform")
    if transform is None:
        return tokens
    if not isinstance(transform, Mapping):
        raise ValueError("latent_transform must be a mapping")
    kind = transform.get("type")
    if kind == "token_permutation_inverse":
        if tokens.ndim != 3:
            raise ValueError("permuted tokens must have shape [B,L,D]")
        permutation = transform.get("permutation")
        if torch.is_tensor(permutation):
            permutation = permutation.tolist()
        if not isinstance(permutation, (list, tuple)):
            raise ValueError("token permutation transform requires permutation")
        permutation = [int(value) for value in permutation]
        if sorted(permutation) != list(range(tokens.shape[1])):
            raise ValueError("token permutation is not a complete sequence permutation")
        physical = torch.empty_like(tokens)
        indices = torch.tensor(permutation, device=tokens.device, dtype=torch.long)
        physical[:, indices] = tokens
        return physical
    if kind not in {"pca_inverse", "linear_inverse"}:
        raise ValueError(f"unsupported latent transform: {kind}")
    if tokens.ndim != 3:
        raise ValueError("linear coefficients must have shape [B,L,D]")
    mean = transform.get("mean")
    basis = transform.get("basis")
    physical_shape = transform.get("physical_shape")
    if not torch.is_tensor(mean) or not torch.is_tensor(basis):
        raise ValueError("linear transform requires tensor mean and basis")
    if not isinstance(physical_shape, (list, tuple)) or len(physical_shape) != 2:
        raise ValueError("linear transform requires a two-dimensional physical_shape")
    rank = tokens.shape[1] * tokens.shape[2]
    physical_count = int(physical_shape[0]) * int(physical_shape[1])
    if tuple(basis.shape) != (physical_count, rank) or mean.numel() != physical_count:
        raise ValueError("linear transform tensors do not match prior/physical shapes")
    coefficients = tokens.float().flatten(1)
    physical = coefficients @ basis.to(tokens.device, dtype=torch.float32).T
    physical = physical + mean.to(tokens.device, dtype=torch.float32)
    return physical.reshape(
        tokens.shape[0], int(physical_shape[0]), int(physical_shape[1])
    )


def latent_transform_fingerprint(payload: Mapping[str, Any]) -> Optional[dict]:
    """Return JSON-safe transform metadata without serializing tensor values."""

    transform = payload.get("latent_transform")
    if transform is None:
        return None
    if not isinstance(transform, Mapping):
        raise ValueError("latent_transform must be a mapping")
    if transform.get("type") == "token_permutation_inverse":
        permutation = transform.get("permutation")
        if torch.is_tensor(permutation):
            permutation = permutation.tolist()
        if not isinstance(permutation, (list, tuple)):
            raise ValueError("token permutation transform requires permutation")
        return {
            "type": "token_permutation_inverse",
            "permutation": [int(value) for value in permutation],
            "source": transform.get("source"),
            "ordering": transform.get("ordering"),
        }
    if transform.get("type") not in {"pca_inverse", "linear_inverse"}:
        raise ValueError("unsupported latent transform")
    basis = transform.get("basis")
    if not torch.is_tensor(basis):
        raise ValueError("linear transform is missing its basis")
    return {
        "type": str(transform["type"]),
        "physical_shape": [int(value) for value in transform["physical_shape"]],
        "rank": int(basis.shape[1]),
        "source": transform.get("source"),
    }
