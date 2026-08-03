"""Representation controls for local and globally supported real DCT tokens."""

from __future__ import annotations

import torch

from control_pixel_diffusion import (
    full_dctify,
    full_idctify,
    full_hartleyify,
    full_ihartleyify,
    orthonormal_dct_matrix,
    orthonormal_hartley_matrix,
    orbit_order_permutation,
    patch_dctify,
    patch_idctify,
)
from frequency import FrequencyCodec, FrequencyCodecConfig


def test_dct_matrices_are_orthonormal() -> None:
    for size in (4, 32):
        matrix = orthonormal_dct_matrix(size)
        identity = torch.eye(size)
        assert torch.allclose(matrix @ matrix.T, identity, atol=1e-5, rtol=1e-5)


def test_hartley_matrices_are_real_and_orthonormal() -> None:
    for size in (4, 32):
        matrix = orthonormal_hartley_matrix(size)
        identity = torch.eye(size)
        assert not matrix.is_complex()
        assert torch.allclose(matrix @ matrix.T, identity, atol=2e-5, rtol=2e-5)


def test_patch_dct_roundtrip_shape_and_energy() -> None:
    images = torch.randn(3, 3, 32, 32)
    tokens = patch_dctify(images, patch=4)
    decoded = patch_idctify(tokens, patch=4, size=32)
    assert tokens.shape == (3, 64, 48)
    assert torch.allclose(decoded, images, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        tokens.square().sum(), images.square().sum(), atol=2e-3, rtol=2e-5
    )


def test_full_dct_roundtrip_shape_and_energy() -> None:
    images = torch.randn(2, 3, 32, 32)
    tokens = full_dctify(images, patch=4)
    decoded = full_idctify(tokens, patch=4, size=32)
    assert tokens.shape == (2, 64, 48)
    assert torch.allclose(decoded, images, atol=3e-5, rtol=3e-5)
    assert torch.allclose(
        tokens.square().sum(), images.square().sum(), atol=3e-3, rtol=3e-5
    )


def test_full_hartley_roundtrip_shape_and_energy() -> None:
    images = torch.randn(2, 3, 32, 32)
    tokens = full_hartleyify(images, patch=4)
    decoded = full_ihartleyify(tokens, patch=4, size=32)
    assert tokens.shape == (2, 64, 48)
    assert torch.allclose(decoded, images, atol=6e-5, rtol=6e-5)
    assert torch.allclose(
        tokens.square().sum(), images.square().sum(), atol=5e-3, rtol=5e-5
    )


def test_square_spiral_orbit_permutation_is_bijective() -> None:
    codec = FrequencyCodec(FrequencyCodecConfig(ordering="radial"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    assert permutation.shape == (codec.seq_len_int,)
    assert torch.equal(permutation.sort().values, torch.arange(codec.seq_len_int))


def test_patch_dct_preserves_linear_gaussian_bridge() -> None:
    data = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(data)
    time = torch.tensor([0.0, 0.2, 0.7, 1.0])[:, None, None, None]
    pixel_bridge = time * data + (1.0 - time) * noise
    transformed_bridge = patch_dctify(pixel_bridge, patch=4)
    expected = (
        time.flatten()[:, None, None] * patch_dctify(data, patch=4)
        + (1.0 - time.flatten()[:, None, None]) * patch_dctify(noise, patch=4)
    )
    assert torch.allclose(transformed_bridge, expected, atol=2e-5, rtol=2e-5)
