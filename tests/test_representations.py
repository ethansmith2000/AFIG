from __future__ import annotations

import pytest
import torch

from scripts.conditioning_context_ablation import (
    shuffled_eigen_context,
    shuffled_token_context,
)
from progressive_tokenizer.representations import (
    PIXEL_PATCHES,
    TOKENIZER_LATENTS,
    decode_representation,
    invert_latent_transform,
    latent_transform_fingerprint,
    patchify,
    representation_type,
    unpatchify,
)


def pixel_payload(size: int = 8, patch: int = 4) -> dict:
    return {
        "representation_type": PIXEL_PATCHES,
        "representation_config": {
            "image_size": size,
            "patch_size": patch,
            "in_channels": 3,
        },
    }


def test_pixel_patch_roundtrip() -> None:
    images = torch.randn(3, 3, 8, 8)
    tokens = patchify(images, 4)
    assert tokens.shape == (3, 4, 48)
    torch.testing.assert_close(unpatchify(tokens, pixel_payload()["representation_config"]), images)
    torch.testing.assert_close(decode_representation(tokens, pixel_payload()), images)


def test_legacy_payload_defaults_to_tokenizer_latents() -> None:
    assert representation_type({"tokenizer_checkpoint": "old.pt"}) == TOKENIZER_LATENTS


def test_pixel_decode_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="pixel tokens"):
        decode_representation(torch.randn(2, 3, 48), pixel_payload())


def test_invert_pca_latent_transform() -> None:
    coefficients = torch.tensor([[[2.0, -1.0]]])
    payload = {
        "latent_transform": {
            "type": "pca_inverse",
            "physical_shape": [2, 2],
            "mean": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "basis": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
            ),
            "source": "basis.pt",
        }
    }
    reconstructed = invert_latent_transform(coefficients, payload)
    expected = torch.tensor([[[3.0, 1.0], [4.0, 5.5]]])
    torch.testing.assert_close(reconstructed, expected)
    assert latent_transform_fingerprint(payload) == {
        "type": "pca_inverse",
        "physical_shape": [2, 2],
        "rank": 2,
        "source": "basis.pt",
    }


def test_token_permutation_transform_roundtrip() -> None:
    physical = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    permutation = [2, 0, 3, 1]
    prior = physical[:, permutation]
    payload = {
        "latent_transform": {
            "type": "token_permutation_inverse",
            "permutation": permutation,
            "source": "unit-test",
            "ordering": "descending_content_rms",
        }
    }
    restored = invert_latent_transform(prior, payload)
    torch.testing.assert_close(restored, physical)
    assert latent_transform_fingerprint(payload) == payload["latent_transform"]


def test_eigen_context_ablation_changes_only_selected_subspace() -> None:
    noisy = torch.tensor([[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]]])
    selected_basis = torch.tensor([[1.0], [0.0]])
    shuffled, ablated = shuffled_eigen_context(
        noisy,
        time=0.5,
        coordinate_mean=torch.zeros(2),
        basis=selected_basis,
    )
    torch.testing.assert_close(shuffled[:, 0, 0], torch.tensor([3.0, 1.0, 2.0]))
    torch.testing.assert_close(shuffled[:, 0, 1], noisy[:, 0, 1])
    torch.testing.assert_close(ablated[:, 0, 0], torch.zeros(3))
    torch.testing.assert_close(ablated[:, 0, 1], noisy[:, 0, 1])


def test_token_context_ablation_uses_population_mean() -> None:
    noisy = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    expected_mean = torch.full((1, 4, 2), 7.0)
    shuffled, ablated = shuffled_token_context(noisy, 2, expected_mean)
    torch.testing.assert_close(shuffled[0, :2], noisy[2, :2])
    torch.testing.assert_close(shuffled[:, 2:], noisy[:, 2:])
    torch.testing.assert_close(ablated[:, :2], expected_mean[:, :2].expand(3, -1, -1))
    torch.testing.assert_close(ablated[:, 2:], noisy[:, 2:])
