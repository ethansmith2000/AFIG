from pathlib import Path

import torch

from autoencoder_models import AutoencoderConfig, SpatialAutoencoder
from train_spatial_latent_hartley_ar import (
    block_dct_support_tokens,
    compact_isometric_fft_tokens,
    frequency_major_local_dct,
    group_radial_hartley_tiles,
    latent_maps_to_tokens,
    load_spatial_ae,
    restore_block_dct_support_tokens,
    restore_compact_isometric_fft_tokens,
    restore_local_dct_raster,
    tokens_to_latent_maps,
    ungroup_radial_hartley_tiles,
)
from train_hartley_ar import HartleyTileAR


def test_latent_hartley_roundtrip() -> None:
    maps = torch.randn(3, 8, 8, 8)
    mean = torch.randn(1, 8, 1, 1)
    std = torch.rand(1, 8, 1, 1) + 0.25
    tokens = latent_maps_to_tokens(maps, mean, std, patch=2)
    restored = tokens_to_latent_maps(tokens, mean, std, patch=2, size=8)
    assert tokens.shape == (3, 16, 32)
    torch.testing.assert_close(restored, maps, atol=5e-5, rtol=5e-5)


def test_latent_spatial_patch_roundtrip() -> None:
    maps = torch.randn(3, 4, 8, 8)
    mean = torch.randn(1, 4, 1, 1)
    std = torch.rand(1, 4, 1, 1) + 0.25
    tokens = latent_maps_to_tokens(maps, mean, std, patch=2, basis="spatial")
    restored = tokens_to_latent_maps(
        tokens, mean, std, patch=2, size=8, basis="spatial"
    )
    assert tokens.shape == (3, 16, 16)
    torch.testing.assert_close(restored, maps)


def test_latent_local_dct_roundtrip() -> None:
    maps = torch.randn(3, 4, 8, 8)
    mean = torch.randn(1, 4, 1, 1)
    std = torch.rand(1, 4, 1, 1) + 0.25
    tokens = latent_maps_to_tokens(maps, mean, std, patch=2, basis="patch_dct")
    restored = tokens_to_latent_maps(
        tokens, mean, std, patch=2, size=8, basis="patch_dct"
    )
    assert tokens.shape == (3, 16, 16)
    torch.testing.assert_close(restored, maps, atol=1e-5, rtol=1e-5)


def test_load_spatial_ae_checkpoint(tmp_path: Path) -> None:
    config = AutoencoderConfig(
        mode="spatial_downsample",
        spatial_resolution=32,
        spatial_downsample=4,
        spatial_latent_channels=8,
        spatial_base_channels=16,
    )
    model = SpatialAutoencoder(config)
    checkpoint = tmp_path / "ae.pt"
    torch.save({"config": config.fingerprint(), "model": model.state_dict()}, checkpoint)
    loaded = load_spatial_ae(str(checkpoint), torch.device("cpu"))
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(images)["reconstruction"]
        actual = loaded(images)["reconstruction"]
    torch.testing.assert_close(actual, expected)


def test_raster_order_does_not_permute_local_tokens() -> None:
    model = HartleyTileAR(
        width=32,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        diff_width=32,
        diff_depth=1,
        inference_steps=2,
        grid=4,
        token_dim=16,
        token_order="raster",
        gradient_checkpointing=False,
    )
    tokens = torch.randn(2, 16, 16)
    torch.testing.assert_close(model.order_tokens(tokens), tokens)
    torch.testing.assert_close(model.restore_raster(tokens), tokens)


def test_grouped_radial_hartley_tiles_roundtrip() -> None:
    raster = torch.randn(3, 16, 16)
    grouped = group_radial_hartley_tiles(raster, grid=4, tiles_per_token=4)
    assert grouped.shape == (3, 4, 64)
    restored = ungroup_radial_hartley_tiles(
        grouped, grid=4, tiles_per_token=4
    )
    torch.testing.assert_close(restored, raster)


def test_frequency_major_local_dct_roundtrip() -> None:
    raster = torch.randn(3, 16, 16)
    frequency_major = frequency_major_local_dct(raster, size=8, patch=2)
    assert frequency_major.shape == raster.shape
    restored = restore_local_dct_raster(frequency_major, size=8, patch=2)
    torch.testing.assert_close(restored, raster)


def test_frequency_major_local_dct_map_roundtrip() -> None:
    maps = torch.randn(3, 4, 8, 8)
    mean = torch.randn(1, 4, 1, 1)
    std = torch.rand(1, 4, 1, 1) + 0.25
    tokens = latent_maps_to_tokens(
        maps, mean, std, patch=2, basis="patch_dct_freq_major"
    )
    restored = tokens_to_latent_maps(
        tokens,
        mean,
        std,
        patch=2,
        size=8,
        basis="patch_dct_freq_major",
    )
    assert tokens.shape == (3, 16, 16)
    torch.testing.assert_close(restored, maps, atol=1e-5, rtol=1e-5)


def test_fixed_shape_block_dct_support_roundtrip_and_energy() -> None:
    maps = torch.randn(3, 4, 8, 8)
    for support in (2, 4, 8):
        tokens = block_dct_support_tokens(maps, support=support, token_dim=16)
        restored = restore_block_dct_support_tokens(
            tokens,
            size=8,
            channels=4,
            support=support,
            token_dim=16,
        )
        assert tokens.shape == (3, 16, 16)
        torch.testing.assert_close(restored, maps, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            tokens.square().sum(), maps.square().sum(), atol=2e-4, rtol=2e-5
        )


def test_full_dct_tile_roundtrip_and_energy() -> None:
    maps = torch.randn(3, 4, 8, 8)
    mean = torch.randn(1, 4, 1, 1)
    std = torch.rand(1, 4, 1, 1) + 0.25
    normalized = (maps - mean) / std
    tokens = latent_maps_to_tokens(
        maps, mean, std, patch=2, basis="full_dct_tiles"
    )
    restored = tokens_to_latent_maps(
        tokens, mean, std, patch=2, size=8, basis="full_dct_tiles"
    )
    assert tokens.shape == (3, 16, 16)
    torch.testing.assert_close(restored, maps, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        tokens.square().sum(), normalized.square().sum(), atol=2e-4, rtol=2e-5
    )


def test_compact_isometric_fft_roundtrip_and_energy() -> None:
    maps = torch.randn(3, 4, 8, 8)
    tokens = compact_isometric_fft_tokens(maps, token_dim=16)
    restored = restore_compact_isometric_fft_tokens(
        tokens, size=8, channels=4, token_dim=16
    )
    assert tokens.shape == (3, 16, 16)
    torch.testing.assert_close(restored, maps, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        tokens.square().sum(), maps.square().sum(), atol=2e-4, rtol=2e-5
    )


def test_sequence_rope_is_fp32_and_has_one_row_per_token() -> None:
    model = HartleyTileAR(
        width=32,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        diff_width=32,
        diff_depth=1,
        inference_steps=2,
        grid=4,
        token_dim=16,
        token_order="raster",
        rope_mode="sequence",
        gradient_checkpointing=False,
    )
    assert model.rope_cos.shape[0] == 16
    assert model.rope_cos.dtype == torch.float32
