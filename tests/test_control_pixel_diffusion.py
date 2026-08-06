"""Representation controls for local and globally supported real DCT tokens."""

from __future__ import annotations

import torch

from control_pixel_diffusion import (
    build_compact_isometric_codec,
    compact_active_scalar_layout,
    compact_isometric_fft_to_tokens,
    compact_isometric_orbit_mask,
    compact_isometric_tokens_to_images,
    compact_scalar_fft_to_tokens,
    compact_scalar_tokens_to_images,
    compact_scale_homogeneous_permutation,
    fit_compact_scalar_rms,
    fit_compact_phase_preserving_scale,
    full_dctify,
    full_idctify,
    full_hartleyify,
    full_ihartleyify,
    flow_interpolate_and_velocity,
    orthonormal_dct_matrix,
    orthonormal_hartley_matrix,
    orbit_order_permutation,
    patch_dctify,
    patch_grid_dctify,
    patch_grid_idctify,
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


def test_patch_grid_dct_roundtrip_shape_energy_and_global_support() -> None:
    images = torch.randn(3, 3, 32, 32)
    tokens = patch_grid_dctify(images, patch=4)
    decoded = patch_grid_idctify(tokens, patch=4, size=32)
    assert tokens.shape == (3, 64, 48)
    torch.testing.assert_close(decoded, images, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        tokens.square().sum(), images.square().sum(), atol=3e-3, rtol=3e-5
    )

    impulse = torch.zeros(1, 3, 32, 32)
    impulse[:, 0, 0, 0] = 1.0
    support = patch_grid_dctify(impulse, patch=4)[0, :, 0].abs() > 1e-7
    assert int(support.sum()) == 64


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


def test_compact_isometric_fft_roundtrip_shape_and_energy() -> None:
    for size in (8, 16, 32):
        codec = build_compact_isometric_codec(size, torch.device("cpu"))
        permutation = orbit_order_permutation(codec, "square_spiral")
        images = torch.randn(3, 3, size, size)
        tokens = compact_isometric_fft_to_tokens(codec, images, permutation)
        decoded = compact_isometric_tokens_to_images(codec, tokens, permutation)
        assert tokens.shape == (3, (3 * size * size) // 48, 48)
        assert torch.allclose(decoded, images, atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            tokens.square().sum(dim=(1, 2)),
            images.square().sum(dim=(1, 2, 3)),
            atol=3e-4,
            rtol=2e-6,
        )


def test_corrected_compact_scalar_layouts_roundtrip_energy_and_bridge() -> None:
    codec = build_compact_isometric_codec(32, torch.device("cpu"))
    images = torch.randn(9, 3, 32, 32)
    scalar_rms, orbit_rms = fit_compact_scalar_rms(codec, images)
    assert scalar_rms.shape == (codec.seq_len_int, 6)
    permutations = (
        orbit_order_permutation(codec, "square_spiral"),
        compact_scale_homogeneous_permutation(codec, orbit_rms),
    )
    for permutation in permutations:
        layout_orbit, layout_component = compact_active_scalar_layout(
            codec, permutation
        )
        tokens = compact_scalar_fft_to_tokens(
            codec, images, layout_orbit, layout_component
        )
        decoded = compact_scalar_tokens_to_images(
            codec, tokens, layout_orbit, layout_component
        )
        assert tokens.shape == (9, 64, 48)
        torch.testing.assert_close(decoded, images, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            tokens.square().sum(dim=(1, 2)),
            images.square().sum(dim=(1, 2, 3)),
            atol=3e-4,
            rtol=2e-6,
        )

        noise = torch.randn_like(images)
        time = 0.37
        mixed_image = time * images + (1.0 - time) * noise
        mixed_tokens = compact_scalar_fft_to_tokens(
            codec, mixed_image, layout_orbit, layout_component
        )
        image_tokens = compact_scalar_fft_to_tokens(
            codec, images, layout_orbit, layout_component
        )
        noise_tokens = compact_scalar_fft_to_tokens(
            codec, noise, layout_orbit, layout_component
        )
        torch.testing.assert_close(
            mixed_tokens,
            time * image_tokens + (1.0 - time) * noise_tokens,
            atol=2e-5,
            rtol=2e-5,
        )


def test_corrected_gridlocal_layout_keeps_self_conjugates_inline() -> None:
    codec = build_compact_isometric_codec(32, torch.device("cpu"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    layout_orbit, layout_component = compact_active_scalar_layout(codec, permutation)
    dc = ((codec.ky == 0) & (codec.kx == 0)).nonzero().item()
    assert torch.equal(layout_orbit[:3], torch.full((3,), dc))
    assert torch.equal(layout_component[:3], torch.arange(3))
    # The next scalar belongs to the next physical square-spiral orbit, rather
    # than to a prepended Nyquist self-conjugate coefficient.
    assert int(layout_orbit[3]) == int(permutation[1])
    assert not bool(codec.is_self_conjugate[layout_orbit[3]])


def test_compact_isometric_orbit_mask_tracks_mixed_self_conjugate_units() -> None:
    codec = build_compact_isometric_codec(32, torch.device("cpu"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    selected = torch.zeros(codec.seq_len_int, dtype=torch.bool)
    dc_index = ((codec.kx == 0) & (codec.ky == 0)).nonzero().item()
    selected[dc_index] = True
    mask = compact_isometric_orbit_mask(codec, selected, permutation)
    assert mask.shape == (64, 48)
    assert int(mask.sum()) == 3

    selected[:] = ~codec.is_self_conjugate
    mask = compact_isometric_orbit_mask(codec, selected, permutation)
    assert int(mask.sum()) == 6 * int((~codec.is_self_conjugate).sum())


def test_compact_isometric_fft_preserves_linear_gaussian_bridge() -> None:
    codec = build_compact_isometric_codec(32, torch.device("cpu"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    data = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(data)
    time = torch.tensor([0.0, 0.2, 0.7, 1.0])[:, None, None, None]
    pixel_bridge = time * data + (1.0 - time) * noise
    transformed_bridge = compact_isometric_fft_to_tokens(
        codec, pixel_bridge, permutation
    )
    expected = (
        time.flatten()[:, None, None]
        * compact_isometric_fft_to_tokens(codec, data, permutation)
        + (1.0 - time.flatten()[:, None, None])
        * compact_isometric_fft_to_tokens(codec, noise, permutation)
    )
    assert torch.allclose(transformed_bridge, expected, atol=2e-5, rtol=2e-5)


def test_compact_phase_preserving_scale_roundtrip_and_unit_rms() -> None:
    codec = build_compact_isometric_codec(32, torch.device("cpu"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    images = torch.randn(32, 3, 32, 32)
    packed = compact_isometric_fft_to_tokens(codec, images, permutation)
    scale, global_rms = fit_compact_phase_preserving_scale(
        codec, images, permutation, exponent=0.8
    )
    normalized = packed / scale / global_rms
    restored = compact_isometric_tokens_to_images(
        codec, normalized * global_rms * scale, permutation
    )
    torch.testing.assert_close(restored, images, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        normalized.square().mean(), torch.tensor(1.0), atol=2e-5, rtol=2e-5
    )


def test_compact_fft_supports_matched_64_by_12_res16_layout() -> None:
    codec = build_compact_isometric_codec(16, torch.device("cpu"))
    permutation = orbit_order_permutation(codec, "square_spiral")
    images = torch.randn(8, 3, 16, 16)
    tokens = compact_isometric_fft_to_tokens(
        codec, images, permutation, token_dim=12
    )
    assert tokens.shape == (8, 64, 12)
    restored = compact_isometric_tokens_to_images(codec, tokens, permutation)
    torch.testing.assert_close(restored, images, atol=2e-5, rtol=2e-5)


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


def test_trig_vp_bridge_preserves_isotropic_gaussian_variance() -> None:
    generator = torch.Generator().manual_seed(123)
    data = torch.randn(200_000, 1, generator=generator)
    noise = torch.randn(200_000, 1, generator=generator)
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        time = torch.full((data.shape[0],), value)
        bridge, _ = flow_interpolate_and_velocity(data, noise, time, "trig_vp")
        assert abs(float(bridge.var(unbiased=False)) - 1.0) < 0.015


def test_flow_path_velocity_matches_finite_difference() -> None:
    data = torch.tensor([[[-1.5, 0.25]]])
    noise = torch.tensor([[[0.5, -0.75]]])
    time = torch.tensor([0.37])
    step = 1e-4
    for path in ("linear", "trig_vp"):
        value, velocity = flow_interpolate_and_velocity(data, noise, time, path)
        next_value, _ = flow_interpolate_and_velocity(
            data, noise, time + step, path
        )
        finite_difference = (next_value - value) / step
        torch.testing.assert_close(velocity, finite_difference, atol=2e-3, rtol=2e-3)
