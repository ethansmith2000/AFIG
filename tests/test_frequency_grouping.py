"""Tests for compact-FFT radial-sector grouping."""

import torch

from frequency_grouping import (
    build_radial_sector_layout,
    pack_radial_sectors,
    unpack_radial_sectors,
)


def test_radial_sector_layout_never_crosses_rings_and_roundtrips() -> None:
    radius = torch.tensor([0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    layout = build_radial_sector_layout(radius, group_size=3)
    assert layout.group_count == 5
    assert layout.group_size == 3
    for group in range(layout.group_count):
        active = layout.indices[group, layout.mask[group]]
        assert radius[active].unique().tolist() == [layout.radius[group].item()]

    tokens = torch.arange(2 * radius.numel() * 6).reshape(2, -1, 6).float()
    groups = pack_radial_sectors(tokens, layout.indices, layout.mask)
    restored = unpack_radial_sectors(
        groups, layout.indices, layout.mask, radius.numel()
    )
    torch.testing.assert_close(restored, tokens)


def test_cifar_compact_fft_group_size_four_has_134_steps() -> None:
    # The exact compact 32x32 ring counts. This guards the intended experimental
    # interface: 514 coefficients become 134 ring-sector prediction steps.
    counts = torch.tensor(
        [1, 4, 8, 10, 12, 20, 18, 24, 28, 28, 34, 32, 40, 46, 44, 48, 38, 30, 18, 16, 8, 6, 1]
    )
    radius = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    layout = build_radial_sector_layout(radius, group_size=4)
    assert layout.group_count == 134
