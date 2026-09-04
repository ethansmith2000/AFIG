from __future__ import annotations

import pytest
import torch

from scripts.analyze_generation_trajectory import (
    first_sustained_time,
    parse_named_paths,
    radial_masks,
)


def test_radial_masks_partition_fft_grid() -> None:
    masks = radial_masks(32, torch.device("cpu"))
    coverage = torch.stack(list(masks.values())).sum(dim=0)
    torch.testing.assert_close(coverage, torch.ones_like(coverage))
    assert int(masks["r0-2"].sum()) > 0
    assert int(masks["r17+"].sum()) > 0


def test_first_sustained_time_rejects_temporary_crossing() -> None:
    times = [0.0, 0.1, 0.2, 0.3]
    assert first_sustained_time(
        [1.0, 0.2, 0.3, 0.1], times, threshold=0.25, below=True
    ) == pytest.approx(0.3)
    assert first_sustained_time(
        [0.0, 0.95, 0.8, 0.92], times, threshold=0.9, below=False
    ) == pytest.approx(0.3)
    assert first_sustained_time(
        [1.0, 0.9, 0.8, 0.7], times, threshold=0.5, below=True
    ) is None


def test_named_paths_require_unique_labels() -> None:
    parsed = parse_named_paths(["control=/tmp/a", "soft=/tmp/b"], "checkpoint")
    assert list(parsed) == ["control", "soft"]
    with pytest.raises(ValueError, match="unique"):
        parse_named_paths(["control=/tmp/a", "control=/tmp/b"], "checkpoint")
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_named_paths(["missing-separator"], "checkpoint")
