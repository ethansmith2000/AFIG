"""Deterministic radial-sector grouping for compact Fourier coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RadialSectorLayout:
    """Padded groups that never cross an integer-radius ring boundary."""

    indices: torch.Tensor
    mask: torch.Tensor
    radius: torch.Tensor

    @property
    def group_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.indices.shape[1])


def build_radial_sector_layout(
    radius_bin: torch.Tensor,
    group_size: int,
) -> RadialSectorLayout:
    """Partition each radius into consecutive, fixed-capacity sectors.

    The source sequence ordering is retained within each radius. Padding points
    at coefficient zero but is always masked, so packing and unpacking are exact.
    """
    radius_bin = radius_bin.detach().long().cpu().reshape(-1)
    if radius_bin.numel() == 0:
        raise ValueError("radius_bin must be non-empty")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if int(radius_bin.min()) < 0:
        raise ValueError("radius bins must be nonnegative")

    chunks: list[torch.Tensor] = []
    radii: list[int] = []
    for radius in range(int(radius_bin.max()) + 1):
        positions = torch.nonzero(radius_bin == radius, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        for start in range(0, positions.numel(), group_size):
            chunks.append(positions[start : start + group_size])
            radii.append(radius)

    indices = torch.zeros(len(chunks), group_size, dtype=torch.long)
    mask = torch.zeros(len(chunks), group_size, dtype=torch.bool)
    for group, positions in enumerate(chunks):
        indices[group, : positions.numel()] = positions
        mask[group, : positions.numel()] = True
    return RadialSectorLayout(
        indices=indices,
        mask=mask,
        radius=torch.tensor(radii, dtype=torch.long),
    )


def pack_radial_sectors(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,L,D]`` coefficient tokens into ``[B,G,K,D]`` groups."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B,L,D]")
    gathered = tokens[:, indices]
    return gathered * mask.to(device=tokens.device, dtype=tokens.dtype)[None, :, :, None]


def unpack_radial_sectors(
    groups: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    sequence_length: int,
) -> torch.Tensor:
    """Invert :func:`pack_radial_sectors` exactly on every active entry."""
    if groups.ndim != 4 or groups.shape[1:3] != indices.shape:
        raise ValueError("groups must have shape [B,G,K,D] matching the layout")
    result = torch.zeros(
        groups.shape[0],
        sequence_length,
        groups.shape[-1],
        device=groups.device,
        dtype=groups.dtype,
    )
    active_indices = indices[mask]
    result[:, active_indices] = groups[:, mask]
    return result
