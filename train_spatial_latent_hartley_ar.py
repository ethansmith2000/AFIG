"""AR flow over Hartley tiles of a learned compressive spatial AE latent map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from autoencoder_models import AutoencoderConfig, SpatialAutoencoder
from control_pixel_diffusion import (
    full_dctify,
    full_hartleyify,
    full_idctify,
    full_ihartleyify,
    patch_dctify,
    patch_idctify,
)
from train_hartley_ar import HartleyTileAR, hartley_tile_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diff_width", type=int, default=768)
    parser.add_argument("--diff_depth", type=int, default=3)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--diagnostic_steps", type=int, default=250)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--validation_images", type=int, default=16)
    parser.add_argument("--latent_patch", type=int, default=2)
    parser.add_argument("--stats_images", type=int, default=4096)
    parser.add_argument(
        "--latent_basis",
        choices=[
            "hartley",
            "spatial",
            "patch_dct",
            "patch_dct_freq_major",
            "block_dct",
            "full_dct_tiles",
            "compact_fft",
        ],
        default="hartley",
    )
    parser.add_argument(
        "--token_order",
        choices=["auto", "radial", "raster"],
        default="auto",
    )
    parser.add_argument(
        "--tiles_per_token",
        type=int,
        default=1,
        help=(
            "For radial Hartley targets, concatenate this many consecutive "
            "frequency tiles into one jointly decoded AR token."
        ),
    )
    parser.add_argument(
        "--dct_support",
        type=int,
        default=2,
        help="Spatial support for latent_basis=block_dct.",
    )
    parser.add_argument(
        "--block_dct_token_dim",
        type=int,
        default=16,
        help="Fixed exported token width for block-DCT support controls.",
    )
    parser.add_argument(
        "--compact_fft_token_dim",
        type=int,
        default=16,
        help="Fixed exported token width for the compact isometric FFT bridge.",
    )
    parser.add_argument(
        "--rope_mode",
        choices=["frequency_2d", "sequence"],
        default="frequency_2d",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_spatial_ae(path: str, device: torch.device) -> SpatialAutoencoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = AutoencoderConfig(**payload["config"])
    if config.mode != "spatial_downsample":
        raise ValueError("Phase-D bridge requires a spatial_downsample checkpoint")
    model = SpatialAutoencoder(config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_images(
    autoencoder: SpatialAutoencoder,
    images: torch.Tensor,
    *,
    sample_posterior: bool = False,
) -> torch.Tensor:
    mean, logvar = autoencoder.encode(images)
    if sample_posterior and autoencoder.config.variational:
        mean = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
    return mean.float()


@torch.no_grad()
def fit_channel_stats(
    autoencoder: SpatialAutoencoder,
    dataset,
    count: int,
    device: torch.device,
    batch_size: int = 256,
    sample_posterior: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    count = min(count, len(dataset))
    for start in range(0, count, batch_size):
        images = torch.stack(
            [dataset[index][0] for index in range(start, min(start + batch_size, count))]
        ).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            chunks.append(
                encode_images(
                    autoencoder, images, sample_posterior=sample_posterior
                ).cpu()
            )
    latents = torch.cat(chunks)
    channel_mean = latents.mean(dim=(0, 2, 3), keepdim=True)
    channel_std = latents.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    return channel_mean.to(device), channel_std.to(device)


def latent_maps_to_tokens(
    maps: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    patch: int,
    basis: str = "hartley",
    dct_support: int = 2,
    block_dct_token_dim: int = 16,
    compact_fft_token_dim: int = 16,
) -> torch.Tensor:
    normalized = (maps - channel_mean) / channel_std
    if basis == "hartley":
        return full_hartleyify(normalized, patch)
    if basis == "patch_dct":
        return patch_dctify(normalized, patch)
    if basis == "patch_dct_freq_major":
        local_dct = patch_dctify(normalized, patch)
        return frequency_major_local_dct(local_dct, normalized.shape[-1], patch)
    if basis == "full_dct_tiles":
        raster = full_dctify(normalized, patch)
        order = dct_tile_order(normalized.shape[-1] // patch, normalized.device)
        return raster[:, order]
    if basis == "block_dct":
        return block_dct_support_tokens(
            normalized,
            support=dct_support,
            token_dim=block_dct_token_dim,
        )
    if basis == "compact_fft":
        return compact_isometric_fft_tokens(
            normalized, token_dim=compact_fft_token_dim
        )
    if basis != "spatial":
        raise ValueError(f"Unknown latent basis: {basis}")
    batch, channels, height, width = normalized.shape
    if height % patch or width % patch:
        raise ValueError("latent map dimensions must be divisible by patch")
    return (
        normalized.unfold(2, patch, patch)
        .unfold(3, patch, patch)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(batch, (height // patch) * (width // patch), channels * patch**2)
    )


def tokens_to_latent_maps(
    raster_tokens: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    patch: int,
    size: int,
    basis: str = "hartley",
    dct_support: int = 2,
    block_dct_token_dim: int = 16,
    compact_fft_token_dim: int = 16,
) -> torch.Tensor:
    if basis == "hartley":
        normalized = full_ihartleyify(raster_tokens, patch, size)
    elif basis == "patch_dct":
        normalized = patch_idctify(raster_tokens, patch, size)
    elif basis == "patch_dct_freq_major":
        local_dct = restore_local_dct_raster(raster_tokens, size, patch)
        normalized = patch_idctify(local_dct, patch, size)
    elif basis == "full_dct_tiles":
        grid = size // patch
        order = dct_tile_order(grid, raster_tokens.device)
        raster = torch.empty_like(raster_tokens)
        raster[:, order] = raster_tokens
        normalized = full_idctify(raster, patch, size)
    elif basis == "block_dct":
        normalized = restore_block_dct_support_tokens(
            raster_tokens,
            size=size,
            channels=channel_mean.shape[1],
            support=dct_support,
            token_dim=block_dct_token_dim,
        )
    elif basis == "compact_fft":
        normalized = restore_compact_isometric_fft_tokens(
            raster_tokens,
            size=size,
            channels=channel_mean.shape[1],
            token_dim=compact_fft_token_dim,
        )
    elif basis == "spatial":
        batch = raster_tokens.shape[0]
        channels = channel_mean.shape[1]
        side = size // patch
        expected = (side * side, channels * patch**2)
        if raster_tokens.shape[1:] != expected:
            raise ValueError(
                f"spatial tokens have shape {tuple(raster_tokens.shape[1:])}, "
                f"expected {expected}"
            )
        normalized = (
            raster_tokens.reshape(batch, side, side, channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, size, size)
        )
    else:
        raise ValueError(f"Unknown latent basis: {basis}")
    return normalized * channel_std + channel_mean


def group_radial_hartley_tiles(
    raster_tokens: torch.Tensor,
    grid: int,
    tiles_per_token: int,
) -> torch.Tensor:
    """Order individual Hartley tiles radially, then concatenate groups."""
    if tiles_per_token < 1 or grid * grid % tiles_per_token:
        raise ValueError("tiles_per_token must divide the Hartley tile count")
    order = hartley_tile_order(grid).to(raster_tokens.device)
    ordered = raster_tokens[:, order]
    batch, tile_count, tile_dim = ordered.shape
    return ordered.reshape(
        batch, tile_count // tiles_per_token, tile_dim * tiles_per_token
    )


def ungroup_radial_hartley_tiles(
    grouped_tokens: torch.Tensor,
    grid: int,
    tiles_per_token: int,
) -> torch.Tensor:
    """Undo :func:`group_radial_hartley_tiles` back to frequency raster."""
    batch, group_count, group_dim = grouped_tokens.shape
    tile_count = grid * grid
    if tiles_per_token < 1 or tile_count % tiles_per_token:
        raise ValueError("tiles_per_token must divide the Hartley tile count")
    if group_count != tile_count // tiles_per_token or group_dim % tiles_per_token:
        raise ValueError("unexpected grouped Hartley token shape")
    ordered = grouped_tokens.reshape(
        batch, tile_count, group_dim // tiles_per_token
    )
    order = hartley_tile_order(grid).to(grouped_tokens.device)
    raster = torch.empty_like(ordered)
    raster[:, order] = ordered
    return raster


def frequency_major_local_dct(
    raster_tokens: torch.Tensor,
    size: int,
    patch: int,
) -> torch.Tensor:
    """Regroup local DCT values by subband, then nearby spatial blocks.

    The mapping preserves token count and dimension when each output token groups
    a ``patch x patch`` neighborhood of spatial DCT blocks. Sequence order is DCT
    subband first (DC to higher local frequencies), then spatial block raster.
    """
    batch, token_count, token_dim = raster_tokens.shape
    grid = size // patch
    if token_count != grid * grid or grid % patch:
        raise ValueError("local DCT grid is incompatible with frequency-major grouping")
    if token_dim % (patch * patch):
        raise ValueError("local DCT token dimension is incompatible with patch")
    channels = token_dim // (patch * patch)
    block_grid = grid // patch
    coefficients = raster_tokens.reshape(
        batch, grid, grid, channels, patch, patch
    ).reshape(
        batch,
        block_grid,
        patch,
        block_grid,
        patch,
        channels,
        patch,
        patch,
    )
    return coefficients.permute(0, 6, 7, 1, 3, 2, 4, 5).reshape(
        batch, patch * patch * block_grid * block_grid, token_dim
    )


def restore_local_dct_raster(
    frequency_major_tokens: torch.Tensor,
    size: int,
    patch: int,
) -> torch.Tensor:
    """Invert :func:`frequency_major_local_dct`."""
    batch, token_count, token_dim = frequency_major_tokens.shape
    grid = size // patch
    block_grid = grid // patch
    if grid % patch or token_count != grid * grid:
        raise ValueError("frequency-major token count is incompatible with grid")
    if token_dim % (patch * patch):
        raise ValueError("frequency-major token dimension is incompatible with patch")
    channels = token_dim // (patch * patch)
    coefficients = frequency_major_tokens.reshape(
        batch,
        patch,
        patch,
        block_grid,
        block_grid,
        patch,
        patch,
        channels,
    )
    return coefficients.permute(0, 3, 5, 4, 6, 7, 1, 2).reshape(
        batch, grid * grid, token_dim
    )


def dct_frequency_order(support: int, device: torch.device) -> torch.Tensor:
    """Return a deterministic low-to-high order on a DCT frequency grid."""
    if support <= 0:
        raise ValueError("DCT support must be positive")
    entries = []
    for u in range(support):
        for v in range(support):
            entries.append((u * u + v * v, u + v, u, v))
    entries.sort()
    return torch.tensor(
        [u * support + v for _, _, u, v in entries],
        device=device,
        dtype=torch.long,
    )


def dct_tile_order(grid: int, device: torch.device) -> torch.Tensor:
    """Low-to-high order for contiguous tiles of a nonnegative DCT plane."""
    if grid <= 0:
        raise ValueError("DCT tile grid must be positive")
    entries = []
    for y in range(grid):
        for x in range(grid):
            entries.append((y * y + x * x, y + x, y, x))
    entries.sort()
    return torch.tensor(
        [y * grid + x for _, _, y, x in entries],
        device=device,
        dtype=torch.long,
    )


def compact_isometric_fft_tokens(
    normalized_maps: torch.Tensor,
    *,
    token_dim: int = 16,
) -> torch.Tensor:
    """Exactly pack a real map's Hermitian FFT into fixed-width AR tokens.

    Non-self-conjugate real/imaginary coordinates receive the required sqrt(2)
    factor, so this is an orthonormal real basis change. Self-conjugate orbits
    are paired first; ordinary radial orbits follow. For C4's 4x8x8 map this is
    32 eight-dimensional orbit units, exported as 16x16 tokens.
    """
    from frequency import build_orbit_table

    batch, channels, height, width = normalized_maps.shape
    if height != width:
        raise ValueError("compact FFT currently requires square latent maps")
    total_values = channels * height * width
    if token_dim <= 0 or total_values % token_dim:
        raise ValueError("compact FFT token width must divide latent scalar count")
    table = build_orbit_table(height, width, ordering="radial")
    ky = table["ky"].to(normalized_maps.device)
    kx = table["kx"].to(normalized_maps.device)
    is_self = table["is_self_conjugate"].to(normalized_maps.device)
    spectrum = torch.fft.fft2(normalized_maps.float(), norm="ortho")
    coeffs = spectrum[:, :, ky, kx].permute(0, 2, 1)
    self_values = coeffs[:, is_self].real
    if self_values.numel() and (self_values.shape[1] * channels) % (2 * channels):
        raise RuntimeError("self-conjugate FFT values cannot be paired")
    self_units = self_values.reshape(batch, -1, 2 * channels)
    ordinary = coeffs[:, ~is_self]
    ordinary_units = math.sqrt(2.0) * torch.cat(
        [ordinary.real, ordinary.imag], dim=-1
    )
    units = torch.cat([self_units, ordinary_units], dim=1)
    compact = units.reshape(batch, -1)
    if compact.shape[1] != total_values:
        raise RuntimeError(
            f"compact FFT exported {compact.shape[1]} values, expected {total_values}"
        )
    return compact.reshape(batch, total_values // token_dim, token_dim)


def restore_compact_isometric_fft_tokens(
    tokens: torch.Tensor,
    *,
    size: int,
    channels: int,
    token_dim: int = 16,
) -> torch.Tensor:
    """Invert :func:`compact_isometric_fft_tokens` exactly."""
    from frequency import build_orbit_table

    batch = tokens.shape[0]
    total_values = channels * size * size
    if tokens.shape[1:] != (total_values // token_dim, token_dim):
        raise ValueError("unexpected compact FFT token shape")
    table = build_orbit_table(size, size, ordering="radial")
    ky = table["ky"].to(tokens.device)
    kx = table["kx"].to(tokens.device)
    partner_ky = table["partner_ky"].to(tokens.device)
    partner_kx = table["partner_kx"].to(tokens.device)
    is_self = table["is_self_conjugate"].to(tokens.device)
    num_self = int(is_self.sum().item())
    if num_self % 2:
        raise RuntimeError("self-conjugate FFT values cannot be paired")
    units = tokens.float().reshape(batch, -1, 2 * channels)
    self_unit_count = num_self // 2
    self_values = units[:, :self_unit_count].reshape(batch, num_self, channels)
    ordinary_units = units[:, self_unit_count:]
    ordinary = torch.complex(
        ordinary_units[..., :channels] / math.sqrt(2.0),
        ordinary_units[..., channels:] / math.sqrt(2.0),
    )
    values = torch.zeros(
        batch,
        ky.numel(),
        channels,
        dtype=torch.complex64,
        device=tokens.device,
    )
    values[:, is_self] = torch.complex(self_values, torch.zeros_like(self_values))
    values[:, ~is_self] = ordinary
    spectrum = torch.zeros(
        batch,
        channels,
        size,
        size,
        dtype=torch.complex64,
        device=tokens.device,
    )
    spectrum[:, :, ky, kx] = values.permute(0, 2, 1)
    non_self = ~is_self
    spectrum[:, :, partner_ky[non_self], partner_kx[non_self]] = torch.conj(
        values[:, non_self].permute(0, 2, 1)
    )
    return torch.fft.ifft2(spectrum, norm="ortho").real


def block_dct_support_tokens(
    normalized_maps: torch.Tensor,
    *,
    support: int,
    token_dim: int,
) -> torch.Tensor:
    """Export fixed-width low-to-high groups from local DCT blocks.

    Token order is frequency-group major, then spatial-block raster. For a
    4x8x8 latent map and token_dim=16, supports 2/4/8 all export 16x16 values.
    """
    batch, channels, height, width = normalized_maps.shape
    if height != width or height % support:
        raise ValueError("block-DCT support must divide a square latent map")
    if token_dim <= 0 or token_dim % channels:
        raise ValueError("block-DCT token width must be divisible by channels")
    frequencies_per_token = token_dim // channels
    if support * support % frequencies_per_token:
        raise ValueError("token width must divide each block's DCT frequencies")

    raster = patch_dctify(normalized_maps, support)
    block_count = (height // support) ** 2
    frequency_groups = support * support // frequencies_per_token
    coefficients = raster.reshape(
        batch, block_count, channels, support * support
    )
    order = dct_frequency_order(support, normalized_maps.device)
    ordered = coefficients.index_select(-1, order).reshape(
        batch,
        block_count,
        channels,
        frequency_groups,
        frequencies_per_token,
    )
    return ordered.permute(0, 3, 1, 2, 4).reshape(
        batch, frequency_groups * block_count, token_dim
    )


def restore_block_dct_support_tokens(
    tokens: torch.Tensor,
    *,
    size: int,
    channels: int,
    support: int,
    token_dim: int,
) -> torch.Tensor:
    """Invert :func:`block_dct_support_tokens`."""
    if size % support or token_dim % channels:
        raise ValueError("invalid block-DCT inverse geometry")
    frequencies_per_token = token_dim // channels
    if support * support % frequencies_per_token:
        raise ValueError("token width must divide each block's DCT frequencies")
    block_count = (size // support) ** 2
    frequency_groups = support * support // frequencies_per_token
    expected = (frequency_groups * block_count, token_dim)
    if tokens.shape[1:] != expected:
        raise ValueError(
            f"block-DCT tokens have shape {tuple(tokens.shape[1:])}, expected {expected}"
        )

    ordered = tokens.reshape(
        tokens.shape[0],
        frequency_groups,
        block_count,
        channels,
        frequencies_per_token,
    ).permute(0, 2, 3, 1, 4).reshape(
        tokens.shape[0], block_count, channels, support * support
    )
    order = dct_frequency_order(support, tokens.device)
    coefficients = torch.empty_like(ordered)
    coefficients[..., order] = ordered
    raster = coefficients.reshape(
        tokens.shape[0], block_count, channels * support * support
    )
    return patch_idctify(raster, support, size)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.width = 64
        args.layers = 1
        args.heads = 4
        args.ff_mult = 2
        args.diff_width = 64
        args.diff_depth = 1
        args.inference_steps = 2
        args.preview_steps = 1
        args.diagnostic_steps = 1
        args.checkpoint_steps = 0
        args.validation_images = 2
        args.stats_images = 8

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_spatial_ae(args.ae_checkpoint, device)
    config = autoencoder.config
    token_order = args.token_order
    if token_order == "auto":
        token_order = "radial" if args.latent_basis == "hartley" else "raster"
    latent_size = config.spatial_resolution // config.spatial_downsample
    if latent_size % args.latent_patch:
        raise ValueError("latent map size must be divisible by latent_patch")
    physical_grid = latent_size // args.latent_patch
    if args.tiles_per_token != 1 and (
        args.latent_basis != "hartley" or token_order != "radial"
    ):
        raise ValueError(
            "tiles_per_token > 1 currently requires radial Hartley targets"
        )
    if args.latent_basis in ("block_dct", "compact_fft"):
        if latent_size % args.dct_support:
            if args.latent_basis == "block_dct":
                raise ValueError("dct_support must divide the latent map size")
        total_values = config.spatial_latent_channels * latent_size**2
        fixed_token_dim = (
            args.block_dct_token_dim
            if args.latent_basis == "block_dct"
            else args.compact_fft_token_dim
        )
        if total_values % fixed_token_dim:
            raise ValueError("spectral token width must divide latent scalar count")
        group_count = total_values // fixed_token_dim
        base_token_dim = fixed_token_dim
    else:
        group_count = physical_grid**2 // args.tiles_per_token
        base_token_dim = config.spatial_latent_channels * args.latent_patch**2
    model_grid = math.isqrt(group_count)
    if model_grid**2 != group_count:
        raise ValueError("grouped AR token count must be a perfect square")
    model_token_order = "raster" if args.tiles_per_token > 1 else token_order

    plain = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    channel_mean, channel_std = fit_channel_stats(
        autoencoder, plain, args.stats_images, device
    )
    train_set = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    test_set = datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transforms.ToTensor()
    )
    validation_images = torch.stack(
        [test_set[index][0] for index in range(args.validation_images)]
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        validation_maps = encode_images(autoencoder, validation_images)
    validation_raster = latent_maps_to_tokens(
        validation_maps,
        channel_mean,
        channel_std,
        args.latent_patch,
        basis=args.latent_basis,
        dct_support=args.dct_support,
        block_dct_token_dim=args.block_dct_token_dim,
        compact_fft_token_dim=args.compact_fft_token_dim,
    )

    model = HartleyTileAR(
        width=args.width,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        diff_width=args.diff_width,
        diff_depth=args.diff_depth,
        inference_steps=args.inference_steps,
        grid=model_grid,
        token_dim=base_token_dim * args.tiles_per_token,
        token_order=model_token_order,
        rope_mode=args.rope_mode,
    ).to(device)
    if args.tiles_per_token > 1:
        validation_tokens = group_radial_hartley_tiles(
            validation_raster, physical_grid, args.tiles_per_token
        )
    else:
        validation_tokens = model.order_tokens(validation_raster)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"latent {args.latent_basis} AR: {model.seq_len} x {model.token_dim}; "
        f"latent={config.spatial_latent_channels}x{latent_size}x{latent_size}; "
        f"order={token_order}; rope={args.rope_mode}; "
        f"tiles_per_token={args.tiles_per_token}; dct_support={args.dct_support}; "
        f"params={parameter_count / 1e6:.1f}M"
    )
    print(
        "latent channel mean="
        + ",".join(f"{value:.4f}" for value in channel_mean.flatten().tolist())
    )
    print(
        "latent channel std="
        + ",".join(f"{value:.4f}" for value in channel_std.flatten().tolist())
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        ae_reconstruction = autoencoder.decode(validation_maps.to(next(autoencoder.parameters()).dtype))
    save_image(
        torch.cat([validation_images, ae_reconstruction.float()], dim=0).clamp(0, 1),
        output_dir / "ae_reconstruction.png",
        nrow=args.validation_images,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def schedule(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history = []
    progress = tqdm(total=args.steps, desc="latent-hartley-ar")
    global_step = 0
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                maps = encode_images(autoencoder, images)
                raster = latent_maps_to_tokens(
                    maps,
                    channel_mean,
                    channel_std,
                    args.latent_patch,
                    basis=args.latent_basis,
                    dct_support=args.dct_support,
                    block_dct_token_dim=args.block_dct_token_dim,
                    compact_fft_token_dim=args.compact_fft_token_dim,
                )
                if args.tiles_per_token > 1:
                    tokens = group_radial_hartley_tiles(
                        raster, physical_grid, args.tiles_per_token
                    )
                else:
                    tokens = model.order_tokens(raster)
            model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(tokens)
                loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0 or global_step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=scheduler.get_last_lr()[0],
                )
            record: Dict[str, float] = {
                "step": global_step,
                "loss": float(loss.detach()),
            }
            if args.diagnostic_steps and global_step % args.diagnostic_steps == 0:
                model.eval()
                cpu_state = torch.random.get_rng_state()
                cuda_state = (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                )
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    clean = model(validation_tokens)["loss"]
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                shuffled_history = validation_tokens.roll(1, 0)[:, :-1]
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    shuffled = model(
                        validation_tokens, history_override=shuffled_history
                    )["loss"]
                record.update(
                    clean=float(clean),
                    shuffled=float(shuffled),
                    gap=float(shuffled - clean),
                )
                print(
                    f"DIAGNOSTIC step={global_step} clean={float(clean):.6f} "
                    f"shuffled={float(shuffled):.6f} gap={float(shuffled-clean):.6f}"
                )
            history.append(record)
            if args.preview_steps and global_step % args.preview_steps == 0:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    ordered = model.generate(
                        2 if args.smoke else 16,
                        args.inference_steps,
                        generator,
                    )
                    if args.tiles_per_token > 1:
                        raster = ungroup_radial_hartley_tiles(
                            ordered.float(), physical_grid, args.tiles_per_token
                        )
                    else:
                        raster = model.restore_raster(ordered.float())
                    maps = tokens_to_latent_maps(
                        raster,
                        channel_mean,
                        channel_std,
                        args.latent_patch,
                        latent_size,
                        basis=args.latent_basis,
                        dct_support=args.dct_support,
                        block_dct_token_dim=args.block_dct_token_dim,
                        compact_fft_token_dim=args.compact_fft_token_dim,
                    )
                    decoded = autoencoder.decode(maps.to(next(autoencoder.parameters()).dtype))
                save_image(
                    decoded.float().clamp(0, 1),
                    output_dir / f"samples_{global_step}.png",
                    nrow=2 if args.smoke else 4,
                )
            if args.checkpoint_steps and global_step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                        "channel_mean": channel_mean.cpu(),
                        "channel_std": channel_std.cpu(),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "ae_checkpoint": str(Path(args.ae_checkpoint).resolve()),
                "latent_basis": args.latent_basis,
                "token_order": token_order,
                "rope_mode": args.rope_mode,
                "tiles_per_token": args.tiles_per_token,
                "dct_support": args.dct_support,
                "block_dct_token_dim": args.block_dct_token_dim,
                "compact_fft_token_dim": args.compact_fft_token_dim,
                "channel_mean": channel_mean.flatten().cpu().tolist(),
                "channel_std": channel_std.flatten().cpu().tolist(),
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
