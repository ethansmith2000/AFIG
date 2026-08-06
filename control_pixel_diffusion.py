"""Control: the same transformer and training budget, on pixel patches.

The joint latent model does not memorize yet produces texture mush, and CIFAR-10's
50k images are known to support excellent generation (DDPM FID 3.17 at ~36M
params, EDM FID 1.79 at ~56M). So "not enough data" cannot by itself explain the
failure -- unless our *architecture and training budget* are the limiting factor
rather than the representation.

This isolates that. Identical bidirectional transformer blocks, identical
rectified-flow objective, identical width/depth/steps/batch/schedule as the joint
latent runs. The only change is what a token is: a 4x4 pixel patch (64 tokens of
48 dims) instead of a frequency latent (53 tokens of 64 dims).

Outcome reading:
  coherent images  -> data and architecture are sufficient; the frequency latent
                      representation is what breaks generation
  texture mush     -> the transformer/budget/data combination is the limit, and
                      the representation is exonerated
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from causal_transformer import CausalTransformerBlock
from diffusion_decoder import FinalLayer, TimestepEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument(
        "--representation",
        choices=[
            "pixels",
            "patch_dct",
            "patch_grid_dct",
            "full_dct",
            "full_hartley",
            "fft_whitened",
            "fft_global",
            "fft_global_spiral",
            "fft_compact_isometric_spiral",
            "fft_compact_isometric_gridlocal",
            "fft_compact_isometric_scale",
            "fft_compact_scaled_spiral",
        ],
        default="pixels",
        help=(
            "pixels: 4x4 spatial patches. patch_dct: an orthonormal DCT inside "
            "each spatial patch. patch_grid_dct: patchify first, then apply an "
            "orthonormal DCT across the patch grid independently for each of the "
            "48 within-patch features. full_dct: one global orthonormal image DCT, "
            "grouped into 4x4 frequency patches. full_hartley: a real, periodic, "
            "globally supported orthonormal Fourier-family basis on the same "
            "frequency grid. fft_whitened: per-orbit whitened FFT. fft_global: "
            "FFT with only a global scalar mean/std. fft_global_spiral: the same "
            "FFT values reordered by square spiral before grouping, making each "
            "48-D token more local in the 2-D frequency plane. "
            "fft_compact_isometric_spiral: an exact orthonormal real packing of "
            "standardized pixels, with inactive Hermitian coordinates removed "
            "before reshaping to exactly 64x48 (legacy self-first layout). "
            "fft_compact_isometric_gridlocal: the same active scalars ordered "
            "inline by square-spiral frequency location, without prepending the "
            "self-conjugate values. fft_compact_isometric_scale: the same scalar "
            "layout with orbit units ordered by train-only uncentered RMS. "
            "fft_compact_scaled_spiral: the "
            "same packing divided by phase-preserving radial/RGB RMS scales."
        ),
    )
    parser.add_argument("--codec_stats", default="autoencoder_runs/codec_stats_32.pt")
    parser.add_argument("--orbits_per_token", type=int, default=8)
    parser.add_argument("--compact_token_dim", type=int, default=48)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--preview_steps", type=int, default=5000)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--spectral_scale_exponent", type=float, default=0.8)
    return parser.parse_args()


class PatchDiffusion(nn.Module):
    def __init__(self, tokens: int, dim: int, args: argparse.Namespace):
        super().__init__()
        width = args.width
        self.tokens, self.dim = tokens, dim
        self.flow_path = getattr(args, "flow_path", "linear")
        if self.flow_path not in ("linear", "trig_vp"):
            raise ValueError(f"Unknown flow path: {self.flow_path}")
        self.input_projection = nn.Linear(dim, width)
        self.position = nn.Parameter(torch.zeros(tokens, width))
        self.time_embed = TimestepEmbedder(width)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=args.num_heads,
                    ff_mult=args.ff_mult,
                    dropout=0.0,
                    conditional_film=True,
                    causal=False,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.final_layer = FinalLayer(width, dim)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def velocity(self, noisy: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(noisy) + self.position.to(noisy.dtype)
        condition = self.time_embed(flow_time * 999.0).unsqueeze(1).expand_as(hidden)
        for layer in self.layers:
            hidden, _ = layer(hidden, condition=condition)
        return self.final_layer(hidden, condition)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        t = torch.rand(batch, device=x.device)
        noise = torch.randn_like(x)
        noisy, target_velocity = flow_interpolate_and_velocity(
            x, noise, t, self.flow_path
        )
        return (self.velocity(noisy, t) - target_velocity).square().mean()

    @torch.no_grad()
    def sample(self, count: int, steps: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(count, self.tokens, self.dim, device=device)
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full((count,), index / steps, device=device)
            v = self.velocity(x, t)
            proposal = x + dt * v
            if index + 1 < steps:
                nt = torch.full((count,), (index + 1) / steps, device=device)
                x = x + 0.5 * dt * (v + self.velocity(proposal, nt))
            else:
                x = proposal
        return x


def flow_interpolate_and_velocity(
    data: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a base-to-data stochastic interpolant and its path velocity.

    ``linear`` is the existing rectified-flow bridge. ``trig_vp`` uses sine and
    cosine coefficients whose squared sum is one, so an isotropic Gaussian data
    distribution remains isotropic Gaussian throughout the bridge instead of
    pinching to half variance at the midpoint.
    """
    view = time.reshape((time.shape[0],) + (1,) * (data.ndim - 1))
    if path == "linear":
        return view * data + (1.0 - view) * noise, data - noise
    if path == "trig_vp":
        angle = 0.5 * math.pi * view
        data_weight = torch.sin(angle)
        noise_weight = torch.cos(angle)
        noisy = data_weight * data + noise_weight * noise
        velocity = 0.5 * math.pi * (
            noise_weight * data - data_weight * noise
        )
        return noisy, velocity
    raise ValueError(f"Unknown flow path: {path}")


def build_codec(args, device):
    """Codec for the FFT modes; whiten_exponent selects whitened vs global."""
    from frequency import FrequencyCodec, FrequencyCodecConfig

    payload = torch.load(args.codec_stats, map_location="cpu", weights_only=False)
    config_dict = dict(payload["config"])
    config_dict["whiten_exponent"] = 1.0 if args.representation == "fft_whitened" else 0.0
    codec = FrequencyCodec(FrequencyCodecConfig(**config_dict))
    codec.load_exported(payload)
    return codec.to(device).eval()


def orbit_order_permutation(codec, ordering: str) -> torch.Tensor:
    """Indices that express codec orbits in another deterministic ordering."""
    from frequency import build_orbit_table

    table = build_orbit_table(
        codec.config.height, codec.config.width, ordering=ordering
    )
    codec_index = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(codec.ky.cpu(), codec.kx.cpu()))
    }
    permutation = torch.tensor(
        [codec_index[(int(y), int(x))] for y, x in zip(table["ky"], table["kx"])],
        dtype=torch.long,
        device=codec.ky.device,
    )
    if permutation.unique().numel() != codec.seq_len_int:
        raise RuntimeError(f"{ordering} orbit order is not a bijection")
    return permutation


def build_compact_isometric_codec(size: int, device: torch.device):
    """Raw orthonormal Hermitian codec; fitted normalization is intentionally unused."""
    from frequency import FrequencyCodec, FrequencyCodecConfig

    codec = FrequencyCodec(
        FrequencyCodecConfig(
            height=size,
            width=size,
            ordering="radial",
            value_transform="identity",
            coordinate_packing="isometric",
        )
    )
    return codec.to(device).eval()


def compact_isometric_fft_to_tokens(codec, images, permutation, token_dim: int = 48):
    """Pack 3,072 active orthonormal FFT coordinates as 512 six-D units.

    Every ordinary Hermitian orbit remains one RGB-real/RGB-imag unit. The four
    three-D self-conjugate coefficients are paired into two six-D units, giving
    exactly 512 units -> 64 tokens of eight units -> 48 dimensions.
    """
    raw = codec.encode_raw(images)[:, permutation]
    self_conjugate = codec.is_self_conjugate[permutation]
    ordinary_units = raw[:, ~self_conjugate]
    self_units = raw[:, self_conjugate, :3].reshape(raw.shape[0], 2, 6)
    units = torch.cat([self_units, ordinary_units], dim=1)
    compact = units.reshape(raw.shape[0], -1)
    expected = images.shape[1] * images.shape[2] * images.shape[3]
    if compact.shape[1] != expected or expected % token_dim:
        raise RuntimeError(
            f"Expected token_dim={token_dim} to divide {expected} values, "
            f"got {compact.shape[1]}"
        )
    return compact.reshape(raw.shape[0], expected // token_dim, token_dim)


def compact_isometric_tokens_to_images(codec, grouped, permutation):
    """Invert :func:`compact_isometric_fft_to_tokens` exactly."""
    batch = grouped.shape[0]
    self_conjugate = codec.is_self_conjugate[permutation]
    expected = int(codec.component_mask.sum().item())
    compact = grouped.reshape(batch, -1)
    if compact.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} compact values, got {compact.shape[1]}"
        )
    num_self = int(self_conjugate.sum().item())
    if num_self % 2:
        raise RuntimeError("Self-conjugate coefficient count must be even")
    self_unit_count = num_self // 2
    unit_count = int((~self_conjugate).sum().item()) + self_unit_count
    units = compact.reshape(batch, unit_count, 6)
    self_values = units[:, :self_unit_count].reshape(batch, num_self, 3)
    ordinary_values = units[:, self_unit_count:]
    ordered = torch.zeros(
        batch,
        codec.seq_len_int,
        6,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    ordered[:, self_conjugate, :3] = self_values
    ordered[:, ~self_conjugate] = ordinary_values
    codec_order = torch.empty_like(ordered)
    codec_order[:, permutation] = ordered
    return codec.decode_raw(codec_order)


def compact_active_scalar_layout(
    codec, orbit_permutation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an active-only scalar layout without the legacy self-first splice.

    Each orbit appears at the location selected by ``orbit_permutation``.  The
    three real coordinates of a self-conjugate orbit are emitted inline; an
    ordinary orbit emits its three real then three imaginary coordinates.  The
    result is therefore an exact permutation of the ``3*H*W`` orthonormal real
    FFT coordinates, even though a 48-scalar token boundary may occasionally
    split an orbit.
    """
    orbit_permutation = orbit_permutation.to(
        device=codec.ky.device, dtype=torch.long
    )
    if orbit_permutation.shape != (codec.seq_len_int,):
        raise ValueError(
            f"orbit_permutation must be [{codec.seq_len_int}], got "
            f"{tuple(orbit_permutation.shape)}"
        )
    if orbit_permutation.unique().numel() != codec.seq_len_int:
        raise ValueError("orbit_permutation must be a bijection")
    orbit_indices: list[int] = []
    component_indices: list[int] = []
    mask = codec.component_mask.bool()
    for orbit in orbit_permutation.tolist():
        for component in mask[orbit].nonzero(as_tuple=False).flatten().tolist():
            orbit_indices.append(orbit)
            component_indices.append(component)
    expected = 3 * codec.config.height * codec.config.width
    if len(orbit_indices) != expected:
        raise RuntimeError(
            f"Expected {expected} active FFT scalars, got {len(orbit_indices)}"
        )
    return (
        torch.tensor(orbit_indices, device=codec.ky.device, dtype=torch.long),
        torch.tensor(component_indices, device=codec.ky.device, dtype=torch.long),
    )


def compact_scalar_fft_to_tokens(
    codec,
    images: torch.Tensor,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    token_dim: int = 48,
) -> torch.Tensor:
    """Gather an explicit active-scalar FFT permutation into fixed-width tokens."""
    raw = codec.encode_raw(images)
    compact = raw[:, layout_orbit, layout_component]
    if compact.shape[1] % token_dim:
        raise ValueError(
            f"token_dim={token_dim} does not divide {compact.shape[1]} active values"
        )
    return compact.reshape(raw.shape[0], -1, token_dim)


def compact_scalar_tokens_to_images(
    codec,
    grouped: torch.Tensor,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
) -> torch.Tensor:
    """Invert :func:`compact_scalar_fft_to_tokens` exactly."""
    compact = grouped.reshape(grouped.shape[0], -1)
    if compact.shape[1] != layout_orbit.numel():
        raise ValueError(
            f"Expected {layout_orbit.numel()} compact values, got {compact.shape[1]}"
        )
    raw = torch.zeros(
        grouped.shape[0],
        codec.seq_len_int,
        6,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    raw[:, layout_orbit, layout_component] = compact
    return codec.decode_raw(raw)


@torch.no_grad()
def fit_compact_scalar_rms(
    codec, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit train-only uncentered scalar and orbit RMS values."""
    raw = codec.encode_raw(images.to(codec.ky.device))
    scalar_rms = raw.square().mean(dim=0).sqrt()
    mask = codec.component_mask.bool()
    orbit_rms = (
        (scalar_rms.square() * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp_min(1)
    ).sqrt()
    return scalar_rms, orbit_rms


def compact_scale_homogeneous_permutation(
    codec, orbit_rms: torch.Tensor
) -> torch.Tensor:
    """Order complete orbit units by fitted RMS, with deterministic tie breaks."""
    if orbit_rms.shape != (codec.seq_len_int,):
        raise ValueError(
            f"orbit_rms must be [{codec.seq_len_int}], got {tuple(orbit_rms.shape)}"
        )
    radius = codec.radius.detach().cpu()
    ky = codec.ky.detach().cpu()
    kx = codec.kx.detach().cpu()
    rms = orbit_rms.detach().cpu()
    order = sorted(
        range(codec.seq_len_int),
        key=lambda index: (
            -float(rms[index]),
            float(radius[index]),
            int(ky[index]),
            int(kx[index]),
        ),
    )
    return torch.tensor(order, device=codec.ky.device, dtype=torch.long)


def compact_layout_diagnostics(
    codec,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    scalar_rms: torch.Tensor,
    token_dim: int = 48,
) -> dict[str, float]:
    """Summarize scale and physical-frequency heterogeneity within tokens."""
    if layout_orbit.numel() % token_dim:
        raise ValueError("layout size must be divisible by token_dim")
    scalar_scale = scalar_rms[layout_orbit, layout_component].reshape(-1, token_dim)
    ratios = scalar_scale.max(dim=-1).values / scalar_scale.min(dim=-1).values.clamp_min(
        1e-12
    )
    radius = codec.radius[layout_orbit].reshape(-1, token_dim)
    radius_spread = radius.max(dim=-1).values - radius.min(dim=-1).values

    # Mean pairwise toroidal distance between distinct frequency orbits in each
    # token. Repeated RGB/real/imag coordinates count once.
    token_orbits = layout_orbit.reshape(-1, token_dim)
    toroidal_means = []
    height, width = codec.config.height, codec.config.width
    for token in token_orbits:
        unique = token.unique()
        if unique.numel() < 2:
            toroidal_means.append(torch.zeros((), device=unique.device))
            continue
        ys = codec.ky[unique].float()
        xs = codec.kx[unique].float()
        dy = (ys[:, None] - ys[None, :]).abs()
        dx = (xs[:, None] - xs[None, :]).abs()
        dy = torch.minimum(dy, height - dy)
        dx = torch.minimum(dx, width - dx)
        distances = (dy.square() + dx.square()).sqrt()
        triangle = torch.triu_indices(
            unique.numel(), unique.numel(), offset=1, device=unique.device
        )
        toroidal_means.append(distances[triangle[0], triangle[1]].mean())
    toroidal = torch.stack(toroidal_means)
    return {
        "scale_ratio_median": float(ratios.median()),
        "scale_ratio_worst": float(ratios.max()),
        "radius_spread_median": float(radius_spread.median()),
        "radius_spread_worst": float(radius_spread.max()),
        "toroidal_distance_median": float(toroidal.median()),
        "toroidal_distance_worst": float(toroidal.max()),
    }


def compact_isometric_orbit_mask(
    codec,
    selected_orbits: torch.Tensor,
    permutation: torch.Tensor,
    token_dim: int = 48,
) -> torch.Tensor:
    """Pack a codec-order orbit selection into the compact scalar layout.

    The compact layout pairs the four three-scalar self-conjugate coefficients
    into two six-scalar units before appending ordinary complex coefficients.
    Consequently, a low-frequency selection cannot safely be represented as a
    simple token prefix.  This helper mirrors the packing exactly and can select
    individual RGB coordinates inside those first two mixed-support units.
    """
    selected_orbits = selected_orbits.to(
        device=permutation.device, dtype=torch.bool
    )
    if selected_orbits.shape != (codec.seq_len_int,):
        raise ValueError(
            f"selected_orbits must be [{codec.seq_len_int}], got "
            f"{tuple(selected_orbits.shape)}"
        )
    selected = selected_orbits[permutation]
    self_conjugate = codec.is_self_conjugate[permutation]
    active = torch.zeros(
        codec.seq_len_int,
        6,
        device=permutation.device,
        dtype=torch.bool,
    )
    active[~self_conjugate] = selected[~self_conjugate, None]
    active[self_conjugate, :3] = selected[self_conjugate, None]
    num_self = int(self_conjugate.sum().item())
    if num_self % 2:
        raise RuntimeError("Self-conjugate coefficient count must be even")
    self_mask = active[self_conjugate, :3].reshape(num_self // 2, 6)
    ordinary_mask = active[~self_conjugate]
    packed = torch.cat([self_mask, ordinary_mask], dim=0).reshape(-1)
    if packed.numel() % token_dim:
        raise ValueError(
            f"token_dim={token_dim} does not divide {packed.numel()} active values"
        )
    return packed.reshape(packed.numel() // token_dim, token_dim)


@torch.no_grad()
def fit_compact_phase_preserving_scale(
    codec,
    images: torch.Tensor,
    permutation: torch.Tensor,
    exponent: float,
    token_dim: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit uncentered radial/RGB RMS scales without changing complex angles.

    Real and imaginary coordinates of an ordinary coefficient share one positive
    divisor. Statistics are pooled within integer radius and RGB channel. The
    final scalar RMS only fixes the overall coordinate scale.
    """
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("spectral_scale_exponent must be in [0, 1]")
    images = images.to(codec.ky.device)
    raw = codec.encode_raw(images)[:, permutation]
    is_self = codec.is_self_conjugate[permutation]
    radius_bin = codec.radius_bin[permutation]
    orbit_scale = torch.empty(
        raw.shape[1], 3, device=raw.device, dtype=raw.dtype
    )
    for radius in radius_bin.unique(sorted=True):
        selected = radius_bin == radius
        ordinary_selected = selected & ~is_self
        for channel in range(3):
            values = [raw[:, selected, channel].reshape(-1)]
            if bool(ordinary_selected.any()):
                values.append(raw[:, ordinary_selected, channel + 3].reshape(-1))
            rms = torch.cat(values).square().mean().sqrt().clamp_min(1e-6)
            orbit_scale[selected, channel] = rms
    self_units = orbit_scale[is_self].reshape(2, 6)
    ordinary_scale = orbit_scale[~is_self]
    ordinary_units = torch.cat([ordinary_scale, ordinary_scale], dim=-1)
    coordinate_scale = torch.cat([self_units, ordinary_units], dim=0).reshape(
        -1, token_dim
    )
    coordinate_scale = coordinate_scale.pow(exponent)
    packed = compact_isometric_fft_to_tokens(
        codec, images, permutation, token_dim=token_dim
    )
    global_rms = (packed / coordinate_scale).square().mean().sqrt().clamp_min(1e-6)
    return coordinate_scale, global_rms


def fft_to_tokens(codec, images, orbits_per_token, permutation=None):
    """[B,3,32,32] -> [B, ceil(L/g), g*6], zero-padded on the orbit axis."""
    tokens = codec.encode(images)
    if permutation is not None:
        tokens = tokens[:, permutation]
    batch, orbits, components = tokens.shape
    groups = -(-orbits // orbits_per_token)
    padded = torch.zeros(
        batch, groups * orbits_per_token, components,
        device=tokens.device, dtype=tokens.dtype,
    )
    padded[:, :orbits] = tokens
    return padded.reshape(batch, groups, orbits_per_token * components)


def tokens_to_images(codec, grouped, orbits_per_token, permutation=None):
    batch, groups, _ = grouped.shape
    tokens = grouped.reshape(batch, groups * orbits_per_token, 6)[:, : codec.seq_len_int]
    if permutation is not None:
        codec_order = torch.empty_like(tokens)
        codec_order[:, permutation] = tokens
        tokens = codec_order
    return codec.decode(tokens)


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    x = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
    return x


def unpatchify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    batch = tokens.shape[0]
    grid = size // patch
    patch_area = patch * patch
    if tokens.shape[-1] % patch_area:
        raise ValueError("token dimension must be divisible by patch area")
    channels = tokens.shape[-1] // patch_area
    x = tokens.reshape(batch, grid, grid, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, size, size)
    return x


def orthonormal_dct_matrix(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix C."""
    sample = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequency = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    matrix = torch.cos(math.pi * frequency * sample[None, :] / size)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix.to(dtype=dtype)


def dct_2d(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal 2-D DCT over the final two dimensions."""
    return torch.einsum("ki,...ij,lj->...kl", matrix, values, matrix)


def idct_2d(coefficients: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Invert :func:`dct_2d` over the final two dimensions."""
    return torch.einsum("ki,...kl,lj->...ij", matrix, coefficients, matrix)


def patch_dctify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Encode spatially local patches in a real orthonormal frequency basis."""
    batch, channels, height, width = images.shape
    grid_h, grid_w = height // patch, width // patch
    patches = images.reshape(
        batch, channels, grid_h, patch, grid_w, patch
    ).permute(0, 2, 4, 1, 3, 5)
    matrix = orthonormal_dct_matrix(
        patch, device=images.device, dtype=images.dtype
    )
    coefficients = dct_2d(patches, matrix)
    return coefficients.reshape(batch, grid_h * grid_w, channels * patch * patch)


def patch_idctify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Decode spatially local DCT tokens back to normalized pixels."""
    batch = tokens.shape[0]
    grid = size // patch
    patch_area = patch * patch
    if tokens.shape[-1] % patch_area:
        raise ValueError("token dimension must be divisible by patch area")
    channels = tokens.shape[-1] // patch_area
    coefficients = tokens.reshape(batch, grid, grid, channels, patch, patch)
    matrix = orthonormal_dct_matrix(
        patch, device=tokens.device, dtype=tokens.dtype
    )
    patches = idct_2d(coefficients, matrix)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels, size, size
    )


def patch_grid_dctify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Globally mix patch positions while preserving within-patch feature slots.

    This is the missing orthogonal bridge between local pixel tokens and global
    spectral tokens: the 64-token axis is transformed, while each token's 48
    RGB/within-patch features retain exactly the pixel arm's interpretation.
    """
    tokens = patchify(images, patch)
    batch, count, features = tokens.shape
    grid = math.isqrt(count)
    if grid * grid != count:
        raise ValueError("patch_grid_dct requires a square patch grid")
    matrix = orthonormal_dct_matrix(
        grid, device=images.device, dtype=images.dtype
    )
    planes = tokens.reshape(batch, grid, grid, features).permute(0, 3, 1, 2)
    coefficients = dct_2d(planes, matrix)
    return coefficients.permute(0, 2, 3, 1).reshape(batch, count, features)


def patch_grid_idctify(
    tokens: torch.Tensor, patch: int, size: int
) -> torch.Tensor:
    """Invert :func:`patch_grid_dctify` exactly."""
    batch, count, features = tokens.shape
    grid = size // patch
    if count != grid * grid:
        raise ValueError(f"expected {grid * grid} patch-grid tokens, got {count}")
    matrix = orthonormal_dct_matrix(
        grid, device=tokens.device, dtype=tokens.dtype
    )
    coefficients = tokens.reshape(batch, grid, grid, features).permute(0, 3, 1, 2)
    patches = idct_2d(coefficients, matrix)
    spatial_tokens = patches.permute(0, 2, 3, 1).reshape(batch, count, features)
    return unpatchify(spatial_tokens, patch, size)


def full_dctify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Apply one global image DCT and group its plane into frequency patches."""
    matrix = orthonormal_dct_matrix(
        images.shape[-1], device=images.device, dtype=images.dtype
    )
    return patchify(dct_2d(images, matrix), patch)


def full_idctify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Invert globally supported DCT frequency-patch tokens."""
    coefficients = unpatchify(tokens, patch, size)
    matrix = orthonormal_dct_matrix(
        size, device=tokens.device, dtype=tokens.dtype
    )
    return idct_2d(coefficients, matrix)


def orthonormal_hartley_matrix(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the real orthonormal DHT matrix using cas(x)=cos(x)+sin(x)."""
    sample = torch.arange(size, device=device, dtype=torch.float32)
    frequency = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    angle = 2.0 * math.pi * frequency * sample[None, :] / size
    return ((angle.cos() + angle.sin()) / math.sqrt(size)).to(dtype=dtype)


def full_hartleyify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Apply a separable global real Hartley transform and frequency-patch it."""
    matrix = orthonormal_hartley_matrix(
        images.shape[-1], device=images.device, dtype=images.dtype
    )
    return patchify(dct_2d(images, matrix), patch)


def full_ihartleyify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Invert globally supported Hartley frequency-patch tokens."""
    coefficients = unpatchify(tokens, patch, size)
    matrix = orthonormal_hartley_matrix(
        size, device=tokens.device, dtype=tokens.dtype
    )
    return idct_2d(coefficients, matrix)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    transform_ops = [transforms.RandomHorizontalFlip()]
    if args.image_size != 32:
        transform_ops.append(
            transforms.Resize((args.image_size, args.image_size), antialias=True)
        )
    transform_ops.append(transforms.ToTensor())
    transform = transforms.Compose(transform_ops)
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
        drop_last=True, persistent_workers=True,
    )

    # One global scalar mean/std, the standard treatment for pixels -- which
    # preserves the eigenspectrum, unlike per-frequency whitening.
    sample_images = torch.stack([dataset[i][0] for i in range(4096)], dim=0)
    codec = None
    fft_permutation = None
    compact_layout_orbit = None
    compact_layout_component = None
    compact_layout_metrics = None
    compact_modes = (
        "fft_compact_isometric_spiral",
        "fft_compact_isometric_gridlocal",
        "fft_compact_isometric_scale",
        "fft_compact_scaled_spiral",
    )
    if args.representation in (
        "pixels",
        "patch_dct",
        "patch_grid_dct",
        "full_dct",
        "full_hartley",
        "fft_compact_isometric_spiral",
        "fft_compact_isometric_gridlocal",
        "fft_compact_isometric_scale",
        "fft_compact_scaled_spiral",
    ):
        mean = float(sample_images.mean())
        std = float(sample_images.std())
        tokens = (args.image_size // args.patch) ** 2
        dim = 3 * args.patch * args.patch
        if args.representation in compact_modes:
            codec = build_compact_isometric_codec(args.image_size, device)
            total_values = 3 * args.image_size * args.image_size
            if total_values % args.compact_token_dim:
                raise ValueError("compact_token_dim must divide image scalar count")
            tokens = total_values // args.compact_token_dim
            dim = args.compact_token_dim
            fft_permutation = orbit_order_permutation(codec, "square_spiral")
            if args.representation in (
                "fft_compact_isometric_gridlocal",
                "fft_compact_isometric_scale",
            ):
                scalar_rms, orbit_rms = fit_compact_scalar_rms(
                    codec, (sample_images.to(device) - mean) / std
                )
                if args.representation == "fft_compact_isometric_scale":
                    fft_permutation = compact_scale_homogeneous_permutation(
                        codec, orbit_rms
                    )
                compact_layout_orbit, compact_layout_component = (
                    compact_active_scalar_layout(codec, fft_permutation)
                )
                compact_layout_metrics = compact_layout_diagnostics(
                    codec,
                    compact_layout_orbit,
                    compact_layout_component,
                    scalar_rms,
                    token_dim=args.compact_token_dim,
                )
                with open(
                    os.path.join(args.output_dir, "compact_layout.json"), "w"
                ) as handle:
                    json.dump(compact_layout_metrics, handle, indent=2)
                print(
                    "compact layout diagnostics: "
                    + json.dumps(compact_layout_metrics, sort_keys=True)
                )
    else:
        codec = build_codec(args, device)
        if args.representation == "fft_global_spiral":
            fft_permutation = orbit_order_permutation(codec, "square_spiral")
        with torch.no_grad():
            probe = fft_to_tokens(
                codec,
                sample_images.to(device),
                args.orbits_per_token,
                fft_permutation,
            )
        # The codec already centres and scales; one residual global standardization
        # keeps every mode on the same footing for the diffusion process.
        mean = float(probe.mean())
        std = float(probe.std())
        tokens, dim = probe.shape[1], probe.shape[2]
    compact_coordinate_scale = None
    compact_global_rms = None
    if args.representation == "fft_compact_scaled_spiral":
        compact_coordinate_scale, compact_global_rms = fit_compact_phase_preserving_scale(
            codec,
            (sample_images.to(device) - mean) / std,
            fft_permutation,
            args.spectral_scale_exponent,
            token_dim=args.compact_token_dim,
        )
        print(
            f"compact phase-preserving scale exponent={args.spectral_scale_exponent:.3f} "
            f"coordinate min/max={float(compact_coordinate_scale.min()):.4f}/"
            f"{float(compact_coordinate_scale.max()):.4f} "
            f"global_rms={float(compact_global_rms):.4f}"
        )
    print(f"representation={args.representation} normalization: mean {mean:.4f} std {std:.4f}")
    model = PatchDiffusion(tokens, dim, args).to(device)
    print(f"tokens {tokens} x dim {dim}; params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=torch.cuda.is_available(),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 1.0 - 0.75 * progress

    step = 0
    history = []
    scaler_dtype = torch.bfloat16
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_at(step)
            images = images.to(device)
            if args.representation == "pixels":
                x = patchify((images - mean) / std, args.patch)
            elif args.representation == "patch_dct":
                x = patch_dctify((images - mean) / std, args.patch)
            elif args.representation == "patch_grid_dct":
                x = patch_grid_dctify((images - mean) / std, args.patch)
            elif args.representation == "full_dct":
                x = full_dctify((images - mean) / std, args.patch)
            elif args.representation == "full_hartley":
                x = full_hartleyify((images - mean) / std, args.patch)
            elif args.representation in (
                "fft_compact_isometric_spiral",
                "fft_compact_isometric_gridlocal",
                "fft_compact_isometric_scale",
                "fft_compact_scaled_spiral",
            ):
                with torch.no_grad():
                    if compact_layout_orbit is None:
                        x = compact_isometric_fft_to_tokens(
                            codec,
                            (images - mean) / std,
                            fft_permutation,
                            token_dim=args.compact_token_dim,
                        )
                    else:
                        x = compact_scalar_fft_to_tokens(
                            codec,
                            (images - mean) / std,
                            compact_layout_orbit,
                            compact_layout_component,
                            token_dim=args.compact_token_dim,
                        )
                    if compact_coordinate_scale is not None:
                        x = x / compact_coordinate_scale / compact_global_rms
            else:
                with torch.no_grad():
                    x = (
                        fft_to_tokens(
                            codec, images, args.orbits_per_token, fft_permutation
                        )
                        - mean
                    ) / std
            with torch.autocast("cuda", dtype=scaler_dtype):
                loss = model.loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 500 == 0:
                history.append({"step": step, "loss": float(loss.detach())})
                if step % 2500 == 0:
                    print(f"  step {step:>6} loss {float(loss.detach()):.4f}")
            if args.preview_steps and step % args.preview_steps == 0:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
                    samples = model.sample(16, args.inference_steps, device)
                raw = samples.float()
                if args.representation == "pixels":
                    decoded = unpatchify(raw, args.patch, args.image_size) * std + mean
                elif args.representation == "patch_dct":
                    decoded = patch_idctify(raw, args.patch, args.image_size) * std + mean
                elif args.representation == "patch_grid_dct":
                    decoded = patch_grid_idctify(raw, args.patch, args.image_size) * std + mean
                elif args.representation == "full_dct":
                    decoded = full_idctify(raw, args.patch, args.image_size) * std + mean
                elif args.representation == "full_hartley":
                    decoded = full_ihartleyify(raw, args.patch, args.image_size) * std + mean
                elif args.representation in (
                    "fft_compact_isometric_spiral",
                    "fft_compact_isometric_gridlocal",
                    "fft_compact_isometric_scale",
                    "fft_compact_scaled_spiral",
                ):
                    if compact_coordinate_scale is not None:
                        raw = raw * compact_global_rms * compact_coordinate_scale
                    if compact_layout_orbit is None:
                        normalized = compact_isometric_tokens_to_images(
                            codec, raw, fft_permutation
                        )
                    else:
                        normalized = compact_scalar_tokens_to_images(
                            codec,
                            raw,
                            compact_layout_orbit,
                            compact_layout_component,
                        )
                    decoded = normalized * std + mean
                else:
                    raw = raw * std + mean
                    decoded = tokens_to_images(
                        codec, raw, args.orbits_per_token, fft_permutation
                    )
                save_image(
                    decoded.clamp(0, 1),
                    os.path.join(args.output_dir, f"preview_{step:07d}.png"),
                    nrow=8,
                )
                model.train()

    with open(os.path.join(args.output_dir, "history.json"), "w") as handle:
        json.dump(
            {
                "history": history,
                "mean": mean,
                "std": std,
                "spectral_scale_exponent": args.spectral_scale_exponent,
                "compact_global_rms": (
                    float(compact_global_rms)
                    if compact_global_rms is not None
                    else None
                ),
                "compact_layout_metrics": compact_layout_metrics,
            },
            handle,
            indent=2,
        )
    torch.save(
        {
            "model": model.state_dict(),
            "compact_orbit_permutation": (
                fft_permutation.detach().cpu() if fft_permutation is not None else None
            ),
            "compact_layout_orbit": (
                compact_layout_orbit.detach().cpu()
                if compact_layout_orbit is not None
                else None
            ),
            "compact_layout_component": (
                compact_layout_component.detach().cpu()
                if compact_layout_component is not None
                else None
            ),
        },
        os.path.join(args.output_dir, "final.pt"),
    )
    print("done")


if __name__ == "__main__":
    main()
