"""Canonical Hermitian Fourier codec for continuous AFIG tokens.

Uses an orthonormal FFT, 514 conjugacy-orbit representatives for 32x32,
integer Euclidean-radius bins, optional asinh value transform, and
dataset-level radial whitening.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


CODEC_VERSION = 2
NUM_CHANNELS = 3
TOKEN_DIM = 6  # RGB real + RGB imag


@dataclass(frozen=True)
class FrequencyCodecConfig:
    height: int = 32
    width: int = 32
    ordering: str = "radial"  # radial | square_spiral
    value_transform: str = "identity"  # identity | asinh
    normalization: str = "radial_whiten"  # radial_whiten | radial_standardize | orbit_whiten | orbit_standardize | global_ecs
    centering: str = "all"  # all | self_conjugate_std | self_conjugate_rms
    mean_policy: str = "legacy"  # legacy | per_orbit | pooled_ordinary | self_only
    scale_policy: str = "legacy"  # legacy | centered_std | uncentered_rms
    covariance_eps: float = 1e-5
    fft_norm: str = "ortho"
    # Partial whitening exponent for orbit_standardize.  Tokens are divided by
    # sigma**whiten_exponent and then by a single global scale, so values stay
    # O(1) at any exponent.
    #   1.0 = full per-frequency whitening (previous behaviour)
    #   0.0 = raw FFT with one global scale, preserving the natural 1/f
    #         eigenspectrum the way per-pixel normalization of an image does
    # Values in between interpolate.  Only meaningful for orbit_standardize.
    whiten_exponent: float = 1.0
    # ``isometric`` multiplies both Cartesian coordinates of every ordinary
    # Hermitian orbit by sqrt(2).  Together with the self-conjugate mask this is
    # an orthonormal real packing: iid pixel Gaussian noise becomes iid Gaussian
    # noise over the 3072 active token coordinates, and Euclidean energy is exact.
    coordinate_packing: str = "legacy"  # legacy | isometric
    # DCTdiff-style entropy-consistent scaling: use one robust bound derived from
    # the DC distribution for every frequency.  Only used by global_ecs.
    ecs_percentile: float = 98.25

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def _signed_freq(k: int, size: int) -> int:
    return k if k <= size // 2 else k - size


def _partner(ky: int, kx: int, height: int, width: int) -> Tuple[int, int]:
    return ((-ky) % height, (-kx) % width)


def _orbit_key(ky: int, kx: int, height: int, width: int) -> Tuple[Tuple[int, int], ...]:
    p = _partner(ky, kx, height, width)
    pts = sorted([(ky, kx), p])
    return tuple(pts)


def _angle(ky_s: float, kx_s: float) -> float:
    return math.atan2(ky_s, kx_s)


def build_orbit_table(
    height: int,
    width: int,
    ordering: str = "radial",
) -> Dict[str, torch.Tensor]:
    """Build canonical conjugacy-orbit representatives and metadata."""
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Only even height/width are supported in this codec.")

    seen = set()
    reps: List[Tuple[int, int]] = []
    self_conjugate: List[bool] = []

    for ky in range(height):
        for kx in range(width):
            key = _orbit_key(ky, kx, height, width)
            if key in seen:
                continue
            seen.add(key)
            # Deterministic representative: lexicographically smaller (ky, kx).
            rep = key[0]
            partner = key[-1]
            reps.append(rep)
            self_conjugate.append(rep == partner)

    expected = (height * width + 4) // 2 if height == width else None
    # For square even images: (N^2 - 4)/2 + 4 = N^2/2 + 2 = 514 for N=32.
    if height == width:
        expected = height * width // 2 + 2
        if len(reps) != expected:
            raise RuntimeError(f"Expected {expected} orbits, got {len(reps)}")

    ky = torch.tensor([r[0] for r in reps], dtype=torch.long)
    kx = torch.tensor([r[1] for r in reps], dtype=torch.long)
    is_self = torch.tensor(self_conjugate, dtype=torch.bool)

    ky_s = torch.tensor([_signed_freq(int(y), height) for y in ky], dtype=torch.float32)
    kx_s = torch.tensor([_signed_freq(int(x), width) for x in kx], dtype=torch.float32)
    radius = torch.sqrt(ky_s * ky_s + kx_s * kx_s)
    angle = torch.atan2(ky_s, kx_s)
    radius_bin = torch.floor(radius).to(torch.long)

    if ordering == "radial":
        # Sort by radius, then angle, then (ky, kx) for stability.
        order = sorted(
            range(len(reps)),
            key=lambda i: (
                float(radius[i]),
                float(angle[i]),
                int(ky[i]),
                int(kx[i]),
            ),
        )
    elif ordering == "square_spiral":
        order = _square_spiral_order(ky, kx, height, width, is_self)
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    order_t = torch.tensor(order, dtype=torch.long)
    ky = ky[order_t]
    kx = kx[order_t]
    is_self = is_self[order_t]
    ky_s = ky_s[order_t]
    kx_s = kx_s[order_t]
    radius = radius[order_t]
    angle = angle[order_t]
    radius_bin = radius_bin[order_t]

    # Component mask: [L, 6] — imag dims inactive for self-conjugate points.
    component_mask = torch.ones(len(reps), TOKEN_DIM, dtype=torch.float32)
    component_mask[is_self, 3:] = 0.0

    partner_ky = ((-ky) % height)
    partner_kx = ((-kx) % width)
    conjugate_multiplicity = torch.where(
        is_self,
        torch.ones_like(ky, dtype=torch.float32),
        torch.full_like(ky, 2, dtype=torch.float32),
    )

    return {
        "ky": ky,
        "kx": kx,
        "partner_ky": partner_ky,
        "partner_kx": partner_kx,
        "is_self_conjugate": is_self,
        "ky_signed": ky_s,
        "kx_signed": kx_s,
        "radius": radius,
        "angle": angle,
        "radius_bin": radius_bin,
        "component_mask": component_mask,
        "conjugate_multiplicity": conjugate_multiplicity,
        "seq_len": torch.tensor(len(reps), dtype=torch.long),
        "num_self_conjugate": torch.tensor(int(is_self.sum().item()), dtype=torch.long),
    }


def _square_spiral_order(
    ky: torch.Tensor,
    kx: torch.Tensor,
    height: int,
    width: int,
    is_self: torch.Tensor,
) -> List[int]:
    """Approximate legacy L-inf spiral using unshifted coords mapped to centered.

    Centers at (0,0) in signed frequency space and expands L-inf shells.
    Within a shell, walk in a deterministic angular/shell order.
    """
    ky_s = [_signed_freq(int(y), height) for y in ky]
    kx_s = [_signed_freq(int(x), width) for x in kx]
    linf = [max(abs(y), abs(x)) for y, x in zip(ky_s, kx_s)]

    # For each shell, order by angle then radius.
    indices = list(range(len(ky)))
    indices.sort(
        key=lambda i: (
            linf[i],
            _angle(ky_s[i], kx_s[i]),
            math.hypot(ky_s[i], kx_s[i]),
            int(ky[i]),
            int(kx[i]),
        )
    )
    return indices


class FrequencyCodec(nn.Module):
    """Encode/decode RGB images <-> continuous Fourier tokens with whitening."""

    def __init__(self, config: Optional[FrequencyCodecConfig] = None):
        super().__init__()
        self.config = config or FrequencyCodecConfig()
        if self.config.fft_norm != "ortho":
            raise ValueError("Only fft_norm='ortho' is supported.")
        if self.config.value_transform not in ("identity", "asinh"):
            raise ValueError(f"Unknown value_transform: {self.config.value_transform}")
        if self.config.normalization not in (
            "radial_whiten",
            "radial_standardize",
            "orbit_whiten",
            "orbit_standardize",
            "global_ecs",
        ):
            raise ValueError(f"Unknown normalization: {self.config.normalization}")
        if self.config.coordinate_packing not in ("legacy", "isometric"):
            raise ValueError(
                f"Unknown coordinate_packing: {self.config.coordinate_packing}"
            )
        if not 50.0 < self.config.ecs_percentile < 100.0:
            raise ValueError("ecs_percentile must be in (50, 100)")
        if self.config.normalization == "global_ecs" and (
            self.config.coordinate_packing != "isometric"
            or self.config.value_transform != "identity"
        ):
            raise ValueError(
                "global_ecs requires coordinate_packing='isometric' and "
                "value_transform='identity'"
            )
        if self.config.centering not in (
            "all",
            "self_conjugate_std",
            "self_conjugate_rms",
        ):
            raise ValueError(f"Unknown centering policy: {self.config.centering}")
        if self.config.mean_policy not in (
            "legacy",
            "per_orbit",
            "pooled_ordinary",
            "self_only",
        ):
            raise ValueError(f"Unknown mean policy: {self.config.mean_policy}")
        if self.config.scale_policy not in (
            "legacy",
            "centered_std",
            "uncentered_rms",
        ):
            raise ValueError(f"Unknown scale policy: {self.config.scale_policy}")
        if not 0.0 <= self.config.whiten_exponent <= 1.0:
            raise ValueError(
                f"whiten_exponent must be in [0,1], got {self.config.whiten_exponent}"
            )
        if (
            self.config.whiten_exponent != 1.0
            and self.config.normalization != "orbit_standardize"
        ):
            raise ValueError(
                "whiten_exponent < 1 is only implemented for orbit_standardize; it "
                "would be silently ignored by the other normalization paths."
            )
        explicit_policy = (
            self.config.mean_policy != "legacy"
            or self.config.scale_policy != "legacy"
        )
        if (
            (self.config.centering != "all" or explicit_policy)
            and self.config.normalization != "orbit_standardize"
        ):
            raise ValueError(
                "Configurable mean/scale policies require orbit_standardize."
            )
        if (
            (self.config.centering != "all" or explicit_policy)
            and self.config.value_transform != "identity"
        ):
            raise ValueError(
                "Configurable mean/scale policies require value_transform='identity'."
            )

        table = build_orbit_table(
            self.config.height,
            self.config.width,
            ordering=self.config.ordering,
        )
        for name, tensor in table.items():
            self.register_buffer(name, tensor, persistent=True)

        self.seq_len_int = int(table["seq_len"].item())
        self.max_radius_bin = int(table["radius_bin"].max().item())
        self.num_bins = self.max_radius_bin + 1

        # Whitening / transform state (filled by fit / load).
        self.register_buffer("is_fitted", torch.tensor(False), persistent=True)
        # Non-persistent for strict compatibility with model state dicts written
        # before Phase A.  export_state() carries both values explicitly.
        self.register_buffer(
            "global_pixel_mean", torch.tensor(0.0), persistent=False
        )
        self.register_buffer("global_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer(
            "bin_counts",
            torch.zeros(self.num_bins, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "bin_mean",
            torch.zeros(self.num_bins, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "bin_cov",
            torch.zeros(self.num_bins, TOKEN_DIM, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "bin_chol",
            torch.zeros(self.num_bins, TOKEN_DIM, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "bin_chol_inv",
            torch.zeros(self.num_bins, TOKEN_DIM, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "bin_std",
            torch.ones(self.num_bins, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "asinh_scale",
            torch.ones(self.num_bins, TOKEN_DIM),
            persistent=True,
        )
        orbit_eye = torch.eye(TOKEN_DIM).unsqueeze(0).repeat(self.seq_len_int, 1, 1)
        self.register_buffer(
            "orbit_counts",
            torch.zeros(self.seq_len_int, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "orbit_mean",
            torch.zeros(self.seq_len_int, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "orbit_cov",
            torch.zeros_like(orbit_eye),
            persistent=True,
        )
        self.register_buffer(
            "orbit_sqrt",
            orbit_eye.clone(),
            persistent=True,
        )
        self.register_buffer(
            "orbit_inv_sqrt",
            orbit_eye.clone(),
            persistent=True,
        )
        self.register_buffer(
            "orbit_std",
            torch.ones(self.seq_len_int, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "orbit_asinh_scale",
            torch.ones(self.seq_len_int, TOKEN_DIM),
            persistent=True,
        )
        self.register_buffer(
            "config_fingerprint",
            torch.tensor(0, dtype=torch.long),  # placeholder; metadata in state_dict extras
            persistent=True,
        )
        self._meta: Dict[str, Any] = {
            "codec_version": CODEC_VERSION,
            "config": self.config.fingerprint(),
        }
        self._orbit_metric_cache: Dict[Tuple[float, str], torch.Tensor] = {}
        self._orbit_scale_metric_cache: Dict[Tuple[float, str], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def seq_len(self) -> int:
        return self.seq_len_int

    @property
    def component_mask_tensor(self) -> torch.Tensor:
        return self.component_mask

    @property
    def uses_orbit_statistics(self) -> bool:
        return self.config.normalization in ("orbit_whiten", "orbit_standardize")

    def assert_fitted(self) -> None:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("FrequencyCodec statistics are not fitted/loaded.")

    def validate_compatible(self, meta: Dict[str, Any]) -> None:
        if meta.get("codec_version") != CODEC_VERSION:
            raise ValueError(
                f"Incompatible codec version: {meta.get('codec_version')} vs {CODEC_VERSION}"
            )
        cfg = meta.get("config", {})
        for key, value in self.config.fingerprint().items():
            # Mean/scale policies are deterministic post-fit transforms of the
            # same orbit moments, so one fitted statistics payload can serve all
            # policy ablations.
            # whiten_exponent belongs to the same category: it rescales the same
            # fitted moments, so one statistics payload serves every exponent.
            # The live exponent is still recorded in the AE checkpoint's codec
            # config and hashed into the layout fingerprint when non-default, so
            # generative checkpoints remain bound to the representation they were
            # trained on.
            if key in ("mean_policy", "scale_policy", "whiten_exponent"):
                continue
            # Version-2 codec payloads predate configurable centering and are
            # exactly equivalent to the new default.
            defaults = {
                "centering": "all",
                "coordinate_packing": "legacy",
                "ecs_percentile": 98.25,
            }
            observed = cfg.get(key, defaults.get(key))
            if observed != value:
                raise ValueError(
                    f"Incompatible codec config field {key}: {observed} vs {value}"
                )

    # ------------------------------------------------------------------
    # Core FFT encode / decode (pre-whitening)
    # ------------------------------------------------------------------
    def fft(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> complex [B, 3, H, W]."""
        if images.ndim != 4 or images.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")
        if images.shape[-2] != self.config.height or images.shape[-1] != self.config.width:
            raise ValueError(
                f"Expected spatial size {(self.config.height, self.config.width)}, "
                f"got {tuple(images.shape[-2:])}"
            )
        return torch.fft.fft2(images.float(), norm=self.config.fft_norm)

    def extract_tokens_raw(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Extract unnormalized Cartesian tokens [B, L, 6] from complex spectrum."""
        # spectrum: [B, 3, H, W] complex
        coeffs = spectrum[:, :, self.ky, self.kx]  # [B, 3, L]
        real = coeffs.real  # [B, 3, L]
        imag = coeffs.imag  # [B, 3, L]
        # Force imag=0 on self-conjugate points from GT (numerical noise).
        imag = imag * (~self.is_self_conjugate)[None, None, :]
        tokens = torch.cat([real, imag], dim=1).permute(0, 2, 1).contiguous()  # [B, L, 6]
        if self.config.coordinate_packing == "isometric":
            tokens = tokens * self.coordinate_scale(
                device=tokens.device, dtype=tokens.dtype
            )[None]
        return tokens

    def coordinate_scale(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Per-orbit scalar used by the real Cartesian packing, shape [L,1]."""
        if device is None:
            device = self.is_self_conjugate.device
        if dtype is None:
            dtype = torch.float32
        if self.config.coordinate_packing == "legacy":
            return torch.ones(self.seq_len_int, 1, device=device, dtype=dtype)
        ordinary = (~self.is_self_conjugate).to(device=device, dtype=dtype)
        return (1.0 + (math.sqrt(2.0) - 1.0) * ordinary)[:, None]

    def tokens_to_spectrum(self, tokens: torch.Tensor) -> torch.Tensor:
        """Place tokens into a full Hermitian spectrum [B, 3, H, W] complex."""
        b = tokens.shape[0]
        device = tokens.device
        dtype = tokens.dtype
        if self.config.coordinate_packing == "isometric":
            tokens = tokens / self.coordinate_scale(
                device=device, dtype=dtype
            )[None]
        real = tokens[..., :3].permute(0, 2, 1)  # [B, 3, L]
        imag = tokens[..., 3:].permute(0, 2, 1)  # [B, 3, L]
        imag = imag * (~self.is_self_conjugate).to(device=device)[None, None, :]

        spectrum = torch.zeros(
            b,
            NUM_CHANNELS,
            self.config.height,
            self.config.width,
            dtype=torch.complex64,
            device=device,
        )
        complex_vals = torch.complex(real.to(torch.float32), imag.to(torch.float32))
        spectrum[:, :, self.ky, self.kx] = complex_vals

        # Fill conjugate partners for non-self orbits.
        non_self = ~self.is_self_conjugate
        if bool(non_self.any()):
            pk = self.partner_ky[non_self]
            px = self.partner_kx[non_self]
            spectrum[:, :, pk, px] = torch.conj(complex_vals[:, :, non_self])

        # Self-conjugate: ensure imag is exactly 0.
        if bool(self.is_self_conjugate.any()):
            sky = self.ky[self.is_self_conjugate]
            skx = self.kx[self.is_self_conjugate]
            spectrum[:, :, sky, skx] = torch.complex(
                spectrum[:, :, sky, skx].real,
                torch.zeros_like(spectrum[:, :, sky, skx].real),
            )
        return spectrum

    def ifft(self, spectrum: torch.Tensor) -> torch.Tensor:
        images = torch.fft.ifft2(spectrum, norm=self.config.fft_norm).real
        return images

    def encode_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_tokens_raw(self.fft(images))

    def decode_raw(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ifft(self.tokens_to_spectrum(tokens))

    # ------------------------------------------------------------------
    # Value transform
    # ------------------------------------------------------------------
    def apply_value_transform(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.value_transform == "identity":
            return tokens
        scales = (
            self.orbit_asinh_scale
            if self.uses_orbit_statistics
            else self.asinh_scale[self.radius_bin]
        )
        return torch.asinh(tokens / scales.clamp_min(1e-8))

    def invert_value_transform(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.value_transform == "identity":
            return tokens
        scales = (
            self.orbit_asinh_scale
            if self.uses_orbit_statistics
            else self.asinh_scale[self.radius_bin]
        )
        return torch.sinh(tokens) * scales

    def _invert_value_transform_at(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Invert value transform for selected orbit positions [T]."""
        if self.config.value_transform == "identity":
            return tokens
        scales = (
            self.orbit_asinh_scale[positions]
            if self.uses_orbit_statistics
            else self.asinh_scale[self.radius_bin[positions]]
        )
        return torch.sinh(tokens) * scales

    # ------------------------------------------------------------------
    # Whitening
    # ------------------------------------------------------------------
    def orbit_uncentered_rms(self) -> torch.Tensor:
        """Per-orbit RGB complex RMS, shared across real/imaginary parts."""
        second_moment = torch.zeros(
            self.seq_len_int,
            NUM_CHANNELS,
            device=self.orbit_cov.device,
            dtype=self.orbit_cov.dtype,
        )
        count = self.orbit_counts.to(dtype=self.orbit_cov.dtype).clamp_min(1.0)
        covariance_to_population = ((count - 1.0) / count)[:, None]
        for channel in range(NUM_CHANNELS):
            real = (
                covariance_to_population[:, 0]
                * self.orbit_cov[:, channel, channel]
                + self.orbit_mean[:, channel].square()
            )
            imag = (
                covariance_to_population[:, 0]
                * self.orbit_cov[:, channel + NUM_CHANNELS, channel + NUM_CHANNELS]
                + self.orbit_mean[:, channel + NUM_CHANNELS].square()
            )
            imag_active = self.component_mask[:, channel + NUM_CHANNELS]
            second_moment[:, channel] = (real + imag) / (1.0 + imag_active)
        rgb = second_moment.clamp_min(self.config.covariance_eps).sqrt()
        scale = torch.cat([rgb, rgb], dim=-1)
        return scale * self.component_mask + (1.0 - self.component_mask)

    def effective_mean_policy(self) -> str:
        if self.config.mean_policy != "legacy":
            return self.config.mean_policy
        return "per_orbit" if self.config.centering == "all" else "self_only"

    def effective_scale_policy(self) -> str:
        if self.config.scale_policy != "legacy":
            return self.config.scale_policy
        return (
            "uncentered_rms"
            if self.config.centering == "self_conjugate_rms"
            else "centered_std"
        )

    def orbit_scale_for_policy(self, policy: str) -> torch.Tensor:
        if policy == "centered_std":
            return self.orbit_std
        if policy != "uncentered_rms":
            raise ValueError(f"Unknown orbit scale policy: {policy}")
        rms = self.orbit_uncentered_rms()
        self_mask = self.is_self_conjugate[:, None]
        return torch.where(self_mask, self.orbit_std, rms)

    def orbit_scaled_offset(
        self,
        mean_policy: str,
        scale_policy: str,
    ) -> torch.Tensor:
        """Return the affine offset applied after per-orbit scaling."""
        scale = self.orbit_scale_for_policy(scale_policy).clamp_min(1e-8)
        per_orbit = self.orbit_mean / scale
        self_mask = self.is_self_conjugate[:, None]
        if mean_policy == "per_orbit":
            offset = per_orbit
        elif mean_policy == "self_only":
            offset = per_orbit * self_mask.to(per_orbit.dtype)
        elif mean_policy == "pooled_ordinary":
            ordinary = ~self.is_self_conjugate
            pooled = per_orbit[ordinary].mean(dim=0, keepdim=True)
            offset = pooled.expand_as(per_orbit).clone()
            offset[self.is_self_conjugate] = per_orbit[self.is_self_conjugate]
        else:
            raise ValueError(f"Unknown orbit mean policy: {mean_policy}")
        return offset * self.component_mask

    def _orbit_normalization_mean(self) -> torch.Tensor:
        scale = self._orbit_normalization_scale()
        offset = self.orbit_scaled_offset(
            self.effective_mean_policy(),
            self.effective_scale_policy(),
        )
        return offset * scale

    def normalization_mean(self) -> torch.Tensor:
        """Affine mean subtracted before the configured normalization."""
        if self.config.normalization == "global_ecs":
            mean = torch.zeros(
                self.seq_len_int,
                TOKEN_DIM,
                device=self.ky.device,
                dtype=self.global_pixel_mean.dtype,
            )
            dc = (self.ky == 0) & (self.kx == 0)
            dc_value = self.global_pixel_mean * math.sqrt(
                self.config.height * self.config.width
            )
            mean[dc, :NUM_CHANNELS] = dc_value
            return mean
        if self.uses_orbit_statistics:
            return self._orbit_normalization_mean()
        return self.bin_mean[self.radius_bin]

    def _orbit_normalization_scale(self) -> torch.Tensor:
        return self.orbit_scale_for_policy(self.effective_scale_policy())

    def _orbit_partial_scale(self) -> torch.Tensor:
        """Divisor for orbit_standardize: sigma**alpha times one global scale.

        Kept separate from ``_orbit_normalization_scale`` because that value also
        feeds the mean computation, where the un-exponentiated policy scale must
        be used or the recovered mean is wrong.

        After dividing by sigma**alpha a component's variance is sigma**(2-2a),
        so the global scale that restores unit average variance is
        sqrt(mean(sigma**(2-2a))) over active components -- available in closed
        form from the fitted statistics, with no extra data pass.  Inactive
        components keep a divisor of exactly 1.
        """
        scale = self._orbit_normalization_scale().clamp_min(1e-8)
        exponent = float(self.config.whiten_exponent)
        if exponent == 1.0:
            return scale
        mask = self.component_mask
        partial = scale.pow(exponent)
        residual_variance = scale.pow(2.0 * (1.0 - exponent))
        active = mask.sum().clamp_min(1.0)
        global_scale = ((residual_variance * mask).sum() / active).sqrt().clamp_min(1e-8)
        partial = partial * global_scale
        # Inactive components must not be rescaled; they are forced to zero later.
        return partial * mask + (1.0 - mask)

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        self.assert_fitted()
        bins = self.radius_bin
        mean = self.normalization_mean()
        centered = tokens - mean
        mask = self.component_mask

        if self.config.normalization == "global_ecs":
            out = centered / self.global_scale.clamp_min(1e-8)
        elif self.config.normalization == "orbit_whiten":
            out = torch.einsum("lrc,blc->blr", self.orbit_inv_sqrt, centered)
        elif self.config.normalization == "orbit_standardize":
            out = centered / self._orbit_partial_scale().clamp_min(1e-8)
        elif self.config.normalization == "radial_standardize":
            std = self.bin_std[bins].clamp_min(1e-8)
            out = centered / std
        else:
            # Whiten: L^{-1} (x - mu) where cov ≈ L L^T
            Linv = self.bin_chol_inv[bins]  # [L, 6, 6]
            out = torch.einsum("lrc,blc->blr", Linv, centered)

        out = out * mask
        return out

    def denormalize(self, tokens: torch.Tensor) -> torch.Tensor:
        self.assert_fitted()
        bins = self.radius_bin
        mean = self.normalization_mean()
        mask = self.component_mask

        if self.config.normalization == "global_ecs":
            out = tokens * self.global_scale + mean
        elif self.config.normalization == "orbit_whiten":
            out = torch.einsum("lrc,blc->blr", self.orbit_sqrt, tokens) + mean
        elif self.config.normalization == "orbit_standardize":
            out = tokens * self._orbit_partial_scale() + mean
        elif self.config.normalization == "radial_standardize":
            std = self.bin_std[bins]
            out = tokens * std + mean
        else:
            L = self.bin_chol[bins]  # [L, 6, 6]
            out = torch.einsum("lrc,blc->blr", L, tokens) + mean

        # Zero inactive imag components.
        out = out * mask + mean * (1.0 - mask) * 0.0
        # More carefully: inactive dims should be exactly 0 in raw Cartesian space
        # after denorm+inverse transform. Force imag of self-conjugate to 0.
        out = out.clone()
        out[..., 3:] = out[..., 3:] * (~self.is_self_conjugate).to(out.dtype)[None, :, None]
        return out

    def denormalize_at(
        self, tokens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize tokens for selected orbit positions.

        tokens: [B, T, 6], positions: [T] long indices into the orbit table.
        Returns transformed-space Cartesian tokens [B, T, 6].
        """
        self.assert_fitted()
        bins = self.radius_bin[positions]
        mean = self.normalization_mean()[positions]
        mask = self.component_mask[positions]
        is_self = self.is_self_conjugate[positions]

        if self.config.normalization == "global_ecs":
            out = tokens * self.global_scale + mean
        elif self.config.normalization == "orbit_whiten":
            sqrt = self.orbit_sqrt[positions]
            out = torch.einsum("trc,btc->btr", sqrt, tokens) + mean
        elif self.config.normalization == "orbit_standardize":
            out = tokens * self._orbit_partial_scale()[positions] + mean
        elif self.config.normalization == "radial_standardize":
            std = self.bin_std[bins]
            out = tokens * std + mean
        else:
            chol = self.bin_chol[bins]
            out = torch.einsum("trc,btc->btr", chol, tokens) + mean

        out = out * mask
        out = out.clone()
        out[..., 3:] = out[..., 3:] * (~is_self).to(out.dtype)[None, :, None]
        return out

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Full encode: image -> normalized continuous tokens [B, L, 6]."""
        raw = self.encode_raw(images)
        transformed = self.apply_value_transform(raw)
        return self.normalize(transformed)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full decode: normalized tokens -> image [B, 3, H, W]."""
        denorm = self.denormalize(tokens)
        raw = self.invert_value_transform(denorm)
        # Ensure self-conjugate imag is zero after inverse transform.
        raw = raw.clone()
        raw[..., 3:] = raw[..., 3:] * (~self.is_self_conjugate).to(raw.dtype)[None, :, None]
        return self.decode_raw(raw)

    # ------------------------------------------------------------------
    # Radial power weights + polar history features
    # ------------------------------------------------------------------
    def bin_active_dims(self) -> torch.Tensor:
        """Return [num_bins] count of active Cartesian components per radius bin."""
        counts = torch.zeros(self.num_bins, device=self.radius_bin.device, dtype=torch.long)
        for b_idx in range(self.num_bins):
            positions = self.radius_bin == b_idx
            if bool(positions.any()):
                active = self.component_mask[positions].amax(dim=0) > 0
                counts[b_idx] = int(active.sum().item())
            else:
                counts[b_idx] = TOKEN_DIM
        return counts

    def bin_expected_centered_power(self) -> torch.Tensor:
        """Per-bin expected centered power: tr(Σ_b) / d_b.

        Returns [num_bins] float tensor. Sparse/unfitted bins fall back to 1.
        """
        self.assert_fitted()
        powers = torch.ones(self.num_bins, device=self.bin_cov.device, dtype=self.bin_cov.dtype)
        active_dims = self.bin_active_dims()
        for b_idx in range(self.num_bins):
            if int(self.bin_counts[b_idx].item()) < 2:
                continue
            positions = self.radius_bin == b_idx
            active = self.component_mask[positions].amax(dim=0) > 0
            d_b = int(active.sum().item())
            if d_b <= 0:
                continue
            cov = self.bin_cov[b_idx]
            # Trace over active submatrix only.
            tr = cov.diag()[active].sum().clamp_min(0.0)
            powers[b_idx] = tr / float(d_b)
        return powers

    def channel_amplitude_scale(self) -> torch.Tensor:
        """Per-bin per-RGB expected amplitude RMS from fitted covariance.

        Returns [num_bins, 3]. Uses sqrt(Var(re_c) + Var(im_c)), with imag var
        treated as 0 when that component is inactive in the bin.
        """
        self.assert_fitted()
        scales = torch.ones(
            self.num_bins, NUM_CHANNELS, device=self.bin_cov.device, dtype=self.bin_cov.dtype
        )
        for b_idx in range(self.num_bins):
            if int(self.bin_counts[b_idx].item()) < 2:
                continue
            positions = self.radius_bin == b_idx
            active = self.component_mask[positions].amax(dim=0) > 0  # [6]
            cov = self.bin_cov[b_idx]
            for c in range(NUM_CHANNELS):
                var_re = cov[c, c].clamp_min(0.0)
                if bool(active[c + 3].item()):
                    var_im = cov[c + 3, c + 3].clamp_min(0.0)
                else:
                    var_im = var_re.new_tensor(0.0)
                scales[b_idx, c] = torch.sqrt(var_re + var_im).clamp_min(1e-4)
        return scales

    def radial_loss_weights(self, exponent: float = 0.5) -> torch.Tensor:
        """Normalized per-orbit tempered radial-power weights [L], mean 1.

        ``exponent=1`` recovers expected-power weighting, which is extremely
        concentrated on CIFAR-10's DC token.  The default square root weights
        by expected amplitude instead, preserving natural low-frequency
        emphasis without allowing one coefficient to dominate the objective.
        """
        if not 0.0 <= exponent <= 1.0:
            raise ValueError(f"radial loss exponent must be in [0, 1], got {exponent}")
        powers = self.bin_expected_centered_power()  # [num_bins]
        weights = powers[self.radius_bin].clamp_min(1e-8).pow(exponent)
        return weights / weights.mean().clamp_min(1e-8)

    @torch.no_grad()
    def orbit_covariance_power_metric(self, exponent: float) -> torch.Tensor:
        """Globally normalize m_i Sigma_i^exponent without per-orbit rescaling."""
        self.assert_fitted()
        if self.config.normalization != "orbit_whiten":
            raise ValueError("Covariance-power metrics require orbit_whiten statistics.")
        if not 0.0 <= exponent <= 1.0:
            raise ValueError(f"Covariance exponent must be in [0,1], got {exponent}.")
        cache_key = (float(exponent), str(self.orbit_cov.device))
        cached = self._orbit_metric_cache.get(cache_key)
        if cached is not None:
            return cached

        metric = torch.zeros_like(self.orbit_cov)
        for position in range(self.seq_len_int):
            active = self.component_mask[position].bool()
            indices = active.nonzero(as_tuple=False).flatten()
            sub = self.orbit_cov[position][indices][:, indices]
            eigenvalues, eigenvectors = torch.linalg.eigh(sub.float())
            powered = (
                eigenvectors
                * eigenvalues.clamp_min(self.config.covariance_eps)
                .pow(exponent)
                .unsqueeze(0)
            ) @ eigenvectors.T
            metric[position][indices[:, None], indices[None, :]] = powered.to(
                metric.dtype
            )
        metric = metric * self.conjugate_multiplicity[:, None, None]
        global_scale = self.seq_len_int / torch.diagonal(
            metric, dim1=-2, dim2=-1
        ).sum().clamp_min(self.config.covariance_eps)
        metric = metric * global_scale
        self._orbit_metric_cache[cache_key] = metric
        return metric

    @torch.no_grad()
    def orbit_scale_power_metric(self, exponent: float) -> torch.Tensor:
        """Diagonal physical-error weights for orbit-standardized coordinates.

        For normalized error ``e`` and fitted diagonal scale ``s``, physical
        squared error is ``s^2 e^2``. Tempering by ``exponent`` gives
        ``s^(2*exponent)``. Conjugate multiplicity is included and one global
        scale makes the expected loss of unit-variance errors equal to one.
        """
        self.assert_fitted()
        if self.config.normalization != "orbit_standardize":
            raise ValueError("Scale-power metrics require orbit_standardize statistics.")
        if not 0.0 <= exponent <= 1.0:
            raise ValueError(f"Scale exponent must be in [0,1], got {exponent}.")
        cache_key = (float(exponent), str(self.orbit_std.device))
        cached = self._orbit_scale_metric_cache.get(cache_key)
        if cached is not None:
            return cached

        # Must use the same divisor ``normalize`` applies, otherwise physical
        # error is mis-stated whenever whiten_exponent < 1.  At exponent 0 the
        # divisor is a constant, so this metric becomes uniform -- correct,
        # because the normalized space then already carries the natural magnitude
        # hierarchy and plain MSE there is physical MSE.
        scale = self._orbit_partial_scale()
        metric = (
            scale.float().pow(2.0 * exponent)
            * self.conjugate_multiplicity[:, None].float()
            * self.component_mask.float()
        )
        global_scale = self.seq_len_int / metric.sum().clamp_min(
            self.config.covariance_eps
        )
        metric = metric * global_scale
        self._orbit_scale_metric_cache[cache_key] = metric
        return metric

    def history_cartesian_features(
        self,
        normalized_tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        *,
        mean_policy: str,
        scale_policy: str,
    ) -> torch.Tensor:
        """Build history Cartesian features independently of diffusion coordinates."""
        self.assert_fitted()
        if self.config.normalization != "orbit_standardize":
            raise ValueError(
                "Independent Cartesian history policies require orbit_standardize."
            )
        if self.config.value_transform != "identity":
            raise ValueError(
                "Independent Cartesian history policies require identity values."
            )
        if normalized_tokens.ndim != 3 or normalized_tokens.shape[-1] != TOKEN_DIM:
            raise ValueError(
                f"Expected normalized tokens [B,T,6], got {tuple(normalized_tokens.shape)}"
            )
        t = normalized_tokens.shape[1]
        device = normalized_tokens.device
        if positions is None:
            positions = torch.arange(t, device=device, dtype=torch.long)
        else:
            positions = positions.to(device=device, dtype=torch.long)
        if positions.numel() != t:
            raise ValueError(f"positions length {positions.numel()} != token length {t}")

        transformed = self.denormalize_at(normalized_tokens, positions)
        raw = self._invert_value_transform_at(transformed, positions)
        scale_all = self.orbit_scale_for_policy(scale_policy)
        offset_all = self.orbit_scaled_offset(mean_policy, scale_policy)
        scale = scale_all[positions].to(device=device, dtype=raw.dtype)
        offset = offset_all[positions].to(device=device, dtype=raw.dtype)
        features = raw / scale.clamp_min(1e-8)[None, :, :] - offset[None, :, :]
        mask = self.component_mask[positions].to(device=device, dtype=raw.dtype)
        return features * mask[None, :, :]

    def phase_preserving_history_features(
        self,
        normalized_tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility wrapper: self-only mean and centered standard deviation."""
        return self.history_cartesian_features(
            normalized_tokens,
            positions,
            mean_policy="self_only",
            scale_policy="centered_std",
        )

    def polar_history_features(
        self,
        normalized_tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic physical-space polar features for history conditioning.

        normalized_tokens: [B, T, 6] in whitened/standardized space.
        positions: optional [T] orbit indices (defaults to 0..T-1).

        Returns [B, T, 9] with per RGB channel:
          [log1p(a), g*cos(θ), g*sin(θ)], a = amp / expected_rms, g = a/(1+a).
        """
        self.assert_fitted()
        if normalized_tokens.ndim != 3 or normalized_tokens.shape[-1] != TOKEN_DIM:
            raise ValueError(
                f"Expected normalized tokens [B,T,6], got {tuple(normalized_tokens.shape)}"
            )
        b, t, _ = normalized_tokens.shape
        device = normalized_tokens.device
        dtype = normalized_tokens.dtype
        if positions is None:
            positions = torch.arange(t, device=device, dtype=torch.long)
        else:
            positions = positions.to(device=device, dtype=torch.long)
            if positions.numel() != t:
                raise ValueError(
                    f"positions length {positions.numel()} != token length {t}"
                )

        denorm = self.denormalize_at(normalized_tokens, positions)
        raw = self._invert_value_transform_at(denorm, positions)
        # Re-apply self-conjugate imag mask after inverse transform.
        is_self = self.is_self_conjugate[positions]
        raw = raw.clone()
        raw[..., 3:] = raw[..., 3:] * (~is_self).to(raw.dtype)[None, :, None]

        if self.uses_orbit_statistics:
            cov = self.orbit_cov[positions]
            mean = self.orbit_mean[positions]
            amp_scale = torch.stack(
                [
                    (
                        cov[:, c, c]
                        + mean[:, c].square()
                        + cov[:, c + 3, c + 3]
                        + mean[:, c + 3].square()
                    )
                    .clamp_min(0.0)
                    .sqrt()
                    .clamp_min(1e-4)
                    for c in range(NUM_CHANNELS)
                ],
                dim=-1,
            )
        else:
            bins = self.radius_bin[positions]
            amp_scale = self.channel_amplitude_scale()[bins]  # [T, 3]
        amp_scale = amp_scale.to(device=device, dtype=dtype).clamp_min(1e-4)

        feats = []
        for c in range(NUM_CHANNELS):
            re = raw[..., c]
            im = raw[..., c + 3]
            amp = torch.sqrt(re * re + im * im)
            a = amp / amp_scale[:, c].unsqueeze(0)
            gate = a / (1.0 + a)
            # Stable phase; gate→0 suppresses meaningless phase near zero amp.
            cos_t = torch.where(amp > 0, re / amp.clamp_min(1e-12), torch.ones_like(re))
            sin_t = torch.where(amp > 0, im / amp.clamp_min(1e-12), torch.zeros_like(im))
            feats.extend([torch.log1p(a), gate * cos_t, gate * sin_t])
        return torch.stack(feats, dim=-1)  # [B, T, 9]

    # ------------------------------------------------------------------
    # Statistics fitting
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, images_iter: Iterable[torch.Tensor], max_batches: Optional[int] = None) -> None:
        """Fit value-transform scales and radial whitening from training images.

        images_iter yields batches of shape [B, 3, H, W] in [0, 1] float.
        Two-pass: (1) asinh scales / accumulate moments on transformed tokens,
        (2) finalize Cholesky factors.
        """
        device = self.ky.device
        # Pass 1: accumulate raw second moment for asinh scales if needed,
        # and accumulate mean / second-moment for whitening.
        n_bins = self.num_bins
        count = torch.zeros(n_bins, dtype=torch.double, device=device)
        sum_x = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
        sum_xx = torch.zeros(n_bins, TOKEN_DIM, TOKEN_DIM, dtype=torch.double, device=device)
        sum_sq = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)

        # Optional preliminary pass for asinh scales on RAW tokens.
        if self.config.value_transform == "asinh":
            raw_count = torch.zeros(n_bins, dtype=torch.double, device=device)
            raw_sum_sq = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
            for bi, batch in enumerate(images_iter):
                if max_batches is not None and bi >= max_batches:
                    break
                batch = batch.to(device=device, dtype=torch.float32)
                raw = self.encode_raw(batch)  # [B, L, 6]
                bins = self.radius_bin
                for b_idx in range(n_bins):
                    sel = bins == b_idx
                    if not bool(sel.any()):
                        continue
                    vals = raw[:, sel, :].reshape(-1, TOKEN_DIM).double()
                    # Mask inactive imag dims
                    mask = self.component_mask[sel][0]  # same for all in bin? not necessarily
                    # Use per-position masks averaged carefully: take from first matching
                    m = self.component_mask[sel]  # [n_pos, 6]
                    # Expand: for each selected position, mask applies
                    vals = vals * m.repeat(raw.shape[0], 1).double()
                    raw_count[b_idx] += vals.shape[0]
                    raw_sum_sq[b_idx] += (vals * vals).sum(dim=0)
            # RMS scale per bin/component; inactive dims stay 1.
            for b_idx in range(n_bins):
                if raw_count[b_idx] > 0:
                    rms = torch.sqrt(raw_sum_sq[b_idx] / raw_count[b_idx].clamp_min(1.0))
                    rms = torch.clamp(rms, min=1e-4)
                    # For inactive components in this bin, if all positions are self-conjugate
                    positions = self.radius_bin == b_idx
                    active = self.component_mask[positions].max(dim=0).values
                    rms = torch.where(active > 0, rms, torch.ones_like(rms))
                    self.asinh_scale[b_idx] = rms.float()
            # Need to re-iterate; caller must provide a fresh iterator.
            raise RuntimeError(
                "asinh fit requires a restartable iterator. Call fit_from_loader instead."
            )

        for bi, batch in enumerate(images_iter):
            if max_batches is not None and bi >= max_batches:
                break
            batch = batch.to(device=device, dtype=torch.float32)
            raw = self.encode_raw(batch)
            transformed = self.apply_value_transform(raw)
            self._accumulate_moments(transformed, count, sum_x, sum_xx, sum_sq)

        self._finalize_moments(count, sum_x, sum_xx, sum_sq)

    @torch.no_grad()
    def _fit_orbit_from_loader(
        self,
        loader: Sequence[Any],
        max_batches: Optional[int],
        device: torch.device,
    ) -> None:
        length = self.seq_len_int
        raw_sum_sq = torch.zeros(length, TOKEN_DIM, dtype=torch.double, device=device)
        raw_count = 0
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            raw = self.encode_raw(images.to(device=device, dtype=torch.float32)).double()
            raw_sum_sq += raw.square().sum(dim=0)
            raw_count += raw.shape[0]
        if raw_count == 0:
            raise ValueError("No samples seen while fitting orbit statistics")

        mask = self.component_mask.double()
        if self.config.value_transform == "asinh":
            scales = (raw_sum_sq / raw_count).clamp_min(1e-12).sqrt()
            scales = scales * mask + (1.0 - mask)
        else:
            scales = torch.ones_like(raw_sum_sq)
        self.orbit_asinh_scale.copy_(scales.float())

        sums = torch.zeros(length, TOKEN_DIM, dtype=torch.double, device=device)
        crosses = torch.zeros(length, TOKEN_DIM, TOKEN_DIM, dtype=torch.double, device=device)
        count = 0
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = self.encode_raw(images.to(device=device, dtype=torch.float32)).double()
            if self.config.value_transform == "asinh":
                x = torch.asinh(x / scales.clamp_min(1e-12))
            sums += x.sum(dim=0)
            crosses += torch.einsum("bli,blj->lij", x, x)
            count += x.shape[0]
        if count == 0:
            raise ValueError("No samples seen while fitting orbit statistics")

        means = sums / count
        covariances = (
            crosses - count * torch.einsum("li,lj->lij", means, means)
        ) / max(count - 1, 1)
        eye = torch.eye(TOKEN_DIM, dtype=torch.double, device=device)
        covariance = torch.zeros(length, TOKEN_DIM, TOKEN_DIM, dtype=torch.double, device=device)
        sqrt = eye.unsqueeze(0).repeat(length, 1, 1)
        inv_sqrt = sqrt.clone()
        std = torch.ones(length, TOKEN_DIM, dtype=torch.double, device=device)

        for position in range(length):
            active = self.component_mask[position].bool()
            indices = active.nonzero(as_tuple=False).flatten()
            sub = covariances[position][indices][:, indices]
            eigenvalues, eigenvectors = torch.linalg.eigh(sub)
            eigenvalues = eigenvalues.clamp_min(self.config.covariance_eps)
            regularized = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T
            root = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
            inverse_root = (
                eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
            ) @ eigenvectors.T
            covariance[position][indices[:, None], indices[None, :]] = regularized
            sqrt[position][indices[:, None], indices[None, :]] = root
            inv_sqrt[position][indices[:, None], indices[None, :]] = inverse_root
            std[position, active] = torch.diagonal(regularized).sqrt()

        if self.config.normalization == "orbit_standardize":
            for channel in range(NUM_CHANNELS):
                real_variance = covariance[:, channel, channel]
                imag_variance = covariance[:, channel + NUM_CHANNELS, channel + NUM_CHANNELS]
                imag_active = self.component_mask[:, channel + NUM_CHANNELS].double()
                complex_variance = (
                    real_variance + imag_variance
                ) / (1.0 + imag_active)
                complex_scale = complex_variance.clamp_min(
                    self.config.covariance_eps
                ).sqrt()
                std[:, channel] = complex_scale
                std[:, channel + NUM_CHANNELS] = torch.where(
                    imag_active.bool(),
                    complex_scale,
                    torch.ones_like(complex_scale),
                )

        means = means * mask
        self.orbit_counts.fill_(count)
        self.orbit_mean.copy_(means.float())
        self.orbit_cov.copy_(covariance.float())
        self.orbit_sqrt.copy_(sqrt.float())
        self.orbit_inv_sqrt.copy_(inv_sqrt.float())
        self.orbit_std.copy_(std.float())
        self._orbit_metric_cache.clear()
        self._orbit_scale_metric_cache.clear()
        self.is_fitted.fill_(True)

    @torch.no_grad()
    def fit_from_loader(
        self,
        loader: Sequence[Any],
        max_batches: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Fit using a restartable DataLoader / list of batches."""
        if device is None:
            device = self.ky.device
        else:
            self.to(device)

        if self.config.normalization == "global_ecs":
            self._fit_global_ecs_from_loader(loader, max_batches, device)
            return

        if self.uses_orbit_statistics:
            self._fit_orbit_from_loader(loader, max_batches, device)
            return

        n_bins = self.num_bins

        if self.config.value_transform == "asinh":
            raw_count = torch.zeros(n_bins, dtype=torch.double, device=device)
            raw_sum_sq = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device=device, dtype=torch.float32)
                raw = self.encode_raw(images)
                bins = self.radius_bin
                B = raw.shape[0]
                for b_idx in range(n_bins):
                    sel = bins == b_idx
                    if not bool(sel.any()):
                        continue
                    vals = raw[:, sel, :]  # [B, npos, 6]
                    m = self.component_mask[sel]  # [npos, 6]
                    vals = vals * m[None, :, :]
                    flat = vals.reshape(-1, TOKEN_DIM).double()
                    raw_count[b_idx] += flat.shape[0]
                    raw_sum_sq[b_idx] += (flat * flat).sum(dim=0)
            for b_idx in range(n_bins):
                if raw_count[b_idx] > 0:
                    rms = torch.sqrt(raw_sum_sq[b_idx] / raw_count[b_idx].clamp_min(1.0))
                    rms = torch.clamp(rms, min=1e-4)
                    positions = self.radius_bin == b_idx
                    active = self.component_mask[positions].amax(dim=0)
                    rms = torch.where(active > 0, rms, torch.ones_like(rms))
                    self.asinh_scale[b_idx] = rms.float()

        count = torch.zeros(n_bins, dtype=torch.double, device=device)
        sum_x = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
        sum_xx = torch.zeros(n_bins, TOKEN_DIM, TOKEN_DIM, dtype=torch.double, device=device)
        sum_sq = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)

        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device=device, dtype=torch.float32)
            raw = self.encode_raw(images)
            transformed = self.apply_value_transform(raw)
            self._accumulate_moments(transformed, count, sum_x, sum_xx, sum_sq)

        self._finalize_moments(count, sum_x, sum_xx, sum_sq)

    @torch.no_grad()
    def _fit_global_ecs_from_loader(
        self,
        loader: Sequence[Any],
        max_batches: Optional[int],
        device: torch.device,
    ) -> None:
        """Fit one dataset mean and one robust DC-derived coefficient scale."""
        n_bins = self.num_bins
        count = torch.zeros(n_bins, dtype=torch.double, device=device)
        sum_x = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
        sum_xx = torch.zeros(
            n_bins, TOKEN_DIM, TOKEN_DIM, dtype=torch.double, device=device
        )
        sum_sq = torch.zeros(n_bins, TOKEN_DIM, dtype=torch.double, device=device)
        pixel_sum = torch.zeros((), dtype=torch.double, device=device)
        pixel_count = 0
        dc_chunks = []
        dc = ((self.ky == 0) & (self.kx == 0)).nonzero(
            as_tuple=False
        ).flatten()
        if dc.numel() != 1:
            raise RuntimeError(f"Expected one DC orbit, found {dc.numel()}")

        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device=device, dtype=torch.float32)
            raw = self.encode_raw(images)
            self._accumulate_moments(raw, count, sum_x, sum_xx, sum_sq)
            pixel_sum += images.double().sum()
            pixel_count += images.numel()
            dc_chunks.append(raw[:, dc.item(), :NUM_CHANNELS].double())

        if pixel_count == 0 or not dc_chunks:
            raise ValueError("No samples seen while fitting global ECS statistics")
        self._finalize_moments(count, sum_x, sum_xx, sum_sq)
        pixel_mean = pixel_sum / float(pixel_count)
        dc_values = torch.cat(dc_chunks, dim=0)
        dc_values = dc_values - pixel_mean * math.sqrt(
            self.config.height * self.config.width
        )
        quantile = self.config.ecs_percentile / 100.0
        upper = torch.quantile(dc_values, quantile, dim=0)
        lower = torch.quantile(dc_values, 1.0 - quantile, dim=0)
        scale = torch.maximum(upper.abs(), lower.abs()).max().clamp_min(1e-8)
        self.global_pixel_mean.copy_(pixel_mean.float())
        self.global_scale.copy_(scale.float())
        self.is_fitted.fill_(True)

    def _accumulate_moments(
        self,
        tokens: torch.Tensor,
        count: torch.Tensor,
        sum_x: torch.Tensor,
        sum_xx: torch.Tensor,
        sum_sq: torch.Tensor,
    ) -> None:
        bins = self.radius_bin
        B = tokens.shape[0]
        for b_idx in range(self.num_bins):
            sel = bins == b_idx
            if not bool(sel.any()):
                continue
            vals = tokens[:, sel, :]  # [B, npos, 6]
            m = self.component_mask[sel]  # [npos, 6]
            vals = vals * m[None, :, :]
            flat = vals.reshape(-1, TOKEN_DIM).double()
            count[b_idx] += flat.shape[0]
            sum_x[b_idx] += flat.sum(dim=0)
            sum_xx[b_idx] += flat.T @ flat
            sum_sq[b_idx] += (flat * flat).sum(dim=0)

    def _finalize_moments(
        self,
        count: torch.Tensor,
        sum_x: torch.Tensor,
        sum_xx: torch.Tensor,
        sum_sq: torch.Tensor,
    ) -> None:
        eps = self.config.covariance_eps
        for b_idx in range(self.num_bins):
            n = count[b_idx]
            self.bin_counts[b_idx] = int(n.item())
            if n < 2:
                # Leave identity whitening.
                self.bin_mean[b_idx].zero_()
                self.bin_std[b_idx] = 1.0
                eye = torch.eye(TOKEN_DIM, device=self.bin_chol.device)
                self.bin_cov[b_idx] = eye
                self.bin_chol[b_idx] = eye
                self.bin_chol_inv[b_idx] = eye
                continue

            mean = sum_x[b_idx] / n
            # Unbiased covariance: (sum_xx - n * mean mean^T) / (n - 1)
            cov = (sum_xx[b_idx] - n * torch.outer(mean, mean)) / (n - 1.0)
            # Determine active dims for this bin.
            positions = self.radius_bin == b_idx
            active = self.component_mask[positions].amax(dim=0) > 0  # [6]
            active_idx = active.nonzero(as_tuple=False).flatten()

            # Zero inactive mean/cov rows.
            mean = mean.clone()
            mean[~active] = 0.0
            cov = cov.clone()
            cov[~active, :] = 0.0
            cov[:, ~active] = 0.0

            # Diagonal std for standardize mode.
            var = torch.diag(cov).clamp_min(0.0)
            std = torch.sqrt(var + eps)
            std = torch.where(active, std, torch.ones_like(std))

            # Regularized Cholesky on active submatrix.
            eye_full = torch.eye(TOKEN_DIM, dtype=cov.dtype, device=cov.device)
            chol_full = eye_full.clone()
            chol_inv_full = eye_full.clone()
            if active_idx.numel() > 0:
                sub = cov[active_idx][:, active_idx]
                # Adaptive jitter.
                jitter = eps
                ok = False
                for _ in range(8):
                    try:
                        sub_reg = sub + jitter * torch.eye(
                            sub.shape[0], dtype=sub.dtype, device=sub.device
                        )
                        chol = torch.linalg.cholesky(sub_reg)
                        chol_inv = torch.linalg.solve_triangular(
                            chol,
                            torch.eye(sub.shape[0], dtype=chol.dtype, device=chol.device),
                            upper=False,
                        )
                        ok = True
                        break
                    except RuntimeError:
                        jitter *= 10.0
                if not ok:
                    # Fall back to diagonal.
                    diag = torch.sqrt(torch.diag(sub).clamp_min(eps))
                    chol = torch.diag(diag)
                    chol_inv = torch.diag(1.0 / diag)
                chol_full[active_idx[:, None], active_idx[None, :]] = chol
                chol_inv_full[active_idx[:, None], active_idx[None, :]] = chol_inv

            self.bin_mean[b_idx] = mean.float()
            self.bin_cov[b_idx] = cov.float()
            self.bin_std[b_idx] = std.float()
            self.bin_chol[b_idx] = chol_full.float()
            self.bin_chol_inv[b_idx] = chol_inv_full.float()

        self.is_fitted.fill_(True)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        self.assert_fitted()
        return {
            "codec_version": CODEC_VERSION,
            "config": self.config.fingerprint(),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "global_pixel_mean": self.global_pixel_mean.detach().cpu(),
            "global_scale": self.global_scale.detach().cpu(),
        }

    def load_exported(self, payload: Dict[str, Any], strict: bool = True) -> None:
        self.validate_compatible(payload)
        self.load_state_dict(payload["state_dict"], strict=strict)
        self.global_pixel_mean.copy_(
            torch.as_tensor(payload.get("global_pixel_mean", 0.0)).to(
                device=self.global_pixel_mean.device,
                dtype=self.global_pixel_mean.dtype,
            )
        )
        self.global_scale.copy_(
            torch.as_tensor(payload.get("global_scale", 1.0)).to(
                device=self.global_scale.device,
                dtype=self.global_scale.dtype,
            )
        )
        self._orbit_metric_cache.clear()
        self._orbit_scale_metric_cache.clear()
        self.is_fitted.fill_(True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def hermitian_violation(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Mean |F(k) - conj(F(-k))| over the spectrum."""
        h, w = self.config.height, self.config.width
        ky = torch.arange(h, device=spectrum.device)
        kx = torch.arange(w, device=spectrum.device)
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        partner = spectrum[:, :, (-grid_y) % h, (-grid_x) % w]
        return (spectrum - torch.conj(partner)).abs().mean()

    def position_metadata(self) -> Dict[str, torch.Tensor]:
        return {
            "ky_signed": self.ky_signed,
            "kx_signed": self.kx_signed,
            "radius": self.radius,
            "angle": self.angle,
            "radius_bin": self.radius_bin,
            "is_self_conjugate": self.is_self_conjugate.float(),
            "conjugate_multiplicity": self.conjugate_multiplicity,
            "component_mask": self.component_mask,
        }
