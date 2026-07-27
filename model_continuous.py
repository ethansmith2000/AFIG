"""Continuous causal Transformer for AFIG with diffusion-loss head.

Teacher-forced training over normalized Fourier tokens, KV-cached
autoregressive generation, optional Gaussian history corruption, and
documented stubs for future CFG / chunk / radial-band grouping.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig
from frequency import FrequencyCodec, FrequencyCodecConfig, TOKEN_DIM


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    max_seq_len: int = 515  # BOS + 514
    gradient_checkpointing: bool = False

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorruptionConfig:
    history_corruption: str = "none"  # none | gaussian
    history_corruption_prob: float = 1.0
    history_noise_min: float = 0.0
    history_noise_max: float = 0.05
    history_noise_ramp_fraction: float = 0.2
    # TODO(corruption): masked history replacement with missingness embedding
    # TODO(corruption): rollout_mix — stopgrad model-sampled prefix replacement

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolarHistoryConfig:
    """Deterministic polar features fused into history token embeddings.

    Features are derived from denormalized (physical) Cartesian history and do
    not change the 6D Cartesian diffusion targets.
    """

    enabled: bool = False
    mode: str = "log_amp_gated_phase"  # reserved for future modes

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryFeatureConfig:
    """Deterministic representation of completed Transformer history."""

    cartesian_mode: str = "centered"  # centered | phase_preserving

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrequencyConditioningConfig:
    """Known-frequency conditioning shared by input, backbone, and decoder."""

    enabled: bool = False
    num_frequencies: int = 4
    max_frequency: float = 8.0
    input_addition: bool = True
    rms_normalize: bool = False
    transformer_film: bool = True
    diffusion_target_conditioning: bool = True
    backbone_position_mode: str = "legacy_hybrid"
    input_scale_init: float = 0.1
    # TODO(position): add 2D RoPE as a separate relative-geometry ablation.

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationConfig:
    num_inference_steps: int = 20
    eta: float = 0.0
    temperature: float = 1.0
    cfg_enabled: bool = False
    cfg_scale: float = 1.0
    # TODO(cfg): class-condition dropout in the Transformer for CFG.
    grouping: str = "coefficient"
    # TODO(grouping): FixedChunkGrouping — joint denoise over K coefficients
    # TODO(grouping): RadialBandGrouping — block-AR over integer-radius bands

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContinuousModelConfig:
    codec: FrequencyCodecConfig = field(default_factory=FrequencyCodecConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    diffusion: DiffusionDecoderConfig = field(default_factory=DiffusionDecoderConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    polar_history: PolarHistoryConfig = field(default_factory=PolarHistoryConfig)
    history_features: HistoryFeatureConfig = field(default_factory=HistoryFeatureConfig)
    frequency_conditioning: FrequencyConditioningConfig = field(
        default_factory=FrequencyConditioningConfig
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "codec": self.codec.fingerprint(),
            "transformer": self.transformer.fingerprint(),
            "diffusion": self.diffusion.fingerprint(),
            "corruption": self.corruption.fingerprint(),
            "polar_history": self.polar_history.fingerprint(),
            "history_features": self.history_features.fingerprint(),
            "frequency_conditioning": self.frequency_conditioning.fingerprint(),
            "generation": self.generation.fingerprint(),
        }


class PositionFiLM(nn.Module):
    """Zero-initialized position-dependent scale and shift."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        shift, scale = self.net(position).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        dropout: float = 0.0,
        position_film: bool = False,
    ):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(width)
        self.position_film = PositionFiLM(width) if position_film else None
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        position_condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, n, _ = x.shape
        h = self.norm(x)
        if self.position_film is not None:
            if position_condition is None:
                raise ValueError("position_condition is required when position FiLM is enabled")
            h = self.position_film(h, position_condition)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Full-sequence training uses is_causal; cached decoding attends to all
        # keys already present (past + current), so is_causal is False when
        # the query length is 1 (or when past exists).
        is_causal = kv_cache is None and n > 1
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(b, n, -1)
        out = x + self.out_proj(attn)
        new_cache = (k, v) if use_cache else None
        return out, new_cache


class FeedForward(nn.Module):
    def __init__(
        self,
        width: int,
        mult: int = 4,
        dropout: float = 0.0,
        position_film: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.position_film = PositionFiLM(width) if position_film else None
        self.net = nn.Sequential(
            nn.Linear(width, width * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mult, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        position_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        if self.position_film is not None:
            if position_condition is None:
                raise ValueError("position_condition is required when position FiLM is enabled")
            h = self.position_film(h, position_condition)
        return x + self.net(h)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        position_film: bool = False,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(width, num_heads, dropout, position_film)
        self.ff = FeedForward(width, ff_mult, dropout, position_film)

    def forward(
        self,
        x: torch.Tensor,
        position_condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x, new_cache = self.attn(
            x,
            position_condition=position_condition,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = self.ff(x, position_condition=position_condition)
        return x, new_cache


class FrequencyPositionEmbed(nn.Module):
    """Functional frequency features plus a learned per-orbit residual."""

    def __init__(
        self,
        width: int,
        max_seq_len: int,
        functional: bool = False,
        num_frequencies: int = 4,
        max_frequency: float = 8.0,
        mode: str = "legacy_hybrid",
        signed_coordinates: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        if max_frequency < 1.0:
            raise ValueError("max_frequency must be >= 1")
        if mode not in ("legacy_hybrid", "random_table", "sincos_table"):
            raise ValueError(f"Unknown position embedding mode: {mode}")
        self.functional = functional and mode == "legacy_hybrid"
        self.mode = mode
        self.seq_embed = nn.Embedding(max_seq_len, width)
        if mode == "sincos_table":
            if signed_coordinates is None:
                raise ValueError("sincos_table requires signed_coordinates")
            table = self._sincos_table(
                signed_coordinates,
                width=width,
                max_seq_len=max_seq_len,
            )
            with torch.no_grad():
                self.seq_embed.weight.copy_(table)
            self.register_buffer("frequency_bands", torch.empty(0), persistent=False)
            self.meta_mlp = None
        elif mode == "random_table":
            nn.init.normal_(self.seq_embed.weight, std=0.02)
            with torch.no_grad():
                self.seq_embed.weight[0].zero_()
            self.register_buffer("frequency_bands", torch.empty(0), persistent=False)
            self.meta_mlp = None
        elif functional:
            nn.init.normal_(self.seq_embed.weight, std=0.02)
            bands = torch.logspace(
                0.0,
                math.log2(max_frequency),
                num_frequencies,
                base=2.0,
            )
            self.register_buffer("frequency_bands", bands, persistent=False)
            # normalized kx, ky, radius, cos(angle), sin(angle), is_self,
            # plus sin/cos bands for the first three continuous coordinates.
            meta_dim = 6 + 2 * 3 * num_frequencies
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
        else:
            self.register_buffer("frequency_bands", torch.empty(0), persistent=False)
            meta_dim = 5
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )

    @staticmethod
    def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
        if dim % 2 != 0:
            raise ValueError("1D sin/cos dimension must be even")
        half = dim // 2
        omega = torch.arange(half, dtype=torch.float32, device=positions.device)
        omega = 1.0 / (10000.0 ** (omega / max(float(half), 1.0)))
        phase = positions.float()[:, None] * omega[None, :]
        return torch.cat([phase.sin(), phase.cos()], dim=-1)

    @classmethod
    def _sincos_table(
        cls,
        signed_coordinates: torch.Tensor,
        width: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        if width % 4 != 0:
            raise ValueError("2D sin/cos width must be divisible by 4")
        if signed_coordinates.ndim != 2 or signed_coordinates.shape[-1] != 2:
            raise ValueError("signed_coordinates must have shape [L,2]")
        if signed_coordinates.shape[0] + 1 > max_seq_len:
            raise ValueError("Position table is too short for signed coordinates")
        ky, kx = signed_coordinates.unbind(dim=-1)
        values = torch.cat(
            [
                cls._sincos_1d(ky, width // 2),
                cls._sincos_1d(kx, width // 2),
            ],
            dim=-1,
        )
        table = torch.zeros(max_seq_len, width, dtype=torch.float32)
        table[1 : values.shape[0] + 1] = values.cpu()
        return table

    def _features(self, meta: torch.Tensor) -> torch.Tensor:
        if not self.functional:
            return meta
        continuous = meta[..., :3]
        phase = math.pi * continuous[..., None] * self.frequency_bands
        fourier = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        return torch.cat([meta, fourier], dim=-1)

    def forward(
        self,
        seq_idx: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.seq_embed(seq_idx)
        if meta is not None and self.meta_mlp is not None:
            functional = self.meta_mlp(self._features(meta))
            # BOS has no frequency coordinate; its learned table entry is enough.
            functional = functional * (seq_idx != 0).unsqueeze(-1).to(functional.dtype)
            emb = emb + functional
        return emb


class ContinuousFFTDecoder(nn.Module):
    """Causal Transformer + diffusion decoder over continuous Fourier tokens."""

    def __init__(
        self,
        config: Optional[ContinuousModelConfig] = None,
        codec: Optional[FrequencyCodec] = None,
    ):
        super().__init__()
        self.config = config or ContinuousModelConfig()
        # Keep diffusion conditions aligned with the Transformer representation.
        expected_target_condition_dim = (
            self.config.transformer.width
            if (
                self.config.frequency_conditioning.enabled
                and self.config.frequency_conditioning.diffusion_target_conditioning
            )
            else 0
        )
        if (
            self.config.diffusion.z_channels != self.config.transformer.width
            or self.config.diffusion.target_condition_dim
            != expected_target_condition_dim
        ):
            diff_fp = dict(self.config.diffusion.fingerprint())
            diff_fp["z_channels"] = self.config.transformer.width
            diff_fp["target_condition_dim"] = expected_target_condition_dim
            self.config.diffusion = DiffusionDecoderConfig(**diff_fp)

        self.codec = codec if codec is not None else FrequencyCodec(self.config.codec)
        if self.config.diffusion.loss_metric == "orbit_covariance_power":
            if self.config.codec.normalization != "orbit_whiten":
                raise ValueError(
                    "orbit_covariance_power requires codec normalization='orbit_whiten'."
                )
            if self.config.codec.value_transform != "identity":
                raise ValueError(
                    "orbit_covariance_power initially requires value_transform='identity'."
                )
        if self.config.diffusion.loss_metric == "orbit_scale_power":
            if self.config.codec.normalization != "orbit_standardize":
                raise ValueError(
                    "orbit_scale_power requires codec normalization='orbit_standardize'."
                )
            if self.config.codec.value_transform != "identity":
                raise ValueError(
                    "orbit_scale_power initially requires value_transform='identity'."
                )
        if (
            self.config.diffusion.learned_output_gain
            and self.config.codec.normalization != "orbit_standardize"
        ):
            raise ValueError(
                "learned_output_gain currently requires orbit_standardize."
            )
        if self.config.diffusion.learned_output_gain:
            self.output_log_gain = nn.Parameter(
                torch.zeros(self.codec.seq_len, 3)
            )
        else:
            self.register_parameter("output_log_gain", None)
        if self.config.history_features.cartesian_mode not in (
            "centered",
            "phase_preserving",
        ):
            raise ValueError(
                "history_features.cartesian_mode must be centered or phase_preserving"
            )
        tcfg = self.config.transformer
        self.width = tcfg.width
        self.token_proj = nn.Linear(TOKEN_DIM, tcfg.width)
        self.polar_proj: Optional[nn.Linear]
        if self.config.polar_history.enabled:
            if self.config.polar_history.mode != "log_amp_gated_phase":
                raise ValueError(
                    f"Unsupported polar_history.mode={self.config.polar_history.mode}. "
                    "Supported: log_amp_gated_phase."
                )
            # Optional zero-initialized modules must not perturb shared-weight RNG.
            with torch.random.fork_rng(devices=[]):
                self.polar_proj = nn.Linear(9, tcfg.width)
            # Zero-init so enabling polar is a soft residual at start.
            nn.init.zeros_(self.polar_proj.weight)
            nn.init.zeros_(self.polar_proj.bias)
        else:
            self.polar_proj = None
        self.bos = nn.Parameter(torch.zeros(1, 1, tcfg.width))
        nn.init.normal_(self.bos, std=0.02)
        pcfg = self.config.frequency_conditioning
        if pcfg.backbone_position_mode not in (
            "legacy_hybrid",
            "random_table",
            "sincos_table",
            "none",
        ):
            raise ValueError(
                f"Unknown backbone_position_mode={pcfg.backbone_position_mode}"
            )
        signed_coordinates = torch.stack(
            [self.codec.ky_signed, self.codec.kx_signed],
            dim=-1,
        )
        self.pos_embed = FrequencyPositionEmbed(
            tcfg.width,
            tcfg.max_seq_len,
            functional=pcfg.enabled,
            num_frequencies=pcfg.num_frequencies,
            max_frequency=pcfg.max_frequency,
            mode="legacy_hybrid",
        )
        backbone_mode = (
            "legacy_hybrid"
            if pcfg.backbone_position_mode == "none"
            else pcfg.backbone_position_mode
        )
        with torch.random.fork_rng(devices=[]):
            self.backbone_pos_embed = FrequencyPositionEmbed(
                tcfg.width,
                tcfg.max_seq_len,
                functional=pcfg.enabled,
                num_frequencies=pcfg.num_frequencies,
                max_frequency=pcfg.max_frequency,
                mode=backbone_mode,
                signed_coordinates=signed_coordinates,
            )
        self.input_position_scale = nn.Parameter(
            torch.tensor(float(pcfg.input_scale_init))
        )
        self.position_norm = (
            nn.RMSNorm(tcfg.width, elementwise_affine=False)
            if pcfg.enabled and pcfg.rms_normalize
            else nn.Identity()
        )
        use_position_film = pcfg.enabled and pcfg.transformer_film
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    tcfg.width,
                    tcfg.num_heads,
                    tcfg.ff_mult,
                    tcfg.dropout,
                    position_film=use_position_film,
                )
                for _ in range(tcfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(tcfg.width)
        self.diffusion = DiffusionDecoder(self.config.diffusion)
        self.gradient_checkpointing = tcfg.gradient_checkpointing

        # Register codec buffers as part of this module tree for checkpointing.
        # The codec is already an nn.Module; assign directly.
        # (Already set above.)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _fuse_polar_features(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add optional polar history projection onto Cartesian token embeddings."""
        if self.polar_proj is None:
            return x
        polar = self.codec.polar_history_features(tokens, positions=positions)
        return x + self.polar_proj(polar.to(dtype=x.dtype))

    def _history_cartesian_features(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.history_features.cartesian_mode == "centered":
            return tokens
        return self.codec.phase_preserving_history_features(tokens, positions)

    def _uses_backbone_position(self) -> bool:
        pcfg = self.config.frequency_conditioning
        return (
            (not pcfg.enabled or pcfg.input_addition)
            and pcfg.backbone_position_mode != "none"
        )

    def _backbone_position_embedding(
        self,
        seq_idx: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        position = self.position_norm(self.backbone_pos_embed(seq_idx, meta))
        if self.config.frequency_conditioning.backbone_position_mode in (
            "random_table",
            "sincos_table",
        ):
            position = position * self.input_position_scale.to(position.dtype)
        return position

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _token_meta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return [L, 5] metadata: kx_s, ky_s, radius, angle, is_self."""
        meta = torch.stack(
            [
                self.codec.kx_signed,
                self.codec.ky_signed,
                self.codec.radius,
                self.codec.angle,
                self.codec.is_self_conjugate.float(),
            ],
            dim=-1,
        )
        return meta.to(device=device, dtype=dtype)

    def _position_meta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return legacy metadata or normalized functional coordinates."""
        if not self.config.frequency_conditioning.enabled:
            return self._token_meta(device, dtype)
        half_w = self.config.codec.width / 2.0
        half_h = self.config.codec.height / 2.0
        max_radius = math.sqrt(half_w * half_w + half_h * half_h)
        meta = torch.stack(
            [
                self.codec.kx_signed / half_w,
                self.codec.ky_signed / half_h,
                self.codec.radius / max_radius,
                self.codec.angle.cos(),
                self.codec.angle.sin(),
                self.codec.is_self_conjugate.float(),
            ],
            dim=-1,
        )
        return meta.to(device=device, dtype=dtype)

    def _bos_meta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        meta_dim = 6 if self.config.frequency_conditioning.enabled else 5
        return torch.zeros(1, meta_dim, device=device, dtype=dtype)

    def target_position_condition(
        self,
        positions: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Embed known target orbit indices as [B,T,width] conditions."""
        if not self.config.frequency_conditioning.enabled:
            return None
        device = positions.device
        meta = self._position_meta(device, dtype)[positions]
        seq_idx = positions + 1  # index 0 is reserved for BOS
        condition = self.pos_embed(seq_idx, meta)
        condition = self.position_norm(condition)
        return condition[None, :, :].expand(batch_size, -1, -1)

    def diffusion_output_gain(self, positions: torch.Tensor) -> Optional[torch.Tensor]:
        if self.output_log_gain is None:
            return None
        rgb_gain = self.output_log_gain[positions].exp()
        return torch.cat([rgb_gain, rgb_gain], dim=-1)

    # ------------------------------------------------------------------
    # History corruption
    # ------------------------------------------------------------------
    def corrupt_history(
        self,
        tokens: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        training_progress: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optionally corrupt teacher-forced history tokens (not BOS).

        tokens: [B, L, 6] clean targets.
        Returns corrupted_tokens [B, L, 6], corruption_strength [B].
        """
        cfg = self.config.corruption
        b = tokens.shape[0]
        device = tokens.device
        strength = torch.zeros(b, device=device, dtype=tokens.dtype)

        if cfg.history_corruption == "none":
            return tokens, strength
        if cfg.history_corruption != "gaussian":
            # Reserved: masked, rollout_mix
            raise NotImplementedError(
                f"history_corruption={cfg.history_corruption} is stubbed. "
                "Supported: none, gaussian. "
                "TODO(corruption): masked / rollout_mix."
            )

        if not 0.0 <= training_progress <= 1.0:
            raise ValueError("training_progress must be in [0, 1].")
        if cfg.history_noise_ramp_fraction > 0:
            ramp = min(training_progress / cfg.history_noise_ramp_fraction, 1.0)
        else:
            ramp = 1.0
        noise_min = cfg.history_noise_min * ramp
        noise_max = cfg.history_noise_max * ramp

        # Isotropic normalized noise becomes position-colored after ZCA inversion.
        u = torch.rand(b, device=device, generator=generator)
        active = u < cfg.history_corruption_prob
        if bool(active.any()):
            s = torch.empty(b, device=device, dtype=tokens.dtype)
            s.uniform_(noise_min, noise_max, generator=generator)
            strength = torch.where(active, s, strength)
            noise = torch.randn(
                tokens.shape, device=device, dtype=tokens.dtype, generator=generator
            )
            mask = self.codec.component_mask[: tokens.shape[1]].to(
                device=device,
                dtype=tokens.dtype,
            )
            noise = noise * mask[None, :, :]
            tokens = tokens + strength[:, None, None] * noise
            tokens = tokens * mask[None, :, :]
        return tokens, strength

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    def embed_tokens(
        self,
        tokens: torch.Tensor,
        include_bos: bool = True,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed continuous tokens, optionally prepending BOS.

        tokens: [B, T, 6] where T is history length (may be < seq_len).
        For teacher forcing, pass corrupted history of length L (=514) to get
        inputs of length L+? Wait: teacher forcing uses [BOS, x0..x_{L-2}] length L,
        targeting [x0..x_{L-1}].

        This helper embeds a sequence of continuous tokens (no BOS) and
        optionally prepends BOS.
        """
        b, t, _ = tokens.shape
        device = tokens.device
        dtype = self.token_proj.weight.dtype
        meta_all = self._position_meta(device, dtype)
        if positions is None:
            # Assume tokens correspond to the first t frequency positions.
            positions = torch.arange(t, device=device)
            token_meta = meta_all[:t]
        else:
            token_meta = meta_all[positions]

        cartesian = self._history_cartesian_features(tokens, positions)
        x = self.token_proj(cartesian.to(dtype=dtype))
        x = self._fuse_polar_features(x, tokens, positions)

        # Sequence indices: if include_bos later, token positions start at 1.
        add_input_position = self._uses_backbone_position()
        if include_bos:
            bos = self.bos.to(dtype=dtype).expand(b, -1, -1)
            if add_input_position:
                seq_idx = positions + 1
                pos = self._backbone_position_embedding(seq_idx, token_meta)
                x = x + pos[None, :, :]
                bos_pos = self._backbone_position_embedding(
                    torch.zeros(1, device=device, dtype=torch.long),
                    self._bos_meta(device, dtype),
                )
                bos = bos + bos_pos[None, :, :]
            x = torch.cat([bos, x], dim=1)
        else:
            if add_input_position:
                seq_idx = (
                    positions
                    if positions is not None
                    else torch.arange(t, device=device)
                )
                # When used for step decode after BOS, positions are absolute seq indices.
                pos = self._backbone_position_embedding(seq_idx, token_meta)
                x = x + pos[None, :, :]
        return x

    def forward_backbone(
        self,
        x: torch.Tensor,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        position_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        pcfg = self.config.frequency_conditioning
        if pcfg.enabled and pcfg.transformer_film:
            if position_condition is None:
                past_length = 0
                if kv_caches is not None and kv_caches[0] is not None:
                    past_length = kv_caches[0][0].shape[2]
                positions = torch.arange(
                    past_length,
                    past_length + x.shape[1],
                    device=x.device,
                )
                position_condition = self.target_position_condition(
                    positions,
                    batch_size=1,
                    dtype=x.dtype,
                )
        else:
            position_condition = None

        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            cache_i = None if kv_caches is None else kv_caches[i]
            if self.gradient_checkpointing and self.training and not use_cache:
                # Checkpoint without cache.
                def _run(module, inp, pos):
                    out, _ = module(
                        inp,
                        position_condition=pos,
                        kv_cache=None,
                        use_cache=False,
                    )
                    return out

                if position_condition is None:
                    def _run_without_position(module, inp):
                        out, _ = module(inp, kv_cache=None, use_cache=False)
                        return out

                    x = checkpoint(_run_without_position, layer, x, use_reentrant=False)
                else:
                    x = checkpoint(
                        _run,
                        layer,
                        x,
                        position_condition,
                        use_reentrant=False,
                    )
                new_caches.append(None)  # type: ignore
            else:
                x, new_cache = layer(
                    x,
                    position_condition=position_condition,
                    kv_cache=cache_i,
                    use_cache=use_cache,
                )
                if use_cache:
                    new_caches.append(new_cache)  # type: ignore
        x = self.final_norm(x)
        return x, (new_caches if use_cache else None)

    def forward(
        self,
        tokens: torch.Tensor,
        corrupt: bool = True,
        training_progress: float = 1.0,
        history_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced training forward.

        tokens: clean normalized targets [B, L, 6], L = codec.seq_len.
        Returns dict with loss and diagnostics.
        """
        if self.config.generation.cfg_enabled:
            raise NotImplementedError(
                "CFG is disabled/stubbed. Set generation.cfg_enabled=False. "
                "TODO(cfg): class-condition dropout for CFG."
            )
        if self.config.generation.grouping != "coefficient":
            raise NotImplementedError(
                f"grouping={self.config.generation.grouping} is stubbed. "
                "Only 'coefficient' is implemented. "
                "TODO(grouping): FixedChunkGrouping / RadialBandGrouping."
            )

        b, l, d = tokens.shape
        if l != self.codec.seq_len or d != TOKEN_DIM:
            raise ValueError(
                f"Expected tokens [B,{self.codec.seq_len},{TOKEN_DIM}], got {tuple(tokens.shape)}"
            )

        history = (
            tokens[:, :-1, :]
            if history_override is None
            else history_override
        )
        if history.shape != tokens[:, :-1, :].shape:
            raise ValueError(
                "history_override must have shape "
                f"{tuple(tokens[:, :-1, :].shape)}, got {tuple(history.shape)}"
            )
        if corrupt and self.training:
            history, corr_strength = self.corrupt_history(
                history,
                training_progress=training_progress,
            )
        else:
            corr_strength = torch.zeros(b, device=tokens.device, dtype=tokens.dtype)

        # Inputs: [BOS, x0, ..., x_{L-2}] length L; targets: all L tokens.
        x = self.embed_tokens(history, include_bos=True)  # [B, L, width]
        target_condition = None
        if self.config.frequency_conditioning.enabled:
            target_positions = torch.arange(l, device=tokens.device)
            target_condition = self.target_position_condition(
                target_positions,
                batch_size=1,
                dtype=x.dtype,
            )
        h, _ = self.forward_backbone(
            x,
            use_cache=False,
            position_condition=target_condition,
        )  # [B, L, width]
        # h[:, i] conditions token i.
        z = h
        diffusion_target_condition = (
            target_condition.expand(b, -1, -1)
            if (
                target_condition is not None
                and self.config.diffusion.target_condition_dim > 0
            )
            else None
        )
        radial_weights = None
        if self.config.diffusion.radial_power_weighting:
            radial_weights = self.codec.radial_loss_weights(
                exponent=self.config.diffusion.radial_power_exponent
            )
        covariance_metric = None
        if self.config.diffusion.loss_metric == "orbit_covariance_power":
            covariance_metric = self.codec.orbit_covariance_power_metric(
                self.config.diffusion.orbit_covariance_exponent
            )
        component_metric = None
        if self.config.diffusion.loss_metric == "orbit_scale_power":
            component_metric = self.codec.orbit_scale_power_metric(
                self.config.diffusion.orbit_scale_exponent
            )
        output_gain = self.diffusion_output_gain(
            torch.arange(l, device=tokens.device)
        )
        loss_out = self.diffusion.compute_loss(
            target=tokens,
            z=z,
            target_condition=diffusion_target_condition,
            component_mask=self.codec.component_mask,
            radius_bin=self.codec.radius_bin,
            radial_weights=radial_weights,
            covariance_metric=covariance_metric,
            component_metric=component_metric,
            output_gain=output_gain,
        )
        loss_out["corruption_strength"] = corr_strength.detach()
        return loss_out

    @torch.no_grad()
    def predict_x0_diagnostics(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        history_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic teacher-forced x0 prediction for a held-out panel."""
        batch, length, _ = tokens.shape
        if length != self.codec.seq_len_int:
            raise ValueError(
                f"Expected {self.codec.seq_len_int} tokens, got {length}"
            )
        history = tokens[:, :-1] if history_override is None else history_override
        if history.shape != tokens[:, :-1].shape:
            raise ValueError(
                "history_override must have shape "
                f"{tuple(tokens[:, :-1].shape)}, got {tuple(history.shape)}"
            )
        hidden, _ = self.forward_backbone(
            self.embed_tokens(history, include_bos=True)
        )
        positions = torch.arange(length, device=tokens.device)
        target_condition = self.target_position_condition(
            positions,
            batch_size=batch,
            dtype=hidden.dtype,
        )
        if self.config.diffusion.target_condition_dim == 0:
            target_condition = None
        output_gain = self.diffusion_output_gain(positions)
        return self.diffusion.predict_x0_deterministic(
            tokens,
            hidden,
            timesteps,
            noise,
            component_mask=self.codec.component_mask,
            output_gain=output_gain,
            target_condition=target_condition,
        )

    # ------------------------------------------------------------------
    # Cached generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Run BOS through the backbone and return (z0, caches)."""
        bos = self.bos.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        if self._uses_backbone_position():
            bos_pos = self._backbone_position_embedding(
                torch.zeros(1, device=device, dtype=torch.long),
                self._bos_meta(device, dtype),
            )
            bos = bos + bos_pos[None, :, :]
        x = bos
        h, caches = self.forward_backbone(x, kv_caches=None, use_cache=True)
        return h[:, -1, :], caches

    @torch.no_grad()
    def forward_step(
        self,
        token: torch.Tensor,
        position: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Consume one normalized token at absolute frequency index `position`.

        token: [B, 6]
        Returns condition for the *next* token and updated caches.
        """
        b = token.shape[0]
        device = token.device
        dtype = self.token_proj.weight.dtype
        # Sequence index for this token embedding is position+1 (0 is BOS).
        seq_idx = torch.tensor([position + 1], device=device, dtype=torch.long)
        meta = self._position_meta(device, dtype)[position : position + 1]
        positions = torch.tensor([position], device=device, dtype=torch.long)
        tok = token.to(dtype=dtype)[:, None, :]
        cartesian = self._history_cartesian_features(tok, positions)
        x = self.token_proj(cartesian)
        x = self._fuse_polar_features(x, tok, positions)
        if self._uses_backbone_position():
            pos = self._backbone_position_embedding(seq_idx, meta)
            x = x + pos[None, :, :]
        h, new_caches = self.forward_backbone(x, kv_caches=kv_caches, use_cache=True)
        return h[:, -1, :], new_caches  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        eta: Optional[float] = None,
        return_tokens: bool = False,
        progress: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.config.generation.cfg_enabled:
            raise NotImplementedError(
                "CFG is disabled/stubbed. TODO(cfg): class-condition dropout."
            )
        if self.config.generation.grouping != "coefficient":
            raise NotImplementedError(
                "Only grouping='coefficient' is implemented. TODO(grouping)."
            )
        self.codec.assert_fitted()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        steps = num_inference_steps or self.config.generation.num_inference_steps
        temp = self.config.generation.temperature if temperature is None else temperature
        eta_v = self.config.generation.eta if eta is None else eta
        n_tokens = self.codec.seq_len if max_tokens is None else min(max_tokens, self.codec.seq_len)

        z, caches = self.init_cache(batch_size, device, dtype)
        tokens = []
        mask = self.codec.component_mask.to(device=device)

        iterator = range(n_tokens)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="generate")

        import time

        t_backbone = 0.0
        t_denoise = 0.0

        for i in iterator:
            t0 = time.perf_counter()
            target_condition = None
            if self.config.diffusion.target_condition_dim > 0:
                target_condition = self.target_position_condition(
                    torch.tensor([i], device=device),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                target_condition = target_condition[:, 0, :]
            output_gain = self.diffusion_output_gain(
                torch.tensor([i], device=device)
            )
            if output_gain is not None:
                output_gain = output_gain.expand(batch_size, -1)
            sample = self.diffusion.sample(
                z,
                target_condition=target_condition,
                component_mask=mask[i],
                generator=generator,
                num_inference_steps=steps,
                eta=eta_v,
                temperature=temp,
                output_gain=output_gain,
            )
            t_denoise += time.perf_counter() - t0
            tokens.append(sample)
            if i + 1 < n_tokens:
                t0 = time.perf_counter()
                z, caches = self.forward_step(sample, position=i, kv_caches=caches)
                t_backbone += time.perf_counter() - t0

        token_seq = torch.stack(tokens, dim=1)  # [B, n_tokens, 6]
        if n_tokens < self.codec.seq_len:
            pad = torch.zeros(
                batch_size,
                self.codec.seq_len - n_tokens,
                TOKEN_DIM,
                device=device,
                dtype=token_seq.dtype,
            )
            pad = pad * mask[None, n_tokens:, :]
            token_seq = torch.cat([token_seq, pad], dim=1)
        images = self.codec.decode(token_seq.float())
        out: Dict[str, Any] = {
            "images": images,
            "backbone_seconds": t_backbone,
            "denoise_seconds": t_denoise,
            "num_tokens_sampled": n_tokens,
        }
        if return_tokens:
            out["tokens"] = token_seq
        return out

    @torch.no_grad()
    def generate_uncached_prefix(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        eta: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Slow reference sampler recomputing the full prefix each step (for tests)."""
        self.codec.assert_fitted()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        steps = num_inference_steps or self.config.generation.num_inference_steps
        mask = self.codec.component_mask.to(device=device)
        tokens_so_far: List[torch.Tensor] = []

        n_tokens = self.codec.seq_len_int
        if max_tokens is not None:
            n_tokens = min(n_tokens, max_tokens)
        for i in range(n_tokens):
            if not tokens_so_far:
                z, _ = self.init_cache(batch_size, device, dtype)
            else:
                hist = torch.stack(tokens_so_far, dim=1)
                x = self.embed_tokens(hist, include_bos=True)
                h, _ = self.forward_backbone(x, use_cache=False)
                z = h[:, -1, :]
            target_condition = None
            if self.config.diffusion.target_condition_dim > 0:
                target_condition = self.target_position_condition(
                    torch.tensor([i], device=device),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                target_condition = target_condition[:, 0, :]
            output_gain = self.diffusion_output_gain(
                torch.tensor([i], device=device)
            )
            if output_gain is not None:
                output_gain = output_gain.expand(batch_size, -1)
            sample = self.diffusion.sample(
                z,
                target_condition=target_condition,
                component_mask=mask[i],
                generator=generator,
                num_inference_steps=steps,
                eta=eta,
                temperature=temperature,
                output_gain=output_gain,
            )
            tokens_so_far.append(sample)
        return torch.stack(tokens_so_far, dim=1)


# ---------------------------------------------------------------------------
# Future grouping contracts (documentation only)
# ---------------------------------------------------------------------------

class FixedChunkGrouping:
    """TODO(grouping): pack K consecutive radial coefficients into one AR token.

    Contract sketch:
      - group_size K, pad final group, mask padded dims in diffusion loss
      - joint denoiser input dim = 6K
      - AR steps = ceil(514 / K)
    """


class RadialBandGrouping:
    """TODO(grouping): block-autoregress over integer-radius bands.

    Contract sketch:
      - known lower-radius bands attend bidirectionally
      - mask tokens for all coefficients in the next band
      - sample the band jointly (shared denoiser or small band attention)
    """
