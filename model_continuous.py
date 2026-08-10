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

from causal_transformer import apply_rope, build_rope_tables
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig
from factorized_polar_decoder import (
    FactorizedPolarConfig,
    FactorizedPolarDecoder,
    cartesian_to_polar_coordinates,
    polar_to_cartesian,
)
from frequency import FrequencyCodec, FrequencyCodecConfig, TOKEN_DIM
from frequency_grouping import (
    build_radial_sector_layout,
    pack_radial_sectors,
    unpack_radial_sectors,
)


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    max_seq_len: int = 515  # BOS + 514
    gradient_checkpointing: bool = False
    qk_norm: bool = True
    attention_rope: str = "frequency_2d"  # none | sequence | frequency_2d
    rope_base: float = 10000.0
    position_film: bool = False

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
    mode: str = "log_amp_gated_phase"
    fusion: str = "add"  # add | replace
    amplitude_transform: str = "match_decoder"
    amplitude_transform_parameter: float = 1.0
    log_epsilon: float = 1e-4

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryFeatureConfig:
    """Deterministic representation of completed Transformer history."""

    cartesian_mode: str = "centered"  # centered | phase_preserving
    mean_policy: str = "legacy"  # legacy | per_orbit | pooled_ordinary | self_only
    scale_policy: str = "legacy"  # legacy | centered_std | uncentered_rms

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
    grouping: str = "coefficient"  # coefficient | radial_sector
    group_size: int = 1

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
    factorized_polar: FactorizedPolarConfig = field(default_factory=FactorizedPolarConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "codec": self.codec.fingerprint(),
            "transformer": self.transformer.fingerprint(),
            "diffusion": self.diffusion.fingerprint(),
            "corruption": self.corruption.fingerprint(),
            "polar_history": self.polar_history.fingerprint(),
            "history_features": self.history_features.fingerprint(),
            "factorized_polar": self.factorized_polar.fingerprint(),
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
        qk_norm: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(width)
        self.position_film = PositionFiLM(width) if position_film else None
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=True)
            if qk_norm
            else None
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=True)
            if qk_norm
            else None
        )
        self.out_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        position_condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)  # type: ignore[operator]

        if rope is not None:
            cos, sin = rope
            # Attention slot j predicts frequency j. During cached decoding the
            # cache length is therefore also the target-frequency table offset.
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            if offset + n > cos.shape[0]:
                raise ValueError("RoPE tables are shorter than the attended sequence")
            q = apply_rope(q, cos[offset : offset + n], sin[offset : offset + n])
            k = apply_rope(k, cos[offset : offset + n], sin[offset : offset + n])

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Full-sequence training uses is_causal; cached decoding attends to all
        # keys already present (past + current), so is_causal is False when
        # the query length is 1 (or when past exists).
        is_causal = self.causal and kv_cache is None and n > 1
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
        qk_norm: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(
            width,
            num_heads,
            dropout,
            position_film,
            qk_norm=qk_norm,
            causal=causal,
        )
        self.ff = FeedForward(width, ff_mult, dropout, position_film)

    def forward(
        self,
        x: torch.Tensor,
        position_condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x, new_cache = self.attn(
            x,
            position_condition=position_condition,
            kv_cache=kv_cache,
            use_cache=use_cache,
            rope=rope,
        )
        x = self.ff(x, position_condition=position_condition)
        return x, new_cache


class ContinuousFFTDecoder(nn.Module):
    """Causal Transformer + diffusion decoder over continuous Fourier tokens."""

    def __init__(
        self,
        config: Optional[ContinuousModelConfig] = None,
        codec: Optional[FrequencyCodec] = None,
        factorized_amplitude_mean: Optional[torch.Tensor] = None,
        factorized_amplitude_std: Optional[torch.Tensor] = None,
        factorized_amplitude_source_rms: Optional[torch.Tensor] = None,
        history_amplitude_mean: Optional[torch.Tensor] = None,
        history_amplitude_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config or ContinuousModelConfig()
        if self.config.generation.grouping not in ("coefficient", "radial_sector"):
            raise ValueError("generation.grouping must be coefficient or radial_sector")
        if self.config.generation.group_size <= 0:
            raise ValueError("generation.group_size must be positive")
        if (
            self.config.generation.grouping == "coefficient"
            and self.config.generation.group_size != 1
        ):
            raise ValueError("coefficient grouping requires group_size=1")
        if (
            self.config.generation.grouping == "radial_sector"
            and not self.config.factorized_polar.enabled
        ):
            raise ValueError("radial_sector grouping currently requires factorized_polar")
        coefficients_per_token = (
            1
            if self.config.generation.grouping == "coefficient"
            else int(self.config.generation.group_size)
        )
        # The Transformer hidden state is the decoder's sole condition. Absolute
        # frequency identity enters once through the learned prediction slot.
        expected_target_condition_dim = 0
        expected_target_dim = TOKEN_DIM * coefficients_per_token
        if (
            self.config.diffusion.z_channels != self.config.transformer.width
            or self.config.diffusion.target_condition_dim
            != expected_target_condition_dim
            or self.config.diffusion.target_dim != expected_target_dim
        ):
            diff_fp = dict(self.config.diffusion.fingerprint())
            diff_fp["z_channels"] = self.config.transformer.width
            diff_fp["target_condition_dim"] = expected_target_condition_dim
            diff_fp["target_dim"] = expected_target_dim
            self.config.diffusion = DiffusionDecoderConfig(**diff_fp)

        self.codec = codec if codec is not None else FrequencyCodec(self.config.codec)
        if self.config.generation.grouping == "coefficient":
            group_indices = torch.arange(self.codec.seq_len_int)[:, None]
            group_mask = torch.ones_like(group_indices, dtype=torch.bool)
            group_radius = self.codec.radius_bin.clone().long()
        else:
            layout = build_radial_sector_layout(
                self.codec.radius_bin,
                coefficients_per_token,
            )
            group_indices = layout.indices
            group_mask = layout.mask
            group_radius = layout.radius
        self.coefficients_per_token = coefficients_per_token
        self.sequence_length = int(group_indices.shape[0])
        self.register_buffer("group_indices", group_indices, persistent=True)
        self.register_buffer("group_mask", group_mask, persistent=True)
        self.register_buffer("group_radius", group_radius, persistent=True)
        decoder_amplitude_mean = (
            torch.zeros(3, dtype=torch.float32)
            if factorized_amplitude_mean is None
            else factorized_amplitude_mean.detach().float().reshape(3)
        )
        decoder_amplitude_std = (
            torch.ones(3, dtype=torch.float32)
            if factorized_amplitude_std is None
            else factorized_amplitude_std.detach().float().reshape(3)
        )
        decoder_amplitude_source_rms = (
            torch.ones(self.codec.seq_len_int, 3, dtype=torch.float32)
            if factorized_amplitude_source_rms is None
            else factorized_amplitude_source_rms.detach().float().reshape(
                self.codec.seq_len_int, 3
            )
        )
        polar_amplitude_mean = (
            decoder_amplitude_mean
            if history_amplitude_mean is None
            else history_amplitude_mean.detach().float().reshape(3)
        )
        polar_amplitude_std = (
            decoder_amplitude_std
            if history_amplitude_std is None
            else history_amplitude_std.detach().float().reshape(3)
        )
        if (
            not bool(torch.isfinite(decoder_amplitude_mean).all())
            or not bool(torch.isfinite(decoder_amplitude_std).all())
            or bool((decoder_amplitude_std <= 0).any())
            or not bool(torch.isfinite(polar_amplitude_mean).all())
            or not bool(torch.isfinite(polar_amplitude_std).all())
            or bool((polar_amplitude_std <= 0).any())
        ):
            raise ValueError(
                "polar amplitude-coordinate statistics must be finite with std > 0"
            )
        # Keep history-coordinate statistics at model scope.  The trunk may use
        # polar history even when the decoder remains Cartesian.
        self.register_buffer(
            "polar_history_amplitude_coordinate_mean",
            polar_amplitude_mean,
            persistent=True,
        )
        self.register_buffer(
            "polar_history_amplitude_coordinate_std",
            polar_amplitude_std,
            persistent=True,
        )
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
        if self.config.diffusion.phase_aux_weight > 0.0 and (
            self.config.codec.normalization != "orbit_standardize"
            or self.config.codec.value_transform != "identity"
        ):
            raise ValueError(
                "Phase auxiliary requires identity orbit_standardize coordinates."
            )
        if self.config.history_features.cartesian_mode not in (
            "centered",
            "phase_preserving",
            "policy",
        ):
            raise ValueError(
                "history_features.cartesian_mode must be centered, phase_preserving, or policy"
            )
        if self.config.history_features.mean_policy not in (
            "legacy",
            "per_orbit",
            "pooled_ordinary",
            "self_only",
        ):
            raise ValueError("Unknown history mean policy")
        if self.config.history_features.scale_policy not in (
            "legacy",
            "centered_std",
            "uncentered_rms",
        ):
            raise ValueError("Unknown history scale policy")
        if self.config.history_features.cartesian_mode == "policy" and (
            self.config.history_features.mean_policy == "legacy"
            or self.config.history_features.scale_policy == "legacy"
        ):
            raise ValueError("Policy history mode requires explicit mean and scale policies")
        tcfg = self.config.transformer
        if tcfg.attention_rope not in ("none", "sequence", "frequency_2d"):
            raise ValueError(
                "transformer.attention_rope must be none, sequence, or frequency_2d"
            )
        if tcfg.rope_base <= 0.0:
            raise ValueError("transformer.rope_base must be positive")
        if self.config.polar_history.fusion not in ("add", "replace"):
            raise ValueError("polar_history.fusion must be add or replace")
        if not self.config.polar_history.enabled and self.config.polar_history.fusion != "add":
            raise ValueError("polar_history.fusion=replace requires polar history features")
        supported_polar_modes = (
            "log_amp_gated_phase",
            "standardized_log_amp_gated_phase",
            "physical_standardized_log_amp_phase",
        )
        if self.config.polar_history.enabled and self.config.polar_history.mode not in supported_polar_modes:
            raise ValueError(
                f"Unsupported polar_history.mode={self.config.polar_history.mode}. "
                f"Supported: {supported_polar_modes}."
            )
        if self.config.polar_history.amplitude_transform not in (
            "match_decoder",
            "log_eps",
            "log1p",
            "inverse_softplus",
            "power",
            "raw",
        ):
            raise ValueError("Unsupported polar-history amplitude transform")
        if self.config.polar_history.amplitude_transform_parameter <= 0.0:
            raise ValueError("polar-history amplitude transform parameter must be positive")
        if self.config.polar_history.log_epsilon <= 0.0:
            raise ValueError("polar-history log epsilon must be positive")
        if (
            self.config.polar_history.mode
            == "physical_standardized_log_amp_phase"
            and self.config.factorized_polar.coordinate_mode
            != "physical_standardized"
        ):
            raise ValueError(
                "physical standardized polar history requires the matching "
                "factorized coordinate mode"
            )
        self.width = tcfg.width
        coefficient_input_dim = (
            9
            if self.config.polar_history.enabled
            and self.config.polar_history.fusion == "replace"
            else TOKEN_DIM
        )
        token_input_dim = coefficient_input_dim * self.coefficients_per_token
        self.token_proj = nn.Linear(token_input_dim, tcfg.width)
        self.polar_proj: Optional[nn.Linear]
        if self.config.polar_history.enabled and self.config.polar_history.fusion == "add":
            # Optional zero-initialized modules must not perturb shared-weight RNG.
            with torch.random.fork_rng(devices=[]):
                self.polar_proj = nn.Linear(
                    9 * self.coefficients_per_token, tcfg.width
                )
            # Zero-init so enabling polar is a soft residual at start.
            nn.init.zeros_(self.polar_proj.weight)
            nn.init.zeros_(self.polar_proj.bias)
        else:
            self.polar_proj = None
        self.bos = nn.Parameter(torch.zeros(1, 1, tcfg.width))
        nn.init.normal_(self.bos, std=0.02)
        if self.sequence_length > tcfg.max_seq_len:
            raise ValueError("transformer.max_seq_len is shorter than the codec sequence")
        self.slot_embed = nn.Embedding(self.sequence_length, tcfg.width)
        nn.init.normal_(self.slot_embed.weight, std=0.02)
        coefficient_coordinates = torch.stack(
            [self.codec.ky_signed, self.codec.kx_signed],
            dim=-1,
        )
        if self.coefficients_per_token == 1:
            signed_coordinates = coefficient_coordinates
        else:
            coordinate_mask = self.group_mask[..., None].to(coefficient_coordinates.dtype)
            signed_coordinates = (
                coefficient_coordinates[self.group_indices] * coordinate_mask
            ).sum(1) / coordinate_mask.sum(1).clamp_min(1)
        if tcfg.attention_rope == "frequency_2d":
            rope_coordinates = signed_coordinates.round().to(torch.int64)
        elif tcfg.attention_rope == "sequence":
            rope_coordinates = torch.arange(self.sequence_length, dtype=torch.int64)
        else:
            rope_coordinates = torch.empty(0, dtype=torch.int64)
        self.register_buffer(
            "_attention_rope_coordinates",
            rope_coordinates,
            persistent=False,
        )
        if tcfg.attention_rope == "none":
            rope_cos = torch.empty(0, dtype=torch.float32)
            rope_sin = torch.empty(0, dtype=torch.float32)
        else:
            rope_cos, rope_sin = build_rope_tables(
                rope_coordinates,
                tcfg.width // tcfg.num_heads,
                base=tcfg.rope_base,
            )
        # These buffers are deliberately fp32. _attention_rope_tables rebuilds
        # them from integer coordinates if a whole-module dtype cast changes them.
        self.register_buffer("_attention_rope_cos", rope_cos, persistent=False)
        self.register_buffer("_attention_rope_sin", rope_sin, persistent=False)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    tcfg.width,
                    tcfg.num_heads,
                    tcfg.ff_mult,
                    tcfg.dropout,
                    position_film=tcfg.position_film,
                    qk_norm=tcfg.qk_norm,
                )
                for _ in range(tcfg.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(tcfg.width)
        self.diffusion: Optional[DiffusionDecoder]
        self.factorized_decoder: Optional[FactorizedPolarDecoder]
        if self.config.factorized_polar.enabled:
            if self.config.codec.value_transform != "identity":
                raise ValueError("factorized_polar initially requires identity values")
            expected_normalization = (
                "global_standardize"
                if self.config.factorized_polar.coordinate_mode
                == "physical_standardized"
                else "global_ecs"
            )
            if self.config.codec.normalization != expected_normalization:
                raise ValueError(
                    "factorized_polar coordinate mode "
                f"{self.config.factorized_polar.coordinate_mode} requires "
                f"normalization={expected_normalization}"
            )
            if self.config.codec.coordinate_packing != "isometric":
                raise ValueError("factorized_polar initially requires isometric packing")
            self.diffusion = None
            self.factorized_decoder = FactorizedPolarDecoder(
                self.config.diffusion,
                self.config.factorized_polar,
                condition_width=tcfg.width,
                amplitude_coordinate_mean=decoder_amplitude_mean,
                amplitude_coordinate_std=decoder_amplitude_std,
                amplitude_source_rms=decoder_amplitude_source_rms,
                coefficients_per_token=self.coefficients_per_token,
            )
        else:
            self.diffusion = DiffusionDecoder(self.config.diffusion)
            self.factorized_decoder = None
        self.gradient_checkpointing = tcfg.gradient_checkpointing

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Polar-v3 checkpoints written before trunk/decoder decoupling stored
        # these moments only inside the factorized decoder.  Promote them to the
        # model-level history buffers when loading so their trunk features remain
        # exactly unchanged.
        defaults = {
            "polar_history_amplitude_coordinate_mean": (
                "factorized_decoder.amplitude_coordinate_mean"
            ),
            "polar_history_amplitude_coordinate_std": (
                "factorized_decoder.amplitude_coordinate_std"
            ),
        }
        for root_suffix, decoder_suffix in defaults.items():
            root_key = prefix + root_suffix
            if root_key not in state_dict:
                decoder_key = prefix + decoder_suffix
                state_dict[root_key] = state_dict.get(
                    decoder_key, getattr(self, root_suffix)
                )
        # Generation groups are deterministic products of the codec ordering
        # and generation config. Checkpoints written before grouping support do
        # not contain these buffers, so reconstruct them without weakening
        # strict loading for learned parameters.
        for suffix in ("group_indices", "group_mask", "group_radius"):
            key = prefix + suffix
            if key not in state_dict:
                state_dict[key] = getattr(self, suffix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Register codec buffers as part of this module tree for checkpointing.
        # The codec is already an nn.Module; assign directly.
        # (Already set above.)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _attention_rope_tables(
        self, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return fp32 attention RoPE tables on ``device``.

        The integer coordinate buffer survives ``model.bfloat16()`` exactly. If
        that operation casts the cached trigonometric tables, rebuild rather
        than promoting already-rounded bf16 values back to fp32.
        """
        tcfg = self.config.transformer
        if tcfg.attention_rope == "none":
            return None
        cos = self._attention_rope_cos
        sin = self._attention_rope_sin
        if cos.device != device or cos.dtype != torch.float32:
            cos, sin = build_rope_tables(
                self._attention_rope_coordinates.to(device=device),
                tcfg.width // tcfg.num_heads,
                base=tcfg.rope_base,
            )
            self._attention_rope_cos = cos
            self._attention_rope_sin = sin
        return cos, sin

    def _fuse_polar_features(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add optional polar history projection onto Cartesian token embeddings."""
        if self.polar_proj is None:
            return x
        polar = self._polar_history_features(tokens, positions)
        return x + self.polar_proj(polar.to(dtype=x.dtype))

    def _polar_history_features(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.polar_history.mode == "log_amp_gated_phase":
            return self.codec.polar_history_features(tokens, positions=positions)
        if self.config.polar_history.mode == "physical_standardized_log_amp_phase":
            scale = self.factorized_amplitude_scale(positions)[None]
            history_transform = self.config.polar_history.amplitude_transform
            if history_transform == "match_decoder":
                history_transform = self.config.factorized_polar.amplitude_transform
                history_parameter = (
                    self.config.factorized_polar.amplitude_transform_parameter
                )
                history_log_epsilon = self.config.factorized_polar.log_epsilon
            else:
                history_parameter = (
                    self.config.polar_history.amplitude_transform_parameter
                )
                history_log_epsilon = self.config.polar_history.log_epsilon
            coordinate, phase, _ = cartesian_to_polar_coordinates(
                tokens,
                scale,
                history_log_epsilon,
                self.polar_history_amplitude_coordinate_mean,
                self.polar_history_amplitude_coordinate_std,
                history_transform,
                history_parameter,
            )
            cosine = torch.cos(phase)
            sine = torch.sin(phase)
            return torch.stack(
                [
                    coordinate[..., 0],
                    cosine[..., 0],
                    sine[..., 0],
                    coordinate[..., 1],
                    cosine[..., 1],
                    sine[..., 1],
                    coordinate[..., 2],
                    cosine[..., 2],
                    sine[..., 2],
                ],
                dim=-1,
            )
        return self.codec.polar_history_features(
            tokens,
            positions=positions,
            log_epsilon=self.config.factorized_polar.log_epsilon,
            amplitude_coordinate_mean=(
                self.polar_history_amplitude_coordinate_mean
                if self.config.factorized_polar.amplitude_standardization != "none"
                else None
            ),
            amplitude_coordinate_std=(
                self.polar_history_amplitude_coordinate_std
                if self.config.factorized_polar.amplitude_standardization != "none"
                else None
            ),
        )

    def _history_cartesian_features(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.history_features.cartesian_mode == "centered":
            return tokens
        if self.config.history_features.cartesian_mode == "phase_preserving":
            return self.codec.phase_preserving_history_features(tokens, positions)
        return self.codec.history_cartesian_features(
            tokens,
            positions,
            mean_policy=self.config.history_features.mean_policy,
            scale_policy=self.config.history_features.scale_policy,
        )

    def pack_groups(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pack coefficient tokens into the configured atomic prediction units."""
        return pack_radial_sectors(tokens, self.group_indices, self.group_mask)

    def unpack_groups(self, groups: torch.Tensor) -> torch.Tensor:
        """Restore grouped coefficients to the codec's exact sequence order."""
        return unpack_radial_sectors(
            groups,
            self.group_indices,
            self.group_mask,
            self.codec.seq_len_int,
        )

    def _group_history_features(
        self,
        groups: torch.Tensor,
        group_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Represent all K completed coefficients without ordering them internally."""
        if groups.ndim != 4 or groups.shape[2:] != (
            self.coefficients_per_token,
            TOKEN_DIM,
        ):
            raise ValueError("groups must have shape [B,G,K,6]")
        batch, length = groups.shape[:2]
        if group_positions is None:
            group_positions = torch.arange(length, device=groups.device)
        group_indices = self.group_indices[group_positions]
        group_mask = self.group_mask[group_positions]
        positions = group_indices.reshape(-1).to(groups.device)
        flat = groups.reshape(batch, length * self.coefficients_per_token, TOKEN_DIM)
        mask = group_mask.reshape(1, -1, 1).to(
            device=groups.device, dtype=groups.dtype
        )
        if (
            self.config.polar_history.enabled
            and self.config.polar_history.fusion == "replace"
        ):
            features = self._polar_history_features(flat, positions)
        else:
            features = self._history_cartesian_features(flat, positions)
        features = features * mask
        return features.reshape(batch, length, -1)

    def embed_groups(
        self,
        groups: torch.Tensor,
        include_bos: bool = True,
        group_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed completed groups shifted onto their next prediction slots."""
        batch, length = groups.shape[:2]
        if group_positions is None:
            group_positions = torch.arange(length, device=groups.device)
        features = self._group_history_features(groups, group_positions)
        x = self.token_proj(features.to(dtype=self.token_proj.weight.dtype))
        if self.polar_proj is not None:
            selected_indices = self.group_indices[group_positions]
            selected_mask = self.group_mask[group_positions]
            flat_positions = selected_indices.reshape(-1).to(groups.device)
            flat = groups.reshape(
                batch, length * self.coefficients_per_token, TOKEN_DIM
            )
            polar = self._polar_history_features(flat, flat_positions)
            polar = polar * selected_mask.reshape(1, -1, 1).to(
                device=groups.device, dtype=polar.dtype
            )
            x = x + self.polar_proj(polar.reshape(batch, length, -1).to(x.dtype))
        prediction_slots = group_positions + 1
        if prediction_slots.numel() and int(prediction_slots.max()) >= self.sequence_length:
            raise ValueError("Grouped history extends beyond the final prediction slot")
        x = x + self.slot_embed(prediction_slots)[None].to(x.dtype)
        if include_bos:
            bos = self.bos.to(dtype=x.dtype).expand(batch, -1, -1)
            bos = bos + self.slot_embed(
                torch.zeros(1, device=groups.device, dtype=torch.long)
            )[None].to(x.dtype)
            x = torch.cat([bos, x], dim=1)
        return x

    def prediction_slot_condition(
        self,
        positions: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the one shared identity embedding for prediction slots."""
        condition = self.slot_embed(positions).to(dtype=dtype)
        return condition[None, :, :].expand(batch_size, -1, -1)

    def diffusion_output_gain(self, positions: torch.Tensor) -> Optional[torch.Tensor]:
        if self.output_log_gain is None:
            return None
        rgb_gain = self.output_log_gain[positions].exp()
        return torch.cat([rgb_gain, rgb_gain], dim=-1)

    def factorized_amplitude_scale(self, positions: torch.Tensor) -> torch.Tensor:
        """Per-position RGB RMS amplitude in packed physical coordinates."""
        if self.config.factorized_polar.coordinate_mode == "physical_standardized":
            return torch.ones(
                positions.numel(),
                3,
                device=positions.device,
                dtype=self.codec.global_scale.dtype,
            )
        if self.codec.uses_orbit_statistics:
            rms_per_active_component = self.codec.orbit_uncentered_rms()[positions, :3]
            imag_active = self.codec.component_mask[positions, 3:]
            return rms_per_active_component * torch.sqrt(1.0 + imag_active)
        # global_ecs deliberately fits no per-orbit normalization. Its radial
        # moment table still gives a coarse, shared scale for log coordinates.
        return self.codec.channel_amplitude_scale()[self.codec.radius_bin[positions]]

    def factorized_cartesian_target(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return the Cartesian coordinate used by the factorized decoder."""
        if self.config.factorized_polar.coordinate_mode == "physical_standardized":
            return tokens
        return self.codec.normalized_to_raw_at(tokens, positions)

    def _apply_phase_auxiliary(
        self,
        loss_out: Dict[str, torch.Tensor],
        batch: int,
        length: int,
    ) -> None:
        weight = self.config.diffusion.phase_aux_weight
        if weight <= 0.0:
            return
        predicted = loss_out.pop("predicted_x0_for_phase").float()
        target = loss_out.pop("target_x0_for_phase").float()
        positions = torch.arange(length, device=predicted.device).repeat(batch)
        positions = positions.repeat(self.config.diffusion.diffusion_batch_mul)

        scale = self.codec._orbit_normalization_scale()[positions].float()
        mean = self.codec._orbit_normalization_mean()[positions].float()
        predicted = predicted * scale + mean
        target = target * scale + mean

        pred_real, pred_imag = predicted[:, :3], predicted[:, 3:]
        target_real, target_imag = target[:, :3], target[:, 3:]
        eps = 1e-6
        pred_amp = (
            pred_real.square() + pred_imag.square() + eps * eps
        ).sqrt()
        target_amp = (
            target_real.square() + target_imag.square() + eps * eps
        ).sqrt()
        cosine = (
            pred_real * target_real + pred_imag * target_imag
        ) / (pred_amp * target_amp).clamp_min(eps)
        cosine = cosine.clamp(-1.0, 1.0)

        expected_amp = (
            math.sqrt(2.0)
            * self.codec.orbit_uncentered_rms()[positions, :3].float()
        ).clamp_min(eps)
        relative_amp = target_amp / expected_amp
        gate = relative_amp / (
            relative_amp + self.config.diffusion.phase_aux_gate
        )
        phase_per_token = (gate * (1.0 - cosine)).sum(dim=-1) / gate.sum(
            dim=-1
        ).clamp_min(eps)
        ordinary = ~self.codec.is_self_conjugate[positions]
        timestep_weights = loss_out["snr_weights"].float()
        phase_loss = (
            phase_per_token[ordinary] * timestep_weights[ordinary]
        ).mean()
        base_loss = loss_out["loss"]
        loss_out["_base_loss_component"] = base_loss
        loss_out["_phase_loss_component"] = phase_loss
        loss_out["base_loss"] = loss_out["loss"].detach()
        loss_out["phase_aux_loss"] = phase_loss.detach()
        loss_out["loss"] = base_loss + weight * phase_loss

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
        if positions is None:
            # Assume tokens correspond to the first t frequency positions.
            positions = torch.arange(t, device=device)

        if self.config.polar_history.enabled and self.config.polar_history.fusion == "replace":
            history_features = self._polar_history_features(tokens, positions)
            x = self.token_proj(history_features.to(dtype=dtype))
        else:
            cartesian = self._history_cartesian_features(tokens, positions)
            x = self.token_proj(cartesian.to(dtype=dtype))
            x = self._fuse_polar_features(x, tokens, positions)
        # A history coefficient x_i occupies prediction slot i+1 and predicts
        # x_{i+1}. Fixed ordering makes this one slot ID sufficient for both roles.
        prediction_slots = positions + 1
        if prediction_slots.numel() and int(prediction_slots.max()) >= self.codec.seq_len:
            raise ValueError("History extends beyond the final prediction slot")
        x = x + self.slot_embed(prediction_slots)[None, :, :].to(dtype=dtype)

        if include_bos:
            bos = self.bos.to(dtype=dtype).expand(b, -1, -1)
            bos_slot = torch.zeros(1, device=device, dtype=torch.long)
            bos = bos + self.slot_embed(bos_slot)[None, :, :].to(dtype=dtype)
            x = torch.cat([bos, x], dim=1)
        return x

    def forward_backbone(
        self,
        x: torch.Tensor,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        position_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        rope = self._attention_rope_tables(x.device)
        if self.config.transformer.position_film:
            if position_condition is None:
                past_length = 0
                if kv_caches is not None and kv_caches[0] is not None:
                    past_length = kv_caches[0][0].shape[2]
                positions = torch.arange(
                    past_length,
                    past_length + x.shape[1],
                    device=x.device,
                )
                position_condition = self.prediction_slot_condition(
                    positions,
                    batch_size=x.shape[0],
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
                        rope=rope,
                    )
                    return out

                if position_condition is None:
                    def _run_without_position(module, inp):
                        out, _ = module(
                            inp,
                            kv_cache=None,
                            use_cache=False,
                            rope=rope,
                        )
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
                    rope=rope,
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
        b, l, d = tokens.shape
        if l != self.codec.seq_len or d != TOKEN_DIM:
            raise ValueError(
                f"Expected tokens [B,{self.codec.seq_len},{TOKEN_DIM}], got {tuple(tokens.shape)}"
            )
        if self.config.generation.grouping == "radial_sector":
            return self._forward_grouped(
                tokens,
                corrupt=corrupt,
                training_progress=training_progress,
                history_override=history_override,
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
        h, _ = self.forward_backbone(x, use_cache=False)  # [B, L, width]
        # h[:, i] conditions token i.
        z = h
        positions = torch.arange(l, device=tokens.device)
        if self.factorized_decoder is not None:
            raw_target = self.factorized_cartesian_target(tokens, positions)
            slot_condition = self.prediction_slot_condition(
                positions, batch_size=b, dtype=z.dtype
            )
            loss_out = self.factorized_decoder.compute_loss(
                raw_target=raw_target,
                z=z,
                slot_condition=slot_condition,
                amplitude_scale=self.factorized_amplitude_scale(positions),
                is_self_conjugate=self.codec.is_self_conjugate[positions],
                radius_bin=self.codec.radius_bin,
            )
            loss_out["corruption_strength"] = corr_strength.detach()
            return loss_out

        assert self.diffusion is not None
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
        output_gain = self.diffusion_output_gain(positions)
        loss_out = self.diffusion.compute_loss(
            target=tokens,
            z=z,
            target_condition=None,
            component_mask=self.codec.component_mask,
            radius_bin=self.codec.radius_bin,
            radial_weights=radial_weights,
            covariance_metric=covariance_metric,
            component_metric=component_metric,
            output_gain=output_gain,
        )
        self._apply_phase_auxiliary(loss_out, b, l)
        loss_out["corruption_strength"] = corr_strength.detach()
        return loss_out

    def _forward_grouped(
        self,
        tokens: torch.Tensor,
        *,
        corrupt: bool,
        training_progress: float,
        history_override: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced block AR over radial sectors of K coefficients."""
        if self.factorized_decoder is None:
            raise RuntimeError("radial-sector grouping requires factorized decoder")
        if corrupt and self.training and self.config.corruption.history_corruption != "none":
            raise NotImplementedError("grouped history corruption is not implemented")
        batch = tokens.shape[0]
        grouped = self.pack_groups(tokens)
        history = grouped[:, :-1] if history_override is None else history_override
        if history.shape != grouped[:, :-1].shape:
            raise ValueError(
                "grouped history_override must have shape "
                f"{tuple(grouped[:, :-1].shape)}, got {tuple(history.shape)}"
            )
        hidden, _ = self.forward_backbone(
            self.embed_groups(history, include_bos=True), use_cache=False
        )
        group_positions = torch.arange(self.sequence_length, device=tokens.device)
        slot_condition = self.prediction_slot_condition(
            group_positions, batch_size=batch, dtype=hidden.dtype
        )
        coefficient_positions = self.group_indices.to(tokens.device)
        flat_positions = coefficient_positions.reshape(-1)
        raw_flat = self.factorized_cartesian_target(
            grouped.reshape(
                batch,
                self.sequence_length * self.coefficients_per_token,
                TOKEN_DIM,
            ),
            flat_positions,
        )
        raw_target = raw_flat.reshape_as(grouped) * self.group_mask.to(
            device=tokens.device, dtype=raw_flat.dtype
        )[None, :, :, None]
        amplitude_scale = self.factorized_amplitude_scale(flat_positions).reshape(
            self.sequence_length, self.coefficients_per_token, 3
        )
        self_conjugate = self.codec.is_self_conjugate[flat_positions].reshape(
            self.sequence_length, self.coefficients_per_token
        ) & self.group_mask
        loss_out = self.factorized_decoder.compute_loss(
            raw_target=raw_target,
            z=hidden,
            slot_condition=slot_condition,
            amplitude_scale=amplitude_scale,
            is_self_conjugate=self_conjugate,
            active_coefficient_mask=self.group_mask,
            coefficient_positions=coefficient_positions,
            radius_bin=self.group_radius,
        )
        loss_out["corruption_strength"] = torch.zeros(
            batch, device=tokens.device, dtype=tokens.dtype
        )
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
        if self.factorized_decoder is not None:
            raw_target = self.factorized_cartesian_target(tokens, positions)
            slot_condition = self.prediction_slot_condition(
                positions, batch_size=batch, dtype=hidden.dtype
            )
            log_amp, phase = self.factorized_decoder.predict_coordinates_deterministic(
                raw_target=raw_target,
                z=hidden,
                slot_condition=slot_condition,
                amplitude_scale=self.factorized_amplitude_scale(positions),
                is_self_conjugate=self.codec.is_self_conjugate[positions],
                timesteps=timesteps,
                noise=noise,
            )
            raw_prediction = polar_to_cartesian(
                log_amp.reshape(-1, 3),
                phase.reshape(-1, 3),
                self.factorized_amplitude_scale(positions)[None]
                .expand(batch, -1, -1)
                .reshape(-1, 3),
                self.codec.is_self_conjugate[positions][None]
                .expand(batch, -1)
                .reshape(-1),
                self.config.factorized_polar.log_epsilon,
                (
                    self.factorized_decoder.amplitude_coordinate_mean
                    if self.config.factorized_polar.amplitude_standardization != "none"
                    else None
                ),
                (
                    self.factorized_decoder.amplitude_coordinate_std
                    if self.config.factorized_polar.amplitude_standardization != "none"
                    else None
                ),
                self.config.factorized_polar.amplitude_transform,
                self.config.factorized_polar.amplitude_transform_parameter,
            ).reshape(batch, length, 6)
            if (
                self.config.factorized_polar.coordinate_mode
                == "physical_standardized"
            ):
                return raw_prediction * self.codec.component_mask[positions][None]
            return self.codec.raw_to_normalized_at(raw_prediction, positions)

        assert self.diffusion is not None
        output_gain = self.diffusion_output_gain(positions)
        return self.diffusion.predict_x0_deterministic(
            tokens,
            hidden,
            timesteps,
            noise,
            component_mask=self.codec.component_mask,
            output_gain=output_gain,
            target_condition=None,
        )

    # ------------------------------------------------------------------
    # Cached generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Run BOS through the backbone and return (z0, caches)."""
        bos = self.bos.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        slot = torch.zeros(1, device=device, dtype=torch.long)
        bos = bos + self.slot_embed(slot)[None, :, :].to(dtype=dtype)
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
        device = token.device
        dtype = self.token_proj.weight.dtype
        positions = torch.tensor([position], device=device, dtype=torch.long)
        tok = token.to(dtype=dtype)[:, None, :]
        if self.config.polar_history.enabled and self.config.polar_history.fusion == "replace":
            history_features = self._polar_history_features(tok, positions)
            x = self.token_proj(history_features.to(dtype=dtype))
        else:
            cartesian = self._history_cartesian_features(tok, positions)
            x = self.token_proj(cartesian)
            x = self._fuse_polar_features(x, tok, positions)
        prediction_slot = torch.tensor([position + 1], device=device)
        x = x + self.slot_embed(prediction_slot)[None, :, :].to(dtype=dtype)
        h, new_caches = self.forward_backbone(x, kv_caches=kv_caches, use_cache=True)
        return h[:, -1, :], new_caches  # type: ignore

    @torch.no_grad()
    def forward_group_step(
        self,
        group: torch.Tensor,
        group_position: int,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Consume one completed radial sector and condition the next sector."""
        expected = (self.coefficients_per_token, TOKEN_DIM)
        if group.ndim != 3 or group.shape[1:] != expected:
            raise ValueError(f"group must have shape [B,{expected[0]},{expected[1]}]")
        positions = torch.tensor([group_position], device=group.device)
        x = self.embed_groups(
            group[:, None],
            include_bos=False,
            group_positions=positions,
        )
        h, new_caches = self.forward_backbone(
            x, kv_caches=kv_caches, use_cache=True
        )
        return h[:, -1], new_caches  # type: ignore

    @torch.no_grad()
    def sample_token(
        self,
        z: torch.Tensor,
        position: int,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float,
        eta: float,
    ) -> torch.Tensor:
        """Sample one normalized Cartesian history token from either decoder."""
        device = z.device
        if self.factorized_decoder is not None:
            positions = torch.tensor([position], device=device, dtype=torch.long)
            slot = self.prediction_slot_condition(
                positions, batch_size=z.shape[0], dtype=z.dtype
            )[:, 0]
            log_amp, phase = self.factorized_decoder.sample_coordinates(
                z=z,
                slot_condition=slot,
                generator=generator,
                steps=steps,
                temperature=temperature,
                is_self_conjugate=(
                    self.codec.is_self_conjugate[positions].expand(z.shape[0])
                ),
                positions=positions.expand(z.shape[0]),
            )
            scale = self.factorized_amplitude_scale(positions).expand(z.shape[0], -1)
            is_self = self.codec.is_self_conjugate[positions].expand(z.shape[0])
            raw = polar_to_cartesian(
                log_amp,
                phase,
                scale,
                is_self,
                self.config.factorized_polar.log_epsilon,
                (
                    self.factorized_decoder.amplitude_coordinate_mean
                    if self.config.factorized_polar.amplitude_standardization != "none"
                    else None
                ),
                (
                    self.factorized_decoder.amplitude_coordinate_std
                    if self.config.factorized_polar.amplitude_standardization != "none"
                    else None
                ),
                self.config.factorized_polar.amplitude_transform,
                self.config.factorized_polar.amplitude_transform_parameter,
            )
            if (
                self.config.factorized_polar.coordinate_mode
                == "physical_standardized"
            ):
                mask = self.codec.component_mask[positions].to(raw)[0]
                return raw * mask
            return self.codec.raw_to_normalized_at(raw[:, None], positions)[:, 0]

        assert self.diffusion is not None
        output_gain = self.diffusion_output_gain(
            torch.tensor([position], device=device)
        )
        if output_gain is not None:
            output_gain = output_gain.expand(z.shape[0], -1)
        return self.diffusion.sample(
            z,
            target_condition=None,
            component_mask=self.codec.component_mask[position],
            generator=generator,
            num_inference_steps=steps,
            eta=eta,
            temperature=temperature,
            output_gain=output_gain,
        )

    @torch.no_grad()
    def sample_group(
        self,
        z: torch.Tensor,
        group_position: int,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float,
    ) -> torch.Tensor:
        """Jointly sample all active coefficients in one radial sector."""
        if self.factorized_decoder is None:
            raise RuntimeError("grouped sampling requires factorized decoder")
        device = z.device
        group_slot = torch.tensor([group_position], device=device)
        slot = self.prediction_slot_condition(
            group_slot, batch_size=z.shape[0], dtype=z.dtype
        )[:, 0]
        positions = self.group_indices[group_position].to(device)
        active = self.group_mask[group_position].to(device)
        self_conjugate = self.codec.is_self_conjugate[positions] & active
        log_amp, phase = self.factorized_decoder.sample_coordinates(
            z=z,
            slot_condition=slot,
            generator=generator,
            steps=steps,
            temperature=temperature,
            is_self_conjugate=self_conjugate[None].expand(z.shape[0], -1),
            positions=positions[None].expand(z.shape[0], -1),
            active_coefficient_mask=active[None].expand(z.shape[0], -1),
        )
        scale = self.factorized_amplitude_scale(positions)[None].expand(
            z.shape[0], -1, -1
        )
        raw = polar_to_cartesian(
            log_amp.reshape(-1, 3),
            phase.reshape(-1, 3),
            scale.reshape(-1, 3),
            self_conjugate[None].expand(z.shape[0], -1).reshape(-1),
            self.config.factorized_polar.log_epsilon,
            self.factorized_decoder.amplitude_coordinate_mean
            if self.config.factorized_polar.amplitude_standardization != "none"
            else None,
            self.factorized_decoder.amplitude_coordinate_std
            if self.config.factorized_polar.amplitude_standardization != "none"
            else None,
            self.config.factorized_polar.amplitude_transform,
            self.config.factorized_polar.amplitude_transform_parameter,
        ).reshape(z.shape[0], self.coefficients_per_token, TOKEN_DIM)
        if self.config.factorized_polar.coordinate_mode != "physical_standardized":
            raw = self.codec.raw_to_normalized_at(raw, positions)
        component_mask = self.codec.component_mask[positions] * active[:, None]
        return raw * component_mask.to(raw)[None]

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
        self.codec.assert_fitted()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        steps = num_inference_steps or self.config.generation.num_inference_steps
        temp = self.config.generation.temperature if temperature is None else temperature
        eta_v = self.config.generation.eta if eta is None else eta
        if self.config.generation.grouping == "radial_sector":
            return self._generate_grouped(
                batch_size=batch_size,
                generator=generator,
                steps=steps,
                temperature=temp,
                return_tokens=return_tokens,
                progress=progress,
                max_groups=max_tokens,
            )
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
            sample = self.sample_token(
                z=z,
                position=i,
                generator=generator,
                steps=steps,
                temperature=temp,
                eta=eta_v,
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
    def _generate_grouped(
        self,
        *,
        batch_size: int,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float,
        return_tokens: bool,
        progress: bool,
        max_groups: Optional[int],
    ) -> Dict[str, Any]:
        """Cached autoregressive rollout over 134 four-coefficient sectors."""
        import time

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        group_count = (
            self.sequence_length
            if max_groups is None
            else min(int(max_groups), self.sequence_length)
        )
        z, caches = self.init_cache(batch_size, device, dtype)
        generated: list[torch.Tensor] = []
        iterator = range(group_count)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="generate-groups")
        t_backbone = 0.0
        t_denoise = 0.0
        for group_position in iterator:
            start = time.perf_counter()
            group = self.sample_group(
                z,
                group_position,
                generator,
                steps,
                temperature,
            )
            t_denoise += time.perf_counter() - start
            generated.append(group)
            if group_position + 1 < group_count:
                start = time.perf_counter()
                z, caches = self.forward_group_step(
                    group, group_position, caches
                )
                t_backbone += time.perf_counter() - start

        groups = torch.stack(generated, dim=1)
        if group_count < self.sequence_length:
            padding = torch.zeros(
                batch_size,
                self.sequence_length - group_count,
                self.coefficients_per_token,
                TOKEN_DIM,
                device=device,
                dtype=groups.dtype,
            )
            groups = torch.cat([groups, padding], dim=1)
        token_seq = self.unpack_groups(groups)
        images = self.codec.decode(token_seq.float())
        out: Dict[str, Any] = {
            "images": images,
            "backbone_seconds": t_backbone,
            "denoise_seconds": t_denoise,
            "num_groups_sampled": group_count,
            "num_tokens_sampled": int(self.group_mask[:group_count].sum()),
        }
        if return_tokens:
            out["tokens"] = token_seq
            out["groups"] = groups
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
            sample = self.sample_token(
                z=z,
                position=i,
                generator=generator,
                steps=steps,
                temperature=temperature,
                eta=eta,
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
