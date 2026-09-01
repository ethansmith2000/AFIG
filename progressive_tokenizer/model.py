"""Deterministic whole-image tokenizer with learned whole-image registers.

The encoder maps a CIFAR image to an ordered sequence of continuous registers.
The decoder uses spatial output queries to reconstruct an image from either the
complete sequence or an explicitly selected prefix.  The initial training gate
uses only complete sequences; prefix reconstruction is a separate objective in
the trainer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenizerConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_latents: int = 32
    latent_dim: int = 64
    width: int = 512
    num_heads: int = 8
    encoder_depth: int = 8
    pool_depth: int = 2
    decoder_depth: int = 8
    mlp_ratio: float = 4.0
    pool_type: str = "residual"
    qk_norm: str = "rms"
    cross_attention_bias: bool = False
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    rope_theta: float = 10_000.0
    variational: bool = False
    log_variance_floor: float = -8.0
    hard_log_variance_clamp: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_latents <= 0 or self.latent_dim <= 0:
            raise ValueError("num_latents and latent_dim must be positive")
        if self.width <= 0 or self.width % self.num_heads:
            raise ValueError("width must be positive and divisible by num_heads")
        if self.head_dim % 4:
            raise ValueError("head dimension must be divisible by four for 2-D RoPE")
        if self.encoder_depth <= 0 or self.pool_depth <= 0 or self.decoder_depth <= 0:
            raise ValueError("all model depths must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.pool_type not in {"residual", "cross_only", "register_tokens"}:
            raise ValueError(
                "pool_type must be residual, cross_only, or register_tokens"
            )
        if self.pool_type in {"cross_only", "register_tokens"} and self.pool_depth != 1:
            raise ValueError(
                f"{self.pool_type} pooling requires pool_depth=1"
            )
        if self.qk_norm not in {"rms", "l2_temperature"}:
            raise ValueError("qk_norm must be rms or l2_temperature")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size**2

    @property
    def head_dim(self) -> int:
        return self.width // self.num_heads

    def fingerprint(self) -> dict:
        return asdict(self)


class Rotary2D(nn.Module):
    """Fixed-grid 2-D rotary tables built and cached in float32."""

    def __init__(self, grid_size: int, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("2-D RoPE requires head_dim divisible by four")
        axis_dim = head_dim // 2
        inv_freq = theta ** (
            -torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
        )
        coords = torch.arange(grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        def axis_tables(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            angles = values.flatten()[:, None] * inv_freq[None, :]
            angles = torch.repeat_interleave(angles, 2, dim=-1)
            return angles.cos(), angles.sin()

        cos_y, sin_y = axis_tables(yy)
        cos_x, sin_x = axis_tables(xx)
        cos = torch.cat((cos_y, cos_x), dim=-1)[None, None]
        sin = torch.cat((sin_y, sin_x), dim=-1)[None, None]
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
        paired = x.reshape(*x.shape[:-1], -1, 2)
        first, second = paired.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.cos.shape[-2] or x.shape[-1] != self.cos.shape[-1]:
            raise ValueError(
                f"RoPE expected [..., {self.cos.shape[-2]}, {self.cos.shape[-1]}], "
                f"received {tuple(x.shape)}"
            )
        # The angle evaluation and multiply happen in fp32 even under bf16 autocast.
        rotated = x.float() * self.cos + self._rotate_pairs(x.float()) * self.sin
        return rotated.to(dtype=x.dtype)


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float, dropout: float):
        super().__init__()
        hidden = int(width * ratio)
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
        qk_norm: str,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.out_dropout = nn.Dropout(projection_dropout)
        self.attention_dropout = attention_dropout
        self.qk_norm = qk_norm
        if qk_norm == "rms":
            self.query_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=True
            )
            self.key_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=True
            )
            self.register_parameter("logit_scale", None)
        elif qk_norm == "l2_temperature":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
            self.logit_scale = nn.Parameter(
                torch.full((num_heads,), math.log(math.sqrt(self.head_dim)))
            )
        else:
            raise ValueError(f"unsupported QK normalization: {qk_norm}")

    def forward(self, x: torch.Tensor, rope: Optional[Rotary2D] = None) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q = self.query_norm(q)
        k = self.key_norm(k)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
        attention_scale = None
        if self.qk_norm == "l2_temperature":
            q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
            k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype)
            scale = self.logit_scale.exp().clamp(max=100.0).to(q.dtype)
            q = q * scale[None, :, None, None]
            attention_scale = 1.0
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=attention_scale,
        )
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.out_dropout(self.out(output))


class CrossAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
        qk_norm: str,
        bias: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q = nn.Linear(width, width, bias=bias)
        self.kv = nn.Linear(width, 2 * width, bias=bias)
        self.out = nn.Linear(width, width)
        self.out_dropout = nn.Dropout(projection_dropout)
        self.attention_dropout = attention_dropout
        self.qk_norm = qk_norm
        if qk_norm == "rms":
            self.query_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=True
            )
            self.key_norm = nn.RMSNorm(
                self.head_dim, eps=1e-6, elementwise_affine=True
            )
            self.register_parameter("logit_scale", None)
        elif qk_norm == "l2_temperature":
            self.query_norm = nn.Identity()
            self.key_norm = nn.Identity()
            self.logit_scale = nn.Parameter(
                torch.full((num_heads,), math.log(math.sqrt(self.head_dim)))
            )
        else:
            raise ValueError(f"unsupported QK normalization: {qk_norm}")

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_length, width = queries.shape
        memory_length = memory.shape[1]
        q = self.q(queries).reshape(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        kv = self.kv(memory).reshape(
            batch, memory_length, 2, self.num_heads, self.head_dim
        )
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q = self.query_norm(q)
        k = self.key_norm(k)
        attention_scale = None
        if self.qk_norm == "l2_temperature":
            q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
            k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype)
            scale = self.logit_scale.exp().clamp(max=100.0).to(q.dtype)
            q = q * scale[None, :, None, None]
            attention_scale = 1.0
        attention_mask = None
        if memory_mask is not None:
            if memory_mask.shape != (batch, memory_length):
                raise ValueError(
                    f"memory_mask must have shape {(batch, memory_length)}, "
                    f"received {tuple(memory_mask.shape)}"
                )
            attention_mask = memory_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=attention_scale,
        )
        output = output.transpose(1, 2).reshape(batch, query_length, width)
        return self.out_dropout(self.out(output))


def _norm(width: int) -> nn.LayerNorm:
    return nn.LayerNorm(width, elementwise_affine=False)


class EncoderBlock(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.attention_norm = _norm(config.width)
        self.attention = SelfAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
            config.qk_norm,
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(
            config.width, config.mlp_ratio, config.projection_dropout
        )

    def forward(self, x: torch.Tensor, rope: Optional[Rotary2D]) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope)
        return x + self.ffn(self.ffn_norm(x))


class RegisterAdapter(nn.Module):
    """Register-only refinement matched to one terminal cross-attention.

    The two bias choices match the projection parameters of ``CrossAttention``
    when cross-attention Q/K/V projections have no bias. Two identity-initialized
    head-coordinate scales match its affine RMS-QK parameters in the selected
    RMS-QK Stage-A configuration and give the adapter a non-spatial channel
    calibration rather than leaving unused parameter padding.
    """

    def __init__(self, width: int, head_dim: int):
        super().__init__()
        if width % head_dim:
            raise ValueError("register-adapter width must be divisible by head_dim")
        self.input_scale = nn.Parameter(torch.ones(head_dim))
        self.hidden_scale = nn.Parameter(torch.ones(head_dim))
        self.input = nn.Linear(width, 2 * width, bias=False)
        self.output = nn.Linear(2 * width, width, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        input_scale = self.input_scale.repeat(
            values.shape[-1] // self.input_scale.numel()
        )
        hidden = F.gelu(self.input(values * input_scale), approximate="tanh")
        hidden_scale = self.hidden_scale.repeat(
            hidden.shape[-1] // self.hidden_scale.numel()
        )
        return self.output(hidden * hidden_scale)


class PerceiverPoolBlock(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.cross_query_norm = _norm(config.width)
        self.cross_memory_norm = _norm(config.width)
        self.cross_attention = CrossAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
            config.qk_norm,
            config.cross_attention_bias,
        )
        self.self_norm = _norm(config.width)
        self.self_attention = SelfAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
            config.qk_norm,
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(
            config.width, config.mlp_ratio, config.projection_dropout
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        queries = queries + self.cross_attention(
            self.cross_query_norm(queries), self.cross_memory_norm(memory)
        )
        queries = queries + self.self_attention(self.self_norm(queries))
        return queries + self.ffn(self.ffn_norm(queries))


class DecoderBlock(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.self_norm = _norm(config.width)
        self.self_attention = SelfAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
            config.qk_norm,
        )
        self.cross_query_norm = _norm(config.width)
        self.cross_memory_norm = _norm(config.width)
        self.cross_attention = CrossAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
            config.qk_norm,
            config.cross_attention_bias,
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(
            config.width, config.mlp_ratio, config.projection_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        rope: Rotary2D,
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.self_attention(self.self_norm(x), rope)
        x = x + self.cross_attention(
            self.cross_query_norm(x), self.cross_memory_norm(memory), memory_mask
        )
        return x + self.ffn(self.ffn_norm(x))


class ProgressiveTokenizer(nn.Module):
    """Deterministic image <-> ordered continuous-register autoencoder."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.encoder_position = nn.Parameter(
            torch.empty(1, config.num_patches, config.width)
        )
        self.encoder_rope = Rotary2D(
            config.grid_size, config.head_dim, config.rope_theta
        )
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.encoder_depth)
        )
        self.encoder_norm = _norm(config.width)

        self.pool_queries = nn.Parameter(
            torch.empty(1, config.num_latents, config.width)
        )
        self.register_joint_block: Optional[EncoderBlock] = None
        self.register_adapter_norm: Optional[nn.LayerNorm] = None
        self.register_adapter: Optional[RegisterAdapter] = None
        if config.pool_type == "residual":
            self.pool_blocks = nn.ModuleList(
                PerceiverPoolBlock(config) for _ in range(config.pool_depth)
            )
            self.pool_query_norm = None
            self.pool_memory_norm = None
            self.pool_attention = None
        elif config.pool_type == "cross_only":
            self.pool_blocks = nn.ModuleList()
            self.pool_query_norm = _norm(config.width)
            self.pool_memory_norm = _norm(config.width)
            self.pool_attention = CrossAttention(
                config.width,
                config.num_heads,
                config.attention_dropout,
                config.projection_dropout,
                config.qk_norm,
                config.cross_attention_bias,
            )
        else:
            # A true register-token alternative to Perceiver pooling. Patches
            # and learned registers share a bidirectional block, after which a
            # register-only adapter replaces the baseline terminal cross read.
            # The launcher reallocates one patch-only encoder block here
            # (e7+j1), keeping the v8/v12 parameter count exact.
            self.pool_blocks = nn.ModuleList()
            self.pool_query_norm = None
            self.pool_memory_norm = None
            self.pool_attention = None
            self.register_joint_block = EncoderBlock(config)
            self.register_adapter_norm = _norm(config.width)
            self.register_adapter = RegisterAdapter(config.width, config.head_dim)
        self.latent_norm = _norm(config.width)
        projection_dim = (
            2 * config.latent_dim if config.variational else config.latent_dim
        )
        self.latent_projection = nn.Linear(config.width, projection_dim)

        self.latent_input = nn.Linear(config.latent_dim, config.width)
        self.latent_position = nn.Parameter(
            torch.empty(1, config.num_latents, config.width)
        )
        self.output_token = nn.Parameter(torch.empty(1, 1, config.width))
        self.output_position = nn.Parameter(
            torch.empty(1, config.num_patches, config.width)
        )
        self.decoder_rope = Rotary2D(
            config.grid_size, config.head_dim, config.rope_theta
        )
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.decoder_depth)
        )
        self.decoder_norm = _norm(config.width)
        self.patch_output = nn.Linear(config.width, config.patch_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.encoder_position, std=0.02)
        nn.init.trunc_normal_(self.pool_queries, std=0.02)
        nn.init.trunc_normal_(self.latent_position, std=0.02)
        nn.init.trunc_normal_(self.output_token, std=0.02)
        nn.init.trunc_normal_(self.output_position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                nn.init.trunc_normal_(module.weight, std=math.sqrt(1.0 / fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_distribution(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (mean, log-variance); log-variance is None when deterministic."""

        expected = (
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"received {tuple(images.shape)}"
            )
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        patches = patches + self.encoder_position
        for block in self.encoder_blocks:
            patches = block(patches, self.encoder_rope)
        patches = self.encoder_norm(patches)

        queries = self.pool_queries.expand(images.shape[0], -1, -1)
        if self.config.pool_type == "residual":
            for block in self.pool_blocks:
                queries = block(queries, patches)
        elif self.config.pool_type == "cross_only":
            if (
                self.pool_attention is None
                or self.pool_query_norm is None
                or self.pool_memory_norm is None
            ):
                raise RuntimeError("cross-only pooling modules were not constructed")
            queries = self.pool_attention(
                self.pool_query_norm(queries), self.pool_memory_norm(patches)
            )
        else:
            if (
                self.register_joint_block is None
                or self.register_adapter_norm is None
                or self.register_adapter is None
            ):
                raise RuntimeError("register-token pooling modules were not constructed")
            # The patch tokens already carry learned 2-D positions and have
            # passed through the 2-D-RoPE patch trunk. The mixed block uses no
            # shared RoPE because patches and registers inhabit different
            # geometries (2-D space versus a learned register/scale axis).
            joint = torch.cat((patches, queries), dim=1)
            joint = self.register_joint_block(joint, None)
            queries = joint[:, self.config.num_patches :]
            queries = queries + self.register_adapter(
                self.register_adapter_norm(queries)
            )
        projected = self.latent_projection(self.latent_norm(queries))
        if not self.config.variational:
            return projected, None
        mean, log_variance = projected.chunk(2, dim=-1)
        return mean, self._bound_log_variance(log_variance)

    def _bound_log_variance(self, log_variance: torch.Tensor) -> torch.Tensor:
        """Bound log-variance below without killing the gradient at the floor.

        A hard clamp is an absorbing trap: reconstruction always pushes sigma
        down, and once the pre-clamp value passes the floor the clamp emits
        zero gradient, so even the KL's restoring force is disconnected. The
        v5 "vae" arm collapsed exactly this way -- 99.97% of logvars sat at
        -8.0, sigma pinned at a constant 0.018, and the KL's sigma-term was a
        constant 3.500 (= 0.5*(e^-8 - 1 + 8)) contributing no gradient. That
        arm was therefore not variational: it was a deterministic encoder plus
        fixed 1.8% jitter.

        Softplus keeps the same floor but leaves the gradient alive, so the KL
        can still push sigma back up and the floor stops being absorbing.
        """

        floor = self.config.log_variance_floor
        if self.config.hard_log_variance_clamp:
            return log_variance.clamp(floor, -floor)
        bounded = floor + torch.nn.functional.softplus(log_variance - floor)
        return -floor - torch.nn.functional.softplus(-floor - bounded)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Deterministic latents: the posterior mean under a variational encoder."""

        return self.encode_distribution(images)[0]

    def _prefix_mask(
        self,
        latents: torch.Tensor,
        prefix_lengths: Optional[Union[int, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if prefix_lengths is None:
            return None
        if isinstance(prefix_lengths, int):
            prefix_lengths = torch.full(
                (latents.shape[0],),
                prefix_lengths,
                device=latents.device,
                dtype=torch.long,
            )
        else:
            prefix_lengths = prefix_lengths.to(device=latents.device, dtype=torch.long)
        if prefix_lengths.shape != (latents.shape[0],):
            raise ValueError(
                f"prefix_lengths must be scalar or shape {(latents.shape[0],)}, "
                f"received {tuple(prefix_lengths.shape)}"
            )
        invalid_prefix = (prefix_lengths < 1) | (
            prefix_lengths > self.config.num_latents
        )
        if not torch.compiler.is_compiling() and bool(invalid_prefix.any()):
            raise ValueError(
                f"prefix lengths must lie in [1, {self.config.num_latents}]"
            )
        positions = torch.arange(self.config.num_latents, device=latents.device)
        return positions[None, :] < prefix_lengths[:, None]

    def decode(
        self,
        latents: torch.Tensor,
        prefix_lengths: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        expected = (self.config.num_latents, self.config.latent_dim)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                f"latents must have shape [B, {expected[0]}, {expected[1]}], "
                f"received {tuple(latents.shape)}"
            )
        memory_mask = self._prefix_mask(latents, prefix_lengths)
        memory = self.latent_input(latents) + self.latent_position
        patches = self.output_token + self.output_position
        patches = patches.expand(latents.shape[0], -1, -1)
        for block in self.decoder_blocks:
            patches = block(
                patches, memory, self.decoder_rope, memory_mask
            )
        patch_pixels = self.patch_output(self.decoder_norm(patches))
        batch = latents.shape[0]
        grid = self.config.grid_size
        size = self.config.patch_size
        channels = self.config.in_channels
        images = patch_pixels.reshape(batch, grid, grid, channels, size, size)
        return images.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, self.config.image_size, self.config.image_size
        )

    def forward(
        self,
        images: torch.Tensor,
        prefix_lengths: Optional[Union[int, torch.Tensor]] = None,
        *,
        include_full_reconstruction: bool = False,
        noise_mode: Optional[str] = None,
        noise_scales: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mean, log_variance = self.encode_distribution(images)
        if self.config.variational and self.training:
            latents = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latents = mean
        decoded_latents = latents
        if noise_mode is not None:
            if noise_scales is None or tuple(noise_scales.shape) != tuple(
                latents.shape[:2]
            ):
                raise ValueError(
                    "noise_scales must have shape [batch, num_latents] when "
                    "noise_mode is set"
                )
            # Reference scale is the batch latent RMS, kept IN-GRAPH: a detached
            # reference lets gradients treat the noise as constant, creating a
            # runaway amplitude treadmill (observed: latent RMS 50x baseline).
            # In-graph, amplitude is exactly gauge-neutral for both noise modes.
            reference = latents.float().square().mean().sqrt().to(latents.dtype)
            scales = noise_scales[..., None].to(latents.dtype)
            noise = torch.randn_like(latents) * reference
            if noise_mode == "mix":
                decoded_latents = scales * latents + (1.0 - scales) * noise
            elif noise_mode == "add":
                decoded_latents = latents + scales * noise
            else:
                raise ValueError("noise_mode must be mix or add")
        reconstruction = self.decode(decoded_latents, prefix_lengths)
        output = {"latents": latents, "reconstruction": reconstruction}
        if self.config.variational:
            output["mean"] = mean
            output["log_variance"] = log_variance
        if include_full_reconstruction:
            if prefix_lengths is None and noise_mode is None:
                output["full_reconstruction"] = reconstruction
            else:
                output["full_reconstruction"] = self.decode(latents)
        return output

    @property
    def exported_token_count(self) -> int:
        return self.config.num_latents
