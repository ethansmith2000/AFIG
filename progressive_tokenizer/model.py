"""Deterministic whole-image tokenizer with Perceiver pooling.

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
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    rope_theta: float = 10_000.0

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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.out_dropout = nn.Dropout(projection_dropout)
        self.attention_dropout = attention_dropout
        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), math.log(math.sqrt(self.head_dim)))
        )

    def forward(self, x: torch.Tensor, rope: Optional[Rotary2D] = None) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        if rope is not None:
            q = rope.rotate(q)
            k = rope.rotate(k)
        q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
        k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype)
        scale = self.logit_scale.exp().clamp(max=100.0).to(q.dtype)
        q = q * scale[None, :, None, None]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=1.0,
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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)
        self.out_dropout = nn.Dropout(projection_dropout)
        self.attention_dropout = attention_dropout
        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), math.log(math.sqrt(self.head_dim)))
        )

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
        q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype)
        k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype)
        scale = self.logit_scale.exp().clamp(max=100.0).to(q.dtype)
        q = q * scale[None, :, None, None]
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
            scale=1.0,
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
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(
            config.width, config.mlp_ratio, config.projection_dropout
        )

    def forward(self, x: torch.Tensor, rope: Rotary2D) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope)
        return x + self.ffn(self.ffn_norm(x))


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
        )
        self.self_norm = _norm(config.width)
        self.self_attention = SelfAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
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
        )
        self.cross_query_norm = _norm(config.width)
        self.cross_memory_norm = _norm(config.width)
        self.cross_attention = CrossAttention(
            config.width,
            config.num_heads,
            config.attention_dropout,
            config.projection_dropout,
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
        self.pool_blocks = nn.ModuleList(
            PerceiverPoolBlock(config) for _ in range(config.pool_depth)
        )
        self.latent_norm = _norm(config.width)
        self.latent_projection = nn.Linear(config.width, config.latent_dim)

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

    def encode(self, images: torch.Tensor) -> torch.Tensor:
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
        for block in self.pool_blocks:
            queries = block(queries, patches)
        return self.latent_projection(self.latent_norm(queries))

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
        if bool(((prefix_lengths < 1) | (prefix_lengths > self.config.num_latents)).any()):
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
    ) -> dict[str, torch.Tensor]:
        latents = self.encode(images)
        reconstruction = self.decode(latents, prefix_lengths)
        return {"latents": latents, "reconstruction": reconstruction}

    @property
    def exported_token_count(self) -> int:
        return self.config.num_latents
