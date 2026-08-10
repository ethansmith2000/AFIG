"""Representation-neutral causal Transformer blocks with KV caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CausalTransformerConfig:
    width: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    max_seq_len: int = 53
    gradient_checkpointing: bool = False
    qk_norm: bool = False

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def build_rope_tables(
    coordinates: torch.Tensor, head_dim: int, base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary tables for arbitrary real-valued position coordinates.

    ``coordinates`` is [L] for 1-D or [L, 2] for axial 2-D, in which case the
    rotation pairs are split evenly between the two axes.  Coordinates are real
    rather than integer indices so a latent's pooled ``radius_center`` and
    angular centre can be used directly -- the frequency-space geometry, rather
    than a sequence index that conflates ring with sector.

    Returns cos and sin of shape [L, head_dim // 2], one entry per rotation pair.
    """
    if head_dim % 2:
        raise ValueError("RoPE requires an even head_dim")
    # RoPE phase construction must not inherit the activation/autocast dtype.
    # In bf16, integer positions above 256 are not all exactly representable;
    # converting q.dtype positions back to fp32 after that point cannot recover
    # the lost coordinates.  Callers provide fp32 coordinates, and this disabled
    # autocast block keeps the complete table fp32 until apply_rope casts at use.
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        coordinates = coordinates.to(dtype=torch.float32)
        pairs = head_dim // 2
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        axes = coordinates.shape[-1]
        if pairs % axes:
            raise ValueError(
                f"head_dim//2={pairs} must divide evenly among {axes} axes"
            )
        per_axis = pairs // axes
        angles = []
        for axis in range(axes):
            exponent = torch.arange(
                per_axis, dtype=torch.float32, device=coordinates.device
            )
            inverse_frequency = base ** (-exponent / max(per_axis, 1))
            angles.append(
                coordinates[:, axis : axis + 1] * inverse_frequency[None, :]
            )
        angle = torch.cat(angles, dim=-1)
        return angle.cos().float(), angle.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs of x by the given angles.

    x: [B, H, L, D];  cos/sin: [L, D//2].  Rotation acts on q and k only, so the
    residual stream is untouched -- this shapes attention geometry rather than
    token identity.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


class ConditionalFiLM(nn.Module):
    """Zero-initialized scale/shift modulation for known token metadata."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.net(condition).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class AdaLNZeroModulation(nn.Module):
    """Canonical zero-initialized modulation and residual gates."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.net(condition).chunk(6, dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        dropout: float = 0.0,
        conditional_film: bool = False,
        causal: bool = True,
        qk_norm: bool = False,
        affine_free_layer_norm: bool = False,
    ):
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(
            width, elementwise_affine=not affine_free_layer_norm
        )
        self.conditional_film = ConditionalFiLM(width) if conditional_film else None
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        norm_factory = (
            lambda: nn.LayerNorm(self.head_dim, elementwise_affine=False)
            if affine_free_layer_norm
            else nn.RMSNorm(self.head_dim, elementwise_affine=True)
        )
        self.q_norm = norm_factory() if qk_norm else None
        self.k_norm = norm_factory() if qk_norm else None
        self.out_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        normalized_input: bool = False,
        residual: bool = True,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch, length, _ = x.shape
        hidden = x if normalized_input else self.norm(x)
        if self.conditional_film is not None:
            if condition is None:
                raise ValueError("condition is required when metadata FiLM is enabled")
            hidden = self.conditional_film(hidden, condition)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            # Autocast leaves RMSNorm's learned weight in fp32.  Passing a bf16
            # activation and fp32 weight to nn.RMSNorm disables the fused CUDA
            # path, so cast the small affine vector at use while preserving its
            # fp32 master parameter and gradient.
            if isinstance(self.q_norm, nn.RMSNorm):
                query = F.rms_norm(
                    query,
                    self.q_norm.normalized_shape,
                    self.q_norm.weight.to(query.dtype),
                    self.q_norm.eps,
                )
                assert isinstance(self.k_norm, nn.RMSNorm)
                key = F.rms_norm(
                    key,
                    self.k_norm.normalized_shape,
                    self.k_norm.weight.to(key.dtype),
                    self.k_norm.eps,
                )
            else:
                query = self.q_norm(query)
                assert self.k_norm is not None
                key = self.k_norm(key)
        if rope is not None:
            cos, sin = rope
            # With a KV cache only the newest token(s) are passed in, so the
            # rotation must be indexed by absolute position, not by 0..length.
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            if offset + length > cos.shape[0]:
                raise ValueError("RoPE tables are shorter than the attended sequence")
            query = apply_rope(query, cos[offset : offset + length], sin[offset : offset + length])
            key = apply_rope(key, cos[offset : offset + length], sin[offset : offset + length])
        if kv_cache is not None:
            key = torch.cat([kv_cache[0], key], dim=2)
            value = torch.cat([kv_cache[1], value], dim=2)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and kv_cache is None and length > 1,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, -1)
        output = self.out_proj(attention)
        return (x + output if residual else output), (key, value) if use_cache else None


class FeedForward(nn.Module):
    def __init__(
        self,
        width: int,
        mult: int,
        dropout: float,
        conditional_film: bool = False,
        affine_free_layer_norm: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(
            width, elementwise_affine=not affine_free_layer_norm
        )
        self.conditional_film = ConditionalFiLM(width) if conditional_film else None
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
        condition: Optional[torch.Tensor] = None,
        normalized_input: bool = False,
        residual: bool = True,
    ) -> torch.Tensor:
        hidden = x if normalized_input else self.norm(x)
        if self.conditional_film is not None:
            if condition is None:
                raise ValueError("condition is required when metadata FiLM is enabled")
            hidden = self.conditional_film(hidden, condition)
        output = self.net(hidden)
        return x + output if residual else output


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        conditional_film: bool = False,
        causal: bool = True,
        qk_norm: bool = False,
        adaln_zero: bool = False,
    ):
        super().__init__()
        if adaln_zero and conditional_film:
            raise ValueError("adaln_zero replaces conditional_film")
        self.adaln_zero = adaln_zero
        self.attn = CausalSelfAttention(
            width,
            num_heads,
            dropout,
            conditional_film,
            causal,
            qk_norm,
            affine_free_layer_norm=adaln_zero,
        )
        self.ff = FeedForward(
            width,
            ff_mult,
            dropout,
            conditional_film,
            affine_free_layer_norm=adaln_zero,
        )
        self.adaln = AdaLNZeroModulation(width) if adaln_zero else None

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        if self.adaln_zero:
            if condition is None or self.adaln is None:
                raise ValueError("condition is required for AdaLN-Zero")
            (
                shift_attention,
                scale_attention,
                gate_attention,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = self.adaln(condition)
            hidden = self.attn.norm(x)
            hidden = hidden * (1.0 + scale_attention) + shift_attention
            attention, new_cache = self.attn(
                hidden,
                kv_cache=kv_cache,
                use_cache=use_cache,
                rope=rope,
                normalized_input=True,
                residual=False,
            )
            x = x + gate_attention * attention
            hidden = self.ff.norm(x)
            hidden = hidden * (1.0 + scale_ffn) + shift_ffn
            feed_forward = self.ff(
                hidden, normalized_input=True, residual=False
            )
            return x + gate_ffn * feed_forward, new_cache
        x, new_cache = self.attn(x, condition, kv_cache, use_cache, rope)
        return self.ff(x, condition), new_cache
