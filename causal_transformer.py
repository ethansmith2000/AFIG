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

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


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


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        dropout: float = 0.0,
        conditional_film: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        self.causal = causal
        self.norm = nn.LayerNorm(width)
        self.conditional_film = ConditionalFiLM(width) if conditional_film else None
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        batch, length, _ = x.shape
        hidden = self.norm(x)
        if self.conditional_film is not None:
            if condition is None:
                raise ValueError("condition is required when metadata FiLM is enabled")
            hidden = self.conditional_film(hidden, condition)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
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
        return x + self.out_proj(attention), (key, value) if use_cache else None


class FeedForward(nn.Module):
    def __init__(
        self,
        width: int,
        mult: int,
        dropout: float,
        conditional_film: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conditional_film = ConditionalFiLM(width) if conditional_film else None
        self.net = nn.Sequential(
            nn.Linear(width, width * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mult, width),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden = self.norm(x)
        if self.conditional_film is not None:
            if condition is None:
                raise ValueError("condition is required when metadata FiLM is enabled")
            hidden = self.conditional_film(hidden, condition)
        return x + self.net(hidden)


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        conditional_film: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(
            width, num_heads, dropout, conditional_film, causal
        )
        self.ff = FeedForward(width, ff_mult, dropout, conditional_film)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        x, new_cache = self.attn(x, condition, kv_cache, use_cache)
        return self.ff(x, condition), new_cache
