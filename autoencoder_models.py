"""Autoencoder/VAE models for compressed AFIG representations.

The frequency models preserve radial-order causality while compressing either
fixed contiguous chunks or adaptive angular sectors inside integer-radius
rings. Pooling queries are temporary computation and are not exported tokens.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from frequency import build_orbit_table


@dataclass(frozen=True)
class AutoencoderConfig:
    mode: str = "causal_k"  # causal_k | causal_ring | spatial_downsample
    variational: bool = False
    latent_dim: int = 64
    model_width: int = 128
    perceiver_width: int = 256
    perceiver_heads: int = 4
    ring_transformer_layers: int = 2
    ring_block_causal: bool = False
    depth: int = 0  # 0 selects the minimum full-sequence receptive field
    kernel_size: int = 3
    group_size: int = 4
    pooler: str = "perceiver_sector"  # flat_mlp | perceiver_full | perceiver_sector
    target_tokens_per_latent: int = 16
    max_ring_latents: int = 4
    group_conditioning: str = "none"  # none | film | low_rank | film_low_rank
    conditioning_rank: int = 16
    spatial_resolution: int = 32
    spatial_downsample: int = 4
    spatial_latent_channels: int = 8
    spatial_base_channels: int = 64

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def _balanced_splits(indices: Sequence[int], parts: int) -> List[List[int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    n = len(indices)
    return [
        list(indices[(n * part) // parts : (n * (part + 1)) // parts])
        for part in range(parts)
    ]


class GroupLayout(nn.Module):
    """Static mapping between Fourier positions, parents, and exported latents."""

    def __init__(
        self,
        *,
        seq_len: int,
        mode: str,
        group_size: int,
        radius_bin: torch.Tensor,
        target_tokens_per_latent: int,
        max_ring_latents: int,
        pooler: str,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if target_tokens_per_latent <= 0 or max_ring_latents <= 0:
            raise ValueError("ring compression settings must be positive")
        if mode not in ("causal_k", "causal_ring"):
            raise ValueError(f"Unsupported frequency grouping mode: {mode}")
        if pooler not in ("flat_mlp", "perceiver_full", "perceiver_sector"):
            raise ValueError(f"Unknown pooler: {pooler}")
        parent_members: List[List[int]] = []
        latent_members: List[List[int]] = []
        latent_parent: List[int] = []
        latent_slot: List[int] = []
        token_parent = torch.empty(seq_len, dtype=torch.long)
        token_latent = torch.empty(seq_len, dtype=torch.long)

        if mode == "causal_k":
            for start in range(0, seq_len, group_size):
                members = list(range(start, min(start + group_size, seq_len)))
                parent = len(parent_members)
                parent_members.append(members)
                latent_members.append(members)
                latent_parent.append(parent)
                latent_slot.append(0)
                token_parent[members] = parent
                token_latent[members] = len(latent_members) - 1
        else:
            unique_radii = torch.unique(radius_bin, sorted=True).tolist()
            for radius in unique_radii:
                members = torch.nonzero(radius_bin == radius, as_tuple=False).flatten().tolist()
                parent = len(parent_members)
                parent_members.append(members)
                token_parent[members] = parent
                count = min(
                    max_ring_latents,
                    max(1, math.ceil(len(members) / target_tokens_per_latent)),
                )
                sectors = _balanced_splits(members, count)
                for slot, sector in enumerate(sectors):
                    latent_members.append(
                        members if pooler == "perceiver_full" else sector
                    )
                    latent_parent.append(parent)
                    latent_slot.append(slot)
                    token_latent[sector] = len(latent_members) - 1

        max_members = max(len(members) for members in latent_members)
        gather_indices = torch.zeros(len(latent_members), max_members, dtype=torch.long)
        gather_mask = torch.zeros(len(latent_members), max_members, dtype=torch.bool)
        for latent, members in enumerate(latent_members):
            gather_indices[latent, : len(members)] = torch.tensor(members)
            gather_mask[latent, : len(members)] = True

        parent_counts = torch.bincount(
            torch.tensor(latent_parent, dtype=torch.long),
            minlength=len(parent_members),
        )
        max_parent_latents = int(parent_counts.max().item())
        parent_latent_indices = torch.zeros(
            len(parent_members), max_parent_latents, dtype=torch.long
        )
        parent_latent_mask = torch.zeros(
            len(parent_members), max_parent_latents, dtype=torch.bool
        )
        offsets = torch.zeros(len(parent_members), dtype=torch.long)
        for latent, parent in enumerate(latent_parent):
            slot = int(offsets[parent].item())
            parent_latent_indices[parent, slot] = latent
            parent_latent_mask[parent, slot] = True
            offsets[parent] += 1

        self.seq_len = seq_len
        self.num_parents = len(parent_members)
        self.num_latents = len(latent_members)
        self.max_members = max_members
        self.max_parent_latents = max_parent_latents
        self.register_buffer("gather_indices", gather_indices, persistent=True)
        self.register_buffer("gather_mask", gather_mask, persistent=True)
        self.register_buffer(
            "latent_parent", torch.tensor(latent_parent, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "latent_slot", torch.tensor(latent_slot, dtype=torch.long), persistent=True
        )
        self.register_buffer("token_parent", token_parent, persistent=True)
        self.register_buffer("token_latent", token_latent, persistent=True)
        self.register_buffer(
            "parent_latent_indices", parent_latent_indices, persistent=True
        )
        self.register_buffer("parent_latent_mask", parent_latent_mask, persistent=True)
        self.register_buffer("parent_counts", parent_counts, persistent=True)

    @property
    def compression_ratio(self) -> float:
        return self.seq_len / self.num_latents


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("Causal kernels must have size >= 2")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x.transpose(1, 2), (self.left_padding, 0))
        return self.conv(x).transpose(1, 2)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, C]; cache stores the preceding normalized inputs.
        if cache is None:
            cache = x.new_zeros(x.shape[0], self.left_padding, x.shape[-1])
        window = torch.cat([cache, x], dim=1)
        y = self.conv(window.transpose(1, 2)).transpose(1, 2)
        new_cache = window[:, -self.left_padding :].detach()
        return y, new_cache


class ConditionalAdapter(nn.Module):
    """FiLM and/or low-rank conditional residual modulation."""

    def __init__(self, width: int, condition_dim: int, mode: str, rank: int):
        super().__init__()
        if mode not in ("none", "film", "low_rank", "film_low_rank"):
            raise ValueError(f"Unknown conditioning mode: {mode}")
        self.use_film = mode in ("film", "film_low_rank")
        self.use_low_rank = mode in ("low_rank", "film_low_rank")
        self.film = nn.Linear(condition_dim, 2 * width) if self.use_film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if self.use_low_rank:
            self.down = nn.Linear(width, rank, bias=False)
            self.gate = nn.Linear(condition_dim, rank)
            self.up = nn.Linear(rank, width, bias=False)
            nn.init.zeros_(self.up.weight)
        else:
            self.down = None
            self.gate = None
            self.up = None

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
        if condition is None:
            return x
        if self.film is not None:
            scale, shift = self.film(condition).chunk(2, dim=-1)
            x = x * (1.0 + scale) + shift
        if self.down is not None and self.gate is not None and self.up is not None:
            x = x + self.up(self.down(x) * torch.tanh(self.gate(condition)))
        return x


class CausalTCNBlock(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_size: int,
        dilation: int,
        condition_dim: int = 0,
        conditioning: str = "none",
        conditioning_rank: int = 16,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.condition = ConditionalAdapter(
            width,
            max(condition_dim, 1),
            conditioning if condition_dim > 0 else "none",
            conditioning_rank,
        )
        self.conv = CausalConv1d(width, kernel_size, dilation)
        self.channel = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GLU(dim=-1),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.channel[-1].weight)
        nn.init.zeros_(self.channel[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.condition(self.norm(x), condition)
        return x + self.channel(F.silu(self.conv(hidden)))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
        condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.condition(self.norm(x), condition)
        y, new_cache = self.conv.forward_step(hidden, cache)
        return x + self.channel(F.silu(y)), new_cache


class CausalTCN(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        kernel_size: int,
        condition_dim: int = 0,
        conditioning: str = "none",
        conditioning_rank: int = 16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    width,
                    kernel_size,
                    dilation=2**layer,
                    condition_dim=condition_dim,
                    conditioning=conditioning,
                    conditioning_rank=conditioning_rank,
                )
                for layer in range(depth)
            ]
        )
        self.receptive_field = 1 + (kernel_size - 1) * (2**depth - 1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, condition)
        return x

    def forward_streaming(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        caches: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        outputs = []
        for index in range(x.shape[1]):
            step = x[:, index : index + 1]
            step_condition = (
                None if condition is None else condition[:, index : index + 1]
            )
            for layer, block in enumerate(self.blocks):
                step, caches[layer] = block.forward_step(
                    step, caches[layer], step_condition
                )
            outputs.append(step)
        return torch.cat(outputs, dim=1)


class PositionFeatures(nn.Module):
    def __init__(
        self,
        metadata: Dict[str, torch.Tensor],
        width: int,
        *,
        project: bool = True,
    ):
        super().__init__()
        radius = metadata["radius"].float()
        kx = metadata["kx_signed"].float()
        ky = metadata["ky_signed"].float()
        axis = ((kx == 0) | (ky == 0)).float()
        empirical_scale = metadata.get("empirical_scale", torch.ones_like(radius)).float()
        log_scale = torch.log(empirical_scale.clamp_min(1e-8))
        log_scale = (log_scale - log_scale.mean()) / log_scale.std().clamp_min(1e-6)
        features = torch.stack(
            [
                kx / kx.abs().max().clamp_min(1),
                ky / ky.abs().max().clamp_min(1),
                radius / radius.max().clamp_min(1),
                torch.sin(metadata["angle"].float()),
                torch.cos(metadata["angle"].float()),
                metadata["is_self_conjugate"].float(),
                axis,
                log_scale,
            ],
            dim=-1,
        )
        self.register_buffer("features", features, persistent=True)
        self.width = width
        self.proj = (
            nn.Sequential(
                nn.Linear(features.shape[-1], width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            if project
            else None
        )

    def forward(self) -> torch.Tensor:
        if self.proj is None:
            return self.features.new_zeros(self.features.shape[0], self.width)
        return self.proj(self.features)

    @property
    def condition_dim(self) -> int:
        return self.features.shape[-1]


class QKRMSAttention(nn.Module):
    """Multi-head attention with per-head Q/K normalization and raw values."""

    def __init__(
        self,
        query_width: int,
        context_width: int,
        width: int,
        heads: int,
        *,
        affine_free_layer_norm: bool = False,
    ):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Attention width must be positive and divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(query_width, width)
        self.key = nn.Linear(context_width, width)
        self.value = nn.Linear(context_width, width)
        if affine_free_layer_norm:
            self.query_norm = nn.LayerNorm(
                self.head_dim, elementwise_affine=False
            )
            self.key_norm = nn.LayerNorm(
                self.head_dim, elementwise_affine=False
            )
        else:
            self.query_norm = nn.RMSNorm(self.head_dim)
            self.key_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(width, query_width)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_count = queries.shape[:2]
        context_count = context.shape[1]
        query = self.query(queries).reshape(
            batch, query_count, self.heads, self.head_dim
        )
        key = self.key(context).reshape(
            batch, context_count, self.heads, self.head_dim
        )
        value = self.value(context).reshape(
            batch, context_count, self.heads, self.head_dim
        )
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            else:
                raise ValueError("Attention mask must have shape [Q,K] or [B,Q,K]")
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, query_count, self.width)
        return self.output(attended)


class MetadataConditioner(nn.Module):
    """Encode the complete physical/group metadata once for all AdaLN blocks."""

    def __init__(self, condition_dim: int, width: int, expansion: int = 4):
        super().__init__()
        hidden = expansion * width
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class AdaLNZeroModulation(nn.Module):
    """Canonical affine-free AdaLN-Zero modulation for attention and MLP."""

    def __init__(self, width: int, condition_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * width),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self.net(condition).chunk(6, dim=-1)


def _clean_norm(width: int, enabled: bool) -> nn.Module:
    if enabled:
        return nn.LayerNorm(width, elementwise_affine=False)
    return nn.RMSNorm(width)


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale) + shift


class BlockCausalTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        condition_dim: int,
        conditioning: str,
        conditioning_rank: int,
    ):
        super().__init__()
        self.clean_adaln = conditioning == "adaln_zero"
        self.attention_norm = _clean_norm(width, self.clean_adaln)
        self.attention_condition = (
            None
            if self.clean_adaln
            else ConditionalAdapter(
                width, condition_dim, conditioning, conditioning_rank
            )
        )
        self.attention = QKRMSAttention(
            width,
            width,
            width,
            heads,
            affine_free_layer_norm=self.clean_adaln,
        )
        self.ffn_norm = _clean_norm(width, self.clean_adaln)
        self.ffn_condition = (
            None
            if self.clean_adaln
            else ConditionalAdapter(
                width, condition_dim, conditioning, conditioning_rank
            )
        )
        self.adaln = (
            AdaLNZeroModulation(width, condition_dim)
            if self.clean_adaln
            else None
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )

    def forward(
        self,
        states: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.clean_adaln:
            assert self.adaln is not None
            (
                shift_attention,
                scale_attention,
                gate_attention,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = self.adaln(condition)
            hidden = _modulate(
                self.attention_norm(states), shift_attention, scale_attention
            )
            states = states + gate_attention * self.attention(hidden, hidden, mask)
            hidden = _modulate(self.ffn_norm(states), shift_ffn, scale_ffn)
            return states + gate_ffn * self.ffn(hidden)
        assert self.attention_condition is not None
        assert self.ffn_condition is not None
        hidden = self.attention_condition(self.attention_norm(states), condition)
        states = states + self.attention(hidden, hidden, mask)
        hidden = self.ffn_condition(self.ffn_norm(states), condition)
        return states + self.ffn(hidden)


class CausalCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        condition_dim: int,
        conditioning: str,
        conditioning_rank: int,
    ):
        super().__init__()
        self.clean_adaln = conditioning == "adaln_zero"
        self.query_norm = _clean_norm(width, self.clean_adaln)
        self.context_norm = _clean_norm(width, self.clean_adaln)
        self.attention = QKRMSAttention(
            width,
            width,
            width,
            heads,
            affine_free_layer_norm=self.clean_adaln,
        )
        self.ffn_norm = _clean_norm(width, self.clean_adaln)
        self.condition = (
            None
            if self.clean_adaln
            else ConditionalAdapter(
                width, condition_dim, conditioning, conditioning_rank
            )
        )
        self.adaln = (
            AdaLNZeroModulation(width, condition_dim)
            if self.clean_adaln
            else None
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.clean_adaln:
            assert self.adaln is not None
            (
                shift_attention,
                scale_attention,
                gate_attention,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = self.adaln(condition)
            hidden = _modulate(
                self.query_norm(queries), shift_attention, scale_attention
            )
            queries = queries + gate_attention * self.attention(
                hidden, self.context_norm(context), mask
            )
            hidden = _modulate(self.ffn_norm(queries), shift_ffn, scale_ffn)
            return queries + gate_ffn * self.ffn(hidden)
        assert self.condition is not None
        queries = queries + self.attention(
            self.query_norm(queries), self.context_norm(context), mask
        )
        hidden = self.condition(self.ffn_norm(queries), condition)
        return queries + self.ffn(hidden)


class GroupPooler(nn.Module):
    def __init__(
        self,
        layout: GroupLayout,
        width: int,
        latent_dim: int,
        variational: bool,
        pooler: str,
        perceiver_width: int = 256,
        perceiver_heads: int = 4,
        condition_dim: int = 0,
        conditioning: str = "none",
        conditioning_rank: int = 16,
    ):
        super().__init__()
        self.layout = layout
        self.width = width
        self.pooler = pooler
        self.variational = variational
        self.perceiver_width = perceiver_width
        self.perceiver_heads = perceiver_heads
        posterior_dim = latent_dim * (2 if variational else 1)
        if pooler == "flat_mlp":
            self.flat = nn.Sequential(
                nn.Linear(layout.max_members * width, width),
                nn.SiLU(),
                nn.Linear(width, posterior_dim),
            )
            self.query = None
            self.key = None
            self.value = None
            self.out = None
            self.query_norm = None
            self.key_norm = None
            self.attention_out = None
            self.ffn_norm = None
            self.ffn = None
            self.flat_condition = (
                nn.Linear(condition_dim, layout.max_members * width)
                if condition_dim > 0 and conditioning != "none"
                else None
            )
            if self.flat_condition is not None:
                nn.init.zeros_(self.flat_condition.weight)
                nn.init.zeros_(self.flat_condition.bias)
        else:
            if perceiver_width <= 0 or perceiver_heads <= 0:
                raise ValueError("Perceiver width and head count must be positive")
            if perceiver_width % perceiver_heads:
                raise ValueError("Perceiver width must be divisible by its head count")
            self.flat = None
            self.flat_condition = None
            self.query = nn.Parameter(
                torch.randn(layout.num_latents, perceiver_width)
                / math.sqrt(perceiver_width)
            )
            self.key = nn.Linear(width, perceiver_width)
            self.value = nn.Linear(width, perceiver_width)
            head_dim = perceiver_width // perceiver_heads
            self.query_norm = nn.RMSNorm(head_dim)
            self.key_norm = nn.RMSNorm(head_dim)
            self.attention_out = nn.Linear(perceiver_width, perceiver_width)
            self.ffn_norm = nn.RMSNorm(perceiver_width)
            self.ffn = nn.Sequential(
                nn.Linear(perceiver_width, 4 * perceiver_width),
                nn.SiLU(),
                nn.Linear(4 * perceiver_width, perceiver_width),
            )
            self.out = nn.Sequential(
                nn.RMSNorm(perceiver_width),
                nn.Linear(perceiver_width, posterior_dim),
            )
        self.output_condition = ConditionalAdapter(
            perceiver_width if self.query is not None else width,
            max(condition_dim, 1),
            conditioning if condition_dim > 0 else "none",
            conditioning_rank,
        )

    def forward(
        self,
        states: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        gathered = states[:, self.layout.gather_indices]  # [B,G,N,W]
        mask = self.layout.gather_mask[None, :, :, None].to(gathered.dtype)
        if self.flat is not None:
            flattened = (gathered * mask).flatten(2)
            if self.flat_condition is not None and condition is not None:
                flattened = flattened + self.flat_condition(condition)
            pooled = self.flat(flattened)
        else:
            assert (
                self.query is not None
                and self.key is not None
                and self.value is not None
                and self.query_norm is not None
                and self.key_norm is not None
                and self.attention_out is not None
                and self.ffn_norm is not None
                and self.ffn is not None
            )
            heads = self.perceiver_heads
            head_dim = self.perceiver_width // heads
            residual_query = self.query[None].expand(batch, -1, -1)
            query = residual_query.reshape(
                batch, self.layout.num_latents, heads, head_dim
            )
            query = self.query_norm(query)
            key = self.key(gathered).reshape(
                batch, self.layout.num_latents, self.layout.max_members, heads, head_dim
            ).permute(0, 1, 3, 2, 4)
            key = self.key_norm(key)
            value = self.value(gathered).reshape(
                batch, self.layout.num_latents, self.layout.max_members, heads, head_dim
            ).permute(0, 1, 3, 2, 4)
            attended = F.scaled_dot_product_attention(
                query[:, :, :, None, :],
                key,
                value,
                attn_mask=self.layout.gather_mask[None, :, None, None, :],
                dropout_p=0.0,
            )
            attended = attended.squeeze(3).transpose(2, 3).reshape(
                batch, self.layout.num_latents, self.perceiver_width
            )
            pooled_state = residual_query + self.attention_out(attended)
            assert self.out is not None
            pooled_state = self.output_condition(pooled_state, condition)
            pooled_state = pooled_state + self.ffn(self.ffn_norm(pooled_state))
            pooled = self.out(pooled_state)
        if self.variational:
            mean, logvar = pooled.chunk(2, dim=-1)
            logvar = logvar.clamp(-12.0, 8.0)
        else:
            mean = pooled
            logvar = torch.zeros_like(mean)
        return mean, logvar


class CoordinateUnpooler(nn.Module):
    def __init__(
        self,
        layout: GroupLayout,
        width: int,
        token_dim: int,
        condition_dim: int = 0,
        conditioning: str = "none",
        conditioning_rank: int = 16,
    ):
        super().__init__()
        self.layout = layout
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.condition = ConditionalAdapter(
            width,
            max(condition_dim, 1),
            conditioning if condition_dim > 0 else "none",
            conditioning_rank,
        )
        self.output = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, token_dim),
        )

    def forward(
        self,
        latent_states: torch.Tensor,
        position_states: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Gather every latent belonging to each output token's parent group.
        parent_latents = self.layout.parent_latent_indices[self.layout.token_parent]
        parent_mask = self.layout.parent_latent_mask[self.layout.token_parent]
        gathered = latent_states[:, parent_latents]  # [B,L,M,W]
        query = self.query(position_states)[None, :, None, :]
        query = query.expand(latent_states.shape[0], -1, -1, -1)
        key = self.key(gathered)
        value = self.value(gathered)
        decoded = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=parent_mask[None, :, None, :],
            dropout_p=0.0,
        )
        decoded = self.condition(decoded.squeeze(2) + position_states[None], condition)
        return self.output(decoded)


class SequentialRingEncoder(nn.Module):
    """Block-causal coefficient encoder with configurable sector/ring blocks."""

    def __init__(
        self,
        layout: GroupLayout,
        width: int,
        latent_dim: int,
        token_dim: int,
        layers: int,
        heads: int,
        variational: bool,
        condition_dim: int,
        conditioning: str,
        conditioning_rank: int,
        ring_block_causal: bool,
    ):
        super().__init__()
        self.layout = layout
        self.width = width
        self.variational = variational
        self.clean_adaln = conditioning == "adaln_zero"
        self.pool_query_residual = not self.clean_adaln
        self.input = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList(
            [
                BlockCausalTransformerBlock(
                    width,
                    heads,
                    condition_dim,
                    conditioning,
                    conditioning_rank,
                )
                for _ in range(layers)
            ]
        )
        self.queries = nn.Parameter(
            torch.randn(layout.num_latents, width) / math.sqrt(width)
        )
        self.query_condition = (
            None
            if self.clean_adaln
            else ConditionalAdapter(
                width, condition_dim, conditioning, conditioning_rank
            )
        )
        self.pool_norm = _clean_norm(width, self.clean_adaln)
        self.pool_attention = QKRMSAttention(
            width,
            width,
            width,
            heads,
            affine_free_layer_norm=self.clean_adaln,
        )
        self.pool_ffn_norm = _clean_norm(width, self.clean_adaln)
        self.pool_ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )
        posterior_dim = latent_dim * (2 if variational else 1)
        self.output = nn.Sequential(
            _clean_norm(width, self.clean_adaln),
            nn.Linear(width, posterior_dim),
        )
        token_sector = (
            layout.token_parent if ring_block_causal else layout.token_latent
        )
        self.register_buffer(
            "block_causal_mask",
            token_sector[:, None] >= token_sector[None, :],
            persistent=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        position_states: torch.Tensor,
        token_condition: torch.Tensor,
        latent_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.input(tokens) + position_states[None].to(tokens.dtype)
        for block in self.blocks:
            states = block(states, token_condition, self.block_causal_mask)

        batch = states.shape[0]
        gathered = states[:, self.layout.gather_indices]
        gathered = gathered.reshape(
            batch * self.layout.num_latents, self.layout.max_members, self.width
        )
        queries = self.queries[None].expand(batch, -1, -1)
        if self.query_condition is not None:
            queries = self.query_condition(queries, latent_condition)
        flat_queries = queries.reshape(batch * self.layout.num_latents, 1, self.width)
        flat_mask = self.layout.gather_mask[:, None, :].expand(
            -1, batch, -1
        ).transpose(0, 1).reshape(
            batch * self.layout.num_latents, 1, self.layout.max_members
        )
        attended = self.pool_attention(
            self.pool_norm(flat_queries), self.pool_norm(gathered), flat_mask
        )
        pooled = attended if not self.pool_query_residual else flat_queries + attended
        pooled = pooled + self.pool_ffn(self.pool_ffn_norm(pooled))
        posterior = self.output(pooled.squeeze(1)).reshape(
            batch, self.layout.num_latents, -1
        )
        if self.variational:
            mean, logvar = posterior.chunk(2, dim=-1)
            logvar = logvar.clamp(-12.0, 8.0)
        else:
            mean = posterior
            logvar = torch.zeros_like(mean)
        return mean, logvar


class SequentialRingDecoder(nn.Module):
    """Sector- or ring-block latent mixer and coefficient decoder."""

    def __init__(
        self,
        layout: GroupLayout,
        width: int,
        latent_dim: int,
        token_dim: int,
        layers: int,
        heads: int,
        condition_dim: int,
        conditioning: str,
        conditioning_rank: int,
        ring_block_causal: bool,
    ):
        super().__init__()
        self.layout = layout
        self.clean_adaln = conditioning == "adaln_zero"
        self.input = nn.Linear(latent_dim, width)
        self.latent_blocks = nn.ModuleList(
            [
                BlockCausalTransformerBlock(
                    width,
                    heads,
                    condition_dim,
                    conditioning,
                    conditioning_rank,
                )
                for _ in range(layers)
            ]
        )
        self.coordinate_blocks = nn.ModuleList(
            [
                CausalCrossAttentionBlock(
                    width,
                    heads,
                    condition_dim,
                    conditioning,
                    conditioning_rank,
                )
                for _ in range(layers)
            ]
        )
        self.output = nn.Sequential(
            _clean_norm(width, self.clean_adaln),
            nn.Linear(width, token_dim),
        )
        latent_ids = torch.arange(layout.num_latents)
        latent_group = (
            layout.latent_parent if ring_block_causal else latent_ids
        )
        self.register_buffer(
            "latent_causal_mask",
            latent_group[:, None] >= latent_group[None, :],
            persistent=True,
        )
        coordinate_limit = (
            layout.token_parent[:, None]
            if ring_block_causal
            else layout.token_latent[:, None]
        )
        latent_limit = (
            layout.latent_parent[None, :]
            if ring_block_causal
            else latent_ids[None, :]
        )
        self.register_buffer(
            "coordinate_causal_mask",
            latent_limit <= coordinate_limit,
            persistent=True,
        )

    def forward(
        self,
        latents: torch.Tensor,
        position_states: torch.Tensor,
        token_condition: torch.Tensor,
        latent_condition: torch.Tensor,
    ) -> torch.Tensor:
        states = self.input(latents)
        for block in self.latent_blocks:
            states = block(states, latent_condition, self.latent_causal_mask)
        queries = position_states[None].to(latents.dtype).expand(
            latents.shape[0], -1, -1
        )
        for block in self.coordinate_blocks:
            queries = block(
                queries,
                states,
                token_condition,
                self.coordinate_causal_mask,
            )
        return self.output(queries)


class CausalFrequencyAutoencoder(nn.Module):
    def __init__(
        self,
        config: AutoencoderConfig,
        metadata: Dict[str, torch.Tensor],
        component_mask: torch.Tensor,
    ):
        super().__init__()
        if config.mode not in ("causal_k", "causal_ring"):
            raise ValueError("CausalFrequencyAutoencoder requires a frequency mode")
        if config.mode == "causal_ring" and config.pooler != "perceiver_sector":
            raise ValueError(
                "The sequential ring codec requires sector-local Perceiver pooling"
            )
        self.clean_ring_v2 = config.group_conditioning == "adaln_zero"
        if self.clean_ring_v2 and config.mode != "causal_ring":
            raise ValueError("adaln_zero conditioning is specific to causal_ring")
        self.config = config
        self.layout = GroupLayout(
            seq_len=component_mask.shape[0],
            mode=config.mode,
            group_size=config.group_size,
            radius_bin=metadata["radius_bin"],
            target_tokens_per_latent=config.target_tokens_per_latent,
            max_ring_latents=config.max_ring_latents,
            pooler=config.pooler,
        )
        self.register_buffer("component_mask", component_mask.float(), persistent=True)
        depth = config.depth
        if depth <= 0:
            depth = math.ceil(
                math.log2(
                    (component_mask.shape[0] - 1)
                    / max(config.kernel_size - 1, 1)
                    + 1
                )
            )
        self.effective_depth = (
            config.ring_transformer_layers if config.mode == "causal_ring" else depth
        )
        codec_width = (
            config.perceiver_width
            if config.mode == "causal_ring"
            else config.model_width
        )
        self.position = PositionFeatures(
            metadata,
            codec_width,
            project=not self.clean_ring_v2,
        )
        base_condition = self.position.features
        condition_dim = self.position.condition_dim + 2
        latent_members = base_condition[self.layout.gather_indices]
        latent_mask = self.layout.gather_mask[:, :, None].to(latent_members.dtype)
        latent_count = latent_mask.sum(dim=1).clamp_min(1.0)
        latent_mean = (latent_members * latent_mask).sum(dim=1) / latent_count
        latent_size = latent_count / float(self.layout.max_members)
        latent_parent_id = self.layout.latent_parent[:, None].float() / max(
            self.layout.num_parents - 1, 1
        )
        latent_condition = torch.cat(
            [latent_mean, latent_size, latent_parent_id], dim=-1
        )

        parent_sum = base_condition.new_zeros(
            self.layout.num_parents, base_condition.shape[-1]
        )
        parent_sum.index_add_(0, self.layout.token_parent, base_condition)
        parent_token_count = torch.bincount(
            self.layout.token_parent, minlength=self.layout.num_parents
        ).to(base_condition.dtype)
        parent_mean = parent_sum / parent_token_count[:, None].clamp_min(1.0)
        parent_size = (
            parent_token_count[:, None] / parent_token_count.max().clamp_min(1.0)
        )
        parent_id = torch.arange(
            self.layout.num_parents, dtype=base_condition.dtype
        )[:, None] / max(self.layout.num_parents - 1, 1)
        parent_condition = torch.cat([parent_mean, parent_size, parent_id], dim=-1)
        token_condition = parent_condition[self.layout.token_parent].clone()
        token_condition[:, : base_condition.shape[-1]] = base_condition
        self.register_buffer("token_condition", token_condition, persistent=True)
        self.register_buffer("latent_condition", latent_condition, persistent=True)
        self.register_buffer("parent_condition", parent_condition, persistent=True)
        self.metadata_conditioner = (
            MetadataConditioner(condition_dim, codec_width)
            if self.clean_ring_v2
            else None
        )
        block_condition_dim = codec_width if self.clean_ring_v2 else condition_dim
        if config.mode == "causal_ring":
            self.ring_encoder = SequentialRingEncoder(
                self.layout,
                config.perceiver_width,
                config.latent_dim,
                component_mask.shape[-1],
                config.ring_transformer_layers,
                config.perceiver_heads,
                config.variational,
                block_condition_dim,
                config.group_conditioning,
                config.conditioning_rank,
                config.ring_block_causal,
            )
            self.ring_decoder = SequentialRingDecoder(
                self.layout,
                config.perceiver_width,
                config.latent_dim,
                component_mask.shape[-1],
                config.ring_transformer_layers,
                config.perceiver_heads,
                block_condition_dim,
                config.group_conditioning,
                config.conditioning_rank,
                config.ring_block_causal,
            )
            self.token_proj = None
            self.encoder = None
            self.pool = None
            self.latent_proj = None
            self.parent_decoder = None
            self.unpool = None
            return

        self.ring_encoder = None
        self.ring_decoder = None
        self.token_proj = nn.Linear(component_mask.shape[-1], config.model_width)
        self.encoder = CausalTCN(
            config.model_width,
            depth,
            config.kernel_size,
            condition_dim=condition_dim,
            conditioning=config.group_conditioning,
            conditioning_rank=config.conditioning_rank,
        )
        self.pool = GroupPooler(
            self.layout,
            config.model_width,
            config.latent_dim,
            config.variational,
            config.pooler,
            perceiver_width=config.perceiver_width,
            perceiver_heads=config.perceiver_heads,
            condition_dim=condition_dim,
            conditioning=config.group_conditioning,
            conditioning_rank=config.conditioning_rank,
        )
        self.latent_proj = nn.Linear(config.latent_dim, config.model_width)
        self.parent_decoder = CausalTCN(
            config.model_width,
            depth,
            config.kernel_size,
            condition_dim=condition_dim,
            conditioning=config.group_conditioning,
            conditioning_rank=config.conditioning_rank,
        )
        self.unpool = CoordinateUnpooler(
            self.layout,
            config.model_width,
            component_mask.shape[-1],
            condition_dim=condition_dim,
            conditioning=config.group_conditioning,
            conditioning_rank=config.conditioning_rank,
        )

    @property
    def exported_token_count(self) -> int:
        return self.layout.num_latents

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = tokens * self.component_mask[None].to(tokens.dtype)
        token_condition = self.token_condition
        latent_condition = self.latent_condition
        if self.metadata_conditioner is not None:
            token_condition = self.metadata_conditioner(token_condition)
            latent_condition = self.metadata_conditioner(latent_condition)
        token_condition = token_condition[None].expand(tokens.shape[0], -1, -1)
        latent_condition = latent_condition[None].expand(tokens.shape[0], -1, -1)
        if self.config.mode == "causal_ring":
            assert self.ring_encoder is not None
            return self.ring_encoder(
                masked,
                self.position(),
                token_condition,
                latent_condition,
            )
        assert self.token_proj is not None and self.encoder is not None
        assert self.pool is not None
        states = self.token_proj(masked) + self.position()[None].to(tokens.dtype)
        states = self.encoder(states, token_condition)
        return self.pool(states, latent_condition)

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        sample_posterior: bool,
    ) -> torch.Tensor:
        if self.config.variational and sample_posterior:
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "causal_ring":
            assert self.ring_decoder is not None
            token_condition = self.token_condition
            latent_condition = self.latent_condition
            if self.metadata_conditioner is not None:
                token_condition = self.metadata_conditioner(token_condition)
                latent_condition = self.metadata_conditioner(latent_condition)
            token_condition = token_condition[None].expand(
                latents.shape[0], -1, -1
            )
            latent_condition = latent_condition[None].expand(
                latents.shape[0], -1, -1
            )
            reconstruction = self.ring_decoder(
                latents,
                self.position(),
                token_condition,
                latent_condition,
            )
            return reconstruction * self.component_mask[None].to(
                reconstruction.dtype
            )
        assert self.latent_proj is not None and self.parent_decoder is not None
        assert self.unpool is not None
        latent_states = self.latent_proj(latents)
        parent_sum = latent_states.new_zeros(
            latent_states.shape[0],
            self.layout.num_parents,
            latent_states.shape[-1],
        )
        parent_sum.index_add_(1, self.layout.latent_parent, latent_states)
        counts = self.layout.parent_counts.to(parent_sum.dtype).clamp_min(1)
        parent_condition = self.parent_condition[None].expand(
            latents.shape[0], -1, -1
        )
        parent_states = self.parent_decoder(
            parent_sum / counts[None, :, None], parent_condition
        )
        contextual_latents = (
            latent_states + parent_states[:, self.layout.latent_parent]
        )
        token_condition = self.token_condition[None].expand(
            latents.shape[0], -1, -1
        )
        reconstruction = self.unpool(
            contextual_latents, self.position(), token_condition
        )
        return reconstruction * self.component_mask[None].to(reconstruction.dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sample_posterior: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(tokens)
        if sample_posterior is None:
            sample_posterior = self.training
        latents = self.sample(mean, logvar, sample_posterior)
        reconstruction = self.decode(latents)
        kl_per_dim = (
            0.5 * (mean.square() + logvar.exp() - 1.0 - logvar)
            if self.config.variational
            else torch.zeros_like(mean)
        )
        return {
            "reconstruction": reconstruction,
            "latents": latents,
            "mean": mean,
            "logvar": logvar,
            "kl_per_dim": kl_per_dim,
        }

    @torch.no_grad()
    def export_latents(
        self,
        tokens: torch.Tensor,
        *,
        sample_posterior: bool = False,
    ) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(tokens)
        latents = self.sample(mean, logvar, sample_posterior)
        return {
            "latents": latents,
            "mean": mean,
            "logvar": logvar,
            "latent_parent": self.layout.latent_parent,
            "token_parent": self.layout.token_parent,
            "gather_indices": self.layout.gather_indices,
            "gather_mask": self.layout.gather_mask,
        }


class SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RealLatentFFT(nn.Module):
    """Hermitian orbit extraction for real maps with arbitrary channel count."""

    def __init__(self, height: int, width: int):
        super().__init__()
        table = build_orbit_table(height, width, ordering="radial")
        for name in ("ky", "kx", "partner_ky", "partner_kx", "is_self_conjugate"):
            self.register_buffer(name, table[name], persistent=True)
        self.height = height
        self.width = width
        self.seq_len = int(table["seq_len"].item())

    def encode(self, maps: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(maps.float(), norm="ortho")
        coeffs = spectrum[:, :, self.ky, self.kx]
        imag = coeffs.imag * (~self.is_self_conjugate)[None, None]
        return torch.cat([coeffs.real, imag], dim=1).permute(0, 2, 1).contiguous()

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        channels = tokens.shape[-1] // 2
        if tokens.shape[-1] != 2 * channels:
            raise ValueError("Latent FFT tokens must contain real and imaginary halves")
        real = tokens[..., :channels].permute(0, 2, 1).float()
        imag = tokens[..., channels:].permute(0, 2, 1).float()
        imag = imag * (~self.is_self_conjugate)[None, None]
        values = torch.complex(real, imag)
        spectrum = torch.zeros(
            tokens.shape[0],
            channels,
            self.height,
            self.width,
            dtype=torch.complex64,
            device=tokens.device,
        )
        spectrum[:, :, self.ky, self.kx] = values
        non_self = ~self.is_self_conjugate
        spectrum[:, :, self.partner_ky[non_self], self.partner_kx[non_self]] = (
            values[:, :, non_self].conj()
        )
        return torch.fft.ifft2(spectrum, norm="ortho").real


class SpatialAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        if config.mode != "spatial_downsample":
            raise ValueError("SpatialAutoencoder requires spatial_downsample mode")
        if config.spatial_downsample not in (2, 4, 8):
            raise ValueError("spatial_downsample must be 2, 4, or 8")
        levels = int(math.log2(config.spatial_downsample))
        base = config.spatial_base_channels
        encoder: List[nn.Module] = [nn.Conv2d(3, base, 3, padding=1)]
        channels = base
        for _ in range(levels):
            encoder.extend(
                [
                    SpatialResidualBlock(channels),
                    nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
                ]
            )
            channels *= 2
        encoder.append(SpatialResidualBlock(channels))
        posterior_channels = config.spatial_latent_channels * (
            2 if config.variational else 1
        )
        encoder.append(nn.Conv2d(channels, posterior_channels, 3, padding=1))
        self.encoder = nn.Sequential(*encoder)

        self.latent_in = nn.Conv2d(config.spatial_latent_channels, channels, 3, padding=1)
        decoder: List[nn.Module] = [SpatialResidualBlock(channels)]
        for _ in range(levels):
            decoder.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, channels // 2, 3, padding=1),
                    SpatialResidualBlock(channels // 2),
                ]
            )
            channels //= 2
        decoder.extend([nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder)
        latent_resolution = config.spatial_resolution // config.spatial_downsample
        self.latent_fft = RealLatentFFT(latent_resolution, latent_resolution)
        self.config = config

    @property
    def exported_token_count(self) -> int:
        return self.latent_fft.seq_len

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = self.encoder(images)
        if self.config.variational:
            mean, logvar = posterior.chunk(2, dim=1)
            logvar = logvar.clamp(-12.0, 8.0)
        else:
            mean = posterior
            logvar = torch.zeros_like(mean)
        return mean, logvar

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        sample_posterior: bool,
    ) -> torch.Tensor:
        if self.config.variational and sample_posterior:
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.latent_in(latents))

    def forward(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(images)
        if sample_posterior is None:
            sample_posterior = self.training
        latents = self.sample(mean, logvar, sample_posterior)
        reconstruction = self.decode(latents)
        kl_per_dim = (
            0.5 * (mean.square() + logvar.exp() - 1.0 - logvar)
            if self.config.variational
            else torch.zeros_like(mean)
        )
        return {
            "reconstruction": reconstruction,
            "latents": latents,
            "latent_tokens": self.latent_fft.encode(latents),
            "mean": mean,
            "logvar": logvar,
            "kl_per_dim": kl_per_dim,
        }

    @torch.no_grad()
    def export_latents(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = False,
    ) -> Dict[str, torch.Tensor]:
        mean, logvar = self.encode(images)
        latents = self.sample(mean, logvar, sample_posterior)
        return {
            "latents": latents,
            "latent_tokens": self.latent_fft.encode(latents),
            "mean": mean,
            "logvar": logvar,
        }


class ImageAutoencoderAdapter(nn.Module):
    """Adapt an existing real-spatial image AE to Hermitian latent FFT tokens."""

    def __init__(
        self,
        model: nn.Module,
        *,
        latent_height: int,
        latent_width: int,
        scaling_factor: Optional[float] = None,
        input_range: str = "minus_one_one",
    ):
        super().__init__()
        if input_range not in ("zero_one", "minus_one_one"):
            raise ValueError(f"Unknown image AE input range: {input_range}")
        self.model = model
        self.input_range = input_range
        configured_scale = getattr(getattr(model, "config", None), "scaling_factor", 1.0)
        self.scaling_factor = float(
            configured_scale if scaling_factor is None else scaling_factor
        )
        self.latent_fft = RealLatentFFT(latent_height, latent_width)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        latent_height: int,
        latent_width: int,
        subfolder: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "ImageAutoencoderAdapter":
        from diffusers import AutoencoderKL

        kwargs: Dict[str, Any] = {}
        if subfolder is not None:
            kwargs["subfolder"] = subfolder
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = AutoencoderKL.from_pretrained(model_name_or_path, **kwargs)
        return cls(
            model,
            latent_height=latent_height,
            latent_width=latent_width,
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        prepared = (
            images * 2.0 - 1.0
            if self.input_range == "minus_one_one"
            else images
        )
        parameter = next(self.model.parameters(), None)
        return (
            prepared
            if parameter is None
            else prepared.to(device=parameter.device, dtype=parameter.dtype)
        )

    def _restore_images(self, images: torch.Tensor) -> torch.Tensor:
        return (images + 1.0) * 0.5 if self.input_range == "minus_one_one" else images

    @staticmethod
    def _latent_distribution(encoded: Any) -> Any:
        return getattr(encoded, "latent_dist", encoded[0] if isinstance(encoded, tuple) else encoded)

    def encode(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior = self._latent_distribution(self.model.encode(self._prepare_images(images)))
        if sample_posterior and hasattr(posterior, "sample"):
            raw = posterior.sample()
        elif hasattr(posterior, "mode"):
            raw = posterior.mode()
        else:
            raw = posterior
        mean = getattr(posterior, "mean", raw)
        logvar = getattr(posterior, "logvar", torch.zeros_like(mean))
        return (
            raw * self.scaling_factor,
            mean * self.scaling_factor,
            logvar,
        )

    def decode(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        parameter = next(self.model.parameters(), None)
        model_latents = scaled_latents / self.scaling_factor
        if parameter is not None:
            model_latents = model_latents.to(
                device=parameter.device, dtype=parameter.dtype
            )
        decoded = self.model.decode(model_latents)
        sample = getattr(decoded, "sample", decoded[0] if isinstance(decoded, tuple) else decoded)
        return self._restore_images(sample)

    def forward(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = False,
        latent_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latents, mean, logvar = self.encode(
            images, sample_posterior=sample_posterior
        )
        tokens = self.latent_fft.encode(latents) if latent_tokens is None else latent_tokens
        roundtrip_latents = self.latent_fft.decode(tokens)
        reconstruction = self.decode(roundtrip_latents)
        return {
            "reconstruction": reconstruction,
            "latents": latents,
            "latent_tokens": tokens,
            "mean": mean,
            "logvar": logvar,
        }


class LatentFourierNormalizer(nn.Module):
    """Phase-preserving per-orbit normalization for real latent-map FFTs."""

    def __init__(
        self,
        bridge: RealLatentFFT,
        channels: int,
        *,
        center_ordinary: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.center_ordinary = center_ordinary
        self.eps = eps
        mask = torch.ones(bridge.seq_len, 2 * channels)
        mask[bridge.is_self_conjugate, channels:] = 0.0
        self.register_buffer("component_mask", mask)
        self.register_buffer("is_self_conjugate", bridge.is_self_conjugate.clone())
        self.register_buffer("mean", torch.zeros(bridge.seq_len, 2 * channels))
        self.register_buffer("scale", torch.ones(bridge.seq_len, channels))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != 2 * self.channels:
            raise ValueError(
                f"Expected [N,L,{2 * self.channels}] latent tokens, got {tuple(tokens.shape)}"
            )
        values = tokens.float()
        empirical_mean = values.mean(dim=0)
        mean = torch.zeros_like(empirical_mean)
        if self.center_ordinary:
            mean.copy_(empirical_mean)
        else:
            mean[self.is_self_conjugate, : self.channels] = empirical_mean[
                self.is_self_conjugate, : self.channels
            ]
        centered = values - mean[None]
        real = centered[..., : self.channels]
        imag = centered[..., self.channels :]
        paired_power = 0.5 * (real.square() + imag.square()).mean(dim=0)
        scale = paired_power.clamp_min(self.eps).sqrt()
        self_scale = real[:, self.is_self_conjugate].square().mean(dim=0).sqrt()
        scale[self.is_self_conjugate] = self_scale.clamp_min(self.eps)
        self.mean.copy_(mean * self.component_mask)
        self.scale.copy_(scale)
        self.is_fitted.fill_(True)

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("Latent Fourier normalizer has not been fitted")
        paired_scale = torch.cat([self.scale, self.scale], dim=-1)
        return (
            (tokens - self.mean[None].to(tokens.dtype))
            / paired_scale[None].to(tokens.dtype).clamp_min(self.eps)
            * self.component_mask[None].to(tokens.dtype)
        )

    def denormalize(self, tokens: torch.Tensor) -> torch.Tensor:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("Latent Fourier normalizer has not been fitted")
        paired_scale = torch.cat([self.scale, self.scale], dim=-1)
        return (
            tokens * paired_scale[None].to(tokens.dtype)
            + self.mean[None].to(tokens.dtype)
        ) * self.component_mask[None].to(tokens.dtype)


class LatentCausalProbe(nn.Module):
    """Small teacher-forced probe for next exported-latent predictability."""

    def __init__(self, latent_dim: int, width: int = 128):
        super().__init__()
        self.input = nn.Linear(latent_dim, width)
        self.rnn = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, latent_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(latents)
        shifted[:, 1:] = latents[:, :-1]
        hidden, _ = self.rnn(self.input(shifted))
        return self.output(hidden)

    def loss(self, latents: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(latents), latents)
