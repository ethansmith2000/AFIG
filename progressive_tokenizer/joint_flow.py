"""Joint rectified flow for complete progressive-token sequences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class JointFlowConfig:
    sequence_length: int = 32
    token_dim: int = 64
    width: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qk_norm: str = "rms"
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.sequence_length <= 0 or self.token_dim <= 0:
            raise ValueError("sequence_length and token_dim must be positive")
        if self.width <= 0 or self.width % self.num_heads:
            raise ValueError("width must be positive and divisible by num_heads")
        if (self.width // self.num_heads) % 2:
            raise ValueError("attention head width must be even for RoPE")
        if self.depth <= 0 or self.mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive")
        if self.qk_norm not in {"rms", "l2_temperature"}:
            raise ValueError("qk_norm must be rms or l2_temperature")

    def fingerprint(self) -> dict:
        return asdict(self)


def _norm(width: int) -> nn.LayerNorm:
    return nn.LayerNorm(width, elementwise_affine=False)


class Rotary1D(nn.Module):
    """Fixed float32 rotary tables for the ordered latent slots."""

    def __init__(self, length: int, head_dim: int, theta: float):
        super().__init__()
        inv_frequency = theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        positions = torch.arange(length, dtype=torch.float32)
        angles = positions[:, None] * inv_frequency[None, :]
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
        paired = values.reshape(*values.shape[:-1], -1, 2)
        first, second = paired.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-2:] != self.cos.shape[-2:]:
            raise ValueError("attention tensor does not match the RoPE table")
        rotated = (
            values.float() * self.cos
            + self._rotate_pairs(values.float()) * self.sin
        )
        return rotated.to(values.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, width: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = time.float()[:, None] * frequencies[None, :] * 1000.0
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class QKNormAttention(nn.Module):
    def __init__(self, width: int, num_heads: int, qk_norm: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
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
        self, values: torch.Tensor, rope: Rotary1D, *, causal: bool = False
    ) -> torch.Tensor:
        batch, length, width = values.shape
        qkv = self.qkv(values).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self.query_norm(query)
        key = self.key_norm(key)
        query = rope(query)
        key = rope(key)
        attention_scale = None
        if self.qk_norm == "l2_temperature":
            query = F.normalize(query.float(), dim=-1).to(value.dtype)
            key = F.normalize(key.float(), dim=-1).to(value.dtype)
            scale = self.logit_scale.exp().clamp(max=100.0).to(value.dtype)
            query = query * scale[None, :, None, None]
            attention_scale = 1.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=causal,
            scale=attention_scale,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, width: int, ratio: float):
        super().__init__()
        hidden = int(width * ratio)
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(values).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


def _modulate(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale[:, None]) + shift[:, None]


class AdaLNZeroBlock(nn.Module):
    def __init__(self, config: JointFlowConfig):
        super().__init__()
        self.attention_norm = _norm(config.width)
        self.attention = QKNormAttention(
            config.width, config.num_heads, config.qk_norm
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(config.width, config.mlp_ratio)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.width, 6 * config.width)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, values: torch.Tensor, condition: torch.Tensor, rope: Rotary1D
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.modulation(condition).chunk(6, dim=-1)
        values = values + attention_gate[:, None] * self.attention(
            _modulate(self.attention_norm(values), attention_shift, attention_scale),
            rope,
        )
        return values + ffn_gate[:, None] * self.ffn(
            _modulate(self.ffn_norm(values), ffn_shift, ffn_scale)
        )


class FinalLayer(nn.Module):
    def __init__(self, width: int, token_dim: int):
        super().__init__()
        self.norm = _norm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, token_dim)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.output(_modulate(self.norm(values), shift, scale))


class JointRectifiedFlow(nn.Module):
    """Bidirectional DiT-style velocity field over all latent tokens."""

    def __init__(self, config: Optional[JointFlowConfig] = None):
        super().__init__()
        self.config = config or JointFlowConfig()
        config = self.config
        self.input = nn.Linear(config.token_dim, config.width)
        self.position = nn.Parameter(
            torch.empty(1, config.sequence_length, config.width)
        )
        self.time = TimestepEmbedding(config.width)
        self.rope = Rotary1D(
            config.sequence_length,
            config.width // config.num_heads,
            config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            AdaLNZeroBlock(config) for _ in range(config.depth)
        )
        self.final = FinalLayer(config.width, config.token_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def predict_velocity(
        self, noisy_latents: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.token_dim)
        if noisy_latents.ndim != 3 or tuple(noisy_latents.shape[1:]) != expected:
            raise ValueError(f"noisy_latents must have shape [B,{expected[0]},{expected[1]}]")
        condition = self.time(time)
        values = self.input(noisy_latents) + self.position
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                values = checkpoint(
                    block, values, condition, self.rope, use_reentrant=False
                )
            else:
                values = block(values, condition, self.rope)
        return self.final(values, condition)

    def forward(
        self,
        clean_latents: torch.Tensor,
        *,
        time: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean_latents.shape[0]
        if time is None:
            time = torch.rand(batch, device=clean_latents.device)
        if noise is None:
            noise = torch.randn_like(clean_latents)
        time_view = time[:, None, None].to(clean_latents.dtype)
        noisy = (1.0 - time_view) * noise + time_view * clean_latents
        target = clean_latents - noise
        prediction = self.predict_velocity(noisy, time)
        squared_error = (prediction.float() - target.float()).square()
        return {
            "loss": squared_error.mean(),
            "per_token_mse": squared_error.mean(dim=(0, 2)).detach(),
            "prediction_rms": prediction.float().square().mean().sqrt().detach(),
            "target_rms": target.float().square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        steps: int = 50,
        solver: str = "heun",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        parameter = self.input.weight
        values = torch.randn(
            batch_size,
            self.config.sequence_length,
            self.config.token_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (batch_size,), index / steps, device=values.device, dtype=torch.float32
            )
            velocity = self.predict_velocity(values, time)
            if solver == "heun" and index + 1 < steps:
                proposal = values + step_size * velocity
                next_time = torch.full_like(time, (index + 1) / steps)
                next_velocity = self.predict_velocity(proposal, next_time)
                values = values + 0.5 * step_size * (velocity + next_velocity)
            else:
                values = values + step_size * velocity
        return values


# --- per-register conditioning -------------------------------------------
# Retained from the removed rolling engine: a static per-register timestep
# offset (t_i = clamp(t + delta_i, 0, 1)) needs an AdaLN condition that varies
# along the sequence, [B, N, W] rather than [B, W]. The rolling/diffusion-
# forcing machinery itself is gone; see commit f906d4a.

def _modulate_tokens(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale) + shift


class PerTokenAdaLNZeroBlock(nn.Module):
    """AdaLN-Zero block whose condition varies per token ([B, N, W])."""

    def __init__(self, config, causal: bool = False):
        super().__init__()
        self.attention_norm = _norm(config.width)
        self.attention = QKNormAttention(
            config.width, config.num_heads, config.qk_norm
        )
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(config.width, config.mlp_ratio)
        # taken as an explicit argument: JointFlowConfig has no `causal`
        # field, and reading one off the config raised AttributeError on the
        # first instantiation after the rolling engine was removed.
        self.causal = causal
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.width, 6 * config.width)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, values: torch.Tensor, condition: torch.Tensor, rope: Rotary1D
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.modulation(condition).chunk(6, dim=-1)
        values = values + attention_gate * self.attention(
            _modulate_tokens(
                self.attention_norm(values), attention_shift, attention_scale
            ),
            rope,
            causal=self.causal,
        )
        return values + ffn_gate * self.ffn(
            _modulate_tokens(self.ffn_norm(values), ffn_shift, ffn_scale)
        )


class PerTokenFinalLayer(nn.Module):
    def __init__(self, width: int, token_dim: int):
        super().__init__()
        self.norm = _norm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, token_dim)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.output(_modulate_tokens(self.norm(values), shift, scale))
