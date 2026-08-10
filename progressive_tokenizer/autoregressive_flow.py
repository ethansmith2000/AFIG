"""Autoregressive prior with a conditional rectified-flow token head."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .joint_flow import (
    FeedForward,
    QKNormAttention,
    Rotary1D,
    TimestepEmbedding,
    _norm,
)


@dataclass(frozen=True)
class AutoregressiveFlowConfig:
    sequence_length: int = 32
    token_dim: int = 64
    width: int = 512
    trunk_depth: int = 12
    head_depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    rope_theta: float = 10_000.0
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.sequence_length <= 0 or self.token_dim <= 0:
            raise ValueError("sequence_length and token_dim must be positive")
        if self.width <= 0 or self.width % self.num_heads:
            raise ValueError("width must be positive and divisible by num_heads")
        if (self.width // self.num_heads) % 2:
            raise ValueError("attention head width must be even for RoPE")
        if self.trunk_depth <= 0 or self.head_depth <= 0:
            raise ValueError("trunk_depth and head_depth must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    def fingerprint(self) -> dict:
        return asdict(self)


class CausalTrunkBlock(nn.Module):
    def __init__(self, config: AutoregressiveFlowConfig):
        super().__init__()
        self.attention_norm = _norm(config.width)
        self.attention = QKNormAttention(config.width, config.num_heads)
        self.ffn_norm = _norm(config.width)
        self.ffn = FeedForward(config.width, config.mlp_ratio)

    def forward(self, values: torch.Tensor, rope: Rotary1D) -> torch.Tensor:
        values = values + self.attention(
            self.attention_norm(values), rope, causal=True
        )
        return values + self.ffn(self.ffn_norm(values))


class CausalTokenTrunk(nn.Module):
    """Shifted-input causal trunk aligned to predict each target register."""

    def __init__(self, config: AutoregressiveFlowConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.token_dim, config.width)
        self.bos = nn.Parameter(torch.empty(1, 1, config.width))
        self.target_position = nn.Parameter(
            torch.empty(1, config.sequence_length, config.width)
        )
        self.rope = Rotary1D(
            config.sequence_length,
            config.width // config.num_heads,
            config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            CausalTrunkBlock(config) for _ in range(config.trunk_depth)
        )
        self.final_norm = _norm(config.width)
        nn.init.trunc_normal_(self.bos, std=0.02)
        nn.init.trunc_normal_(self.target_position, std=0.02)

    def forward(self, completed_tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.token_dim)
        if completed_tokens.ndim != 3 or tuple(completed_tokens.shape[1:]) != expected:
            raise ValueError(
                f"completed_tokens must have shape [B,{expected[0]},{expected[1]}]"
            )
        batch = completed_tokens.shape[0]
        shifted = self.input(completed_tokens[:, :-1])
        values = torch.cat((self.bos.expand(batch, -1, -1), shifted), dim=1)
        values = values + self.target_position
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                values = checkpoint(
                    block, values, self.rope, use_reentrant=False
                )
            else:
                values = block(values, self.rope)
        return self.final_norm(values)


class ConditionalFlowBlock(nn.Module):
    def __init__(self, width: int, ratio: float):
        super().__init__()
        self.norm = _norm(width)
        self.ffn = FeedForward(width, ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        modulated = self.norm(values) * (1.0 + scale) + shift
        return values + gate * self.ffn(modulated)


class ConditionalFlowHead(nn.Module):
    def __init__(self, config: AutoregressiveFlowConfig):
        super().__init__()
        width = config.width
        self.config = config
        self.input = nn.Linear(config.token_dim, width)
        self.time = TimestepEmbedding(width)
        self.condition_fusion = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.blocks = nn.ModuleList(
            ConditionalFlowBlock(width, config.mlp_ratio)
            for _ in range(config.head_depth)
        )
        self.final_norm = _norm(width)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 2 * width)
        )
        self.output = nn.Linear(width, config.token_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def predict_velocity(
        self,
        noisy_tokens: torch.Tensor,
        time: torch.Tensor,
        trunk_condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_tokens.shape[:-1] != time.shape:
            raise ValueError("time must match every noisy token except its channel axis")
        if trunk_condition.shape[:-1] != noisy_tokens.shape[:-1]:
            raise ValueError("trunk_condition leading axes must match noisy tokens")
        leading = noisy_tokens.shape[:-1]
        noisy_flat = noisy_tokens.reshape(-1, noisy_tokens.shape[-1])
        condition_flat = trunk_condition.reshape(-1, trunk_condition.shape[-1])
        time_flat = time.reshape(-1)
        condition = self.condition_fusion(
            torch.cat((condition_flat, self.time(time_flat)), dim=-1)
        )
        values = self.input(noisy_flat)
        for block in self.blocks:
            values = block(values, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        modulated = self.final_norm(values) * (1.0 + scale) + shift
        output = self.output(modulated)
        return output.reshape(*leading, self.config.token_dim)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        steps: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError("single-token sampling condition must be [B,W]")
        parameter = self.input.weight
        values = torch.randn(
            condition.shape[0],
            self.config.token_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (values.shape[0],),
                index / steps,
                device=values.device,
                dtype=torch.float32,
            )
            velocity = self.predict_velocity(values, time, condition)
            if index + 1 < steps:
                proposal = values + step_size * velocity
                next_time = torch.full_like(time, (index + 1) / steps)
                next_velocity = self.predict_velocity(proposal, next_time, condition)
                values = values + 0.5 * step_size * (velocity + next_velocity)
            else:
                values = values + step_size * velocity
        return values


class AutoregressiveRectifiedFlow(nn.Module):
    def __init__(self, config: Optional[AutoregressiveFlowConfig] = None):
        super().__init__()
        self.config = config or AutoregressiveFlowConfig()
        self.trunk = CausalTokenTrunk(self.config)
        self.head = ConditionalFlowHead(self.config)

    def forward(
        self,
        clean_tokens: torch.Tensor,
        *,
        time: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        condition = self.trunk(clean_tokens)
        if time is None:
            time = torch.rand(clean_tokens.shape[:-1], device=clean_tokens.device)
        if noise is None:
            noise = torch.randn_like(clean_tokens)
        noisy = (1.0 - time[..., None]) * noise + time[..., None] * clean_tokens
        target = clean_tokens - noise
        prediction = self.head.predict_velocity(noisy, time, condition)
        squared_error = (prediction.float() - target.float()).square()
        return {
            "loss": squared_error.mean(),
            "per_token_mse": squared_error.mean(dim=(0, 2)).detach(),
            "prediction_rms": prediction.float().square().mean().sqrt().detach(),
            "target_rms": target.float().square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        *,
        steps: int = 50,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        parameter = self.trunk.input.weight
        tokens = torch.zeros(
            batch_size,
            self.config.sequence_length,
            self.config.token_dim,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        for index in range(self.config.sequence_length):
            condition = self.trunk(tokens)[:, index]
            tokens[:, index] = self.head.sample(
                condition, steps=steps, generator=generator
            )
        return tokens
