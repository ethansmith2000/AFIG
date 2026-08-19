"""Rolling (per-token-time) rectified flow over progressive-token sequences.

Reimplementation of the lost 2026-08-14 rolling prior from its recovered W&B
spec: a headless bidirectional transformer identical to the joint prior except
that every AdaLN condition is per token. Training samples a frontier position
and assigns each register the local data time

    t_i = clamp(frontier - i / overlap, 0, 1)

so early registers denoise before later ones; the loss is flat MSE over the
active registers only (those with 0 < t_i < 1). Sampling advances the frontier
from 0 to its full duration, giving each register `steps_per_token` solver
steps while it is active. With overlap == sequence_length the schedule is a
full-sequence skew; a small overlap approaches one-register-at-a-time AR with
full attention over partially denoised history.
"""

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
class RollingFlowConfig:
    sequence_length: int = 64
    token_dim: int = 16
    width: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qk_norm: str = "rms"
    rope_theta: float = 10_000.0
    overlap: float = 8.0
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
        if self.overlap <= 0:
            raise ValueError("overlap must be positive")

    @property
    def frontier_duration(self) -> float:
        return (self.sequence_length - 1) / self.overlap + 1.0

    def fingerprint(self) -> dict:
        return asdict(self)


def _modulate_tokens(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale) + shift


class PerTokenAdaLNZeroBlock(nn.Module):
    """AdaLN-Zero block whose condition varies per token ([B, N, W])."""

    def __init__(self, config: RollingFlowConfig):
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
        values = values + attention_gate * self.attention(
            _modulate_tokens(
                self.attention_norm(values), attention_shift, attention_scale
            ),
            rope,
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


class RollingRectifiedFlow(nn.Module):
    """Bidirectional velocity field with an independent time per register."""

    def __init__(self, config: Optional[RollingFlowConfig] = None):
        super().__init__()
        self.config = config or RollingFlowConfig()
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
            PerTokenAdaLNZeroBlock(config) for _ in range(config.depth)
        )
        self.final = PerTokenFinalLayer(config.width, config.token_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def local_times(self, frontier: torch.Tensor) -> torch.Tensor:
        """Per-register data time for frontier positions [B]."""

        if frontier.ndim != 1:
            raise ValueError("frontier must have shape [B]")
        index_time = (
            torch.arange(
                self.config.sequence_length,
                device=frontier.device,
                dtype=torch.float32,
            )
            / self.config.overlap
        )
        return (frontier.float()[:, None] - index_time[None, :]).clamp(0.0, 1.0)

    def predict_velocity(
        self, noisy_latents: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.config.sequence_length, self.config.token_dim)
        if noisy_latents.ndim != 3 or tuple(noisy_latents.shape[1:]) != expected:
            raise ValueError(
                f"noisy_latents must have shape [B,{expected[0]},{expected[1]}]"
            )
        if times.shape != noisy_latents.shape[:2]:
            raise ValueError("times must have shape [B, sequence_length]")
        condition = self.time(times.reshape(-1)).reshape(
            times.shape[0], times.shape[1], -1
        )
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
        frontier: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean_latents.shape[0]
        if frontier is None:
            frontier = (
                torch.rand(batch, device=clean_latents.device)
                * self.config.frontier_duration
            )
        if noise is None:
            noise = torch.randn_like(clean_latents)
        times = self.local_times(frontier)
        time_view = times[..., None].to(clean_latents.dtype)
        noisy = (1.0 - time_view) * noise + time_view * clean_latents
        target = clean_latents - noise
        prediction = self.predict_velocity(noisy, times)
        squared_error = (prediction.float() - target.float()).square()
        active = ((times > 0.0) & (times < 1.0)).float()
        active_scalars = (active.sum() * clean_latents.shape[-1]).clamp_min(1.0)
        loss = (squared_error * active[..., None]).sum() / active_scalars
        per_token_active = active.sum(dim=0)
        per_token_mse = (squared_error.mean(dim=2) * active).sum(
            dim=0
        ) / per_token_active.clamp_min(1.0)
        return {
            "loss": loss,
            "per_token_mse": per_token_mse.detach(),
            "per_token_active": per_token_active.detach(),
            "prediction_rms": (
                (prediction.float().square() * active[..., None]).sum()
                / active_scalars
            )
            .sqrt()
            .detach(),
            "target_rms": (
                (target.float().square() * active[..., None]).sum() / active_scalars
            )
            .sqrt()
            .detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        steps_per_token: int = 50,
        solver: str = "heun",
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if steps_per_token <= 0:
            raise ValueError("steps_per_token must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        duration = self.config.frontier_duration
        total_steps = max(1, round(steps_per_token * duration))
        parameter = self.input.weight
        values = torch.randn(
            batch_size,
            self.config.sequence_length,
            self.config.token_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        for index in range(total_steps):
            frontier_now = torch.full(
                (batch_size,),
                index * duration / total_steps,
                device=values.device,
                dtype=torch.float32,
            )
            frontier_next = torch.full_like(
                frontier_now, (index + 1) * duration / total_steps
            )
            times_now = self.local_times(frontier_now)
            times_next = self.local_times(frontier_next)
            step_sizes = (times_next - times_now)[..., None].to(values.dtype)
            velocity = self.predict_velocity(values, times_now)
            if solver == "heun" and index + 1 < total_steps:
                proposal = values + step_sizes * velocity
                next_velocity = self.predict_velocity(proposal, times_next)
                values = values + 0.5 * step_sizes * (velocity + next_velocity)
            else:
                values = values + step_sizes * velocity
        return values
