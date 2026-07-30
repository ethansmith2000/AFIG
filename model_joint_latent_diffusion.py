"""Full-sequence diffusion over frozen autoencoder latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from causal_transformer import CausalTransformerBlock, CausalTransformerConfig
from diffusion_decoder import FinalLayer, TimestepEmbedder
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


@dataclass
class JointLatentDiffusionConfig:
    sequence_length: int = LATENT_SEQUENCE_LENGTH
    token_dim: int = LATENT_TOKEN_DIM
    metadata_dim: int = 11
    transformer: CausalTransformerConfig = field(
        default_factory=CausalTransformerConfig
    )
    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    flow_solver: str = "heun"

    def fingerprint(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["transformer"] = self.transformer.fingerprint()
        return payload


def joint_config_from_dict(payload: Dict[str, Any]) -> JointLatentDiffusionConfig:
    return JointLatentDiffusionConfig(
        sequence_length=int(payload["sequence_length"]),
        token_dim=int(payload["token_dim"]),
        metadata_dim=int(payload["metadata_dim"]),
        transformer=CausalTransformerConfig(**payload["transformer"]),
        num_train_timesteps=int(payload["num_train_timesteps"]),
        num_inference_steps=int(payload["num_inference_steps"]),
        flow_solver=str(payload["flow_solver"]),
    )


class JointLatentDiffusionModel(nn.Module):
    """Bidirectional DiT-style rectified flow on all latent tokens at once."""

    def __init__(self, config: Optional[JointLatentDiffusionConfig] = None):
        super().__init__()
        self.config = config or JointLatentDiffusionConfig()
        cfg = self.config
        if cfg.sequence_length != LATENT_SEQUENCE_LENGTH:
            raise ValueError(f"Expected {LATENT_SEQUENCE_LENGTH} latent tokens")
        if cfg.token_dim != LATENT_TOKEN_DIM:
            raise ValueError(f"Expected {LATENT_TOKEN_DIM}-D latent tokens")
        if cfg.transformer.max_seq_len < cfg.sequence_length:
            raise ValueError("Transformer max_seq_len is shorter than the latent sequence")
        if cfg.flow_solver not in ("euler", "heun"):
            raise ValueError("flow_solver must be euler or heun")

        width = cfg.transformer.width
        self.input_projection = nn.Linear(cfg.token_dim + cfg.metadata_dim, width)
        self.time_embed = TimestepEmbedder(width)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=cfg.transformer.num_heads,
                    ff_mult=cfg.transformer.ff_mult,
                    dropout=cfg.transformer.dropout,
                    conditional_film=True,
                    causal=False,
                )
                for _ in range(cfg.transformer.num_layers)
            ]
        )
        self.final_layer = FinalLayer(width, cfg.token_dim)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _metadata_batch(
        self, latents: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(latents.shape[0], -1, -1)
        expected = (
            latents.shape[0],
            self.config.sequence_length,
            self.config.metadata_dim,
        )
        if metadata.shape != expected:
            raise ValueError("metadata must be [L,M] or [B,L,M]")
        return metadata.to(device=latents.device, dtype=latents.dtype)

    def predict_velocity(
        self,
        noisy_latents: torch.Tensor,
        flow_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latents.ndim != 3 or noisy_latents.shape[1:] != (
            self.config.sequence_length,
            self.config.token_dim,
        ):
            raise ValueError("noisy_latents must be [B,53,64]")
        if flow_time.shape != (noisy_latents.shape[0],):
            raise ValueError("flow_time must be [B]")
        metadata = self._metadata_batch(noisy_latents, metadata)
        hidden = self.input_projection(torch.cat([noisy_latents, metadata], dim=-1))
        timestep = flow_time * float(self.config.num_train_timesteps - 1)
        condition = self.time_embed(timestep).unsqueeze(1).expand_as(hidden)
        for layer in self.layers:
            if self.config.transformer.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda value, cond: layer(value, cond)[0],
                    hidden,
                    condition,
                    use_reentrant=False,
                )
            else:
                hidden, _ = layer(hidden, condition=condition)
        return self.final_layer(hidden, condition)

    def forward(
        self, latents: torch.Tensor, metadata: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch = latents.shape[0]
        timestep = torch.randint(
            0,
            self.config.num_train_timesteps,
            (batch,),
            device=latents.device,
        )
        flow_time = (timestep.float() + 0.5) / float(
            self.config.num_train_timesteps
        )
        noise = torch.randn_like(latents)
        time_view = flow_time[:, None, None]
        noisy = time_view * latents + (1.0 - time_view) * noise
        target_velocity = latents - noise
        prediction = self.predict_velocity(noisy, flow_time, metadata)
        per_component = (prediction.float() - target_velocity.float()).square()
        per_position = per_component.mean(dim=-1)
        return {
            "loss": per_position.mean(),
            "per_position": per_position.detach(),
            "unweighted_mse": per_component.mean().detach(),
            "prediction_rms": prediction.float().square().mean().sqrt().detach(),
            "target_rms": target_velocity.float().square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def generate_latents(
        self,
        batch_size: int,
        metadata: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        steps = num_inference_steps or self.config.num_inference_steps
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        parameter = self.input_projection.weight
        sample = torch.randn(
            batch_size,
            self.config.sequence_length,
            self.config.token_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        dt = 1.0 / float(steps)
        for index in range(steps):
            time = torch.full(
                (batch_size,),
                index / float(steps),
                device=sample.device,
                dtype=torch.float32,
            )
            velocity = self.predict_velocity(sample, time, metadata)
            if self.config.flow_solver == "heun" and index + 1 < steps:
                proposal = sample + dt * velocity
                next_time = torch.full_like(time, (index + 1) / float(steps))
                next_velocity = self.predict_velocity(proposal, next_time, metadata)
                sample = sample + 0.5 * dt * (velocity + next_velocity)
            else:
                sample = sample + dt * velocity
        return sample
