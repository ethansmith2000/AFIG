"""Full-sequence diffusion over frozen autoencoder latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from causal_transformer import (
    CausalTransformerBlock,
    CausalTransformerConfig,
    build_rope_tables,
)
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
    # Per-position conditioning.  The per-block FiLM path already accepts a
    # [B,L,W] condition, but it is fed a position-constant timestep embedding, so
    # attention is permutation-equivariant apart from the metadata channels.
    # These add a learned identity signal at the input and/or at every block.
    position_embedding_input: bool = False
    position_embedding_film: bool = False
    # Training-time timestep distribution.  "uniform" reproduces prior runs.
    # "snr_interpolate" draws t = u**(1 + 2*alpha), concentrating samples toward
    # t -> 0 (high noise), where the model measurably fails to beat a linear
    # Gaussian predictor and where sampling commits global structure.
    timestep_sampling: str = "uniform"
    timestep_sampling_alpha: float = 0.0
    # Rotary embeddings on q/k only, leaving the residual stream untouched.
    # Absolute position embeddings give identifiability; RoPE shapes attention
    # geometry.  "radius_angle" uses each latent's pooled frequency-space
    # coordinates, which 1-D over sequence index would conflate (radial ordering
    # mixes ring with sector slot).
    rope: str = "none"  # none | sequence | radius_angle
    rope_base: float = 10000.0

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
        # Default False keeps checkpoints written before these options loadable.
        position_embedding_input=bool(payload.get("position_embedding_input", False)),
        position_embedding_film=bool(payload.get("position_embedding_film", False)),
        timestep_sampling=str(payload.get("timestep_sampling", "uniform")),
        timestep_sampling_alpha=float(payload.get("timestep_sampling_alpha", 0.0)),
        rope=str(payload.get("rope", "none")),
        rope_base=float(payload.get("rope_base", 10000.0)),
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

        # Learned absolute identity per latent position.  Zero-initialized so the
        # model starts identical to the position-constant baseline.
        self.position_embedding_input = None
        self.position_embedding_film = None
        if cfg.position_embedding_input:
            self.position_embedding_input = nn.Parameter(
                torch.zeros(cfg.sequence_length, width)
            )
        if cfg.position_embedding_film:
            self.position_embedding_film = nn.Parameter(
                torch.zeros(cfg.sequence_length, width)
            )

    def _rope_tables(
        self, metadata: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Build rotary tables from the per-position frequency coordinates.

        POSITION_FEATURE_SCHEMA index 5/6 are angle_center_sin/cos and index 8 is
        radius_center, so the pooled polar coordinates of each latent's sector are
        available directly.  Angle is recovered with atan2 and used in radians;
        radius is already normalized to [0,1] and is scaled up so neighbouring
        rings are distinguishable at the lowest rotary frequency.
        """
        mode = self.config.rope
        if mode == "none":
            return None
        cfg = self.config
        head_dim = cfg.transformer.width // cfg.transformer.num_heads
        positions = metadata[0] if metadata.ndim == 3 else metadata
        if mode == "sequence":
            coordinates = torch.arange(
                cfg.sequence_length, device=positions.device, dtype=torch.float32
            )
        elif mode == "radius_angle":
            angle = torch.atan2(positions[:, 5], positions[:, 6])
            radius = positions[:, 8] * float(cfg.sequence_length)
            coordinates = torch.stack([radius, angle], dim=-1)
        else:
            raise ValueError(f"Unknown rope mode: {mode}")
        return build_rope_tables(coordinates, head_dim, cfg.rope_base)

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
        # Preserve the source coordinate precision for RoPE.  The metadata copy
        # concatenated with activations may be bf16, but rotary angles must be
        # derived from fp32 coordinates before that cast.
        rope = self._rope_tables(
            metadata.to(device=noisy_latents.device, dtype=torch.float32)
        )
        metadata = self._metadata_batch(noisy_latents, metadata)
        hidden = self.input_projection(torch.cat([noisy_latents, metadata], dim=-1))
        if self.position_embedding_input is not None:
            hidden = hidden + self.position_embedding_input.to(hidden.dtype)
        timestep = flow_time * float(self.config.num_train_timesteps - 1)
        condition = self.time_embed(timestep).unsqueeze(1).expand_as(hidden)
        if self.position_embedding_film is not None:
            # Makes the per-block FiLM modulation position-dependent instead of
            # broadcasting one timestep embedding to all 53 positions.
            condition = condition + self.position_embedding_film.to(condition.dtype)
        for layer in self.layers:
            if self.config.transformer.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda value, cond, layer=layer: layer(value, cond, rope=rope)[0],
                    hidden,
                    condition,
                    use_reentrant=False,
                )
            else:
                hidden, _ = layer(hidden, condition=condition, rope=rope)
        return self.final_layer(hidden, condition)

    def forward(
        self, latents: torch.Tensor, metadata: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch = latents.shape[0]
        if self.config.timestep_sampling == "uniform":
            timestep = torch.randint(
                0,
                self.config.num_train_timesteps,
                (batch,),
                device=latents.device,
            )
        elif self.config.timestep_sampling == "snr_interpolate":
            uniform = torch.rand(batch, device=latents.device)
            skewed = uniform.pow(1.0 + 2.0 * self.config.timestep_sampling_alpha)
            timestep = (skewed * self.config.num_train_timesteps).long().clamp_(
                0, self.config.num_train_timesteps - 1
            )
        else:
            raise ValueError(f"Unknown timestep_sampling: {self.config.timestep_sampling}")
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
