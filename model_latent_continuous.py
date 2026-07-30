"""Minimal continuous autoregressive model for target-12 AE latents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from causal_transformer import (
    CausalTransformerBlock,
    CausalTransformerConfig,
    KVCache,
)
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig


LATENT_SEQUENCE_LENGTH = 53
LATENT_TOKEN_DIM = 64


@dataclass
class LatentContinuousConfig:
    sequence_length: int = LATENT_SEQUENCE_LENGTH
    token_dim: int = LATENT_TOKEN_DIM
    metadata_dim: int = 11
    transformer: CausalTransformerConfig = field(
        default_factory=CausalTransformerConfig
    )
    diffusion: DiffusionDecoderConfig = field(
        default_factory=lambda: DiffusionDecoderConfig(
            target_dim=LATENT_TOKEN_DIM,
            z_channels=512,
            target_condition_dim=11,
            condition_fusion="concat_mlp",
            width=512,
            depth=3,
        )
    )
    transformer_metadata_film: bool = False
    context_dropout_probability: float = 0.1
    latent_loss_weighting: str = "unweighted"

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "token_dim": self.token_dim,
            "metadata_dim": self.metadata_dim,
            "transformer": self.transformer.fingerprint(),
            "diffusion": self.diffusion.fingerprint(),
            "transformer_metadata_film": self.transformer_metadata_film,
            "context_dropout_probability": self.context_dropout_probability,
            "latent_loss_weighting": self.latent_loss_weighting,
        }


def latent_config_from_dict(payload: Dict[str, Any]) -> LatentContinuousConfig:
    return LatentContinuousConfig(
        sequence_length=int(payload["sequence_length"]),
        token_dim=int(payload["token_dim"]),
        metadata_dim=int(payload["metadata_dim"]),
        transformer=CausalTransformerConfig(**payload["transformer"]),
        diffusion=DiffusionDecoderConfig(**payload["diffusion"]),
        transformer_metadata_film=bool(payload["transformer_metadata_film"]),
        context_dropout_probability=float(payload["context_dropout_probability"]),
        latent_loss_weighting=str(payload.get("latent_loss_weighting", "unweighted")),
    )


class LatentContinuousModel(nn.Module):
    """Causal Transformer plus per-token diffusion decoder.

    Input position ``i`` contains latent ``i-1`` (or zeros at BOS), metadata
    for target ``i``, and a BOS flag. There are no learned position tables.
    """

    def __init__(
        self,
        config: Optional[LatentContinuousConfig] = None,
        loss_component_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config or LatentContinuousConfig()
        cfg = self.config
        if cfg.sequence_length != LATENT_SEQUENCE_LENGTH:
            raise ValueError(f"Expected exactly {LATENT_SEQUENCE_LENGTH} latent tokens")
        if cfg.token_dim != LATENT_TOKEN_DIM:
            raise ValueError(f"Expected exactly {LATENT_TOKEN_DIM}-D latent tokens")
        if cfg.transformer.max_seq_len < cfg.sequence_length:
            raise ValueError("Transformer max_seq_len is shorter than the latent sequence")
        if cfg.diffusion.target_dim != cfg.token_dim:
            raise ValueError("Diffusion target_dim must match latent token_dim")
        if cfg.diffusion.z_channels != cfg.transformer.width:
            raise ValueError("Diffusion z_channels must match Transformer width")
        if cfg.diffusion.target_condition_dim != cfg.metadata_dim:
            raise ValueError("Diffusion target_condition_dim must match metadata_dim")
        if cfg.diffusion.condition_fusion != "concat_mlp":
            raise ValueError("Latent decoder requires condition_fusion='concat_mlp'")
        if cfg.latent_loss_weighting not in (
            "unweighted",
            "raw_variance",
            "decoder_sensitivity",
        ):
            raise ValueError(
                f"Unknown latent_loss_weighting={cfg.latent_loss_weighting}"
            )
        if cfg.latent_loss_weighting == "unweighted":
            weights = torch.ones(cfg.sequence_length, cfg.token_dim)
        else:
            if loss_component_weights is None:
                raise ValueError(
                    f"{cfg.latent_loss_weighting} requires loss component weights"
                )
            weights = loss_component_weights.detach().float()
            if weights.shape != (cfg.sequence_length, cfg.token_dim):
                raise ValueError("loss component weights must be [53,64]")
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("loss component weights must be finite and positive")
            weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer(
            "loss_component_weights", weights, persistent=True
        )

        width = cfg.transformer.width
        self.input_projection = nn.Linear(
            cfg.token_dim + cfg.metadata_dim + 1, width
        )
        self.metadata_projection = (
            nn.Linear(cfg.metadata_dim, width, bias=False)
            if cfg.transformer_metadata_film
            else None
        )
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=cfg.transformer.num_heads,
                    ff_mult=cfg.transformer.ff_mult,
                    dropout=cfg.transformer.dropout,
                    conditional_film=cfg.transformer_metadata_film,
                )
                for _ in range(cfg.transformer.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(width)
        self.null_context = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.null_context, std=0.02)
        self.diffusion = DiffusionDecoder(cfg.diffusion)

    def _validate_inputs(
        self, tokens: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (
            self.config.sequence_length,
            self.config.token_dim,
        ):
            raise ValueError(
                f"tokens must be [B,{self.config.sequence_length},"
                f"{self.config.token_dim}], got {tuple(tokens.shape)}"
            )
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        if metadata.shape != (
            tokens.shape[0],
            self.config.sequence_length,
            self.config.metadata_dim,
        ):
            raise ValueError("metadata must be [L,M] or [B,L,M]")
        return metadata.to(device=tokens.device, dtype=tokens.dtype)

    def shifted_features(
        self, tokens: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        metadata = self._validate_inputs(tokens, metadata)
        previous = torch.zeros_like(tokens)
        previous[:, 1:] = tokens[:, :-1]
        bos = torch.zeros(
            tokens.shape[0],
            self.config.sequence_length,
            1,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        bos[:, 0] = 1.0
        return torch.cat([previous, metadata, bos], dim=-1)

    def shifted_inputs(
        self, tokens: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        return self.input_projection(self.shifted_features(tokens, metadata))

    def forward_backbone(
        self,
        inputs: torch.Tensor,
        metadata: torch.Tensor,
        kv_caches: Optional[List[Optional[KVCache]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]]]:
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        condition = (
            self.metadata_projection(metadata)
            if self.metadata_projection is not None
            else None
        )
        caches: List[KVCache] = []
        hidden = inputs
        for index, layer in enumerate(self.layers):
            cache = None if kv_caches is None else kv_caches[index]
            if (
                self.config.transformer.gradient_checkpointing
                and self.training
                and not use_cache
            ):
                if condition is None:
                    hidden = checkpoint(
                        lambda value: layer(value)[0],
                        hidden,
                        use_reentrant=False,
                    )
                else:
                    hidden = checkpoint(
                        lambda value, cond: layer(value, cond)[0],
                        hidden,
                        condition,
                        use_reentrant=False,
                    )
                new_cache = None
            else:
                hidden, new_cache = layer(
                    hidden,
                    condition=condition,
                    kv_cache=cache,
                    use_cache=use_cache,
                )
            if new_cache is not None:
                caches.append(new_cache)
        return self.final_norm(hidden), caches if use_cache else None

    def apply_context_dropout(
        self,
        hidden: torch.Tensor,
        force_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if force_mask is None:
            probability = (
                self.config.context_dropout_probability if self.training else 0.0
            )
            mask = torch.rand(
                hidden.shape[0], hidden.shape[1], 1, device=hidden.device
            ) < probability
        else:
            mask = force_mask.to(device=hidden.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            if mask.shape != hidden.shape[:2] + (1,):
                raise ValueError("context dropout mask must be [B,L] or [B,L,1]")
        null = self.null_context.to(dtype=hidden.dtype).view(1, 1, -1)
        return torch.where(mask, null, hidden), mask

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        context_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        metadata = self._validate_inputs(tokens, metadata)
        inputs = self.shifted_inputs(tokens, metadata)
        hidden, _ = self.forward_backbone(inputs, metadata)
        conditioned, dropped = self.apply_context_dropout(
            hidden, force_mask=context_dropout_mask
        )
        output = self.diffusion.compute_loss(
            target=tokens,
            z=conditioned,
            target_condition=metadata,
            component_metric=(
                self.loss_component_weights / float(self.config.token_dim)
                if self.config.latent_loss_weighting != "unweighted"
                else None
            ),
        )
        output["context_drop_fraction"] = dropped.float().mean().detach()
        output["context_null_gap"] = (
            hidden - self.null_context.to(dtype=hidden.dtype)
        ).square().mean().sqrt().detach()
        output["hidden"] = hidden
        return output

    @torch.no_grad()
    def init_cache(
        self, batch_size: int, metadata: torch.Tensor
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        if metadata.shape != (
            self.config.sequence_length,
            self.config.metadata_dim,
        ):
            raise ValueError("metadata must be [L,M]")
        dtype = self.input_projection.weight.dtype
        device = self.input_projection.weight.device
        target_metadata = metadata[0].to(device=device, dtype=dtype)
        previous = torch.zeros(batch_size, self.config.token_dim, device=device, dtype=dtype)
        bos = torch.ones(batch_size, 1, device=device, dtype=dtype)
        inputs = self.input_projection(
            torch.cat(
                [previous, target_metadata[None].expand(batch_size, -1), bos],
                dim=-1,
            )
        ).unsqueeze(1)
        hidden, caches = self.forward_backbone(
            inputs,
            target_metadata.view(1, 1, -1).expand(batch_size, -1, -1),
            use_cache=True,
        )
        assert caches is not None
        return hidden[:, 0], caches

    @torch.no_grad()
    def forward_step(
        self,
        previous_latent: torch.Tensor,
        target_index: int,
        target_metadata: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        if not 0 < target_index < self.config.sequence_length:
            raise ValueError("target_index must be in [1, sequence_length)")
        metadata = target_metadata.to(
            device=previous_latent.device, dtype=previous_latent.dtype
        )
        bos = torch.zeros(previous_latent.shape[0], 1, device=previous_latent.device, dtype=previous_latent.dtype)
        inputs = self.input_projection(
            torch.cat(
                [previous_latent, metadata[None].expand(previous_latent.shape[0], -1), bos],
                dim=-1,
            )
        ).unsqueeze(1)
        hidden, caches = self.forward_backbone(
            inputs,
            metadata.view(1, 1, -1).expand(previous_latent.shape[0], -1, -1),
            kv_caches=kv_caches,
            use_cache=True,
        )
        assert caches is not None
        return hidden[:, 0], caches

    @torch.no_grad()
    def generate_latents(
        self,
        batch_size: int,
        metadata: torch.Tensor,
        cfg_scale: float = 1.0,
        cfg_norm_match: bool = False,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        hidden, caches = self.init_cache(batch_size, metadata)
        generated = []
        for index in range(self.config.sequence_length):
            target_metadata = metadata[index].to(
                device=hidden.device, dtype=hidden.dtype
            )[None].expand(batch_size, -1)
            latent = self.diffusion.sample(
                hidden,
                target_condition=target_metadata,
                unconditional_z=self.null_context.to(dtype=hidden.dtype)[None].expand(
                    batch_size, -1
                ),
                cfg_scale=cfg_scale,
                cfg_norm_match=cfg_norm_match,
                generator=generator,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
            )
            generated.append(latent)
            if index + 1 < self.config.sequence_length:
                hidden, caches = self.forward_step(
                    latent, index + 1, metadata[index + 1], caches
                )
        self.train(was_training)
        return torch.stack(generated, dim=1)
