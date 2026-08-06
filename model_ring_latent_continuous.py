"""Autoregressive latent generator with joint diffusion inside frequency rings."""

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
    build_rope_tables,
)
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


RING_SEQUENCE_LENGTH = 23
MAX_RING_LATENTS = 4


@dataclass
class RingLatentContinuousConfig:
    """Configuration for ``p(Z_r | Z_<r)`` over padded latent ring blocks."""

    latent_sequence_length: int = LATENT_SEQUENCE_LENGTH
    latent_dim: int = LATENT_TOKEN_DIM
    ring_sequence_length: int = RING_SEQUENCE_LENGTH
    max_ring_latents: int = MAX_RING_LATENTS
    grouping: str = "ring"  # ring | token
    transformer: CausalTransformerConfig = field(
        default_factory=lambda: CausalTransformerConfig(
            max_seq_len=RING_SEQUENCE_LENGTH,
            qk_norm=True,
        )
    )
    diffusion: DiffusionDecoderConfig = field(
        default_factory=lambda: DiffusionDecoderConfig(
            target_dim=MAX_RING_LATENTS * LATENT_TOKEN_DIM,
            z_channels=512,
            target_condition_dim=0,
            condition_fusion="add",
            width=512,
            depth=6,
            objective="flow",
            prediction_type="v_prediction",
            component_reduction="fixed_dim",
        )
    )
    context_dropout_probability: float = 0.1
    rope_base: float = 10000.0

    def fingerprint(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["transformer"] = self.transformer.fingerprint()
        payload["diffusion"] = self.diffusion.fingerprint()
        return payload


def ring_latent_config_from_dict(payload: Dict[str, Any]) -> RingLatentContinuousConfig:
    return RingLatentContinuousConfig(
        latent_sequence_length=int(payload["latent_sequence_length"]),
        latent_dim=int(payload["latent_dim"]),
        ring_sequence_length=int(payload["ring_sequence_length"]),
        max_ring_latents=int(payload["max_ring_latents"]),
        grouping=str(payload.get("grouping", "ring")),
        transformer=CausalTransformerConfig(**payload["transformer"]),
        diffusion=DiffusionDecoderConfig(**payload["diffusion"]),
        context_dropout_probability=float(payload["context_dropout_probability"]),
        rope_base=float(payload.get("rope_base", 10000.0)),
    )


def _latent_slots(latent_parent: torch.Tensor, ring_count: int) -> torch.Tensor:
    """Return each latent's zero-based slot inside its parent ring."""
    counters = [0] * ring_count
    slots = []
    for parent in latent_parent.tolist():
        parent = int(parent)
        if not 0 <= parent < ring_count:
            raise ValueError("latent_parent contains an out-of-range ring")
        slots.append(counters[parent])
        counters[parent] += 1
    return torch.tensor(slots, dtype=torch.long)


class RingLatentContinuousModel(nn.Module):
    """Causal ring trunk plus a joint variable-width ring diffusion head.

    The 53 normalized AE latents are packed into 23 padded ring vectors.  At
    target ring ``r``, the Transformer input contains the complete sampled ring
    ``r-1`` (or a BOS marker for ring zero).  A single diffusion call denoises
    every active latent in ring ``r`` jointly, so no artificial sector order is
    introduced within a ring.

    Ring identity enters once through a learned target slot.  Sequence RoPE is
    used only in q/k attention geometry, and QK normalization is controlled by
    the neutral Transformer config.  The first baseline deliberately omits a
    second physical-metadata conditioning path.
    """

    def __init__(
        self,
        latent_parent: torch.Tensor,
        config: Optional[RingLatentContinuousConfig] = None,
    ):
        super().__init__()
        self.config = config or RingLatentContinuousConfig()
        cfg = self.config
        if min(
            cfg.latent_sequence_length,
            cfg.latent_dim,
            cfg.ring_sequence_length,
            cfg.max_ring_latents,
        ) <= 0:
            raise ValueError("Latent and grouping dimensions must be positive")
        if cfg.grouping not in ("ring", "token"):
            raise ValueError("grouping must be ring or token")
        if cfg.transformer.max_seq_len < cfg.ring_sequence_length:
            raise ValueError("Transformer max_seq_len is shorter than the ring sequence")
        ring_dim = cfg.max_ring_latents * cfg.latent_dim
        if cfg.diffusion.target_dim != ring_dim:
            raise ValueError("Diffusion target_dim must equal max_ring_latents * latent_dim")
        if cfg.diffusion.z_channels != cfg.transformer.width:
            raise ValueError("Diffusion z_channels must match Transformer width")
        if cfg.diffusion.target_condition_dim != 0 or cfg.diffusion.condition_fusion != "add":
            raise ValueError("Ring decoder baseline uses no redundant target metadata path")
        if cfg.diffusion.component_reduction != "fixed_dim":
            raise ValueError("Ring decoder requires fixed_dim reduction for padded blocks")

        latent_parent = latent_parent.detach().long().cpu()
        if latent_parent.shape != (cfg.latent_sequence_length,):
            raise ValueError("latent_parent must have one ring id per exported latent")
        if cfg.grouping == "token":
            latent_parent = torch.arange(cfg.latent_sequence_length)
            if cfg.ring_sequence_length != cfg.latent_sequence_length:
                raise ValueError("Token grouping requires one group per latent")
            if cfg.max_ring_latents != 1:
                raise ValueError("Token grouping requires max_ring_latents=1")
        slots = _latent_slots(latent_parent, cfg.ring_sequence_length)
        counts = torch.bincount(
            latent_parent, minlength=cfg.ring_sequence_length
        )
        if (counts == 0).any():
            raise ValueError("Every ring must export at least one latent")
        if int(counts.max()) > cfg.max_ring_latents:
            raise ValueError("A ring exceeds max_ring_latents")

        gather_indices = torch.zeros(
            cfg.ring_sequence_length, cfg.max_ring_latents, dtype=torch.long
        )
        slot_mask = torch.zeros_like(gather_indices, dtype=torch.bool)
        for latent_index, (parent, slot) in enumerate(
            zip(latent_parent.tolist(), slots.tolist())
        ):
            gather_indices[parent, slot] = latent_index
            slot_mask[parent, slot] = True
        component_mask = slot_mask[..., None].expand(
            -1, -1, cfg.latent_dim
        ).reshape(cfg.ring_sequence_length, ring_dim).clone()

        self.register_buffer("latent_parent", latent_parent, persistent=True)
        self.register_buffer("latent_slot", slots, persistent=True)
        self.register_buffer("ring_counts", counts, persistent=True)
        self.register_buffer("gather_indices", gather_indices, persistent=True)
        self.register_buffer("ring_slot_mask", slot_mask, persistent=True)
        self.register_buffer("ring_component_mask", component_mask, persistent=True)

        width = cfg.transformer.width
        self.input_projection = nn.Linear(ring_dim + 1, width)
        self.target_slot = nn.Embedding(cfg.ring_sequence_length, width)
        nn.init.normal_(self.target_slot.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=cfg.transformer.num_heads,
                    ff_mult=cfg.transformer.ff_mult,
                    dropout=cfg.transformer.dropout,
                    conditional_film=False,
                    causal=True,
                    qk_norm=cfg.transformer.qk_norm,
                )
                for _ in range(cfg.transformer.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(width)
        self.null_context = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.null_context, std=0.02)
        self.diffusion = DiffusionDecoder(cfg.diffusion)

        rope_cos, rope_sin = build_rope_tables(
            torch.arange(cfg.ring_sequence_length, dtype=torch.float32),
            width // cfg.transformer.num_heads,
            base=cfg.rope_base,
        )
        self.register_buffer("rope_cos", rope_cos.float(), persistent=False)
        self.register_buffer("rope_sin", rope_sin.float(), persistent=False)

    @property
    def ring_dim(self) -> int:
        return self.config.max_ring_latents * self.config.latent_dim

    def pack_rings(self, latents: torch.Tensor) -> torch.Tensor:
        expected = (self.config.latent_sequence_length, self.config.latent_dim)
        if latents.ndim != 3 or latents.shape[1:] != expected:
            raise ValueError(f"latents must be [B,{expected[0]},{expected[1]}]")
        gathered = latents[:, self.gather_indices]
        gathered = gathered * self.ring_slot_mask[None, :, :, None]
        return gathered.reshape(latents.shape[0], self.config.ring_sequence_length, -1)

    def unpack_rings(self, rings: torch.Tensor) -> torch.Tensor:
        expected = (self.config.ring_sequence_length, self.ring_dim)
        if rings.ndim != 3 or rings.shape[1:] != expected:
            raise ValueError(f"rings must be [B,{expected[0]},{expected[1]}]")
        padded = rings.reshape(
            rings.shape[0],
            self.config.ring_sequence_length,
            self.config.max_ring_latents,
            self.config.latent_dim,
        )
        return padded[:, self.latent_parent, self.latent_slot]

    def shifted_inputs_from_rings(self, rings: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(rings)
        previous[:, 1:] = rings[:, :-1]
        bos = torch.zeros(
            rings.shape[0],
            self.config.ring_sequence_length,
            1,
            device=rings.device,
            dtype=rings.dtype,
        )
        bos[:, 0] = 1.0
        positions = torch.arange(
            self.config.ring_sequence_length, device=rings.device
        )
        projected = self.input_projection(torch.cat([previous, bos], dim=-1))
        return projected + self.target_slot(positions).to(projected.dtype)

    def shifted_inputs(self, latents: torch.Tensor) -> torch.Tensor:
        return self.shifted_inputs_from_rings(self.pack_rings(latents))

    def _rope(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope_cos.device != device:
            coordinates = torch.arange(
                self.config.ring_sequence_length, device=device, dtype=torch.float32
            )
            cos, sin = build_rope_tables(
                coordinates,
                self.config.transformer.width // self.config.transformer.num_heads,
                base=self.config.rope_base,
            )
            self.rope_cos = cos.float()
            self.rope_sin = sin.float()
        return self.rope_cos, self.rope_sin

    def forward_backbone(
        self,
        inputs: torch.Tensor,
        kv_caches: Optional[List[Optional[KVCache]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]]]:
        hidden = inputs
        caches: List[KVCache] = []
        rope = self._rope(inputs.device)
        for index, layer in enumerate(self.layers):
            cache = None if kv_caches is None else kv_caches[index]
            if (
                self.config.transformer.gradient_checkpointing
                and self.training
                and not use_cache
            ):
                hidden = checkpoint(
                    lambda value, layer=layer: layer(value, rope=rope)[0],
                    hidden,
                    use_reentrant=False,
                )
                new_cache = None
            else:
                hidden, new_cache = layer(
                    hidden,
                    kv_cache=cache,
                    use_cache=use_cache,
                    rope=rope,
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
            probability = self.config.context_dropout_probability if self.training else 0.0
            mask = torch.rand(
                hidden.shape[0], hidden.shape[1], 1, device=hidden.device
            ) < probability
        else:
            mask = force_mask.to(device=hidden.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            if mask.shape != hidden.shape[:2] + (1,):
                raise ValueError("context dropout mask must be [B,R] or [B,R,1]")
        null = self.null_context.to(dtype=hidden.dtype).view(1, 1, -1)
        return torch.where(mask, null, hidden), mask

    def forward(
        self,
        latents: torch.Tensor,
        context_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rings = self.pack_rings(latents)
        hidden, _ = self.forward_backbone(self.shifted_inputs_from_rings(rings))
        conditioned, dropped = self.apply_context_dropout(
            hidden, force_mask=context_dropout_mask
        )
        output = self.diffusion.compute_loss(
            target=rings,
            z=conditioned,
            component_mask=self.ring_component_mask,
        )
        output["context_drop_fraction"] = dropped.float().mean().detach()
        output["context_null_gap"] = (
            hidden - self.null_context.to(dtype=hidden.dtype)
        ).square().mean().sqrt().detach()
        output["hidden"] = hidden
        return output

    @torch.no_grad()
    def init_cache(self, batch_size: int) -> Tuple[torch.Tensor, List[KVCache]]:
        dtype = self.input_projection.weight.dtype
        device = self.input_projection.weight.device
        previous = torch.zeros(batch_size, self.ring_dim, device=device, dtype=dtype)
        bos = torch.ones(batch_size, 1, device=device, dtype=dtype)
        inputs = self.input_projection(torch.cat([previous, bos], dim=-1))
        inputs = inputs + self.target_slot.weight[0].to(dtype)[None]
        hidden, caches = self.forward_backbone(inputs[:, None], use_cache=True)
        assert caches is not None
        return hidden[:, 0], caches

    @torch.no_grad()
    def forward_step(
        self,
        previous_ring: torch.Tensor,
        target_ring: int,
        kv_caches: List[KVCache],
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        if not 0 < target_ring < self.config.ring_sequence_length:
            raise ValueError("target_ring must be inside the ring sequence")
        bos = torch.zeros(
            previous_ring.shape[0], 1, device=previous_ring.device, dtype=previous_ring.dtype
        )
        inputs = self.input_projection(torch.cat([previous_ring, bos], dim=-1))
        inputs = inputs + self.target_slot.weight[target_ring].to(previous_ring.dtype)[None]
        hidden, caches = self.forward_backbone(
            inputs[:, None], kv_caches=kv_caches, use_cache=True
        )
        assert caches is not None
        return hidden[:, 0], caches

    @torch.no_grad()
    def generate_latents(
        self,
        batch_size: int,
        cfg_scale: float = 1.0,
        cfg_norm_match: bool = False,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        hidden, caches = self.init_cache(batch_size)
        generated = []
        for ring in range(self.config.ring_sequence_length):
            mask = self.ring_component_mask[ring].to(hidden.dtype)[None].expand(
                batch_size, -1
            )
            block = self.diffusion.sample(
                hidden,
                component_mask=mask,
                unconditional_z=self.null_context.to(dtype=hidden.dtype)[None].expand(
                    batch_size, -1
                ),
                cfg_scale=cfg_scale,
                cfg_norm_match=cfg_norm_match,
                generator=generator,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
            )
            block = block * mask
            generated.append(block)
            if ring + 1 < self.config.ring_sequence_length:
                hidden, caches = self.forward_step(block, ring + 1, caches)
        self.train(was_training)
        return self.unpack_rings(torch.stack(generated, dim=1))
