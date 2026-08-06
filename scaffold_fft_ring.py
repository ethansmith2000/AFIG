"""Scaffold-conditioned causal generation of exact compact-FFT residual rings.

The deterministic scaffold is encoded on its local 4x4 patch grid.  Exact
compact-FFT residual coordinates are then generated in radial rings: the trunk
is causal between rings, while one diffusion MLP denoises every active scalar
inside the current ring jointly.  No sector ordering is imposed within a ring,
and all RGB real/imaginary coordinates of an orbit remain in the same step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from causal_transformer import CausalTransformerBlock, KVCache, build_rope_tables
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig


@dataclass(frozen=True)
class ScaffoldFFTRingConfig:
    local_tokens: int = 64
    patch_dim: int = 48
    ring_count: int = 23
    max_ring_dim: int = 288
    width: int = 768
    scaffold_layers: int = 4
    ring_layers: int = 8
    num_heads: int = 12
    ff_mult: int = 4
    diffusion_width: int = 768
    diffusion_depth: int = 6
    diffusion_batch_mul: int = 1
    num_inference_steps: int = 20
    rope_base: float = 10000.0

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def build_ring_indices(
    scalar_ring: torch.Tensor, ring_count: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded scalar indices, active mask, and counts for each ring."""
    scalar_ring = scalar_ring.detach().long().cpu()
    if scalar_ring.ndim != 1 or scalar_ring.numel() == 0:
        raise ValueError("scalar_ring must be a non-empty one-dimensional tensor")
    if int(scalar_ring.min()) < 0 or int(scalar_ring.max()) >= ring_count:
        raise ValueError("scalar_ring contains an out-of-range ring id")
    counts = torch.bincount(scalar_ring, minlength=ring_count)
    if (counts == 0).any():
        raise ValueError("every radial ring must contain at least one scalar")
    max_ring_dim = int(counts.max())
    indices = torch.zeros(ring_count, max_ring_dim, dtype=torch.long)
    mask = torch.zeros(ring_count, max_ring_dim, dtype=torch.bool)
    for ring in range(ring_count):
        active = torch.nonzero(scalar_ring == ring, as_tuple=False).flatten()
        indices[ring, : active.numel()] = active
        mask[ring, : active.numel()] = True
    return indices, mask, counts


class ScaffoldFFTRingModel(nn.Module):
    """Local scaffold encoder plus causal radial-ring diffusion decoder."""

    def __init__(
        self,
        scalar_ring: torch.Tensor,
        config: Optional[ScaffoldFFTRingConfig] = None,
    ):
        super().__init__()
        self.config = config or ScaffoldFFTRingConfig()
        cfg = self.config
        if cfg.width % cfg.num_heads:
            raise ValueError("width must be divisible by num_heads")
        head_dim = cfg.width // cfg.num_heads
        if head_dim % 4:
            raise ValueError("head dimension must be divisible by four for 2-D RoPE")
        side = int(round(cfg.local_tokens**0.5))
        if side * side != cfg.local_tokens:
            raise ValueError("local_tokens must form a square patch grid")

        indices, mask, counts = build_ring_indices(scalar_ring, cfg.ring_count)
        if indices.shape[1] != cfg.max_ring_dim:
            raise ValueError(
                f"max_ring_dim={cfg.max_ring_dim}, but layout requires {indices.shape[1]}"
            )
        self.register_buffer("scalar_ring", scalar_ring.detach().long(), persistent=True)
        self.register_buffer("ring_indices", indices, persistent=True)
        self.register_buffer("ring_component_mask", mask, persistent=True)
        self.register_buffer("ring_counts", counts, persistent=True)

        self.scaffold_projection = nn.Linear(cfg.patch_dim, cfg.width)
        self.scaffold_position = nn.Parameter(torch.zeros(cfg.local_tokens, cfg.width))
        self.scaffold_layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=cfg.width,
                    num_heads=cfg.num_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=0.0,
                    causal=False,
                    qk_norm=True,
                )
                for _ in range(cfg.scaffold_layers)
            ]
        )
        self.scaffold_norm = nn.LayerNorm(cfg.width)

        # The extra scalar is an explicit BOS flag.  Target-ring identity is
        # injected after projection, where it cannot be drowned out by a large
        # raw coefficient and where each shared trunk computation is identifiable.
        self.ring_input_projection = nn.Linear(cfg.max_ring_dim + 1, cfg.width)
        self.target_slot = nn.Embedding(cfg.ring_count, cfg.width)
        nn.init.normal_(self.target_slot.weight, std=0.02)
        self.ring_layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=cfg.width,
                    num_heads=cfg.num_heads,
                    ff_mult=cfg.ff_mult,
                    dropout=0.0,
                    causal=True,
                    qk_norm=True,
                )
                for _ in range(cfg.ring_layers)
            ]
        )
        self.ring_norm = nn.LayerNorm(cfg.width)

        diffusion_config = DiffusionDecoderConfig(
            target_dim=cfg.max_ring_dim,
            z_channels=cfg.width,
            width=cfg.diffusion_width,
            depth=cfg.diffusion_depth,
            objective="flow",
            prediction_type="v_prediction",
            loss_metric="normalized",
            component_reduction="fixed_dim",
            diffusion_batch_mul=cfg.diffusion_batch_mul,
            num_inference_steps=cfg.num_inference_steps,
            flow_solver="heun",
        )
        self.diffusion = DiffusionDecoder(diffusion_config)

        y, x = torch.meshgrid(
            torch.arange(side, dtype=torch.float32),
            torch.arange(side, dtype=torch.float32),
            indexing="ij",
        )
        scaffold_cos, scaffold_sin = build_rope_tables(
            torch.stack([y.flatten(), x.flatten()], dim=-1),
            head_dim,
            base=cfg.rope_base,
        )
        # The already-bidirectional scaffold memory is placed before the causal
        # ring sequence.  It uses zero phase in this second attention stack;
        # radial sequence RoPE is reserved for ring-to-ring attention geometry.
        combined_coordinates = torch.cat(
            [
                torch.zeros(cfg.local_tokens, dtype=torch.float32),
                torch.arange(cfg.ring_count, dtype=torch.float32),
            ]
        )
        ring_cos, ring_sin = build_rope_tables(
            combined_coordinates, head_dim, base=cfg.rope_base
        )
        self.register_buffer("scaffold_rope_cos", scaffold_cos.float(), persistent=False)
        self.register_buffer("scaffold_rope_sin", scaffold_sin.float(), persistent=False)
        self.register_buffer("ring_rope_cos", ring_cos.float(), persistent=False)
        self.register_buffer("ring_rope_sin", ring_sin.float(), persistent=False)

    @property
    def scalar_dim(self) -> int:
        return int(self.scalar_ring.numel())

    def pack_rings(self, fft_state: torch.Tensor) -> torch.Tensor:
        """Pack flattened exact FFT scalars into padded radial ring vectors."""
        if fft_state.ndim < 2 or fft_state.shape[0] <= 0:
            raise ValueError("fft_state must have a non-empty batch dimension")
        flat = fft_state.reshape(fft_state.shape[0], -1)
        if flat.shape[1] != self.scalar_dim:
            raise ValueError(
                f"FFT state has {flat.shape[1]} scalars, expected {self.scalar_dim}"
            )
        rings = flat[:, self.ring_indices]
        return rings * self.ring_component_mask.to(rings.dtype)[None]

    def unpack_rings(self, rings: torch.Tensor) -> torch.Tensor:
        expected = (self.config.ring_count, self.config.max_ring_dim)
        if rings.ndim != 3 or rings.shape[1:] != expected:
            raise ValueError(f"rings must have shape [B,{expected[0]},{expected[1]}]")
        flat = torch.zeros(
            rings.shape[0], self.scalar_dim, device=rings.device, dtype=rings.dtype
        )
        for ring in range(self.config.ring_count):
            count = int(self.ring_counts[ring])
            flat[:, self.ring_indices[ring, :count]] = rings[:, ring, :count]
        return flat

    def encode_scaffold(self, scaffold_patches: torch.Tensor) -> torch.Tensor:
        expected = (self.config.local_tokens, self.config.patch_dim)
        if scaffold_patches.ndim != 3 or scaffold_patches.shape[1:] != expected:
            raise ValueError(
                f"scaffold_patches must have shape [B,{expected[0]},{expected[1]}]"
            )
        hidden = self.scaffold_projection(scaffold_patches)
        hidden = hidden + self.scaffold_position.to(hidden.dtype)[None]
        rope = (self.scaffold_rope_cos, self.scaffold_rope_sin)
        for layer in self.scaffold_layers:
            hidden, _ = layer(hidden, rope=rope)
        return self.scaffold_norm(hidden)

    def shifted_ring_inputs(self, rings: torch.Tensor) -> torch.Tensor:
        if rings.ndim != 3 or rings.shape[2] != self.config.max_ring_dim:
            raise ValueError("rings must be [B,R,max_ring_dim]")
        length = rings.shape[1]
        if not 1 <= length <= self.config.ring_count:
            raise ValueError("ring prefix length is out of range")
        previous = torch.zeros_like(rings)
        previous[:, 1:] = rings[:, :-1]
        bos = torch.zeros(
            rings.shape[0], length, 1, device=rings.device, dtype=rings.dtype
        )
        bos[:, 0] = 1.0
        hidden = self.ring_input_projection(torch.cat([previous, bos], dim=-1))
        slots = self.target_slot.weight[:length].to(hidden.dtype)
        return hidden + slots[None]

    def ring_conditions_from_memory(
        self, scaffold_memory: torch.Tensor, rings: torch.Tensor
    ) -> torch.Tensor:
        ring_inputs = self.shifted_ring_inputs(rings)
        hidden = torch.cat([scaffold_memory, ring_inputs], dim=1)
        length = hidden.shape[1]
        rope = (self.ring_rope_cos[:length], self.ring_rope_sin[:length])
        for layer in self.ring_layers:
            hidden, _ = layer(hidden, rope=rope)
        hidden = self.ring_norm(hidden)
        return hidden[:, self.config.local_tokens :]

    @torch.no_grad()
    def init_ring_cache(self, scaffold_memory: torch.Tensor) -> List[KVCache]:
        """Cache the fixed scaffold prefix once for sequential generation."""
        hidden = scaffold_memory
        caches: List[KVCache] = []
        rope = (
            self.ring_rope_cos[: self.config.local_tokens],
            self.ring_rope_sin[: self.config.local_tokens],
        )
        for layer in self.ring_layers:
            hidden, cache = layer(hidden, use_cache=True, rope=rope)
            assert cache is not None
            caches.append(cache)
        return caches

    @torch.no_grad()
    def ring_condition_step(
        self,
        previous_ring: torch.Tensor,
        target_ring: int,
        caches: List[KVCache],
    ) -> Tuple[torch.Tensor, List[KVCache]]:
        """Consume BOS/previous ring and return one target-ring condition."""
        if not 0 <= target_ring < self.config.ring_count:
            raise ValueError("target_ring is out of range")
        if len(caches) != len(self.ring_layers):
            raise ValueError("one KV cache is required per causal ring layer")
        bos = torch.zeros(
            previous_ring.shape[0], 1, device=previous_ring.device, dtype=previous_ring.dtype
        )
        if target_ring == 0:
            bos.fill_(1.0)
            previous_ring = torch.zeros_like(previous_ring)
        hidden = self.ring_input_projection(
            torch.cat([previous_ring, bos], dim=-1)
        )
        hidden = hidden + self.target_slot.weight[target_ring].to(hidden.dtype)[None]
        hidden = hidden[:, None]
        new_caches: List[KVCache] = []
        for layer, cache in zip(self.ring_layers, caches):
            hidden, new_cache = layer(
                hidden,
                kv_cache=cache,
                use_cache=True,
                rope=(self.ring_rope_cos, self.ring_rope_sin),
            )
            assert new_cache is not None
            new_caches.append(new_cache)
        return self.ring_norm(hidden[:, 0]), new_caches

    def ring_conditions(
        self, scaffold_patches: torch.Tensor, rings: torch.Tensor
    ) -> torch.Tensor:
        return self.ring_conditions_from_memory(self.encode_scaffold(scaffold_patches), rings)

    def forward(
        self, fft_state: torch.Tensor, scaffold_patches: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        rings = self.pack_rings(fft_state)
        hidden = self.ring_conditions(scaffold_patches, rings)
        output = self.diffusion.compute_loss(
            target=rings,
            z=hidden,
            component_mask=self.ring_component_mask,
        )
        output["hidden"] = hidden
        return output

    @torch.no_grad()
    def generate_fft(
        self,
        scaffold_patches: torch.Tensor,
        *,
        teacher_history_fft: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        memory = self.encode_scaffold(scaffold_patches)
        caches = self.init_ring_cache(memory)
        batch = scaffold_patches.shape[0]
        rings = torch.zeros(
            batch,
            self.config.ring_count,
            self.config.max_ring_dim,
            device=scaffold_patches.device,
            dtype=memory.dtype,
        )
        teacher_rings = (
            self.pack_rings(teacher_history_fft).to(memory.dtype)
            if teacher_history_fft is not None
            else None
        )
        previous = rings[:, 0]
        for ring in range(self.config.ring_count):
            condition, caches = self.ring_condition_step(previous, ring, caches)
            mask = self.ring_component_mask[ring].to(memory.dtype)[None].expand(batch, -1)
            rings[:, ring] = self.diffusion.sample(
                condition,
                component_mask=mask,
                generator=generator,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
            ) * mask
            previous = (
                teacher_rings[:, ring] if teacher_rings is not None else rings[:, ring]
            )
        self.train(was_training)
        return self.unpack_rings(rings)
