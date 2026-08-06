"""AR global-Hartley targets conditioned on a local partial inverse transform."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from causal_transformer import build_rope_tables
from control_pixel_diffusion import full_ihartleyify, patchify
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig
from model_continuous import TransformerBlock
from train_hartley_ar import hartley_tile_order


class SpatializedPrefixHartleyAR(nn.Module):
    """Predict global Hartley tiles from local patches of the known prefix.

    Target position ``p`` sees coefficients ``0..p-1`` in radial order. Unknown
    coefficients are exactly zero, the partial coefficient plane is inverted,
    and the resulting spatial map is exposed as local raster patches. The target
    remains the global Hartley tile at ``p``.
    """

    def __init__(
        self,
        *,
        width: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        diff_width: int,
        diff_depth: int,
        inference_steps: int,
        latent_size: int = 8,
        patch: int = 2,
        channels: int = 4,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if latent_size % patch:
            raise ValueError("latent_size must be divisible by patch")
        self.latent_size = latent_size
        self.patch = patch
        self.channels = channels
        self.grid = latent_size // patch
        self.seq_len = self.grid**2
        self.token_dim = channels * patch**2
        self.width = width
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_proj = nn.Linear(self.token_dim, width)
        self.spatial_slot = nn.Embedding(self.seq_len, width)
        self.target_slot = nn.Embedding(self.seq_len, width)
        nn.init.normal_(self.spatial_slot.weight, std=0.02)
        nn.init.normal_(self.target_slot.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    width,
                    num_heads,
                    ff_mult,
                    dropout=0.0,
                    position_film=False,
                    qk_norm=True,
                    causal=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(width)

        order = hartley_tile_order(self.grid)
        coords = torch.tensor(
            [(index // self.grid, index % self.grid) for index in range(self.seq_len)],
            dtype=torch.int64,
        )
        rope_cos, rope_sin = build_rope_tables(coords, width // num_heads)
        prefix_mask = torch.arange(self.seq_len)[None, :] < torch.arange(
            self.seq_len
        )[:, None]
        self.register_buffer("tile_order", order, persistent=True)
        self.register_buffer("rope_cos", rope_cos.float(), persistent=False)
        self.register_buffer("rope_sin", rope_sin.float(), persistent=False)
        self.register_buffer("prefix_mask", prefix_mask, persistent=False)

        self.diffusion = DiffusionDecoder(
            DiffusionDecoderConfig(
                target_dim=self.token_dim,
                z_channels=width,
                target_condition_dim=width,
                width=diff_width,
                depth=diff_depth,
                objective="flow",
                prediction_type="v_prediction",
                loss_space="native",
                loss_weighting="none",
                component_reduction="fixed_dim",
                flow_solver="heun",
                snr_scale=1.0,
                diffusion_batch_mul=1,
                num_inference_steps=inference_steps,
            )
        )

    def order_tokens(self, raster_tokens: torch.Tensor) -> torch.Tensor:
        return raster_tokens[:, self.tile_order]

    def restore_raster(self, ordered_tokens: torch.Tensor) -> torch.Tensor:
        raster = torch.empty_like(ordered_tokens)
        raster[:, self.tile_order] = ordered_tokens
        return raster

    def partial_spatial_tokens(
        self,
        ordered_history: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[B,K,S,D]`` local states for selected target positions."""
        batch, length, dim = ordered_history.shape
        if length != self.seq_len or dim != self.token_dim:
            raise ValueError(
                f"history has shape {tuple(ordered_history.shape[1:])}, expected "
                f"({self.seq_len}, {self.token_dim})"
            )
        if positions is None:
            positions = torch.arange(self.seq_len, device=ordered_history.device)
        positions = positions.to(device=ordered_history.device, dtype=torch.long)
        mask = self.prefix_mask[positions].to(ordered_history.dtype)
        partial_ordered = ordered_history[:, None] * mask[None, :, :, None]
        raster = torch.zeros_like(partial_ordered)
        raster[:, :, self.tile_order] = partial_ordered
        partial_maps = full_ihartleyify(
            raster.flatten(0, 1), self.patch, self.latent_size
        )
        spatial = patchify(partial_maps, self.patch)
        return spatial.reshape(batch, positions.numel(), self.seq_len, self.token_dim)

    def encode_states(
        self,
        spatial_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode local partial maps and return trunk/decoder conditions."""
        batch, count, spatial_len, dim = spatial_states.shape
        if spatial_len != self.seq_len or dim != self.token_dim:
            raise ValueError("unexpected spatialized-prefix shape")
        positions = positions.to(device=spatial_states.device, dtype=torch.long)
        target = self.target_slot(positions)
        target = target[None].expand(batch, -1, -1).reshape(batch * count, self.width)
        hidden = self.patch_proj(spatial_states.flatten(0, 1))
        hidden = hidden + self.spatial_slot.weight[None].to(hidden.dtype)
        hidden = hidden + target[:, None].to(hidden.dtype)
        rope = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                def run(module, values):
                    result, _ = module(values, rope=rope)
                    return result

                hidden = checkpoint(run, layer, hidden, use_reentrant=False)
            else:
                hidden, _ = layer(hidden, rope=rope)
        z = self.final_norm(hidden).mean(dim=1)
        return z, target.to(z.dtype)

    def forward(
        self,
        ordered_targets: torch.Tensor,
        history_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        history = ordered_targets if history_override is None else history_override
        positions = torch.arange(self.seq_len, device=ordered_targets.device)
        states = self.partial_spatial_tokens(history, positions)
        z, condition = self.encode_states(states, positions)
        return self.diffusion.compute_loss(
            target=ordered_targets.flatten(0, 1),
            z=z,
            target_condition=condition,
        )

    @torch.no_grad()
    def generate(
        self,
        count: int,
        steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = self.target_slot.weight.device
        ordered = torch.zeros(
            count, self.seq_len, self.token_dim, device=device
        )
        for position in range(self.seq_len):
            positions = torch.tensor([position], device=device)
            states = self.partial_spatial_tokens(ordered, positions)
            z, condition = self.encode_states(states, positions)
            ordered[:, position] = self.diffusion.sample(
                z,
                target_condition=condition,
                generator=generator,
                num_inference_steps=steps,
            )
        return ordered
