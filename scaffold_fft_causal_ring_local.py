"""Causal radial-ring flow with exact FFT state and aligned local computation.

The generative state remains in the corrected compact isometric FFT layout.  A
training example selects one target ring: earlier rings are clean, the target
ring is on its rectified-flow path, and later rings remain at their Gaussian
base.  Velocity is evaluated after transforming the complete hybrid state to
local image patches, then transformed back and masked to the target ring.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from train_scaffold_fft_residual import (
    ScaffoldResidualDenoiser,
    fft_state_to_images,
    images_to_fft_state,
)
from control_pixel_diffusion import patchify, unpatchify


def validate_scalar_rings(
    scalar_ring: torch.Tensor, *, expected_values: int | None = None
) -> torch.Tensor:
    """Validate and return a contiguous one-dimensional radial-ring layout."""
    if scalar_ring.ndim != 1 or scalar_ring.numel() == 0:
        raise ValueError("scalar_ring must be a non-empty one-dimensional tensor")
    scalar_ring = scalar_ring.long()
    if expected_values is not None and scalar_ring.numel() != expected_values:
        raise ValueError(
            f"scalar_ring has {scalar_ring.numel()} values, expected {expected_values}"
        )
    if int(scalar_ring.min()) != 0:
        raise ValueError("scalar ring ids must start at zero")
    ring_count = int(scalar_ring.max()) + 1
    counts = torch.bincount(scalar_ring.detach().cpu(), minlength=ring_count)
    if (counts == 0).any():
        raise ValueError("scalar ring ids must be contiguous and non-empty")
    return counts


def ring_masks(
    scalar_ring: torch.Tensor,
    target_ring: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return earlier/current/later masks for a per-example target ring."""
    if target_ring.ndim != 1:
        raise ValueError("target_ring must be one-dimensional")
    ring_count = int(scalar_ring.max()) + 1
    if target_ring.numel() == 0 or int(target_ring.min()) < 0:
        raise ValueError("target_ring contains an invalid ring id")
    if int(target_ring.max()) >= ring_count:
        raise ValueError("target_ring contains an out-of-range ring id")
    ids = scalar_ring.to(target_ring.device)[None]
    targets = target_ring.long()[:, None]
    return ids < targets, ids == targets, ids > targets


def sample_target_rings(
    batch_size: int,
    ring_counts: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample rings proportional to active scalar count.

    Combining this distribution with a within-ring mean gives every physical
    compact-FFT scalar the same population loss coefficient.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ring_counts.ndim != 1 or ring_counts.numel() == 0:
        raise ValueError("ring_counts must be a non-empty vector")
    if (ring_counts <= 0).any():
        raise ValueError("every ring count must be positive")
    probabilities = ring_counts.to(device=device, dtype=torch.float32)
    probabilities = probabilities / probabilities.sum()
    return torch.multinomial(
        probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    )


def assemble_causal_ring_state(
    target_fft_state: torch.Tensor,
    noise_fft_state: torch.Tensor,
    flow_time: torch.Tensor,
    target_ring: torch.Tensor,
    scalar_ring: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the asynchronous causal state and current-ring flow target."""
    if target_fft_state.shape != noise_fft_state.shape:
        raise ValueError("target and noise FFT states must have the same shape")
    if target_fft_state.ndim != 3:
        raise ValueError("FFT states must have shape [B,T,D]")
    batch = target_fft_state.shape[0]
    if flow_time.shape != (batch,) or target_ring.shape != (batch,):
        raise ValueError("flow_time and target_ring must have shape [B]")
    flat_target = target_fft_state.reshape(batch, -1)
    flat_noise = noise_fft_state.reshape(batch, -1)
    validate_scalar_rings(scalar_ring, expected_values=flat_target.shape[1])
    earlier, current, _ = ring_masks(scalar_ring, target_ring)
    time = flow_time.to(flat_target.dtype)[:, None]

    # Future coordinates retain exactly the sampled base noise.  Earlier
    # coordinates are teacher-forced data, and only the current ring is mixed.
    flat_state = torch.where(earlier, flat_target, flat_noise)
    current_state = time * flat_target + (1.0 - time) * flat_noise
    flat_state = torch.where(current, current_state, flat_state)
    target_velocity = flat_target - flat_noise
    return (
        flat_state.reshape_as(target_fft_state),
        target_velocity.reshape_as(target_fft_state),
        current.reshape_as(target_fft_state),
    )


class CausalRingLocalDenoiser(ScaffoldResidualDenoiser):
    """The passing local denoiser plus an explicit target-ring condition."""

    def __init__(
        self,
        tokens: int,
        patch_dim: int,
        ring_count: int,
        args: Any,
    ):
        super().__init__(tokens=tokens, patch_dim=patch_dim, args=args)
        if ring_count <= 0:
            raise ValueError("ring_count must be positive")
        self.ring_count = ring_count
        self.target_ring_embedding = nn.Embedding(ring_count, args.width)
        # Loading the passing joint model plus this zero embedding preserves its
        # exact initial function for every ring.
        nn.init.zeros_(self.target_ring_embedding.weight)

    def velocity_local(
        self,
        noisy_residual_patches: torch.Tensor,
        scaffold_patches: torch.Tensor,
        flow_time: torch.Tensor,
        target_ring: torch.Tensor,
    ) -> torch.Tensor:
        expected = (self.tokens, self.patch_dim)
        if noisy_residual_patches.shape[1:] != expected:
            raise ValueError(
                "noisy residual patches have shape "
                f"{tuple(noisy_residual_patches.shape[1:])}, expected {expected}"
            )
        if scaffold_patches.shape != noisy_residual_patches.shape:
            raise ValueError("scaffold and noisy residual patch shapes must match")
        if target_ring.shape != (noisy_residual_patches.shape[0],):
            raise ValueError("target_ring must have shape [B]")
        if int(target_ring.min()) < 0 or int(target_ring.max()) >= self.ring_count:
            raise ValueError("target_ring contains an out-of-range ring id")

        hidden = (
            self.residual_projection(noisy_residual_patches)
            + self.scaffold_projection(scaffold_patches)
            + self.position.to(noisy_residual_patches.dtype)
        )
        global_condition = self.time_embed(flow_time * 999.0)
        global_condition = global_condition + self.target_ring_embedding(target_ring)
        condition = global_condition.unsqueeze(1).expand_as(hidden)
        for layer in self.layers:
            hidden, _ = layer(hidden, condition=condition)
        return self.final_layer(hidden, condition)


def load_joint_denoiser_weights(
    model: CausalRingLocalDenoiser,
    joint_state: dict[str, torch.Tensor],
) -> None:
    """Load the passing joint denoiser, allowing only the new ring embedding."""
    incompatible = model.load_state_dict(joint_state, strict=False)
    expected_missing = {"target_ring_embedding.weight"}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"unexpected missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    nn.init.zeros_(model.target_ring_embedding.weight)


def causal_ring_dual_domain_velocity(
    model: CausalRingLocalDenoiser,
    codec,
    noisy_fft_state: torch.Tensor,
    scaffold_patches: torch.Tensor,
    flow_time: torch.Tensor,
    target_ring: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
) -> torch.Tensor:
    """Evaluate full FFT velocity through aligned local image patches."""
    noisy_images = fft_state_to_images(
        codec, noisy_fft_state, layout_orbit, layout_component
    )
    local_velocity = model.velocity_local(
        patchify(noisy_images, patch),
        scaffold_patches,
        flow_time,
        target_ring,
    )
    velocity_images = unpatchify(local_velocity, patch, image_size)
    return images_to_fft_state(
        codec,
        velocity_images,
        layout_orbit,
        layout_component,
        token_dim,
    )


def causal_ring_flow_loss(
    model: CausalRingLocalDenoiser,
    codec,
    target_fft_state: torch.Tensor,
    scaffold_patches: torch.Tensor,
    scalar_ring: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
    target_ring: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
    noise_fft_state: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """One unbiased physical-scalar causal-ring training objective."""
    batch = target_fft_state.shape[0]
    ring_counts = validate_scalar_rings(
        scalar_ring, expected_values=target_fft_state[0].numel()
    )
    if target_ring is None:
        target_ring = sample_target_rings(
            batch,
            ring_counts,
            device=target_fft_state.device,
            generator=generator,
        )
    if flow_time is None:
        flow_time = torch.rand(
            batch, device=target_fft_state.device, generator=generator
        )
    if noise_fft_state is None:
        noise_fft_state = torch.randn(
            target_fft_state.shape,
            device=target_fft_state.device,
            dtype=target_fft_state.dtype,
            generator=generator,
        )
    noisy, target_velocity, current_mask = assemble_causal_ring_state(
        target_fft_state,
        noise_fft_state,
        flow_time,
        target_ring,
        scalar_ring,
    )
    predicted_velocity = causal_ring_dual_domain_velocity(
        model,
        codec,
        noisy,
        scaffold_patches,
        flow_time,
        target_ring,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=patch,
        image_size=image_size,
        token_dim=token_dim,
    )
    squared = (predicted_velocity - target_velocity).square()
    mask = current_mask.to(squared.dtype)
    per_example = (squared * mask).sum(dim=(1, 2)) / mask.sum(
        dim=(1, 2)
    ).clamp_min(1.0)
    return {
        "loss": per_example.mean(),
        "per_example_loss": per_example.detach(),
        "target_ring": target_ring.detach(),
        "flow_time": flow_time.detach(),
        "current_mask": current_mask,
    }


def masked_ring_heun_step(
    model: CausalRingLocalDenoiser,
    codec,
    state: torch.Tensor,
    scaffold_patches: torch.Tensor,
    scalar_ring: torch.Tensor,
    *,
    target_ring_index: int,
    time: float,
    dt: float,
    use_heun: bool,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
) -> torch.Tensor:
    """Advance one solver step while changing only the selected FFT ring."""
    if not 0 <= target_ring_index <= int(scalar_ring.max()):
        raise ValueError("target_ring_index is out of range")
    batch = state.shape[0]
    target_ring = torch.full(
        (batch,), target_ring_index, device=state.device, dtype=torch.long
    )
    current_mask = (scalar_ring.to(state.device) == target_ring_index).reshape(
        1, *state.shape[1:]
    )
    mask = current_mask.to(state.dtype)
    flow_time = torch.full(
        (batch,), time, device=state.device, dtype=torch.float32
    )
    velocity = causal_ring_dual_domain_velocity(
        model,
        codec,
        state,
        scaffold_patches,
        flow_time,
        target_ring,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=patch,
        image_size=image_size,
        token_dim=token_dim,
    )
    proposal = state + dt * velocity * mask
    if not use_heun:
        return proposal
    next_time = torch.full(
        (batch,), time + dt, device=state.device, dtype=torch.float32
    )
    next_velocity = causal_ring_dual_domain_velocity(
        model,
        codec,
        proposal,
        scaffold_patches,
        next_time,
        target_ring,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=patch,
        image_size=image_size,
        token_dim=token_dim,
    )
    return state + 0.5 * dt * (velocity + next_velocity) * mask


@torch.no_grad()
def sample_causal_ring_fft(
    model: CausalRingLocalDenoiser,
    codec,
    scaffold_patches: torch.Tensor,
    scalar_ring: torch.Tensor,
    *,
    layout_orbit: torch.Tensor,
    layout_component: torch.Tensor,
    patch: int,
    image_size: int,
    token_dim: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate low-to-high FFT rings with a fixed future Gaussian state."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    count = scaffold_patches.shape[0]
    state = torch.randn(
        count,
        3 * image_size * image_size // token_dim,
        token_dim,
        device=scaffold_patches.device,
        dtype=torch.float32,
        generator=generator,
    )
    validate_scalar_rings(scalar_ring, expected_values=state[0].numel())
    dt = 1.0 / steps
    ring_count = int(scalar_ring.max()) + 1
    for ring in range(ring_count):
        for index in range(steps):
            state = masked_ring_heun_step(
                model,
                codec,
                state,
                scaffold_patches,
                scalar_ring,
                target_ring_index=ring,
                time=index / steps,
                dt=dt,
                use_heun=index + 1 < steps,
                layout_orbit=layout_orbit,
                layout_component=layout_component,
                patch=patch,
                image_size=image_size,
                token_dim=token_dim,
            )
    return state


def model_args_from_joint_checkpoint(arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        width=int(arguments["width"]),
        num_layers=int(arguments["num_layers"]),
        num_heads=int(arguments["num_heads"]),
        ff_mult=int(arguments["ff_mult"]),
    )
