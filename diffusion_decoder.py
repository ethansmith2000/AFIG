"""Conditional diffusion-loss head for continuous AFIG tokens.

Implements an AdaLN residual MLP denoiser with Diffusers DDPM/DDIM
schedulers, ε / v-prediction objectives, optional Min-SNR weighting,
component masking, and a diffusion batch multiplier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.training_utils import compute_snr


@dataclass(frozen=True)
class DiffusionDecoderConfig:
    target_dim: int = 6
    z_channels: int = 512
    target_condition_dim: int = 0
    width: int = 512
    depth: int = 3
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # epsilon | v_prediction
    loss_weighting: str = "none"  # none | min_snr
    min_snr_gamma: float = 5.0
    # Independent of loss_weighting: multiplies per-token MSE by radial
    # expected-centered-power weights (normalized mean 1 across orbits).
    radial_power_weighting: bool = False
    radial_power_exponent: float = 0.5
    diffusion_batch_mul: int = 4
    num_inference_steps: int = 20
    clip_sample: bool = False
    # Flow matching is intentionally not selectable yet.
    # TODO(flow): implement rectified-flow training + Euler sampling.

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(dtype=self.mlp[0].weight.dtype))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        target_condition_dim: int,
        num_res_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.target_condition_embed = (
            nn.Linear(target_condition_dim, model_channels)
            if target_condition_dim > 0
            else None
        )
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        target_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        y = self.time_embed(t) + self.cond_embed(c)
        if self.target_condition_embed is not None:
            if target_condition is None:
                raise ValueError(
                    "target_condition is required when target_condition_dim > 0"
                )
            y = y + self.target_condition_embed(target_condition)
        elif target_condition is not None:
            raise ValueError(
                "target_condition was provided but target_condition_dim is 0"
            )
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class DiffusionDecoder(nn.Module):
    """Per-token conditional diffusion loss + DDIM sampler."""

    def __init__(self, config: Optional[DiffusionDecoderConfig] = None):
        super().__init__()
        self.config = config or DiffusionDecoderConfig()
        if self.config.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(
                f"Unsupported prediction_type={self.config.prediction_type}. "
                "Supported: epsilon, v_prediction. "
                "TODO(flow): add flow-matching objective."
            )
        if self.config.loss_weighting not in ("none", "min_snr"):
            raise ValueError(f"Unknown loss_weighting={self.config.loss_weighting}")
        if not 0.0 <= self.config.radial_power_exponent <= 1.0:
            raise ValueError(
                "radial_power_exponent must be in [0, 1], "
                f"got {self.config.radial_power_exponent}"
            )

        self.net = SimpleMLPAdaLN(
            in_channels=self.config.target_dim,
            model_channels=self.config.width,
            out_channels=self.config.target_dim,
            z_channels=self.config.z_channels,
            target_condition_dim=self.config.target_condition_dim,
            num_res_blocks=self.config.depth,
        )

        common = dict(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule=self.config.beta_schedule,
            prediction_type=self.config.prediction_type,
            clip_sample=self.config.clip_sample,
        )
        self.train_scheduler = DDPMScheduler(**common)
        self.sample_scheduler = DDIMScheduler(**common)

    def _expand_mask(
        self,
        component_mask: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if component_mask is None:
            return torch.ones(batch, self.config.target_dim, device=device, dtype=dtype)
        if component_mask.ndim == 1:
            component_mask = component_mask[None, :].expand(batch, -1)
        elif component_mask.ndim == 2 and component_mask.shape[0] == 1:
            component_mask = component_mask.expand(batch, -1)
        return component_mask.to(device=device, dtype=dtype)

    def _prepare_target_batch(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        component_mask: Optional[torch.Tensor] = None,
        radius_bin: Optional[torch.Tensor] = None,
        radial_weights: Optional[torch.Tensor] = None,
        target_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Flatten [B,L,...] and apply diffusion batch multiplier.

        Returns flat target/z/target condition/mask/radius bin/radial weights.
        """
        if target.ndim == 3:
            b, l, d = target.shape
            target = target.reshape(b * l, d)
            z = z.reshape(b * l, -1)
            if target_condition is not None:
                if target_condition.ndim == 2:
                    target_condition = target_condition[None, :, :].expand(b, -1, -1)
                target_condition = target_condition.reshape(b * l, -1)
            if component_mask is not None:
                if component_mask.ndim == 2:
                    component_mask = component_mask[None, :, :].expand(b, -1, -1)
                component_mask = component_mask.reshape(b * l, -1)
            if radius_bin is not None:
                if radius_bin.ndim == 1:
                    radius_bin = radius_bin[None, :].expand(b, -1)
                radius_bin = radius_bin.reshape(b * l)
            if radial_weights is not None:
                if radial_weights.ndim == 1:
                    radial_weights = radial_weights[None, :].expand(b, -1)
                radial_weights = radial_weights.reshape(b * l)
        elif target.ndim != 2:
            raise ValueError(f"Expected target [B,L,D] or [N,D], got {tuple(target.shape)}")

        mul = self.config.diffusion_batch_mul
        if mul > 1:
            target = target.repeat(mul, 1)
            z = z.repeat(mul, 1)
            if target_condition is not None:
                target_condition = target_condition.repeat(mul, 1)
            if component_mask is not None:
                component_mask = component_mask.repeat(mul, 1)
            if radius_bin is not None:
                radius_bin = radius_bin.repeat(mul)
            if radial_weights is not None:
                radial_weights = radial_weights.repeat(mul)
        return target, z, target_condition, component_mask, radius_bin, radial_weights

    def compute_loss(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        component_mask: Optional[torch.Tensor] = None,
        radius_bin: Optional[torch.Tensor] = None,
        radial_weights: Optional[torch.Tensor] = None,
        target_condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized diffusion loss.

        target/z: [B, L, D]/[B, L, C] or already flat [N, D]/[N, C].
        radial_weights: optional [L] or [B, L] or flat [N] per-token multipliers
        (e.g. normalized radial expected-centered-power). Composed multiplicatively
        with Min-SNR when both are enabled.
        """
        if self.config.radial_power_weighting and radial_weights is None:
            raise ValueError(
                "radial_power_weighting=True requires radial_weights to be provided."
            )
        if not self.config.radial_power_weighting:
            radial_weights = None

        (
            target,
            z,
            target_condition,
            component_mask,
            radius_bin,
            radial_weights,
        ) = self._prepare_target_batch(
            target,
            z,
            component_mask,
            radius_bin,
            radial_weights,
            target_condition,
        )
        n = target.shape[0]
        device = target.device
        dtype = target.dtype
        mask = self._expand_mask(component_mask, n, device, dtype)

        noise = torch.randn_like(target) * mask
        timesteps = torch.randint(
            0,
            self.train_scheduler.config.num_train_timesteps,
            (n,),
            device=device,
            dtype=torch.long,
        )
        noisy = self.train_scheduler.add_noise(target.float(), noise.float(), timesteps)
        noisy = noisy.to(dtype=dtype) * mask

        if self.config.prediction_type == "epsilon":
            model_target = noise
        else:
            model_target = self.train_scheduler.get_velocity(
                target.float(), noise.float(), timesteps
            ).to(dtype=dtype)
        model_target = model_target * mask

        pred = self.net(noisy, timesteps, z, target_condition=target_condition) * mask
        per_dim = (pred.float() - model_target.float()) ** 2
        # Mean over active components only.
        denom = mask.sum(dim=-1).clamp_min(1.0)
        per_example = (per_dim * mask).sum(dim=-1) / denom

        if self.config.loss_weighting == "min_snr":
            snr_weights = self._min_snr_weights(timesteps)
        else:
            snr_weights = torch.ones_like(per_example)

        if radial_weights is not None:
            radial_w = radial_weights.to(device=device, dtype=per_example.dtype)
            if radial_w.shape != per_example.shape:
                raise ValueError(
                    f"radial_weights shape {tuple(radial_w.shape)} != "
                    f"per_example shape {tuple(per_example.shape)}"
                )
        else:
            radial_w = torch.ones_like(per_example)

        weights = snr_weights * radial_w
        weighted = per_example * weights
        radial_weighted = per_example * radial_w

        loss = weighted.mean()
        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "unweighted_mse": per_example.mean().detach(),
            "weighted_loss": weighted.mean().detach(),
            "radial_weighted_mse": radial_weighted.mean().detach(),
            "per_example": per_example.detach(),
            "timesteps": timesteps.detach(),
            "weights": weights.detach(),
            "snr_weights": snr_weights.detach(),
            "radial_weights": radial_w.detach(),
        }
        if radius_bin is not None:
            out["radius_bin"] = radius_bin.detach()
        return out

    def _min_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        snr = compute_snr(self.train_scheduler, timesteps)
        gamma = self.config.min_snr_gamma
        mse_loss_weights = torch.stack(
            [snr, gamma * torch.ones_like(timesteps, dtype=snr.dtype)], dim=1
        ).min(dim=1)[0]
        if self.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr.clamp_min(1e-8)
        else:
            mse_loss_weights = mse_loss_weights / (snr + 1)
        return mse_loss_weights

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        component_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        temperature: float = 1.0,
        target_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM sample tokens conditioned on z.

        z: [N, C] flat history conditions. Returns [N, D] normalized tokens.
        target_condition: optional [N, P] known target-frequency features.
        temperature scales the initial Gaussian noise (deterministic DDIM keeps eta=0).
        """
        if z.ndim != 2:
            raise ValueError(f"sample expects flat z [N,C], got {tuple(z.shape)}")
        n = z.shape[0]
        device = z.device
        dtype = z.dtype
        steps = num_inference_steps or self.config.num_inference_steps
        mask = self._expand_mask(component_mask, n, device, dtype)

        # Clone scheduler config so callers can change steps safely.
        scheduler = DDIMScheduler.from_config(self.sample_scheduler.config)
        scheduler.set_timesteps(steps, device=device)

        latents = torch.randn(
            n,
            self.config.target_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        latents = latents * float(temperature) * mask.float()

        for t in scheduler.timesteps:
            t_batch = torch.full((n,), int(t), device=device, dtype=torch.long)
            model_input = latents.to(dtype=dtype) * mask
            model_output = self.net(
                model_input,
                t_batch,
                z,
                target_condition=target_condition,
            ).float() * mask.float()
            out = scheduler.step(
                model_output,
                t,
                latents,
                eta=eta,
                generator=generator,
            )
            latents = out.prev_sample * mask.float()

        return (latents * mask.float()).to(dtype=dtype)


# ---------------------------------------------------------------------------
# Future extension points (documented stubs)
# ---------------------------------------------------------------------------

class FlowMatchingDecoderStub:
    """TODO(flow): rectified-flow / flow-matching per-token decoder.

    Intended interface:
      - train with x_t = (1-t)*x0 + t*noise, target = noise - x0
      - sample with FlowMatchEulerDiscreteScheduler
    Not implemented in the first continuous release.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Flow matching is stubbed. Use prediction_type in {epsilon, v_prediction}."
        )
