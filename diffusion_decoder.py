"""Conditional diffusion/flow head for continuous AFIG tokens."""

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
    objective: str = "ddpm"  # ddpm | flow
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    rescale_betas_zero_snr: bool = False
    timestep_spacing: str = "leading"
    prediction_type: str = "epsilon"  # epsilon | v_prediction | x0
    loss_space: str = "native"  # native | v (x0 prediction only)
    loss_weighting: str = "none"  # none | min_snr | logit_normal
    min_snr_gamma: float = 5.0
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    flow_t_eps: float = 0.05
    flow_solver: str = "heun"  # euler | heun
    # Independent of loss_weighting: multiplies per-token MSE by radial
    # expected-centered-power weights (normalized mean 1 across orbits).
    radial_power_weighting: bool = False
    radial_power_exponent: float = 0.5
    loss_metric: str = "normalized"  # normalized | orbit_covariance_power | orbit_scale_power
    orbit_covariance_exponent: float = 0.0
    orbit_scale_exponent: float = 0.0
    learned_output_gain: bool = False
    phase_aux_weight: float = 0.0
    phase_aux_gate: float = 0.1
    input_timestep_conditioning: str = "none"  # none | film
    input_projection_init: str = "xavier"  # xavier | kaiming_linear
    diffusion_batch_mul: int = 4
    num_inference_steps: int = 20
    clip_sample: bool = False

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
        input_timestep_conditioning: str,
        input_projection_init: str,
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
        self.input_timestep_conditioning = input_timestep_conditioning
        self.input_projection_init = input_projection_init
        if input_timestep_conditioning == "film":
            # Keep all subsequently constructed shared layers bit-identical to
            # the no-FiLM arm under the same global seed.
            with torch.random.fork_rng(devices=[]):
                self.input_time_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(model_channels, 2 * model_channels, bias=True),
                )
        else:
            self.input_time_modulation = None
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if (
                    self.input_time_modulation is not None
                    and module is self.input_time_modulation[-1]
                ):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if self.input_projection_init == "kaiming_linear":
            nn.init.kaiming_uniform_(
                self.input_proj.weight,
                a=0.0,
                mode="fan_in",
                nonlinearity="linear",
            )
            nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if self.input_time_modulation is not None:
            nn.init.zeros_(self.input_time_modulation[-1].weight)
            nn.init.zeros_(self.input_time_modulation[-1].bias)
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
        time = self.time_embed(t)
        if self.input_time_modulation is not None:
            scale, shift = self.input_time_modulation(time).chunk(2, dim=-1)
            x = modulate(x, shift, scale)
        y = time + self.cond_embed(c)
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
        if self.config.objective not in ("ddpm", "flow"):
            raise ValueError(f"Unsupported objective={self.config.objective}")
        if self.config.prediction_type not in ("epsilon", "v_prediction", "x0"):
            raise ValueError(
                f"Unsupported prediction_type={self.config.prediction_type}. "
                "Supported: epsilon, v_prediction, x0."
            )
        if self.config.objective == "flow" and self.config.prediction_type == "epsilon":
            raise ValueError("Flow objective supports x0 or v_prediction outputs.")
        if self.config.loss_space not in ("native", "v"):
            raise ValueError(f"Unsupported loss_space={self.config.loss_space}")
        if self.config.loss_space == "v" and self.config.prediction_type != "x0":
            raise ValueError("loss_space='v' requires prediction_type='x0'.")
        if self.config.loss_weighting not in ("none", "min_snr", "logit_normal"):
            raise ValueError(f"Unknown loss_weighting={self.config.loss_weighting}")
        if self.config.loss_weighting == "logit_normal" and self.config.objective != "flow":
            raise ValueError("logit_normal weighting is only supported for flow.")
        if self.config.logit_normal_std <= 0:
            raise ValueError("logit_normal_std must be positive.")
        if not 0.0 < self.config.flow_t_eps < 0.5:
            raise ValueError("flow_t_eps must be in (0, 0.5).")
        if self.config.flow_solver not in ("euler", "heun"):
            raise ValueError(f"Unsupported flow_solver={self.config.flow_solver}")
        if self.config.input_timestep_conditioning not in ("none", "film"):
            raise ValueError(
                "input_timestep_conditioning must be 'none' or 'film'."
            )
        if self.config.phase_aux_weight < 0.0:
            raise ValueError("phase_aux_weight must be non-negative")
        if self.config.phase_aux_gate <= 0.0:
            raise ValueError("phase_aux_gate must be positive")
        if self.config.phase_aux_weight > 0.0 and not (
            self.config.prediction_type == "x0"
            and self.config.loss_space == "native"
        ):
            raise ValueError(
                "Phase auxiliary currently requires native x0 prediction."
            )
        if self.config.input_projection_init not in ("xavier", "kaiming_linear"):
            raise ValueError(
                "input_projection_init must be 'xavier' or 'kaiming_linear'."
            )
        if self.config.timestep_spacing not in ("leading", "trailing", "linspace"):
            raise ValueError(
                f"Unsupported timestep_spacing={self.config.timestep_spacing}"
            )
        if not 0.0 <= self.config.radial_power_exponent <= 1.0:
            raise ValueError(
                "radial_power_exponent must be in [0, 1], "
                f"got {self.config.radial_power_exponent}"
            )
        if self.config.loss_metric not in (
            "normalized",
            "orbit_covariance_power",
            "orbit_scale_power",
        ):
            raise ValueError(f"Unsupported loss_metric={self.config.loss_metric}")
        if not 0.0 <= self.config.orbit_covariance_exponent <= 1.0:
            raise ValueError("orbit_covariance_exponent must be in [0, 1].")
        if not 0.0 <= self.config.orbit_scale_exponent <= 1.0:
            raise ValueError("orbit_scale_exponent must be in [0, 1].")
        if self.config.loss_metric in (
            "orbit_covariance_power",
            "orbit_scale_power",
        ):
            if (
                self.config.objective != "ddpm"
                or self.config.prediction_type != "x0"
                or self.config.loss_space != "native"
            ):
                raise ValueError(
                    f"{self.config.loss_metric} initially requires DDPM native x0 prediction."
                )
            if self.config.radial_power_weighting:
                raise ValueError(
                    f"{self.config.loss_metric} is mutually exclusive with radial weighting."
                )

        self.net = SimpleMLPAdaLN(
            in_channels=self.config.target_dim,
            model_channels=self.config.width,
            out_channels=self.config.target_dim,
            z_channels=self.config.z_channels,
            target_condition_dim=self.config.target_condition_dim,
            num_res_blocks=self.config.depth,
            input_timestep_conditioning=self.config.input_timestep_conditioning,
            input_projection_init=self.config.input_projection_init,
        )

        scheduler_prediction_type = {
            "epsilon": "epsilon",
            "v_prediction": "v_prediction",
            "x0": "sample",
        }[self.config.prediction_type]
        common = dict(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule=self.config.beta_schedule,
            rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
            timestep_spacing=self.config.timestep_spacing,
            prediction_type=scheduler_prediction_type,
            clip_sample=self.config.clip_sample,
        )
        self.train_scheduler = DDPMScheduler(**common)
        self.sample_scheduler = DDIMScheduler(**common)
        self.register_buffer(
            "logit_normal_weights",
            self._build_logit_normal_weights(),
            persistent=False,
        )

    def _build_logit_normal_weights(self) -> torch.Tensor:
        """Mean-one logit-normal density over uniform flow-time bins."""
        n = self.config.num_train_timesteps
        t = (torch.arange(n, dtype=torch.float64) + 0.5) / float(n)
        logit = torch.log(t) - torch.log1p(-t)
        mean = self.config.logit_normal_mean
        std = self.config.logit_normal_std
        density = torch.exp(-0.5 * ((logit - mean) / std).square())
        density = density / (std * math.sqrt(2.0 * math.pi) * t * (1.0 - t))
        density = density / density.mean().clamp_min(1e-12)
        return density.float()

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
        covariance_metric: Optional[torch.Tensor] = None,
        component_metric: Optional[torch.Tensor] = None,
        output_gain: Optional[torch.Tensor] = None,
        target_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
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
            if covariance_metric is not None:
                if covariance_metric.ndim == 3:
                    covariance_metric = covariance_metric[None, :, :, :].expand(
                        b, -1, -1, -1
                    )
                covariance_metric = covariance_metric.reshape(b * l, d, d)
            if component_metric is not None:
                if component_metric.ndim == 2:
                    component_metric = component_metric[None, :, :].expand(b, -1, -1)
                component_metric = component_metric.reshape(b * l, d)
            if output_gain is not None:
                if output_gain.ndim == 2:
                    output_gain = output_gain[None, :, :].expand(b, -1, -1)
                output_gain = output_gain.reshape(b * l, d)
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
            if covariance_metric is not None:
                covariance_metric = covariance_metric.repeat(mul, 1, 1)
            if component_metric is not None:
                component_metric = component_metric.repeat(mul, 1)
            if output_gain is not None:
                output_gain = output_gain.repeat(mul, 1)
        return (
            target,
            z,
            target_condition,
            component_mask,
            radius_bin,
            radial_weights,
            covariance_metric,
            component_metric,
            output_gain,
        )

    def compute_loss(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        component_mask: Optional[torch.Tensor] = None,
        radius_bin: Optional[torch.Tensor] = None,
        radial_weights: Optional[torch.Tensor] = None,
        covariance_metric: Optional[torch.Tensor] = None,
        component_metric: Optional[torch.Tensor] = None,
        output_gain: Optional[torch.Tensor] = None,
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
        if self.config.loss_metric == "orbit_covariance_power":
            if covariance_metric is None:
                raise ValueError(
                    "orbit_covariance_power requires covariance_metric matrices."
                )
        else:
            covariance_metric = None
        if self.config.loss_metric == "orbit_scale_power":
            if component_metric is None:
                raise ValueError(
                    "orbit_scale_power requires per-component metric weights."
                )
        else:
            component_metric = None
        if self.config.learned_output_gain and output_gain is None:
            raise ValueError("learned_output_gain requires output_gain values.")
        if not self.config.learned_output_gain:
            output_gain = None

        (
            target,
            z,
            target_condition,
            component_mask,
            radius_bin,
            radial_weights,
            covariance_metric,
            component_metric,
            output_gain,
        ) = self._prepare_target_batch(
            target,
            z,
            component_mask,
            radius_bin,
            radial_weights,
            covariance_metric,
            component_metric,
            output_gain,
            target_condition,
        )
        n = target.shape[0]
        device = target.device
        dtype = target.dtype
        mask = self._expand_mask(component_mask, n, device, dtype)

        noise = torch.randn_like(target) * mask
        if self.config.objective == "ddpm":
            timesteps = torch.randint(
                0,
                self.train_scheduler.config.num_train_timesteps,
                (n,),
                device=device,
                dtype=torch.long,
            )
            noisy = self.train_scheduler.add_noise(target.float(), noise.float(), timesteps)
            noisy = noisy.to(dtype=dtype) * mask
            time_condition = timesteps
        else:
            timesteps = torch.randint(
                0,
                self.config.num_train_timesteps,
                (n,),
                device=device,
                dtype=torch.long,
            )
            flow_t = (timesteps.float() + 0.5) / float(
                self.config.num_train_timesteps
            )
            flow_t_col = flow_t[:, None]
            noisy = (
                flow_t_col * target.float()
                + (1.0 - flow_t_col) * noise.float()
            ).to(dtype=dtype) * mask
            time_condition = flow_t * float(self.config.num_train_timesteps - 1)

        raw_pred = self.net(
            noisy,
            time_condition,
            z,
            target_condition=target_condition,
        )
        if output_gain is not None:
            raw_pred = raw_pred * output_gain.to(device=device, dtype=raw_pred.dtype)
        raw_pred = raw_pred * mask

        if self.config.objective == "ddpm":
            if self.config.prediction_type == "epsilon":
                pred = raw_pred
                model_target = noise
            elif self.config.prediction_type == "v_prediction":
                pred = raw_pred
                model_target = self.train_scheduler.get_velocity(
                    target.float(), noise.float(), timesteps
                ).to(dtype=dtype)
            else:
                if self.config.loss_space == "native":
                    pred = raw_pred
                    model_target = target
                else:
                    alpha_bar = self.train_scheduler.alphas_cumprod.to(device)[timesteps]
                    alpha = alpha_bar.sqrt()[:, None]
                    sigma = (1.0 - alpha_bar).sqrt().clamp_min(1e-6)[:, None]
                    pred = ((alpha * noisy.float() - raw_pred.float()) / sigma).to(
                        dtype=dtype
                    )
                    model_target = self.train_scheduler.get_velocity(
                        target.float(), noise.float(), timesteps
                    ).to(dtype=dtype)
        elif self.config.prediction_type == "x0":
            if self.config.loss_space == "native":
                pred = raw_pred
                model_target = target
            else:
                denom = (1.0 - flow_t_col).clamp_min(self.config.flow_t_eps)
                pred = ((raw_pred.float() - noisy.float()) / denom).to(dtype=dtype)
                model_target = ((target.float() - noisy.float()) / denom).to(
                    dtype=dtype
                )
        else:
            pred = raw_pred
            model_target = target - noise

        model_target = model_target * mask
        error = (pred.float() - model_target.float()) * mask.float()
        per_dim = error.square()
        # Mean over active components only.
        denom = mask.sum(dim=-1).clamp_min(1.0)
        per_example = (per_dim * mask).sum(dim=-1) / denom
        if covariance_metric is not None:
            covariance_per_example = torch.einsum(
                "ni,nij,nj->n",
                error,
                covariance_metric.to(device=device, dtype=error.dtype),
                error,
            )
        else:
            covariance_per_example = per_example
        if component_metric is not None:
            scale_per_example = (
                per_dim
                * component_metric.to(device=device, dtype=per_dim.dtype)
            ).sum(dim=-1)
        else:
            scale_per_example = per_example
        if self.config.loss_metric == "orbit_covariance_power":
            metric_per_example = covariance_per_example
        elif self.config.loss_metric == "orbit_scale_power":
            metric_per_example = scale_per_example
        else:
            metric_per_example = per_example

        if self.config.loss_weighting == "min_snr":
            snr_weights = self._min_snr_weights(
                timesteps,
                flow_t=flow_t if self.config.objective == "flow" else None,
            )
        elif self.config.loss_weighting == "logit_normal":
            snr_weights = self.logit_normal_weights[timesteps].to(
                device=device,
                dtype=per_example.dtype,
            )
        else:
            snr_weights = torch.ones_like(metric_per_example)

        if radial_weights is not None:
            radial_w = radial_weights.to(device=device, dtype=metric_per_example.dtype)
            if radial_w.shape != metric_per_example.shape:
                raise ValueError(
                    f"radial_weights shape {tuple(radial_w.shape)} != "
                    f"per_example shape {tuple(metric_per_example.shape)}"
                )
        else:
            radial_w = torch.ones_like(metric_per_example)

        weights = snr_weights * radial_w
        weighted = metric_per_example * weights
        radial_weighted = per_example * radial_w

        loss = weighted.mean()
        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "unweighted_mse": per_example.mean().detach(),
            "covariance_metric_loss": covariance_per_example.mean().detach(),
            "scale_metric_loss": scale_per_example.mean().detach(),
            "weighted_loss": weighted.mean().detach(),
            "radial_weighted_mse": radial_weighted.mean().detach(),
            "per_example": metric_per_example.detach(),
            "normalized_per_example": per_example.detach(),
            "timesteps": timesteps.detach(),
            "weights": weights.detach(),
            "snr_weights": snr_weights.detach(),
            "radial_weights": radial_w.detach(),
        }
        if self.config.phase_aux_weight > 0.0:
            out["predicted_x0_for_phase"] = raw_pred
            out["target_x0_for_phase"] = target
        if radius_bin is not None:
            out["radius_bin"] = radius_bin.detach()
        if self.config.objective == "flow":
            out["flow_times"] = flow_t.detach()
        return out

    @torch.no_grad()
    def predict_x0_deterministic(
        self,
        target: torch.Tensor,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        component_mask: Optional[torch.Tensor] = None,
        output_gain: Optional[torch.Tensor] = None,
        target_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict normalized x0 at explicitly supplied times and noise.

        This path is evaluation-only and deliberately bypasses diffusion batch
        multiplication and random sampling so different objectives can be
        compared on one fixed held-out panel.
        """
        original_shape = target.shape
        if target.ndim == 3:
            b, length, dim = target.shape
            target = target.reshape(b * length, dim)
            z = z.reshape(b * length, -1)
            timesteps = timesteps.reshape(b * length)
            noise = noise.reshape(b * length, dim)
            if component_mask is not None:
                if component_mask.ndim == 2:
                    component_mask = component_mask[None].expand(b, -1, -1)
                component_mask = component_mask.reshape(b * length, dim)
            if output_gain is not None:
                if output_gain.ndim == 2:
                    output_gain = output_gain[None].expand(b, -1, -1)
                output_gain = output_gain.reshape(b * length, dim)
            if target_condition is not None:
                if target_condition.ndim == 2:
                    target_condition = target_condition[None].expand(b, -1, -1)
                target_condition = target_condition.reshape(b * length, -1)
        elif target.ndim != 2:
            raise ValueError("target must have shape [B,L,D] or [N,D]")

        n = target.shape[0]
        mask = self._expand_mask(
            component_mask,
            n,
            target.device,
            target.dtype,
        )
        noise = noise.to(device=target.device, dtype=target.dtype) * mask
        timesteps = timesteps.to(device=target.device, dtype=torch.long)
        if self.config.objective == "ddpm":
            noisy = self.train_scheduler.add_noise(
                target.float(), noise.float(), timesteps
            ).to(target.dtype)
            time_condition = timesteps
        else:
            flow_t = (timesteps.float() + 0.5) / float(
                self.config.num_train_timesteps
            )
            noisy = (
                flow_t[:, None] * target.float()
                + (1.0 - flow_t[:, None]) * noise.float()
            ).to(target.dtype)
            time_condition = flow_t * float(self.config.num_train_timesteps - 1)
        noisy = noisy * mask

        raw = self.net(
            noisy,
            time_condition,
            z,
            target_condition=target_condition,
        )
        if output_gain is not None:
            raw = raw * output_gain.to(device=raw.device, dtype=raw.dtype)
        raw = raw * mask

        if self.config.prediction_type == "x0":
            x0 = raw
        elif self.config.objective == "ddpm":
            alpha_bar = self.train_scheduler.alphas_cumprod.to(raw.device)[timesteps]
            alpha = alpha_bar.sqrt()[:, None]
            sigma = (1.0 - alpha_bar).sqrt()[:, None]
            if self.config.prediction_type == "epsilon":
                x0 = (
                    noisy.float() - sigma * raw.float()
                ) / alpha.clamp_min(1e-6)
            else:
                x0 = alpha * noisy.float() - sigma * raw.float()
        else:
            x0 = noisy.float() + (1.0 - flow_t[:, None]) * raw.float()
        x0 = x0.to(target.dtype) * mask
        return x0.reshape(original_shape)

    def _min_snr_weights(
        self,
        timesteps: torch.Tensor,
        flow_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.objective == "ddpm":
            snr = compute_snr(self.train_scheduler, timesteps)
        else:
            t = flow_t
            if t is None:
                t = (timesteps.float() + 0.5) / float(
                    self.config.num_train_timesteps
                )
            snr = (t / (1.0 - t).clamp_min(1e-8)).square()
        gamma = self.config.min_snr_gamma
        mse_loss_weights = torch.stack(
            [snr, gamma * torch.ones_like(timesteps, dtype=snr.dtype)], dim=1
        ).min(dim=1)[0]
        loss_prediction_type = (
            "v_prediction"
            if self.config.loss_space == "v"
            else self.config.prediction_type
        )
        if loss_prediction_type == "x0":
            mse_loss_weights = mse_loss_weights / gamma
        elif loss_prediction_type == "epsilon":
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
        output_gain: Optional[torch.Tensor] = None,
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
        if self.config.learned_output_gain and output_gain is None:
            raise ValueError("learned_output_gain requires output_gain values.")
        if not self.config.learned_output_gain:
            output_gain = None
        elif output_gain is not None:
            output_gain = output_gain.to(device=device, dtype=dtype)

        if self.config.objective == "flow":
            return self._sample_flow(
                z=z,
                target_condition=target_condition,
                mask=mask,
                generator=generator,
                steps=steps,
                temperature=temperature,
                output_gain=output_gain,
            )

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
            )
            if output_gain is not None:
                model_output = model_output * output_gain
            model_output = model_output.float() * mask.float()
            out = scheduler.step(
                model_output,
                t,
                latents,
                eta=eta,
                generator=generator,
            )
            latents = out.prev_sample * mask.float()

        return (latents * mask.float()).to(dtype=dtype)

    def _sample_flow(
        self,
        z: torch.Tensor,
        target_condition: Optional[torch.Tensor],
        mask: torch.Tensor,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float,
        output_gain: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Integrate the JiT convention from noise at t=0 to data at t=1."""
        n = z.shape[0]
        device = z.device
        dtype = z.dtype
        state = torch.randn(
            n,
            self.config.target_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        state = state * float(temperature) * mask.float()
        dt = 1.0 / float(steps)

        def velocity(x: torch.Tensor, t_value: float) -> torch.Tensor:
            time_condition = torch.full(
                (n,),
                t_value * float(self.config.num_train_timesteps - 1),
                device=device,
                dtype=torch.float32,
            )
            raw = self.net(
                x.to(dtype=dtype) * mask,
                time_condition,
                z,
                target_condition=target_condition,
            )
            if output_gain is not None:
                raw = raw * output_gain
            raw = raw.float() * mask.float()
            if self.config.prediction_type == "x0":
                denom = max(1.0 - t_value, self.config.flow_t_eps)
                return (raw - x) / denom
            return raw

        for i in range(steps):
            t_value = i / float(steps)
            v0 = velocity(state, t_value)
            proposed = (state + dt * v0) * mask.float()
            if self.config.flow_solver == "heun" and i + 1 < steps:
                v1 = velocity(proposed, (i + 1) / float(steps))
                state = state + 0.5 * dt * (v0 + v1)
                state = state * mask.float()
            else:
                state = proposed

        return state.to(dtype=dtype) * mask
