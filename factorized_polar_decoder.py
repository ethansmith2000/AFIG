"""Amplitude-first, intrinsic circular decoder for native complex FFT tokens."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from diffusion_decoder import DiffusionDecoderConfig, SimpleMLPAdaLN


@dataclass(frozen=True)
class FactorizedPolarConfig:
    enabled: bool = False
    log_epsilon: float = 1e-4
    amplitude_standardization: str = "none"  # none | global | channel
    amplitude_loss_weight: float = 1.0
    phase_loss_weight: float = 1.0
    cartesian_loss_weight: float = 0.1
    phase_gate: float = 0.1
    phase_predicted_amplitude_probability: float = 0.5
    phase_process: str = "geodesic_flow"
    phase_sigma_min: float = 0.01 * math.pi
    phase_sigma_max: float = math.pi

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Map radians to [-pi, pi) without changing local derivatives."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def wrapped_normal_score(
    displacement: torch.Tensor,
    sigma: torch.Tensor,
    aliases: int = 4,
) -> torch.Tensor:
    """Score of a wrapped-normal perturbation kernel on ``S1``.

    ``displacement`` is noisy angle minus clean angle, wrapped or unwrapped.
    The finite image sum is effectively exact for sigma <= pi with four aliases
    on either side (the omitted boundary terms are below 3e-11 at sigma=pi).
    """
    displacement = wrap_angle(displacement.float())
    sigma = sigma.float().clamp_min(1e-6)
    while sigma.ndim < displacement.ndim:
        sigma = sigma.unsqueeze(-1)
    shifts = torch.arange(
        -aliases,
        aliases + 1,
        device=displacement.device,
        dtype=displacement.dtype,
    )
    lifts = displacement.unsqueeze(-1) + 2.0 * math.pi * shifts
    log_weights = -0.5 * (lifts / sigma.unsqueeze(-1)).square()
    weights = torch.softmax(log_weights, dim=-1)
    return -(weights * lifts).sum(dim=-1) / sigma.square()


@torch.no_grad()
def wrapped_normal_score_norm_table(
    timesteps: int,
    sigma_min: float,
    sigma_max: float,
    *,
    grid_points: int = 4096,
) -> torch.Tensor:
    """Deterministic Fisher-information table used by torsional DSM weighting."""
    t = (torch.arange(timesteps, dtype=torch.float64) + 0.5) / timesteps
    sigma = torch.exp(
        math.log(sigma_max) + t * (math.log(sigma_min) - math.log(sigma_max))
    )
    grid = (
        (torch.arange(grid_points, dtype=torch.float64) + 0.5)
        / grid_points
        * (2.0 * math.pi)
        - math.pi
    )
    aliases = torch.arange(-4, 5, dtype=torch.float64)
    result = []
    for sigma_chunk in sigma.split(64):
        lifts = grid[None, :, None] + 2.0 * math.pi * aliases[None, None, :]
        scaled = lifts / sigma_chunk[:, None, None]
        log_terms = -0.5 * scaled.square()
        log_density = torch.logsumexp(log_terms, dim=-1)
        alias_weights = torch.softmax(log_terms, dim=-1)
        score = -(
            alias_weights * lifts
        ).sum(dim=-1) / sigma_chunk[:, None].square()
        density_weights = torch.softmax(log_density, dim=-1)
        result.append((density_weights * score.square()).sum(dim=-1))
    return torch.cat(result).float().clamp_min(1e-8)


def polar_to_cartesian(
    log_relative_amplitude: torch.Tensor,
    phase: torch.Tensor,
    amplitude_scale: torch.Tensor,
    is_self_conjugate: torch.Tensor,
    log_epsilon: float,
    amplitude_coordinate_mean: Optional[torch.Tensor] = None,
    amplitude_coordinate_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Invert decoder coordinates into packed physical Cartesian coefficients."""
    if (amplitude_coordinate_mean is None) != (amplitude_coordinate_std is None):
        raise ValueError("amplitude coordinate mean and std must be provided together")
    if amplitude_coordinate_mean is not None:
        mean = amplitude_coordinate_mean.to(
            device=log_relative_amplitude.device,
            dtype=log_relative_amplitude.dtype,
        )
        std = amplitude_coordinate_std.to(
            device=log_relative_amplitude.device,
            dtype=log_relative_amplitude.dtype,
        )
        log_relative_amplitude = log_relative_amplitude * std + mean
    relative_amplitude = (
        torch.exp(log_relative_amplitude.clamp(min=-16.0, max=8.0))
        - float(log_epsilon)
    ).clamp_min(0.0)
    amplitude = relative_amplitude * amplitude_scale
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    imag = imag * (~is_self_conjugate).to(imag.dtype)[:, None]
    return torch.cat([real, imag], dim=-1)


class FactorizedPolarDecoder(nn.Module):
    """Amplitude-first generation over R^3 and (S1)^3.

    Amplitude follows an ordinary Euclidean flow in log-relative coordinates.
    Phase uses either shortest-geodesic Riemannian flow matching or a
    wrapped-normal Brownian score process. The sampled amplitude conditions the
    phase head in both cases.
    """

    def __init__(
        self,
        base: DiffusionDecoderConfig,
        config: FactorizedPolarConfig,
        condition_width: int,
        amplitude_coordinate_mean: Optional[torch.Tensor] = None,
        amplitude_coordinate_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if base.objective != "flow" or base.prediction_type != "v_prediction":
            raise ValueError(
                "factorized_polar requires objective=flow and prediction_type=v_prediction"
            )
        if base.loss_weighting != "none":
            raise ValueError("factorized_polar currently requires loss_weighting=none")
        if base.snr_scale != 1.0:
            raise ValueError(
                "factorized_polar defines separate base measures; set snr_scale=1"
            )
        if not 0.0 <= config.phase_predicted_amplitude_probability <= 1.0:
            raise ValueError("phase_predicted_amplitude_probability must be in [0, 1]")
        if config.log_epsilon <= 0.0 or config.phase_gate <= 0.0:
            raise ValueError("log_epsilon and phase_gate must be positive")
        if config.amplitude_standardization not in ("none", "global", "channel"):
            raise ValueError(
                "amplitude_standardization must be none, global, or channel"
            )
        if config.phase_process not in ("geodesic_flow", "wrapped_normal_score"):
            raise ValueError(f"Unknown phase process: {config.phase_process}")
        if not 0.0 < config.phase_sigma_min < config.phase_sigma_max:
            raise ValueError("phase sigma bounds must satisfy 0 < min < max")

        self.base = base
        self.config = config
        mean = (
            torch.zeros(3, dtype=torch.float32)
            if amplitude_coordinate_mean is None
            else amplitude_coordinate_mean.detach().float().reshape(3)
        )
        std = (
            torch.ones(3, dtype=torch.float32)
            if amplitude_coordinate_std is None
            else amplitude_coordinate_std.detach().float().reshape(3)
        )
        if bool((std <= 0).any()) or not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(std).all()
        ):
            raise ValueError("amplitude coordinate statistics must be finite with std > 0")
        self.register_buffer("amplitude_coordinate_mean", mean, persistent=True)
        self.register_buffer("amplitude_coordinate_std", std, persistent=True)
        common = dict(
            model_channels=base.width,
            z_channels=condition_width,
            num_res_blocks=base.depth,
            input_timestep_conditioning=base.input_timestep_conditioning,
            input_projection_init=base.input_projection_init,
            condition_fusion="add",
        )
        self.amplitude_net = SimpleMLPAdaLN(
            in_channels=3,
            out_channels=3,
            target_condition_dim=condition_width,
            **common,
        )
        # The direct condition is [learned target slot, sampled/predicted log amp].
        self.phase_net = SimpleMLPAdaLN(
            in_channels=6,
            out_channels=3,
            target_condition_dim=condition_width + 3,
            **common,
        )
        score_norm = (
            wrapped_normal_score_norm_table(
                int(base.num_train_timesteps),
                config.phase_sigma_min,
                config.phase_sigma_max,
            )
            if config.phase_process == "wrapped_normal_score"
            else torch.ones(int(base.num_train_timesteps))
        )
        self.register_buffer("phase_score_norm", score_norm, persistent=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # ``phase_score_norm`` is deterministic from the config and was added
        # after the first geodesic-flow checkpoints were written.  Supply the
        # reconstructed table when loading those checkpoints so strict loading
        # remains backward compatible.
        defaults = {
            prefix + "phase_score_norm": self.phase_score_norm,
            prefix + "amplitude_coordinate_mean": self.amplitude_coordinate_mean,
            prefix + "amplitude_coordinate_std": self.amplitude_coordinate_std,
        }
        for key, value in defaults.items():
            if key not in state_dict:
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def num_train_timesteps(self) -> int:
        return int(self.base.num_train_timesteps)

    @staticmethod
    def _flatten_and_repeat(
        value: torch.Tensor,
        multiplier: int,
    ) -> torch.Tensor:
        value = value.reshape(-1, value.shape[-1])
        return value.repeat(multiplier, 1) if multiplier > 1 else value

    def target_coordinates(
        self,
        raw_cartesian: torch.Tensor,
        amplitude_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real, imag = raw_cartesian[..., :3], raw_cartesian[..., 3:]
        amplitude = torch.sqrt(real.float().square() + imag.float().square())
        scale = amplitude_scale.to(device=amplitude.device, dtype=amplitude.dtype)
        relative = amplitude / scale.clamp_min(1e-8)
        log_amplitude = torch.log(relative + float(self.config.log_epsilon))
        if self.config.amplitude_standardization != "none":
            mean = self.amplitude_coordinate_mean.to(log_amplitude)
            std = self.amplitude_coordinate_std.to(log_amplitude)
            log_amplitude = (log_amplitude - mean) / std
        phase = torch.atan2(imag.float(), real.float())
        return log_amplitude, phase, relative

    def compute_loss(
        self,
        raw_target: torch.Tensor,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        amplitude_scale: torch.Tensor,
        is_self_conjugate: torch.Tensor,
        radius_bin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train both flows and a globally normalized Cartesian reconstruction."""
        if raw_target.ndim != 3 or raw_target.shape[-1] != 6:
            raise ValueError("raw_target must have shape [B,L,6]")
        batch, length, _ = raw_target.shape
        mul = int(self.base.diffusion_batch_mul)

        log_amp, target_phase, relative_amp = self.target_coordinates(
            raw_target, amplitude_scale[None]
        )
        log_amp = self._flatten_and_repeat(log_amp, mul)
        target_phase = self._flatten_and_repeat(target_phase, mul)
        relative_amp = self._flatten_and_repeat(relative_amp, mul)
        raw_flat = self._flatten_and_repeat(raw_target.float(), mul)
        z_flat = self._flatten_and_repeat(z, mul)
        slot_flat = self._flatten_and_repeat(slot_condition, mul)
        scale_flat = self._flatten_and_repeat(
            amplitude_scale[None].expand(batch, -1, -1), mul
        ).float()
        self_flat = is_self_conjugate[None, :, None].expand(batch, -1, 1)
        self_flat = self._flatten_and_repeat(self_flat.float(), mul)[:, 0].bool()

        n = log_amp.shape[0]
        device = log_amp.device
        amp_timestep = torch.randint(
            0, self.num_train_timesteps, (n,), device=device, dtype=torch.long
        )
        amp_t = (amp_timestep.float() + 0.5) / float(self.num_train_timesteps)
        amp_noise = torch.randn_like(log_amp)
        amp_noisy = amp_t[:, None] * log_amp + (1.0 - amp_t[:, None]) * amp_noise
        amp_velocity_target = log_amp - amp_noise
        amp_velocity = self.amplitude_net(
            amp_noisy,
            amp_t * float(self.num_train_timesteps - 1),
            z_flat,
            target_condition=slot_flat,
        )
        amp_per_example = (amp_velocity.float() - amp_velocity_target).square().mean(-1)
        predicted_log_amp = amp_noisy.float() + (1.0 - amp_t[:, None]) * amp_velocity.float()

        choose_predicted = torch.rand(n, 1, device=device) < float(
            self.config.phase_predicted_amplitude_probability
        )
        phase_amp_condition = torch.where(
            choose_predicted,
            predicted_log_amp.detach(),
            log_amp.float(),
        )
        phase_condition = torch.cat(
            [slot_flat, phase_amp_condition.to(slot_flat.dtype)], dim=-1
        )

        phase_timestep = torch.randint(
            0, self.num_train_timesteps, (n,), device=device, dtype=torch.long
        )
        phase_t = (phase_timestep.float() + 0.5) / float(self.num_train_timesteps)
        if self.config.phase_process == "geodesic_flow":
            base_phase = (
                torch.rand(n, 3, device=device, dtype=torch.float32)
                * (2.0 * math.pi)
                - math.pi
            )
            phase_target = wrap_angle(target_phase.float() - base_phase)
            noisy_phase = wrap_angle(base_phase + phase_t[:, None] * phase_target)
            phase_normalizer = torch.ones(n, 1, device=device)
        else:
            sigma = self._phase_sigma(phase_t).unsqueeze(-1)
            phase_noise = torch.randn_like(target_phase.float())
            displacement = sigma * phase_noise
            noisy_phase = wrap_angle(target_phase.float() + displacement)
            phase_target = wrapped_normal_score(displacement, sigma)
            phase_normalizer = self.phase_score_norm[phase_timestep].unsqueeze(-1)
        phase_input = torch.cat([torch.cos(noisy_phase), torch.sin(noisy_phase)], dim=-1)
        phase_output = self.phase_net(
            phase_input.to(z_flat.dtype),
            phase_t * float(self.num_train_timesteps - 1),
            z_flat,
            target_condition=phase_condition,
        ).float()
        angular_error = phase_output - phase_target
        gate = relative_amp.square() / (
            relative_amp.square() + float(self.config.phase_gate) ** 2
        )
        phase_per_example = (
            gate * angular_error.square() / phase_normalizer
        ).sum(-1) / gate.sum(-1).clamp_min(1e-6)
        if self.config.phase_process == "geodesic_flow":
            predicted_phase = wrap_angle(
                noisy_phase + (1.0 - phase_t[:, None]) * phase_output
            )
        else:
            predicted_phase = wrap_angle(noisy_phase + sigma.square() * phase_output)

        predicted_raw = polar_to_cartesian(
            predicted_log_amp,
            predicted_phase,
            scale_flat,
            self_flat,
            self.config.log_epsilon,
            (
                self.amplitude_coordinate_mean
                if self.config.amplitude_standardization != "none"
                else None
            ),
            (
                self.amplitude_coordinate_std
                if self.config.amplitude_standardization != "none"
                else None
            ),
        )
        component_mask = torch.ones_like(predicted_raw)
        component_mask[self_flat, 3:] = 0.0
        cartesian_error = (predicted_raw - raw_flat) * component_mask
        cartesian_per_example = cartesian_error.square().sum(-1) / component_mask.sum(-1)
        # One global denominator preserves the physical frequency hierarchy.
        target_energy = (
            raw_flat.square() * component_mask
        ).sum(-1) / component_mask.sum(-1)
        cartesian_normalizer = target_energy.mean().detach().clamp_min(1e-8)
        normalized_cartesian = cartesian_per_example / cartesian_normalizer

        total_per_example = (
            float(self.config.amplitude_loss_weight) * amp_per_example
            + float(self.config.phase_loss_weight) * phase_per_example
            + float(self.config.cartesian_loss_weight) * normalized_cartesian
        )
        ones = torch.ones_like(total_per_example)
        out: Dict[str, torch.Tensor] = {
            "loss": total_per_example.mean(),
            "unweighted_mse": total_per_example.mean().detach(),
            "weighted_loss": total_per_example.mean().detach(),
            "per_example": total_per_example.detach(),
            "normalized_per_example": total_per_example.detach(),
            "timesteps": amp_timestep.detach(),
            "weights": ones,
            "snr_weights": ones,
            "amplitude_flow_loss": amp_per_example.mean().detach(),
            "phase_flow_loss": phase_per_example.mean().detach(),
            "cartesian_reconstruction_loss": normalized_cartesian.mean().detach(),
            "phase_predicted_amplitude_fraction": choose_predicted.float().mean().detach(),
        }
        if radius_bin is not None:
            rb = radius_bin[None].expand(batch, -1).reshape(-1)
            out["radius_bin"] = rb.repeat(mul).detach() if mul > 1 else rb.detach()
        return out

    def _amplitude_velocity(
        self,
        state: torch.Tensor,
        t: float,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
    ) -> torch.Tensor:
        time = torch.full(
            (state.shape[0],),
            t * float(self.num_train_timesteps - 1),
            device=state.device,
            dtype=torch.float32,
        )
        return self.amplitude_net(
            state.to(z.dtype), time, z, target_condition=slot_condition
        ).float()

    def _phase_velocity(
        self,
        phase: torch.Tensor,
        t: float,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time = torch.full(
            (phase.shape[0],),
            t * float(self.num_train_timesteps - 1),
            device=phase.device,
            dtype=torch.float32,
        )
        phasor = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        output = self.phase_net(
            phasor.to(z.dtype), time, z, target_condition=condition
        ).float()
        if self.config.phase_process == "wrapped_normal_score":
            sigma = self._phase_sigma(t)
            return (
                math.log(self.config.phase_sigma_max / self.config.phase_sigma_min)
                * sigma**2
                * output
            )
        return output

    def _phase_sigma(self, generative_time: torch.Tensor | float) -> torch.Tensor | float:
        log_max = math.log(self.config.phase_sigma_max)
        log_ratio = math.log(self.config.phase_sigma_min) - log_max
        if isinstance(generative_time, torch.Tensor):
            return torch.exp(log_max + generative_time.float() * log_ratio)
        return math.exp(log_max + float(generative_time) * log_ratio)

    @torch.no_grad()
    def sample_coordinates(
        self,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequentially sample log amplitude and circular phase."""
        n = z.shape[0]
        log_amp = torch.randn(
            n, 3, device=z.device, dtype=torch.float32, generator=generator
        ) * float(temperature)
        dt = 1.0 / float(steps)
        for index in range(steps):
            t = index / float(steps)
            v0 = self._amplitude_velocity(log_amp, t, z, slot_condition)
            proposed = log_amp + dt * v0
            if self.base.flow_solver == "heun" and index + 1 < steps:
                v1 = self._amplitude_velocity(proposed, t + dt, z, slot_condition)
                log_amp = log_amp + 0.5 * dt * (v0 + v1)
            else:
                log_amp = proposed

        condition = torch.cat([slot_condition, log_amp.to(slot_condition.dtype)], dim=-1)
        phase = (
            torch.rand(n, 3, device=z.device, dtype=torch.float32, generator=generator)
            * (2.0 * math.pi)
            - math.pi
        )
        for index in range(steps):
            t = index / float(steps)
            v0 = self._phase_velocity(phase, t, z, condition)
            proposed = wrap_angle(phase + dt * v0)
            if self.base.flow_solver == "heun" and index + 1 < steps:
                v1 = self._phase_velocity(proposed, t + dt, z, condition)
                phase = wrap_angle(phase + 0.5 * dt * (v0 + v1))
            else:
                phase = proposed
        return log_amp.to(z.dtype), phase.to(z.dtype)

    @torch.no_grad()
    def predict_coordinates_deterministic(
        self,
        raw_target: torch.Tensor,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        amplitude_scale: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-shot endpoint estimates for the fixed held-out diagnostic panel."""
        original_shape = raw_target.shape[:-1]
        log_amp, target_phase, _ = self.target_coordinates(
            raw_target, amplitude_scale[None]
        )
        log_amp = log_amp.reshape(-1, 3)
        target_phase = target_phase.reshape(-1, 3)
        z = z.reshape(-1, z.shape[-1])
        slot_condition = slot_condition.reshape(-1, slot_condition.shape[-1])
        timesteps = timesteps.reshape(-1).to(device=z.device, dtype=torch.long)
        noise = noise.reshape(-1, 6).float()
        t = (timesteps.float() + 0.5) / float(self.num_train_timesteps)

        amp_noise = noise[:, :3]
        amp_noisy = t[:, None] * log_amp + (1.0 - t[:, None]) * amp_noise
        amp_velocity = self.amplitude_net(
            amp_noisy.to(z.dtype),
            t * float(self.num_train_timesteps - 1),
            z,
            target_condition=slot_condition,
        ).float()
        predicted_log_amp = amp_noisy + (1.0 - t[:, None]) * amp_velocity

        condition = torch.cat(
            [slot_condition, predicted_log_amp.to(slot_condition.dtype)], dim=-1
        )
        if self.config.phase_process == "geodesic_flow":
            # Gaussian CDF maps the pre-existing deterministic noise panel to a
            # deterministic uniform base measure for each circle.
            uniform = 0.5 * (1.0 + torch.erf(noise[:, 3:] / math.sqrt(2.0)))
            base_phase = uniform * (2.0 * math.pi) - math.pi
            angular_target = wrap_angle(target_phase - base_phase)
            noisy_phase = wrap_angle(base_phase + t[:, None] * angular_target)
        else:
            sigma = self._phase_sigma(t).unsqueeze(-1)
            noisy_phase = wrap_angle(target_phase + sigma * noise[:, 3:])
        phasor = torch.cat([torch.cos(noisy_phase), torch.sin(noisy_phase)], dim=-1)
        phase_output = self.phase_net(
            phasor.to(z.dtype),
            t * float(self.num_train_timesteps - 1),
            z,
            target_condition=condition,
        ).float()
        if self.config.phase_process == "geodesic_flow":
            predicted_phase = wrap_angle(
                noisy_phase + (1.0 - t[:, None]) * phase_output
            )
        else:
            predicted_phase = wrap_angle(noisy_phase + sigma.square() * phase_output)
        return (
            predicted_log_amp.reshape(*original_shape, 3).to(z.dtype),
            predicted_phase.reshape(*original_shape, 3).to(z.dtype),
        )
