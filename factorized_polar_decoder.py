"""Amplitude-first, intrinsic circular decoder for native complex FFT tokens."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_decoder import DiffusionDecoderConfig, SimpleMLPAdaLN


@dataclass(frozen=True)
class FactorizedPolarConfig:
    enabled: bool = False
    log_epsilon: float = 1e-4
    amplitude_transform: str = "log_eps"  # log_eps | log1p | inverse_softplus | power | raw
    # Knee for log1p/inverse_softplus; exponent for power.
    amplitude_transform_parameter: float = 1.0
    amplitude_standardization: str = "none"  # none | global | channel
    # Base Gaussian for the amplitude flow. ``frequency_rms`` scales each
    # frequency/RGB source coordinate by the RMS signal power that remains
    # after the shared population standardization.
    amplitude_source_scale: str = "unit"  # unit | frequency_rms
    condition_fusion: str = "add"  # add | joint_mlp
    amplitude_loss_weight: float = 1.0
    phase_loss_weight: float = 1.0
    cartesian_loss_weight: float = 0.1
    phase_gate: float = 0.1
    phase_predicted_amplitude_probability: float = 0.5
    phase_process: str = "geodesic_flow"
    phase_sigma_min: float = 0.01 * math.pi
    phase_sigma_max: float = math.pi
    # ``relative_raw`` is the legacy v1/v2 coordinate: raw-image FFT amplitude
    # divided by a radial/RGB RMS. ``physical_standardized`` instead operates on
    # the exact isometric FFT of globally standardized pixels, with no
    # frequency-dependent divisor.
    coordinate_mode: str = "relative_raw"
    amplitude_prediction_type: str = "v_prediction"  # v_prediction | x0
    phase_weighting: str = "relative_gate"  # relative_gate | physical_energy
    self_conjugate_sign: str = "phase"  # phase | bernoulli

    def fingerprint(self) -> Dict[str, Any]:
        return asdict(self)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Map radians to [-pi, pi) without changing local derivatives."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def transform_amplitude(
    amplitude: torch.Tensor,
    transform: str,
    *,
    log_epsilon: float,
    parameter: float,
) -> torch.Tensor:
    """Map a nonnegative amplitude to the decoder's Euclidean coordinate."""
    amplitude = amplitude.float().clamp_min(0.0)
    if transform == "log_eps":
        return torch.log(amplitude + float(log_epsilon))
    if transform == "log1p":
        return torch.log1p(amplitude / float(parameter))
    if transform == "inverse_softplus":
        scaled = (amplitude + float(log_epsilon)) / float(parameter)
        # softplus^{-1}(scaled), written stably for both tiny and large scaled.
        return scaled + torch.log(-torch.expm1(-scaled))
    if transform == "power":
        return amplitude.pow(float(parameter))
    if transform == "raw":
        return amplitude
    raise ValueError(f"Unknown amplitude transform: {transform}")


def inverse_transform_amplitude(
    coordinate: torch.Tensor,
    transform: str,
    *,
    log_epsilon: float,
    parameter: float,
) -> torch.Tensor:
    """Invert a decoder coordinate, projecting outside-support values to zero."""
    coordinate = coordinate.float()
    if transform == "log_eps":
        amplitude = torch.exp(coordinate.clamp(min=-16.0, max=8.0)) - float(
            log_epsilon
        )
    elif transform == "log1p":
        amplitude = float(parameter) * torch.expm1(
            coordinate.clamp(min=-16.0, max=8.0)
        )
    elif transform == "inverse_softplus":
        amplitude = float(parameter) * F.softplus(coordinate) - float(log_epsilon)
    elif transform == "power":
        amplitude = coordinate.clamp_min(0.0).pow(1.0 / float(parameter))
    elif transform == "raw":
        amplitude = coordinate
    else:
        raise ValueError(f"Unknown amplitude transform: {transform}")
    return amplitude.clamp_min(0.0)


def cartesian_to_polar_coordinates(
    raw_cartesian: torch.Tensor,
    amplitude_scale: torch.Tensor,
    log_epsilon: float,
    amplitude_coordinate_mean: Optional[torch.Tensor] = None,
    amplitude_coordinate_std: Optional[torch.Tensor] = None,
    amplitude_transform: str = "log_eps",
    amplitude_transform_parameter: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map packed Cartesian coefficients to the shared polar coordinates.

    This conversion is deliberately independent of the decoder module so the
    Transformer history and token decoder can use different representations.
    """
    if (amplitude_coordinate_mean is None) != (amplitude_coordinate_std is None):
        raise ValueError("amplitude coordinate mean and std must be provided together")
    real, imag = raw_cartesian[..., :3], raw_cartesian[..., 3:]
    amplitude = torch.sqrt(real.float().square() + imag.float().square())
    scale = amplitude_scale.to(device=amplitude.device, dtype=amplitude.dtype)
    relative = amplitude / scale.clamp_min(1e-8)
    log_amplitude = transform_amplitude(
        relative,
        amplitude_transform,
        log_epsilon=log_epsilon,
        parameter=amplitude_transform_parameter,
    )
    if amplitude_coordinate_mean is not None:
        mean = amplitude_coordinate_mean.to(log_amplitude)
        std = amplitude_coordinate_std.to(log_amplitude)
        log_amplitude = (log_amplitude - mean) / std
    phase = torch.atan2(imag.float(), real.float())
    return log_amplitude, phase, relative


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
    amplitude_transform: str = "log_eps",
    amplitude_transform_parameter: float = 1.0,
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
    relative_amplitude = inverse_transform_amplitude(
        log_relative_amplitude,
        amplitude_transform,
        log_epsilon=log_epsilon,
        parameter=amplitude_transform_parameter,
    )
    amplitude = relative_amplitude * amplitude_scale
    real = amplitude * torch.cos(phase)
    imag = amplitude * torch.sin(phase)
    imag = imag * (~is_self_conjugate).to(imag.dtype)[:, None]
    return torch.cat([real, imag], dim=-1)


class ConditionalSignHead(nn.Module):
    """Bernoulli logits for the signs of self-conjugate RGB coefficients."""

    def __init__(
        self,
        condition_width: int,
        hidden_width: int,
        amplitude_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * condition_width + amplitude_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, amplitude_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        log_amplitude: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([z, slot_condition, log_amplitude], dim=-1))


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
        amplitude_source_rms: Optional[torch.Tensor] = None,
        coefficients_per_token: int = 1,
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
        if config.amplitude_transform not in (
            "log_eps",
            "log1p",
            "inverse_softplus",
            "power",
            "raw",
        ):
            raise ValueError(
                "amplitude_transform must be log_eps, log1p, inverse_softplus, "
                "power, or raw"
            )
        if config.amplitude_transform_parameter <= 0.0:
            raise ValueError("amplitude_transform_parameter must be positive")
        if config.amplitude_standardization not in ("none", "global", "channel"):
            raise ValueError(
                "amplitude_standardization must be none, global, or channel"
            )
        if config.amplitude_source_scale not in ("unit", "frequency_rms"):
            raise ValueError(
                "amplitude_source_scale must be unit or frequency_rms"
            )
        if config.condition_fusion not in ("add", "joint_mlp"):
            raise ValueError("factorized condition_fusion must be add or joint_mlp")
        if config.phase_process not in ("geodesic_flow", "wrapped_normal_score"):
            raise ValueError(f"Unknown phase process: {config.phase_process}")
        if config.coordinate_mode not in ("relative_raw", "physical_standardized"):
            raise ValueError(
                f"Unknown factorized coordinate mode: {config.coordinate_mode}"
            )
        if config.amplitude_prediction_type not in ("v_prediction", "x0"):
            raise ValueError(
                "factorized amplitude_prediction_type must be v_prediction or x0"
            )
        if config.phase_weighting not in ("relative_gate", "physical_energy"):
            raise ValueError(
                f"Unknown factorized phase weighting: {config.phase_weighting}"
            )
        if config.self_conjugate_sign not in ("phase", "bernoulli"):
            raise ValueError(
                f"Unknown self-conjugate sign treatment: {config.self_conjugate_sign}"
            )
        if (
            config.coordinate_mode == "physical_standardized"
            and config.amplitude_standardization == "none"
        ):
            raise ValueError(
                "physical_standardized polar coordinates require global or channel "
                "amplitude standardization"
            )
        if not 0.0 < config.phase_sigma_min < config.phase_sigma_max:
            raise ValueError("phase sigma bounds must satisfy 0 < min < max")
        if coefficients_per_token <= 0:
            raise ValueError("coefficients_per_token must be positive")

        self.base = base
        self.config = config
        self.coefficients_per_token = int(coefficients_per_token)
        amplitude_dim = 3 * self.coefficients_per_token
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
        source_rms = (
            torch.ones(1, 3, dtype=torch.float32)
            if amplitude_source_rms is None
            else amplitude_source_rms.detach().float()
        )
        if source_rms.ndim != 2 or source_rms.shape[-1] != 3:
            raise ValueError("amplitude source RMS must have shape [L,3] or [1,3]")
        if bool((source_rms <= 0).any()) or not bool(
            torch.isfinite(source_rms).all()
        ):
            raise ValueError("amplitude source RMS must be finite and positive")
        if config.amplitude_source_scale == "unit":
            source_rms = torch.ones_like(source_rms)
        self.register_buffer("amplitude_source_rms", source_rms, persistent=True)
        common = dict(
            model_channels=base.width,
            z_channels=condition_width,
            num_res_blocks=base.depth,
            input_timestep_conditioning=base.input_timestep_conditioning,
            input_projection_init=base.input_projection_init,
            condition_fusion=config.condition_fusion,
        )
        self.amplitude_net = SimpleMLPAdaLN(
            in_channels=amplitude_dim,
            out_channels=amplitude_dim,
            target_condition_dim=condition_width,
            **common,
        )
        # The direct condition is [learned target slot, sampled/predicted log amp].
        self.phase_net = SimpleMLPAdaLN(
            in_channels=2 * amplitude_dim,
            out_channels=amplitude_dim,
            target_condition_dim=condition_width + amplitude_dim,
            **common,
        )
        self.sign_net = (
            ConditionalSignHead(condition_width, base.width, amplitude_dim)
            if config.self_conjugate_sign == "bernoulli"
            else None
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
            prefix + "amplitude_source_rms": self.amplitude_source_rms,
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
        return cartesian_to_polar_coordinates(
            raw_cartesian,
            amplitude_scale,
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
            self.config.amplitude_transform,
            self.config.amplitude_transform_parameter,
        )

    def source_rms_for_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Return the Gaussian source scale for absolute frequency positions."""
        positions = positions.to(
            device=self.amplitude_source_rms.device, dtype=torch.long
        )
        if self.amplitude_source_rms.shape[0] == 1:
            return self.amplitude_source_rms.expand(positions.numel(), -1)
        if (
            positions.numel()
            and int(positions.max()) >= self.amplitude_source_rms.shape[0]
        ):
            raise ValueError("amplitude source RMS does not cover all positions")
        return self.amplitude_source_rms[positions]

    def compute_loss(
        self,
        raw_target: torch.Tensor,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        amplitude_scale: torch.Tensor,
        is_self_conjugate: torch.Tensor,
        radius_bin: Optional[torch.Tensor] = None,
        active_coefficient_mask: Optional[torch.Tensor] = None,
        coefficient_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train both flows and a globally normalized Cartesian reconstruction."""
        if self.coefficients_per_token > 1:
            return self._compute_grouped_loss(
                raw_target=raw_target,
                z=z,
                slot_condition=slot_condition,
                amplitude_scale=amplitude_scale,
                is_self_conjugate=is_self_conjugate,
                active_coefficient_mask=active_coefficient_mask,
                coefficient_positions=coefficient_positions,
                radius_bin=radius_bin,
            )
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
        device = log_amp.device
        scale_flat = self._flatten_and_repeat(
            amplitude_scale[None].expand(batch, -1, -1), mul
        ).float()
        source_rms = self.source_rms_for_positions(
            torch.arange(length, device=device)
        )
        source_rms_flat = self._flatten_and_repeat(
            source_rms[None].expand(batch, -1, -1), mul
        ).float()
        self_flat = is_self_conjugate[None, :, None].expand(batch, -1, 1)
        self_flat = self._flatten_and_repeat(self_flat.float(), mul)[:, 0].bool()

        n = log_amp.shape[0]
        amp_timestep = torch.randint(
            0, self.num_train_timesteps, (n,), device=device, dtype=torch.long
        )
        amp_t = (amp_timestep.float() + 0.5) / float(self.num_train_timesteps)
        amp_source = torch.randn_like(log_amp) * source_rms_flat
        amp_noisy = amp_t[:, None] * log_amp + (1.0 - amp_t[:, None]) * amp_source
        amp_velocity_target = log_amp - amp_source
        amp_output = self.amplitude_net(
            amp_noisy,
            amp_t * float(self.num_train_timesteps - 1),
            z_flat,
            target_condition=slot_flat,
        ).float()
        if self.config.amplitude_prediction_type == "x0":
            amp_per_example = (amp_output - log_amp.float()).square().mean(-1)
            predicted_log_amp = amp_output
        else:
            amp_per_example = (
                amp_output - amp_velocity_target.float()
            ).square().mean(-1)
            predicted_log_amp = (
                amp_noisy.float() + (1.0 - amp_t[:, None]) * amp_output
            )

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
        ordinary = (~self_flat).to(relative_amp.dtype)[:, None]
        if self.config.phase_weighting == "physical_energy":
            phase_weight = relative_amp.square() * ordinary
            # One scalar normalization retains the cross-frequency hierarchy;
            # unlike a per-token denominator it does not make weak bands count
            # as much as low-frequency, high-energy bands.
            active = ordinary.sum().clamp_min(1.0) * relative_amp.shape[-1]
            weight_mean = (phase_weight.sum() / active).detach().clamp_min(1e-8)
            phase_per_example = (
                phase_weight * angular_error.square() / phase_normalizer
            ).sum(-1) / (relative_amp.shape[-1] * weight_mean)
        else:
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
            self.config.amplitude_transform,
            self.config.amplitude_transform_parameter,
        )
        sign_per_example = torch.zeros_like(amp_per_example)
        if self.sign_net is not None:
            sign_logits = self.sign_net(
                z_flat,
                slot_flat,
                phase_amp_condition.to(slot_flat.dtype),
            ).float()
            target_positive = (raw_flat[:, :3] >= 0.0).float()
            sign_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                sign_logits,
                target_positive,
                reduction="none",
            ).mean(-1)
            sign_per_example = sign_bce * self_flat.to(sign_bce.dtype)

            # For the differentiable Cartesian endpoint estimate, use the
            # Bernoulli mean in {-1,+1}. Actual rollout samples the sign.
            if bool(self_flat.any()):
                coordinate = predicted_log_amp
                if self.config.amplitude_standardization != "none":
                    coordinate = (
                        coordinate * self.amplitude_coordinate_std.to(coordinate)
                        + self.amplitude_coordinate_mean.to(coordinate)
                    )
                predicted_amplitude = inverse_transform_amplitude(
                    coordinate,
                    self.config.amplitude_transform,
                    log_epsilon=self.config.log_epsilon,
                    parameter=self.config.amplitude_transform_parameter,
                ) * scale_flat
                expected_sign = torch.tanh(0.5 * sign_logits)
                predicted_raw = predicted_raw.clone()
                predicted_raw[self_flat, :3] = (
                    predicted_amplitude[self_flat] * expected_sign[self_flat]
                )
                predicted_raw[self_flat, 3:] = 0.0
        component_mask = torch.ones_like(predicted_raw)
        component_mask[self_flat, 3:] = 0.0
        cartesian_error = (predicted_raw - raw_flat) * component_mask
        active_component_count = component_mask.sum(-1)
        if self.base.component_reduction == "fixed_dim":
            cartesian_denom = torch.full_like(
                active_component_count, float(self.base.target_dim)
            )
        else:
            cartesian_denom = active_component_count
        cartesian_per_example = cartesian_error.square().sum(-1) / cartesian_denom
        # One global denominator preserves the physical frequency hierarchy.
        target_energy_sum = (raw_flat.square() * component_mask).sum(-1)
        target_energy = target_energy_sum / cartesian_denom
        cartesian_normalizer = target_energy.mean().detach().clamp_min(1e-8)
        normalized_cartesian = cartesian_per_example / cartesian_normalizer

        total_per_example = (
            float(self.config.amplitude_loss_weight) * amp_per_example
            + float(self.config.phase_loss_weight) * phase_per_example
            + float(self.config.phase_loss_weight) * sign_per_example
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
            "self_conjugate_sign_loss": (
                sign_per_example.sum() / self_flat.sum().clamp_min(1)
            ).detach(),
            "cartesian_reconstruction_loss": normalized_cartesian.mean().detach(),
            "phase_predicted_amplitude_fraction": choose_predicted.float().mean().detach(),
            # Component-level arrays support numerical audits.  Amplitude and
            # phase use independent flow times, so attributing the aggregate
            # objective to ``amp_timestep`` alone is not meaningful.
            "amplitude_timesteps": amp_timestep.detach(),
            "phase_timesteps": phase_timestep.detach(),
            "amplitude_per_example": amp_per_example.detach(),
            "phase_per_example": phase_per_example.detach(),
            "sign_per_example": sign_per_example.detach(),
            "cartesian_per_example": normalized_cartesian.detach(),
            "cartesian_raw_per_example": cartesian_per_example.detach(),
            "target_energy_per_example": target_energy.detach(),
            "target_energy_sum_per_example": target_energy_sum.detach(),
            "active_component_count_per_example": active_component_count.detach(),
        }
        if radius_bin is not None:
            rb = radius_bin[None].expand(batch, -1).reshape(-1)
            out["radius_bin"] = rb.repeat(mul).detach() if mul > 1 else rb.detach()
        return out

    def _compute_grouped_loss(
        self,
        raw_target: torch.Tensor,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        amplitude_scale: torch.Tensor,
        is_self_conjugate: torch.Tensor,
        active_coefficient_mask: Optional[torch.Tensor],
        coefficient_positions: Optional[torch.Tensor],
        radius_bin: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Jointly denoise K coefficients while retaining per-coefficient measure."""
        k = self.coefficients_per_token
        if raw_target.ndim != 4 or raw_target.shape[-2:] != (k, 6):
            raise ValueError(f"raw_target must have shape [B,G,{k},6]")
        batch, groups = raw_target.shape[:2]
        if z.shape[:2] != (batch, groups) or slot_condition.shape[:2] != (
            batch,
            groups,
        ):
            raise ValueError("z and slot_condition must align with grouped targets")
        expected_group_shape = (groups, k)
        if amplitude_scale.shape != (groups, k, 3):
            raise ValueError("amplitude_scale must have shape [G,K,3]")
        if is_self_conjugate.shape != expected_group_shape:
            raise ValueError("is_self_conjugate must have shape [G,K]")
        if active_coefficient_mask is None:
            active_coefficient_mask = torch.ones(
                expected_group_shape,
                device=raw_target.device,
                dtype=torch.bool,
            )
        if active_coefficient_mask.shape != expected_group_shape:
            raise ValueError("active_coefficient_mask must have shape [G,K]")
        if coefficient_positions is None:
            raise ValueError("grouped decoding requires coefficient_positions")
        if coefficient_positions.shape != expected_group_shape:
            raise ValueError("coefficient_positions must have shape [G,K]")

        mul = int(self.base.diffusion_batch_mul)

        def flatten_groups(value: torch.Tensor) -> torch.Tensor:
            value = value.reshape(batch * groups, *value.shape[2:])
            repeats = (mul,) + (1,) * (value.ndim - 1)
            return value.repeat(repeats) if mul > 1 else value

        log_amp, target_phase, relative_amp = self.target_coordinates(
            raw_target, amplitude_scale[None]
        )
        log_amp = flatten_groups(log_amp)
        target_phase = flatten_groups(target_phase)
        relative_amp = flatten_groups(relative_amp)
        raw_flat = flatten_groups(raw_target.float())
        z_flat = self._flatten_and_repeat(z, mul)
        slot_flat = self._flatten_and_repeat(slot_condition, mul)
        scale_flat = flatten_groups(
            amplitude_scale[None].expand(batch, -1, -1, -1)
        ).float()
        active = flatten_groups(
            active_coefficient_mask[None].expand(batch, -1, -1)
        ).bool()
        self_mask = flatten_groups(
            is_self_conjugate[None].expand(batch, -1, -1)
        ).bool() & active
        source_rms = self.source_rms_for_positions(
            coefficient_positions.reshape(-1)
        ).reshape(groups, k, 3)
        source_rms = flatten_groups(source_rms[None].expand(batch, -1, -1, -1))

        active_rgb = active[..., None].expand(-1, -1, 3)
        active_count = active.sum(-1).float().clamp_min(1.0)
        group_weight = active_count
        group_weight_sum = group_weight.sum().clamp_min(1.0)

        def weighted_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * group_weight).sum() / group_weight_sum

        n = log_amp.shape[0]
        device = log_amp.device
        amp_target = log_amp.reshape(n, 3 * k)
        amp_mask = active_rgb.reshape(n, 3 * k).to(amp_target.dtype)
        amp_timestep = torch.randint(
            0, self.num_train_timesteps, (n,), device=device, dtype=torch.long
        )
        amp_t = (amp_timestep.float() + 0.5) / float(self.num_train_timesteps)
        amp_source = (
            torch.randn_like(log_amp) * source_rms * active_rgb.to(log_amp.dtype)
        ).reshape(n, 3 * k)
        amp_noisy = (
            amp_t[:, None] * amp_target + (1.0 - amp_t[:, None]) * amp_source
        ) * amp_mask
        amp_velocity_target = (amp_target - amp_source) * amp_mask
        amp_output = self.amplitude_net(
            amp_noisy,
            amp_t * float(self.num_train_timesteps - 1),
            z_flat,
            target_condition=slot_flat,
        ).float()
        if self.config.amplitude_prediction_type == "x0":
            amp_error = (amp_output - amp_target.float()) * amp_mask
            predicted_amp = amp_output * amp_mask
        else:
            amp_error = (amp_output - amp_velocity_target.float()) * amp_mask
            predicted_amp = (
                amp_noisy.float() + (1.0 - amp_t[:, None]) * amp_output
            ) * amp_mask
        amp_per_example = amp_error.square().sum(-1) / (3.0 * active_count)
        predicted_log_amp = predicted_amp.reshape(n, k, 3)

        choose_predicted = torch.rand(n, 1, 1, device=device) < float(
            self.config.phase_predicted_amplitude_probability
        )
        phase_amp_condition = torch.where(
            choose_predicted,
            predicted_log_amp.detach(),
            log_amp.float(),
        ) * active_rgb
        phase_condition = torch.cat(
            [slot_flat, phase_amp_condition.reshape(n, 3 * k).to(slot_flat.dtype)],
            dim=-1,
        )

        phase_timestep = torch.randint(
            0, self.num_train_timesteps, (n,), device=device, dtype=torch.long
        )
        phase_t = (phase_timestep.float() + 0.5) / float(self.num_train_timesteps)
        if self.config.phase_process == "geodesic_flow":
            base_phase = (
                torch.rand(n, k, 3, device=device, dtype=torch.float32)
                * (2.0 * math.pi)
                - math.pi
            )
            phase_target = wrap_angle(target_phase.float() - base_phase)
            noisy_phase = wrap_angle(base_phase + phase_t[:, None, None] * phase_target)
            phase_normalizer = torch.ones(n, 1, 1, device=device)
        else:
            sigma = self._phase_sigma(phase_t).view(n, 1, 1)
            displacement = sigma * torch.randn_like(target_phase.float())
            noisy_phase = wrap_angle(target_phase.float() + displacement)
            phase_target = wrapped_normal_score(displacement, sigma)
            phase_normalizer = self.phase_score_norm[phase_timestep].view(n, 1, 1)
        phase_input = torch.cat(
            [torch.cos(noisy_phase).reshape(n, 3 * k),
             torch.sin(noisy_phase).reshape(n, 3 * k)],
            dim=-1,
        )
        phase_output = self.phase_net(
            phase_input.to(z_flat.dtype),
            phase_t * float(self.num_train_timesteps - 1),
            z_flat,
            target_condition=phase_condition,
        ).float().reshape(n, k, 3)
        angular_error = (phase_output - phase_target) * active_rgb
        ordinary = (active & ~self_mask)[..., None]
        if self.config.phase_weighting == "physical_energy":
            phase_weight = relative_amp.square() * ordinary
            weight_mean = (
                phase_weight.sum()
                / (ordinary.sum().clamp_min(1) * 3.0)
            ).detach().clamp_min(1e-8)
            phase_per_example = (
                phase_weight * angular_error.square() / phase_normalizer
            ).sum(dim=(-1, -2)) / (3.0 * active_count * weight_mean)
        else:
            gate = (
                relative_amp.square()
                / (relative_amp.square() + float(self.config.phase_gate) ** 2)
            ) * ordinary
            phase_per_example = (
                gate * angular_error.square() / phase_normalizer
            ).sum(dim=(-1, -2)) / gate.sum(dim=(-1, -2)).clamp_min(1e-6)
        if self.config.phase_process == "geodesic_flow":
            predicted_phase = wrap_angle(
                noisy_phase + (1.0 - phase_t[:, None, None]) * phase_output
            )
        else:
            predicted_phase = wrap_angle(noisy_phase + sigma.square() * phase_output)

        predicted_raw = polar_to_cartesian(
            predicted_log_amp.reshape(-1, 3),
            predicted_phase.reshape(-1, 3),
            scale_flat.reshape(-1, 3),
            self_mask.reshape(-1),
            self.config.log_epsilon,
            self.amplitude_coordinate_mean
            if self.config.amplitude_standardization != "none"
            else None,
            self.amplitude_coordinate_std
            if self.config.amplitude_standardization != "none"
            else None,
            self.config.amplitude_transform,
            self.config.amplitude_transform_parameter,
        ).reshape(n, k, 6)

        sign_per_example = torch.zeros_like(amp_per_example)
        if self.sign_net is not None:
            sign_logits = self.sign_net(
                z_flat,
                slot_flat,
                phase_amp_condition.reshape(n, 3 * k).to(slot_flat.dtype),
            ).float().reshape(n, k, 3)
            target_positive = (raw_flat[..., :3] >= 0.0).float()
            sign_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                sign_logits, target_positive, reduction="none"
            )
            sign_per_example = (
                sign_bce * self_mask[..., None]
            ).sum(dim=(-1, -2)) / (3.0 * active_count)
            if bool(self_mask.any()):
                coordinate = predicted_log_amp
                if self.config.amplitude_standardization != "none":
                    coordinate = (
                        coordinate * self.amplitude_coordinate_std.to(coordinate)
                        + self.amplitude_coordinate_mean.to(coordinate)
                    )
                predicted_amplitude = inverse_transform_amplitude(
                    coordinate,
                    self.config.amplitude_transform,
                    log_epsilon=self.config.log_epsilon,
                    parameter=self.config.amplitude_transform_parameter,
                ) * scale_flat
                expected_sign = torch.tanh(0.5 * sign_logits)
                predicted_raw = predicted_raw.clone()
                predicted_raw[..., :3] = torch.where(
                    self_mask[..., None], predicted_amplitude * expected_sign,
                    predicted_raw[..., :3],
                )
                predicted_raw[..., 3:] = torch.where(
                    self_mask[..., None], torch.zeros_like(predicted_raw[..., 3:]),
                    predicted_raw[..., 3:],
                )

        component_mask = active[..., None].expand(-1, -1, 6).clone()
        component_mask[..., 3:] &= ~self_mask[..., None]
        component_mask_f = component_mask.to(predicted_raw.dtype)
        cartesian_error = (predicted_raw - raw_flat) * component_mask_f
        active_component_count = component_mask_f.sum(dim=(-1, -2))
        if self.base.component_reduction == "fixed_dim":
            cartesian_denom = 6.0 * active_count
            population_denom = 6.0 * group_weight_sum
        else:
            cartesian_denom = active_component_count.clamp_min(1.0)
            population_denom = active_component_count.sum().clamp_min(1.0)
        cartesian_per_example = (
            cartesian_error.square().sum(dim=(-1, -2)) / cartesian_denom
        )
        target_energy_sum = (raw_flat.square() * component_mask_f).sum(dim=(-1, -2))
        target_energy = target_energy_sum / cartesian_denom
        cartesian_normalizer = (
            target_energy_sum.sum() / population_denom
        ).detach().clamp_min(1e-8)
        normalized_cartesian = cartesian_per_example / cartesian_normalizer

        total_per_example = (
            float(self.config.amplitude_loss_weight) * amp_per_example
            + float(self.config.phase_loss_weight) * phase_per_example
            + float(self.config.phase_loss_weight) * sign_per_example
            + float(self.config.cartesian_loss_weight) * normalized_cartesian
        )
        total_loss = weighted_mean(total_per_example)
        ones = torch.ones_like(total_per_example)
        self_count = self_mask.sum().clamp_min(1)
        out: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "unweighted_mse": total_loss.detach(),
            "weighted_loss": total_loss.detach(),
            "per_example": total_per_example.detach(),
            "normalized_per_example": total_per_example.detach(),
            "timesteps": amp_timestep.detach(),
            "weights": group_weight.detach(),
            "snr_weights": ones,
            "amplitude_flow_loss": weighted_mean(amp_per_example).detach(),
            "phase_flow_loss": weighted_mean(phase_per_example).detach(),
            "self_conjugate_sign_loss": (
                (sign_bce * self_mask[..., None]).sum() / (3.0 * self_count)
                if self.sign_net is not None
                else torch.zeros((), device=device)
            ).detach(),
            "cartesian_reconstruction_loss": weighted_mean(normalized_cartesian).detach(),
            "phase_predicted_amplitude_fraction": choose_predicted.float().mean().detach(),
            "amplitude_timesteps": amp_timestep.detach(),
            "phase_timesteps": phase_timestep.detach(),
            "amplitude_per_example": amp_per_example.detach(),
            "phase_per_example": phase_per_example.detach(),
            "sign_per_example": sign_per_example.detach(),
            "cartesian_per_example": normalized_cartesian.detach(),
            "cartesian_raw_per_example": cartesian_per_example.detach(),
            "target_energy_per_example": target_energy.detach(),
            "target_energy_sum_per_example": target_energy_sum.detach(),
            "active_component_count_per_example": active_component_count.detach(),
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
        output = self.amplitude_net(
            state.to(z.dtype), time, z, target_condition=slot_condition
        ).float()
        if self.config.amplitude_prediction_type == "x0":
            return (output - state) / max(1.0 - float(t), self.base.flow_t_eps)
        return output

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
        is_self_conjugate: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        active_coefficient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequentially sample log amplitude and circular phase."""
        if self.coefficients_per_token > 1:
            return self._sample_grouped_coordinates(
                z=z,
                slot_condition=slot_condition,
                generator=generator,
                steps=steps,
                temperature=temperature,
                is_self_conjugate=is_self_conjugate,
                positions=positions,
                active_coefficient_mask=active_coefficient_mask,
            )
        n = z.shape[0]
        if positions is None:
            if self.config.amplitude_source_scale != "unit":
                raise ValueError("frequency-scaled amplitude sampling requires positions")
            source_rms = torch.ones(n, 3, device=z.device, dtype=torch.float32)
        else:
            positions = positions.to(device=z.device, dtype=torch.long).reshape(-1)
            if positions.numel() == 1 and n != 1:
                positions = positions.expand(n)
            if positions.numel() != n:
                raise ValueError("positions must have one entry per sampled token")
            source_rms = self.source_rms_for_positions(positions).to(z.device)
        log_amp = torch.randn(
            n, 3, device=z.device, dtype=torch.float32, generator=generator
        ) * source_rms * float(temperature)
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
        if self.sign_net is not None:
            if is_self_conjugate is None:
                raise ValueError(
                    "Bernoulli self-conjugate signs require a position mask"
                )
            is_self = is_self_conjugate.to(
                device=z.device, dtype=torch.bool
            ).reshape(-1)
            if bool(is_self.any()):
                logits = self.sign_net(
                    z, slot_condition, log_amp.to(slot_condition.dtype)
                )
                probability = torch.sigmoid(logits.float())
                uniform = torch.rand(
                    probability.shape,
                    device=probability.device,
                    dtype=probability.dtype,
                    generator=generator,
                )
                positive = uniform < probability
                sign_phase = torch.where(
                    positive,
                    torch.zeros_like(probability),
                    torch.full_like(probability, math.pi),
                )
                phase = torch.where(is_self[:, None], sign_phase, phase)
        return log_amp.to(z.dtype), phase.to(z.dtype)

    @torch.no_grad()
    def _sample_grouped_coordinates(
        self,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        generator: Optional[torch.Generator],
        steps: int,
        temperature: float,
        is_self_conjugate: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        active_coefficient_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.coefficients_per_token
        n = z.shape[0]
        if active_coefficient_mask is None:
            active = torch.ones(n, k, device=z.device, dtype=torch.bool)
        else:
            active = active_coefficient_mask.to(device=z.device, dtype=torch.bool)
            if active.ndim == 1:
                active = active[None].expand(n, -1)
            if active.shape != (n, k):
                raise ValueError("active_coefficient_mask must have shape [N,K]")
        if positions is None:
            if self.config.amplitude_source_scale != "unit":
                raise ValueError("frequency-scaled grouped sampling requires positions")
            source_rms = torch.ones(n, k, 3, device=z.device, dtype=torch.float32)
        else:
            positions = positions.to(device=z.device, dtype=torch.long)
            if positions.ndim == 1:
                positions = positions[None].expand(n, -1)
            if positions.shape != (n, k):
                raise ValueError("positions must have shape [N,K]")
            source_rms = self.source_rms_for_positions(
                positions.reshape(-1)
            ).reshape(n, k, 3).to(z.device)
        amp_mask = active[..., None].expand(-1, -1, 3).reshape(n, 3 * k).float()
        log_amp = (
            torch.randn(
                n, 3 * k, device=z.device, dtype=torch.float32, generator=generator
            )
            * source_rms.reshape(n, 3 * k)
            * float(temperature)
            * amp_mask
        )
        dt = 1.0 / float(steps)
        for index in range(steps):
            t = index / float(steps)
            v0 = self._amplitude_velocity(log_amp, t, z, slot_condition) * amp_mask
            proposed = (log_amp + dt * v0) * amp_mask
            if self.base.flow_solver == "heun" and index + 1 < steps:
                v1 = self._amplitude_velocity(
                    proposed, t + dt, z, slot_condition
                ) * amp_mask
                log_amp = (log_amp + 0.5 * dt * (v0 + v1)) * amp_mask
            else:
                log_amp = proposed

        condition = torch.cat(
            [slot_condition, log_amp.to(slot_condition.dtype)], dim=-1
        )
        phase = (
            torch.rand(
                n, 3 * k, device=z.device, dtype=torch.float32, generator=generator
            )
            * (2.0 * math.pi)
            - math.pi
        ) * amp_mask
        for index in range(steps):
            t = index / float(steps)
            v0 = self._phase_velocity(phase, t, z, condition) * amp_mask
            proposed = wrap_angle(phase + dt * v0) * amp_mask
            if self.base.flow_solver == "heun" and index + 1 < steps:
                v1 = self._phase_velocity(proposed, t + dt, z, condition) * amp_mask
                phase = wrap_angle(phase + 0.5 * dt * (v0 + v1)) * amp_mask
            else:
                phase = proposed

        log_amp = log_amp.reshape(n, k, 3)
        phase = phase.reshape(n, k, 3)
        if self.sign_net is not None:
            if is_self_conjugate is None:
                raise ValueError("Bernoulli signs require a grouped position mask")
            is_self = is_self_conjugate.to(device=z.device, dtype=torch.bool)
            if is_self.ndim == 1:
                is_self = is_self[None].expand(n, -1)
            if is_self.shape != (n, k):
                raise ValueError("is_self_conjugate must have shape [N,K]")
            is_self = is_self & active
            if bool(is_self.any()):
                logits = self.sign_net(
                    z,
                    slot_condition,
                    log_amp.reshape(n, 3 * k).to(slot_condition.dtype),
                ).reshape(n, k, 3)
                probability = torch.sigmoid(logits.float())
                uniform = torch.rand(
                    probability.shape,
                    device=probability.device,
                    dtype=probability.dtype,
                    generator=generator,
                )
                sign_phase = torch.where(
                    uniform < probability,
                    torch.zeros_like(probability),
                    torch.full_like(probability, math.pi),
                )
                phase = torch.where(is_self[..., None], sign_phase, phase)
        return log_amp.to(z.dtype), phase.to(z.dtype)

    @torch.no_grad()
    def predict_coordinates_deterministic(
        self,
        raw_target: torch.Tensor,
        z: torch.Tensor,
        slot_condition: torch.Tensor,
        amplitude_scale: torch.Tensor,
        is_self_conjugate: torch.Tensor,
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
        self_flat = (
            is_self_conjugate[None]
            .expand(raw_target.shape[0], -1)
            .reshape(-1)
            .to(device=z.device, dtype=torch.bool)
        )
        t = (timesteps.float() + 0.5) / float(self.num_train_timesteps)

        source_rms = self.source_rms_for_positions(
            torch.arange(raw_target.shape[1], device=z.device)
        )
        source_rms = (
            source_rms[None]
            .expand(raw_target.shape[0], -1, -1)
            .reshape(-1, 3)
        )
        amp_source = noise[:, :3] * source_rms
        amp_noisy = t[:, None] * log_amp + (1.0 - t[:, None]) * amp_source
        amp_output = self.amplitude_net(
            amp_noisy.to(z.dtype),
            t * float(self.num_train_timesteps - 1),
            z,
            target_condition=slot_condition,
        ).float()
        if self.config.amplitude_prediction_type == "x0":
            predicted_log_amp = amp_output
        else:
            predicted_log_amp = amp_noisy + (1.0 - t[:, None]) * amp_output

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
        if self.sign_net is not None:
            # The held-out condition diagnostic is deterministic. Use the mode
            # of the learned Bernoulli rather than consuming additional RNG.
            sign_logits = self.sign_net(
                z,
                slot_condition,
                predicted_log_amp.to(slot_condition.dtype),
            ).float()
            sign_phase = torch.where(
                sign_logits >= 0.0,
                torch.zeros_like(sign_logits),
                torch.full_like(sign_logits, math.pi),
            )
            predicted_phase = torch.where(
                self_flat[:, None], sign_phase, predicted_phase
            )
        return (
            predicted_log_amp.reshape(*original_shape, 3).to(z.dtype),
            predicted_phase.reshape(*original_shape, 3).to(z.dtype),
        )
