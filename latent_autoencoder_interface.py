"""Frozen target-12 sequential-ring autoencoder interface for latent AFIG."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

from autoencoder_models import AutoencoderConfig, CausalFrequencyAutoencoder
from frequency import FrequencyCodec, FrequencyCodecConfig
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


POSITION_FEATURE_SCHEMA: List[str] = [
    "sequence_index",
    "ring_index",
    "sector_slot",
    "sector_count",
    "coefficient_count",
    "angle_center_sin",
    "angle_center_cos",
    "angle_span",
    "radius_center",
    "kx_center",
    "ky_center",
]


def _load_frozen_autoencoder(
    checkpoint_path: str,
) -> tuple[CausalFrequencyAutoencoder, FrequencyCodec, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = AutoencoderConfig(**payload["config"])
    if config.mode != "causal_ring" or config.target_tokens_per_latent != 12:
        raise ValueError("Latent AFIG requires the target-12 sequential-ring codec")
    if config.latent_dim != LATENT_TOKEN_DIM:
        raise ValueError(f"Expected latent_dim={LATENT_TOKEN_DIM}")
    codec_payload = payload["codec"]
    codec = FrequencyCodec(FrequencyCodecConfig(**codec_payload["config"]))
    codec.load_exported(codec_payload)
    codec_metadata = codec.position_metadata()
    codec_metadata["empirical_scale"] = codec.orbit_scale_for_policy(
        codec.effective_scale_policy()
    ).mean(dim=-1)
    autoencoder = CausalFrequencyAutoencoder(
        config, codec_metadata, codec.component_mask
    )
    incompatible = autoencoder.load_state_dict(payload["model"], strict=False)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith("token_latent")
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible autoencoder checkpoint: "
            f"missing={disallowed_missing}, unexpected={incompatible.unexpected_keys}"
        )
    if autoencoder.exported_token_count != LATENT_SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {LATENT_SEQUENCE_LENGTH} exported latents, got "
            f"{autoencoder.exported_token_count}"
        )
    return autoencoder, codec, payload


def build_position_features(
    autoencoder: CausalFrequencyAutoencoder,
    codec: FrequencyCodec,
) -> torch.Tensor:
    """Build fixed target metadata by pooling each sector's Fourier coordinates."""
    layout = autoencoder.layout
    indices = layout.gather_indices
    mask = layout.gather_mask
    count = mask.sum(dim=1).float().clamp_min(1.0)

    def pooled(values: torch.Tensor) -> torch.Tensor:
        gathered = values[indices] * mask.to(values.dtype)
        return gathered.sum(dim=1) / count.to(values.dtype)

    angles = codec.angle[indices]
    sin_center = (torch.sin(angles) * mask).sum(dim=1) / count
    cos_center = (torch.cos(angles) * mask).sum(dim=1) / count
    center = torch.atan2(sin_center, cos_center)
    delta = torch.atan2(
        torch.sin(angles - center[:, None]),
        torch.cos(angles - center[:, None]),
    )
    positive_inf = torch.full_like(delta, torch.inf)
    negative_inf = torch.full_like(delta, -torch.inf)
    angular_span = (
        torch.where(mask, delta, positive_inf).amin(dim=1)
        - torch.where(mask, delta, negative_inf).amax(dim=1)
    ).abs() / (2.0 * torch.pi)

    parent_count = layout.parent_counts[layout.latent_parent].float()
    slot_denominator = (parent_count - 1.0).clamp_min(1.0)
    features = torch.stack(
        [
            torch.arange(layout.num_latents, dtype=torch.float32)
            / max(layout.num_latents - 1, 1),
            layout.latent_parent.float() / max(layout.num_parents - 1, 1),
            layout.latent_slot.float() / slot_denominator,
            parent_count / float(layout.max_parent_latents),
            count / float(layout.max_members),
            sin_center,
            cos_center,
            angular_span,
            pooled(codec.radius) / codec.radius.max().clamp_min(1.0),
            pooled(codec.kx_signed) / codec.kx_signed.abs().max().clamp_min(1.0),
            pooled(codec.ky_signed) / codec.ky_signed.abs().max().clamp_min(1.0),
        ],
        dim=-1,
    )
    if features.shape != (LATENT_SEQUENCE_LENGTH, len(POSITION_FEATURE_SCHEMA)):
        raise RuntimeError(f"Unexpected position feature shape {tuple(features.shape)}")
    return features


def layout_fingerprint(
    autoencoder: CausalFrequencyAutoencoder, codec: FrequencyCodec
) -> str:
    digest = hashlib.sha256()
    for tensor in (
        autoencoder.layout.gather_indices,
        autoencoder.layout.gather_mask,
        autoencoder.layout.latent_parent,
        autoencoder.layout.latent_slot,
        codec.ky,
        codec.kx,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    # Fields added after existing checkpoints were written are omitted from the
    # hash when they hold their default, so a default-valued new option does not
    # invalidate every prior generative checkpoint.  Non-default values *are*
    # hashed, so partially-whitened codecs can never be confused with the
    # fully-whitened ones.  validate_compatible() still checks the field in all
    # cases, so this only relaxes the hash, not the contract.
    codec_fingerprint = dict(codec.config.fingerprint())
    for field, default in (
        ("whiten_exponent", 1.0),
        ("coordinate_packing", "legacy"),
        ("ecs_percentile", 98.25),
    ):
        if codec_fingerprint.get(field) == default:
            codec_fingerprint.pop(field, None)
    digest.update(
        json.dumps(
            {
                "autoencoder": autoencoder.config.fingerprint(),
                "codec": codec_fingerprint,
                "schema": POSITION_FEATURE_SCHEMA,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


class FrozenLatentAutoencoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        latent_interface_path: str,
        sample_posterior: bool | None = None,
    ):
        super().__init__()
        checkpoint_path = os.path.abspath(checkpoint_path)
        latent_interface_path = os.path.abspath(latent_interface_path)
        autoencoder, codec, checkpoint = _load_frozen_autoencoder(checkpoint_path)
        interface = torch.load(
            latent_interface_path, map_location="cpu", weights_only=False
        )
        interface_checkpoint = os.path.abspath(interface["checkpoint"])
        if os.path.realpath(interface_checkpoint) != os.path.realpath(checkpoint_path):
            raise ValueError(
                "Latent statistics were fitted for a different autoencoder checkpoint"
            )
        mean = interface["latent_mean"].float()
        std = interface["latent_std"].float()
        expected = (LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"latent_mean/std must both be {expected}, got "
                f"{tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Latent statistics contain non-finite values")
        if (std <= 0).any():
            raise ValueError("Latent standard deviations must be positive")
        fitted_sample_posterior = bool(interface.get("sample_posterior", False))
        if (
            sample_posterior is not None
            and sample_posterior != fitted_sample_posterior
        ):
            raise ValueError(
                "Requested posterior sampling does not match the latent interface"
            )

        self.autoencoder = autoencoder
        self.codec = codec
        self.register_buffer("latent_mean", mean, persistent=True)
        self.register_buffer("latent_std", std.clamp_min(1e-6), persistent=True)
        self.register_buffer(
            "position_features",
            build_position_features(autoencoder, codec),
            persistent=True,
        )
        self.checkpoint_path = checkpoint_path
        self.latent_interface_path = latent_interface_path
        self.autoencoder_global_step = int(checkpoint["global_step"])
        self.layout_hash = layout_fingerprint(autoencoder, codec)
        self.probe_validation_mse = float(interface.get("probe_validation_mse", float("nan")))
        self.zero_baseline_mse = float(interface.get("zero_baseline_mse", 1.0))
        self.sample_posterior = fitted_sample_posterior
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()
        self.codec.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.autoencoder.eval()
        self.codec.eval()
        return self

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean.to(latents.dtype)) / self.latent_std.to(
            latents.dtype
        )

    def denormalize(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return (
            normalized_latents * self.latent_std.to(normalized_latents.dtype)
            + self.latent_mean.to(normalized_latents.dtype)
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.codec.encode(images.float())
        latents = self.autoencoder.export_latents(
            tokens, sample_posterior=self.sample_posterior
        )["latents"]
        if latents.shape[1:] != (LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM):
            raise RuntimeError(f"Unexpected latent shape {tuple(latents.shape)}")
        return self.normalize(latents)

    @torch.no_grad()
    def decode_latents(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return self.decode_latents_with_grad(normalized_latents).clamp(0.0, 1.0)

    def decode_latents_with_grad(
        self, normalized_latents: torch.Tensor
    ) -> torch.Tensor:
        expected = (LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM)
        if normalized_latents.ndim != 3 or normalized_latents.shape[1:] != expected:
            raise ValueError(f"normalized_latents must be [B,{expected[0]},{expected[1]}]")
        latents = self.denormalize(normalized_latents)
        tokens = self.autoencoder.decode(latents)
        return self.codec.decode(tokens)

    def checkpoint_contract(self) -> Dict[str, Any]:
        return {
            "ae_checkpoint": self.checkpoint_path,
            "ae_global_step": self.autoencoder_global_step,
            "latent_interface": self.latent_interface_path,
            "latent_mean": self.latent_mean.detach().cpu(),
            "latent_std": self.latent_std.detach().cpu(),
            "layout_fingerprint": self.layout_hash,
            "position_feature_schema": POSITION_FEATURE_SCHEMA,
            "position_features": self.position_features.detach().cpu(),
            "token_dim": LATENT_TOKEN_DIM,
            "sequence_length": LATENT_SEQUENCE_LENGTH,
            "sample_posterior": self.sample_posterior,
        }

    def assert_contract_compatible(self, contract: Dict[str, Any]) -> None:
        required_equal = {
            "layout_fingerprint": self.layout_hash,
            "position_feature_schema": POSITION_FEATURE_SCHEMA,
            "token_dim": LATENT_TOKEN_DIM,
            "sequence_length": LATENT_SEQUENCE_LENGTH,
            "sample_posterior": self.sample_posterior,
        }
        for key, expected in required_equal.items():
            if contract.get(key) != expected:
                raise ValueError(f"Incompatible latent checkpoint field: {key}")
        for key, current in (
            ("latent_mean", self.latent_mean.cpu()),
            ("latent_std", self.latent_std.cpu()),
            ("position_features", self.position_features.cpu()),
        ):
            saved = contract.get(key)
            if not isinstance(saved, torch.Tensor) or not torch.equal(saved.cpu(), current):
                raise ValueError(f"Incompatible latent checkpoint field: {key}")
