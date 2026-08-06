"""Tests for the amplitude-first circular FFT decoder."""

from __future__ import annotations

import math

import torch

from diffusion_decoder import DiffusionDecoderConfig
from factorized_polar_decoder import (
    FactorizedPolarConfig,
    FactorizedPolarDecoder,
    polar_to_cartesian,
    wrap_angle,
    wrapped_normal_score,
    wrapped_normal_score_norm_table,
)
from frequency import FrequencyCodec, FrequencyCodecConfig
from model_continuous import (
    ContinuousFFTDecoder,
    ContinuousModelConfig,
    GenerationConfig,
    PolarHistoryConfig,
    TransformerConfig,
)
from train_continuous import _fit_factorized_amplitude_stats


def _codec() -> FrequencyCodec:
    config = FrequencyCodecConfig(
        height=8,
        width=8,
        ordering="square_spiral",
        normalization="global_ecs",
        coordinate_packing="isometric",
    )
    codec = FrequencyCodec(config)
    generator = torch.Generator().manual_seed(7)
    batches = [torch.rand(8, 3, 8, 8, generator=generator) for _ in range(4)]
    codec.fit_from_loader(batches)
    return codec


def _model(
    codec: FrequencyCodec,
    phase_process: str = "geodesic_flow",
) -> ContinuousFFTDecoder:
    config = ContinuousModelConfig(
        codec=codec.config,
        transformer=TransformerConfig(
            width=32,
            num_layers=1,
            num_heads=4,
            ff_mult=2,
            max_seq_len=codec.seq_len,
            position_film=True,
        ),
        diffusion=DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=32,
            depth=1,
            objective="flow",
            prediction_type="v_prediction",
            snr_scale=1.0,
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            phase_process=phase_process,
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    return ContinuousFFTDecoder(config, codec=codec)


def test_wrap_angle_and_polar_reconstruction() -> None:
    angles = torch.tensor([-3 * math.pi, -math.pi, 0.0, math.pi, 3 * math.pi])
    wrapped = wrap_angle(angles)
    assert bool(((wrapped >= -math.pi) & (wrapped < math.pi)).all())

    log_amp = torch.log(torch.tensor([[1.0, 2.0, 0.5]]) + 1e-4)
    phase = torch.tensor([[0.0, math.pi / 2.0, math.pi]])
    scale = torch.ones(1, 3)
    ordinary = polar_to_cartesian(
        log_amp, phase, scale, torch.tensor([False]), 1e-4
    )
    assert torch.allclose(ordinary[0, :3], torch.tensor([1.0, 0.0, -0.5]), atol=1e-5)
    assert torch.allclose(ordinary[0, 3:], torch.tensor([0.0, 2.0, 0.0]), atol=1e-5)
    self_conjugate = polar_to_cartesian(
        log_amp, phase, scale, torch.tensor([True]), 1e-4
    )
    assert torch.equal(self_conjugate[:, 3:], torch.zeros(1, 3))


def test_standardized_log_amplitude_roundtrip() -> None:
    base = DiffusionDecoderConfig(
        target_dim=6,
        z_channels=16,
        width=16,
        depth=1,
        objective="flow",
        prediction_type="v_prediction",
        snr_scale=1.0,
        diffusion_batch_mul=1,
    )
    config = FactorizedPolarConfig(
        enabled=True,
        log_epsilon=0.1,
        amplitude_standardization="global",
    )
    mean = torch.tensor([-0.7, -0.7, -0.7])
    std = torch.tensor([0.8, 0.8, 0.8])
    decoder = FactorizedPolarDecoder(
        base,
        config,
        condition_width=16,
        amplitude_coordinate_mean=mean,
        amplitude_coordinate_std=std,
    )
    raw = torch.tensor(
        [[[1.0, -0.5, 0.25, 0.5, 0.75, -0.125]]], dtype=torch.float32
    )
    scale = torch.tensor([[[2.0, 1.0, 0.5]]])
    coordinate, phase, _ = decoder.target_coordinates(raw, scale)
    reconstructed = polar_to_cartesian(
        coordinate.reshape(-1, 3),
        phase.reshape(-1, 3),
        scale.reshape(-1, 3),
        torch.tensor([False]),
        config.log_epsilon,
        decoder.amplitude_coordinate_mean,
        decoder.amplitude_coordinate_std,
    ).reshape_as(raw)
    torch.testing.assert_close(reconstructed, raw, atol=2e-6, rtol=2e-6)


def test_global_amplitude_coordinate_fit_preserves_one_common_affine_map() -> None:
    codec = _codec()
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(101))
    payload = _fit_factorized_amplitude_stats(
        [images],
        codec,
        log_epsilon=0.1,
        scope="global",
    )
    assert payload["examples"] == 12
    torch.testing.assert_close(payload["mean"], payload["mean"][0].expand(3))
    torch.testing.assert_close(payload["std"], payload["std"][0].expand(3))

    positions = torch.arange(codec.seq_len)
    raw = codec.normalized_to_raw_at(codec.encode(images), positions).double()
    scale = codec.channel_amplitude_scale()[codec.radius_bin[positions]].double()
    amplitude = torch.sqrt(raw[..., :3].square() + raw[..., 3:].square())
    coordinate = torch.log(amplitude / scale[None] + 0.1)
    standardized = (coordinate - payload["mean"].double()) / payload["std"].double()
    torch.testing.assert_close(standardized.mean(), torch.tensor(0.0, dtype=torch.double), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        standardized.square().mean(),
        torch.tensor(1.0, dtype=torch.double),
        atol=1e-6,
        rtol=0,
    )


def test_wrapped_normal_score_is_periodic_and_locally_gaussian() -> None:
    displacement = torch.tensor([[-0.2, 0.1, 0.35]])
    sigma = torch.tensor([[0.12]])
    score = wrapped_normal_score(displacement, sigma)
    shifted = wrapped_normal_score(displacement + 2.0 * math.pi, sigma)
    torch.testing.assert_close(score, shifted)
    torch.testing.assert_close(
        score,
        -displacement / sigma.square(),
        atol=1e-4,
        rtol=1e-4,
    )
    norm = wrapped_normal_score_norm_table(16, 0.01 * math.pi, math.pi)
    assert norm.shape == (16,)
    assert bool(torch.isfinite(norm).all())
    assert bool((norm > 0).all())


def test_selected_position_codec_roundtrip() -> None:
    codec = _codec()
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(11))
    normalized = codec.encode(images)
    positions = torch.tensor([0, 3, 7, codec.seq_len - 1])
    raw = codec.normalized_to_raw_at(normalized[:, positions], positions)
    recovered = codec.raw_to_normalized_at(raw, positions)
    assert torch.allclose(recovered, normalized[:, positions], atol=2e-5, rtol=2e-5)


def test_factorized_model_trains_and_generates_cartesian_history() -> None:
    codec = _codec()
    model = _model(codec)
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(13))
    tokens = codec.encode(images)
    output = model(tokens, corrupt=False)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["amplitude_flow_loss"])
    assert torch.isfinite(output["phase_flow_loss"])
    assert torch.isfinite(output["cartesian_reconstruction_loss"])
    output["loss"].backward()
    assert model.factorized_decoder is not None
    assert model.diffusion is None
    assert model.factorized_decoder.amplitude_net.input_proj.weight.grad is not None
    assert model.factorized_decoder.phase_net.input_proj.weight.grad is not None
    assert model.slot_embed.weight.grad is not None

    model.eval()
    with torch.no_grad():
        sampled = model.generate(
            batch_size=2,
            generator=torch.Generator().manual_seed(19),
            num_inference_steps=2,
            return_tokens=True,
            max_tokens=4,
        )["tokens"]
    assert sampled.shape == (2, codec.seq_len, 6)
    assert bool(torch.isfinite(sampled).all())
    self_positions = codec.is_self_conjugate.nonzero(as_tuple=False).flatten()
    assert torch.equal(
        sampled[:, self_positions, 3:], torch.zeros_like(sampled[:, self_positions, 3:])
    )


def test_wrapped_normal_factorized_model_trains_and_generates() -> None:
    codec = _codec()
    model = _model(codec, phase_process="wrapped_normal_score")
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(29))
    output = model(codec.encode(images), corrupt=False)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    model.eval()
    with torch.no_grad():
        sampled = model.generate(
            batch_size=2,
            generator=torch.Generator().manual_seed(31),
            num_inference_steps=2,
            return_tokens=True,
            max_tokens=4,
        )["tokens"]
    assert bool(torch.isfinite(sampled).all())


def test_phase_score_table_is_backward_compatible_with_old_checkpoints() -> None:
    codec = _codec()
    source = _model(codec)
    state = source.state_dict()
    keys = [
        "factorized_decoder.phase_score_norm",
        "factorized_decoder.amplitude_coordinate_mean",
        "factorized_decoder.amplitude_coordinate_std",
    ]
    for key in keys:
        assert key in state
        state.pop(key)
    target = _model(codec)
    target.load_state_dict(state, strict=True)


def test_factorized_cached_and_uncached_prefix_match() -> None:
    codec = _codec()
    model = _model(codec).eval()
    with torch.no_grad():
        cached = model.generate(
            batch_size=1,
            generator=torch.Generator().manual_seed(23),
            num_inference_steps=2,
            return_tokens=True,
            max_tokens=4,
        )["tokens"][:, :4]
        uncached = model.generate_uncached_prefix(
            batch_size=1,
            generator=torch.Generator().manual_seed(23),
            num_inference_steps=2,
            max_tokens=4,
        )
    assert torch.allclose(cached, uncached, atol=2e-5, rtol=2e-5)


def test_standardized_polar_history_replaces_cartesian_trunk_input() -> None:
    codec = _codec()
    config = ContinuousModelConfig(
        codec=codec.config,
        transformer=TransformerConfig(
            width=32,
            num_layers=1,
            num_heads=4,
            ff_mult=2,
            max_seq_len=codec.seq_len,
        ),
        diffusion=DiffusionDecoderConfig(
            target_dim=6,
            z_channels=32,
            width=32,
            depth=1,
            objective="flow",
            prediction_type="v_prediction",
            snr_scale=1.0,
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        polar_history=PolarHistoryConfig(
            enabled=True,
            mode="standardized_log_amp_gated_phase",
            fusion="replace",
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            log_epsilon=0.1,
            amplitude_standardization="global",
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    mean = torch.full((3,), -0.4)
    std = torch.full((3,), 0.7)
    model = ContinuousFFTDecoder(
        config,
        codec=codec,
        factorized_amplitude_mean=mean,
        factorized_amplitude_std=std,
    )
    assert model.token_proj.in_features == 9
    assert model.polar_proj is None
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(211))
    tokens = codec.encode(images)
    positions = torch.arange(codec.seq_len - 1)
    history = model._polar_history_features(tokens[:, :-1], positions)
    raw = codec.normalized_to_raw_at(tokens[:, :-1], positions)
    coordinate, _, _ = model.factorized_decoder.target_coordinates(
        raw,
        model.factorized_amplitude_scale(positions)[None],
    )
    torch.testing.assert_close(history[..., 0::3], coordinate)
    output = model(tokens, corrupt=False)
    output["loss"].backward()
    assert model.token_proj.weight.grad is not None

    model.eval()
    with torch.no_grad():
        cached = model.generate(
            batch_size=1,
            generator=torch.Generator().manual_seed(223),
            num_inference_steps=2,
            return_tokens=True,
            max_tokens=4,
        )["tokens"][:, :4]
        uncached = model.generate_uncached_prefix(
            batch_size=1,
            generator=torch.Generator().manual_seed(223),
            num_inference_steps=2,
            max_tokens=4,
        )
    torch.testing.assert_close(cached, uncached, atol=2e-5, rtol=2e-5)
