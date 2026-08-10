"""Tests for the amplitude-first circular FFT decoder."""

from __future__ import annotations

import math

import torch

from diffusion_decoder import DiffusionDecoderConfig
from factorized_polar_decoder import (
    FactorizedPolarConfig,
    FactorizedPolarDecoder,
    cartesian_to_polar_coordinates,
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


def _physical_codec() -> FrequencyCodec:
    config = FrequencyCodecConfig(
        height=8,
        width=8,
        ordering="square_spiral",
        normalization="global_standardize",
        coordinate_packing="isometric",
    )
    codec = FrequencyCodec(config)
    generator = torch.Generator().manual_seed(37)
    batches = [torch.rand(8, 3, 8, 8, generator=generator) for _ in range(4)]
    codec.fit_from_loader(batches)
    return codec


def _model(
    codec: FrequencyCodec,
    phase_process: str = "geodesic_flow",
    component_reduction: str = "active_mean",
    condition_fusion: str = "add",
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
            component_reduction=component_reduction,
            snr_scale=1.0,
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            phase_process=phase_process,
            condition_fusion=condition_fusion,
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    return ContinuousFFTDecoder(config, codec=codec)


def test_joint_condition_fusion_is_wired_to_both_factorized_heads() -> None:
    model = _model(_codec(), condition_fusion="joint_mlp")
    decoder = model.factorized_decoder
    assert decoder is not None
    assert decoder.amplitude_net.joint_condition_mlp is not None
    assert decoder.phase_net.joint_condition_mlp is not None

    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(41))
    output = model(model.codec.encode(images), corrupt=False)
    assert torch.isfinite(output["loss"])


def test_grouped_factorized_decoder_jointly_trains_and_samples_four_coefficients() -> None:
    base = DiffusionDecoderConfig(
        target_dim=24,
        z_channels=16,
        width=16,
        depth=1,
        objective="flow",
        prediction_type="v_prediction",
        component_reduction="fixed_dim",
        diffusion_batch_mul=1,
        num_train_timesteps=16,
        num_inference_steps=2,
    )
    config = FactorizedPolarConfig(
        enabled=True,
        amplitude_transform="raw",
        amplitude_standardization="global",
        coordinate_mode="physical_standardized",
        phase_weighting="physical_energy",
        self_conjugate_sign="bernoulli",
    )
    decoder = FactorizedPolarDecoder(
        base,
        config,
        condition_width=16,
        coefficients_per_token=4,
    )
    raw = torch.randn(2, 3, 4, 6)
    active = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool
    )
    raw = raw * active[None, :, :, None]
    self_conjugate = torch.zeros(3, 4, dtype=torch.bool)
    self_conjugate[0, 0] = True
    raw[:, 0, 0, 3:] = 0.0
    output = decoder.compute_loss(
        raw_target=raw,
        z=torch.randn(2, 3, 16),
        slot_condition=torch.randn(2, 3, 16),
        amplitude_scale=torch.ones(3, 4, 3),
        is_self_conjugate=self_conjugate,
        active_coefficient_mask=active,
        coefficient_positions=torch.arange(12).reshape(3, 4),
        radius_bin=torch.arange(3),
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()

    sampled_amp, sampled_phase = decoder.sample_coordinates(
        z=torch.randn(2, 16),
        slot_condition=torch.randn(2, 16),
        generator=torch.Generator().manual_seed(19),
        steps=2,
        is_self_conjugate=self_conjugate[2].expand(2, -1),
        positions=torch.arange(8, 12).expand(2, -1),
        active_coefficient_mask=active[2].expand(2, -1),
    )
    assert sampled_amp.shape == (2, 4, 3)
    assert sampled_phase.shape == (2, 4, 3)
    assert sampled_amp[:, 2:].abs().max().item() == 0.0
    assert sampled_phase[:, 2:].abs().max().item() == 0.0


def test_radial_sector_model_roundtrip_causality_and_generation() -> None:
    codec = _physical_codec()
    config = ContinuousModelConfig(
        codec=codec.config,
        transformer=TransformerConfig(
            width=32,
            num_layers=1,
            num_heads=4,
            ff_mult=2,
            max_seq_len=codec.seq_len,
            qk_norm=True,
            attention_rope="frequency_2d",
        ),
        diffusion=DiffusionDecoderConfig(
            target_dim=24,
            z_channels=32,
            width=32,
            depth=1,
            objective="flow",
            prediction_type="v_prediction",
            component_reduction="fixed_dim",
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            amplitude_transform="raw",
            amplitude_standardization="global",
            coordinate_mode="physical_standardized",
            phase_weighting="physical_energy",
            self_conjugate_sign="bernoulli",
        ),
        polar_history=PolarHistoryConfig(
            enabled=True,
            mode="physical_standardized_log_amp_phase",
            fusion="replace",
            amplitude_transform="log_eps",
            log_epsilon=0.003,
        ),
        generation=GenerationConfig(
            num_inference_steps=2,
            grouping="radial_sector",
            group_size=4,
        ),
    )
    model = ContinuousFFTDecoder(config, codec=codec)
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(61))
    tokens = codec.encode(images)
    groups = model.pack_groups(tokens)
    torch.testing.assert_close(model.unpack_groups(groups), tokens)
    assert bool((model.group_radius[:, None] == codec.radius_bin[model.group_indices]).logical_or(~model.group_mask).all())

    output = model(tokens, corrupt=False)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()

    model.eval()
    with torch.no_grad():
        full, _ = model.forward_backbone(model.embed_groups(groups[:, :-1]))
        z, caches = model.init_cache(2, torch.device("cpu"), model.token_proj.weight.dtype)
        for group_position in range(model.sequence_length - 1):
            z, caches = model.forward_group_step(
                groups[:, group_position], group_position, caches
            )
        torch.testing.assert_close(z, full[:, -1], atol=2e-5, rtol=2e-5)

        generated = model.generate(
            batch_size=1,
            generator=torch.Generator().manual_seed(67),
            num_inference_steps=2,
            return_tokens=True,
        )
    assert generated["tokens"].shape == (1, codec.seq_len, 6)
    assert generated["groups"].shape[2:] == (4, 6)
    assert torch.isfinite(generated["images"]).all()


def test_fixed_dim_cartesian_metric_counts_each_active_coordinate_once() -> None:
    codec = _codec()
    model = _model(codec, component_reduction="fixed_dim")
    images = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(9))
    tokens = codec.encode(images)
    output = model(tokens, corrupt=False)
    positions = torch.arange(codec.seq_len)
    raw = model.factorized_cartesian_target(tokens, positions).reshape(-1, 6)
    mask = torch.ones_like(raw)
    self_conjugate = codec.is_self_conjugate[None].expand(images.shape[0], -1).reshape(-1)
    mask[self_conjugate, 3:] = 0.0
    expected_sum = (raw.square() * mask).sum(-1)
    torch.testing.assert_close(output["target_energy_sum_per_example"], expected_sum)
    torch.testing.assert_close(
        output["target_energy_per_example"], expected_sum / 6.0
    )
    assert bool((output["active_component_count_per_example"][self_conjugate] == 3).all())


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


def test_candidate_amplitude_transforms_roundtrip() -> None:
    raw = torch.tensor(
        [[[1.0, -0.5, 0.25, 0.5, 0.75, -0.125]]], dtype=torch.float32
    )
    scale = torch.tensor([[[2.0, 1.0, 0.5]]])
    for transform, parameter in (
        ("log1p", 1.0),
        ("inverse_softplus", 2.0),
        ("power", 2.0 / 3.0),
        ("raw", 1.0),
    ):
        coordinate, phase, _ = cartesian_to_polar_coordinates(
            raw,
            scale,
            0.003,
            amplitude_transform=transform,
            amplitude_transform_parameter=parameter,
        )
        reconstructed = polar_to_cartesian(
            coordinate.reshape(-1, 3),
            phase.reshape(-1, 3),
            scale.reshape(-1, 3),
            torch.tensor([False]),
            0.003,
            amplitude_transform=transform,
            amplitude_transform_parameter=parameter,
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
    torch.testing.assert_close(
        standardized.mean(), torch.tensor(0.0, dtype=torch.double), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        standardized.square().mean(),
        torch.tensor(1.0, dtype=torch.double),
        atol=1e-6,
        rtol=0,
    )
    expected_frequency_rms = standardized.square().mean(dim=0).sqrt()
    torch.testing.assert_close(
        payload["standardized_frequency_rms"].double(),
        expected_frequency_rms,
        atol=1e-6,
        rtol=0,
    )
    midpoint_snr = standardized.square().mean(dim=0) / payload[
        "standardized_frequency_rms"
    ].double().square()
    torch.testing.assert_close(
        midpoint_snr, torch.ones_like(midpoint_snr), atol=1e-6, rtol=0
    )


def test_frequency_rms_scales_the_amplitude_flow_source() -> None:
    class ZeroNet(torch.nn.Module):
        def forward(self, value, *args, **kwargs):
            return torch.zeros_like(value)

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
    source_rms = torch.tensor(
        [[0.25, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]]
    )
    decoder = FactorizedPolarDecoder(
        base,
        FactorizedPolarConfig(enabled=True, amplitude_source_scale="frequency_rms"),
        condition_width=16,
        amplitude_source_rms=source_rms,
    )
    decoder.amplitude_net = ZeroNet()
    seed = 181
    expected_generator = torch.Generator().manual_seed(seed)
    expected = torch.randn(2, 3, generator=expected_generator) * source_rms[[0, 2]]
    sampled, _ = decoder.sample_coordinates(
        z=torch.zeros(2, 16),
        slot_condition=torch.zeros(2, 16),
        generator=torch.Generator().manual_seed(seed),
        steps=1,
        positions=torch.tensor([0, 2]),
    )
    torch.testing.assert_close(sampled, expected)


def test_physical_standardized_amplitude_fit_uses_exact_model_coordinates() -> None:
    codec = _physical_codec()
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(103))
    payload = _fit_factorized_amplitude_stats(
        [images],
        codec,
        log_epsilon=0.003,
        scope="global",
        coordinate_mode="physical_standardized",
    )
    tokens = codec.encode(images).double()
    amplitude = torch.sqrt(tokens[..., :3].square() + tokens[..., 3:].square())
    coordinate = torch.log(amplitude + 0.003)
    standardized = (
        coordinate - payload["mean"].double()
    ) / payload["std"].double()
    assert payload["coordinate_mode"] == "physical_standardized"
    torch.testing.assert_close(
        standardized.mean(), torch.tensor(0.0, dtype=torch.double), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        standardized.square().mean(),
        torch.tensor(1.0, dtype=torch.double),
        atol=1e-6,
        rtol=0,
    )


def test_log1p_decoder_coordinate_can_differ_from_log_history() -> None:
    codec = _physical_codec()
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(107))
    common = dict(
        log_epsilon=0.003,
        scope="global",
        coordinate_mode="physical_standardized",
    )
    decoder_stats = _fit_factorized_amplitude_stats(
        [images],
        codec,
        amplitude_transform="log1p",
        amplitude_transform_parameter=1.0,
        **common,
    )
    history_stats = _fit_factorized_amplitude_stats(
        [images], codec, amplitude_transform="log_eps", **common
    )
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
            mode="physical_standardized_log_amp_phase",
            fusion="replace",
            amplitude_transform="log_eps",
            log_epsilon=0.003,
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            log_epsilon=0.003,
            amplitude_transform="log1p",
            amplitude_transform_parameter=1.0,
            amplitude_standardization="global",
            coordinate_mode="physical_standardized",
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    model = ContinuousFFTDecoder(
        config,
        codec=codec,
        factorized_amplitude_mean=decoder_stats["mean"],
        factorized_amplitude_std=decoder_stats["std"],
        history_amplitude_mean=history_stats["mean"],
        history_amplitude_std=history_stats["std"],
    )
    tokens = codec.encode(images[:2])
    positions = torch.arange(codec.seq_len - 1)
    history = model._polar_history_features(tokens[:, :-1], positions)
    expected_history, _, _ = cartesian_to_polar_coordinates(
        tokens[:, :-1],
        torch.ones(1, codec.seq_len - 1, 3),
        0.003,
        history_stats["mean"],
        history_stats["std"],
        "log_eps",
        1.0,
    )
    decoder_coordinate, _, _ = model.factorized_decoder.target_coordinates(
        tokens[:, :-1], torch.ones(1, codec.seq_len - 1, 3)
    )
    torch.testing.assert_close(history[..., 0::3], expected_history)
    assert not torch.allclose(history[..., 0::3], decoder_coordinate)
    output = model(tokens, corrupt=False)
    assert torch.isfinite(output["loss"])


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


def test_physical_polar_x0_with_discrete_sign_trains_and_generates() -> None:
    codec = _physical_codec()
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(107))
    stats = _fit_factorized_amplitude_stats(
        [images],
        codec,
        log_epsilon=0.003,
        scope="global",
        coordinate_mode="physical_standardized",
    )
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
            depth=2,
            objective="flow",
            prediction_type="v_prediction",
            snr_scale=1.0,
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        polar_history=PolarHistoryConfig(
            enabled=True,
            mode="physical_standardized_log_amp_phase",
            fusion="replace",
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=True,
            log_epsilon=0.003,
            amplitude_standardization="global",
            amplitude_loss_weight=0.1,
            phase_loss_weight=0.1,
            cartesian_loss_weight=1.0,
            coordinate_mode="physical_standardized",
            amplitude_prediction_type="x0",
            phase_weighting="physical_energy",
            self_conjugate_sign="bernoulli",
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    model = ContinuousFFTDecoder(
        config,
        codec=codec,
        factorized_amplitude_mean=stats["mean"],
        factorized_amplitude_std=stats["std"],
    )
    tokens = codec.encode(images[:2])
    positions = torch.arange(codec.seq_len - 1)
    history = model._polar_history_features(tokens[:, :-1], positions)
    coordinate, phase, _ = model.factorized_decoder.target_coordinates(
        tokens[:, :-1], torch.ones(1, codec.seq_len - 1, 3)
    )
    torch.testing.assert_close(history[..., 0::3], coordinate)
    torch.testing.assert_close(history[..., 1::3], torch.cos(phase))
    torch.testing.assert_close(history[..., 2::3], torch.sin(phase))

    output = model(tokens, corrupt=False)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.factorized_decoder.sign_net is not None
    assert model.factorized_decoder.sign_net.net[0].weight.grad is not None

    model.eval()
    sampled = model.generate(
        batch_size=2,
        generator=torch.Generator().manual_seed(109),
        num_inference_steps=2,
        return_tokens=True,
        max_tokens=4,
    )["tokens"]
    uncached = model.generate_uncached_prefix(
        batch_size=2,
        generator=torch.Generator().manual_seed(109),
        num_inference_steps=2,
        max_tokens=4,
    )
    torch.testing.assert_close(sampled[:, :4], uncached, atol=2e-5, rtol=2e-5)
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
        "factorized_decoder.amplitude_source_rms",
        "polar_history_amplitude_coordinate_mean",
        "polar_history_amplitude_coordinate_std",
        "group_indices",
        "group_mask",
        "group_radius",
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


def test_physical_polar_history_can_condition_cartesian_decoder() -> None:
    codec = _physical_codec()
    images = torch.rand(12, 3, 8, 8, generator=torch.Generator().manual_seed(151))
    stats = _fit_factorized_amplitude_stats(
        [images],
        codec,
        log_epsilon=0.003,
        scope="global",
        coordinate_mode="physical_standardized",
    )
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
            depth=2,
            objective="flow",
            prediction_type="v_prediction",
            diffusion_batch_mul=1,
            num_train_timesteps=16,
            num_inference_steps=2,
        ),
        polar_history=PolarHistoryConfig(
            enabled=True,
            mode="physical_standardized_log_amp_phase",
            fusion="replace",
        ),
        factorized_polar=FactorizedPolarConfig(
            enabled=False,
            log_epsilon=0.003,
            amplitude_standardization="global",
            coordinate_mode="physical_standardized",
        ),
        generation=GenerationConfig(num_inference_steps=2),
    )
    model = ContinuousFFTDecoder(
        config,
        codec=codec,
        factorized_amplitude_mean=stats["mean"],
        factorized_amplitude_std=stats["std"],
    )
    assert model.factorized_decoder is None
    assert model.diffusion is not None
    tokens = codec.encode(images[:2])
    positions = torch.arange(codec.seq_len - 1)
    history = model._polar_history_features(tokens[:, :-1], positions)
    coordinate, phase, _ = cartesian_to_polar_coordinates(
        tokens[:, :-1],
        torch.ones(1, codec.seq_len - 1, 3),
        0.003,
        stats["mean"],
        stats["std"],
    )
    torch.testing.assert_close(history[..., 0::3], coordinate)
    torch.testing.assert_close(history[..., 1::3], torch.cos(phase))
    torch.testing.assert_close(history[..., 2::3], torch.sin(phase))

    output = model(tokens, corrupt=False)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.token_proj.weight.grad is not None

    model.eval()
    with torch.no_grad():
        cached = model.generate(
            batch_size=1,
            generator=torch.Generator().manual_seed(157),
            num_inference_steps=2,
            return_tokens=True,
            max_tokens=4,
        )["tokens"][:, :4]
        uncached = model.generate_uncached_prefix(
            batch_size=1,
            generator=torch.Generator().manual_seed(157),
            num_inference_steps=2,
            max_tokens=4,
        )
    torch.testing.assert_close(cached, uncached, atol=2e-5, rtol=2e-5)


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
