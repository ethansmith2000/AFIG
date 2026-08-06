import torch

from frequency import FrequencyCodec, FrequencyCodecConfig
from train_joint_phase_oracle import (
    JointPhaseOracle,
    decode_with_phase,
    grouped_frequency_coordinates,
    pad_and_group,
    ungroup_and_trim,
)


def test_grouping_round_trip_with_padding() -> None:
    values = torch.randn(2, 10, 3)
    grouped = pad_and_group(values, group_size=4)
    assert grouped.shape == (2, 3, 12)
    restored = ungroup_and_trim(grouped, group_size=4, length=10, channels=3)
    torch.testing.assert_close(restored, values)


def test_joint_phase_oracle_loss_and_sampling() -> None:
    model = JointPhaseOracle(
        sequence_length=10,
        group_size=4,
        group_coordinates=torch.randn(3, 2),
        width=32,
        num_layers=2,
        num_heads=4,
        ff_mult=2,
        qk_norm=True,
        rope_base=10000.0,
        gradient_checkpointing=True,
    )
    phase = torch.randn(2, 10, 3)
    standardized_log_amplitude = torch.randn_like(phase)
    relative_amplitude = torch.rand_like(phase)
    physical_amplitude = torch.rand_like(phase)
    is_self_conjugate = torch.zeros(10, dtype=torch.bool)
    is_self_conjugate[[0, 7]] = True
    losses = model.loss(
        phase,
        standardized_log_amplitude,
        relative_amplitude,
        physical_amplitude,
        is_self_conjugate,
        phase_gate=0.1,
        cartesian_loss_weight=0.1,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.phase_projection.weight.grad is not None
    model.eval()
    sampled = model.sample(standardized_log_amplitude, steps=2)
    assert sampled.shape == phase.shape
    assert bool((sampled >= -torch.pi).all())
    assert bool((sampled < torch.pi).all())


def test_true_phase_and_amplitude_decode_exactly() -> None:
    codec = FrequencyCodec(
        FrequencyCodecConfig(
            normalization="global_ecs",
            coordinate_packing="isometric",
            value_transform="identity",
            ordering="square_spiral",
        )
    )
    images = torch.rand(2, 3, 32, 32)
    raw = codec.encode_raw(images)
    real, imag = raw[..., :3], raw[..., 3:]
    amplitude = torch.sqrt(real.square() + imag.square())
    phase = torch.atan2(imag, real)
    decoded = decode_with_phase(codec, amplitude, phase)
    torch.testing.assert_close(decoded, images, atol=2e-5, rtol=2e-5)
    coordinates = grouped_frequency_coordinates(codec, group_size=8)
    assert coordinates.shape == (65, 2)
