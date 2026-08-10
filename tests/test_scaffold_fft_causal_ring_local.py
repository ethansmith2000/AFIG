"""Tests for causal FFT-ring scheduling with aligned local computation."""

from types import SimpleNamespace

import torch

from control_pixel_diffusion import patchify
from scaffold_fft_causal_ring_local import (
    CausalRingLocalDenoiser,
    assemble_causal_ring_state,
    causal_ring_flow_loss,
    load_joint_denoiser_weights,
    masked_ring_heun_step,
    ring_masks,
    sample_causal_ring_fft,
    sample_target_rings,
    validate_scalar_rings,
)
from train_scaffold_fft_residual import (
    ScaffoldResidualDenoiser,
    images_to_fft_state,
    make_compact_layout,
)


def tiny_args() -> SimpleNamespace:
    return SimpleNamespace(width=32, num_layers=1, num_heads=4, ff_mult=2)


def tiny_setup():
    codec, layout_orbit, layout_component = make_compact_layout(
        8, torch.device("cpu")
    )
    scalar_ring = codec.radius_bin[layout_orbit]
    model = CausalRingLocalDenoiser(
        tokens=4,
        patch_dim=48,
        ring_count=codec.num_bins,
        args=tiny_args(),
    )
    return model, codec, layout_orbit, layout_component, scalar_ring


def test_ring_layout_and_proportional_sampling() -> None:
    _, codec, layout_orbit, _, scalar_ring = tiny_setup()
    counts = validate_scalar_rings(scalar_ring, expected_values=3 * 8 * 8)
    assert counts.sum() == 3 * 8 * 8
    assert counts.numel() == codec.num_bins

    # Every active component of one Hermitian orbit belongs to one radius ring.
    for orbit in layout_orbit.unique():
        assert scalar_ring[layout_orbit == orbit].unique().numel() == 1

    draws = sample_target_rings(
        100_000,
        counts,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(12),
    )
    empirical = torch.bincount(draws, minlength=counts.numel()).float()
    empirical /= empirical.sum()
    expected = counts.float() / counts.sum()
    torch.testing.assert_close(empirical, expected, atol=0.004, rtol=0.04)


def test_hybrid_state_has_clean_past_noisy_future_and_flowing_current() -> None:
    _, _, _, _, scalar_ring = tiny_setup()
    batch = 3
    target = torch.arange(batch * scalar_ring.numel(), dtype=torch.float32).reshape(
        batch, 4, 48
    )
    noise = -torch.ones_like(target)
    target_ring = torch.tensor([0, 1, int(scalar_ring.max())])
    time = torch.tensor([0.25, 0.5, 0.75])
    state, velocity, current = assemble_causal_ring_state(
        target, noise, time, target_ring, scalar_ring
    )
    earlier, expected_current, later = ring_masks(scalar_ring, target_ring)
    flat_state = state.flatten(1)
    flat_target = target.flatten(1)
    flat_noise = noise.flatten(1)
    expected_mixture = time[:, None] * flat_target + (1.0 - time[:, None]) * flat_noise
    torch.testing.assert_close(flat_state[earlier], flat_target[earlier])
    torch.testing.assert_close(flat_state[later], flat_noise[later])
    torch.testing.assert_close(flat_state[expected_current], expected_mixture[expected_current])
    torch.testing.assert_close(velocity, target - noise)
    assert torch.equal(current.flatten(1), expected_current)

    # Future data is not observable in the constructed state or target-ring loss.
    changed_target = target.clone().flatten(1)
    changed_target[later] += 10_000.0
    changed_target = changed_target.reshape_as(target)
    changed_state, changed_velocity, _ = assemble_causal_ring_state(
        changed_target, noise, time, target_ring, scalar_ring
    )
    torch.testing.assert_close(changed_state, state)
    current_3d = current
    torch.testing.assert_close(
        changed_velocity[current_3d], velocity[current_3d]
    )


def test_joint_checkpoint_initialization_preserves_exact_function() -> None:
    model, _, _, _, _ = tiny_setup()
    joint = ScaffoldResidualDenoiser(tokens=4, patch_dim=48, args=tiny_args())
    load_joint_denoiser_weights(model, joint.state_dict())
    noisy = torch.randn(3, 4, 48)
    scaffold = torch.randn_like(noisy)
    time = torch.rand(3)
    with torch.no_grad():
        expected = joint.velocity_local(noisy, scaffold, time)
        for ring in range(model.ring_count):
            actual = model.velocity_local(
                noisy,
                scaffold,
                time,
                torch.full((3,), ring, dtype=torch.long),
            )
            torch.testing.assert_close(actual, expected)


def test_causal_ring_loss_backward_and_only_current_solver_update() -> None:
    model, codec, layout_orbit, layout_component, scalar_ring = tiny_setup()
    images = torch.randn(2, 3, 8, 8)
    scaffold = patchify(torch.randn_like(images), 4)
    target = images_to_fft_state(
        codec, images, layout_orbit, layout_component, token_dim=48
    )
    chosen_ring = torch.tensor([0, min(2, int(scalar_ring.max()))])
    output = causal_ring_flow_loss(
        model,
        codec,
        target,
        scaffold,
        scalar_ring,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
        target_ring=chosen_ring,
        flow_time=torch.tensor([0.3, 0.7]),
        noise_fft_state=torch.randn_like(target),
    )
    assert output["loss"].ndim == 0 and torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.final_layer.linear.weight.grad is not None
    assert torch.isfinite(model.final_layer.linear.weight.grad).all()

    model.eval()
    state = torch.randn_like(target)
    ring = min(1, int(scalar_ring.max()))
    stepped = masked_ring_heun_step(
        model,
        codec,
        state,
        scaffold,
        scalar_ring,
        target_ring_index=ring,
        time=0.2,
        dt=0.1,
        use_heun=True,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
    )
    inactive = (scalar_ring != ring).reshape(1, 4, 48).expand_as(state)
    torch.testing.assert_close(stepped[inactive], state[inactive])


def test_sequential_sampler_shape_and_finiteness() -> None:
    model, codec, layout_orbit, layout_component, scalar_ring = tiny_setup()
    scaffold = patchify(torch.randn(2, 3, 8, 8), 4)
    sampled = sample_causal_ring_fft(
        model,
        codec,
        scaffold,
        scalar_ring,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
        steps=2,
        generator=torch.Generator().manual_seed(123),
    )
    assert sampled.shape == (2, 4, 48)
    assert torch.isfinite(sampled).all()
