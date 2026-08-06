"""Tests for scaffold-conditioned causal compact-FFT ring generation."""

import torch

from control_pixel_diffusion import patchify
from scaffold_fft_ring import ScaffoldFFTRingConfig, ScaffoldFFTRingModel
from train_scaffold_fft_residual import images_to_fft_state, make_compact_layout


def tiny_model():
    codec, layout_orbit, layout_component = make_compact_layout(
        8, torch.device("cpu")
    )
    scalar_ring = codec.radius_bin[layout_orbit]
    counts = torch.bincount(scalar_ring, minlength=codec.num_bins)
    config = ScaffoldFFTRingConfig(
        local_tokens=4,
        patch_dim=48,
        ring_count=codec.num_bins,
        max_ring_dim=int(counts.max()),
        width=32,
        scaffold_layers=1,
        ring_layers=1,
        num_heads=4,
        ff_mult=2,
        diffusion_width=32,
        diffusion_depth=2,
        diffusion_batch_mul=1,
        num_inference_steps=2,
    )
    return (
        ScaffoldFFTRingModel(scalar_ring, config),
        codec,
        layout_orbit,
        layout_component,
    )


def test_exact_ring_partition_and_roundtrip() -> None:
    model, codec, layout_orbit, layout_component = tiny_model()
    images = torch.randn(3, 3, 8, 8)
    state = images_to_fft_state(
        codec, images, layout_orbit, layout_component, token_dim=48
    )
    rings = model.pack_rings(state)
    restored = model.unpack_rings(rings).reshape_as(state)
    torch.testing.assert_close(restored, state)
    assert int(model.ring_component_mask.sum()) == state[0].numel()
    assert model.ring_rope_cos.dtype == torch.float32
    assert model.scaffold_rope_cos.dtype == torch.float32

    # Each orbit's active Cartesian/RGB components remain in one atomic ring.
    for orbit in layout_orbit.unique():
        rings_for_orbit = model.scalar_ring[layout_orbit == orbit].unique()
        assert rings_for_orbit.numel() == 1


def test_shifted_bos_and_future_ring_causality() -> None:
    model, _, _, _ = tiny_model()
    model.eval()
    scaffold = patchify(torch.randn(2, 3, 8, 8), 4)
    rings = torch.randn(2, model.config.ring_count, model.config.max_ring_dim)
    rings = rings * model.ring_component_mask[None]

    shifted = model.shifted_ring_inputs(rings)
    changed_first = rings.clone()
    changed_first[:, 0] += model.ring_component_mask[0]
    shifted_changed = model.shifted_ring_inputs(changed_first)
    # Ring zero is conditioned on BOS, not on its own teacher-forced target.
    torch.testing.assert_close(shifted[:, 0], shifted_changed[:, 0])
    assert not torch.allclose(shifted[:, 1], shifted_changed[:, 1])

    baseline = model.ring_conditions(scaffold, rings)
    changed_future = rings.clone()
    changed_future[:, 3] += 7.0 * model.ring_component_mask[3]
    perturbed = model.ring_conditions(scaffold, changed_future)
    # A target ring first enters history while predicting the following ring.
    torch.testing.assert_close(baseline[:, :4], perturbed[:, :4])
    assert not torch.allclose(baseline[:, 4], perturbed[:, 4])

    # Cached inference follows the same BOS/shift convention as teacher forcing.
    memory = model.encode_scaffold(scaffold)
    caches = model.init_ring_cache(memory)
    cached = []
    previous = rings[:, 0]
    for ring in range(model.config.ring_count):
        condition, caches = model.ring_condition_step(previous, ring, caches)
        cached.append(condition)
        previous = rings[:, ring]
    torch.testing.assert_close(torch.stack(cached, dim=1), baseline, atol=1e-5, rtol=1e-5)


def test_loss_backward_and_sequential_generation_shape() -> None:
    model, codec, layout_orbit, layout_component = tiny_model()
    images = torch.randn(2, 3, 8, 8)
    scaffold = patchify(torch.randn_like(images), 4)
    state = images_to_fft_state(
        codec, images, layout_orbit, layout_component, token_dim=48
    )
    output = model(state, scaffold)
    assert output["loss"].ndim == 0 and torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.diffusion.net.final_layer.linear.weight.grad is not None
    assert torch.isfinite(model.diffusion.net.final_layer.linear.weight.grad).all()

    generator = torch.Generator().manual_seed(123)
    sampled = model.generate_fft(
        scaffold,
        num_inference_steps=2,
        generator=generator,
    )
    assert sampled.shape == (2, state[0].numel())
    assert torch.isfinite(sampled).all()

    teacher_sampled = model.generate_fft(
        scaffold,
        teacher_history_fft=state,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(123),
    )
    assert teacher_sampled.shape == sampled.shape
    assert torch.isfinite(teacher_sampled).all()
