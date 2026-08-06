"""Tests for the Stage-C oracle-scaffold compact-FFT residual gate."""

from types import SimpleNamespace

import torch

from control_pixel_diffusion import patchify
from train_scaffold_fft_residual import (
    ScaffoldResidualDenoiser,
    dual_domain_flow_loss,
    dual_domain_velocity,
    fft_state_to_images,
    images_to_fft_state,
    make_compact_layout,
    sample_residual_fft,
)


def tiny_args() -> SimpleNamespace:
    return SimpleNamespace(
        width=32,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
    )


def test_dual_domain_state_roundtrip_and_local_identity_velocity() -> None:
    codec, layout_orbit, layout_component = make_compact_layout(
        8, torch.device("cpu")
    )
    images = torch.randn(3, 3, 8, 8)
    state = images_to_fft_state(
        codec, images, layout_orbit, layout_component, token_dim=48
    )
    restored = fft_state_to_images(
        codec, state, layout_orbit, layout_component
    )
    torch.testing.assert_close(restored, images, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        state.square().sum(dim=(1, 2)),
        images.square().sum(dim=(1, 2, 3)),
        atol=3e-4,
        rtol=2e-6,
    )

    class IdentityLocal(ScaffoldResidualDenoiser):
        def velocity_local(self, noisy, scaffold, flow_time):
            return noisy

    identity = IdentityLocal(tokens=4, patch_dim=48, args=tiny_args())
    scaffold = torch.zeros_like(patchify(images, 4))
    velocity = dual_domain_velocity(
        identity,
        codec,
        state,
        scaffold,
        torch.rand(images.shape[0]),
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
    )
    torch.testing.assert_close(velocity, state, atol=2e-5, rtol=2e-5)


def test_scaffold_denoiser_loss_backward_and_heun_shape() -> None:
    codec, layout_orbit, layout_component = make_compact_layout(
        8, torch.device("cpu")
    )
    model = ScaffoldResidualDenoiser(tokens=4, patch_dim=48, args=tiny_args())
    residual = torch.randn(2, 3, 8, 8)
    scaffold_patches = patchify(torch.randn_like(residual), 4)
    state = images_to_fft_state(
        codec, residual, layout_orbit, layout_component, token_dim=48
    )
    loss = dual_domain_flow_loss(
        model,
        codec,
        state,
        scaffold_patches,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.final_layer.linear.weight.grad is not None
    assert torch.isfinite(model.final_layer.linear.weight.grad).all()

    generator = torch.Generator().manual_seed(123)
    sampled = sample_residual_fft(
        model,
        codec,
        scaffold_patches,
        layout_orbit=layout_orbit,
        layout_component=layout_component,
        patch=4,
        image_size=8,
        token_dim=48,
        steps=2,
        generator=generator,
    )
    assert sampled.shape == (2, 4, 48)
    assert torch.isfinite(sampled).all()
