"""Tests for the matched 64-token Hartley autoregressive control."""

from __future__ import annotations

import torch

from control_pixel_diffusion import full_hartleyify, full_ihartleyify
from train_hartley_ar import HartleyTileAR, hartley_tile_order


def _model() -> HartleyTileAR:
    return HartleyTileAR(
        width=64,
        num_layers=2,
        num_heads=4,
        ff_mult=2,
        diff_width=64,
        diff_depth=1,
        inference_steps=2,
        gradient_checkpointing=False,
    )


def test_hartley_tile_order_and_transform_roundtrip() -> None:
    order = hartley_tile_order(8)
    assert torch.equal(order.sort().values, torch.arange(64))
    assert int(order[0]) == 0
    images = torch.randn(2, 3, 32, 32)
    tokens = full_hartleyify(images, 4)
    recovered = full_ihartleyify(tokens, 4, 32)
    assert torch.allclose(images, recovered, atol=5e-5, rtol=5e-5)


def test_hartley_ar_forward_grad_and_order_inverse() -> None:
    model = _model()
    raster = torch.randn(2, 64, 48)
    ordered = model.order_tokens(raster)
    assert torch.equal(model.restore_raster(ordered), raster)
    output = model(ordered)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.token_proj.weight.grad is not None
    assert model.diffusion.net.input_proj.weight.grad is not None
    assert model.slot_embed.weight.grad is not None


def test_hartley_ar_cache_parity_and_generation() -> None:
    model = _model().eval()
    prefix = torch.randn(1, 6, 48)
    with torch.no_grad():
        full, _ = model.forward_backbone(model.embed_history(prefix))
        hidden, caches = model.forward_backbone(
            model.embed_history(torch.empty(1, 0, 48)), use_cache=True
        )
        for position in range(prefix.shape[1]):
            projected = model.token_proj(prefix[:, position])[:, None]
            slot = torch.tensor([position + 1])
            projected = projected + model.slot_embed(slot)[None]
            hidden, caches = model.forward_backbone(
                projected, caches=caches, use_cache=True
            )
        assert torch.allclose(full[:, -1], hidden[:, -1], atol=1e-4, rtol=1e-4)
        samples = model.generate(
            1, 2, torch.Generator().manual_seed(31)
        )
    assert samples.shape == (1, 64, 48)
    assert bool(torch.isfinite(samples).all())
