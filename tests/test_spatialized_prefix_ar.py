import torch

from spatialized_prefix_ar import SpatializedPrefixHartleyAR


def make_model() -> SpatializedPrefixHartleyAR:
    return SpatializedPrefixHartleyAR(
        width=32,
        num_layers=1,
        num_heads=4,
        ff_mult=2,
        diff_width=32,
        diff_depth=1,
        inference_steps=2,
        latent_size=4,
        patch=2,
        channels=2,
        gradient_checkpointing=False,
    )


def test_order_roundtrip_and_partial_prefix_causality() -> None:
    model = make_model()
    raster = torch.randn(2, model.seq_len, model.token_dim)
    ordered = model.order_tokens(raster)
    torch.testing.assert_close(model.restore_raster(ordered), raster)

    position = torch.tensor([2])
    reference = model.partial_spatial_tokens(ordered, position)
    changed_future = ordered.clone()
    changed_future[:, 2:] += 100.0
    torch.testing.assert_close(
        model.partial_spatial_tokens(changed_future, position), reference
    )

    changed_past = ordered.clone()
    changed_past[:, 0] += 1.0
    assert not torch.allclose(
        model.partial_spatial_tokens(changed_past, position), reference
    )


def test_position_zero_has_empty_spatial_state() -> None:
    model = make_model()
    ordered = torch.randn(2, model.seq_len, model.token_dim)
    state = model.partial_spatial_tokens(ordered, torch.tensor([0]))
    torch.testing.assert_close(state, torch.zeros_like(state))


def test_forward_and_generate_shapes() -> None:
    model = make_model()
    ordered = torch.randn(2, model.seq_len, model.token_dim)
    output = model(ordered)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    model.eval()
    generator = torch.Generator().manual_seed(123)
    generated = model.generate(2, 2, generator)
    assert generated.shape == ordered.shape
    assert torch.isfinite(generated).all()
