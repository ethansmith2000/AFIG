from __future__ import annotations

import torch

from progressive_tokenizer import (
    AutoregressiveFlowConfig,
    AutoregressiveRectifiedFlow,
)
from progressive_tokenizer.training import optimizer_parameter_groups
from train_progressive_ar_flow import block_latents, unblock_latents


def tiny_model() -> AutoregressiveRectifiedFlow:
    return AutoregressiveRectifiedFlow(
        AutoregressiveFlowConfig(
            sequence_length=4,
            token_dim=8,
            width=32,
            trunk_depth=2,
            head_depth=2,
            num_heads=4,
            mlp_ratio=2.0,
        )
    )


def test_teacher_forced_shapes_and_gradients() -> None:
    model = tiny_model()
    tokens = torch.randn(3, 4, 8)
    output = model(tokens)
    assert output["loss"].shape == ()
    assert output["per_token_mse"].shape == (4,)
    output["loss"].backward()
    assert model.head.output.weight.grad is not None
    assert float(model.head.output.weight.grad.abs().sum()) > 0


def test_bos_alignment_and_causality() -> None:
    model = tiny_model().eval()
    original = torch.randn(2, 4, 8)
    changed = original.clone()
    changed[:, 1] += 100.0
    first = model.trunk(original)
    second = model.trunk(changed)
    # z_1 enters the shifted stream only at the condition used to predict z_2.
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2], second[:, 2])


def test_bos_condition_is_image_independent() -> None:
    model = tiny_model().eval()
    first = model.trunk(torch.randn(2, 4, 8))[:, 0]
    second = model.trunk(torch.randn(2, 4, 8))[:, 0]
    torch.testing.assert_close(first, second)


def test_zero_initialized_head_starts_at_zero_velocity() -> None:
    model = tiny_model()
    noisy = torch.randn(2, 4, 8)
    time = torch.rand(2, 4)
    condition = model.trunk(torch.randn_like(noisy))
    prediction = model.head.predict_velocity(noisy, time, condition)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


def test_generation_shape_and_seed_determinism() -> None:
    model = tiny_model().eval()
    first = model.generate(
        2, steps=2, generator=torch.Generator().manual_seed(7)
    )
    second = model.generate(
        2, steps=2, generator=torch.Generator().manual_seed(7)
    )
    assert first.shape == (2, 4, 8)
    torch.testing.assert_close(first, second)


def test_consecutive_block_layout_round_trip() -> None:
    physical = torch.randn(3, 64, 16)
    blocked = block_latents(physical, 4)
    assert blocked.shape == (3, 16, 64)
    restored = unblock_latents(blocked, sequence_length=64, token_dim=16)
    torch.testing.assert_close(restored, physical)


def test_block_layout_rejects_non_divisible_sequence() -> None:
    try:
        block_latents(torch.randn(2, 7, 4), 2)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected non-divisible block layout to fail")


def test_weight_decay_excludes_bos_and_includes_projection_weights() -> None:
    model = tiny_model()
    groups = optimizer_parameter_groups(model, weight_decay=0.05)
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    protected = {id(parameter) for parameter in groups[1]["params"]}
    assert id(model.trunk.bos) in protected
    assert id(model.trunk.target_position) in protected
    assert id(model.trunk.input.weight) in decayed
    assert id(model.head.condition_fusion[0].weight) in decayed
    assert decayed.isdisjoint(protected)
    assert len(decayed) + len(protected) == sum(
        parameter.requires_grad for parameter in model.parameters()
    )
