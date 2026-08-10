from __future__ import annotations

import torch

from progressive_tokenizer import (
    AutoregressiveFlowConfig,
    AutoregressiveRectifiedFlow,
)


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
