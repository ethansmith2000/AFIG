from __future__ import annotations

import torch

from progressive_tokenizer import JointFlowConfig, JointRectifiedFlow


def tiny_model(*, checkpointing: bool = False) -> JointRectifiedFlow:
    return JointRectifiedFlow(
        JointFlowConfig(
            sequence_length=4,
            token_dim=8,
            width=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            gradient_checkpointing=checkpointing,
        )
    )


def tiny_warped_model() -> JointRectifiedFlow:
    return JointRectifiedFlow(
        JointFlowConfig(
            sequence_length=4,
            token_dim=8,
            width=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2.0,
            time_parameterization="rational_per_token",
            token_time_scales=(4.0, 2.0, 0.5, 0.25),
        )
    )


def test_flow_shape_and_finite_loss() -> None:
    model = tiny_model()
    clean = torch.randn(3, 4, 8)
    output = model(clean)
    assert output["loss"].shape == ()
    assert output["per_token_mse"].shape == (4,)
    assert torch.isfinite(output["loss"])


def test_linear_bridge_endpoints_and_target() -> None:
    model = tiny_model()
    clean = torch.randn(2, 4, 8)
    noise = torch.randn_like(clean)
    at_noise = torch.zeros(2)
    at_data = torch.ones(2)
    captured = []

    def fake_predict(values, time):
        captured.append(values.detach().clone())
        return torch.zeros_like(values)

    model.predict_velocity = fake_predict  # type: ignore[method-assign]
    model(clean, time=at_noise, noise=noise)
    model(clean, time=at_data, noise=noise)
    torch.testing.assert_close(captured[0], noise)
    torch.testing.assert_close(captured[1], clean)


def test_adaln_zero_starts_with_zero_velocity_and_trains() -> None:
    model = tiny_model()
    clean = torch.randn(2, 4, 8)
    prediction = model.predict_velocity(clean, torch.rand(2))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))
    loss = model(clean)["loss"]
    loss.backward()
    assert model.final.output.weight.grad is not None
    assert float(model.final.output.weight.grad.abs().sum()) > 0


def test_sampling_is_seed_deterministic() -> None:
    model = tiny_model().eval()
    first = model.sample(
        2, steps=3, generator=torch.Generator().manual_seed(123)
    )
    second = model.sample(
        2, steps=3, generator=torch.Generator().manual_seed(123)
    )
    torch.testing.assert_close(first, second)


def test_gradient_checkpointing_path() -> None:
    model = tiny_model(checkpointing=True).train()
    loss = model(torch.randn(2, 4, 8))["loss"]
    loss.backward()
    assert model.final.output.weight.grad is not None


def test_per_token_loss_weighting_preserves_unweighted_diagnostic() -> None:
    model = tiny_model()
    clean = torch.zeros(2, 4, 8)
    noise = torch.zeros_like(clean)

    def fixed_prediction(values, time):
        prediction = torch.zeros_like(values)
        prediction[:, 0] = 1.0
        prediction[:, 1] = 2.0
        prediction[:, 2] = 3.0
        prediction[:, 3] = 4.0
        return prediction

    model.predict_velocity = fixed_prediction  # type: ignore[method-assign]
    weights = torch.tensor([0.25, 0.5, 1.0, 2.25])
    output = model(
        clean,
        time=torch.zeros(2),
        noise=noise,
        token_loss_weights=weights,
    )
    expected_per_token = torch.tensor([1.0, 4.0, 9.0, 16.0])
    torch.testing.assert_close(output["per_token_mse"], expected_per_token)
    torch.testing.assert_close(output["unweighted_loss"], expected_per_token.mean())
    torch.testing.assert_close(output["loss"], (expected_per_token * weights).mean())


def test_rational_time_warp_has_common_endpoints_and_declared_crossings() -> None:
    model = tiny_warped_model()
    endpoints = model.path_time(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(endpoints[0], torch.zeros(4))
    torch.testing.assert_close(endpoints[1], torch.ones(4))
    scales = torch.tensor(model.config.token_time_scales)
    crossing_time = 1.0 / (1.0 + scales)
    crossing = model.path_time(crossing_time)
    torch.testing.assert_close(crossing.diagonal(), torch.full((4,), 0.5))


def test_rational_time_warp_conditions_each_token_and_preserves_base_target() -> None:
    model = tiny_warped_model()
    clean = torch.randn(2, 4, 8)
    noise = torch.randn_like(clean)
    captured = {}

    def fake_predict(values, time):
        captured["values"] = values.detach().clone()
        captured["time"] = time.detach().clone()
        return torch.zeros_like(values)

    model.predict_velocity = fake_predict  # type: ignore[method-assign]
    base_time = torch.tensor([0.25, 0.75])
    output = model(clean, time=base_time, noise=noise)
    path_time = model.path_time(base_time)
    expected_noisy = (
        (1.0 - path_time[:, :, None]) * noise
        + path_time[:, :, None] * clean
    )
    torch.testing.assert_close(captured["values"], expected_noisy)
    torch.testing.assert_close(captured["time"], path_time)
    expected_target_mse = (clean - noise).square().mean()
    torch.testing.assert_close(output["unweighted_loss"], expected_target_mse)


def test_rational_sampling_integrates_each_phi_increment() -> None:
    model = tiny_warped_model().eval()

    def unit_velocity(values, time):
        return torch.ones_like(values)

    model.predict_velocity = unit_velocity  # type: ignore[method-assign]
    generator = torch.Generator().manual_seed(123)
    initial = torch.randn(2, 4, 8, generator=generator)
    generator = torch.Generator().manual_seed(123)
    sampled = model.sample(2, steps=4, solver="euler", generator=generator)
    torch.testing.assert_close(sampled, initial + 1.0)
