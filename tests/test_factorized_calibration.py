import torch

from diagnose_factorized_calibration import calibration_summary


def test_calibration_summary_exposes_conditional_collapse() -> None:
    samples = torch.zeros(3, 2, 1, 6)
    target = torch.zeros(2, 1, 6)
    target[..., 0] = 1.0
    component_mask = torch.ones(1, 6)
    selection = torch.ones(1, dtype=torch.bool)

    summary = calibration_summary(samples, target, component_mask, selection)

    assert summary["sample_to_target_nrmse"] == 1.0
    assert summary["ensemble_mean_to_target_nrmse"] == 1.0
    assert summary["pairwise_spread_nrmse"] == 0.0
    assert summary["spread_to_error_sq_ratio"] == 0.0
    assert summary["normalized_energy_score"] == 1.0
    assert summary["sample_power_to_target_ratio"] == 0.0


def test_energy_score_rewards_spread_that_covers_target() -> None:
    collapsed = torch.zeros(2, 1, 1, 6)
    dispersed = collapsed.clone()
    dispersed[1, ..., 0] = 2.0
    target = torch.zeros(1, 1, 6)
    target[..., 0] = 1.0
    component_mask = torch.ones(1, 6)
    selection = torch.ones(1, dtype=torch.bool)

    collapsed_summary = calibration_summary(
        collapsed, target, component_mask, selection
    )
    dispersed_summary = calibration_summary(
        dispersed, target, component_mask, selection
    )

    assert dispersed_summary["normalized_energy_score"] < collapsed_summary[
        "normalized_energy_score"
    ]
