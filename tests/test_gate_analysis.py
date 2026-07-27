"""Tests for matched-step architecture gate analysis."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")),
)

from analyze_architecture_gates import (  # noqa: E402
    add_control_deltas,
    aggregate_rows,
    control_arm,
    evidence_tier,
    nearest_row,
    parse_arm,
    resolve_attempts,
)


class TestGateAnalysis(unittest.TestCase):
    def test_arm_parsing_controls_and_evidence_tiers(self):
        parsed = parse_arm("arch-h-finalist1-s2-b128-n100000")
        self.assertEqual(parsed["arm"], "h-finalist1")
        self.assertEqual(parsed["seed"], 2)
        self.assertEqual(parsed["budget_steps"], 100000)
        self.assertEqual(control_arm("h-finalist1"), "h-anchor")
        self.assertEqual(control_arm("f-gain"), "f-alpha02")
        self.assertEqual(evidence_tier(5000), "exploratory")
        self.assertEqual(evidence_tier(30000), "medium")
        self.assertEqual(evidence_tier(100000), "confirmation")

    def test_nearest_row_uses_only_prior_steps_and_tolerance(self):
        history = [
            {"_step": 4000, "metric": 4.0},
            {"_step": 5000, "metric": 5.0},
            {"_step": 6000, "metric": 6.0},
        ]
        self.assertEqual(nearest_row(history, 5500, 500)["_step"], 5000)
        self.assertIsNone(nearest_row(history, 5500, 499))
        self.assertEqual(nearest_row(history, 6000, 0)["_step"], 6000)

    def test_newest_attempt_wins_duplicate_name(self):
        old = SimpleNamespace(name="same", created_at="2026-01-01")
        new = SimpleNamespace(name="same", created_at="2026-02-01")
        other = SimpleNamespace(name="other", created_at="2026-01-15")
        selected = resolve_attempts([old, other, new], "newest")
        self.assertEqual(selected, [new, other])
        self.assertEqual(resolve_attempts([old, new], "all"), [new, old])

    def test_paired_seed_deltas_and_direction(self):
        metrics = [
            "spectral/physical_complex_nrmse",
            "spectral/phase_coherence",
            "spectral/log_amplitude_bias",
            "grad_norm",
        ]
        rows = [
            {
                "run_id": "a",
                "arm": "p0",
                "seed": 0,
                "requested_step": 30000,
                metrics[0]: 0.30,
                metrics[1]: 0.60,
                metrics[2]: -0.20,
                metrics[3]: 1.0,
            },
            {
                "run_id": "b",
                "arm": "p1",
                "seed": 0,
                "requested_step": 30000,
                metrics[0]: 0.28,
                metrics[1]: 0.70,
                metrics[2]: -0.10,
                metrics[3]: 0.5,
            },
        ]
        add_control_deltas(rows, metrics)
        self.assertAlmostEqual(rows[1][f"delta/{metrics[0]}"], -0.02)
        self.assertAlmostEqual(rows[1][f"improvement/{metrics[0]}"], 0.02)
        self.assertAlmostEqual(rows[1][f"improvement/{metrics[1]}"], 0.10)
        self.assertAlmostEqual(rows[1][f"improvement/{metrics[2]}"], 0.10)
        self.assertNotIn(f"improvement/{metrics[3]}", rows[1])

        summary = aggregate_rows(rows, metrics)
        p1 = next(row for row in summary if row["arm"] == "p1")
        self.assertEqual(p1["num_seeds"], 1)
        self.assertAlmostEqual(
            p1[f"mean/improvement/{metrics[0]}"],
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
