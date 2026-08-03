"""Precision regressions for rotary position tables."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_transformer import (  # noqa: E402
    CausalTransformerConfig,
    apply_rope,
    build_rope_tables,
)
from model_joint_latent_diffusion import (  # noqa: E402
    JointLatentDiffusionConfig,
    JointLatentDiffusionModel,
)


class TestRoPEPrecision(unittest.TestCase):
    def test_long_tables_stay_fp32_under_bf16_autocast(self):
        coordinates = torch.arange(515, dtype=torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cos, sin = build_rope_tables(coordinates, head_dim=64)

        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        self.assertFalse(torch.equal(cos[511], cos[512]))
        self.assertFalse(torch.equal(sin[511], sin[512]))

    def test_tables_cast_only_when_applied_to_qk(self):
        cos, sin = build_rope_tables(torch.arange(8), head_dim=16)
        q = torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)
        rotated = apply_rope(q, cos, sin)

        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        self.assertEqual(rotated.dtype, torch.bfloat16)

    def test_radius_angle_uses_fp32_metadata_before_activation_cast(self):
        config = JointLatentDiffusionConfig(
            metadata_dim=11,
            transformer=CausalTransformerConfig(
                width=32,
                num_layers=1,
                num_heads=4,
                ff_mult=2,
                max_seq_len=53,
            ),
            num_train_timesteps=20,
            num_inference_steps=2,
            rope="radius_angle",
        )
        model = JointLatentDiffusionModel(config).eval()
        metadata = torch.randn(53, 11, dtype=torch.float32)
        noisy = torch.randn(1, 53, 64, dtype=torch.bfloat16)
        flow_time = torch.tensor([0.5])
        observed = {}
        original = model._rope_tables

        def capture(value):
            observed["input_dtype"] = value.dtype
            tables = original(value)
            observed["table_dtypes"] = tuple(table.dtype for table in tables)
            return tables

        model._rope_tables = capture
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model.predict_velocity(noisy, flow_time, metadata)

        self.assertEqual(observed["input_dtype"], torch.float32)
        self.assertEqual(
            observed["table_dtypes"], (torch.float32, torch.float32)
        )
        self.assertEqual(prediction.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
