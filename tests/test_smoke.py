"""CPU smoke: one training step + short cached generation with tiny preset."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import math

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSmoke(unittest.TestCase):
    def test_raw_weights_and_final_eval_defaults(self):
        from train_continuous import parse_args

        args = parse_args([])
        self.assertFalse(args.use_ema)
        self.assertTrue(args.final_eval)
        smoke = parse_args(["--smoke"])
        from train_continuous import apply_preset

        self.assertFalse(apply_preset(smoke).final_eval)

    def test_live_metric_math_is_finite(self):
        from live_evaluation import StreamingMoments, _fid, _kid

        real = torch.randn(32, 8)
        generated = torch.randn(32, 8) + 0.1
        real_moments = StreamingMoments(8)
        generated_moments = StreamingMoments(8)
        real_moments.update(real)
        generated_moments.update(generated)
        real_mean, real_covariance = real_moments.compute()
        generated_mean, generated_covariance = generated_moments.compute()
        fid = _fid(real_mean, real_covariance, generated_mean, generated_covariance)
        kid = _kid(real, generated, subsets=2, subset_size=16)
        self.assertTrue(math.isfinite(fid))
        self.assertTrue(math.isfinite(kid))

    def test_train_and_generate_smoke(self):
        from train_continuous import main, parse_args

        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--smoke",
                "--output_dir",
                td,
                "--data_root",
                os.path.join(td, "data"),
                "--report_to",
                "none",
                "--dataset",
                "synthetic",
                "--save_final_checkpoint",
            ]
            args = parse_args(argv)
            main(args)

            ckpts = [p for p in os.listdir(td) if p.startswith("checkpoint_")]
            self.assertTrue(len(ckpts) >= 1)
            stats = os.path.join(td, "codec_stats.pt")
            self.assertTrue(os.path.isfile(stats))
            # Short generation smoke via loaded codec + tiny model weights from ckpt.
            from frequency import FrequencyCodec, FrequencyCodecConfig
            from model_continuous import ContinuousFFTDecoder, ContinuousModelConfig, TransformerConfig
            from diffusion_decoder import DiffusionDecoderConfig

            payload = torch.load(os.path.join(td, ckpts[-1]), map_location="cpu", weights_only=False)
            codec = FrequencyCodec(FrequencyCodecConfig(**payload["codec"]["config"]))
            codec.load_exported(payload["codec"])
            cfg = ContinuousModelConfig(
                codec=FrequencyCodecConfig(**payload["codec"]["config"]),
                transformer=TransformerConfig(width=64, num_layers=2, num_heads=4, ff_mult=2),
                diffusion=DiffusionDecoderConfig(
                    z_channels=64, width=64, depth=2, num_inference_steps=2, diffusion_batch_mul=1
                ),
            )
            model = ContinuousFFTDecoder(cfg, codec=codec)
            model.load_state_dict(payload["model"])
            model.eval()
            out = model.generate(batch_size=2, num_inference_steps=2, max_tokens=4)
            self.assertEqual(tuple(out["images"].shape), (2, 3, 32, 32))
            self.assertTrue(torch.isfinite(out["images"]).all())

    def test_hf_cifar_loader_available(self):
        """Use local HuggingFace CIFAR arrow cache when torchvision tarball is missing."""
        from types import SimpleNamespace
        from train_continuous import make_dataloader, _hf_cifar_paths

        paths = _hf_cifar_paths()
        if not paths:
            self.skipTest("No local HF CIFAR arrow cache on this machine")
        args = SimpleNamespace(
            smoke=False,
            synthetic_data=False,
            dataset="huggingface_cifar",
            data_root="/workspace/AFIG/data",
            train_batch_size=4,
            dataloader_num_workers=0,
        )
        dataset, loader = make_dataloader(args)
        self.assertGreaterEqual(len(dataset), 1000)
        batch = next(iter(loader))
        images, labels = batch
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertTrue(torch.isfinite(images).all())


if __name__ == "__main__":
    unittest.main()
