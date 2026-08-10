"""Evaluate a saved direct continuous-FFT checkpoint with fixed/fresh grids."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch
import torchvision.utils as vutils

from frequency import FrequencyCodec
from live_evaluation import evaluate_live
from model_continuous import ContinuousFFTDecoder
from train_continuous import build_model_config, make_dataloader, parse_args as parse_train_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--reference_stats_path",
        default="continuous_runs/cifar10_inception_reference_radial.pt",
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reference_samples", type=int, default=50000)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--fixed_seed", type=int, default=12345)
    parser.add_argument("--fresh_seed", type=int, default=54321)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(path: str, device: torch.device) -> tuple[ContinuousFFTDecoder, Namespace, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Older checkpoints predate some current CLI fields.  Rehydrate them with
    # current defaults, then let every value actually saved by the run win.
    merged = vars(parse_train_args([]))
    merged.update(payload["args"])
    saved_args = Namespace(**merged)
    config = build_model_config(saved_args)
    codec = FrequencyCodec(config.codec)
    codec.load_exported(payload["codec"])
    model = ContinuousFFTDecoder(config, codec=codec)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, saved_args, int(payload["global_step"])


@torch.no_grad()
def save_grid(
    model: ContinuousFFTDecoder,
    path: Path,
    count: int,
    seed: int,
    steps: int,
) -> None:
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.generate(
            batch_size=count,
            generator=generator,
            num_inference_steps=steps,
            return_tokens=False,
            progress=False,
        )
    images = result["images"].float().clamp(0, 1).cpu()
    nrow = max(int(round(count**0.5)), 1)
    vutils.save_image(vutils.make_grid(images, nrow=nrow), path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This evaluator currently expects a CUDA device")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, saved_args, step = load_model(args.checkpoint, device)

    for label, seed in (("fixed", args.fixed_seed), ("fresh", args.fresh_seed)):
        save_grid(
            model,
            output / f"samples_{label}_{seed}.png",
            args.grid_size,
            seed,
            args.num_inference_steps,
        )

    saved_args.smoke = False
    saved_args.synthetic_data = False
    saved_args.train_batch_size = args.batch_size
    saved_args.dataloader_num_workers = min(int(saved_args.dataloader_num_workers), 4)
    dataset, _ = make_dataloader(saved_args)
    reference_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=saved_args.dataloader_num_workers,
        pin_memory=True,
    )
    torch.manual_seed(args.fresh_seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        metrics = evaluate_live(
            model,
            reference_loader=reference_loader,
            num_samples=args.samples,
            batch_size=args.batch_size,
            reference_cache_path=args.reference_stats_path,
            output_dir=str(output),
            num_inference_steps=args.num_inference_steps,
            reference_samples=args.reference_samples,
        )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": step,
        "prediction_type": saved_args.prediction_type,
        "normalization": saved_args.normalization,
        "num_inference_steps": args.num_inference_steps,
        "samples": args.samples,
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
