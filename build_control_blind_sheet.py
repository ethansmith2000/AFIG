#!/usr/bin/env python3
"""Build a frozen-key, multi-seed blind sheet for the direct representation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import make_grid, save_image

from control_pixel_diffusion import (
    PatchDiffusion,
    build_compact_isometric_codec,
    orbit_order_permutation,
)
from evaluate_control_diffusion import _decode, _layout


ARMS = (
    ("pixel_control", "pixels"),
    ("patch_dct_control", "patch_dct"),
    ("full_dct_control", "full_dct"),
    ("full_hartley_control", "full_hartley"),
    ("fft_compact_isometric_spiral_control", "fft_compact_isometric_spiral"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", default="latent_continuous_runs")
    parser.add_argument("--output_dir", default="diagnostics/control_blind_4seed")
    parser.add_argument("--seeds", default="71011,71013,71017,71019")
    parser.add_argument("--images_per_panel", type=int, default=16)
    parser.add_argument("--shuffle_seed", type=int, default=20260805)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if len(seeds) < 4:
        raise ValueError("blind protocol requires at least four seeds")
    if args.images_per_panel <= 0:
        raise ValueError("images_per_panel must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_args = SimpleNamespace(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        flow_path="linear",
    )
    panels: list[torch.Tensor] = []
    identities: list[dict] = []

    for run_name, representation in ARMS:
        run_dir = Path(args.run_root) / run_name
        history = json.loads((run_dir / "history.json").read_text())
        mean, std = float(history["mean"]), float(history["std"])
        arm_args = SimpleNamespace(**vars(args), representation=representation)
        tokens, dim = _layout(arm_args)
        model = PatchDiffusion(tokens, dim, model_args).to(device)
        payload = torch.load(run_dir / "final.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        codec = None
        permutation = None
        if representation == "fft_compact_isometric_spiral":
            codec = build_compact_isometric_codec(args.image_size, device)
            permutation = orbit_order_permutation(codec, "square_spiral")

        for seed in seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                values = model.sample(
                    args.images_per_panel,
                    args.num_inference_steps,
                    device,
                )
            images = _decode(
                values.float(), arm_args, mean, std, codec, permutation
            ).clamp(0.0, 1.0)
            panels.append(
                make_grid(images.cpu(), nrow=4, padding=1, pad_value=0.0)
            )
            identities.append(
                {
                    "run": run_name,
                    "representation": representation,
                    "seed": seed,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    order = list(range(len(panels)))
    random.Random(args.shuffle_seed).shuffle(order)
    ordered_panels = torch.stack([panels[index] for index in order])
    sheet = make_grid(ordered_panels, nrow=len(ARMS), padding=8, pad_value=0.5)
    sheet_path = output_dir / "blind_sheet.png"
    save_image(sheet, sheet_path)
    manifest = {
        "version": 1,
        "created_before_rating": True,
        "panel_order": "row_major",
        "columns": len(ARMS),
        "rows": len(seeds),
        "images_per_panel": args.images_per_panel,
        "inference_steps": args.num_inference_steps,
        "shuffle_seed": args.shuffle_seed,
        "panels": [
            {
                "panel_index": panel_index,
                "row": panel_index // len(ARMS),
                "column": panel_index % len(ARMS),
                **identities[source_index],
            }
            for panel_index, source_index in enumerate(order)
        ],
    }
    key_path = output_dir / "blind_key.json"
    key_path.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(key_path.read_bytes()).hexdigest()
    (output_dir / "blind_key.sha256").write_text(
        f"{digest}  {key_path.name}\n"
    )
    print(f"sheet={sheet_path}")
    print(f"key={key_path}")
    print(f"key_sha256={digest}")


if __name__ == "__main__":
    main()
