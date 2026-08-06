#!/usr/bin/env python3
"""Sample global coordinates through the successful pixel model by conjugacy.

The state and ODE updates stay in a globally supported orthonormal basis.  Only
the velocity evaluation is mapped to the local pixel-token basis expected by the
trained model, then mapped back.  This is a zero-training test of stochastic
state geometry versus native network-interface geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision.utils import save_image

from control_pixel_diffusion import (
    PatchDiffusion,
    build_compact_isometric_codec,
    compact_active_scalar_layout,
    compact_scalar_fft_to_tokens,
    compact_scalar_tokens_to_images,
    orbit_order_permutation,
    patch_grid_dctify,
    patch_grid_idctify,
    patchify,
    unpatchify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixel_checkpoint_dir",
        default="latent_continuous_runs/pixel_control",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--representation",
        choices=["patch_grid_dct", "compact_fft_gridlocal"],
        required=True,
    )
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=71021)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.inference_steps <= 0:
        raise ValueError("count and inference_steps must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.pixel_checkpoint_dir)
    history = json.loads((checkpoint_dir / "history.json").read_text())
    mean, std = float(history["mean"]), float(history["std"])
    device = torch.device(args.device)
    token_count = (args.image_size // args.patch) ** 2
    token_dim = 3 * args.patch**2
    model_args = SimpleNamespace(
        width=args.width,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        flow_path="linear",
    )
    model = PatchDiffusion(token_count, token_dim, model_args).to(device)
    payload = torch.load(
        checkpoint_dir / "final.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    codec = None
    layout_orbit = None
    layout_component = None
    if args.representation == "compact_fft_gridlocal":
        codec = build_compact_isometric_codec(args.image_size, device)
        permutation = orbit_order_permutation(codec, "square_spiral")
        layout_orbit, layout_component = compact_active_scalar_layout(
            codec, permutation
        )

    def local_to_global(tokens: torch.Tensor) -> torch.Tensor:
        normalized = unpatchify(tokens, args.patch, args.image_size)
        if args.representation == "patch_grid_dct":
            return patch_grid_dctify(normalized, args.patch)
        return compact_scalar_fft_to_tokens(
            codec,
            normalized,
            layout_orbit,
            layout_component,
            token_dim=token_dim,
        )

    def global_to_local(tokens: torch.Tensor) -> torch.Tensor:
        if args.representation == "patch_grid_dct":
            normalized = patch_grid_idctify(
                tokens, args.patch, args.image_size
            )
        else:
            normalized = compact_scalar_tokens_to_images(
                codec, tokens, layout_orbit, layout_component
            )
        return patchify(normalized, args.patch)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    local_base = torch.randn(
        args.count, token_count, token_dim, device=device
    )
    state = local_to_global(local_base)
    base_roundtrip = global_to_local(state)
    base_error = float((base_roundtrip - local_base).abs().max())

    dt = 1.0 / args.inference_steps
    with torch.no_grad(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for index in range(args.inference_steps):
            time = torch.full(
                (args.count,), index / args.inference_steps, device=device
            )
            velocity = local_to_global(
                model.velocity(global_to_local(state).to(torch.float32), time)
            )
            proposal = state + dt * velocity
            if index + 1 < args.inference_steps:
                next_time = torch.full(
                    (args.count,),
                    (index + 1) / args.inference_steps,
                    device=device,
                )
                next_velocity = local_to_global(
                    model.velocity(
                        global_to_local(proposal).to(torch.float32), next_time
                    )
                )
                state = state + 0.5 * dt * (velocity + next_velocity)
            else:
                state = proposal

    normalized = unpatchify(
        global_to_local(state.float()), args.patch, args.image_size
    )
    images = normalized * std + mean
    save_image(
        images.clamp(0.0, 1.0),
        output_dir / "samples.png",
        nrow=min(8, args.count),
    )
    metrics = {
        "representation": args.representation,
        "pixel_checkpoint_dir": str(checkpoint_dir.resolve()),
        "count": args.count,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "base_roundtrip_max_abs": base_error,
        "unclipped_min": float(images.min()),
        "unclipped_max": float(images.max()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
