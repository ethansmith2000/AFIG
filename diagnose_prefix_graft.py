"""Assign blame for incoherent samples between the latent prefix and suffix.

Perceptual importance is concentrated in the first few latent positions (the
low-frequency rings): corrupting positions 0-3 damages the image more than
corrupting the remaining 49 combined.  This script grafts real and generated
latents across a split point to find which part of a generated sample is
actually broken.

  * real prefix + generated suffix  -> coherent?  then the suffix is fine
  * generated prefix + real suffix  -> mush?      then the prefix is off-manifold

Both grafts are decoded and scored, and the per-position marginal statistics of
generated latents are compared against real ones for the prefix specifically.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from latent_autoencoder_interface import FrozenLatentAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=1024)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transform
    )
    images = torch.stack(
        [dataset[i][0] for i in range(args.num_images)], dim=0
    ).to(device)
    real = interface.encode_images(images)

    from train_joint_latent_diffusion import load_checkpoint

    model, step = load_checkpoint(args.joint_checkpoint, interface)
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(1234)
    chunks = []
    remaining = args.num_images
    while remaining > 0:
        take = min(512, remaining)
        chunks.append(
            model.generate_latents(
                take, interface.position_features, generator=generator
            ).float()
        )
        remaining -= take
    generated = torch.cat(chunks, dim=0)
    print(f"joint checkpoint step {step}; generated {tuple(generated.shape)}")

    grid_n = args.grid_images
    rows: List[torch.Tensor] = [
        interface.decode_latents(real[:grid_n]),
        interface.decode_latents(generated[:grid_n]),
    ]
    labels = ["real_reconstruction", "pure_generated"]

    report: Dict[str, object] = {"joint_step": step}
    grafts: List[Dict[str, float]] = []
    for split in (2, 4, 8, 16, 24):
        real_prefix = torch.cat([real[:, :split], generated[:, split:]], dim=1)
        generated_prefix = torch.cat([generated[:, :split], real[:, split:]], dim=1)
        decoded_real_prefix = interface.decode_latents(real_prefix)
        decoded_generated_prefix = interface.decode_latents(generated_prefix)
        grafts.append(
            {
                "split": split,
                "real_prefix_generated_suffix_psnr_vs_real_image": float(
                    10.0
                    * torch.log10(
                        1.0 / torch.mean((decoded_real_prefix - images) ** 2)
                    )
                ),
                "generated_prefix_real_suffix_psnr_vs_real_image": float(
                    10.0
                    * torch.log10(
                        1.0 / torch.mean((decoded_generated_prefix - images) ** 2)
                    )
                ),
            }
        )
        rows.append(decoded_real_prefix[:grid_n])
        labels.append(f"REALprefix{split}_genSuffix")
        rows.append(decoded_generated_prefix[:grid_n])
        labels.append(f"GENprefix{split}_realSuffix")
    report["grafts"] = grafts

    # Per-position marginal comparison, focused on the prefix.
    per_position: List[Dict[str, float]] = []
    for position in range(real.shape[1]):
        real_slice = real[:, position]
        generated_slice = generated[:, position]
        per_position.append(
            {
                "position": position,
                "real_rms": float(real_slice.pow(2).mean().sqrt()),
                "generated_rms": float(generated_slice.pow(2).mean().sqrt()),
                "real_kurtosis": float(
                    (real_slice - real_slice.mean()).pow(4).mean()
                    / real_slice.var().pow(2).clamp_min(1e-12)
                ),
                "generated_kurtosis": float(
                    (generated_slice - generated_slice.mean()).pow(4).mean()
                    / generated_slice.var().pow(2).clamp_min(1e-12)
                ),
            }
        )
    report["per_position_marginals"] = per_position

    save_image(
        torch.cat(rows, dim=0),
        os.path.join(args.output_dir, "graft_comparison.png"),
        nrow=grid_n,
    )
    report["grid_row_order"] = labels

    path = os.path.join(args.output_dir, "graft_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Graft PSNR against the source real image ===")
    print(f"{'split':>6} {'real-prefix+gen-suffix':>24} {'gen-prefix+real-suffix':>24}")
    for row in grafts:
        print(
            f"{row['split']:>6d} {row['real_prefix_generated_suffix_psnr_vs_real_image']:>24.2f}"
            f" {row['generated_prefix_real_suffix_psnr_vs_real_image']:>24.2f}"
        )
    print("\n=== Prefix marginals (real vs generated) ===")
    for row in per_position[:8]:
        print(
            f"  pos {row['position']:>2d}: rms {row['real_rms']:.3f} vs {row['generated_rms']:.3f}"
            f"   kurtosis {row['real_kurtosis']:.2f} vs {row['generated_kurtosis']:.2f}"
        )
    print(f"\ngrid rows: {labels}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
