"""Does the AR model recall training images when given their own prefix?

Section 16 showed the AR model's conditional advantage reverses on held-out data,
i.e. it memorized. The obvious objection is that samples look nothing like CIFAR,
whereas a memorizing model "should" emit training data.

The resolution is that memorizing a conditional map p(x_i | x_<i) is not
memorizing the distribution. A discrete AR model can emit training data verbatim
because greedy decoding follows a memorized path exactly through a discrete token
space; in a continuous 64-D space the sampler never re-enters a memorized
trajectory, so the lookup is always queried off-domain and returns noise.

That story makes a falsifiable prediction: teacher-force a *real* prefix and let
the model complete it, and completions from TRAIN prefixes should recover the
source image far better than completions from TEST prefixes. If train and test
completions are equally bad, the memorization reading is wrong.
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
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--prefix_lengths", type=int, nargs="+", default=[0, 4, 8, 16, 24])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def complete_from_prefix(
    model,
    metadata: torch.Tensor,
    real_latents: torch.Tensor,
    prefix_length: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Teacher-force the first `prefix_length` tokens, then sample the rest."""
    batch = real_latents.shape[0]
    hidden, caches = model.init_cache(batch, metadata)
    produced: List[torch.Tensor] = []
    for index in range(model.config.sequence_length):
        if index < prefix_length:
            latent = real_latents[:, index].to(hidden.dtype)
        else:
            target_metadata = metadata[index].to(
                device=hidden.device, dtype=hidden.dtype
            )[None].expand(batch, -1)
            latent = model.diffusion.sample(
                hidden,
                target_condition=target_metadata,
                unconditional_z=model.null_context.to(dtype=hidden.dtype)[None].expand(
                    batch, -1
                ),
                cfg_scale=1.0,
                cfg_norm_match=False,
                generator=generator,
                num_inference_steps=steps,
                temperature=1.0,
            )
        produced.append(latent)
        if index + 1 < model.config.sequence_length:
            hidden, caches = model.forward_step(
                latent, index + 1, metadata[index + 1], caches
            )
    return torch.stack(produced, dim=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)
    from train_latent_continuous import load_latent_checkpoint

    model, step = load_latent_checkpoint(args.ar_checkpoint, interface)
    model = model.to(device).eval()
    metadata = interface.position_features
    print(f"AR checkpoint step {step}")

    transform = transforms.Compose([transforms.ToTensor()])
    splits = {}
    for name, is_train in (("train", True), ("test", False)):
        dataset = torchvision.datasets.CIFAR10(
            args.data_root, train=is_train, download=False, transform=transform
        )
        images = torch.stack(
            [dataset[i][0] for i in range(args.num_images)], dim=0
        ).to(device)
        splits[name] = (images, interface.encode_images(images))

    grid_n = args.grid_images
    rows: List[torch.Tensor] = []
    labels: List[str] = []
    results: List[Dict[str, float]] = []
    for name, (images, latents) in splits.items():
        rows.append(interface.decode_latents(latents[:grid_n]))
        labels.append(f"{name}_reconstruction")
        for prefix_length in args.prefix_lengths:
            generator = torch.Generator(device=device).manual_seed(999)
            completed = complete_from_prefix(
                model, metadata, latents, prefix_length,
                args.num_inference_steps, generator,
            )
            decoded = interface.decode_latents(completed.float())
            mse = float(((decoded - images) ** 2).mean())
            psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
            # Latent-space agreement on the SAMPLED region only.
            if prefix_length < latents.shape[1]:
                suffix_mse = float(
                    ((completed[:, prefix_length:].float() - latents[:, prefix_length:]) ** 2).mean()
                )
            else:
                suffix_mse = 0.0
            results.append(
                {
                    "split": name,
                    "prefix_length": prefix_length,
                    "psnr_vs_source": psnr,
                    "sampled_suffix_latent_mse": suffix_mse,
                }
            )
            rows.append(decoded[:grid_n])
            labels.append(f"{name}_prefix{prefix_length}")

    save_image(
        torch.cat(rows, dim=0),
        os.path.join(args.output_dir, "prefix_recall.png"),
        nrow=grid_n,
    )
    report = {"ar_step": step, "results": results, "grid_row_order": labels}
    with open(os.path.join(args.output_dir, "prefix_recall.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\n{'prefix':>7} {'TRAIN psnr':>12} {'TEST psnr':>11} {'gap':>8} |"
          f" {'TRAIN sufMSE':>13} {'TEST sufMSE':>12}")
    for prefix_length in args.prefix_lengths:
        train = next(r for r in results if r["split"] == "train" and r["prefix_length"] == prefix_length)
        test = next(r for r in results if r["split"] == "test" and r["prefix_length"] == prefix_length)
        print(
            f"{prefix_length:>7d} {train['psnr_vs_source']:>12.2f} {test['psnr_vs_source']:>11.2f}"
            f" {train['psnr_vs_source'] - test['psnr_vs_source']:>+8.2f} |"
            f" {train['sampled_suffix_latent_mse']:>13.4f} {test['sampled_suffix_latent_mse']:>12.4f}"
        )
    print("\nA large train-over-test gap supports memorization;")
    print("equally bad completions would refute it.")


if __name__ == "__main__":
    main()
