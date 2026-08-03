"""Autoregressive sampling with a normalizing-flow head, then decode.

The flow reaches a far lower held-out NLL than the conditional Gaussian, but low
NLL is precisely the known normalizing-flow failure mode: a density can score well
while placing mass off the data manifold. The only test that settles it is
decoding actual samples.

The trunk is the 7,500-step AR checkpoint (the one shown to generalize), frozen.
At each position the trunk hidden state conditions the flow, the flow samples a
latent in one pass, and that latent is fed back into the trunk. This is a legitimate
hybrid: the flow was trained on exactly these contexts, and the trunk consumes
latents regardless of which head produced them.

Also reports prefix-conditioned completions, so flow and diffusion heads can be
compared on the same footing as `diagnose_ar_prefix_recall.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from flow_decoder import ConditionalFlowDecoder, FlowDecoderConfig
from latent_autoencoder_interface import FrozenLatentAutoencoder
from prototype_flow_head import build_contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_images", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--hidden_width", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--grid_images", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.9, 0.7])
    parser.add_argument("--prefix_lengths", type=int, nargs="+", default=[0, 8, 24])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def rollout(
    model,
    flow: ConditionalFlowDecoder,
    metadata: torch.Tensor,
    batch: int,
    real_latents: torch.Tensor | None,
    prefix_length: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    hidden, caches = model.init_cache(batch, metadata)
    produced: List[torch.Tensor] = []
    for index in range(model.config.sequence_length):
        if real_latents is not None and index < prefix_length:
            latent = real_latents[:, index].to(hidden.dtype)
        else:
            latent = flow.sample(
                hidden.float(), temperature=temperature, generator=generator
            ).to(hidden.dtype)
        produced.append(latent)
        if index + 1 < model.config.sequence_length:
            hidden, caches = model.forward_step(
                latent, index + 1, metadata[index + 1], caches
            )
    return torch.stack(produced, dim=1)


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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = interface.position_features
    print(f"frozen AR trunk step {step}")

    train_context, train_target = build_contexts(
        interface, model, args.data_root, True, args.train_images, device
    )
    flow = ConditionalFlowDecoder(
        FlowDecoderConfig(
            token_dim=train_target.shape[-1],
            context_dim=train_context.shape[-1],
            num_layers=args.num_layers,
            hidden_width=args.hidden_width,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    for iteration in range(args.steps):
        index = torch.randint(0, train_context.shape[0], (args.batch_size,), device=device)
        loss = flow.loss(train_target[index], train_context[index])["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if (iteration + 1) % 1500 == 0:
            print(f"  flow step {iteration+1}: train NLL/dim {float(loss):.4f}")
    flow.eval()
    del train_context, train_target

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transform
    )
    grid_n = args.grid_images
    images = torch.stack([dataset[i][0] for i in range(grid_n)], dim=0).to(device)
    real_latents = interface.encode_images(images)

    rows = [interface.decode_latents(real_latents)]
    labels = ["real_reconstruction"]
    results: List[Dict[str, float]] = []

    for temperature in args.temperatures:
        generator = torch.Generator(device=device).manual_seed(555)
        sampled = rollout(
            model, flow, metadata, grid_n, None, 0, temperature, generator
        )
        decoded = interface.decode_latents(sampled.float())
        rows.append(decoded)
        labels.append(f"flow_uncond_T{temperature}")
        results.append(
            {
                "mode": "unconditional",
                "temperature": temperature,
                "latent_rms": float(sampled.float().pow(2).mean().sqrt()),
                "pixel_std": float(decoded.std()),
            }
        )

    for prefix_length in args.prefix_lengths:
        if prefix_length == 0:
            continue
        generator = torch.Generator(device=device).manual_seed(555)
        completed = rollout(
            model, flow, metadata, grid_n, real_latents, prefix_length, 1.0, generator
        )
        decoded = interface.decode_latents(completed.float())
        mse = float(((decoded - images) ** 2).mean())
        rows.append(decoded)
        labels.append(f"flow_prefix{prefix_length}")
        results.append(
            {
                "mode": "prefix",
                "prefix_length": prefix_length,
                "psnr_vs_source": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
                "sampled_suffix_latent_mse": float(
                    ((completed[:, prefix_length:].float() - real_latents[:, prefix_length:]) ** 2).mean()
                ),
            }
        )

    save_image(
        torch.cat(rows, dim=0),
        os.path.join(args.output_dir, "flow_samples.png"),
        nrow=grid_n,
    )
    report = {"ar_step": step, "results": results, "grid_row_order": labels}
    with open(os.path.join(args.output_dir, "flow_samples.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Flow head sampling ===")
    for row in results:
        if row["mode"] == "unconditional":
            print(f"  uncond T={row['temperature']}: latent RMS {row['latent_rms']:.3f}"
                  f"  pixel std {row['pixel_std']:.3f}")
        else:
            print(f"  prefix {row['prefix_length']:>2}: PSNR {row['psnr_vs_source']:.2f} dB"
                  f"  suffix latent MSE {row['sampled_suffix_latent_mse']:.3f}")
    print(f"\ngrid rows: {labels}")


if __name__ == "__main__":
    main()
