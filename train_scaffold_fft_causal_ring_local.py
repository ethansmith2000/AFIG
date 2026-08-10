#!/usr/bin/env python3
"""Fine-tune the passing local FFT residual flow under a causal ring schedule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from control_pixel_diffusion import patchify
from scaffold_fft_causal_ring_local import (
    CausalRingLocalDenoiser,
    causal_ring_flow_loss,
    load_joint_denoiser_weights,
    model_args_from_joint_checkpoint,
    sample_causal_ring_fft,
    validate_scalar_rings,
)
from train_scaffold_fft_residual import (
    deterministic_scaffold,
    fft_state_to_images,
    images_to_fft_state,
    make_compact_layout,
    psnr,
)
from train_spatial_latent_hartley_ar import load_spatial_ae


DEFAULT_INIT = (
    "latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000/"
    "checkpoint_30000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--validation_images", type=int, default=8)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_initialized_model(
    payload: dict,
    *,
    local_tokens: int,
    patch_dim: int,
    ring_count: int,
    device: torch.device,
) -> CausalRingLocalDenoiser:
    arguments = payload["args"]
    model = CausalRingLocalDenoiser(
        tokens=local_tokens,
        patch_dim=patch_dim,
        ring_count=ring_count,
        args=model_args_from_joint_checkpoint(arguments),
    ).to(device)
    load_joint_denoiser_weights(model, payload["model"])
    return model


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.validation_images = 2
        args.preview_steps = 0
        args.checkpoint_steps = 0
        args.inference_steps = 2
        args.warmup = 1
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if args.inference_steps <= 0:
        raise ValueError("inference_steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    init_payload = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    saved_args = init_payload["args"]
    image_size = int(saved_args["image_size"])
    patch = int(saved_args["patch"])
    token_dim = int(saved_args["compact_token_dim"])
    if image_size != 32 or patch != 4 or token_dim != 48:
        raise ValueError("the controlled gate requires the passing 32x32, 4x4, 64x48 setup")

    normalization = init_payload["normalization"]
    required_stats = ("scaffold_mean", "scaffold_std", "residual_mean", "residual_std")
    if any(name not in normalization for name in required_stats):
        raise ValueError("initial checkpoint lacks scaffold/residual scalar statistics")
    autoencoder = load_spatial_ae(init_payload["ae_checkpoint"], device)
    if autoencoder.config.variational:
        raise ValueError("the oracle scaffold gate requires a deterministic AE")

    codec, layout_orbit, layout_component = make_compact_layout(image_size, device)
    saved_orbit = init_payload["compact_layout_orbit"].to(device)
    saved_component = init_payload["compact_layout_component"].to(device)
    if not torch.equal(layout_orbit, saved_orbit) or not torch.equal(
        layout_component, saved_component
    ):
        raise RuntimeError("current compact FFT layout differs from the initialization")
    scalar_ring = codec.radius_bin[layout_orbit]
    ring_counts = validate_scalar_rings(
        scalar_ring, expected_values=3 * image_size * image_size
    ).to(device)
    local_tokens = (image_size // patch) ** 2
    patch_dim = 3 * patch**2
    model = load_initialized_model(
        init_payload,
        local_tokens=local_tokens,
        patch_dim=patch_dim,
        ring_count=ring_counts.numel(),
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "causal-ring local FFT residual: "
        f"rings={ring_counts.numel()}, dimensions={ring_counts.tolist()}, "
        f"params={parameter_count / 1e6:.3f}M"
    )
    print("normalization=" + json.dumps(normalization, sort_keys=True))
    print(
        "schedule=earlier:data,current:linear-flow,future:fixed-base; "
        "ring_sampling=proportional-active-scalars; compute=ifft-local-fft-mask"
    )

    train_set = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    test_set = datasets.CIFAR10(
        args.data_root, train=False, download=False, transform=transforms.ToTensor()
    )
    validation_images = torch.stack(
        [test_set[index][0] for index in range(args.validation_images)]
    ).to(device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        validation_scaffold = deterministic_scaffold(autoencoder, validation_images)

    scaffold_mean = float(normalization["scaffold_mean"])
    scaffold_std = float(normalization["scaffold_std"])
    residual_mean = float(normalization["residual_mean"])
    residual_std = float(normalization["residual_std"])
    validation_scaffold_patches = patchify(
        (validation_scaffold - scaffold_mean) / scaffold_std, patch
    )
    save_image(
        torch.cat([validation_images, validation_scaffold], dim=0).clamp(0, 1),
        output_dir / "scaffold_baseline.png",
        nrow=args.validation_images,
    )
    print(f"validation scaffold PSNR={psnr(validation_images, validation_scaffold):.3f}dB")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.25 + 0.75 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    history: list[dict[str, float | int]] = []
    sampled_ring_counts = torch.zeros_like(ring_counts, dtype=torch.long)
    global_step = 0
    progress = tqdm(total=args.steps, desc="scaffold-fft-causal-ring-local")
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_scale(global_step)
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scaffold = deterministic_scaffold(autoencoder, images)
            residual = (images.float() - scaffold - residual_mean) / residual_std
            scaffold_patches = patchify(
                (scaffold - scaffold_mean) / scaffold_std, patch
            )
            target_fft = images_to_fft_state(
                codec,
                residual,
                layout_orbit,
                layout_component,
                token_dim,
            )
            model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = causal_ring_flow_loss(
                    model,
                    codec,
                    target_fft,
                    scaffold_patches,
                    scalar_ring,
                    layout_orbit=layout_orbit,
                    layout_component=layout_component,
                    patch=patch,
                    image_size=image_size,
                    token_dim=token_dim,
                )
                loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sampled_ring_counts += torch.bincount(
                output["target_ring"], minlength=ring_counts.numel()
            )
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0 or global_step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=optimizer.param_groups[0]["lr"],
                )
            if global_step % 100 == 0 or global_step == args.steps:
                history.append(
                    {
                        "step": global_step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "mean_flow_time": float(output["flow_time"].mean()),
                    }
                )

            if args.preview_steps > 0 and global_step % args.preview_steps == 0:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    sampled_fft = sample_causal_ring_fft(
                        model,
                        codec,
                        validation_scaffold_patches,
                        scalar_ring,
                        layout_orbit=layout_orbit,
                        layout_component=layout_component,
                        patch=patch,
                        image_size=image_size,
                        token_dim=token_dim,
                        steps=args.inference_steps,
                        generator=generator,
                    )
                sampled_normalized = fft_state_to_images(
                    codec, sampled_fft, layout_orbit, layout_component
                )
                completion = (
                    validation_scaffold
                    + sampled_normalized * residual_std
                    + residual_mean
                )
                completion_psnr = psnr(validation_images, completion.clamp(0, 1))
                save_image(
                    torch.cat(
                        [validation_images, validation_scaffold, completion], dim=0
                    ).clamp(0, 1),
                    output_dir / f"preview_{global_step:07d}.png",
                    nrow=args.validation_images,
                )
                print(
                    f"PREVIEW step={global_step} completion_psnr={completion_psnr:.3f}dB"
                )

            if args.checkpoint_steps > 0 and global_step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "kind": "scaffold_fft_causal_ring_local",
                        "step": global_step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "joint_model_args": init_payload["args"],
                        "ae_checkpoint": str(Path(init_payload["ae_checkpoint"]).resolve()),
                        "initialization_checkpoint": str(Path(args.init_checkpoint).resolve()),
                        "normalization": normalization,
                        "compact_layout_orbit": layout_orbit.cpu(),
                        "compact_layout_component": layout_component.cpu(),
                        "scalar_ring": scalar_ring.cpu(),
                        "ring_counts": ring_counts.cpu(),
                        "sampled_ring_counts": sampled_ring_counts.cpu(),
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    with (output_dir / "history.json").open("w") as handle:
        json.dump(
            {
                "kind": "scaffold_fft_causal_ring_local",
                "initialization_checkpoint": str(Path(args.init_checkpoint).resolve()),
                "normalization": normalization,
                "parameter_count": parameter_count,
                "ring_counts": ring_counts.cpu().tolist(),
                "sampled_ring_counts": sampled_ring_counts.cpu().tolist(),
                "schedule": "earlier=data,current=linear-flow,future=fixed-Gaussian",
                "computation": "exact FFT state -> aligned local patches -> FFT velocity -> current-ring mask",
                "history": history,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print("done")


if __name__ == "__main__":
    main()
