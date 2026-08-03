"""Is the joint latent diffusion model overfitting CIFAR-10?

30k steps at batch 256 is ~154 epochs of a 50k-image dataset, so the achieved
`train/flow_mse` of 0.908 may not reflect held-out performance.  Every earlier
comparison against the Gaussian floor used train loss and therefore cannot tell
underfitting from overfitting.  This script measures the same quantity on train
and held-out test latents under an identical, variance-reduced protocol.

For a fair reference, the linear/Gaussian baseline is *also* fit on train and
evaluated on test: A_t = (t*Sigma - (1-t)I) (t^2*Sigma + (1-t)^2 I)^{-1}, applied
in the train covariance eigenbasis.  Both model and baseline then face the same
generalization test.

Paired evaluation: identical timesteps and identical noise realizations are used
for train and test, so the difference between them is not sampling noise.

Also reports autoencoder reconstruction on train vs test, to check whether the
frozen AE itself overfits.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torchvision
from torchvision import transforms

from latent_autoencoder_interface import FrozenLatentAutoencoder
from model_latent_continuous import LATENT_SEQUENCE_LENGTH, LATENT_TOKEN_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent_interface", required=True)
    parser.add_argument("--joint_checkpoint", required=True)
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_eval", type=int, default=10000)
    parser.add_argument("--covariance_images", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_timesteps", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def encode_split(
    interface: FrozenLatentAutoencoder,
    data_root: str,
    train: bool,
    count: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Returns normalized latents and reconstruction PSNR for the split.

    No augmentation: deterministic ToTensor only, so train and test are measured
    on the same footing.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        data_root, train=train, download=False, transform=transform
    )
    count = min(count, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    chunks: List[torch.Tensor] = []
    squared_error = 0.0
    seen = 0
    for images, _ in loader:
        images = images.to(device)
        latents = interface.encode_images(images)
        decoded = interface.decode_latents(latents)
        squared_error += float(((decoded - images) ** 2).mean()) * images.shape[0]
        seen += images.shape[0]
        chunks.append(latents)
    mse = squared_error / max(seen, 1)
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
    return torch.cat(chunks, dim=0), psnr


@torch.no_grad()
def evaluate_flow_mse(
    model,
    latents: torch.Tensor,
    metadata: torch.Tensor,
    times: torch.Tensor,
    seed: int,
    batch_size: int,
) -> Tuple[float, torch.Tensor]:
    """Velocity MSE averaged over a fixed timestep grid with fixed noise.

    The same seed is used for every split so the noise realizations are shared,
    making the train/test difference a paired comparison.
    """
    total = 0.0
    weight = 0
    per_position = torch.zeros(latents.shape[1], dtype=torch.float64, device=latents.device)
    for time_value in times:
        generator = torch.Generator(device=latents.device).manual_seed(
            seed + int(time_value.item() * 1_000_000)
        )
        for start in range(0, latents.shape[0], batch_size):
            batch = latents[start : start + batch_size]
            noise = torch.randn(
                batch.shape, device=batch.device, dtype=batch.dtype, generator=generator
            )
            flow_time = torch.full(
                (batch.shape[0],), float(time_value), device=batch.device
            )
            view = flow_time[:, None, None]
            noisy = view * batch + (1.0 - view) * noise
            target = batch - noise
            prediction = model.predict_velocity(noisy, flow_time, metadata)
            squared = (prediction.float() - target.float()).square()
            total += float(squared.mean()) * batch.shape[0]
            per_position += squared.mean(dim=-1).sum(dim=0).double()
            weight += batch.shape[0]
    return total / max(weight, 1), (per_position / max(weight, 1)).cpu()


@torch.no_grad()
def linear_baseline_mse(
    latents: torch.Tensor,
    eigenvalues: torch.Tensor,
    vectors: torch.Tensor,
    times: torch.Tensor,
    seed: int,
) -> float:
    """MSE of the Gaussian-optimal linear predictor fit on train, evaluated here.

    Works in the train covariance eigenbasis, where A_t is diagonal:
        a_t = (t*lambda - (1-t)) / (t^2*lambda + (1-t)^2)
    """
    flat = latents.reshape(latents.shape[0], -1).double()
    projected = flat @ vectors
    total = 0.0
    weight = 0
    for time_value in times:
        generator = torch.Generator(device=flat.device).manual_seed(
            seed + int(time_value.item() * 1_000_000)
        )
        noise = torch.randn(
            flat.shape, device=flat.device, dtype=flat.dtype, generator=generator
        )
        projected_noise = noise @ vectors
        t = float(time_value)
        noisy = t * projected + (1.0 - t) * projected_noise
        target = projected - projected_noise
        coefficient = (t * eigenvalues - (1.0 - t)) / (
            t**2 * eigenvalues + (1.0 - t) ** 2
        ).clamp_min(1e-12)
        prediction = noisy * coefficient[None, :]
        total += float((prediction - target).square().mean()) * flat.shape[0]
        weight += flat.shape[0]
    return total / max(weight, 1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    interface = FrozenLatentAutoencoder(
        args.checkpoint, args.latent_interface, sample_posterior=False
    ).to(device)

    train_latents, train_psnr = encode_split(
        interface, args.data_root, True, args.num_eval, args.batch_size, device
    )
    test_latents, test_psnr = encode_split(
        interface, args.data_root, False, args.num_eval, args.batch_size, device
    )
    print(f"AE reconstruction PSNR: train {train_psnr:.2f} dB, test {test_psnr:.2f} dB")

    from train_joint_latent_diffusion import load_checkpoint

    model, step = load_checkpoint(args.joint_checkpoint, interface)
    model = model.to(device).eval()
    print(f"joint checkpoint step {step}")

    times = (torch.arange(args.num_timesteps, dtype=torch.float64) + 0.5) / args.num_timesteps
    metadata = interface.position_features

    train_mse, train_per_position = evaluate_flow_mse(
        model, train_latents, metadata, times, 777, args.batch_size
    )
    test_mse, test_per_position = evaluate_flow_mse(
        model, test_latents, metadata, times, 777, args.batch_size
    )

    # Gaussian baseline fit on a large train sample, evaluated on both splits.
    covariance_latents, _ = encode_split(
        interface, args.data_root, True, args.covariance_images, args.batch_size, device
    )
    flat = covariance_latents.reshape(covariance_latents.shape[0], -1).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    del covariance_latents, flat, centered, covariance

    train_linear = linear_baseline_mse(
        train_latents, eigenvalues, vectors, times, 777
    )
    test_linear = linear_baseline_mse(test_latents, eigenvalues, vectors, times, 777)

    report: Dict[str, object] = {
        "joint_step": step,
        "num_eval_per_split": int(train_latents.shape[0]),
        "num_timesteps": args.num_timesteps,
        "ae_psnr_train": train_psnr,
        "ae_psnr_test": test_psnr,
        "ae_psnr_gap": train_psnr - test_psnr,
        "model_flow_mse_train": train_mse,
        "model_flow_mse_test": test_mse,
        "model_generalization_gap": test_mse - train_mse,
        "linear_gaussian_mse_train": train_linear,
        "linear_gaussian_mse_test": test_linear,
        "model_advantage_over_linear_train": train_linear - train_mse,
        "model_advantage_over_linear_test": test_linear - test_mse,
        "per_position_train": [float(x) for x in train_per_position],
        "per_position_test": [float(x) for x in test_per_position],
    }
    path = os.path.join(args.output_dir, "overfitting_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Autoencoder ===")
    print(f"  reconstruction PSNR   train {train_psnr:7.2f}   test {test_psnr:7.2f}"
          f"   gap {train_psnr - test_psnr:+.2f} dB")
    print("\n=== Joint diffusion velocity MSE (paired noise, 32-point t grid) ===")
    print(f"  model                 train {train_mse:7.4f}   test {test_mse:7.4f}"
          f"   gap {test_mse - train_mse:+.4f}")
    print(f"  linear Gaussian       train {train_linear:7.4f}   test {test_linear:7.4f}"
          f"   gap {test_linear - train_linear:+.4f}")
    print(
        f"\n  model advantage over linear:  train {train_linear - train_mse:+.4f}"
        f"   test {test_linear - test_mse:+.4f}"
    )
    if test_mse > test_linear:
        print("\n  -> On held-out data the model is WORSE than a linear Gaussian fit:"
              " overfitting dominates.")
    else:
        print("\n  -> The model still beats the linear fit on held-out data.")
    print("\n=== Largest per-position generalization gaps ===")
    gaps = sorted(
        ((float(t - r), index) for index, (r, t) in enumerate(zip(train_per_position, test_per_position))),
        reverse=True,
    )
    for gap, index in gaps[:8]:
        print(f"  position {index:>2d}: train {train_per_position[index]:.4f}"
              f"  test {test_per_position[index]:.4f}  gap {gap:+.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
