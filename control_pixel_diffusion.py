"""Control: the same transformer and training budget, on pixel patches.

The joint latent model does not memorize yet produces texture mush, and CIFAR-10's
50k images are known to support excellent generation (DDPM FID 3.17 at ~36M
params, EDM FID 1.79 at ~56M). So "not enough data" cannot by itself explain the
failure -- unless our *architecture and training budget* are the limiting factor
rather than the representation.

This isolates that. Identical bidirectional transformer blocks, identical
rectified-flow objective, identical width/depth/steps/batch/schedule as the joint
latent runs. The only change is what a token is: a 4x4 pixel patch (64 tokens of
48 dims) instead of a frequency latent (53 tokens of 64 dims).

Outcome reading:
  coherent images  -> data and architecture are sufficient; the frequency latent
                      representation is what breaks generation
  texture mush     -> the transformer/budget/data combination is the limit, and
                      the representation is exonerated
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from causal_transformer import CausalTransformerBlock
from diffusion_decoder import FinalLayer, TimestepEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/AFIG/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument(
        "--representation",
        choices=[
            "pixels",
            "patch_dct",
            "full_dct",
            "full_hartley",
            "fft_whitened",
            "fft_global",
            "fft_global_spiral",
        ],
        default="pixels",
        help=(
            "pixels: 4x4 spatial patches. patch_dct: an orthonormal DCT inside "
            "each spatial patch. full_dct: one global orthonormal image DCT, "
            "grouped into 4x4 frequency patches. full_hartley: a real, periodic, "
            "globally supported orthonormal Fourier-family basis on the same "
            "frequency grid. fft_whitened: per-orbit whitened FFT. fft_global: "
            "FFT with only a global scalar mean/std. fft_global_spiral: the same "
            "FFT values reordered by square spiral before grouping, making each "
            "48-D token more local in the 2-D frequency plane. All modes use "
            "~64 tokens x 48 dims."
        ),
    )
    parser.add_argument("--codec_stats", default="autoencoder_runs/codec_stats_32.pt")
    parser.add_argument("--orbits_per_token", type=int, default=8)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--preview_steps", type=int, default=5000)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


class PatchDiffusion(nn.Module):
    def __init__(self, tokens: int, dim: int, args: argparse.Namespace):
        super().__init__()
        width = args.width
        self.tokens, self.dim = tokens, dim
        self.input_projection = nn.Linear(dim, width)
        self.position = nn.Parameter(torch.zeros(tokens, width))
        self.time_embed = TimestepEmbedder(width)
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(
                    width=width,
                    num_heads=args.num_heads,
                    ff_mult=args.ff_mult,
                    dropout=0.0,
                    conditional_film=True,
                    causal=False,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.final_layer = FinalLayer(width, dim)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def velocity(self, noisy: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(noisy) + self.position.to(noisy.dtype)
        condition = self.time_embed(flow_time * 999.0).unsqueeze(1).expand_as(hidden)
        for layer in self.layers:
            hidden, _ = layer(hidden, condition=condition)
        return self.final_layer(hidden, condition)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        t = torch.rand(batch, device=x.device)
        noise = torch.randn_like(x)
        view = t[:, None, None]
        noisy = view * x + (1.0 - view) * noise
        return (self.velocity(noisy, t) - (x - noise)).square().mean()

    @torch.no_grad()
    def sample(self, count: int, steps: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(count, self.tokens, self.dim, device=device)
        dt = 1.0 / steps
        for index in range(steps):
            t = torch.full((count,), index / steps, device=device)
            v = self.velocity(x, t)
            proposal = x + dt * v
            if index + 1 < steps:
                nt = torch.full((count,), (index + 1) / steps, device=device)
                x = x + 0.5 * dt * (v + self.velocity(proposal, nt))
            else:
                x = proposal
        return x


def build_codec(args, device):
    """Codec for the FFT modes; whiten_exponent selects whitened vs global."""
    from frequency import FrequencyCodec, FrequencyCodecConfig

    payload = torch.load(args.codec_stats, map_location="cpu", weights_only=False)
    config_dict = dict(payload["config"])
    config_dict["whiten_exponent"] = 1.0 if args.representation == "fft_whitened" else 0.0
    codec = FrequencyCodec(FrequencyCodecConfig(**config_dict))
    codec.load_exported(payload)
    return codec.to(device).eval()


def orbit_order_permutation(codec, ordering: str) -> torch.Tensor:
    """Indices that express codec orbits in another deterministic ordering."""
    from frequency import build_orbit_table

    table = build_orbit_table(
        codec.config.height, codec.config.width, ordering=ordering
    )
    codec_index = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(codec.ky.cpu(), codec.kx.cpu()))
    }
    permutation = torch.tensor(
        [codec_index[(int(y), int(x))] for y, x in zip(table["ky"], table["kx"])],
        dtype=torch.long,
        device=codec.ky.device,
    )
    if permutation.unique().numel() != codec.seq_len_int:
        raise RuntimeError(f"{ordering} orbit order is not a bijection")
    return permutation


def fft_to_tokens(codec, images, orbits_per_token, permutation=None):
    """[B,3,32,32] -> [B, ceil(L/g), g*6], zero-padded on the orbit axis."""
    tokens = codec.encode(images)
    if permutation is not None:
        tokens = tokens[:, permutation]
    batch, orbits, components = tokens.shape
    groups = -(-orbits // orbits_per_token)
    padded = torch.zeros(
        batch, groups * orbits_per_token, components,
        device=tokens.device, dtype=tokens.dtype,
    )
    padded[:, :orbits] = tokens
    return padded.reshape(batch, groups, orbits_per_token * components)


def tokens_to_images(codec, grouped, orbits_per_token, permutation=None):
    batch, groups, _ = grouped.shape
    tokens = grouped.reshape(batch, groups * orbits_per_token, 6)[:, : codec.seq_len_int]
    if permutation is not None:
        codec_order = torch.empty_like(tokens)
        codec_order[:, permutation] = tokens
        tokens = codec_order
    return codec.decode(tokens)


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    x = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)
    return x


def unpatchify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    batch = tokens.shape[0]
    grid = size // patch
    x = tokens.reshape(batch, grid, grid, 3, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, size, size)
    return x


def orthonormal_dct_matrix(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix C."""
    sample = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequency = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    matrix = torch.cos(math.pi * frequency * sample[None, :] / size)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix.to(dtype=dtype)


def dct_2d(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal 2-D DCT over the final two dimensions."""
    return torch.einsum("ki,...ij,lj->...kl", matrix, values, matrix)


def idct_2d(coefficients: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Invert :func:`dct_2d` over the final two dimensions."""
    return torch.einsum("ki,...kl,lj->...ij", matrix, coefficients, matrix)


def patch_dctify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Encode spatially local patches in a real orthonormal frequency basis."""
    batch, channels, height, width = images.shape
    grid_h, grid_w = height // patch, width // patch
    patches = images.reshape(
        batch, channels, grid_h, patch, grid_w, patch
    ).permute(0, 2, 4, 1, 3, 5)
    matrix = orthonormal_dct_matrix(
        patch, device=images.device, dtype=images.dtype
    )
    coefficients = dct_2d(patches, matrix)
    return coefficients.reshape(batch, grid_h * grid_w, channels * patch * patch)


def patch_idctify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Decode spatially local DCT tokens back to normalized pixels."""
    batch = tokens.shape[0]
    grid = size // patch
    coefficients = tokens.reshape(batch, grid, grid, 3, patch, patch)
    matrix = orthonormal_dct_matrix(
        patch, device=tokens.device, dtype=tokens.dtype
    )
    patches = idct_2d(coefficients, matrix)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, size, size)


def full_dctify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Apply one global image DCT and group its plane into frequency patches."""
    matrix = orthonormal_dct_matrix(
        images.shape[-1], device=images.device, dtype=images.dtype
    )
    return patchify(dct_2d(images, matrix), patch)


def full_idctify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Invert globally supported DCT frequency-patch tokens."""
    coefficients = unpatchify(tokens, patch, size)
    matrix = orthonormal_dct_matrix(
        size, device=tokens.device, dtype=tokens.dtype
    )
    return idct_2d(coefficients, matrix)


def orthonormal_hartley_matrix(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the real orthonormal DHT matrix using cas(x)=cos(x)+sin(x)."""
    sample = torch.arange(size, device=device, dtype=torch.float32)
    frequency = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    angle = 2.0 * math.pi * frequency * sample[None, :] / size
    return ((angle.cos() + angle.sin()) / math.sqrt(size)).to(dtype=dtype)


def full_hartleyify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Apply a separable global real Hartley transform and frequency-patch it."""
    matrix = orthonormal_hartley_matrix(
        images.shape[-1], device=images.device, dtype=images.dtype
    )
    return patchify(dct_2d(images, matrix), patch)


def full_ihartleyify(tokens: torch.Tensor, patch: int, size: int) -> torch.Tensor:
    """Invert globally supported Hartley frequency-patch tokens."""
    coefficients = unpatchify(tokens, patch, size)
    matrix = orthonormal_hartley_matrix(
        size, device=tokens.device, dtype=tokens.dtype
    )
    return idct_2d(coefficients, matrix)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    dataset = torchvision.datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
        drop_last=True, persistent_workers=True,
    )

    # One global scalar mean/std, the standard treatment for pixels -- which
    # preserves the eigenspectrum, unlike per-frequency whitening.
    sample_images = torch.stack([dataset[i][0] for i in range(4096)], dim=0)
    codec = None
    fft_permutation = None
    if args.representation in ("pixels", "patch_dct", "full_dct", "full_hartley"):
        mean = float(sample_images.mean())
        std = float(sample_images.std())
        tokens = (32 // args.patch) ** 2
        dim = 3 * args.patch * args.patch
    else:
        codec = build_codec(args, device)
        if args.representation == "fft_global_spiral":
            fft_permutation = orbit_order_permutation(codec, "square_spiral")
        with torch.no_grad():
            probe = fft_to_tokens(
                codec,
                sample_images.to(device),
                args.orbits_per_token,
                fft_permutation,
            )
        # The codec already centres and scales; one residual global standardization
        # keeps every mode on the same footing for the diffusion process.
        mean = float(probe.mean())
        std = float(probe.std())
        tokens, dim = probe.shape[1], probe.shape[2]
    print(f"representation={args.representation} normalization: mean {mean:.4f} std {std:.4f}")
    model = PatchDiffusion(tokens, dim, args).to(device)
    print(f"tokens {tokens} x dim {dim}; params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=torch.cuda.is_available(),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 1.0 - 0.75 * progress

    step = 0
    history = []
    scaler_dtype = torch.bfloat16
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_at(step)
            images = images.to(device)
            if args.representation == "pixels":
                x = patchify((images - mean) / std, args.patch)
            elif args.representation == "patch_dct":
                x = patch_dctify((images - mean) / std, args.patch)
            elif args.representation == "full_dct":
                x = full_dctify((images - mean) / std, args.patch)
            elif args.representation == "full_hartley":
                x = full_hartleyify((images - mean) / std, args.patch)
            else:
                with torch.no_grad():
                    x = (
                        fft_to_tokens(
                            codec, images, args.orbits_per_token, fft_permutation
                        )
                        - mean
                    ) / std
            with torch.autocast("cuda", dtype=scaler_dtype):
                loss = model.loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 500 == 0:
                history.append({"step": step, "loss": float(loss.detach())})
                if step % 2500 == 0:
                    print(f"  step {step:>6} loss {float(loss.detach()):.4f}")
            if args.preview_steps and step % args.preview_steps == 0:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=scaler_dtype):
                    samples = model.sample(16, args.inference_steps, device)
                raw = samples.float()
                if args.representation == "pixels":
                    decoded = unpatchify(raw, args.patch, 32) * std + mean
                elif args.representation == "patch_dct":
                    decoded = patch_idctify(raw, args.patch, 32) * std + mean
                elif args.representation == "full_dct":
                    decoded = full_idctify(raw, args.patch, 32) * std + mean
                elif args.representation == "full_hartley":
                    decoded = full_ihartleyify(raw, args.patch, 32) * std + mean
                else:
                    raw = raw * std + mean
                    decoded = tokens_to_images(
                        codec, raw, args.orbits_per_token, fft_permutation
                    )
                save_image(
                    decoded.clamp(0, 1),
                    os.path.join(args.output_dir, f"preview_{step:07d}.png"),
                    nrow=8,
                )
                model.train()

    with open(os.path.join(args.output_dir, "history.json"), "w") as handle:
        json.dump({"history": history, "mean": mean, "std": std}, handle, indent=2)
    torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "final.pt"))
    print("done")


if __name__ == "__main__":
    main()
