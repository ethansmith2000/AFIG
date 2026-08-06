"""Matched 64-step autoregressive control over full-image Hartley tiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from causal_transformer import build_rope_tables
from control_pixel_diffusion import full_hartleyify, full_ihartleyify
from diffusion_decoder import DiffusionDecoder, DiffusionDecoderConfig
from model_continuous import TransformerBlock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=7e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--diff_width", type=int, default=768)
    parser.add_argument("--diff_depth", type=int, default=3)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--preview_steps", type=int, default=2500)
    parser.add_argument("--diagnostic_steps", type=int, default=250)
    parser.add_argument("--checkpoint_steps", type=int, default=2500)
    parser.add_argument("--validation_images", type=int, default=16)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument(
        "--rope_mode",
        choices=["frequency_2d", "sequence"],
        default="frequency_2d",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def hartley_tile_order(grid: int) -> torch.Tensor:
    """Low-to-high radial order over the toroidal frequency-tile grid."""
    entries = []
    for y in range(grid):
        sy = y if y < grid // 2 else y - grid
        for x in range(grid):
            sx = x if x < grid // 2 else x - grid
            radius = math.hypot(sy, sx)
            angle = math.atan2(sy, sx)
            entries.append((radius, angle, y, x))
    entries.sort()
    return torch.tensor([y * grid + x for _, _, y, x in entries], dtype=torch.long)


class HartleyTileAR(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        diff_width: int,
        diff_depth: int,
        inference_steps: int,
        grid: int = 8,
        token_dim: int = 48,
        token_order: str = "radial",
        rope_mode: str = "frequency_2d",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.seq_len = grid * grid
        self.token_dim = token_dim
        self.width = width
        self.gradient_checkpointing = gradient_checkpointing
        self.token_proj = nn.Linear(token_dim, width)
        self.bos = nn.Parameter(torch.empty(1, 1, width))
        self.slot_embed = nn.Embedding(self.seq_len, width)
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    width,
                    num_heads,
                    ff_mult,
                    dropout=0.0,
                    position_film=True,
                    qk_norm=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(width)
        head_dim = width // num_heads
        if token_order == "radial":
            order = hartley_tile_order(grid)
        elif token_order == "raster":
            order = torch.arange(self.seq_len, dtype=torch.long)
        else:
            raise ValueError(f"Unknown token order: {token_order}")
        if rope_mode == "sequence":
            coordinates_tensor = torch.arange(self.seq_len, dtype=torch.int64)
        elif rope_mode == "frequency_2d":
            coordinates = []
            for flat in order.tolist():
                y, x = divmod(flat, grid)
                if token_order == "radial":
                    y = y if y < grid // 2 else y - grid
                    x = x if x < grid // 2 else x - grid
                coordinates.append((y, x))
            coordinates_tensor = torch.tensor(coordinates, dtype=torch.int64)
        else:
            raise ValueError(f"Unknown rope_mode: {rope_mode}")
        rope_cos, rope_sin = build_rope_tables(coordinates_tensor, head_dim)
        self.register_buffer("tile_order", order, persistent=True)
        self.register_buffer("rope_cos", rope_cos.float(), persistent=False)
        self.register_buffer("rope_sin", rope_sin.float(), persistent=False)
        self.diffusion = DiffusionDecoder(
            DiffusionDecoderConfig(
                target_dim=token_dim,
                z_channels=width,
                target_condition_dim=width,
                width=diff_width,
                depth=diff_depth,
                objective="flow",
                prediction_type="v_prediction",
                loss_space="native",
                loss_weighting="none",
                component_reduction="fixed_dim",
                flow_solver="heun",
                snr_scale=1.0,
                diffusion_batch_mul=1,
                num_inference_steps=inference_steps,
            )
        )

    def order_tokens(self, raster_tokens: torch.Tensor) -> torch.Tensor:
        return raster_tokens[:, self.tile_order]

    def restore_raster(self, ordered_tokens: torch.Tensor) -> torch.Tensor:
        raster = torch.empty_like(ordered_tokens)
        raster[:, self.tile_order] = ordered_tokens
        return raster

    def slot_condition(
        self, positions: torch.Tensor, batch: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.slot_embed(positions)[None].expand(batch, -1, -1).to(dtype)

    def embed_history(self, history: torch.Tensor) -> torch.Tensor:
        batch, length, _ = history.shape
        dtype = self.token_proj.weight.dtype
        bos = self.bos.expand(batch, -1, -1).to(dtype)
        bos = bos + self.slot_embed(torch.zeros(1, device=history.device, dtype=torch.long))[None].to(dtype)
        if length == 0:
            return bos
        projected = self.token_proj(history.to(dtype))
        slots = torch.arange(1, length + 1, device=history.device)
        projected = projected + self.slot_embed(slots)[None].to(dtype)
        return torch.cat([bos, projected], dim=1)

    def forward_backbone(
        self,
        x: torch.Tensor,
        caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        past = 0
        if caches is not None and caches[0] is not None:
            past = caches[0][0].shape[2]
        positions = torch.arange(past, past + x.shape[1], device=x.device)
        condition = self.slot_condition(positions, x.shape[0], x.dtype)
        new_caches = []
        rope = (self.rope_cos, self.rope_sin)
        for index, layer in enumerate(self.layers):
            cache = None if caches is None else caches[index]
            if self.gradient_checkpointing and self.training and not use_cache:
                def run(module, values, position_values):
                    result, _ = module(
                        values,
                        position_condition=position_values,
                        use_cache=False,
                        rope=rope,
                    )
                    return result

                x = checkpoint(run, layer, x, condition, use_reentrant=False)
            else:
                x, new_cache = layer(
                    x,
                    position_condition=condition,
                    kv_cache=cache,
                    use_cache=use_cache,
                    rope=rope,
                )
                if use_cache:
                    new_caches.append(new_cache)
        return self.final_norm(x), (new_caches if use_cache else None)

    def forward(
        self,
        tokens: torch.Tensor,
        history_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        history = tokens[:, :-1] if history_override is None else history_override
        hidden, _ = self.forward_backbone(self.embed_history(history))
        positions = torch.arange(self.seq_len, device=tokens.device)
        condition = self.slot_condition(positions, tokens.shape[0], hidden.dtype)
        output = self.diffusion.compute_loss(
            target=tokens,
            z=hidden,
            target_condition=condition,
        )
        return output

    @torch.no_grad()
    def generate(
        self,
        count: int,
        steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        dtype = self.token_proj.weight.dtype
        device = self.bos.device
        hidden, caches = self.forward_backbone(
            self.embed_history(torch.empty(count, 0, self.token_dim, device=device)),
            use_cache=True,
        )
        z = hidden[:, -1]
        outputs = []
        for position in range(self.seq_len):
            pos = torch.tensor([position], device=device)
            condition = self.slot_condition(pos, count, z.dtype)[:, 0]
            token = self.diffusion.sample(
                z,
                target_condition=condition,
                generator=generator,
                num_inference_steps=steps,
            )
            outputs.append(token)
            if position + 1 < self.seq_len:
                projected = self.token_proj(token.to(dtype))[:, None]
                next_pos = torch.tensor([position + 1], device=device)
                projected = projected + self.slot_embed(next_pos)[None].to(dtype)
                hidden, caches = self.forward_backbone(
                    projected, caches=caches, use_cache=True
                )
                z = hidden[:, -1]
        return torch.stack(outputs, dim=1)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.num_workers = 0
        args.width = 64
        args.layers = 1
        args.heads = 4
        args.ff_mult = 2
        args.diff_width = 64
        args.diff_depth = 1
        args.inference_steps = 2
        args.preview_steps = 1
        args.diagnostic_steps = 1
        args.checkpoint_steps = 0
        args.validation_images = 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plain = datasets.CIFAR10(
        args.data_root, train=True, download=False, transform=transforms.ToTensor()
    )
    probe = torch.stack([plain[index][0] for index in range(min(4096, len(plain)))])
    mean, std = float(probe.mean()), float(probe.std())
    train_set = datasets.CIFAR10(
        args.data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        ),
    )
    loader = torch.utils.data.DataLoader(
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
    validation_tokens_raster = full_hartleyify(
        (validation_images - mean) / std, args.patch
    )

    model = HartleyTileAR(
        width=args.width,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_mult=args.ff_mult,
        diff_width=args.diff_width,
        diff_depth=args.diff_depth,
        inference_steps=args.inference_steps,
        grid=32 // args.patch,
        token_dim=3 * args.patch * args.patch,
        rope_mode=args.rope_mode,
    ).to(device)
    validation_tokens = model.order_tokens(validation_tokens_raster)
    print(
        f"hartley AR: {model.seq_len} x {model.token_dim}; "
        f"mean={mean:.6f} std={std:.6f}; "
        f"params={sum(parameter.numel() for parameter in model.parameters()) / 1e6:.1f}M"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    def schedule(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        progress = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history = []
    progress = tqdm(total=args.steps, desc="hartley-ar")
    global_step = 0
    while global_step < args.steps:
        for images, _ in loader:
            if global_step >= args.steps:
                break
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                raster = full_hartleyify((images - mean) / std, args.patch)
                tokens = model.order_tokens(raster)
            model.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(tokens)
                loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0 or global_step == args.steps:
                progress.set_postfix(
                    loss=float(loss.detach()),
                    grad=float(grad_norm),
                    lr=scheduler.get_last_lr()[0],
                )
            record: Dict[str, float] = {"step": global_step, "loss": float(loss.detach())}
            if args.diagnostic_steps and global_step % args.diagnostic_steps == 0:
                model.eval()
                cpu_state = torch.random.get_rng_state()
                cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    clean = model(validation_tokens)["loss"]
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                shuffled_history = validation_tokens.roll(1, 0)[:, :-1]
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    shuffled = model(
                        validation_tokens, history_override=shuffled_history
                    )["loss"]
                record.update(
                    clean=float(clean),
                    shuffled=float(shuffled),
                    gap=float(shuffled - clean),
                )
                print(
                    f"DIAGNOSTIC step={global_step} clean={float(clean):.6f} "
                    f"shuffled={float(shuffled):.6f} gap={float(shuffled-clean):.6f}"
                )
            history.append(record)
            if args.preview_steps and global_step % args.preview_steps == 0:
                model.eval()
                generator = torch.Generator(device=device).manual_seed(12345)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    ordered = model.generate(16 if not args.smoke else 2, args.inference_steps, generator)
                raster = model.restore_raster(ordered.float())
                decoded = full_ihartleyify(raster, args.patch, 32) * std + mean
                save_image(
                    decoded.clamp(0, 1),
                    output_dir / f"samples_{global_step}.png",
                    nrow=2 if args.smoke else 4,
                )
            if args.checkpoint_steps and global_step % args.checkpoint_steps == 0:
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "args": vars(args),
                        "mean": mean,
                        "std": std,
                    },
                    output_dir / f"checkpoint_{global_step}.pt",
                )
    progress.close()
    (output_dir / "history.json").write_text(
        json.dumps({"mean": mean, "std": std, "history": history}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
