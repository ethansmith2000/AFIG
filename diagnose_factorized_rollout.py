"""Separate factorized-head sampling quality from autoregressive rollout error."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from frequency import FrequencyCodec
from model_continuous import ContinuousFFTDecoder
from train_continuous import build_model_config


def load_model(checkpoint: Path, device: torch.device) -> ContinuousFFTDecoder:
    payload = torch.load(checkpoint, map_location="cpu")
    saved_args = Namespace(**payload["args"])
    config = build_model_config(saved_args)
    codec = FrequencyCodec(config.codec)
    codec.load_exported(payload["codec"])
    model = ContinuousFFTDecoder(config, codec=codec)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    if model.factorized_decoder is None:
        raise ValueError("Checkpoint does not use decoder_geometry=factorized_polar")
    return model


@torch.no_grad()
def oracle_history_sample(
    model: ContinuousFFTDecoder,
    true_tokens: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    position_chunk: int,
) -> torch.Tensor:
    """Sample each token independently while giving it its true causal prefix."""
    batch, length, _ = true_tokens.shape
    hidden, _ = model.forward_backbone(
        model.embed_tokens(true_tokens[:, :-1], include_bos=True),
        use_cache=False,
    )
    outputs: List[torch.Tensor] = []
    decoder = model.factorized_decoder
    assert decoder is not None
    for start in range(0, length, position_chunk):
        stop = min(start + position_chunk, length)
        positions = torch.arange(start, stop, device=true_tokens.device)
        count = stop - start
        z = hidden[:, start:stop].reshape(batch * count, -1)
        slot = model.prediction_slot_condition(
            positions, batch_size=batch, dtype=hidden.dtype
        ).reshape(batch * count, -1)
        log_amp, phase = decoder.sample_coordinates(
            z=z,
            slot_condition=slot,
            generator=generator,
            steps=steps,
        )
        scale = model.factorized_amplitude_scale(positions)[None].expand(
            batch, -1, -1
        ).reshape(batch * count, 3)
        is_self = model.codec.is_self_conjugate[positions][None].expand(
            batch, -1
        ).reshape(batch * count)
        from factorized_polar_decoder import polar_to_cartesian

        raw = polar_to_cartesian(
            log_amp,
            phase,
            scale,
            is_self,
            model.config.factorized_polar.log_epsilon,
            decoder.amplitude_coordinate_mean,
            decoder.amplitude_coordinate_std,
        ).reshape(batch, count, 6)
        outputs.append(model.codec.raw_to_normalized_at(raw, positions))
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def rollout_after_true_prefix(
    model: ContinuousFFTDecoder,
    true_tokens: torch.Tensor,
    prefix_length: int,
    generator: torch.Generator,
    steps: int,
) -> torch.Tensor:
    """Consume a true prefix, then sample the remaining suffix autoregressively."""
    batch, length, _ = true_tokens.shape
    z, caches = model.init_cache(
        batch, true_tokens.device, model.token_proj.weight.dtype
    )
    outputs: List[torch.Tensor] = []
    for position in range(length):
        if position < prefix_length:
            token = true_tokens[:, position]
        else:
            token = model.sample_token(
                z=z,
                position=position,
                generator=generator,
                steps=steps,
                temperature=1.0,
                eta=0.0,
            )
        outputs.append(token)
        if position + 1 < length:
            z, caches = model.forward_step(token, position, caches)
    return torch.stack(outputs, dim=1)


def suffix_metrics(
    model: ContinuousFFTDecoder,
    predicted: torch.Tensor,
    target: torch.Tensor,
    start: int,
) -> Dict[str, float]:
    positions = torch.arange(start, target.shape[1], device=target.device)
    pred = predicted[:, start:]
    truth = target[:, start:]
    mask = model.codec.component_mask[positions][None]
    normalized_mse = (
        ((pred - truth).float().square() * mask).sum()
        / (mask.sum() * pred.shape[0]).clamp_min(1.0)
    ).item()
    pred_raw = model.codec.normalized_to_raw_at(pred, positions).float()
    truth_raw = model.codec.normalized_to_raw_at(truth, positions).float()
    physical_nrmse = (
        ((pred_raw - truth_raw).square() * mask).sum().sqrt()
        / (truth_raw.square() * mask).sum().sqrt().clamp_min(1e-8)
    ).item()
    pred_amp = torch.sqrt(pred_raw[..., :3].square() + pred_raw[..., 3:].square())
    true_amp = torch.sqrt(truth_raw[..., :3].square() + truth_raw[..., 3:].square())
    amp_ratio = (pred_amp.mean() / true_amp.mean().clamp_min(1e-8)).item()
    phase_cos = (
        pred_raw[..., :3] * truth_raw[..., :3]
        + pred_raw[..., 3:] * truth_raw[..., 3:]
    ) / (pred_amp * true_amp).clamp_min(1e-8)
    amp_weight = true_amp / true_amp.mean().clamp_min(1e-8)
    phase_coherence = (
        (phase_cos.clamp(-1, 1) * amp_weight).sum() / amp_weight.sum().clamp_min(1e-8)
    ).item()
    return {
        "normalized_mse": normalized_mse,
        "physical_complex_nrmse": physical_nrmse,
        "mean_amplitude_ratio": amp_ratio,
        "amplitude_weighted_phase_coherence": phase_coherence,
    }


def parse_prefixes(value: str) -> Iterable[int]:
    return (int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--position_chunk", type=int, default=64)
    parser.add_argument("--prefixes", default="32,128,256,384")
    parser.add_argument("--free_cutoffs", default="")
    parser.add_argument("--skip_oracle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset = CIFAR10(
        root=str(args.data_root), train=False, transform=ToTensor(), download=False
    )
    images = torch.stack([dataset[index][0] for index in range(args.num_images)]).to(device)
    true_tokens = model.codec.encode(images)
    outputs: Dict[str, torch.Tensor] = {"reference": true_tokens}
    metrics: Dict[str, Dict[str, float]] = {}

    generator = torch.Generator(device=device).manual_seed(args.seed)
    if not args.skip_oracle:
        oracle = oracle_history_sample(
            model, true_tokens, generator, args.steps, args.position_chunk
        )
        outputs["oracle_history"] = oracle
        metrics["oracle_history"] = suffix_metrics(model, oracle, true_tokens, 0)

    for cutoff in sorted(set(parse_prefixes(args.free_cutoffs))):
        if not 0 < cutoff <= model.codec.seq_len:
            raise ValueError(f"Invalid cutoff {cutoff}; sequence length is {model.codec.seq_len}")
        generator = torch.Generator(device=device).manual_seed(args.seed + cutoff)
        sampled = model.generate(
            batch_size=args.num_images,
            generator=generator,
            num_inference_steps=args.steps,
            return_tokens=True,
            max_tokens=cutoff,
        )["tokens"]
        outputs[f"free_cutoff_{cutoff}"] = sampled

    prefixes = sorted(set(parse_prefixes(args.prefixes)))
    for prefix in prefixes:
        if not 0 <= prefix < model.codec.seq_len:
            raise ValueError(f"Invalid prefix {prefix}; sequence length is {model.codec.seq_len}")
        prefix_only = torch.zeros_like(true_tokens)
        prefix_only[:, :prefix] = true_tokens[:, :prefix]
        outputs[f"prefix_only_{prefix}"] = prefix_only
        generator = torch.Generator(device=device).manual_seed(args.seed + prefix)
        sampled = rollout_after_true_prefix(
            model, true_tokens, prefix, generator, args.steps
        )
        name = f"true_prefix_{prefix}"
        outputs[name] = sampled
        metrics[name] = suffix_metrics(model, sampled, true_tokens, prefix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoded = [images.cpu()]
    row_names = ["reference"]
    for name, tokens in outputs.items():
        if name == "reference":
            continue
        decoded.append(model.codec.decode(tokens.float()).clamp(0, 1).cpu())
        row_names.append(name)
    grid = vutils.make_grid(torch.cat(decoded), nrow=args.num_images, padding=2)
    vutils.save_image(grid, args.output_dir / "oracle_and_prefix_grid.png")
    torch.save(
        {name: value.detach().cpu() for name, value in outputs.items()},
        args.output_dir / "tokens.pt",
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "rows": row_names,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
