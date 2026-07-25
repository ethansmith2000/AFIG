"""Train continuous AFIG (Fourier coefficient AR + diffusion loss).

Example:
  python train_continuous.py
  python train_continuous.py --smoke
  gpu-claim run --owner AFIG --job continuous-train --wait -- \\
    python train_continuous.py --output_dir runs/continuous
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from diffusion_decoder import DiffusionDecoderConfig
from frequency import FrequencyCodec, FrequencyCodecConfig
from model_continuous import (
    ContinuousFFTDecoder,
    ContinuousModelConfig,
    CorruptionConfig,
    FrequencyConditioningConfig,
    GenerationConfig,
    PolarHistoryConfig,
    TransformerConfig,
)

logger = get_logger(__name__)
_std_log = logging.getLogger("afig.train_continuous")


def _log_info(msg: str) -> None:
    try:
        logger.info(msg)
    except Exception:
        _std_log.info(msg)


def _log_warning(msg: str) -> None:
    try:
        logger.warning(msg)
    except Exception:
        _std_log.warning(msg)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continuous AFIG training")
    p.add_argument("--output_dir", type=str, default="continuous_runs")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_weight_decay", type=float, default=0.02)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--allow_tf32", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--tracker_project_name", type=str, default="afig-continuous")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--run_group", type=str, default=None)
    p.add_argument(
        "--run_tags",
        type=str,
        default="",
        help="Comma-separated W&B tags.",
    )
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="Save resumable checkpoints every N steps; 0 disables periodic checkpoints.",
    )
    p.add_argument(
        "--save_final_checkpoint",
        action="store_true",
        help="Save one resumable checkpoint at the end (disabled by default).",
    )
    p.add_argument("--checkpoints_total_limit", type=int, default=5)
    p.add_argument("--validation_steps", type=int, default=500)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument(
        "--final_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run checkpoint-free FID/KID and diagnostics after training.",
    )
    p.add_argument("--final_eval_samples", type=int, default=5000)
    p.add_argument("--final_eval_batch_size", type=int, default=32)
    p.add_argument("--final_eval_reference_samples", type=int, default=50000)
    p.add_argument("--reference_stats_path", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--codec_stats_path", type=str, default=None)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "cifar10", "huggingface_cifar", "synthetic"],
        help="Training data source. 'auto' tries torchvision CIFAR, then local HF arrows, then synthetic. "
        "Smoke defaults to synthetic unless overridden.",
    )

    # Model / objective
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--diff_width", type=int, default=512)
    p.add_argument("--diff_depth", type=int, default=6)
    p.add_argument("--objective", type=str, default="ddpm", choices=["ddpm", "flow"])
    p.add_argument(
        "--rescale_betas_zero_snr",
        action="store_true",
        help="Rescale DDPM betas so the terminal timestep has exactly zero SNR.",
    )
    p.add_argument(
        "--timestep_spacing",
        type=str,
        default="leading",
        choices=["leading", "trailing", "linspace"],
        help="Diffusers inference timestep spacing for DDPM/DDIM sampling.",
    )
    p.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "v_prediction", "x0"],
    )
    p.add_argument("--loss_space", type=str, default="native", choices=["native", "v"])
    p.add_argument(
        "--loss_weighting",
        type=str,
        default="none",
        choices=["none", "min_snr", "logit_normal"],
    )
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--logit_normal_mean", type=float, default=0.0)
    p.add_argument("--logit_normal_std", type=float, default=1.0)
    p.add_argument("--flow_t_eps", type=float, default=0.05)
    p.add_argument("--flow_solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument(
        "--radial_power_weighting",
        action="store_true",
        help="Multiply per-token MSE by normalized tempered radial-power weights.",
    )
    p.add_argument(
        "--radial_power_exponent",
        type=float,
        default=0.5,
        help="Exponent on expected radial power; 0.5 weights by expected amplitude.",
    )
    p.add_argument(
        "--loss_metric",
        type=str,
        default="normalized",
        choices=["normalized", "orbit_covariance_power"],
    )
    p.add_argument("--orbit_covariance_exponent", type=float, default=0.0)
    p.add_argument("--diffusion_batch_mul", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--ordering", type=str, default="radial", choices=["radial", "square_spiral"])
    p.add_argument("--value_transform", type=str, default="identity", choices=["identity", "asinh"])
    p.add_argument(
        "--normalization",
        type=str,
        default="radial_whiten",
        choices=["radial_whiten", "radial_standardize", "orbit_whiten"],
    )
    p.add_argument("--history_corruption", type=str, default="none", choices=["none", "gaussian"])
    p.add_argument("--history_corruption_prob", type=float, default=1.0)
    p.add_argument("--history_noise_min", type=float, default=0.0)
    p.add_argument("--history_noise_max", type=float, default=0.05)
    p.add_argument("--history_noise_ramp_fraction", type=float, default=0.2)
    p.add_argument(
        "--history_polar_features",
        type=str,
        default="none",
        choices=["none", "log_amp_gated_phase"],
        help="Deterministic polar features fused into history embeddings (Cartesian diffusion targets unchanged).",
    )
    p.add_argument(
        "--frequency_conditioning",
        action="store_true",
        help=(
            "Use functional Fourier-coordinate features, target-frequency "
            "diffusion conditioning, and zero-initialized Transformer FiLM."
        ),
    )
    p.add_argument(
        "--position_num_frequencies",
        type=int,
        default=4,
        help="Number of log-spaced sinusoidal bands per kx/ky/radius coordinate.",
    )
    p.add_argument(
        "--position_max_frequency",
        type=float,
        default=8.0,
        help="Highest sinusoidal frequency for normalized Fourier coordinates.",
    )
    p.add_argument(
        "--position-input-addition",
        dest="position_input_addition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the shared frequency representation to input token embeddings.",
    )
    p.add_argument(
        "--position-rms-normalize",
        dest="position_rms_normalize",
        action="store_true",
        help="RMS-normalize frequency representations before conditioning.",
    )
    p.add_argument(
        "--transformer-position-film",
        dest="transformer_position_film",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use zero-initialized target-position FiLM in Transformer blocks.",
    )
    p.add_argument(
        "--diffusion-target-conditioning",
        dest="diffusion_target_conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition diffusion AdaLN directly on the known target frequency.",
    )
    p.add_argument("--use_ema", action="store_true", help="Opt in to EMA evaluation.")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--preset", type=str, default="moderate", choices=["tiny", "moderate", "legacy"])
    p.add_argument("--smoke", action="store_true", help="CPU/tiny one-step smoke train + generate")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Disable tracking/evaluation and report throughput plus peak CUDA memory.",
    )
    return p.parse_args(argv)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.benchmark:
        args.report_to = "none"
        args.final_eval = False
        args.validation_steps = max(args.validation_steps, 10**9)
        args.checkpointing_steps = 0
        args.save_final_checkpoint = False
    if args.smoke:
        args.preset = "tiny"
        args.mixed_precision = "no"
        if args.report_to in ("wandb",):
            args.report_to = "none"
        args.train_batch_size = 2
        args.max_train_steps = 1
        args.num_train_epochs = 1
        args.validation_steps = 1
        args.dataloader_num_workers = 0
        args.diffusion_batch_mul = 1
        args.num_inference_steps = 2
        args.num_validation_images = 2
        args.lr_warmup_steps = 0
        args.output_dir = args.output_dir or "continuous_smoke"
        args.final_eval = False
    if args.preset == "tiny":
        args.width = 64
        args.num_layers = 2
        args.num_heads = 4
        args.ff_mult = 2
        args.diff_width = 64
        args.diff_depth = 2
    elif args.preset == "legacy":
        args.width = 1024
        args.num_layers = 12
        args.num_heads = 16
        args.ff_mult = 3
        args.diff_width = 1024
        args.diff_depth = 3
    # moderate keeps CLI defaults / width 512
    return args


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, v in self.shadow.items():
            if k in state and state[k].shape == v.shape:
                state[k].copy_(v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


def build_model_config(args: argparse.Namespace) -> ContinuousModelConfig:
    return ContinuousModelConfig(
        codec=FrequencyCodecConfig(
            ordering=args.ordering,
            value_transform=args.value_transform,
            normalization=args.normalization,
        ),
        transformer=TransformerConfig(
            width=args.width,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            gradient_checkpointing=args.gradient_checkpointing,
        ),
        diffusion=DiffusionDecoderConfig(
            target_dim=6,
            z_channels=args.width,
            width=args.diff_width,
            depth=args.diff_depth,
            objective=args.objective,
            rescale_betas_zero_snr=args.rescale_betas_zero_snr,
            timestep_spacing=args.timestep_spacing,
            prediction_type=args.prediction_type,
            loss_space=args.loss_space,
            loss_weighting=args.loss_weighting,
            min_snr_gamma=args.min_snr_gamma,
            logit_normal_mean=args.logit_normal_mean,
            logit_normal_std=args.logit_normal_std,
            flow_t_eps=args.flow_t_eps,
            flow_solver=args.flow_solver,
            radial_power_weighting=bool(getattr(args, "radial_power_weighting", False)),
            radial_power_exponent=args.radial_power_exponent,
            loss_metric=args.loss_metric,
            orbit_covariance_exponent=args.orbit_covariance_exponent,
            diffusion_batch_mul=args.diffusion_batch_mul,
            num_inference_steps=args.num_inference_steps,
        ),
        corruption=CorruptionConfig(
            history_corruption=args.history_corruption,
            history_corruption_prob=args.history_corruption_prob,
            history_noise_min=args.history_noise_min,
            history_noise_max=args.history_noise_max,
            history_noise_ramp_fraction=args.history_noise_ramp_fraction,
        ),
        polar_history=PolarHistoryConfig(
            enabled=getattr(args, "history_polar_features", "none") != "none",
            mode=(
                "log_amp_gated_phase"
                if getattr(args, "history_polar_features", "none") == "none"
                else args.history_polar_features
            ),
        ),
        frequency_conditioning=FrequencyConditioningConfig(
            enabled=bool(getattr(args, "frequency_conditioning", False)),
            num_frequencies=args.position_num_frequencies,
            max_frequency=args.position_max_frequency,
            input_addition=bool(getattr(args, "position_input_addition", True)),
            rms_normalize=bool(getattr(args, "position_rms_normalize", False)),
            transformer_film=bool(
                getattr(args, "transformer_position_film", True)
            ),
            diffusion_target_conditioning=bool(
                getattr(args, "diffusion_target_conditioning", True)
            ),
        ),
        generation=GenerationConfig(
            num_inference_steps=args.num_inference_steps,
            eta=0.0,
            temperature=1.0,
            cfg_enabled=False,
            grouping="coefficient",
        ),
    )


def _synthetic_dataset(n: int = 64):
    class _Synth(torch.utils.data.Dataset):
        def __init__(self, n=64):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            g = torch.Generator().manual_seed(idx)
            img = torch.rand(3, 32, 32, generator=g)
            return img, 0

    return _Synth(n=n)


def _hf_cifar_paths() -> list:
    """Known local HuggingFace arrow caches for CIFAR-10 on this machine."""
    candidates = [
        "/workspace/SNRAdam/data/huggingface/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow",
        str(Path.home() / ".cache/huggingface/datasets"),
    ]
    # Also search AFIG/data and workspace HF home.
    extra_roots = [
        Path("/workspace/AFIG/data"),
        Path("/workspace/.hf_home/hub"),
        Path("/workspace/SNRAdam/data/huggingface"),
    ]
    found = []
    for c in candidates:
        if c.endswith(".arrow") and os.path.isfile(c):
            found.append(c)
    for root in extra_roots:
        if root.exists():
            for p in root.rglob("cifar10-train.arrow"):
                found.append(str(p))
    # de-dupe preserve order
    out = []
    seen = set()
    for p in found:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _dataset_from_hf_arrow(arrow_path: str, transform=None):
    from datasets import Dataset
    import numpy as np
    from PIL import Image

    ds = Dataset.from_file(arrow_path)

    class _HFCifar(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, idx):
            ex = ds[int(idx)]
            img = ex["img"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            if transform is not None:
                img = transform(img)
            else:
                arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
                img = arr
            label = int(ex["label"])
            return img, label

    return _HFCifar()


def make_dataloader(args: argparse.Namespace):
    """Build train loader.

    Resolution order:
      1. Explicit --dataset synthetic / smoke default
      2. torchvision CIFAR-10 if local files exist (or download succeeds)
      3. Local HuggingFace CIFAR arrow cache (common on this host)
      4. Synthetic fallback with a warning
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.CenterCrop(32),
        ]
    )

    dataset_choice = getattr(args, "dataset", "auto")
    if args.smoke and dataset_choice == "auto":
        # Fast unit/smoke path; use --dataset cifar10 to force real data.
        dataset_choice = "synthetic"

    if dataset_choice == "synthetic" or getattr(args, "synthetic_data", False):
        dataset = _synthetic_dataset(n=64 if args.smoke else 1024)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        return dataset, loader

    def _torchvision_cifar(download: bool):
        return torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=download, transform=transform
        )

    dataset = None
    batches_dir = os.path.join(args.data_root, "cifar-10-batches-py")

    # 1) Already-extracted torchvision layout.
    if dataset_choice in ("auto", "cifar10") and os.path.isdir(batches_dir):
        dataset = _torchvision_cifar(download=False)
        _log_info(f"Using local torchvision CIFAR-10 at {batches_dir}")

    # 2) Local HuggingFace arrow cache (fast on this host; avoids slow Toronto mirror).
    if dataset is None and dataset_choice in ("auto", "huggingface_cifar"):
        for arrow in _hf_cifar_paths():
            try:
                dataset = _dataset_from_hf_arrow(arrow, transform=transform)
                _log_info(f"Using HuggingFace CIFAR arrow cache: {arrow}")
                break
            except Exception as e:
                _log_warning(f"Failed HF arrow {arrow}: {e}")

    # 3) Force / attempt torchvision download when explicitly requested, or as auto last network try.
    if dataset is None and dataset_choice == "cifar10":
        try:
            _log_info("Downloading CIFAR-10 via torchvision...")
            dataset = _torchvision_cifar(download=True)
        except Exception as e:
            _log_warning(f"torchvision CIFAR download failed: {e}")
            dataset = None
        # Fall back to HF arrows if download fails.
        if dataset is None:
            for arrow in _hf_cifar_paths():
                try:
                    dataset = _dataset_from_hf_arrow(arrow, transform=transform)
                    _log_info(f"CIFAR download failed; using HF arrow cache: {arrow}")
                    break
                except Exception as e:
                    _log_warning(f"Failed HF arrow {arrow}: {e}")
    elif dataset is None and dataset_choice == "auto":
        # Only attempt network download if no local HF cache was found.
        try:
            _log_info("Attempting torchvision CIFAR-10 download...")
            dataset = _torchvision_cifar(download=True)
            _log_info("Loaded CIFAR-10 via torchvision")
        except Exception as e:
            _log_warning(f"torchvision CIFAR unavailable: {e}")
            dataset = None

    # 4) For huggingface_cifar with no cache, also try local torchvision batches.
    if dataset is None and dataset_choice == "huggingface_cifar":
        if os.path.isdir(batches_dir):
            dataset = _torchvision_cifar(download=False)
        else:
            _log_warning("No HF CIFAR arrow cache found.")

    # 5) Synthetic fallback.
    if dataset is None:
        _log_warning(
            "Falling back to synthetic 32x32 images. "
            "Provide CIFAR under --data_root or a local HF arrow cache, "
            "or pass --dataset synthetic explicitly."
        )
        dataset = _synthetic_dataset(n=1024)
        use_workers = 0
    else:
        use_workers = args.dataloader_num_workers

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=use_workers,
        drop_last=True,
    )
    return dataset, loader


def fit_or_load_codec(
    args: argparse.Namespace,
    accelerator: Accelerator,
    train_loader,
    config: ContinuousModelConfig,
) -> FrequencyCodec:
    codec = FrequencyCodec(config.codec)
    stats_path = args.codec_stats_path or os.path.join(args.output_dir, "codec_stats.pt")

    if args.resume_from_checkpoint:
        # Codec comes from checkpoint later; still need a fitted object for construction.
        if os.path.isfile(stats_path):
            payload = torch.load(stats_path, map_location="cpu")
            codec.load_exported(payload)
            return codec

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.isfile(stats_path):
            logger.info(f"Loading codec stats from {stats_path}")
            payload = torch.load(stats_path, map_location="cpu")
            codec.load_exported(payload)
        else:
            logger.info("Fitting codec statistics on CIFAR-10 train split...")
            # Use a dedicated loader so we don't exhaust the training iterator.
            fit_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=args.train_batch_size,
                shuffle=False,
                num_workers=args.dataloader_num_workers,
            )
            max_batches = 8 if args.smoke else None
            codec.fit_from_loader(fit_loader, max_batches=max_batches, device=torch.device("cpu"))
            torch.save(codec.export_state(), stats_path)
            logger.info(f"Wrote codec stats to {stats_path}")
    accelerator.wait_for_everyone()
    if not bool(codec.is_fitted.item()):
        payload = torch.load(stats_path, map_location="cpu")
        codec.load_exported(payload)
    return codec


def save_checkpoint(
    path: str,
    model: ContinuousFFTDecoder,
    optimizer,
    lr_scheduler,
    ema: Optional[ModelEMA],
    args: argparse.Namespace,
    global_step: int,
    config: ContinuousModelConfig,
) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    payload = {
        "version": 1,
        "global_step": global_step,
        "args": vars(args),
        "config": config.fingerprint(),
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "codec": unwrapped.codec.export_state(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: ContinuousFFTDecoder,
    optimizer=None,
    lr_scheduler=None,
    ema: Optional[ModelEMA] = None,
) -> int:
    payload = torch.load(path, map_location="cpu")
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.codec.load_exported(payload["codec"])
    unwrapped.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None and payload.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return int(payload["global_step"])


@torch.no_grad()
def validate(
    model: ContinuousFFTDecoder,
    args: argparse.Namespace,
    accelerator: Accelerator,
    global_step: int,
    ema: Optional[ModelEMA] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    unwrapped = accelerator.unwrap_model(model)
    backup = None
    if ema is not None:
        backup = {
            k: v.detach().clone()
            for k, v in unwrapped.state_dict().items()
            if k in ema.shadow
        }
        ema.copy_to(unwrapped)

    unwrapped.eval()
    out = unwrapped.generate(
        batch_size=args.num_validation_images,
        num_inference_steps=args.num_inference_steps,
        return_tokens=True,
        progress=False,
        max_tokens=max_tokens,
    )
    images = out["images"]
    tokens = out["tokens"]
    # Reconstruct spectrum via public decode path for diagnostics.
    denorm = unwrapped.codec.denormalize(tokens.float())
    raw = unwrapped.codec.invert_value_transform(denorm)
    raw = raw.clone()
    raw[..., 3:] = raw[..., 3:] * (~unwrapped.codec.is_self_conjugate).to(raw.dtype)[None, :, None]
    spectrum = unwrapped.codec.tokens_to_spectrum(raw)
    herm = unwrapped.codec.hermitian_violation(spectrum).item()
    imag_energy = spectrum.imag.abs().mean().item()

    images_clip = images.clamp(0, 1)

    logs = {
        "val/hermitian_violation": herm,
        "val/imag_energy": imag_energy,
        "val/backbone_seconds": out["backbone_seconds"],
        "val/denoise_seconds": out["denoise_seconds"],
    }
    if accelerator.is_main_process:
        try:
            import torchvision.utils as vutils

            grid = vutils.make_grid(images_clip.cpu(), nrow=2)
            os.makedirs(args.output_dir, exist_ok=True)
            vutils.save_image(grid, os.path.join(args.output_dir, f"samples_{global_step}.png"))
        except Exception as e:
            logger.warning(f"Failed to save sample grid: {e}")
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                import wandb

                tracker.log(
                    {
                        "val/samples": [
                            wandb.Image(images_clip[i].cpu()) for i in range(images_clip.shape[0])
                        ],
                        **logs,
                    },
                    step=global_step,
                )

    if backup is not None:
        cur = unwrapped.state_dict()
        for k, v in backup.items():
            cur[k].copy_(v)

    unwrapped.train()
    return logs


def bucket_timestep_loss(timesteps: torch.Tensor, per_example: torch.Tensor, num_buckets: int = 5) -> Dict[str, float]:
    t = timesteps.float()
    t_max = float(t.max().clamp_min(1.0))
    out = {}
    for i in range(num_buckets):
        lo = i / num_buckets * t_max
        hi = (i + 1) / num_buckets * t_max
        sel = (t >= lo) & (t < hi if i < num_buckets - 1 else t <= hi)
        if bool(sel.any()):
            out[f"loss/t_bucket_{i}"] = per_example[sel].mean().item()
    return out


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_args()
    args = apply_preset(args)

    logging_dir = Path(args.output_dir, args.logging_dir)
    log_with = None if args.report_to in (None, "", "none") else args.report_to
    accelerator = Accelerator(
        cpu=args.smoke,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else "no",
        log_with=log_with,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset, train_loader = make_dataloader(args)
    config = build_model_config(args)
    codec = fit_or_load_codec(args, accelerator, train_loader, config)
    if accelerator.is_main_process and args.normalization != "orbit_whiten":
        rw = codec.radial_loss_weights(exponent=args.radial_power_exponent)
        _log_info(
            f"Radial loss weights: mean={rw.mean().item():.4f} "
            f"min={rw.min().item():.4f} max={rw.max().item():.4f} "
            f"exponent={args.radial_power_exponent:.3f} "
            f"(radial_power_weighting={bool(args.radial_power_weighting)})"
        )
        _log_info(
            f"Generative objective: objective={args.objective} "
            f"prediction={args.prediction_type} loss_space={args.loss_space} "
            f"weighting={args.loss_weighting} min_snr_gamma={args.min_snr_gamma:g} "
            f"logit_normal=({args.logit_normal_mean:g}, {args.logit_normal_std:g}) "
            f"zero_terminal_snr={args.rescale_betas_zero_snr} "
            f"timestep_spacing={args.timestep_spacing} "
            f"flow_solver={args.flow_solver}"
        )
        _log_info(
            f"Polar history features: {args.history_polar_features} "
            f"(enabled={args.history_polar_features != 'none'})"
        )
        _log_info(
            f"Frequency conditioning: enabled={bool(args.frequency_conditioning)} "
            f"bands={args.position_num_frequencies} "
            f"max_frequency={args.position_max_frequency:g} "
            f"input_addition={bool(args.position_input_addition)} "
            f"rms_normalize={bool(args.position_rms_normalize)} "
            f"transformer_film={bool(args.transformer_position_film)} "
            f"diffusion_target={bool(args.diffusion_target_conditioning)}"
        )
    model = ContinuousFFTDecoder(config, codec=codec)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    if overrode_max:
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ema = (
        ModelEMA(accelerator.unwrap_model(model), decay=args.ema_decay)
        if args.use_ema
        else None
    )

    if accelerator.is_main_process and log_with is not None:
        metric_suffix = (
            f"-a{args.orbit_covariance_exponent:g}"
            if args.loss_metric == "orbit_covariance_power"
            else (
                f"-radial{args.radial_power_exponent:g}"
                if args.radial_power_weighting
                else ""
            )
        )
        run_name = args.run_name or (
            f"{args.objective}-{args.prediction_type}-{args.normalization}-"
            f"{args.loss_metric}{metric_suffix}-d{args.diff_depth}-b{args.train_batch_size}"
        )
        init_kwargs = {}
        if log_with == "wandb":
            wandb_kwargs: Dict[str, Any] = {"name": run_name}
            if args.run_group:
                wandb_kwargs["group"] = args.run_group
            tags = [tag.strip() for tag in args.run_tags.split(",") if tag.strip()]
            if tags:
                wandb_kwargs["tags"] = tags
            init_kwargs["wandb"] = wandb_kwargs
        accelerator.init_trackers(
            args.tracker_project_name,
            config=vars(args),
            init_kwargs=init_kwargs,
        )

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            cands = sorted(
                [p for p in os.listdir(args.output_dir) if p.startswith("checkpoint_") and p.endswith(".pt")],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            path = os.path.join(args.output_dir, cands[-1]) if cands else None
        if path is not None:
            global_step = load_checkpoint(path, accelerator.unwrap_model(model), optimizer, lr_scheduler, ema)
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed from {path} at step {global_step}")

    progress = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    benchmark_start = None
    benchmark_start_step = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                images = batch[0]
                # Encode under no_grad; tokens are continuous targets.
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)
                    tokens = unwrapped.codec.encode(images)
                out = model(
                    tokens,
                    corrupt=True,
                    training_progress=min(global_step / max(args.max_train_steps, 1), 1.0),
                )
                loss = out["loss"]
                accelerator.backward(loss)
                grad_norm = 0.0
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if ema is not None:
                    ema.update(accelerator.unwrap_model(model))
                progress.update(1)
                global_step += 1
                if args.benchmark and global_step == 1:
                    accelerator.wait_for_everyone()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    benchmark_start = time.perf_counter()
                    benchmark_start_step = global_step

                logs = {
                    "loss": loss.detach().item(),
                    "unweighted_mse": out["unweighted_mse"].item(),
                    "weighted_loss": out["weighted_loss"].item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "corruption_strength": out["corruption_strength"].mean().item(),
                }
                if "covariance_metric_loss" in out:
                    logs["covariance_metric_loss"] = out[
                        "covariance_metric_loss"
                    ].item()
                if "radial_weighted_mse" in out:
                    logs["radial_weighted_mse"] = out["radial_weighted_mse"].item()
                if "radial_weights" in out:
                    logs["radial_weight_mean"] = out["radial_weights"].float().mean().item()
                # Timestep buckets
                logs.update(bucket_timestep_loss(out["timesteps"], out["per_example"]))
                # Radius-bin loss (cheap aggregate)
                if "radius_bin" in out:
                    rb = out["radius_bin"]
                    pe = out["per_example"]
                    rw = out.get("radial_weights")
                    for b in (0, 1, 5, 10, 20):
                        sel = rb == b
                        if bool(sel.any()):
                            logs[f"loss/radius_bin_{b}"] = pe[sel].mean().item()
                            if rw is not None:
                                logs[f"loss/radius_bin_{b}_weighted"] = (
                                    (pe[sel] * rw[sel]).mean().item()
                                )

                progress.set_postfix(**{k: logs[k] for k in ("loss", "lr", "grad_norm")})
                accelerator.log(logs, step=global_step)

                if (
                    accelerator.is_main_process
                    and args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    ckpt = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                    save_checkpoint(
                        ckpt,
                        accelerator.unwrap_model(model),
                        optimizer,
                        lr_scheduler,
                        ema,
                        args,
                        global_step,
                        config,
                    )
                    # Enforce total limit on checkpoint_*.pt files.
                    if args.checkpoints_total_limit is not None:
                        cands = sorted(
                            [p for p in os.listdir(args.output_dir) if p.startswith("checkpoint_") and p.endswith(".pt")],
                            key=lambda x: int(x.split("_")[1].split(".")[0]),
                        )
                        while len(cands) > args.checkpoints_total_limit:
                            os.remove(os.path.join(args.output_dir, cands.pop(0)))

                if global_step % args.validation_steps == 0:
                    max_tokens = 8 if args.smoke else None
                    val_logs = validate(
                        model, args, accelerator, global_step, ema=ema, max_tokens=max_tokens
                    )
                    accelerator.log(val_logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if args.benchmark and accelerator.is_main_process:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - (benchmark_start or time.perf_counter()), 1e-9)
        measured_steps = max(global_step - benchmark_start_step, 0)
        peak_allocated = (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        _log_info(
            "BENCHMARK "
            f"batch={args.train_batch_size} steps={measured_steps} "
            f"steps_per_sec={measured_steps / elapsed:.4f} "
            f"examples_per_sec={measured_steps * args.train_batch_size / elapsed:.2f} "
            f"peak_allocated_gib={peak_allocated:.3f} "
            f"peak_reserved_gib={peak_reserved:.3f}"
        )
    if accelerator.is_main_process and args.save_final_checkpoint:
        final = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
        save_checkpoint(
            final,
            accelerator.unwrap_model(model),
            optimizer,
            lr_scheduler,
            ema,
            args,
            global_step,
            config,
        )
        logger.info(f"Saved final checkpoint to {final}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.final_eval:
        from live_evaluation import evaluate_live

        reference_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.final_eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
        )
        reference_stats_path = args.reference_stats_path or os.path.join(
            args.data_root,
            f"cifar10_inception_reference_{args.ordering}.pt",
        )
        final_metrics = evaluate_live(
            accelerator.unwrap_model(model),
            reference_loader=reference_loader,
            num_samples=args.final_eval_samples,
            batch_size=args.final_eval_batch_size,
            reference_cache_path=reference_stats_path,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            reference_samples=args.final_eval_reference_samples,
        )
        accelerator.log(final_metrics, step=global_step)
        _log_info(f"Final live metrics: {final_metrics}")
    accelerator.wait_for_everyone()
    accelerator.end_training()


def run_smoke_generate_only(output_dir: str) -> None:
    """Helper used by docs: load codec stats and run a 2-sample tiny generate."""
    from model_continuous import ContinuousFFTDecoder, ContinuousModelConfig, TransformerConfig
    from diffusion_decoder import DiffusionDecoderConfig
    from frequency import FrequencyCodecConfig

    stats = torch.load(os.path.join(output_dir, "codec_stats.pt"), map_location="cpu")
    codec = FrequencyCodec(FrequencyCodecConfig(**stats["config"]))
    codec.load_exported(stats)
    cfg = ContinuousModelConfig(
        codec=FrequencyCodecConfig(**stats["config"]),
        transformer=TransformerConfig(width=64, num_layers=2, num_heads=4, ff_mult=2),
        diffusion=DiffusionDecoderConfig(
            z_channels=64, width=64, depth=2, num_inference_steps=2, diffusion_batch_mul=1
        ),
    )
    model = ContinuousFFTDecoder(cfg, codec=codec).eval()
    out = model.generate(batch_size=2, num_inference_steps=2)
    assert out["images"].shape[0] == 2


if __name__ == "__main__":
    main()
