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
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from diffusion_decoder import DiffusionDecoderConfig
from frequency import FrequencyCodec, FrequencyCodecConfig
from model_continuous import (
    ContinuousFFTDecoder,
    ContinuousModelConfig,
    CorruptionConfig,
    GenerationConfig,
    HistoryFeatureConfig,
    PolarHistoryConfig,
    TransformerConfig,
)
from spectral_diagnostics import (
    compute_normalization_phase_distortion,
    compute_perturbation_diagnostics,
    compute_spectral_diagnostics,
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
    p.add_argument(
        "--grad_norm_mode",
        type=str,
        default="clip",
        choices=["off", "track", "clip"],
        help="Disable gradient norms, track without clipping, or clip and track.",
    )
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
        "--logging_steps",
        type=int,
        default=25,
        help="Log routine training metrics every N optimizer steps.",
    )
    p.add_argument(
        "--timing_steps",
        type=int,
        default=100,
        help="Measure one training step with CUDA events every N optimizer steps; 0 disables.",
    )
    p.add_argument(
        "--timestep_histogram_bins",
        type=int,
        default=20,
        help="Number of EMA timestep-loss bins; 0 disables the diagnostic.",
    )
    p.add_argument(
        "--timestep_histogram_decay",
        type=float,
        default=0.98,
        help="Per-update EMA decay for timestep diagnostics.",
    )
    p.add_argument(
        "--timestep_histogram_log_steps",
        type=int,
        default=100,
        help="Log timestep EMA bins every N optimizer steps.",
    )
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
    p.add_argument(
        "--preview_steps",
        type=int,
        default=5000,
        help="Generate and log a small image preview every N optimizer steps; 0 disables.",
    )
    p.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Deprecated alias for --preview_steps.",
    )
    p.add_argument(
        "--preview_seed",
        type=int,
        default=12345,
        help="Fixed sampling seed used for comparable preview grids.",
    )
    p.add_argument(
        "--condition_diagnostic_steps",
        type=int,
        default=500,
        help="Compare clean and batch-shuffled AR history; 0 disables.",
    )
    p.add_argument(
        "--spectral_diagnostic_steps",
        type=int,
        default=1000,
        help="Evaluate the fixed held-out spectral panel every N steps; 0 disables.",
    )
    p.add_argument("--spectral_panel_size", type=int, default=16)
    p.add_argument("--spectral_diagnostic_seed", type=int, default=1729)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument(
        "--final_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt into checkpoint-free FID/KID and diagnostics after training.",
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
    p.add_argument(
        "--qk_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply learned per-head RMSNorm to queries and keys before attention.",
    )
    p.add_argument(
        "--attention_rope",
        choices=["none", "sequence", "frequency_2d"],
        default="frequency_2d",
        help=(
            "Rotary attention geometry: sequence index or signed 2D Fourier "
            "coordinates. Absolute frequency identity remains enabled separately."
        ),
    )
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument(
        "--transformer-position-film",
        dest="transformer_position_film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally modulate each Transformer block with the same learned "
            "prediction-slot embedding; disabled by default."
        ),
    )
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
        "--component_reduction",
        choices=["active_mean", "fixed_dim"],
        default="active_mean",
        help="Normalize masked loss by active coordinates or by the fixed token dimension.",
    )
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
        "--snr_scale",
        type=float,
        default=1.0,
        help="Multiply bridge SNR by this factor inside the token diffusion decoder.",
    )
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
        choices=["normalized", "orbit_covariance_power", "orbit_scale_power"],
    )
    p.add_argument("--orbit_covariance_exponent", type=float, default=0.0)
    p.add_argument("--orbit_scale_exponent", type=float, default=0.0)
    p.add_argument("--phase_aux_weight", type=float, default=0.0)
    p.add_argument("--phase_aux_gate", type=float, default=0.1)
    p.add_argument(
        "--phase_gradient_diagnostic_step",
        type=int,
        default=1,
        help="One-based optimizer step for a phase/base output-gradient ratio; 0 disables.",
    )
    p.add_argument("--diffusion_batch_mul", type=int, default=4)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--ordering", type=str, default="radial", choices=["radial", "square_spiral"])
    p.add_argument("--value_transform", type=str, default="identity", choices=["identity", "asinh"])
    p.add_argument(
        "--normalization",
        type=str,
        default="radial_whiten",
        choices=[
            "radial_whiten",
            "radial_standardize",
            "orbit_whiten",
            "orbit_standardize",
            "global_ecs",
        ],
    )
    p.add_argument(
        "--coordinate_packing",
        choices=["legacy", "isometric"],
        default="legacy",
        help="Real packing of Hermitian FFT coefficients. isometric applies the "
        "sqrt(2) factors needed for exact Euclidean/noise equivalence.",
    )
    p.add_argument(
        "--ecs_percentile",
        type=float,
        default=98.25,
        help="Two-sided robust DC percentile used by global_ecs.",
    )
    p.add_argument(
        "--learned_output_gain",
        action="store_true",
        help="Learn per-orbit RGB log gains shared across real/imaginary outputs.",
    )
    p.add_argument(
        "--centering",
        type=str,
        default="all",
        choices=["all", "self_conjugate_std", "self_conjugate_rms"],
        help="Per-orbit complex centering and scaling policy.",
    )
    p.add_argument(
        "--diffusion_mean_policy",
        default="legacy",
        choices=["legacy", "per_orbit", "pooled_ordinary", "self_only"],
    )
    p.add_argument(
        "--diffusion_scale_policy",
        default="legacy",
        choices=["legacy", "centered_std", "uncentered_rms"],
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
        "--history_cartesian_features",
        type=str,
        default="centered",
        choices=["centered", "phase_preserving", "policy"],
        help="Cartesian coordinate family used only for completed AR history.",
    )
    p.add_argument(
        "--history_mean_policy",
        default="legacy",
        choices=["legacy", "per_orbit", "pooled_ordinary", "self_only"],
    )
    p.add_argument(
        "--history_scale_policy",
        default="legacy",
        choices=["legacy", "centered_std", "uncentered_rms"],
    )
    p.add_argument(
        "--input_timestep_conditioning",
        type=str,
        default="none",
        choices=["none", "film"],
        help="Timestep modulation immediately after the diffusion input projection.",
    )
    p.add_argument(
        "--input_stem_time_film",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clear alias for optional timestep FiLM at the diffusion input stem.",
    )
    p.add_argument(
        "--input_projection_init",
        type=str,
        default="xavier",
        choices=["xavier", "kaiming_linear"],
        help="Initializer for the unbalanced 6-to-width diffusion input projection.",
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
    args = p.parse_args(argv)
    if args.validation_steps is not None:
        args.preview_steps = args.validation_steps
    if args.logging_steps <= 0:
        p.error("--logging_steps must be positive")
    if args.timing_steps < 0:
        p.error("--timing_steps must be non-negative")
    return args


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.benchmark:
        args.report_to = "none"
        args.final_eval = False
        args.preview_steps = max(args.preview_steps, 10**9)
        args.checkpointing_steps = 0
        args.save_final_checkpoint = False
        args.logging_steps = max(args.logging_steps, 10**9)
        args.timing_steps = 0
    if args.smoke:
        args.preset = "tiny"
        args.mixed_precision = "no"
        if args.report_to in ("wandb",):
            args.report_to = "none"
        args.train_batch_size = 2
        args.max_train_steps = 1
        args.num_train_epochs = 1
        args.preview_steps = 1
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
    input_timestep_conditioning = args.input_timestep_conditioning
    if args.input_stem_time_film is not None:
        input_timestep_conditioning = (
            "film" if args.input_stem_time_film else "none"
        )
    return ContinuousModelConfig(
        codec=FrequencyCodecConfig(
            ordering=args.ordering,
            value_transform=args.value_transform,
            normalization=args.normalization,
            centering=args.centering,
            mean_policy=args.diffusion_mean_policy,
            scale_policy=args.diffusion_scale_policy,
            coordinate_packing=args.coordinate_packing,
            ecs_percentile=args.ecs_percentile,
        ),
        transformer=TransformerConfig(
            width=args.width,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            gradient_checkpointing=args.gradient_checkpointing,
            qk_norm=bool(getattr(args, "qk_norm", True)),
            attention_rope=getattr(args, "attention_rope", "frequency_2d"),
            rope_base=float(getattr(args, "rope_base", 10000.0)),
            position_film=bool(getattr(args, "transformer_position_film", False)),
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
            snr_scale=args.snr_scale,
            radial_power_weighting=bool(getattr(args, "radial_power_weighting", False)),
            radial_power_exponent=args.radial_power_exponent,
            loss_metric=args.loss_metric,
            component_reduction=args.component_reduction,
            orbit_covariance_exponent=args.orbit_covariance_exponent,
            orbit_scale_exponent=args.orbit_scale_exponent,
            learned_output_gain=args.learned_output_gain,
            phase_aux_weight=args.phase_aux_weight,
            phase_aux_gate=args.phase_aux_gate,
            input_timestep_conditioning=input_timestep_conditioning,
            input_projection_init=args.input_projection_init,
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
        history_features=HistoryFeatureConfig(
            cartesian_mode=args.history_cartesian_features,
            mean_policy=args.history_mean_policy,
            scale_policy=args.history_scale_policy,
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
        panel_size = int(getattr(args, "spectral_panel_size", 16))
        train_size = max(len(dataset) - min(panel_size, len(dataset) // 4), 1)
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        loader = torch.utils.data.DataLoader(
            train_dataset,
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

    panel_size = int(getattr(args, "spectral_panel_size", 16))
    train_size = max(len(dataset) - min(panel_size, len(dataset) // 4), 1)
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=use_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_workers > 0,
    )
    return dataset, loader


def make_spectral_panel(dataset, panel_size: int) -> torch.Tensor:
    """Stack the deterministic tail split that is excluded from training."""
    if panel_size <= 0:
        raise ValueError("spectral_panel_size must be positive")
    count = min(panel_size, max(len(dataset) - 1, 1))
    start = len(dataset) - count
    images = []
    for index in range(start, len(dataset)):
        item = dataset[index]
        images.append(item[0] if isinstance(item, (list, tuple)) else item)
    return torch.stack(images)


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
    generator = torch.Generator(
        device=next(unwrapped.parameters()).device
    ).manual_seed(args.preview_seed)
    out = unwrapped.generate(
        batch_size=args.num_validation_images,
        generator=generator,
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


@torch.no_grad()
def evaluate_spectral_panel(
    model: ContinuousFFTDecoder,
    panel_images: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run objective-independent metrics with fixed images, times, and noise."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    images = panel_images.to(device=device, dtype=torch.float32)
    tokens = model.codec.encode(images)
    batch, length, _ = tokens.shape
    timesteps = torch.linspace(
        0,
        model.config.diffusion.num_train_timesteps - 1,
        batch,
        device=device,
    ).round().long()
    token_timesteps = timesteps[:, None].expand(batch, length)
    generator = torch.Generator(device=device).manual_seed(
        args.spectral_diagnostic_seed
    )
    noise = torch.randn(
        tokens.shape,
        device=device,
        dtype=tokens.dtype,
        generator=generator,
    )
    noise = noise * model.codec.component_mask[None].to(noise.dtype)
    predicted = model.predict_x0_diagnostics(tokens, token_timesteps, noise)

    metrics = {
        f"spectral/{key}": value
        for key, value in compute_spectral_diagnostics(
            predicted,
            tokens,
            model.codec,
            timesteps=timesteps,
        ).items()
    }
    history_mask = model.codec.component_mask[:-1][None].to(tokens.dtype)
    history_perturbation = (
        noise[:, :-1] * float(args.history_noise_max) * history_mask
    )
    perturbed_history = tokens[:, :-1] + history_perturbation
    perturbed_prediction = model.predict_x0_diagnostics(
        tokens,
        token_timesteps,
        noise,
        history_override=perturbed_history,
    )
    perturbed_metrics = compute_spectral_diagnostics(
        perturbed_prediction,
        tokens,
        model.codec,
        timesteps=timesteps,
    )
    for key, value in perturbed_metrics.items():
        metrics[f"robustness/gaussian/{key}"] = value
        clean_key = f"spectral/{key}"
        if clean_key in metrics:
            metrics[f"robustness/gaussian_delta/{key}"] = value - metrics[clean_key]

    clean_conditions, _ = model.forward_backbone(
        model.embed_tokens(tokens[:, :-1], include_bos=True)
    )
    perturbed_conditions, _ = model.forward_backbone(
        model.embed_tokens(perturbed_history, include_bos=True)
    )
    metrics["robustness/condition_cosine"] = F.cosine_similarity(
        clean_conditions.float(),
        perturbed_conditions.float(),
        dim=-1,
    ).mean()
    metrics["robustness/condition_relative_rms"] = (
        (perturbed_conditions.float() - clean_conditions.float())
        .square()
        .mean()
        .sqrt()
        / clean_conditions.float().square().mean().sqrt().clamp_min(1e-8)
    )
    physical = model.codec.encode_raw(images)
    metrics.update(
        {
            f"normalization/{key}": value
            for key, value in compute_normalization_phase_distortion(
                physical,
                model.codec,
            ).items()
        }
    )
    perturbation = noise * float(args.history_noise_max)
    metrics.update(
        {
            f"perturbation/{key}": value
            for key, value in compute_perturbation_diagnostics(
                tokens,
                noise,
                perturbation,
                codec=model.codec,
                timesteps=timesteps,
            ).items()
        }
    )

    positions = torch.arange(length - 1, device=device)
    history = model._history_cartesian_features(tokens[:, :-1], positions)
    token_projected = model.token_proj(history.to(model.token_proj.weight.dtype))
    metrics["projection/history_input_rms"] = history.float().square().mean().sqrt()
    metrics["projection/token_output_rms"] = (
        token_projected.float().square().mean().sqrt()
    )
    if model.polar_proj is not None:
        polar = model.codec.polar_history_features(tokens[:, :-1], positions)
        polar_projected = model.polar_proj(polar.to(model.polar_proj.weight.dtype))
        metrics["projection/polar_input_rms"] = polar.float().square().mean().sqrt()
        metrics["projection/polar_output_rms"] = (
            polar_projected.float().square().mean().sqrt()
        )
    input_projected = model.diffusion.net.input_proj(
        tokens.to(model.diffusion.net.input_proj.weight.dtype)
    )
    metrics["projection/diffusion_input_rms"] = tokens.float().square().mean().sqrt()
    metrics["projection/diffusion_projected_rms"] = (
        input_projected.float().square().mean().sqrt()
    )
    position_values = model.slot_embed.weight
    metrics["projection/slot_embedding_rms"] = (
        position_values.float().square().mean().sqrt()
    )

    logs: Dict[str, float] = {}
    for key, value in metrics.items():
        scalar = value.detach().float()
        if scalar.numel() == 1 and bool(torch.isfinite(scalar)):
            logs[key] = scalar.item()
    if was_training:
        model.train()
    return logs


def _bucket_means(
    timesteps: torch.Tensor,
    values: torch.Tensor,
    num_buckets: int,
    num_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized per-timestep-bin means on the input device."""
    indices = torch.div(
        timesteps.long() * num_buckets,
        num_timesteps,
        rounding_mode="floor",
    ).clamp_(0, num_buckets - 1)
    counts = torch.bincount(indices, minlength=num_buckets)
    sums = torch.zeros(num_buckets, device=values.device, dtype=torch.float32)
    sums.scatter_add_(0, indices, values.detach().float())
    means = sums / counts.clamp_min(1).to(sums.dtype)
    return means, counts


def bucket_timestep_loss(
    timesteps: torch.Tensor,
    per_example: torch.Tensor,
    num_buckets: int = 5,
    num_timesteps: int = 1000,
) -> Dict[str, float]:
    means, counts = _bucket_means(
        timesteps,
        per_example,
        num_buckets=num_buckets,
        num_timesteps=num_timesteps,
    )
    means_cpu = means.cpu().tolist()
    counts_cpu = counts.cpu().tolist()
    return {
        f"loss/t_bucket_{index}": means_cpu[index]
        for index in range(num_buckets)
        if counts_cpu[index] > 0
    }


class TimestepLossEMA:
    """Low-overhead GPU EMA of raw and effective loss by flow/DDPM time."""

    metric_names = (
        "raw_mse",
        "unweighted_objective",
        "time_weight",
        "weighted_objective",
    )

    def __init__(self, num_buckets: int, num_timesteps: int, decay: float):
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.num_buckets = num_buckets
        self.num_timesteps = num_timesteps
        self.decay = decay
        self.ema: Optional[torch.Tensor] = None
        self.initialized: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, out: Dict[str, torch.Tensor]) -> None:
        timesteps = out["timesteps"]
        series = (
            out["normalized_per_example"],
            out["per_example"],
            out["snr_weights"],
            out["per_example"] * out["weights"],
        )
        means = []
        counts = None
        for values in series:
            bucketed, current_counts = _bucket_means(
                timesteps,
                values,
                num_buckets=self.num_buckets,
                num_timesteps=self.num_timesteps,
            )
            means.append(bucketed)
            if counts is None:
                counts = current_counts
        current = torch.stack(means)
        if self.ema is None:
            self.ema = torch.zeros_like(current)
            self.initialized = torch.zeros(
                self.num_buckets,
                device=current.device,
                dtype=torch.bool,
            )
        assert self.initialized is not None
        valid = counts > 0
        continuing = valid & self.initialized
        starting = valid & ~self.initialized
        self.ema[:, continuing] = (
            self.decay * self.ema[:, continuing]
            + (1.0 - self.decay) * current[:, continuing]
        )
        self.ema[:, starting] = current[:, starting]
        self.initialized |= valid

    def logs(self) -> Dict[str, float]:
        if self.ema is None or self.initialized is None:
            return {}
        values = self.ema.cpu().tolist()
        initialized = self.initialized.cpu().tolist()
        logs: Dict[str, float] = {}
        for metric_index, metric_name in enumerate(self.metric_names):
            for bucket in range(self.num_buckets):
                if initialized[bucket]:
                    logs[
                        f"timestep_ema/{metric_name}/bin_{bucket:02d}"
                    ] = values[metric_index][bucket]
        return logs


class CudaStepTimer:
    """Sparse CUDA-event timer that synchronizes only when a sample is reported."""

    intervals = (
        ("encode", "start", "after_encode"),
        ("forward", "after_encode", "after_forward"),
        ("backward", "after_forward", "after_backward"),
        ("grad_processing", "after_backward", "after_grad"),
        ("optimizer", "after_grad", "after_optimizer"),
        ("train_step_gpu", "start", "after_optimizer"),
    )

    def __init__(self) -> None:
        self.events: Dict[str, torch.cuda.Event] = {}

    def record(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[name] = event

    def logs(self) -> Dict[str, float]:
        self.events["after_optimizer"].synchronize()
        return {
            f"timing/{name}_ms": self.events[start].elapsed_time(self.events[end])
            for name, start, end in self.intervals
        }


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
        dataloader_config=DataLoaderConfiguration(
            non_blocking=torch.cuda.is_available() and not args.smoke,
        ),
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
    spectral_panel_images = (
        make_spectral_panel(dataset, args.spectral_panel_size)
        if args.spectral_diagnostic_steps > 0
        else None
    )
    config = build_model_config(args)
    codec = fit_or_load_codec(args, accelerator, train_loader, config)
    if accelerator.is_main_process and args.loss_metric == "orbit_scale_power":
        scale_weights = codec.orbit_scale_power_metric(args.orbit_scale_exponent)
        active_weights = scale_weights[codec.component_mask.bool()]
        _log_info(
            "Orbit scale-power weights: "
            f"mean={active_weights.mean().item():.4f} "
            f"p99={torch.quantile(active_weights, 0.99).item():.4f} "
            f"max={active_weights.max().item():.4f} "
            f"exponent={args.orbit_scale_exponent:.3f}"
        )
    if accelerator.is_main_process:
        if not args.normalization.startswith("orbit_"):
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
            f"component_reduction={args.component_reduction} "
            f"weighting={args.loss_weighting} min_snr_gamma={args.min_snr_gamma:g} "
            f"logit_normal=({args.logit_normal_mean:g}, {args.logit_normal_std:g}) "
            f"zero_terminal_snr={args.rescale_betas_zero_snr} "
            f"snr_scale={args.snr_scale:g} "
            f"timestep_spacing={args.timestep_spacing} "
            f"flow_solver={args.flow_solver}"
        )
        _log_info(
            f"Polar history features: {args.history_polar_features} "
            f"(enabled={args.history_polar_features != 'none'})"
        )
        _log_info(
            f"Attention geometry: qk_norm={bool(args.qk_norm)} "
            f"rope={args.attention_rope} rope_base={args.rope_base:g} "
            f"slot_embedding=learned transformer_film="
            f"{bool(args.transformer_position_film)} decoder_position_condition=False"
        )
        _log_info(
            f"Representation: centering={args.centering} "
            f"packing={args.coordinate_packing} "
            f"ecs_percentile={args.ecs_percentile:g} "
            f"diffusion_mean={config.codec.mean_policy} "
            f"diffusion_scale={config.codec.scale_policy} "
            f"history_cartesian={args.history_cartesian_features} "
            f"history_mean={config.history_features.mean_policy} "
            f"history_scale={config.history_features.scale_policy} "
            f"input_time={config.diffusion.input_timestep_conditioning} "
            f"input_init={args.input_projection_init} "
            f"adam=({args.adam_beta1:g},{args.adam_beta2:g})"
        )
    model = ContinuousFFTDecoder(config, codec=codec)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    no_decay_names: set[str] = set()
    if model.output_log_gain is not None:
        no_decay_names.add("output_log_gain")
    optimizer_parameters: Any = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if name not in no_decay_names
            ],
            "weight_decay": args.adam_weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if name in no_decay_names
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
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
                f"-a{args.orbit_scale_exponent:g}"
                if args.loss_metric == "orbit_scale_power"
                else (
                    f"-radial{args.radial_power_exponent:g}"
                    if args.radial_power_weighting
                    else ""
                )
            )
        )
        run_name = args.run_name or (
            f"{args.objective}-{args.prediction_type}-{args.normalization}-"
            f"{args.diffusion_mean_policy}-{args.diffusion_scale_policy}-"
            f"{args.loss_metric}{metric_suffix}-pos-slot-rope-{args.attention_rope}-"
            f"hist-{args.history_cartesian_features}-{args.history_mean_policy}-"
            f"{args.history_scale_policy}-stem-"
            f"{config.diffusion.input_timestep_conditioning}-d{args.diff_depth}-"
            f"b{args.train_batch_size}"
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

    if accelerator.is_main_process and spectral_panel_images is not None:
        initial_spectral_logs = evaluate_spectral_panel(
            accelerator.unwrap_model(model),
            spectral_panel_images,
            args,
        )
        if log_with is not None:
            accelerator.log(initial_spectral_logs, step=global_step)
        summary_keys = (
            "normalization/mu_over_z_q50",
            "normalization/mu_over_z_q90",
            "normalization/mu_over_z_q99",
            "normalization/phase_distortion_circular_error",
            "normalization/mu_over_uncentered_rms/q50",
            "normalization/mu_over_uncentered_rms/q90",
            "normalization/pooled_residual_rms/uncentered_rms",
            "normalization/pooled_rms/phase_distortion_circular_error",
            "projection/history_input_rms",
            "projection/diffusion_projected_rms",
        )
        _log_info(
            "SPECTRAL_STARTUP "
            + " ".join(
                f"{key}={initial_spectral_logs[key]:.6g}"
                for key in summary_keys
                if key in initial_spectral_logs
            )
        )
    accelerator.wait_for_everyone()

    progress = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    benchmark_start = None
    benchmark_start_step = 0
    timestep_loss_ema = (
        TimestepLossEMA(
            num_buckets=args.timestep_histogram_bins,
            num_timesteps=config.diffusion.num_train_timesteps,
            decay=args.timestep_histogram_decay,
        )
        if args.timestep_histogram_bins > 0
        else None
    )
    data_wait_started = time.perf_counter()
    throughput_window_started = time.perf_counter()
    throughput_window_step = global_step

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            data_load_ms = (time.perf_counter() - data_wait_started) * 1000.0
            projection_grad_logs: Dict[str, Any] = {}
            timing_logs: Dict[str, float] = {}
            step_timer = None
            with accelerator.accumulate(model):
                next_optimizer_step = global_step + 1
                routine_log_step = (
                    accelerator.sync_gradients
                    and (
                        next_optimizer_step % args.logging_steps == 0
                        or next_optimizer_step >= args.max_train_steps
                    )
                )
                timing_sample = (
                    accelerator.sync_gradients
                    and args.timing_steps > 0
                    and next_optimizer_step % args.timing_steps == 0
                    and torch.cuda.is_available()
                    and not args.smoke
                )
                if timing_sample:
                    step_timer = CudaStepTimer()
                    step_timer.record("start")
                images = batch[0]
                # Encode under no_grad; tokens are continuous targets.
                with torch.no_grad():
                    unwrapped = accelerator.unwrap_model(model)
                    tokens = unwrapped.codec.encode(images)
                if step_timer is not None:
                    step_timer.record("after_encode")
                out = model(
                    tokens,
                    corrupt=True,
                    training_progress=min(global_step / max(args.max_train_steps, 1), 1.0),
                )
                loss = out["loss"]
                if (
                    accelerator.sync_gradients
                    and args.phase_aux_weight > 0.0
                    and args.phase_gradient_diagnostic_step > 0
                    and next_optimizer_step == args.phase_gradient_diagnostic_step
                ):
                    output_weight = accelerator.unwrap_model(
                        model
                    ).diffusion.net.final_layer.linear.weight
                    base_grad = torch.autograd.grad(
                        out["_base_loss_component"],
                        output_weight,
                        retain_graph=True,
                    )[0]
                    phase_grad = torch.autograd.grad(
                        args.phase_aux_weight * out["_phase_loss_component"],
                        output_weight,
                        retain_graph=True,
                    )[0]
                    out["phase_aux_output_grad_ratio"] = (
                        phase_grad.float().norm()
                        / base_grad.float().norm().clamp_min(1e-12)
                    ).detach()
                if step_timer is not None:
                    step_timer.record("after_forward")
                accelerator.backward(loss)
                if step_timer is not None:
                    step_timer.record("after_backward")
                grad_norm = None
                if accelerator.sync_gradients:
                    if args.grad_norm_mode == "clip":
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )
                    elif args.grad_norm_mode == "track":
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            float("inf"),
                        )
                    if routine_log_step:
                        unwrapped_for_grads = accelerator.unwrap_model(model)
                        for name, parameter in (
                            ("token_proj", unwrapped_for_grads.token_proj.weight),
                            (
                                "diffusion_input",
                                unwrapped_for_grads.diffusion.net.input_proj.weight,
                            ),
                            (
                                "diffusion_output",
                                unwrapped_for_grads.diffusion.net.final_layer.linear.weight,
                            ),
                        ):
                            if parameter.grad is not None:
                                projection_grad_logs[
                                    f"projection_grad_rms/{name}"
                                ] = parameter.grad.detach().float().square().mean().sqrt()
                        if (
                            unwrapped_for_grads.polar_proj is not None
                            and unwrapped_for_grads.polar_proj.weight.grad is not None
                        ):
                            projection_grad_logs["projection_grad_rms/polar"] = (
                                unwrapped_for_grads.polar_proj.weight.grad.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            )
                if step_timer is not None:
                    step_timer.record("after_grad")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if step_timer is not None:
                    step_timer.record("after_optimizer")

            if accelerator.sync_gradients:
                if step_timer is not None:
                    timing_logs = step_timer.logs()
                    timing_logs["timing/data_load_ms"] = data_load_ms
                    timing_logs["performance/cuda_memory_allocated_gib"] = (
                        torch.cuda.memory_allocated() / (1024**3)
                    )
                    timing_logs["performance/cuda_memory_reserved_gib"] = (
                        torch.cuda.memory_reserved() / (1024**3)
                    )
                    timing_logs["performance/cuda_peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated() / (1024**3)
                    )
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

                logs: Dict[str, Any] = dict(timing_logs)
                if "phase_aux_output_grad_ratio" in out:
                    logs["phase_aux_output_grad_ratio"] = out[
                        "phase_aux_output_grad_ratio"
                    ].item()
                if routine_log_step:
                    logs.update(
                        {
                            "loss": loss.detach().item(),
                            "unweighted_mse": out["unweighted_mse"].item(),
                            "weighted_loss": out["weighted_loss"].item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "corruption_strength": out[
                                "corruption_strength"
                            ].mean().item(),
                        }
                    )
                    logs.update(
                        {
                            key: value.item() if hasattr(value, "item") else float(value)
                            for key, value in projection_grad_logs.items()
                        }
                    )
                    if grad_norm is not None:
                        logs["grad_norm"] = (
                            grad_norm.item()
                            if hasattr(grad_norm, "item")
                            else float(grad_norm)
                        )
                    if "covariance_metric_loss" in out:
                        logs["covariance_metric_loss"] = out[
                            "covariance_metric_loss"
                        ].item()
                    if "scale_metric_loss" in out:
                        logs["scale_metric_loss"] = out["scale_metric_loss"].item()
                    if "phase_aux_loss" in out:
                        logs["phase_aux_loss"] = out["phase_aux_loss"].item()
                        logs["base_loss"] = out["base_loss"].item()
                    if "radial_weighted_mse" in out:
                        logs["radial_weighted_mse"] = out["radial_weighted_mse"].item()
                    if "radial_weights" in out:
                        logs["radial_weight_mean"] = (
                            out["radial_weights"].float().mean().item()
                        )
                    unwrapped = accelerator.unwrap_model(model)
                    if unwrapped.output_log_gain is not None:
                        gains = unwrapped.output_log_gain.detach().exp()
                        logs["output_gain/mean"] = gains.mean().item()
                        logs["output_gain/min"] = gains.min().item()
                        logs["output_gain/max"] = gains.max().item()
                    logs.update(
                        bucket_timestep_loss(
                            out["timesteps"],
                            out["per_example"],
                            num_timesteps=config.diffusion.num_train_timesteps,
                        )
                    )
                if timestep_loss_ema is not None:
                    timestep_loss_ema.update(out)
                    if (
                        args.timestep_histogram_log_steps > 0
                        and global_step % args.timestep_histogram_log_steps == 0
                    ):
                        logs.update(timestep_loss_ema.logs())
                # Radius-bin loss is cheap on-device but synchronizes when exported.
                if routine_log_step and "radius_bin" in out:
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

                if (
                    args.condition_diagnostic_steps > 0
                    and global_step % args.condition_diagnostic_steps == 0
                ):
                    condition_started = time.perf_counter()
                    cpu_rng_state = torch.random.get_rng_state()
                    cuda_rng_state = (
                        torch.cuda.get_rng_state(tokens.device)
                        if tokens.is_cuda
                        else None
                    )
                    with torch.no_grad(), accelerator.autocast():
                        conditioned = model(tokens, corrupt=False)
                    torch.random.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, tokens.device)
                    shuffled_history = tokens.roll(1, dims=0)[:, :-1, :]
                    with torch.no_grad(), accelerator.autocast():
                        shuffled = model(
                            tokens,
                            corrupt=False,
                            history_override=shuffled_history,
                        )
                    conditioned_loss = conditioned["loss"].item()
                    shuffled_loss = shuffled["loss"].item()
                    logs["condition/clean_loss"] = conditioned_loss
                    logs["condition/shuffled_history_loss"] = shuffled_loss
                    logs["condition/shuffle_gap"] = (
                        shuffled_loss - conditioned_loss
                    )
                    logs["timing/condition_diagnostic_wall_ms"] = (
                        time.perf_counter() - condition_started
                    ) * 1000.0
                    _log_info(
                        "CONDITION_DIAGNOSTIC "
                        f"step={global_step} clean={conditioned_loss:.6f} "
                        f"shuffled={shuffled_loss:.6f} "
                        f"gap={shuffled_loss - conditioned_loss:.6f}"
                    )

                if (
                    accelerator.is_main_process
                    and spectral_panel_images is not None
                    and args.spectral_diagnostic_steps > 0
                    and global_step % args.spectral_diagnostic_steps == 0
                ):
                    spectral_started = time.perf_counter()
                    logs.update(
                        evaluate_spectral_panel(
                            accelerator.unwrap_model(model),
                            spectral_panel_images,
                            args,
                        )
                    )
                    logs["timing/spectral_diagnostic_wall_ms"] = (
                        time.perf_counter() - spectral_started
                    ) * 1000.0

                if routine_log_step:
                    window_elapsed = max(
                        time.perf_counter() - throughput_window_started,
                        1e-9,
                    )
                    window_steps = max(global_step - throughput_window_step, 1)
                    logs["performance/steps_per_sec"] = window_steps / window_elapsed
                    logs["performance/examples_per_sec"] = (
                        window_steps * args.train_batch_size / window_elapsed
                    )
                if logs:
                    progress_keys = ("loss", "lr", "grad_norm")
                    progress.set_postfix(
                        **{key: logs[key] for key in progress_keys if key in logs}
                    )
                    accelerator.log(logs, step=global_step)
                    if routine_log_step:
                        throughput_window_started = time.perf_counter()
                        throughput_window_step = global_step

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

                if (
                    args.preview_steps > 0
                    and global_step % args.preview_steps == 0
                ):
                    preview_started = time.perf_counter()
                    max_tokens = 8 if args.smoke else None
                    val_logs = validate(
                        model, args, accelerator, global_step, ema=ema, max_tokens=max_tokens
                    )
                    val_logs["timing/preview_wall_ms"] = (
                        time.perf_counter() - preview_started
                    ) * 1000.0
                    accelerator.log(val_logs, step=global_step)

                data_wait_started = time.perf_counter()
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
