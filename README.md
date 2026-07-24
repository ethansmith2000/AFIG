# AFIG (Autoregressive Fourier Image Generation)

Continuous-token rewrite of the original quantized Fourier AR project, plus the
legacy discrete baseline.

Blog post for the original idea:
https://www.ethansmith2000.com/post/mimicking-diffusion-models-by-sequencing-frequency-coefficients

## Layout

| Path | Role |
| --- | --- |
| `frequency.py` | Canonical 514-orbit Hermitian Fourier codec + radial whitening |
| `diffusion_decoder.py` | AdaLN MLP diffusion loss head + DDIM sampler |
| `model_continuous.py` | Causal Transformer + KV cache + continuous generation |
| `train_continuous.py` | Accelerate training entrypoint |
| `model.py` / `train_quantized.py` / `utils.py` | Legacy quantized path (kept) |
| `tests/` | Unit tests + CPU smoke |

## Continuous representation

- Orthonormal `fft2(..., norm="ortho")` on CIFAR-10 `3×32×32` images.
- **514** conjugacy-orbit representatives (not the old 544 half-plane cells).
- Each token is **6D Cartesian**: RGB real + RGB imag.
- Four self-conjugate frequencies keep imag components masked to zero.
- Default order: exact Euclidean radius, then angle (`--ordering radial`).
  Legacy L∞ square spiral remains available (`--ordering square_spiral`).
- Normalization bins: **one per integer radius** `floor(sqrt(kx²+ky²))`.
- Default whitening: dataset-level radial Cholesky whitening
  (`--normalization radial_whiten`). Diagonal standardize is also supported.
- Optional value transform: `--value_transform identity|asinh`.

Sequence length is **514** autoregressive coefficient steps (plus a learned BOS).

## Diffusion objective

Configured in `DiffusionDecoderConfig` / CLI:

- `--prediction_type epsilon|v_prediction` (default `epsilon`)
- `--loss_weighting none|min_snr` (default `none`, γ via `--min_snr_gamma 5`)
- `--radial_power_weighting`: multiply per-token whitened MSE by normalized
  tempered radial power `(tr(Σ_b) / d_b)^α` (mean 1 across orbits). The default
  `--radial_power_exponent 0.5` weights by expected amplitude; `1.0` restores
  the much more concentrated expected-power objective. Independent of Min-SNR.
- Cosine (`squaredcos_cap_v2`) schedule, 1000 train steps
- DDIM sampling, default 20 steps, `eta=0`
- Diffusion batch multiplier `--diffusion_batch_mul` (default 4): reuse each
  `(token, condition)` with independent `(t, ε)` draws

Flow matching is intentionally **not** selectable yet (stub only).

## Polar history conditioning

Diffusion targets stay **6D Cartesian**. Optionally enrich AR history embeddings
with deterministic physical-space polar features
(`--history_polar_features log_amp_gated_phase`):

- Denormalize each completed history token, then per RGB channel form
  `[log1p(a), g·cos θ, g·sin θ]` with `a = amp / expected_rms` and `g = a/(1+a)`.
- Projected by a zero-initialized `Linear(9, width)` and added to the Cartesian
  token embedding. Does **not** change the diffusion state manifold.

## Frequency position conditioning

`--frequency_conditioning` enables the positional path used by the larger
exploratory model:

- normalized `(kx, ky, radius)` receive log-spaced sinusoidal features, combined
  with `(cos(angle), sin(angle), is_self_conjugate)` and a learned orbit residual;
- the known target orbit directly conditions the diffusion decoder's AdaLN;
- each Transformer attention/MLP pre-norm receives target-position FiLM whose
  scale/shift projection is zero-initialized.

2D RoPE remains deferred as a separate relative-geometry ablation.

The three routes can be ablated independently:

- `--[no-]position-input-addition`
- `--[no-]transformer-position-film`
- `--[no-]diffusion-target-conditioning`
- `--position-rms-normalize` optionally controls the shared position RMS.

For a clean content-only input stream while retaining conditional position,
use `--no-position-input-addition --position-rms-normalize`.

## Training

Fit/load codec statistics once (main process writes `codec_stats.pt`), then train:

```bash
cd /workspace/AFIG
source /venv/main/bin/activate

# CPU / tiny smoke (synthetic data by default — fast, no download)
python train_continuous.py --smoke --output_dir continuous_smoke --report_to none

# Real CIFAR-10 via torchvision (downloads ~163MB from cs.toronto.edu if missing)
python train_continuous.py --dataset cifar10 --data_root ./data ...

# Or use a local HuggingFace CIFAR arrow cache (already present on this host under SNRAdam)
python train_continuous.py --dataset huggingface_cifar ...

# Full-ish moderate run on GPU (use shared claim helper on this machine)
gpu-claim status
gpu-claim run --owner AFIG --job continuous-train --wait -- \
  python train_continuous.py \
    --output_dir continuous_runs \
    --dataset auto \
    --preset moderate \
    --prediction_type epsilon \
    --loss_weighting none \
    --history_corruption none \
    --mixed_precision bf16

# Exploratory 10×768 with polar history, radial weighting, and frequency conditioning
gpu-claim run --owner AFIG --job continuous-10x768 --wait -- \
  python train_continuous.py \
    --output_dir continuous_runs/hf_cifar_10x768_bs32 \
    --codec_stats_path continuous_runs/hf_cifar_moderate/codec_stats.pt \
    --dataset huggingface_cifar \
    --num_layers 10 --width 768 --num_heads 12 \
    --diff_width 768 --diff_depth 3 \
    --train_batch_size 32 --diffusion_batch_mul 1 \
    --learning_rate 7e-5 --adam_beta2 0.99 \
    --prediction_type v_prediction --loss_weighting min_snr \
    --radial_power_weighting \
    --radial_power_exponent 0.5 \
    --history_polar_features log_amp_gated_phase \
    --frequency_conditioning \
    --history_corruption none \
    --mixed_precision bf16 --gradient_checkpointing --allow_tf32
```

### Data sources (`--dataset`)

| Value | Behavior |
| --- | --- |
| `auto` (default) | torchvision CIFAR if present/downloadable, else local HF arrows, else synthetic |
| `cifar10` | Force torchvision CIFAR-10 under `--data_root` |
| `huggingface_cifar` | Force local HuggingFace `cifar10-train.arrow` cache |
| `synthetic` | Random 32×32 tensors (used by `--smoke` unless overridden) |

Note: the official Toronto CIFAR tarball can be very slow from this host (~50KB/s).
If `./data/cifar-10-batches-py` is missing, prefer `--dataset huggingface_cifar` or let
`auto` pick up the existing arrow cache.

Useful flags:

- `--resume_from_checkpoint PATH|latest`
- `--codec_stats_path PATH` (defaults to `$output_dir/codec_stats.pt`)
- `--history_corruption none|gaussian`
- `--history_polar_features none|log_amp_gated_phase`
- `--frequency_conditioning`
- `--position_num_frequencies 4 --position_max_frequency 8`
- `--[no-]position-input-addition`
- `--[no-]transformer-position-film`
- `--[no-]diffusion-target-conditioning`
- `--position-rms-normalize`
- `--radial_power_weighting`
- `--radial_power_exponent 0.5`
- `--checkpointing_steps 0` (default; set above zero only for resumable runs)
- `--save_final_checkpoint` (off by default)
- `--preset tiny|moderate|legacy`
- `--dataset auto|cifar10|huggingface_cifar|synthetic`

Checkpoint saving is disabled by default. When explicitly enabled, versioned
`.pt` files contain model, optimizer, LR schedule, EMA, codec export, and configs.

### Validation diagnostics

Logged / saved periodically:

- sample image grid `samples_{step}.png`
- Hermitian violation and imaginary reconstruction energy
- backbone vs denoiser wall time
- loss by timestep bucket and selected radius bins (flat and radially weighted)
- radial weight mean / range at startup

## Tests

```bash
cd /workspace/AFIG
python -m unittest discover -s tests -v
```

## Legacy quantized path

```bash
python train_quantized.py
```

Still uses the polar half-spectrum unrolling in `utils.py` with a discrete vocab
and cross-entropy. Narrow compatibility fixes only:

- device-correct `new_empty` in `get_1d_freqs_from_2d`
- optional `context` on `TransformerLayer.forward`
- `topk_sample` / `top_k_sampling` alias

## Future TODOs (stubbed, not implemented)

- **Flow matching** per-token decoder + Euler sampler
- **CFG** via class-condition dropout in the Transformer (do not drop AR state `z`)
- Alternate history corruptions: masked tokens, rollout-mix
- **FixedChunkGrouping**: joint denoise over K consecutive coefficients
- **RadialBandGrouping**: block-AR over integer-radius bands

## Notes for agents on this host

Any exclusive single-GPU train/eval must go through `gpu-claim`
(see `/workspace/GPU_QUEUEING.md`). Do not invent a parallel claim scheme.
