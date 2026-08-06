#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
exponent="${SPECTRAL_SCALE_EXPONENT:-0.8}"
steps="${STEPS:-30000}"
output_dir="${OUTPUT_DIR:-${project_root}/latent_continuous_runs/fft_compact_scaled_spiral_e${exponent}_control}"

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "fft-compact-scaled-spiral-e${exponent}" \
  --wait \
  -- \
  /venv/main/bin/python -u control_pixel_diffusion.py \
  --output_dir "${output_dir}" \
  --representation fft_compact_scaled_spiral \
  --spectral_scale_exponent "${exponent}" \
  --patch 4 \
  --steps "${steps}" \
  --batch_size 256 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12
