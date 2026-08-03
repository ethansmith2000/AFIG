#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/fft_global_spiral_control

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job fft-global-spiral-control \
  --wait \
  -- \
  /venv/main/bin/python control_pixel_diffusion.py \
  --output_dir "${output_dir}" \
  --representation fft_global_spiral \
  --orbits_per_token 8 \
  --steps 30000 \
  --batch_size 256 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12
