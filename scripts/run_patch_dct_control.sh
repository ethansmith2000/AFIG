#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/patch_dct_control

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job patch-dct-control \
  --wait \
  -- \
  /venv/main/bin/python control_pixel_diffusion.py \
  --output_dir "${output_dir}" \
  --representation patch_dct \
  --patch 4 \
  --steps 30000 \
  --batch_size 256 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12
