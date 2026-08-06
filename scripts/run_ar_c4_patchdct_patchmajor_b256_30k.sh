#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/continuous_runs/ar_c4_patchdct_patchmajor_b256_s7_30k

mkdir -p "${output_dir}"
cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job ar-c4-patchdct-patchmajor-b256-30k \
  --wait \
  -- \
  /venv/main/bin/python -u train_spatial_latent_hartley_ar.py \
  --ae_checkpoint autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt \
  --data_root data \
  --output_dir "${output_dir}" \
  --steps 30000 \
  --batch_size 256 \
  --num_workers 4 \
  --width 768 \
  --layers 10 \
  --heads 12 \
  --ff_mult 4 \
  --diff_width 768 \
  --diff_depth 3 \
  --inference_steps 20 \
  --preview_steps 5000 \
  --diagnostic_steps 500 \
  --checkpoint_steps 5000 \
  --validation_images 16 \
  --latent_patch 2 \
  --stats_images 4096 \
  --latent_basis patch_dct \
  --token_order raster \
  --rope_mode frequency_2d \
  --seed 7
