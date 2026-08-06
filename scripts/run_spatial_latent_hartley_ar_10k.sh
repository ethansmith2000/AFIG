#!/bin/bash
set -euo pipefail

cd /workspace/AFIG

ae_checkpoint="${AE_CHECKPOINT:-/workspace/AFIG/autoencoder_runs/ae-spatial-d4-c8-r32-s1-n10000-noise0.1/checkpoint_10000.pt}"
output_dir="${OUTPUT_DIR:-/workspace/AFIG/continuous_runs/ar_spatial_ae_hartley_tiles_10k}"
latent_basis="${LATENT_BASIS:-hartley}"
token_order="${TOKEN_ORDER:-auto}"
tiles_per_token="${TILES_PER_TOKEN:-1}"
latent_patch="${LATENT_PATCH:-2}"
dct_support="${DCT_SUPPORT:-2}"
block_dct_token_dim="${BLOCK_DCT_TOKEN_DIM:-16}"
compact_fft_token_dim="${COMPACT_FFT_TOKEN_DIM:-16}"
rope_mode="${ROPE_MODE:-frequency_2d}"
seed="${SEED:-1}"
steps="${STEPS:-10000}"
run_label="${RUN_LABEL:-ar-spatial-ae-${latent_basis}-${token_order}-g${tiles_per_token}-${steps}}"

exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "${run_label}" \
  --wait -- \
  /venv/main/bin/python -u train_spatial_latent_hartley_ar.py \
  --ae_checkpoint "${ae_checkpoint}" \
  --output_dir "${output_dir}" \
  --data_root /workspace/AFIG/data \
  --seed "${seed}" \
  --steps "${steps}" \
  --batch_size 64 \
  --num_workers 4 \
  --learning_rate 7e-5 \
  --warmup 500 \
  --weight_decay 0.1 \
  --width 768 \
  --layers 10 \
  --heads 12 \
  --ff_mult 4 \
  --diff_width 768 \
  --diff_depth 3 \
  --inference_steps 20 \
  --preview_steps 2500 \
  --diagnostic_steps 250 \
  --checkpoint_steps 2500 \
  --validation_images 16 \
  --latent_patch "${latent_patch}" \
  --dct_support "${dct_support}" \
  --block_dct_token_dim "${block_dct_token_dim}" \
  --compact_fft_token_dim "${compact_fft_token_dim}" \
  --rope_mode "${rope_mode}" \
  --stats_images 4096 \
  --latent_basis "${latent_basis}" \
  --token_order "${token_order}" \
  --tiles_per_token "${tiles_per_token}"
