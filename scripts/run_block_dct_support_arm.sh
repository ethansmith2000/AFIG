#!/bin/bash
set -euo pipefail

project_root=/workspace/AFIG
support="${DCT_SUPPORT:-2}"
seed="${SEED:-7}"
steps="${STEPS:-10000}"
ae_checkpoint="${AE_CHECKPOINT:-${project_root}/autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt}"
output_dir="${OUTPUT_DIR:-${project_root}/continuous_runs/ar_c4_blockdct_support${support}_seqrope_s${seed}_${steps}}"

case "${support}" in
  2|4|8) ;;
  *) echo "DCT_SUPPORT must be one of 2, 4, or 8" >&2; exit 2 ;;
esac

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "c4-blockdct-support${support}-seqrope-s${seed}-${steps}" \
  --wait -- \
  /venv/main/bin/python -u train_spatial_latent_hartley_ar.py \
  --ae_checkpoint "${ae_checkpoint}" \
  --output_dir "${output_dir}" \
  --data_root "${project_root}/data" \
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
  --latent_patch 2 \
  --stats_images 4096 \
  --latent_basis block_dct \
  --token_order raster \
  --tiles_per_token 1 \
  --dct_support "${support}" \
  --block_dct_token_dim 16 \
  --rope_mode sequence
