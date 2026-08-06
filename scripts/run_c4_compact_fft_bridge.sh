#!/bin/bash
set -euo pipefail

project_root=/workspace/AFIG
seed="${SEED:-7}"
steps="${STEPS:-10000}"
ae_checkpoint="${AE_CHECKPOINT:-${project_root}/autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt}"
output_dir="${OUTPUT_DIR:-${project_root}/continuous_runs/ar_c4_compact_fft_seqrope_s${seed}_${steps}}"

cd "${project_root}"
exec env \
  AE_CHECKPOINT="${ae_checkpoint}" \
  OUTPUT_DIR="${output_dir}" \
  LATENT_BASIS=compact_fft \
  TOKEN_ORDER=raster \
  TILES_PER_TOKEN=1 \
  LATENT_PATCH=2 \
  COMPACT_FFT_TOKEN_DIM=16 \
  ROPE_MODE=sequence \
  SEED="${seed}" \
  STEPS="${steps}" \
  RUN_LABEL="c4-compact-fft-seqrope-s${seed}-${steps}" \
  "${project_root}/scripts/run_spatial_latent_hartley_ar_10k.sh"
