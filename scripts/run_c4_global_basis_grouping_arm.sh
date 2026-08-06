#!/bin/bash
set -euo pipefail

project_root=/workspace/AFIG
arm="${ARM:-hartley_quartets}"
seed="${SEED:-7}"
steps="${STEPS:-10000}"
ae_checkpoint="${AE_CHECKPOINT:-${project_root}/autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt}"

case "${arm}" in
  dct_tiles)
    latent_basis=full_dct_tiles
    latent_patch=2
    tiles_per_token=1
    ;;
  hartley_quartets)
    latent_basis=hartley
    latent_patch=1
    tiles_per_token=4
    ;;
  hartley_tiles)
    latent_basis=hartley
    latent_patch=2
    tiles_per_token=1
    ;;
  *)
    echo "ARM must be dct_tiles, hartley_quartets, or hartley_tiles" >&2
    exit 2
    ;;
esac

output_dir="${OUTPUT_DIR:-${project_root}/continuous_runs/ar_c4_${arm}_seqrope_s${seed}_${steps}}"
run_label="c4-${arm}-seqrope-s${seed}-${steps}"

cd "${project_root}"
exec env \
  AE_CHECKPOINT="${ae_checkpoint}" \
  OUTPUT_DIR="${output_dir}" \
  LATENT_BASIS="${latent_basis}" \
  TOKEN_ORDER=auto \
  TILES_PER_TOKEN="${tiles_per_token}" \
  LATENT_PATCH="${latent_patch}" \
  ROPE_MODE=sequence \
  SEED="${seed}" \
  STEPS="${steps}" \
  RUN_LABEL="${run_label}" \
  "${project_root}/scripts/run_spatial_latent_hartley_ar_10k.sh"
