#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
arm="${ARM:-local_dct}"
seed="${SEED:-1}"
steps="${STEPS:-30000}"
ae_checkpoint="${AE_CHECKPOINT:-${project_root}/autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt}"

case "${arm}" in
  local_dct)
    latent_basis=patch_dct
    ;;
  local_dct_freqmajor)
    latent_basis=patch_dct_freq_major
    ;;
  full_dct)
    latent_basis=full_dct_tiles
    ;;
  full_hartley)
    latent_basis=hartley
    ;;
  *)
    echo "ARM must be local_dct, local_dct_freqmajor, full_dct, or full_hartley" >&2
    exit 2
    ;;
esac

output_dir="${OUTPUT_DIR:-${project_root}/continuous_runs/joint_c4_rate_${arm}_s${seed}_${steps}}"

cd "${project_root}"
exec env \
  AE_CHECKPOINT="${ae_checkpoint}" \
  OUTPUT_DIR="${output_dir}" \
  LATENT_BASIS="${latent_basis}" \
  SEED="${seed}" \
  STEPS="${steps}" \
  RUN_LABEL="c4-joint-rate-${arm}-s${seed}-${steps}" \
  "${project_root}/scripts/run_spatial_latent_hartley_joint_10k.sh"
