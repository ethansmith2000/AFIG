#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z64-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
SEED="${SEED:-1}"
STEPS="${STEPS:-10000}"
CODEC_TAG="${CODEC_TAG:-clean-ring-v2-tensor}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/grouped-ring-${CODEC_TAG}-w768-l12-d6-s${SEED}-n${STEPS}}"

export AE_RUN
export AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
export LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface_tensor.pt}"
export CODEC_TAG
export GENERATION_GROUPING="ring"
export SEED
export STEPS
export OUTPUT_DIR

exec gpu-claim run \
  --owner AFIG \
  --job "clean-ring-v2-latent-afig-s${SEED}-n${STEPS}" \
  --wait \
  -- "${ROOT}/scripts/run_ring_block_latent_afig.sh" "$@"
