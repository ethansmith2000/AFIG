#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z32-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
export CODEC_TAG="${CODEC_TAG:-clean-ring-v2-z32-tensor}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/grouped-ring-clean-ring-v2-z32-tensor-w768-l12-d6-s1-n10000}"

exec "${ROOT}/scripts/run_clean_ring_v2_latent_afig.sh" "$@"
