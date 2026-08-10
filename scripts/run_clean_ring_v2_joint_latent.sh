#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z64-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
STEPS="${STEPS:-10000}"

export NORMALIZATION_SCOPE="tensor"
export STEPS
export AE_RUN
export AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
export LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface_tensor.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/joint-clean-ring-v2-mean-tensor-rf-w768-l12-b256-s1-n${STEPS}}"

exec "${ROOT}/scripts/run_joint_latent_normalization_scope.sh"
