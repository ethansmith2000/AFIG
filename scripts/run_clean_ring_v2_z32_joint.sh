#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z32-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/joint-clean-ring-v2-z32-modern-opt-lr4e4-b095-b099-matrixwd-const-w768-l12-b256-s1-n20000}"

exec "${ROOT}/scripts/run_clean_ring_v2_joint_optimizer_arm.sh" lr4e4-b095-b099
