#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z32-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface_tensor.pt}"

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing z32 autoencoder checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${AFIG_GPU_WORKER:-0}" != "1" ]]; then
  exec /workspace/bin/gpu-claim run \
    --owner AFIG \
    --job clean-ring-v2-z32-interface \
    --wait \
    -- \
    env AFIG_GPU_WORKER=1 "${BASH_SOURCE[0]}"
fi

cd "${ROOT}"
exec /venv/main/bin/python -u fit_autoencoder_latent_interface.py \
  --checkpoint "${AE_CHECKPOINT}" \
  --data_root "${ROOT}/data" \
  --batch_size 128 \
  --num_batches 64 \
  --probe_steps 500 \
  --normalization_scope tensor \
  --output "${LATENT_INTERFACE}"
