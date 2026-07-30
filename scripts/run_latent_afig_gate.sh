#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
if [[ -z "${LATENT_INTERFACE:-}" ]]; then
  if [[ -f "${AE_RUN}/latent_interface.pt" ]]; then
    LATENT_INTERFACE="${AE_RUN}/latent_interface.pt"
  else
    LATENT_INTERFACE="${AE_RUN}/latent_interface_cpu.pt"
  fi
fi
SEED="${SEED:-1}"
STEPS="${STEPS:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/t12-seed${SEED}-n${STEPS}}"

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing autoencoder checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${LATENT_INTERFACE}" ]]; then
  echo "Missing latent interface: ${LATENT_INTERFACE}" >&2
  exit 1
fi

exec /venv/main/bin/python "${ROOT}/train_latent_continuous.py" \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_interface "${LATENT_INTERFACE}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max_train_steps "${STEPS}" \
  --run_name "latent-afig-t12-s${SEED}-n${STEPS}" \
  "$@"
