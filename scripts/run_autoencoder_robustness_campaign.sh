#!/bin/bash
set -euo pipefail

mode="${1:?usage: run_autoencoder_robustness_campaign.sh MODE POOLER LATENT_DIM CONDITIONING [STEPS]}"
pooler="${2:?pooler required}"
latent_dim="${3:?latent dimension required}"
conditioning="${4:?conditioning required}"
steps="${5:-30000}"
launcher=/workspace/AFIG/scripts/run_autoencoder_gate.sh
pids=()

launch() {
  "$@" &
  pids+=("$!")
}

for seed in 0 1; do
  common=(
    env
    LATENT_DIM="${latent_dim}"
    POOLER="${pooler}"
    GROUP_CONDITIONING="${conditioning}"
    TOKEN_LOSS_WEIGHT=0.01
    RUN_GROUP=afig-ae-robustness-30k
  )
  launch "${common[@]}" VARIATIONAL=false \
    "${launcher}" "${mode}" "${seed}" "${steps}"
  launch "${common[@]}" VARIATIONAL=false LATENT_NOISE_STD=0.05 \
    "${launcher}" "${mode}" "${seed}" "${steps}"
  launch "${common[@]}" VARIATIONAL=false LATENT_RING_DROPOUT=0.02 \
    LATENT_HIGH_FREQUENCY_DROPOUT=0.05 \
    "${launcher}" "${mode}" "${seed}" "${steps}"
  launch "${common[@]}" VARIATIONAL=true KL_WEIGHT=0.0001 KL_FREE_BITS=0.01 \
    "${launcher}" "${mode}" "${seed}" "${steps}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
