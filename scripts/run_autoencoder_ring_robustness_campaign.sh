#!/bin/bash
set -euo pipefail

launcher="$(dirname "$0")/run_autoencoder_gate.sh"
steps="${STEPS:-30000}"
pids=()

launch() {
  "$@" &
  pids+=("$!")
}

for seed in 0 1; do
  common=(
    env
    LATENT_DIM=64
    TARGET_TOKENS_PER_LATENT=12
    MAX_RING_LATENTS=8
    POOLER=perceiver_sector
    PERCEIVER_WIDTH=256
    PERCEIVER_HEADS=4
    RING_TRANSFORMER_LAYERS=2
    GROUP_CONDITIONING=film_low_rank
    TOKEN_LOSS_WEIGHT=0.01
    LATENT_MOMENT_WEIGHT=0.0001
    RUN_GROUP=afig-ae-ring-t12-robustness-30k
  )
  launch "${common[@]}" VARIATIONAL=false LATENT_NOISE_STD=0.05 \
    "${launcher}" causal_ring "${seed}" "${steps}"
  launch "${common[@]}" VARIATIONAL=false LATENT_RING_DROPOUT=0.02 \
    LATENT_HIGH_FREQUENCY_DROPOUT=0.05 \
    "${launcher}" causal_ring "${seed}" "${steps}"
  launch "${common[@]}" VARIATIONAL=true KL_WEIGHT=0.0001 KL_FREE_BITS=0.01 \
    "${launcher}" causal_ring "${seed}" "${steps}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
