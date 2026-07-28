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
  launch env LATENT_DIM=64 GROUP_SIZE=4 POOLER=perceiver_full \
    PERCEIVER_WIDTH=256 PERCEIVER_HEADS=4 \
    GROUP_CONDITIONING=film_low_rank TOKEN_LOSS_WEIGHT=0.01 \
    RUN_GROUP=afig-ae-perceiver-v2-30k \
    "${launcher}" causal_k "${seed}" "${steps}"
  launch env LATENT_DIM=64 TARGET_TOKENS_PER_LATENT=16 MAX_RING_LATENTS=4 \
    POOLER=perceiver_sector PERCEIVER_WIDTH=256 PERCEIVER_HEADS=4 \
    GROUP_CONDITIONING=film_low_rank TOKEN_LOSS_WEIGHT=0.01 \
    RUN_GROUP=afig-ae-perceiver-v2-30k \
    "${launcher}" causal_ring "${seed}" "${steps}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
