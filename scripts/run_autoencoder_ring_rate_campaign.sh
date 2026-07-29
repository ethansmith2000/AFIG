#!/bin/bash
set -euo pipefail

launcher="$(dirname "$0")/run_autoencoder_gate.sh"
steps="${STEPS:-30000}"
pids=()

for seed in 0 1; do
  for specification in "8:8" "12:8" "16:4"; do
    target="${specification%%:*}"
    cap="${specification##*:}"
    env LATENT_DIM=64 TARGET_TOKENS_PER_LATENT="${target}" \
      MAX_RING_LATENTS="${cap}" POOLER=perceiver_sector \
      PERCEIVER_WIDTH=256 PERCEIVER_HEADS=4 RING_TRANSFORMER_LAYERS=2 \
      GROUP_CONDITIONING=film_low_rank TOKEN_LOSS_WEIGHT=0.01 \
      LATENT_MOMENT_WEIGHT=0.0001 RUN_GROUP=afig-ae-ring-rate-30k \
      "${launcher}" causal_ring "${seed}" "${steps}" &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
