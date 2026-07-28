#!/bin/bash
set -euo pipefail

steps="${1:-30000}"
launcher=/workspace/AFIG/scripts/run_autoencoder_gate.sh
pids=()

launch() {
  "$@" &
  pids+=("$!")
}

for seed in 0 1; do
  launch env LATENT_DIM=64 GROUP_SIZE=4 POOLER=flat_mlp \
    GROUP_CONDITIONING=film_low_rank TOKEN_LOSS_WEIGHT=0.01 \
    RUN_GROUP=afig-ae-followup-30k \
    "${launcher}" causal_k "${seed}" "${steps}"
  launch env LATENT_DIM=64 GROUP_SIZE=4 POOLER=perceiver_full \
    GROUP_CONDITIONING=film_low_rank TOKEN_LOSS_WEIGHT=0.01 \
    RUN_GROUP=afig-ae-followup-30k \
    "${launcher}" causal_k "${seed}" "${steps}"
  launch env LATENT_DIM=128 TARGET_TOKENS_PER_LATENT=16 MAX_RING_LATENTS=4 \
    POOLER=perceiver_sector GROUP_CONDITIONING=film TOKEN_LOSS_WEIGHT=0.01 \
    RUN_GROUP=afig-ae-followup-30k \
    "${launcher}" causal_ring "${seed}" "${steps}"
  launch env LATENT_DIM=128 TARGET_TOKENS_PER_LATENT=16 MAX_RING_LATENTS=4 \
    POOLER=perceiver_sector GROUP_CONDITIONING=low_rank CONDITIONING_RANK=32 \
    TOKEN_LOSS_WEIGHT=0.01 RUN_GROUP=afig-ae-followup-30k \
    "${launcher}" causal_ring "${seed}" "${steps}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
