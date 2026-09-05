#!/usr/bin/env bash
# Queue-managed construction of the one frozen factorized-whitened cache.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

source_cache="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt"
transform="reports/2026-08-26_autoencoder_program/regularized_whitening/selected_transform.pt"
output="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_factorized_whiten16_original_flip.pt"
log="reports/2026-08-26_autoencoder_program/whitened_cache_build.log"

if [[ ! -f "$output" ]]; then
  gpu-claim run --owner AFIG --job v27-factorized-whiten16-cache --wait -- \
    python -u scripts/build_whitened_prior_cache.py \
      --cache "$source_cache" --transform "$transform" --output "$output" \
      --chunk_size 2048 --device cuda > "$log" 2>&1
fi

echo "WHITENED CACHE BUILD COMPLETE $(date -u +%FT%TZ)"
