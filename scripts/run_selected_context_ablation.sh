#!/usr/bin/env bash
# Directly test whether selected-v27 early PCA/token context helps late prediction.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output="reports/2026-08-26_autoencoder_program/generation_trajectory/v27_context_ablation.json"
log="reports/2026-08-26_autoencoder_program/generation_trajectory/v27_context_ablation.log"
if [[ ! -f "$output" ]]; then
  gpu-claim run --owner AFIG --job v27-context-ablation --wait -- \
    python -u scripts/conditioning_context_ablation.py \
      --checkpoint prior_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2-prior-s1/checkpoint_final.pt \
      --cache tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt \
      --output "$output" --seed 1729 --pca_samples 20000 \
      --eval_samples 2048 --batch_size 128 \
      --times 0.2,0.35,0.5,0.65,0.8 --early_snr 4 --late_snr 0.25 \
      --max_band_dims 256 --token_prefixes 8,16,32 \
      > "$log" 2>&1
fi

echo "V27 CONTEXT ABLATION COMPLETE $(date -u +%FT%TZ)"
