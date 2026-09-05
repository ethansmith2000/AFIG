#!/usr/bin/env bash
# Queue-managed Phase-I whitening feasibility audit; no model training.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output="reports/2026-08-26_autoencoder_program/regularized_whitening"
cache="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt"
checkpoint="prior_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2-prior-s1/checkpoint_final.pt"
mkdir -p "$output"

if [[ ! -f "$output/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job v27-regularized-whitening-audit --wait -- \
    python -u scripts/analyze_regularized_whitening.py \
      --cache "$cache" \
      --prior_checkpoint "$checkpoint" \
      --output_dir "$output" \
      --fit_samples 25000 --eval_samples 10000 \
      --roundtrip_samples 1024 --decode_samples 128 --batch_size 64 \
      --gain_caps 4,8,16,32 --betas 0,0.125,0.25,0.5 \
      --device cuda \
      > "$output/analysis.log" 2>&1
fi

echo "REGULARIZED WHITENING AUDIT COMPLETE $(date -u +%FT%TZ)"
