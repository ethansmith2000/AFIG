#!/usr/bin/env bash
# Queue-managed weak-tail and smooth power-whitening analysis; no training.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output="reports/2026-08-26_autoencoder_program/power_whitening"
cache="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt"
transform="reports/2026-08-26_autoencoder_program/regularized_whitening/selected_transform.pt"
mkdir -p "$output"

if [[ ! -f "$output/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job v27-power-whitening-audit --wait -- \
    python -u scripts/analyze_power_whitening.py \
      --cache "$cache" --transform "$transform" --output_dir "$output" \
      --fit_samples 25000 --eval_samples 10000 \
      --roundtrip_samples 1024 --decode_samples 128 --batch_size 64 \
      --gammas 0,0.125,0.25,0.5,0.75,1 --device cuda \
      > "$output/analysis.log" 2>&1
fi

echo "POWER WHITENING AUDIT COMPLETE $(date -u +%FT%TZ)"
