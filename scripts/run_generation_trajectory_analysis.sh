#!/usr/bin/env bash
# Fixed-seed direct sampler-trajectory analysis for the selected and final arms.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output="reports/2026-08-26_autoencoder_program/generation_trajectory"
mkdir -p "$output"

if [[ ! -f "$output/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job generation-trajectory-v27-v34 --wait -- \
    python -u scripts/analyze_generation_trajectory.py \
      --checkpoint v27=prior_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2-prior-s1/checkpoint_final.pt \
      --cache v27=tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt \
      --checkpoint v34-common=prior_runs/v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2-prior-common-s1/checkpoint_final.pt \
      --cache v34-common=tokenizer_runs/v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt \
      --checkpoint v34-soft25=prior_runs/v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2-prior-softsnr25-s1/checkpoint_final.pt \
      --cache v34-soft25=tokenizer_runs/v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt \
      --output_dir "$output" --num_samples 128 --preview_samples 4 \
      --batch_size 32 --steps 50 --pca_samples 10000 --seed 54321 \
      > "$output/analysis.log" 2>&1
fi

echo "GENERATION TRAJECTORY ANALYSIS COMPLETE $(date -u +%FT%TZ)"
