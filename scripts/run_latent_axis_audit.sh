#!/usr/bin/env bash
# Analysis-only v27 latent-axis geometry and known-clean denoising audit.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output="reports/2026-08-26_autoencoder_program/latent_axis_audit"
basis="/tmp/afig-v27-latent-axis-basis.pt"
cache="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt"
checkpoint="prior_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2-prior-s1/checkpoint_final.pt"
mkdir -p "$output"

if [[ ! -f "$output/geometry.json" || ! -f "$basis" ]]; then
  gpu-claim run --owner AFIG --job v27-latent-axis-geometry --wait -- \
    python -u scripts/analyze_latent_axis_geometry.py \
      --cache "$cache" \
      --prior_checkpoint "$checkpoint" \
      --output_dir "$output" \
      --basis_output "$basis" \
      --fit_samples 25000 --eval_samples 10000 \
      --role_samples 256 --preview_samples 2 --batch_size 64 \
      --seed 1729 --device cuda \
      > "$output/geometry.log" 2>&1
fi

if [[ ! -f "$output/known_clean_denoising.json" ]]; then
  gpu-claim run --owner AFIG --job v27-known-clean-denoising --wait -- \
    python -u scripts/analyze_known_clean_denoising.py \
      --cache "$cache" \
      --checkpoint "$checkpoint" \
      --basis "$basis" \
      --output_dir "$output" \
      --eval_samples 2048 --decoded_samples 256 \
      --preview_samples 4 --batch_size 128 \
      --times 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
      --seed 1729 --device cuda \
      > "$output/known_clean_denoising.log" 2>&1
fi

echo "LATENT AXIS AUDIT COMPLETE $(date -u +%FT%TZ)"
