#!/bin/bash
# Queue-managed fixed-cache PCA oracle; no prior training in this phase.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

output_dir="reports/2026-08-26_autoencoder_program/pca_oracle_v9_n64d48"
basis="tokenizer_runs/v9-unordered-vae-n64d48-s1/pca_basis_25k.pt"
mkdir -p "$output_dir"

gpu-claim run --owner AFIG --job e5-pca-oracle-v9-n64d48 --wait -- \
  python -u scripts/evaluate_pca_truncation_oracle.py \
    --cache tokenizer_runs/v9-unordered-vae-n64d48-s1/latents_final_original_flip.pt \
    --basis_output "$basis" \
    --output_dir "$output_dir" \
    > "${output_dir}.log" 2>&1

echo "E5 PCA ORACLE COMPLETE $(date -u +%FT%TZ)"
