#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000
log_dir=${project_root}/logs

mkdir -p "${output_dir}" "${log_dir}"
cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job scaffold-fft-residual-oracle-c4-30k \
  --wait \
  -- \
  /venv/main/bin/python -u train_scaffold_fft_residual.py \
  --ae_checkpoint autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt \
  --output_dir "${output_dir}" \
  --steps 30000 \
  --batch_size 256 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --preview_steps 5000 \
  --checkpoint_steps 5000 \
  --inference_steps 50 \
  --seed 1
