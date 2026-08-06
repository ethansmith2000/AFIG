#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/scaffold_fft_ring_residual_oracle_c4_s1_10000
log_dir=${project_root}/logs

mkdir -p "${output_dir}" "${log_dir}"
cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job scaffold-fft-ring-residual-oracle-c4-10k \
  --wait \
  -- \
  /venv/main/bin/python -u train_scaffold_fft_ring_residual.py \
  --init_checkpoint latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000/checkpoint_30000.pt \
  --output_dir "${output_dir}" \
  --steps 10000 \
  --batch_size 128 \
  --width 768 \
  --scaffold_layers 4 \
  --ring_layers 8 \
  --num_heads 12 \
  --diffusion_width 768 \
  --diffusion_depth 6 \
  --diffusion_batch_mul 1 \
  --preview_steps 2500 \
  --checkpoint_steps 2500 \
  --inference_steps 20 \
  --seed 1
