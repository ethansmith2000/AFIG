#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/scaffold_fft_causal_ring_local_oracle_c4_s1_10000
smoke_dir=${project_root}/diagnostics/scaffold_fft_causal_ring_local_cuda_smoke

cd "${project_root}"
/venv/main/bin/python -u train_scaffold_fft_causal_ring_local.py \
  --output_dir "${smoke_dir}" \
  --device cuda \
  --smoke

exec /venv/main/bin/python -u train_scaffold_fft_causal_ring_local.py \
  --init_checkpoint latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000/checkpoint_30000.pt \
  --output_dir "${output_dir}" \
  --steps 10000 \
  --batch_size 128 \
  --learning_rate 5e-5 \
  --weight_decay 0.1 \
  --warmup 500 \
  --validation_images 8 \
  --preview_steps 2500 \
  --checkpoint_steps 2500 \
  --inference_steps 20 \
  --seed 1
