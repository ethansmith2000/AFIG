#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
checkpoint=${project_root}/latent_continuous_runs/scaffold_fft_causal_ring_local_oracle_c4_s1_10000/checkpoint_10000.pt
output_dir=${project_root}/diagnostics/scaffold_fft_causal_ring_local_oracle_c4_10000_shuffled

mkdir -p "${output_dir}"
cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job scaffold-fft-causal-ring-local-eval10k-shuffled \
  --wait \
  -- \
  /venv/main/bin/python -u evaluate_scaffold_fft_causal_ring_local.py \
  --checkpoint "${checkpoint}" \
  --output_dir "${output_dir}" \
  --num_samples 5000 \
  --batch_size 64 \
  --num_inference_steps 20 \
  --seed 71001 \
  --condition_mode shuffled
