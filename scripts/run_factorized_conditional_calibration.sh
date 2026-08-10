#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

output_dir="diagnostics/polar_v4_conditional_calibration"
mkdir -p "${output_dir}"

for name in logeps_ref log1p_tau1 raw; do
  checkpoint="continuous_runs/ar_fft_polar_v4_ampcoord_${name}_s1_10000/checkpoint_10000.pt"
  /venv/main/bin/python -u diagnose_factorized_calibration.py \
    --checkpoint "${checkpoint}" \
    --output "${output_dir}/${name}.json" \
    --num_images 16 \
    --draws 4 \
    --steps 20 \
    --position_chunk 64 \
    --seed 20260807
done
