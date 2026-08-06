#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/continuous_runs/ar_hartley_tiles_10k

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job ar_hartley-tiles-10000 \
  --wait \
  -- \
  /venv/main/bin/python train_hartley_ar.py \
  --output_dir "${output_dir}" \
  --data_root "${project_root}/data" \
  --seed 1 \
  --steps 10000 \
  --batch_size 32 \
  --num_workers 4 \
  --learning_rate 7e-5 \
  --warmup 500 \
  --weight_decay 0.1 \
  --width 768 \
  --layers 10 \
  --heads 12 \
  --ff_mult 4 \
  --diff_width 768 \
  --diff_depth 3 \
  --inference_steps 20 \
  --preview_steps 2500 \
  --diagnostic_steps 250 \
  --checkpoint_steps 2500 \
  --validation_images 16 \
  --patch 4
