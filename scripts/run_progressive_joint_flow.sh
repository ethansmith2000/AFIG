#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

exec gpu-claim run \
  --owner AFIG \
  --job progressive-joint-flow-s1 \
  --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/n32-d64-prefix-s1/latents_012500.pt \
    --output_dir prior_runs/joint-flow-n32-d64-s1 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_train_steps 20000 \
    "$@"
