#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

gpu-claim run \
  --owner AFIG \
  --job tokenizer-v2-residual-n16-d64 \
  --wait -- \
  /venv/main/bin/python -u train_progressive_tokenizer.py \
    --output_dir tokenizer_runs/v2-residual-n16-d64-s1 \
    --objective progressive \
    --pool_type residual \
    --pool_depth 2 \
    --num_latents 16 \
    --latent_dim 64 \
    --qk_norm rms \
    --train_batch_size 512 \
    --eval_batch_size 512 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_train_steps 15000 \
    --run_group tokenizer-v2-followup \
    --compile
