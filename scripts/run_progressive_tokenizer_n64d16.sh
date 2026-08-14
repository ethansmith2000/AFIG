#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG
mkdir -p logs tokenizer_runs

gpu-claim run \
  --owner AFIG \
  --job tokenizer-v2-cross-n64-d16 \
  --wait -- \
  /venv/main/bin/python -u train_progressive_tokenizer.py \
    --output_dir tokenizer_runs/v2-cross-n64-d16-s1 \
    --objective progressive \
    --pool_type cross_only \
    --pool_depth 1 \
    --num_latents 64 \
    --latent_dim 16 \
    --qk_norm rms \
    --train_batch_size 512 \
    --eval_batch_size 512 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_train_steps 15000 \
    --report_to wandb \
    --tracker_project_name afig-progressive-tokenizer \
    --run_group tokenizer-token-factorization \
    --run_name v2-cross-n64-d16-s1 \
    --compile
