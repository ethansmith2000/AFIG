#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

exec gpu-claim run \
  --owner AFIG \
  --job tokenizer-n32-d64-prefix-s1 \
  --wait -- \
  python -u train_progressive_tokenizer.py \
    --output_dir tokenizer_runs/n32-d64-prefix-s1 \
    --objective progressive \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    "$@"
