#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

exec gpu-claim run \
  --owner AFIG \
  --job tokenizer-n32-d64-full-s1 \
  --wait -- \
  python -u train_progressive_tokenizer.py \
    --output_dir tokenizer_runs/n32-d64-full-s1 \
    --objective full \
    "$@"
