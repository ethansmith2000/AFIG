#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

checkpoint="/workspace/AFIG/tokenizer_runs/n32-d64-full-s1/checkpoint_latest.pt"
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing resumable checkpoint: ${checkpoint}" >&2
  exit 1
fi

exec /workspace/AFIG/scripts/run_progressive_tokenizer_full.sh \
  --resume "${checkpoint}" \
  --learning_rate 5e-5 \
  "$@"
