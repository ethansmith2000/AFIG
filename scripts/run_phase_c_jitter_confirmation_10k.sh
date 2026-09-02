#!/usr/bin/env bash
# Larger-sample evaluation for Phase-C decoder-jitter confirmation arms.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {tokenizer_s1|tokenizer_s3|prior2_v12|prior2_v20}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  tokenizer_s1)
    name="v23-residual-e7p1-det-jitter05-n64d16-s1-prior-s1"
    ;;
  tokenizer_s3)
    name="v24-residual-e7p1-det-jitter05-n64d16-s3-prior-s1"
    ;;
  prior2_v12)
    name="v12-joint-residual-e7p1-n64d16-tokenizer-s2-prior-s2"
    ;;
  prior2_v20)
    name="v20-residual-e7p1-det-jitter05-n64d16-s2-prior-s2"
    ;;
  *)
    echo "usage: $0 {tokenizer_s1|tokenizer_s3|prior2_v12|prior2_v20}" >&2
    exit 2
    ;;
esac

checkpoint="prior_runs/${name}/checkpoint_final.pt"
evaluation="prior_evals/${name}-060000-n10k"
mkdir -p "$evaluation"

if [[ ! -f "$checkpoint" ]]; then
  echo "missing completed prior checkpoint: $checkpoint" >&2
  exit 1
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-eval10k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "$checkpoint" --output_dir "$evaluation" \
      --num_samples 10000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "PHASE C JITTER CONFIRMATION 10K $1 COMPLETE $(date -u +%FT%TZ)"
