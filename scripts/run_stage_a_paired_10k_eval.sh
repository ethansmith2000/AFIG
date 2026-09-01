#!/usr/bin/env bash
# Larger-sample follow-up for the close tokenizer-seed architecture comparisons.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {s2_v8|s2_v12|s3_v12|s3_v13}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  s2_v8)
    name="v8-joint-unordered-vae-tokenizer-s2-prior-s1"
    ;;
  s2_v12)
    name="v12-joint-residual-e7p1-n64d16-tokenizer-s2-prior-s1"
    ;;
  s3_v12)
    name="v12-joint-residual-e7p1-n64d16-tokenizer-s3-prior-s1"
    ;;
  s3_v13)
    name="v13-joint-register-e7j1-n64d16-tokenizer-s3-prior-s1"
    ;;
  *)
    echo "usage: $0 {s2_v8|s2_v12|s3_v12|s3_v13}" >&2
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
      --checkpoint "$checkpoint" \
      --output_dir "$evaluation" \
      --num_samples 10000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "PAIRED 10K EVALUATION $1 COMPLETE $(date -u +%FT%TZ)"
