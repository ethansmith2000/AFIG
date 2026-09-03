#!/usr/bin/env bash
# Larger-sample evaluation for qualifying register-jitter factorial arms.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "s1" && "$1" != "s2" && "$1" != "s3" && "$1" != "residual_s1" ) ]]; then
  echo "usage: $0 {s1|s2|s3|residual_s1}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  s1|s2|s3)
    seed="${1#s}"
    name="v25-register-e7j1-det-jitter05-n64d16-s${seed}-prior-s1"
    ;;
  residual_s1)
    name="v23-residual-e7p1-det-jitter05-n64d16-s1-prior-s1"
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

echo "REGISTER JITTER FACTORIAL 10K $1 COMPLETE $(date -u +%FT%TZ)"
