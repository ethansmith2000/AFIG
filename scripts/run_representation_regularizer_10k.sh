#!/usr/bin/env bash
# Larger-sample follow-up for a qualifying weak representation regularizer.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "marginal" && "$1" != "slot" ) ]]; then
  echo "usage: $0 {marginal|slot}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  marginal)
    name="v26-residual-e7p1-det-jitter05-kurt1e4-n64d16-s1-prior-s1"
    ;;
  slot)
    name="v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s1-prior-s1"
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

echo "REPRESENTATION REGULARIZER 10K $1 COMPLETE $(date -u +%FT%TZ)"
