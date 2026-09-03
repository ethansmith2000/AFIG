#!/usr/bin/env bash
# Larger-sample follow-up for a qualifying learned latent-shape arm.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "n32d32" && "$1" != "n128d8" ) ]]; then
  echo "usage: $0 {n32d32|n128d8}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  n32d32)
    name="v30-residual-e7p1-det-jitter05-slotbal2e3-n32d32-s2-prior-s1"
    ;;
  n128d8)
    name="v31-residual-e7p1-det-jitter05-slotbal2e3-n128d8-s2-prior-s1"
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

echo "LEARNED LATENT SHAPE 10K $1 COMPLETE $(date -u +%FT%TZ)"
