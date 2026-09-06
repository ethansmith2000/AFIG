#!/usr/bin/env bash
# Frozen larger-sample evaluations for qualifying axial-ZCA factorial arms.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {common_uniform|ordered_uniform|ordered_weighted}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

case "$1" in
  common_uniform)
    name="v37-v27-axial-zca1-common-uniform-prior-s1"
    ;;
  ordered_uniform)
    name="v37-v27-axial-zca1-ordered-uniform-prior-s1"
    ;;
  ordered_weighted)
    name="v37-v27-axial-zca1-ordered-flowweight-prior-s1"
    ;;
  *)
    echo "arm did not satisfy the frozen 10k continuation rule: $1" >&2
    exit 2
    ;;
esac

checkpoint="prior_runs/${name}/checkpoint_final.pt"
evaluation="prior_evals/${name}-060000-n10k"
mkdir -p "$evaluation"
[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-eval10k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "$checkpoint" --output_dir "$evaluation" \
      --num_samples 10000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "ZCA 10K $1 COMPLETE $(date -u +%FT%TZ)"
