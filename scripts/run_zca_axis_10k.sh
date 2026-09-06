#!/usr/bin/env bash
# Frozen 10k evaluations for qualifying cells of the expanded ZCA-axis matrix.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {channel|sequence|flattened} {common_uniform|ordered_uniform|common_weighted|ordered_weighted}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

variant="$1"
arm="$2"
case "${variant}/${arm}" in
  channel/common_uniform|channel/ordered_uniform|channel/common_weighted|channel/ordered_weighted) ;;
  sequence/common_uniform|sequence/ordered_uniform|sequence/common_weighted|sequence/ordered_weighted) ;;
  flattened/ordered_uniform|flattened/common_weighted|flattened/ordered_weighted) ;;
  *) echo "cell did not satisfy the frozen 10k rule: ${variant}/${arm}" >&2; exit 2 ;;
esac

case "$arm" in
  common_uniform) suffix="common-uniform" ;;
  ordered_uniform) suffix="ordered-uniform" ;;
  common_weighted) suffix="common-flowweight" ;;
  ordered_weighted) suffix="ordered-flowweight" ;;
esac

name="v38-v27-${variant}-zca1-${suffix}-prior-s1"
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

echo "ZCA AXIS 10K ${variant}/${arm} COMPLETE $(date -u +%FT%TZ)"
