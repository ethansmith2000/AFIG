#!/usr/bin/env bash
# Larger-sample evaluation only for arms admitted by the frozen 5k gate.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "control" && "$1" != "soft25" ) ]]; then
  echo "usage: $0 {control|soft25}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

tokenizer_name="v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2"
case "$1" in
  control) prior_name="${tokenizer_name}-prior-common-s1" ;;
  soft25) prior_name="${tokenizer_name}-prior-softsnr25-s1" ;;
esac
prior="prior_runs/${prior_name}"
evaluation="prior_evals/${prior_name}-060000-n10k"
mkdir -p "$evaluation"

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  echo "missing completed prior: ${prior}/checkpoint_final.pt" >&2
  exit 1
fi
if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${prior_name}-eval10k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" --output_dir "$evaluation" \
      --num_samples 10000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "INPUT REGISTER SNR $1 10K COMPLETE $(date -u +%FT%TZ)"
