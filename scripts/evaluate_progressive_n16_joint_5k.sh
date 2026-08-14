#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG
mkdir -p logs prior_evals

evaluate_arm() {
  local name="$1"

  gpu-claim run \
    --owner AFIG \
    --job "joint-eval5k-${name}" \
    --wait -- \
    /venv/main/bin/python -u evaluate_progressive_joint_flow.py \
      --checkpoint "prior_runs/v2-joint-${name}-s1/checkpoint_005000.pt" \
      --output_dir "prior_evals/v2-joint-${name}-005000" \
      --num_samples 5000 \
      --batch_size 256 \
      --sample_steps 50
}

names=(residual-n16-d64 cross-n16-d64)
pids=()

evaluate_arm residual-n16-d64 >logs/v2_joint_residual_n16_d64_eval5k.log 2>&1 &
pids+=("$!")
evaluate_arm cross-n16-d64 >logs/v2_joint_cross_n16_d64_eval5k.log 2>&1 &
pids+=("$!")

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${names[$index]}"
  else
    arm_status="$?"
    echo "failed ${names[$index]} status=${arm_status}" >&2
    status="$arm_status"
  fi
done
exit "$status"
