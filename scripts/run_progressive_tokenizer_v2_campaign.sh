#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

max_steps="${1:-15000}"
batch_size="${2:-512}"
mkdir -p logs tokenizer_runs

run_arm() {
  local name="$1"
  local pool_type="$2"
  local pool_depth="$3"
  local num_latents="$4"
  local latent_dim="$5"

  gpu-claim run \
    --owner AFIG \
    --job "tokenizer-v2-${name}" \
    --wait -- \
    /venv/main/bin/python -u train_progressive_tokenizer.py \
      --output_dir "tokenizer_runs/v2-${name}-s1" \
      --objective progressive \
      --pool_type "$pool_type" \
      --pool_depth "$pool_depth" \
      --num_latents "$num_latents" \
      --latent_dim "$latent_dim" \
      --qk_norm rms \
      --train_batch_size "$batch_size" \
      --eval_batch_size "$batch_size" \
      --num_workers 4 \
      --learning_rate 1e-4 \
      --warmup_steps 1000 \
      --max_train_steps "$max_steps" \
      --compile
}

names=(
  residual-n32-d64
  cross-n32-d64
  cross-n16-d64
  cross-n32-d32
)
pids=()

run_arm residual-n32-d64 residual 2 32 64 \
  >logs/tokenizer_v2_residual_n32_d64.log 2>&1 &
pids+=("$!")
run_arm cross-n32-d64 cross_only 1 32 64 \
  >logs/tokenizer_v2_cross_n32_d64.log 2>&1 &
pids+=("$!")
run_arm cross-n16-d64 cross_only 1 16 64 \
  >logs/tokenizer_v2_cross_n16_d64.log 2>&1 &
pids+=("$!")
run_arm cross-n32-d32 cross_only 1 32 32 \
  >logs/tokenizer_v2_cross_n32_d32.log 2>&1 &
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
