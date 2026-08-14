#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG
mkdir -p logs prior_runs

run_arm() {
  local name="$1"
  local tokenizer_run="$2"
  local cache="tokenizer_runs/${tokenizer_run}/latents_final_original_flip.pt"
  local output="prior_runs/v2-joint-${name}-s1"

  gpu-claim run \
    --owner AFIG \
    --job "cache-${name}-original-flip" \
    --wait -- \
    /venv/main/bin/python -u cache_progressive_latents.py \
      --tokenizer_checkpoint "tokenizer_runs/${tokenizer_run}/checkpoint_final.pt" \
      --output "$cache" \
      --batch_size 512 \
      --num_workers 4 \
      --include_horizontal_flip

  gpu-claim run \
    --owner AFIG \
    --job "joint-${name}" \
    --wait -- \
    /venv/main/bin/python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" \
      --output_dir "$output" \
      --width 512 \
      --depth 12 \
      --num_heads 8 \
      --qk_norm rms \
      --batch_size 256 \
      --num_workers 4 \
      --learning_rate 1e-4 \
      --warmup_steps 1000 \
      --max_train_steps 20000 \
      --run_group joint-prior-v2-n16d64 \
      --compile
}

names=(residual-n16-d64 cross-n16-d64)
pids=()

run_arm residual-n16-d64 v2-residual-n16-d64-s1 \
  >logs/v2_joint_residual_n16_d64.log 2>&1 &
pids+=("$!")
run_arm cross-n16-d64 v2-cross-n16-d64-s1 \
  >logs/v2_joint_cross_n16_d64.log 2>&1 &
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
