#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

tokenizer_run="tokenizer_runs/v2-cross-n32-d32-s1"
cache="${tokenizer_run}/latents_final_original_flip.pt"
output="prior_runs/v2-joint-cross-n32-d32-s1"

mkdir -p logs prior_runs

if [[ ! -f "$cache" ]]; then
  gpu-claim run \
    --owner AFIG \
    --job cache-cross-n32-d32-original-flip \
    --wait -- \
    /venv/main/bin/python -u cache_progressive_latents.py \
      --tokenizer_checkpoint "${tokenizer_run}/checkpoint_final.pt" \
      --output "$cache" \
      --batch_size 512 \
      --num_workers 4 \
      --include_horizontal_flip
fi

gpu-claim run \
  --owner AFIG \
  --job joint-cross-n32-d32-60k \
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
    --max_train_steps 60000 \
    --report_to wandb \
    --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-token-factorization \
    --run_name v2-joint-cross-n32-d32-s1 \
    --compile
