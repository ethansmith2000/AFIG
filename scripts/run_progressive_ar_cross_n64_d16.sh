#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG
mkdir -p logs prior_runs prior_evals

cache=tokenizer_runs/v2-cross-n64-d16-s1/latents_final_original_flip.pt
output=prior_runs/v2-ar-cross-n64-d16-s1

gpu-claim run \
  --owner AFIG \
  --job ar-cross-n64-d16-60k \
  --wait -- \
  /venv/main/bin/python -u train_progressive_ar_flow.py \
    --latent_cache "$cache" \
    --output_dir "$output" \
    --width 512 \
    --trunk_depth 12 \
    --head_depth 6 \
    --num_heads 8 \
    --qk_norm rms \
    --batch_size 256 \
    --num_workers 4 \
    --learning_rate 1e-4 \
    --warmup_steps 1000 \
    --max_train_steps 60000 \
    --report_to wandb \
    --tracker_project_name afig-progressive-tokenizer \
    --run_group ar-prior-token-factorization \
    --run_name v2-ar-cross-n64-d16-s1 \
    --compile

gpu-claim run \
  --owner AFIG \
  --job ar-eval60k-cross-n64-d16 \
  --wait -- \
  /venv/main/bin/python -u evaluate_progressive_joint_flow.py \
    --checkpoint "$output/checkpoint_final.pt" \
    --output_dir prior_evals/v2-ar-cross-n64-d16-060000 \
    --num_samples 5000 \
    --batch_size 256 \
    --sample_steps 50
