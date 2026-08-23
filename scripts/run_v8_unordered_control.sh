#!/bin/bash
# Matched no-prefix control: tokenizer -> cache -> joint prior -> decoded FID.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

name="v8-unordered-vae-s1"
tokenizer_dir="tokenizer_runs/${name}"
prior_dir="prior_runs/v8-joint-unordered-vae-s1"
evaluation_dir="prior_evals/v8-joint-unordered-vae-060000"

mkdir -p tokenizer_runs prior_runs prior_evals

gpu-claim run --owner AFIG --job "${name}-tokenizer" --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective full \
    --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 \
    --variational --kl_weight 1e-4 --hard_log_variance_clamp \
    --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group decisive-controls-v8 --run_name "$name" \
    --output_dir "$tokenizer_dir" \
    > "${tokenizer_dir}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${tokenizer_dir}/checkpoint_final.pt" "${name}-tokenizer" \
  >> "${tokenizer_dir}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "${name}-cache" --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint "${tokenizer_dir}/checkpoint_final.pt" \
    --output "${tokenizer_dir}/latents_final_original_flip.pt" \
    --include_horizontal_flip \
    > "${tokenizer_dir}/cache.log" 2>&1

python -u scripts/analyze_axis_scorecard.py \
  --cache "${tokenizer_dir}/latents_final_original_flip.pt" \
  --output "${tokenizer_dir}/axis_scorecard.json" \
  > "${tokenizer_dir}/axis_scorecard.log" 2>&1

gpu-claim run --owner AFIG --job "v8-joint-unordered-vae" --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "${tokenizer_dir}/latents_final_original_flip.pt" \
    --output_dir "$prior_dir" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group decisive-controls-v8 --run_name v8-joint-unordered-vae-s1 \
    --compile \
    > "${prior_dir}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${prior_dir}/checkpoint_final.pt" v8-joint-unordered-vae \
  >> "${prior_dir}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v8-eval-unordered-vae" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior_dir}/checkpoint_final.pt" \
    --output_dir "$evaluation_dir" \
    > "${evaluation_dir}.log" 2>&1

echo "UNORDERED CONTROL COMPLETE $(date -u +%FT%TZ)"
