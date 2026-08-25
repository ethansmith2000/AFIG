#!/bin/bash
# No-dimensional-bottleneck control: 64x48 unordered tokenizer -> cache ->
# matched joint prior -> decoded FID -> decoder sensitivity.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

name="v9-unordered-vae-n64d48-s1"
tokenizer_dir="tokenizer_runs/${name}"
prior_dir="prior_runs/v9-joint-unordered-vae-n64d48-s1"
evaluation_dir="prior_evals/v9-joint-unordered-vae-n64d48-060000"

mkdir -p tokenizer_runs prior_runs prior_evals

gpu-claim run --owner AFIG --job "${name}-tokenizer" --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective full \
    --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 48 \
    --variational --kl_weight 1e-4 --hard_log_variance_clamp \
    --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group latent-rate-v9 --run_name "$name" \
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

gpu-claim run --owner AFIG --job "v9-joint-unordered-vae-n64d48" --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "${tokenizer_dir}/latents_final_original_flip.pt" \
    --output_dir "$prior_dir" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group latent-rate-v9 --run_name v9-joint-unordered-vae-n64d48-s1 \
    --compile \
    > "${prior_dir}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${prior_dir}/checkpoint_final.pt" v9-joint-unordered-vae-n64d48 \
  >> "${prior_dir}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v9-eval-unordered-vae-n64d48" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior_dir}/checkpoint_final.pt" \
    --output_dir "$evaluation_dir" \
    > "${evaluation_dir}.log" 2>&1

gpu-claim run --owner AFIG --job "v9-unordered-n64d48-decoder-sensitivity" --wait -- \
  python -u scripts/decoder_sensitivity.py \
    --cache "${tokenizer_dir}/latents_final_original_flip.pt" \
    --output "${tokenizer_dir}/decoder_sensitivity.json" \
    > "${tokenizer_dir}/decoder_sensitivity.log" 2>&1

echo "UNORDERED 64x48 RATE CONTROL COMPLETE $(date -u +%FT%TZ)"
