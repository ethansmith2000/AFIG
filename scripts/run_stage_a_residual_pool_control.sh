#!/bin/bash
# Parameter-matched encoder allocation: move one patch block into register refinement.
set -euo pipefail

cd /workspace/AFIG
source /venv/main/bin/activate

name="v12-unordered-vae-residual-e7p1-n64d16-s1"
tokenizer="tokenizer_runs/${name}"
mkdir -p tokenizer_runs

gpu-claim run --owner AFIG --job "${name}-tokenizer" --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective full \
    --encoder_depth 7 --pool_type residual --pool_depth 1 \
    --num_latents 64 --latent_dim 16 \
    --variational --kl_weight 1e-4 --hard_log_variance_clamp \
    --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group autoencoder-stage-a --run_name "$name" \
    --output_dir "$tokenizer" \
    > "${tokenizer}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${tokenizer}/checkpoint_final.pt" "${name}-tokenizer" \
  >> "${tokenizer}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "${name}-cache" --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint "${tokenizer}/checkpoint_final.pt" \
    --output "${tokenizer}/latents_final_original_flip.pt" \
    --include_horizontal_flip \
    > "${tokenizer}/cache.log" 2>&1

python -u scripts/analyze_axis_scorecard.py \
  --cache "${tokenizer}/latents_final_original_flip.pt" \
  --output "${tokenizer}/axis_scorecard.json" \
  > "${tokenizer}/axis_scorecard.log" 2>&1

gpu-claim run --owner AFIG --job "${name}-sensitivity" --wait -- \
  python -u scripts/decoder_sensitivity.py \
    --cache "${tokenizer}/latents_final_original_flip.pt" \
    --output "${tokenizer}/decoder_sensitivity.json" \
    > "${tokenizer}/decoder_sensitivity.log" 2>&1

echo "STAGE A RESIDUAL POOL CONTROL COMPLETE $(date -u +%FT%TZ)"
