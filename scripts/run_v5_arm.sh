#!/bin/bash
# Full v5 chain for one shaping arm: tokenizer -> backup -> cache -> scorecard
# -> joint prior -> backup -> FID eval. Run under setsid; survives teardown.
set -u
arm="$1"; shift
cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate
tok="tokenizer_runs/v5-${arm}-s1"
pri="prior_runs/v5-joint-${arm}-s1"
mkdir -p prior_runs prior_evals

gpu-claim run --owner AFIG --job "v5-${arm}-tok" --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v5-rebuild --run_name "v5-${arm}-s1" \
    --output_dir "$tok" "$@" > "${tok}.launch.log" 2>&1 \
|| { echo "TOKENIZER FAILED ${arm}"; exit 1; }
python -u scripts/backup_wandb_file.py "${tok}/checkpoint_final.pt" "v5-${arm}-tokenizer" >> "${tok}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v5-${arm}-cache" --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint "${tok}/checkpoint_final.pt" \
    --output "${tok}/latents_final_original_flip.pt" \
    --include_horizontal_flip >> "${tok}/cache.log" 2>&1 \
|| { echo "CACHE FAILED ${arm}"; exit 1; }
python -u scripts/analyze_axis_scorecard.py \
  --cache "${tok}/latents_final_original_flip.pt" \
  --output "${tok}/axis_scorecard.json" > "${tok}/axis_scorecard.log" 2>&1

gpu-claim run --owner AFIG --job "v5-joint-${arm}" --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "${tok}/latents_final_original_flip.pt" \
    --output_dir "$pri" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v5-rebuild --run_name "v5-joint-${arm}-s1" \
    --compile > "${pri}.launch.log" 2>&1 \
|| { echo "JOINT FAILED ${arm}"; exit 1; }
python -u scripts/backup_wandb_file.py "${pri}/checkpoint_final.pt" "v5-${arm}-joint" >> "${pri}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v5-eval-${arm}" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${pri}/checkpoint_final.pt" \
    --output_dir "prior_evals/v5-joint-${arm}-060000" \
    > "prior_evals/v5-joint-${arm}-060000.log" 2>&1
echo "ARM ${arm} COMPLETE $(date -u +%FT%TZ)"
