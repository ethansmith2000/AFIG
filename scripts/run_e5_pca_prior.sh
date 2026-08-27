#!/bin/bash
# Matched 60k prior for the selected rank-1536 PCA control, then 5k FID/KID.
set -euo pipefail

cd /workspace/AFIG
source /venv/main/bin/activate

cache="tokenizer_runs/v9-unordered-vae-n64d48-s1/latents_pca_r1536_n64d24_original_flip.pt"
prior="prior_runs/e5-joint-pca-r1536-n64d24-s1"
evaluation="prior_evals/e5-joint-pca-r1536-n64d24-060000"

mkdir -p prior_runs prior_evals

gpu-claim run --owner AFIG --job e5-pca-r1536-prior --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "$cache" \
    --output_dir "$prior" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-e5-pca --run_name e5-joint-pca-r1536-n64d24-s1 \
    --compile > "${prior}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${prior}/checkpoint_final.pt" e5-joint-pca-r1536-n64d24 \
  >> "${prior}.launch.log" 2>&1

gpu-claim run --owner AFIG --job e5-pca-r1536-eval --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior}/checkpoint_final.pt" \
    --output_dir "$evaluation" \
    --num_samples 5000 --batch_size 256 --sample_steps 50 \
    > "${evaluation}.log" 2>&1

echo "E5 PCA PRIOR COMPLETE $(date -u +%FT%TZ)"
