#!/bin/bash
# Matched pixel-space control: 4x4 pixel patches -> joint RF -> image-space FID.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

cache="pixel_runs/v8-cifar10-patches-original-flip.pt"
prior_dir="prior_runs/v8-joint-pixels-s1"
evaluation_dir="prior_evals/v8-joint-pixels-060000"

mkdir -p pixel_runs prior_runs prior_evals

if [[ ! -f "$cache" ]]; then
  /venv/main/bin/python -u cache_cifar_pixel_patches.py \
    --output "$cache" --patch_size 4 --include_horizontal_flip \
    > pixel_runs/v8-cifar10-patches.cache.log 2>&1
fi

gpu-claim run --owner AFIG --job v8-joint-pixels --wait -- \
  /venv/main/bin/python -u train_progressive_joint_flow.py \
    --latent_cache "$cache" \
    --output_dir "$prior_dir" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group decisive-controls-v8 --run_name v8-joint-pixels-s1 \
    --compile \
    > "${prior_dir}.launch.log" 2>&1

/venv/main/bin/python -u scripts/backup_wandb_file.py \
  "${prior_dir}/checkpoint_final.pt" v8-joint-pixels \
  >> "${prior_dir}.launch.log" 2>&1

gpu-claim run --owner AFIG --job v8-eval-pixels --wait -- \
  /venv/main/bin/python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior_dir}/checkpoint_final.pt" \
    --output_dir "$evaluation_dir" \
    > "${evaluation_dir}.log" 2>&1

echo "PIXEL CONTROL COMPLETE $(date -u +%FT%TZ)"
