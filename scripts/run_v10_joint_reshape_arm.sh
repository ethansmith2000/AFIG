#!/bin/bash
# Exact fixed-representation layout control for the v8 unordered 64x16 cache.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <sequence-length>" >&2
  exit 2
fi

sequence_length="$1"
case "$sequence_length" in
  32) token_dim=32 ;;
  128) token_dim=8 ;;
  *) echo "sequence-length must be one of: 32, 128" >&2; exit 2 ;;
esac

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

cache="tokenizer_runs/v8-unordered-vae-s1/latents_final_original_flip.pt"
name="v10-joint-unordered-reshape-n${sequence_length}d${token_dim}-s1"
prior_dir="prior_runs/${name}"
evaluation_dir="prior_evals/v10-joint-unordered-reshape-n${sequence_length}d${token_dim}-060000"

mkdir -p prior_runs prior_evals

gpu-claim run --owner AFIG --job "$name" --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "$cache" \
    --layout_sequence_length "$sequence_length" \
    --output_dir "$prior_dir" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group rate-shape-v10 --run_name "$name" \
    --compile \
    > "${prior_dir}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${prior_dir}/checkpoint_final.pt" "$name" \
  >> "${prior_dir}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v10-eval-reshape-n${sequence_length}d${token_dim}" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior_dir}/checkpoint_final.pt" \
    --output_dir "$evaluation_dir" \
    > "${evaluation_dir}.log" 2>&1

echo "RESHAPE N${sequence_length}D${token_dim} COMPLETE $(date -u +%FT%TZ)"
