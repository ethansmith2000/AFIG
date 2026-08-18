#!/bin/bash
set -u
cd /workspace/AFIG
source /venv/main/bin/activate

setsid nohup gpu-claim run --owner AFIG --job v4-joint-det --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-det-s2/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-det-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-det-s1 \
    --compile > prior_runs/v4-joint-det-s1.launch.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-joint-frontier --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-frontier-s2/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-frontier-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-frontier-s1 \
    --compile > prior_runs/v4-joint-frontier-s1.launch.log 2>&1 &

sleep 2
ps aux | grep '[g]pu-claim run' | grep AFIG | sed 's/.*--job \([^ ]*\).*/\1/' | sort -u
