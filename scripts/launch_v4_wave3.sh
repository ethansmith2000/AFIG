#!/bin/bash
# Wave 3: FID evals for both finished joints, ramp-s2 relaunch,
# det-s2 + frontier-s2 cache -> scorecard -> joint chains.
set -u
cd /workspace/AFIG
source /venv/main/bin/activate
mkdir -p prior_evals

setsid nohup gpu-claim run --owner AFIG --job eval-joint-vae-60k --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint prior_runs/v4-joint-vae-kl1e4-s1/checkpoint_final.pt \
    --output_dir prior_evals/v4-joint-vae-kl1e4-060000 \
    > prior_evals/v4-joint-vae-kl1e4-060000.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job eval-joint-energycv-60k --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint prior_runs/v4-joint-energycv-s1/checkpoint_final.pt \
    --output_dir prior_evals/v4-joint-energycv-060000 \
    > prior_evals/v4-joint-energycv-060000.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-ramp-s2 --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v4-shaping --run_name v4-ramp-s2 \
    --output_dir tokenizer_runs/v4-ramp-s2 --latent_shaping ramp \
    > tokenizer_runs/v4-ramp-s2.launch.log 2>&1 &

setsid nohup bash -c '
source /venv/main/bin/activate
gpu-claim run --owner AFIG --job cache-v4-det-s2 --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint tokenizer_runs/v4-det-s2/checkpoint_final.pt \
    --output tokenizer_runs/v4-det-s2/latents_final_original_flip.pt \
    --include_horizontal_flip >> tokenizer_runs/v4-det-s2/cache.log 2>&1 \
&& python -u scripts/analyze_axis_scorecard.py \
    --cache tokenizer_runs/v4-det-s2/latents_final_original_flip.pt \
    --output tokenizer_runs/v4-det-s2/axis_scorecard.json \
    > tokenizer_runs/v4-det-s2/axis_scorecard.log 2>&1 \
&& gpu-claim run --owner AFIG --job v4-joint-det --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-det-s2/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-det-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-det-s1 \
    --compile > prior_runs/v4-joint-det-s1.launch.log 2>&1
' > tokenizer_runs/v4-det-s2-chain.log 2>&1 &

setsid nohup bash -c '
cd /workspace/AFIG && source /venv/main/bin/activate
gpu-claim run --owner AFIG --job cache-v4-frontier-s2 --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint tokenizer_runs/v4-frontier-s2/checkpoint_final.pt \
    --output tokenizer_runs/v4-frontier-s2/latents_final_original_flip.pt \
    --include_horizontal_flip >> tokenizer_runs/v4-frontier-s2/cache.log 2>&1 \
&& python -u scripts/analyze_axis_scorecard.py \
    --cache tokenizer_runs/v4-frontier-s2/latents_final_original_flip.pt \
    --output tokenizer_runs/v4-frontier-s2/axis_scorecard.json \
    > tokenizer_runs/v4-frontier-s2/axis_scorecard.log 2>&1 \
&& gpu-claim run --owner AFIG --job v4-joint-frontier --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-frontier-s2/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-frontier-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-frontier-s1 \
    --compile > prior_runs/v4-joint-frontier-s1.launch.log 2>&1
' > tokenizer_runs/v4-frontier-s2-chain.log 2>&1 &

sleep 2
ps aux | grep '[g]pu-claim run' | grep AFIG | sed 's/.*--job \([^ ]*\).*/\1/' | sort
