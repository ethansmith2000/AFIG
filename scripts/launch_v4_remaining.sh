#!/bin/bash
# Explicit relaunch of all outstanding v4 work (no loops/vars in job names).
set -u
cd /workspace/AFIG
source /venv/main/bin/activate

for pid in $(ps aux | grep '[g]pu-claim' | grep -F 'rname' | awk '{print $2}'); do
  kill "$pid" 2>/dev/null
done

setsid nohup bash -c '
source /venv/main/bin/activate
gpu-claim run --owner AFIG --job cache-v4-energycv --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint tokenizer_runs/v4-energycv-s1/checkpoint_final.pt \
    --output tokenizer_runs/v4-energycv-s1/latents_final_original_flip.pt \
    --include_horizontal_flip >> tokenizer_runs/v4-energycv-s1/cache.log 2>&1 \
&& python -u scripts/analyze_axis_scorecard.py \
    --cache tokenizer_runs/v4-energycv-s1/latents_final_original_flip.pt \
    --output tokenizer_runs/v4-energycv-s1/axis_scorecard.json \
    > tokenizer_runs/v4-energycv-s1/axis_scorecard.log 2>&1 \
&& gpu-claim run --owner AFIG --job v4-joint-energycv --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-energycv-s1/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-energycv-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-energycv-s1 \
    --compile > prior_runs/v4-joint-energycv-s1.launch.log 2>&1
' > tokenizer_runs/v4-energycv-chain.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-joint-vae-kl1e4 --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v4-vae-kl1e4-s1/latents_final_original_flip.pt \
    --output_dir prior_runs/v4-joint-vae-kl1e4-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v4-shaping --run_name v4-joint-vae-kl1e4-s1 \
    --compile > prior_runs/v4-joint-vae-kl1e4-s1.launch.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-det-s2 --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v4-shaping --run_name v4-det-s2 \
    --output_dir tokenizer_runs/v4-det-s2 --seed 2 \
    > tokenizer_runs/v4-det-s2.launch.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-frontier-s2 --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v4-shaping --run_name v4-frontier-s2 \
    --output_dir tokenizer_runs/v4-frontier-s2 \
    --latent_shaping frontier --frontier_overlap 8 \
    > tokenizer_runs/v4-frontier-s2.launch.log 2>&1 &

setsid nohup gpu-claim run --owner AFIG --job v4-ramp-s2 --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v4-shaping --run_name v4-ramp-s2 \
    --output_dir tokenizer_runs/v4-ramp-s2 --latent_shaping ramp \
    > tokenizer_runs/v4-ramp-s2.launch.log 2>&1 &

sleep 2
echo "AFIG pollers now:"
ps aux | grep '[g]pu-claim' | grep AFIG | awk '{for(i=11;i<=NF;i++) if($i=="--job") print $(i+1)}'
