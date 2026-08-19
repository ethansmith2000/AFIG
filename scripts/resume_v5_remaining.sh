#!/bin/bash
# Resume the two teardown-killed v5 runs and finish their chains.
set -u
cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

# vae: resume joint 40k->60k, then backup + eval
setsid nohup bash -c '
export PATH="$PATH:/workspace/bin" && source /venv/main/bin/activate
gpu-claim run --owner AFIG --job v5-joint-vae-resume --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache tokenizer_runs/v5-vae-kl1e4-s1/latents_final_original_flip.pt \
    --output_dir prior_runs/v5-joint-vae-kl1e4-s1 \
    --resume prior_runs/v5-joint-vae-kl1e4-s1/checkpoint_latest.pt \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v5-rebuild --run_name v5-joint-vae-kl1e4-s1 \
    --compile >> prior_runs/v5-joint-vae-kl1e4-s1.launch.log 2>&1 \
&& python -u scripts/backup_wandb_file.py prior_runs/v5-joint-vae-kl1e4-s1/checkpoint_final.pt v5-vae-kl1e4-joint >> prior_runs/v5-joint-vae-kl1e4-s1.launch.log 2>&1 \
&& gpu-claim run --owner AFIG --job v5-eval-vae --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint prior_runs/v5-joint-vae-kl1e4-s1/checkpoint_final.pt \
    --output_dir prior_evals/v5-joint-vae-kl1e4-060000 \
    > prior_evals/v5-joint-vae-kl1e4-060000.log 2>&1
' > chain_vae_resume.log 2>&1 &

# frontier: resume tokenizer 7.5k->15k, then full downstream chain
setsid nohup bash -c '
cd /workspace/AFIG && export PATH="$PATH:/workspace/bin" && source /venv/main/bin/activate
tok=tokenizer_runs/v5-frontier-s1
gpu-claim run --owner AFIG --job v5-frontier-tok-resume --wait -- \
  python -u train_progressive_tokenizer.py \
    --objective progressive --pool_type cross_only --pool_depth 1 \
    --num_latents 64 --latent_dim 16 --learning_rate 1e-4 \
    --train_batch_size 512 --max_train_steps 15000 --warmup_steps 1000 \
    --run_group tokenizer-v5-rebuild --run_name v5-frontier-s1 \
    --output_dir $tok --latent_shaping frontier --frontier_overlap 8 \
    --resume $tok/checkpoint_latest.pt >> ${tok}.launch.log 2>&1 \
&& python -u scripts/backup_wandb_file.py $tok/checkpoint_final.pt v5-frontier-tokenizer >> ${tok}.launch.log 2>&1 \
&& gpu-claim run --owner AFIG --job v5-frontier-cache --wait -- \
  python -u cache_progressive_latents.py \
    --tokenizer_checkpoint $tok/checkpoint_final.pt \
    --output $tok/latents_final_original_flip.pt \
    --include_horizontal_flip >> $tok/cache.log 2>&1 \
&& python -u scripts/analyze_axis_scorecard.py \
    --cache $tok/latents_final_original_flip.pt \
    --output $tok/axis_scorecard.json > $tok/axis_scorecard.log 2>&1 \
&& gpu-claim run --owner AFIG --job v5-joint-frontier --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache $tok/latents_final_original_flip.pt \
    --output_dir prior_runs/v5-joint-frontier-s1 \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v5-rebuild --run_name v5-joint-frontier-s1 \
    --compile > prior_runs/v5-joint-frontier-s1.launch.log 2>&1 \
&& python -u scripts/backup_wandb_file.py prior_runs/v5-joint-frontier-s1/checkpoint_final.pt v5-frontier-joint >> prior_runs/v5-joint-frontier-s1.launch.log 2>&1 \
&& gpu-claim run --owner AFIG --job v5-eval-frontier --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint prior_runs/v5-joint-frontier-s1/checkpoint_final.pt \
    --output_dir prior_evals/v5-joint-frontier-060000 \
    > prior_evals/v5-joint-frontier-060000.log 2>&1
' > chain_frontier_resume.log 2>&1 &
sleep 3
ps aux | grep '[g]pu-claim run' | grep AFIG | sed 's/.*--job \([^ ]*\).*/\1/' | sort -u
