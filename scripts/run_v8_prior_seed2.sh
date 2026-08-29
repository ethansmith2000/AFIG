#!/bin/bash
# Paired independent prior seed for the frozen cross-only v8 baseline tokenizer.
set -euo pipefail

cd /workspace/AFIG
source /venv/main/bin/activate

tokenizer="tokenizer_runs/v8-unordered-vae-s1"
prior="prior_runs/v8-joint-unordered-vae-prior-s2"
evaluation="prior_evals/v8-joint-unordered-vae-prior-s2-060000"
mkdir -p prior_runs prior_evals

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${prior}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${prior}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job v8-cross-only-prior-s2 --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "${tokenizer}/latents_final_original_flip.pt" \
      --output_dir "$prior" --seed 2 \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-stage-a-confirmation \
      --run_name v8-joint-cross-only-prior-s2 \
      --compile "${resume_args[@]}" >> "${prior}.launch.log" 2>&1
fi

if ! timeout --signal=TERM 300s python -u scripts/backup_wandb_file.py \
  "${prior}/checkpoint_final.pt" v8-joint-cross-only-prior-s2 \
  >> "${prior}.launch.log" 2>&1; then
  echo '{"backup_skipped":"v8-joint-cross-only-prior-s2","warning":"upload timeout or failure"}' \
    >> "${prior}.launch.log"
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job v8-cross-only-eval-s2 --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" \
      --output_dir "$evaluation" \
      --num_samples 5000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "V8 CROSS-ONLY PRIOR SEED 2 COMPLETE $(date -u +%FT%TZ)"
