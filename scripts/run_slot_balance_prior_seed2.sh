#!/usr/bin/env bash
# Prior-seed robustness check for the mixed tokenizer-seed-2 slot result.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

tokenizer_name="v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
cache="tokenizer_runs/${tokenizer_name}/latents_final_original_flip.pt"
prior_name="${tokenizer_name}-prior-s2"
prior="prior_runs/${prior_name}"
evaluation="prior_evals/${prior_name}-060000"
mkdir -p "$prior" "$evaluation"

if [[ ! -f "$cache" ]]; then
  echo "missing completed latent cache: $cache" >&2
  exit 1
fi

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${prior}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${prior}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${prior_name}-train" --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" --output_dir "$prior" --seed 2 \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-slot-balance-prior-seed-confirmation \
      --run_name "$prior_name" --compile "${resume_args[@]}" \
      >> "${prior}.launch.log" 2>&1
fi

if [[ ! -f "${prior}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${prior}/checkpoint_final.pt" "${prior_name}-checkpoint" \
    >> "${prior}.launch.log" 2>&1 || true
  touch "${prior}/wandb_backup_attempted"
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${prior_name}-eval5k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" --output_dir "$evaluation" \
      --num_samples 5000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "SLOT BALANCE PRIOR-SEED-2 CONFIRMATION COMPLETE $(date -u +%FT%TZ)"
