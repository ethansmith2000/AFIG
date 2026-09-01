#!/usr/bin/env bash
# Independent prior-seed replication for the seed-3 v12/v13 architecture pair.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "v12" && "$1" != "v13" ) ]]; then
  echo "usage: $0 {v12|v13}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
case "$arm" in
  v12)
    tokenizer="tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s3"
    name="v12-joint-residual-e7p1-n64d16-tokenizer-s3-prior-s2"
    ;;
  v13)
    tokenizer="tokenizer_runs/v13-unordered-vae-register-e7j1-n64d16-s3"
    name="v13-joint-register-e7j1-n64d16-tokenizer-s3-prior-s2"
    ;;
esac

cache="${tokenizer}/latents_final_original_flip.pt"
prior="prior_runs/${name}"
evaluation="prior_evals/${name}-060000-n10k"
mkdir -p "$prior" "$evaluation"

if [[ ! -f "$cache" ]]; then
  echo "missing frozen tokenizer-seed-3 cache: $cache" >&2
  exit 1
fi

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${prior}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${prior}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${name}-train" --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" \
      --output_dir "$prior" --seed 2 \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-stage-a-seed3-prior-replication \
      --run_name "$name" \
      --compile "${resume_args[@]}" >> "${prior}.launch.log" 2>&1
fi

if [[ ! -f "${prior}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${prior}/checkpoint_final.pt" "${name}-checkpoint" \
    >> "${prior}.launch.log" 2>&1 || true
  touch "${prior}/wandb_backup_attempted"
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-eval10k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" \
      --output_dir "$evaluation" \
      --num_samples 10000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "TOKENIZER SEED 3 PRIOR SEED 2 ${arm} COMPLETE $(date -u +%FT%TZ)"
