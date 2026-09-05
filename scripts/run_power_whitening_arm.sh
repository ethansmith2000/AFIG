#!/usr/bin/env bash
# One queue-managed train-plus-FID5k arm of the smooth power-whitening screen.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {g0|g025|g05|g1}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
cache_root="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
failed="${cache_root}/power_whitening_cache_failed"
case "$arm" in
  g0)
    name="v36-v27-factorized-powerg0-common-uniform-prior-s1"
    ;;
  g025)
    name="v36-v27-factorized-powerg025-common-uniform-prior-s1"
    ;;
  g05)
    name="v36-v27-factorized-powerg05-common-uniform-prior-s1"
    ;;
  g1)
    name="v36-v27-factorized-powerg1-common-uniform-prior-s1"
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

cache="${cache_root}/latents_factorized_power${arm}_original_flip.pt"
ready="${cache_root}/power_whitening_cache_${arm}_ready"
while [[ ! -f "$ready" ]]; do
  if [[ -f "$failed" ]]; then
    echo "${arm} cancelled because power-whitening cache construction failed" >&2
    exit 1
  fi
  sleep 30
done
[[ -f "$cache" ]] || { echo "ready marker exists but cache is missing" >&2; exit 1; }

prior="prior_runs/${name}"
evaluation="prior_evals/${name}-060000"
mkdir -p "$prior" "$evaluation"

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${prior}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${prior}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${name}-train" --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" --output_dir "$prior" --seed 1 \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
      --time_parameterization global --token_loss_weighting uniform \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-v27-power-whitening-screen \
      --run_name "$name" --compile "${resume_args[@]}" \
      >> "${prior}.launch.log" 2>&1
fi

if [[ ! -f "${prior}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${prior}/checkpoint_final.pt" "${name}-checkpoint" \
    >> "${prior}.launch.log" 2>&1 || true
  touch "${prior}/wandb_backup_attempted"
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-eval5k" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" --output_dir "$evaluation" \
      --num_samples 5000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "POWER WHITENING ${arm} COMPLETE $(date -u +%FT%TZ)"
