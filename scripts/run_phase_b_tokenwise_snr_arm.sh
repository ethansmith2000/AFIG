#!/usr/bin/env bash
# Clean tokenwise-SNR study on an exactly invertible content-RMS-ordered cache.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "control" && "$1" != "warp" && "$1" != "warp_weighted" ) ]]; then
  echo "usage: $0 {control|warp|warp_weighted}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
source_cache="tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s2/latents_final_original_flip.pt"
cache="tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s2/latents_final_original_flip_snr_ordered.pt"
if [[ ! -f "$cache" ]]; then
  python -u scripts/make_token_snr_ordered_cache.py \
    --input "$source_cache" --output "$cache"
fi

groups="11,11,11,11,10,10"
crossings="0.17456502825094655,0.38296835060017503,0.5262492263492242,0.627482253623356,0.742521533181472,0.8473513676508831"
image_band_variances="22.35894171624479,2.595906901503802,0.8104326396466207,0.35244474417636107,0.120244085390767,0.03245329262163484"

case "$arm" in
  control)
    name="v16-joint-v12s2-snr-ordered-common-time-prior-s1"
    objective_args=(--time_parameterization global --token_loss_weighting uniform)
    ;;
  warp)
    name="v17-joint-v12s2-snr-ordered-rational-time-prior-s1"
    objective_args=(
      --time_parameterization rational_per_token
      --token_group_sizes "$groups"
      --token_snr1_crossings "$crossings"
      --token_loss_weighting uniform
    )
    ;;
  warp_weighted)
    name="v18-joint-v12s2-snr-ordered-rational-time-weighted-prior-s1"
    objective_args=(
      --time_parameterization rational_per_token
      --token_group_sizes "$groups"
      --token_snr1_crossings "$crossings"
      --token_loss_weighting explicit
      --token_loss_group_weights "$image_band_variances"
    )
    ;;
esac

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
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-phase-b-tokenwise-snr \
      --run_name "$name" \
      "${objective_args[@]}" \
      --compile "${resume_args[@]}" >> "${prior}.launch.log" 2>&1
fi

if [[ ! -f "${prior}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${prior}/checkpoint_final.pt" "${name}-checkpoint" \
    >> "${prior}.launch.log" 2>&1 || true
  touch "${prior}/wandb_backup_attempted"
fi

if [[ ! -f "${evaluation}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-eval" --wait -- \
    python -u evaluate_progressive_joint_flow.py \
      --checkpoint "${prior}/checkpoint_final.pt" \
      --output_dir "$evaluation" \
      --num_samples 5000 --batch_size 256 --sample_steps 50 --seed 54321 \
      > "${evaluation}.log" 2>&1
fi

echo "PHASE B TOKENWISE SNR ${arm} COMPLETE $(date -u +%FT%TZ)"
