#!/usr/bin/env bash
# One queue-managed train-plus-FID5k arm of the expanded ZCA-axis matrix.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {channel|sequence|flattened} {common_uniform|ordered_uniform|common_weighted|ordered_weighted}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

variant="$1"
arm="$2"
case "$variant" in
  channel|sequence|flattened) ;;
  *) echo "unsupported new-cache variant: $variant" >&2; exit 2 ;;
esac

cache_root="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
cache="${cache_root}/latents_${variant}_zca_g1_original_flip.pt"
cache_ready="${cache_root}/zca_${variant}_g1_ready"
cache_failed="${cache_root}/zca_axis_caches_failed"
protocol="reports/2026-08-26_autoencoder_program/zca_axis_matrix_protocol.json"
while [[ ! -f "$cache_ready" ]]; do
  if [[ -f "$cache_failed" ]]; then
    echo "${variant}/${arm} cancelled because ZCA cache construction failed" >&2
    exit 1
  fi
  sleep 30
done
[[ -f "$cache" ]] || { echo "ready marker exists but cache is missing" >&2; exit 1; }

mapfile -t objective_values < <(
  /venv/main/bin/python - "$protocol" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1]))["token_intervention"]
print(",".join(str(value) for value in selection["group_sizes"]))
print(",".join(f"{value:.17g}" for value in selection["snr1_crossings"]))
print(",".join(f"{value:.17g}" for value in selection["flow_target_energy_loss_weights"]))
PY
)
groups="${objective_values[0]}"
crossings="${objective_values[1]}"
weights="${objective_values[2]}"

case "$arm" in
  common_uniform)
    suffix="common-uniform"
    objective_args=(--time_parameterization global --token_loss_weighting uniform)
    ;;
  ordered_uniform)
    suffix="ordered-uniform"
    objective_args=(
      --time_parameterization rational_per_token
      --token_group_sizes "$groups" --token_snr1_crossings "$crossings"
      --token_snr_logit_strength 1 --token_loss_weighting uniform
    )
    ;;
  common_weighted)
    suffix="common-flowweight"
    objective_args=(
      --time_parameterization global --token_group_sizes "$groups"
      --token_loss_weighting explicit --token_loss_group_weights "$weights"
    )
    ;;
  ordered_weighted)
    suffix="ordered-flowweight"
    objective_args=(
      --time_parameterization rational_per_token
      --token_group_sizes "$groups" --token_snr1_crossings "$crossings"
      --token_snr_logit_strength 1 --token_loss_weighting explicit
      --token_loss_group_weights "$weights"
    )
    ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

name="v38-v27-${variant}-zca1-${suffix}-prior-s1"
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
      --run_group autoencoder-v27-zca-axis-matrix \
      --run_name "$name" "${objective_args[@]}" \
      --compile "${resume_args[@]}" >> "${prior}.launch.log" 2>&1
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

echo "ZCA AXIS FACTORIAL ${variant}/${arm} COMPLETE $(date -u +%FT%TZ)"
