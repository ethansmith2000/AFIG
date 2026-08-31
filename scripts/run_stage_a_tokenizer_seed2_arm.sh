#!/usr/bin/env bash
# Second-tokenizer-seed confirmation for the parameter-matched Stage-A pair.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "v8" && "$1" != "v12" ) ]]; then
  echo "usage: $0 {v8|v12}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
common_train_args=(
  --seed 2
  --objective full
  --num_latents 64 --latent_dim 16
  --variational --kl_weight 1e-4 --hard_log_variance_clamp
  --qk_norm rms
  --learning_rate 1e-4
  --train_batch_size 512 --eval_batch_size 512 --num_workers 4
  --max_train_steps 15000 --warmup_steps 1000
  --checkpoint_every 2500
  --report_to wandb --tracker_project_name afig-progressive-tokenizer
  --run_group autoencoder-stage-a-tokenizer-seed2
  --compile
)

if [[ "$arm" == "v8" ]]; then
  name="v8-unordered-vae-s2"
  architecture_args=(--encoder_depth 8 --pool_type cross_only --pool_depth 1)
else
  name="v12-unordered-vae-residual-e7p1-n64d16-s2"
  architecture_args=(--encoder_depth 7 --pool_type residual --pool_depth 1)
fi

tokenizer="tokenizer_runs/${name}"
cache="${tokenizer}/latents_final_original_flip.pt"
mkdir -p "$tokenizer"

if [[ ! -f "${tokenizer}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${tokenizer}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${tokenizer}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${name}-tokenizer" --wait -- \
    python -u train_progressive_tokenizer.py \
      "${common_train_args[@]}" "${architecture_args[@]}" \
      --run_name "$name" --output_dir "$tokenizer" \
      "${resume_args[@]}" \
      >> "${tokenizer}.launch.log" 2>&1
fi

# The workspace is not volume-backed. Upload is best-effort so a telemetry
# outage cannot suppress the scientific screen; artifact presence is verified
# separately after the arm completes.
if [[ ! -f "${tokenizer}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${tokenizer}/checkpoint_final.pt" "${name}-tokenizer" \
    >> "${tokenizer}.launch.log" 2>&1 || true
  touch "${tokenizer}/wandb_backup_attempted"
fi

if [[ ! -f "$cache" ]]; then
  gpu-claim run --owner AFIG --job "${name}-cache" --wait -- \
    python -u cache_progressive_latents.py \
      --tokenizer_checkpoint "${tokenizer}/checkpoint_final.pt" \
      --output "$cache" --include_horizontal_flip \
      > "${tokenizer}/cache.log" 2>&1
fi

if [[ ! -f "${tokenizer}/axis_scorecard.json" ]]; then
  python -u scripts/analyze_axis_scorecard.py \
    --cache "$cache" --output "${tokenizer}/axis_scorecard.json" \
    > "${tokenizer}/axis_scorecard.log" 2>&1
fi

if [[ ! -f "${tokenizer}/decoder_sensitivity.json" ]]; then
  gpu-claim run --owner AFIG --job "${name}-sensitivity" --wait -- \
    python -u scripts/decoder_sensitivity.py \
      --cache "$cache" --output "${tokenizer}/decoder_sensitivity.json" \
      > "${tokenizer}/decoder_sensitivity.log" 2>&1
fi

echo "STAGE A TOKENIZER SEED 2 ${arm} COMPLETE $(date -u +%FT%TZ)"
