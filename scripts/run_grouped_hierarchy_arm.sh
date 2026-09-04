#!/usr/bin/env bash
# Grouped cumulative low-pass versus additive DoG hierarchy arms.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "cumulative" && "$1" != "innovation" ) ]]; then
  echo "usage: $0 {cumulative|innovation}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
case "$arm" in
  cumulative)
    tokenizer_name="v32-residual-e7p1-det-jitter05-slotbal2e3-hier-cum3e2-n64d16-s2"
    hierarchy_weight="0.030"
    ;;
  innovation)
    tokenizer_name="v33-residual-e7p1-det-jitter05-slotbal2e3-hier-innov23e3-n64d16-s2"
    hierarchy_weight="0.023"
    ;;
esac

tokenizer="tokenizer_runs/${tokenizer_name}"
cache="${tokenizer}/latents_final_original_flip.pt"
prior_name="${tokenizer_name}-prior-s1"
prior="prior_runs/${prior_name}"
evaluation="prior_evals/${prior_name}-060000"
hierarchy_analysis="${tokenizer}/hierarchy_analysis"
mkdir -p "$tokenizer" "$prior" "$evaluation" "$hierarchy_analysis"

if [[ ! -f "${tokenizer}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${tokenizer}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${tokenizer}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${tokenizer_name}-tokenizer" --wait -- \
    python -u train_progressive_tokenizer.py \
      --seed 2 --objective full \
      --patch_size 4 --num_latents 64 --latent_dim 16 \
      --encoder_depth 7 --pool_type residual --pool_depth 1 \
      --no-variational --decoder_jitter_std 0.05 \
      --slot_balance_weight 0.002 \
      --hierarchy_mode "$arm" \
      --hierarchy_group_sizes 11,11,11,11,10,10 \
      --hierarchy_blur_sigmas 8,4,2,1,0.5,0 \
      --hierarchy_loss_weight "$hierarchy_weight" \
      --hierarchy_batch_size 128 \
      --qk_norm rms --learning_rate 1e-4 \
      --train_batch_size 512 --eval_batch_size 512 --num_workers 4 \
      --max_train_steps 15000 --warmup_steps 1000 --checkpoint_every 2500 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-grouped-hierarchy \
      --run_name "$tokenizer_name" --output_dir "$tokenizer" \
      --compile "${resume_args[@]}" >> "${tokenizer}.launch.log" 2>&1
fi

if [[ ! -f "${tokenizer}/wandb_backup_attempted" ]]; then
  timeout --signal=TERM 600s python -u scripts/backup_wandb_file.py \
    "${tokenizer}/checkpoint_final.pt" "${tokenizer_name}-tokenizer" \
    >> "${tokenizer}.launch.log" 2>&1 || true
  touch "${tokenizer}/wandb_backup_attempted"
fi

if [[ ! -f "$cache" ]]; then
  gpu-claim run --owner AFIG --job "${tokenizer_name}-cache" --wait -- \
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
  gpu-claim run --owner AFIG --job "${tokenizer_name}-sensitivity" --wait -- \
    python -u scripts/decoder_sensitivity.py \
      --cache "$cache" --output "${tokenizer}/decoder_sensitivity.json" \
      > "${tokenizer}/decoder_sensitivity.log" 2>&1
fi

if [[ ! -f "${hierarchy_analysis}/metrics.json" ]]; then
  gpu-claim run --owner AFIG --job "${tokenizer_name}-hierarchy-audit" --wait -- \
    python -u scripts/analyze_grouped_hierarchy.py \
      --checkpoint "${tokenizer}/checkpoint_final.pt" \
      --output "${hierarchy_analysis}/metrics.json" \
      --examples 512 --batch_size 64 \
      > "${hierarchy_analysis}.log" 2>&1
fi

# Reconstruction is a permissive codec-health veto, not a selector.
if ! python - "$tokenizer" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = json.loads((root / "metrics_final.json").read_text())
sensitivity = json.loads((root / "decoder_sensitivity.json").read_text())
psnr = float(metrics["prefix"]["64"]["psnr_db"])
rfid = float(sensitivity["sigmas"]["0.0"]["reconstruction_fid"])
if not math.isfinite(psnr) or not math.isfinite(rfid):
    raise SystemExit(f"non-finite codec health: PSNR={psnr}, rFID={rfid}")
if psnr < 28.0 and rfid > 25.0:
    raise SystemExit(f"codec outside health envelope: PSNR={psnr}, rFID={rfid}")
print(json.dumps({"codec_health": "pass", "psnr": psnr, "clean_rfid": rfid}, sort_keys=True))
PY
then
  touch "${tokenizer}/codec_health_gate_failed"
  echo "GROUPED HIERARCHY ${arm} STOPPED AT CODEC HEALTH GATE $(date -u +%FT%TZ)"
  exit 0
fi

if [[ ! -f "${prior}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${prior}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${prior}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${prior_name}-train" --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" --output_dir "$prior" --seed 1 \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-grouped-hierarchy \
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

echo "GROUPED HIERARCHY ${arm} COMPLETE $(date -u +%FT%TZ)"
