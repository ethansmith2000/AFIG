#!/usr/bin/env bash
# End-to-end learned latent-factorization arms at a fixed 1,024 coordinates.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "n32d32" && "$1" != "n128d8" ) ]]; then
  echo "usage: $0 {n32d32|n128d8}" >&2
  exit 2
fi

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

arm="$1"
case "$arm" in
  n32d32)
    num_latents=32
    latent_dim=32
    tokenizer_name="v30-residual-e7p1-det-jitter05-slotbal2e3-n32d32-s2"
    ;;
  n128d8)
    num_latents=128
    latent_dim=8
    tokenizer_name="v31-residual-e7p1-det-jitter05-slotbal2e3-n128d8-s2"
    ;;
esac

tokenizer="tokenizer_runs/${tokenizer_name}"
cache="${tokenizer}/latents_final_original_flip.pt"
prior_name="${tokenizer_name}-prior-s1"
prior="prior_runs/${prior_name}"
evaluation="prior_evals/${prior_name}-060000"
mkdir -p "$tokenizer" "$prior" "$evaluation"

if [[ ! -f "${tokenizer}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${tokenizer}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${tokenizer}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${tokenizer_name}-tokenizer" --wait -- \
    python -u train_progressive_tokenizer.py \
      --seed 2 --objective full \
      --patch_size 4 --num_latents "$num_latents" --latent_dim "$latent_dim" \
      --encoder_depth 7 --pool_type residual --pool_depth 1 \
      --no-variational --decoder_jitter_std 0.05 \
      --slot_balance_weight 0.002 \
      --qk_norm rms --learning_rate 1e-4 \
      --train_batch_size 512 --eval_batch_size 512 --num_workers 4 \
      --max_train_steps 15000 --warmup_steps 1000 --checkpoint_every 2500 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-learned-latent-shape \
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

# Reconstruction is only a permissive codec-health veto.
if ! python - "$tokenizer" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = json.loads((root / "metrics_final.json").read_text())
sensitivity = json.loads((root / "decoder_sensitivity.json").read_text())
full_prefix = max(metrics["prefix"], key=int)
psnr = float(metrics["prefix"][full_prefix]["psnr_db"])
rfid = float(sensitivity["sigmas"]["0.0"]["reconstruction_fid"])
if not math.isfinite(psnr) or not math.isfinite(rfid):
    raise SystemExit(f"non-finite codec health: PSNR={psnr}, rFID={rfid}")
if psnr < 28.0 and rfid > 25.0:
    raise SystemExit(f"codec outside health envelope: PSNR={psnr}, rFID={rfid}")
print(json.dumps({"codec_health": "pass", "psnr": psnr, "clean_rfid": rfid}, sort_keys=True))
PY
then
  touch "${tokenizer}/codec_health_gate_failed"
  echo "LEARNED LATENT SHAPE ${arm} STOPPED AT CODEC HEALTH GATE $(date -u +%FT%TZ)"
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
      --run_group autoencoder-learned-latent-shape \
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

echo "LEARNED LATENT SHAPE ${arm} COMPLETE $(date -u +%FT%TZ)"
