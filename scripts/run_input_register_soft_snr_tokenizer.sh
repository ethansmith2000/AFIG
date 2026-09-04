#!/usr/bin/env bash
# Full-depth input-register tokenizer shared by the common and soft-SNR priors.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

tokenizer_name="v34-inputreg-e8-det-jitter05-slotbal2e3-n64d16-s2"
tokenizer="tokenizer_runs/${tokenizer_name}"
cache="${tokenizer}/latents_final_original_flip.pt"
ready="${tokenizer}/campaign_tokenizer_ready"
failed="${tokenizer}/codec_health_gate_failed"
mkdir -p "$tokenizer"

if [[ ! -f "${tokenizer}/checkpoint_final.pt" ]]; then
  resume_args=()
  if [[ -f "${tokenizer}/checkpoint_latest.pt" ]]; then
    resume_args=(--resume "${tokenizer}/checkpoint_latest.pt")
  fi
  gpu-claim run --owner AFIG --job "${tokenizer_name}-tokenizer" --wait -- \
    python -u train_progressive_tokenizer.py \
      --seed 2 --objective full \
      --patch_size 4 --num_latents 64 --latent_dim 16 \
      --encoder_depth 8 --pool_type input_register_tokens --pool_depth 1 \
      --no-variational --decoder_jitter_std 0.05 \
      --slot_balance_weight 0.002 \
      --qk_norm rms --learning_rate 1e-4 \
      --train_batch_size 512 --eval_batch_size 512 --num_workers 4 \
      --max_train_steps 15000 --warmup_steps 1000 --checkpoint_every 2500 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group autoencoder-input-register-soft-snr \
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

# Reconstruction remains a permissive health veto; generation selects.
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
  touch "$failed"
  echo "INPUT REGISTER TOKENIZER STOPPED AT CODEC HEALTH GATE $(date -u +%FT%TZ)"
  exit 0
fi

touch "$ready"
echo "INPUT REGISTER TOKENIZER READY $(date -u +%FT%TZ)"
