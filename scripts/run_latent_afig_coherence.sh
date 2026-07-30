#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/t12-rf-w768-l12-d6-b256-s1-n30000}"

exec /venv/main/bin/python "${ROOT}/train_latent_continuous.py" \
  --ae_checkpoint "${AE_RUN}/checkpoint_30000.pt" \
  --latent_interface "${AE_RUN}/latent_interface.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --seed 1 \
  --train_batch_size 256 \
  --max_train_steps 30000 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --diffusion_width 768 \
  --diffusion_depth 6 \
  --objective flow \
  --prediction_type v_prediction \
  --learning_rate 1e-4 \
  --lr_scheduler linear_floor \
  --lr_end_ratio 0.25 \
  --lr_warmup_steps 2000 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --max_grad_norm 1.0 \
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --cfg_norm_match \
  --checkpointing_steps 7500 \
  --preview_steps 2500 \
  --run_name latent-afig-t12-rf-w768-l12-d6-b256-s1-n30000 \
  "$@"
