#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG

AE_CHECKPOINT="${AE_CHECKPOINT:-/workspace/AFIG/autoencoder_runs/ae-spatial-perceptual-c4-deterministic-noise01-10k/checkpoint_10000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/AFIG/continuous_runs/ar_spatialized_prefix_hartley_perceptual_c4_s7_10k}"
SEED="${SEED:-7}"
STEPS="${STEPS:-10000}"
RUN_LABEL="${RUN_LABEL:-ar-spatialized-prefix-hartley-perceptual-c4-s${SEED}-${STEPS}}"

exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "${RUN_LABEL}" \
  --wait -- \
  /venv/main/bin/python -u train_spatialized_prefix_hartley_ar.py \
    --ae_checkpoint "${AE_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_root /workspace/AFIG/data \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --batch_size 32 \
    --num_workers 4 \
    --learning_rate 7e-5 \
    --warmup 500 \
    --weight_decay 0.1 \
    --width 768 \
    --layers 10 \
    --heads 12 \
    --ff_mult 4 \
    --diff_width 768 \
    --diff_depth 3 \
    --inference_steps 20 \
    --preview_steps 2500 \
    --diagnostic_steps 250 \
    --checkpoint_steps 2500 \
    --validation_images 16 \
    --latent_patch 2 \
    --stats_images 4096
