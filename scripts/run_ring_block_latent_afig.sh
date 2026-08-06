#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-ringblock}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface.pt}"
CODEC_TAG="${CODEC_TAG:-ringblock-codec}"
GENERATION_GROUPING="${GENERATION_GROUPING:-ring}"
SEED="${SEED:-1}"
STEPS="${STEPS:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/grouped-${GENERATION_GROUPING}-${CODEC_TAG}-w768-l12-d6-s${SEED}-n${STEPS}}"

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing ring-block autoencoder checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${LATENT_INTERFACE}" ]]; then
  echo "Missing fitted latent interface: ${LATENT_INTERFACE}" >&2
  exit 1
fi

exec /venv/main/bin/python "${ROOT}/train_ring_latent_continuous.py" \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_interface "${LATENT_INTERFACE}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max_train_steps "${STEPS}" \
  --train_batch_size 128 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --diffusion_width 768 \
  --diffusion_depth 6 \
  --diffusion_batch_mul 2 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --max_grad_norm 1.0 \
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --flow_solver heun \
  --generation_grouping "${GENERATION_GROUPING}" \
  --cfg_norm_match \
  --checkpointing_steps 2500 \
  --preview_steps 2500 \
  --run_name "grouped-${GENERATION_GROUPING}-${CODEC_TAG}-w768-l12-d6-s${SEED}-n${STEPS}" \
  "$@"
