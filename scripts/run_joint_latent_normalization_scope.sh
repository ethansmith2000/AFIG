#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCOPE="${NORMALIZATION_SCOPE:-channel}"
STEPS="${STEPS:-30000}"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-vae-kl0.0001}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface_posterior_mean_${SCOPE}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/joint-vae-mean-${SCOPE}-rf-w768-l12-b256-s1-n${STEPS}}"

case "${SCOPE}" in
  channel|tensor) ;;
  *)
    echo "NORMALIZATION_SCOPE must be channel or tensor, got: ${SCOPE}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing autoencoder checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi

if [[ "${AFIG_GPU_WORKER:-0}" != "1" ]]; then
  exec /workspace/bin/gpu-claim run \
    --owner AFIG \
    --job "joint-latent-norm-${SCOPE}-${STEPS}" \
    --wait \
    -- \
    env AFIG_GPU_WORKER=1 NORMALIZATION_SCOPE="${SCOPE}" STEPS="${STEPS}" \
      AE_RUN="${AE_RUN}" AE_CHECKPOINT="${AE_CHECKPOINT}" \
      LATENT_INTERFACE="${LATENT_INTERFACE}" OUTPUT_DIR="${OUTPUT_DIR}" \
      "${BASH_SOURCE[0]}"
fi

cd "${ROOT}"

if [[ ! -f "${LATENT_INTERFACE}" ]]; then
  /venv/main/bin/python -u fit_autoencoder_latent_interface.py \
    --checkpoint "${AE_CHECKPOINT}" \
    --data_root "${ROOT}/data" \
    --normalization_scope "${SCOPE}" \
    --output "${LATENT_INTERFACE}"
fi

exec /venv/main/bin/python -u train_joint_latent_diffusion.py \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_interface "${LATENT_INTERFACE}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${ROOT}/data" \
  --seed 1 \
  --train_batch_size 256 \
  --dataloader_num_workers 4 \
  --max_train_steps "${STEPS}" \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --learning_rate 1e-4 \
  --lr_scheduler linear_floor \
  --lr_end_ratio 0.25 \
  --lr_warmup_steps 2000 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --max_grad_norm 1.0 \
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --flow_solver heun \
  --checkpointing_steps 7500 \
  --preview_steps 2500 \
  --preview_images 16 \
  --report_to none \
  --run_name "joint-latent-${SCOPE}-rf-w768-l12-b256-s1-n${STEPS}"
