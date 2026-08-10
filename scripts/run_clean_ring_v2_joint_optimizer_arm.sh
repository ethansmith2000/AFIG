#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {lr4e4-b095-b0999|lr1e4-b095-b0999|lr4e4-b095-b099}" >&2
  exit 2
fi

ARM="$1"
case "${ARM}" in
  lr4e4-b095-b0999)
    LEARNING_RATE="4e-4"
    ADAM_BETA2="0.999"
    ;;
  lr1e4-b095-b0999)
    LEARNING_RATE="1e-4"
    ADAM_BETA2="0.999"
    ;;
  lr4e4-b095-b099)
    LEARNING_RATE="4e-4"
    ADAM_BETA2="0.99"
    ;;
  *)
    echo "Unknown optimizer arm: ${ARM}" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z64-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
LATENT_INTERFACE="${LATENT_INTERFACE:-${AE_RUN}/latent_interface_tensor.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/latent_continuous_runs/joint-clean-ring-v2-modern-opt-${ARM}-matrixwd-const-w768-l12-b256-s1-n20000}"

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing autoencoder checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${LATENT_INTERFACE}" ]]; then
  echo "Missing latent interface: ${LATENT_INTERFACE}" >&2
  exit 1
fi

if [[ "${AFIG_GPU_WORKER:-0}" != "1" ]]; then
  exec /workspace/bin/gpu-claim run \
    --owner AFIG \
    --job "joint-modern-opt-${ARM}" \
    --wait \
    -- \
    env AFIG_GPU_WORKER=1 "${BASH_SOURCE[0]}" "${ARM}"
fi

cd "${ROOT}"

exec /venv/main/bin/python -u train_joint_latent_diffusion.py \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_interface "${LATENT_INTERFACE}" \
  --output_dir "${OUTPUT_DIR}" \
  --data_root "${ROOT}/data" \
  --seed 1 \
  --train_batch_size 256 \
  --dataloader_num_workers 4 \
  --max_train_steps 20000 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --learning_rate "${LEARNING_RATE}" \
  --adam_beta1 0.95 \
  --adam_beta2 "${ADAM_BETA2}" \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 2000 \
  --weight_decay 0.02 \
  --weight_decay_mode matrix_only \
  --mixed_precision bf16 \
  --max_grad_norm 1.0 \
  --num_train_timesteps 1000 \
  --num_inference_steps 50 \
  --flow_solver heun \
  --qk_norm \
  --rope radius_angle \
  --position_embedding_film \
  --block_conditioning adaln_zero \
  --checkpointing_steps 5000 \
  --preview_steps 2500 \
  --preview_images 16 \
  --report_to none \
  --run_name "joint-clean-ring-v2-modern-opt-${ARM}"
