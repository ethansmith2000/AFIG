#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {lr4e4-b095-b0999|lr1e4-b095-b0999|lr4e4-b095-b099|previous}" >&2
  exit 2
fi

ARM="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z64-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize"
case "${ARM}" in
  lr4e4-b095-b0999|lr1e4-b095-b0999|lr4e4-b095-b099)
    RUN_DIR="${ROOT}/latent_continuous_runs/joint-clean-ring-v2-modern-opt-${ARM}-matrixwd-const-w768-l12-b256-s1-n20000"
    ;;
  previous)
    RUN_DIR="${ROOT}/latent_continuous_runs/joint-clean-ring-v2-modern-posfilm-rope2d-qknorm-adalnzero-ema-w768-l12-b256-s1-n20000"
    ;;
  *)
    echo "Unknown evaluation arm: ${ARM}" >&2
    exit 2
    ;;
esac

if [[ "${AFIG_GPU_WORKER:-0}" != "1" ]]; then
  exec /workspace/bin/gpu-claim run \
    --owner AFIG \
    --job "joint-opt-eval-${ARM}" \
    --wait \
    -- \
    env AFIG_GPU_WORKER=1 "${BASH_SOURCE[0]}" "${ARM}"
fi

cd "${ROOT}"
exec /venv/main/bin/python -u evaluate_joint_latent_diffusion.py \
  --checkpoint "${RUN_DIR}/checkpoint_final.pt" \
  --ae_checkpoint "${AE_RUN}/checkpoint_30000.pt" \
  --latent_interface "${AE_RUN}/latent_interface_tensor.pt" \
  --output_dir "${ROOT}/diagnostics/joint_optimizer_5k/${ARM}" \
  --reference_cache "${ROOT}/continuous_runs/cifar10_inception_reference_radial.pt" \
  --num_samples 5000 \
  --batch_size 128 \
  --num_inference_steps 50 \
  --seed 71001 \
  --preview_images 64
