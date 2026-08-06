#!/bin/bash
set -euo pipefail

cd /workspace/AFIG

ae_checkpoint="${AE_CHECKPOINT:-/workspace/AFIG/autoencoder_runs/ae-spatial-d4-c8-r32-s1-n10000-noise0.1/checkpoint_10000.pt}"
output_dir="${OUTPUT_DIR:-/workspace/AFIG/continuous_runs/joint_spatial_ae_hartley_tiles_10k}"
flow_path="${FLOW_PATH:-linear}"
latent_basis="${LATENT_BASIS:-hartley}"
seed="${SEED:-1}"
steps="${STEPS:-10000}"
run_label="${RUN_LABEL:-joint-spatial-ae-${latent_basis}-${flow_path}-${steps}}"
posterior_args=()
if [[ "${SAMPLE_POSTERIOR:-false}" == true ]]; then
  posterior_args+=(--sample_posterior)
fi

exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "${run_label}" \
  --wait -- \
  /venv/main/bin/python -u train_spatial_latent_hartley_joint.py \
  --ae_checkpoint "${ae_checkpoint}" \
  --output_dir "${output_dir}" \
  --data_root /workspace/AFIG/data \
  --seed "${seed}" \
  --steps "${steps}" \
  --batch_size 256 \
  --num_workers 4 \
  --learning_rate 1e-4 \
  --warmup 2000 \
  --weight_decay 0.1 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --inference_steps 50 \
  --preview_steps 2500 \
  --checkpoint_steps 2500 \
  --latent_patch 2 \
  --stats_images 4096 \
  --flow_path "${flow_path}" \
  --latent_basis "${latent_basis}" \
  "${posterior_args[@]}"
