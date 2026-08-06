#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
source_checkpoint=${SOURCE_CHECKPOINT:-${project_root}/continuous_runs/ar_fft_factorized_polar_v2_eps01_global_10k/checkpoint_10000.pt}
output_dir=${OUTPUT_DIR:-${project_root}/continuous_runs/joint_phase_oracle_true_amplitude_10k}

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job joint-phase-oracle-true-amplitude-10000 \
  --wait \
  -- \
  /venv/main/bin/python -u train_joint_phase_oracle.py \
  --source_checkpoint "${source_checkpoint}" \
  --output_dir "${output_dir}" \
  --data_root "${project_root}/data" \
  --group_size 8 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --ff_mult 4 \
  --qk_norm \
  --rope_base 10000 \
  --gradient_checkpointing \
  --max_train_steps 10000 \
  --train_batch_size 256 \
  --dataloader_num_workers 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_steps 500 \
  --phase_gate 0.1 \
  --cartesian_loss_weight 0.1 \
  --preview_steps 2500 \
  --checkpointing_steps 2500 \
  --num_validation_images 8 \
  --num_inference_steps 20 \
  --logging_steps 50 \
  --seed 1
