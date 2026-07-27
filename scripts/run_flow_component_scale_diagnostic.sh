#!/bin/bash
set -euo pipefail

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job orbit-component-flow-diagnostic --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/orbit_component_flow_diagnostic" \
  --codec_stats_path "continuous_runs/orbit_standardize_component_codec_stats.pt" \
  --dataset huggingface_cifar \
  --max_train_steps 750 \
  --preview_steps 0 \
  --condition_diagnostic_steps 250 \
  --no-final_eval \
  --report_to none \
  --num_layers 10 \
  --width 768 \
  --num_heads 12 \
  --diff_width 768 \
  --diff_depth 6 \
  --train_batch_size 128 \
  --diffusion_batch_mul 1 \
  --learning_rate 1e-4 \
  --adam_beta2 0.99 \
  --grad_norm_mode track \
  --objective flow \
  --prediction_type v_prediction \
  --loss_space native \
  --loss_weighting logit_normal \
  --flow_solver heun \
  --normalization orbit_standardize \
  --history_polar_features log_amp_gated_phase \
  --frequency_conditioning \
  --no-position-input-addition \
  --position-rms-normalize \
  --no-transformer-position-film \
  --diffusion-target-conditioning \
  --history_corruption none \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --allow_tf32
