#!/bin/bash
set -euo pipefail

variant="${1:?expected fixed or learned}"
gain_args=()
case "${variant}" in
  fixed)
    ;;
  learned)
    gain_args=(--learned_output_gain)
    ;;
  *)
    echo "unknown output-gain variant: ${variant}" >&2
    exit 2
    ;;
esac

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job "orbit-std-flow-${variant}-30k" --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/orbit_standardize_flow_${variant}_30k" \
  --codec_stats_path "continuous_runs/orbit_standardize_codec_stats.pt" \
  --dataset huggingface_cifar \
  --max_train_steps 30000 \
  --preview_steps 5000 \
  --num_layers 10 \
  --width 768 \
  --num_heads 12 \
  --diff_width 768 \
  --diff_depth 6 \
  --train_batch_size 128 \
  --diffusion_batch_mul 1 \
  --learning_rate 1e-4 \
  --adam_beta2 0.99 \
  --objective flow \
  --prediction_type v_prediction \
  --loss_space native \
  --loss_weighting logit_normal \
  --logit_normal_mean 0.0 \
  --logit_normal_std 1.0 \
  --flow_solver heun \
  --normalization orbit_standardize \
  "${gain_args[@]}" \
  --history_polar_features log_amp_gated_phase \
  --frequency_conditioning \
  --no-position-input-addition \
  --position-rms-normalize \
  --no-transformer-position-film \
  --diffusion-target-conditioning \
  --history_corruption none \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --allow_tf32 \
  --run_name "orbit-standardize-flow-${variant}-30k" \
  --run_group "orbit-standardize-output-gain" \
  --run_tags "flow,depth6,raw,clean,orbit-standardize,${variant}" \
  --reference_stats_path "continuous_runs/cifar10_inception_reference_radial.pt"
