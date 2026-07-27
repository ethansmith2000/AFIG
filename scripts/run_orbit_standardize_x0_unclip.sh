#!/bin/bash
set -euo pipefail

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job orbit-std-x0-unclip-30k --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/orbit_standardize_x0_unclip_30k" \
  --codec_stats_path "continuous_runs/orbit_standardize_codec_stats.pt" \
  --dataset huggingface_cifar \
  --max_train_steps 30000 \
  --preview_steps 5000 \
  --condition_diagnostic_steps 0 \
  --num_layers 10 \
  --width 768 \
  --num_heads 12 \
  --diff_width 768 \
  --diff_depth 6 \
  --train_batch_size 128 \
  --diffusion_batch_mul 1 \
  --learning_rate 1e-4 \
  --adam_beta2 0.99 \
  --grad_norm_mode clip \
  --objective ddpm \
  --prediction_type x0 \
  --loss_space native \
  --loss_weighting min_snr \
  --min_snr_gamma 0.2 \
  --timestep_spacing linspace \
  --num_inference_steps 20 \
  --normalization orbit_standardize \
  --loss_metric normalized \
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
  --run_name "orbit-standardize-x0-unclip-30k" \
  --run_group "orbit-standardize-x0" \
  --run_tags "ddpm,x0,minsnr-0.2,unclip-cosine,linspace,depth6,raw,clean,orbit-standardize" \
  --reference_stats_path "continuous_runs/cifar10_inception_reference_radial.pt"
