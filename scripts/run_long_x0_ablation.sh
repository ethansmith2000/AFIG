#!/bin/bash
set -euo pipefail

variant="${1:?expected minsnr-b128, minsnr-b256, unweighted-b128, or alpha02-b128}"

batch_size=128
max_steps=100000
warmup_steps=500
preview_steps=5000
loss_args=(--loss_weighting min_snr --min_snr_gamma 0.2 --loss_metric normalized)

case "${variant}" in
  minsnr-b128)
    ;;
  minsnr-b256)
    batch_size=256
    max_steps=50000
    warmup_steps=250
    preview_steps=2500
    ;;
  unweighted-b128)
    loss_args=(--loss_weighting none --loss_metric normalized)
    ;;
  alpha02-b128)
    loss_args=(
      --loss_weighting min_snr
      --min_snr_gamma 0.2
      --loss_metric orbit_scale_power
      --orbit_scale_exponent 0.2
    )
    ;;
  *)
    echo "unknown long-run variant: ${variant}" >&2
    exit 2
    ;;
esac

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job "long-x0-${variant}" --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/long_x0_${variant}" \
  --codec_stats_path "continuous_runs/orbit_standardize_codec_stats.pt" \
  --dataset huggingface_cifar \
  --max_train_steps "${max_steps}" \
  --preview_steps "${preview_steps}" \
  --preview_seed 12345 \
  --condition_diagnostic_steps 0 \
  --no-final_eval \
  --num_layers 10 \
  --width 768 \
  --num_heads 12 \
  --diff_width 768 \
  --diff_depth 6 \
  --train_batch_size "${batch_size}" \
  --diffusion_batch_mul 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps "${warmup_steps}" \
  --adam_beta2 0.99 \
  --grad_norm_mode clip \
  --objective ddpm \
  --prediction_type x0 \
  --loss_space native \
  "${loss_args[@]}" \
  --timestep_spacing linspace \
  --num_inference_steps 20 \
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
  --allow_tf32 \
  --run_name "long-x0-${variant}" \
  --run_group "long-x0-matched-budget" \
  --run_tags "long,x0,unclip-cosine,linspace,depth6,raw,clean,orbit-standardize,${variant}"
