#!/bin/bash
set -euo pipefail

mode="${1:?expected radial-a02, orbit-a0, orbit-a02, or orbit-a1}"
cd /workspace/AFIG

normalization="orbit_whiten"
codec_stats="continuous_runs/orbit_whiten_codec_stats.pt"
loss_args=(--loss_metric orbit_covariance_power)
case "${mode}" in
  radial-a02)
    normalization="radial_whiten"
    codec_stats="continuous_runs/radial_whiten_v2_codec_stats.pt"
    loss_args=(--loss_metric normalized --radial_power_weighting --radial_power_exponent 0.2)
    ;;
  orbit-a0)
    loss_args+=(--orbit_covariance_exponent 0.0)
    ;;
  orbit-a02)
    loss_args+=(--orbit_covariance_exponent 0.2)
    ;;
  orbit-a1)
    loss_args+=(--orbit_covariance_exponent 1.0)
    ;;
  *)
    echo "unknown controlled baseline: ${mode}" >&2
    exit 2
    ;;
esac

exec gpu-claim run --owner AFIG --job "${mode}-30k" --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/per_orbit_${mode}_30k" \
  --codec_stats_path "${codec_stats}" \
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
  --objective ddpm \
  --prediction_type x0 \
  --loss_space native \
  --loss_weighting min_snr \
  --min_snr_gamma 0.2 \
  --rescale_betas_zero_snr \
  --timestep_spacing trailing \
  --normalization "${normalization}" \
  "${loss_args[@]}" \
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
  --run_name "per-orbit-${mode}-30k" \
  --run_group "per-orbit-baseline" \
  --run_tags "baseline,depth6,raw,clean,${mode}" \
  --reference_stats_path "continuous_runs/cifar10_inception_reference_radial.pt"
