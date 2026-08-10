#!/usr/bin/env bash
set -euo pipefail

arm="${1:?usage: run_fft_polar_v3_interface_gate.sh polar_trunk_cart_head-or-cart_trunk_polar_head [GPU] [STEPS]}"
gpu="${2:-any}"
steps="${3:-10000}"

case "${arm}" in
  polar_trunk_cart_head)
    decoder_geometry="cartesian"
    history_polar_features="physical_standardized_log_amp_phase"
    history_polar_fusion="replace"
    ;;
  cart_trunk_polar_head)
    decoder_geometry="factorized_polar"
    history_polar_features="none"
    history_polar_fusion="add"
    ;;
  *)
    echo "unknown interface arm: ${arm}" >&2
    exit 2
    ;;
esac

project_root=/workspace/AFIG
output_dir="${project_root}/continuous_runs/ar_fft_polar_v3_${arm}_s1_${steps}"
codec_stats="${project_root}/autoencoder_runs/codec_stats_32_global_standardize.pt"

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "fft-polar-v3-${arm}-s1-${steps}" \
  --gpu "${gpu}" \
  --wait \
  -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "${output_dir}" \
  --codec_stats_path "${codec_stats}" \
  --dataset huggingface_cifar \
  --data_root "${project_root}/data" \
  --seed 1 \
  --train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --max_train_steps "${steps}" \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_weight_decay 0.02 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --allow_tf32 \
  --width 768 \
  --num_layers 10 \
  --num_heads 12 \
  --ff_mult 4 \
  --qk_norm \
  --attention_rope frequency_2d \
  --rope_base 10000 \
  --no-transformer-position-film \
  --diff_width 768 \
  --diff_depth 6 \
  --decoder_geometry "${decoder_geometry}" \
  --factorized_coordinate_mode physical_standardized \
  --factorized_log_epsilon 0.003 \
  --factorized_amplitude_standardization global \
  --factorized_amplitude_prediction_type v_prediction \
  --factorized_amplitude_loss_weight 0.1 \
  --factorized_phase_loss_weight 0.1 \
  --factorized_cartesian_loss_weight 1 \
  --factorized_phase_weighting physical_energy \
  --factorized_phase_predicted_amplitude_probability 0.5 \
  --factorized_phase_process geodesic_flow \
  --factorized_self_conjugate_sign bernoulli \
  --objective flow \
  --prediction_type v_prediction \
  --loss_space native \
  --component_reduction fixed_dim \
  --loss_weighting none \
  --flow_solver heun \
  --flow_t_eps 0.05 \
  --snr_scale 1 \
  --diffusion_batch_mul 1 \
  --num_inference_steps 20 \
  --normalization global_standardize \
  --coordinate_packing isometric \
  --value_transform identity \
  --history_polar_features "${history_polar_features}" \
  --history_polar_fusion "${history_polar_fusion}" \
  --history_cartesian_features centered \
  --input_timestep_conditioning none \
  --history_corruption none \
  --logging_steps 25 \
  --timing_steps 100 \
  --timestep_histogram_bins 20 \
  --condition_diagnostic_steps 250 \
  --spectral_diagnostic_steps 500 \
  --spectral_panel_size 16 \
  --preview_steps 2500 \
  --preview_seed 12345 \
  --num_validation_images 16 \
  --checkpointing_steps 2500 \
  --checkpoints_total_limit 4 \
  --save_final_checkpoint \
  --no-final_eval \
  --report_to wandb \
  --tracker_project_name afig-continuous \
  --run_name "fft-polar-v3-${arm}-s1-${steps}" \
  --run_group fft-polar-v3-interface-gate \
  --run_tags "direct-fft,polar-v3,interface-gate,${arm},amplitude-v,physical-standardized,isometric,depth6,seed-1"
