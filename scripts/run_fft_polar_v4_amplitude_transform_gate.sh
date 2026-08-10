#!/usr/bin/env bash
set -euo pipefail

arm="${1:?usage: run_fft_polar_v4_amplitude_transform_gate.sh ARM [GPU] [STEPS]}"
gpu="${2:-any}"
steps="${3:-10000}"

case "${arm}" in
  logeps_ref)
    amplitude_transform="log_eps"
    transform_parameter="1"
    ;;
  log1p_tau1)
    amplitude_transform="log1p"
    transform_parameter="1"
    ;;
  inverse_softplus_tau2)
    amplitude_transform="inverse_softplus"
    transform_parameter="2"
    ;;
  inverse_softplus_tau5)
    amplitude_transform="inverse_softplus"
    transform_parameter="5"
    ;;
  power_p033)
    amplitude_transform="power"
    transform_parameter="0.3333333333333333"
    ;;
  power_p05)
    amplitude_transform="power"
    transform_parameter="0.5"
    ;;
  power_p067)
    amplitude_transform="power"
    transform_parameter="0.6666666666666666"
    ;;
  power_p08)
    amplitude_transform="power"
    transform_parameter="0.8"
    ;;
  raw)
    amplitude_transform="raw"
    transform_parameter="1"
    amplitude_source_scale="unit"
    run_group="fft-polar-v4-amplitude-transform-gate"
    ;;
  raw_frequency_rms)
    amplitude_transform="raw"
    transform_parameter="1"
    amplitude_source_scale="frequency_rms"
    run_group="fft-polar-v4-amplitude-source-gate"
    ;;
  raw_joint_condition)
    amplitude_transform="raw"
    transform_parameter="1"
    amplitude_source_scale="unit"
    condition_fusion="joint_mlp"
    run_group="fft-polar-v4-condition-fusion-gate"
    ;;
  *)
    echo "unknown amplitude-transform arm: ${arm}" >&2
    echo "expected logeps_ref, log1p_tau1, inverse_softplus_tau2, inverse_softplus_tau5, power_p033, power_p05, power_p067, power_p08, raw, raw_frequency_rms, or raw_joint_condition" >&2
    exit 2
    ;;
esac

amplitude_source_scale="${amplitude_source_scale:-unit}"
condition_fusion="${condition_fusion:-add}"
run_group="${run_group:-fft-polar-v4-amplitude-transform-gate}"
resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

project_root=/workspace/AFIG
output_dir="${project_root}/continuous_runs/ar_fft_polar_v4_ampcoord_${arm}_s1_${steps}"
codec_stats="${project_root}/autoencoder_runs/codec_stats_32_global_standardize.pt"

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "fft-polar-v4-ampcoord-${arm}-s1-${steps}" \
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
  "${resume_args[@]}" \
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
  --decoder_geometry factorized_polar \
  --factorized_coordinate_mode physical_standardized \
  --factorized_amplitude_transform "${amplitude_transform}" \
  --factorized_amplitude_transform_parameter "${transform_parameter}" \
  --factorized_log_epsilon 0.003 \
  --factorized_amplitude_standardization global \
  --factorized_amplitude_source_scale "${amplitude_source_scale}" \
  --factorized_condition_fusion "${condition_fusion}" \
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
  --history_polar_features physical_standardized_log_amp_phase \
  --history_polar_fusion replace \
  --history_polar_amplitude_transform log_eps \
  --history_polar_amplitude_transform_parameter 1 \
  --history_polar_log_epsilon 0.003 \
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
  --run_name "fft-polar-v4-ampcoord-${arm}-s1-${steps}" \
  --run_group "${run_group}" \
  --run_tags "direct-fft,polar-v4,amplitude-transform-${arm},amplitude-source-${amplitude_source_scale},condition-fusion-${condition_fusion},decoder-only-transform,log-history,physical-standardized,isometric,velocity,intrinsic-phase,bernoulli-sign,cartesian-primary,fixed-dim,depth6,seed-1"
