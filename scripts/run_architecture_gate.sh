#!/bin/bash
set -euo pipefail

arm="${1:?usage: run_architecture_gate.sh ARM [SEED] [STEPS]}"
seed="${2:-0}"
steps="${3:-5000}"

batch_size="${BATCH_SIZE:-128}"
position_mode="${POSITION_MODE:-sincos_table}"
adam_beta1="${ADAM_BETA1:-0.9}"
adam_beta2="${ADAM_BETA2:-0.99}"
polar="${POLAR_MODE:-log_amp_gated_phase}"
target_position="${TARGET_POSITION:-true}"
history_cartesian="${HISTORY_CARTESIAN:-centered}"
centering="${CENTERING:-all}"
input_time="none"
input_init="xavier"
history_corruption="none"
history_noise_max="0.05"
loss_metric="${LOSS_METRIC:-normalized}"
loss_alpha="${LOSS_ALPHA:-0.0}"
learned_gain="${LEARNED_GAIN:-false}"

case "${arm}" in
  P0) position_mode="none" ;;
  P1) position_mode="random_table" ;;
  P2) position_mode="sincos_table" ;;
  B-default) ;;
  B-beta) adam_beta1="0.95"; adam_beta2="0.995" ;;
  B-polar-off) polar="none" ;;
  B-target-off) target_position=false ;;
  R0) history_cartesian="centered" ;;
  R1) history_cartesian="phase_preserving" ;;
  R2) history_cartesian="phase_preserving"; polar="none" ;;
  N0) centering="all"; history_cartesian="phase_preserving"; polar="none" ;;
  N1) centering="self_conjugate_std"; history_cartesian="phase_preserving"; polar="none" ;;
  N2) centering="self_conjugate_rms"; history_cartesian="phase_preserving"; polar="none" ;;
  S0) input_time="none" ;;
  S1) input_time="film" ;;
  S1-kaiming) input_time="film"; input_init="kaiming_linear" ;;
  F-alpha0) loss_metric="orbit_scale_power"; loss_alpha="0.0" ;;
  F-alpha02) loss_metric="orbit_scale_power"; loss_alpha="0.2" ;;
  F-alpha1) loss_metric="orbit_scale_power"; loss_alpha="1.0" ;;
  F-gain) loss_metric="orbit_scale_power"; loss_alpha="${LOSS_ALPHA:-0.2}"; learned_gain=true ;;
  G-clean) history_corruption="none" ;;
  G-noise) history_corruption="gaussian" ;;
  H-anchor)
    position_mode="none"
    history_cartesian="centered"
    polar="log_amp_gated_phase"
    centering="all"
    input_time="none"
    loss_metric="normalized"
    loss_alpha="0.0"
    learned_gain=false
    ;;
  H-finalist1)
    position_mode="random_table"
    history_cartesian="phase_preserving"
    polar="none"
    centering="all"
    input_time="none"
    loss_metric="orbit_scale_power"
    loss_alpha="0.2"
    learned_gain=true
    ;;
  H-finalist2)
    position_mode="random_table"
    history_cartesian="centered"
    polar="log_amp_gated_phase"
    centering="all"
    input_time="none"
    loss_metric="orbit_scale_power"
    loss_alpha="0.2"
    learned_gain=true
    ;;
  *)
    echo "unknown architecture arm: ${arm}" >&2
    exit 2
    ;;
esac

warmup_steps="${WARMUP_STEPS:-500}"
if (( steps < warmup_steps )); then
  warmup_steps=$((steps / 5))
  if (( warmup_steps < 10 )); then warmup_steps=10; fi
fi
preview_steps="${PREVIEW_STEPS:-5000}"
if (( steps < preview_steps )); then preview_steps=0; fi
diagnostic_steps="${SPECTRAL_STEPS:-1000}"
if (( steps < diagnostic_steps )); then
  diagnostic_steps=$((steps / 5))
  if (( diagnostic_steps < 25 )); then diagnostic_steps=25; fi
fi

target_args=(--diffusion-target-conditioning)
if [[ "${target_position}" == false ]]; then
  target_args=(--no-diffusion-target-conditioning)
fi
gain_args=()
if [[ "${learned_gain}" == true ]]; then gain_args=(--learned_output_gain); fi

run_name="arch-${arm,,}-s${seed}-b${batch_size}-n${steps}"
stats="continuous_runs/codec_stats_${centering}_heldout${SPECTRAL_PANEL_SIZE:-16}.pt"

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job "${run_name}" --wait -- \
  /venv/main/bin/python -u train_continuous.py \
  --output_dir "continuous_runs/${run_name}" \
  --codec_stats_path "${stats}" \
  --dataset huggingface_cifar \
  --seed "${seed}" \
  --max_train_steps "${steps}" \
  --preview_steps "${preview_steps}" \
  --preview_seed 12345 \
  --spectral_diagnostic_steps "${diagnostic_steps}" \
  --spectral_panel_size "${SPECTRAL_PANEL_SIZE:-16}" \
  --spectral_diagnostic_seed 1729 \
  --condition_diagnostic_steps 0 \
  --logging_steps "${LOGGING_STEPS:-25}" \
  --timing_steps "${TIMING_STEPS:-100}" \
  --no-final_eval \
  --checkpointing_steps 0 \
  --num_layers 10 \
  --width 768 \
  --num_heads 12 \
  --diff_width 768 \
  --diff_depth 6 \
  --train_batch_size "${batch_size}" \
  --diffusion_batch_mul 1 \
  --learning_rate 1e-4 \
  --lr_warmup_steps "${warmup_steps}" \
  --adam_beta1 "${adam_beta1}" \
  --adam_beta2 "${adam_beta2}" \
  --grad_norm_mode clip \
  --objective ddpm \
  --prediction_type x0 \
  --loss_space native \
  --loss_weighting min_snr \
  --min_snr_gamma 0.2 \
  --loss_metric "${loss_metric}" \
  --orbit_scale_exponent "${loss_alpha}" \
  "${gain_args[@]}" \
  --timestep_spacing linspace \
  --num_inference_steps 20 \
  --value_transform identity \
  --normalization orbit_standardize \
  --centering "${centering}" \
  --history_cartesian_features "${history_cartesian}" \
  --history_polar_features "${polar}" \
  --frequency_conditioning \
  --backbone_position_mode "${position_mode}" \
  --position-input-addition \
  --input_position_scale_init 0.1 \
  --position-rms-normalize \
  --no-transformer-position-film \
  "${target_args[@]}" \
  --input_timestep_conditioning "${input_time}" \
  --input_projection_init "${input_init}" \
  --history_corruption "${history_corruption}" \
  --history_corruption_prob 1.0 \
  --history_noise_min 0.0 \
  --history_noise_max "${history_noise_max}" \
  --history_noise_ramp_fraction 0.2 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --allow_tf32 \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "${run_name}" \
  --run_group "afig-coefficient-architecture-gates" \
  --run_tags "architecture-gates,${arm},seed-${seed},position-${position_mode},centering-${centering},history-${history_cartesian},polar-${polar},input-time-${input_time},input-init-${input_init},adam-${adam_beta1}-${adam_beta2},loss-${loss_metric}-${loss_alpha},batch-${batch_size},depth6"
