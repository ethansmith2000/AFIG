#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
phase_process=${PHASE_PROCESS:-geodesic_flow}
output_dir=${OUTPUT_DIR:-${project_root}/continuous_runs/ar_fft_factorized_polar_v2_full_eps01_global_d6_10k}

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "ar-fft-factorized-v2-full-${phase_process}-10000" \
  --wait \
  -- \
  /venv/main/bin/python train_continuous.py \
  --output_dir "${output_dir}" \
  --dataset huggingface_cifar \
  --data_root "${project_root}/data" \
  --seed 1 \
  --train_batch_size 32 \
  --dataloader_num_workers 4 \
  --max_train_steps 10000 \
  --learning_rate 7e-5 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --adam_weight_decay 0.1 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --width 768 \
  --num_layers 10 \
  --num_heads 12 \
  --ff_mult 4 \
  --qk_norm \
  --attention_rope frequency_2d \
  --rope_base 10000 \
  --transformer-position-film \
  --diff_width 768 \
  --diff_depth 6 \
  --decoder_geometry factorized_polar \
  --factorized_log_epsilon 0.1 \
  --factorized_amplitude_standardization global \
  --factorized_amplitude_loss_weight 1 \
  --factorized_phase_loss_weight 1 \
  --factorized_cartesian_loss_weight 0.1 \
  --factorized_phase_gate 0.1 \
  --factorized_phase_predicted_amplitude_probability 0.5 \
  --factorized_phase_process "${phase_process}" \
  --objective flow \
  --prediction_type v_prediction \
  --loss_space native \
  --component_reduction fixed_dim \
  --loss_weighting none \
  --flow_solver heun \
  --snr_scale 1 \
  --diffusion_batch_mul 1 \
  --num_inference_steps 20 \
  --ordering square_spiral \
  --normalization global_ecs \
  --coordinate_packing isometric \
  --ecs_percentile 98.25 \
  --history_polar_features standardized_log_amp_gated_phase \
  --history_polar_fusion replace \
  --history_cartesian_features centered \
  --logging_steps 25 \
  --timing_steps 100 \
  --condition_diagnostic_steps 250 \
  --spectral_diagnostic_steps 500 \
  --spectral_panel_size 16 \
  --preview_steps 2500 \
  --num_validation_images 16 \
  --checkpointing_steps 2500 \
  --checkpoints_total_limit 4 \
  --save_final_checkpoint \
  --report_to wandb \
  --tracker_project_name afig-continuous \
  --run_name "phase-b-ar-fft-factorized-v2-full-eps01-global-d6-${phase_process}-10k"
