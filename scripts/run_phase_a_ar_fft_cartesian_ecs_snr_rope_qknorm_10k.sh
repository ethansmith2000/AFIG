#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/continuous_runs/ar_fft_cartesian_ecs_snr_rope_qknorm_10k

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job ar_fft_cartesian_ecs_snr-rope-qknorm-10000 \
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
  --diff_width 768 \
  --diff_depth 3 \
  --objective flow \
  --prediction_type v_prediction \
  --loss_space native \
  --component_reduction fixed_dim \
  --loss_weighting none \
  --flow_solver heun \
  --snr_scale 4 \
  --diffusion_batch_mul 1 \
  --num_inference_steps 20 \
  --normalization global_ecs \
  --coordinate_packing isometric \
  --ecs_percentile 98.25 \
  --frequency_conditioning \
  --position_num_frequencies 4 \
  --position_max_frequency 8 \
  --backbone_position_mode legacy_hybrid \
  --position-input-addition \
  --transformer-position-film \
  --diffusion-target-conditioning \
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
  --run_name phase-a-ar-fft-cartesian-ecs-snr-rope-qknorm-10k
