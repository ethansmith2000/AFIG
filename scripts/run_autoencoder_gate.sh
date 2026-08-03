#!/bin/bash
set -euo pipefail

mode="${1:?usage: run_autoencoder_gate.sh causal_k|causal_ring|spatial_downsample [SEED] [STEPS]}"
seed="${2:-0}"
steps="${3:-30000}"
case "${mode}" in
  causal_k|causal_ring|spatial_downsample) ;;
  *)
    echo "unknown autoencoder mode: ${mode}" >&2
    exit 2
    ;;
esac

resolution="${RESOLUTION:-32}"
batch_size="${BATCH_SIZE:-128}"
latent_dim="${LATENT_DIM:-64}"
variational="${VARIATIONAL:-false}"
kl_weight="${KL_WEIGHT:-0.0}"
conditioning="${GROUP_CONDITIONING:-film_low_rank}"
latent_noise="${LATENT_NOISE_STD:-0.0}"
ring_dropout="${LATENT_RING_DROPOUT:-0.0}"
high_dropout="${LATENT_HIGH_FREQUENCY_DROPOUT:-0.0}"

mode_args=()
case "${mode}" in
  causal_k)
    group_size="${GROUP_SIZE:-4}"
    pooler="${POOLER:-flat_mlp}"
    pool_suffix=""
    if [[ "${pooler}" == perceiver_* ]]; then
      pool_suffix="-p${PERCEIVER_WIDTH:-256}h${PERCEIVER_HEADS:-4}"
    fi
    run_name="ae-causal-k-k${group_size}-${pooler}${pool_suffix}-${conditioning}-z${latent_dim}-r${resolution}-s${seed}-n${steps}"
    mode_args=(
      --group_size "${group_size}"
      --pooler "${pooler}"
    )
    ;;
  causal_ring)
    target_tokens="${TARGET_TOKENS_PER_LATENT:-16}"
    max_latents="${MAX_RING_LATENTS:-4}"
    pooler="${POOLER:-perceiver_sector}"
    pool_suffix=""
    if [[ "${pooler}" == perceiver_* ]]; then
      pool_suffix="-p${PERCEIVER_WIDTH:-256}h${PERCEIVER_HEADS:-4}-seq${RING_TRANSFORMER_LAYERS:-2}"
    fi
    run_name="ae-causal-ring-t${target_tokens}-m${max_latents}-${pooler}${pool_suffix}-${conditioning}-z${latent_dim}-r${resolution}-s${seed}-n${steps}"
    mode_args=(
      --pooler "${pooler}"
      --target_tokens_per_latent "${target_tokens}"
      --max_ring_latents "${max_latents}"
    )
    ;;
  spatial_downsample)
    downsample="${SPATIAL_DOWNSAMPLE:-4}"
    latent_channels="${SPATIAL_LATENT_CHANNELS:-8}"
    run_name="ae-spatial-d${downsample}-c${latent_channels}-r${resolution}-s${seed}-n${steps}"
    mode_args=(
      --spatial_downsample "${downsample}"
      --spatial_latent_channels "${latent_channels}"
      --spatial_base_channels "${SPATIAL_BASE_CHANNELS:-64}"
    )
    ;;
esac
if [[ "${variational}" == true ]]; then
  run_name="${run_name}-vae-kl${kl_weight}"
fi
whiten_exponent="${WHITEN_EXPONENT:-1.0}"
if [[ "${whiten_exponent}" != 1.0 ]]; then
  run_name="${run_name}-wx${whiten_exponent}"
fi
if [[ "${latent_noise}" != 0 && "${latent_noise}" != 0.0 ]]; then
  run_name="${run_name}-noise${latent_noise}"
fi
if [[ "${ring_dropout}" != 0 && "${ring_dropout}" != 0.0 ]] || \
   [[ "${high_dropout}" != 0 && "${high_dropout}" != 0.0 ]]; then
  run_name="${run_name}-drop${ring_dropout}-${high_dropout}"
fi
output_dir="autoencoder_runs/${run_name}"

vae_args=(--no-variational)
if [[ "${variational}" == true ]]; then
  vae_args=(
    --variational
    --kl_weight "${kl_weight}"
    --kl_free_bits "${KL_FREE_BITS:-0.0}"
  )
fi

warmup="${WARMUP_STEPS:-500}"
if (( steps < warmup )); then warmup=$((steps / 5)); fi

cd /workspace/AFIG
exec gpu-claim run --owner AFIG --job "${run_name}" --wait -- \
  /venv/main/bin/python -u train_autoencoder.py \
  --mode "${mode}" \
  --dataset "${DATASET:-huggingface_cifar}" \
  --data_root "${DATA_ROOT:-/workspace/AFIG/data}" \
  --resolution "${resolution}" \
  --output_dir "${output_dir}" \
  --codec_stats_path "autoencoder_runs/codec_stats_${resolution}.pt" \
  --whiten_exponent "${whiten_exponent}" \
  --seed "${seed}" \
  --max_train_steps "${steps}" \
  --train_batch_size "${batch_size}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lr_warmup_steps "${warmup}" \
  --model_width "${MODEL_WIDTH:-128}" \
  --latent_dim "${latent_dim}" \
  --perceiver_width "${PERCEIVER_WIDTH:-256}" \
  --perceiver_heads "${PERCEIVER_HEADS:-4}" \
  --ring_transformer_layers "${RING_TRANSFORMER_LAYERS:-2}" \
  --group_conditioning "${conditioning}" \
  --conditioning_rank "${CONDITIONING_RANK:-16}" \
  --depth "${CAUSAL_DEPTH:-0}" \
  --kernel_size "${KERNEL_SIZE:-3}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --allow_tf32 \
  --eval_steps "${EVAL_STEPS:-1000}" \
  --preview_steps "${PREVIEW_STEPS:-5000}" \
  --logging_steps "${LOGGING_STEPS:-25}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-0}" \
  --save_final_checkpoint \
  --reconstruction_loss "${RECONSTRUCTION_LOSS:-mse}" \
  --token_loss_weight "${TOKEN_LOSS_WEIGHT:-0.01}" \
  --image_loss_weight "${IMAGE_LOSS_WEIGHT:-1.0}" \
  --fourier_loss_weight 0 \
  --log_amplitude_weight "${LOG_AMPLITUDE_WEIGHT:-0.0}" \
  --phase_loss_weight "${PHASE_LOSS_WEIGHT:-0.0}" \
  --phase_loss_gate "${PHASE_LOSS_GATE:-0.1}" \
  --radial_log_power_weight "${RADIAL_LOG_POWER_WEIGHT:-0.0}" \
  --loss_gradient_diagnostic_steps "${LOSS_GRADIENT_DIAGNOSTIC_STEPS:-1000}" \
  --latent_noise_std "${latent_noise}" \
  --latent_ring_dropout "${ring_dropout}" \
  --latent_high_frequency_dropout "${high_dropout}" \
  --latent_moment_weight "${LATENT_MOMENT_WEIGHT:-0.0}" \
  --report_to "${REPORT_TO:-wandb}" \
  --tracker_project_name "${WANDB_PROJECT:-afig-autoencoder}" \
  --run_group "${RUN_GROUP:-afig-autoencoder-reconstruction-gates}" \
  --run_name "${run_name}" \
  "${mode_args[@]}" \
  "${vae_args[@]}"
