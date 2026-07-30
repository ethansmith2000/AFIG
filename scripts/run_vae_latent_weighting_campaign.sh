#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-vae-kl0.0001}"
AE_CHECKPOINT="${AE_RUN}/checkpoint_30000.pt"
LATENT_INTERFACE="${AE_RUN}/latent_interface_posterior_mean.pt"
LOSS_WEIGHTS="${AE_RUN}/latent_loss_weights_posterior_mean.pt"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-${ROOT}/latent_continuous_runs/vae-posterior-mean-weighting}"

if [[ ! -f "${AE_CHECKPOINT}" ]]; then
  echo "Missing VAE checkpoint: ${AE_CHECKPOINT}" >&2
  exit 1
fi

# Validate reconstruction and fit every campaign artifact from deterministic
# posterior means. Posterior sampling is intentionally not enabled here.
/venv/main/bin/python "${ROOT}/evaluate_autoencoder_checkpoint.py" \
  "${AE_CHECKPOINT}" \
  --data_root "${ROOT}/data"

/venv/main/bin/python "${ROOT}/fit_autoencoder_latent_interface.py" \
  --checkpoint "${AE_CHECKPOINT}" \
  --output "${LATENT_INTERFACE}"

/venv/main/bin/python "${ROOT}/fit_latent_loss_weights.py" \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_interface "${LATENT_INTERFACE}" \
  --output "${LOSS_WEIGHTS}"

common_args=(
  --ae_checkpoint "${AE_CHECKPOINT}"
  --latent_interface "${LATENT_INTERFACE}"
  --seed 1
  --train_batch_size 256
  --max_train_steps 30000
  --width 768
  --num_layers 12
  --num_heads 12
  --ff_mult 4
  --diffusion_width 768
  --diffusion_depth 6
  --objective flow
  --prediction_type v_prediction
  --learning_rate 1e-4
  --lr_scheduler linear_floor
  --lr_end_ratio 0.25
  --lr_warmup_steps 2000
  --weight_decay 0.02
  --mixed_precision bf16
  --max_grad_norm 1.0
  --num_train_timesteps 1000
  --num_inference_steps 50
  --cfg_norm_match
  --checkpointing_steps 7500
  --preview_steps 2500
)

for weighting in unweighted raw_variance decoder_sensitivity; do
  extra_args=()
  if [[ "${weighting}" != "unweighted" ]]; then
    extra_args+=(--latent_loss_weights "${LOSS_WEIGHTS}")
  fi
  /venv/main/bin/python "${ROOT}/train_latent_continuous.py" \
    "${common_args[@]}" \
    "${extra_args[@]}" \
    --latent_loss_weighting "${weighting}" \
    --output_dir "${CAMPAIGN_DIR}/${weighting}" \
    --run_name "latent-afig-vae-mean-${weighting}-rf-w768-l12-d6-b256-s1-n30000"
done
