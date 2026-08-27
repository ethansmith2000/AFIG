#!/bin/bash
# Isolate token-SNR scaling from its implicit MSE allocation with 1/a_i^2 weights.
set -euo pipefail

alpha="${1:-05}"
case "$alpha" in
  05)
    cache="tokenizer_runs/v5-vae-kl1e4-s1/latents_pow05.pt"
    alpha_label="0.50"
    ;;
  083)
    cache="tokenizer_runs/v5-vae-kl1e4-s1/latents_pow083.pt"
    alpha_label="0.83"
    ;;
  *)
    echo "usage: $0 [05|083]" >&2
    exit 2
    ;;
esac

cd /workspace/AFIG
source /venv/main/bin/activate

prior="prior_runs/v11-joint-pow${alpha}-compensated-vae-s1"
evaluation="prior_evals/v11-joint-pow${alpha}-compensated-vae-060000"
mkdir -p prior_runs prior_evals

gpu-claim run --owner AFIG --job "v11-pow${alpha}-comp-prior" --wait -- \
  python -u train_progressive_joint_flow.py \
    --latent_cache "$cache" \
    --output_dir "$prior" \
    --width 512 --depth 12 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --token_loss_weighting inverse_token_scale_squared \
    --checkpoint_every 2500 --keep_numbered_checkpoints 0 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group joint-prior-v11-compensated-snr \
    --run_name "v11-joint-pow${alpha}-compensated-vae-s1" \
    --compile > "${prior}.launch.log" 2>&1

python -u scripts/backup_wandb_file.py \
  "${prior}/checkpoint_final.pt" "v11-pow${alpha}-compensated-joint" \
  >> "${prior}.launch.log" 2>&1

gpu-claim run --owner AFIG --job "v11-pow${alpha}-comp-eval" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${prior}/checkpoint_final.pt" \
    --output_dir "$evaluation" \
    --num_samples 5000 --batch_size 256 --sample_steps 50 \
    > "${evaluation}.log" 2>&1

echo "COMPENSATED TOKEN SCALE alpha=${alpha_label} COMPLETE $(date -u +%FT%TZ)"
