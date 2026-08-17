#!/bin/bash
# v4 shaping campaign: cache finished tokenizer arms, run the axis scorecard,
# then queue matched 60k joint priors. Detached-safe (run under setsid).
set -u
cd /workspace/AFIG
source /venv/main/bin/activate

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(vae-kl1e4-s1 energycv-s1)

for arm in "${ARMS[@]}"; do
  run="tokenizer_runs/v4-${arm}"
  cache="${run}/latents_final_original_flip.pt"
  if [ ! -f "$cache" ]; then
    gpu-claim run --owner AFIG --job "cache-v4-${arm}" --wait -- \
      python -u cache_progressive_latents.py \
        --tokenizer_checkpoint "${run}/checkpoint_final.pt" \
        --output "$cache" --include_horizontal_flip \
      >> "${run}/cache.log" 2>&1 || { echo "CACHE FAILED ${arm}"; continue; }
  fi
  python -u scripts/analyze_axis_scorecard.py \
    --cache "$cache" --output "${run}/axis_scorecard.json" \
    > "${run}/axis_scorecard.log" 2>&1
  name="${arm%-s*}"
  setsid nohup gpu-claim run --owner AFIG --job "v4-joint-${name}" --wait -- \
    python -u train_progressive_joint_flow.py \
      --latent_cache "$cache" \
      --output_dir "prior_runs/v4-joint-${name}-s1" \
      --width 512 --depth 12 --num_heads 8 --qk_norm rms \
      --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
      --warmup_steps 1000 --max_train_steps 60000 \
      --report_to wandb --tracker_project_name afig-progressive-tokenizer \
      --run_group joint-prior-v4-shaping --run_name "v4-joint-${name}-s1" \
      --compile \
    > "prior_runs/v4-joint-${name}-s1.launch.log" 2>&1 &
  echo "queued joint prior for ${arm}"
done
echo "PIPELINE LAUNCH DONE $(date -u +%FT%TZ)" > /workspace/AFIG/tokenizer_runs/v4_pipeline_done.marker
