#!/bin/bash
# One generation-engine run on the energycv cache: train -> backup -> FID eval.
# usage: run_v5_engine.sh <name> <trainer> [extra args...]
set -u
name="$1"; trainer="$2"; shift 2
cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate
cache="${ENGINE_CACHE:-tokenizer_runs/v5-vae-kl1e4-s1/latents_final_original_flip.pt}"
out="prior_runs/v5-${name}-vae-s1"
gpu-claim run --owner AFIG --job "v5-${name}-vae" --wait -- \
  python -u "$trainer" \
    --latent_cache "$cache" --output_dir "$out" \
    --width 512 --num_heads 8 --qk_norm rms \
    --batch_size 256 --num_workers 4 --learning_rate 1e-4 \
    --warmup_steps 1000 --max_train_steps 60000 \
    --report_to wandb --tracker_project_name afig-progressive-tokenizer \
    --run_group engine-comparison-v5 --run_name "v5-${name}-vae-s1" \
    --compile "$@" > "${out}.launch.log" 2>&1 \
|| { echo "ENGINE ${name} FAILED"; exit 1; }
python -u scripts/backup_wandb_file.py "${out}/checkpoint_final.pt" "v5-${name}-vae" >> "${out}.launch.log" 2>&1
gpu-claim run --owner AFIG --job "v5-eval-${name}" --wait -- \
  python -u evaluate_progressive_joint_flow.py \
    --checkpoint "${out}/checkpoint_final.pt" \
    --output_dir "prior_evals/v5-${name}-vae-060000" \
    > "prior_evals/v5-${name}-vae-060000.log" 2>&1
echo "ENGINE ${name} COMPLETE $(date -u +%FT%TZ)"
