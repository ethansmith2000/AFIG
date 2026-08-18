#!/bin/bash
# Detached supervisor: advances the v4 campaign as dependencies complete.
set -u
cd /workspace/AFIG
source /venv/main/bin/activate
for i in $(seq 1 1440); do
  # FID evals for joints as they finish
  for name in det frontier; do
    ck="prior_runs/v4-joint-${name}-s1/checkpoint_final.pt"
    out="prior_evals/v4-joint-${name}-060000"
    if [ -f "$ck" ] && [ ! -e "$out/metrics.json" ] && [ ! -e "$out.launched" ]; then
      touch "$out.launched"
      setsid nohup gpu-claim run --owner AFIG --job "eval-joint-${name}-60k" --wait -- \
        python -u evaluate_progressive_joint_flow.py \
          --checkpoint "$ck" --output_dir "$out" > "$out.log" 2>&1 &
    fi
  done
  # ramp-s2: cache -> scorecard -> joint once tokenizer finishes
  if [ -f tokenizer_runs/v4-ramp-s2/metrics_final.json ] && [ ! -e tokenizer_runs/v4-ramp-s2/chain.launched ]; then
    touch tokenizer_runs/v4-ramp-s2/chain.launched
    setsid nohup bash scripts/run_v4_cache_and_joint.sh ramp-s2 > tokenizer_runs/v4-ramp-s2-chain.log 2>&1 &
  fi
  # ramp joint eval
  ck="prior_runs/v4-joint-ramp-s1/checkpoint_final.pt"
  out="prior_evals/v4-joint-ramp-060000"
  if [ -f "$ck" ] && [ ! -e "$out/metrics.json" ] && [ ! -e "$out.launched" ]; then
    touch "$out.launched"
    setsid nohup gpu-claim run --owner AFIG --job eval-joint-ramp-60k --wait -- \
      python -u evaluate_progressive_joint_flow.py \
        --checkpoint "$ck" --output_dir "$out" > "$out.log" 2>&1 &
  fi
  sleep 60
done
