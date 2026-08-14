#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AFIG
mkdir -p logs audit_2026-08-12/eval_metrics/ar_n64d16_prefix

checkpoint=prior_runs/v2-ar-cross-n64-d16-s1/checkpoint_final.pt
cache=tokenizer_runs/v2-cross-n64-d16-s1/latents_final_original_flip.pt
prefixes=(1 2 4 8 16 32 48)
pids=()

for prefix in "${prefixes[@]}"; do
  gpu-claim run \
    --owner AFIG \
    --job "ar-n64d16-oracle-prefix-${prefix}" \
    --wait -- \
    /venv/main/bin/python -u evaluate_progressive_ar_prefix.py \
      --checkpoint "$checkpoint" \
      --latent_cache "$cache" \
      --output "audit_2026-08-12/eval_metrics/ar_n64d16_prefix/prefix_${prefix}.json" \
      --prefix_length "$prefix" \
      --num_samples 5000 \
      --batch_size 256 \
      --sample_steps 50 \
      >"logs/ar_n64d16_oracle_prefix_${prefix}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed prefix ${prefixes[$index]}"
  else
    arm_status="$?"
    echo "failed prefix ${prefixes[$index]} status=${arm_status}" >&2
    status="$arm_status"
  fi
done
exit "$status"
