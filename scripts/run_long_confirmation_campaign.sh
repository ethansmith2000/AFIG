#!/bin/bash
set -uo pipefail

cd /workspace/AFIG

pids=()

terminate_children() {
  if ((${#pids[@]} > 0)); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap terminate_children INT TERM

for arm in H-anchor H-finalist1 H-sincos; do
  for seed in 0 1; do
    GRADIENT_CHECKPOINTING=false \
      scripts/run_architecture_gate.sh "${arm}" "${seed}" 100000 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
