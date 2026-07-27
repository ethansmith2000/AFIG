#!/bin/bash
set -euo pipefail

phase="${1:?usage: run_architecture_campaign.sh smoke|A|B|C|D|E|F|G|H}"
steps_override="${2:-}"
seed="${SEED:-0}"
runner="/workspace/AFIG/scripts/run_architecture_gate.sh"

case "${phase}" in
  smoke)
    arms=(P1 P2 R1 N1 N2 S1 S1-kaiming F-alpha02 F-gain G-noise)
    steps="${steps_override:-75}"
    ;;
  A) arms=(P0 P1 P2); steps="${steps_override:-5000}" ;;
  B) arms=(B-default B-beta B-polar-off B-target-off); steps="${steps_override:-5000}" ;;
  C) arms=(R0 R1 R2); steps="${steps_override:-5000}" ;;
  D) arms=(N0 N1 N2); steps="${steps_override:-5000}" ;;
  E) arms=(S0 S1); steps="${steps_override:-5000}" ;;
  F) arms=(F-alpha0 F-alpha02 F-alpha1 F-gain); steps="${steps_override:-5000}" ;;
  G) arms=(G-clean G-noise); steps="${steps_override:-5000}" ;;
  H) arms=(H-anchor H-finalist1 H-finalist2); steps="${steps_override:-100000}" ;;
  *)
    echo "unknown campaign phase: ${phase}" >&2
    exit 2
    ;;
esac

pids=()
for arm in "${arms[@]}"; do
  "${runner}" "${arm}" "${seed}" "${steps}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
