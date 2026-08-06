#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG

cd "${project_root}"
exec env \
  ARM=local_dct_freqmajor \
  SEED=1 \
  STEPS=30000 \
  OUTPUT_DIR="${project_root}/continuous_runs/joint_c4_rate_local_dct_freqmajor_s1_30000" \
  "${project_root}/scripts/run_c4_joint_rate_arm.sh"
