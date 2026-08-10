#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
output_dir=${project_root}/latent_continuous_runs/scaffold_fft_causal_ring_local_oracle_c4_s1_10000

mkdir -p "${output_dir}"
cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job scaffold-fft-causal-ring-local-oracle-c4-10k \
  --wait \
  -- \
  /bin/bash scripts/run_scaffold_fft_causal_ring_local_claimed.sh
