#!/usr/bin/env bash
# Queue-managed construction of all four frozen power-whitened latent caches.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

cache_root="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
source_cache="${cache_root}/latents_final_original_flip.pt"
transform="reports/2026-08-26_autoencoder_program/regularized_whitening/selected_transform.pt"
log="reports/2026-08-26_autoencoder_program/power_whitening_cache_build.log"
failed="${cache_root}/power_whitening_cache_failed"

# Retain one lifetime claim across the suite so the four short transformations
# cannot be separated by unrelated queue entrants.
if [[ "${AFIG_POWER_CACHE_CLAIMED:-0}" != "1" ]]; then
  exec gpu-claim run --owner AFIG --job v27-power-whitening-caches --wait -- \
    env AFIG_POWER_CACHE_CLAIMED=1 "$0"
fi

trap 'touch "$failed"' ERR
mkdir -p "$(dirname "$log")"
: > "$log"

labels=(g0 g025 g05 g1)
gammas=(0 0.25 0.5 1)
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  gamma="${gammas[$index]}"
  output="${cache_root}/latents_factorized_power${label}_original_flip.pt"
  ready="${cache_root}/power_whitening_cache_${label}_ready"

  if [[ ! -f "$output" ]]; then
    python -u scripts/build_power_whitened_prior_cache.py \
      --cache "$source_cache" --transform "$transform" --output "$output" \
      --gamma "$gamma" --chunk_size 2048 --device cuda >> "$log" 2>&1
  fi

  python - "$output" "$gamma" <<'PY' >> "$log" 2>&1
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_gamma = float(sys.argv[2])
assert payload["latent_transform"]["type"] == "linear_inverse"
assert payload["whitening_config"]["type"] == "factorized_power"
assert math.isclose(payload["whitening_config"]["gamma"], expected_gamma)
assert payload["train_latents"].shape[1:] == (64, 16)
assert math.isfinite(float(payload["statistics"]["global_std"]))
assert float(payload["statistics"]["global_std"]) > 0
assert payload["whitening_config"]["cache_roundtrip_relative_rms"] <= 0.002
print(f"validated {sys.argv[1]} gamma={expected_gamma}")
PY
  touch "$ready"
done

rm -f "$failed"
trap - ERR
echo "POWER WHITENING CACHE SUITE COMPLETE $(date -u +%FT%TZ)" | tee -a "$log"
