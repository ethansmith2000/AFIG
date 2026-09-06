#!/usr/bin/env bash
# Queue-managed construction of channel, sequence, and flattened gamma-1 caches.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

cache_root="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
source_cache="${cache_root}/latents_final_original_flip.pt"
geometry="reports/2026-08-26_autoencoder_program/zca_whitening/zca_geometry.pt"
log="reports/2026-08-26_autoencoder_program/zca_axis_cache_build.log"
failed="${cache_root}/zca_axis_caches_failed"

# Keep one lifetime lock across these short builds so the suite cannot occupy
# multiple GPUs or lose its place between variants.
if [[ "${AFIG_ZCA_AXIS_CACHE_CLAIMED:-0}" != "1" ]]; then
  exec gpu-claim run --owner AFIG --job v27-zca-axis-caches --wait -- \
    env AFIG_ZCA_AXIS_CACHE_CLAIMED=1 "$0"
fi

trap 'touch "$failed"' ERR
mkdir -p "$(dirname "$log")"
: > "$log"

for variant in channel sequence flattened; do
  output="${cache_root}/latents_${variant}_zca_g1_original_flip.pt"
  ready="${cache_root}/zca_${variant}_g1_ready"
  if [[ ! -f "$output" ]]; then
    python -u scripts/build_zca_prior_cache.py \
      --cache "$source_cache" --geometry "$geometry" --output "$output" \
      --variant "$variant" --gamma 1 --chunk_size 2048 --device cuda \
      >> "$log" 2>&1
  fi

  python - "$output" "$variant" <<'PY' >> "$log" 2>&1
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
variant = sys.argv[2]
assert payload["latent_transform"]["type"] == "linear_inverse"
assert payload["whitening_config"]["type"] == f"{variant}_zca_power"
assert payload["whitening_config"]["variant"] == variant
assert payload["whitening_config"]["gamma"] == 1.0
assert payload["whitening_config"]["clean_token_magnitude_rescaling"] is False
assert payload["train_latents"].shape[1:] == (64, 16)
assert math.isfinite(float(payload["statistics"]["global_std"]))
assert float(payload["statistics"]["global_std"]) > 0
assert payload["whitening_config"]["cache_roundtrip_relative_rms"] <= 0.002
print(f"validated {variant}: {payload['whitening_config']}")
PY
  touch "$ready"
done

rm -f "$failed"
trap - ERR
echo "ZCA AXIS CACHE SUITE COMPLETE $(date -u +%FT%TZ)" | tee -a "$log"
