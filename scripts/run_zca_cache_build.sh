#!/usr/bin/env bash
# Queue-managed construction of the selected axial gamma-1 ZCA cache.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

cache_root="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2"
source_cache="${cache_root}/latents_final_original_flip.pt"
geometry="reports/2026-08-26_autoencoder_program/zca_whitening/zca_geometry.pt"
output="${cache_root}/latents_axial_zca_g1_original_flip.pt"
log="reports/2026-08-26_autoencoder_program/zca_cache_build.log"
ready="${cache_root}/zca_cache_g1_ready"
failed="${cache_root}/zca_cache_g1_failed"
trap 'touch "$failed"' ERR

if [[ ! -f "$output" ]]; then
  gpu-claim run --owner AFIG --job v27-axial-zca-g1-cache --wait -- \
    python -u scripts/build_zca_prior_cache.py \
      --cache "$source_cache" --geometry "$geometry" --output "$output" \
      --gamma 1 --chunk_size 2048 --device cuda > "$log" 2>&1
fi

/venv/main/bin/python - "$output" <<'PY'
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert payload["latent_transform"]["type"] == "linear_inverse"
assert payload["whitening_config"]["type"] == "axial_zca_power"
assert payload["whitening_config"]["gamma"] == 1.0
assert payload["whitening_config"]["clean_token_magnitude_rescaling"] is False
assert payload["train_latents"].shape[1:] == (64, 16)
assert math.isfinite(float(payload["statistics"]["global_std"]))
assert float(payload["statistics"]["global_std"]) > 0
assert payload["whitening_config"]["cache_roundtrip_relative_rms"] <= 0.002
PY
touch "$ready"
rm -f "$failed"
trap - ERR

echo "ZCA CACHE BUILD COMPLETE $(date -u +%FT%TZ)"
