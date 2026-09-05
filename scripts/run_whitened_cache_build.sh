#!/usr/bin/env bash
# Queue-managed construction of the one frozen factorized-whitened cache.
set -euo pipefail

cd /workspace/AFIG
export PATH="$PATH:/workspace/bin"
source /venv/main/bin/activate

source_cache="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_final_original_flip.pt"
transform="reports/2026-08-26_autoencoder_program/regularized_whitening/selected_transform.pt"
output="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/latents_factorized_whiten16_original_flip.pt"
log="reports/2026-08-26_autoencoder_program/whitened_cache_build.log"
ready="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/whitened_cache_ready"
failed="tokenizer_runs/v27-residual-e7p1-det-jitter05-slotbal2e3-n64d16-s2/whitened_cache_failed"
trap 'touch "$failed"' ERR

if [[ ! -f "$output" ]]; then
  gpu-claim run --owner AFIG --job v27-factorized-whiten16-cache --wait -- \
    python -u scripts/build_whitened_prior_cache.py \
      --cache "$source_cache" --transform "$transform" --output "$output" \
      --chunk_size 2048 --device cuda > "$log" 2>&1
fi

/venv/main/bin/python - "$output" <<'PY'
import math
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert payload["latent_transform"]["type"] == "linear_inverse"
assert payload["whitening_config"]["type"] == "factorized_sequence_channel"
assert payload["whitening_config"]["relative_gain_cap"] == 16.0
assert payload["train_latents"].shape[1:] == (64, 16)
assert math.isfinite(float(payload["statistics"]["global_std"]))
assert float(payload["statistics"]["global_std"]) > 0
assert payload["whitening_config"]["cache_roundtrip_relative_rms"] <= 0.002
PY
touch "$ready"
rm -f "$failed"
trap - ERR

echo "WHITENED CACHE BUILD COMPLETE $(date -u +%FT%TZ)"
