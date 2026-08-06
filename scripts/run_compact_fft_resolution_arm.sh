#!/usr/bin/env bash
set -euo pipefail

project_root=/workspace/AFIG
arm="${ARM:-scaled}"
image_size="${IMAGE_SIZE:-16}"
steps="${STEPS:-10000}"
exponent="${SPECTRAL_SCALE_EXPONENT:-0.8}"
patch="${PATCH:-4}"
compact_token_dim="${COMPACT_TOKEN_DIM:-48}"

case "${arm}" in
  pixel)
    representation=pixels
    ;;
  unscaled)
    representation=fft_compact_isometric_spiral
    ;;
  scaled)
    representation=fft_compact_scaled_spiral
    ;;
  *)
    echo "ARM must be pixel, unscaled, or scaled" >&2
    exit 2
    ;;
esac

output_dir="${OUTPUT_DIR:-${project_root}/latent_continuous_runs/res${image_size}_${arm}_s1_${steps}}"

cd "${project_root}"
exec /workspace/bin/gpu-claim run \
  --owner AFIG \
  --job "res${image_size}-${arm}-s1-${steps}" \
  --wait \
  -- \
  /venv/main/bin/python -u control_pixel_diffusion.py \
  --output_dir "${output_dir}" \
  --representation "${representation}" \
  --spectral_scale_exponent "${exponent}" \
  --compact_token_dim "${compact_token_dim}" \
  --image_size "${image_size}" \
  --patch "${patch}" \
  --steps "${steps}" \
  --batch_size 256 \
  --width 768 \
  --num_layers 12 \
  --num_heads 12 \
  --preview_steps 2500
