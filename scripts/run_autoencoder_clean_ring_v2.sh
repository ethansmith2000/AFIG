#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Clean frequency-ring VAE gate:
# - one global pixel affine followed by exact isometric compact FFT;
# - full-ring bidirectionality and strict causality between rings;
# - affine-free LayerNorm, QK LayerNorm, and canonical AdaLN-Zero;
# - one shared 10 -> 4d -> d physical-metadata conditioner;
# - learned pooling queries guide attention but are not added to pooled values;
# - weak LDM-style KL regularization, evaluated by its actual weighted share.
export CODEC_NORMALIZATION="global_standardize"
export RING_BLOCK_CAUSAL="true"
export GROUP_CONDITIONING="adaln_zero"
export RING_TRANSFORMER_LAYERS="${RING_TRANSFORMER_LAYERS:-6}"
export TARGET_TOKENS_PER_LATENT="${TARGET_TOKENS_PER_LATENT:-12}"
export MAX_RING_LATENTS="${MAX_RING_LATENTS:-8}"
export LATENT_DIM="${LATENT_DIM:-64}"
export VARIATIONAL="true"
export KL_WEIGHT="${KL_WEIGHT:-0.000001}"
export KL_FREE_BITS="${KL_FREE_BITS:-0.0}"
export TOKEN_LOSS_WEIGHT="${TOKEN_LOSS_WEIGHT:-0.01}"
export IMAGE_LOSS_WEIGHT="${IMAGE_LOSS_WEIGHT:-1.0}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export PREVIEW_STEPS="${PREVIEW_STEPS:-2500}"

exec "${project_root}/scripts/run_autoencoder_gate.sh" causal_ring "${1:-1}" "${2:-30000}"
