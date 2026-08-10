#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AE_RUN="${AE_RUN:-${ROOT}/autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq6-adaln_zero-z32-r32-s1-n30000-ringblock-vae-kl0.000001-global_standardize}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${AE_RUN}/checkpoint_30000.pt}"
TEST_METRICS="${AE_RUN}/cifar10_test_metrics.json"
MIN_PSNR="${MIN_PSNR:-34.0}"

while true; do
  # supervisorctl returns a non-zero status when the process is EXITED; that is
  # the success state this watcher is waiting for, so inspect its text instead.
  status="$(supervisorctl status afig-clean-ring-v2-z32-ae || true)"
  if [[ "${status}" == *"EXITED"* && -f "${AE_CHECKPOINT}" ]]; then
    break
  fi
  if [[ "${status}" == *"FATAL"* ]]; then
    echo "z32 AE service failed: ${status}" >&2
    exit 1
  fi
  sleep 30
done

cd "${ROOT}"
/workspace/bin/gpu-claim run \
  --owner AFIG \
  --job clean-ring-v2-z32-test \
  --wait \
  -- \
  /venv/main/bin/python -u evaluate_autoencoder_checkpoint.py \
    "${AE_CHECKPOINT}" \
    --data_root "${ROOT}/data" \
    --batch_size 128 \
    --num_workers 4

/venv/main/bin/python - "${TEST_METRICS}" "${MIN_PSNR}" <<'PY'
import json
import math
import sys

path, threshold_text = sys.argv[1:]
metrics = json.load(open(path))
psnr = float(metrics["reconstruction/psnr"])
mse = float(metrics["reconstruction/pixel_mse"])
threshold = float(threshold_text)
print(f"z32 held-out reconstruction: PSNR={psnr:.4f} dB pixel_MSE={mse:.8f}")
if not (math.isfinite(psnr) and math.isfinite(mse) and psnr >= threshold):
    raise SystemExit(
        f"z32 reconstruction gate failed: PSNR {psnr:.4f} < {threshold:.4f} dB"
    )
PY

"${ROOT}/scripts/fit_clean_ring_v2_z32_interface.sh"

supervisorctl start afig-clean-ring-v2-z32-joint
supervisorctl start afig-clean-ring-v2-z32-ar
