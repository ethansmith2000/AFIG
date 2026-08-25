# AFIG experiment journal

This is the chronological index for experiments that consume meaningful GPU
time or change a project-level conclusion. Detailed plans, evidence, and
verdicts live under `reports/` and are linked here.

## Operating rules

- Every exclusive GPU phase runs through the shared lifetime-locking launcher
  documented in `/workspace/GPU_QUEUEING.md`:
  `gpu-claim run --owner AFIG --job <name> --wait -- <command>`.
- `/tmp/vast_gpu_claims` and its held flock are authoritative. `nvidia-smi`
  memory or project-local metadata is never treated as a reservation.
- Before launch, record the hypothesis, control, configuration, decision metric,
  and stopping/continuation rule. After completion, record exact artifact paths,
  FID/KID and reconstruction metrics, known failures, and the resulting verdict.
- Decoded FID is the selection metric. FID-5k gaps below roughly 2 points are
  unresolved and require another seed; fixed-seed repeatability is not decision
  noise.
- Commit code before long runs. Preserve optimizer-bearing resume checkpoints,
  and commit durable metrics/reports after evaluation.

## Campaigns

- **2026-08-25 — v9 dimensional-rate control:** launched the predeclared
  unordered learned `64x48` tokenizer and matched joint prior. This removes the
  `1,024 / 3,072` dimensional bottleneck while exactly matching the pixel arm's
  literal token/feature shape. The chained launcher also records reconstruction,
  axis utilization, decoded FID/KID, and decoder sensitivity through the shared
  GPU queue. Launcher: `scripts/run_v9_unordered_rate_control.sh`; live record:
  [v8 decisive-controls plan](reports/2026-08-23_v8_decisive_controls/plan.md).
- **2026-08-23 — v8 decisive controls:** matched unordered tokenizer, matched
  pixel-space joint RF, direct context ablation, and planned latent rate/shape
  decomposition. [Plan and live record](reports/2026-08-23_v8_decisive_controls/plan.md)
- **2026-08-23 — conditioning proxy correction:** energy correlation is a
  dependence proxy, not direct conditioning gain.
  [Results and correction](reports/2026-08-23_conditioning_axes/results.md)
- **2026-08-23 — external review:** adversarial review and corrected premises.
  [Review](reports/2026-08-23_external_review/review.md)
- **2026-08-21 — rolling exposure diagnosis:**
  [Results](reports/2026-08-21_rolling_exposure_diagnosis/results.md)
- **2026-08-19 — v5 shaping verdict:**
  [Results](reports/2026-08-19_v5_shaping_verdict/results.md)
