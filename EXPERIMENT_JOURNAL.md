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

- **2026-08-26 — autoencoder representation program:** prefix-decoding visual,
  completed fixed-cache PCA oracle, and staged encoder/objective/regularization
  exploration. The selected PCA prior control retains 1,536 dimensions (98.71%
  variance, clean rFID 4.65). [Plan](reports/2026-08-26_autoencoder_program/plan.md)
- **2026-08-25 — v9/v10 rate and shape controls:** the no-dimensional-bottleneck
  unordered learned `64x48` arm completed at FID 33.05 despite reconstruction
  FID 3.04. The learned `64x8`/`64x32` rate points and exact
  `32x32`/`128x8` prior-only reshapes also completed. `64x16` is the measured
  rate/modelability optimum; exact reshaping shows native token factorization
  matters independently of scalar count. All chained phases used the shared
  GPU queue. Full matrix and verdict:
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
