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

- **2026-08-29 — seed-2 prior confirmation complete:** after resuming from the
  optimizer-bearing step-22,500 checkpoints, both frozen-tokenizer priors
  reached step 60,000 and completed paired 5k evaluations. The residual pool
  again wins: FID **25.48**/KID **0.01634** versus cross-only FID 27.21/KID
  0.01891, a 1.73-point FID and 0.00257 KID improvement. Across the two paired
  5k prior seeds, mean FID is **26.31** versus 28.57 (mean paired gain 2.26).
  The seed-2 direction replicates, but its 1.73-point gap narrowly misses the
  predeclared greater-than-2 single-run confirmation gate. The residual pool
  remains the leading architecture; the next discriminating test is a second
  paired tokenizer seed rather than another prior seed on the same caches.
  Final metrics are durable in Git commit `8668d96`. W&B retained the cross-only
  seed-2 prior checkpoint but the residual upload crashed; neither checkpoint
  was transferred locally, and the missing residual weights do not affect the
  completed verdict.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md)
- **2026-08-28 — residual-pool generative promotion:** the parameter-matched
  residual register pool improves matched-prior FID from 29.93 to **27.13** at
  5k samples and from 27.38 to **24.85** in a paired 10k evaluation. The
  2.53-point larger-sample gain clears the predeclared modelability gate, while
  KID improves from 0.02040 to 0.01910. This is now the leading tokenizer
  architecture. Paired prior seed-2 confirmations on the frozen residual and
  cross-only caches are running; they test prior-training stochasticity before
  spending compute on a second tokenizer seed.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md)
- **2026-08-29 — seed-2 confirmation recovery:** both paired prior confirmations
  were externally interrupted after step 23k; neither produced an evaluation.
  Their optimizer-bearing step-22,500 checkpoints are valid. Launchers now
  auto-resume and skip completed phases, and supervisor owns persistent queue
  waiters so the jobs continue when GPUs become available without bypassing
  `/workspace/GPU_QUEUEING.md`.
- **2026-08-27 — first autoencoder-program decisions:** the raw rank-1,536 PCA
  prior is rejected at FID 170.53/KID 0.1692 despite its 4.65 reconstruction
  FID oracle. Its retained coefficients have a 14.9x token-RMS range (221.8x
  first/last token power), and the trained flow remains at MSE 1.192; this is a
  representation/modelability failure, not an inverse-decoder bug. The
  compensated alpha-0.50 schedule is also rejected at FID 39.89, only 1.04
  better than its uncompensated version and 4.04 worse than the flat progressive
  prior. Conversely, the parameter-matched residual-pool tokenizer clears its
  promotion gate: PSNR 37.20 versus 35.88 and rFID improves by
  0.73/0.71/0.59 at sigma 0/.10/.20. Its matched 60k prior is now running.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md)
- **2026-08-27 — prefix increments and isolated representation controls:** the
  512-example decoder-increment audit is complete. Progressive prefix training
  makes additions reliably useful but does not create a monotonic frequency
  ladder: token-index/spectral-centroid Spearman is 0.070, adjacent centroids
  ascend 49.2% of the time, and the decoder path length is 8.01x its direct
  displacement. Two matched 60k priors are running through `gpu-claim`: the
  selected rank-1,536 PCA cache (`64x24`, unchanged `64x48` decoder after exact
  inverse projection) and the progressive alpha-0.50 token-scale cache with
  mean-one `1/a_i^2` flow-loss compensation. The PCA gate is FID 29.93 +/- 2;
  the schedule gate is the flat progressive FID 35.85 +/- 2. A parameter-exact
  Stage-A tokenizer control is also queued: it reallocates one of eight patch
  transformer blocks to one residual register-refinement block (`e7+p1`),
  retaining 60,056,784 parameters and every other v8 setting. It stops after
  the 15k distortion/robustness screen unless it moves the measured frontier.
  Artifacts and live
  status: [autoencoder program](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-08-26 — autoencoder representation program:** prefix-decoding visual,
  completed fixed-cache PCA oracle, staged encoder/objective/regularization
  exploration, and CPU-only image/latent SNR crossing maps. Raw-image radial
  bands cross SNR 1 from `t=0.175` through `0.847`, with one color mode carrying
  93.7-95.8% of band variance and per-image adjacent order above 99.5%. Learned
  active tokens instead cross mostly near `t=0.5`; progressive prefix semantics
  therefore do not currently imply an image-like magnitude schedule. The
  selected PCA prior control retains 1,536 dimensions (98.71% variance, clean
  rFID 4.65). [Plan](reports/2026-08-26_autoencoder_program/plan.md)
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
