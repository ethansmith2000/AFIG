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

- **2026-09-02 — complete three-seed architecture verdict:** at 10k samples,
  v8/v12/v13 FIDs are 27.38/24.85/31.18 for tokenizer seed 1,
  28.07/29.59/33.29 for seed 2, and 40.04/35.75/33.74 for seed 3. Each arm wins
  one seed. V12 has the best mean FID/KID (30.06/0.02309), while v13 has by far
  the smallest FID standard deviation (1.12) and best worst-case FID (33.74).
  V13's seed-3 win also repeats with prior seed 2, 32.04 versus v12's 35.24.
  Select v12 residual as the expected-performance baseline and retain v13 as
  the stability control; the scientifically important finding is substantial
  architecture-by-tokenizer-seed interaction.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — open GPUs assigned to architecture robustness:** v13 tokenizer
  seed 2 completed its 15k tokenizer phase at PSNR 34.68 and is in artifact/
  diagnostic processing before its matched prior. In parallel, frozen
  tokenizer-seed-3 v12/v13 priors are training with prior seed 2 at W&B
  [`6pnsnulc`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/6pnsnulc)
  and [`1lxjnhll`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/1lxjnhll),
  while v13 tokenizer seed 1 is training at
  [`hh07ajxg`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/hh07ajxg).
  These jobs use three additional `gpu-claim` locks and leave GPUs 5--7 free.
  [Full design](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — larger-sample architecture verdict:** paired 10k evaluation
  preserves the seed-2 reversal: v8 FID/KID 28.07/0.02046 versus v12
  29.59/0.02255, so residual loses by 1.52 FID. At tokenizer seed 3, v13
  register tokens reach FID/KID **33.74/0.02743** versus v12
  35.75/0.02762, a 2.01-FID win with the KID direction now agreeing. V13 is
  therefore the seed-3 winner, but it has only one tokenizer seed. The next
  efficient confirmation is v13 tokenizer seed 2 against the already completed
  seed-2 v8/v12 controls. Its end-to-end tokenizer/prior chain started on GPU 0
  at 2026-09-01 21:10 UTC under `gpu-claim`; tokenizer W&B run
  [`m49qqabv`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/m49qqabv).
  [Durable comparison](reports/2026-08-26_autoencoder_program/matched_prior_architecture_comparison.json)
  and [full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — five matched priors complete; two comparisons remain close:**
  on tokenizer seed 2, v8 reaches FID/KID 30.31/0.02019 versus v12
  31.97/0.02270, so residual pooling is 1.66 FID worse. On tokenizer seed 3,
  v8/v12/v13 reach FID 42.60/37.74/36.35 and KID
  0.03427/0.02704/0.02777. Both alternative encoders clearly beat the poor v8
  seed, but v13's 1.38-FID advantage over v12 is below the decision threshold
  and KID slightly favors v12. All checkpoints are backed up to W&B. Paired
  10k evaluations were therefore run for seed-2 v8/v12 and seed-3 v12/v13;
  their verdict is recorded above.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — reconstruction gate corrected before seed-3 results:** clean
  reconstruction and decoder sensitivity are codec-health diagnostics, not
  selectors for latent quality. The former `>=0.5` rFID-improvement promotion
  rule could reject a representation with slightly worse distortion but a much
  easier prior distribution. Every finite, semantically coherent arm in the
  permissive historical health envelope will therefore receive a matched 60k
  joint prior; paired decoded FID/KID makes the architecture decision. This
  correction reopens the healthy tokenizer-seed-2 v8/v12 caches for the paired
  generative comparison previously skipped.
  The paired seed-2 priors launched as W&B `oytltgh0` (v8) and `4um6874y`
  (v12). The three healthy seed-3 priors launched as `kir5jvch` (v8),
  `gtoqstp3` (v12), and `hj15kjc0` (v13). All five are supervisor-owned,
  resumable, and hold lifetime GPU claims.
  [Current roadmap](ROADMAP.md) and
  [full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — seed-3 tokenizers complete and all pass codec health:** v8,
  v12, and v13 reached 15k and completed caches, axis diagnostics, sensitivity,
  and W&B checkpoint backup. PSNR is 31.76/33.84/36.39 and clean rFID is
  10.59/7.65/5.77 respectively. Flattened effective rank moves in the opposite
  direction at 137.96/196.06/344.02, illustrating why reconstruction cannot
  select latent quality. All three therefore advance to paired prior-seed-1
  60k joint flows under the corrected protocol.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-01 — seed-3 register-formation screen predeclared:** the next
  Stage-A screen compares three parameter-exact `64x16` full-only tokenizers:
  v8 cross-only (`e8`), v12 residual pooling (`e7+p1`), and a new true
  register-token arm (`e7+j1`) in which patches and learned registers share a
  bidirectional block before register-only refinement. All have 60,056,784
  parameters, seed 3, and a 15k budget. The original reconstruction-improvement
  gate in this entry was superseded before results by the codec-health rule
  above; every healthy arm receives a matched prior. The
  follow-on prior experiment is now specified as an endpoint-preserving
  token/group time warp with unchanged clean magnitudes, per-token log-SNR
  conditioning, base-displacement prediction, and independently controlled
  loss weights; it is not part of this representation screen.
  All three resumable arms launched at `2026-09-01T18:59:32Z` through
  `gpu-claim` under supervisor. W&B runs are `38zmd550` (v8), `qn4em7zo`
  (v12), and `etkl1hma` (v13); each completed compilation and entered steady
  training before handoff.
  [Current roadmap](ROADMAP.md) and
  [full record](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-08-31 — tokenizer-seed-2 architecture confirmation fails:** the
  parameter-matched v8/v12 pair completed 15k steps plus full-test
  reconstruction and latent-noise sensitivity. Residual pooling is essentially
  tied but slightly worse at seed 2: PSNR 35.15 versus 35.27, and rFID is worse
  by 0.07/0.05/0.11/0.13 at sigma 0/.05/.10/.20 (0.18 better only at sigma
  .40). It therefore does not clear the unchanged 0.5-rFID Stage-A gate, and no
  matched priors will be trained on these caches. The strong seed-1 checkpoint
  and its prior results remain real, but the improvement cannot yet be assigned
  robustly to the residual-pool architecture. Both seed-2 tokenizer checkpoints
  are preserved as W&B artifacts.
  [Full record](reports/2026-08-26_autoencoder_program/plan.md)
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
