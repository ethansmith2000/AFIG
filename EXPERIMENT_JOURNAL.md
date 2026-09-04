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

- **2026-09-04 — generation is coarse-to-fine in distributed subspaces, not
  native token order:** compare
  selected v27, v34 common time, and v34 soft25 from identical standardized
  Gaussian seeds through their actual 50-step Heun samplers. At eleven times,
  save the noisy latent state and the velocity field's predicted-clean endpoint;
  decode fixed examples and measure stabilization in image pixels, RGB means,
  Inception features, seven complex-FFT radial bands, 64 native tokens, and five
  population-PCA bands. Early settling is treated as a conditioning candidate,
  not conditioning proof; a paired selected-v27 intervention separately
  shuffles resolved PCA directions across examples and scores the unresolved
  velocity target. The run completed under queued GPU locks. In v27, FFT bands
  settle monotonically from radius 0-2 at `t=.1` through radius 17+ at `.6`;
  PCA bands settle from ranks 1-32 at `.2` through ranks 513-1024 at `.5`, while
  all 64 native tokens settle together at `.3`. Correct leading-PC context
  reduces unresolved-direction MSE versus shuffled context by 9.24% at `.35`
  and 16.80% at `.5`. Soft25 creates token-index timing (correlation `.574`)
  but leaves frequency timing unchanged, delays Inception settling from `.7`
  to `.8`, and was already worse in FID. The natural useful hierarchy is thus
  a distributed leading-mode/coarse-to-residual/detail order, not a token
  prefix. This analysis does not alter the completed v27 selection.
  [Result and figures](reports/2026-08-26_autoencoder_program/generation_trajectory/results.md);
  [exact protocol](reports/2026-08-26_autoencoder_program/generation_trajectory_protocol.json).

- **2026-09-04 — full-depth input registers reach parity; soft SNR is not
  selected:** correct
  the earlier late-register scope by placing 64 learned registers beside the
  patch embeddings before encoder block 1 and carrying them through all eight
  bidirectional blocks. V34 remains parameter-exact with v27 and retains
  deterministic `64x16`, decoder jitter `.05`, slot balance `.002`, and full
  reconstruction. Its single frozen cache receives common-time and softened
  tokenwise-time priors. The soft schedule keeps clean magnitudes and uniform
  loss unchanged and maps the CIFAR crossing anchors toward `.5` with
  `sigmoid(.25*logit(anchor))`, giving `.404/.470/.507/.533/.566/.606`.
  Architecture and schedule effects have separate exact controls and frozen
  5k-to-10k gates. Thirty-six directly affected and 53 active-path tests, CPU
  and queued prior smokes, and queued batch-512 eager/compiled checks pass at
  27.71/19.17 GiB. One initial tiny
  smoke accidentally inherited CUDA outside the queue for roughly seven
  seconds; it caused no observed failure, was rerun CPU-only, and all subsequent
  GPU work used `gpu-claim`.
  Launched from commit `4eeef6c` at `2026-09-04T17:26:12Z` under supervisor;
  tokenizer W&B run [`rheoy5s1`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/rheoy5s1)
  and both matched priors completed successfully. The tokenizer reached `36.37`
  dB PSNR, `5.77` clean rFID, and effective
  rank `347.84`. At 5k, common time is effectively tied with v27 at
  `26.884/.01777` versus `26.743/.01754` FID/KID; soft25 is weaker at
  `28.296/.01873`, a `+1.412` FID and `+.000958` KID delta from its exact-cache
  control. Both remain inside the frozen two-FID band, so their 10k evaluations
  launched through `gpu-claim` at `2026-09-04T21:00:00Z`.
  At 10k, common v34 reaches **24.274/.01747**, only `-0.260/-0.000174`
  FID/KID versus v27's **24.534/.01765**. This is a credible tie, not the
  predeclared two-FID architecture gain, and comes with greater encoder
  attention compute, lower PSNR (`36.37` versus `37.87`), and 7.84% lower
  effective rank. Soft25 reaches **25.599/.01796**, concordantly worse than
  common v34 by `+1.324/+0.000488`. Retain v27 residual Perceiver pooling and
  common time; do not replicate either v34 arm. Preserve both mechanisms as
  tested controls, then proceed to repository/artifact cleanup.
  [Exact protocol](reports/2026-08-26_autoencoder_program/input_register_soft_snr_screen.json).
- **2026-09-04 — grouped Gaussian/DoG hierarchy succeeds mechanically but is
  rejected for generation:** both six-group objectives produce a clear
  coarse-to-fine decoder. Cumulative target MSE falls 94.6% to `0.0026397`;
  innovation MSE falls 97.2% to `0.00142382` while increment cosine rises from
  `0.29543` to `0.91528`. Both remain usable codecs (PSNR `34.44/35.04`, clean
  rFID `6.76/6.65`), but matched 5k generation is decisively worse:
  cumulative **33.820/0.02227** and innovation **31.391/0.02299** FID/KID versus
  v27 **26.743/0.01754**. Both exceed the frozen two-FID stop boundary with
  worse KID, so no 10k or block-causal follow-up runs. Effective rank contracts
  from `377.43` to `238.14/227.90`, consistent with an ordering/modelability
  tradeoff. Retain flat v27 for generation; treat explicit hierarchy as a
  variable-rate branch or seek a separate hierarchical readout. W&B tokenizer/
  prior runs: `kc8ug18o`/`cp50vzal` and `obu5zpi3`/`ovht4gi9`.
  [Exact protocol](reports/2026-08-26_autoencoder_program/grouped_hierarchy_screen.json).
- **2026-09-04 — learned latent-factorization screen complete:** native
  `32x32` is decisively destructive at 44.324/0.03289 FID/KID-5k. Native
  `128x8` is healthy and retains the control effective rank, but reaches
  27.891/0.01742 at 5k and 25.484/0.01741 at 10k versus `64x16` at
  26.743/0.01754 and 24.534/0.01765. It costs roughly twice the prior compute
  and misses the concordant replication gate. Retain `64x16`; end-to-end native
  training does not erase the literal factorization preference.
  [Exact screen](reports/2026-08-26_autoencoder_program/learned_latent_shape_screen.json).

- **2026-09-03 — decoder-objective screen predeclared:** on the promoted v27
  tokenizer-seed-2 residual+jitter+slot design, isolate weak radial log-power
  matching from frozen LPIPS-Alex feature matching. A signal-relative `1e-3`
  floor makes the radial term finite near the spectral noise floor; raw FFT MSE
  is excluded as a duplicate of pixel MSE. Frozen-checkpoint gradient
  calibration sets weights `6e-5` and `0.02`, giving approximately 10% of the
  pixel-loss output gradient with only 1.34%/0.75% scalar contributions. Both
  healthy codecs receive matched priors; generation selects. Only a >=2-FID
  10k gain with lower KID earns seed replication. Twenty focused tests and four
  end-to-end smokes pass.
  Both supervisor chains acquired shared GPU locks at 19:19 UTC and began with
  finite objective values. Tokenizer W&B runs are radial
  [`xlu5z1n6`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/xlu5z1n6)
  and perceptual
  [`5471b4xt`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/5471b4xt).
  [Exact protocol](reports/2026-08-26_autoencoder_program/decoder_objective_screen.json).
- **2026-09-03 — weak representation-regularizer screen predeclared:** on the
  weak seed-1 residual+jitter checkpoint, compare marginal kurtosis-to-Gaussian
  regularization at weight `1e-4` with scale-invariant sample-varying slot-power
  balancing at weight `0.002`. Frozen-cache calibration makes their expected
  mature contributions 7-10% of reconstruction loss. They are separate arms,
  add no parameters, and retain the full matched-prior generation protocol;
  reconstruction is only a health veto. A seed-1 10k gain of at least two FID
  with KID agreeing is required before seeds 2/3. Both supervisor chains
  acquired shared GPU locks at 04:30 UTC and emitted finite, control-like
  reconstruction trajectories. Tokenizer W&B runs are marginal
  [`gmudbr8y`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/gmudbr8y)
  and slot balance
  [`h43ihnea`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/h43ihnea).
  Both complete chains are healthy. At 5k, marginal reaches
  **27.199/0.01754** FID/KID and slot balance reaches **27.232/0.01915**,
  improving on v23 by 2.122 and 2.090 FID respectively with KID agreeing.
  Marginal kurtosis falls 11.1x; slot-power dispersion falls 90.8x and yields
  an almost flat 0.937-0.965 slot-RMS range, so both named mechanisms are
  verified. Both advance to the predeclared 10k test. Seed-1 evidence remains
  insufficient for global promotion.
  [Exact screen](reports/2026-08-26_autoencoder_program/representation_regularizer_screen.json).
  The 10k check separates the arms: slot balancing reaches
  **24.166/0.01817**, a 2.538-FID and 0.00104-KID improvement over v23, and
  earns tokenizer-seed-2/3 replication. Marginal reaches
  **24.798/0.01756**, but its 1.907-FID gain misses the fixed threshold by
  0.093 and stops. Replication is paired to residual+jitter v20/v24 with prior
  seed 1; promotion requires two of three FID wins, improved mean FID and KID,
  and no >2-FID concordant regression.
  Both replication chains acquired shared lifetime locks at 14:28 UTC and
  began finite training near 4.3k images/s. Tokenizer W&B runs are seed 2
  [`wi7qqx8s`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/wi7qqx8s)
  and seed 3
  [`v2avcbeh`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/v2avcbeh).
  Both chains complete through 5k. Seed 2 reaches **26.743/0.01754** versus
  27.743/0.01603: FID improves by 1.000 but KID disagrees. Seed 3 reaches
  **25.171/0.01665** versus 27.755/0.01830, improving by 2.584 FID and
  0.00165 KID. Both advance to 10k under the frozen continuation rule. Codec
  health passes and the slot-balance mechanism repeats strongly; these remain
  diagnostics rather than selection evidence.
  Final 10k FID/KID for slot balance is **24.166/0.01817**,
  **24.534/0.01765**, and **22.548/0.01617** at tokenizer seeds 1/2/3. Paired
  FID improves by 2.538/0.819/2.497; KID improves at seeds 1 and 3 but
  regresses at seed 2. Mean FID improves 25.701 -> **23.749**, and mean KID
  improves 0.017623 -> **0.017328**. The frozen gate passes: three FID wins,
  better mean FID/KID, and no >2-FID regression. Promote slot-power balancing
  at weight `0.002` into the leading residual-pool + sigma-0.05 decoder-jitter
  tokenizer, with the seed-2 KID interaction retained as a caveat. The most
  efficient robustness follow-up is a prior-seed-2 run on the mixed seed-2
  slot cache against the existing v20 prior-seed-2 control.
  That follow-up is now predeclared: train only the v27 seed-2 prior with seed
  2 and compare to the existing v20 prior-2 control at 25.738/0.01384 (5k) and
  22.861/0.01315 (10k). The standard continuation rule applies. The 10k
  outcome is classified as full confirmation, FID-only confirmation, prior
  sensitivity, or revocation using the exact thresholds in the plan.
  The supervisor-owned chain acquired a shared lifetime GPU lock at 16:18 UTC
  and began finite training near 21.5 steps/s. Prior W&B run:
  [`ptfm3rzt`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/ptfm3rzt).
  At 5k it reaches **25.708/0.01648** versus 25.738/0.01384: FID is tied
  (-0.030) and KID is worse (+0.00263). The frozen metric-disagreement rule
  advances it to 10k, but this checkpoint is not a robustness confirmation.
  At 10k it reaches **23.375/0.01640** versus **22.861/0.01315**, losing
  0.514 FID and 0.00325 KID. This is the frozen prior-sensitive outcome and is
  below the >=2-FID revocation boundary. Slot balancing remains the leading
  expected-value modifier, but its tokenizer/prior-seed interaction is real and
  the result must not be described as universal.
  [Full specification](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-03 — register formation x decoder jitter factorial complete:**
  three new seed-matched tokenizers combine v13's bidirectional
  patch/register formation with deterministic sigma-0.05 decoder-only jitter.
  Existing v13 hard-VAE and v23/v20/v24 residual-jitter runs supply both main-
  effect controls. Every healthy seed receives the unchanged prior-seed-1 60k
  flow and FID/KID-5k; qualifying paired comparisons advance to 10k when improved,
  within two FID, or metric-discordant. Promotion requires a three-seed mean or
  stability frontier improvement, not better reconstruction. All three chains
  acquired shared GPU locks at 01:34 UTC and entered finite training. Tokenizer
  W&B runs are seed 1
  [`0zoq4bhb`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/0zoq4bhb),
  seed 2
  [`17pc720a`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/17pc720a),
  and seed 3
  [`e3d4l17z`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/e3d4l17z).
  All 5k screens completed. Register-minus-residual jitter FID deltas are
  -2.585/+3.120/+1.386 at seeds 1/2/3; KID agrees in direction at every seed.
  The combined arm has worse mean, variance, worst-case FID, and mean KID, so
  it cannot clear either global promotion route. Seed 2 stops; seeds 1 and 3
  qualify for 10k, and seed 1 also triggers the missing residual-jitter 10k
  control.
  The larger evaluation confirms both local directions: seed 1 register+jitter
  reaches **23.972/0.01774** versus residual+jitter **26.705/0.01921**, while
  seed 3 reaches 26.579/0.02052 versus **25.045/0.01798**. Register formation
  therefore remains architecture-by-seed dependent and does not improve the
  aggregate jitter frontier. Retain residual+jitter as the expected-value lead
  and hard-VAE v13 as the stability control.
  [Full specification](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-03 — decoder-jitter robustness confirmation complete:** the
  v20 sigma-0.05 result is being tested at fresh tokenizer seeds 1 and 3 with
  matched prior seed 1, and the frozen seed-2 v12/v20 caches are being compared
  with fresh prior seed 2. All four chains use the shared lifetime GPU queue,
  resumable checkpoints, and supervisor ownership. Generation FID/KID selects;
  reconstruction remains only a permissive health veto. The 5k screen advances
  promising, close, or metric-discordant pairs to 10k. Global promotion
  requires at least two of three tokenizer-seed FID wins, improved mean FID and
  KID, no clear concordant per-seed regression, and preservation of direction
  under prior seed 2. The four chains acquired GPUs 3--6 at 18:16 UTC. W&B
  runs are tokenizer seed 1
  [`ahrvj0vr`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/ahrvj0vr),
  tokenizer seed 3
  [`zudow065`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/zudow065),
  v12 prior seed 2
  [`c3l0pdrh`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/c3l0pdrh),
  and v20 prior seed 2
  [`whl85sj2`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/whl85sj2).
  All 5k screens completed. At tokenizer seeds 1/2/3, jitter-minus-v12 FID is
  +2.186/-4.228/-9.982 and KID agrees in every case; jitter therefore wins two
  of three seeds and improves mean FID by 4.008, but seed 1 narrowly crosses
  the declared clear-regression boundary. Under prior seed 2 at tokenizer seed
  2, jitter repeats strongly at 25.738/0.01384 versus 32.453/0.02258. Seed 3
  and both prior-seed-2 arms advance to 10k; seed 1 stops at 5k as predeclared.
  At 10k, seed-3 jitter reaches **25.045/0.01798** versus v12
  **35.750/0.02762**, while the seed-2 prior-seed-2 pair reaches
  **22.861/0.01315** versus **30.133/0.02244**. Thus the gain repeats across a
  second tokenizer seed and a second prior seed, but it is not universal due
  to seed 1. Retain v12 as the conservative control and carry deterministic
  sigma-0.05 decoder jitter as the leading expected-value experimental design.
  [Full specification](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-02 — Phase C posterior/noise study complete:** the selected v12
  seed-2 residual baseline is held fixed while four arms test clean
  deterministic latents, deterministic decoder-input jitter at sigma 0.05 and
  0.10, and the same-KL soft-floor posterior. Jitter levels are predeclared
  from v12's measured decoder-sensitivity curve rather than selected from
  generation. Training now records posterior log-variance/sigma quantiles,
  per-token sigma, and near-floor mass; a soft arm with at least 95% of values
  within 0.05 logvar of the floor fails the mechanism gate. Thirty-four focused
  tests and deterministic-jitter/soft-posterior end-to-end CPU smokes pass. All
  four supervisor chains entered `gpu-claim --wait` at 07:59 UTC; the node's
  eight GPUs were already held by other projects, so the jobs initially waited
  rather than oversubscribing. All chains later completed. Pure deterministic,
  10% jitter, and soft-VAE arms reach FID/KID-5k
  47.195/0.03633, 34.229/0.02528, and 33.147/0.02381. The 5% jitter arm reaches
  27.743/0.01603 and confirms at 10k with **25.353/0.01568** versus v12's
  **29.588/0.02255**. This is the first clear Phase-C win and now requires an
  independent tokenizer-seed replication. Tokenizer/prior W&B runs are
  [`w2kvua0n`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/w2kvua0n)
  and [`exzxq6zm`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/exzxq6zm).
  [Full specification](reports/2026-08-26_autoencoder_program/plan.md).
- **2026-09-02 — fine-stem and clean tokenwise-SNR follow-ons complete:**
  the tokenizer now separates encoder and decoder patch sizes. Two v12 seed-2
  arms test direct 256-token `2x2` encoding and a local `2x2 -> 8x8` convolutional
  stem while keeping the `4x4` decoder. The prior now supports smooth rational
  per-token time paths, tokenwise AdaLN conditioning, base-displacement targets,
  and `Delta phi_i` sampling. Phase B uses an exactly invertible content-RMS
  token ordering and three parameter-exact arms: common time, CIFAR-radial
  groupwise time, and the same time warp with declared radial-variance loss
  weights. Thirty-two focused tests and end-to-end queued GPU smokes pass. All
  five long runs started through `gpu-claim` at 02:16 UTC and have emitted
  finite optimization records: direct-fine tokenizer
  [`s1bvu09a`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/s1bvu09a),
  local-fine tokenizer
  [`m3lnfesl`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/m3lnfesl),
  common-time control
  [`a5dyvamz`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/a5dyvamz),
  rational-time
  [`14o0ldkj`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/14o0ldkj),
  and rational-time plus radial loss allocation
  [`kl6zbjog`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/kl6zbjog).
  All chains completed. At 10k, the local-convolutional stem ties v12:
  FID/KID 29.485/0.02232 versus 29.588/0.02255. Direct `2x2` encoding is clearly
  worse at 5k (38.153/0.02948). On the reordered v12 cache, common time reaches
  30.461/0.02409 at 10k versus the original 29.588/0.02255; at 5k, rational
  tokenwise time worsens its matched control by 4.666 FID and adding radial
  loss allocation worsens it by 14.787. Retain v12 unchanged. Reject direct
  fine patching and the tested image-frequency-derived token schedule/weight
  transfer; treat the local stem as a neutral alternative, not an improvement.
  [Full specification](reports/2026-08-26_autoencoder_program/plan.md).
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
