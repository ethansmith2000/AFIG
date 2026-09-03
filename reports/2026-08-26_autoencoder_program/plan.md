# Autoencoder representation program (2026-08-26)

## Current conclusion

The best learned full-generation point is the unordered `64x16` tokenizer:
moderate dimensional compression removes nuisance variation while retaining an
acceptable reconstruction floor. Increasing width improves reconstruction and
flat flow MSE but not decoded FID; exact reshaping proves that the tokenizer's
native register boundary also matters.

The next program should optimize a three-way objective rather than reconstruction
alone:

1. **Distortion:** clean reconstruction FID/PSNR.
2. **Modelability:** matched-prior decoded FID at a fixed compute budget.
3. **Robustness:** decoder sensitivity to plausible off-manifold latent error.

Effective rank and slot utilization are diagnostics, not selection metrics.

## Figure contract — prefix decoding

- **Question:** how does decoding evolve as the first `k` latent tokens become
  available, and what behavior was purchased by nested-prefix training?
- **Evidence:** the first six fixed preview examples from the v5 progressive and
  v8 unordered checkpoints; columns are reference and `k=1,2,4,8,16,32,64`.
- **Takeaway:** progressive training provides coherent coarse-to-fine decoding;
  the unordered model purchases better full-generation FID at the cost of useful
  partial decoding. This is a product tradeoff, not evidence that prefix ordering
  improves full-length generative modelability.
- **Artifact:** `prefix_decode_comparison.png`, reproducibly built by
  `scripts/build_prefix_decode_comparison.py`.

## E5 — fixed-cache PCA rate control

Start from the completed unordered `64x48` cache and fit PCA on a fixed training
subset. Inverse-project retained coefficients before the unchanged decoder.

1. Oracle-only ranks: `128, 256, 512, 768, 1024, 1536, 2048, 3072`.
2. Record reconstruction FID/PSNR and retained variance for every rank.
3. Select at most two ranks that bracket the `64x16` clean rFID of 6.08.
4. Only those ranks receive matched 60k priors.

This varies effective rate inside one trained representation. It therefore
separates the causal effect of spectral truncation from independent tokenizer
training and tests whether concentrated high-rate codes recover modelability.

Status: completed at `2026-08-26T01:31:45Z` through the shared GPU queue. The
full-rank validation reproduced the existing clean result (rFID 3.040 versus
3.040 previously; PSNR 45.25 versus 45.30), so the truncation curve is valid.

- Launcher: `scripts/run_e5_pca_oracle.sh`.
- Evaluator: `scripts/evaluate_pca_truncation_oracle.py`.
- Output: `pca_oracle_v9_n64d48/metrics.json` and `reconstructions.png`.
- Basis: `tokenizer_runs/v9-unordered-vae-n64d48-s1/pca_basis_25k.pt`.

| retained rank | variance | PSNR | clean rFID |
|---:|---:|---:|---:|
| 128 | 53.42% | 20.53 | 121.93 |
| 256 | 65.90% | 22.52 | 82.09 |
| 512 | 80.68% | 25.36 | 35.91 |
| 768 | 88.98% | 27.60 | 18.78 |
| 1,024 | 93.85% | 29.77 | 11.56 |
| 1,536 | 98.71% | 35.89 | 4.65 |
| 2,048 | 99.99% | 45.24 | 3.04 |
| 3,072 | 100.00% | 45.25 | 3.04 |

The `64x16` nonlinear tokenizer reaches clean rFID 6.08 with 1,024 scalars,
substantially better rate-distortion than top-1,024 PCA on the high-rate code.
The 1,536-PC point is the selected generative control: it preserves a better
oracle floor than `64x16` while removing half the high-rate coordinates. Train
it as 64 tokens by 24 coefficients so token count remains native; inverse PCA
before the unchanged `64x48` decoder.

## Autoencoder exploration sequence

Do not launch a broad architecture/objective grid. Preserve the `64x16` unordered
baseline and change one causal axis per stage.

### Stage A — encoder allocation and latent formation

- Compare the current one-layer cross-attention pool with deeper latent pooling
  and a convolutional/local stem feeding the same 64 latent queries.
- Hold decoder, latent shape, parameter budget, data, steps, and full-only
  objective fixed where possible.
- Measure whether each encoder actually reduces clean rFID without inflating
  effective rank, dead slots, or decoder sensitivity.

### Stage B — posterior/noise parameterization

- Replace the historically collapsed hard-clamped pseudo-VAE with explicit
  deterministic latents plus controlled decoder-input jitter as the clean
  baseline.
- Separately test a soft-floor variational posterior whose variance remains
  trainable. Log the full log-variance distribution and reject arms pinned to a
  boundary.
- Sweep only a small number of noise levels chosen around the observed prior
  error scale; do not infer robustness from clean reconstruction.

### Stage C — representation regularization

- Test controlled spectral concentration or PCA-aligned penalties, but target a
  rate-distortion region rather than blindly minimizing effective rank.
- Treat balanced slot usage and dead-slot penalties separately from spectral
  rank; the `64x32/48` runs show that they are not the same quantity.
- Consider slot dropout only if variable-rate behavior is desired, and report it
  as a product objective because prefix training already showed a full-FID tax.

### Stage D — decoder and perceptual objective

- Change decoder capacity only after encoder-side controls identify a promising
  representation; otherwise encoder and decoder effects are inseparable.
- Compare pixel MSE with a restrained perceptual or frequency-aware term while
  guarding against visually plausible but information-losing reconstructions.
- Always retain clean rFID and latent-noise sensitivity; PSNR alone is not a
  promotion criterion.

## Promotion gates

Every tokenizer gets the same 15k budget and must report clean rFID/PSNR,
decoder sensitivity, effective rank, coordinate spread, slot RMS, and posterior
statistics. Only arms that improve the distortion/robustness frontier receive a
prior screen. Final selection uses decoded FID/KID under the matched prior recipe,
followed by a larger-sample evaluation and another training seed for claimed
improvements.

## Architecture and SNR review (2026-08-26)

### Patch resolution is currently coupled across encoder and decoder

`patch_size=4` gives an `8x8` grid: 64 encoder patch tokens and 64 decoder
output queries. A `2x2` patch is a sensible CIFAR stem candidate, but changing
the existing flag naively would produce 256 tokens on **both** sides. That
quadruples tokenwise work and makes each self-attention matrix 16 times larger,
while simultaneously changing the encoder, decoder, output factorization, and
parameter-to-token compute balance.

The `4x4 -> width` convolution is not itself a scalar information bottleneck:
each 48-scalar RGB patch is linearly lifted to width 512. Its possible weakness
is loss of within-patch interaction granularity. Therefore the controlled test
is to separate `encoder_patch_size` from `decoder_patch_size`: compare a `2x2`
encoder stem against the existing `4x4` stem while retaining 64 latent tokens,
the `8x8` decoder query grid, decoder, objective, and training budget. A local
convolutional stem that converts the `16x16` fine grid to an `8x8` transformer
grid is an additional compute-matched alternative.

### Letting latent registers participate throughout the encoder

Concatenating learned latent registers with image-patch tokens before the
encoder blocks is a valid architecture. At `patch_size=4`, the transformer
would process 128 tokens and the final 64 register states would be projected to
the bottleneck. This gives registers repeated image reads, mutual communication,
and iterative write/refinement instead of the selected model's single terminal
cross-attention read.

There is already a narrower version of this hypothesis in the code:
`pool_type=residual` repeatedly performs register-to-patch cross-attention,
register self-attention, and an FFN. The selected `cross_only` pool performs one
cross-attention and has no register communication. The clean first comparison
is therefore `cross_only` versus a depth-matched residual Perceiver pool. Full
patch/register concatenation is the follow-up if bidirectional patch-register
updates add value beyond iterative register pooling.

A causal mask over registers alone does **not** create successive information:
if every register can read every image patch, every register can still encode
the complete image independently. Causality becomes meaningful only when paired
with an innovation constraint, a residual reconstruction path, explicit band
targets, or a token-specific rate/noise budget.

### Ways to impose order besides random-prefix reconstruction

1. **Nested dropout/prefix reconstruction (current):** establishes functional
   partial decodability but allows the nonlinear decoder to rewrite the whole
   image at every prefix. It has already shown a full-generation tax.
2. **Grouped successive refinement:** group tokens into a small number of
   stages; each group predicts an additive residual after the previous groups.
   This supplies actual innovation semantics and avoids 64 serial stages.
3. **Explicit frequency supervision:** match successive reconstructions to
   cumulative low-pass targets, or match additive decoder increments to a
   lossless low-pass/band-pass pyramid. Keep phase and orientation.
4. **Block-causal register formation:** later groups may use earlier register
   groups while forming innovations. Image-patch memory should remain globally
   bidirectional.
5. **Hierarchical variational/rate constraints:** allocate KL, noise, dropout,
   or dimensional capacity by group so later groups cannot cheaply duplicate
   earlier information.
6. **Monotonic/leave-one-group-out constraints:** penalize regressions in
   prefix distortion and measure the unique contribution of each group. These
   are weaker than additive residual decoding but useful diagnostics.

### Relation to FAR, VAR, NFIG, and the local blur/DoG evidence

- FAR uses cumulative filtered images `x_i = LP_i(x)`, bidirectional modeling
  within a frequency level, and autoregression across levels. In its continuous
  version it predicts the full clean token distribution conditional on `x_i`
  and filters that prediction to obtain the next level. This is closer to
  cumulative low-pass conditioning than to assigning independent DoG bands to
  latent tokens.
- VAR performs residual multi-scale quantization in the encoder feature map:
  quantize a coarse residual, subtract its upsampled decoded contribution, and
  add all scale contributions during reconstruction. This is close to the
  proposed grouped additive latent decoder, but discrete and organized by
  spatial resolution.
- NFIG is the closest explicit frequency construction: FFT masks split the
  encoder feature map into disjoint bands, residual quantization uses increasing
  token-map resolutions, and a block-causal transformer predicts low-to-high
  frequency groups.
- The neighboring `joint_diffusion` project already tested the exact telescoping
  input basis `[x-G1x, G1x-G2x, G2x-G4x, G4x]`. It reached FID 22.57, near its
  RGB baseline, while the redundant cumulative blur input
  `[x, G1x, G2x, G4x]` reached 19.26. This was input conditioning rather than
  tokenizer design, but it warns that an invertible DoG basis is not
  optimization-neutral and supports testing cumulative targets alongside
  additive bands.

### Measured noise-floor crossings

New CPU-only diagnostics use the exact rectified-flow convention
`z_t=(1-t)eps+t z` and exact tensor-wide scalar normalization used by the
matched priors. The artifacts are:

- `scripts/analyze_image_spectral_snr.py` and `image_spectral_snr.json`;
- `scripts/analyze_token_snr_crossings.py` and `token_snr_crossings.json`.

For raw CIFAR images, an orthonormal FFT and frequency-specific 3x3 RGB
cross-spectral covariance make unit isotropic pixel noise remain variance 1 in
every unit color/frequency direction. Radial average-direction crossings are:

| radial band | r1-2 | r3-4 | r5-6 | r7-8 | r9-12 | r13-16 |
|---|---:|---:|---:|---:|---:|---:|
| population `t*` | 0.175 | 0.383 | 0.526 | 0.628 | 0.743 | 0.847 |
| strongest color mode `t*` | 0.112 | 0.270 | 0.398 | 0.500 | 0.630 | 0.766 |
| strongest-mode variance share | 93.7% | 93.9% | 94.2% | 94.8% | 95.5% | 95.8% |

Adjacent radial-energy ordering holds for
`99.984%, 99.796%, 99.514%, 99.972%, 99.994%` of individual images. This
simultaneously quantifies the broad coarse-to-fine SNR schedule, strong RGB
covariance (one dominant luma-like color direction at every band), and stable
per-sample order.

The learned-token result is qualitatively different:

- The progressive v5 representation's per-token population crossings occupy
  only `t*=0.503..0.555` over the 5th-95th percentile after removing each
  token's population mean. The literal `sqrt(mean_dim(z^2))` statistic requested
  in the discussion, including token-specific means after tensor-wide prior
  normalization, gives `t*=0.469..0.537`. The first content-centered token
  crosses at 0.496 and the final 32-token block averages 0.540. Only 3/64 tokens
  are above content SNR 1 at `t=0.5` (30/64 under literal observed RMS), but all
  64 are above it by `t=0.65`. Adjacent token energy is descending only 50.4%
  of the time.
- The unordered v8 representation has 55 active tokens crossing tightly near
  `t=0.469..0.494` under literal observed RMS and nine nearly dead tokens
  crossing near 0.850 (indices `1,8,17,27,43,52,57,58,63`). The dead-slot
  pattern is highly consistent across samples, but not progressive order;
  adjacent descent is 51.1%.
- Within-token effective rank is retained in the artifact so aggregate token
  energy cannot hide a single active feature direction.

Thus the progressive tokenizer's blurry-to-fine prefix decoding is currently
mostly a **decoder/content-allocation property**, not a magnitude-driven
noise-floor schedule resembling image spectra. Per-token SNR is the correct
primary semantic view for a token schedule; flattened PCA remains the
rotation-invariant view of total prior difficulty, and within-token spectra are
the required guardrail.

### Schedule gauge: rescaling versus per-token noise

For token RMS `r_i`, noise standard deviation `sigma_i`, and the RF path above,
the amplitude form is

`SNRamp_i(t) = t r_i / ((1-t) sigma_i)`.

Its square is the conventional power SNR based on variance. Both reach one at
the same time:

`t_i* = sigma_i / (sigma_i + r_i)`.

There are two equivalent ways to impose target crossings after per-token scalar
RMS normalization:

1. Scale token `i` by `a_i`, retain unit noise, and invert `a_i` before decoding.
2. Leave the token unchanged and use noise standard deviation
   `sigma_i = 1/a_i`.

For a desired crossing `t_i*`, the required amplitude ratio is
`a_i = (1-t_i*)/t_i*` under unit noise. A final tensor-wide standard deviation
does preserve all relative token scales, but changes their common absolute
gauge. To retain exact target crossings, normalize the target amplitude vector
to global RMS one before declaring the final schedule, or absorb the common
factor into the base noise scale.

Full `d x d` whitening inside every token is a stronger intervention than
needed: it destroys within-token eigenvalue structure. Start with token-specific
centering and one scalar RMS per token; retain the internal feature covariance.

The completed power-law rescale sweep already applied static per-token scaling,
recomputed the final tensor-wide mean/std, and inverted the scale before decode.
It was negative relative to the same v5 joint baseline (FID 35.85): alpha
0.25/0.50/0.83 produced FID 39.95/40.93/47.18. This instrument welded together
the desired SNR shift and the scale-induced loss allocation. It does **not**
answer whether a schedule shift with separately chosen loss weights helps.

The clean next instrument is either a static scale with per-token loss weight
`w_i=1/a_i^2`, which expresses loss in the original decoded units, or a smooth
per-token log-SNR/time warp with common noise/data endpoints and explicit
tokenwise time conditioning. Avoid clamped time offsets: prior rolling results
already show that tokens freezing at endpoints creates exposure and
degenerate-gradient problems. If the goal is to reproduce image-like coupling
rather than isolate the schedule, retain an additional uncompensated or
timestep-aware weighting arm; do not conflate that with the pure-schedule arm.

### Resulting experimental order

1. Audit additive prefix increments
   `D(z_{<=k})-D(z_{<=k-1})` by radial and oriented spectrum; current prefix
   images alone do not prove additive residual semantics.
2. Run the isolated register-communication control: full-only `64x16`, existing
   `4x4` image grid and decoder, residual Perceiver pool, matched parameter and
   15k budget.
3. Decouple encoder and decoder patch sizes and test a `2x2` encoder stem without
   changing the bottleneck or output grid.
4. Only then introduce 4-8 explicit progressive groups, comparing cumulative
   low-pass supervision against additive band/residual supervision. Promote by
   clean rFID/PSNR and robustness first, then matched-prior FID.

## Prefix-increment audit (completed 2026-08-27)

The one-token image-space increments
`Delta_k = D(z_<=k) - D(z_<=k-1)` were measured for all 64 slots on the first
512 held-out CIFAR-10 images. The audit records absolute increment RMS,
population MSE gain, the fraction of individual examples improved, residual
alignment, radial/oriented FFT power, and the decoder path length. The fixed
examples are dataset-order examples, not selected outcomes.

- Script: `scripts/analyze_prefix_increments.py`.
- Metrics: `prefix_increment_audit/metrics.json`.
- Overview: `prefix_increment_audit/prefix_increment_overview.png`.
- All 64 signed increments: `prefix_increment_audit/prefix_increment_contact_sheet.png`.

| measurement | progressive v5 | unordered v8 |
|---|---:|---:|
| population-positive token steps | 64/64 | 64/64 |
| mean fraction of examples improved | 98.98% | 97.38% |
| tokens for 50% / 90% total error reduction | 31 / 57 | 30 / 55 |
| centroid-index Spearman | 0.070 | 0.378 |
| adjacent centroids ascending | 49.2% | 44.4% |
| decoder path/direct-displacement ratio | 8.01x | 6.75x |
| energy-weighted centroid, token quartiles | 4.10, 3.17, 3.74, 3.88 | 2.41, 2.28, 2.42, 2.47 |

The unordered unweighted centroid correlation is driven by very small/dead-slot
increments late in the sequence; energy-weighted quartiles are essentially
flat. The progressive representation is also not a clean low-to-high ladder:
its first quartile is spectrally broad, the per-token order is indistinguishable
from alternating at the adjacent level, and later increments revise spatially
localized content. Prefix training therefore supplies **useful successive
decoder refinement**, but not additive frequency semantics. If explicit
spectral roles are pursued, use 4-8 supervised groups and an additive/cumulative
comparison rather than assigning 64 individual frequency bands.

## Running matched-prior controls (launched 2026-08-27)

### E5 rank-1,536 PCA prior

`scripts/build_pca_prior_cache.py` produced an exact-transform prior cache with
shape `64x24`; generated coefficients are inverse-projected to `64x48` before
the unchanged v9 decoder. The cache retains 98.7126% variance and a 512-example
validation gives 35.81 dB PSNR, consistent with the oracle. The matched 60k
prior and 5k evaluation are running via `scripts/run_e5_pca_prior.sh`.

- Cache: `tokenizer_runs/v9-unordered-vae-n64d48-s1/latents_pca_r1536_n64d24_original_flip.pt`.
- Prior: `prior_runs/e5-joint-pca-r1536-n64d24-s1`.
- Evaluation: `prior_evals/e5-joint-pca-r1536-n64d24-060000`.
- Decision gate: relative to unordered `64x16` FID 29.93, `<=27.93` is a clear
  win, `27.93..31.93` requires another seed, and `>=31.93` is a clear loss at
  the current 5k evaluation resolution.

### Compensated alpha-0.50 token-SNR control

The existing power-law token-scale cache is trained with normalized
`w_i = 1/a_i^2`, separating the imposed crossing schedule from the scale's
implicit squared-error allocation. The 64 weights have mean 1 and range
`0.03077..1.96923`. The matched 60k prior and 5k evaluation are running via
`scripts/run_compensated_token_scale_prior.sh 05`.

- Prior: `prior_runs/v11-joint-pow05-compensated-vae-s1`.
- Evaluation: `prior_evals/v11-joint-pow05-compensated-vae-060000`.
- Decision gate: relative to flat progressive FID 35.85, `<=33.85` is a clear
  positive schedule signal, `33.85..37.85` requires another seed, and
  `>=37.85` rejects the isolated alpha-0.50 schedule. The historical
  uncompensated alpha-0.50 result is FID 40.93.

### Stage A parameter-matched register communication

The v8 control has eight patch-encoder blocks followed by one terminal
cross-attention read. A one-block residual pool adds register self-attention and
an FFN as well as the image read; that extra work is exactly one transformer
block. The controlled arm therefore uses seven patch blocks plus one residual
register block. It has exactly the same 60,056,784 parameters as v8 and keeps
the decoder, `4x4` patches, `64x16` bottleneck, full-only objective, variational
settings, seed, batch size, and 15k budget fixed.

- Launcher: `scripts/run_stage_a_residual_pool_control.sh`.
- Output: `tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s1`.
- Hypothesis: moving one encoder block from patch-only computation to iterative
  register communication improves the distortion/robustness frontier.
- Baseline: PSNR 35.88 dB; sensitivity rFID at sigma
  `0/.05/.10/.20/.40 = 6.08/6.30/7.20/11.86/35.54`.
- Promotion gate: train a matched prior only if the arm improves at least one of
  clean, sigma-0.10, or sigma-0.20 rFID by 0.5 without worsening another by more
  than 0.5. A uniformly near-tied arm is not promoted because it does not
  justify the topology change.

## First control results and decisions (2026-08-27)

### Rank-1,536 PCA: rejected in its raw tensor-wide gauge

The completed matched prior reaches FID **170.53** and KID **0.16918**, far
outside the loss gate. Samples are locally textured but globally incoherent;
the failure is also visible in the final flow MSE of 1.192 versus 0.893 for the
unordered `64x16` baseline. The inverse transform was independently validated
before training at 35.81 dB on 512 held-out examples, so this is not an inverse
projection or decoder-shape error.

The retained PCA coordinates create a severe token gauge under the standard
tensor-wide normalization: token RMS ranges from 5.52 to 0.37 (14.9x), and the
first/last token power ratio is 221.8x. Thus the raw result rejects the claim
that concentrating 1,536 leading PCs is automatically easy for the matched
prior. It does not distinguish high scalar rate from the ordered variance
hierarchy; a per-token scalar-standardized PCA cache would be the clean follow-up
if that distinction remains decision-relevant.

- Evaluation: `prior_evals/e5-joint-pca-r1536-n64d24-060000/metrics.json`.
- Verdict: do not repeat this raw gauge or promote a second seed.

### Compensated alpha-0.50 schedule: rejected

The inverse-square compensated arm reaches FID **39.89** and KID **0.03001**.
Compensation recovers only 1.04 FID from the uncompensated alpha-0.50 result
(40.93) and remains 4.04 worse than the flat progressive baseline (35.85). The
predeclared rejection threshold was 37.85.

This says the earlier negative result was not primarily caused by implicit MSE
allocation. Static magnitude scheduling itself is harmful in this instrument;
do not run alpha 0.83. A future schedule test should use an explicit smooth
tokenwise time/noise parameterization rather than another static rescale.

- Evaluation: `prior_evals/v11-joint-pow05-compensated-vae-060000/metrics.json`.
- Verdict: reject the static token-scale family.

### Residual register pool: promoted

The parameter-exact `e7+p1` reallocation improves the entire measured
distortion/robustness curve:

| metric | cross-only v8 | residual pool v12 | change |
|---|---:|---:|---:|
| PSNR | 35.88 | **37.20** | +1.32 dB |
| rFID, sigma 0 | 6.08 | **5.35** | -0.73 |
| rFID, sigma .05 | 6.30 | **5.56** | -0.74 |
| rFID, sigma .10 | 7.20 | **6.49** | -0.71 |
| rFID, sigma .20 | 11.86 | **11.27** | -0.59 |
| rFID, sigma .40 | 35.54 | **34.22** | -1.33 |

Effective rank rises from 241.95 to 314.34 and top-128 variance share falls
from 66.66% to 60.37%. This is acceptable at the promotion stage because rank
is diagnostic and the arm improves both distortion and off-manifold robustness.
The matched prior is running through
`scripts/run_stage_a_residual_pool_prior.sh`; decoded FID determines whether the
better representation is also more modelable.

### Residual register pool: generative gate cleared (2026-08-28)

The matched seed-1 prior improves the 5k evaluation from FID 29.93/KID 0.02045
to **FID 27.13/KID 0.01885**, a 2.79-point FID gain. This clears the
predeclared `<=27.93` strong-win threshold.

A paired 10k evaluation using the same solver, 50 steps, and random seed reduces
sampling uncertainty and preserves the result:

| tokenizer/prior | samples | FID | KID | clipping |
|---|---:|---:|---:|---:|
| cross-only v8 | 10,000 | 27.38 | 0.02040 | 0.405% |
| residual pool v12 | 10,000 | **24.85** | **0.01910** | 0.462% |
| difference | | **-2.53** | **-0.00130** | +0.057 pp |

The architecture therefore improves all three selected criteria at fixed
parameter count: distortion, decoder robustness, and matched-prior modelability.
This supports repeated register reads and register communication, specifically
the parameter reallocation tested here; it does not yet establish that full
patch-register concatenation or a deeper pool is better.

- 5k result: `prior_evals/v12-joint-unordered-vae-residual-e7p1-n64d16-060000/metrics.json`.
- Paired 10k results:
  `prior_evals/v12-joint-unordered-vae-residual-e7p1-n64d16-060000-n10k/metrics.json`
  and `prior_evals/v8-joint-unordered-vae-060000-n10k/metrics.json`.
- Next confirmation: paired seed-2 priors on both frozen tokenizer caches via
  `scripts/run_stage_a_prior_seed2.sh` and `scripts/run_v8_prior_seed2.sh`.
  A repeated advantage larger than 2 FID is the confirmation gate. This tests
  prior stochasticity only; a second paired tokenizer seed remains necessary
  before treating the architecture effect size as seed-robust.

Seed-2 recovery note (2026-08-29): both prior processes were externally stopped
near step 23k before evaluation. Each has a valid optimizer-bearing step-22,500
checkpoint. The two launchers now resume automatically, skip already completed
training/evaluation phases, and are managed as persistent supervisor jobs. They
remain ordinary `gpu-claim --wait` clients and will not contend with the GPUs
currently held by other projects.

Seed-2 final result (2026-08-29): both resumptions reached step 60,000 and the
paired 5k evaluations completed with the same evaluation seed and solver:

| frozen tokenizer / prior seed 2 | samples | FID | KID | clipping |
|---|---:|---:|---:|---:|
| cross-only v8 | 5,000 | 27.21 | 0.01891 | 0.489% |
| residual pool v12 | 5,000 | **25.48** | **0.01634** | 0.482% |
| difference | | **-1.73** | **-0.00257** | -0.007 pp |

The effect replicates directionally across prior-training seeds. Averaging the
two paired 5k comparisons gives FID 28.57 for cross-only and **26.31** for the
residual pool, a mean paired advantage of **2.26 FID**. However, the seed-2 gap
itself is 1.73 rather than the predeclared greater-than-2 confirmation
threshold, so record this as a strong directional replication rather than a
literal clearance of that gate. Together with the paired 10k seed-1 gain of
2.53, it is enough to keep the residual pool promoted. The next useful
uncertainty is tokenizer-training stochasticity: train a second parameter-matched
v8/v12 tokenizer pair, screen distortion and robustness, and only then train
matched priors if the representation-side gain repeats.

- Seed-2 metrics:
  `prior_evals/v12-joint-residual-e7p1-n64d16-prior-s2-060000/metrics.json`
  and
  `prior_evals/v8-joint-unordered-vae-prior-s2-060000/metrics.json`.
- The evaluation artifacts are committed in `8668d96`. After migration to a
  fresh non-volume-backed machine, neither checkpoint is local. W&B retained
  the cross-only seed-2 prior artifact, while the residual upload run crashed
  before registering an artifact; no retraining is required to recover the
  completed verdict.

### Tokenizer-seed-2 representation confirmation (2026-08-31)

This is the next predeclared uncertainty after the two frozen-cache prior
comparisons. Train a fresh parameter-matched tokenizer pair with seed 2, holding
the 15k budget and every non-topology setting fixed:

| arm | patch encoder | register pool | parameters | output |
|---|---:|---|---:|---|
| v8 cross-only | 8 blocks | one terminal cross read | 60,056,784 | `tokenizer_runs/v8-unordered-vae-s2` |
| v12 residual | 7 blocks | one residual read/refinement block | 60,056,784 | `tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s2` |

- Shared recipe: CIFAR-10, full-only objective, `64x16`, variational posterior,
  KL weight `1e-4`, historical hard log-variance clamp, batch 512, LR `1e-4`,
  1k warmup, 15k steps, BF16, and identical seed 2.
- Launcher: `scripts/run_stage_a_tokenizer_seed2_arm.sh {v8|v12}`. Each training,
  cache, and decoder-sensitivity phase obtains a lifetime GPU lock through
  `gpu-claim`; optimizer checkpoints make the jobs resumable.
- Evidence: full-test PSNR and latent statistics from `metrics_final.json`, plus
  reconstruction FID at sigma `0/.05/.10/.20/.40` from
  `decoder_sensitivity.json`. Axis scorecards remain diagnostic rather than
  selection metrics.
- Decision gate: apply the original within-seed Stage-A rule unchanged. Promote
  seed-2 caches to matched priors only if residual pooling improves at least one
  of clean, sigma-0.10, or sigma-0.20 rFID by 0.5 without worsening another by
  more than 0.5. A directionally consistent but sub-threshold screen is
  suggestive, not sufficient for another 120k prior-training steps.

#### Tokenizer-seed-2 result: gate not cleared

Both arms reached step 15,000 and completed the full cache, axis, and decoder
sensitivity phases. Values below are residual minus cross-only, so positive rFID
is worse:

| metric | v8 cross-only | v12 residual | residual change |
|---|---:|---:|---:|
| PSNR | **35.267** | 35.153 | -0.114 dB |
| rFID, sigma 0 | **6.398** | 6.464 | +0.067 |
| rFID, sigma .05 | **6.645** | 6.699 | +0.054 |
| rFID, sigma .10 | **7.472** | 7.585 | +0.113 |
| rFID, sigma .20 | **11.741** | 11.866 | +0.125 |
| rFID, sigma .40 | 32.295 | **32.110** | -0.184 |
| flattened effective rank | 241.18 | 253.65 | +12.47 |

This does not satisfy the 0.5-rFID promotion rule on clean, sigma .10, or sigma
.20; residual pooling is slightly behind on all three. Stop before matched-prior
training, as predeclared. The seed-1 residual checkpoint's large gains and its
two successful prior comparisons remain valid checkpoint-level observations,
but the topology effect is not tokenizer-seed robust: all generative evidence so
far uses that one tokenizer cache. Do not describe residual pooling itself as a
confirmed architecture improvement without another independent tokenizer seed.

- Durable comparison: `tokenizer_seed2_comparison.json` in this report folder.
- Training runs: W&B `7qrfzjq9` (v8) and `pl5hyam0` (v12).
- Checkpoint artifacts:
  `v8-unordered-vae-s2-tokenizer:v0` and
  `v12-unordered-vae-residual-e7p1-n64d16-s2-tokenizer:v0`.

## Seed-3 register-formation screen (predeclared 2026-09-01)

The mixed seed-1/seed-2 residual result leaves encoder allocation unresolved.
Use a third within-seed control to close that question while adding the
previously untested alternative of learned registers participating directly in
the encoder sequence.

| arm | patch-only blocks | register formation | parameters | output |
|---|---:|---|---:|---|
| v8 | 8 | terminal cross-attention | 60,056,784 | `tokenizer_runs/v8-unordered-vae-s3` |
| v12 | 7 | residual Perceiver read + register self-attention + FFN | 60,056,784 | `tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s3` |
| v13 | 7 | one joint patch/register block + matched register adapter | 60,056,784 | `tokenizer_runs/v13-unordered-vae-register-e7j1-n64d16-s3` |

The v13 arm is an alternative to Perceiver pooling. After seven ordinary
2-D-RoPE patch blocks and the existing affine-free patch norm, concatenate the
64 learned register tokens with the 64 patch tokens. One bidirectional
self-attention/FFN block lets both modalities update each other. Its mixed block
does not apply one shared RoPE because patch coordinates are spatial while
register coordinates are a learned scale/identity axis. Retain the register
states and apply a register-only ratio-2 adapter whose parameter count exactly
matches the removed terminal cross-attention. Learned register queries retain
the historical random truncated-normal initialization; structured scale
embeddings are a later isolated control, not bundled into v13.

- Shared recipe: seed 3, CIFAR-10, full-only objective, `64x16`, historical
  hard-clamped variational settings and KL `1e-4`, batch 512, LR `1e-4`, 1k
  warmup, 15k steps, BF16, identical decoder and data order.
- Launcher: `scripts/run_stage_a_tokenizer_seed3_arm.sh {v8|v12|v13}`. Every
  training/cache/sensitivity phase obtains a lifetime lock through `gpu-claim`;
  checkpoint-bearing training is resumable and final weights are backed up to
  W&B.
- Evidence: full-test PSNR/rFID, sigma `0/.05/.10/.20/.40` decoder sensitivity,
  flattened effective rank, slot RMS, coordinate spread, and posterior
  statistics.
- Corrected gate, declared before seed-3 results: reconstruction and sensitivity
  establish only that the codec is healthy. They do not select latent quality.
  Every finite arm with semantically coherent reconstructions in the permissive
  historical envelope receives a matched 60k joint prior. `PSNR < 28 dB`
  together with clean `rFID > 25`, non-finite statistics, or visibly corrupted
  decoding triggers manual failure review; no improvement over v8 is required.
- Decision: compare paired decoded FID/KID using the same prior seed and sample
  seed. Reconstruction, robustness, rank, and utilization explain the result
  but cannot promote or reject a healthy representation. A v13 architecture
  claim additionally requires a second tokenizer seed and larger-sample paired
  evaluation.
- Launch status: all three supervisor-owned launchers acquired lifetime GPU
  claims at `2026-09-01T18:59:32Z` and entered training. W&B runs are
  `38zmd550` (v8), `qn4em7zo` (v12), and `etkl1hma` (v13). The mixed v13 arm
  completed its longer first compile and reached the same approximately 4.2k
  images/s steady-state throughput as the established arms.

### Reconstruction-gate correction and tokenizer-seed-2 recovery

The original Stage-A `0.5` rFID rule was too strong: it treated reconstruction
quality as a proxy for latent modelability. The rate sweep already shows why
that inference is unsafe—reconstruction and generated FID can move in opposite
directions. Reconstruction should veto a broken codec, not rank healthy latent
distributions.

This correction occurs before seed-3 results and therefore does not select an
outcome post hoc. It also changes the earlier tokenizer-seed-2 stop decision.
Both v8 and v12 seed-2 codecs are healthy despite being tied on reconstruction,
so train matched prior-seed-1 joint flows on both frozen caches and evaluate
with the same 5k sample seed. Outputs:

- `prior_runs/v8-joint-unordered-vae-tokenizer-s2-prior-s1`;
- `prior_runs/v12-joint-residual-e7p1-n64d16-tokenizer-s2-prior-s1`;
- corresponding `prior_evals/*-060000` directories.

Launcher: `scripts/run_stage_a_tokenizer_seed2_prior_arm.sh {v8|v12}`. The
primary comparison is paired decoded FID/KID; a difference below roughly two
FID remains unresolved and a promising result receives a paired 10k evaluation.

### Seed-3 tokenizer results: all healthy, all advance to priors

All three arms reached step 15,000 and completed cache, axis, sensitivity, and
checkpoint-backup phases. The results reinforce the gate correction because
reconstruction improves monotonically while flattened effective rank worsens
monotonically:

| metric | v8 cross-only | v12 residual | v13 register tokens |
|---|---:|---:|---:|
| PSNR | 31.763 | 33.835 | **36.395** |
| rFID, sigma 0 | 10.588 | 7.645 | **5.766** |
| rFID, sigma .05 | 10.996 | 7.914 | **6.099** |
| rFID, sigma .10 | 12.255 | 9.022 | **7.293** |
| rFID, sigma .20 | 18.329 | 13.925 | **12.919** |
| rFID, sigma .40 | 44.065 | **36.514** | 40.007 |
| flattened effective rank | **137.96** | 196.06 | 344.02 |

Every arm is finite, semantically reconstructive, and comfortably inside the
codec-health envelope. Train paired prior-seed-1 60k joint flows for all three;
do not infer the ordering from either reconstruction or rank. Launcher:
`scripts/run_stage_a_tokenizer_seed3_prior_arm.sh {v8|v12|v13}`. Outputs use
`prior_runs/{v8,v12,v13}-joint-*-tokenizer-s3-prior-s1` with corresponding
`prior_evals/*-060000` directories and the same evaluation seed 54321.

Launch record: the tokenizer-seed-2 priors acquired GPU claims at
`2026-09-01T19:39:21Z`; W&B runs are `oytltgh0` (v8) and `4um6874y` (v12).
The tokenizer-seed-3 priors acquired claims at `2026-09-01T19:41:00Z`; W&B
runs are `kir5jvch` (v8), `gtoqstp3` (v12), and `hj15kjc0` (v13). All are
supervisor-owned and resumable through optimizer-bearing checkpoints.

### Matched-prior 5k results and paired 10k decision

All five priors reached step 60,000, completed the fixed-seed 5k evaluation,
and uploaded final checkpoint artifacts:

| tokenizer seed | architecture | FID | KID | clipping |
|---:|---|---:|---:|---:|
| 2 | v8 cross-only | **30.311** | **0.02019** | 0.754% |
| 2 | v12 residual | 31.970 | 0.02270 | 0.484% |
| 3 | v8 cross-only | 42.596 | 0.03427 | 0.413% |
| 3 | v12 residual | 37.737 | **0.02704** | 0.823% |
| 3 | v13 register tokens | **36.355** | 0.02777 | 0.379% |

The seed-2 residual arm loses by 1.66 FID and 0.00252 KID, reversing the seed-1
direction but remaining inside the approximately two-FID unresolved region. At
seed 3, v12 and v13 beat v8 by 4.86 and 6.24 FID respectively. The v13-v12 gap
is only 1.38 FID and KID changes sign, so their ordering is unresolved.

Run paired 10k evaluations with the same 50-step Heun solver and sample seed for
seed-2 v8/v12 and seed-3 v12/v13. Launcher:
`scripts/run_stage_a_paired_10k_eval.sh {s2_v8|s2_v12|s3_v12|s3_v13}`.
Treat v8 as clearly rejected only within tokenizer seed 3; do not infer a
general architecture order until the two close comparisons complete.

### Paired 10k results and next confirmation

| tokenizer seed | architecture | FID-10k | KID-10k |
|---:|---|---:|---:|
| 2 | v8 cross-only | **28.073** | **0.02046** |
| 2 | v12 residual | 29.588 | 0.02255 |
| 3 | v12 residual | 35.750 | 0.02762 |
| 3 | v13 register tokens | **33.740** | **0.02743** |

The seed-2 residual loss remains 1.52 FID and 0.00209 KID, consistent with its
5k direction. At seed 3, the register-token arm improves 2.01 FID over residual
and now also improves KID slightly. Thus v13 is the selected seed-3 architecture
checkpoint, while residual pooling remains strongly tokenizer-seed dependent.

Do not yet call v13 architecture-robust. Train the parameter-exact v13 tokenizer
with seed 2, using the unchanged historical recipe, then train its prior-seed-1
joint flow if it passes only the permissive codec-health check. Compare against
the completed seed-2 v8/v12 priors above. End-to-end resumable launcher:
`scripts/run_stage_a_register_seed2_confirmation.sh`.

Launch record: started 2026-09-01 21:10 UTC on GPU 0 through `gpu-claim` and
supervisor program `afig_stage_a_v13_register_s2_confirmation`. The tokenizer
has 60,056,784 parameters as expected and is active at W&B run
[`m49qqabv`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/m49qqabv).

### Parallel robustness block

Available GPUs permit two independent checks without waiting for the seed-2
tokenizer chain:

1. Freeze the existing tokenizer-seed-3 v12/v13 caches and train both matched
   priors with prior seed 2. Evaluate both at 10k with the same seed, 50-step
   Heun solver, and decoder. This measures whether the 2.01-FID v13 advantage is
   stable to prior optimization stochasticity.
2. Train the parameter-exact v13 arm at tokenizer seed 1 and run its matched
   prior-seed-1 10k evaluation. Compare with the durable seed-1 v8/v12 controls
   (27.38/24.85 FID). Together with active tokenizer seed 2 and completed seed 3,
   this gives v13 all three tokenizer seeds used by the controls.

Launchers are resumable and queue-compliant:
`scripts/run_stage_a_seed3_prior_seed2_arm.sh {v12|v13}` and
`scripts/run_stage_a_register_seed1_confirmation.sh`. Reconstruction remains a
permissive health veto; only matched-prior generation determines architecture
selection.

Launch record (2026-09-01 21:44 UTC): the prior-seed-2 v12 and v13 runs are
active at W&B [`6pnsnulc`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/6pnsnulc)
and [`1lxjnhll`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/1lxjnhll).
V13 tokenizer seed 1 is active at
[`hh07ajxg`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/hh07ajxg).
The earlier v13 tokenizer seed 2 reached step 15,000 with PSNR 34.68 and is
currently uploading its artifact before cache construction and diagnostics.
GPU assignments are dynamic lifetime locks; at launch the three new jobs used
GPUs 0, 1, and 4, leaving GPUs 5--7 free for later phases or other projects.

Completion record (2026-09-02): all primary prior-seed-1 evaluations now use
10k samples.

| architecture | tokenizer seed 1 | tokenizer seed 2 | tokenizer seed 3 | mean FID | FID std | worst FID |
|---|---:|---:|---:|---:|---:|---:|
| v8 cross-only | 27.381 | **28.073** | 40.037 | 31.830 | 5.810 | 40.037 |
| v12 residual | **24.851** | 29.588 | 35.750 | **30.063** | 4.462 | 35.750 |
| v13 register tokens | 31.176 | 33.294 | **33.740** | 32.737 | **1.118** | **33.740** |

The per-seed winners are v12, v8, and v13 respectively. Mean KID is
0.02498/0.02309/0.02538, again favoring v12. Thus v12 is the primary baseline
for expected performance: it has the best mean FID/KID and beats v8 in two of
three seeds. V13 is not the mean-performance winner, but its narrow 2.56-FID
range and best worst-case value make it the retained stability control.

The seed-3 prior-seed-2 replication remains important: v13 reaches FID/KID
32.038/0.02452 versus v12 at 35.243/0.02883, repeating the seed-3 v13 win by
3.205 FID. This rules out prior-seed noise as the explanation for that local
result. It does not erase the cross-tokenizer-seed interaction. Exact durable
values and aggregates are in `matched_prior_architecture_comparison.json`.

### Clean tokenwise-SNR follow-on (launched 2026-09-02)

Keep the clean latent cache and tensor-wide normalization unchanged. For group
`i`, use `phi_i(t)=a_i*t/(1-t+a_i*t)` and noised state
`(1-phi_i)eps_i + phi_i*z_i`. Condition each token on its own `phi_i` or
log-SNR, predict the comparable base displacement `z_i-eps_i`, and sample with
`Delta phi_i`. This avoids asking shared pre-LayerNorm token projections and a
shared output head to span artificial orders of magnitude.

Compare common schedule/uniform loss, groupwise warp/uniform loss, and the same
warp with a separately declared normalized importance profile. Begin with 4-8
semantic groups. Static cache magnitude scaling, clamped offsets, and
global-time derivative targets are not controls for this experiment: the first
was already negative, the second recreates rolling exposure, and the third
reintroduces the output-scale problem the clean parameterization is intended to
remove.

### Fine-stem and tokenwise-SNR launch specification (2026-09-02)

The encoder/decoder patch coupling is now removed. With decoder `patch_size=4`
fixed, the two tokenizer-seed-2 v12 residual arms are:

| arm | encoder path | encoder transformer tokens | decoder queries | parameters |
|---|---|---:|---:|---:|
| v14 direct fine | non-overlapping `2x2` lift | 256 | 64 | 60,136,656 |
| v15 local fine | `2x2` lift, depthwise `3x3/2`, pointwise mix | 64 | 64 | 60,306,128 |

The historical v12 seed-2 control has 60,056,784 parameters and FID-10k 29.59.
The +0.13%/+0.42% parameter deltas are small and explicitly reported. Each arm
uses the unchanged 15k codec-health screen and receives a matched 60k prior if
healthy; reconstruction is not the selector.

For Phase B, source
`tokenizer_runs/v12-unordered-vae-residual-e7p1-n64d16-s2/latents_final_original_flip.pt`
is permuted by descending population content RMS. The inverse permutation is
serialized with the cache and applied before the unchanged decoder. This gives
the prior adjacent high-to-low-energy groups while exactly preserving images.
The common-time arm on this same cache controls both permutation and RoPE
adjacency.

Six group sizes are `11/11/11/11/10/10`. Target crossings are the measured
CIFAR radial population values
`0.174565/0.382968/0.526249/0.627482/0.742522/0.847351`. For crossing `t_i*`,
the rational-path scale is `a_i=(1-t_i*)/t_i*`, and
`phi_i(t)=a_i*t/(1-t+a_i*t)`. Every token starts at noise and ends at data.
The network conditions on each `phi_i`, predicts `z_i-eps_i`, and Heun/Euler
sampling advances with `Delta phi_i`.

Three parameter-exact prior-seed-1 arms isolate the factors:

1. reordered cache, common linear time, uniform loss;
2. reordered cache, rational groupwise time, uniform loss;
3. the same rational time plus mean-one group weights proportional to the six
   measured CIFAR radial variances
   `22.35894/2.59591/0.81043/0.35244/0.12024/0.03245`.

This is not static magnitude scaling: clean latent magnitudes and tensor-wide
normalization remain unchanged. Compare decoded FID/KID at 5k; use 10k whenever
a relevant gap is below two FID or FID/KID disagree. Launchers:
`scripts/run_stage_a_fine_stem_arm.sh {direct2|local2}` and
`scripts/run_phase_b_tokenwise_snr_arm.sh {control|warp|warp_weighted}`.

Launch record: all five supervisor chains acquired lifetime `gpu-claim` locks
at 2026-09-02 02:16 UTC. The direct-fine and local-fine tokenizer W&B runs are
[`s1bvu09a`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/s1bvu09a)
and [`m3lnfesl`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/m3lnfesl).
The common-time, rational-time, and rational-time-plus-weighting prior runs are
[`a5dyvamz`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/a5dyvamz),
[`14o0ldkj`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/14o0ldkj),
and [`kl6zbjog`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/kl6zbjog).
The first post-compile health check found all runs writing finite optimization
history; no gate had failed, and GPUs 5--7 remained unclaimed.

### Fine-stem and tokenwise-SNR 5k outcome (2026-09-02)

All chains completed successfully. Against the matched v12 tokenizer-seed-2
control at FID/KID-5k 31.970/0.02270, the direct `2x2` stem reaches
38.153/0.02948 and is rejected. The local-convolutional stem reaches
31.894/0.02279: effectively tied, with opposite negligible FID/KID directions.
Its PSNR/rFID 35.119/6.453 also ties v12 at 35.153/6.464, so reconstruction does
not provide a reason to override the generation result. Per protocol, only the
local arm advances to a paired 10k evaluation.

On the exactly reordered v12 cache, the common-time control reaches
32.622/0.02330. Rational tokenwise time worsens this to 37.287/0.02631; adding
the radial-variance loss allocation worsens it further to 47.408/0.03704. Both
negative gaps exceed two FID and KID agrees, so neither receives a larger
evaluation. The common-time control remains within 0.65 FID of the original
cache and receives a 10k check to quantify whether ordering/RoPE adjacency is
neutral. The result rejects these aggressive image-frequency-derived token
trajectories and especially their direct loss-allocation analogy; it does not
show that every gentler learned tokenwise schedule must fail.

The paired 10k checks are complete. V15 local reaches FID/KID
29.48494/0.022322 versus v12 at 29.58849/0.022549: deltas -0.104/-0.000227,
far below the decision threshold. It is a neutral alternative, not a promoted
design. V16 reordered common-time reaches 30.46105/0.024094: deltas
+0.873/+0.001545 versus the original cache. This is consistent with a modest
ordering/RoPE cost, though the FID gap alone remains below resolution. In
either interpretation, v17 and v18 remain decisive regressions relative to
their exact reordered control. Retain v12 unchanged. Durable exact metrics and
deltas are in `fine_stem_tokenwise_snr_comparison.json`.

### Phase C posterior/noise launch specification (2026-09-02)

Use the selected v12 residual architecture and tokenizer seed 2. The existing
hard-clamped v12 checkpoint remains the exact historical control: KL `1e-4`,
posterior sigma effectively fixed at `exp(-4)=0.0183`, and a live mean-square
penalty. Do not retrain it. Four new tokenizer arms hold every non-posterior
setting fixed:

1. v19 deterministic with clean decoder inputs;
2. v20 deterministic with additive decoder-input jitter sigma 0.05;
3. v21 deterministic with additive decoder-input jitter sigma 0.10;
4. v22 variational with the differentiable `-8` soft floor and KL `1e-4`.

The jitter sigma multiplies the in-graph batch latent RMS, matching the units
of the decoder-sensitivity audit. Noise is present only for the training
reconstruction; clean posterior means are evaluated, cached, and modeled by
the prior. The 0.05/0.10 values bracket the measured v12 region before the
sigma-0.20 sensitivity curve steepens. This isolates jitter from the KL mean
penalty while v22 versus historical v12 isolates the hard versus soft bound.

Deterministic arms have 60,048,576 parameters versus 60,056,784 for both VAE
arms. The -0.014% delta is negligible but explicit. Final variational metrics
include global bounded-logvar and sigma quantiles, sigma RMS, mass within
0.01/0.05/0.10 of the floor, and per-token sigma means. V22 advances only if
less than 95% of posterior values lie within 0.05 logvar of the floor. This is
a mechanism-validity gate, not a reconstruction ranking. Every other healthy
arm advances through cache, axis scorecard, sigma `0/.05/.10/.20/.40`
sensitivity, matched prior-seed-1 training for 60k steps, and paired
FID/KID-5k. Reconstruction only vetoes a broken codec. Launcher:
`scripts/run_phase_c_posterior_arm.sh {det|jitter05|jitter10|softvae}`.

Launch record: the four supervisor-owned chains entered the shared queue at
2026-09-02 07:59 UTC. All eight GPUs were held by other projects at submission,
so each chain remains a `gpu-claim --wait` process until a lifetime lock becomes
available. No project-local reservation or raw CUDA launch is used.

### Phase C 5k outcome (2026-09-02)

All four chains completed. The existing v12 seed-2 control is FID/KID-5k
31.970/0.02270. Pure deterministic v19 is a decisive regression at
47.195/0.03633. Adding decoder jitter is strongly non-monotonic: sigma 0.05
reaches **27.743/0.01603**, while sigma 0.10 reaches 34.229/0.02528. Thus the
5% arm improves 4.228 FID with KID agreeing, whereas doubling the intervention
over-regularizes. The 5% arm advances to the declared paired 10k confirmation;
the other deterministic arms do not.

Soft-floor v22 passes its mechanism gate: only 5.53% of log-variances lie
within 0.05 of the floor, mean sigma is 0.169, median sigma is 0.0408, and p95
sigma is 0.985. Its posterior is strongly slot-adaptive rather than globally
noisy: several low-content slots learn sigma near one while most active slots
remain near-deterministic. Generation is FID/KID 33.147/0.02381, not an
improvement over v12, so it is not promoted despite being a valid variational
mechanism.

The diagnostics clarify the 5% result without selecting it. Relative to pure
deterministic, v20 raises flattened effective rank 174.73 -> 294.27 and greatly
flattens slot-energy variation; it is also much less noise-sensitive. Yet v12
and v21 show that neither rank nor decoder robustness alone predicts the FID
ordering. The causal result is the matched v19/v20 intervention: modest decoder
jitter changes the learned clean representation enough to improve prior
modelability, while too much jitter loses the gain.

The paired 10k result confirms v20: FID/KID **25.35326/0.015676** versus v12
at **29.58849/0.022549**. The deltas, -4.23523 FID and -0.006873 KID, reproduce
the 5k deltas almost exactly. Promote sigma-0.05 decoder jitter from screen
winner to leading candidate, but do not yet replace v12 globally: the prior
architecture study demonstrated large tokenizer-seed interactions, so an
independent tokenizer seed remains mandatory. Exact values are serialized in
`phase_c_posterior_comparison.json`.

### Phase C decoder-jitter confirmation specification (2026-09-02)

Resolve both known stochastic axes before changing the baseline. Four
supervisor-owned, queue-compliant chains are predeclared:

| arm | tokenizer cache | tokenizer seed | prior seed | matched control |
|---|---|---:|---:|---|
| tokenizer_s1 | new v23 deterministic jitter-0.05 | 1 | 1 | durable v12 seed-1 result |
| tokenizer_s3 | new v24 deterministic jitter-0.05 | 3 | 1 | local v12 seed-3 result |
| prior2_v12 | frozen v12 residual | 2 | 2 | paired with prior2_v20 |
| prior2_v20 | frozen v20 jitter-0.05 | 2 | 2 | paired with prior2_v12 |

The new tokenizers exactly retain the v20 architecture, `64x16` shape,
decoder, data pipeline, 15k budget, optimizer, and sigma-0.05 in-graph
latent-RMS-scaled decoder jitter. Their clean caches receive the unchanged 60k
joint prior. The seed-2 replication does not retrain either tokenizer.
Launcher:
`scripts/run_phase_c_jitter_confirmation.sh {tokenizer_s1|tokenizer_s3|prior2_v12|prior2_v20}`.
Qualifying larger-sample evaluations use
`scripts/run_phase_c_jitter_confirmation_10k.sh <same-arm>`.

Every arm first receives FID/KID-5k using seed 54321, 50-step Heun sampling,
and the same decoder protocol. Advance a pair to 10k when jitter improves FID,
lies within two FID of its matched control, or FID and KID disagree. The seed-1
historical controls are FID/KID 27.13/0.01885 at 5k and
24.85115/0.0191017 at 10k; seed-2 controls are 31.97039/0.0227030 and
29.58849/0.0225492; seed-3 controls are 37.73726/0.0270355 and
35.74954/0.0276222. These values were fixed before the new results.

Generation selects. Reconstruction and sensitivity are retained only as
mechanism/health evidence; the permissive broken-codec veto is `PSNR < 28`
together with clean rFID `> 25`. Global promotion requires all of the
following on the paired 10k evidence: jitter wins FID in at least two of three
tokenizer seeds; its three-seed mean FID and mean KID both improve; it has no
greater-than-two-FID loss with concordantly worse KID at any seed; and the
prior-seed-2 seed-2 pair preserves the improvement direction. Failure retains
v12 globally even if the original v20 checkpoint remains excellent.

Launch record: all four supervisor chains acquired lifetime locks at
2026-09-02 18:16 UTC and began with finite losses. W&B runs are
[`ahrvj0vr`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/ahrvj0vr)
(tokenizer seed 1),
[`zudow065`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/zudow065)
(tokenizer seed 3),
[`c3l0pdrh`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/c3l0pdrh)
(v12 prior seed 2), and
[`whl85sj2`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/whl85sj2)
(v20 prior seed 2). Initial assignments were GPUs 3--6; GPU 7 remained free.

### Phase C decoder-jitter confirmation 5k outcome (2026-09-03)

All chains completed cleanly. Both new tokenizers pass the codec-health veto:
seed 1 reaches PSNR/rFID 37.437/5.380 and seed 3 reaches 37.834/5.210. These
numbers do not select the representation.

| tokenizer seed | v12 FID/KID-5k | jitter-0.05 FID/KID-5k | jitter FID delta | decision |
|---:|---:|---:|---:|---|
| 1 | 27.135/0.01885 | 29.321/0.01967 | +2.186 | stop at 5k |
| 2 | 31.970/0.02270 | 27.743/0.01603 | -4.228 | existing 10k positive |
| 3 | 37.737/0.02704 | 27.755/0.01830 | -9.982 | advance to 10k |

Jitter wins two of three tokenizer seeds. Across all three it improves mean
FID 32.281 -> 28.273 and mean KID 0.02286 -> 0.01800, but the seed-1 loss is a
concordant 2.186 FID and therefore narrowly fails the predeclared no-clear-
regression clause. Do not run seed 1 at 10k or declare a global replacement
from this screen.

The independent prior-seed-2 pair at tokenizer seed 2 strongly repeats the
original direction: v20 reaches FID/KID 25.738/0.01384 versus v12 at
32.453/0.02258, deltas -6.715/-0.008735. Advance both members of this pair and
the tokenizer-seed-3 jitter arm to 10k. Exact values and decisions are in
`phase_c_jitter_confirmation.json`.

### Phase C decoder-jitter confirmation 10k outcome

All three qualifying evaluations completed:

| comparison | v12 FID/KID-10k | jitter-0.05 FID/KID-10k | FID delta | KID delta |
|---|---:|---:|---:|---:|
| tokenizer seed 2, prior seed 1 | 29.588/0.02255 | 25.353/0.01568 | -4.235 | -0.00687 |
| tokenizer seed 3, prior seed 1 | 35.750/0.02762 | 25.045/0.01798 | -10.704 | -0.00964 |
| tokenizer seed 2, prior seed 2 | 30.133/0.02244 | **22.861/0.01315** | -7.272 | -0.00929 |

The second tokenizer seed and second prior seed both strengthen rather than
merely preserve the original result. Modest decoder-input noise therefore has
a real representation-modelability effect; the result cannot be explained by
one favorable prior optimization or one favorable tokenizer checkpoint. The
seed-1 5k regression remains equally real, so the effect is not universal.

Honor the declared conservative gate by retaining hard-VAE v12 as the global
control. For expected generation quality, however, deterministic residual
pooling with sigma-0.05 decoder-only jitter is now the leading experimental
design: it wins two of three tokenizer seeds, improves three-seed mean FID/KID
at 5k, and replicates under the second prior seed. Carry both into the next
phase and report tokenizer-seed interaction explicitly. This is stronger than
a checkpoint-local result but weaker than universal architecture dominance.

### Register formation x decoder jitter factorial (predeclared 2026-09-03)

The next experiment crosses the two useful empirical properties rather than
adding a new semantic objective. V13's true register-token encoder has the
lowest architecture-seed FID variance but weak mean performance; deterministic
sigma-0.05 decoder jitter has the strongest mean performance but one local
seed regression. Train their combination at tokenizer seeds 1, 2, and 3.

The new v25 arms use seven ordinary patch blocks followed by the existing
bidirectional patch/register block and parameter-matched register adapter,
`64x16` clean deterministic latents, and decoder-input Gaussian jitter with
sigma 0.05 times the in-graph latent RMS. Everything else remains fixed:
CIFAR-10, `4x4` patches, full-only objective, batch 512, LR `1e-4`, 1k warmup,
15k tokenizer steps, and a prior-seed-1 60k joint flow. Outputs are
`tokenizer_runs/v25-register-e7j1-det-jitter05-n64d16-s{1,2,3}` and matching
`prior_runs/*-prior-s1`.

The factorial controls are already complete:

| architecture | posterior/decoder training | seeds |
|---|---|---|
| v12 residual | hard-floor VAE | 1, 2, 3 |
| v13 register tokens | hard-floor VAE | 1, 2, 3 |
| v23/v20/v24 residual | deterministic + jitter 0.05 | 1, 2, 3 |
| v25 register tokens | deterministic + jitter 0.05 | 1, 2, 3 (new) |

Generation is primary. Reconstruction, sensitivity, effective rank, and slot
statistics explain outcomes but only invoke the established broken-codec veto.
At 5k, compare v25 to the exact-seed residual-jitter controls:
29.321/0.01967, 27.743/0.01603, and 27.755/0.01830 for seeds 1/2/3. A seed
advances to 10k if v25 improves FID, lies within two FID, or FID and KID
disagree. Qualifying seed 1 also triggers a matched 10k evaluation of v23,
which stopped at 5k under the earlier question's gate.

Promote register+jitter as the expected-quality lead only if all seeds advance,
it wins at least two seeds, mean FID improves by at least two with mean KID
agreeing, and no seed regresses by more than two FID. Promote it as the
stability lead if worst-case FID improves by at least two, mean FID cost stays
below one, and mean KID is non-worse. Otherwise retain residual+jitter for
expected value and hard-VAE v13 as the stability control. Launchers:
`scripts/run_register_jitter_factorial.sh {s1|s2|s3}` and
`scripts/run_register_jitter_factorial_10k.sh {s1|s2|s3|residual_s1}`.

Launch record: the three supervisor-owned chains acquired shared lifetime GPU
locks at 2026-09-03 01:34 UTC. After the longer register-token compilation,
all emitted finite losses at approximately 4.2k images/s. Tokenizer W&B runs
are [`0zoq4bhb`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/0zoq4bhb),
[`17pc720a`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/17pc720a),
and [`e3d4l17z`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/e3d4l17z)
for seeds 1, 2, and 3 respectively. Five GPUs remained free at submission.

### Register formation x decoder jitter 5k outcome

All tokenizers, diagnostics, matched priors, and screens completed. The codecs
are healthy: seed-1/2/3 PSNR is 37.549/34.381/37.496 and clean rFID is
5.533/7.255/5.375. As predeclared, these do not rank the representations.

| seed | residual+jitter FID/KID | register+jitter FID/KID | register FID delta | decision |
|---:|---:|---:|---:|---|
| 1 | 29.321/0.01967 | **26.736/0.01802** | -2.585 | advance both to 10k |
| 2 | **27.743/0.01603** | 30.863/0.02061 | +3.120 | stop |
| 3 | **27.755/0.01830** | 29.141/0.02030 | +1.386 | advance register to 10k |

KID agrees with every FID direction. Register+jitter has mean/std/worst FID
28.913/1.692/30.863 versus residual+jitter at 28.273/0.741/29.321, and mean
KID is 0.01964 versus 0.01800. Thus it fails both declared global promotion
routes: seed 2 is a clear concordant regression, and stability also worsens.
Residual+jitter remains the expected-value lead and hard-VAE v13 remains the
stability control.

Complete the local uncertainties without reopening the global gate: run 10k
for register+jitter seeds 1 and 3, plus the residual-jitter seed-1 control that
was absent under the earlier experiment's stopping rule. Exact screen evidence
is serialized in `register_jitter_factorial.json`.

### Register formation x decoder jitter 10k outcome

The qualifying checks preserve rather than overturn the screen. At seed 1,
register+jitter reaches FID/KID **23.97155/0.017738** versus residual+jitter
26.70471/0.019208, a clear -2.733 FID and -0.001470 KID win. At seed 3,
register+jitter reaches 26.57875/0.020521 versus residual+jitter
**25.04512/0.017985**, a +1.534 FID and +0.002536 KID loss. Seed 2 remains
stopped at its decisive +3.120-FID 5k regression.

The factorial conclusion is therefore clean. Decoder jitter transfers to the
register-token architecture—it improves v13 dramatically—but register
formation itself changes the jitter result in opposite directions across
seeds and worsens mean, variance, worst-case FID, and mean KID at 5k. It is not
the expected-quality or stability lead. Retain deterministic residual pooling
with sigma-0.05 decoder jitter for expected value and hard-VAE v13 as the
separate stability control. The seed-1 v25 checkpoint is an excellent local
result, not evidence for universal register-token superiority.

### Weak representation-regularizer screen (predeclared 2026-09-03)

Defer explicit progressive semantics and first test whether a weak latent
distribution objective can repair the selected residual+jitter design's seed-1
regression. Use two independent parameter-free arms on the exact v23 seed-1
recipe:

| arm | penalty | weight | frozen-v23 value | estimated mature contribution |
|---|---|---:|---:|---:|
| v26 marginal | mean squared per-coordinate `(kurtosis - 3)` | `1e-4` | 0.5273 | `5.27e-5` |
| v27 slot | variance of normalized sample-varying slot power | `0.002` | 0.03624 | `7.25e-5` |

The baseline terminal full reconstruction loss is typically about `7e-4`, so
both interventions are intentionally weak at roughly 7-10% of that scale.
The slot term batch-centers every token-coordinate, averages variance over its
16 channels, and penalizes dispersion across the 64 slots after division by
mean slot power. It is invariant to global latent scale and cannot be satisfied
by per-slot constants. The marginal term changes distributional shape without
directly balancing slots. Do not combine them until one works alone.

Every non-regularizer detail remains fixed: residual `e7+p1`, deterministic
`64x16` latents, sigma-0.05 latent-RMS decoder jitter, tokenizer seed 1, full
objective, 15k steps, and matched prior seed 1 for 60k steps. Parameter counts
are identical. The implementation reports both regularizer values separately;
18 focused tests and two queue-compliant end-to-end smokes pass.

Generation selects. Both codecs advance unless they trip the established
broken-codec veto. Compare FID/KID-5k to v23 at 29.32129/0.019673. Improvement,
a gap within two FID, or metric disagreement advances to the existing v23 10k
control at 26.70471/0.019208. Only a 10k improvement of at least two FID with
KID agreeing justifies seed-2/3 replication. A seed-1 result alone cannot
replace residual+jitter globally. Launchers:
`scripts/run_representation_regularizer_arm.sh {marginal|slot}` and
`scripts/run_representation_regularizer_10k.sh {marginal|slot}`.

Launch record: both supervisor-owned chains acquired shared lifetime GPU locks
at 2026-09-03 04:30 UTC. Their first finite records follow the unregularized
reconstruction trajectory. At step 25, marginal Gaussianity contributes about
`4.4e-5`; slot balance contributes `3.8e-6` before any late slot imbalance has
formed. Tokenizer W&B runs are
[`gmudbr8y`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/gmudbr8y)
and [`h43ihnea`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/h43ihnea).

The 5k generation screen is complete. Marginal Gaussianity records
**27.19883/0.0175446** FID/KID, deltas of **-2.12246/-0.0021285** from v23.
Slot balancing records **27.23151/0.0191532**, deltas of
**-2.08978/-0.0005199**. Both therefore advance to 10k. The interventions did
what they were intended to do: the marginal penalty falls from the frozen-v23
0.5273 calibration to 0.04745 at the terminal train batch, and slot-power
dispersion falls from 0.03624 to 0.000399. Slot balancing also compresses the
full-test slot-RMS range from v23's 0.322-0.989 to 0.937-0.965. Reconstruction
remains healthy (38.25/38.24 dB and clean rFID 5.10/5.15) but does not enter the
ranking decision. Exact evidence is in `representation_regularizer_screen.json`.

At 10k, the arms separate. Slot balancing reaches
**24.16647/0.0181675**, improving on v23 by **2.53825 FID and 0.0010405 KID**;
it clears the fixed gate and advances to tokenizer seeds 2 and 3. Marginal
Gaussianity reaches **24.79815/0.0175552**, an improvement of 1.90656 FID and
0.0016528 KID. Despite the favorable KID, it misses the exact 2-FID gate by
0.09344 and stops without replication.

For the slot replication, hold prior seed 1 and every non-tokenizer detail
fixed. Pair seed 2 with residual+jitter v20 (5k 27.74280/0.0160268; 10k
25.35326/0.0156761) and seed 3 with v24 (5k 27.75492/0.0183005; 10k
25.04512/0.0179849). The existing continuation rule decides 10k evaluation.
Promote slot balancing globally only if it wins FID in at least two of three
tokenizer seeds, improves three-seed mean FID and KID, and has no paired
concordant regression greater than two FID.

Both replication chains launched through supervisor and the shared lifetime
GPU queue at 2026-09-03 14:28 UTC. Initial losses are finite and throughput is
approximately 4.3k images/s. Tokenizer W&B runs are seed 2
[`wi7qqx8s`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/wi7qqx8s)
and seed 3
[`v2avcbeh`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/v2avcbeh).

Both replication chains completed through FID/KID-5k. Seed 2 records
**26.74299/0.0175380** against v20's 27.74280/0.0160268: FID improves by
0.99981 while KID worsens by 0.0015113, so metric disagreement advances it to
10k without counting it as a win. Seed 3 records **25.17054/0.0166531** against
v24's 27.75492/0.0183005, improving by 2.58439 FID and 0.0016474 KID. It is a
clear replication and also advances.

Both codecs are healthy at 37.87/37.74 dB and clean rFID 5.31/5.29. More
importantly for intervention fidelity, terminal slot penalties are
0.000352/0.000426 and slot-RMS ranges are only 0.897-0.919 and 0.804-0.825.
These diagnostics verify that the mechanism repeats; the paired 10k generation
results decide global promotion.

### Slot-balance replication final outcome

The 10k results pass the frozen global promotion rule. Slot balancing reaches
FID/KID **24.16647/0.0181675**, **24.53411/0.0176470**, and
**22.54780/0.0161682** at tokenizer seeds 1/2/3. Their paired residual+jitter
controls are 26.70471/0.0192080, 25.35326/0.0156761, and
25.04512/0.0179849. Thus FID improves at every seed by
2.53825/0.81915/2.49733. KID improves at seeds 1 and 3 by
0.0010405/0.0018167 but worsens at seed 2 by 0.0019708.

Across seeds, mean FID moves from 25.70103 to **23.74946**, a 1.95158-point
gain, while mean KID moves from 0.0176230 to **0.0173276**, a smaller but
favorable 0.0002954 change. There are three FID wins, two concordant FID/KID
wins, and no FID regression. Promote the scale-invariant sample-varying
slot-power penalty at weight `0.002` into the leading deterministic residual
pool, `64x16`, sigma-0.05 decoder-jitter tokenizer. Keep the seed-2 KID
regression explicit: the result is a robust FID/mean improvement, not a claim
that every metric improves at every seed.

All matched priors in this tokenizer-seed replication used prior seed 1. The
highest-value robustness follow-up is one prior-seed-2 run on the mixed seed-2
slot-balanced cache, paired against the already completed v20 seed-2/prior-2
control. It requires no new tokenizer and directly tests whether the weakest
pairing survives prior stochasticity.

### Prior-seed-2 confirmation (predeclared 2026-09-03)

Train one new prior, not a tokenizer: use the verified v27 tokenizer-seed-2
slot-balanced cache with prior seed 2. The paired v20 tokenizer-seed-2/prior-2
control is already complete at FID/KID 25.737998/0.0138424 at 5k and
22.860635/0.0131494 at 10k. Architecture, optimizer, 60k budget, sampling seed,
and evaluation recipe remain exact.

Apply the standard 5k continuation rule: advance when candidate FID improves,
is within two points, or FID/KID disagree. At 10k, improving FID and KID is a
full confirmation; improving FID with worse KID is FID-only confirmation;
losing FID by less than two or with metric disagreement establishes prior-seed
sensitivity; losing at least two FID with KID also worse revokes slot balance as
the default. Launchers are `scripts/run_slot_balance_prior_seed2.sh` and
`scripts/run_slot_balance_prior_seed2_10k.sh`.

The supervisor-owned chain acquired a shared lifetime GPU lock at 2026-09-03
16:18 UTC and entered finite training near 21.5 steps/s. Prior W&B run:
[`ptfm3rzt`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/ptfm3rzt).

The 5k evaluation reaches FID/KID **25.70832/0.0164760** against the paired
control's 25.73800/0.0138424. The -0.02968 FID difference is an effective tie,
while KID worsens by 0.0026337. This does not confirm robustness, but the frozen
continuation rule advances the metric-discordant result to 10k.

The 10k result is **23.37484/0.0164034** versus the paired control's
**22.86064/0.0131494**, deltas of +0.51420 FID and +0.0032540 KID. This lands
in the frozen prior-sensitive category: it is a concordant loss, but far below
the two-FID revocation boundary. At tokenizer seed 2 across prior seeds 1 and
2, slot balancing changes mean FID by -0.15247 (effectively neutral) and mean
KID by +0.0026124 (worse).

Retain the prior three-tokenizer-seed promotion as the expected-value decision,
but narrow the claim. Slot balancing is a leading modifier with two strong
tokenizer-seed wins, not a universally robust improvement. No additional seed
sweeps are warranted before testing a genuinely different objective; carry the
unregularized residual+jitter checkpoint as the paired control in future work.

### Decoder-objective screen (predeclared 2026-09-03)

With encoder formation now stable enough to attribute a decoder objective,
compare two isolated additions on the exact v27 tokenizer-seed-2 configuration.
V28 adds per-image/channel radial log-power matching with a target-relative
`1e-3` floor; v29 adds frozen LPIPS-0.1 AlexNet feature distance on native
`[-1,1]` images. Raw complex FFT MSE is deliberately excluded because Parseval
makes it a rescaled duplicate of pixel MSE. Do not combine the objectives.

Calibrate on 128 frozen v27 test reconstructions by gradient with respect to the
decoder output, targeting 10% of the pixel-MSE gradient norm. Raw median
auxiliary/base ratios are 1657.18 for radial and 4.6705 for LPIPS. Rounded
weights `6e-5` and `0.02` yield approximately 9.94%/9.34% weighted gradient
ratios and only 1.34%/0.75% scalar-loss ratios. Exact calibration is in
`decoder_objective_calibration.json`.

Hold residual `e7+p1`, deterministic `64x16` latents, sigma-0.05 decoder
jitter, slot balance `0.002`, tokenizer seed 2, 15k budget, prior seed 1, and
60k prior recipe fixed. The exact v27 paired control is 26.74299/0.0175380 at
5k and 24.53411/0.0176470 at 10k. Reconstruction is only a health veto. Advance
an arm to 10k if FID improves, is within two FID, or FID/KID disagree. Only a
10k gain of at least two FID with KID agreeing earns tokenizer seeds 1 and 3.
Twenty focused tests, both CPU end-to-end smokes, and both full 60M-parameter
batch-512 GPU smokes pass. Launchers:
`scripts/run_decoder_objective_arm.sh {radial|perceptual}` and
`scripts/run_decoder_objective_10k.sh {radial|perceptual}`.

Both supervisor-owned chains acquired shared lifetime GPU locks at 2026-09-03
19:19 UTC and entered finite training. Tokenizer W&B runs are radial
[`xlu5z1n6`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/xlu5z1n6)
and perceptual
[`5471b4xt`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/5471b4xt).

Both chains completed successfully. Full-prefix PSNR is 38.10 dB for radial
and 38.31 dB for perceptual, with clean reconstruction FID 5.09 and 5.21; these
are codec-health checks only. The terminal weighted auxiliary terms remain
small at approximately 1.55% and 1.13% of pixel MSE, confirming that the screen
tested restrained decoder signals rather than replacements for reconstruction.

At 5k generated samples, radial obtains **25.84728/0.0157119**, deltas of
-0.89571 FID and -0.0018262 KID from v27. Perceptual obtains
**27.61312/0.0192295**, deltas of +0.87013 FID and +0.0016915 KID. Radial is a
concordant early improvement. Perceptual is a concordant early loss but remains
inside the predeclared two-FID continuation margin. Accordingly, both—not just
the visually or reconstructively preferred arm—advance to the 10k evaluation.
