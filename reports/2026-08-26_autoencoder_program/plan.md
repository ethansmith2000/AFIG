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

### Clean tokenwise-SNR follow-on (specified, not launched)

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
