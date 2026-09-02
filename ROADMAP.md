# AFIG roadmap — 2026-09-02

This is the current decision roadmap. Historical campaigns and corrected
premises remain in `EXPERIMENT_JOURNAL.md`, `REVIEW_BRIEF.md`, and `reports/`.

## Goal

Learn a compact continuous whole-image representation that jointly optimizes:

1. **Distortion:** clean reconstruction FID and PSNR.
2. **Modelability:** decoded FID/KID under one matched joint prior recipe.
3. **Robustness:** decoder sensitivity to realistic off-manifold latent error.

Progressive/variable-rate decoding is a separate product objective. It is not
assumed to improve full-length generation.

## Established decisions

- The reproducible baseline is an unordered/full-only `64x16` tokenizer with a
  terminal cross-attention pool and a bidirectional joint rectified-flow prior.
- Nested-prefix training provides coherent partial decoding but taxes full
  generation: FID 35.85 versus 29.93 for the matched unordered control.
- Joint generation is the retained engine. Rolling and causal AR generation
  lose through exposure to self-generated clean context.
- `64x16` is the best tested rate point. Wider `64x32/48` codes reconstruct
  better but generate worse; exact `32x32/128x8` reshapes prove literal token
  factorization matters independently of scalar count.
- Raw PCA concentration and static token-magnitude schedules are rejected in
  their tested gauges. The loss-compensated alpha-0.50 scale still reached FID
  39.89 versus the flat progressive control's 35.85.
- No latent-formation arm wins every tokenizer seed. Across three matched 10k
  evaluations, residual pooling has the best mean FID/KID and beats cross-only
  in two seeds; register tokens have the lowest variance and best worst-case
  FID but the worst mean. Carry residual as the primary baseline and registers
  as the stability control.
- The historical hard-clamped "VAE" is effectively deterministic plus tiny
  fixed jitter and a mean penalty. A clean posterior/noise study remains open.

Decoded FID decides model selection. FID-5k gaps below roughly two points are
unresolved. Every exclusive GPU phase uses `/workspace/GPU_QUEUEING.md` and
`gpu-claim`; code and predeclared gates land in Git before long runs.

## Phase A — encoder formation (active)

### A1. Seed-3 three-arm register screen

Train three full-only `64x16` tokenizers with seed 3, 15k steps, and exactly
`60,056,784` parameters each:

| arm | patch-only trunk | latent formation | purpose |
|---|---:|---|---|
| v8 | 8 blocks | one terminal cross read | reproducible control |
| v12 | 7 blocks | cross read + register self-attention + FFN | final residual-pool replication |
| v13 | 7 blocks | patches and registers share one bidirectional block, then a matched register adapter | true register-token alternative |

The v13 mixed block uses learned patch positions and learned register identities
without a shared RoPE: patch position is 2-D, while register position is a
learned/scale axis. This changes only the reallocated eighth encoder block.

**Codec-health gate, corrected before results.** Reconstruction does not select
latent quality. It only vetoes a broken codec. Every arm with finite training,
semantically coherent reconstructions, and metrics in a permissive historical
envelope receives a matched prior. `PSNR < 28 dB` together with clean
`rFID > 25`, non-finite statistics, or visibly corrupted decoding triggers a
manual failure review; no sub-threshold advantage over v8 is required.
Clean/noisy rFID, PSNR, effective rank, slot RMS, and posterior statistics are
diagnostics used to interpret the generative result.

**Selection rule.** Latent quality is selected by paired matched-prior decoded
FID/KID. Train a 60k joint prior for every healthy v8/v12/v13 seed-3 cache. Use
the same prior seed and generated-sample seed within the three-way comparison;
require a larger 10k evaluation and another tokenizer seed before an
architecture-level v13 claim.

This correction also reopens the completed tokenizer-seed-2 v8/v12 caches. They
were both healthy but stopped because v12 did not improve reconstruction. A
paired prior-seed-1 comparison now tests whether latent modelability differed
despite the reconstruction tie.

**Tokenizer status:** complete. The seed-3 v8/v12/v13 PSNR values are
31.76/33.84/36.39 and clean rFID values are 10.59/7.65/5.77. All are healthy.
Flattened effective rank is 137.96/196.06/344.02, moving opposite to
reconstruction quality and making the matched-prior comparison especially
diagnostic. All three prior-seed-1 runs and the reopened paired
tokenizer-seed-2 priors completed under the corrected rule.

**Final 10k result:** each architecture wins one tokenizer seed.

| architecture | seed 1 | seed 2 | seed 3 | mean | population std | worst |
|---|---:|---:|---:|---:|---:|---:|
| v8 cross-only | 27.38 | **28.07** | 40.04 | 31.83 | 5.81 | 40.04 |
| v12 residual | **24.85** | 29.59 | 35.75 | **30.06** | 4.46 | 35.75 |
| v13 registers | 31.18 | 33.29 | **33.74** | 32.74 | **1.12** | **33.74** |

Mean KID is likewise best for v12: 0.02309 versus 0.02498 for v8 and 0.02538
for v13. At tokenizer seed 3, v13's advantage over v12 also repeats with prior
seed 2 (32.04 versus 35.24 FID), so that local win is real rather than prior
noise. The overall conclusion is architecture-by-tokenizer-seed interaction,
not a universal ordering. Select v12 residual for expected performance; retain
v13 as the stability control. Durable exact metrics:
`reports/2026-08-26_autoencoder_program/matched_prior_architecture_comparison.json`.

### A2. Fine/local image stem

With v12 residual selected as the primary baseline, decouple encoder and
decoder patch sizes. Compare the current `4x4` encoder with a `2x2` encoder or
local convolutional stem reduced to the same `8x8` transformer grid. Hold the
`64x16` bottleneck, decoder output grid, objective, budget, and residual latent
formation fixed; retain v13 only as a stability control where useful.

**Complete:** tokenizer seed 2 is matched to
the existing v12 seed-2 control.
V14 directly processes 256 non-overlapping `2x2` encoder tokens while retaining
64 `4x4` decoder queries (60,136,656 parameters). V15 uses a `2x2` lift plus a
depthwise-separable local reduction to the historical `8x8` transformer grid
and unchanged decoder (60,306,128 parameters). Both retain the 15k tokenizer,
permissive codec-health, 60k prior-seed-1, and FID/KID-5k protocol. Parameter
deltas from v12 are only +0.13%/+0.42% and are reported rather than hidden.
The direct-fine and local-fine W&B runs are
[`s1bvu09a`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/s1bvu09a)
and [`m3lnfesl`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/m3lnfesl).
Both completed. Direct `2x2` encoding is rejected at FID/KID-5k
38.153/0.02948. The local stem reaches 31.894/0.02279 at 5k and
29.485/0.02232 at 10k versus v12's 31.970/0.02270 and 29.588/0.02255. That is
an effective tie, not evidence for promotion; keep v12 unchanged. Notably, the
local stem's flattened effective rank rises from 253.65 to 280.95 without a
generation improvement, another reason not to use representation diagnostics
as selectors.

## Phase B — clean tokenwise-SNR prior (complete)

The image/latent analogy is retained, but clean token magnitudes will remain in
their learned gauge. Natural-image frequency modes differ structurally under
spatial operators; forcing several orders of magnitude through shared
tokenwise projections and pre-LayerNorm blocks is an avoidable conditioning
burden.

For token or group `i`, use the endpoint-preserving warp

`phi_i(t) = a_i t / (1 - t + a_i t)`

and noised state

`x_i(t) = (1 - phi_i(t)) eps_i + phi_i(t) z_i`.

Condition every token on its own `phi_i(t)` or log-SNR. Predict the comparable
base displacement `u_i = z_i - eps_i`, not the global-time derivative
`phi'_i(t) u_i`; sample with increments `Delta phi_i`. This keeps the shared
input/output projections from handling artificial orders-of-magnitude clean
targets.

Run matched prior arms on one frozen cache from v12 tokenizer seed 2. Token IDs
are exactly permuted by descending content RMS and inverted before decoding;
this makes adjacent groups meaningful without changing the representation.
The six groups contain `11/11/11/11/10/10` tokens. Their declared SNR=1 crossing
times are copied, without fitting to FID, from the six CIFAR radial population
bands: `0.1746/0.3830/0.5262/0.6275/0.7425/0.8474`.

The matched arms are:

1. common schedule `phi_i(t)=t`, uniform loss;
2. groupwise warp, uniform loss (schedule only);
3. the same warp with normalized explicit importance weights proportional to
   the corresponding CIFAR radial variances (schedule + loss allocation).

All three priors are parameter-exact. The common-time arm controls for the
permutation and changed RoPE adjacency; warp versus common isolates schedule,
and weighted versus unweighted warp isolates loss allocation. Flow loss does
not select. Compare decoded FID/KID-5k, and rerun at 10k when a gap is below two
FID or when FID and KID disagree.

The common-time, rational-time, and rational-time-plus-weighting W&B runs are
[`a5dyvamz`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/a5dyvamz),
[`14o0ldkj`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/14o0ldkj),
and [`kl6zbjog`](https://wandb.ai/ethansmith2000/afig-progressive-tokenizer/runs/kl6zbjog).
All three completed. The reordered common-time control reaches FID/KID
32.622/0.02330 at 5k and 30.461/0.02409 at 10k, slightly worse than the original
v12 cache at 29.588/0.02255 at 10k. Rational groupwise time reaches
37.287/0.02631 at 5k, 4.666 FID worse than its exact control. Adding the
frequency-variance loss allocation reaches 47.408/0.03704, 14.787 FID worse.
Reject both tested transfers; the weighting analogy is especially harmful.
Exact metrics are in
`reports/2026-08-26_autoencoder_program/fine_stem_tokenwise_snr_comparison.json`.

This is distinct from the rejected static cache rescale: the clean endpoint and
tensor-wide latent gauge are unchanged. Clamped time offsets are prohibited
because they recreate the rolling exposure pathology.

## Phase C — posterior/noise parameterization (ready to queue)

Keep the selected v12 residual encoder, `64x16` shape, seed 2, decoder, data
order, 15k tokenizer budget, and matched 60k prior fixed. The existing v12 arm
is the historical control: hard-floor VAE, KL `1e-4`, effectively fixed 1.8%
jitter plus a mean penalty. Four new arms isolate the missing mechanisms:

| arm | posterior / decoder training noise | parameters |
|---|---|---:|
| v19 | deterministic, clean decoder input | 60,048,576 |
| v20 | deterministic, additive `sigma=0.05` decoder jitter | 60,048,576 |
| v21 | deterministic, additive `sigma=0.10` decoder jitter | 60,048,576 |
| v22 | differentiable soft-floor VAE, KL `1e-4` | 60,056,784 |

Jitter is measured in tensor-wide latent-RMS units and is used only while
training the decoder/encoder; evaluation and cached prior targets remain clean.
The two levels come from v12's measured sensitivity curve: rFID is
`6.464/6.699/7.585/11.866` at sigma `0/.05/.10/.20`, so 0.05 and 0.10 span the
useful region before degradation accelerates. The deterministic parameter
delta versus v12 is only -8,208 (-0.014%) and is reported explicitly.

For v22, log bounded log-variance quantiles, sigma quantiles and RMS, three
near-floor fractions, and per-token sigma means. A fraction `>=0.95` within
0.05 log-variance of the `-8` floor fails the named soft-posterior mechanism
and stops before prior training. Reconstruction otherwise remains only a
permissive health veto; every healthy valid-mechanism arm receives the matched
prior and FID/KID-5k evaluation. Use 10k only for a sub-two-FID promising gap
or metric disagreement.

## Phase D — explicit progressive semantics (conditional)

Only pursue this phase if partial decoding is itself valuable or Phase B gives
a positive schedule signal. Use 4-8 groups and compare:

- cumulative low-pass targets;
- additive DoG/band/residual targets;
- block-causal register formation paired with an innovation constraint;
- structured scale embeddings versus independent learned slot identities.

Do not return to 64 individually supervised frequency tokens. Sinusoidal/RoPE
nearness can encode a scale coordinate but cannot create scale semantics by
itself.

## Phase E — representation and decoder objectives

After the encoder and posterior stabilize:

- test controlled spectral concentration and slot utilization separately;
- change decoder capacity only after encoder-side attribution is clear;
- compare pixel MSE with restrained perceptual/frequency-aware objectives;
- retain clean rFID, PSNR, decoder sensitivity, and matched-prior decoded FID.

Effective rank, flow MSE, slot balance, and PSNR are diagnostics rather than
standalone promotion criteria.

## Promotion protocol

1. 15k tokenizer codec-health screen; reconstruction is not a latent-quality
   ranking metric.
2. Full-test clean rFID/PSNR and sigma `0/.05/.10/.20/.40` sensitivity.
3. Matched 60k bidirectional joint prior for every healthy planned arm.
4. Paired FID/KID-5k screen, then 10k for a promising difference.
5. Independent tokenizer and prior seeds before an architecture-level claim.
6. Preserve optimizer-bearing resumes, final metrics, and W&B checkpoint
   artifacts; append every decision to `EXPERIMENT_JOURNAL.md`.
