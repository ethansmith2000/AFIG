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
  only `t*=0.503..0.555` over the 5th-95th percentile. The first token crosses
  at 0.496 and the final 32-token block averages 0.540. Only 3/64 tokens are
  above SNR 1 at `t=0.5`, but all 64 are above it by `t=0.65`. Adjacent token
  energy is descending only 50.4% of the time.
- The unordered v8 representation has 55 active tokens crossing tightly near
  `t=0.48..0.505` and nine nearly dead tokens crossing near 0.855 (indices
  `1,8,17,27,43,52,57,58,63`). The dead-slot pattern is highly consistent
  across samples, but not progressive order; adjacent descent is 51.1%.
- Within-token effective rank is retained in the artifact so aggregate token
  energy cannot hide a single active feature direction.

Thus the progressive tokenizer's blurry-to-fine prefix decoding is currently
mostly a **decoder/content-allocation property**, not a magnitude-driven
noise-floor schedule resembling image spectra. Per-token SNR is the correct
primary semantic view for a token schedule; flattened PCA remains the
rotation-invariant view of total prior difficulty, and within-token spectra are
the required guardrail.

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
