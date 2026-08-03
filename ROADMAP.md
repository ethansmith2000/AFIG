# AFIG research roadmap

Last updated: 2026-08-03.

This is the forward-looking decision document. `DIAGNOSIS.md` is the evidence
log and `HANDOFF_BRIEF.md` is the operational summary. The purpose here is not
to defend the current pipeline; it is to identify the smallest experiments that
tell us what a workable autoregressive frequency generator must look like.

## 1. Current conclusion

The existing AFIG representation stack is not modelable by the tested generator
at the available budget. This is established visually, not by MSE:

- a 115.5M rectified-flow transformer on 4x4 pixel patches produces recognizable
  CIFAR-10 objects by 5k steps and coherent classes by 30k;
- the same 115.5M model on per-orbit-whitened FFT coefficients, without the AE,
  remains texture mush at 30k;
- the same model on FFT coefficients without per-orbit variance scaling, also
  without the AE, remains nearly the same texture mush at 30k.
- the matched local 4x4 DCT control succeeds about as well as pixels, and a
  full-image DCT control also reaches recognizable objects by 30k, though more
  slowly and less coherently.

This substantially rules out the current AE and variance whitening as the sole
cause. The full-DCT result also rules out global support as a fatal obstruction,
although its gap from patch DCT shows that global support makes optimization
harder. It does **not** show that autoregressive Fourier generation is impossible.
The remaining confound includes periodic Fourier geometry, radial/Hermitian token
composition, complex coordinate geometry, and frequency-specific inductive bias.

## 2. Working principles

1. **Images decide gates.** Every arm must save decoded fixed-seed and fresh-seed
   samples. A better loss with universally broken samples is not progress.
2. **Keep joint and AR evidence separate.** DCTdiff is a successful joint
   frequency diffusion design. FAR is the closer precedent for a causal trunk
   plus a conditional continuous-token diffusion decoder.
3. **Change one conceptual layer at a time.** Representation/noise controls come
   before a new AE; a new AE comes before expensive trunk architecture searches.
4. **Preserve the natural hierarchy initially.** Do not per-frequency-whiten a
   representation whose coefficient energy carries perceptual and SNR meaning.
5. **Stop AR runs before memorization.** The existing AR path is strongest around
   7.5k steps and crosses into harmful conditioning around 21k. Evaluate early.
6. **Wavelets are out of the immediate plan.** They remain a fallback, not the
   current direction.

## 3. Phase A: direct AR transfer baseline

### Question

Can the current causal transformer plus single-token diffusion decoder generate
raw Fourier tokens when given the most literature-supported normalization and
noise geometry, without an autoencoder?

### Representation

- Use Cartesian real/imaginary coefficients.
- Order frequency groups low-to-high as in the existing radial/orbit layout.
- Use one robust global coefficient scale, derived from a high percentile of the
  DC distribution in the spirit of DCTdiff's entropy-consistent scaling.
- Do not use per-frequency or per-orbit variance normalization.
- Make centering explicit. The existing `fft_global` control still subtracts
  per-orbit complex means; it is not a pure global affine normalization.
- Pass exact target metadata to both trunk and diffusion head: sequence index,
  radius, angle as sin/cos, `(kx, ky)`, conjugacy/self-conjugacy, component mask,
  and physical coefficient scale.

### Noise-equivalence audit

Before training, verify that the coefficient-space bridge is exactly the Fourier
transform of the pixel-space bridge:

1. sample white Gaussian noise in pixel space;
2. apply the same orthonormal FFT packing used for data;
3. form the interpolation and velocity in those packed coordinates;
4. confirm inverse-transform equality numerically across timesteps.

This avoids ambiguity from Hermitian pairs, self-conjugate components, and the
relative variance of real and imaginary coordinates. If an analytic packed
Gaussian is used later, it must reproduce this reference, including the required
square-root-of-two factors.

### Initial arms

Run these sequentially rather than as a broad sweep:

1. `ar_fft_cartesian_ecs`: exact packed noise, one robust global scale, current
   AR trunk and metadata-conditioned token decoder.
2. `ar_fft_cartesian_ecs_snr`: the same model with a representation-level SNR
   shift. Start from the DCTdiff idea, but treat its published constant as an
   initialization rather than a value guaranteed to transfer from local DCT.

Do not add static frequency loss weights to these first arms. Earlier AR
weighting concentrated gradients onto a few tokens and worsened memorization.

### Phase A result (2026-08-03)

Both initial arms completed short runs configured to end at 2.5k.

| arm | final held-out clean | shuffled | gap | decoded result |
|---|---:|---:|---:|---|
| `ar_fft_cartesian_ecs` | 0.012045 | 0.013190 | 0.001144 | texture mush |
| `ar_fft_cartesian_ecs_snr` (`c=4`) | 0.023469 | 0.026515 | 0.003047 | texture mush |

The implementation uses an isometric Hermitian packing whose 3,072 active real
coordinates preserve pixel-space L2 energy exactly. ECS fitted one pixel mean
(`0.4733601`) and one DC-derived scale (`10.43208`) with no per-frequency
centering or variance scaling. Round-trip, energy, Gaussian-bridge, velocity, and
physical-decode tests pass.

The baseline normalized history RMS was only `0.02513` against unit Gaussian
noise. Multiplying bridge SNR by `c=4` increased the held-out context advantage
substantially: the final shuffle gap was 2.7x larger, and at step 2k it was
`0.003821` versus `0.001602`. Therefore SNR geometry was a real suppressor of AR
conditioning. It was not the complete generative bottleneck: all 16 decoded
samples in both fixed-seed grids remain low-frequency colored texture with no
recognizable CIFAR-10 objects.

**The visual result is not yet a stop decision.** These were 2.5k-total runs, so
their cosine schedules reached zero learning rate at the preview. They are not
equivalent to the 2.5k checkpoint of a 10k or 30k run. The pixel control's first
saved preview is at 5k; it is already object-like there, but there is no matched
2.5k pixel artifact. It is therefore unjustified to conclude from these grids
that the raw FFT arm cannot emerge with ordinary training length.

The promoted c=4 arm then completed a fresh 10k run whose cosine schedule spans
the full trajectory:

| step | held-out clean | shuffled | gap | decoded result |
|---:|---:|---:|---:|---|
| 2,500 | 0.023732 | 0.027525 | 0.003793 | texture, no clear objects |
| 5,000 | 0.021503 | 0.027910 | 0.006406 | texture, no clear objects |
| 7,500 | 0.021457 | 0.028950 | 0.007492 | texture, no clear objects |
| 10,000 | 0.019638 | 0.026835 | 0.007197 | texture, no clear objects |

The corrected trajectory resolves the 2.5k ambiguity. Causal context use grows
strongly and remains positive, but the fixed-seed grids change little after 5k.
A fresh 16-sample seed at 10k has the same failure mode. In contrast, the pixel
control's stored 5k and 10k grids already contain recognizable animals, vehicles,
and scenes. The final held-out spectral panel also reports log-amplitude bias
`-0.961` and phase coherence `0.250`.

Close this Cartesian/ECS/c=4 arm under the planned early-training budget. This
does not establish that global Fourier generation is impossible; it establishes
that correct Cartesian geometry, global scaling, stronger SNR, exact frequency
metadata, and a context-using causal trunk are not sufficient in this design.
Proceed to Phase B rather than extending the same arm automatically to 30k.

### Budget and gates

- Save checkpoints and decoded samples at 2.5k, 5k, 7.5k, and 10k.
- Primary gate: recognizable object structure in unconditional samples.
- Secondary gates: held-out conditional-versus-null advantage, train/test gap,
  prefix graft, and per-position sample inspection.
- Stop an arm if held-out conditioning reverses or samples remain unchanged after
  the early generalization peak. Do not extend automatically to 30k.

### Interpretation

- Coherent samples mean the old failure was largely normalization/noise geometry;
  continue refining the raw AR path before building an AE.
- Continued mush means either global token composition or joint amplitude/phase
  modeling remains the obstacle; proceed to Phase B.

## 4. Phase B: factor the complex distribution

Cartesian L2 has a meaningful implicit geometry:

`|z - z_hat|^2 = r^2 + r_hat^2 - 2 r r_hat cos(phi - phi_hat)`.

It automatically downweights phase error near zero magnitude. Per-frequency
whitening removed much of that useful coupling. Still, an explicit autoregressive
factorization may be easier to model:

`p(r, phi | context) = p(r | context) p(phi | r, context)`.

### Prototype sequence

For each frequency group, emit two tokens:

1. an amplitude token, using `log1p(magnitude)` or log-power in Euclidean space;
2. a phase token conditioned on the sampled amplitude and all earlier groups.

This approximately doubles the sequence length, which is acceptable at the
current scale. The factorization is more important than preserving 53 steps.

### Phase head requirements

- Use unit phasors, a wrapped-normal/von-Mises family, or an explicitly circular
  diffusion. Do not apply ordinary scalar Gaussian diffusion directly to angles.
- Integrate phase modulo `2*pi`.
- Gate phase loss and phase sampling when amplitude is near zero.
- Train the phase head on sampled/noisy amplitudes as well as teacher-forced true
  amplitudes, otherwise amplitude errors create a new exposure-bias boundary.
- Keep phase vectors for all RGB components in a group joint initially; their
  cross-channel and cross-frequency correlations matter.

### Translation/phase gauge

A spatial translation rotates every Fourier phase coherently by a
frequency-dependent angle. If per-group phase prediction remains unstable, add a
small global pose/translation (phase-gauge) variable before the frequency
sequence, then generate residual phases conditioned on it. This is preferable to
asking every shared token head to rediscover the same global group action.

### Gate

Compare the structured amplitude-then-phase arm against the Cartesian Phase A
arm at the same early-step budget. Promote it only for visibly better coherence,
not for a lower phase or amplitude loss in isolation.

## 5. Phase C: diagnostic locality controls

These are diagnostics, not a commitment to abandon the full FFT premise:

1. **Per-patch orthonormal DCT.** This is a within-token orthogonal rotation of
   the successful pixel patches and should be almost equally modelable. Failure
   would point to an implementation error.
2. **Full-image DCT grouped by frequency.** This is real but globally supported.
   Comparing it with local DCT tests global support; comparing it with Cartesian
   FFT tests complex/Hermitian geometry and boundary conditions.

Together with the completed pixel and FFT controls, this forms a compact matrix:

| representation | local support | complex geometry | status |
|---|---:|---:|---|
| pixel patches | yes | no | coherent |
| local block DCT | yes | no | proposed |
| full-image DCT | no | no | proposed |
| full-image FFT | no | yes | mush under two tested normalizations |

Run this phase if Phase A and B remain ambiguous. It should identify what the AE
must change instead of merely showing that another representation works.

## 6. Phase D: structured, compressive autoencoder

The AE remains the likely long-term route. Its job is to create a generatively
useful coordinate system, not simply preserve the existing FFT coordinates.

### Required properties

- Genuine compression: fewer latent degrees of freedom than the 3072 image
  values, rather than the current 3392-dimensional expansion.
- Perceptual lossiness: discard detail that is expensive to model and cheap to
  perception.
- Explicit structure where Phase B supports it: amplitude/structure latents,
  phase/geometry latents, and possibly a global pose/phase-gauge latent.
- A smooth aggregate distribution suitable for the intended AR decoder.
- Frequency and position metadata that remains exact after encoding, or learned
  token identities if the encoder deliberately abandons direct frequency meaning.
- Robustness: neighborhoods around valid latents must decode to coherent images.

### Training objectives

Use a balanced combination of pixel reconstruction, perceptual reconstruction,
complex/spectral consistency, and a meaningful prior regularizer. Reconstruction
PSNR is a constraint, not the optimization target. A 35 dB near-lossless AE that
remains unmodelable is worse for this project than a softer AE whose generated
latents decode coherently.

### AE gates

1. Reconstruction remains recognizably faithful on held-out images.
2. Latent dimensionality is truly compressive.
3. Moderate latent perturbations decode smoothly.
4. A small, fixed-budget generator produces recognizable objects early.

Gate 4 is mandatory before scaling either the AE or generator. It prevents
another long campaign around a reconstruction-good but generation-bad latent.

## 7. Architecture escalation, only after representation gates

If a representation is demonstrably modelable but the shared AR decoder remains
the bottleneck, consider:

- radial-band-specific input/output adapters or a small mixture of experts;
- hypernetwork/FiLM weights conditioned on exact `(kx, ky)` and band identity;
- separate amplitude and phase decoder families;
- a joint phase corrector over a completed ring;
- explicit complex- or translation-equivariant layers.

Do not begin with another general transformer conditioning sweep. The nine prior
conditioning arms produced only a small RoPE gain with no visual repair.

## 8. External anchors

- DCTdiff (joint DCT diffusion): <https://arxiv.org/abs/2412.15032>
- FAR (AR transformer plus continuous-token diffusion loss):
  <https://arxiv.org/abs/2503.05305>
- Time Series Diffusion in the Frequency Domain (mirrored Brownian motion):
  <https://proceedings.mlr.press/v235/crabbe24a.html>
- Fourier Image Transformer (explicit frequency coordinates and circular phase
  loss): <https://arxiv.org/abs/2104.02555>
- Generating Images with Sparse Representations (explicit DCT frequency,
  position, and value tokens): <https://proceedings.mlr.press/v139/nash21a.html>

## 9. Immediate next action

The slim fixed-order AR baseline completed 10k steps. It used one learned
prediction-slot table, QKNorm, fp32 2-D `(ky, kx)` RoPE, no functional metadata,
no direct decoder position condition, and Transformer position FiLM off. Its
final held-out clean/shuffled/gap values were `0.017984 / 0.025342 / 0.007358`.
This is the best clean conditional diagnostic in the raw-FFT series, but all
four decoded checkpoints remain texture mush. The fixed ordering and redundant
metadata paths are therefore closed as primary explanations.

Run the matched local-DCT control now in
`scripts/run_patch_dct_control.sh`: the successful 4x4 pixel patches are changed
only by an orthonormal DCT inside each patch. It preserves 64 tokens x 48 dims,
spatial support, Gaussian noise, the linear flow bridge, L2, model size, and
training schedule.

**Result (2026-08-03): patch DCT succeeds.** Recognizable CIFAR objects appear by
5k and remain coherent through 30k. Its final training loss is `0.3114`, nearly
identical to the pixel control's `0.3116`. An orthonormal local frequency basis
is therefore not intrinsically hostile to this model or flow objective.

**Result (2026-08-03): full-image DCT also succeeds, but later and less cleanly.**
The 5k grid is mostly texture, structure emerges around 10k, and the 30k grid has
recognizable vehicles, animals, and scenes. Final training loss is `0.3505`,
versus `0.3114` for patch DCT and `0.3116` for pixels. Global basis support is
therefore a real difficulty but not a sufficient explanation for universal mush.

This result does **not** by itself isolate complex phase. Full DCT uses contiguous
4x4 tiles of a real frequency plane, while the failed FFT controls use tokens of
eight radial Hermitian orbits. Basis family and token composition still move
together.

The active bridge is `full_hartley`, launched through the shared queue by
`scripts/run_full_hartley_control.sh`. The orthonormal separable Hartley transform
is real, global, periodic, and Fourier-family, but it retains the full-DCT model's
64 contiguous 4x4 frequency-grid tokens.

In parallel, `fft_global_spiral` changes only the grouping of the failed
`fft_global` coefficients. The codec still produces the identical normalized 514
Hermitian orbits, but a square-spiral permutation is applied before packing eight
orbits into each 48-D token and inverted before decode. This reduces mean
within-token frequency distance from `7.43` to `2.84` (`7.43` to `2.14` for a
true 4x4 grid tile) while preserving coefficient values exactly.

- If Hartley succeeds, periodic global Fourier coordinates are modelable and the
  radial token packing becomes the leading defect. Read `fft_global_spiral`
  directly: success validates local complex-orbit grouping; failure means the
  remaining real-versus-complex/Hermitian distinction still matters.
- If Hartley fails while full DCT succeeds, the periodic translation/phase gauge
  is implicated. Then Phase B amplitude-before-phase is the direct intervention.
- In either case, compare decoded images at 5k before trusting the scalar loss.

The 2-D RoPE coordinates remain a useful attention chart, not an assumption that
Fourier dependencies are monotonically local.
