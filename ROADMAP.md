# AFIG research roadmap

Last updated: 2026-08-05.

This is the forward-looking decision document. `DIAGNOSIS.md` is the evidence
log and `HANDOFF_BRIEF.md` is the operational summary. The purpose here is not
to defend the current pipeline; it is to identify the smallest experiments that
tell us what a workable autoregressive frequency generator must look like.
**Section 11 is the post-audit source of truth; sections 3--10 retain the
historical experiment sequence and its inline corrections.**

## 1. Current conclusion

**Quantitative recalibration (5k samples, shared 50k CIFAR reference):** pixel
and patch-DCT score FID `31.67/31.38`; token-axis patch-grid DCT `130.57`; full
DCT `112.82`; full Hartley `156.36`; legacy-self-first compact FFT `164.60`.
Corrected grid-local/scale-homogeneous compact FFT score `171.38/214.04`.
KID is `0.01991/0.02038/0.11172/0.09547/0.14135/0.14690/0.15681/0.20414`
in the same order. This supersedes
binary visual wording that called all full-support C4/raw endpoints “passes.”
Full DCT sometimes produces recognizable fragments, but every raw full-support
arm is quantitatively far below the local tier. Hartley and compact FFT occupy
the same poor tier; their small difference should not carry a mechanism claim.

**Decisive state-versus-interface control:** the successful pixel model was
conjugated by the exact corrected compact FFT with no retraining. Gaussian base
samples, the evolving Heun state, and returned velocities all remain in compact
FFT coordinates; each velocity call alone is inverse-transformed to local pixel
patches and transformed back. Its 64 new samples are clean pixel-tier objects,
with `2.15e-6` maximum base round-trip error. Therefore global FFT coordinates
are a valid Euclidean diffusion/flow **state space**. The failure is their direct
use as the native token/computation interface of this shared transformer at the
tested budget. This is now the load-bearing distinction.

The newest matched controls retract the previous support-first diagnosis.
**Global support is a graded difficulty, not the binary cause of failure.** On
the same C4 aggregate and identical 16 x 16 AR interface, 2x2, 4x4, and full 8x8
block DCT all generate recognizable fixed- and fresh-seed CIFAR structure. A
second matched matrix also passes with full DCT or Hartley and either contiguous
frequency tiles or radial frequency quartets. This is established visually, not
by MSE:

- a 115.5M rectified-flow transformer on 4x4 pixel patches produces recognizable
  CIFAR-10 objects by 5k steps and coherent classes by 30k;
- the same 115.5M model on per-orbit-whitened FFT coefficients, without the AE,
  remains texture mush at 30k;
- the same model on FFT coefficients without per-orbit variance scaling, also
  without the AE, remains nearly the same texture mush at 30k.
- the matched local 4x4 DCT control succeeds about as well as pixels, and a
  full-image DCT control also reaches recognizable objects by 30k, though more
  slowly and less coherently.
- full-image Hartley coefficients in contiguous frequency tiles are also
  recognizable, while both a locality-matched FFT regrouping and an exact
  compact isometric FFT packing remain mush at 30k.
- the matched 64-step autoregressive Hartley-tile model remains object-free
  speckle through 10k even though its held-out shuffle gap grows to `0.117`.
- a genuinely compressive 8x8x8 spatial AE reaches 30.9 dB and is locally robust;
  joint diffusion over its 16 global Hartley tiles is rough texture/pseudo-scenes,
  while the exact same old MSE latent map in 16 local raster tokens generates
  recognizable objects at 10k and under a fresh seed;
- the same large global/local quality gap repeats with the new perceptual C8 and
  12x-compressive perceptual C4 codecs; local raster and local 2x2 DCT both pass;
- the C4 block-DCT support sweep passes at 2x2, 4x4, and 8x8 support. Quality and
  held-out context use weaken with support, but the full-support endpoint remains
  semantic;
- the matched full-support basis/grouping matrix passes for DCT tiles, DCT
  radial quartets, Hartley tiles, and Hartley radial quartets, including fresh
  seed 54321;
- the same local DCT values also pass when reordered subband-first, so low-to-high
  spectral order is compatible with causal generation when target support stays
  local;
- changing the global C4 Hartley factorization to 64 scalar steps, four 64-D
  bands, or one 256-D target produces no visual repair. The matched one-token
  spatial target also fails, identifying the capacity of that endpoint rather
  than a Fourier-specific obstruction.

This retracts the earlier inference that the compressive AE aggregate itself was
not modelable, and also the later claim that global support was the load-bearing
confound. Local support remains easier, but is not required for generation under
the short standardized C4 interface. Real frequency coordinates, Gaussian flow,
and Euclidean MSE are compatible even at full support. The old MSE codec
succeeding locally still shows that perceptual training is useful for rate and
reconstruction tradeoffs, but is not what repaired generation.

The first trunk/decoder representation split has also completed. It retained the
same global low-to-high Hartley targets but fed the trunk local patches of the
zero-filled inverse-transformed prefix. This **did not improve** the raw global
arm: fixed and fresh outputs are visually near-identical (pixel correlations
`0.979` and `0.976`). This arm was not fully matched (batch `32` versus `64`,
`82.9M` versus `106.5M` parameters), and the prefix-graft PSNR is near a
different-image baseline, so it does not close trunk input geometry. The later
support and basis/grouping controls
show that error support is not the sole issue either. A local token error does
stay local, which plausibly explains its quality advantage, but full real
orthogonal spectral coordinates are modelable at the matched endpoint.

Global real Fourier generation is feasible, and exact native-complex generation
is feasible on both C4 and 16-pixel raw endpoints. Phase-preserving radial scaling
does not repair the 32-pixel compact FFT, while the unscaled 16-pixel endpoint
passes under both 16 x 48 and 64 x 12 layouts. The remaining bottleneck therefore
scales with added high-frequency dependency/detail, not complex coordinates,
global support, normalization, or token count alone. Joint/bidirectional ring
generation failed visually on two codecs. A 134-step, 64-D sequential arm and a
134-step, 16-D arm also fail through 10k under fixed and fresh seeds. Their
conditional metrics were measured on training batches, so these are
implementation-level negatives rather than proof that AR horizon, joint-ring
width, or moderate compression cannot help. Both bounded polar-v2 controls
likewise fail visually, but their context gaps were training-batch measurements
and their polar objective only weakly enforced the physical Cartesian metric.
The true-amplitude joint-phase oracle completed with nonsemantic sampled phases;
its negative is exploratory and cannot close spectrum-level factorization under
the present loss.

## Independent-audit reset (2026-08-05)

Two independent reviewers converged on the following changes:

- all historical `clean/shuffled/gap` tables from `train_continuous.py` are
  training-batch, not held-out, including ECS/SNR/slim and every factorized-polar
  or wrapped-normal arm; the diagnostic has now been fixed to use the excluded
  deterministic tail panel;
- the ring/grouped conditional losses are training-batch too;
- the 64-step Hartley AR saw 24x fewer image exposures than the joint Hartley
  control and used a different schedule/solver budget;
- the compact FFT negative is confounded by prepending paired self-conjugate
  units, which puts DC and Nyquist-scale values in token zero (`~634:1` scale
  ratio), while legacy 65-token FFT layouts contain 48 padded target values;
- the polar phase objective averages gated relative phase approximately equally
  across frequencies; its `0.1 x` Cartesian auxiliary is the only physical
  cross-frequency energy term;
- direct global-affine orthonormal losses are comparable and show a graded ladder
  (trailing-20 means: pixel `0.3511`, patch DCT `0.3517`, full DCT `0.3909`,
  Hartley `0.4056`, compact FFT `0.4151`);
- no FID has yet been computed, and single-seed unblinded visual labels are not a
  sufficient protocol for marginal pass/rough distinctions.

Accordingly, the leading hypothesis is graded interface/rate friction—especially
co-tokenization and high-noise commitment—not a theorem that global support or
native complex coordinates are unusable.

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

| arm | final training-batch clean | shuffled | gap | decoded result |
|---|---:|---:|---:|---|
| `ar_fft_cartesian_ecs` | 0.012045 | 0.013190 | 0.001144 | texture mush |
| `ar_fft_cartesian_ecs_snr` (`c=4`) | 0.023469 | 0.026515 | 0.003047 | texture mush |

The implementation uses an isometric Hermitian packing whose 3,072 active real
coordinates preserve pixel-space L2 energy exactly. ECS fitted one pixel mean
(`0.4733601`) and one DC-derived scale (`10.43208`) with no per-frequency
centering or variance scaling. Round-trip, energy, Gaussian-bridge, velocity, and
physical-decode tests pass.

The baseline normalized history RMS was only `0.02513` against unit Gaussian
noise. Multiplying bridge SNR by `c=4` increased the training-batch shuffle gap:
the final gap was 2.7x larger, and at step 2k it was
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

| step | training-batch clean | shuffled | gap | decoded result |
|---:|---:|---:|---:|---|
| 2,500 | 0.023732 | 0.027525 | 0.003793 | texture, no clear objects |
| 5,000 | 0.021503 | 0.027910 | 0.006406 | texture, no clear objects |
| 7,500 | 0.021457 | 0.028950 | 0.007492 | texture, no clear objects |
| 10,000 | 0.019638 | 0.026835 | 0.007197 | texture, no clear objects |

The corrected trajectory resolves the 2.5k visual ambiguity. The
**training-batch** shuffle gap grows, but held-out context use was not measured;
the fixed-seed grids change little after 5k.
A fresh 16-sample seed at 10k has the same failure mode. In contrast, the pixel
control's stored 5k and 10k grids already contain recognizable animals, vehicles,
and scenes. The separately held-out spectral panel also reports log-amplitude bias
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

For each frequency group, use one transformer step and two conditional decoder
substeps:

1. sample three log amplitudes from a Euclidean rectified-flow head;
2. sample three phases from an intrinsic circular head conditioned on the sampled
   amplitudes and all earlier groups;
3. reconstruct the complex RGB coefficient, convert it to the Cartesian history
   representation, and advance the trunk once.

This preserves the 514-step causal sequence. It doubles decoder work, not trunk
sequence length, and implements
`p(r, phi | h) = p(r | h) p(phi | r, h)` explicitly.

The trunk and decoder need not share coordinates. The first prototype keeps the
trunk history Cartesian to avoid changing two statistical interfaces at once;
only the decoder uses log-amplitude/circular-phase coordinates. A sampled
coefficient is the canonical bridge between them.

### Phase head requirements

- Use uniform base phase and an intrinsic tangent velocity on `S1`; integrate the
  scalar angle modulo `2*pi`. Network inputs may use unit phasors, but scalar angle
  differences never receive ordinary unwrapped MSE.
- Reconstructing Cartesian coefficients supplies the exact magnitude-aware error:
  `|z_hat-z|^2 = r_hat^2+r^2-2 r_hat r cos(delta_phi)`. Combine this with normalized
  log-amplitude flow loss rather than inverse-Jacobian weights.
- Gate phase supervision continuously by magnitude; self-conjugate frequencies
  remain real and receive the corresponding sign/phase treatment.
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

The first arm runs for 10k steps and saves at 2.5k, 5k, 7.5k, and 10k. It keeps
QKNorm, fp32 2-D RoPE, the learned absolute target slot, and enables the existing
zero-init per-block position FiLM. Target identity also conditions both decoder
heads directly. These are deliberate defaults for frequency-specific computation,
not a claim that positional conditioning alone repairs the representation.

### Phase B result (2026-08-04)

The 10k `ar_fft_factorized_polar_10k` arm completed. It kept one 514-step
Transformer sequence, Cartesian ECS history, QKNorm, fp32 2-D RoPE, learned
target slots, and position FiLM. Its decoder used a Gaussian Euclidean flow over
coarse-radial-normalized log amplitude followed by a uniform-base shortest-path
circular flow over phase. Half of phase-training examples used the amplitude
head's predicted endpoint, and a 0.1-weight globally normalized Cartesian loss
was the only term carrying the physical cross-frequency hierarchy.

| step | clean | shuffled | gap | phase coherence | physical NRMSE | decoded result |
|---:|---:|---:|---:|---:|---:|---|
| 2,500 | 2.821 | 3.456 | 0.635 | 0.715 | 0.405 | high-frequency speckle |
| 5,000 | 2.498 | 3.839 | 1.341 | 0.792 | 0.385 | bands/regions, no objects |
| 7,500 | 2.422 | 4.084 | 1.662 | 0.818 | 0.381 | same failure |
| 10,000 | 2.364 | 4.129 | 1.765 | 0.824 | 0.378 | same failure |

On a separately held-out spectral panel, log-amplitude MAE falls to `0.278` with
bias `+0.021`; phase coherence is `0.824`. The clean/shuffled table itself is
training-batch, not held-out. The formulation changes coordinate prediction but
still fails the load-bearing visual gate: every free sample is texture/speckle.

The rollout diagnostic sharpens the failure. Thirty-two true low-frequency
coefficients with a zero suffix already decode to blurred recognizable objects;
sampling the remaining suffix destroys them. With true prefixes of
128/256/384, objects increasingly survive, but sampled suffixes consistently add
harmful texture despite mean amplitude ratios of `1.023/1.000/1.028`. Fully
generated cutoffs show coarse blobs at 32, weak structure by 128, and progressively
more incoherent texture through 384. This is not a simple excess-amplitude bug.
It is a cross-frequency phase/history consistency problem amplified across 514
causal steps.

Do not extend this exact arm to 30k. The matched 64-step AR Hartley control below
also fails its early visual gate, so the plan moves to the structured compressive
AE rather than another phase-loss sweep.

### Matched 64-step Hartley AR result (2026-08-04)

`ar_hartley_tiles_10k` uses the already-successful real full-image Hartley
representation, packed as 64 contiguous 4x4 frequency tiles. Its 106.7M-parameter
causal model uses the same target slots, QKNorm, fp32 2-D RoPE, position FiLM,
direct decoder slot condition, and 20-step flow solver as the Phase-B trunk. This
removes native-complex phase geometry and shortens the rollout from 514 to 64.

| step | clean | shuffled | gap | decoded result |
|---:|---:|---:|---:|---|
| 2,500 | 0.471 | 0.540 | 0.069 | mottled speckle, no objects |
| 5,000 | 0.461 | 0.548 | 0.087 | same failure |
| 7,500 | 0.498 | 0.598 | 0.101 | same failure |
| 10,000 | 0.427 | 0.544 | 0.117 | same failure |

The genuinely held-out gap shows that the model uses ordered causal history, but
the fixed-seed grids remain qualitatively unchanged from 2.5k to 10k. Contrast
this with the bidirectional full-Hartley control, which becomes recognizable at
roughly 10k--15k. That contrast is budget-confounded: the AR run saw `0.32M`
images versus `7.68M` for joint, with a different schedule and solver count. The
result closes only this early-budget arm; it does not isolate causality or global
support.

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
| local block DCT | yes | no | coherent |
| full-image DCT | no | no | recognizable, delayed/weaker |
| full-image Hartley | no | no | recognizable jointly; failed in 64-step AR |
| full-image FFT | no | yes | mush under all tested direct variants |

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

### First Phase-D gate

Start with the smallest already-supported bridge before inventing a large new
codec: a deterministic 4x spatial-downsample AE with an 8-channel 8x8 real
latent map. This is 512 latent scalars, a true 6x compression from 3,072 pixels.
Train the decoder with latent noise so neighborhoods around valid codes remain
decodable, save the checkpoint, and measure held-out reconstruction plus latent
perturbation panels. Then train a fixed-budget AR generator over low-to-high
Hartley tiles of that latent map (16 tiles x 32 values for 2x2 tiles). This keeps
the representation frequency-domain and globally supported while giving the AE
permission to discard perceptually expensive detail and reducing the causal
horizon to 16. Only add a heavier perceptual objective or native-complex latent
factorization after this bridge says which part is limiting.

### First Phase-D result (2026-08-04)

The robust deterministic bridge AE passes its reconstruction-side gates:

- 8x8x8 = 512 real scalars, exactly 6x compression;
- held-out pixel MSE `0.000812`, PSNR `30.91` dB;
- adding 10% latent-RMS noise raises MSE only to `0.000950`;
- reconstructions are visually faithful.

It fails the mandatory generator gate in both causal and non-causal forms. The
16-step 106.6M AR Hartley model ends at clean/shuffled/gap
`1.340 / 1.646 / 0.306`; all 2.5k--10k grids are smooth texture. A matched 115.4M
joint flow on the identical 16x32 tokens ends near loss `1.286` and is also
object-free. Therefore rollout is not the sole problem and decoder-local
robustness is not aggregate modelability.

`diagnose_spatial_ae_latents.py` shows why ordinary normalization is not the next
move. The deterministic latent has nearly Gaussian scalar marginals (skew `0.02`,
excess kurtosis `0.32`), coordinate standard deviations near one, and only a
`2.22x` Hartley tile-RMS range. Image validity lives in subtle joint dependencies,
and Gaussian-like codes outside that set decode to texture.

The MSE-only VAE bracket also closes simple KL tuning:

| AE | PSNR | raw KL/dim | offdiag corr RMS | covariance condition | prior/generator gate |
|---|---:|---:|---:|---:|---|
| deterministic + latent noise | 30.91 | -- | 0.321 | 9.12 | AR and joint fail |
| beta=`1e-3` | 30.27 | 1.258 | 0.045 | 2.23 | prior fails; sampled- and mean-posterior joint fail |
| beta=`1e-2` | 25.07 | 0.390 | 0.296 | 2335 | partial posterior collapse; prior fails |
| beta=`1e-2`, 0.5 free bits | 25.29 | 0.508 | 0.114 | 2.76 | collapse repaired; prior still fails |

The beta=`1e-3` sampled-posterior joint flow is somewhat more scene-like than the
deterministic arm, but remains unrecognizable at every gate and ends at loss
`1.461`. A paired posterior audit showed that sampling injects RMS `0.422` noise,
accounts for `18.1%` of aggregate variance, and costs `1.22` dB reconstruction,
so a posterior-mean control was necessary. It also fails: posterior-mean linear
and trigonometric-VP flows end at losses `1.458` and `2.219`, respectively, and
decode into essentially the same broken basin. A deterministic-AE trigonometric
arm changes the texture but still produces no objects. The loss values are not
cross-path comparable; the visual conclusion is. Posterior sampling and the
midpoint variance pinch of independent linear flow are not sufficient causes.

**Retraction after the local-token controls.** "Do not train another generator on
this MSE-only family" was too broad. It was correct only for the tested *global
Hartley tokenization*. The original deterministic MSE C8 codec generates
recognizable objects when the same normalized 8x8x8 latent map is exposed as 16
local raster tokens. The fixed 10k grid and a fresh seed both pass visually.
Therefore Gaussian-like marginals, subtle aggregate dependence, and MSE-only
codec training do not make the latent distribution intrinsically unmodelable.

The perceptual codec campaign is still informative, but for a different reason.
The deterministic C8 codec reaches `30.84` dB and the C4 codec reaches `26.71` dB
at 256 scalars (12x compression) while retaining class and pose. The perceptual
VAE with beta=`1e-3` and 0.5 free bits drives posterior noise nearly to zero and
does not provide a useful prior. Most importantly, the deterministic perceptual
C4 codec repeats the basis split exactly:

| codec / joint tokens | final loss | fixed and fresh samples |
|---|---:|---|
| old MSE C8 / full-map Hartley | `1.286` | rough texture/pseudo-scenes |
| old MSE C8 / local raster | `1.187` | recognizable objects |
| perceptual C8 / full-map Hartley | `1.304` | rough, much weaker than local |
| perceptual C8 / local raster | `1.223` | recognizable objects |
| perceptual C4 / full-map Hartley | `1.306` | rough, much weaker than local |
| perceptual C4 / local raster | `1.220` | recognizable objects |
| perceptual C4 / local 2x2 DCT | `1.208` | recognizable objects |

The local raster and patch-DCT C4 arms use the same seed, model, schedule, latent
statistics, linear flow, token count, and token dimensionality as the failed C4
Hartley arm. Standardized Gaussian bases passed through either decoder remain
texture, so the recognizable samples are learned rather than a decoder prior.

The matched 16-step AR over the C4 local-DCT tokens also passes: at 10k it ends at
train loss `1.146` and held-out clean/shuffled/gap
`1.338 / 1.819 / 0.481`. Both its fixed and fresh seed grids are recognizably
object-like. Local tokenization therefore repairs causal rollout too; the joint
result is not an artifact of bidirectional attention.

The matched wrapped-normal Brownian score process on phase also fails visually at
10k. Its training-batch clean/shuffled/gap is `1.354 / 2.193 / 0.839`; the old
metric does not establish held-out context use. Combined with the weak Cartesian
auxiliary, this does not close phase geometry broadly.

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
- Riemannian Flow Matching (geodesic conditional paths on manifolds):
  <https://arxiv.org/abs/2302.03660>
- Torsional Diffusion (wrapped-normal Brownian diffusion and score matching on a
  hypertorus): <https://arxiv.org/abs/2206.01729>
- FlowMM (Riemannian flow matching on periodic torus coordinates):
  <https://arxiv.org/abs/2406.04713>

## 9. Immediate next action

The slim fixed-order AR baseline completed 10k steps. It used one learned
prediction-slot table, QKNorm, fp32 2-D `(ky, kx)` RoPE, no functional metadata,
no direct decoder position condition, and Transformer position FiLM off. Its
final training-batch clean/shuffled/gap values were
`0.017984 / 0.025342 / 0.007358`. This is the
best old condition diagnostic in the raw-FFT series, but all
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

**Result: full Hartley succeeds at roughly the full-DCT tier.** It develops
object/scene structure by 10k--15k and retains recognizable structure at 30k,
though it remains substantially weaker than patch DCT and pixels. Its final loss
is `0.3647`. A real, global, periodic Fourier-family basis is therefore modelable
in contiguous 4x4 frequency-grid tokens.

`fft_global_spiral` changed only the grouping of the failed
`fft_global` coefficients. The codec still produces the identical normalized 514
Hermitian orbits, but a square-spiral permutation is applied before packing eight
orbits into each 48-D token and inverted before decode. This reduces mean
within-token frequency distance from `7.43` to `2.84` (`7.43` to `2.14` for a
true 4x4 grid tile) while preserving coefficient values exactly.

**Result: spiral grouping does not repair Cartesian FFT.** Its 30k grid remains
substantially mushy and close in failure mode to radial `fft_global`, despite a
lower final loss (`0.3414`) than Hartley. Radial grouping alone is not the cause,
and scalar loss again selects the visually worse representation.

**Result: the compact isometric FFT control fails visually but is packing-confounded.**
`fft_compact_isometric_spiral` applied the isometric sqrt(2) Hermitian
packing to the same globally standardized pixels as the successful pixel/DCT/
Hartley arms, keeps ordinary complex orbits as indivisible six-D units, combines
the four self-conjugate three-D coefficients into two units, and packs exactly
3,072 active coordinates as 64x48. It has no per-orbit centering, whitening,
inactive dimensions, or padding. Round-trip, energy, and linear Gaussian-bridge
tests pass, yet the 30k grid remains rough/mushy. The audit found that the two
prepended self-conjugate units put DC and Nyquist-scale values together in token
zero (`~634:1` within-token scale ratio). The result therefore motivates a
corrected packing/co-tokenization control rather than closing compact FFT. The
trailing-20 loss mean is `0.4151`; `0.3830` was one final minibatch.

The compact coefficient distribution makes the remaining mismatch concrete. On
4,096 CIFAR-10 images, active scalar absolute values have p50/p90/p99/p99.9/max
`0.112/0.686/3.363/12.705/58.047`; complex amplitudes have
`0.145/0.752/3.651/14.989/58.047`. Median amplitude falls from `11.28` at DC to
`0.0456` at radii 16--23; DC-orbit RMS is about 594x the maximum-radius RMS.
Within-coordinate standardized tails are much milder, so most of the apparent
heavy tail is a mixture of known, strongly frequency-dependent scales rather
than an unbounded marginal at each frequency.

Normalization/packing controls now stop. The native-complex Phase-B factorization
improves teacher-forced geometry substantially but fails free rollout as described
above. The subsequent 64-token AR Hartley control also fails through 10k despite
a genuinely held-out context gap (`0.117`). It was nevertheless trained on 24x
fewer image exposures than the 30k joint Hartley control, so the contrast does
not rule out causality, phase geometry, or horizon under a matched budget.

The Phase-D codec and generator matrix is now complete. Its original
global-Hartley arms are much weaker than the matched local-raster and local-DCT
arms, including the old MSE C8 codec and a causal C4 run. The C4 global AR reaches
rough structure, so this is a quality hierarchy rather than a binary impossibility.
This supersedes the interim perceptual-AE recommendation. The spatialized-prefix
and global output-grouping ablations are also complete.

### Superseded execution plan (2026-08-04)

Retained below as the reasoning record. Section 10 is the current execution
queue after the amplitude-coordinate, polar-history, and ring-codec review.

The representation and grouping matrix has passed its decision gate. Do not
launch another KL, whitening, phase-process, timestep, or generic conditioning
sweep before exploiting it.

1. **Robust working baseline:** keep the deterministic perceptual C4 codec and
   patch-major local 2x2 DCT tokens as the default 16-step AR. It is 12x
   compressive, frequency-domain, and the strongest causal latent arm. The
   frequency-major variant is a viable coarse-to-fine alternative, but is
   visibly softer (`1.195 / 1.407 / 1.738 / 0.332`) than patch-major
   (`1.146 / 1.338 / 1.819 / 0.481`).
2. **Completed grouping bracket:** raw global 2x2 Hartley tiles are rough but
   partly semantic. Four-tile bands are worse (`1.308 / 1.553 / 1.710 / 0.157`).
   Scalar frequency tokens remain rough despite a lower clean loss
   (`1.180 / 1.364 / 1.684 / 0.320`). A single 256-D Hartley target is rough, as
   is the orthogonally matched single spatial target. Therefore neither exact
   frequency identity, small fixed bands, nor eliminating AR exposure is a
   stand-alone repair.
3. **Next scientific gate — fixed-shape support sweep:** use block DCTs on the
   same 4x8x8 normalized C4 map while preserving 16 tokens x 16 dims. The 2x2
   endpoint is the passing baseline. Add 4x4 local blocks split into low-to-high
   coefficient groups, then one full 8x8 block split the same way. This changes
   coefficient support radius without changing real-vs-complex geometry, total
   dimensions, AR horizon, or diffusion-head width.
4. **If intermediate local blocks pass:** build a multiscale local spectral
   hierarchy rather than jumping directly to full FFT tokens—coarse local DC and
   low bands first, then progressively finer blocks/residuals. Keep errors local
   at every emitted step.
5. **Practical AE route:** the C4 codec already proves the escape hatch. Generate
   local spectral latents, decode deterministically to the image, and apply an
   orthonormal FFT/Hartley transform when coefficient output is required. Audit
   coefficient error by radius and preserve the natural energy hierarchy. Train
   a new codec only to improve rate/fidelity or impose a multiscale latent layout;
   use posterior means/deterministic decoding and no strong Gaussian KL.
6. **Native-complex return is conditional:** revisit log-amplitude plus intrinsic
   phase heads only after a local/multiscale real spectral hierarchy passes. The
   wrapped-normal negative says phase geometry is not the first bottleneck, not
   that complex geometry is irrelevant.

The 2-D RoPE coordinates remain a useful attention chart, not an assumption that
Fourier dependencies are monotonically local. Learned target slots identify the
coefficient and RoPE oils attention; neither replaces a target factorization that
keeps jointly important global coefficients mutually consistent.

## 10. Revised execution queue (2026-08-05)

The recent review changes the implementation details and softens two earlier
claims, but it does not overturn the completed control matrix:

- overcomplete latents are not intrinsically bad; the 53 x 64 ring latent is a
  failed *particular geometry*, not evidence that every useful AE must contain
  fewer scalars than pixels;
- the completed factorized-polar run did not exhaust that family. Its amplitude
  coordinate used a per-radius/RGB RMS divisor, `log(a + 1e-4)` without fitted
  population centering/scaling, a depth-3 decoder, and Cartesian-only trunk
  history. Those are real remaining confounds, although the wrapped-normal and
  rollout results still make phase noising alone a low-priority explanation.

The queue is ordered by expected information gain. Independent arms at the same
priority may run concurrently when `gpu-claim` reports free devices.

### Priority 1: fixed-shape block-DCT support sweep — completed

This remains the cleanest test of the current load-bearing hypothesis. Use the
same deterministic perceptual C4 codec and hold the generator interface at
exactly 16 tokens x 16 values:

| arm | DCT support | spatial blocks | frequency groups per block | status |
|---|---:|---:|---:|---|
| `support2` | 2x2 | 16 | 1 | passes; `1.337 / 1.829 / 0.492` |
| `support4` | 4x4 | 4 | 4 | passes; `1.405 / 1.744 / 0.340` |
| `support8` | 8x8 | 1 | 16 | passes; `1.479 / 1.758 / 0.279` |

Each token contains four DCT frequencies across four latent channels. Larger
supports are split in radial low-to-high DCT order. Because the three layouts do
not share a meaningful spatial 2-D token grid, all three arms use the same 1-D
sequence RoPE plus learned target slots; the existing spatial-RoPE 2x2 run is not
used as the sole matched reference. Model width/depth, diffusion head, data,
seed, schedule, solver, and channel normalization remain fixed.

All three fixed and fresh grids are recognizably semantic. The monotonic decrease
in shuffle gap says larger support makes causal context harder to exploit, but
the full 8x8 endpoint rules out support radius as a binary explanation.

A follow-up 2x2 basis/grouping matrix also completed at the same 16 x 16 shape:

| basis / grouping | clean | shuffled | gap | visual gate |
|---|---:|---:|---:|---|
| full DCT / radial quartets (`support8`) | `1.479` | `1.758` | `0.279` | pass |
| full DCT / contiguous 2x2 tiles | `1.445` | `1.753` | `0.309` | pass |
| full Hartley / radial quartets | `1.462` | `1.684` | `0.222` | pass |
| full Hartley / contiguous 2x2 tiles | `1.436` | `1.761` | `0.325` | pass |

Therefore neither real basis family nor these two grouping laws is a binary
separator. Keep local DCT as the strongest practical reference, but do not build
the scientific roadmap around a universal locality theorem.

### Priority 2: compact native-complex C4 bridge — completed, passes

Apply an exact orthonormal Hermitian FFT packing to the same population-
standardized 4 x 8 x 8 C4 latent map. Pair the four self-conjugate coefficient
vectors, keep every ordinary four-channel real/imaginary orbit intact with the
required sqrt(2) scale, and export exactly 256 active values as 16 x 16 tokens in
radial order. Use the same sequence RoPE, learned slots, model, flow head,
schedule, seed, and AE as the passing matrix.

The arm passes at every saved gate and under fresh seed 54321. Its final held-out
clean/shuffled/gap is `1.458 / 1.679 / 0.221`. Round-trip, energy, CPU end-to-end
smoke, and shape tests also pass. Cartesian complex coordinates, Hermitian
packing, implicit phase, Gaussian linear flow, and Euclidean loss are therefore
all modelable on this short standardized aggregate.

### Priority 3: phase-preserving raw spectral scaling — completed, negative

The passing C4 bridge exposes a large endpoint-statistics difference. On the
same 4,096 images after spatial-domain population normalization:

| endpoint | DC / highest-radius RMS | active `|x|` p50 / p90 / p99 / p99.9 |
|---|---:|---|
| raw CIFAR compact FFT | `594.10x` | `0.112 / 0.686 / 3.363 / 12.705` |
| standardized C4 compact FFT | `3.66x` | `0.547 / 1.541 / 3.216 / 5.722` |

Run the exact failed raw 64 x 48 compact-FFT control with one uncentered positive
RMS divisor shared by real and imaginary coordinates for each radial-bin/RGB
pair. Raise the divisor to exponent `0.8`; this leaves roughly
`594^(1-0.8) = 3.6x` residual radial hierarchy, deliberately matching C4 while
retaining some natural spectrum weighting. One final global RMS sets total
coordinate energy to one. There is no complex mean subtraction, angle change,
padding, inactive coordinate, or covariance rotation.

Implementation: `scripts/run_fft_compact_scaled_spiral_control.sh`. Unit tests
verify inverse round-trip and unit RMS; the GPU smoke passes. At both 5k and 10k
the decoded grid remains texture and is close to the unscaled control (correlation
`0.943/0.940`, MAE `0.068/0.060`). It remains rough/mushy through 30k; final loss
is `0.7905` in the changed coordinate system and is not cross-arm comparable.
Scale hierarchy alone has failed the repair gate.

### Priority 4: raw-resolution matrix — completed, all pass

Run three matched 16 x 16-image arms, each with 16 tokens x 48 dimensions and
the original 115.5M bidirectional model: pixel patches, unscaled compact FFT, and
the exponent-0.8 scaled compact FFT. All use the same resized CIFAR data, seed,
schedule, and 10k budget. The compact codec's exact round-trip and energy tests
now cover image sizes 8, 16, and 32; the scaled 16-pixel GPU smoke passes.

All three arms produce recognizable low-resolution objects/scenes by 5k and pass
clearly at 7.5k/10k. Scaled and unscaled FFT grids correlate `0.968--0.970`
throughout, so scale normalization is again not the repair. Reducing 32-pixel
detail to 16 pixels changes compact FFT from persistent texture to semantic
generation. This leaves resolution/high-frequency dependency burden and token
granularity (16 x 48 versus 64 x 48) coupled.

### Priority 5: matched 64-token granularity separator — completed, passes

Keep the now-passing 16-pixel endpoint but reshape its exact compact FFT from
16 x 48 to 64 x 12. Compare against matched 2x2 pixel patches, also 64 x 12.
Both use the same joint model, seed, schedule, and 10k budget. The compact token
reshape remains exactly invertible and passed unit and end-to-end GPU smokes.

Both arms pass through 10k. The compact-FFT grids become almost identical to the
16 x 48 tokenization as training proceeds: cross-layout correlation
`0.976 / 0.986 / 0.990 / 0.994` at 2.5k/5k/7.5k/10k. Sixty-four attention tokens
and smaller token width are therefore not the cause at fixed low resolution.
The added 16--32-pixel high-frequency dependency structure is the leading raw
obstruction.

Implementation: `scripts/run_compact_fft_resolution_arm.sh` with `PATCH=2` for
pixels and `COMPACT_TOKEN_DIM=12` for FFT.

### Priority 6: true ring-block codec and generator — completed, negative

Priority 5 confirms a resolution/high-frequency burden, so promote the ring
design ahead of more phase-noise work. The operative solution hypothesis is:
generate a low-frequency block jointly, condition later rings causally on all
completed lower rings, and allow bidirectional mixing/denoising within the active
ring. This directly matches the successful low-resolution control and avoids
requiring hundreds of independently sampled scalar phases to remain coherent.

Treat this as a staged redesign, not a simultaneous AE/generator sweep:

1. Change the current sector-causal AE mask to ring-block causal: coefficients
   and exported latents within the same ring mix bidirectionally, while later
   rings remain hidden. Compare against the existing 53 x 64 deterministic codec
   at matched seed and training budget.
2. Gate on held-out reconstruction, perturbation panels, latent effective rank,
   and decoder smoothness. Metadata and learned Perceiver queries remain free
   conditioning and are not reconstruction targets.
3. If the codec passes, generate approximately 23 ring blocks autoregressively,
   denoising all `K_r` latents in the current ring jointly. Training uses one
   fixed block-causal mask over the flattened 53-token layout; a padded
   `[B, rings, K_max, D]` view is needed only by the variable-width ring decoder.
4. Then test latent width/token-count tradeoffs. Include 64-D overcomplete codes
   as a legitimate reference alongside 16-D and 8-D variants; judge geometry and
   sample quality rather than scalar compression alone.

Use deterministic posterior means as generator targets. Fixed decoder-input
noise is allowed as local smoothness regularization because it applies no KL
pressure to Gaussianize the clean aggregate. Do not reopen a generic KL sweep.

Implementation status: `AutoencoderConfig.ring_block_causal` and
`--ring_block_causal` now switch the existing target-12 codec from sector-causal
to ring-block masks without changing its 53 exported sector latents. Encoder
tokens and decoder latents mix bidirectionally within a radius; coordinate decode
can see every latent in its own and earlier rings. Default `false` preserves old
checkpoints and is omitted from legacy layout hashes. Mask, legacy-interface, and
end-to-end GPU smoke tests pass. The matched seed-1, 30k, 53 x 64 deterministic
codec completed under
`autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-ringblock/`.
Test PSNR improves from `28.05 dB` at 10k to `31.82 dB` at 20k and `32.85 dB`
at 30k, but misses the legacy codec's matched `34.77 dB` by `1.92 dB`.
Equal-noise decoder robustness is only `0.1--0.2 dB` better, 90%-energy PCA rank
is `166` rather than `168`, the ordinary causal-probe gain improves slightly
(`10.95%` versus `10.14%`), and the ring-summary probe worsens (`9.44%` versus
`14.13%`). The codec does not establish a better generative geometry and is not
promoted over the legacy deterministic codec.

The gated generator is also implemented in
`model_ring_latent_continuous.py` and `train_ring_latent_continuous.py`. It packs
the 53 normalized latents into 23 causal ring steps with at most four 64-D
latents per step. One 256-D masked diffusion head denoises every active latent
in the current ring jointly. The Transformer consumes only completed earlier
rings; learned absolute ring slots supply identity, fp32 sequence RoPE supplies
relative attention geometry, and QK normalization stabilizes attention logits.
No second physical-metadata path is enabled in this first arm. Packing,
causality, cache parity, gradients, deterministic sampling, checkpoint
contracts, a two-step GPU integration smoke, and a full-width/batch one-step GPU
smoke all pass.

Matched generator arms then ran for 10k on both the 34.77 dB legacy codec and
the 32.85 dB ring-block codec. Both remain texture/pseudo-scenes under fixed and
fresh seed 54321 at every gate. Final teacher-forced conditional x0 MSE is
`0.217` for the legacy codec and `0.205` for the ring-block codec, but the latter
numeric advantage never appears in free rollout. CFG above one mostly increases
contrast. Jointly denoising a complete ring and shortening the rollout from 53
sector steps to 23 ring steps is insufficient.

### Priority 7: more autoregressive decisions — completed, negative

The failed ring result strengthens the deferred hypothesis that 53 decisions
may already be too few, while 23 is worse: each diffusion call must resolve too
much conditional structure. Reuse the existing target-4 deterministic codec,
which exports 134 x 64 latents and reaches `49.57 dB` test PSNR. Generate those
134 latents one at a time with the same learned target slots, fp32 sequence
RoPE, QK normalization, depth-6 flow head, optimizer, and 10k visual gates. This
changes causal granularity without first changing latent width or retraining a
codec.

The frozen interface and grouped generator now support arbitrary latent counts,
dimensions, and ring or token grouping. Dynamic-interface tests and a full-
width/batch GPU smoke pass. The 134-step, 64-D run remains texture/pseudo-scenes
through 10k and fresh seed 54321. Its final teacher-forced conditional x0 MSE is
only `0.140`, substantially better than the 23-step arms, but free rollout is not.
More decisions make each prediction easier while extending exposure to sampled
history; granularity alone does not repair the family.

### Priority 8: more tokens plus a smaller latent target — completed, negative

Test the combined version of the user's hypothesis rather than extrapolating
from the 64-D negative. Train the same target-4, 134-token sector-causal codec at
16 latent dimensions. It exports 2,144 scalars for 3,072 pixels (1.43x scalar
compression) and makes each eventual diffusion target four times smaller. The
codec reaches `32.76 dB` held-out at 10k, already comparable to the 32.85 dB
ring-block codec at 30k. At 20k it reaches `34.07 dB`, pixel MSE
`3.926e-4`, physical FFT NRMSE `0.03677`, phase circular error `0.01379`, and
radial-power relative error `0.01182`. The deterministic interface fitted to
that exact checkpoint has a single-token linear probe gain of `2.36%` and a
ring-summary gain of `11.78%` over their zero baselines. The final 30k codec
reaches `34.34 dB`, only `0.27 dB` beyond 20k.

The reconstruction gate therefore passed before the codec's final 30k
checkpoint. The matched 134-step generator now uses `checkpoint_20000.pt` and
`latent_interface_20000.pt`, target width 16, learned target slots, fp32 RoPE,
QK normalization, and a depth-6 flow head. It runs at
`latent_continuous_runs/grouped-token-t4-z16-c20k-w768-l12-d6-s1-n10000/`.
Keep the codec training to 30k as a reconstruction endpoint, but do not silently
swap its checkpoint under this generator contract. Only after this arm should
an 8-D codec be considered. Its visual failure is an implementation-level
negative; the training-batch condition metric cannot close token count, AR depth,
joint-ring width, or moderate learned compression generally.

All fixed-seed gates from 2.5k through 10k remain texture/pseudo-scenes under
CFG 1.0--2.0, and fresh seed 54321 fails at both 5k and 10k. The 5k and 10k
fresh grids are
`diagnostics/grouped_token_t4_z16_c20k_{5k,10k}_fresh_54321.png`. Final
training-batch conditional/null x0 MSE is `0.254/0.365` with a reported gap of
`0.336`; this does not establish held-out context use. A matched perturbation audit
also weakens the simple "smaller is easier" story: z16 and z64 need `172/178`
PCA dimensions for 90% sample energy, while z16 is roughly 4 dB less robust to
the same standardized latent perturbation (`34.47` vs `38.47 dB` at sigma 0.1).
Do not run z8 as the next arm: this branch compacted redundant coordinates but
did not simplify the sampled manifold or stabilize decoded error.

### Priority 9: direct raw compact-FFT ring transfer — not promoted

The ring construction is not specific to the AE. Exact 32 x 32 compact FFT has
the same 23 integer-radius rings, with `3--288` active real coordinates per ring
and 3,072 total values. That is closely matched to the prepared AE generator's
23 rings, `64--256` active values per ring, and 3,392 total values.

After the AE ring generator reaches a visual gate, transfer the same causal
trunk and joint masked diffusion design directly to the raw isometric compact
FFT. Use the ordinary global pixel mean/std and no per-frequency scaling in the
first arm; the completed exponent-0.8 control already shows that radial scaling
does not repair the endpoint. Pad each target ring to 288 values, keep exact
active-coordinate masks, and preserve all Hermitian/self-conjugate coordinates
without pairing self-conjugate values from different rings.

This is the clean AE-necessity separator:

- both generators pass: ring factorization is sufficient and the AE is optional;
- AE passes but raw fails: the learned transport/decoder regularization is doing
  essential work beyond the causal schedule;
- neither passes: within-ring jointness is not the missing dependency law;
- raw passes but AE fails: the learned latent geometry or decoder, not Fourier
  coefficients, is the new bottleneck.

The prerequisite visual gate failed on both codecs. Do not launch this arm now:
it would repeat a factorization already shown insufficient. Retain the tight
shape match as an AE-necessity separator only if another ring model first passes.

### Priority 10 result: bounded factorized-polar v2 — both controls fail

Do not launch a broad phase-process or loss sweep. First fit and save amplitude
coordinate statistics, then run two cumulative controls:

1. **Coordinate control:** retain radial/RGB RMS scaling and Cartesian trunk
   history, but replace the near-singular `log(a + 1e-4)` coordinate with
   `log(a + epsilon)` at `epsilon=0.1` in RMS-relative units and fit one
   population mean/std (or RGB-shared-across-frequency statistics) for the
   resulting log amplitude. Keep the old depth-3 heads so this arm isolates the
   coordinate distribution.
2. **Full polar interface:** start from the coordinate control, use depth-6
   amplitude and phase heads, and replace Cartesian history with
   `[u, g(a)cos(phi), g(a)sin(phi)]`. Use the completed amplitude and phase
   directly at rollout rather than round-tripping through `atan2`. The phase
   gate suppresses undefined near-zero phase; a globally normalized Cartesian
   auxiliary was intended to supply the physical `a^2 dphi^2` metric and natural
   spectrum hierarchy. **Audit correction:** at weight `0.1`, it is only a weak
   auxiliary; the main phase loss averages tokens approximately equally.

The 10k training-set audit found that current RMS-relative amplitude has
p10/median/p90/p99 `0.207/0.615/1.480/3.268`. After standardization,
`epsilon=0.1` gives skew `0.044` and excess kurtosis `0.075`, while current
`log1p(a)` (equivalent knee `epsilon=1`) gives `1.071/1.957`. Thus `0.1` is a
measured starting point, not an arbitrary constant.

The coordinate control is implemented in
`scripts/run_phase_b_ar_fft_factorized_polar_v2_coordinate_10k.sh` and completed at
`continuous_runs/ar_fft_factorized_polar_v2_eps01_global_10k/`. It fits one
population affine map across all frequencies and RGB channels, preserving the
natural hierarchy. On 49,984 training examples the fitted coordinate mean/std
are `-0.34504/0.64290`; these statistics are saved separately and embedded as
checkpoint buffers. The implementation passes standardized polar/Cartesian
round-trip, fitted-moment, backward-compatibility, train/generate smoke, and the
full test suite.

| coordinate-only step | training-batch clean | shuffled | gap | fixed visual |
|---:|---:|---:|---:|---|
| 2,500 | `2.914` | `3.915` | `1.001` | speckle/texture |
| 5,000 | `2.733` | `4.335` | `1.602` | smoother texture, no objects |
| 7,500 | `2.563` | `4.569` | `2.006` | same failure |
| 10,000 | `2.388` | `4.686` | `2.297` | same failure |

The final fixed and fresh grids are non-semantic. The corrected coordinate is
not inert: relative to the old arm, its fresh oracle-history physical NRMSE
improves from `0.609` to `0.563`, and true-prefix-384 phase coherence improves
from `0.384` to `0.429`. Those conditional gains do not survive free generation.
This shows that the coordinate change was not a visual repair. Because the old
and new context tables were training-batch and the heads differ in conditioning,
it does not cleanly quantify the coordinate's causal effect.

The cumulative full interface completed at
`continuous_runs/ar_fft_factorized_polar_v2_full_eps01_global_d6_10k/`. It
replaces Cartesian trunk input with standardized
`[u, g(a)cos(phi), g(a)sin(phi)]` and uses depth-6 heads. Its launcher is
`scripts/run_phase_b_ar_fft_factorized_polar_v2_full_10k.sh`.

| full-interface step | training-batch clean | shuffled | gap | fixed visual |
|---:|---:|---:|---:|---|
| 2,500 | `2.478` | `4.230` | `1.752` | speckle/texture |
| 5,000 | `2.423` | `4.716` | `2.292` | same failure |
| 7,500 | `2.249` | `4.556` | `2.307` | coarse regions, no objects |
| 10,000 | `2.255` | `4.797` | `2.541` | same failure |

The deeper true-polar interface improves final teacher-forced phase coherence
to `0.853` from the coordinate arm's `0.829`, log-amplitude MAE to `0.246` from
`0.258`, and raises history-input RMS from `0.025` to `0.650`. Every fixed sample
still lies in the same texture/pseudo-scene basin. Standardized polar history,
frequency-deep conditioning, and depth-6 decoder heads did not repair this arm,
but the loss and conditioning confounds prevent a broader closure. The final
fresh/prefix diagnostic is recorded separately in
`diagnostics/factorized_polar_v2_full_10k/`.

Only if the full interface visibly improves free samples should normalization
scope become the next arm: compare the radial/RGB divisor with one common RGB
scale. Per-frequency variance standardization is not the default because it
equalizes the spectrum's conditional SNR. Likewise, compare `x0`, epsilon, and
velocity only after a representation passes a visual gate; first use
algebraically equivalent timestep weights, then native recipes. The full arm did
not pass, so neither sweep is promoted.

### Priority 11 result: complete-amplitude conditional joint-phase oracle fails under the current loss

The full polar-interface arm remained non-semantic at 10k, so this bounded oracle
was run.
The current per-frequency decoder implements
`p(a_i, phi_i | x_<i) = p(a_i | x_<i) p(phi_i | a_i, x_<i)`, but it never lets
the phase stage see the complete amplitude field. That is materially different
from the proposed spectrum-level factorization

`p(a_1:L, phi_1:L) = p(a_1:L) p(phi_1:L | a_1:L)`.

Do not build the complete two-stage generator first. Run the smaller oracle:

1. encode held-out images with the same exact compact FFT and standardized
   `log(a + 0.1)` coordinate used by polar v2;
2. give a bidirectional phase model the **complete true amplitude field**;
3. sample all ordinary phases jointly with an intrinsic circular process,
   retaining the amplitude gate and globally normalized Cartesian auxiliary;
4. decode true amplitudes plus sampled phases and compare them visually with
   true-phase and uniform-phase controls.

This is conditional generation, not a claim of unconditional success. Its job
is to test whether a globally coherent phase field is modelable once amplitude
uncertainty and causal phase rollout are removed. Use grouped orbit tokens for
compute, exact intra-group masks, learned absolute group identity, QKNorm, and
fp32 2-D frequency RoPE. Do not independently center `cos(phi)` or `sin(phi)`.

Promotion rule:

- recognizable held-out samples: train an amplitude-only low-to-high AR pass,
  then condition the passing joint phase stage on its complete sampled field;
- broken samples even with true amplitudes: archive this implementation-level
  negative, but do not close the 1,028-step amplitude-then-phase chain until the
  phase objective makes full reconstructed Cartesian error primary;
- a joint phase pass followed by failure under sampled amplitudes: improve the
  amplitude generator or its train/test conditioning, not the phase geometry.

The implementation is `train_joint_phase_oracle.py`, launched by
`scripts/run_joint_phase_oracle_10k.sh`. It passed focused CPU tests, a real
checkpoint/interface smoke, and a full 115M-parameter batch-256 GPU smoke before
launching at `continuous_runs/joint_phase_oracle_true_amplitude_10k/`. The full
repository suite passes `173` tests plus `3` subtests.

This run was already active when the audit arrived. It completed cleanly at 10k:
true phases reproduce references, uniform phases destroy structure, and sampled
phases remain nonsemantic texture at every saved gate and fresh seed 54321. The
last-20 mean +/- SEM is total `1.9652 +/- 0.0097`, phase
`1.9586 +/- 0.0096`, Cartesian `0.06564 +/- 0.00135`. Its primary phase term uses
relative-amplitude gating and the Cartesian loss has weight `0.1`; consequently
this negative is not decisive. Do not launch a follow-up from it before the
post-audit queue below is discussed.

The alternating 1,028-token design remains a later separator. It inserts a full
Transformer update between `a_i` and `phi_i`, but unlike the oracle above it does
not expose `phi_i` to future amplitudes and therefore does not test the strongest
version of the user's proposed amplitude-first factorization.

### Deferred until a visual pass

- native parameterization recipes (`x0`, epsilon, velocity, score) beyond the
  equivalent-weight control;
- Mamba/DeltaNet or mixture-of-experts trunks;
- learned frequency-specific output adapters;
- a global pose/translation phase-gauge latent;
- larger models or longer runs whose only justification is a lower scalar loss.

The promotion rule remains visual: a direction earns further capacity only when
fixed and fresh unconditional samples become more coherent. Conditional loss,
shuffle gaps, phase coherence, and reconstruction metrics explain failures but
do not override broken images.

## 11. Consolidated post-review decision plan (current source of truth)

### What the newest controls remove from the live explanation

- **Frozen latent normalization is not the repair.** Shared-channel and
  tensor-wide affines lower the 53 x 64 joint-flow loss by about half, yet their
  30k fresh samples remain nonsemantic and correlate `0.9943` with each other.
- **A small correct low-frequency prefix is not enough.** Supplying 72% of target
  spectral energy still leaves texture; the learned unknown spectrum is worse
  than zero fill at every tested cutoff. The model uses the prefix weakly but does
  not turn it into a coherent conditional completion.
- **BOS/shift/cache alignment is correct.** Training uses
  `[BOS,x0,...,xL-2] -> [x0,...,xL-1]`; inference uses the identical BOS and slot
  zero before sampling `x0`. Targeted causal/cache tests pass. Exposure bias can
  hurt AR, but it cannot explain the matched bidirectional raw-FFT failure.
- **A global schedule is not a representation-specific geometric fix.** For the
  globally affined orthonormal pixel/DCT/Hartley/compact controls, linear flow,
  Gaussian noise, and MSE are exactly basis-invariant as full vectors. Their
  different behavior must enter through the tokenized network interface and its
  optimization, not through the mathematical forward path alone.

### Ranked working hypotheses and decisive falsifiers

| rank | hypothesis | evidence it explains | cheapest decisive separator |
|---:|---|---|---|
| 1 | **Global token-axis mixing creates architectural friction.** A shared token map and attention stack learn local patch variables easily but struggle when each token is a global mixture. | pixel FID `31.67` becomes `130.57` after only an orthonormal DCT across its 64-token axis; conjugating the pixel model recovers clean global-state samples | completed and confirmed; move to explicit local-domain/dual-domain computation |
| 2 | **Native FFT co-tokenization adds secondary friction.** Which scalars share a token and their scale/support heterogeneity modulate quality beyond globality. | corrected grid-local remains weak at FID `171.38`; scale-homogeneous worsens to `214.04` despite better token scale ratios | completed: self placement is not the repair; preserve locality and stop packing sweeps |
| 3 | **Emergence-rate difference is real but insufficient.** Global interfaces learn more slowly and remain worse at matched budget. | C4 local/full-DCT/Hartley FID improves `72/92/94 -> 62/82/86` from 10k to 30k, leaving a `19--24` gap | completed: change interface before considering a longer schedule |
| 4 | **High-noise structure commitment is the downstream failure mode of 1--3.** The network does not infer the coherent global scaffold while SNR is low, then renders detail onto a bad scaffold. | held-out joint-SNR diagnostic; Gaussian-fit decode; low-frequency clamp; teacher-forced metrics improve without free samples | time-stratified sample metrics after the interface/rate controls; do not launch another generic SNR sweep first |
| 5 | **The AE must learn under a hierarchy-preserving FFT affine.** A frozen-boundary change cannot undo a representation learned from per-orbit-standardized inputs. | valid scope objection; old AE training used per-orbit centering/standardization | completed negative: `global_standardize` AE (`32.54 dB`) plus 30k tensor-wide joint generator |
| 6 | **Polar geometry is required.** | Cartesian loss naturally gates phase by amplitude, while present polar objectives did not; no current positive separator favors polar | revisit only with full-weight reconstructed-Cartesian loss, discrete self-conjugate signs, and matched conditioning |

### Stage A: calibrate the evidence before ranking marginal arms — complete

1. Evaluate the existing pixel, patch-DCT, full-DCT, full-Hartley, and compact-FFT
   30k checkpoints with 5,000 generated samples each against the same cached
   50,000-image CIFAR Inception reference. Report FID/KID plus sample moments.
2. Generate a blind interleaved sheet from at least four new seeds for those same
   arms. Freeze the anonymized key before rating.
3. Continue reporting trailing-window mean and SEM. Loss comparisons are valid
   only for the globally affined orthonormal controls above.
4. Keep the repaired held-out history diagnostic, but do not spend GPU time
   recomputing obsolete training-batch tables unless they affect a live decision.

The purpose is calibration, not metric substitution. Grossly broken samples can
be rejected visually; FID/KID is required for the disputed Hartley-versus-compact
and local-versus-global rankings.

**First calibration result.** The five direct 30k evaluations are complete at
`diagnostics/control_fid/`. Pixel and patch-DCT are statistically/visually the
same strong tier (`31.67/31.38` FID). Full DCT is much worse (`112.82`), and full
Hartley/compact FFT are worse again (`156.36/164.60`). KID agrees exactly. This
confirms a large local-versus-global interface gap and demotes the marginal
Hartley-versus-FFT distinction. The four-new-seed blind sheet was keyed and
SHA256-frozen before rating. All eight panels rated semantic were pixel or
patch-DCT; all twelve rated weak were full-DCT, Hartley, or compact FFT
(`8 TP / 12 TN / 0 FP / 0 FN`). Stage A is therefore complete for this direct
matrix. The orthogonal token-mixing bridge was justified and is now complete.

### Stage B1: orthogonal token-mixing bridge

Add one `patch_grid_dct` direct control. Patchify exactly as the successful pixel
arm, then apply an orthonormal 8 x 8 DCT **across the 64 patch positions** for
each of the 48 within-patch features independently. This keeps shape, scalar
count, normalization, loss, and within-token feature semantics fixed while making
every target token globally supported.

- If it tracks pixels/patch-DCT, global support and high-noise commitment are not
  sufficient; native FFT co-tokenization/boundary structure becomes primary.
- If it falls toward compact FFT, transforming the token axis globally is itself
  the architectural friction; prioritize a hierarchical/local scaffold rather
  than another Fourier coordinate recipe.

This is a cleaner bridge than comparing unrelated FFT and DCT packings because it
changes one exact orthogonal axis of the successful pixel representation.

**Final result:** `patch_grid_dct` tracks the weak full-support tier, not pixels
or patch-DCT. Its 30k FID/KID is `130.57/0.11172`, versus
`31.67/0.01991` for pixels and `31.38/0.02038` for patch-DCT. Trailing loss is
`0.3994 +/- 0.0040` SEM. At 30k its grid still consists mainly of texture and
coarse fragments. This closes the one-variable separator: a global orthogonal
mixture of the token axis is itself difficult for the shared token transformer.
The positive conjugacy controls show that the difficulty belongs to native
computation, not the coordinate state or Gaussian path.

One zero-training positive control conjugates the
successful pixel model by either patch-grid DCT or corrected compact FFT. The
diffusion state and Heun updates remain in global coordinates, but each velocity
call is exactly untransformed to local patches and transformed back. Semantic
samples here would prove that global coordinates are a valid Gaussian/flow state
space while isolating the failure to their *native compute interface*. This is
the direct precursor to a dual-domain denoiser; it is not proposed as a strict
frequency-wise AR solution. Both compact FFT and patch-grid DCT are complete and
produce the same clean pixel-tier sample set, with `2.15e-6` and `5.25e-6` base
round-trip errors respectively.

### Stage B2: corrected compact packing and co-tokenization pair

Version the legacy `self_first` packing rather than mutating old checkpoints. The
corrected packer must be an exact permutation of the 3,072 active isometric FFT
scalars with one pixel-space population affine, no padding, and no per-orbit
centering or whitening. Self-conjugate RGB values stay at their physical
frequency location instead of being prepended as DC/Nyquist pairs.

Train two 64 x 48 arms differing only in permutation:

1. **grid-local:** order coefficient units by their 2-D toroidal frequency
   neighborhood;
2. **scale-homogeneous:** order the same units by train-only uncentered RMS/radius
   to minimize within-token dynamic range.

Save exact round-trip/Parseval/Gaussian-bridge tests and per-token worst/median
scale ratio, radial spread, and toroidal distance. A material visual/FID gain from
either arm localizes the defect to composition. No gain sends us to Stage B3
rather than another normalization sweep.

**Final result:** the corrected active-scalar layout is versioned
separately from the legacy self-first packer. Seventeen focused tests cover exact
inverse, Parseval energy, linear Gaussian-bridge commutation, and inline
self-conjugate placement. The grid-local layout has median/worst within-token
scale ratio `2.69/11.38` and median/worst toroidal distance `3.33/8.17`; the
scale-homogeneous layout changes those to `1.12/7.36` and `8.86/10.57`.
Therefore the intended locality-versus-statistical-homogeneity trade is measured,
not just named. Both matched 30k jobs completed at
`latent_continuous_runs/fft_compact_isometric_{gridlocal,scale}_control/`.
Grid-local FID/KID is `171.38/0.15681`; scale-homogeneous is
`214.04/0.20414`, versus legacy self-first `164.60/0.14690`. Trailing losses are
`0.4150 +/- 0.0037` and `0.4320 +/- 0.0037`. Correcting self placement does not
repair generation, while trading frequency locality for scale homogeneity makes
it materially worse. Co-tokenization modulates quality but is not the primary
failure; stop packing permutations and proceed to Stage C.

### Stage B3: matched C4 optimization-rate control

Retrain the successful C4 local-DCT and the relevant full-support DCT/Hartley
joint endpoints from scratch on identical 30k trajectories. Match batch, model,
image exposures, LR trajectory, solver, and evaluation seeds. Measure blind grids
and 5k FID/KID at 10k/20k/30k.

- Closing gap: support/composition mainly changes rate.
- Stable or widening gap after both plateau: the tokenized architecture has a
  durable global-interface bias.

Only after this separator may the best corrected raw control receive a fresh 90k
schedule. Match image exposures, not merely optimizer steps.

**Matched trio complete:** the joint C4 trainer accepts the already-tested
`full_dct_tiles` inverse path. A CPU end-to-end smoke passed, and the three 30k
arms ran from the same deterministic perceptual C4 checkpoint as
`continuous_runs/joint_c4_rate_{local_dct,full_dct,full_hartley}_s1_30000/`.
Every arm is 16 tokens x 16 values, 115M parameters, batch 256, seed 1, and uses
the same linear flow, schedule, Heun solver, preview seeds, and image exposure.
`evaluate_spatial_latent_joint.py` loads any saved step and applies the same
5,000-sample CIFAR FID/KID, moment, gradient, and radial-power protocol as the raw
direct controls; an end-to-end checkpoint smoke passed.

The common-protocol result is:

| C4 joint basis | FID 10k | FID 20k | FID 30k | KID 10k | KID 20k | KID 30k | trailing loss 30k |
|---|---:|---:|---:|---:|---:|---:|---:|
| local 2x2 DCT | 72.38 | 65.49 | 62.39 | 0.0781 | 0.0683 | 0.0639 | 1.2206 |
| full-map DCT tiles | 91.61 | 86.29 | 81.80 | 0.0966 | 0.0905 | 0.0844 | 1.2700 |
| full-map Hartley tiles | 94.47 | 89.27 | 86.09 | 0.0996 | 0.0942 | 0.0899 | 1.2832 |

All arms improve by roughly 8--10 FID from 10k to 30k, so global coordinates do
benefit from exposure; the local/global deficit does not close and remains
`19.4--23.7` FID. This closes “the old 10k comparison was only too short” as the
primary explanation at matched 30k budget. It does not claim an infinite-budget
asymptote, but it removes the rationale for a raw 90k extension before changing
the compute interface.

### Stage C: architecture response if representation controls remain negative

Do not jump directly to Mamba, DeltaNet, MoE, or a larger transformer. The
conjugated pixel controls now identify the required operation more specifically:
a position-conditioned feature adapter cannot invert a dense mixture across the
token axis. Prioritize cross-token/local-domain computation.

1. **Engineering baseline, already valid:** generate deterministic C4
   patch-local DCT latents, decode to pixels, and apply the exact FFT afterward.
   This is the strongest unconditional route and should remain the reference.
2. **Joint dual-domain baseline:** keep Gaussian state and solver updates in
   compact FFT coordinates, apply the exact IFFT to local patches for denoising,
   and FFT the predicted velocity back. The zero-training conjugacy controls
   already prove this construction at pixel-model quality. A trainable version
   is needed only if frequency-domain features are also to participate inside
   the network; it is not needed to prove feasibility.
3. **Strict Fourier-AR route:** generate the local C4/coarse scaffold first. Then
   predict a Fourier residual or completed spectrum conditioned on the entire
   scaffold. Make causality apply **between** radial rings or blocks; denoise a
   ring jointly/bidirectionally inside the current block. Variable ring widths
   can use padding plus an attention/loss mask, or fixed learned queries with the
   mask carried through the decoder.
4. **First conditional gate:** use the deterministic reconstruction of a real
   image as the scaffold and train only the residual/global completion model. If
   it cannot improve or preserve that scaffold, do not add rollout exposure. If
   it passes, replace the oracle scaffold with samples from the passing local C4
   generator and train with a mixture of encoded and generated scaffolds.
5. Retain fp32 sequence RoPE, QK normalization, and a learned absolute target
   slot in any causal refinement trunk. Position-conditioned input/output
   adapters are a secondary ablation within that model, not the primary repair.
6. Revisit amplitude/phase heads only inside a passing scaffold-conditioned
   refinement. At that point full reconstructed-Cartesian loss can preserve the
   natural energy hierarchy, while intrinsic phase treatment addresses a bounded
   decoder problem instead of being asked to discover global structure alone.

**Stage C1 gate passes through 30k.** The implemented model uses the deterministic
C4 reconstruction of each real image as an oracle scaffold. Its target is the
pixel residual after one train-population tensor-wide scalar affine
(`residual_std=0.04823`). Gaussian state, linear-flow targets, MSE, and all Heun
updates live in the corrected 64 x 48 compact isometric FFT. Each velocity call
applies the exact inverse FFT, forms aligned local 4 x 4 residual and scaffold
patches, runs the matched 115.51M bidirectional transformer, and transforms the
local pixel velocity back to the FFT state. There is no per-frequency centering,
whitening, or scale schedule. Exact inverse, Parseval, identity-velocity
conjugacy, gradient, Heun, CPU end-to-end, and checkpoint/evaluator smokes pass;
the repository suite is `182 passed, 3 subtests passed`.

Under the common 5,000-sample, seed-71001, 50-step protocol, the deterministic
real-image scaffold itself is FID/KID `37.32/0.03646`. Adding a sampled FFT
residual improves to `28.00/0.01656`, `19.44/0.01003`, `14.33/0.00609`, and
`12.63/0.00512` at 5k/10k/20k/30k. Radial power relative error falls from
`0.401` to `0.109/0.0417/0.0461/0.0511`.
Paired completion PSNR is lower than the scaffold because the residual is
stochastic, but improves from `23.83` to `24.27/24.52/24.95 dB` at 10k/20k/30k
(scaffold `26.29 dB`). Treat this as a conditional refinement gate, not an
unconditional CIFAR score. Final sampled training loss is `0.6608`; the last 20
logged points are `0.6620 +/- 0.0041` SEM.

Zero-shot attachment to the completed 30k local-DCT C4 generator also passes,
without generated-scaffold training. The unconditional scaffold baseline is
FID/KID `62.39/0.06393`; the 5k/10k/20k/30k refiners improve it to
`47.83/0.03295`, `40.34/0.02715`, `35.76/0.02290`, and `34.69/0.02254`.
At 10k/20k/30k radial error falls from `0.426` to `0.0628/0.0965/0.0458`.
Object/layout remains fixed in paired sheets while local detail is added. This is
the first trainable confirmation that Fourier coordinates are a
sound stochastic state and that local-domain computation repairs the dominant
native-token failure. The scheduled 30k curve is complete and selects 30k for
both conditional and unconditional use. The two-stage unconditional result is
only `3.02` FID behind the direct 30k pixel control (`31.67`), despite the C4
scaffold beginning at `62.39`. Do not invent supervised pairs for free generated
scaffolds: an unconditional C4 sample has no corresponding real high-resolution
target. Exposure training needs a principled paired corruption or joint
objective, not arbitrary nearest-neighbour targets.

The matched 10k shuffled-condition audit rules out an unconditional texture
overlay as the explanation. It keeps the scaffold receiving the sampled residual
and every Gaussian seed fixed, but rolls only the scaffold patches supplied to
the denoiser within each batch. Oracle completion FID worsens from aligned
`19.44` to shuffled `45.44` (untouched scaffold `37.32`), and generated-scaffold
FID worsens from `40.34` to `65.95` (untouched scaffold `62.39`). Oracle paired
PSNR also falls `24.27 -> 23.51 dB`. The local denoiser is using the matching
scaffold condition, not merely sampling independent high-frequency texture.

The refiner also transfers zero-shot to the project's passing 10k,
16-step frequency-major local-DCT AR scaffold. Common FID/KID improves from
`91.19/0.09966` to `73.66/0.05844` at refiner 20k and `71.70/0.05707` at 30k;
final radial error is `0.0431` versus scaffold `0.443`. This is a real repair,
but the AR scaffold is much weaker than the matched 30k joint C4 scaffold
(`62.39` FID). For that pipeline the next bottleneck is the coarse AR generator,
not the Fourier residual stage.

**The first exposure interpretation is superseded by a token-composition
control.** The old 10k AR used batch 64 (`0.64M` image exposures), whereas every
joint rate arm used batch 256. Retraining the unchanged 16-step frequency-major
local-DCT AR at batch 256 improved scaffold FID/KID from `91.19/0.09966` to
`84.18/0.09207`; attaching the frozen 30k FFT refiner improved the old/new
pipelines from `71.70/0.05707` to `63.25/0.04869`. The earlier inference that
the remaining `11.80` FID difference from joint local DCT was an AR penalty was
not valid: the joint arm kept all four 2 x 2 DCT modes of a local patch in one
token, while the AR arm split modes into frequency-major tokens.

The missing 2 x 2 separator is now complete. A new joint arm uses the exact
frequency-major regrouping, and fresh 30k batch-256 AR trajectories compare
patch-major and frequency-major packing with the same 16 x 16 shape, model,
seed, schedule, RoPE, QK normalization, decoder, and exposure. Common-protocol
scaffold FID is:

| step | joint patch-major | joint frequency-major | AR patch-major | AR frequency-major |
|---:|---:|---:|---:|---:|
| 5k | 82.47 | 95.52 | **77.44** | 89.48 |
| 10k | 72.38 | 86.65 | **75.48** | 84.54 |
| 20k | 65.49 | 78.22 | **75.83** | 84.06 |
| 30k | **62.39** | 74.70 | 75.11 | 82.53 |

Within the joint family, splitting each local patch across subband tokens costs
`13.05/14.27/12.73/12.30` FID at 5k/10k/20k/30k. Within the AR family, the same
change costs `12.04/9.06/8.23/7.41` FID. This is a replicated, durable
token-composition effect, not an emergence-rate artifact. Joint-versus-AR is not
an exact architectural separator because their trunks, heads, solvers, and LR
paths differ; at 30k the remaining same-packing gaps are `12.72` FID for
patch-major and `7.83` for frequency-major. Causality/conditional decoding may
therefore still cost quality, but substantially less than the old confounded
comparison implied at useful AR checkpoints.

The frozen-refiner AR pipeline confirms the same ranking. Patch-major scaffold
FID/KID and completion FID/KID are `77.44/0.08461 -> 52.69/0.03959` at 5k,
`75.48/0.08259 -> 53.34/0.04042` at 10k,
`75.83/0.08402 -> 54.34/0.04093` at 20k, and
`75.11/0.08242 -> 54.06/0.04054` at 30k. The standalone scaffold selects 30k by
only `0.37` FID over 10k, while the end-to-end pipeline selects 5k. Later AR
checkpoints become smoother and their radial mismatch grows; lower training MSE
does not imply a better scaffold. At 30k patch/frequency-major train loss is
`0.3532/0.3815`, yet held-out clean loss is `4.984/5.251`.

The design consequence is stronger than a generic preference for locality:
keep mutually dependent local frequency components atomic. Do not double the
coarse trunk sequence into separate power and phase tokens by default. If polar
heads return, sample amplitude and phase sequentially *inside one decoder step*,
then commit the completed bundle to history. Low-to-high Fourier causality belongs
in the scaffold-conditioned residual route, with causal rings/blocks and
bidirectional denoising inside the current block.

If strict frequency-wise AR is required, the principled unit is therefore a
conditional ring/block after a local scaffold—not a longer flat 514-step chain
or a component-split coarse sequence.

**Stage C2 strict-ring gate is complete and negative.** The controlled implementation
keeps the Stage-C1 deterministic C4 oracle scaffold and its exact tensor-wide
population statistics. A four-layer bidirectional 8 x 8 patch encoder exposes
the complete aligned scaffold as prefix memory. An eight-layer QKNorm causal
trunk then advances through the 23 exact radial bins of the corrected compact
isometric FFT. Its input at ring zero is an explicit BOS; at every later target
it is the preceding *completed* ring. Learned target slots provide absolute ring
identity after input projection, fp32 2-D RoPE is used in the scaffold encoder,
and fp32 sequence RoPE is used for ring attention.

Each target ring is one masked vector containing all active RGB real/imaginary
coordinates, with widths
`3,24,48,60,72,120,108,144,168,168,204,192,240,276,264,288,222,180,108,96,48,36,3`.
A depth-6 AdaLN diffusion MLP jointly denoises the complete vector with a
20-step Heun flow solve. The mask and fixed-288 reduction give every physical
FFT scalar the same common loss coefficient; padding contributes no loss and
rings are not equalized. Complete orbit bundles are never split across history
steps. The 106.09M model reuses only the passing C1 scaffold projection and
learned patch slots; the scaffold encoder, causal trunk, and diffusion head are
fresh.

Tests cover exact scalar packing/inversion, one-ring ownership of every orbit,
BOS train/inference alignment, no future-ring leakage, cached inference matching
full teacher forcing, finite backward, and sequential sampling. CPU, tiny CUDA,
full-width batch-32, full-width batch-128, and evaluator smokes pass. The 10k
run completed in 10.0 minutes with checkpoints at 2.5k intervals under
`latent_continuous_runs/scaffold_fft_ring_residual_oracle_c4_s1_10000/`.
Decoded completions remain destructively textured through 10k. Common
5,000-sample oracle results are:

| checkpoint | completion FID / KID | paired PSNR | radial error |
|---:|---:|---:|---:|
| untouched scaffold | 37.32 / 0.03646 | 26.29 dB | 0.4012 |
| 2.5k, sampled history | 75.82 / 0.06124 | 23.49 dB | 0.0439 |
| 5k, sampled history | 78.80 / 0.06318 | 23.29 dB | 0.1371 |
| 10k, sampled history | 74.53 / 0.05963 | 23.42 dB | 0.0515 |

The low radial error beside very poor FID is important: marginal spectrum
allocation can look correct while cross-frequency phase/coherence destroys the
image. Teacher-forcing true prior rings improves 5k/10k FID only
`78.80 -> 75.90` and `74.53 -> 70.31`; exposure bias is real but secondary.
Increasing the final solver from 20 to the C1-matched 50 Heun steps changes FID
`74.53 -> 75.40`, ruling out under-integration. Shuffling only the scaffold
prefix unexpectedly improves FID `74.53 -> 70.91` and leaves paired PSNR nearly
unchanged (`23.42 -> 23.41 dB`). Unlike C1's large aligned/shuffled separation,
the static ring-summary path has not learned useful aligned conditioning.
Final sampled train loss is `0.4600`; the last 20 logged losses are
`0.4722 +/- 0.0032` SEM and do not rank the decoded result.

This is not evidence that radial causal scheduling itself is impossible. C2
simultaneously replaced C1's evolving aligned local-patch interaction with one
768-D static target summary and a global MLP. The next clean separator, if strict
Fourier causality remains required, is an asynchronous causal-ring schedule with
C1's local dual-domain denoiser: completed lower rings stay at data, the current
ring follows its flow path, future rings stay at base noise, and every velocity
call IFFTs the whole state to aligned residual/scaffold patches before masking
the returned FFT velocity to the current ring. Sample rings proportional to
their active scalar counts and provide an explicit target-ring embedding. That
tests causal factorization while retaining the computation already known to
work. Do not launch more depth, solver, normalization, or polar variations of
the failed static ring MLP.

### Active normalization closure and deferred branches

The matched `global_standardize` 53 x 64 VAE retrain is complete. It uses one
train-population pixel mean/std followed by isometric FFT packing, with no
per-orbit centering/whitening. Held-out reconstruction is `32.54 dB`, below the
old AE's `34.76 dB`; its tensor-wide interface has ordinary/ring probe gains
`0.5906/0.6769`. The one matched tensor-wide joint generator is complete. Record
its visual result, but do not let it block the interface separators. It is now
complete and negative at 30k under both the fixed preview and a 64-image fresh
seed `54321` grid (`latent_rms=1.0047`). The stronger AE-training normalization
objection is closed as a primary repair.

Defer generic KL/VAE sweeps, polar parameterization sweeps, `x0`/epsilon/v
ablations, frequency-specific experts, wavelets, and new sequence trunks until a
separator above moves decoded samples. Keep the deterministic C4 patch-DCT route
as the engineering baseline throughout.

## 12. Archived post-audit queue (superseded by section 11)

The phase oracle that was already running when the audit arrived finished
negative under its current loss. The queue below is retained to preserve the
reasoning trail; completed results have been folded into section 11.

### Priority 0: repair the evidence protocol

1. Keep the `train_continuous.py` fix that evaluates clean/shuffled history on
   the deterministic tail panel excluded from optimization. Add the held-out
   panel size to every log and never call a training-batch metric “held out.”
2. Re-evaluate representative saved raw checkpoints on that panel: ECS/SNR/slim,
   original polar, wrapped-normal, and both polar-v2 endpoints. This is a
   measurement pass, not retraining.
3. Report the mean and SEM of a trailing loss window, not a single minibatch.
   Compare losses only when the transform is orthonormal and the global affine,
   active dimensions, reduction, and objective are identical.
4. Build a blind interleaved sheet across at least four new seeds for the pixel,
   patch-DCT, full-DCT, full-Hartley, compact-FFT, and relevant C4 endpoints.
   Record labels before revealing arm identity.
5. Use the existing `live_evaluation.py`/CIFAR Inception reference to compute
   FID/KID from at least 5,000 generated images for distinctions that affect a
   promotion decision. FID complements decoded inspection; it does not replace it.
6. Repair launchers whose deleted flags make them unreproducible, and record
   batch size, parameters, image exposures, learning-rate trajectory, solver,
   inference steps, normalization, and packing version in every comparison.

### Priority 1: latent population-normalization scope — frozen boundary complete, negative

The original 53 x 64 ring interface fits a separate mean/std for every token
position and latent channel. It therefore removes the position-wise diagonal
hierarchy even though the AE restores much of the covariance eigenspectrum. Two
scopes discussed by the user were not trained:

- one tensor-wide population mean/std, broadcast to all 3,392 latent values;
- one population mean/std per latent channel, shared across all 53 positions.

The latter is the preferred first baseline because latent channels are a learned
feature gauge while sequence position carries frequency identity. On the existing
posterior-mean interface, tensor-wide and shared-channel normalization retain
position-RMS ratios of `1.98x` and `1.92x`; the current position-by-channel affine
sets them to approximately one by construction. Train both scopes against the
same frozen AE and identical 30k joint model. This isolates generator-boundary
normalization without retraining the codec. If either visibly helps, repeat the
winner on the `whiten_exponent=0` AE, whose training input retained the raw FFT
eigenspectrum, before considering a new AE.

Do not describe this as a missing raw-FFT tensor-scale control. Scalar pixel
normalization followed by the orthonormal compact FFT already gives packed-tensor
mean/std `0.0014/1.0000` and preserves the spectrum. Literal subtraction of one
coefficient-wide mean from every real and imaginary value is not geometry-safe.
An RGB-paired scale shared by real/imag and frequency remains a possible later
raw control, preferably folded into the corrected packing matrix.

Implementation: `fit_autoencoder_latent_interface.py --normalization_scope
{channel,tensor}` and `scripts/run_joint_latent_normalization_scope.sh`.

Both 30k arms are complete. Fresh samples remain nonsemantic and the two grids
correlate `0.9943`. Last-20 losses are `0.45823 +/- 0.00206` (shared channel) and
`0.45493 +/- 0.00208` (tensor), versus `0.90010 +/- 0.00263` under the old
position-by-channel affine, but cross-affine losses are not comparable and the
visual distribution barely changes. Do not promote either frozen-boundary scope
as a repair.

This does not close normalization during AE training. A new geometry-safe
`global_standardize` codec applies one train-population pixel mean/std before an
isometric FFT, with no per-orbit centering or whitening. The matched 53 x 64 VAE
retrain completed at `32.54 dB`; its tensor-wide joint generator is complete and
negative. See
section 11 for the live interpretation.

### Priority 2: zero-training low-frequency clamp — completed, negative separator

Use the existing compact-FFT checkpoint and replace/clamp a ground-truth prefix
of low-frequency rings throughout sampling, sweeping a few cutoffs. Compare
against true-spectrum and unconditional controls under the same held-out images.
This is conditional and cannot establish unconditional quality. It cheaply asks
whether the model can generate high-frequency detail once the high-noise stage no
longer has to commit the dominant global structure. If a small low-frequency
prefix changes mush into coherent completions, high-noise/rate allocation becomes
the leading mechanism. If it does not, proceed without interpreting it as a
packing test.

The forward-consistent held-out clamp is complete for radius `0/2/4/8`. The
cutoffs reveal `24/72/85/95%` of target energy. Small cutoffs remain texture;
large cutoffs are recognizable only when the oracle low-pass row already carries
the object layout. Generated unknown coefficients improve MSE versus the
unconditional sample by only `6--16%` and are worse than zero fill at every
cutoff. Full-spectrum clamp reconstruction is exact. Proceed to corrected
packing/co-tokenization rather than treating high-frequency cleanup as the sole
bottleneck. Implementation and artifacts:
`sample_compact_fft_low_frequency_clamp.py` and
`diagnostics/compact_fft_low_frequency_clamp/`.

### Priority 3: corrected compact packing and co-tokenization matrix

Do not mutate the legacy packing in place; old checkpoints need an explicit
`legacy_self_first` version. Add a corrected version with these invariants:

- exact active-only orthonormal real FFT packing and inverse;
- one global pixel mean/std and no per-orbit centering or whitening;
- no inactive or padded targets;
- self-conjugate values assigned near comparable-radius/scale values rather than
  prepended as DC-plus-Nyquist units;
- measured per-token scale ratio, radial spread, toroidal frequency distance,
  and exact Gaussian-bridge tests saved with the run.

Train a minimal two-arm permutation control with the same 64 x 48 values, model,
batch, and schedule: (a) spatially contiguous 2x2/grid-local frequency grouping,
and (b) radius/scale-homogeneous grouping. This directly tests the trade between
frequency nearness and within-token statistical homogeneity. It is more
informative than another normalization sweep because only co-tokenization moves.

### Priority 4: matched C4 optimization-rate control

Before spending on a 90k raw run, retrain the existing C4 local-DCT and global
DCT/Hartley endpoints on matched 30k trajectories from scratch, with identical
batch, parameter count, learning-rate schedule, solver, and evaluation protocol.
The current 10k visual ranking may be rate rather than asymptotic modelability.
Use blind multi-seed grids and >=5k FID/KID. This is the cheaper way to determine
whether support mainly changes convergence speed or leaves a durable quality gap.

### Priority 5: longer direct raw controls, only if still needed

If Priority 3 remains worse after corrected packing and Priority 4 shows that
global representations mainly need more optimization, extend the best corrected
compact-FFT and matched real Hartley/DCT direct controls from a fresh 90k schedule.
Match image exposures, not merely optimizer steps. This supersedes reading the
10k-batch-32 Hartley AR versus 30k-batch-256 joint contrast as matched evidence.

### Priority 6: physically matched polar control

Continue polar work only as a controlled geometry experiment:

- make reconstructed Cartesian error the primary full-weight objective, with
  log-amplitude and circular coordinate losses as auxiliaries;
- alternatively implement the local pullback metric
  `(a + s*epsilon)^2 du^2 + a^2 dphi^2` explicitly;
- handle self-conjugate coefficients as discrete real signs, not free circle
  phases;
- give Cartesian and polar heads identical target-slot conditioning and depth;
- remove the train/sample amplitude-conditioning mismatch;
- population-standardize the log-amplitude coordinate with saved train-only
  statistics, while preserving the physical spectrum hierarchy in the primary
  loss.

Only after that comparison should `x0`, epsilon, velocity/flow, timestep
weighting, alternating amplitude/phase tokens, or all-amplitudes-first decoding
be ablated. A negative from the current oracle is insufficient motivation for a
broad polar sweep.

### Stable baseline and deferred work

Keep the deterministic C4 autoencoder with patch-local DCT latents as the working
engineering route. It provides deterministic encode/decode, local frequency
tokens, and a successful generator; a final full-image FFT can be applied after
decode when global coefficients are required. Defer new KL/VAE sweeps,
Mamba/DeltaNet trunks, wavelets, learned per-frequency experts, and generic loss
or schedule sweeps until one of the separators above identifies a concrete
bottleneck.
