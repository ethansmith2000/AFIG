# Why latent AFIG samples are texture-like

Date: 2026-07-30; control results updated 2026-08-05. Diagnostics in
`diagnostics/`, scripts named `diagnose_*.py`.

AE under test: `ae-causal-ring-t12-...-vae-kl0.0001/checkpoint_30000.pt` with
`latent_interface_posterior_mean.pt` (posterior-mean policy, 34.76 dB test PSNR
reproduced here). Generative model under test:
`joint-vae-mean-rf-w768-l12-b256-s1-n30000/checkpoint_final.pt`.

## Independent-audit correction (2026-08-05)

Two reviewers independently re-derived the main claims from code and artifacts.
Their overlap exposed several load-bearing confounds. These corrections supersede
the corresponding wording in the historical sections below:

1. In `train_continuous.py`, every previously reported `clean/shuffled/gap`
   diagnostic used the minibatch that had just received an optimizer update. It
   was **not held out**. This affects the raw Cartesian ECS/SNR/slim runs, the
   factorized-polar and wrapped-normal runs, and both polar-v2 runs. Their decoded
   fixed/fresh samples and separately evaluated held-out spectral panels remain
   valid; their old context-gap tables must be read as **training-batch** values.
   The trainer now evaluates the deterministic tail panel excluded from training
   and logs `HELDOUT_CONDITION_DIAGNOSTIC` plus its panel size.
2. The grouped/ring generator's conditional and null x0 losses are also
   training-batch measurements. Its visual failure remains, but those numbers do
   not establish held-out context use.
3. The 64-step Hartley AR and 30k joint Hartley comparison is not budget matched:
   `10k x 32 = 0.32M` versus `30k x 256 = 7.68M` image exposures, a 24x gap,
   with different schedules and 20 versus 50 solver steps. It is an early-budget
   negative, not evidence that causality and global support are jointly fatal.
4. The compact isometric FFT control has a token-composition defect. It prepends
   two six-value units made from the four self-conjugate frequencies; token zero
   therefore mixes DC and Nyquist-scale values. The measured within-token
   standard-deviation ratio is about `634:1`. Its transform remains exactly
   invertible and isometric, but this control does **not** close compact packing.
   The legacy 65-token FFT layouts additionally pad 514 orbits to 520, leaving
   48 dead target coordinates after normalization.
5. The old claim that direct-control losses are generally incomparable was too
   broad. For the orthonormally matched global-affine controls they are comparable:
   mean of the final 20 logged losses is pixel `0.3511`, patch DCT `0.3517`, full
   DCT `0.3909`, full Hartley `0.4056`, and compact FFT `0.4151` (SEM about
   `0.004`). This graded loss ladder agrees with the visual ordering. Losses
   remain incomparable across the legacy representations with different fitted
   target scales.
6. The polar phase term does not preserve the natural cross-frequency energy
   hierarchy. The phase gate is normalized across RGB within each token, after
   which tokens are averaged equally; only the `0.1 x` reconstructed-Cartesian
   auxiliary carries physical cross-frequency weighting. A serious polar retry
   must make full reconstructed Cartesian error the primary loss (or implement
   the pullback metric explicitly) and treat self-conjugate signs discretely.
7. Visual labels were assigned from single fixed/fresh grids without a blinded,
   multi-seed protocol, and no FID has yet been computed. Fine distinctions such
   as “rough” versus “pass” are hypotheses until blind interleaved rating and a
   >=5k-sample FID/KID evaluation are run.

The strongest conclusions that survive are narrower: local C4 DCT remains the
best engineering baseline; the old AE aggregate-modelability claim is retracted;
and direct raw compact/native FFT remains harder under the implementations tried.
The current mechanism is best described as graded interface/rate friction and
co-tokenization heterogeneity, not “global Fourier tokens are impossible.”

The later zero-training conjugacy control makes this sharper. A final pixel model
was wrapped so the Gaussian base, all Heun states, and all solver updates live in
the exact corrected compact-FFT coordinates, while only velocity evaluation is
mapped to local 4 x 4 pixel tokens and back. Sixty-four fresh samples are clean
pixel-tier CIFAR objects; base round-trip max error is `2.15e-6`. Thus compact FFT
is not intrinsically a bad Euclidean noising or solver space. It is a poor native
token/computation interface for the tested shared transformer.

## Conclusion

The most controlled current result is now a representation-interface result, not
a locality theorem. For the same perceptual C4 latent map, model, population
channel normalization, Gaussian linear flow, MSE, seed, 16 x 16 interface, and
10k schedule, every real spectral layout in the new matrix passes visually:

- block DCT with 2x2, 4x4, or full 8x8 spatial support;
- full DCT grouped as radial quartets or contiguous frequency tiles;
- full Hartley grouped as radial quartets or contiguous frequency tiles.

Fixed and fresh seeds are recognizably semantic in every arm. Local support is
somewhat crisper and its held-out shuffle gap is larger, so support remains a
graded optimization/quality factor, but full support is not the binary cause of
failure. Likewise, neither real basis nor the tested grouping law separates pass
from fail.

This strengthens the earlier retraction: the AE aggregate distribution is
modelable, and real global frequency coordinates are modelable under the short
standardized C4 interface. The unresolved contrast is native compact Cartesian
FFT over raw pixels (64 x 48, failed at 30k) versus the C4 interface. An exact
compact FFT over C4, with all 256 active values packed isometrically as 16 x 16,
passes at every saved gate and under fresh seed. Its final held-out
clean/shuffled/gap is `1.458 / 1.679 / 0.221`.

The latest native-complex controls sharpen this without reversing it. Replacing
the old near-singular amplitude coordinate with population-standardized
`log(a + 0.1)` improves held-out spectral and prefix metrics, but free samples
remain texture through 10k. Replacing Cartesian history with true standardized
gated-polar features and doubling decoder depth changes the **training-batch**
clean/shuffled/gap to `2.255 / 4.797 / 2.541` and held-out phase coherence to
`0.853`; it also remains nonsemantic. Because the context metric was not held
out and the polar loss underweights physical cross-frequency geometry, these
arms establish only that the coordinate and history variants did not visibly
repair generation. Section 20 records the exact result and audit correction.

The first trunk/decoder representation split has now been tested. It kept global
low-to-high Hartley targets but fed local patches of the inverse-transformed known
prefix to the trunk. It did not improve the earlier raw-global C4 arm: fixed and
fresh grids are nearly identical and held-out context use is weaker. However,
the arm used batch `32` versus `64` and `82.9M` versus `106.5M` parameters, while
its graft PSNR is near the different-image baseline. It therefore does not close
mixing spatial sites in the *trunk input*. The older output
grouping bracket also remains valid as evidence that four very wide targets, 64
very narrow targets, and one 256-D target are poor interfaces. It does not imply
that every full-support factorization is poor: the new fixed-width matrix passes.
Token width, sequence length, endpoint statistics, and basis geometry must be
kept matched before assigning causality.

**Historical scope corrections remain important.** The original joint model
does not memorize, while the old AR weighting arms do and their conditional
advantage reverses on held-out data (section 16). Perceptual loss weighting
worsened that overfit. Diffusion forcing alone does not address the joint failure.
The older sections below are retained as an evidence/reasoning log; read their
inline retractions rather than treating every intermediate implication as current.

## Direct-representation controls: native-complex FFT is harder, but not the final localization

`control_pixel_diffusion.py`. Identical bidirectional transformer blocks, identical
rectified-flow objective, identical width/depth/steps/batch/schedule and the same
50k CIFAR-10 images. The only change is what a token is: a 4x4 pixel patch (64
tokens x 48 dims) instead of a frequency latent (53 x 64).

**At 5,000 steps the pixel control already produces recognizable objects, and at
the full 30,000 it produces clearly recognizable CIFAR classes** -- birds, cars,
animals, ships with coherent backgrounds. The frequency-latent path produces
texture mush at 30,000.

This settles the "is 50k images enough" question directly rather than by
inference. It is enough -- CIFAR-10 is a solved generative benchmark at this scale
(DDPM FID 3.17 at ~36M params, EDM FID 1.79 at ~56M, both on these same 50k
images) -- and the same transformer and budget reach coherent structure on pixels
in a sixth of the training. Neither data, architecture, nor compute is the
bottleneck.

Eight matched direct-representation controls have now completed at 30,000 steps:

| representation | shape | params | 30k decoded result |
|---|---:|---:|---|
| 4x4 pixel patches | 64 x 48 | 115.5M | recognizable CIFAR classes |
| per-patch 4x4 DCT | 64 x 48 | 115.5M | recognizable CIFAR classes |
| full-image DCT, 4x4 frequency tiles | 64 x 48 | 115.5M | recognizable, but delayed/weaker |
| full-image Hartley, 4x4 frequency tiles | 64 x 48 | 115.5M | recognizable, but weaker |
| per-orbit-whitened FFT | 65 x 48 | 115.5M | texture mush |
| FFT without per-orbit variance scaling | 65 x 48 | 115.5M | texture mush |
| locality-regrouped legacy FFT | 65 x 48 | 115.5M | texture mush |
| compact active-only isometric FFT | 64 x 48 | 115.5M | rough/mushy |

Artifacts:
`latent_continuous_runs/pixel_control/preview_0030000.png`,
`latent_continuous_runs/patch_dct_control/preview_0030000.png`,
`latent_continuous_runs/full_dct_control/preview_0030000.png`,
`latent_continuous_runs/full_hartley_control/preview_0030000.png`,
`latent_continuous_runs/control_fft_whitened/preview_0030000.png`, and
`latent_continuous_runs/control_fft_global/preview_0030000.png`,
`latent_continuous_runs/fft_global_spiral/preview_0030000.png`, and
`latent_continuous_runs/fft_compact_isometric_spiral_control/preview_0030000.png`.

The two FFT outputs are not merely both broken; under the paired fixed seed they
are visually very similar (same-channel RGB correlations 0.930--0.940). Their
loss values differ because their legacy target scales differ and must not be
compared. By contrast, loss is comparable across the globally affined
orthonormal pixel/patch-DCT/full-DCT/full-Hartley/compact-FFT controls; the
audit's trailing-20 means form the graded ladder reported above.

This closes two important branches. The autoencoder is **not necessary for the
failure**: direct FFT coefficients fail without it. Per-orbit variance whitening
is also **not the sole cause**: removing that scaling produces essentially the
same broken result. Because the direct FFT controls have the same 115.5M
parameters as the pixel control, the old ~85M latent-model capacity caveat does
not apply to this comparison.

The DCT controls narrow this further. An orthonormal local frequency basis is as
modelable as pixels (`0.3114` versus `0.3116` final loss), while a full-image DCT
eventually produces recognizable objects with a higher final loss (`0.3505`).
Global support is therefore a difficulty but not a fatal obstruction. The
remaining matched-control confound is periodic Fourier geometry versus token
composition: DCT uses contiguous 4x4 tiles of a real frequency plane, while FFT
uses eight radial Hermitian orbits per token. These controls do not establish
that Fourier-domain generation is impossible or that complex phase alone is the
cause.

The token-composition audit quantifies the remaining mismatch. With distance on
the frequency torus modulo Hermitian conjugacy, the eight radial FFT orbits in a
token have mean pair distance `7.43`, mean diameter `13.66`, and only `9.4%`
neighboring pairs. A contiguous 4x4 frequency tile has corresponding values
`2.14`, `4.24`, and `35%`. Radial grouping does preserve magnitude hierarchy
more tightly (mean radial spread `0.34` versus `3.91`), so this is a trade rather
than an unconditionally better ordering. The completed `fft_global_spiral`
control kept every FFT coefficient fixed while reducing mean pair distance to
`2.84`, but its 30k samples remained mushy. Local regrouping alone is therefore
insufficient.
See `diagnose_token_composition.py` and
`diagnostics/token_composition.json`.

Historical normalization caveat: `fft_global` is a shorthand. The implementation
sets the variance-whitening exponent to zero and uses a global residual scale, but
the codec still subtracts per-orbit complex means. It therefore isolates
per-orbit **variance scaling**, not every form of frequency-dependent
normalization. The compact control resolves the affine-normalization caveat: it uses one
pixel mean/std, `fft2(norm="ortho")`, exact sqrt(2) Hermitian packing, no fitted
per-orbit statistics, no inactive coordinates, and no padding. Round trip, L2
energy, and the Gaussian bridge are exact, yet its 30k output remains mushy.
It does **not** resolve token composition: the encoder prepends the paired
self-conjugate units, mixing DC and Nyquist values in token zero (about `634:1`
within-token scale ratio). A corrected compact layout is therefore required
before treating this negative as exact (`0.4151` trailing-20 loss mean; the old
single final-batch value was `0.3830`).

The compact distribution also quantifies why native complex coefficients remain
statistically awkward despite exact Euclidean geometry. On 4,096 CIFAR-10 images,
complex-amplitude p50/p90/p99/p99.9/max is
`0.145/0.752/3.651/14.989/58.047`; median amplitude falls from `11.28` at DC to
`0.0456` at radii 16--23, and DC-orbit RMS is about 594x maximum-radius RMS.
Within-coordinate standardization produces far milder tails. Thus much of the
global heavy tail is a mixture across known frequency-specific scales. Exact
Cartesian noising is legal -- the packing is an orthonormal chart of pixel space
-- but it is evidently not an easy statistical substrate for the shared decoder.

### The exact raw-FFT AR control does not emerge by 10k

Two new short 2.5k-step arms exercise the normalization/noise audit proposed
above. They use the existing 514-group causal AR path, an isometric Hermitian
real/imaginary packing (3,072 active real coordinates), one global pixel mean,
one robust DC-derived scale, exact target-frequency metadata, and no frequency
loss weights.
Tests verify pixel/Fourier L2 equality, round trip, Gaussian-bridge equality,
velocity equality, and physical decode.

| arm | final training-batch clean | shuffled | gap | decoded result |
|---|---:|---:|---:|---|
| Cartesian + ECS | 0.012045 | 0.013190 | 0.001144 | texture mush |
| Cartesian + ECS + 4x SNR | 0.023469 | 0.026515 | 0.003047 | texture mush |

The fitted global affine values were pixel mean `0.4733601` and scale `10.43208`.
The baseline normalized history RMS was `0.02513` against unit Gaussian noise.
The 4x-SNR arm scales the internal clean endpoint by two, so its bridge SNR is
exactly four times larger while external token coordinates and physical decode
remain unchanged.

On the just-updated training batch, the SNR change increased the shuffle gap:
its final gap is 2.7x the baseline, and its step-2k gap is `0.003821` versus
`0.001602`. The audit found that this was not a held-out context test. Both
16-image decoded grids contain only low-frequency colored
texture and no recognizable objects. The grids are numerically different (pixel
MAE `7.26/255`) but qualitatively fail in the same way.

**Retraction of the initial stop decision.** Both runs were configured with
`max_train_steps=2500`, so cosine decay reached zero learning rate exactly at the
only preview. This is not the same optimization trajectory as a 2.5k checkpoint
from a 10k/30k run. The matched pixel control first saved at 5k and is object-like
there; it has no 2.5k artifact. Texture at 2.5k therefore does not close the raw
FFT branch. What these runs establish is narrower: exact geometry is operational,
c=4 improves causal conditioning, and neither short run had emerged by its end.
A fresh c=4 run with a 10k cosine schedule is required before interpreting the
remaining failure as global support or joint magnitude/phase modeling.

That corrected 10k run has now completed:

| step | training-batch clean | shuffled | gap | decoded result |
|---:|---:|---:|---:|---|
| 2,500 | 0.023732 | 0.027525 | 0.003793 | texture, no clear objects |
| 5,000 | 0.021503 | 0.027910 | 0.006406 | texture, no clear objects |
| 7,500 | 0.021457 | 0.028950 | 0.007492 | texture, no clear objects |
| 10,000 | 0.019638 | 0.026835 | 0.007197 | texture, no clear objects |

This separates two facts that the short run could not, with an audit correction.
First, the AR trunk increasingly uses causal context on its current training
batch; this does not establish held-out context use. Second, none of that gain
becomes recognizable unconditional structure. The fixed-seed grids change little
after 5k, and a new 16-sample seed at 10k fails in the same way. The pixel
control, by contrast, is recognizably object-like in its stored 5k and 10k grids.

The final teacher-forced spectral panel is consistent with the proposed Phase B
factorization: log-amplitude bias is `-0.961`, log-amplitude MAE is `1.174`, phase
coherence is `0.250`, and physical complex NRMSE is `0.423`. These diagnostics do
not prove that factorization will work, but they show that a low normalized MSE
coexists with severe errors in the physical amplitude/phase quantities sampling
must compose across 514 steps.

This closes the present Cartesian/ECS/c=4 arm under the planned early-training
budget, not the broader Fourier premise. Correct Hermitian noise geometry, one
global scale, a 4x SNR shift, explicit frequency metadata, and genuine context
use are not sufficient in this implementation.

### Product-space amplitude/phase AR improves conditional modeling, not samples

The next 10k arm changed the decoder coordinates rather than applying another
Cartesian normalization. For each of 514 native complex frequency groups it
samples normalized log amplitude with a Euclidean flow, then samples phase with a
uniform-base intrinsic circular flow conditioned on that amplitude. The sampled
coefficient is converted back to Cartesian ECS coordinates for trunk history.
The trunk uses QKNorm, fp32 2-D RoPE, learned target slots, and per-block position
FiLM; both decoder heads also receive the target slot directly.

| step | training-batch clean | shuffled | gap | held-out phase coherence | held-out physical NRMSE | samples |
|---:|---:|---:|---:|---:|---:|---|
| 2,500 | 2.8205 | 3.4556 | 0.6351 | 0.715 | 0.405 | speckle |
| 5,000 | 2.4981 | 3.8387 | 1.3406 | 0.792 | 0.385 | no objects |
| 7,500 | 2.4223 | 4.0842 | 1.6619 | 0.818 | 0.381 | no objects |
| 10,000 | 2.3639 | 4.1293 | 1.7654 | 0.824 | 0.378 | no objects |

Final log-amplitude MAE is `0.278` with bias `+0.021`, versus the Cartesian
arm's `1.174` and `-0.961`; phase coherence rises from `0.250` to `0.824`.
The factorization repairs much of the held-out teacher-forced coordinate problem,
while its large shuffle gap is only a training-batch result. It does **not**
repair free generation: all four fixed-seed grids remain high-frequency
texture/speckle.

The matched rollout audit (`diagnose_factorized_rollout.py`) rules out a simple
amplitude explosion. With true prefixes of 32/128/256/384 coefficients, sampled
suffix mean-amplitude ratios are `0.971/1.023/1.000/1.028`. Nevertheless, a true
32-coefficient prefix plus zero suffix is already a blurred recognizable image,
while rolling out the model suffix destroys it. Longer true prefixes survive but
gain harmful texture. Fully generated cutoffs show only coarse blobs at 32, weak
structure at 128, then increasing speckle through 384. This localizes the live
failure more narrowly than the aggregate loss: sampled suffixes break
cross-frequency phase/history consistency over the 514-step causal horizon
without a simple amplitude explosion. It rules out phase learnability or total
power *alone*. It does not fully close amplitude-coordinate normalization,
because the completed head used coarse radial/RGB scaling followed by an
unfitted log coordinate; section 18 records that remaining confound.

### A real 64-step Hartley AR also fails

`train_hartley_ar.py` tests whether the remaining failure is specifically native
complex geometry or the 514-step scalar-frequency rollout. It uses the same real,
full-image orthonormal Hartley transform that succeeds in the bidirectional
control, but generates 64 contiguous 4x4 frequency tiles causally. The 106.7M
model includes QKNorm, fp32 2-D RoPE, learned target slots, per-block position
FiLM, and direct target-slot conditioning of its flow decoder.

| step | held-out clean | shuffled | gap | samples |
|---:|---:|---:|---:|---|
| 2,500 | 0.4707 | 0.5398 | 0.0691 | mottled speckle |
| 5,000 | 0.4605 | 0.5478 | 0.0873 | no objects |
| 7,500 | 0.4977 | 0.5984 | 0.1007 | no objects |
| 10,000 | 0.4266 | 0.5440 | 0.1173 | no objects |

The run is stable, the held-out shuffle gap grows, and every scheduled checkpoint
is saved. Yet the fixed-seed grids remain qualitatively stationary from 2.5k to
10k. This shows that an entirely real Fourier-family basis and a 64-step grouped
horizon did not yield coherent rollout at this budget. The comparison with the
joint Hartley run is **not compute matched**: AR saw about `0.32M` image exposures
(`10k x 32`) versus `7.68M` (`30k x 256`) for joint, alongside different learning
rate and solver schedules. It therefore does not localize the cause to the
interaction of causality and global support; it is an early-budget negative that
needs a matched exposure/schedule extension before supporting that claim.

### Compression and local decoder robustness are not enough

The first Phase-D bridge uses the existing spatial AE architecture with a true
512-scalar bottleneck (8 channels at 8x8, 6x compression), 10% latent-noise
training, and channel moment regularization. At 10k its held-out reconstruction
MSE/PSNR is `0.000812 / 30.91 dB`; perturbing latents by 10% RMS raises MSE only
to `0.000950`. The stored reconstructions are visually faithful.

Generation still fails in two matched forms. A 106.6M, 16-step AR over low-to-high
2x2 Hartley tiles of the latent map reaches held-out clean/shuffled/gap
`1.340 / 1.646 / 0.306`, yet every grid is smooth texture. A 115.4M bidirectional
joint flow over the same 16x32 tokens ends at loss `1.286` and is also object-free.
At the time this separated decoder smoothness from the tested joint generator and
ruled out AR rollout as the sole failure. It did **not**, as later claimed,
establish that the aggregate was intrinsically unmodelable: the local-token
controls below use the same endpoint and succeed.

The latent audit (`diagnose_spatial_ae_latents.py`, 4,096 images) finds normalized
scalar p50/p90/p99/p99.9/max `0.655/1.637/2.691/3.596/6.330`, skew `0.021`, and
excess kurtosis `0.315`. Coordinate standard deviations cluster near one and
Hartley tile RMS spans only `2.22x`. Thus the new failure is not heavy tails or a
large frequency scale mixture. As with the original near-lossless AE, image
validity occupies subtle joint structure inside an innocuous Gaussian-like
marginal; Gaussian-like off-manifold points decode to texture.

MSE-only VAE regularization provides a useful negative bracket. Beta=`1e-3`
keeps 30.27 dB reconstruction and improves off-diagonal correlation RMS from
`0.321` to `0.045` (condition `9.12` to `2.23`), but direct prior samples remain
texture and a joint flow on sampled posteriors remains unrecognizable through
10k (final loss `1.461`). Beta=`1e-2` lowers reconstruction to 25.07 dB and causes
partial posterior collapse (mean covariance condition about `2335`). Adding 0.5
free bits repairs the condition to `2.76` and yields 25.29 dB, but prior samples
still fail. More KL is therefore not the missing ingredient.

### Posterior sampling and the linear flow path are not sufficient explanations

The first beta=`1e-3` generator used sampled posterior latents. That distinction
is material, so its failure did not initially close generation from the cleaner
posterior mean. `diagnose_spatial_vae_posterior.py` measures both policies on the
same 4,096 images. The posterior mean has RMS `0.898` and excess kurtosis `0.971`;
sampling adds RMS `0.422` noise, contributes `18.1%` of aggregate variance, and
reduces excess kurtosis to `0.409`. It also lowers reconstruction from `29.89` to
`28.67` dB. Posterior sampling therefore made the training target measurably more
Gaussian and less reconstructive; it was not a harmless implementation detail.

Three matched joint 10k controls now separate that policy from flow-path geometry:

| latent endpoint | path | final loss | decoded result |
|---|---|---:|---|
| beta=`1e-3` posterior sample | linear | `1.461` | blurred texture/scenes, no objects |
| beta=`1e-3` posterior mean | linear | `1.458` | same visual basin |
| beta=`1e-3` posterior mean | trigonometric VP | `2.219` | same visual basin |
| deterministic noise-trained AE | trigonometric VP | `1.978` | different, sharper texture; no objects |

The loss values across linear and trigonometric paths are not comparable because
the velocity target changes. The decoded grids are the gate: posterior-mean
training does not repair the VAE, and the variance-preserving trigonometric path
does not repair either learned representation. The trigonometric interpolant is
still a useful control: for independent unit-Gaussian endpoints it keeps every
intermediate marginal at unit variance, whereas the independent linear bridge
pinches variance to one half at its midpoint. Its failure here says that this
bridge pathology is not the sole source of the texture basin.

Artifacts: `diagnostics/spatial_vae_kl1e3_posterior.json`,
`continuous_runs/joint_spatial_vae_kl1e3_mean_linear_10k/`,
`continuous_runs/joint_spatial_vae_kl1e3_mean_trig_vp_10k/`, and
`continuous_runs/joint_spatial_ae_hartley_trig_vp_10k/`.

### Local tokenization overturns the aggregate-modelability inference

The failed Phase-D joint arms all applied a full 2-D Hartley transform to the
latent map before forming tokens. That made every token globally supported. The
following controls hold the endpoint latent map fixed and change only the basis
and token support:

| endpoint | tokenization | final loss | 10k visual result |
|---|---|---:|---|
| old MSE C8 | full-map Hartley | `1.286` | rough texture/pseudo-scenes |
| old MSE C8 | local raster | `1.187` | recognizable objects |
| perceptual C8 | full-map Hartley | `1.304` | rough, much weaker than local |
| perceptual C8 | local raster | `1.223` | recognizable objects |
| perceptual C4 | full-map Hartley | `1.306` | rough, much weaker than local |
| perceptual C4 | local raster | `1.220` | recognizable objects |
| perceptual C4 | local 2x2 DCT | `1.208` | recognizable objects |

All joint arms use 16 tokens and the same 115.4M `PatchDiffusion`; the relevant
C4 comparison also fixes seed, schedule, linear flow, channel statistics, and
latent endpoint exactly. Local raster and local DCT are related by an orthonormal
rotation inside each token, so white Gaussian noise, the flow bridge, and squared
error are preserved. This directly separates "frequency values" from "global
support across tokens." Both local representations pass from fixed and fresh
seeds.

Decoder priors do not explain the result. Standardized-Gaussian maps decoded by
the old MSE C8 and perceptual C4 codecs yield texture/pseudo-scenes, substantially
worse than the learned local samples. The flows learned image structure.

The codec objective is also not the repair: the old deterministic MSE C8 endpoint
succeeds locally. LPIPS/Charbonnier enabled the C4 codec to retain class and pose
at 256 scalars (12x compression, `26.71` dB), which makes it an attractive
baseline, but perceptual training is not necessary for modelability.

The 16-step causal C4 local-DCT arm provides the matching AR test. It produces
recognizable objects from the first stored checkpoint and retains them through
training, unlike the global-Hartley latent AR. At 10k it has train loss `1.146`
and held-out clean/shuffled/gap `1.338 / 1.819 / 0.481`. A fresh seed is also
clearly object-like. This rules out the joint model's bidirectional attention as
the explanation for the local result.

The exact C4 global AR comparison adds an important scope correction. Raw global
Hartley is not completely object-free with this codec: fixed and fresh grids have
rough recognizable structure, but remain far below local DCT. Raw global ends at
train/clean/shuffled/gap `1.237 / 1.441 / 1.752 / 0.310`. A spatialized-prefix
variant zero-fills the unknown global suffix, inverts the known prefix, and gives
16 local patches to an 82.9M noncausal spatial trunk at every causal step. It ends
at `1.237 / 1.493 / 1.659 / 0.166` and does not improve samples. Fixed and fresh
pixel correlations between the two global arms are `0.979` and `0.976`.

This falsifies the simplest version of the trunk-interface hypothesis. Local
tokenization changes not only the state the trunk reads but the support of the
quantity independently emitted by the diffusion head. The latter now looks
load-bearing: local errors remain local, whereas global errors perturb the whole
map and must compose consistently across steps.

The output-composition bracket makes that statement more precise:

| target factorization | sequence | train / clean / shuffled / gap | visual result |
|---|---:|---:|---|
| local 2x2 DCT, patch-major | 16 x 16 | `1.146 / 1.338 / 1.819 / 0.481` | recognizable, strongest AR arm |
| local 2x2 DCT, frequency-major | 16 x 16 | `1.195 / 1.407 / 1.738 / 0.332` | recognizable, somewhat softer |
| global Hartley, one coefficient | 64 x 4 | `1.180 / 1.364 / 1.684 / 0.320` | rough; no repair |
| global Hartley, one 2x2 tile | 16 x 16 | `1.237 / 1.441 / 1.752 / 0.310` | rough but partly recognizable |
| global Hartley, four tiles jointly | 4 x 64 | `1.308 / 1.553 / 1.710 / 0.157` | worse than one tile |
| global Hartley, all tiles jointly | 1 x 256 | `1.464 / 1.428 / 1.428 / 0` | rough |
| local spatial map, all values jointly | 1 x 256 | `1.459 / 1.500 / 1.500 / 0` | rough |

The zero shuffle gaps in the final two rows are by construction: a one-token
model has no history to shuffle. The two one-token targets are related by a fixed
orthonormal Hartley rotation and both fail in the same broad way. They show that
the three-layer diffusion MLP is not a sufficient unconditional 256-D generator;
they do not show that a fully joint Fourier model is impossible.

Frequency-major local DCT is the more consequential positive. It emits all local
DC groups before the three higher 2x2 subbands while preserving local spatial
support. Fixed and fresh grids remain semantic, although they are softer than the
patch-major arm. Thus coarse-to-fine frequency order itself is viable. Removing
within-token frequency mixing is not sufficient in the global arm: the 64-step
scalar Hartley model has a lower clean loss than the passing frequency-major arm
but still looks worse. Loss and shuffle gap again do not select sample quality.

Artifacts:
`continuous_runs/joint_spatial_ae_mse_c8_raster_linear_s7_10k/`,
`continuous_runs/joint_spatial_ae_perceptual_c8_raster_linear_s7_10k/`,
`continuous_runs/joint_spatial_ae_perceptual_c4_raster_linear_s7_10k/`,
`continuous_runs/joint_spatial_ae_perceptual_c4_patchdct_linear_s7_10k/`, and
`continuous_runs/ar_spatial_ae_perceptual_c4_patchdct_raster_s7_10k/`; fresh AR
artifact `diagnostics/ar_spatial_ae_perceptual_c4_patchdct_fresh_54321.png`.
Global comparison artifacts:
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_radial_s7_10k/`,
`continuous_runs/ar_spatialized_prefix_hartley_perceptual_c4_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_band4_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_scalar_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_all16_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_spatial_all1_s7_10k/`,
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_fresh_54321.png`, and
`diagnostics/ar_spatialized_prefix_hartley_perceptual_c4_fresh_54321.png`.
The frequency-major local run and fresh sample are
`continuous_runs/ar_spatial_ae_perceptual_c4_patchdct_freqmajor_s7_10k/` and
`diagnostics/ar_spatial_ae_perceptual_c4_patchdct_freqmajor_fresh_54321.png`.
Fresh grouping endpoints are
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_band4_fresh_54321.png`,
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_scalar_fresh_54321.png`,
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_all16_fresh_54321.png`, and
`diagnostics/ar_spatial_ae_perceptual_c4_spatial_all1_fresh_54321.png`.

### Established intrinsic phase diffusion does not repair the global FFT arm

`ar_fft_factorized_wrapped_normal_score_10k` replaces shortest-geodesic phase
flow with the wrapped-normal Brownian score construction used by Torsional
Diffusion: exact finite-image wrapped-normal scores, denoising score matching,
and probability-flow ODE sampling on the circle. It ends at **training-batch**
clean/shuffled/gap `1.354 / 2.193 / 0.839`; its fixed grids remain speckle through
10k. Thus phase geometry was treated intrinsically, but the old metric does not
establish held-out context use. The visual failure persists, while the audit's
loss-geometry correction prevents this arm from closing phase modeling broadly.

## Evidence

### 1. The decoder is healthy and tolerant of latent error

`diagnose_latent_robustness.py`. Gaussian noise of scale sigma added to real
normalized latents (injected per-token MSE = sigma^2), PSNR vs the clean
reconstruction:

| sigma | latent MSE | PSNR | appearance |
|---|---:|---:|---|
| 0.10 | 0.010 | 36.50 | indistinguishable |
| 0.20 | 0.040 | 30.64 | slightly soft |
| 0.35 | 0.122 | 25.86 | degraded, still clearly recognizable |
| 0.50 | 0.250 | 22.74 | rough, recognizable |
| 1.00 | 1.000 | 17.21 | **texture mush -- matches our samples** |

The AR model's conditional latent MSE is 0.128, i.e. sigma ~= 0.36, which decodes
to *recognizable* images. So decoder hypersensitivity is ruled out; but our
samples look like sigma ~= 1.0, meaning sampled latents carry far less usable
information than the training loss implies.

### 2. Perceptual importance is concentrated ~1000:1 in the first few tokens

Per-position sensitivity at sigma=0.5 (PSNR vs clean reconstruction, lower = more
damage): position 1 is 28.4 dB, position 0 is 31.4 dB, falling monotonically to
~59 dB by position 44. Position 1 alone is ~1300x more perceptually damaging than
position 44.

Corrupting only positions 0-3 (27.42 dB at sigma=0.35) does **more** damage than
corrupting the entire 49-token remainder (30.86 dB). The joint trainer weights all
53 positions equally (`per_position.mean()`).

### 3. The generator matches the latent distribution's measurable structure

`diagnose_latent_structure.py`. Because normalized latents have unit per-dimension
variance, an N(0, I) model attains a known rectified-flow velocity MSE of
2 - pi/2 = 1.5708; any gain must come from correlation. Bayes-optimal MSE for
Gaussian data with the real covariance:

| model | RF velocity MSE |
|---|---:|
| no structure, N(0, I) | 1.5708 |
| within-token only (block-diagonal Sigma) | 1.1255 |
| all second order (full Sigma) | 1.0021 |
| **achieved by joint model** | **0.9083** |

The model is *past* the full-covariance optimum, and its samples reproduce real
correlation structure closely -- within-token RMS correlation 0.2133 vs 0.2123,
cross-token 0.0474 vs 0.0516. Cross-token *magnitude* (energy) coupling is also
reproduced: 0.375 vs 0.399 (independently shuffled latents give 0.001).

Per-position marginals match too (RMS ~1.0, kurtosis within a few percent).
So the failure is not distributional in any low-order sense.

### 4. But it is barely past Gaussian, and Gaussian decodes to mush

`diagnose_position_floor.py`. Compared per position against the Bayes-optimal
*linear* Gaussian predictor, the model beats it at all 53 positions -- but only by
0.05 to 0.17, and the margin is *largest* at the prefix (position 1: 0.635 vs
0.803). The model is roughly 10% of the way past second order overall, and about
21% of the way at position 1.

This does *not* by itself rule out loss reweighting: capturing 21% of the
available structure at the prefix while needing most of it is consistent with the
prefix simply needing far more capacity or signal. What it does show is that the
prefix is not being *ignored* -- it is the best-modeled part of the sequence in
relative terms, and still nowhere near good enough. Treat the reweighting
evidence as ambiguous rather than negative.

`diagnose_gaussian_sample.py`. Sampling an exact Gaussian fit N(mu, Sigma) of the
real latents and decoding produces texture mush closely resembling our generative
samples (`diagnostics/gaussian_sample/gaussian_vs_prior.png`, row 2 vs row 3).
Combined with the above: the samples are mush because the model never got
meaningfully past Gaussian, and all perceptual structure lives in the
non-Gaussian part it has barely touched.

### 5. The break is specifically in the low-frequency prefix, in both models

`diagnose_prefix_graft.py`. Grafting real and generated latents across a split,
PSNR against the source real image:

| split | real prefix + generated suffix | generated prefix + real suffix |
|---:|---:|---:|
| 4 | 17.00 | 10.20 |
| 8 | 20.36 | 9.74 |
| 16 | 24.03 | 9.53 |
| 24 | **27.49** | **9.45** |

Generated high-frequency suffixes are perfectly usable -- replacing 29 of 53
tokens with generated ones still gives 27.5 dB and recognizable images. Generated
low-frequency prefixes are catastrophic even when 29 real tokens are supplied.
The generated prefix is off-manifold.

`diagnose_cross_token.py` corroborates from the other side: independently
shuffling real latents from position 16 onward leaves images recognizable, while
shuffling from position 8 destroys them.

### 6. Not overfitting -- measured on a held-out split

`diagnose_overfitting.py`. 30k steps at batch 256 is ~154 epochs of 50k images,
so every train-loss comparison above needed validating. Identical timestep grid
and identical noise realizations for both splits (paired comparison), 10000
images each, no augmentation:

| quantity | train | test | gap |
|---|---:|---:|---:|
| AE reconstruction PSNR | 35.12 dB | 35.09 dB | +0.03 dB |
| joint model velocity MSE | 0.9050 | 0.9098 | +0.0048 |
| linear Gaussian (fit on train) | 1.0013 | 1.0158 | +0.0144 |
| model advantage over linear | +0.0963 | +0.1059 | -- |

Neither the AE nor the generative model memorizes; the largest per-position gap
is 0.0096. The model's edge over the Gaussian baseline is slightly *larger* on
held-out data, so the "~10% past Gaussian" finding holds out of sample.

That an ~85M-parameter model shows a 0.5% train/test gap after ~154 epochs is
itself evidence of **underfitting**: the objective is so dominated by
incompressible high-frequency content that the model cannot memorize it.

### 7. The latent is NOT incompressible -- correcting an earlier claim

`diagnose_compressibility.py`, 50000 train images. Linear R^2 predicting each
token from all 3328 *other* dimensions (in-sample optimism ~0.031, subtract it):

| | mean | prefix[0:16] | suffix[16:] |
|---|---:|---:|---:|
| token values | 0.538 | 0.530 | 0.541 |
| token log-energy | 0.562 | 0.558 | 0.564 |

About 54% of every token's variance is linearly predictable from the other
tokens. An earlier draft of this document called the high-frequency content
"incompressible"; that was wrong, and it contradicted the cross-token energy
coupling of 0.399 measured in section 3. There is substantial cross-frequency
redundancy.

Sanity check that the measurement tracks something real: position 0 (DC/lowest
ring) has value R^2 0.852 but energy R^2 0.051 -- overall image brightness is
independent of everything else, while its detailed pattern given that energy is
highly predictable.

**Why the conclusion still holds.** The linear/Gaussian baseline exploits all of
this redundancy by construction, and the model beats that baseline. So the 54% is
already captured; the missing ingredient is specifically the higher-order
structure where coherence lives. What does *not* depend on the compressibility
question is section 2: perceptual sensitivity is concentrated ~1000:1 in the first
four tokens while the loss is a uniform mean over 53, giving them ~7.5% of the
gradient.

**What this changes.** The route-B compressive-AE argument partly rested on the
incompressibility claim. It survives only reframed -- a lossy AE would help by
removing perceptually *irrelevant* dimensions from the objective, not unpredictable
ones -- which is a weaker case. This measurement shifts the recommendation toward
route A (cascade).

### 8. Latent capacity is allocated proportional to content, not to perception

`diagnose_ring_allocation.py`. The layout rule in `GroupLayout` is
`latents_per_ring = min(8, max(1, ceil(ring_size / 12)))`, so latents per ring
grow with the number of Fourier coefficients in the ring.

An earlier draft of this section called the allocation "inverted" or "backwards".
That was wrong. Latents per ring (1,1,1,1,1,2,2,2,3,3,3,3,4,4,4,4,4,3,2,2,1,1,1)
track coefficients per ring (1,4,8,10,12,20,18,24,28,28,34,32,40,46,44,48,38,30,
18,16,8,6,1) closely, which is exactly the ~1:1 ratio measured below. The
allocation is **proportional to information content**; it is merely *orthogonal to
perceptual value*. That is a narrower claim, and it is the one the data supports.

Verified exactly: 514 orbits carrying 3 or 6 real components each = 3072 real
values, against 3392 latent dims. Per-ring compression (real values per latent
dim) is 0.84-1.13 for rings 3-20 -- **~1:1 at every radius**. The AE is a
near-identity re-coding by construction; the rule targets a constant
coefficients-per-latent ratio and therefore never compresses differentially.

| | rings 0-4 | rings 12-22 |
|---|---:|---:|
| latents per ring | 1, 1, 1, 1, 1 | 4, 4, 4, 4, 4, ... |
| share of latent dims | 9.4% | 56.6% |
| share of perceptual damage | **77.1%** | **3.0%** |

**The fix is not more capacity for low rings.** Rings 0-4 hold only 207 of 3072
real values (6.7%) while carrying 77% of the perceptual damage, and already
receive 320 latent dims for those 207 values -- a 1.55x surplus. They are not
capacity-starved.

The waste is at the top end: rings 12-22 spend 1920 latent dims (57%) on 3% of
perceptual value. High rings should go *down* from 4 latents toward 1, or keep one
latent per ring with a width that scales with importance (low rings 64-dim, high
rings 8-16-dim, instead of a uniform 64). That would take the latent from 3392 to
roughly 800-1000 dims at little perceptual cost -- real compression, concentrated
where the content is perceptually irrelevant.

### 9. Spectral-amplitude loss weighting is a good but imperfect proxy

Same script. Against directly measured perceptual damage (corrupt one latent,
measure decoded pixel MSE), population spectral amplitude gives Spearman 0.854 and
log-Pearson 0.668. Since the codec whitens by dividing out the population standard
deviation, weighting whitened squared error by population variance restores an
approximately pixel-space L2 objective. This also explains the shape of the
existing `decoder_sensitivity` weights, which this measurement independently
reproduces.

Two defects if used directly as a loss:

* **Position 52** (ring 22, a single self-conjugate Nyquist orbit) gets spectral
  weight 1.47 -- about average -- against measured damage of 1.6e-5, an error of
  10^4.95. Ring 16 shows the same anomaly.
* **Systematic mid-ring bias**: positions 19 and 27-38 receive spectral weight
  ~1.2 log units *below* their measured damage, underweighted ~16x.

Preferred alternative: use the measured damage weights directly (one decode per
position, low-noise, end-to-end), saved as the `measured_damage` key in
`diagnostics/ring_allocation/spectral_weights.pt`. Dynamic range is 8.6e5, so
temper it (`w**alpha`, alpha ~ 0.5, plus a floor) or training collapses onto ~6
tokens.

### 10. Normalization, tail geometry, and positional structure

`diagnose_normalization.py`, 20000 train images.

**How the raw FFT is normalized.** Codec config is `normalization=orbit_standardize`,
`centering=all` (so effective `mean_policy=per_orbit`, `scale_policy=centered_std`),
`fft_norm=ortho`, `ordering=radial`, `value_transform=identity`. Each of the ~3084
(orbit, component) slots therefore gets its own dataset mean and standard
deviation. The asinh compression in `frequency.py` is implemented but **disabled**.

**FFT tokens are heavy-tailed, increasingly so with radius.**

| radius | kurtosis | P(abs z > 4) | max abs z |
|---:|---:|---:|---:|
| 0 | 3.16 | 3.3e-05 | 4.0 |
| 5 | 4.54 | 1.7e-03 | 9.8 |
| 13 | 5.70 | 2.2e-03 | 28.2 |
| 19 | 6.43 | 3.8e-03 | 11.9 |
| 22 | 8.45 | 5.4e-03 | 10.5 |

Overall kurtosis 5.16 against a Gaussian 3; P(abs z > 4) is ~36x the Gaussian rate.
Per-orbit standardization fully solves the *scale* imbalance across frequencies
(every frequency gets its own std); what remains is *within*-frequency shape, i.e.
sparsity, which is what the disabled asinh transform would target, and it is worst
exactly where asinh would bite hardest.

**A single global latent mean/std: mechanically defensible, but it does not buy
natural weighting.** Raw pre-normalization latents:

* per-position std spans 0.2918-0.5674 -- ratio only **1.94x**
* per-dim std spans 0.0629-1.1817 -- ratio 18.79x
* latent kurtosis **3.27**, essentially Gaussian (unlike the FFT tokens)
* global std 0.9600, which exceeds every per-position std, so large per-position
  *mean* offsets dominate and would still need removing

Correlation of natural per-position latent scale against measured perceptual
importance: **log-Pearson -0.38, Spearman -0.46 -- negative.** The AE equalized the
spectral hierarchy out of its latent scale (to within 2x) and mildly inverted it,
so global scalar normalization would give slightly *worse*-than-uniform implicit
weighting. Perceptual weighting has to be explicit (the J^T J pull-back of pixel
MSE), or the AE has to change so its latent scale carries that information.

**No positional embedding anywhere.** `causal_transformer.py` contains no RoPE, no
learned position table and no ALiBi (`max_seq_len` is used only for masking). Both
generative models use a *single* shared `input_projection` and a single shared
output head across all 53 positions. For the bidirectional joint model this means
attention is permutation-equivariant and the only signal distinguishing the 53
positions is the 11 hand-designed metadata floats passed through that shared linear
map. Not degenerate -- the metadata includes `sequence_index` and `ring_index` --
but the model must re-derive position identity from a very thin channel, in a
problem defined by absolute position rather than translation invariance. A learned
per-position embedding costs 53 x 768 ~ 40k params; per-position input/output
projections ~6M. The causal AR model is less affected, since its mask already
breaks the permutation symmetry.

### 11. The model is Gaussian-equivalent exactly where structure is decided

`diagnose_snr.py`, held-out test latents, velocity MSE resolved by timestep
instead of averaged. Convention: `noisy = t*latents + (1-t)*noise`, so **t=0 is
pure noise and t=1 is clean**; low t is early in sampling.

| t | nominal SNR | no-structure | Gaussian floor | model | model - floor | excess as % of available gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.003 | 1.1050 | 1.0769 | 1.0794 | **+0.0025 (worse)** | 0% |
| 0.14 | 0.027 | 1.3172 | 1.1427 | 1.1330 | -0.0097 | 5.6% |
| 0.50 | 1.000 | 2.0000 | 1.0030 | 0.8801 | -0.1229 | 12.3% |
| 0.86 | 37.7 | 1.3172 | 0.8496 | 0.7121 | -0.1375 | **29.4%** |

The model's ability to exceed second-order structure grows monotonically as noise
falls, and is zero-to-negative at high noise -- at t=0.05 it is *worse* than a
linear Gaussian predictor.

**This is the mechanism behind the whole failure.** Diffusion sampling commits
global structure in the earliest, highest-noise steps. In that regime this model is
Gaussian-equivalent, so the structural commitment is effectively a draw from the
Gaussian fit -- which section 4 shows decodes to mush. The later low-noise steps,
where the model does beat Gaussian, then faithfully render detail onto broken
structure. That is exactly why generated suffixes graft at 27.5 dB while generated
prefixes give 9.45 dB (section 5).

Nominal SNR is `t^2/(1-t)^2`, **identical at every position by construction**, since
per-position standardization forces unit variance everywhere. The schedule spends
equal resolution effort on all 53 positions while perceptual need is ~1000:1
concentrated.

Important nuance, not a counterargument: absolute MSE stakes at high noise are
small -- available gain is 0.028 at t=0.05 versus 0.997 at t=0.50. A uniform-in-t
objective therefore correctly judges high noise to be nearly worthless *in MSE
terms*, and that is the trap: those 0.028 carry the entire structural decision,
because at high noise the velocity field selects which basin the trajectory enters.
Training samples t uniformly (`randint(0, num_train_timesteps)`) and the sampler
steps uniformly, so both allocate ~15% of their effort to the region that decides
everything, weighted by a metric that says it does not matter. Same reasoning as
SD3/Flux logit-normal timestep weighting, pointing here toward high noise.

### 12. The codec whitening destroys the spectral hierarchy (supersedes section 12b)

`diagnose_hierarchy_chain.py`. Variance spread measured in **both** the coordinate
and eigen bases for every stage, using 5th-vs-95th-percentile ratios (min/max
eigenvalue ratios are dominated by numerically-tiny directions and are not
meaningful). t* = 1/(1+sqrt(lambda)) is the time a direction crosses SNR=1.

| space | coord var ratio | eigen var ratio | PR | coord t* spread | eigen t* spread |
|---|---:|---:|---:|---:|---:|
| pixels | 1.58 | 1.996e4 | 9.2 | 0.042 | 0.704 |
| FFT raw | 762 | 1.995e4 | 5.3 | 0.720 | 0.720 |
| **FFT whitened (AE input)** | 1.18 | **5.12e2** | **457.4** | 0.019 | **0.368** |
| latents raw | 3.52 | 8.10e3 | 260.1 | 0.132 | 0.586 |
| latents normalized | 1.07 | 7.07e3 | 284.5 | 0.008 | 0.508 |

**Pixels and raw FFT have identical eigenspectra** (1.996e4 vs 1.995e4), as they
must, since the orthonormal FFT is a rotation. The two bases behave very
differently though: pixels have coord spread 0.042 against eigen spread 0.704,
while raw FFT has the two *equal* at 0.720. That is empirical confirmation that
**the FFT basis is approximately the eigenbasis of natural images**, and therefore
that per-frequency whitening *is* eigenvalue whitening: eigen ratio 2e4 -> 5.1e2,
PR 5.3 -> 457, eigen t* spread 0.720 -> 0.368.

This resolves the apparent tension between "networks need O(1) inputs" and
"preserve the hierarchy": per-coordinate whitening was never required for O(1)
values. Pixels are already O(1) per coordinate (ratio 1.58) while keeping the full
2e4 eigenspectrum. Per-pixel normalization is harmless; per-*frequency*
normalization is maximally destructive. The basis is what makes the difference.

**The autoencoder is exonerated -- it partially restores hierarchy**, taking eigen
ratio 5.12e2 -> 8.10e3 and t* spread 0.368 -> 0.586 by introducing correlations its
whitened input lacked. Latent per-dim normalization is then nearly free
(8.10e3 -> 7.07e3). Both "mechanisms" track the eigenspectrum, which is set at the
whitening step; measuring one in the coordinate basis and the other in the
eigenbasis (as section 12b did) confounded them.

### 12b. Superseded: whitening costs implicit weighting, not conditioning order

`diagnose_snr_staggering.py`. Under isotropic noise an eigendirection with
variance lambda crosses SNR=1 at `t* = 1/(1+sqrt(lambda))`. A wide spread of t*
means the representation resolves coarse structure early and detail late by
itself -- the natural schedule that makes pixel-space image diffusion work.

| space | eigenvalue range | diagonal range | participation ratio | t* spread (p05->p95) |
|---|---:|---:|---:|---:|
| raw FFT, pre-whitening | 8.06e7 | 9.31e5 | 5.3 | 0.720 (0.122->0.842) |
| raw latents, pre-normalization | 8.39e9 | 353 | 260.1 | 0.586 (0.173->0.759) |
| normalized latents (current) | 9.59e9 | 1.18 | 284.5 | 0.508 (0.096->0.604) |

Two mechanisms are often conflated here, and they behave differently:

* **Implicit loss weighting via magnitude -- destroyed.** Covariance diagonal range
  collapses 353 -> 1.18 under per-dim normalization. Higher-variance dimensions no
  longer claim a larger share of the gradient.
* **Staggered SNR schedules / conditioning order -- mostly survives.** Eigenvalue
  dynamic range stays ~1e10 and t* spread only drops 0.586 -> 0.508. Normalization
  sets the covariance *diagonal* to 1, but staggering is governed by *eigenvalues*,
  which it barely touches.

So de-normalizing (or partial whitening, Sigma^-alpha) would mainly restore
mechanism 1 -- and per section 10 the per-position component of natural scale is
anti-correlated with measured perceptual importance (Spearman -0.46), so that
restored weighting points the wrong way.

**RETRACTED.** This section claimed the AE compressed the natural hierarchy ~2600x.
That was wrong in both stage and sign: the AE's input is already whitened, so it
never saw the hierarchy, and it in fact partially *restores* one. It also compared
mechanism 1 in the coordinate basis against mechanism 2 in the eigenbasis, which
confounded two views of the same phenomenon. Section 12 supersedes it.

### 13. Results: cheap conditioning, schedule and whitening changes all failed

Four arms trained to 30k steps against the same AE and config, plus an AE
whitening sweep. All re-scored on a **common** protocol (held-out, uniform 32-point
t grid, paired noise, 4096 images) -- `train/flow_mse` is *not* comparable across
arms that change the training t distribution.

| arm | test MSE | vs linear floor | delta vs baseline |
|---|---:|---:|---:|
| baseline | 0.9107 | +0.1076 | -- |
| position embedding (input + per-block FiLM) | 0.9094 | +0.1088 | -0.0013 |
| timestep shift (t = u^2) | 0.9144 | +0.1039 | +0.0037 |
| both | 0.9134 | +0.1048 | +0.0027 |
| alpha=0.5 latents + position embedding | 0.9106 | +0.1074 | -0.0001 |

(The alpha=0.5 row is scored against its own AE's linear floor, since its latents
differ; the comparison is self-consistent.)

**RoPE, added later, is the first change that consistently helps.** An earlier
draft of this section dismissed RoPE as "the same category of fix that just failed
four times". That was wrong -- it had never been tested, and absolute position
embeddings are a different mechanism: they add to the residual stream for
identifiability, whereas RoPE rotates q/k to shape attention geometry.

| arm | test MSE | vs linear | delta vs base | perc. weighted delta | prefix 0-3 delta |
|---|---:|---:|---:|---:|---:|
| baseline | 0.9107 | +0.1076 | -- | -- | -- |
| absolute position embedding only | 0.9094 | +0.1088 | -0.0012 | -0.0015 | -0.0014 |
| rope (sequence index) | 0.9059 | +0.1123 | -0.0047 | -0.0030 | -0.0021 |
| rope (radius, angle) | 0.9056 | +0.1127 | -0.0051 | -0.0038 | -0.0028 |
| **rope (radius, angle) + absolute** | **0.9042** | **+0.1141** | **-0.0065** | **-0.0054** | **-0.0044** |

RoPE alone helps ~4x more than absolute embeddings alone, and the two are
**additive** (-0.0051 + -0.0012 = -0.0063 against -0.0065 measured), consistent
with them serving distinct roles. 2-D (radius, angle) edges out 1-D sequence index,
in the predicted direction but barely. No arm overfits (train/test gaps +0.0017).

Calibration: advantage over linear moves 0.1076 -> 0.1141, i.e. ~10.6% -> ~11.2%
past Gaussian. Real and replicated across three arms, but small against the gap
that needs closing.

Position conditioning gives a 0.14% improvement -- noise level. The timestep shift
is genuinely *worse* on uniform-t evaluation, not merely re-scaled: it spends
gradient where available gain is small. Previews for the position arm and for the
AR `--transformer_metadata_film` arm remain texture-like and unchanged in
character. The model sits at ~+0.11 past linear in every case, i.e. the ~10% past
Gaussian plateau of section 4 is untouched.

**Whitening sweep: alpha does not reach the latents.**

| alpha | AE *input* eigen ratio | normalized *latent* eigen ratio | latent t* spread | AE test PSNR |
|---:|---:|---:|---:|---:|
| 1.0 | 5.12e2 | 7.07e3 | 0.508 | 34.95 dB |
| 0.5 | 1.97e3 | 1.12e4 | 0.498 | 35.08 dB |
| 0.0 | 1.995e4 | 1.54e4 | 0.489 | 33.05 dB |

Implementation verified: at alpha=0 the AE input reproduces raw FFT exactly (coord
ratio 7.627e2, eigen ratio 1.995e4). The input hierarchy swings **39x** across the
sweep while the latent hierarchy the generative model sees moves only **2.2x**, with
t* spread flat. The autoencoder absorbs the change -- expected, since its loss is
pixel-space (weight 1.0 against token weight 0.01) and its internal RMSNorms wash
out input scale.

(t* spread is not comparable across differently-*scaled* spaces, since t* depends
on absolute variance and the global divisor shifts it. The scale-invariant eigen
ratio is the measure to read.)

**Consequences.** For the latent variants the whitening lever is near-inert, and
alpha=0 pays 1.9 dB for it; alpha=0.5 is a mild free win (better PSNR, 1.6x more
hierarchy) but should not be expected to fix coherence. For the direct-FFT variants
alpha is decisive, since that 39x swing *is* the modelled representation.

Taken together these eliminate a class of cheap explanations by measurement: the
plateau is not caused by missing positional conditioning, not by the timestep
distribution, and not by the codec whitening *on the latent path*. Changing the
latent geometry requires changing the AE's objective or architecture, not its
input.

### 14. Loss weighting cannot substitute for the variance hierarchy (retraction)

Earlier sections recommended per-position perceptual loss weighting (the J^T J
diagonal / `decoder_sensitivity`) as a way to recover what whitening removes.
**That recommendation is wrong**, and the reason is structural.

For a Gaussian direction of variance lambda the Bayes-optimal RF velocity residual
is `d_t = (lam+1) - (t*lam-(1-t))^2 / (t^2*lam + (1-t)^2)`. Dividing each curve by
its own mean factors out any constant weight; if weighting could mimic variance the
curves would coincide:

| t | lam=0.01 | lam=1 | lam=100 |
|---:|---:|---:|---:|
| 0.02 | 0.070 | 0.658 | 6.668 |
| 0.26 | 0.122 | 1.027 | 0.913 |
| 0.50 | 0.264 | 1.264 | 0.264 |
| 0.74 | 0.913 | 1.027 | 0.122 |
| 0.98 | 6.668 | 0.658 | 0.070 |

They are not proportional but qualitatively different functions: lam=100 is
monotonically decreasing (crosses SNR=1 at t*=0.09, resolved early), lam=1 is
peaked at t=0.5, lam=0.01 is monotonically increasing. Best-constant-rescale shape
mismatch is 92-189%.

Whitening forces every direction to lam=1 and therefore to the peaked profile. A
constant per-position weight rescales that peak but cannot convert it into a
monotone-decreasing one. This retroactively explains why the `decoder_sensitivity`
AR arm did not help (section: weighting campaign) and is consistent with the
timestep-shift arm failing (section 13).

**The distinction that matters: loss weighting changes gradient allocation; the SNR
profile changes what is inferable when.** Only the second reorders the sampling
trajectory. This also rules out a t-dependent 2-D weight table w(p,t): it could
match the error *shape* while still leaving the forward process, and hence the
resolution order, unchanged. The only levers that genuinely alter the dynamics are
retaining real variance differences, or per-position noise schedules.

Corollary for asinh: MSE presumes physical-space symmetry (predicting 2 against a
target of 3 equals predicting 4 against 3). Under asinh/log it does not -- the
over-prediction is exponentially larger in physical terms -- so asinh in the
target/loss space yields systematically asymmetric physical error. It is defensible
only as a trunk *input* recoding, where no loss is taken.

### 15. Mean subtraction does not meaningfully corrupt phase

`diagnose_phase_centering.py`. Power lives in a complex coefficient's norm and
phase in its angle, so subtractive normalization can rotate phase arbitrarily --
for z = 0.25+0.25i a mean of 0.1 barely moves the angle while a mean of 0.5
inverts it. Whether that matters depends on |mean| relative to spread.

| ring | self-conj | \|mean\|/std | median dphase | p95 dphase | frac > 90 deg |
|---:|---|---:|---:|---:|---:|
| 0 (DC) | True | 1.76 | 180.00 | 180.00 | 0.518 |
| 1 | False | 0.105 | 3.94 | 30.51 | 0.008 |
| 4 | False | 0.026 | 0.82 | 8.09 | 0.0009 |
| 12 | False | 0.0085 | 0.19 | 2.19 | 0.0005 |
| 21 | False | 0.0037 | 0.22 | 1.28 | 0.0000 |

The DC row is a non-issue: it is self-conjugate and purely real, so it has no
complex phase, "180 degrees" is a sign flip, and the mean being removed is image
brightness. For every genuinely complex orbit |mean|/std is 0.003-0.105 and the
induced rotation is a median of 0.2-3.9 degrees with under 1% of coefficients
moving past 90 degrees. Natural images carry no systematic alignment, so non-DC
orbit means are tiny relative to spread and centering is nearly phase-neutral.

Caveat: the worst case is ring 1 (3.94 deg median, 30.5 deg at p95), which is the
*most* perceptually important position (section 2). If that is worth eliminating,
the fix is not a polar representation -- `mean_policy` already offers `self_only`,
which centers only the self-conjugate real orbits and leaves complex ones
untouched, preserving phase exactly.

Related: Cartesian MSE already carries a sensible phase weighting, since
|z - zhat|^2 = r^2 + rhat^2 - 2*r*rhat*cos(theta - thetahat) penalizes phase error
in proportion to r*rhat. Phase is weighted by energy automatically.

### 16. The AR model memorizes; its conditional advantage does not generalize

`diagnose_ar_generalization.py`, on `vae-posterior-mean-weighting/unweighted`.
Paired protocol: identical timesteps and noise across splits and across the
conditional/null settings.

| split | conditional | null | gap |
|---|---:|---:|---:|
| train | 0.5148 | 1.1208 | +0.6060 |
| test | **1.5001** | 1.1184 | **-0.3817** |

On held-out images the AR context makes predictions **worse than no context at
all**. The measurement validates itself: null-context values agree across splits
to 0.2% (1.1208 vs 1.1184), as they must since null context carries no
image-specific information, while only the conditional branch diverges.

This overturns a load-bearing claim in the project handoff -- "diagnostics show AR
context helps (conditional MSE ~0.12 vs null ~0.26-0.30)". Those numbers came from
training batches. Held out, the effect reverses.

It was first noticed indirectly: a *linear* readout from the frozen trunk's hidden
state to its target scores train R^2 = 0.66 against test R^2 = 0.005. With 434k
samples and 769 predictors in-sample optimism should be ~0.002, so that gap cannot
be regression overfitting.

**Two distinct failure modes have been conflated.** The joint model generalizes
(0.5% train/test gap, section 6) but sits ~10% past Gaussian -- underfitting. The
AR model massively overfits. Conditioning on an image's exact prefix is close to a
unique identifier and 154 epochs of 50k images is ample to memorize it, whereas
the joint model only ever sees noised versions of every token.

**Refinement (`diagnose_ar_prefix_recall.py`).** An earlier draft of this section
claimed the memorized map is "queried off-domain and returns garbage", which would
predict that teacher-forcing a real *training* prefix yields strong recall. Tested
directly, the train-over-test advantage is real but modest:

| prefix | TRAIN psnr | TEST psnr | gap | TRAIN suffix latent MSE | TEST suffix latent MSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 9.12 | 9.32 | -0.19 | 2.184 | 2.173 |
| 8 | 22.38 | 21.15 | +1.23 | 1.705 | 1.859 |
| 24 | 29.70 | 28.58 | +1.12 | 1.463 | 1.707 |

Independent unit-variance vectors score 2.0 on suffix latent MSE. Even given a real
24-token *training* prefix the sampled suffix reaches only 1.46, about 27% of
variance explained; unconditionally it is 2.18, worse than independent. So
memorization alone does not explain the samples.

**All three AR weighting arms memorize, and perceptual weighting makes it worse.**

| arm | train cond | train null | test cond | test null | test gap |
|---|---:|---:|---:|---:|---:|
| unweighted | 0.5148 | 1.1208 | 1.5001 | 1.1184 | -0.3817 |
| raw_variance | 0.5045 | 1.1134 | 1.6286 | 1.1117 | -0.5170 |
| decoder_sensitivity | 0.4266 | 0.9297 | 2.1008 | 0.9247 | -1.1761 |

`decoder_sensitivity` overfits ~3x worse than `unweighted`: best train conditional,
worst test conditional. Its weights span a 1e6 ratio and concentrate the loss on
roughly six tokens, leaving far fewer effective constraints to memorize. So the
perceptual weighting recommended earlier in this document actively *increased*
memorization. The whole weighting campaign was scored on a memorized signal and
its cross-arm comparisons should not be cited. (Absolute values are not comparable
across arms because the loss itself is reweighted; only within-arm train-vs-test
is meaningful.)

**The AR conditional advantage is real early and is destroyed by training length.**
`wd=0.1 + brightness augmentation`, held-out, across checkpoints (~5.1 epochs per
1000 steps):

| step | epochs | train cond | test cond | test null | test gap |
|---:|---:|---:|---:|---:|---:|
| 7,500 | 38 | 0.8034 | 0.8567 | 1.1702 | **+0.3135** |
| 15,000 | 77 | 0.6577 | 0.9788 | 1.1385 | +0.1597 |
| 22,500 | 115 | 0.5672 | 1.1712 | 1.1265 | -0.0447 |
| 30,000 | 154 | 0.5084 | 1.3832 | 1.1204 | -0.2628 |

At 7,500 steps the model genuinely generalizes: held-out conditional 0.857 against
null 1.170, a +0.31 advantage, with a train/test gap of only 0.05. The advantage
then decays monotonically and crosses zero near 21,000 steps (~107 epochs).
**Every AR run in this project used 30,000 steps**, i.e. 3-4x past the point where
conditioning stopped helping.

Regularization knobs did not fix it at 30k steps: `wd=0.1` alone was *worse* than
the `wd=0.02` baseline (-0.5757 vs -0.3817) and brightness augmentation only
partially helped (-0.2628). Caveat: the new arms also enable
`--transformer_metadata_film` while the older ones do not, so the weight-decay
comparison is confounded. Training length is the dominant factor either way.

**But early stopping does not fix the samples.** The 7,500-step preview is still
texture mush. Generalizing on denoising-given-side-information is a different
achievement from being able to generate; the underfitting of section 4 is
untouched. What early stopping does provide is a trunk whose contexts are not
memorization-laden, which is a precondition for a clean flow-vs-diffusion
comparison.

**The model is simultaneously overfit and underfit, on two different tasks**, and
that resolves the natural objection "if it overfits, why do samples look nothing
like CIFAR?":

* *Diagnostic / training task*: given the true prefix **and a noised version of the
  true target**, predict the target. At moderate noise x_t already carries much of
  x0. Memorization lets the prefix act as a key identifying the training image, so
  the model snaps to the stored value -- train 0.51 against test 1.50.
* *Sampling task*: given the prefix **only**, produce the target from pure noise.

The first is far easier and memorization inflates it further, so conditional MSE
0.128 never implied generative capability. At the task sampling actually requires
the model is badly underfit. Overfitting a denoising objective does not yield a
generative model, and in a continuous space a memorized conditional map yields no
recoverable trajectory either way.

Caveat on scale: the reported values are the model's training objective
(v-prediction MSE), not the x0 MSE quoted in the handoff, because the returned
dict exposes `loss`. The train/test comparison is unaffected since both use the
same key.

### 17. Latent geometry

40000 training images, full covariance: 847 dimensions carry 90% of the variance,
1744 carry 99%, participation ratio 287. (An earlier 256-sample PCA suggested far
lower dimensionality; that estimate was rank-limited and should be ignored.)
The latent distribution genuinely occupies a high-dimensional space -- consistent
with an autoencoder that compresses nothing.

### 18. Factorized amplitude-coordinate audit

The completed factorized decoder did not population-standardize its log-amplitude
target. With physical amplitude `a_phys` and the fitted per-radius/RGB RMS
`s[r,c]`, it used

`u = log(a_phys / s[r,c] + 1e-4)`.

The corresponding optional trunk feature was different again:
`log1p(a_phys / s[r,c])`, followed by a bounded phase-reliability gate. That
polar feature was disabled in the completed run, which used Cartesian history.

On 10,000 CIFAR-10 training images, RMS-relative amplitude has
p10/median/p90/p99 `0.207/0.615/1.480/3.268`. The existing decoder coordinate has
mean/std `-0.550/0.806`; no tensor-wide or per-RGB affine was fitted after the
log. RGB means and standard deviations differ by less than `0.016/0.004`, so a
single population affine or RGB statistics shared across all frequencies are
both plausible first controls.

The epsilon is a meaningful loss/geometry parameter. For
`y = log(a + epsilon)`, values below epsilon enter the approximately linear
region while values above it retain logarithmic relative-error geometry. After
population standardization:

| epsilon in RMS-relative units | fraction `a < epsilon` | skew | excess kurtosis |
|---:|---:|---:|---:|
| `1e-4` | ~0% | `-0.554` | `1.378` |
| `0.05` | `0.69%` | `-0.147` | `0.198` |
| `0.10` | `2.61%` | `0.044` | `0.075` |
| `0.20` | `9.43%` | `0.289` | `0.151` |
| `1.0` (`log1p(a)`) | `75.65%` | `1.071` | `1.957` |

Thus `epsilon=0.1` is the measured starting point for the next coordinate
control. This audit does not claim that a nearly symmetric marginal will repair
generation. It identifies a concrete mismatch between the Gaussian amplitude
base, the decoder target, and the proposed trunk coordinate that the first
factorized arm did not test.

### 19. Fixed-shape support and real-basis/grouping controls

The matched C4 block-DCT support sweep changes the earlier causal reading. All
arms use the deterministic perceptual 4 x 8 x 8 latent, 256 active scalars,
16 tokens x 16 values, sequence RoPE, seed 7, and the same AR model/schedule.

| support | clean | shuffled | gap | fixed/fresh visual gate |
|---:|---:|---:|---:|---|
| 2x2 | `1.337` | `1.829` | `0.492` | pass |
| 4x4 | `1.405` | `1.744` | `0.340` | pass |
| 8x8 (full map) | `1.479` | `1.758` | `0.279` | pass |

Increasing support weakens context use and visual sharpness monotonically, but
the full-map endpoint remains recognizably semantic. Global support is therefore
a graded difficulty, not the binary obstruction previously implied.

A matched full-support matrix then varied real basis and grouping. Full DCT and
full Hartley both pass when grouped either as contiguous 2x2 frequency-plane
tiles or radial quartets. Their final clean/shuffled/gap values are respectively
`1.445/1.753/0.309`, `1.436/1.761/0.325`, and `1.462/1.684/0.222` for DCT tiles,
Hartley tiles, and Hartley quartets; the DCT-quartet cell is the support8 arm
above. Fresh seed 54321 passes in every cell. Neither real basis family nor these
grouping laws is a binary separator.

The native-complex bridge passes. An exact isometric compact FFT over the same C4
map preserves all 256 values and the 16 x 16 AR shape, and generates recognizable
fixed and fresh samples. Final clean/shuffled/gap is
`1.458 / 1.679 / 0.221`. Cartesian complex coordinates, Hermitian packing,
implicit phase, Gaussian linear flow, and Euclidean loss are therefore modelable
on the short standardized aggregate.

The corresponding scale audit reveals a 594.10x DC-to-highest-radius RMS ratio
for raw standardized CIFAR FFT, versus only 3.66x for standardized C4 FFT. Active
coordinate p50/p90/p99/p99.9 is `0.112/0.686/3.363/12.705` raw versus
`0.547/1.541/3.216/5.722` for C4. This promotes a phase-preserving raw scaling
control: shared positive real/imag radial-RGB RMS divisors at exponent 0.8, no
centering, leaving approximately the C4 residual hierarchy. Its 5k and 10k grids
remain texture and correlate `0.943/0.940` with the unscaled grids. It remains
rough/mushy at 30k (final loss `0.7905`, not cross-coordinate comparable). Scale
hierarchy alone therefore fails the repair gate.

The 16 x 16 resolution separator passes in every arm. Pixel, unscaled compact
FFT, and scaled compact FFT at 16 tokens x 48 dimensions all generate semantic
low-resolution scenes by 5k and pass clearly by 7.5k/10k. Scaled/unscaled grids
correlate `0.968--0.970`. Removing 16--32-pixel detail repairs raw compact FFT;
normalization does not.

The final cheap separator kept 16-pixel data fixed and compared 64 x 12 unscaled
compact FFT with matched 2x2 pixel patches. Both pass through 10k. Compact-FFT
samples correlate `0.976/0.986/0.990/0.994` with the 16 x 48 layout across the
four gates. The added high-frequency dependency burden—not 64 attention tokens
or token width—is now the leading obstruction and directly motivates ring-block
generation.

The first ring-block codec arm completed. It preserves the
existing target-12 layout and 53 x 64 exported latents, changing only attention
masks from bidirectional-within-sector to bidirectional-within-ring while staying
causal between radius bins. The default remains sector-causal for checkpoint
compatibility. Unit, legacy-interface, and end-to-end smoke tests pass. Held-out
test PSNR improves from `28.05 dB` at 10k to `31.82 dB` at 20k and `32.85 dB` at
30k, missing the legacy codec's `34.77 dB` by `1.92 dB`. Equal-noise decoder
robustness and PCA rank are nearly unchanged, while its ring-summary causal
probe is worse.

The corresponding generator is implemented but gated on that final codec. It
packs the same 53 x 64 normalized latents into 23 padded ring vectors (at most
four latents or 256 values per ring), consumes completed rings causally, and
jointly denoises all active values in the target ring. Learned absolute ring
slots provide identity; fp32 sequence RoPE and QK normalization shape and
stabilize attention. There is deliberately no redundant physical-metadata
conditioning path in the first arm. Packing, masks, causal/cache parity,
gradients, deterministic sampling, checkpoint contracts, reduced GPU
integration, and full-configuration one-step GPU smokes pass. Matched 10k runs
on both the legacy and ring-block codecs remain texture/pseudo-scenes under
fixed and fresh seeds. Final conditional x0 MSE is `0.217/0.205`; the lower
teacher-forced number is measured on training batches and does not repair free
rollout. The visual negative survives; held-out context use was not tested.

There is a direct raw follow-up with unusually tight shape matching. The exact
32 x 32 compact FFT has 23 integer-radius rings containing `3--288` active real
coordinates per ring, versus the AE ring generator's 23 rings and `64--256`
active values. If the AE generator passes, running the same joint-within-ring
architecture on raw coefficients will separate the benefit of the causal ring
law from the benefit of learned AE transport. This separator takes priority over
another phase/noise sweep, but its prerequisite ring visual gate failed, so it
is not launched.

The next separator increased AR depth. An existing target-4 codec
exports 134 x 64 latents and reconstructs at `49.57 dB`. The frozen interface
and grouped generator were generalized to dynamic token counts/dimensions, and
a 134-step single-latent arm used learned target slots, fp32 sequence RoPE, QK
normalization, and a depth-6 flow head. It remains texture/pseudo-scenes through
10k and fresh seed despite training-batch conditional x0 MSE `0.140`. More
decisions alone do not repair the visual result.

The combined test trains the same target-4 layout at 16 latent dimensions.
This yields 134 x 16 = 2,144 scalars (1.43x compression) and four-times-smaller
diffusion targets. It reaches `32.76/34.07 dB` held-out at 10k/20k; the 20k
metrics also include pixel MSE `3.926e-4`, physical FFT NRMSE `0.03677`, phase
circular error `0.01379`, and radial-power relative error `0.01182`. The fitted
deterministic interface gives `2.36%` single-token and `11.78%` ring-summary
linear-probe improvement over zero. This clears the reconstruction gate. The
matched 134-step, 16-D generator uses the exact 20k checkpoint and interface.
The codec's final 30k endpoint reaches `34.34 dB`, only `0.27 dB`
above 20k. The generator remains texture/pseudo-scenes at every fixed gate from
2.5k through 10k under CFG 1.0--2.0, and fresh seed 54321 fails at 5k and 10k.
Final training-batch conditional/null x0 MSE is `0.254/0.365` (reported relative
gap `0.336`). It does not establish held-out context use; the fixed/fresh visual
failure is the valid negative.

Matched latent perturbation audits further refine the compression hypothesis.
The z16 20k code needs `172/2144` PCA dimensions for 90% sample energy and `239`
for 99%; z64 at target-4 needs `178/8576` and `241`. Compression removes large
amounts of redundant coordinate expansion but scarcely changes this sampled
intrinsic rank. At normalized noise sigma `0.1/0.2/0.35`, z16 decoding reaches
`34.47/28.55/23.74 dB` versus z64's `38.47/32.46/27.59 dB` relative to clean
reconstruction. The smaller code is therefore more noise-sensitive, not an
obviously easier diffusion endpoint. Reports are in
`diagnostics/latent_robustness_t4_z16_c20k/` and
`diagnostics/latent_robustness_t4_z64_c30k/`.

## Implication

**Superseded in part.** This section originally proposed a cascade/prefix model
and a compressive AE as the two routes. The cascade was dropped at the user's
request as an evasion of the interesting question, and the compressive-AE argument
rested on claims later corrected (section 7). The pixel control at the top of this
document supersedes both: it shows directly that the representation stack, not the
data, the architecture, the budget, the objective, or any of the modeling
machinery, is what prevents generation.

What survives from the original reasoning is narrower. The tested AE is nearly
lossless (34.9 dB) and exports 3392 scalars for 3072 image values, so it did not
earn an obvious rate advantage or discard much expensive detail. That makes the
specific code a weak escape hatch. It does **not** follow that overcomplete
latents are intrinsically harder: expansion can untangle a manifold, add useful
redundancy, or create smoother coordinates. Future AE comparisons therefore gate
on reconstruction, perturbation robustness, latent geometry, and generated
images rather than requiring scalar compression. Lossiness remains one useful
way to discard perceptually irrelevant detail, not a universal prerequisite.

The completed no-AE controls show that neither the current autoencoder nor
per-orbit variance whitening is required to produce the failure. The short raw-AR
controls additionally verify the exact Hermitian Gaussian measure and show that a
4x SNR shift increases the **training-batch** shuffle gap. The old diagnostic did
not measure held-out conditioning. The corrected 10k visual trajectory still
shows that this training-side change does not become coherent generation within
the planned early window.

The forward sequence is maintained in `ROADMAP.md`. Full-image Hartley succeeded
at roughly the full-DCT tier, while both square-spiral legacy FFT and compact exact
raw FFT remained mushy. The later compact C4 FFT pass reopens one narrowly
specified normalization question: its radial RMS hierarchy is 3.66x rather than
raw CIFAR's 594x. The active control changes only uncentered positive real/imag
scales to match that hierarchy. This is not a return to per-orbit complex
centering or covariance whitening. The earlier amplitude-before-phase arm and
64-step Hartley AR remain useful negative results, but complex phase is no longer
an intrinsic explanation.

**Superseded implication.** The first structured compressive bridge did not fail
the generator gate in the abstract; its *global-Hartley tokenization* failed.
The unchanged old MSE endpoint succeeds in local raster tokens. Perceptual C8 and
C4 codecs reproduce the same global-weak/local-strong split, and within-token DCT
also passes. The wrapped-normal phase-score control fails, so changing phase
process again is not the next move.

The spatialized-prefix and output-grouping controls are complete. Spatializing the
known prefix does not improve the matched raw-global C4 arm. Changing the global
Hartley horizon from 16 steps to 4, 64, or 1 also does not produce a visual repair;
the four-band arm is worse, the scalar arm remains rough, and the one-token
Hartley endpoint fails alongside its one-token spatial control. Small fixed bands,
exact frequency-pure tokens, and removal of AR exposure are therefore closed as
stand-alone fixes.

The forward path is no longer justified as a locality-only hierarchy. Keep the
passing C4 local-DCT AR as the strongest practical baseline, and record that full
support DCT/Hartley and compact FFT also pass at the matched short interface.
Phase-preserving raw scaling fails, while 16-pixel compact FFT passes at both
16 x 48 and 64 x 12. The added high-frequency dependency structure is therefore
the leading raw obstruction. Matched 23-step ring generators then fail on both
the legacy and ring-block codecs, so joint-within-ring staging is insufficient.
Both 134-step sequential separators fail. The second combines those decisions
with 16-D compressed latents from the 34.07 dB 20k codec and still remains
texture/pseudo-scenes through 10k and fresh seed. The
deterministic spatial codec remains a
valid practical escape hatch: generate its spectral latents, decode, then apply
a deterministic FFT when global coefficients are required. Wavelets remain
outside the immediate plan.

### 20. Factorized-polar v2 result and spectrum-level factorization gate

The two cumulative factorized-polar v2 arms are complete. The coordinate-only
arm is at
`continuous_runs/ar_fft_factorized_polar_v2_eps01_global_10k/`. Relative
amplitude uses `log(a + 0.1)` followed by one fitted population mean/std across
all frequencies and RGB channels (`-0.34504/0.64290` on 49,984 examples). It
otherwise retains the old Cartesian trunk history, depth-3 amplitude/phase
heads, geodesic phase flow, and physical Cartesian auxiliary, so it isolates the
measured amplitude-coordinate defect.

Its **training-batch** clean/shuffled/gap trajectory is
`2.914/3.915/1.001`, `2.733/4.335/1.602`, `2.563/4.569/2.006`, and
`2.388/4.686/2.297` at 2.5k/5k/7.5k/10k. Every fixed grid and fresh seed 54321
remains non-semantic. The coordinate correction nevertheless improves the old
arm's fresh oracle-history physical NRMSE from `0.609` to `0.563` and its
true-prefix-384 phase coherence from `0.384` to `0.429`. This closes the old
log-amplitude tail as the primary cause, not as an irrelevant measurement.

The cumulative arm at
`continuous_runs/ar_fft_factorized_polar_v2_full_eps01_global_d6_10k/` uses
depth-6 heads and replaces Cartesian history with standardized gated-polar
features. Its **training-batch** clean/shuffled/gap trajectory is
`2.478/4.230/1.752`, `2.423/4.716/2.292`, `2.249/4.556/2.307`, and
`2.255/4.797/2.541`. Final phase coherence improves from the coordinate arm's
`0.829` to `0.853`, log-amplitude MAE improves from `0.258` to `0.246`, and trunk
history-input RMS rises from `0.025` to `0.650`. All four fixed grids remain in
the same texture/pseudo-scene basin. The final fresh/prefix audit is
`diagnostics/factorized_polar_v2_full_10k/`.

These visual negatives show that standardized amplitude coordinates, true polar
history, and depth-6 per-token heads did not repair these arms. They do not
constitute clean individual ablations: the context gaps were not held out, the
Cartesian head lacked the polar head's direct slot condition, training conditions
phase on true or intermediate predicted amplitude while sampling always uses the
integrated endpoint, and the `0.1 x` Cartesian auxiliary is the only term that
couples phase error to physical cross-frequency energy. A future polar test must
make reconstructed Cartesian error primary before it can support a broader
geometry conclusion.

It also does not test the user's
stronger spectrum-level factorization
`p(a_1:L, phi_1:L) = p(a_1:L) p(phi_1:L | a_1:L)`, because each current phase is
sampled before future amplitudes exist. The completed oracle therefore exposes the
complete **true** amplitude field to a bidirectional intrinsic joint-phase model
and asks whether sampled phases decode into recognizable held-out images. It is
implemented in `train_joint_phase_oracle.py`, launched by
`scripts/run_joint_phase_oracle_10k.sh`, and writes to
`continuous_runs/joint_phase_oracle_true_amplitude_10k/`. It passed exact
true-phase reconstruction, focused CPU tests, a real-interface smoke, a full
115M-parameter batch-256 GPU smoke, and the repository suite (`173` tests plus
`3` subtests) before launch. Its primary phase term is still relative-amplitude
gated and the physical Cartesian term is weighted only `0.1`, so a negative is
exploratory rather than a decisive rejection of spectrum-level factorization.

The oracle completed 10k steps. Exact true phases reproduce the reference row,
uniform phases destroy structure as expected, and sampled phases remain
nonsemantic colored texture at 2.5k/5k/7.5k/10k and fresh seed 54321. The final
20 logged points have total loss `1.9652 +/- 0.0097` SEM, phase loss
`1.9586 +/- 0.0096`, and Cartesian loss `0.06564 +/- 0.00135`; the scalar decline
does not become recognizable conditional samples. Artifacts are
`continuous_runs/joint_phase_oracle_true_amplitude_10k/samples_*.png` and
`checkpoint_final.pt`. This is a negative for the current relative-phase-dominant
objective, not for all-amplitudes-first generation under a physical metric.

The conditional phase samples failed visually. Archive the result but do not use
it to close the 1,028-step amplitude/phase chain until the physical-loss confound
is repaired. Retain the local C4 DCT transport as the practical baseline.

### 21. Population-normalization scope and low-frequency clamp

The original 53 x 64 posterior-mean interface fitted a separate affine for every
token position and latent channel. Two missing frozen-AE controls now fit either
one affine per latent channel shared over all 53 positions or one affine over the
entire 53 x 64 tensor. They preserve position-RMS ratios of `1.92x` and `1.98x`,
respectively, rather than setting the ratio to one. Their fitted causal-probe
improvements are `0.7565/0.7607`, versus `0.0887` for the position-by-channel
interface; much of the position-dependent mean/scale is therefore trivially
predictable.

Both matched 115M joint models completed 30k. Fresh seed 54321 remains
nonsemantic for both scopes at 7.5k, 15k, and 30k. Final channel-versus-tensor
image correlation is `0.9943`; correlations against the old failed interface are
`0.9271/0.9197`. Last-20 training losses are `0.45823 +/- 0.00206` and
`0.45493 +/- 0.00208` SEM, versus `0.90010 +/- 0.00263` for the old interface,
but these losses are not physically comparable across affine measures. The
visual result is the gate: broader population normalization makes the numerical
flow task easier without repairing generation. Artifacts are
`diagnostics/joint_vae_mean_{channel,tensor}_30000_fresh_54321.png`.

This was a frozen-codec boundary test. The original AE itself was trained on
`orbit_standardize`; even its exponent-zero variant retains per-orbit centering.
The stronger user-proposed control is now implemented as codec normalization
`global_standardize`: one train-population pixel mean/std, followed by exact
isometric FFT packing, with no per-orbit centering or whitening. It has unit
active-tensor RMS while preserving the natural spectrum. Literal subtraction of
the arithmetic mean of every packed real/imag coefficient is deliberately not
used because it moves every complex origin. The matched 53 x 64 VAE retrain is at
`autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-vae-kl0.0001-global_standardize/`.
It completed at held-out pixel MSE `0.0005572` / PSNR `32.54 dB`, about `2.2 dB`
below the old orbit-standardized AE. The tensor-wide posterior-mean interface
has global mean/std `0.00089/0.80984`, ordinary causal-probe gain `0.5906`, and
ring-summary gain `0.6769`. Its matched joint generator is complete at
`latent_continuous_runs/joint-vae-globalstandardized-mean-tensor-rf-w768-l12-b256-s1-n30000/`.
It completed negative at 30k: fixed previews remain nonsemantic and a 64-image
fresh seed `54321` grid is likewise texture-like (`latent_rms=1.0047`) at
`diagnostics/joint_vae_globalstandardized_tensor_30000_fresh_54321.png`. This
closes the stronger AE-training normalization objection as a primary repair.

The zero-training compact-FFT low-frequency clamp is also complete. It enforces
the exact linear bridge for held-out known coordinates after every Heun update;
the mask follows individual active scalars, so the legacy DC/Nyquist mixed units
cannot contaminate the cutoff. Radius cutoffs `0/2/4/8` expose
`0.10%/2.44%/6.74%/24.32%` of scalars and
`23.61%/72.17%/85.07%/95.06%` of target energy. Unknown-coordinate MSE improves
only `5.7%/12.1%/10.8%/15.9%` over unconditional sampling and remains
`91.6%/52.5%/51.6%/38.2%` worse than zero-filling the unknown spectrum. Small
prefixes remain texture; large cutoffs become recognizable only after the oracle
low-pass image already contains the layout. This supports weak use of supplied
global structure but not a high-frequency-only failure. The exact full clamp has
zero token error. See `diagnostics/compact_fft_low_frequency_clamp/`.

### 22. BOS alignment and quantitative direct-control calibration

The coefficient AR first prediction is aligned. During training the backbone sees
`[BOS, x0, ..., xL-2]` and its slots target `[x0, ..., xL-1]`. During sampling,
`init_cache()` runs the same learned BOS plus `slot_embed[0]`, samples `x0`, and
then `forward_step(x0, position=0)` adds `slot_embed[1]` before predicting `x1`.
The Hartley/C4 AR path uses the same shift. Ring AR uses a zero previous-ring
vector plus an explicit BOS bit and target-ring slot. Joint diffusion has no BOS.
Five focused BOS/slot/causal/cache-parity tests pass. In the default Cartesian
arm the separate learned BOS and slot-zero vectors are partly gauge-redundant
because their sum is what the first slot observes; this does not create a
train/inference mismatch or image-dependent leakage.

The first common 5,000-sample evaluation of the globally affined orthonormal
direct controls is complete. Every arm uses seed `71001`, 50-step Heun sampling,
the same `torch-fidelity` Inception extractor, and the cached 50,000-image CIFAR
reference at `continuous_runs/cifar10_inception_reference_radial.pt`:

| representation | FID (5k) | KID (5k) | radial power relative error |
|---|---:|---:|---:|
| pixels | 31.668 | 0.01991 | 0.0313 |
| patch-local DCT | 31.378 | 0.02038 | 0.0545 |
| patch-grid DCT, token-axis global mix | 130.569 | 0.11172 | 0.1527 |
| full-image DCT | 112.822 | 0.09547 | 0.1296 |
| full-image Hartley | 156.363 | 0.14135 | 0.0569 |
| compact isometric FFT, legacy self-first | 164.603 | 0.14690 | 0.2956 |

This resolves the audit's visual-rating objection in the important direction.
Pixel and patch-DCT are one strong tier. Full DCT occasionally contains objects
but is far worse as a distribution; Hartley and compact FFT occupy the same poor
tier. The old language that all full-support endpoints “pass” was too generous,
and the small Hartley-versus-compact difference cannot support a phase- or
complex-specific mechanism. The large local-versus-global gap is real at this
budget. Artifacts and 64-sample grids are under `diagnostics/control_fid/`; the
reusable evaluator is `evaluate_control_diffusion.py`.

The blind visual protocol independently agrees. Four new seeds per arm produced
20 shuffled 4 x 4 panels. `blind_key.json` was SHA256-frozen before viewing; the
pre-reveal rating marked eight panels semantic and twelve weak. After reveal,
all eight semantic panels were pixel/patch-DCT and all twelve weak panels were
full-DCT/Hartley/compact FFT (`8 TP, 12 TN, 0 FP, 0 FN`). Artifacts, key hash,
and pre-reveal ratings are in `diagnostics/control_blind_4seed/`.

The resulting highest-information separator is `patch_grid_dct`: start from the
successful 64 x 48 pixel patches and apply an orthonormal 8 x 8 DCT only across
the 64 patch positions, independently for every within-patch feature. It changes
local tokens into globally supported tokens without changing shape, scalar
measure, or the 48-D feature interpretation. Exact round-trip, Parseval, global
impulse support, unit tests, and a GPU smoke pass. The matched 30k run is at
`latent_continuous_runs/patch_grid_dct_control/`.

The bridge is complete and discriminative. At 5k its samples are colored texture,
while the matched pixel and patch-DCT arms already contain recognizable objects;
at 30k it still contains mainly texture and coarse fragments. Its 5,000-sample
FID/KID is `130.57/0.11172`, versus `31.67/0.01991` for pixels. Trailing loss is
`0.3994 +/- 0.0040` SEM. This elevates global token-axis mixing from a broad
correlation to the leading architectural mechanism. The conjugacy result below
localizes that mechanism to native computation rather than state/noise geometry.

The next independent separator is implemented and complete. The legacy compact
packer remains unchanged; two corrected layouts emit every active orthonormal FFT
scalar inline, including self-conjugate values at their ordered physical/scale
location. Both are exact permutations of the same 3,072 values with no padding,
whitening, centering, or rescaling. Grid-local ordering yields median/worst
within-token RMS ratios `2.69/11.38` and toroidal distances `3.33/8.17`.
Scale-homogeneous ordering yields `1.12/7.36` and `8.86/10.57`. Exact inverse,
Parseval, Gaussian-bridge, and layout tests pass (`180` repository tests plus
three subtests overall). The matched runs are
`latent_continuous_runs/fft_compact_isometric_gridlocal_control/` and
`latent_continuous_runs/fft_compact_isometric_scale_control/`.
Their 30k FID/KID is `171.38/0.15681` and `214.04/0.20414`, versus legacy
self-first `164.60/0.14690`. Trailing losses are `0.4150 +/- 0.0037` and
`0.4320 +/- 0.0037`. The grid-local correction does not repair generation, and
scale-homogeneous grouping is substantially worse despite lower radial-power
error (`0.0719` versus `0.1164`). Self placement and token scale spread were real
confounds but not causes; frequency locality is preferable when forced to choose,
and further packing permutations are not justified.

The matched C4 optimization-rate bracket is complete.
It uses the same deterministic perceptual C4 checkpoint and identical 16 x 16
joint-flow architecture, batch 256, seed, LR path, 50-step Heun solver, and 30k
image exposure for local DCT, full-map DCT tiles, and full-map Hartley tiles. A
CPU end-to-end full-DCT smoke passed. Runs are under
`continuous_runs/joint_c4_rate_{local_dct,full_dct,full_hartley}_s1_30000/`.
Shared 5,000-sample FID at 10k/20k/30k is
`72.38/65.49/62.39` for local DCT, `91.61/86.29/81.80` for full DCT, and
`94.47/89.27/86.09` for Hartley. Corresponding KID trajectories are
`0.0781/0.0683/0.0639`, `0.0966/0.0905/0.0844`, and
`0.0996/0.0942/0.0899`. Trailing 30k loss is
`1.2206/1.2700/1.2832`. More exposure improves all arms similarly, while the
global penalty remains `19.4--23.7` FID. The old 10k global comparison was
undertrained, but insufficient exposure is not the primary cause of its gap.

A zero-retraining conjugacy control also loads the successful pixel
model while keeping the sampled state and Heun updates in either patch-grid-DCT
or corrected compact-FFT coordinates. Each velocity evaluation applies the exact
inverse global transform, calls the local pixel-token network, and transforms the
velocity back. A positive result would distinguish a bad native transformer
interface from a bad stochastic state/noising geometry and motivate explicit
dual-domain computation. Both arms are complete and strongly positive: each
produces the same clean pixel-tier sample set, with base round-trip max error
`2.15e-6` for compact FFT and `5.25e-6` for patch-grid DCT. This closes global
coordinates as a pathological Gaussian/flow state space and isolates the direct
token/computation interface.

### 23. Trainable dual-domain scaffold gate

The Stage-C conditional control confirms the zero-training conjugacy result with
a newly trained model. A deterministic C4 reconstruction supplies the complete
coarse scaffold. The target is the standardized pixel residual, but Gaussian
state, flow interpolation and target, FFT-space MSE, and 50-step Heun integration
all stay in the exact corrected compact isometric FFT. Every learned velocity
evaluation is conjugated into 64 aligned local 4 x 4 pixel patches and receives
the aligned scaffold patch through a separate input projection. The predicted
local velocity is conjugated back before the solver update. Both scaffold and
residual use one train-population scalar affine; nothing is centered or whitened
per frequency. Fitted scaffold mean/std is `0.47198/0.24758`; residual mean/std
is `0.000469/0.048233`. The model is 115.51M parameters.

The common 5,000-sample result through 30k is:

| condition/source | untouched scaffold | refiner 5k | refiner 10k | refiner 20k | refiner 30k |
|---|---:|---:|---:|---:|---:|
| deterministic scaffold from each held-out real image | 37.32 / 0.03646 | 28.00 / 0.01656 | 19.44 / 0.01003 | 14.33 / 0.00609 | 12.63 / 0.00512 |
| unconditional 30k local-DCT C4 sample | 62.39 / 0.06393 | 47.83 / 0.03295 | 40.34 / 0.02715 | 35.76 / 0.02290 | 34.69 / 0.02254 |

The first row is a conditional reconstruction/refinement gate and must not be
reported as an unconditional CIFAR generator. The second row is the honest
unconditional two-stage pipeline and uses the oracle-trained refiner zero-shot.
Both improve decisively. For held-out oracle scaffolds, radial-power error falls
from `0.401` to `0.109/0.0417/0.0461/0.0511` across 5k/10k/20k/30k. For
generated scaffolds it falls from `0.426` to `0.127/0.0628/0.0965/0.0458`.
Paired images preserve object and layout;
the early residual is over-textured rather than semantically collapsed. This
distinguishes the failure from every native global-token arm: global FFT is a
valid state and output space, while a shared transformer operating directly on
dense globally supported token mixtures is the dominant obstruction.

A same-noise 10k condition-permutation control closes the most important cheap
alternative explanation. The sampled residual is still added to its original
scaffold, but the denoiser sees scaffold patches rolled by one example within
each batch. Held-out oracle FID changes from aligned `19.44` to shuffled `45.44`
(untouched scaffold `37.32`); generated-scaffold FID changes from `40.34` to
`65.95` (untouched scaffold `62.39`). Oracle paired completion PSNR falls from
`24.27` to `23.51 dB`. Thus the gain is not an independent texture prior pasted
onto a structure-preserving skip connection. The refiner materially uses aligned
scaffold content, and incorrect content is actively harmful.

The 20k and 30k refiners were then attached unchanged to the passing 10k frequency-major
local-DCT AR model. Its unconditional scaffold FID/KID is `91.19/0.09966`; the
refined outputs are `73.66/0.05844` and `71.70/0.05707`. Final radial error
improves `0.443 -> 0.0431`.
Therefore the refiner transfers to the actual AR front end, but that front end is
substantially weaker than the 30k joint local-DCT C4 generator. The AR pipeline's
dominant remaining error is coarse scaffold generation rather than Fourier
residual geometry.

The old AR checkpoint was not exposure-matched: 10k steps at batch 64 saw only
`0.64M` images, versus `2.56M` for a 10k joint arm. An unchanged batch-256
frequency-major AR retrain completed in 12.9 minutes and improved scaffold
FID/KID from `91.19/0.09966` to `84.18/0.09207`. With the selected frozen 30k
FFT refiner, the end-to-end result improved from `71.70/0.05707` to
`63.25/0.04869`.

**Retraction of the first interpretation.** Calling the remaining `11.80` FID
difference from the 10k joint local-DCT arm an “AR cost” conflated causal
factorization with token composition. Joint local DCT stores every channel and
all four 2 x 2 DCT modes for one patch together; frequency-major AR distributes
those modes across different tokens. The missing controls are complete:

| step | joint patch-major | joint frequency-major | AR patch-major | AR frequency-major |
|---:|---:|---:|---:|---:|
| 5k | 82.47 | 95.52 | **77.44** | 89.48 |
| 10k | 72.38 | 86.65 | **75.48** | 84.54 |
| 20k | 65.49 | 78.22 | **75.83** | 84.06 |
| 30k | **62.39** | 74.70 | 75.11 | 82.53 |

These are common 5,000-sample scaffold FIDs. Within-family comparisons isolate
packing: frequency-major costs joint `13.05/14.27/12.73/12.30` FID and AR
`12.04/9.06/8.23/7.41` FID through 5k/10k/20k/30k. The effect replicates under
bidirectional and causal trunks and survives exposure. Joint-versus-AR remains
architecture/schedule-confounded, so the 30k same-packing gaps (`12.72`
patch-major, `7.83` frequency-major) are suggestive rather than a pure causal
measurement.

The patch-major AR plus frozen refiner scores scaffold/completion FID/KID
`77.44/0.08461 -> 52.69/0.03959` at 5k,
`75.48/0.08259 -> 53.34/0.04042` at 10k,
`75.83/0.08402 -> 54.34/0.04093` at 20k, and
`75.11/0.08242 -> 54.06/0.04054` at 30k. Thus 30k narrowly selects the standalone
scaffold, while 5k selects the two-stage pipeline. Later scaffolds lose gradient
and radial power even as train MSE falls. Final patch/frequency-major last-20
train loss is `0.3532 +/- 0.0013` and `0.3815 +/- 0.0015` SEM, whereas held-out
clean loss is `4.984/5.251`. Exposure and context use matter, but loss cannot
rank these samples.

This identifies what the positive local solution preserves: the input/output
adapters see a coherent local bundle with its coupled low/high DCT modes, while
the trunk models dependencies between complete patches. Component splitting is
therefore not a safe default. A future amplitude-then-phase decoder should keep
both substeps inside one atomic token and publish only the completed bundle to
history. Strict low-to-high Fourier causality should operate after the local
scaffold, over causal rings/blocks with bidirectional within-block denoising.

Artifacts are
`latent_continuous_runs/scaffold_fft_residual_oracle_c4_s1_30000/`,
`diagnostics/scaffold_fft_residual_oracle_c4_{5000,10000,20000,30000}/`, and
`diagnostics/generated_c4_scaffold_fft_refinement_refiner{5000,10000,20000,30000}/`; the
condition controls add the corresponding `_shuffled` directories.
The AR transfer is
`diagnostics/ar_c4_patchdct_freqmajor_fft_refinement_refiner{20000,30000}/`.
The first exposure control and its midpoint/final evaluations are
`continuous_runs/ar_c4_patchdct_freqmajor_b256_s7_10k/` and
`diagnostics/ar_c4_patchdct_freqmajor_b256_{5000,10000}_fft_refinement_refiner30000/`.
The composition controls are
`continuous_runs/joint_c4_rate_local_dct_freqmajor_s1_30000/`,
`continuous_runs/ar_c4_patchdct_freqmajor_b256_s7_30k/`, and
`continuous_runs/ar_c4_patchdct_patchmajor_b256_s7_30k/`; common evaluations are
under the corresponding `diagnostics/*30ksched*` and final `*30000*` directories.
Training completed at the predeclared 30k endpoint in 51.5 minutes. Final loss is
`0.6608`; the last 20 logged losses are `0.6620 +/- 0.0041` SEM. The 30k
checkpoint is best on oracle, joint-generated, and AR-generated FID/KID and is
the selected refiner. Do not return to another normalization, phase, or packing
sweep as the next move.

### Strict-ring separator (2026-08-06, complete negative)

The next experiment implements the consequence of the composition control
without reopening representation geometry. It predicts the same globally
standardized exact compact-FFT residual as Stage C1, conditioned on the same
deterministic C4 oracle scaffold. The difference is learned computation: the
scaffold is first encoded bidirectionally as 64 local 4 x 4 patches, then used as
prefix memory for a causal 23-ring FFT trunk. Ring zero receives an explicit BOS
at both train and inference. Ring `r>0` receives completed ring `r-1`; a learned
absolute target slot says which ring is being predicted. QKNorm and fp32 RoPE
are active.

Within a ring, one depth-6 diffusion MLP denoises the entire padded vector
jointly. Active widths range from 3 to 288 exact scalar coordinates. The mask
partitions all 3,072 coordinates exactly, every RGB real/imaginary orbit bundle
belongs to one ring, and fixed-dimension reduction preserves equal physical
coordinate measure rather than equalizing rings. This is therefore causal only
between completed radial blocks and fully connected within the current block.

The model has 106.09M parameters. Only the C1 scaffold linear projection and
learned local patch positions are copied; all 12 new transformer layers and the
diffusion head are fresh. The causal-invariance test perturbs a future target
ring and confirms earlier target conditions are bitwise unchanged. A separate
test walks the inference KV cache and matches the full teacher-forced target
conditions. Packing, backward, generation, full-width batch-128, and evaluator
smokes pass. The bounded 10k job and 2.5k checkpoints live at
`latent_continuous_runs/scaffold_fft_ring_residual_oracle_c4_s1_10000/`.
It completed in 10.0 minutes. Final loss is `0.45995`; the last 20 logged losses
are `0.47217 +/- 0.00320` SEM.

Decoded inspection is negative at every gate. On the common 5,000-sample oracle
panel, untouched scaffold FID/KID is `37.32/0.03646`; sampled-history completion
is `75.82/0.06124`, `78.80/0.06318`, and `74.53/0.05963` at 2.5k/5k/10k.
Paired PSNR is `23.49/23.29/23.42 dB` versus scaffold `26.29 dB`. Radial error is
nevertheless `0.0439/0.1371/0.0515`, showing that a nearly right marginal power
spectrum does not imply coherent images.

Three diagnostics localize the failure. First, true-ring teacher history only
improves 5k/10k FID to `75.90/70.31`, so exposure bias is not the dominant
obstruction. Second, 50 rather than 20 Heun steps changes final FID
`74.53 -> 75.40`; solver under-integration is closed. Third, shuffling the
scaffold prefix improves final FID to `70.91` and barely changes paired PSNR
(`23.42 -> 23.41 dB`). This reverses the passing C1 condition audit and means the
static target-ring summary does not use aligned scaffold content productively.

Do not interpret C2 as a pure causal-factorization control. It also removed
C1's evolving spatially aligned noisy-residual/scaffold interaction and replaced
it with a 768-D summary plus a shared global MLP. The next strict-causality
separator should retain C1's local dual-domain transformer under an asynchronous
ring schedule: lower rings complete, current ring diffusing, future rings at
base noise, and only the current returned FFT velocity active. This directly
tests causality without discarding the computation known to work.
