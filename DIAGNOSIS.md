# Why latent AFIG samples are texture-like

Date: 2026-07-30; control results updated 2026-08-03. Diagnostics in
`diagnostics/`, scripts named `diagnose_*.py`.

AE under test: `ae-causal-ring-t12-...-vae-kl0.0001/checkpoint_30000.pt` with
`latent_interface_posterior_mean.pt` (posterior-mean policy, 34.76 dB test PSNR
reproduced here). Generative model under test:
`joint-vae-mean-rf-w768-l12-b256-s1-n30000/checkpoint_final.pt`.

## Conclusion

The generative models are not failing at rollout, at overfitting, at decoder
fragility, or at matching the latent distribution's low-order statistics. They
are failing because **they have only learned a slightly-better-than-Gaussian
model of the latent distribution, and a Gaussian fit of this latent space decodes
to exactly the texture mush we see in samples.**

Ruled out by measurement, each with a named script: decoder hypersensitivity (1),
low-order distribution mismatch (3), and AR exposure bias (5, since the
bidirectional model fails identically).

**Important scope correction.** The paragraph above and sections 1-15 describe the
**joint** model, which genuinely does not memorize (section 6, 0.5% train/test
gap). The **AR** model is a different story: section 16 shows it memorizes badly,
and its conditional advantage *reverses* on held-out data. The two paths have
different failure modes and were conflated in earlier drafts. Any claim here that
cites AR training diagnostics -- including the "conditional MSE 0.128" figure used
in section 1 -- is measuring a memorized signal and should not be relied on.

The objective is misallocated relative to perception. 53 x 64 = 3392 latent
dimensions for a 3 x 32 x 32 = 3072-dimensional image is 0.91x -- an *expansion* --
and per-dimension whitening rescales all 3392 dimensions to unit variance. So the
loss is a flat mean over 3392 dimensions while perceptual importance is
concentrated roughly 1000:1 in the first few tokens: the four tokens that dominate
image structure receive ~7.5% of the gradient.

(Section 7 corrects an earlier, stronger version of this claim. The latent is
*not* incompressible -- ~54% of each token's variance is linearly predictable from
the other tokens. But that linear redundancy is already fully exploited by the
Gaussian baseline the model beats, so it is not the missing ingredient.)

**Diffusion forcing will not fix this.** It targets AR exposure bias, but the
bidirectional joint-diffusion model has no rollout and no exposure bias, and it
fails identically and in precisely the same place (see graft test below).

## The decisive controls: the current representation is the problem

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

Five matched direct-representation controls have now completed at 30,000 steps:

| representation | shape | params | 30k decoded result |
|---|---:|---:|---|
| 4x4 pixel patches | 64 x 48 | 115.5M | recognizable CIFAR classes |
| per-patch 4x4 DCT | 64 x 48 | 115.5M | recognizable CIFAR classes |
| full-image DCT, 4x4 frequency tiles | 64 x 48 | 115.5M | recognizable, but delayed/weaker |
| per-orbit-whitened FFT | 65 x 48 | 115.5M | texture mush |
| FFT without per-orbit variance scaling | 65 x 48 | 115.5M | texture mush |

Artifacts:
`latent_continuous_runs/pixel_control/preview_0030000.png`,
`latent_continuous_runs/patch_dct_control/preview_0030000.png`,
`latent_continuous_runs/full_dct_control/preview_0030000.png`,
`latent_continuous_runs/control_fft_whitened/preview_0030000.png`, and
`latent_continuous_runs/control_fft_global/preview_0030000.png`.

The two FFT outputs are not merely both broken; under the paired fixed seed they
are visually very similar (same-channel RGB correlations 0.930--0.940). Their
loss values differ because their target scales differ and must not be compared.

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
than an unconditionally better ordering. The active `fft_global_spiral` control
keeps every FFT coefficient fixed while reducing mean pair distance to `2.84`.
See `diagnose_token_composition.py` and
`diagnostics/token_composition.json`.

Normalization caveat: `fft_global` is a historical shorthand. The implementation
sets the variance-whitening exponent to zero and uses a global residual scale, but
the codec still subtracts per-orbit complex means. It therefore isolates
per-orbit **variance scaling**, not every form of frequency-dependent
normalization. Section 15 suggests the retained centering is small for ordinary
complex orbits, but an exact unitary-noise control remains worthwhile.

### The exact raw-FFT AR control does not emerge by 10k

Two new short 2.5k-step arms exercise the normalization/noise audit proposed
above. They use the existing 514-group causal AR path, an isometric Hermitian
real/imaginary packing (3,072 active real coordinates), one global pixel mean,
one robust DC-derived scale, exact target-frequency metadata, and no frequency
loss weights.
Tests verify pixel/Fourier L2 equality, round trip, Gaussian-bridge equality,
velocity equality, and physical decode.

| arm | final held-out clean | shuffled | gap | decoded result |
|---|---:|---:|---:|---|
| Cartesian + ECS | 0.012045 | 0.013190 | 0.001144 | texture mush |
| Cartesian + ECS + 4x SNR | 0.023469 | 0.026515 | 0.003047 | texture mush |

The fitted global affine values were pixel mean `0.4733601` and scale `10.43208`.
The baseline normalized history RMS was `0.02513` against unit Gaussian noise.
The 4x-SNR arm scales the internal clean endpoint by two, so its bridge SNR is
exactly four times larger while external token coordinates and physical decode
remain unchanged.

The SNR change measurably improved use of causal context: its final held-out
shuffle gap is 2.7x the baseline, and its step-2k gap is `0.003821` versus
`0.001602`. But both 16-image decoded grids contain only low-frequency colored
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

| step | held-out clean | shuffled | gap | decoded result |
|---:|---:|---:|---:|---|
| 2,500 | 0.023732 | 0.027525 | 0.003793 | texture, no clear objects |
| 5,000 | 0.021503 | 0.027910 | 0.006406 | texture, no clear objects |
| 7,500 | 0.021457 | 0.028950 | 0.007492 | texture, no clear objects |
| 10,000 | 0.019638 | 0.026835 | 0.007197 | texture, no clear objects |

This separates two facts that the short run could not. First, the AR trunk does
learn and increasingly use held-out causal context; the shuffle gap roughly
doubles from 2.5k to 7.5k and does not reverse at 10k. Second, none of that gain
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

## Implication

**Superseded in part.** This section originally proposed a cascade/prefix model
and a compressive AE as the two routes. The cascade was dropped at the user's
request as an evasion of the interesting question, and the compressive-AE argument
rested on claims later corrected (section 7). The pixel control at the top of this
document supersedes both: it shows directly that the representation stack, not the
data, the architecture, the budget, the objective, or any of the modeling
machinery, is what prevents generation.

What survives from the original reasoning: the AE is nearly lossless (34.9 dB) and
non-compressive -- 3392 latent dims for 3072 real values -- so latent generative
modeling here is about as hard as modeling 32x32 pixels at 35 dB fidelity, with no
compression benefit to show for it. Standard latent diffusion works because the AE
is *lossy*: it discards perceptually irrelevant high-frequency detail, leaving a
smooth, low-dimensional, modelable latent. That is about perceptual relevance, not
predictability -- per section 7 the discarded detail is roughly half predictable.

The completed no-AE controls show that neither the current autoencoder nor
per-orbit variance whitening is required to produce the failure. The short raw-AR
controls additionally verify the exact Hermitian Gaussian measure and show that a
4x SNR shift improves held-out conditioning substantially. The corrected 10k
trajectory shows that this improvement still does not become coherent generation
within the planned early generalization window.

The forward experimental sequence is maintained in `ROADMAP.md`. The active
bridge is a full-image separable Hartley transform: real, orthonormal, global,
periodic, and packed in the same contiguous frequency-grid tokens as full DCT.
If it succeeds, radial/Hermitian token composition becomes the first repair; if
it fails beside full DCT, proceed to explicit amplitude-before-phase modeling
with circular phase geometry. Then build a structured, genuinely compressive AE
whose generation gate is tested early. Wavelets are not in the immediate plan.
