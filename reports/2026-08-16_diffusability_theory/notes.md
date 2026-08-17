# Diffusability of representations — theory notes

Snapshot date: 2026-08-16 UTC. Discussion notes (Ethan + Claude), logged so the
running theory thread survives session loss. Companion measurement script:
`spectral_consistency.py` in this directory (CIFAR-10 train, luma channel, 50k
images; white-noise baseline matched in per-pixel variance).

Status of claims: sections 1–3 are derivations or measurements; sections 4–6
are working hypotheses and design instruments, to be judged by decoded FID as
per the project's standing metric rules.

## 1. Basis, rotation, and SNR: resolving the disagreement

Setup: rectified-flow path `z_t = (1-t) eps + t z` with isotropic `eps`, flat
coordinate MSE on the velocity target.

Two different quantities were being discussed:

- **Directional SNR** — signal energy along a fixed unit direction over noise
  variance along that direction. Isotropic noise has variance 1 along *every*
  unit direction, so a direction with total signal norm 10 has directional SNR
  100 whether its energy sits in one coordinate or is spread across many. A
  single linear layer (matched filter) recovers it either way. Because
  isotropic Gaussian noise and flat MSE are both rotation-invariant, a pure
  rotation of the data composed with rotating the model's input/output
  projections leaves the objective and the information content exactly
  unchanged.
- **Per-coordinate SNR** — what any coordinate-diagonal operation sees. Before
  rotation, each coordinate is a blend of directions at wildly different SNRs;
  after rotation into the eigenbasis, each coordinate is SNR-pure.

Both statements in the original exchange are true about their respective
quantities. The operative question is which quantity the training machinery is
sensitive to:

- Sensitive only to directional SNR (rotation-invariant): the forward process,
  the flat MSE loss, weight decay on linear maps (Frobenius norm is
  orthogonally invariant).
- Sensitive to per-coordinate SNR (rotation-dependent): **Adam's per-coordinate
  second-moment preconditioning**, any per-coordinate loss weighting or
  normalization, elementwise clipping/quantization (e.g. fp16 caches).

So the honest summary: *rotation changes nothing about the problem, but it
changes what diagonal interventions can express.* Rotation is the enabling move
that makes SNR-targeted weighting/scaling/scheduling expressible as per-
coordinate operations. It also means whitening ablations are never pure: Adam
already applies an implicit per-coordinate (not per-direction) preconditioning,
and it is timestep-blind.

## 2. Why scale/weight interventions would not decouple, and a t-aware fix

Observed lesson: whitening/scale interventions change the data, which moves
each direction's noise-floor crossing **both** in training and in generation —
they alter the curriculum and the sampler's resolving order at once. Static
loss re-weighting is the opposite: it leaves the data alone but is blind to
timestep. A direction with variance `lambda` crosses SNR=1 at
`t* = 1/(1+sqrt(lambda))`; a weak direction that only resolves near `t≈0.9`
spends ~90% of uniformly-sampled training steps fully drowned, where its loss
sits at the irreducible floor and its static weight is wasted amplifying
floor-level noise.

The decoupled instrument has to be **direction- and timestep-aware**. The
second-order (Gaussian) version is closed-form. For an eigendirection with
variance `lambda`, observation `y = t s + (1-t) n`, velocity target
`v = s - n`, the Gaussian-optimal (Wiener) velocity MSE floor is

```text
F(lambda, t) = lambda + 1 - (t*lambda - (1-t))^2 / (t^2*lambda + (1-t)^2)
```

with sanity limits `F(., 0) = lambda` and `F(., 1) = 1`. Proposed weighting:
train in the (freely available, section 1) eigenbasis and weight each
(direction, t) cell by `w = 1 / max(F(lambda, t), eps)`. A merely
Gaussian-optimal model then incurs ~uniform loss over all cells, and the
gradient concentrates exactly on **excess-over-Gaussian** error — the
non-Gaussian structure — at each direction's own resolution window. This
leaves the data, the resolving order, and the sampler untouched; only credit
assignment changes.

Caveats: it is still a second-order allocation heuristic (higher-order
structure is what it is *for*, not what it models); it interacts with Adam;
clamp the floor; gate by decoded FID, per the standing rule that geometry
statistics are diagnostics, not objectives.

## 3. Measured: per-sample spectral consistency of CIFAR (axis A)

Question: how consistent is the power-law structure per-image, versus only in
population average? Radial band powers (per-FFT-bin mean, luma), 50k train
images:

| statistic | r1-2 | r3-4 | r5-6 | r7-8 | r9-12 | r13-16 |
|---|---:|---:|---:|---:|---:|---:|
| mean log10 power | 3.628 | 2.724 | 2.229 | 1.873 | 1.412 | 0.828 |
| per-image std of log10 power | 0.361 | 0.292 | 0.275 | 0.268 | 0.260 | 0.286 |
| P(band_k > band_k+1) per image | 1.000 | 0.997 | 0.993 | 1.000 | 1.000 | — |

Per-image spectral slope (log power vs log radius, r in [1,12]):
`-2.92 ± 0.38`. White-noise control: adjacent-band ordering ~0.48–0.50 (chance),
slope `0.00 ± 0.17`.

Cross-band correlation of per-image log-energies decays smoothly with spectral
distance (adjacent 0.83–0.91; most distant 0.50), versus ~0 for noise. So the
conjectured "neighborhood correlation in eigenspace" is real, and it is
**causally linked to schedule consistency**: the ordering margin between
adjacent bands is a difference of two log-energies whose fluctuations are
~0.9-correlated, so `Var(log p_k - log p_{k+1})` is far smaller than either
band's own variance. Correlated fluctuation is *why* the per-sample resolving
order almost never flips despite ~2x (1 sigma) per-band energy spread.

Contrast: pitched audio concentrates energy at a per-sample fundamental and
harmonics — cross-band dependence is strong, but *which* bands dominate varies
per sample, so the resolving order under a fixed noise schedule is
unpredictable. Gaussian noise has neither property.

## 4. Two-axis picture of a diffusable representation

Working hypothesis for "what makes images a lucky modality," stated as two
separable, measurable axes:

- **Axis A — schedule consistency.** The map from timestep to "which
  directions are resolving now" is (nearly) the same for every sample, because
  per-sample energy ordering matches the population ordering (section 3). The
  fixed global noise schedule is then a valid curriculum for every sample
  simultaneously.
- **Axis B — cross-stage predictive value.** Directions that resolve early
  carry information about directions that resolve later, so each denoising
  stage inherits usable conditioning.

Key structural observation: for a **stationary Gaussian** signal, the Fourier
basis diagonalizes the covariance — every frequency is independent, so axis B
is exactly zero. Coarse-to-fine order exists but is *worthless*: nothing
resolved helps anything unresolved. All of images' axis-B value is therefore
carried by **higher-order statistics** — cross-frequency phase alignment
(edges, contours, object structure), not the power spectrum. This is the
precise version of "don't stop at second-order stats": second-order structure
sets the schedule (axis A); the payoff of following that schedule (axis B) is
entirely non-Gaussian.

Consequences:

- Population spectra / covariance eigenvalues can tell you the *order* a
  representation will resolve in, but never whether that order is *useful*.
- Axis B is measurable with existing project machinery generalized to the
  diffusion frame: at matched noise level t, how much does (nonlinearly)
  conditioning on the currently-resolved content reduce error on the
  not-yet-resolved content, versus unconditional? ("Conditioning gain curve" —
  the nonlinear, t-indexed version of the Gate-H prefix-predictability tables.)
- A candidate latent should be scored on both axes before priors are trained
  on it. A VAE pushed toward N(0, I) destroys both axes in the limit (order
  and dependence both vanish as the prior's job migrates into the decoder);
  the useful operating point is enough smoothing to kill pathological
  fine-scale geometry, not full Gaussianization.

## 5. Instruments for coaxing an ordered latent

Goal from the discussion: a latent whose directions ascend the noise floor in a
designed order (along token index, or eigendirection within/across tokens),
so the diffusion/crescendo schedule and the code agree by construction.

- **Index-ramped bottleneck noise** (most direct): inject noise with per-token
  std `sigma_i` ascending in token index during autoencoder training. Lineage:
  nested dropout (Rippel et al. 2014) provably induces exact PCA ordering in
  the linear case; a continuous noise ramp is its smooth analog and doubles as
  VAE-style smoothing. This simultaneously (a) forces high-variance,
  early-resolving content into early tokens, (b) sets the marginal-variance
  ordering that determines noise-floor crossings, (c) provides decoder
  robustness in exactly the region priors will actually sample.
- **Co-design with the crescendo frontier**: choose `sigma_i` so that token i's
  content crosses SNR=1 at the frontier time the rolling/diffusion-forcing
  sampler reaches token i. Teacher (tokenizer) and generator (frontier
  schedule) then share one schedule instead of the schedule being fit
  post-hoc to whatever the code happened to learn.
- **Eigen-level ramp**: same idea applied in the latent covariance eigenbasis
  (online/EMA-estimated rotation) rather than token index — heavier machinery,
  only worth it if the token-level ramp proves too coarse.
- Cautions carried over from the VAE exploration: watch total information
  (reconstruction ceiling), and watch axis B — smoothing that Gaussianizes
  away cross-direction dependence removes exactly the structure that makes
  the resolving order valuable. Rank rising "naturally" under a VAE is not by
  itself a win; score candidates on axes A and B plus decoded FID.

## 6. Channel-replication diffusion (side project) — analysis notes

Proposal: replicate the RGB image k times across channels, independent noise
per copy; variants with static invertible transforms per copy (flips,
rotations, color jitter); aggregate per-copy predictions.

- **Plain duplication is (to first order) a schedule shift.** k independent
  noisy views of the same x0 at time t carry the information of one view at
  k-fold SNR — a matched filter (channel-averaging, trivially expressible in
  the first conv) realizes it. Any experiment needs the control: single-copy
  model with the equivalently shifted noise schedule. Gains over that control
  measure genuine architectural/ensembling benefit, not "less noise is
  easier." (Coding-theory framing: pure duplication is a repetition code —
  the weakest code; transform-diversity is what makes the code nontrivial.)
- **Transformed copies are the interesting arm.** Averaging per-copy x0
  estimates after inverse transforms is score averaging over a group orbit —
  a projection of the score onto the (approximately) equivariant subspace.
  Known variance-reduction/regularization effect; at sampling time the copies
  additionally act as a self-consistency constraint on the trajectory.
- **Aggregation must happen in x0 space**, as noted: convert each copy's
  velocity to its x0 estimate, apply the inverse static transform, average,
  then map back per-copy (each copy keeps its own noise/trajectory).
  Non-augmented exact copies' estimates should agree up to noise; divergence
  during sampling is itself a useful diagnostic. Consider an explicit
  consistency projection every m steps (reset copies to transforms of the
  averaged x0 estimate re-noised to the current t) versus trusting the model.
- **Static (per-clone-fixed) transforms: agreed** — clone identity must be
  learnable by position. Randomized transforms would need the transform as
  conditioning.
- Transform menu beyond jitter/flip/rotation: 90-degree rotations and flips
  (exact, lossless), RGB channel permutations, a decorrelated color-space copy
  (YUV: one clone carries luma/chroma), integer translations (exact under
  periodic FFT-style handling), and — most interesting for AFIG — **blur-level
  copies**: clones at different blur/resolution levels have different
  effective spectra, so they cross the noise floor at different times; the
  coarse clone resolves first and conditions the fine clone. That is a joint,
  single-state version of cascaded diffusion, and connects this side project
  directly to the crescendo/progressive-token program. Closest precedent:
  Matryoshka Diffusion Models (joint multi-resolution denoising); also
  cascaded diffusion and relay diffusion.
- Predicted failure mode to watch: the network collapses the copies by
  averaging them in the first layer (recovering the schedule-shift
  equivalent). Per-copy weights (grouped/separate stems, as proposed) or
  limited early cross-copy mixing keep the ensemble nontrivial.
- Suggested sweep order: (1) k in {2, 4, 8} plain duplication **with** the
  schedule-shift control; (2) flips/rotations with x0-space inverse-transform
  averaging; (3) blur-pyramid clones; (4) consistency-projection on/off.

## 7. Addendum (same day, follow-up discussion)

**Loss weighting parked** by preference for latent shaping. Noted for the
record: a compound weight in global and directional SNR was previously
explored as a heuristic acknowledging that resolved global signal should
raise the demand on unresolved directions. The Gaussian floor of section 2
fixes credit allocation only; it cannot see conditioning value that has no
second-order signature.

**Correction to section 5 (index-ramped noise).** The ramp — like the prefix
mask — is a *content allocation* instrument: it decides what each token
knows, i.e. reconstruction dominance. It does not set noise-floor crossing
order (marginal energy) nor conditional/causal order (higher-order). In-house
evidence that these dissociate: Gate B/G produced strong functional ordering
with only mild energy ordering (slot stds 0.735–1.149). Three independent
levers:

1. **Content allocation** — prefix mask, sigma-ramp, rate objectives.
2. **Schedule gauge** — when each token/direction ascends the floor is pure
   second-order energy, and for a learned latent it is an invertible gauge:
   per-token (or per-eigendirection) population scaling applied before the
   prior and inverted before decode changes the resolving schedule with zero
   information change. Tested so far: natural gauge (accidental) and
   per-coordinate whitening = the *flat* gauge (slightly negative). The
   **designed crescendo gauge is untested**: rescale the existing cache
   per-token to a prescribed descending profile (matched to the Gate-H
   functional order or a power law), retrain the same joint prior, decode
   with the inverse. No tokenizer retraining; isolates schedule design from
   content.
3. **Conditional order** — not settable by scaling; needs content-side
   instruments.

**Frontier-noise AE training (main new proposal).** The prefix-masking
objective is the sigma-to-infinity limit of crescendo noising (masking the
suffix = infinite noise on it). Generalize: sample a frontier position, noise
each token with the sigma the crescendo sampler would have there (early
clean, frontier partial, late heavy), decoder reconstructs. This trains the
encoder to make partially-resolved prefixes maximally informative about (and
decodable with) the still-noisy suffix — i.e. it shapes the causal
conditioning order at the sampler's actual state distribution, which the
plain ramp does not. It also confers decoder robustness exactly along the
generation trajectory and co-designs the tokenizer with the diffusion-forcing
frontier by construction. The aggressive alternative — a jointly-trained
causal probe predicting token i from its prefix with gradient into the
encoder — has a degenerate optimum (later tokens most predictable when
empty); try frontier-noise first.

**Blur clarification:** blur-level clones apply to the pixel-space
channel-replication project only. There is no meaningful blur along latent
token/feature axes; the prefix/frontier structure is the latent analog.

**Latent axis-A measurement recipe** (runs once caches are rebuilt; ~10-line
change to `spectral_consistency.py`): primary axis = flattened 1024-D
population eigenbasis banded by eigenvalue rank (1–8 / 9–32 / 33–128 /
129–512 / 513–1024). Per sample: band energies, then (a) ordering consistency
vs population — margin-normalized flip probability or Spearman, since the
eigen-spectrum is locally flatter than the pixel spectrum and raw
P(band_k > band_k+1) is not comparable; (b) cross-band log-energy correlation
(eigen-neighborhood structure); (c) per-direction activity CV (std/mean of
per-sample energy). Secondary axes: token-slot energies, feature-cov
eigenbasis within tokens, functional order (Gate H, done). **Prediction:**
the latent looks more audio-like than image-like — mid-rank eigendirections
are present-or-absent semantic features, so high activity CV, weak per-sample
ordering consistency, weak eigen-neighborhood correlation. If confirmed, this
mechanistically accounts for part of the latent diffusability tax and routes
the fix: the gauge (lever 2) can enforce the population schedule but cannot
reduce per-sample activity variance — only content shaping (levers 1/3) can.
The measurement is therefore a decision procedure for where to spend effort.
