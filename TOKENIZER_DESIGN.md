# Progressive tokenizer design

## Representation

The canonical bottleneck contains 32 ordered whole-image vectors of width 64:

```text
z = E(x),       z.shape = [batch, 32, 64]
x_hat = D(z)
```

This is 2,048 continuous coordinates for a 3,072-scalar CIFAR image. The initial
autoencoder is deterministic in both directions. The encoder ends in an
affine-free LayerNorm followed by an unconstrained learned linear projection;
there is no normalization after that projection.

## Architecture

The encoder patchifies the image into an 8-by-8 grid, applies eight width-512
bidirectional Transformer blocks, and lets 32 unique learned Perceiver queries
pool the resulting features. Two pooling blocks alternate query-to-image cross
attention, query self-attention, and feed-forward computation.

The autoencoder decoder begins with 64 content-free spatial queries, represented
by a shared learned output token plus a unique 2-D output position. Eight
width-512 decoder blocks alternate spatial self-attention, cross-attention to the
continuous latent sequence, and feed-forward computation. A final normalization
and patch projection reconstruct the image.

This decoder is not the future autoregressive model or its stochastic token
head. It is only the deterministic decoding half of the autoencoder.

## Gate A: complete-sequence reconstruction

The first objective is ordinary pixel MSE in the `[-1, 1]` training space:

```text
L_full = MSE(x, D(E(x))).
```

Its purpose is to establish the reconstruction ceiling and inspect the emergent
latent distribution without imposing an ordering regularizer prematurely.

The initial optimizer uses a 1,000-step linear warmup to `2e-4`. The first run
showed a clear late-step instability after its best 5k checkpoint, so the
canonical continuation restores that checkpoint and fine-tunes at `5e-5`. The
held-out curve reached a second plateau around 9--10k, after which the run steps
down again to `2e-5` for its reconstruction-ceiling phase.

## Gate B: progressive prefixes

For an ordered prefix of length `K`, the same decoder masks every latent after
`z_K`:

```text
x_hat_K = D(z_1, ..., z_K)
L = L_full + E_K[MSE(x, x_hat_K)],   K uniform in {1, ..., 31}.
```

Every training example always receives the complete reconstruction loss and one
random shorter-prefix loss. Token order is fixed; no token permutation is used.
The expectation is a compute-efficient estimate of training every possible
prefix.

Gate B trains the same architecture jointly from scratch. A 100-step CIFAR
pilot initialized from Gate A showed severe interference: while prefix quality
began improving, complete-sequence PSNR fell from 41.18 to 29.69 dB even though
the warmup had reached only `5e-6`. The ordered objective must therefore shape
the encoder and decoder together instead of attempting to rewrite a converged,
distributed code.

The completed 30k Gate-B checkpoint reaches full-test prefix PSNR
`20.54 / 22.61 / 25.14 / 28.84 / 31.95 / 35.73` dB at
`K = 1 / 2 / 4 / 8 / 16 / 32`. This is a smooth successive-refinement code,
with a 5.45 dB complete-reconstruction cost relative to the unordered Gate-A
ceiling.

## Gate A result

The selected 25k checkpoint reconstructs the complete CIFAR-10 test set at
41.18 dB PSNR (pixel MSE `7.63e-5`). It outperforms the old 53-by-64 ring codec's
38.27 dB while using 2,048 instead of 3,392 latent scalars. A chronological 30k
checkpoint reached 40.75 dB and is retained separately.

The unconstrained clean latent has global mean `0.071`, standard deviation
`0.935`, covariance effective rank `21.7 / 64`, and coordinate standard
deviations spanning `0.503--1.330`. Slot RMS lies in `0.884--1.056`. Thus the
classic final-normalization-plus-projection interface naturally learned a
well-scaled code without a hard spherical constraint or injected noise.

As expected, the complete-sequence objective did not spontaneously order the
registers: prefix PSNR is 10.62, 10.68, 11.60, 12.79, 17.13, and 41.18 dB at
lengths 1, 2, 4, 8, 16, and 32. Gate B directly tests whether asymmetric prefix
reconstruction can turn the same high-quality code into successive refinement.

## Deferred constraints

Latent AWGN, explicit power constraints, semantic teacher losses, and soft
continuous prefix gates are ablations rather than v1 defaults. They should be
introduced only in response to measured failure modes such as fragile latent
encoding, one-token information concentration, tail starvation, or poor prior
modelability.

## Measurements

Final evaluation reports reconstruction at prefix lengths 1, 2, 4, 8, 16, and
32, together with:

- normalized and physical-pixel MSE;
- PSNR;
- coordinate standard-deviation range;
- latent covariance effective rank;
- mean per-token peak-to-RMS ratio;
- RMS by ordered latent slot.

The first bottleneck comparison holds the total coordinate count fixed at 1,024:
`16 x 64` tests fewer, richer autoregressive decisions, while `32 x 32` keeps
the 32-step ordering and makes each individual decision smaller.

## Gate C: joint generative positive control

Generative modeling begins with all 32 tokens noised and denoised jointly. This
removes autoregressive exposure bias and the capacity of a small per-token head
as confounds. The frozen 12.5k Gate-B encoder produces a fixed CIFAR latent
cache. Its train split has tensor-wide mean `0.0645`, standard deviation
`0.7649`, and range `[-4.19, 4.47]`; slot standard deviations span only
`0.672--0.833`. The baseline therefore uses one population mean and scale for
the complete tensor, not coordinate or slot whitening.

The model is a 12-layer, width-512 bidirectional DiT-style rectified flow over
the `32 x 64` tensor. It combines learned absolute slot embeddings with fp32
1-D RoPE in attention, uses QKNorm and canonical AdaLN-Zero, and predicts the
velocity of the straight path

```text
z_t = (1 - t) epsilon + t z,
v_target = z - epsilon.
```

Training samples `t` uniformly and applies ordinary unweighted MSE. Sampling
uses 50-step Heun integration. EMA, CFG, per-coordinate whitening, timestep
weighting, and class conditioning are absent from the first control.

The joint control improved monotonically but slowly: 5k/10k/15k/20k FID was
`127.64 / 94.27 / 80.53 / 75.30`, with corresponding KID
`0.1288 / 0.0882 / 0.0719 / 0.0654`. At 20k the decoded samples contain
recognizable animals, vehicles, and outdoor layouts, establishing that the
continuous latent is generatively viable. Its quality remains substantially
behind the earlier pixel and patch-local controls, so modelability is not yet a
solved property of the tokenizer.

## Gate D: ordered autoregressive prior

The causal baseline factorizes the learned sequence in its intended order. The
trunk receives the exact shifted sequence

```text
input:  [BOS, z_1, ..., z_31]
target: [z_1, z_2, ..., z_32].
```

Learned target-position embeddings identify which register each residual state
must predict, while 1-D RoPE and QKNorm shape causal attention. No separate
source-token metadata is added: the fixed one-step shift and target identity are
sufficient. Unit tests perturb each target and verify that it cannot influence
its own condition.

The output distribution is a shared six-block conditional rectified-flow MLP.
It concatenates the width-512 trunk state with the timestep embedding, passes
them through a two-layer fusion MLP, and supplies the resulting condition to
canonical AdaLN-Zero blocks. Training noises all 32 teacher-forced targets in
parallel; inference generates one complete 64-D register at a time with 50-step
Heun integration. The normalization, optimizer, flow path, lack of EMA, and
flat loss weighting match Gate C.

The initial comparison used the tokenizer frozen at 12.5k so modeling could
begin before the codec run completed. AR 5k/10k/15k/20k FID was
`139.44 / 123.08 / 113.21 / 105.90`, with KID
`0.1390 / 0.1205 / 0.1092 / 0.1013`. At every matched checkpoint this is worse
than joint flow, even though AR teacher-forced validation MSE is much lower.
For example, at 10k AR validation MSE is `0.289` versus joint flow's `0.424`,
while FID is `123.08` versus `94.27`. This was initially interpreted as direct
evidence of exposure bias. **Retracted:** joint and teacher-forced conditional
flow MSE have different irreducible floors and are not comparable across these
factorizations. Later continuation also improves AR FID while worsening its
teacher-forced validation MSE.

The AR cache contains one fixed orientation for each of 50k training images.
By 20k, training MSE fell to `0.215` while held-out teacher-forced MSE worsened
from `0.287` at 15k to `0.319`. This was initially called generative overfitting.
**Retracted as a model-selection claim:** a later 16-token AR continuation shows
that decoded FID can improve monotonically while teacher-forced validation MSE
worsens. The metric still detects conditional-regression generalization, but it
does not identify the best generative checkpoint.

## Gate E: pooling and bottleneck audit

The first tokenizer established reconstruction and generative viability, but its
`32 x 64` continuous code is only a modest dimensional bottleneck and its
64-coordinate covariance had effective rank about 25 at the frozen 12.5k
checkpoint. The v2 audit therefore tests whether a stricter pooling operation
and a 1,024-coordinate code are easier to model.

New attention defaults replace unit-L2 Q/K vectors plus a learned per-head log
temperature with learned affine RMSNorm on each head dimension. RMSNorm scales
start at one and scaled-dot-product attention supplies the ordinary
`1 / sqrt(head_dim)` factor. Legacy checkpoints retain their original attention
form when loaded. Training forwards use `torch.compile(mode="default",
fullgraph=True)` by default; eager execution remains available with
`--no-compile`.

The new `cross_only` pool uses one learned-query cross-attention operation and
exports the attended value result directly through the existing final norm and
latent projection. It deliberately removes the learned-query residual, latent
self-attention, and pooling FFN. The four controlled arms are:

- two-block residual pool, `32 x 64`, as the new-attention control;
- cross-only pool, `32 x 64`, isolating the pooling rule;
- cross-only pool, `16 x 64`;
- cross-only pool, `32 x 32`.

All four retain patch size 4, width 512, eight encoder and decoder blocks,
horizontal flips, BF16, and the progressive prefix objective. They train with
batch 512 for 15k steps, which exposes the models to the same number of images
as 30k steps at the old batch 256. The two smaller representations have the
same 1,024-coordinate budget, separating token count from token width.

All four runs completed. Held-out prefix PSNR at the native token boundaries is:

| pool and code | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---:|---:|---:|---:|---:|---:|
| cross-only, `16 x 64` | 20.37 | 22.39 | 24.84 | 27.95 | 32.62 | -- |
| cross-only, `32 x 32` | 18.75 | 20.53 | 22.54 | 25.12 | 29.06 | 32.48 |
| cross-only, `32 x 64` | 19.19 | 20.75 | 23.60 | 27.10 | 31.31 | 34.83 |
| residual, `32 x 64` | 20.24 | 22.27 | 24.96 | 27.80 | 32.28 | 34.95 |

The equal-size `16 x 64` and `32 x 32` codes are much closer when compared at
equal coordinate budget rather than equal token count. At 64, 128, 256, and 512
available coordinates, the `32 x 32` code leads by only 0.16, 0.15, 0.28, and
1.11 dB; at the complete 1,024-coordinate code, `16 x 64` leads by 0.14 dB.
The main benefit of `16 x 64` is therefore reducing the number of stochastic AR
decisions from 32 to 16, not increasing reconstruction capacity.

Cross-only pooling is a negative result at `32 x 64`. The residual pool is
better at every measured prefix and its coordinate covariance has effective
rank `29.45 / 64`, versus `19.54 / 64` for cross-only. The reconstruction panels
also show earlier object identity under residual pooling. Removing the learned
query residual and pooling FFN did not produce a cleaner or more economical
code; instead it encouraged a lower-rank one.

The missing factorial cell, residual pooling with a `16 x 64` code, completed at
`20.42 / 22.67 / 25.22 / 29.03 / 31.81` dB for
`K = 1 / 2 / 4 / 8 / 16`. Relative to cross-only `16 x 64`, it is better by
0.29, 0.38, and 1.08 dB at K=2, 4, and 8, but worse by 0.81 dB at the complete
code. Its coordinate covariance effective rank is `30.42 / 64`, versus
`28.02 / 64` for cross-only. Thus residual pooling gives stronger intermediate
prefixes and a richer code at a modest complete-reconstruction cost.

The final selection is being made by modelability rather than PSNR alone. Both
`16 x 64` tokenizers are encoded into deterministic 100k-example training
caches containing each CIFAR image and its horizontal flip; test caches retain
the original 10k images only. Each cache uses one tensor-wide population mean
and standard deviation. Matched 20k-step joint rectified-flow priors then test
whether the residual code's stronger prefixes or the cross-only code's slightly
better reconstruction ceiling translates into a better learned distribution.

The joint-prior result selects cross-only `16 x 64`:

| tokenizer | 5k FID / KID | 20k FID / KID | final validation flow MSE |
|---|---:|---:|---:|
| residual `16 x 64` | 121.19 / 0.11727 | 85.49 / 0.07694 | **0.552** |
| cross-only `16 x 64` | **119.94 / 0.11706** | **74.97 / 0.06691** | 0.588 |

This is a consequential metric disagreement. Residual pooling is consistently
easier to fit in standardized latent-space MSE, yet its decoded distribution is
10.52 FID worse at 20k. The decoded sample panels agree with FID. Therefore,
latent flow MSE, covariance effective rank, and early-prefix reconstruction are
diagnostics rather than tokenizer-selection objectives. The cross-only code is
the selected generative representation. It essentially matches the previous
32-by-64 joint prior's 20k FID of 75.30 while using half the coordinates and half
the token count.

The next ordered control trains the unchanged causal prior on this selected
cross-only `16 x 64` cache. Its exact alignment is
`[BOS, z_1, ..., z_15] -> [z_1, ..., z_16]`; training uses the same tensor-wide
normalization, 100k original-plus-flip cache, optimizer, six-block flow head,
and lack of EMA/CFG as the joint comparison. This measures the combined effect
of the factorization, sequential exposure, and the smaller AdaLN-conditioned
per-token head; it does not isolate exposure bias by itself.

The 20k AR run reaches FID `100.08` and KID `0.09510`, compared with joint FID
`74.97` and KID `0.06691`. Reducing the sequence from 32 to 16 tokens improves
over the earlier 32-token AR result (FID `105.90`) but leaves a substantial
joint-versus-AR gap. The held-out teacher-forced flow MSE bottoms near 14k at
about `0.415`, then worsens to `0.430` at 20k while training loss continues to
fall. Token 1 has the largest raw conditional MSE, but later evidence shows that
cross-slot MSE is dominated by different irreducible floors and must not be read
as a direct ranking of modeling failure.

## Gate F: independent audit and long-run correction

The independent audit is preserved in `AUDIT_RESPONSE_2026-08-11.md`, with
scripts, metrics, and checkpoints under `audit_2026-08-11/`. Its principal
measurements reproduce against the project artifacts:

- full-tokenizer reconstruction FID/KID is `6.15 / 0.00295`;
- isotropic standardized latent noise degrades reconstruction smoothly:
  sigma `0.05 / 0.10 / 0.20 / 0.40` gives FID
  `7.18 / 11.47 / 32.87 / 113.23`;
- the unchanged joint run continued to 40k/60k reaches FID
  `59.31 / 56.53`, versus `74.97` at 20k;
- the unchanged AR run continued to 40k reaches FID `90.40`, despite worsening
  teacher-forced validation MSE;
- replacing the generated AR prefix with 0/1/2/4/8 real tokens gives FID
  `100.39 / 88.81 / 75.56 / 58.42 / 28.09`;
- per-coordinate whitening is a null-to-negative result by 20k
  (`80.18` versus raw `74.97`), with the caveat that whitening ran eager after
  a compile failure;
- the flattened latent covariance has effective rank about `92 / 1024`, and a
  matched full-covariance Gaussian decodes at FID `145.98`;
- PCA truncation to 64/128/256/512 directions gives FID
  `127.80 / 82.23 / 44.39 / 13.53`.

These results retire decoder reconstruction quality and a sharp off-manifold
failure as explanations at current sample quality. They do **not** prove that
the tokenizer representation is generatively cost-free: reconstruction fidelity
and density modelability are different questions, and the joint model still
lags the reconstruction floor and legacy pixel controls.

The joint continuation proves that 20k was an inadequate training budget. The
AR prefix-replacement curve rejects a purely first-token account and shows
distributed degradation. It does not completely distinguish weak one-step
conditionals from exposure accumulated within each generated suffix; that needs
an oracle-history per-stage diagnostic or a direct conditioning-bandwidth
ablation. The 512-D single-vector AdaLN interface of the shared token MLP is now
the leading architectural suspect, not a demonstrated cause.

The revised near-term controls are:

1. test the representation factorization at fixed 1,024-coordinate budget:
   first `32 x 32`, then `64 x 16` if it helps, using matched joint priors;
2. establish a properly trained joint ceiling and a matched current-code pixel
   baseline at comparable budgets;
3. keep unconditional generation as the canonical task—class conditioning and
   CFG may be useful ceilings but change the distribution and are not baseline
   fixes;
4. repair the BOS weight-decay grouping before new runs;
5. only after the joint ceiling is established, test noisy teacher-forced history
   against a richer cross-attention or block-joint AR head;
6. defer ImageNet scaling until the CIFAR representation/recipe tax is measured.

## Gate G: latent geometry and the token-width hypothesis

The selected cross-only `16 x 64` training cache was audited before choosing a
normalization or token-specific noise schedule. Its single tensor-wide
standardization leaves only modest slotwise scale differences: slot scalar
standard deviations range from `0.8796` to `1.1850`, and slot scalar means stay
within `[-0.0968, 0.0816]`. The first 4 and 8 slots contain approximately 31%
and 57% of centered scalar variance, versus 25% and 50% under a uniform split.
Thus a per-slot scalar mean/std is a valid small ablation, but there is no large
slotwise marginal mismatch for it to repair. The stronger per-coordinate
normalization has already been null-to-negative at 20k.

The important asymmetry is instead inside and across tokens. Individual 64-D
slots have covariance effective ranks from `11.0` to `30.7`, and the flattened
1,024-D code has effective rank only `91.7`. Its covariance eigenspectrum is
steep: the top `1 / 8 / 32 / 128 / 256 / 512` directions explain
`16.9 / 44.1 / 65.7 / 85.8 / 93.9 / 98.7` percent of variance. Under the
rectified-flow path

`y_t = (1 - t) epsilon + t y`,

an eigen-direction with variance `lambda` has `SNR = lambda (t/(1-t))^2`.
The corresponding `SNR = 1` crossing occurs at approximately `t=0.089` for
PC1, `0.220` for PC8, `0.364` for PC32, `0.553` for PC128, `0.675` for PC256,
`0.812` for the median direction, and `0.977` for the weakest direction. The
learned code therefore already has a very strong coarse-to-fine emergence in
its covariance eigenbasis. PCA truncation confirms that those directions are
not merely algebraic: 64/128/256/512 PCs decode at FID
`127.8 / 82.2 / 44.4 / 13.5`, respectively.

That eigen-order is **not** the learned token order. Only about 57--60% of the
top-PC energy lies in the first 8 of 16 slots, and dominant PCs are spread
across the sequence. Nor is the token sequence locally correlated: only 17.9%
of covariance Frobenius energy is inside token blocks, while distant token
pairs are often more strongly coupled than adjacent pairs. The nested prefix
objective therefore does not create a PCA-like innovation order. This covariance
result alone does not determine whether the nonlinear decoder gives the prefixes
a functional coarse-to-fine order; that requires measuring reconstruction by
spatial-frequency band.

This supports a clean token-factorization experiment. At a fixed 1,024 scalar
budget, compare `16 x 64`, the already-trained `32 x 32` tokenizer, and, if the
second arm helps, a new `64 x 16` tokenizer. The existing `32 x 32` tokenizer
has essentially the same complete reconstruction PSNR as `16 x 64`
(`32.48` versus `32.62` dB), so its matched joint prior isolates token count
from scalar budget with little decoder-ceiling confounding. Its newly cached
training latents have a per-coordinate covariance effective rank of
`14.0 / 32`, versus `28.0 / 64` for `16 x 64`: channel redundancy scales almost
exactly with width rather than disappearing in the wider token. A `64 x 16` code
also aligns one latent register with each 4x4 source patch in count, without
requiring the registers themselves to be spatial. Run joint generation first;
AR would conflate any representation gain with a longer exposure path.

The decision rule is decoded FID at matched steps and recipe. If `32 x 32`
materially beats `16 x 64`, train and test `64 x 16`; if it ties or loses, do
not expand to 64/96/128 registers yet. Per-slot scalar normalization is secondary
and should only follow a positive token-count result. Token-dependent SNR or
diffusion-forcing schedules remain deferred until the desired ordering is
defined explicitly (prefix importance, covariance PCs, or a learned schedule).

## Gate H: axis-separated and spectral-prefix audit

Flattened covariance is relevant to the joint flow's global geometry, but it is
not the right statistic for asking whether token positions behave like a
correlated sequence and feature coordinates behave like correlated channels.
The audit in `scripts/analyze_progressive_latent_axes.py` therefore reports them
separately and measures the decoder's Fourier reconstruction error for every
prefix.

For the selected `16 x 64` code, the sequence-position covariance has effective
rank `8.60 / 16`, mean off-diagonal correlation `0.332`, and mean adjacent
correlation `0.347`. The feature covariance has effective rank `25.04 / 64`,
mean absolute off-diagonal correlation `0.166`, and 95th-percentile absolute
correlation `0.411`. Thus both explicit axes are substantially correlated. The
sequence is not locally stationary, however: correlation and block coupling do
not decay monotonically with sequence distance, and several distant pairs are
more strongly coupled than neighbors.

Full-prefix predictability is much stronger than one-neighbor predictability.
After the first few registers, a linear model of the complete prior prefix leaves
only about 6--26% of a target slot's variance unexplained, whereas using the
immediately previous slot alone often leaves 40--85%. The code is consequently
hierarchical and prefix-conditioned, but not approximately first-order Markov.
This favors full causal attention over a local convolution or recurrence.

Most importantly, decoder behavior **is** strongly coarse-to-fine even though
the raw token covariance is not PCA-ordered. Relative to the improvement
available between prefix 1 and the complete reconstruction, the prefix at which
half the Fourier-band error has been removed is:

| code | radius 0--3 | radius 4--7 | radius 8--11 | radius 12+ |
|---|---:|---:|---:|---:|
| `16 x 64` | 2 | 3 | 5 | 7 |
| `32 x 32` | 2 | 4 | 8 | 13 |

For `16 x 64`, token 2 has already removed 55% of recoverable radius-0--3 error
but only 11% of radius-12+ error; by token 7 the corresponding numbers are 94%
and 50%. This directly validates the intended low-to-high-frequency progressive
semantics. The earlier inference that the hierarchy was not aligned with token
order is therefore narrowed: it is not an eigenvalue/PCA ordering, but it is a
clear functional frequency ordering through the nonlinear decoder.

The `32 x 32` code spreads essentially the same progression across more steps.
Its sequence and feature effective ranks are `9.88 / 32` and `11.44 / 32`, so
doubling token count does not double independent sequence structure. It creates
smaller, more redundant increments. The matched 60k-step joint prior reaches FID
`54.57` and KID `0.04472`, compared with `56.53 / 0.04584` for `16 x 64`.
Thus the narrower, longer factorization provides a real but modest gain at fixed
scalar budget; it does not by itself close the representation gap. This positive
gate is sufficient to train the next `64 x 16` tokenizer before deciding whether
still longer sequences are useful. Token-indexed rolling noise now has an
empirically meaningful direction, but remains downstream of the joint comparison
so that two changes are not conflated.

The gated `64 x 16` tokenizer completed at `34.06` dB full reconstruction,
compared with `32.62` dB for `16 x 64` and `32.48` dB for `32 x 32`. At equal
coordinate budgets of 64/128/256/512/1024, its PSNR is
`20.57 / 22.66 / 25.21 / 28.86 / 34.06` dB. It therefore retains the
progressive code while raising the full decoder ceiling.

Its axis geometry also shows the limit of simply adding registers. Sequence
effective rank is only `13.78 / 64`, feature effective rank is `4.54 / 16`, and
several late registers are almost completely linearly determined by the full
prefix. Two cross-register block couplings exceed `0.999`. Spectral half-recovery
occurs at registers `3 / 6 / 15 / 27` for radius bands
`0--3 / 4--7 / 8--11 / 12+`, preserving the same coarse-to-fine progression on
a finer grid. The code is best described as a highly redundant refinement path,
not 64 independent factors. Its matched 60k joint prior is the stopping gate:
do not extrapolate to 96 or 128 registers without a clear decoded-FID gain.

The matched joint result is decisively positive. At 60k steps, `64 x 16` reaches
FID/KID `39.37 / 0.02843`, compared with `54.57 / 0.04472` for `32 x 32` and
`56.53 / 0.04584` for `16 x 64`. All arms use 1,024 latent scalars, the same
512-wide 12-block unconditional joint transformer, tensor-wide population
standardization, 50-step Heun sampling, and the same 5k-sample evaluation
protocol. The gain is therefore attributable to the learned token-width
factorization plus its associated decoder, not class conditioning or CFG.

`64 x 16` is now the selected representation. Its fixed-order causal prior is
the next modeling gate, using the existing 12-block causal trunk and six-block
conditional flow head. This intentionally tests the complete factorization:
the narrower 16-D token should ease each conditional denoising problem, while
the 64-step sequence increases exposure and sampling cost. Do not infer its AR
outcome from the joint gain.

The 60k causal result reaches FID/KID `81.68 / 0.07359`. It improves on the
earlier `16 x 64` AR result (`90.40` FID at 40k), but remains `42.30` FID behind
its own `64 x 16` joint ceiling. The representation change therefore transfers
only partially to fixed-order generation: it improves the joint prior by 17.16
FID relative to `16 x 64`, but the available AR comparison by only 8.72 FID.
Sample grids are correspondingly more washed, hybridized, and texture-confused
than joint samples.

Teacher-forced conditional MSE follows the functional hierarchy. The first
eight slots range approximately `0.63--1.11`, while much of the late sequence is
`0.10--0.30`, with a few nearly deterministic slots near `0.026`. Cross-slot MSE
is not a direct measure of modeling error because irreducible conditional entropy
changes with position, but the profile confirms that the most visually
consequential coarse registers are also the broadest conditional distributions.
The next diagnostic is an oracle-prefix rollout at several fractions of the
64-token sequence. If short real prefixes rapidly close the decoded-FID gap,
compounding is the main amplifier and rolling/noisy-context training is justified;
if long real prefixes still generate weak suffixes, the conditional head or
fixed-order factorization remains the primary limitation.

The 5k-sample oracle-prefix sweep gives:

| exact prefix | AR-completed FID | prefix-only FID |
|---:|---:|---:|
| 0 | 81.68 | -- |
| 1 | 80.25 | 191.35 |
| 2 | 77.56 | 132.02 |
| 4 | 70.58 | 96.59 |
| 8 | 58.14 | 63.92 |
| 16 | 40.98 | 38.31 |
| 32 | 19.88 | 17.76 |
| 48 | 10.27 | 9.88 |
| 64 | -- | 7.22 |

Replacing successively longer prefixes improves the rollout smoothly, so
generated-context compounding is real and distributed rather than a first-token
failure. The prefix-only control adds an important qualification. AR suffixes
substantially improve prefixes of length 1--8, but after 16 real registers they
slightly worsen the already-good prefix reconstruction (`+2.67`, `+2.13`, and
`+0.38` FID at prefixes 16, 32, and 48). The falling oracle curve therefore
cannot be credited entirely to strong suffix conditionals: the true prefix itself
supplies progressively more of the final image.

This isolates two failures. Early/middle generated errors compound through the
64-step trajectory, while late conditional samples add detail that is plausible
in isolation but insufficiently aligned to the particular scaffold. A simple
teacher-forcing-noise fix targets only the first. The next efficient modeling
control should jointly generate consecutive register blocks (initially four
registers per 64-D block, producing a 16-step AR process). It preserves the
selected tokenizer and exact scalar code, reduces exposure length fourfold, and
lets correlated neighboring refinement stages coordinate inside one flow head.
Rolling/diffusion-forcing remains the more general follow-up if blockwise AR is
positive.

The first blockwise control is now `block_size=4`: consecutive physical
registers are reshaped from `[64,16]` to `[16,64]` for the causal prior and
restored exactly before tokenizer decoding. No latent coordinate, normalization,
tokenizer weight, or scalar budget changes. The model retains the 512-wide
12-block trunk and six-block flow head; only the stochastic factorization changes.

This control is negative. At 60k steps it reaches FID/KID
`109.66 / 0.10258`, substantially worse than the single-register causal result
`81.68 / 0.07359`. Earlier checkpoints are also poor (`140.25` FID at 10k,
`130.96` at 20k, and `117.35` at 40k), so this is not merely a late-checkpoint
regression. Naively shortening the sequence trades 16-D conditional targets for
64-D targets whose neighboring refinement stages must be modeled jointly; that
head difficulty more than offsets the reduction in exposure length. This result
rules out concatenation as the useful form of blockwise generation. It does not
rule out rolling/diffusion-forcing, where multiple registers remain separate
attention tokens and interact throughout denoising.

## Current prior diagnosis and next gates

The single-register checkpoint sweep also weakens an optimizer-only explanation.
Decoded FID is `103.47 / 89.43 / 81.60 / 81.68` at
`10k / 20k / 40k / 60k`. Held-out teacher-forced flow MSE begins worsening much
earlier, but rollout quality continues to improve through 40k and then plateaus
within the approximately 0.3-FID protocol noise. The widening train/validation
loss gap is real, yet early stopping at the validation minimum would have selected
a much worse generator. Constant learning rate and ordinary regression overfit
are therefore not sufficient causes of the AR gap.

The current training and sampling baseline is deliberately plain:

- 512-wide, 12-block causal trunk with eight heads, RMS QK normalization, RoPE,
  a learned absolute target-position embedding, and a six-block conditional
  AdaLN flow MLP;
- tensor-wide population latent mean/std normalization;
- independent uniform time per target token, straight rectified-flow path
  `x_t=(1-t)eps+t*x_0`, velocity target `x_0-eps`, and flat MSE;
- no min-SNR weighting, logit-normal time sampling, CFG, EMA, class conditioning,
  or per-token loss weights;
- AdamW at `1e-4`, betas `(0.9,0.995)`, matrix-weight-only decay `0.05`, batch
  256, BF16, 1k warmup followed by constant LR, gradient clipping at 1.0, and
  50-step Heun sampling.

Position is identifiable, but only indirectly at the diffusion head. The trunk
adds a unique learned target-position vector before its causal blocks, so its
output `h_i` can carry slot identity. The head then fuses only `h_i` and the time
embedding. A clean low-cost control is to pass the same slot embedding directly
to the head's condition-fusion MLP, for example `MLP([h_i,e_i,e_t])`. This tests
whether target identity is being attenuated by content aggregation without
removing the useful positional input to attention.

The observed loss allocation argues against a special first-token weight as the
next move. Token 1 already contributes about 4.75% of total validation MSE versus
1.56% under a uniform token share; tokens 1--8 contribute about 29.2%. Supplying
the exact first token improves rollout FID by only 1.43, while progressively longer
oracle prefixes improve it smoothly. The failure is distributed across the coarse
prefix and its propagation. If weighting is revisited, a modest normalized weight
over the first 8--16 functional registers is more defensible than heavily weighting
token 1, but it cannot manufacture the context missing at BOS and must be judged by
decoded FID rather than teacher-forced loss.

Current global normalization leaves only modest marginal slot-scale variation:
slot scalar standard deviations span about `0.735--1.149` (a `2.44x` variance
ratio), while feature standard deviations span `0.539--1.453`. A previous
per-coordinate whitening control was null-to-negative, and whitening does not
remove cross-coordinate dependence. Per-slot scalar normalization remains a
possible controlled ablation, but it also erases part of the learned progressive
energy hierarchy and is not currently a leading explanation.

Token-dependent SNR is more promising when treated as a generation schedule than
as normalization. A standalone position-dependent time warp must be used
consistently in training and integration, including its chain-rule velocity, and
should not be chosen from marginal variance alone. Conditional innovation and the
measured coarse-to-fine prefix curve are the relevant quantities. The preferred
form is rolling diffusion/diffusion-forcing: keep the physical 16-D registers as
separate attention tokens, expose the trunk to noisy histories with a time/quality
embedding per register, and advance a moving low-to-high-frequency denoising
frontier. This directly targets the clean-history exposure mismatch without the
64-D head confound of block concatenation.

The near-term order is:

1. add a direct learned target-slot condition to the flow head and verify that it
   changes slot-conditioned behavior;
2. add a small noisy-history/teacher-forcing robustness control, separately from
   any token-dependent schedule;
3. implement rolling diffusion with per-register noise/time conditioning while
   retaining 16-D token heads;
4. only then test modest prefix-aware loss weighting or per-slot scalar
   normalization if diagnostics still point to those mechanisms.

There is no trustworthy scalar "latent quality" score. The working scorecard is
decoder ceiling (34.06 dB and 7.22 reconstruction FID at the current evaluation
size), progressive prefix reconstruction and spectral half-recovery, decoder
sensitivity to off-manifold latent noise, joint-prior FID (39.37), conditional
innovation by slot, and final AR rollout FID. Covariance, correlation, and
effective rank describe the code's geometry, but low rank can mean either useful
redundancy or a modeling liability and must not be optimized in isolation.
