# AFIG roadmap — 2026-09-03

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
  fixed jitter and a mean penalty. The completed posterior/noise study promotes
  deterministic sigma-0.05 decoder jitter for expected value while retaining
  the hard-VAE path as a conservative stability control.
- Scale-invariant sample-varying slot-power balancing at weight `0.002` is now
  part of the leading residual+jitter tokenizer. At 10k it improves FID at all
  three tokenizer seeds and improves mean FID/KID; marginal kurtosis
  regularization did not clear its replication gate.

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

## Phase C — posterior/noise parameterization (screen complete; replication next)

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

All four supervisor-owned chains entered the shared queue at 2026-09-02 07:59
UTC. Every GPU was already claimed by another project, so the jobs are waiting
inside `gpu-claim --wait` and will start without oversubscription as capacity
opens.

**Outcome:** all chains and evaluations completed. Pure deterministic v19 is a
clear regression at FID/KID-5k 47.195/0.03633. Decoder jitter is sharply
non-monotonic: sigma 0.05 reaches 27.743/0.01603, while sigma 0.10 reaches
34.229/0.02528. The v20 gain over v12 repeats at 10k: **25.353/0.01568** versus
**29.588/0.02255**, a 4.235-FID and 0.00687-KID improvement. This becomes the
leading candidate, pending an independent tokenizer seed.

Soft-floor v22 is mechanism-valid rather than boundary-pinned: 5.53% of values
are within 0.05 logvar of the floor, median/mean/p95 sigma are
0.0408/0.169/0.985, and high noise is allocated mostly to low-content slots.
Its FID/KID-5k 33.147/0.02381 does not improve v12, so it is not promoted.
Exact metrics are in
`reports/2026-08-26_autoencoder_program/phase_c_posterior_comparison.json`.

### Phase C decoder-jitter robustness confirmation

Do not promote v20 from one favorable tokenizer seed. Run the exact
deterministic sigma-0.05 recipe at tokenizer seeds 1 and 3, each with a matched
prior seed 1, and compare them with the durable v12 controls at the same seeds.
In parallel, freeze the existing seed-2 v12 and v20 caches and train both with
prior seed 2. This separates tokenizer-training interaction from prior-training
luck without spending compute to reproduce frozen caches.

All four arms first receive the fixed-seed FID/KID-5k evaluation. Advance a
paired comparison to 10k if jitter improves FID, is within two FID of its
control, or FID and KID disagree. Reconstruction, decoder sensitivity,
effective rank, and utilization remain explanatory diagnostics; they only veto
a broken codec (`PSNR < 28` together with clean reconstruction FID `> 25`) and
cannot promote a representation.

Replace v12 as the expected-performance baseline only if the prior-seed-1
jitter arm wins FID at least two of three tokenizer seeds, improves both mean
FID and mean KID across those seeds, has no greater-than-two-FID regression
with concordantly worse KID, and the seed-2 prior-seed-2 pair preserves the
direction. Otherwise retain v12 and report v20 as a checkpoint-local result.
Launcher:
`scripts/run_phase_c_jitter_confirmation.sh {tokenizer_s1|tokenizer_s3|prior2_v12|prior2_v20}`;
larger-sample follow-up:
`scripts/run_phase_c_jitter_confirmation_10k.sh <same-arm>`.

Launch record (2026-09-02 18:16 UTC): all four chains acquired shared
`gpu-claim` locks and entered finite training. Tokenizer seeds 1 and 3 use W&B
`ahrvj0vr` and `zudow065`; the v12/v20 prior-seed-2 pair uses `c3l0pdrh` and
`whl85sj2`. Supervisor owns every chain and one GPU remained free at launch.

**5k screen:** jitter is seed-sensitive but favorable in expectation. At
tokenizer seeds 1/2/3 its FID deltas versus v12 are
`+2.186/-4.228/-9.982`; corresponding KID deltas are
`+0.000825/-0.006676/-0.008735`. It wins two of three seeds and improves the
three-seed mean from FID/KID 32.281/0.02286 to 28.273/0.01800. However, the
seed-1 concordant regression is just beyond the declared two-FID boundary, so
that arm stops at 5k and the conservative global replacement condition is not
yet met. The prior-seed-2 seed-2 replication is strongly positive:
25.738/0.01384 versus v12 32.453/0.02258. Run 10k for tokenizer seed 3 and
both prior-seed-2 arms. Exact screen values are in
`reports/2026-08-26_autoencoder_program/phase_c_jitter_confirmation.json`.

**10k outcome and selection:** the positive directions strengthen. At
tokenizer seed 3, jitter reaches FID/KID **25.045/0.01798** versus v12
**35.750/0.02762**, deltas -10.704/-0.00964. At tokenizer seed 2 with prior
seed 2, jitter reaches **22.861/0.01315** versus **30.133/0.02244**, deltas
-7.272/-0.00929. Together with the prior-seed-1 seed-2 gain, this establishes
that the effect is neither one-prior-seed luck nor one-tokenizer-seed luck.
It is still tokenizer-seed dependent because seed 1 regresses. Honor the
conservative gate: retain v12 as the global control, while making
deterministic sigma-0.05 decoder jitter the leading expected-value design for
the next representation experiments. Always report the interaction rather
than claiming universal dominance.

### Register formation x decoder jitter factorial

Test whether v13's stable true register-token formation and Phase C's strong
decoder-jitter result combine constructively. Train three new `64x16`
tokenizers at seeds 1/2/3 with seven patch blocks, one bidirectional
patch/register block plus the parameter-matched register adapter,
deterministic latents, and decoder-only latent-RMS-scaled Gaussian jitter
`sigma=0.05`. Hold the 15k tokenizer and matched 60k prior-seed-1 protocols
fixed.

This completes the two-factor table: v12 is residual/hard-VAE, v13 is
register/hard-VAE, v23/v20/v24 are residual/jitter, and v25 is
register/jitter. Reconstruction and latent statistics remain diagnostics and
the existing permissive codec-health veto is unchanged. Generation FID/KID
selects.

Each seed receives FID/KID-5k. Compare first against its exact-seed residual-
jitter control: 29.321/0.01967, 27.743/0.01603, and 27.755/0.01830 for seeds
1/2/3. Advance a v25 seed to 10k if it improves FID, lies within two FID of
that control, or FID/KID disagree. If seed 1 advances, also evaluate its v23
residual-jitter control at 10k. Register+jitter becomes the expected-quality
lead only if all three seeds advance, it wins at least two, improves mean FID
by at least two with mean KID agreeing, and has no greater-than-two-FID
regression. It becomes the stability lead if worst-case FID improves by at
least two with mean FID cost below one and mean KID non-worse. Otherwise retain
residual+jitter as the expected-value lead and v13 hard-VAE as the stability
control.

Launchers: `scripts/run_register_jitter_factorial.sh {s1|s2|s3}` and
`scripts/run_register_jitter_factorial_10k.sh {s1|s2|s3|residual_s1}`.

Launch record (2026-09-03 01:34 UTC): all three supervisor-owned chains
acquired lifetime `gpu-claim` locks and began with finite optimization records
at roughly 4.2k images/s after compilation. Tokenizer W&B runs are
`0zoq4bhb`, `17pc720a`, and `e3d4l17z` for seeds 1/2/3. Five GPUs were free at
submission.

**5k screen:** register+jitter reaches FID/KID
26.736/0.01802, 30.863/0.02061, and 29.141/0.02030 at seeds 1/2/3. Relative
to residual+jitter, the FID deltas are -2.585/+3.120/+1.386 and KID agrees in
all cases. Its aggregate mean/std/worst FID is 28.913/1.692/30.863 versus
28.273/0.741/29.321 for residual+jitter; mean KID is also worse,
0.01964 versus 0.01800. Seed 2 decisively fails and stops, so register+jitter
cannot become either the expected-quality or stability lead. Seeds 1 and 3
receive 10k checks under the declared gate; seed 1 also receives the missing
v23 residual-jitter 10k control. Exact values are in
`reports/2026-08-26_autoencoder_program/register_jitter_factorial.json`.

**10k outcome:** seed 1 confirms a real local register+jitter win at
23.972/0.01774 versus residual+jitter 26.705/0.01921, deltas
-2.733/-0.00147. Seed 3 confirms a mild loss at 26.579/0.02052 versus
25.045/0.01798, deltas +1.534/+0.00254. Together with the decisive seed-2 5k
loss, this rejects register+jitter as a global mean or stability improvement.
Jitter itself does transfer well to register formation relative to hard-VAE
v13, but the architecture-by-seed interaction remains. Keep residual+jitter
as the expected-value lead and v13 hard-VAE only as the stability control.

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

### Weak representation-regularizer screen

Start on residual+jitter tokenizer seed 1, the only seed where the selected
design lost to hard-VAE v12. Run two isolated, parameter-free penalties:

- v26 marginal Gaussianity: existing per-coordinate batch kurtosis-to-3
  penalty with weight `1e-4`;
- v27 slot utilization: new scale-invariant variance of sample-varying
  per-slot power with weight `0.002`.

The frozen v23 cache measures these raw penalties at 0.5273 and 0.03624, so
their mature loss contributions are approximately `5.27e-5` and `7.25e-5`—
roughly 7-10% of the typical terminal reconstruction loss near `7e-4`. This is
a deliberately weak intervention. Slot balancing centers each token-coordinate
over the batch before measuring power, so constant offsets cannot satisfy it;
normalization by mean slot power makes it invariant to the latent's global
scale. Do not combine the penalties in this screen.

Hold the v23 deterministic residual encoder, sigma-0.05 decoder jitter,
`64x16` shape, seed 1, 15k budget, and prior-seed-1 60k recipe fixed. Both
healthy codecs receive priors regardless of reconstruction ordering. Compare
against v23 FID/KID 29.321/0.01967 at 5k and 26.705/0.01921 at 10k. Advance
an arm to 10k if it improves FID, lies within two FID, or FID/KID disagree.
Only a 10k gain of at least two FID with KID agreeing earns tokenizer-seed-2/3
replication; nothing from seed 1 alone changes the global design.

Launchers: `scripts/run_representation_regularizer_arm.sh {marginal|slot}` and
`scripts/run_representation_regularizer_10k.sh {marginal|slot}`.

Launch record (2026-09-03 04:30 UTC): both supervisor-owned chains acquired
lifetime GPU locks and began finite training at approximately 4.2k images/s.
Marginal Gaussianity is W&B `gmudbr8y`; slot balance is `h43ihnea`. Their early
reconstruction trajectories match the control, and the regularizer terms are
small by design.

5k result: both arms are healthy and both clear the continuation gate. Marginal
Gaussianity reaches **27.199/0.01754** FID/KID, improving on v23 by
**2.122 FID and 0.00213 KID**. Slot balancing reaches **27.232/0.01915**,
improving by **2.090 FID and 0.00052 KID**. The named mechanisms are also
measurably active: marginal kurtosis penalty falls 11.1x, while slot-power
dispersion falls 90.8x and the slot-RMS range tightens to 0.937-0.965. These
diagnostics establish intervention fidelity, not selection. Advance both to
10k; only the predeclared >=2-FID gain with lower KID can earn seed-2/3
replication.

10k result: slot balancing is the sole replication winner at
**24.166/0.01817**, improving on v23 by **2.538 FID and 0.00104 KID**.
Marginal Gaussianity remains promising at **24.798/0.01756**, but its
1.907-FID gain misses the fixed 2-point gate by 0.093 and stops. Replicate slot
balancing at tokenizer seeds 2 and 3 with prior seed 1. Compare against the
paired residual+jitter controls v20 and v24; apply the same 5k continuation
rule. Global promotion requires at least two of three tokenizer-seed FID wins,
better mean FID and KID, and no concordant regression larger than two FID.

Replication launch (2026-09-03 14:28 UTC): both supervisor chains acquired
shared lifetime GPU locks and began finite training near 4.3k images/s. W&B
tokenizer runs are seed 2 `wi7qqx8s` and seed 3 `v2avcbeh`.

Replication 5k result: seed 2 reaches **26.743/0.01754** versus v20
27.743/0.01603, a 1.000-FID improvement with KID disagreement. Seed 3 reaches
**25.171/0.01665** versus v24 27.755/0.01830, a clear 2.584-FID and
0.00165-KID improvement. Both advance to paired 10k evaluations under the
predeclared continuation rule. Slot-power dispersion remains tightly
controlled at 0.000352/0.000426 and both codecs pass the health veto; neither
diagnostic selects the result.

**Final 10k result and promotion:** slot balancing reaches FID/KID
**24.166/0.01817**, **24.534/0.01765**, and **22.548/0.01617** at tokenizer
seeds 1/2/3, versus residual+jitter controls 26.705/0.01921,
25.353/0.01568, and 25.045/0.01798. FID improves at all three seeds by
2.538/0.819/2.497. KID improves at seeds 1 and 3 but regresses at seed 2.
Mean FID improves from 25.701 to **23.749** (-1.952), and mean KID improves
slightly from 0.017623 to **0.017328**. This passes every frozen promotion
condition: at least two FID wins, improved mean FID and KID, and no >2-FID
regression. Promote slot balancing at weight `0.002` into the leading
deterministic residual-pool + sigma-0.05 decoder-jitter design. Preserve the
seed-2 KID interaction as a caveat rather than claiming universal improvement.

### Prior-seed-2 robustness check

Use the already verified tokenizer-seed-2 slot cache and train only a fresh
prior with seed 2. Its exact paired v20 seed-2/prior-2 control already exists at
FID/KID 25.738/0.01384 (5k) and 22.861/0.01315 (10k). Advance the candidate
from 5k to 10k if FID improves, is within two points, or FID/KID disagree.
At 10k, improving both metrics is full confirmation; improving FID with worse
KID is FID-only confirmation; a sub-two-point or metric-discordant FID loss
marks prior sensitivity; a >=2-FID loss with worse KID revokes default
promotion. This isolates prior stochasticity without another tokenizer.

Launch record (2026-09-03 16:18 UTC): the supervisor-owned chain acquired a
shared lifetime GPU lock and entered finite prior training at roughly 21.5
steps/s. W&B run: `ptfm3rzt`.

5k result: slot balance reaches **25.708/0.01648** versus the prior-seed-2
control's 25.738/0.01384. FID is effectively tied (-0.030) while KID worsens
by 0.00263. This is not confirmation, but it advances to 10k under the frozen
within-two/metric-disagreement rule.

**10k outcome:** slot balance reaches **23.375/0.01640** versus
**22.861/0.01315**, losing 0.514 FID and 0.00325 KID. This is the predeclared
prior-sensitive outcome, not the >=2-FID concordant loss required to revoke
default promotion. Across the two priors at tokenizer seed 2, FID is effectively
neutral on average (-0.152) while KID is worse by 0.00261. Keep slot balancing
as the leading expected-value modifier because the three-tokenizer-seed gate
passed, but do not describe it as universally robust; its benefit is strongly
seed-interactive.

### Roadmap after the robustness check

1. **Restrained decoder objectives:** with encoder formation stabilized, isolate
   a small frequency-aware reconstruction term from a small perceptual term.
   Hold the selected encoder, latent shape, jitter, slot penalty, decoder, and
   prior recipe fixed. Reconstruction remains a health veto; matched-prior
   FID/KID selects.
2. **Grouped explicit information allocation:** only if variable-rate or
   interpretable partial decoding is valuable, use 4-8 additive residual/DoG
   groups with cumulative low-pass targets. Pair any block-causal register mask
   with an innovation constraint; causality alone does not create hierarchy.
3. **Latent factorization under the stabilized baseline:** revisit nearby
   fixed-budget shapes such as `32x32`, `64x16`, and `128x8` after closing the
   objective test. Slot balancing may change the earlier token-count/channel-
   width tradeoff, but the established `64x16` point remains the control.
4. **Decoder capacity/locality:** test only after an objective or hierarchy arm
   gives a representation-side gain, so encoder and decoder effects remain
   attributable.

Deprioritize marginal kurtosis, static token-magnitude schedules, the tested
tokenwise time warps, direct `2x2` patching, and register+jitter without a new
mechanistic hypothesis; each has already missed a declared gate.

### Learned latent-factorization screen (predeclared 2026-09-03)

The decoder-objective screen is closed before changing shape. Retrain native
`32x32` and `128x8` tokenizers at the same 1,024-coordinate capacity against
the established native `64x16` v27 seed-2 control. Hold residual `e7+p1`,
deterministic decoding with sigma-0.05 training jitter, slot balance `0.002`,
pixel MSE, optimizer, batches, steps, and matched prior seed fixed. Do not carry
radial or LPIPS into this screen.

This is not a rerun of the completed exact-cache reshapes. Those held the v8
representation and decoder fixed and found FID 43.12/29.925/32.28 for
`32x32/64x16/128x8`; the new screen permits the encoder and decoder to organize
natively around each factorization. That answers whether learned geometry can
compensate for the prior's literal-layout preference.

Reconstruction remains a health veto. Both healthy arms receive 60k matched
priors. Advance 5k to 10k when FID improves, lies within two points of the v27
control, or FID/KID disagree. Only a concordant 10k gain of at least two FID
earns tokenizer-seed-1/3 replication; otherwise retain `64x16`.

Preflight: both full 60M-parameter tokenizers pass batch-512 GPU
forward/backward smokes at 18.9/23.9 GiB peak allocation for `32x32/128x8`,
and 20 focused tokenizer tests pass. The tokenizer parameter differences from
the 60,048,576-parameter control are -0.027% and +0.095%; hidden architecture
and all train settings remain fixed.

Launch record (2026-09-03 22:55 UTC): both supervisor-owned chains acquired
shared lifetime GPU locks and began tokenizer training without contention.
Tokenizer W&B runs are `bm5ih5cx` (`32x32`) and `y67pbcqn` (`128x8`).
After compilation they entered finite training near 4.64k and 3.87k images/s,
respectively.

5k gate (2026-09-04): both tokenizers pass the codec-health veto. `32x32`
has PSNR 34.75 dB, clean rFID 6.60, and flat effective rank 266.62; `128x8`
has 37.38 dB, 5.19, and 374.75, versus control rank 377.43. Generation sharply
rejects `32x32`: **44.324/0.03289**, worse than control by 17.581 FID and
0.01536 KID. Native end-to-end training therefore does not compensate for its
literal prior layout penalty. `128x8` reaches **27.891/0.01742**, 1.148 FID
worse but 0.00012 KID better than control. It advances to 10k under both the
frozen within-two and metric-disagreement clauses. Reconstruction did not make
either decision.

**10k outcome:** `128x8` reaches **25.484/0.01741** versus
**24.534/0.01765** for `64x16`, repeating the 5k pattern with +0.950 FID and
-0.00024 KID. It misses the concordant two-FID replication gate and its prior
trains at only 10.77 steps/s versus 19.99 for the control. No arm is replicated
or promoted. Retain native `64x16`: `32x32` is decisively destructive, while
`128x8` costs roughly twice as much in prior training for slightly worse FID.

### Decoder-objective screen (predeclared)

Use tokenizer seed 2 and the promoted residual+jitter+slot configuration. Hold
all architecture, optimization, latent, and prior details fixed against v27;
add one decoder-side objective per arm:

- v28 radial: per-image/channel radial log-power error with a target-relative
  `1e-3` power floor, weight `6e-5`;
- v29 perceptual: frozen LPIPS-0.1 AlexNet features, weight `0.02`.

Raw complex FFT MSE is excluded because an orthonormal FFT makes it equivalent
to pixel MSE. The signal-relative radial floor prevents near-zero/noise-floor
bands from dominating gradients. On 128 frozen v27 reconstructions, the two
rounded coefficients contribute approximately 9.94% and 9.34% of the pixel-MSE
output-gradient norm, while their scalar losses are only about 1.34% and 0.75%
of MSE. Do not combine them in this screen.

Both healthy tokenizers receive matched prior-seed-1 60k runs regardless of
reconstruction ordering. Compare to v27 seed 2 at FID/KID
26.743/0.01754 (5k) and 24.534/0.01765 (10k). Advance when FID improves, is
within two points, or FID/KID disagree. Only a 10k gain of at least two FID
with lower KID earns tokenizer-seed-1/3 replication. Twenty focused tests, two
CPU end-to-end smokes, and two full-architecture batch-512 GPU smokes pass.

Launch record (2026-09-03 19:19 UTC): both supervisor-owned chains acquired
shared lifetime locks and entered finite tokenizer training. Radial runs near
3.6k images/s and LPIPS near 4.3k images/s after compilation. W&B tokenizer
runs are `xlu5z1n6` and `5471b4xt`.

5k gate (2026-09-03 21:00 UTC): both tokenizers are healthy, with full-prefix
PSNR 38.10/38.31 dB and clean reconstruction FID 5.09/5.21 for radial/LPIPS.
The auxiliaries remain weak at the end of training: their weighted scalar
contributions are approximately 1.55%/1.13% of pixel MSE. Radial reaches
**25.847/0.01571**, improving the control by 0.896 FID and 0.00183 KID. LPIPS
reaches **27.613/0.01923**, worsening by 0.870 FID and 0.00169 KID, but remains
inside the frozen two-FID continuation margin. Both advance to the 10k read;
this is a generation-metric decision, not reconstruction ranking.

**10k outcome:** radial reaches **23.686/0.01586** versus the control's
**24.534/0.01765**, a concordant improvement of 0.849 FID and 0.00179 KID.
The nearly identical 5k and 10k FID deltas make this useful directional evidence,
but it is below the frozen two-FID threshold for seed replication. LPIPS reaches
**24.771/0.01881**, losing 0.237 FID and 0.00116 KID. Neither arm is promoted or
replicated. Retain radial log-power as a plausible weak modifier; reject LPIPS
for this configuration, and do not silently combine either objective with the
next experiment.

## Promotion protocol

1. 15k tokenizer codec-health screen; reconstruction is not a latent-quality
   ranking metric.
2. Full-test clean rFID/PSNR and sigma `0/.05/.10/.20/.40` sensitivity.
3. Matched 60k bidirectional joint prior for every healthy planned arm.
4. Paired FID/KID-5k screen, then 10k for a promising difference.
5. Independent tokenizer and prior seeds before an architecture-level claim.
6. Preserve optimizer-bearing resumes, final metrics, and W&B checkpoint
   artifacts; append every decision to `EXPERIMENT_JOURNAL.md`.
