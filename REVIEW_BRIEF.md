# AFIG — external review brief (as of 2026-08-22)

Repo: `/workspace/AFIG`. Everything below is measured on CIFAR-10 unless stated.
We want adversarial review: design critique, mechanism analysis, and ideas.
Please challenge premises, not just implementation.

---

## 1. The goal

Learn a **progressive continuous image tokenizer**: an autoencoder producing an
*ordered* sequence of whole-image registers, where a prefix of the sequence is
itself a valid (coarser) encoding of the image. Then learn a generative prior
over that latent sequence and decode.

The bet is that an ordered latent is *more diffusable* than an unordered one:
that a code whose registers resolve in a consistent order should be easier for a
diffusion/flow prior to model, in the same way that natural images are easy
because their low frequencies dominate and resolve first.

Configuration throughout: **64 registers x 16 channels** (1,024 scalars per
32x32 image), cross-attention-only pooling, nested-prefix training objective.

## 2. Method

**Tokenizer.** Encoder produces 64 ordered registers. Nested-prefix objective:
reconstruct from every prefix length, so register 0 alone must reconstruct
something, registers 0..1 better, etc. `progressive_tokenizer/model.py`.
Optional variational mode (we cache the posterior *mean*, but the decoder is
trained on sampled z).

**Priors** (all rectified flow, path `z_t = (1-t)eps + t*z`, target `z - eps`,
50-step Heun sampling; all ~70.3M params at width 512, depth 12):
- **joint** — bidirectional DiT, one shared timestep for all registers.
  `progressive_tokenizer/joint_flow.py`
- **rolling / diffusion forcing** — per-register time
  `t_i = clamp(frontier - i/overlap, 0, 1)`; a frontier sweeps the sequence.
  `progressive_tokenizer/rolling_flow.py`
- **autoregressive** — causal trunk + per-token conditional flow head.
  `progressive_tokenizer/autoregressive_flow.py`

**Metric protocol (important).** Decoded **FID** on 5,000 samples against the
CIFAR-10 test set decides everything. Reconstruction PSNR/dB is treated as a
capacity check only, never a selection metric. Latent MSE was retired for cross-model decisions. **All results are
single-seed.**

**Protocol-noise correction (2026-08-23).** The long-quoted "+-0.3 FID protocol
noise" was measured by *re-running the same eval with the sampler hard-seeded*
(`--seed 54321`), so it captures only bf16/atomic nondeterminism, not decision
noise. Independent-seed FID-5k variance at these levels is realistically
+-1-2 FID, and training-seed variance is on top of that. Every gap below ~2 FID
in this document should be read as unresolved, including several adjacent pairs
in the v5 matrix.

## 3. Results so far

### Latent shaping (v5 matrix — 5 tokenizers, identical architecture and recipe, judged by an identical 60k-step joint prior)

| arm | FID | eff-rank | tok-profile-corr | chan-order-consist. | PSNR |
|---|---:|---:|---:|---:|---:|
| vae (KL 1e-4)* | **35.85** | 100.7 | 0.312 | 0.537 | ~33.9 |
| energycv (kurtosis->3 penalty) | 39.03 | 26.3 | 0.172 | 0.551 | 34.02 |
| ramp (sigma-ramp latent noise) | 40.72 | 519.6 | 0.725 | 0.508 | — |
| det (control) | 41.16 | 61.0 | 0.587 | 0.540 | 32.88 |
| frontier (frontier-noise) | 47.60 | 48.8 | **0.908** | 0.514 | 31.41 |

- Token-axis *schedule consistency* rank-correlates with FID at Spearman +0.8
  (wrong sign). **Under-powered**: n=5 gives exact permutation p=0.13
  two-sided, and one rank swap destroys it. `tok-profile-corr` is also largely
  explained by each arm's Gaussian surrogate (frontier 0.908 raw vs 0.937
  surrogate -- *less* schedule-consistent than its own second-order surrogate),
  so the correlation is mostly with second-order energy-profile shape, which
  this same document calls gauge. Both arms that shaped the schedule did fail;
  the statistic supporting it is weak.
- `channel_eigen_order_consistency` sits at 0.508-0.551 in all five arms.
  **This is not evidence of anything.** 0.5 is not the chance level: the
  statistic compares lambda_k*chi2_1 vs lambda_k+1*chi2_1 on single samples,
  which for a Gaussian is (2/pi)*arctan(sqrt(lambda_k/lambda_k+1)) -- given
  these near-degenerate adjacent eigenvalues its entire feasible range is
  ~0.50-0.60 for *any* distribution. Every arm matches its own Gaussian
  surrogate to within +-0.005 (vae 0.5372 vs 0.5391; det 0.5404 vs 0.5412;
  energycv 0.5505 vs 0.5540; ramp 0.5081 vs 0.5034; frontier 0.5138 vs 0.5101),
  and the cross-arm spread is fully explained by spectral flatness. "No
  intervention moved it" was guaranteed a priori by a near-zero-power test.
- Reconstruction dB is non-predictive: energycv has the best PSNR and loses 3.2
  FID to vae.

***The "vae" arm is effectively deterministic** -- and that is the normal LDM
regime, not a defect. Verified on `checkpoint_final.pt`: 99.97% of posterior
log-variances sit at the -8.0 clamp, sigma is a constant 0.0183 against
posterior-mean RMS 1.096 (~1.7% relative noise), and the KL's sigma-term is a
constant 3.5001 (the analytic floor). SD's VAE uses KL ~1e-6 with effectively
deterministic posteriors, so this is the design point rather than a collapse to
be fixed. The narrower consequences: the arm is "small fixed noise + mu-L2",
not rate-constrained coding; any KL-weight sweep is degenerate (every weight
losing the race gives the same sigma); `kl_per_dim` is not a rate monitor (3.5
of 4.1 nats is constant); and the clamp passes zero gradient past the floor, so
sigma can never recover (softplus bound added for future runs).

### Latent higher-order structure

Raw channel coordinates are near-Gaussian (kurtosis ~3.0-3.2) and the
fourth-moment pair slice `E[z_i^2 z_j^2]` sits within 11-15% of its Gaussian
value `L_ii L_jj + 2 L_ij^2` — so there is little for an energy-covariance
regularizer to bite on. But rotating into the channel eigenbasis exposes
elevated kurtosis, loosely increasing with rank. **Corrected figures**: the
6.05 maximum is the *det* arm's (vae's max is 5.78), and the earlier
"top-4 2.8-3.7 / bottom-4 5.2-5.6" range matched no single arm -- per arm the
top-4 are det [2.84, 2.22, 2.48, 3.50], vae [2.49, 4.58, 4.58, 3.13], energycv
[3.33, 3.72, 3.35, 2.43]. "Rank-ordered" is a loose trend, not monotone (vae's
rank-2 direction is already 4.58). **Omitted and awkward**: frontier's
eigen-kurtosis is 36-88 at essentially every rank, and frontier is the worst
FID arm -- that datapoint is not reconciled with "the non-Gaussianity lives in
the low-variance directions". Pooling across 64 token positions with
heterogeneous covariances also manufactures kurtosis by scale-mixing; per-token
estimates stay elevated, so the qualitative claim survives, but with an
unquantified max-over-16-directions selection bias at n=20k.

### Decoder sensitivity (reconstruction FID vs injected latent noise)

| arm | gen FID | s=0 | 0.05 | 0.1 | 0.2 | 0.4 | x@0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vae | 35.85 | 6.9 | 8.0 | 13.2 | 39.2 | 124.4 | **1.91** |
| energycv | 39.03 | 7.1 | 12.5 | 35.7 | 112.9 | 214.0 | **5.04** |
| det | 41.16 | 8.2 | 11.3 | 25.0 | 81.3 | 189.9 | 3.04 |
| ramp | 40.72 | 7.4 | 7.7 | 8.7 | 13.4 | 30.1 | **1.17** |
| frontier | 47.60 | 9.6 | 14.5 | 34.1 | 96.1 | 155.2 | 3.56 |

Two factors, not one: energycv buys reconstruction with brittle directions
(steepest curve, mid FID); ramp has an almost noise-immune decoder and still
loses (robustness alone is insufficient). vae is the only arm strong on both
robustness and density modelability.

### Generation engines (all on the identical vae cache, 60k steps)

| engine | FID |
|---|---:|
| joint | **35.85** |
| roll64 + jitter/prefix-noise | 48.35 |
| roll64 baseline | 51.69 |
| roll8 + independent-t 0.25 | 57.26 |
| roll8 + overlap-jitter | 62.78 |
| roll8 + jitter/prefix-noise | 67.45 |
| roll8 baseline | 70.85 |
| AR + history-noise/reliability | 77.07 |
| AR + RF head | 79.38 |

**The central diagnostic.** Per-register MSE at end of training, and weighted by
each register's energy share (16.5 / 13.0 / 26.1 / 22.8 / 21.5 % for the bands):

| model | 0-8 | 8-16 | 16-32 | 32-48 | 48-64 | energy-wtd | FID |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 0.563 | 0.640 | 0.677 | 0.632 | 0.574 | 0.621 (worst) | **35.85** |
| roll64 | 0.970 | 0.801 | 0.554 | 0.317 | 0.190 | 0.522 | 51.69 |
| roll8 | 0.978 | 0.571 | 0.390 | 0.266 | 0.166 | **0.434 (best)** | 70.85 |

Rolling achieves *lower* training loss and *far worse* samples. Its late-register
accuracy is conditional on ground-truth clean context; at inference that context
is the model's own output, which explains only about half its variance
(MSE ~0.98 against target variance 2.0). Every engine orders monotonically on
"how much it leans on ground-truth clean context", and that ordering is exactly
reversed between loss and FID.

Consequences we drew: rolling/joint training losses are not comparable; the
active-register loss mask is correct (unmasking dilutes the useful gradient ~8x
because ~87% of registers sit at degenerate schedule endpoints); causal
attention hurts, and hurts more the nearer the schedule is to joint.

**Every intervention that widened the training distribution helped**
(independent per-register times -13.6 FID, overlap jitter -8.1, context
corruption -3.3). **Every structural change hurt** (unmasking +3.1, causal
+9.9 at overlap 8 and +54 at overlap 64). Ordered *generation* has never beaten
the plain bidirectional prior.

## 4. The current hypothesis (where we want the hardest scrutiny)

Measured on the vae cache: register 0 carries only **1.59x** the energy of
register 63, and under RF (`t_cross = 1/(1+sqrt(lambda))`) all 64 registers
cross the noise floor within **10.4% of the schedule**. Natural images have a
~10^3 low/high-frequency energy ratio, spreading crossings over ~94% of the
schedule.

**Corrected 2026-08-23 (two independent reviews + recomputation).** The
premise of this section is wrong, and two numbers in it were wrong.

*The code is not missing an SNR spectrum.* The 10.4% figure is the
**register-index marginal** -- it averages 16 channels into one energy per
register, discarding the spectrum diffusion actually sees. In the flattened
1024-dim eigenbasis, after the prior's scalar normalisation, the same cache has
lambda_max/lambda_min = 3.3e5 and a **crossing spread of 0.905**, against ~0.92
for CIFAR pixels. Per-coordinate spread within a register is already 0.35. The
code is SNR-ordered in its covariance eigenbasis; it is only **not
token-aligned**, and flat-MSE training is invariant to that alignment except
through per-token layers, Adam, and loss weighting.

*Arithmetic error.* For energy ratio R the maximum crossing separation is
`(R^0.25 - 1)/(R^0.25 + 1)`. R = 10^3 gives **0.698**, not the 0.94 claimed
here; 0.94 needs R ~ 1.1e6. The natural-image 10^3 *band* ratio and the ~0.92
*per-coordinate* spread (driven by a ~3.7e5 DC-to-Nyquist ratio) were
conflated.

*Reporting error.* `rescale_cache.py` printed the profile multiplier as the
energy ratio, but it multiplies the 1.59 the cache already had. The arms
labelled 8x / 64x / 996x are really **12.8x / 102x / 1588x**.

*Likelier mechanism for the observed loss.* The rescaling degrades
conditioning rather than merely reweighting: effective rank falls
100.0 -> 84.8 -> 58.4 -> 31.8 and the condition number rises 3.3e5 -> 7.0e6
across alpha = 0 / 0.25 / 0.5 / 0.83. alpha = 0.5 measured **FID 40.93** vs
35.85 flat.

**So this sweep tests "does token-axis alignment help this architecture", not
"does adding a missing spectrum help", and a negative result does not refute
the ordering thesis.** If a mild alignment test is still wanted, target the
*actual* post-scale ratio at alpha ~ 0.1-0.2.

For a genuine SNR-only arm, do **not** use `t_i = clamp(t + delta_i, 0, 1)`:
that is the rolling frontier schedule we removed, with the same exposure
pathology, and it fails to map noise to data at common endpoints. Use the
endpoint-preserving warp `phi_i(t) = a_i*t / (1 - t + a_i*t)`, train on
`(1-phi_i)eps + phi_i*z` predicting `z - eps`, and sample with increments
`d phi_i`. That reproduces the scaled-data SNR curve exactly without touching
latent magnitudes.

The thesis quantity that scaling provably cannot touch -- and that has never
been measured -- is **axis B**, the conditioning-gain curve (zero for a
stationary Gaussian by construction). It is measurable on the existing cache
with no training run. Measure it before any thesis-level conclusion.

## 5. Questions we would most like attacked

1. **[partly answered]** The rescale is not a no-op (scalar-only
   standardisation confirmed), but it conflates schedule shift with loss
   reweighting. The clean decomposition is C' = magnitude scaling `a_i` plus a
   compensating loss weight `w_i = 1/a_i^2`, which cancels the implicit
   reweighting and isolates the pure schedule shift in decoded units. A
   per-register *time offset* is not a viable alternative: `t_i = clamp(t +
   delta_i, 0, 1)` is literally the rolling frontier schedule we removed, with
   the same exposure pathology.
2. "Ordered code helps (5.3 FID)" is **unsupported on both ends**: the winning
   arm is not the intervention it is named after (see above), and the two arms
   that actually shaped ordering (ramp, frontier) tied or lost to the det
   control. All five arms share the nested-prefix objective, so the matrix
   cannot speak to whether progressiveness helps at all. The missing control is
   a **no-prefix tokenizer at matched everything**; if it also reaches ~35.85,
   the progressive premise contributes nothing measurable to diffusability.
   Ordered *generation* still costs 16-44 FID, so "generate jointly" stands --
   but a progressive code justified only by full-length FID is dead weight by
   construction, and would need a metric that rewards it (FID vs prefix length
   under one joint prior).
3. **[malformed, retired]** The metric has ~zero power (see above). The
   replacement is the theory's **axis B** -- the conditioning-gain curve: how
   much does knowing the already-resolved directions reduce uncertainty about
   the rest? It is zero for a stationary Gaussian by construction, is exactly
   what scaling cannot touch, is measurable on the existing cache with no
   training run, and has never been measured. Do this before any
   thesis-level conclusion.
4. The heavy tails sit in the low-variance channel eigendirections. Harmful
   structure to remove, or useful sparse detail to preserve?
5. We have **no external baseline** -- no pixel-space diffusion at matched
   budget. The latent is only 3:1 compression, so the honest comparison is the
   same 64-token DiT on 4x4 pixel patches; literature pixel CIFAR sits at
   FID 2-3 at comparable budgets, i.e. the representation tax may be ~10x the
   largest effect this whole matrix measured. Note `train_continuous.py`'s
   FID uses the CIFAR **train** split as reference while every progressive
   evaluator uses the **test** split -- those numbers are incommensurable and
   must never share a table.
6. Anything in `rolling_flow.py` / `joint_flow.py` / `train_progressive_*.py`
   that looks wrong. We have already found and fixed: an amplitude treadmill
   from a detached RMS reference, prefix noise silently enlarging the
   supervision set, and a sampler overlap override that did not reach
   `local_times`.

## 6. Repo pointers

- models: `progressive_tokenizer/{model,joint_flow,rolling_flow,autoregressive_flow}.py`
- trainers: `train_progressive_{tokenizer,joint_flow,rolling_flow,ar_flow}.py`
- eval: `evaluate_progressive_joint_flow.py`, `live_evaluation.py`
- analysis: `scripts/{analyze_axis_scorecard,channel_pair_slice,decoder_sensitivity}.py`
- write-ups: `reports/*/` — especially `2026-08-19_v5_shaping_verdict/` and
  `2026-08-21_rolling_exposure_diagnosis/`
- tests: `tests/test_progressive_{rolling,joint}_flow.py`, `tests/test_ar_variants.py`
  (6 legacy test modules fail to import — they reference `frequency` and
  `diffusion_decoder` from an earlier project phase; ignore them)
