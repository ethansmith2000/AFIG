# External adversarial review of REVIEW_BRIEF.md (2026-08-23)

Reviewer: Claude (Fable 5), full-repo review with three parallel code-audit passes
(tokenizer side, prior/eval side, analysis side) plus independent measurements run
on the archived v5 caches. Scope: the brief's six questions, the §4 hypothesis, and
the in-flight magnitude sweep (pow05 done at 40.93; pow025 running; pow083 queued).

Everything below excludes the three already-known bugs (amplitude treadmill,
prefix-noise supervision enlargement, rolling overlap override).

---

## Verdict in one paragraph

The engine-comparison work (rolling/AR vs joint) is the strongest part of the
project — clean design, correct diagnosis, right conclusion. But the two premises
now carrying the program are both unsupported by the data on hand: (1) the §4
hypothesis mis-states what the code lacks — measured in the basis that
rotation-invariant RF training actually responds to, the vae latent's noise-floor
crossings already span 92% of the schedule, statistically indistinguishable from
CIFAR pixels — so the magnitude sweep is testing token-axis *alignment* plus an
implicit loss reweighting, not "giving the code an SNR order"; and (2) "ordered
code helps, 5.3 FID" is not established — the winning arm's intervention is not
ordering (all five arms share the nested-prefix objective), and a code audit shows
the vae arm's posterior collapsed to the logvar clamp, so even the "KL smoothing"
attribution is wrong in mechanism. Meanwhile there is no pixel baseline for a
latent that compresses only 3:1, and literature values suggest the representation
tax is ~10× larger than every effect measured in the matrix. The single most
valuable next runs are the E2 pixel control and a no-prefix tokenizer control, not
more shaping arms.

---

## 1. The §4 hypothesis measures the wrong basis (Q1, and the running sweep)

**New measurement** (this review; reproducible in ~20 lines on
`tokenizer_runs/v5-vae-kl1e4-s1/latents_final_original_flip.pt`):

| quantity | vae latent (1024-D eigenbasis) | CIFAR-10 pixels (3072-D eigenbasis) |
|---|---:|---:|
| eigenvalue range | 226 → 0.0004 (5.7×10⁵) | 4.4×10⁷ |
| t* = 1/(1+√λ) full spread | **91.8%** of schedule | 96.3% |
| middle-98% spread | **74.4%** | 76.3% |

The analysis-side audit independently found the per-coordinate (unrotated)
crossing spread is already 35%. The brief's 10.4% is real but it is a statement
about the **register-index marginal** — i.e., about whether the SNR ordering is
*aligned with the token axis* — not about whether the code has SNR ordering. By
the project's own theory doc (§1, rotation invariance of forward process + flat
MSE), the training problem is invariant to that alignment; alignment only matters
through (a) token-diagonal machinery (per-token layers, LayerNorm, Adam's
per-coordinate second moments), (b) the implicit loss weighting, (c)
discretization allocation. Those are second-order effects, and 40.93 vs 35.85 at
α=0.5 is consistent with them being net-negative once the weighting distortion is
included.

Consequences for reading the sweep:

- **Do not conclude "the ordering thesis is dead" from a null/negative sweep.**
  The correct conclusion available from this instrument is: "token-axis SNR
  alignment, welded to a_i²-implicit loss reweighting, does not help." The thesis
  quantity that was never touched — and never measured — is **axis B**
  (cross-stage predictive value, theory doc §4). Per-register scaling provably
  cannot create axis-B structure: conditional dependence is invariant to
  per-coordinate scaling. Your own §4 stationary-Gaussian argument says order
  without conditioning value is worthless. The sweep can therefore kill the
  *instrument*, not the *thesis*; the thesis dies (or lives) on an axis-B
  measurement.
- **Per-sample ordering** (this review, computed on the rescaled caches):
  band-level per-sample P(E_k > E_{k+1}) is 0.40–0.95 on the original cache,
  0.93–1.00 at α=0.83 — so pow083 does replicate image-like *band-level* axis-A
  consistency (images: 0.99–1.00). Adjacent-register ordering stays near chance
  (0.59) because adjacent scale ratios are ~1%. So if pow083 also loses, the
  token-axis schedule story is exhausted at both population and per-sample level.
- **Mechanistic forecast for large α**: register 63's target is ≈ −ε; the model's
  cheap optimum there is v ≈ −z_t/(1−t), whose ODE integrates to z₁ ≈ 0 — the
  register decodes (after inverse scale) to nothing. So α→∞ degrades gracefully
  into **prefix truncation**, and the sweep interpolates joint → prefix-only
  decoding. You can predict the α=0.83 ceiling from the existing prefix-recon
  curves. A secondary architectural penalty: the pre-LN trunk + shared final
  layer must emit per-position output scales spanning 996×, which weight-shared
  LN'd transformers do awkwardly.
- **The corrected gauge argument cuts backwards too**: the v5 verdict explained
  frontier's failure as "buying a quantity that was free (gauge)". If finite
  capacity makes rescaling a real intervention (the premise of this sweep), then
  the profile was not free for frontier either, and its 6.4-FID loss needs the
  CV²=49.6 mechanism, not the gauge argument. One of the two documents should be
  amended for consistency.

**Q1 direct answers.** The rescale argument is right that it is not a no-op; it is
wrong about *why* it might help. Profile choice is second-order once you see the
eigenbasis numbers. On avoiding starvation: the surgical fix is a per-register
loss weight **w_i = 1/a_i²** on the scaled loss — this exactly cancels the
implicit variance weighting, leaving pure schedule shift *in original decoded
units*. That arm (call it C′) is a one-line loss change on the running setup, and
it is the cleanest "SNR ordering alone" test in the A/B/C family.

## 2. Arm B as specified re-imports rolling's pathology (the A/B/C plan)

`t_i = clamp(t + δ_i, 0, 1)` **is** the rolling frontier schedule
(δ_i = −i/overlap, frontier = t). At sampling, registers with positive offsets hit
t_i = 1 mid-trajectory and freeze; later registers then condition on frozen model
output — exactly the exposure mechanism the 2026-08-21 diagnosis identified as
rolling's killer, plus the degenerate-endpoint gradient dilution the same report
quantified (~87% of registers at endpoints for roll8). The overlap sweep already
sampled this axis end to end: spread 0 (joint) 35.85 → spread 1 (roll64) 51.69,
monotone. Arm B at small spread is a new point on a measured monotone curve.

If B is still wanted, change the schedule family: a **smooth monotone per-register
time warp** t_i = f_i(t) with f_i(0)=0, f_i(1)=1 (e.g. t^{γ_i}, or a logistic
shift in log-SNR). This shifts each register's crossing with no clamping, no
freeze-out, no degenerate endpoints, and no self-conditioning on frozen output.
That is precisely the per-frequency-schedule construction of Blurring Diffusion
(Hoogeboom & Salimans 2023; also Rissanen et al.'s inverse heat dissipation) — and
their pixel-space result is the most important external datapoint for this whole
program: **explicit coarse-to-fine per-frequency schedules bought little or
nothing over isotropic diffusion on CIFAR**, in the modality where the ordering
premise is maximally true. Expectations for the latent version should be set
accordingly.

Cheapest version of the decomposition: skip B's new machinery entirely and run
A + w_i = 1/a_i² (C′ above). If C′ ≈ joint and A < joint, the entire A deficit is
the loss weighting; if C′ > joint, schedule shift alone helps — either way the two
welded knobs separate with zero new sampler code.

Bug directly in this path: `PerTokenAdaLNZeroBlock` (joint_flow.py:350) reads
`config.causal`, which `JointFlowConfig` no longer defines — the first arm-B
instantiation raises AttributeError. Fix or give it an explicit constructor arg.

## 3. "Ordered code helps (5.3 FID)" is not established (Q2, first half)

Three independent holes:

1. **The 5.3 FID is vae − det, and vae is not an ordering intervention.** Both
   arms that explicitly shaped ordering (ramp, frontier) tied or lost against the
   det control. The nested-prefix objective — the thing that makes the code
   "ordered" — is common to the arms, so the v5 matrix never varied it. The
   matrix shows *latent smoothing* helps; it says nothing about ordering.
2. **The vae arm is a collapsed pseudo-VAE** (tokenizer audit, verified
   empirically on `checkpoint_final.pt`): 99.98% of posterior logvars sit exactly
   at the −8.0 clamp (model.py:510), σ = e⁻⁴ ≈ 0.018 fixed, noise/signal ≈ 1.9%.
   The KL is a per-dim mean, so "KL 1e-4" is ~1e-7 per-image in the summed
   convention — collapse was guaranteed, and the clamp passes zero gradient once
   railed (a one-way trap). The winning intervention is actually "deterministic
   encoder + fixed 1.8% decoder-input jitter + weak L2 on μ", and its 5.3-FID
   mechanism is **not understood** — the decoder-robustness story (x@0.1 = 1.91)
   is a 6–12× extrapolation beyond the noise it trained at. Any KL-weight sweep
   in this regime is silently degenerate (every sufficiently small weight
   produces the same clamped σ).
3. **The control that would test the premise is missing**: same architecture,
   same (fixed) KL, `--objective` without the nested-prefix term. If that
   matches 35.85, the progressive premise contributes nothing measurable to
   diffusability. One tokenizer + one prior run — cheaper than any shaping arm
   and more informative than the rest of the α sweep.

## 4. Q2, second half: yes — generate jointly

The engine ladder is the most trustworthy result in the repo: monotone in
"reliance on ground-truth clean context", the exposure diagnosis is mechanistic
and confirmed by interventions (every distribution-widening fix helped, every
structural ordering move hurt), and it agrees with external evidence (diffusion
forcing/AR-diffusion help on temporally-causal data like video; per-frequency and
cascaded schedules ≈ neutral on single images; subspace diffusion modest). Ordered
*generation* of a whole-image latent has no remaining support here.

But note what "keep the progressive tokenizer, generate jointly" is then for:
progressive codes buy variable-rate decoding, coarse-first preview, adaptive
compute, editability — none of which FID measures. If decoded FID at full length
stays the only selection metric, the nested-prefix structure is dead weight by
construction (see §3.3's control). Either adopt a metric that rewards
progressiveness (e.g., FID-vs-prefix-length curve under one joint prior at
matched budget) or be explicit that ordering is a product feature, not a
diffusability feature.

## 5. Q5: the missing pixel baseline discounts everything, severely

The latent is 1,024 scalars for 3,072 pixels — **3:1 compression**. At that ratio
the pixel-space alternative is the *same* 64-token transformer on 4×4 patches
(48 dims/token vs 16). Literature pixel-space CIFAR at comparable or smaller
budgets: DDPM 3.17 (36M), EDM 1.79 (56M) FID-50k. Even discounting for FID-5k
bias, 60k steps, and no EMA-tuning, a matched pixel DiT should land far below 35.
The representation tax is plausibly ~10× larger than the largest effect the v5
matrix measured (5.3 FID). Until E2 runs, every intra-latent conclusion is
optimizing within a regime the trivial baseline may dominate outright.

Practicalities: `control_pixel_diffusion.py` is from the legacy phase (imports
`diffusion_decoder`, broken). The clean E2 is `train_progressive_joint_flow.py`
pointed at a patchified-pixel "cache" — same trainer, sampler, and evaluator.
Do **not** reuse `train_continuous.py`'s `evaluate_live` numbers: it builds its
FID reference from the **train** split (train_continuous.py:894) while the
progressive evaluators use the 10k **test** reference — the two are not on one
scale (prior/eval audit, confirmed).

Protocol noise: the ±0.3 figure is same-seed repeatability (sampler generator
hard-seeded to 54321; a rerun replays the same latents). Independent-seed FID-5k
noise at these levels is realistically ±1–2, and every headline number is also a
single *training* seed. Several v5 gaps (0.4–2.1 FID) sit inside that. "The
ordering is real at every gap" is not currently supported; re-evaluate at 2–3
sampler seeds (change `--seed`) for adjacent pairs, and put a second training
seed on vae and det.

## 6. Q3: the immovable metric is immovable by construction

Analysis audit, decisive: `channel_eigen_order_consistency` compares λ_k·χ²₁ vs
λ_{k+1}·χ²₁ per (sample, token); for the vae spectrum its Gaussian value is
0.5295 analytically, and its feasible dynamic range is ~0.50–0.60 **for any
distribution** (a 4× eigenvalue gap yields only 0.705; mid-spectrum eigenvectors
are rotationally ill-determined besides). Each arm's measured value equals its
*own Gaussian surrogate* within ±0.005 — the entire cross-arm spread 0.508–0.551
is second-order spectrum shape. "Pinned at chance across five objectives" was
guaranteed a priori and is therefore not evidence for the conditional-order
theory; §2 of the v5 verdict over-reads a near-zero-power test.

So Q3 is malformed as posed. Replace, don't move: (a) report latent-minus-
surrogate deltas for every scorecard metric as standard practice (the same audit
shows `tok-profile-corr` is also largely surrogate: frontier 0.908 vs surrogate
0.937 — frontier is *less* schedule-consistent than its own Gaussian); (b)
measure **axis B directly** — the conditioning-gain curve already specified in
the theory doc §4/§7: at matched t, error on unresolved directions with vs
without (nonlinear) conditioning on resolved content. That is the quantity the
entire ordering thesis rides on, and it has never been measured. Related: the
Spearman +0.8 "anti-prediction" is p = 0.13 two-sided at n = 5 — drop the claim
or present it as a trend.

## 7. Q4: the heavy tails look like sparse detail — measure before removing

Kurtosis 5–6 in low-variance eigendirections is Laplace-range: present-or-absent
sparse features. That is the signature of natural images themselves in any
bandpass/wavelet basis (kurtosis ≫ 6 is normal there), and images are the
existence proof that diffusion copes with kurtotic low-SNR directions — *when
they are strongly conditioned by resolved context*. So the default should be
preserve, and the deciding measurement is again axis B restricted to those
directions, plus an information test: decode with the bottom-k eigendirections
zeroed (or marginal-resampled) and read the reconstruction-FID cost. In-house
cautionary evidence already exists: energycv Gaussianized coordinate kurtosis,
gained PSNR, produced the most brittle decoder (x@0.1 = 5.04), and lost FID.

Data hygiene before this section circulates: the brief's specific numbers are
misquoted — 6.05 is the **det** arm's max (vae: 5.78); "top-4: 2.8–3.7" matches
no arm; and frontier's eigen-kurtosis of 36–88 across essentially all ranks is
omitted despite sitting awkwardly beside "the non-Gaussianity lives in the
low-variance directions." Also note the pooled estimator mixes 64 token
positions with heterogeneous covariances (scale-mixing manufactures kurtosis);
the per-token estimates stay elevated, so the qualitative claim survives.

## 8. New code findings (Q6) — beyond the three known bugs

| # | sev | finding | where |
|---|---|---|---|
| 1 | CRITICAL | vae posterior collapsed to logvar clamp; winning arm is not the named intervention; clamp passes zero gradient (absorbing). Fix: soft floor (e.g. −8+softplus), monitor σ, recalibrate kl_weight in summed convention; rerun arm | model.py:510, train_progressive_tokenizer.py:534-541 |
| 2 | HIGH | AR trainer checkpoint omits `token_scale`; evaluator reads it from the checkpoint → any AR magnitude run decodes un-inverted latents silently (`token_scale_applied: false` is the only trace). Fix before launching AR arms | train_progressive_ar_flow.py:233-250 vs train_progressive_joint_flow.py:210-231 |
| 3 | HIGH | ±0.3 "protocol noise" is same-seed repeatability, not decision noise; sub-1.5-FID margins need 2–3 sampler seeds | evaluate_progressive_joint_flow.py:53; AUDIT_RESPONSE:21 |
| 4 | MED | `train_continuous.py` FID reference = train split; progressive evaluators = test split → cross-family FID incomparable. Matters for E2 | train_continuous.py:894 |
| 5 | MED | prefix/recon evaluators and decoder_sensitivity are token_scale-unaware → oracle floors on rescaled caches are wrong; add divide-or-fail guard | evaluate_progressive_prefix_only.py:35, evaluate_progressive_ar_prefix.py:52-76, scripts/decoder_sensitivity.py |
| 6 | MED | both trainers' in-training previews decode rescaled latents without the inverse — pow083's previews will look broken while the run is healthy; don't kill or journal from previews | train_progressive_joint_flow.py:184-192, train_progressive_ar_flow.py:203-215 |
| 7 | MED | shaping arms (ramp/frontier) never trained with prefix masks but evaluated with them; their prefix-PSNR curves are OOD probes (masked-absent ≠ noised-present) — don't cite as "frontier failed to become prefix-decodable" | train_progressive_tokenizer.py:483-507 vs 599-621 |
| 8 | LOW | `PerTokenAdaLNZeroBlock` reads nonexistent `config.causal` → AttributeError on first arm-B use | joint_flow.py:350 |
| 9 | LOW | per-dim-mean KL convention (see #1); document to avoid literature-β confusion | train_progressive_tokenizer.py:538-541 |
| 10 | OPS | `checkpoint_latest.pt` (only optimizer-bearing ckpt) deleted after final → completed runs unresumable; this failure mode has bitten the project before | train_progressive_tokenizer.py:702-704 |
| 11 | NOTE | per-register band-MSE table's generating code was deleted with rolling but IS recoverable (`git show f906d4a`); regenerate or pin the commit hash in the report before publication | reports/2026-08-21 |

Verified clean (don't re-audit): Heun sampler (true 50 steps, correct sign/
endpoints for z−ε); scalar-only standardization (the α sweep is NOT silently
undone — per-register scale survives into model space; verified in the pow05
checkpoint); inverse scale applied exactly once on the joint path; rescale_cache
recomputes global stats; unconditional generation confirmed; identical
FID extractor/quantization across progressive evaluators; cache deterministic,
encoder in eval mode; nested-prefix masking semantics correct (with the noted
linear supervision decay and prefix-term dominance ~6.5× as design levers, and
register 63 receiving only full-loss gradient).

## 9. What I would run, in order

1. **E2 pixel baseline** through the progressive evaluator path (patchified
   4×4 pixels as a 64×48 cache; same trainer/sampler/eval; test-split
   reference). ~1 engine-run of GPU. Decides whether the program's regime is
   dominated by a trivial baseline.
2. **No-prefix control**: vae-fixed (post-clamp-fix) tokenizer minus the nested
   objective → same prior → FID. Tests the actual premise of the project.
3. **Fix the clamp, rerun the vae arm** — the headline 35.85's mechanism is
   currently unexplained; also re-check the mean-vs-sample caching question
   after the fix (it is only currently moot *because* of the collapse).
4. Read the α sweep with the §1 framing (alignment+weighting, not ordering);
   add **C′ = A + w_i = 1/a_i²** as the one-line decomposition arm; use smooth
   time warps, never clamped offsets, if B is built.
5. **Measure axis B** (conditioning-gain curve, theory doc §7 recipe) on the
   existing cache — the single most thesis-relevant unmeasured quantity, and it
   needs no training run.
6. Re-calibrate eval noise (2–3 sampler seeds) and add one training-seed
   replicate for vae/det before publishing any table with sub-2-FID gaps.
