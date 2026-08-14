# Independent method audit — AFIG progressive tokenizer + flow priors

Auditor: Claude (independent pass, 2026-08-11). Scope: method, objectives,
normalization, optimizer, train/inference alignment, evaluation logic — plus
new diagnostic measurements run on this box today. All new measurements use the
repo's own FID protocol (`live_evaluation.InceptionFeatures`, 10k-test
reference, 5k generated samples unless noted), so they are directly comparable
to the numbers in the brief. Scripts, metrics, and the new checkpoints are
preserved under `audit_2026-08-11/` (checkpoints: joint@60k, AR@40k,
whitened-joint@20k).

## 0. What was verified vs inferred vs new

Verified against code/artifacts: every number in the brief that I checked
reproduces — tokenizer prefix PSNRs and latent stats
(`tokenizer_runs/v2-cross-n16-d64-s1/metrics_final.json`), joint/AR FID at
5k/20k, the AR teacher-forced validation minimum near 14k (0.4148) rising to
0.4297 at 20k, per-token MSE profiles, cache statistics
(mean −0.0477 / std 0.6797), and the exact shifted alignment
(`tests/test_progressive_ar_flow.py` correctly tests causality and BOS
independence). Re-running the AR 20k eval reproduced FID 100.39 vs the recorded
100.08 — protocol noise is ~0.3 FID, which calibrates small differences.

New measurements from this audit:

| # | Diagnostic | Result |
|---|---|---|
| D1 | Tokenizer reconstruction, 10k test, full 16 tokens | **FID 6.15 / KID 0.0030** (PSNR 32.62) |
| D2 | Decode with iso latent noise σ=0.05/0.1/0.2/0.4 (standardized units) | FID 7.2 / 11.5 / 32.9 / 113.2 |
| D3 | σ=0.2 noise on token 1 only / tokens 9–16 only | FID 7.0 / 15.2 |
| D4 | Full-covariance Gaussian fit to 1024-D latents, decoded | **FID 146.0** |
| D5 | AR FID at 12.5k / 15k / 20k / **40k** | 109.5 / 103.8 / 100.1 / **90.4** (monotone ↓) |
| D6 | Joint FID at 20k / **40k / 60k** (same run continued, constant LR) | 74.97 / **59.31 / 56.53** |
| D7 | Joint uniform checkpoint-average 12.5k–20k | FID 81.1 (worse than 75.0) |
| D8 | AR prefix-replacement, real prefix length j=0/1/2/4/8 | **FID 100.4 / 88.8 / 75.6 / 58.4 / 28.1** |
| D9 | AR slot-1 generated marginal vs real (5k each) | sliced-W2 0.0039; cov-Frobenius gap 12%; mean gap 7% of ‖μ‖ |
| D10 | AR head sampling temperature τ=0.9 / 0.8 (zero training) | FID 96.7 / 98.8 (vs 100.4 at τ=1) |
| D11 | Joint retrained on per-coordinate whitened cache, 5k / 20k | FID **112.1** / **80.2** (raw: 119.9 / 75.0) |
| D12 | Flattened 1024-D latent covariance | effective rank ≈ **92**; top-32 PCs = 66% var; top-256 = 94% |
| D13 | Per-coordinate std after tensor-wide standardization | 0.40–2.76 (≈47× variance spread) |
| D14 | Coordinate kurtosis / innovation kurtosis | 2.9–3.9 / 3.3–4.1 (near-Gaussian) |
| D15 | Slot-k residual variance after linear regression on prefix | slot2 0.30; slots 8–16: 0.06–0.12 |
| D16 | PCA-truncated latents decoded: 64/128/256/512 PCs | FID 127.8 / 82.2 / 44.4 / 13.5 (PSNR 20.3/22.5/25.4/29.5) |
| — | AR TF val MSE during 20k→40k continuation | degrades 0.43 → 0.51 while FID improves 100→90 |
| — | Joint val flow MSE during continuation | still falling at 60k (0.588 → 0.575) |

Caveat on D11: the whitened run trained eager (`--no-compile`, after a flaky
Triton failure) while the baseline was compiled — a small numerics confound;
read it as "no benefit at 20k," not as a precise −5.

## 1. What the method is trying to achieve (restated)

Replace literal Fourier coefficients with a learned, deterministic, *nested*
code: 16 ordered 64-D whole-image registers where every prefix decodes to a
progressively better reconstruction — a short sequence with a meaningful
coarse-to-fine causal order. Then test, separately: (Q1) does the code
preserve the image distribution in generatively usable form; (Q2) can a joint
rectified flow learn the 1024-D latent distribution; (Q3) does the ordered AR
factorization survive, or do first-token difficulty and exposure effects
dominate. The joint model is the positive control that removes AR confounds.

## 2. Answers to the three separated questions, on today's evidence

**Q1 — tokenizer: answered, and it is not the bottleneck.** Reconstruction FID
of the full test set through the frozen tokenizer is 6.15 (D1). Every
generative number in play (56–146) sits far above this floor. The decoder is
also not fragile: isotropic latent noise degrades it smoothly with no cliff
(D2). To explain even FID 56 by decode alone, prior samples would need
≈σ0.25-equivalent latent error. Hypotheses 1 and 2 are effectively retired at
the current quality level; revisit only when generation reaches FID ≈ 15–20.
One notable slot asymmetry (D3): a fixed noise budget hurts ~2× more on late
tokens than on token 1 — decode-side, the *late* tokens are the sensitive
ones (texture), the mirror image of the distribution-side difficulty.

**Q2 — joint modelability: yes, and the model was mostly just undertrained;
but a real representation/objective tax remains.** A full-covariance Gaussian
fit decodes at FID 146 (D4); the joint flow at 20k reaches 75 — it has learned
substantial non-Gaussian structure. Simply continuing the same run (constant
LR, no changes) gives 59.3 at 40k and 56.5 at 60k (D6): 18.4 points were
undertraining. The curve is decelerating (−15.7, then −2.8 per 20k), so the
current recipe is heading to a plateau around ~50 — still well above both the
reconstruction floor (6.2) and the legacy pixel-control region (~31). The
remaining gap is the real object of study. Per-coordinate whitening is not the
missing piece (D11: helps at 5k, gone by 20k). The next candidates are recipe-
level (EMA + LR decay + logit-normal t + class conditioning — all standard,
all cheap here) and, if those saturate, code smoothness itself (latent-noise /
KL-regularized tokenizer variants).

**Q3 — the ordered AR factorization loses 25–31 FID vs joint at matched
steps, and the loss is *distributed*, not first-token-dominated.** The
decisive measurement is D8 — condition the frozen AR 20k model on real
test-cache prefixes of length j and generate the rest:
FID 100.4 / 88.8 / 75.6 / 58.4 / 28.1 for j=0/1/2/4/8. Readings:

- Token 1 contributes ~12 points — real, but not dominant. Its generated
  marginal is close to correct (D9). "First-token difficulty" (hypothesis 5)
  was largely a misreading of the per-token MSE table: token-1 MSE ≈1.15 is
  mostly *irreducible conditional entropy* of the one unconditional slot (its
  val MSE stays flat at 1.14 across the entire run while everything else
  overfits — floors differ per slot, so cross-slot MSE comparison is
  uninformative).
- Even with 8 real tokens — zero exposure bias for those slots, and remaining
  slots 90%+ linearly predictable from the prefix (D15) — generating the tail
  still costs 28 FID vs the 6.2 floor. **Per-stage conditionals are weak
  independent of exposure.**
- The j-curve is smooth (no elbow): gradual compounding on top of mediocre
  conditionals, not a catastrophic distribution-shift cascade.

Structurally, joint-vs-AR is not a pure factorization comparison: the joint
model applies 70M parameters of full attention to every token at every
denoising step, while the AR head is a 26M MLP whose entire view of history is
one 512-D trunk vector injected only through AdaLN shift/scale/gate. The
factorization change is confounded with a large drop in per-token conditioning
bandwidth and compute. Additional AR facts: FID improves monotonically through
40k (D5) even as TF val MSE badly degrades (0.43→0.51) — so the 20k "overfit"
checkpoint was never past-best, and the joint–AR gap *widens* with training
(25 → 31 at 40k). Temperature τ=0.9 buys ~3.7 points free (D10) — conditionals
are slightly diffuse, but temperature is not a major lever without CFG.

## 3. Strongest and weakest assumptions in the causal argument

Strongest (verified): a deterministic nested code with smooth prefix gains is
achievable at high PSNR; the joint control is fair and informative; the
caching/normalization/eval plumbing is correct (no alignment or leakage bugs
found; the brief's numbers all reproduce).

Weakest:

1. **"Latent-MSE differences carry evidence between models."** The design doc
   states the AR-vs-joint val-MSE comparison is "direct evidence of exposure
   bias" (Gate D). It is not: teacher-forced conditional MSE and shared-t
   joint MSE have different irreducible floors and are incommensurable. Both
   directions of this metric failed empirically today: AR FID improved while
   TF-val MSE worsened (D5), and the joint model's *better* FID came with
   *worse* val flow MSE than residual pooling's. Retire latent MSE for all
   cross-model decisions; use decoded FID (evals cost ~10 min here).
2. **"20k steps is a meaningful budget."** 20k × 256 is ~7 GPU-minutes on this
   box and ~5% of a standard CIFAR diffusion budget. D6 shows −18.4 FID from
   continuation alone. Any "X worse than Y at 20k" conclusion is provisional.
3. **"A meaningful causal order should help generation."** The nested
   objective concentrates conditional entropy in the earliest slots — exactly
   where the AR machinery is weakest (unconditional slot, MLP head) — and
   makes late slots nearly deterministic linear maps (D15). Nothing measured
   shows the ordering *buying* anything generatively; it is currently a pure
   cost vs joint. The order's genuine value (anytime decoding, short
   sequences, interpretability) is real but should not be conflated with a
   generative advantage.
4. **"One scalar normalization avoids distorting semantics."** For a learned
   code, per-coordinate scales are gauge artifacts of the final linear layer.
   Empirically the choice barely matters at 20k (D11) — which is itself the
   answer: normalization is not where the tax is. The deeper geometry (D12,
   D16) is: effective rank 92 with perceptually load-bearing low-variance
   directions means flat MSE *underweights* precisely the dimensions that
   carry texture. That argues for loss-side reweighting (or invertible
   PCA-rotation + scale) rather than data-side scalar choices — but note
   PCA *truncation* is not free (256 PCs → FID 44), so any rank reduction
   must be validated through decode.

## 4. Ranked bottlenecks

1. **Prior training scale / recipe (hypothesis 4).** Confidence: high —
   measured. −18.4 FID (joint) and −9.7 (AR) from continuation alone; both
   curves still falling at the end; val flow MSE still improving at 60k.
   Counterevidence: deceleration means scale alone saturates near ~50 (joint).
   Cheapest next test: one long joint run (200k ≈ 70 GPU-min) with per-step
   EMA (0.9995), cosine or step decay, and logit-normal t — three standard
   levers, one run.
2. **AR per-stage conditional capacity / conditioning bandwidth (hypothesis 6,
   broadened).** Confidence: medium-high. Evidence: D8's j=8 result (28 vs 6.2
   floor with zero exposure); 512-D AdaLN-only bottleneck vs full attention;
   gap widening with training. Counterevidence: none found today.
   Cheapest decisive test: cross-attention conditioning for the head (or the
   joint-block-AR hybrid, §7-E3), compared at matched steps.
3. **Error compounding / exposure (hypothesis 5, reduced).** Confidence:
   medium. Evidence: smooth j-curve, ~12 FID per early real token, slot-1
   marginal near-correct yet replacing it helps — small drifts amplify.
   Counterevidence: conditionals weak even on real prefixes, so exposure is an
   amplifier, not the root cause. Cheapest test: fine-tune AR with σ≈0.05–0.1
   Gaussian noise on teacher-forced *inputs* (within the decoder's measured
   tolerance, D2) — doubles as the fix and as a TF-overfitting regularizer.
4. **Residual joint plateau — representation/objective interaction
   (hypothesis 3, reframed).** Confidence: medium. Evidence: decelerating D6
   curve toward ~50 vs pixel-region ~31; D12/D16 geometry (flat MSE
   underweights perceptually critical low-variance directions).
   Counterevidence: whitening null (D11); near-Gaussian marginals/innovations
   (D14). Cheapest tests: (a) matched pixel baseline to pin the true tax
   (brief control #3 — the legacy number is confounded: width 768, 30k steps,
   wd 0.1 per `control_pixel_diffusion.py`); (b) after E1's recipe run, an
   eigenspace-reweighted loss or PCA-rotated (not truncated) cache.
5. **No EMA / constant LR.** Confidence: medium for long runs, low for 20k.
   Coarse checkpoint-averaging hurt (D7) because the trajectory is still
   descending fast — that discredits the proxy, not per-step EMA. Fold into
   the E1 run rather than testing separately.
6. **Tokenizer distribution ceiling (hypothesis 1).** Refuted at current
   scale: rFID 6.15 (D1). Revisit below FID ~20.
7. **Decoder off-manifold fragility (hypothesis 2).** Weakened: smooth
   degradation (D2), no cliff. Robustness training would buy little now.
8. **Small-data limits (hypothesis 8).** Not binding for the joint model (val
   MSE still falling at 60k, no overfitting). AR TF-overfitting is real but
   does not impede FID. Defer ImageNet-32 until 1–4 move; agreed with the
   brief's own instinct that scaling now would mask, not explain.
9. **Progressive objective produces pathological residuals (hypothesis 7).**
   Refuted: innovations near-Gaussian (kurtosis 3.3–4.1), weakly
   heteroskedastic (|ρ| ≤ 0.22). The objective's real distributional effect is
   entropy *ordering* (see §3.3), not residual pathology.

## 5. Mathematical / objective / optimizer notes

- Rectified-flow implementation is correct (path, velocity target, Heun with
  final Euler step, t-convention). Uniform-t + flat MSE is the vanilla
  recipe; logit-normal t is the one standard upgrade worth folding into E1.
- Flow-MSE comparisons across representations or factorizations are
  uninformative (different floors) — see §3.1. Make this a project rule.
- Optimizer: AdamW(0.9, 0.995), constant LR 1e-4, clip 1.0 — adequate at this
  scale. For 10× longer runs add decay and/or EMA. One grouping inconsistency:
  `trunk.bos` **is weight-decayed** (ndim-3, no protected name fragment),
  contrary to the stated policy
  (`progressive_tokenizer/training.py:optimizer_parameter_groups`; verified
  empirically). Impact small; the name-fragment mechanism is brittle — prefer
  explicit parameter tagging.
- Numerics: fp16 cache quantization (~5e-4 relative) and bf16 sampling are
  negligible at current FID scales; one fp32-sampling eval on the best
  checkpoint would close the question. Keep eager/compiled runs unpooled in
  comparisons (the D11 caveat).
- Eval protocol is internally consistent (canonical torch-fidelity extractor,
  fixed reference, deterministic seeds; measured protocol noise ~0.3 FID).
  Caveats: (a) absolute FIDs (10k-test reference, 5k samples) are not
  comparable to literature numbers — add a 50k-train-reference variant before
  quoting externally; (b) single-seed selection: cross-only beat residual by
  10.5 FID with one run each and near-identical 5k FIDs; re-confirm once the
  recipe improves, and note the ranking might differ for AR (selection used
  the joint prior only).

## 6. Implementation risks (file / symbol)

No correctness bugs found in the training/eval path. Notes:

- `progressive_tokenizer/training.py:optimizer_parameter_groups` — `trunk.bos`
  decayed (§5); fragment matching silently reclassifies any future parameter
  whose name contains "norm"/"position"/etc.
- `progressive_tokenizer/model.py:_prefix_mask` — range validation skipped
  under `torch.compiler.is_compiling()`; safe with the current trainer, unsafe
  if reused with untrusted prefix tensors.
- `evaluate_progressive_joint_flow.py` — `setdefault("qk_norm",
  "l2_temperature")` silently switches architecture for configs missing the
  key; fine today, footgun later.
- `cache_progressive_latents.py:latent_statistics` — `values.flatten(0, 0)` is
  a no-op (cosmetic).
- Val metrics use 2048 fixed examples with fixed noise — good for
  comparability; quoted val-MSE gaps carry that sampling width.
- AR alignment/causality verified correct (tests + independent reading).

## 7. Three highest-information next experiments

(Several of the brief's proposed controls are now done: #1 rFID = 6.15; #2
joint continuation → 56.5 at 60k and decelerating; #4 per-coordinate
whitening null; #5 smooth decoder sensitivity; #6 AR 20k/40k were never
past-best. The three below are what remains highest-value.)

**E1 — One properly-trained joint run: 200k steps + per-step EMA (≈0.9995) +
LR decay + logit-normal t, on the raw cache.** ~1–2 GPU-hours total.
Outcomes: (a) FID reaches ≈30–40 → the entire gap was recipe/scale; the
representation is vindicated at this size, and the AR question can be posed
against a trustworthy ceiling; (b) plateaus at 45–55 → a genuine
representation tax exists; proceed to loss-side geometry fixes (eigenspace
reweighting / PCA-rotation) and, if those fail, tokenizer smoothness
regularization (latent noise or small KL) — with rFID 6.15 there is ~25 dB of
slack to spend on making the code smoother; (c) (unlikely) worse → EMA/decay
interact badly at this scale; fall back to plain longer training, which D6
already validates.

**E2 — Matched pixel-space baseline under the current harness** (brief
control #3: same 512-wide DiT, optimizer, 60k+ steps, solver, eval).
Outcomes: (a) pixel ≈ 30–35 at matched budget → the latent code costs ~20–25
FID at parity; representation work is the priority; (b) pixel ≈ 50–60 →
the tax is mostly recipe, both regimes scale together; prioritize E1
conclusions and revisit the "compact-FFT lagged pixels" narrative, since the
legacy pixel number was confounded (bigger, longer); (c) pixel ≫ latent
(unlikely) → latent code is genuinely favorable; go straight to AR repair.

**E3 — AR repair, two arms, gated on E1's ceiling.** Arm 1 (exposure):
fine-tune the AR model with σ≈0.05–0.1 noise on teacher-forced trunk inputs,
targets clean; if the j=0↔j=8 spread (D8) compresses substantially, exposure
compounding was the amplifier and this is the fix; if unchanged, exposure is
closed as a cause. Arm 2 (bandwidth): give the head real conditioning — cross-
attention over all previous trunk states, or a joint-block-AR hybrid (joint
flow over slots 1–4, AR across 4-slot blocks); if per-stage replacement FID
approaches the floor, the shared-MLP bottleneck was the cause; if neither arm
closes the gap toward the joint model, the ordered factorization itself —
entropy concentrated early over a weak unconditional stage — is the wrong
shape, and the project should pivot to joint or masked/any-order generation
over these registers (MAR-style), keeping the nested code for its anytime-
decoding value rather than as an AR substrate.

## 8. Additional observations and ideas

- **Class conditioning is free.** The caches already store labels; a class-
  conditional joint run + CFG is the single biggest known FID lever in this
  regime and would disambiguate "can't model the distribution" from "can't
  cover unconditional multimodality at this scale."
- **Loss-side geometry.** D16 shows perceptual content concentrated in
  low-variance latent directions that flat MSE underweights. An invertible
  PCA rotation with per-direction loss reweighting (or simply training in
  PCA-rotated coordinates, which reweights nothing but at least axis-aligns
  the spectrum) is the principled version of "whitening" — D11 only tested
  scale equalization, not rotation.
- **Slot 5–6 anomaly.** Slots 5–6 are simultaneously the joint model's
  hardest (val MSE 0.67/0.77), the AR model's most-overfitting
  (Δ +0.08/+0.09), and locally innovation-rich (D15: 0.26/0.25 residual vs
  slot 4's 0.15). The learned ordering is not information-monotone there;
  worth decoding with slots 5–6 mean-filled to see what they carry before
  leaning on the coarse-to-fine narrative.
- **Diffusion forcing** (per-token noise levels on trunk inputs, embedded in
  the trunk) unifies E3's two arms and yields a model that can interpolate
  between joint and AR generation at inference — arguably the natural
  endpoint for this architecture family.
- **Metric hygiene**: select checkpoints by decoded FID only; quote ±0.3
  protocol noise; two seeds for any selection decision; never pool
  eager/compiled results.
- **MAR precedent** (Li et al. 2024, continuous tokens + diffusion head) is
  the closest published system; its findings (temperature, CFG, random-order
  masked AR ≫ fixed raster order, trunk capacity > head capacity) map directly
  onto E3 and are worth mining before building custom fixes.

## 9. Bottom line

The tokenizer is fine (rFID 6.15, graceful under noise). The joint prior was
mostly undertrained — a free 18.4-FID gain was sitting in the same run — and
its remaining plateau (~50) is the true open problem; per-coordinate
normalization is not the cause, and loss/geometry or recipe (EMA, decay,
t-sampling, class conditioning) are the live suspects. The AR gap is not a
first-token story and not primarily classical exposure bias: per-stage
conditionals are weak everywhere (28 FID even on all-real 8-prefixes), with
compounding as an amplifier — pointing at the single-vector AdaLN conditioning
bottleneck of the shared MLP head. Teacher-forced validation MSE should be
retired as a decision metric in both directions; it failed twice today.
Establish the joint ceiling first (E1/E2); only then is the ordered-AR
question (E3) cleanly posed.
