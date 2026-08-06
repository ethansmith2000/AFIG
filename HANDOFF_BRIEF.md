# AFIG Handoff Brief

Last updated: 2026-08-05. Repo: `/workspace/AFIG`.
Supersedes the 2026-07-30 brief (kept at `HANDOFF_BRIEF_20260730.md.bak`), several
of whose central claims turned out to be wrong — see section 6.

**You are explicitly invited to audit this rather than inherit it.** Section 7
lists the claims I hold least confidently, ranked. Several turns of this
investigation were spent chasing hypotheses that measurement later killed, and at
least four of those corrections came from the user pushing back on my reasoning
rather than from me catching it. A fresh reading may find more.

---

## 1. The one-paragraph state of the world

AFIG's original whitened-FFT -> 53 x 64 AE-latent generator produces texture-like
mush. The strongest positive route is now a deterministic C4 autoencoder whose
patch-local DCT latents generate recognizable fixed/fresh CIFAR structure; the
same aggregate also passes under several real full-support DCT/Hartley layouts.
That retracts both “the AE aggregate is unmodelable” and “global support is a
binary blocker.” Direct raw native FFT remains clearly harder in the runs tried,
and the audited packing objection is now closed: corrected grid-local and
scale-homogeneous layouts score FID `171.38/214.04`, no better than legacy
self-first `164.60`. Two independent audits also
found that all old raw `train_continuous.py` shuffle-gap tables—including every
polar arm—were training-batch rather than held-out, the Hartley AR/joint contrast
had a 24x exposure mismatch, and the polar objective did not make physical
Cartesian geometry primary. Therefore the current explanation is graded
interface/rate and co-tokenization friction, not an impossibility result for
global FFT tokens. A common 5k-sample evaluation now makes the quality gap
unambiguous: pixel/patch-DCT FID is `31.67/31.38`, full DCT `112.82`, full
Hartley `156.36`, and compact FFT `164.60`; KID agrees. The exact token-axis
patch-grid DCT bridge is also weak at FID `130.57`, despite changing nothing but
an orthonormal mix across the successful pixel model's 64 tokens. “Full support passes” was
too generous, while Hartley-versus-compact is only a marginal distinction within
the same poor tier. A decisive zero-training control then conjugated the passing
pixel model into exact compact-FFT state coordinates: it retains clean pixel-tier
samples (`2.15e-6` base round-trip error) when velocity computation is performed
in local patches and transformed back. Global FFT is therefore a valid flow state
space but a poor native interface for this transformer. `ROADMAP.md` section 11
gives the corrected queue.

---

## 2. Read these in this order

1. `DIAGNOSIS.md` — the full evidence log, ~1,480 lines, 20 sections. **Read the
   independent-audit correction and Conclusion first**; the numbered sections are supporting
   measurements. Retractions and scope corrections are marked inline — pay
   attention to those, they are where the reasoning went wrong and got fixed.
2. `ROADMAP.md` — read the audit reset and section 11 first; earlier phases are
   retained as a reasoning history.
3. This file.
4. `README.md` predates all of this; treat as stale. `thing.md` is unrelated scratch.

---

## 3. Completed controls / runtime state

The compact raw-FFT, perceptual-codec, posterior/path, wrapped-normal phase-score,
local-token, support, real-basis/grouping, and C4 compact-FFT suites are complete.
The scaled raw 32-pixel control completed negative at 30k; the 16-pixel 16 x 48
and 64 x 12 matrices passed. The matched ring-block codec ends at `32.85 dB`,
below the legacy codec's `34.77 dB`, without a material robustness/rank gain.
Matched 23-step joint-ring generators on both codecs remain texture/pseudo-scenes
through 10k and fresh seed. A 134-step run on the existing 49.57 dB target-4
codec also fails; its final conditional x0 MSE `0.140` was measured on a training
batch. The 134 x 16 codec
reaches `32.76/34.07/34.34 dB` held-out after 10k/20k/30k. Its matched generator fails
at `latent_continuous_runs/grouped-token-t4-z16-c20k-w768-l12-d6-s1-n10000/`;
the exact codec/interface contract is `checkpoint_20000.pt` plus
`latent_interface_20000.pt`. Every fixed gate through 10k and fresh seed 54321
remain texture-like. Check
`/workspace/bin/gpu-claim status` before launching. All launches use the shared
queue helper.

The polar-v2 coordinate/full runs are complete at
`continuous_runs/ar_fft_factorized_polar_v2_eps01_global_10k/` and
`continuous_runs/ar_fft_factorized_polar_v2_full_eps01_global_d6_10k/`.
Their final **training-batch** clean/shuffled/gap values are `2.388/4.686/2.297` and
`2.255/4.797/2.541`; every fixed gate remains nonsemantic. The coordinate
fresh-prefix diagnostic is complete, and the full diagnostic writes to
`diagnostics/factorized_polar_v2_full_10k/`. The conditional phase oracle completed
negative at `continuous_runs/joint_phase_oracle_true_amplitude_10k/` under the
relative-phase-dominant objective described in the audit correction.

| run | tokens | params | representation | 30k result |
|---|---:|---:|---|---|
| `pixel_control` | 64 x 48 | 115.5M | local 4x4 RGB patches | recognizable CIFAR objects |
| `control_fft_whitened` | 65 x 48 | 115.5M | per-orbit whitened FFT, no AE | texture-like mush |
| `control_fft_global` | 65 x 48 | 115.5M | no per-orbit variance scaling, no AE | texture-like mush |
| `patch_dct_control` | 64 x 48 | 115.5M | local 4x4 DCT | recognizable objects |
| `full_dct_control` | 64 x 48 | 115.5M | global real DCT, 4x4 frequency tiles | recognizable, weaker |
| `full_hartley_control` | 64 x 48 | 115.5M | global periodic real basis, 4x4 tiles | recognizable, weaker |
| `fft_global_spiral` | 65 x 48 | 115.5M | locality-regrouped legacy FFT | texture-like mush |
| `fft_compact_isometric_spiral` | 64 x 48 | 115.5M | active-only isometric FFT; self-first packing confound | rough/mushy |

Final previews:

- `latent_continuous_runs/pixel_control/preview_0030000.png`
- `latent_continuous_runs/control_fft_whitened/preview_0030000.png`
- `latent_continuous_runs/control_fft_global/preview_0030000.png`

The FFT grids are visually very similar (whole-grid MAE 0.0707, MSE 0.01048;
same-channel RGB correlations 0.930--0.940), although neither is coherent. This
closes the original capacity caveat as well as the claim that the current AE or
per-orbit variance whitening is the sole cause. Losses are comparable across the
globally affined orthonormal pixel/DCT/Hartley/compact controls, but not across
legacy codecs with different fitted scales. Decoded samples remain primary.

Important naming caveat: `fft_global` sets `whiten_exponent=0`, so it removes
per-orbit variance scaling but still subtracts each orbit's complex mean. It is
not a pure single-global-mean/std transform. Both FFT codecs were verified to
round-trip through the grouping at 6.9e-15 MSE.

---

### Independent-audit corrections to keep in working memory

- `train_continuous.py` condition diagnostics historically used the just-updated
  training batch. They now use the excluded deterministic tail panel. Old
  ECS/SNR/slim, factorized, wrapped-normal, and polar-v2 shuffle tables must be
  relabeled training-batch; separately held-out spectral/prefix diagnostics are
  still valid.
- Ring/grouped conditional/null losses were also training-batch. Their sample
  grids remain valid negatives.
- Hartley AR at `10k x batch32` saw 24x fewer image exposures than joint Hartley
  at `30k x batch256`; do not call that a matched causal-versus-joint result.
- Compact packing is exactly isometric but puts the paired self-conjugate units
  first, mixing DC and Nyquist in token zero (`~634:1` within-token scale ratio).
  Preserve this layout for old checkpoints and add a versioned corrected layout.
  Legacy 65-token FFT representations also contain 48 padded coordinates.
- The polar phase gate is normalized within each token/RGB triplet and then
  averaged across tokens. It does not weight low-frequency phase more than high;
  only the `0.1 x` Cartesian auxiliary does. A future polar control must use
  reconstructed Cartesian error as the primary loss and model self-conjugate
  signs discretely.
- The spatialized-prefix arm is unmatched in batch and parameter count, and its
  graft PSNR is near the different-image baseline; it does not close trunk input
  geometry.
- No FID has been computed. Single-seed, unblinded “pass/rough” labels need blind
  multi-seed grids and >=5k FID/KID before supporting marginal quality claims.

## 4. What is ruled out, with the script that did it

Each row is the narrow scope supported by its measurement. The audit removed
several broader closures; do not silently promote these to impossibility claims.

| hypothesis | verdict | script |
|---|---|---|
| decoder too fragile for latent error | no — sigma=0.35 noise still decodes recognizably | `diagnose_latent_robustness.py` |
| generator mismatches the latent distribution | no — matches correlations, energy coupling, marginals, kurtosis | `diagnose_latent_structure.py`, `diagnose_cross_token.py` |
| joint model memorizes | no — 0.5% train/test gap | `diagnose_overfitting.py` |
| AR exposure bias is the sole explanation | no — a bidirectional joint model also fails; AR exposure bias itself remains possible | `diagnose_prefix_graft.py` |
| latent is incompressible | no — ~54% of each token linearly predictable from the others | `diagnose_compressibility.py` |
| one static loss weight can restore the full SNR dynamics | no — error-vs-t curves for different variance are not proportional (92–189% shape mismatch) | inline derivation, section 14 |
| whitening matters on the *latent* path | no — 39x input swing -> 2.2x latent change; alpha=0.5 gate identical | `diagnose_hierarchy_chain.py`, section 13 |
| phase corrupted by mean subtraction | mostly no — median 0.2–3.9 degrees on complex orbits | `diagnose_phase_centering.py` |
| exact likelihood beats MSE as an objective | no — flow gains 0.73 nats/dim NLL, zero improvement in samples | `prototype_flow_head.py`, `prototype_flow_sample.py` |
| not enough data | no — pixel control succeeds on the same 50k images | `control_pixel_diffusion.py` |
| current autoencoder is the sole cause | no — direct FFT controls bypass it and still fail | `control_pixel_diffusion.py --representation fft_*` |
| per-orbit variance whitening is the sole cause | no — `fft_global` removes it and still fails | `control_pixel_diffusion.py --representation fft_global` |
| pixel/FFT result is a capacity mismatch | no — all three direct controls use the same 115.5M model | completed control configs |

---

## 5. What actually helped

Only one thing, and modestly:

**RoPE + zero-init absolute position embeddings**, on the joint model. Test MSE
0.9107 -> 0.9042; advantage over the linear/Gaussian floor 0.1076 -> 0.1141. RoPE
alone helps ~4x more than absolute embeddings alone, and the two are **additive**.
2-D RoPE over (radius, angle) beats 1-D over sequence index, though barely.
Implemented in `causal_transformer.py` (`build_rope_tables`, `apply_rope`), exposed
as `--rope {none,sequence,radius_angle}` plus `--position_embedding_input` /
`--position_embedding_film`.

Context for how small that is: the model sits ~11% past a Gaussian fit either way,
and every sample is still mush.

**Raw AR follow-up (2026-08-03):** QKNorm plus fp32 2-D `(ky, kx)` RoPE completed
10k steps. The training-batch clean diagnostic moved only 0.019638 -> 0.019317 and the
shuffled-history gap 0.007197 -> 0.007800. Decoded samples remained broken and
visually extremely close to the non-RoPE baseline. Keep RoPE/QKNorm as sound
attention defaults; do not mistake this for a generative repair.

The subsequent cleanup hard-removed the raw AR model's redundant functional
metadata encoder and independent decoder-position condition. It now uses one
learned prediction-slot table. Per-block position FiLM remains available but is
off by default and reuses that same table. The matched launcher is
`scripts/run_phase_a_ar_fft_cartesian_ecs_snr_slim_10k.sh`.

That slim run completed: final clean/shuffled/gap was
`0.017984 / 0.025342 / 0.007358`, yet every checkpoint remained texture mush.

**Patch-DCT result:** success. Objects are recognizable at 5k and coherent at
30k; final loss `0.3114` essentially matches the pixel control's `0.3116`.
Frequency coordinates and the flow objective are therefore viable when basis
support stays local.

**Full-DCT result:** success, but delayed and weaker. Its 5k grid is mostly
texture, structure emerges by 10k, and recognizable vehicles, animals, and
scenes appear at 30k; final loss is `0.3505`. Global support is a difficulty, not
a fatal obstruction. This still does not isolate complex phase because full DCT
uses contiguous 4x4 frequency-plane tokens while the FFT controls use eight
radial Hermitian orbits per token.

**Full-Hartley result:** success at roughly the full-DCT tier. It develops
object/scene structure by 10k--15k and retains recognizable structure at 30k,
though it is much weaker than patch DCT/pixels; final loss is `0.3647`. Global
periodic Fourier-family coordinates are modelable in real 4x4 grid tokens.

**Direct grouping result:** `fft_global_spiral` used exactly the same codec values,
normalization, shape, and model as failed `fft_global`, but permutes the 514
orbits into square-spiral order before grouping eight per token, then inverts the
permutation before decode. The regrouping round trip is exact. The accompanying
`diagnose_token_composition.py` shows mean within-token frequency distance drops
from `7.43` to `2.84`. Its 30k output remains mushy; local regrouping alone does
not repair Cartesian FFT. Its visually worse output nevertheless has lower final
loss (`0.3414`) than Hartley, so do not use that loss to rank representations.

**Compact-control result:** `fft_compact_isometric_spiral` used the same pixel
mean/std as pixels, DCT, and Hartley; exact sqrt(2) isometric Hermitian packing;
square-spiral ordinary-orbit units; and exactly 3,072 active coordinates reshaped
to 64x48, with no inactive/padded dimensions or fitted per-orbit normalization.
Round-trip, energy, and Gaussian-bridge tests pass. Its 30k samples are still
rough/mushy. The audit found that self-conjugate units are prepended, mixing DC
and Nyquist-scale values in token zero. The transform closes the Gaussian-measure
and affine-normalization questions, but **reopens packing/co-tokenization**. Its
trailing-20 loss mean is `0.4151`; `0.3830` was one final minibatch.

The scale hierarchy is nevertheless extreme in physical isometric coordinates.
Across 4,096 images, complex-amplitude p50/p90/p99/p99.9/max is
`0.145/0.752/3.651/14.989/58.047`; median amplitude falls from `11.28` at DC to
`0.0456` at radii 16--23. Most of the marginal tail comes from mixing known
frequency-conditioned scales. Phase B therefore models normalized log amplitude
in Euclidean space and phase intrinsically on the circle.

**Also fixed, on the AR path:** all AR runs were trained 3–4x too long. At 7,500
steps the model genuinely generalizes (held-out conditional 0.857 vs null 1.170,
**+0.31**); by 30,000 conditioning is actively *harmful* (**-0.26**), crossing over
near 21k. Use `checkpoint_7500.pt` from
`latent_continuous_runs/regularized/ar-wd0.1-bright/` if you need a trunk whose
contexts are not memorization-laden.

**Phase-B native-complex result (2026-08-04): visually failed, diagnostically
important.** `ar_fft_factorized_polar_10k` retained 514 Cartesian history steps
but replaced the decoder with Euclidean flow over normalized log amplitude and
an amplitude-conditioned intrinsic circular phase flow. It also enabled
position FiLM and direct target-slot conditions in both heads. At 10k,
training-batch clean/shuffled/gap is `2.3639 / 4.1293 / 1.7654`; held-out phase coherence reaches `0.824`,
log-amplitude MAE `0.278` with bias `+0.021`, and physical complex NRMSE `0.378`.
Those are large conditional improvements. All 2.5k/5k/7.5k/10k samples remain
speckled texture with no recognizable objects.

`diagnose_factorized_rollout.py` explains more than the aggregate loss. A true
32-coefficient low-frequency prefix with zero suffix is already recognizably
image-like; rolling out the learned suffix destroys it. True 128/256/384 prefixes
survive progressively better, but sampled suffixes add harmful texture while
their mean amplitude ratios stay near one. Fully generated cutoffs progress from
coarse blobs at 32 to weak structure at 128 and increasingly incoherent texture
thereafter. The remaining problem is cross-frequency phase/history consistency
across a long rollout, not merely amplitude scale.

**Matched Hartley AR result (2026-08-04): failed the early visual gate.**
`ar_hartley_tiles_10k` generates 64 real 4x4 full-image Hartley tiles in radial
order with a 106.7M-parameter causal trunk, QKNorm, fp32 2-D RoPE, target slots,
position FiLM, and a directly slot-conditioned flow decoder. Its held-out
clean/shuffled/gap progresses from `0.4707 / 0.5398 / 0.0691` at 2.5k to
`0.4266 / 0.5440 / 0.1173` at 10k. Ordered history matters, but all four grids are
nearly stationary mottled speckle with no recognizable objects. Removing complex
phase geometry and shortening 514 causal steps to 64 is therefore insufficient.
This is specifically an AR result: the matched bidirectional Hartley control does
produce recognizable structure at 10k--30k.

**Phase-D compressive bridge result (2026-08-04; inference superseded below):
reconstruction succeeds; global-Hartley modelability fails.** A 4x-downsample
8-channel spatial AE exports an 8x8x8 map
(512 scalars, 6x compression). With 10% latent-noise training and channel-moment
regularization it reaches 30.91 dB held-out PSNR; 10% test perturbations raise
pixel MSE only from `0.000812` to `0.000950`. Reconstructions remain faithful.

Its 16-step Hartley AR nevertheless remains smooth texture through 10k despite a
final clean/shuffled/gap of `1.340 / 1.646 / 0.306`. A matched joint flow on the
same 16x32 tokens also remains object-free (final loss `1.286`), so rollout is not
the sole cause. The latent marginal is not badly scaled or heavy-tailed: skew
`0.02`, excess kurtosis `0.32`, coordinate scales near one, and tile-RMS range
only `2.22x`. The hard information is in subtle dependencies.

The MSE-VAE rate bracket does not fix it. Beta=`1e-3` retains 30.27 dB and greatly
improves covariance (`offdiag RMS 0.045`, condition 2.23), but direct prior samples
and both sampled- and mean-posterior joint flows remain texture. Beta=`1e-2` drops to 25.07 dB
and partially collapses (condition about 2335); 0.5 free bits repairs the condition
to 2.76 but not prior samples. This closes simple KL tuning.

**Local-token correction (2026-08-04): the same aggregate is modelable.** Three
matched joint comparisons isolate token support:

| codec | full-map Hartley | local raster | local 2x2 DCT |
|---|---|---|---|
| old MSE C8, 512 scalars | texture (`1.286`) | objects (`1.187`) | -- |
| perceptual C8, 512 scalars | texture (`1.304`) | objects (`1.223`) | -- |
| perceptual C4, 256 scalars | texture (`1.306`) | objects (`1.220`) | objects (`1.208`) |

The local C4 raster and DCT arms differ only by an orthonormal transform inside
each token. Their Gaussian noise, linear flow, MSE, model, schedule, seed, and
latent endpoint are otherwise identical. Fresh-seed grids pass for every local
joint arm checked; Gaussian-base decodes remain texture. Thus the AE distribution
was not intrinsically off-limits. The transform that mixes every spatial site
across tokens was the decisive change.

The causal control agrees: the 16-step perceptual-C4 local-DCT AR produces
recognizable classes from its first stored grid, unlike the 16-step global-
Hartley AR. It ends at train loss `1.146` and held-out clean/shuffled/gap
`1.338 / 1.819 / 0.481`. The fresh-seed grid is at
`diagnostics/ar_spatial_ae_perceptual_c4_patchdct_fresh_54321.png` and is at
least as clearly object-like as the fixed preview.

**Scope correction from the exact C4 AR comparison.** The previous sentence's
"global-Hartley AR" referred to the old C8 codec. With the C4 perceptual codec,
raw global-Hartley AR reaches rough but partly recognizable structure. It ends at
train/clean/shuffled/gap `1.237 / 1.441 / 1.752 / 0.310`; it is materially below
the local-DCT arm, but no longer object-free.

The proposed spatialized-prefix bridge was implemented and tested. At every
global radial target position it zero-fills unknown tiles, inverse-transforms the
known prefix, and runs a noncausal spatial transformer over 16 local patches; the
overall generator remains causal. It ends at
`1.237 / 1.493 / 1.659 / 0.166`. Its fixed and fresh outputs correlate `0.979`
and `0.976` with the raw-global arm and show no visual improvement. The smaller
context gap points the same way. Trunk representation alone is therefore not the
repair; the remaining local/global gap follows the output factorization and its
error support.

The final grouping bracket is also complete. Four adjacent Hartley tiles per
target are worse at `1.308 / 1.553 / 1.710 / 0.157`. One frequency per token
remains rough at `1.180 / 1.364 / 1.684 / 0.320`, despite the attractive loss.
All 16 tiles in one target remain rough at `1.464 / 1.428`; the matched single
spatial target is rough at `1.459 / 1.500`, showing that this endpoint is limited
by the unconditional 256-D diffusion head. In the positive direction, the local
DCT values reordered frequency-major still generate semantic fixed and fresh
grids at `1.195 / 1.407 / 1.738 / 0.332`. Low-to-high order passes; global support
does not become easy under a different fixed chunk size.

The wrapped-normal phase-score AR is a visual negative. At 10k it reaches
training-batch clean/shuffled/gap `1.354 / 2.193 / 0.839`, but every grid remains
speckle. The intrinsic phase process works mechanically; the old metric does not
establish held-out context use, and the polar physical-loss confound remains.

---

## 6. Claims the previous brief got wrong

Worth knowing, because they shaped a lot of prior work:

- **"AR diagnostics show context helps (conditional MSE ~0.12 vs null ~0.30)."**
  Measured on training batches. Held out, the effect **reverses**. All three
  weighting arms memorize; `decoder_sensitivity` worst by ~3x, because its
  1e6-ratio weights concentrate the loss onto ~6 tokens. The whole weighting
  campaign was scored on a memorized signal — do not cite its cross-arm comparisons.
- **"Diffusion forcing is the endorsed next step."** It targets exposure bias,
  which the joint model rules out. Dropped.
- **"The AE is not obviously broken (34–35 dB)."** True but misleading — good
  reconstruction says nothing about whether the latent space is *modelable*.
- **"The compressive AE aggregate is unmodelable."** Retracted. That conclusion
  came from global-Hartley tokens. The unchanged old MSE latent map succeeds in
  local raster tokens, so the failed generator diagnosed the representation/model
  interface, not the aggregate distribution in the abstract.

---

## 7. Claims I hold least confidently — audit these first

Ranked by how much they would change the plan if wrong.

1. **How much of the raw compact-FFT failure is co-tokenization.** The old
   active-only packing is isometric but has a `~634:1` within-token scale defect
   in token zero. No completed run isolates corrected self-conjugate placement
   from frequency-local versus radius-homogeneous grouping.
2. **Whether the apparent local-quality advantage is mostly optimization rate.**
   The labels are unblinded/single-seed and no FID exists. A matched 30k C4
   local/global comparison may narrow or preserve the gap.
3. **Whether polar geometry helps once the loss is physically matched.** The
   completed phase losses average tokens approximately equally, the Cartesian
   term has weight `0.1`, Cartesian/polar heads are not conditioning matched, and
   train/sample amplitude conditioning differs. Current polar negatives are not
   the clean test we intended.
4. **How much causality matters after matching image exposures and solver budget.**
   The 64-step Hartley AR used 24x fewer examples than joint Hartley. Its visual
   failure is real; the causal-versus-joint inference is not settled.
5. **"~10% past Gaussian" as the summary statistic.** Exactly computable and held
   out, but MSE-past-Gaussian is a weak proxy for sample quality — a model could be
   barely past Gaussian and still sample well if its errors were structured
   favourably. The load-bearing evidence is the direct kind: Gaussian samples decode
   to mush resembling ours, and the prefix graft.
6. **The flow prototype is minimal** — affine couplings, 12 layers, fixed
   permutations, 6k steps on 8k images' contexts. Its NLL is dramatically better
   than Gaussian and its samples are not. I read that as "the objective is not the
   bottleneck"; a stronger flow (spline couplings, more data) could be argued for.
   I do not think it changes the conclusion; you might.
7. **Perceptual weighting from decoder sensitivity.** I advocated it for several
   turns, then retracted twice — first because loss weighting cannot reproduce SNR
   dynamics, then because it measurably worsens overfitting. If you see an argument
   for it, read both retractions first.

---

## 8. Repo navigation

### Core pipeline (pre-existing, with additions marked)
| file | role |
|---|---|
| `frequency.py` | FFT codec, orbit layout, whitening. **New:** `whiten_exponent` partial whitening |
| `autoencoder_models.py` | causal ring AE/VAE; `GroupLayout` holds the ring->latent allocation rule |
| `latent_autoencoder_interface.py` | frozen AE adapter, contract validation, layout fingerprint |
| `causal_transformer.py` | transformer blocks. **New:** RoPE |
| `diffusion_decoder.py` | per-token AdaLN diffusion head |
| `model_latent_continuous.py` | AR latent model (causal trunk + diffusion head) |
| `model_joint_latent_diffusion.py` | joint bidirectional latent diffusion. **New:** position embeddings, RoPE, timestep sampling |
| `train_*.py` | trainers. **New:** `--rope`, `--position_embedding_*`, `--timestep_weighting`, `--augment_brightness`, `--whiten_exponent`; weight decay 0.02 -> 0.1; fused AdamW |

### Diagnostics added (standalone; write JSON and usually PNG to `diagnostics/`)
`diagnose_latent_robustness.py`, `diagnose_latent_structure.py`,
`diagnose_cross_token.py`, `diagnose_prefix_graft.py`, `diagnose_position_floor.py`,
`diagnose_gaussian_sample.py`, `diagnose_overfitting.py`,
`diagnose_compressibility.py`, `diagnose_ring_allocation.py`,
`diagnose_normalization.py`, `diagnose_snr.py`, `diagnose_snr_staggering.py`,
`diagnose_token_composition.py`,
`diagnose_hierarchy_chain.py`, `diagnose_phase_centering.py`,
`diagnose_ar_generalization.py`, `diagnose_ar_prefix_recall.py`

### New models / controls
| file | role |
|---|---|
| `flow_decoder.py` | conditional normalizing-flow head. Verified: invertibility 1.8e-7, log-det vs autograd 6.6e-7, exact N(0,I) at zero init |
| `prototype_flow_head.py` | flow vs conditional-Gaussian NLL on frozen AR contexts |
| `prototype_flow_sample.py` | AR rollout with the flow head, decoded |
| `control_pixel_diffusion.py` | matched direct controls; includes the orthogonal `patch_grid_dct` token-axis bridge and versioned corrected compact active-scalar layouts |
| `factorized_polar_decoder.py` | amplitude-first Euclidean flow plus intrinsic circular phase flow; v2 adds checkpointed population-standardized log amplitude |
| `scripts/run_phase_b_ar_fft_factorized_polar_v2_coordinate_10k.sh` | epsilon=0.1/global-standardization coordinate-only gate |
| `scripts/run_phase_b_ar_fft_factorized_polar_v2_full_10k.sh` | depth-6 decoder and standardized gated-polar trunk replacement |
| `train_joint_phase_oracle.py` | bidirectional intrinsic phase flow conditioned on the complete true amplitude field |
| `scripts/run_joint_phase_oracle_10k.sh` | batch-256, 115M-parameter conditional phase oracle gate |
| `fit_autoencoder_latent_interface.py --normalization_scope` | fits position-by-channel, shared-channel, or tensor-wide population latent affines |
| `scripts/run_joint_latent_normalization_scope.sh` | matched joint generator for the missing shared-channel/tensor latent controls |
| `sample_joint_latent_diffusion.py` | explicit fresh-seed sampler for joint latent checkpoints |
| `evaluate_control_diffusion.py` | common 5k FID/KID/moment evaluator for saved direct-control checkpoints |
| `evaluate_spatial_latent_joint.py` | the same protocol for decoded joint C4 checkpoints at any saved step |
| `sample_conjugated_pixel_control.py` | zero-training global-state/local-compute positive control for patch-grid DCT and compact FFT |
| `build_control_blind_sheet.py` | four-new-seed shuffled panel sheet with a pre-rating SHA256-frozen key |
| `scripts/run_patch_grid_dct_control.sh` | matched 30k global token-axis DCT bridge |
| `scripts/run_compact_fft_{gridlocal,scale}_control.sh` | matched corrected compact-FFT co-tokenization pair |
| `scripts/run_c4_joint_rate_arm.sh` | matched 30k C4 joint local-DCT/full-DCT/full-Hartley rate bracket |

### Key artifacts
- AE: `autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-vae-kl0.0001/`
  (plus `-wx0.5` and `-wx0.0` from the whitening sweep; wx0.5 is 35.08 dB, slightly
  *better* than the 34.95 dB baseline)
- Joint baseline: `latent_continuous_runs/joint-vae-mean-rf-w768-l12-b256-s1-n30000/`
- Conditioning arms: `latent_continuous_runs/conditioning/`
- Regularized AR: `latent_continuous_runs/regularized/`
- Pixel control: `latent_continuous_runs/pixel_control/`
- Direct quantitative evaluation: `diagnostics/control_fid/`
- Frozen-key blind sheet and pre-reveal ratings: `diagnostics/control_blind_4seed/`
- Active token-axis bridge: `latent_continuous_runs/patch_grid_dct_control/`
- Completed corrected packing pair: `latent_continuous_runs/fft_compact_isometric_{gridlocal,scale}_control/`; metrics in `diagnostics/control_fid/compact_{gridlocal,scale}_corrected/`
- Completed C4 rate bracket: `continuous_runs/joint_c4_rate_{local_dct,full_dct,full_hartley}_s1_30000/`; metrics in `diagnostics/c4_rate_fid/`
- Conjugated global-state controls: `diagnostics/conjugated_pixel/{patch_grid_dct,compact_fft_gridlocal}/`

---

## 9. Operational notes

- **GPU jobs must use `gpu-claim`** (`/workspace/bin/gpu-claim`, docs in
  `/workspace/GPU_QUEUEING.md`). The box is **shared with other projects** — expect
  to queue; `gpu-claim run ... --wait` handles it.
- Launch long jobs **detached**, or they die with your shell:
  `setsid nohup gpu-claim run ... > log 2>&1 < /dev/null & disown`. I lost a batch
  of eval jobs this way.
- Python `/venv/main/bin/python`. Tests: `python -m pytest tests/ -q` (currently
  173 pass plus 3 subtests).
- W&B: `ethansmith2000/afig-latent-continuous` and `.../afig-autoencoder`.
- Trainers log progress with `\r`; read it via
  `tail -c 2000 log | tr '\r' '\n' | grep -o "[0-9]*/30000 \[[^]]*\]" | tail -1`.
- The AE trainer logs only to W&B, so an idle-looking log does not mean it is stuck
  — check `nvidia-smi`.
- **Backward compatibility is easy to break.** Adding one field to
  `FrequencyCodecConfig` broke every existing checkpoint twice: via
  `validate_compatible`, then via `layout_fingerprint`. The fix pattern lives in
  `latent_autoencoder_interface.py` — default-valued new fields are omitted from the
  hash. Always reload the baseline joint checkpoint after touching configs.
- At handoff the Read tool's hook was intermittently timing out on this host;
  writing files through a Bash heredoc was a working fallback.

---

## 10. What to do next

`ROADMAP.md` section 11 is now the source of truth. The sequence below is retained
as background and is superseded where the independent-audit correction disagrees.

1. **Keep the passing reference:** perceptual C4 codec, 16 local 2x2-DCT tokens,
   joint and AR generators. It is 12x compressive and already frequency-domain.
   Patch-major is the stronger default; frequency-major is a passing but softer
   coarse-to-fine variant.
2. **Completed negative bracket:** spatialized history does not help global
   Hartley. Neither do four-tile 64-D bands, 64 scalar-frequency steps, or one
   256-D target containing the full map. The matched one-token spatial model also
   fails, preventing a Fourier-specific reading of that endpoint.
3. **Completed support gate:** the fixed 16 x 16 C4 block-DCT layouts at 2x2,
   4x4, and 8x8 support all pass fixed and fresh visual gates. Context use weakens
   with support, but full support remains modelable.
4. **Completed real-basis/grouping gate:** full DCT and Hartley both pass as
   radial quartets and contiguous 2x2 frequency tiles. Exact compact FFT on C4
   also passes at the same 16 x 16 shape; the later raw 594x-hierarchy packing
   controls completed negative (section 12).
5. **Practical AE route:** the current deterministic C4 codec is already a working
   escape hatch. Generate local spectral latents, deterministically decode, then
   FFT/Hartley-transform the image when global coefficients are the required
   output. A new AE is justified for rate/fidelity or an explicit multiscale
   latent layout, not as another KL sweep.

The posterior-policy/path controls are complete. Sampling the beta=`1e-3`
posterior adds RMS `0.422` noise, supplies `18.1%` of aggregate variance, and
costs `1.22` dB, but removing it does not fix generation. Posterior-mean linear
and trigonometric-VP joint runs both remain in the same blurred texture basin;
the deterministic-AE trigonometric run also fails. See
`diagnostics/spatial_vae_kl1e3_posterior.json` and the three
`continuous_runs/joint_spatial_*_{linear,trig_vp}_10k` directories.

The perceptual codecs and Brownian-torus control are complete. The C8
deterministic codec reaches `30.84` dB; the C4 deterministic codec reaches
`26.71` dB while preserving semantics; the nominal perceptual VAE drives its
posterior standard deviation to RMS `0.046` and is effectively deterministic.
Do not interpret the passing local generations as evidence for stronger KL or
LPIPS: the old MSE C8 local arm passes too.

Key fresh-seed artifacts are
`diagnostics/joint_spatial_ae_mse_c8_raster_fresh_54321.png`,
`diagnostics/joint_spatial_ae_perceptual_c4_raster_fresh_54321.png`, and
`diagnostics/joint_spatial_ae_perceptual_c4_patchdct_fresh_54321.png`, plus the
causal
`diagnostics/ar_spatial_ae_perceptual_c4_patchdct_fresh_54321.png`.
Use `sample_spatial_latent_joint.py` and `sample_spatial_latent_ar.py` for new
seeds rather than trusting the preview seed.

The completed global C4 artifacts are
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_radial_s7_10k/` and
`continuous_runs/ar_spatialized_prefix_hartley_perceptual_c4_s7_10k/`, with fresh
grids `diagnostics/ar_spatial_ae_perceptual_c4_hartley_fresh_54321.png` and
`diagnostics/ar_spatialized_prefix_hartley_perceptual_c4_fresh_54321.png`.
The final grouping runs are
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_band4_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_scalar_s7_10k/`,
`continuous_runs/ar_spatial_ae_perceptual_c4_hartley_all16_s7_10k/`, and
`continuous_runs/ar_spatial_ae_perceptual_c4_spatial_all1_s7_10k/`. The passing
frequency-major local run is
`continuous_runs/ar_spatial_ae_perceptual_c4_patchdct_freqmajor_s7_10k/`, with
fresh grid
`diagnostics/ar_spatial_ae_perceptual_c4_patchdct_freqmajor_fresh_54321.png`.
Fresh grouping grids are
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_band4_fresh_54321.png`,
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_scalar_fresh_54321.png`,
`diagnostics/ar_spatial_ae_perceptual_c4_hartley_all16_fresh_54321.png`, and
`diagnostics/ar_spatial_ae_perceptual_c4_spatial_all1_fresh_54321.png`.

Wavelets are intentionally outside the immediate plan. Architecture escalation
comes only after representation-matched baselines; the current evidence does not
justify another broad conditioning sweep.

Things I would *not* spend on: another generic AE/KL campaign; another prefix
variant; more fixed global-Hartley group sizes; extending a rough global AR with
the same factorization; more conditioning arms; loss reweighting; another phase
process; timestep schedule changes; the normalizing-flow direction; or diffusion
forcing as a stand-alone repair.

One framing worth keeping, from the user: with every sample broken, **no MSE
variant is a trustworthy guide.** We are trying to fix brokenness, not optimize a
metric. Decode and look at images before believing any number in this document.

---

## 11. August 5 design delta

Three conclusions changed after reviewing the native-complex and ring-codec
interfaces in detail:

1. **The support gate passed and changed the queue.** The 2x2/4x4/8x8 block-DCT
   arms all generate recognizable fixed and fresh samples at a matched 16 x 16
   interface. A DCT/Hartley x radial-quartet/tile matrix also passes in every
   cell. Support, real basis, and these grouping laws affect quality but are not
   binary causes.
2. **The compact C4 FFT bridge passes.** Exact isometric Hermitian packing of the
   standardized C4 map as 16 x 16 produces recognizable fixed and fresh samples;
   final clean/shuffled/gap is `1.458 / 1.679 / 0.221`. Complex Cartesian geometry
   is not intrinsically unmodelable.
3. **Factorized polar was a bounded secondary branch, but is not cleanly closed.**
   The coordinate control replaces `log(a + 1e-4)` with
   globally standardized `log(a + 0.1)`; the cumulative control adds depth-6
   heads and true gated-polar history. Both remain texture through 10k. Their
   shuffle gaps were training-batch and the physical Cartesian term was only a
   `0.1 x` auxiliary, so use a future full-weight physical control before a broad
   geometry conclusion.
4. **Overcompleteness is no longer treated as an AE disqualifier.** The old
   53 x 64 ring code did not produce an easier generative geometry, but scalar
   expansion can be useful. The ring return first changes sector causality to
   bidirectional-within-ring/block-causal-between-rings at the existing width;
   only then does it compare 64-D, 16-D, and 8-D allocations and a 23-step joint
   ring decoder.

The phase-preserving raw scaling control remains texture through 30k and
correlates `0.943/0.940` with the unscaled compact grids at 5k/10k. The 16-pixel
matrix passes in all three arms: pixel,
unscaled compact FFT, and scaled compact FFT all generate semantic scenes at
16 x 48; scaled/unscaled correlations are `0.968--0.970`. The 64 x 12 matched
pixel/FFT separator also passes. Compact-FFT samples correlate
`0.976/0.986/0.990/0.994` with the 16 x 48 layout across the four gates. The
added high-frequency dependency burden, not 64 tokens, is now the leading
obstruction. The bidirectional-within-ring, causal-between-ring codec completed at
`autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-ringblock/`.
Its held-out PSNR is `28.05/31.82/32.85 dB` at 10k/20k/30k, versus `34.77 dB`
for the legacy codec.
The generator in `model_ring_latent_continuous.py` packs those latents into 23
ring steps (maximum four latents per ring), denoises each active ring jointly,
and uses learned ring slots plus fp32 sequence RoPE and QK normalization. Tests,
checkpoint round-trip, reduced GPU integration smoke, and full-configuration
one-step GPU smoke pass. The audit found nearly unchanged rank/robustness and
worse ring-summary predictability.

Both codec versions of that generator failed visually through 10k and fresh seed
54321; final training-batch conditional x0 MSE is `0.217/0.205`. The target-4,
134 x 64 sequential arm also fails fixed/fresh samples despite training-batch
conditional x0 MSE `0.140`.
The combined codec uses 2,144 latent scalars and reaches
`32.76/34.07/34.34 dB` held-out at 10k/20k/30k. The fitted 20k interface has `2.36%` ordinary next-token
probe gain and `11.78%` ring-summary gain over zero; the matched 134-step, 16-D
generator remains texture/pseudo-scenes at every fixed gate through 10k. The matched
robustness audit finds `172` versus z64's `178` PCA dimensions for 90% energy,
but about 4 dB worse pixel robustness at the same standardized latent noise.
The generator remains broken through 10k and fresh seed; final training-batch
conditional/null x0 MSE is `0.254/0.365` (reported gap `0.336`). Do not promote
z8 next.
If a ring generator later passes visually, the next
separator is the same 23-step architecture on exact
raw compact FFT. Raw 32 x 32 has 23 radius rings with
`3--288` active coordinates per ring, closely matching the AE generator's
`64--256`; this directly tests whether ring scheduling suffices or learned AE
transport is essential. It is not promoted after the matched ring failures.

The amplitude audit on 10,000 training images supports the new epsilon rather
than merely motivating it qualitatively. In current RMS-relative units,
`epsilon=0.1` places only 2.61% of amplitudes below the transform knee and gives
standardized skew/excess-kurtosis `0.044/0.075`. Current `log1p(a)` implicitly
uses a knee of one, places 75.65% below it, and gives `1.071/1.957`.
The coordinate-only arm completed at
`continuous_runs/ar_fft_factorized_polar_v2_eps01_global_10k/`. It retains
Cartesian trunk history and depth-3 heads, changes epsilon to `0.1`, and fits one
global population affine map. Full-train mean/std are `-0.34504/0.64290`. Its
fixed and fresh samples remain non-semantic through 10k despite final gap
`2.297` on the training batch. The cumulative depth-6 arm with true standardized polar trunk
replacement also completed negative at
`continuous_runs/ar_fft_factorized_polar_v2_full_eps01_global_d6_10k/`; its final
training-batch gap is `2.541`, held-out phase coherence is `0.853`, and the fixed grids remain in the
same texture/pseudo-scene basin. The complete-true-amplitude joint-phase oracle
completed. True-phase rows recover the references, but sampled-phase rows remain
nonsemantic texture through 10k and fresh seed. Its last-20
total/phase/Cartesian means are `1.9652/1.9586/0.06564`. Because its Cartesian
physical term is weighted only `0.1`, this negative is exploratory.

Do not population-center `cos(phi)` and `sin(phi)` independently. For Cartesian
history, positive scaling shared by real and imaginary components preserves
phase; frequency-dependent complex mean subtraction changes the geometric
origin even when it remains algebraically invertible. For polar history, fitted
shifts/scales apply to log amplitude only, while phase stays on the circle.

## 12. Current breakpoint after independent review

The true-amplitude phase oracle has finished; launch nothing from its result
automatically. The consolidated breakpoint is:

1. the direct-control evidence calibration is complete: 5k FID/KID and a
   four-new-seed blind sheet both cleanly separate pixel/patch-DCT from
   full-DCT/Hartley/compact FFT. The blind reveal is `8/12/0/0` TP/TN/FP/FN;
2. the frozen-AE 53 x 64 shared-channel/tensor normalization pair is complete and
   visually negative at 30k despite much lower coordinate loss. The stronger
   `global_standardize` AE retrain is complete at `32.54 dB` held-out versus the
   old AE's `34.76 dB`; its tensor-wide matched joint generator is complete and
   nonsemantic at 30k under fixed and fresh seed `54321` samples;
3. the no-training low-frequency clamp is complete and negative as a repair:
   small prefixes stay texture, while large prefixes are recognizable only after
   the oracle low-pass already supplies layout;
4. the matched `patch_grid_dct` 30k bridge is complete. It changes only the
   successful pixel arm's 64-token axis from local patches to an orthonormal
   global mixture while preserving each token's 48-feature interpretation. Its
   final FID/KID is `130.57/0.11172`, versus pixel `31.67/0.01991`;
5. the corrected compact self-conjugate placement and active-scalar layouts are
   complete. Grid-local/scale-homogeneous FID is `171.38/214.04`, versus legacy
   `164.60`; stop packing permutations;
6. the matched 30k C4 local-DCT/full-DCT/full-Hartley optimization-rate controls
   are complete. FID improves for all arms from 10k to 30k, but the global gap
   remains `19.4--23.7`; do not spend on a raw 90k extension before changing the
   compute interface;
7. conjugating the passing pixel model into patch-grid-DCT or compact-FFT state
   coordinates produces the same clean sample set (`5.25e-6/2.15e-6` base
   round-trip error). Global coordinates are a valid flow state; native global
   token computation is the problem;
8. the first trainable Stage-C gate completed through 30k. It keeps residual
   Gaussian state, flow targets, loss, and Heun updates in exact compact FFT
   coordinates, but computes every velocity on aligned local 4 x 4 patches
   conditioned on the deterministic C4 scaffold. Oracle-scaffold FID/KID improves
   `37.32/0.03646 -> 28.00/0.01656 -> 19.44/0.01003 -> 14.33/0.00609 ->
   12.63/0.00512` at scaffold/5k/10k/20k/30k;
9. zero-shot attachment to unconditional generated C4 scaffolds also passes:
   `62.39/0.06393 -> 47.83/0.03295 -> 40.34/0.02715 -> 35.76/0.02290 ->
   34.69/0.02254` through 30k. This is the honest unconditional pipeline
   comparison and selects refiner 30k. Free generated scaffolds are unpaired, so do not
   fabricate supervised residual targets for them. A same-noise shuffled-context
   control worsens oracle/generated FID from aligned `19.44/40.34` to
   `45.44/65.95`, proving the refiner uses aligned scaffold content rather than
   adding an unconditional texture prior;
10. the same 20k/30k refiners improve the actual 10k frequency-major local-DCT AR
   scaffold from `91.19/0.09966` to `73.66/0.05844` and `71.70/0.05707`. The
   repair transfers, but this coarse AR front end is now the larger bottleneck;
11. the old AR arm was exposure-mismatched at batch 64. An unchanged batch-256
   10k frequency-major retrain improves scaffold FID/KID to `84.18/0.09207` and
   the frozen-refiner pipeline to `63.25/0.04869`. Exposure matters, but the old
   claim that its `11.80` FID difference from joint local DCT measured an AR
   penalty is retracted: it also changed patch-major to frequency-major token
   composition;
12. the missing 2 x 2 matrix is complete. Frequency-major versus patch-major
   costs joint FID `13.05/14.27/12.73/12.30` and AR FID
   `12.04/9.06/8.23/7.41` at 5k/10k/20k/30k. Final joint
   patch/frequency-major FID is `62.39/74.70`; final AR is `75.11/82.53`.
   This replicated within-family separator makes atomic local frequency bundles
   the new coarse-generator default;
13. patch-major AR plus the frozen FFT refiner scores scaffold/completion FID/KID
   `77.44/0.08461 -> 52.69/0.03959` at 5k and
   `75.11/0.08242 -> 54.06/0.04054` at 30k. Select 30k only for standalone
   scaffold FID; select 5k for the complete pipeline. Later checkpoints smooth
   and lose radial/gradient energy despite lower training loss;
14. do not double the coarse AR sequence into amplitude and phase tokens by
   default. If polar decoding returns, make amplitude and phase specialized
   subheads inside one atomic decoder step, then commit the completed bundle to
   trunk history;
15. for strict Fourier AR, condition causal rings/blocks on the full scaffold
   while allowing bidirectional denoising within each current ring. Revisit
   polar only inside this now-passing scaffold-conditioned refinement, with
   full-weight reconstructed-Cartesian loss, discrete self-conjugate signs, and
   matched head conditioning.
16. that strict-ring separator completed negative. It uses 23 exact
   radial compact-FFT residual blocks after a bidirectional local scaffold
   encoder, explicit BOS/shift alignment, QKNorm, fp32 RoPE, absolute target
   slots, and one depth-6 joint diffusion decode per ring. It retains C1's one
   tensor-wide scaffold/residual statistics and never splits an orbit bundle.
   Tests prove exact packing, causality, cached-inference equivalence, backward,
   and sampling. Sampled-history 2.5k/5k/10k FID is
   `75.82/78.80/74.53` versus untouched scaffold `37.32`. Teacher history only
   improves final FID to `70.31`; 50 Heun steps gives `75.40`; shuffled scaffold
   conditioning gives `70.91`, showing the aligned static prefix is not being
   used productively. Low final radial error `0.0515` beside broken images is a
   warning not to trust spectrum marginals. Run directory:
   `latent_continuous_runs/scaffold_fft_ring_residual_oracle_c4_s1_10000/`;
17. if strict causality is still desired, the next separator should keep C1's
   aligned local dual-domain denoiser and change only its schedule: completed
   lower rings, current ring on its flow path, future rings at base noise, with
   the returned FFT velocity masked to the current ring. Sample target rings in
   proportion to active scalar count and add explicit ring identity. Do not run
   more normalization/depth/solver/polar variants of the failed static ring MLP.

Keep the deterministic C4 patch-DCT codec/generator as the practical baseline.
All GPU launches must go through `/workspace/bin/gpu-claim` and follow
`/workspace/GPU_QUEUEING.md`.
