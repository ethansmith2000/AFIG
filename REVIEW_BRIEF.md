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
capacity check only, never a selection metric. Protocol noise ~ +-0.3 FID.
Latent MSE was retired for cross-model decisions. Most results are single-seed.

## 3. Results so far

### Latent shaping (v5 matrix — 5 tokenizers, identical architecture and recipe, judged by an identical 60k-step joint prior)

| arm | FID | eff-rank | tok-profile-corr | chan-order-consist. | PSNR |
|---|---:|---:|---:|---:|---:|
| vae (KL 1e-4) | **35.85** | 100.7 | 0.312 | 0.537 | ~33.9 |
| energycv (kurtosis->3 penalty) | 39.03 | 26.3 | 0.172 | 0.551 | 34.02 |
| ramp (sigma-ramp latent noise) | 40.72 | 519.6 | 0.725 | 0.508 | — |
| det (control) | 41.16 | 61.0 | 0.587 | 0.540 | 32.88 |
| frontier (frontier-noise) | 47.60 | 48.8 | **0.908** | 0.514 | 31.41 |

- Token-axis *schedule consistency* anti-predicts FID (Spearman +0.8, wrong
  sign). Both arms that explicitly shaped the resolving schedule failed.
- `channel_eigen_order_consistency` sits at **0.508-0.551 (chance) in all five
  arms** — no second-order intervention moved it.
- Reconstruction dB is non-predictive: energycv has the best PSNR and loses 3.2
  FID to vae.

### Latent higher-order structure

Raw channel coordinates are near-Gaussian (kurtosis ~3.0-3.2) and the
fourth-moment pair slice `E[z_i^2 z_j^2]` sits within 11-15% of its Gaussian
value `L_ii L_jj + 2 L_ij^2` — so there is little for an energy-covariance
regularizer to bite on. But rotating into the channel eigenbasis exposes
kurtosis up to 6.05, rank-ordered: top-4 eigendirections 2.8-3.7, bottom-4
5.2-5.6. The non-Gaussianity is real and lives in the low-variance directions;
the code has arranged for its working coordinates to be the most Gaussian view.

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

So the code is semantically ordered but **not** ordered in the SNR sense that
diffusion actually responds to. The proposal is to impose the missing spectrum
directly — scale register `i` by `a_i` (e.g. a power law), train the joint prior
on the rescaled cache, and invert the scale before decoding.

Note this is a no-op only for an infinite-capacity, infinite-step model: the
forward noise is isotropic, so scaling data per coordinate is exactly a
per-coordinate noise schedule, which changes resolution order, the effective
per-coordinate loss weighting, and discretization error allocation. A post-hoc
rescale costs no AE training, so it cleanly separates "ordering helps" from
"ordering is a reparameterization".

## 5. Questions we would most like attacked

1. Is the post-hoc rescale argument right, or is there a reason it must be a
   no-op that we are missing? What profile `a_i` would you impose, and how would
   you avoid simply starving the low-magnitude registers of loss weight until
   they are never learned?
2. Ordered code helps (5.3 FID from shaping) but ordered *generation* costs
   16-44 FID. Is the progressive-tokenizer premise worth keeping at all, or is
   the right conclusion "use the ordered code, generate jointly"?
3. `channel_eigen_order_consistency` is pinned at chance across five very
   different objectives. Is there any objective that could move it?
4. The heavy tails sit in the low-variance channel eigendirections. Harmful
   structure to remove, or useful sparse detail to preserve?
5. We have **no external baseline** — no pixel-space diffusion at matched budget.
   How much should that discount everything above?
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
