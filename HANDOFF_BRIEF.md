# AFIG Handoff Brief

Last updated: 2026-08-03. Repo: `/workspace/AFIG`.
Supersedes the 2026-07-30 brief (kept at `HANDOFF_BRIEF_20260730.md.bak`), several
of whose central claims turned out to be wrong — see section 6.

**You are explicitly invited to audit this rather than inherit it.** Section 7
lists the claims I hold least confidently, ranked. Several turns of this
investigation were spent chasing hypotheses that measurement later killed, and at
least four of those corrections came from the user pushing back on my reasoning
rather than from me catching it. A fresh reading may find more.

---

## 1. The one-paragraph state of the world

AFIG models CIFAR-10 by encoding images into whitened FFT tokens, compressing them
with a frozen autoencoder into 53 latent tokens of 64 dims, and training a
generative model on those latents. Samples have always been texture-like mush.

After ~16 diagnostics and 10 training arms, the controls that matter are now
complete. The *same* 115.5M-parameter bidirectional transformer, 50k CIFAR images,
rectified-flow objective, width/depth/steps/batch/schedule produces recognizable
objects from local 4x4 **pixel patches**, but produces mush from direct FFT tokens
both with per-orbit whitening and without per-orbit variance scaling. All three
ran for 30k steps; the two FFT controls bypass the autoencoder.

So the current failure is in the **global-frequency representation/model pairing**,
not merely the autoencoder, per-orbit variance whitening, model capacity, data,
objective, or compute budget. This does *not* establish that Fourier generation is
impossible. Global token support, complex/Hermitian geometry and noising, and
weight sharing across absolute frequency positions remain confounded. The staged
response is in `ROADMAP.md`; `DIAGNOSIS.md` is the supporting evidence log.

---

## 2. Read these in this order

1. `DIAGNOSIS.md` — the full evidence log, ~770 lines, 17 sections. **Read the
   Conclusion and the control section first**; the numbered sections are supporting
   measurements. Retractions and scope corrections are marked inline — pay
   attention to those, they are where the reasoning went wrong and got fixed.
2. `ROADMAP.md` — the staged experimental plan derived from the completed controls.
3. This file.
4. `README.md` predates all of this; treat as stale. `thing.md` is unrelated scratch.

---

## 3. Completed controls / runtime state

The two direct FFT controls finished on 2026-08-03. Their GPU claims were released;
no AFIG experiment is currently queued or running.

| run | tokens | params | representation | 30k result |
|---|---:|---:|---|---|
| `pixel_control` | 64 x 48 | 115.5M | local 4x4 RGB patches | recognizable CIFAR objects |
| `control_fft_whitened` | 65 x 48 | 115.5M | per-orbit whitened FFT, no AE | texture-like mush |
| `control_fft_global` | 65 x 48 | 115.5M | no per-orbit variance scaling, no AE | texture-like mush |

Final previews:

- `latent_continuous_runs/pixel_control/preview_0030000.png`
- `latent_continuous_runs/control_fft_whitened/preview_0030000.png`
- `latent_continuous_runs/control_fft_global/preview_0030000.png`

The FFT grids are visually very similar (whole-grid MAE 0.0707, MSE 0.01048;
same-channel RGB correlations 0.930--0.940), although neither is coherent. This
closes the original capacity caveat as well as the claim that the current AE or
per-orbit variance whitening is the sole cause. Losses across representations are
not directly comparable; decoded samples are the primary result.

Important naming caveat: `fft_global` sets `whiten_exponent=0`, so it removes
per-orbit variance scaling but still subtracts each orbit's complex mean. It is
not a pure single-global-mean/std transform. Both FFT codecs were verified to
round-trip through the grouping at 6.9e-15 MSE.

---

## 4. What is ruled out, with the script that did it

Each was a live hypothesis that measurement killed. Do not re-run without a reason.

| hypothesis | verdict | script |
|---|---|---|
| decoder too fragile for latent error | no — sigma=0.35 noise still decodes recognizably | `diagnose_latent_robustness.py` |
| generator mismatches the latent distribution | no — matches correlations, energy coupling, marginals, kurtosis | `diagnose_latent_structure.py`, `diagnose_cross_token.py` |
| joint model memorizes | no — 0.5% train/test gap | `diagnose_overfitting.py` |
| AR exposure bias | no — bidirectional joint model fails identically and in the same place | `diagnose_prefix_graft.py` |
| latent is incompressible | no — ~54% of each token linearly predictable from the others | `diagnose_compressibility.py` |
| loss weighting can restore SNR dynamics | **impossible in principle** — error-vs-t curves for different variance are not proportional (92–189% shape mismatch) | inline derivation, section 14 |
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
10k steps. The held-out clean diagnostic moved only 0.019638 -> 0.019317 and the
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
The next active control is `full_hartley` in `control_pixel_diffusion.py`,
launched by `scripts/run_full_hartley_control.sh` through `gpu-claim`. It keeps
the successful full-DCT token grid and model while changing the real orthogonal
basis to a global periodic Fourier-family basis.

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

**Active bridge:** `full_hartley` under
`scripts/run_full_hartley_control.sh`. It is a real orthonormal, global periodic
Fourier-family transform packed into the same contiguous 4x4 frequency-plane
tokens as full DCT. Success points first to radial/Hermitian token composition;
failure beside successful full DCT points to the periodic translation/phase
gauge and justifies the amplitude-before-phase intervention.

**Active direct grouping control:** `fft_global_spiral` under
`scripts/run_fft_global_spiral_control.sh`. It uses exactly the same codec values,
normalization, shape, and model as failed `fft_global`, but permutes the 514
orbits into square-spiral order before grouping eight per token, then inverts the
permutation before decode. The regrouping round trip is exact. The accompanying
`diagnose_token_composition.py` shows mean within-token frequency distance drops
from `7.43` to `2.84`; this directly tests whether radial token composition, not
Fourier coefficients themselves, caused the matched joint failure.

**Also fixed, on the AR path:** all AR runs were trained 3–4x too long. At 7,500
steps the model genuinely generalizes (held-out conditional 0.857 vs null 1.170,
**+0.31**); by 30,000 conditioning is actively *harmful* (**-0.26**), crossing over
near 21k. Use `checkpoint_7500.pt` from
`latent_continuous_runs/regularized/ar-wd0.1-bright/` if you need a trunk whose
contexts are not memorization-laden.

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

---

## 7. Claims I hold least confidently — audit these first

Ranked by how much they would change the plan if wrong.

1. **"The representation is the problem" is still too broad.** The completed
   controls show that the current AE and per-orbit variance scaling are not
   *necessary* for failure. Patch DCT and full DCT now show that neither local
   frequency coordinates nor global support are fatal. They still do not separate
   periodic Fourier geometry from radial/Hermitian token composition or complex
   coordinates. The active Hartley bridge separates the first two. Until it is
   read visually, say "FFT representation/model pairing," not "Fourier is
   impossible" or "phase is isolated."
2. **The direct controls do not yet validate the proposed AR transfer.** The joint
   controls establish a failure and remove major confounds; they do not establish
   that robust global scaling, exact coefficient metadata, or an amplitude/phase
   factorization will repair the AR + single-token diffusion-decoder path. That is
   the purpose of roadmap phases A and B.
3. **"~10% past Gaussian" as the summary statistic.** Exactly computable and held
   out, but MSE-past-Gaussian is a weak proxy for sample quality — a model could be
   barely past Gaussian and still sample well if its errors were structured
   favourably. The load-bearing evidence is the direct kind: Gaussian samples decode
   to mush resembling ours, and the prefix graft.
4. **The flow prototype is minimal** — affine couplings, 12 layers, fixed
   permutations, 6k steps on 8k images' contexts. Its NLL is dramatically better
   than Gaussian and its samples are not. I read that as "the objective is not the
   bottleneck"; a stronger flow (spline couplings, more data) could be argued for.
   I do not think it changes the conclusion; you might.
5. **Perceptual weighting from decoder sensitivity.** I advocated it for several
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
| `control_pixel_diffusion.py` | **the decisive control.** `--representation {pixels,fft_whitened,fft_global}` |

### Key artifacts
- AE: `autoencoder_runs/ae-causal-ring-t12-m8-perceiver_sector-p256h4-seq2-film_low_rank-z64-r32-s1-n30000-vae-kl0.0001/`
  (plus `-wx0.5` and `-wx0.0` from the whitening sweep; wx0.5 is 35.08 dB, slightly
  *better* than the 34.95 dB baseline)
- Joint baseline: `latent_continuous_runs/joint-vae-mean-rf-w768-l12-b256-s1-n30000/`
- Conditioning arms: `latent_continuous_runs/conditioning/`
- Regularized AR: `latent_continuous_runs/regularized/`
- Pixel control: `latent_continuous_runs/pixel_control/`

---

## 9. Operational notes

- **GPU jobs must use `gpu-claim`** (`/workspace/bin/gpu-claim`, docs in
  `/workspace/GPU_QUEUEING.md`). The box is **shared with other projects** — expect
  to queue; `gpu-claim run ... --wait` handles it.
- Launch long jobs **detached**, or they die with your shell:
  `setsid nohup gpu-claim run ... > log 2>&1 < /dev/null & disown`. I lost a batch
  of eval jobs this way.
- Python `/venv/main/bin/python`. Tests: `python -m pytest tests/ -q` (currently
  130 pass plus 3 subtests).
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

`ROADMAP.md` is now the source of truth. The sequence is:

1. **Direct AR transfer:** raw Cartesian FFT targets, one robust DC-derived global
   scale, no per-orbit variance normalization, explicit coefficient metadata, and
   a test proving the pixel-white-noise/orthonormal-FFT equivalence including the
   required sqrt(2) packing factors. Run only to 2.5k first, then inspect images;
   continue through 5k/7.5k/10k only if justified.
2. **Geometry intervention:** if that baseline still fails, factor each frequency
   into amplitude then conditional circular phase, with different targets, losses,
   noising/integration, and magnitude-gated phase supervision.
3. **Locality/basis controls:** pixel patches, per-patch DCT, and full-image DCT
   have completed. Read the active full-image Hartley bridge next; it separates
   periodic Fourier geometry from the radial/Hermitian FFT token layout without
   introducing angular diffusion.
4. **Structured compressive AE:** build the desired escape hatch only after the
   preceding controls specify what it must repair. It must be genuinely
   compressive, perceptually trained, generatively smooth, frequency-aware, and
   pass an early generator visual gate.

Wavelets are intentionally outside the immediate plan. Architecture escalation
comes only after representation-matched baselines; the current evidence does not
justify another broad conditioning sweep.

Things I would *not* spend on: more conditioning/architecture arms (nine ran, one
small win), loss reweighting (insufficient in principle), timestep schedule changes
(measurably worse), the normalizing-flow direction (tested, no sample improvement),
diffusion forcing (targets a ruled-out cause).

One framing worth keeping, from the user: with every sample broken, **no MSE
variant is a trustworthy guide.** We are trying to fix brokenness, not optimize a
metric. Decode and look at images before believing any number in this document.
