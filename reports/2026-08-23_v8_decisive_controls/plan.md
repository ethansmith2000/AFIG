# v8 decisive controls — plan and live record (2026-08-23)

## Question

Does the nested-prefix representation improve full-length diffusability, and
how much of the current gap is due to learned representation geometry,
dimensional bottleneck, or literal token/feature factorization?

## Shared protocol

All GPU phases use the lifetime-locking launcher from
`/workspace/GPU_QUEUEING.md`; no phase binds a GPU directly. Canonical prior
recipe: unconditional joint RF, width 512, depth 12, 8 heads, RMS QK norm,
batch 256, AdamW 1e-4, 1k warmup, 60k steps, BF16/TF32, tensor-wide scalar
normalization, 50-step Heun, decoded FID/KID on 5,000 samples with seed 54321.

The fixed evaluation seed controls the sample comparison but does not estimate
decision variance. Treat differences below approximately 2 FID as unresolved.

## E1 — matched pixel-space baseline (complete)

**Hypothesis.** If raw pixels substantially beat the selected learned latent at
the same prior recipe, a representation tax exists. This arm does not by itself
separate geometry from dimensional rate.

- Representation: reversible raster `4x4` patches, `64x48 = 3,072` scalars.
- Train cache: CIFAR-10 originals plus deterministic horizontal flips (100k).
- Test cache: CIFAR-10 test split (10k).
- Cache: `pixel_runs/v8-cifar10-patches-original-flip.pt`.
- Prior: `prior_runs/v8-joint-pixels-s1`.
- Evaluation: `prior_evals/v8-joint-pixels-060000`.
- Launcher/log: `scripts/run_v8_pixel_control.sh`,
  `prior_runs/v8-joint-pixels-s1.launch.log`.
- Completed: `2026-08-23T10:32:45Z`.
- FID/KID: **27.4104 / 0.01627** at 60k steps and 5k generated samples.
- Standardized sample RMS: `0.9454`; decoded clipping fraction: `0.00706`.
- Initial comparison: this beats the selected progressive VAE latent's 35.85
  FID by 8.44. The gap establishes a representation-and/or-dimensional-rate
  tax under the matched prior recipe; it does not yet separate those causes.

## E2 — matched unordered tokenizer (complete)

**Hypothesis.** Removing nested-prefix reconstruction while holding architecture,
nominal rate, stochasticity, optimizer, cache augmentation, and joint prior
fixed measures whether progressiveness helps or taxes full-length generation.

- Tokenizer: `64x16 = 1,024` continuous scalars, cross-only pooling.
- Objective: full reconstruction only; variational KL weight `1e-4`.
- Historical compatibility: hard `[-8,8]` log-variance clamp, matching the v5
  progressive VAE rather than silently mixing in the repaired soft bound.
- Tokenizer: `tokenizer_runs/v8-unordered-vae-s1`.
- Prior: `prior_runs/v8-joint-unordered-vae-s1`.
- Evaluation: `prior_evals/v8-joint-unordered-vae-060000`.
- Launcher/log: `scripts/run_v8_unordered_control.sh`,
  `tokenizer_runs/v8-unordered-vae-s1.launch.log`.
- Tokenizer completed: full test PSNR **35.88 dB** versus approximately 34.04
  dB for the progressive VAE. Several unordered slots have RMS near 0.16 while
  most are near 1.0, so the full-only objective did not use all nominal token
  positions uniformly; this is a measured representation consequence, not a
  reason to alter the arm post hoc.
- Cache and axis scorecard completed. Flattened covariance effective rank is
  **241.95**, versus approximately 100 for the selected progressive VAE.
- The matched 60k joint prior completed at **FID/KID 29.9251 / 0.02045**.
  This beats the progressive VAE's 35.85 by 5.93 and sits 2.51 behind raw
  pixels at 27.41. The progressive objective therefore accounts for most of
  the original pixel/latent gap under this recipe.
- Clean reconstruction FID is **6.081**. Noise sensitivity at sigma
  `0.05 / 0.1 / 0.2 / 0.4` is FID
  `6.305 / 7.204 / 11.861 / 35.544`. The progressive VAE's corresponding
  curve was `8.0 / 13.2 / 39.2 / 124.4` (clean 6.9), so most of the 5.93
  generative improvement cannot be assigned to the approximately 0.8-point
  clean reconstruction-floor improvement. The unordered representation is
  dramatically more tolerant of plausible prior error.

## E3 — trained-prior context ablation (complete)

**Question.** Does the trained joint denoiser actually use correct early context
to reduce late-direction velocity MSE relative to batch-shuffled or
mean-ablated context?

This measures context used by one trained model, not intrinsic Bayes conditioning
gain of the representation. Report covariance-eigenband and literal-token-prefix
ablations separately.

- Existing checkpoint: `prior_runs/v5-joint-vae-kl1e4-s1/checkpoint_final.pt`.
- Existing cache: `tokenizer_runs/v5-vae-kl1e4-s1/latents_final_original_flip.pt`.
- Output: `reports/2026-08-23_context_ablation/v5-vae-joint.json`.
- Tool/log: `scripts/conditioning_context_ablation.py`, `context_v8_vae.log`.
- First attempt claimed a GPU and exited before loading data because direct
  script execution omitted the repository root from `sys.path`. The launcher
  path was repaired and smoke-checked; the diagnostic was requeued through
  `gpu-claim`. No checkpoint or experimental output was modified by the failed
  attempt.
- The repaired run completed on 2,048 held-out examples. Correct early
  eigen-context reduces late-direction velocity MSE relative to shuffled
  context by `2.7 / 11.8 / 20.1 / 34.5 / 61.6%` at
  `t = 0.2 / 0.35 / 0.5 / 0.65 / 0.8`. The number of high-SNR early
  directions grows from 1 to 256 across that curve.
- Literal token-prefix context is also used. At `t=0.5`, true versus shuffled
  context improves tail MSE by `22.2%` for prefix 8, `44.1%` for prefix 16,
  and `68.7%` for prefix 32.
- Verdict: the trained joint prior demonstrably uses sample-specific early and
  prefix context. This rejects the earlier inference from near-zero top-band
  energy correlation that “early directions condition nothing.” It does not
  show that this utility is unique to progressive latents, nor that it offsets
  the distortion and decoder brittleness imposed by prefix training.

## E4 — literal latent shape at fixed representation (planned, not queued)

The selected latent has a nominal dimensional bottleneck of `1,024 / 3,072 =
1/3`, but continuous unquantized coordinates do not define a bit rate. Separate:

1. **Dimensional rate:** total scalars `S = tokens x features`.
2. **Factorization:** token/feature aspect ratio at fixed `S`.

First perform an exact reshape of the same v5 VAE cache and invert the reshape
before the unchanged decoder:

| layout | scalars | purpose |
|---|---:|---|
| `32x32` | 1,024 | fewer, wider attention tokens |
| `64x16` | 1,024 | existing baseline |
| `128x8` | 1,024 | more, narrower attention tokens |
| `16x64` | 1,024 | optional endpoint if the curve needs it |

This holds the images, learned representation, scalar order, tokenizer,
decoder, normalization, and information exactly fixed. Record both equal-step
FID and throughput/GPU time because attention cost grows quadratically with
token count.

Then isolate dimensional rate with a learned `64x48` autoencoder arm. Comparing
raw pixels `64x48`, learned `64x48`, and learned `64x16` separates learned
geometry/decoder tax from the nominal dimensional bottleneck. Choose the
progressive or unordered objective for that rate arm only after E2 resolves
whether prefix training is beneficial.

## Predeclared interpretation

| Observation | Interpretation |
|---|---|
| unordered approximately equals progressive | no measured full-FID benefit from prefix ordering |
| unordered beats progressive | nested-prefix training is a full-generation tax |
| progressive beats unordered by more than noise | ordering thesis survives; repeat another seed |
| pixels beat both learned `64x16` arms | representation and/or dimensional-rate tax exists |
| exact cache reshape changes FID | joint-prior token/feature factorization matters |
| learned `64x48` approximately equals pixels | dimensional bottleneck explains most representation tax |
| learned `64x48` still loses to pixels | learned geometry or decoder robustness remains a tax |

## Interim verdict after E1--E3

The original 8.44-point pixel-versus-progressive gap is mostly an
objective-induced representation tax, not evidence that 1,024 continuous
coordinates are inherently insufficient. Removing prefix training at the same
nominal dimensional rate closes 5.93 points and increases effective rank from
about 100 to 242. Clean reconstruction improves, but the larger change is
decoder robustness away from the encoder manifold.

The remaining 2.51-point raw-pixel advantage is unresolved at one seed and
conflates dimensional bottleneck, learned coordinate geometry, and the decoder.
The decisive next rate control is an unordered learned `64x48` tokenizer and
matched prior, which has the same literal `64x48` prior shape and 3,072 scalar
dimensions as the pixel arm.

## E4a — unordered learned `64x48` rate control (launched 2026-08-25)

This is the direct no-dimensional-bottleneck comparison selected by E2. It uses
the same unordered/full-reconstruction objective as the winning `64x16` learned
arm, but expands each of the 64 tokens from 16 to 48 features. The resulting
3,072 continuous coordinates and literal `64x48` prior input shape exactly match
the pixel control. Tokenizer steps (15k), prior architecture and steps
(`512x12`, 60k), data, augmentation, optimizer settings, evaluation, and seed are
otherwise matched.

- Tokenizer/cache: `tokenizer_runs/v9-unordered-vae-n64d48-s1/`.
- Prior: `prior_runs/v9-joint-unordered-vae-n64d48-s1/`.
- Evaluation: `prior_evals/v9-joint-unordered-vae-n64d48-060000/`.
- Launcher: `scripts/run_v9_unordered_rate_control.sh`.
- Queue protocol: every exclusive phase is a separate lifetime-locked
  `gpu-claim run --owner AFIG ... --wait` invocation.
- Primary decision: compare decoded FID to pixel `27.41` and unordered `64x16`
  `29.93`. A gap below about 2 FID remains unresolved at one seed.
- Secondary diagnostics: clean reconstruction, KID, clipping, effective rank,
  coordinate/slot utilization, and the same latent-noise decoder-sensitivity
  curve used for `64x16`.

Interpretation is predeclared: reaching the pixel arm supports dimensional rate
as the residual limitation; remaining materially worse supports learned
geometry or decoder robustness as an additional tax. This arm cannot identify
bit-rate compression because the latents remain continuous and unquantized.

## Completion checklist

For every arm append: completion timestamp, training seed, checkpoint and cache
paths, final reconstruction metrics, FID/KID, clipping fraction, training
throughput, GPU time, failures/retries, and a one-paragraph verdict. Update
`EXPERIMENT_JOURNAL.md` and commit the durable evidence.
