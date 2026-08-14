# AFIG progressive-prior diagnostic source notes

Snapshot date: 2026-08-13 UTC.

## Canonical project evidence

- `TOKENIZER_DESIGN.md`: experiment design, tokenizer reconstruction and prefix
  measurements, latent-axis diagnostics, matched joint-prior results, oracle-prefix
  sweep, and current interpretation.
- `prior_evals/v2-ar-cross-n64-d16-{010000,020000,040000,060000}/metrics.json`:
  decoded 5,000-sample FID/KID for the single-register causal prior.
- `prior_evals/v2-block-ar-cross-n64-d16-b4-{010000,020000,040000,060000}/metrics.json`:
  decoded 5,000-sample FID/KID for the four-register concatenated causal prior.
- `prior_evals/v2-joint-cross-n64-d16-s1-final/metrics.json` and the matched
  `16x64` and `32x32` joint evaluation directories: joint-prior comparisons.
- `prior_runs/v2-ar-cross-n64-d16-s1/metrics_latest.json` and numbered training
  histories: teacher-forced loss and per-token MSE.
- `evaluate_progressive_ar_prefix.py` and `evaluate_progressive_prefix_only.py`:
  the two sides of the oracle-prefix diagnostic.
- `progressive_tokenizer/autoregressive_flow.py`,
  `train_progressive_ar_flow.py`, and
  `scripts/run_progressive_ar_cross_n64_d16.sh`: architecture, objective,
  optimizer, and sampling configuration.

## Transformations used in the report

- All FID/KID values are copied from completed metric JSON files or the canonical
  design log. FID comparisons use 5,000 generated samples unless explicitly noted.
- Cumulative loss shares are sums of the final held-out per-token MSE divided by
  the sum across all 64 tokens. The uniform reference is `prefix_length / 64`.
- The AR joint gap is `81.6782 - 39.3743 = 42.3039` FID.
- The single-register checkpoint curve uses the same checkpoint family, evaluator,
  50-step Heun sampler, sample count, and reference statistics at every step.
- The four-register block curve changes only the stochastic factorization from
  `[64,16]` targets to `[16,64]` targets before restoring the original tokenizer
  layout for decoding.

## Interpretation limits

- FID differences below roughly 0.3 are treated as protocol noise based on prior
  repeated evaluations in this project.
- Teacher-forced flow MSE contains position-dependent irreducible conditional
  variance and is not directly comparable across sequence positions.
- Covariance, correlation, and effective-rank statistics characterize the latent
  geometry; they are not standalone latent-quality objectives.
- Reconstruction FID 7.22 is from the current 5,000-sample prefix-only evaluation;
  it should not be treated as a precise replacement for prior 10,000-sample codec
  reconstruction measurements.
