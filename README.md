# AFIG — continuous whole-image latents

AFIG studies learned continuous image tokenizers and matched generative priors
on CIFAR-10. The current question is no longer whether prefix ordering alone
makes a representation easier to generate; it does not under the tested setup.
The program now optimizes the joint frontier of reconstruction distortion,
matched-prior modelability, and decoder robustness.

See [ROADMAP.md](ROADMAP.md) for active decisions and
[EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) for the chronological evidence.

## Reproducible baseline

```text
32x32 RGB image
  -> 64 non-overlapping 4x4 patch features
  -> 8 patch Transformer blocks
  -> 64 learned queries with one terminal cross-attention read
  -> 64x16 continuous latent
  -> 64 spatial decoder queries
  -> reconstructed image
```

- Tokenizer objective: full reconstruction only.
- Encoder/decoder width 512, eight heads, QK RMS normalization.
- One tensor-wide latent mean/std for the prior.
- Prior: 12-block width-512 bidirectional rectified flow.
- Sampling: 50-step Heun.
- Selection: decoded FID/KID, with reconstruction and latent-noise sensitivity
  as representation gates.

The matched historical control retains the hard-clamped variational head and KL
`1e-4`, but it is effectively deterministic: almost all posterior log-variance
values pin to the floor. A clean deterministic/jitter/soft-floor comparison is
an explicit future stage rather than a silent recipe change.

## Main results

- Removing nested-prefix training improves matched-prior FID from 35.85 to
  29.93. Prefix training buys useful partial decoding, not better full-length
  generation.
- Joint bidirectional generation decisively beats rolling and causal AR
  variants, whose training benefits from ground-truth context do not survive
  sampling.
- Learned latent width has a U-shaped modelability curve: `64x16` is the best
  tested point, while `64x32/48` reconstruct better but generate worse.
- Exact fixed-cache reshapes show that the native `64x16` token boundary matters
  independently of scalar count.
- Static token-magnitude schedules and raw high-rank PCA concentration are
  negative in their tested gauges.
- Residual register refinement produced the best seed-1 checkpoint and repeated
  across two prior seeds, but did not repeat at tokenizer seed 2. It remains a
  checkpoint-level result, not a confirmed architecture effect.

## Active experiment

The current Stage-A screen uses tokenizer seed 3 and exactly 60,056,784
parameters per arm:

1. v8: eight patch blocks plus terminal cross-attention;
2. v12: seven patch blocks plus one residual Perceiver register block;
3. v13: seven patch blocks plus one joint patch/register block and a matched
   register-only adapter.

Reconstruction is now only a permissive codec-health check. Every healthy arm
receives a matched prior, and decoded FID/KID selects latent quality. This avoids
discarding a representation that reconstructs slightly worse but has a simpler
or more useful generative distribution. Details and stop rules are in
`reports/2026-08-26_autoencoder_program/plan.md`.

## Running safely

All exclusive GPU work must use the shared lifetime queue:

```bash
gpu-claim status
gpu-claim run --owner AFIG --job <name> --wait -- <command>
```

Do not bind a GPU directly. The complete protocol is
`/workspace/GPU_QUEUEING.md`.

Seed-3 arms are resumable end-to-end launchers:

```bash
scripts/run_stage_a_tokenizer_seed3_arm.sh v8
scripts/run_stage_a_tokenizer_seed3_arm.sh v12
scripts/run_stage_a_tokenizer_seed3_arm.sh v13
```

Each launcher trains, uploads the final tokenizer checkpoint to W&B, builds the
augmented latent cache, computes the axis diagnostics, and evaluates decoder
sensitivity. Training, caching, and sensitivity phases acquire separate
lifetime GPU claims.

## Tests

```bash
/venv/main/bin/python -m pytest -q tests/test_progressive_tokenizer.py
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python train_progressive_tokenizer.py \
  --smoke --pool_type register_tokens --pool_depth 1 \
  --output_dir /tmp/afig-register-smoke
```

The former Fourier-generation and rolling-prior implementations remain
recoverable from Git history; they are not active generation tracks.
