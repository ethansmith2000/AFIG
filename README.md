# AFIG — continuous whole-image latents

AFIG studies learned continuous image tokenizers and matched generative priors
on CIFAR-10. The current question is no longer whether prefix ordering alone
makes a representation easier to generate; it does not under the tested setup.
The program now optimizes the joint frontier of reconstruction distortion,
matched-prior modelability, and decoder robustness.

See [ROADMAP.md](ROADMAP.md) for active decisions and
[EXPERIMENT_JOURNAL.md](EXPERIMENT_JOURNAL.md) for the chronological evidence.

## Current leading design

```text
32x32 RGB image
  -> 64 non-overlapping 4x4 patch features
  -> 7 patch Transformer blocks
  -> 64 learned queries with a cross read + residual latent self-attention/FFN
  -> deterministic 64x16 continuous latent
  -> scale-invariant sample-varying slot-power balance penalty (weight 0.002)
  -> decoder training with latent-RMS Gaussian jitter (sigma 0.05)
  -> 64 spatial decoder queries
  -> reconstructed image
```

- Tokenizer objective: full reconstruction plus weak slot-power balancing.
- Encoder/decoder width 512, eight heads, QK RMS normalization.
- One tensor-wide latent mean/std for the prior.
- Prior: 12-block width-512 bidirectional rectified flow.
- Sampling: 50-step Heun.
- Selection: decoded FID/KID, with reconstruction and latent-noise sensitivity
  as representation gates.

The historical hard-clamped variational control is effectively deterministic:
almost all posterior log-variance values pin to the floor. Controlled posterior
experiments selected decoder-only jitter for expected generation quality while
retaining the hard-VAE path as a conservative stability control.

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
- Decoder-only sigma-0.05 latent jitter improves expected matched-prior
  generation across tokenizer and prior seeds, although one tokenizer seed
  regresses; it is the retained expected-value training design.
- Weak slot-power balancing improves 10k FID at all three tokenizer seeds. Mean
  FID improves from 25.70 to 23.75 and mean KID from 0.01762 to 0.01733, so it
  is now part of the leading design. KID regresses at seed 2 and remains an
  explicit interaction caveat.

## Current follow-up

The slot-balance campaign is complete and passes its predeclared multi-seed
promotion gate. The most efficient remaining robustness check is a prior-seed-2
run on the mixed tokenizer-seed-2 slot-balanced cache, paired with the already
completed unregularized seed-2/prior-2 control. This tests prior stochasticity
without training another tokenizer. Reconstruction remains only a permissive
codec-health veto; decoded FID/KID selects representation quality. Details and
exact stop rules are in `reports/2026-08-26_autoencoder_program/plan.md`.

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
