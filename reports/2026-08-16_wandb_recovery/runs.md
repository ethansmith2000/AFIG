# W&B recovery — post-2026-08-13 campaign inventory

Recovered 2026-08-16 from `ethansmith2000/afig-progressive-tokenizer` after
the previous instance (and all local `tokenizer_runs/`, `prior_runs/`, caches,
and `prior_evals` for these arms) was lost. W&B holds **metric histories,
final tokenizer evaluations, and sample-preview images only** — the only
artifact type is `wandb-history`; **no checkpoints or latent caches were
logged**. Anything below needing decoded FID therefore requires retraining to
decide.

## Run inventory (created 2026-08-14 through 2026-08-16)

| run | group | state | last step |
|---|---|---|---:|
| v2-ar-headpos-n64-d16-s1 | ar-prior-conditioning | killed | 26.9k |
| v2-ar-history-robust-n64-d16-s1 | ar-prior-exposure | finished | 40k |
| v2-ar-history-unlabeled-n64-d16-s1 | ar-prior-exposure | killed | 9.6k |
| v2-rolling-cross-n64-d16-o8-s1 | rolling-prior | finished | 20k |
| v2-rolling-cross-n64-d16-o64-s1 | rolling-prior | finished | 20k |
| v2-rolling-cross-n64-d16-o8-b768-s1 | rolling-prior | finished | 10k |
| v2-ar-tokenwise-boundary-n64-d16-s1 | ar-prior-coordinate-basis | killed | 4.2k |
| v2-ar-tokenwise-centered-boundary-n64-d16-s1 | ar-prior-coordinate-basis | finished | 20k |
| v3-vae-cross-n64-d16-s1 | tokenizer-latent-shaping | finished | 15k |
| v3-ar-vae-cross-n64-d16-s1 | ar-prior-token-factorization | finished | 60k |
| v3-cross-n128-d8-s1 | tokenizer-token-factorization | finished | 15k |
| v3-vae-kl1e3-cross-n64-d16-s1 | tokenizer-latent-shaping | finished | 15k |
| v3-vae-kl1e4-cross-n64-d16-s1 | tokenizer-latent-shaping | finished | 15k |
| v3-cross-causal-n64-d16-s1 | tokenizer-latent-shaping | finished | 15k |
| v3-ar-cross-n128-d8-s1 | ar-prior-tokenizer-shaping | finished | 60k |
| v3-ar-vae-kl1e4-cross-n64-d16-s1 | ar-prior-tokenizer-shaping | finished | 60k |
| v3-ar-vae-kl1e3-cross-n64-d16-s1 | ar-prior-tokenizer-shaping | finished | 60k |
| v3-ar-cross-causal-n64-d16-s1 | ar-prior-tokenizer-shaping | finished | 60k |

## Recovered v3 tokenizer final evaluations (10k test)

Held-out prefix PSNR (dB) at K tokens; v2 deterministic cross-only `64x16`
baseline for reference (from `TOKENIZER_DESIGN.md`: 20.57 / 22.66 / 25.21 /
28.86 / 34.06 dB at 64/128/256/512/1024 coordinates, i.e. K=4/8/16/32/64).

| tokenizer | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 | K=64 (full) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3-vae (weighted_kl ~1e-6) | 17.34 | 18.84 | 20.51 | 22.58 | 25.38 | 29.07 | 34.36 |
| v3-vae-kl1e4 | 17.46 | 18.91 | 20.63 | 22.69 | 25.38 | 28.91 | 34.66 |
| v3-vae-kl1e3 | 17.43 | 18.83 | 20.57 | 22.58 | 25.33 | 29.30 | 32.19 |
| v3-cross-causal | 17.27 | 18.73 | 20.47 | 22.46 | 25.35 | 28.73 | 32.45 |
| v3-cross-n128-d8 | 15.93 | 17.20 | 18.59 | 20.13 | 22.20 | 24.91 | 31.06 @K=128 |

Readings (reconstruction side only): the VAE variants at kl<=1e-4 match the
deterministic v2 ceiling (34.4-34.7 vs 34.06 dB) with essentially the same
prefix curve; kl1e-3 costs ~2.4 dB of ceiling; the causal-encoder variant
costs ~1.6 dB; `128x8` loses ~3 dB of ceiling and is behind at every matched
coordinate budget.

## Decision status

The four 60k v3 AR priors (n128d8, vae-kl1e4, vae-kl1e3, cross-causal, plus
the earlier v3-ar-vae) have **no decoded FID anywhere** — their
`prior_evals/` outputs were on the lost box, and W&B summaries carry only
teacher-forced losses, which are inadmissible across different tokenizers /
normalizations under the project's standing metric rules. The
tokenizer-shaping comparison is therefore currently **undecided**, and
deciding it requires re-training tokenizers -> caches -> priors -> FID evals
(checkpoints were never uploaded). Likewise the rolling-prior (crescendo),
history-robust, headpos, and tokenwise-normalization arms have training
histories but no decoded verdicts recorded here; if decoded FIDs were
computed before the box died, they exist only in memory/terminal history and
should be transcribed into the journal from recollection with an
"unverified" flag, or re-measured.

Operational rule going forward (workspace is not a volume on the current
box): after every FID eval, sync `prior_evals/` into git, and upload selected
checkpoints + caches as W&B artifacts or to HF Hub before the next arm
starts.

## The code is lost too — recovered design specs from configs

The last commit (2026-08-14T00:27Z) predates every run above. None of the
following features exist in the committed `progressive_tokenizer/` stack or
trainers; the W&B configs below are the surviving reimplementation spec.

- **Rolling prior** (`v2-rolling-*`): NOT the trunk+head AR model — a
  12-block width-512 headless model (70,293,520 params, same as the joint
  prior) with per-token data time
  `local_data_time = clamp(frontier - token_index / overlap, 0, 1)`,
  flat MSE **over active registers only**. Arms: `active_window=8`
  (frontier_duration 8.875), `active_window=64` (frontier_duration 1.984375,
  i.e. a full-sequence skewed schedule), and an o8 batch-768/10k variant.
  This is the "per-token-time trunk-as-denoiser" design: attention over
  partially-denoised history at every step, AR-ness expressed purely in the
  noise schedule.
- **History-robust AR** (`v2-ar-history-robust`): teacher-forced trunk inputs
  noised with `history_noise_max=0.1, min=0, probability=0.75,
  ramp_steps=4000`, plus `history_reliability_conditioning=true` — a
  per-token reliability/noise-level embedding into the trunk (a
  diffusion-forcing precursor). The `history-unlabeled` arm removed the
  reliability embedding and was killed at 9.6k.
- **Headpos AR** (`v2-ar-headpos`): `head_position_conditioning=true` —
  target-slot embedding passed directly to the head's condition fusion
  (near-term control #1). Killed at ~27k.
- **Tokenwise-boundary AR** (`v2-ar-tokenwise-*`):
  `head_tokenwise_boundary_projections=true` — per-token input/output
  projections in the flow head; the surviving variant is the "centered"
  one (finished 20k).
- **v3 tokenizers**: `variational=true` with `kl_weight` in
  {1e-6, 1e-4, 1e-3} on the `64x16` cross-only architecture (pool_depth 1),
  plus `pool_type=cross_causal` (causal pooling — register i computed
  without access to later registers), plus `128x8` deterministic. All 15k
  steps, batch 512, otherwise the v2 recipe.
- **v3 AR arms** all reuse the unchanged trunk+head AR prior (76.6M params)
  at 60k on the respective caches; per-cache tensor-wide normalization
  constants are recorded in each run config.
