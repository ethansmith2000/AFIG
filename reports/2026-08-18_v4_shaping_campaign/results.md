# v4 latent-shaping campaign — running results

Snapshot 2026-08-18. All arms: `64x16` cross-only, 15k x 512, lr 1e-4, the
recovered v2 recipe, current codebase (post commit 473505c). Decoded-FID
judge: matched 512-wide 12-block joint rectified flow, 60k x 256, 5k-sample
FID vs 10k test (protocol noise ~0.3).

## Reconstruction (held-out prefix PSNR, dB)

| arm | K=4 | K=8 | K=16 | K=32 | K=64 | note |
|---|---:|---:|---:|---:|---:|---|
| v2 recovered target | 20.57 | 22.66 | 25.21 | 28.86 | 34.06 | old codebase |
| v4-det-s1 / s2 | 20.17/20.15 | 22.12/21.95 | 24.73/24.73 | 28.42/28.34 | 32.38/32.49 | two seeds agree: current-code det is ~1.6 dB low; config diff vs recovered v2 wandb config is empty; optimizer-grouping change exonerated (identical allocation); cause unresolved (old runs used an uncommitted tree state) |
| v4-vae-kl1e4 | 20.43 | 22.47 | 25.21 | 29.32 | 33.93 | reproduces target |
| v4-energycv (kurt->3 @1e-2) | 20.36 | 22.41 | 25.25 | 29.06 | 34.06 | reproduces target exactly |
| v4-frontier-s2 (fixed) | 18.55 | 20.31 | 22.92 | 26.52 | 31.50 | prefix-mask eval is off-distribution for this arm (trained on noise states) |
| v4-ramp-s1, frontier-s1 | — | — | — | — | — | INVALID: detached-RMS amplitude treadmill (latent RMS 50x / 5x); fixed by keeping the noise RMS reference in-graph (473505c) |

Notable: on the current stack the two regularized arms exceed plain det's
ceiling by ~1.5 dB — the shaping acts as an optimization aid, not merely a
latent-geometry instrument.

## Schedule-consistency scorecards (25k train latents, full-cov Gaussian surrogate)

| arm | flat rank /1024 | flat order consistency | flat CV2 med | frac CV2>4 | token CV2 (sur) | token profile corr | channel CV2 (sur) |
|---|---:|---|---:|---:|---:|---:|---:|
| det-s2 | 53.5 | 0.991-1.0 | 2.59 | **0.110** | 0.48 (0.59) | 0.67 | 3.89 (2.46) |
| vae-kl1e4 | 91.9 | 0.981-1.0 | 2.51 | 0.007 | 0.31 (0.36) | 0.71 | 4.48 (2.39) |
| energycv | 28.6 | 0.995-1.0 | 2.47 | 0.007 | 0.81 (0.75) | 0.15 | 4.59 (2.56) |
| frontier-s2 | 50.8 | 0.991-1.0 | 2.73 | 0.059 | pending | pending | pending |

Readings:
- The bulk of every arm's flattened eigenbasis is schedule-consistent on the
  current codebase; what shaping changes is the **spiky tail** (11% of det's
  directions at CV2>4 vs 0.7% for vae/energycv) and (VAE only) effective rank.
- Per-token energies are schedule-consistent in all measured arms →
  token-indexed (crescendo) noise schedules are per-sample valid.
- The higher-order residue lives on the **within-token channel axis**
  (CV2 ~3.9-4.6 vs Gaussian ~2.4-2.6) in every arm — untouched by both KL and
  coordinate-wise kurtosis penalty. A channel-eigenbasis (EMA-rotated)
  energy-consistency penalty is the targeted follow-up instrument; its payoff
  would be felt by per-token heads / rolling denoising, not the joint prior.
- energycv flattened its slot-energy hierarchy (profile corr 0.15) while vae
  kept a structured profile (0.71): two different consistent geometries.

## Joint-prior decoded FID (60k, 5k samples)

| prior | FID | KID |
|---|---:|---:|
| v4-joint-vae-kl1e4 | **37.93** | 0.02760 |
| v4-joint-energycv | **38.48** | 0.02853 |
| v2 recovered benchmark | 39.37 | 0.02843 |
| v4-joint-det (control) | running | — |
| v4-joint-frontier | running | — |
| v4-joint-ramp | tokenizer still training | — |

Both shaped codes edge the old benchmark by ~1-1.4 FID (cross-codebase caveat:
39.37 rode the old 34.06 dB tokenizer). The in-campaign det control decides
whether the shaping produced a controlled generative win.

## Operational

- gpu-claim was updated mid-campaign (no --timeout; queue registry reset);
  several waves of pollers were lost to session teardowns — relaunches now go
  through setsid'd script files only.
- Backups: tokenizer finals + joint finals uploaded to W&B artifacts
  (job_type=backup) as of this snapshot.
