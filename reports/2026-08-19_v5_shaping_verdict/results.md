# v5 shaping matrix — closed verdict (2026-08-19)

Five tokenizer arms, all `64x16` cross-only, identical 60.0M architecture,
identical recipe, same box and same commit; each cached with horizontal-flip
views, then judged by an identical 60k-step joint prior and 5k-sample decoded
FID. This is the first fully param-matched, control-included shaping matrix.

## Verdict

| arm | FID | KID | eff-rank | tok-profile-corr | chan-order-consist. | chan CV2 | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| vae-kl1e4 | **35.85** | 0.02627 | 100.7 | 0.312 | 0.537 | 4.76 | ~33.9 |
| energycv | 39.03 | 0.03014 | 26.3 | 0.172 | 0.551 | 4.23 | 34.02 |
| ramp | 40.72 | 0.03050 | 519.6 | 0.725 | 0.508 | 4.84 | — |
| det (control) | 41.16 | 0.03138 | 61.0 | 0.587 | 0.540 | 5.12 | 32.88 |
| frontier | 47.60 | 0.03871 | 48.8 | **0.908** | 0.514 | 49.58 | 31.41 |

Protocol noise is +-0.3 FID, so the ordering is real at every gap.

## 1. Schedule-consistency shaping is a dead end (and the gauge argument holds)

Rank-correlating `token_profile_correlation_mean` against FID gives Spearman
**+0.8** — the wrong sign. The two arms that explicitly shaped the *resolving
schedule* are the two that failed:

- **ramp** raised profile correlation 0.587 -> 0.725 and landed 40.72, inside
  noise of the 41.16 control. Zero payoff.
- **frontier** drove it to 0.908 — by far the most schedule-structured code the
  project has produced — and paid **1.5 dB of reconstruction and 6.4 FID** for
  it.

This is the gauge freedom confirmed experimentally rather than on paper. The
per-token energy profile is settable by an invertible per-token rescale, so it
cannot move the conditional order; frontier spent real encoder capacity buying a
quantity that was free all along, and the bill arrived in both reconstruction
and generation. **Standing rule: no further token-axis / schedule shaping.**

Caveat on ramp's eff-rank 519.6: the sigma-ramp injects noise into the cached
latents, flattening the spectrum. That number is an artifact of the
intervention, not evidence of a richer code.

## 2. The immovable number

`channel_eigen_order_consistency_mean` reads **0.508–0.551 in all five arms** —
chance, in every one. Five very different objectives (KL, coordinate kurtosis
penalty, sigma-ramp, frontier noise, plain reconstruction) and not one of them
moved which channel eigendirection resolves first on a given sample.

That is exactly what the theory predicts: conditional order is not settable by
any second-order intervention. It also identifies where the remaining headroom
is. Both arms that *won* act on channel-axis statistics — vae on the marginal,
energycv on the kurtosis diagonal — and both cut channel CV2 relative to the
control. Token axis: gauge. Channel axis: live.

## 3. Frontier's channel CV2 of 49.58

Ten times every other arm. Previously discounted as a pooling artifact (its own
Gaussian surrogate reads 41.1), but the absolute level says the frontier noise
pushed information into extreme within-token per-channel dynamic range. A prior
must then model a code with ~50x activity variance along the channel axis. This
is a plausible mechanism for the 6.4-FID loss independent of the gauge argument
above, and the two readings are compatible.

## 4. Reconstruction dB is confirmed non-predictive

energycv has the best PSNR of the matrix (34.02) and loses to vae (~33.9) by
3.2 FID. frontier's 31.41 dB and its last-place FID happen to agree, but
det-vs-energycv (32.88 -> 34.02 dB buying 2.1 FID) and vae-vs-energycv
(dB tie, 3.2 FID apart) confirm the standing rule: **dB is a capacity check,
never a selection metric.** Any AE capacity or architecture change must re-run
the matched-prior judge.

## Follow-ups this sets up

1. Channel pair-slice measurement `E[z_i^2 z_j^2]` on all five caches
   (`scripts/channel_pair_slice.py`) — decides whether an off-diagonal
   energy-covariance regularizer has anything to bite on.
2. Decoder-sensitivity curves (det / vae / energycv) — separates "vae widened
   the decoder's runway" from "vae smoothed the density".
3. Engine comparison (roll8 / roll64 / ar / arrobust) on the vae cache against
   the 35.85 joint reference.
