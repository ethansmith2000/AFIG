# Regularized whitening feasibility

Date: 2026-09-05

Status: complete. The audit trained no model. Its frozen numerical gates pass
and authorize the matched prior factorial described below.

## Decision

Use the factorized sequence/channel readout with a relative forward-gain cap of
`16`. It rotates the latent into 64 sequence eigenmodes, orders those tokens by
population power, uses the channel eigenmodes as their 16 features, and applies
a capped diagonal whitening gain to every product coordinate. The decoder
inverts the complete transform before consuming a latent, so its physical
interface is unchanged.

For the ordered-time arms, use beta `.25` and the 64 exact SNR=1 crossings in
`metrics.json`. They span `.4332-.6045`, corresponding to only a `2x` range of
rational-path odds. For the weighted-loss arms, use the mean-one rectified-flow
target-energy weights. They span `.7326-1.3913`, a `1.899x` range. The clean
whitened coordinates remain unit-scale; the original power hierarchy is used
only to derive the explicit clock and weighting.

## Whitening result

All statistics below use 25,000 training latents to fit the bases and the
disjoint 10,000 CIFAR-10 test latents for evaluation.

| readout | gain cap | held-out effective rank | off-diagonal Frobenius fraction | numerical verdict |
|---|---:|---:|---:|---|
| original native coordinates | — | 369.88 | .9439 | baseline |
| factorized | 4 | 584.59 | .6126 | pass |
| factorized | 8 | 720.76 | .6068 | pass |
| **factorized** | **16** | **794.98** | **.5844** | **selected** |
| factorized | 32 | 795.29 | .5842 | rejected by gain cap |
| flattened PCA | 4 | 629.50 | .2547 | pass |
| flattened PCA | 8 | 810.61 | .3252 | pass |
| flattened PCA | 16 | 926.86 | .3653 | healthy control |
| flattened PCA | 32 | 931.17 | .3676 | rejected by gain cap |

Cap 8 reaches only 82.5% of the factorized effective-rank improvement obtained
at cap 16, so it misses the frozen 95% rule. Cap 32 adds only `.31` effective
rank and exceeds the maximum permitted gain. Cap 16 is consequently the
smallest qualifying factorized cap.

Flattened PCA whitens more completely, as expected, but the factorized readout
passes every gate while preserving the directly observed 64-step sequence-mode
ordering. The remaining factorized off-diagonal covariance is intentional
evidence that the transform is not being mistaken for full whitening.

The selected transform's held-out coordinate variances have median `1.018` and
5th/95th percentiles `.978/1.051`. A near-null terminal sequence constraint is
not forced to unit variance: the minimum remains `.000329`, exactly the weak
tail the gain cap is meant to protect.

## Inversion safety

The selected factorized basis has maximum orthogonality error `7.15e-7` after
numerically re-orthogonalizing the float32 eigenspaces. Its errors are:

- float32 latent round-trip relative RMS: `7.09e-7`;
- simulated float16-cache latent round-trip relative RMS: `2.06e-4`;
- decoded-image pixel delta RMS after the float16 round trip: `.001211`.

All are comfortably within the frozen `1e-5/.002/.002` gates. An initial audit
pass correctly caught that TF32 matrix multiplication and the raw float32
sequence eigenbasis were slightly too imprecise for the first gate. The final
run disables TF32 for the audit and uses QR only to repair the numerical
orthogonality of the same ordered eigenspaces; it does not change their order
or the scientific protocol.

## Schedule and loss choice

The power floor used for gain cap 16 is also applied before deriving the clock
and weights, preventing the single near-null terminal direction from setting
the dynamic range.

| beta | odds range | SNR=1 crossing range | signal-metric weight range | flow-target weight range | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 1x | .500-.500 | 1x | 1x | exact common control |
| .125 | 1.414x | .466-.553 | 2x | 1.395x | pass |
| **.25** | **2x** | **.433-.605** | **4x** | **1.899x** | **strongest full pass** |
| .5 | 4x | .369-.700 | 16x | 3.321x | rejects flow-target bound |

The weighted arm uses the gentler flow-target profile rather than the
signal-only profile because the actual rectified-flow target is clean latent
minus unit noise. This also avoids repeating the aggressive image-variance
weight transfer that previously failed.

## Authorized experiment

Build one exact transformed float16 cache and train four parameter-matched
priors at tokenizer seed 2 / prior seed 1:

1. common time, uniform loss;
2. ordered beta-.25 time, uniform loss;
3. common time, flow-target-energy loss;
4. ordered beta-.25 time, flow-target-energy loss.

The unwhitened selected v27 common-time prior remains the external baseline.
Whitened common/uniform isolates the transform; the remaining differences form
the schedule-by-loss factorial. Model selection remains decoded FID/KID, not
the covariance statistics in this audit.

## Artifacts

- [Exact metrics](metrics.json)
- [Frozen selected transform](selected_transform.pt)
- [Held-out covariance spectra](heldout_covariance_spectra.png)
- [Effective-rank sweep](whitening_effective_rank.png)
- [Schedule profiles](schedule_profiles.png)
- [Frozen protocol](../regularized_whitening_protocol.json)
