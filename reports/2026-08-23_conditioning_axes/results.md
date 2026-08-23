# Axis A / Axis B: what a wide SNR spectrum is missing (2026-08-23)

Prompted by the observation that "the latent already has a crossing spread of
0.905" settles nothing: a wide spectrum is near-automatic for any anisotropic
high-dimensional data. What makes an SNR spectrum *useful* for diffusion is two
further properties, both measurable on an existing cache with no training run
(`scripts/conditioning_axes.py`):

- **Axis A -- consistency.** Does an individual sample resolve in the
  population's order? If not, the schedule is only right on average.
- **Axis B -- conditioning gain.** Do directions that clear the noise floor
  early help predict the ones that clear it later? In the population eigenbasis
  the directions are uncorrelated *by construction*, so the entire linear
  conditioning gain is exactly zero and any real value is higher-order. Energy
  (squared-magnitude) correlation is therefore a clean probe: exactly 0 for a
  Gaussian of any covariance, strongly positive for natural images, whose
  scale-mixture structure is what makes coarse-to-fine conditioning pay.

## Results (20k samples, each source in its own eigenbasis)

| source | FID | crossing spread | axis A | axis B | S/N | top8 vs 8-32 | top8 vs 32-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CIFAR-10 pixels** | — | 0.963 | **0.831** | **0.117** | 16.6 | **+0.048** | **+0.037** |
| vae-kl1e4 | 35.85 | 0.905 | 0.712 | 0.050 | 7.0 | +0.002 | −0.009 |
| energycv | 39.03 | — | 0.728 | 0.046 | 6.6 | +0.020 | 0.000 |
| ramp | 40.72 | — | 0.635 | 0.052 | 7.4 | +0.003 | 0.000 |
| det | 41.16 | — | 0.731 | 0.049 | 7.0 | +0.021 | +0.003 |
| frontier | 47.60 | — | 0.778 | **0.067** | 9.5 | +0.019 | +0.020 |

S/N is the off-diagonal energy correlation over the finite-sample null
(1/sqrt(N) = 0.0071), so every value is real signal, not estimation noise.

## What holds

**The latent-vs-pixels gap is large and consistent.** Every arm sits at axis B
0.046-0.067 against CIFAR's 0.117 -- 2-2.5x weaker -- and every arm's top-8
directions condition later ones at ~0 against CIFAR's +0.048/+0.037. We have the
magnitudes without the dependence that makes early resolution informative. This
is a credible mechanism for "why this latent is harder to diffuse than pixels",
and it is *higher-order*, so no magnitude rescaling can address it.

## What does NOT hold

**Within the five arms, axis B does not predict FID -- it mildly anti-predicts.**
Frontier has the highest axis A (0.778) and axis B (0.067) and the worst FID;
vae has the best FID and the weakest top-8 conditioning (+0.002 / -0.009). So
axis B is *not* the variable separating our arms. Frontier is plausibly
dominated by other pathologies (channel kurtosis 54, brittlest decoder) but that
is a hypothesis. n=5, single seed, and the cross-arm spread (0.046-0.067) is
small next to the gap to pixels (0.117). Do not repeat the v5 mistake of
promoting an n=5 correlation to a mechanism.

## Predictions this makes

1. **Matched-budget pixel baseline.** If pixels beat 35.85 substantially, the
   axis-B gap is the leading explanation for the representation tax.
2. **Unordered control.** vae's *negative* top8-vs-later correlation looks like
   competition for a fixed energy budget, plausibly induced by the nested-prefix
   objective. Prediction: a non-progressive AE at matched everything shows
   *positive* top8-vs-later energy correlation. If so, the progressive objective
   is destroying the cross-scale conditioning it was meant to create -- which
   would be the sharpest result the project has produced.
3. **An AE objective targeting axis B directly** (make top eigendirection
   magnitudes predictive of the rest, i.e. induce scale-mixture structure) is
   the first shaping intervention aimed at a quantity that scaling provably
   cannot reach.
