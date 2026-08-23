# Consistency and energy dependence beyond a wide SNR spectrum (2026-08-23)

> **Correction:** the quantity originally called “Axis B conditioning gain”
> below is squared-energy correlation in a sample-fitted eigenbasis. It is a
> useful higher-order **dependence proxy**, not a direct measurement of how much
> early-resolved values improve prediction or denoising of later values. Raw
> pixel/latent aggregates are also not directly comparable because their
> dimensions, eigenvalue spectra, and band proportions differ. Claims of
> “2–2.5x weaker conditioning,” “early directions condition nothing,” and a
> mechanism that rescaling “provably cannot reach” are retracted pending the
> held-out true-vs-shuffled context ablation.

Prompted by the observation that "the latent already has a crossing spread of
0.905" settles nothing: a wide spectrum is near-automatic for any anisotropic
high-dimensional data. What makes an SNR spectrum *useful* for diffusion is two
further properties, both measurable on an existing cache with no training run
(`scripts/conditioning_axes.py`):

- **Axis A -- consistency.** Does an individual sample resolve in the
  population's order? If not, the schedule is only right on average.
- **Energy-dependence proxy.** Squared-magnitude correlation across directions
  detects higher-order dependence, including global scale mixtures. It does not
  by itself show that early directions reduce uncertainty about later ones.

## Results (20k samples, each source in its own eigenbasis)

| source | FID | crossing spread | axis A | energy-dep. proxy | historical ratio | top8 vs 8-32 | top8 vs 32-128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CIFAR-10 pixels** | — | 0.963 | **0.831** | **0.117** | 16.6 | **+0.048** | **+0.037** |
| vae-kl1e4 | 35.85 | 0.905 | 0.712 | 0.050 | 7.0 | +0.002 | −0.009 |
| energycv | 39.03 | — | 0.728 | 0.046 | 6.6 | +0.020 | 0.000 |
| ramp | 40.72 | — | 0.635 | 0.052 | 7.4 | +0.003 | 0.000 |
| det | 41.16 | — | 0.731 | 0.049 | 7.0 | +0.021 | +0.003 |
| frontier | 47.60 | — | 0.778 | **0.067** | 9.5 | +0.019 | +0.020 |

The former “S/N” column divides an absolute mean correlation by the null
standard deviation of one correlation estimate. It is not an aggregate
significance statistic because the pairwise estimates are dependent; it is
retained only to reproduce the historical table.

## What holds

**The descriptive latent-vs-pixels gap is large and consistent.** Every arm has
lower pairwise squared-energy dependence than raw CIFAR pixels under these
particular bands. This motivates a direct context-utility test, but does not yet
establish weaker conditioning or explain the FID gap.

## What does NOT hold

**Within the five arms, the proxy does not predict FID -- it mildly
anti-predicts.**
Frontier has the highest axis A (0.778) and axis B (0.067) and the worst FID;
vae has the best FID and the weakest top-8 conditioning (+0.002 / -0.009). So
axis B is *not* the variable separating our arms. Frontier is plausibly
dominated by other pathologies (channel kurtosis 54, brittlest decoder) but that
is a hypothesis. n=5, single seed, and the cross-arm spread (0.046-0.067) is
small next to the gap to pixels (0.117). Do not repeat the v5 mistake of
promoting an n=5 correlation to a mechanism.

## Predictions this makes

1. **Direct context ablation.** Compare held-out late-direction denoising MSE
   with correct, batch-shuffled, and mean-ablated early context. Repeat for
   eigenbands and literal token prefixes.
2. **Matched-budget pixel baseline.** Establish whether there is a
   representation tax under the current architecture and budget.
3. **Unordered control.** Establish whether nested-prefix training helps or
   hurts full-length modelability. Its energy-dependence sign is exploratory,
   not a predeclared success criterion.

Do not train an objective against this proxy unless direct conditioning utility
is first established and shown to relate to the decision metric.
