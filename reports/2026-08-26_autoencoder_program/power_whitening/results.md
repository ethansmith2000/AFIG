# Smooth power-whitening audit

Date: 2026-09-05

Status: complete. All four predeclared training exponents pass the numerical
gate; this analysis authorizes the common-time/uniform-loss gamma screen.

## Answer

Unrestricted factorized whitening is numerically healthy. Its `892.6x`
relative forward gain is large by design, but neither source-float16
quantization nor inversion error indicates that it is merely elevating stored
rounding noise. Smooth power interpolation produces the intended monotone
covariance-rank curve and should be tested generatively rather than rejected by
gain magnitude alone.

This is factorized—not complete flattened—whitening. Gamma 1 equalizes the
1,024 product-coordinate marginal variances, while residual nonseparable
correlations remain.

## Gamma sweep

For original product-coordinate power `p_j`, use
`gain_j proportional to p_j^(-gamma/2)` and one global scale. The transformed
training spectrum is proportional to `p_j^(1-gamma)`.

| gamma | relative gain range | held-out effective rank | held-out off-diagonal covariance fraction | numerical gate |
|---:|---:|---:|---:|---|
| 0 | 1.00x | 369.88 | .4056 | pass |
| .125 | 2.34x | 472.86 | .4712 | pass |
| .25 | 5.47x | 565.94 | .5314 | pass |
| .5 | 29.88x | 700.63 | .5991 | pass |
| .75 | 163.31x | 770.96 | .6048 | pass |
| 1 | 892.63x | 795.95 | .5982 | pass |

Gamma 0 is not the native latent: it is the sequence/channel rotation-only
control. Its covariance eigenvalues are unchanged, but it isolates whether
placing ordered sequence eigenmodes at literal token positions is itself hard
for the prior. Gamma 1 has held-out coordinate-variance minimum/median/maximum
`.941/1.001/1.068`, confirming essentially complete diagonal whitening.

The increasing off-diagonal fraction is not a contradiction. Equalizing weak
coordinate variances increases the relative contribution of correlations that
the factorized basis cannot diagonalize. The earlier full covariance audit
already measured this nonseparable residual.

## Is the weak tail stable signal?

In the frozen product basis, two disjoint halves of the training set have
coordinate-power Spearman `.99924` and token-power Spearman `.99973`. The
median absolute half-to-half log-power difference is `.0155` per coordinate
and `.0090` per token; even the respective maxima are only `.0794` and `.0229`.

Sequence eigenspaces are also stable. Mean squared principal overlap is `.9998`
for ranks 1-2, stays between `.9828` and `.9968` through ranks 3-63, and is
`1.0000` for the final near-null mode. The final mode is therefore a stable
structural constraint, not a randomly rotating sample-covariance artifact.

## Float16 and decoder safety

A conservative proxy propagates each stored float16 value's local rounding
step variance through the factorized basis. No coordinate has estimated
signal-to-quantization-noise ratio below 1,000; the weakest coordinate is
`1,650` and the weakest 16-feature token is `4,392`. Median coordinate SNR is
`13.1 million`. This does not prove that every weak direction is useful, but it
strongly rejects the simple claim that full whitening only promotes float16
rounding noise.

Across every gamma, float32 relative latent round-trip error is about `7.1e-7`,
simulated transformed-float16 error about `2.07e-4`, and decoded pixel delta
RMS about `.00121`. Full whitening is no worse numerically than gamma 0.

## Authorized screen

Build four exact caches and train only common-time, uniform-loss priors for
gamma `0/.25/.5/1`. This isolates:

1. ordered factorized rotation without whitening;
2. mild log-spectrum compression;
3. half whitening;
4. intentional complete factorized whitening.

The cap-16 common/uniform result at FID `88.54` remains an additional nonlinear
reference. No beta timing or loss weighting enters this screen. FID/KID—not
effective rank or training loss—selects whether any gamma deserves a later
ordered-time comparison.

## Artifacts

- [Exact metrics](metrics.json)
- [Gamma covariance spectra](gamma_covariance_spectra.png)
- [Rank and gain curve](gamma_rank_and_gain.png)
- [Frozen protocol](../power_whitening_protocol.json)
