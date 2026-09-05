# Rotate-back/ZCA whitening audit

## Decision

The corrected transform is numerically healthy and tests a materially different
object from the rejected PCA-coordinate screen. Proceed with the predeclared
axial gamma-1 cache and the complete common/ordered-time by uniform/tempered-loss
factorial.

Gamma zero is exactly the native representation for every variant. Full axial
ZCA increases held-out flattened effective rank from `369.88` to `725.34` while
rotating back into the native `64x16` axes. Its inverse survives float16 cache
storage at `2.25e-4` relative latent RMS and changes decoded pixels by only
`.001206` RMS. These are representation diagnostics and numerical gates, not
evidence that the transformed latent will generate better.

## Gamma-1 geometry

| transform | effective rank | covariance off-diagonal fraction | gain range | native correlation | relative displacement | mean matching-token map energy |
|---|---:|---:|---:|---:|---:|---:|
| channel | 610.37 | .8283 | 3.21x | .9349 | .3605 | 1.0000 |
| sequence | 627.27 | .7567 | 215.83x | .9262 | .3838 | .9441 |
| axial | 725.34 | .6646 | 693.58x | .8428 | .5879 | .9441 |
| flattened | 943.63 | .3772 | 1,866.16x | .8051 | .6315 | .8773 |

Channel ZCA never mixes tokens. Sequence and axial ZCA retain an average
`94.41%` of each output token's linear-map energy on its matching native input
token; the median is `96.09%`, with a minimum of `51.06%` for the most mixed
slot. Flattened ZCA is the strongest covariance correction but is analysis-only:
it mixes arbitrary token-feature coordinates and retains less native-token
self-attribution.

The axial transform nearly saturates the separate channel and sequence ranks
(`15.41/16` and `62.86/64`) but does not fully whiten the complete covariance.
That is expected: a Kronecker transform removes the fitted marginal axial
spectra, while non-separable token-channel interactions remain. The literal
token-power ratio is still `1.70x`, so the audit does not claim that axial ZCA
makes every token identical.

## Numerical gates

| gate | limit | worst observation | result |
|---|---:|---:|---|
| gamma-0 identity, maximum absolute error | 1e-6 | 0 | pass |
| float32 inverse, relative latent RMS | 1e-5 | 9.66e-6 | pass |
| simulated float16-cache inverse, relative latent RMS | .002 | 2.25e-4 | pass |
| decoded pixel delta RMS | .002 | .001208 | pass |

The first queued audit attempt stopped before analysis because `torch.kron`
received a non-contiguous QR basis. Commit `05f1bfb` made those eigenspaces
contiguous; the queue-managed rerun completed on GPU 5 at
`2026-09-05T21:44:53Z`.

## Authorized generative test

Build one float16 axial gamma-1 cache with an exact serialized inverse into the
native decoder gauge. Train four 60k prior-seed-1 arms:

1. common time / uniform loss;
2. beta-.25 softened token time / uniform loss;
3. common time / beta-.25 tempered token loss;
4. softened token time / tempered token loss.

The time crossings and loss weights remain exactly those frozen in the protocol.
There is no explicit clean-token magnitude rescaling. Paired decoded FID/KID-5k,
not covariance rank, reconstruction, or flow loss, decides the outcome.

Artifacts: [protocol](../zca_whitening_protocol.json),
[`metrics.json`](metrics.json), [`zca_geometry.pt`](zca_geometry.pt), and
[`zca_rank_and_token_attribution.png`](zca_rank_and_token_attribution.png).
