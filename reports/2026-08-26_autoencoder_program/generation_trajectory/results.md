# Direct generation-trajectory result

Date: 2026-09-04
Status: complete; descriptive analysis only, with one selected-prior context
intervention. This result does not reopen the completed v27 model selection.

## Answer

The selected v27 prior already generates in a strong coarse-to-fine and
high-variance-to-low-variance order, but that order is distributed across the
64 latent tokens rather than stored in token index. The first decoded structure
is global color and low spatial frequency; leading latent population PCs settle
next, followed by progressively higher image frequencies and lower-variance
latent directions. Recognizable semantic features stabilize later still.

The ordering is also useful context, not merely a visualization artifact. In a
held-out intervention on v27, retaining the correct already-resolved leading-PC
coefficients instead of shuffling them between examples reduces the denoiser's
MSE on unresolved directions by 9.24% at `t=.35` and 16.80% at `t=.50`.

By contrast, the softened tokenwise schedule makes native register indices
settle in an imposed order, but does not make image-frequency bands emerge
earlier and delays the Inception-feature threshold from `.7` to `.8`. Combined
with its worse FID, this says that an arbitrary slot clock is not the helpful
conditioning hierarchy.

## Method

- Generate 128 examples from v27, v34 common time, and v34 soft25 with the
  actual 50-step Heun sampler and the same standardized Gaussian RNG stream.
- At `t=0,.1,...,1`, retain both the live noisy state and the model's predicted
  clean endpoint, `z_hat = z_t + (1-path_time) * v_theta`; the path coordinate
  is applied per token for rational tokenwise time.
- Decode fixed examples and compare each estimate with its own final generated
  endpoint. Measure pixels, RGB means, CIFAR-population-centered Inception
  features, complex FFT bands, native latent tokens, and training-population
  PCA bands.
- Define frequency settling as the first persistent complex correlation of
  at least `.90`; define PCA/token settling as the first persistent endpoint-
  relative MSE of at most `.25`; define semantic settling as the first
  persistent centered Inception correlation of at least `.90`.
- Separately evaluate the v27 denoiser on 2,048 held-out real latents. Preserve,
  shuffle between examples, or mean-ablate already-resolved PCA coefficients,
  then score velocity MSE only in unresolved directions.

The predicted-clean view is the primary visualization. Decoding the live noisy
state is intentionally off-manifold and remains visually noisy much longer.

## Emergence order

| Representation component | v27 | v34 common | v34 soft25 |
|---|---:|---:|---:|
| FFT radius 0-2 | .1 | .1 | .1 |
| FFT radius 3-4 | .3 | .3 | .3 |
| FFT radius 5-6 | .3 | .3 | .3 |
| FFT radius 7-8 | .3 | .3 | .3 |
| FFT radius 9-12 | .4 | .4 | .4 |
| FFT radius 13-16 | .5 | .5 | .5 |
| FFT radius 17+ | .6 | .7 | .7 |
| PCA ranks 1-8 | .2 | .2 | .1 |
| PCA ranks 9-32 | .2 | .2 | .2 |
| PCA ranks 33-128 | .3 | .3 | .3 |
| PCA ranks 129-512 | .4 | .4 | .4 |
| PCA ranks 513-1024 | .5 | .5 | .5 |
| Centered Inception features | .7 | .7 | .8 |

V27's population spectrum is concentrated enough to supply a natural order:
the top 8, 32, and 128 PCs explain 21.44%, 36.74%, and 58.55% of latent
variance. At `t=.1`, the v27 top-8-PC estimate already correlates `.950` with
its endpoint and the lowest FFT band correlates `.902`; the latent tail and
high-frequency image bands remain unresolved. The fixed examples visibly move
from color/layout to recognizable object and then texture/detail.

Native token settling does not carry this hierarchy:

| Run | Token settling counts | token index vs. settling time |
|---|---|---:|
| v27 | all 64 at `.3` | undefined: all tied |
| v34 common | 6 at `.2`, 58 at `.3` | `-.061` |
| v34 soft25 | 1 at `.2`, 48 at `.3`, 15 at `.4` | `.574` |

Soft25 therefore succeeds mechanically at separating assigned slot clocks, but
the separation is not aligned with faster decoded-frequency or semantic
resolution.

## Does early structure condition what comes later?

For the PCA intervention, "early" means population directions with power
SNR >= 4 and "late" means unresolved directions with power SNR <= .25. Both
sets are selected without using the intervention outcomes.

| Time | resolved PCs used | unresolved PCs scored | gain vs shuffled context | gain vs mean ablation |
|---:|---:|---:|---:|---:|
| .20 | 0 | -- | skipped | skipped |
| .35 | 6 | 256 | 9.24% | 9.02% |
| .50 | 31 | 256 | 16.80% | 21.20% |
| .65 | 183 | 16 | -3.07% | 51.87% |
| .80 | 256 of 738 | 16 | -2.47% | 56.38% |

The clean model-use read is the middle of the trajectory: true sample-specific
leading-PC context improves prediction of a large unresolved subspace. At
`.65/.80`, only 16 near-null tail coordinates meet the late criterion;
shuffling is slightly favorable there, so those rows do not support a claim of
sample-specific late-tail conditioning. The large ablation effect only says
that removing context entirely is disruptive.

As a control, correct native prefixes of 8/16/32 arbitrary v27 tokens reduce
remaining-token MSE versus shuffled prefixes by 8.78/17.87/34.95% at `.35` and
10.75/21.04/40.84% at `.50`. Tokens clearly exchange global information, but
because v27 tokens settle together and the prefixes are arbitrary, this is
mutual distributed context rather than a native token sequence.

## Design implication

Keep v27's flat `64x16` prior-facing representation and common-time path. If an
explicit hierarchy is revisited, align it to conditional innovation in a
learned subspace or multiscale decoded readout: global/leading modes first,
residual/detail modes later. A low-risk experiment is an auxiliary ordered
readout or basis-aware head attached to the flat latent, without forcing the
main 64 tokens into causal or magnitude-coded roles. Such a mechanism should
first demonstrate context gain and preserved effective rank before receiving a
matched-prior generation campaign.

## Artifacts

- [Predicted-clean contact sheet](predicted_clean_contact_sheet.png)
- [Live noisy-state contact sheet](noisy_state_contact_sheet.png)
- [Frequency emergence](frequency_emergence.png)
- [PCA emergence](pca_emergence.png)
- [Token emergence](token_emergence.png)
- [Trajectory summary](trajectory_summary.png)
- [Full trajectory metrics](metrics.json)
- [V27 context-ablation metrics](v27_context_ablation.json)
- [Frozen protocol](../generation_trajectory_protocol.json)
