# V27 latent-axis geometry audit

Date: 2026-09-04

Status: complete. This phase trained no models and does not change the selected
v27 tokenizer or common-time prior.

## Answer

V27 contains a strong and highly sample-consistent magnitude hierarchy, but it
does not live in native token power. It lives in distributed sequence and
flattened covariance modes. Those modes also have a clear decoded role:
leading bands predominantly change global color and low spatial frequencies,
while progressively weaker bands move toward medium and high-frequency image
structure. When real held-out latents are noised, their analytic magnitude-
derived SNR crossings predict the trained prior's empirical recovery order.

Native tokens are the opposite. Slot balancing makes their population powers
nearly identical, their samplewise power ranking is at chance, their known-
clean recovery curves overlap, and every power-ranked token group changes a
broad mixture of decoded frequencies. Scheduling current token indices
therefore imposes a clock on slots that do not already possess the corresponding
information roles.

## Spectrum shape and sample consistency

All covariances are computed after the prior checkpoint's tensor-wide
standardization and removal of an elementwise training-population mean. Bases
use 25,000 training latents; all consistency statistics use the disjoint 10,000
test latents.

| View | Effective rank | Stable rank | strongest / median | strongest / weakest | median per-sample mode-rank correlation |
|---|---:|---:|---:|---:|---:|
| channel, 16 modes | 12.16 | 4.55 | 5.98x | 10.33x | .826 |
| sequence, 64 modes | 45.35 | 7.45 | 12.54x | 46,581x | .649 |
| flattened, 1,024 modes | 376.55 | 16.71 | 137.54x | 3.48 million x | .442 |
| native token power, 64 groups | 64.00 | 62.32 | 1.026x | 1.053x | .024 |

The extreme sequence and flattened condition numbers come from tiny terminal
directions, not a five-component representation. The sequence spectrum has a
broad head and ordinary tail down to mode 63, followed by one nearly-null mode
at `0.000184`. Its first 8/16/32 modes explain 39.0/53.1/73.0% of sequence-axis
power. The flattened spectrum is also broad: its first 1/8/32/128/512 modes
explain 5.99/21.32/36.45/57.90/87.74%. The final 24 flattened directions carry
only 0.088% of total power.

Single-mode ranks fluctuate, as expected. Broad population-rank bands are much
more reliable:

| Adjacent band comparisons | descending probability |
|---|---|
| channel `1`, `2`, `3-4`, `5-8`, `9-16` | 82.95%, 68.82%, 89.95%, 98.68% |
| sequence `1-2`, `3-4`, `5-8`, `9-16`, `17-32`, `33-64` | 87.81%, 72.88%, 85.36%, 88.75%, 95.04% |
| flattened `1-8`, `9-32`, `33-128`, `129-512`, `513-1024` | 97.15%, 98.86%, 99.92%, 100% |
| eight native-token power bands | 49.46-52.52%, approximately chance |

This answers the audio-style consistency question: the individual latent
coefficients vary, but the broad flattened and sequence spectra are reliably
ordered for individual images. Native token power is not.

## Factorized versus full covariance

The best scalar fit of `C_sequence kron C_channel` to the complete flattened
covariance has covariance cosine `.8001`, squared cosine `.6401`, and relative
Frobenius residual `.5999`. Thus the marginal factorization captures about 64%
of squared covariance alignment while leaving about 36% in nonseparable
token-channel interactions.

Factorized channel/sequence whitening remains an attractive architectural
transform, but it is not full whitening. Full whitening is the scientific
control; however, naively whitening the complete covariance would amplify the
weakest direction about `sqrt(3.48e6) = 1,866x` relative to the strongest. It
requires shrinkage, an eigenvalue floor, or a retained-subspace treatment.

## What the bands do in image space

For 256 held-out examples, replace one band with the same band from another
example and decode with the unchanged tokenizer. The intervention preserves the
band marginal but breaks its sample-specific relation to the remaining latent.
Because the hybrid can be off the joint latent manifold, the robust conclusion
comes from the systematic progression across bands rather than any single
changed image.

| Band order | decoded FFT power in radii 0-4 | decoded FFT power in radii 13+ |
|---|---|---|
| sequence, strongest to weakest | 90.7%, 78.5%, 69.6%, 64.5%, 56.1%, 43.7% | 1.9%, 4.3%, 5.9%, 6.3%, 6.7%, 7.9% |
| flattened, strongest to weakest | 92.9%, 84.4%, 65.1%, 30.2%, 22.8% | 1.8%, 3.7%, 4.1%, 5.6%, 28.6% |

Both progressions are perfectly monotone in these coarse summaries. The
leading sequence pair and flattened top-eight also produce large RGB-mean
changes (`.347` and `.362` RMS); the corresponding tails produce only `.017`
and `.008`. Channel modes show the same overall head-to-tail tendency but are
less monotone in the middle. Native token power groups have no comparable
progression: every group changes a broad, mostly low-frequency mixture.

Inception-feature changes remain material across many rank bands, including
middle and tail bands. The result is best described as global/color/coarse to
residual/fine, not as a claim that all semantic information lives in the head.

## Recovery from known noisy real latents

Noise 2,048 held-out latents with one fixed Gaussian realization per example
and ask the selected v27 prior for a clean endpoint at `t=.1,.2,...,.9`. A band
settles at the first measured time from which its relative MSE remains at most
`.25`.

| View and ordered bands | analytic population SNR=1 times | empirical recovery times |
|---|---|---|
| channel `1`; `2`; `3-4`; `5-8`; `9-16` | .348, .403, .445, .513, .600 | .4, .4, .5, .6, .7 |
| sequence `1-2`; `3-4`; `5-8`; `9-16`; `17-32`; `33-64` | .281, .386, .434, .486, .529, .577 | .3, .4, .5, .5, .6, .6 |
| flattened `1-8`; `9-32`; `33-128`; `129-512`; `513-1024` | .161, .283, .398, .529, .669 | .3, .4, .5, .6, .7 |
| native token power bands | .499-.503 | all .5-.6, with no magnitude order |

Spearman correlation between analytic and empirical band order is `1.000` for
flattened, `.971` for sequence, and `.975` for channel. The exact times are not
identical—the denoiser generally needs more signal than the analytic SNR=1
crossing—but magnitude accurately predicts the order. Native token powers are
too uniform to provide such an order.

Decoded recovery against the known clean reconstruction is independently
coarse-to-fine: FFT radius 0-2 exceeds `.90` correlation at `t=.3`, radii 3-4
at `.5`, radii 5-8 at `.6`, radii 9-16 at `.7`, and radius 17+ at `.8`.
Centered Inception correlation exceeds `.90` at `.9`.

## Decision and next analysis

Do not apply another schedule to native v27 token indices. The current tokens
are equal-power distributed mixtures, not an image-frequency-like hierarchy.

The most interpretable candidate is now the sequence eigenbasis: it yields 64
global modes, preserves a `64x16` matrix, has sample-consistent power bands, and
has a monotone decoded coarse-to-fine role. But sequence-plus-channel whitening
leaves substantial nonseparable covariance. Before training, construct and
audit two regularized invertible transforms:

1. sequence-PCA plus channel whitening, retaining sequence rank as token order;
2. regularized flattened PCA whitening, grouping nearby original eigenvalue
   ranks into 64 unit-scale tokens as the full-covariance control.

For each, sweep a predeclared covariance shrinkage or maximum inverse-gain cap,
measure held-out residual covariance, verify numerical inverse decoding, and
derive the dynamic ranges of softened SNR odds and loss weights from the
pre-whitening powers. Only then freeze one whitening transform and launch the
`common/ordered time x uniform/latent-derived loss` prior factorial.

## Artifacts

- [Complete spectra](axis_spectra.png)
- [Analytic SNR=1 curves](axis_snr1_crossings.png)
- [Decoded role frequency map](decoded_role_frequency.png)
- [Decoded role contact sheet](decoded_role_contact_sheet.png)
- [Known-clean axis recovery](known_clean_axis_recovery.png)
- [Known-clean contact sheet](known_clean_contact_sheet.png)
- [Exact geometry metrics](geometry.json)
- [Exact known-clean metrics](known_clean_denoising.json)
- [Frozen protocol](../latent_axis_audit_protocol.json)

