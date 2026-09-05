# Smooth power-whitening generative screen

Date: 2026-09-05

Status: all four 60k-step priors and their FID/KID-5k evaluations completed.
Gamma `.25` is the sole arm admitted to the predeclared 10k confirmation.

## Result

| representation | gamma | FID-5k | KID-5k | decoded clip fraction |
|---|---:|---:|---:|---:|
| native v27 control | — | **26.74** | **.01754** | — |
| factorized power | 0 | 94.69 | .07557 | .0157 |
| factorized power | .25 | **72.99** | **.05852** | .0119 |
| factorized power | .5 | 94.32 | .07506 | .0139 |
| factorized power | 1 | 77.70 | .06457 | .0088 |
| factorized cap-16 reference | — | 88.54 | .07337 | — |

Gamma `.25` wins both generative metrics. It improves FID by `21.70` over
gamma 0 and by `15.54` over the previous cap-16 common/uniform arm, but remains
`46.25` worse than the native latent. Gamma 1 is second among the power arms at
FID `77.70`; gamma `.5` is essentially tied with the rotation-only failure.

## Interpretation

The decisive observation is gamma 0. It applies an orthonormal factorized
sequence/channel rotation without changing the covariance eigenvalue spectrum,
yet FID degrades from `26.74` to `94.69`. Therefore the poor result cannot be
attributed primarily to weak-direction amplification, inversion error, or
float16 precision. Moving global sequence eigenmodes into literal token
positions destroys useful alignment between the learned representation and the
prior's token-wise/shared-weight inductive bias.

Some spectrum compression is useful *inside that already difficult coordinate
system*: gamma `.25` substantially rescues gamma 0. It is nevertheless not
competitive with leaving the native coordinates intact. This argues against
factorized post-hoc rotation/whitening as the main path.

Training was numerically healthy. Every arm reached 60,000 steps at about 21
steps/s; final evaluation flow MSE rises monotonically from `1.120` at gamma 0
to `1.294` at gamma 1. Generative FID does not follow that ordering—gamma 1
beats gamma `.5` by `16.61` FID despite worse flow MSE—again validating the
decision not to select representation changes by reconstruction or training
loss alone.

## Selection

Only gamma `.25` advances to 10k: it is best on both FID and KID, no other arm
is within two FID, and the two metrics do not disagree. The 10k result estimates
the size of the failure more precisely; it cannot overturn the screen-level
rejection relative to native v27.

Exact machine-readable values are in
[power_whitening_screen_results.json](../power_whitening_screen_results.json).
