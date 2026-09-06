# Axial-ZCA schedule × loss factorial

## 5k result

Rotate-back whitening is dramatically healthier than exposing PCA coordinates,
but it has not improved generation. The axial gamma-1 common/uniform arm reaches
FID/KID `27.246/.01917`, close to but worse than native v27 at
`26.743/.01754`. This is a useful result: a `693.6x` axial eigengain range is
tolerated when the transform returns to native token-feature coordinates.

| axial-ZCA arm | FID-5k | KID-5k | FID delta vs common/uniform |
|---|---:|---:|---:|
| common time / uniform loss | **27.246** | **.01917** | — |
| ordered time / uniform loss | 29.099 | .02144 | +1.853 |
| common time / tempered loss | 30.037 | .02152 | +2.791 |
| ordered time / tempered loss | 28.740 | .02085 | +1.494 |

At uniform loss, the softened token schedule worsens FID and KID. At common
time, the tempered loss allocation also worsens both metrics. The joint arm is
non-additive: timing improves the weighted arm by `1.298` FID, and weighting
improves the ordered arm by `.360`, producing a `-3.151` FID interaction. This
looks like mutual compensation rather than a useful hierarchy because the
combination still loses to common/uniform by `1.494` FID and `.00168` KID.

All sample grids are visually coherent; no arm failed codec inversion or
sampling. Final flow losses are not used for selection.

## Frozen continuation decision

Three paired 10k evaluations are authorized:

- common/uniform is only `.503` FID behind native v27;
- ordered/uniform is `1.853` FID behind its exact-cache common-time control;
- ordered/weighted is within two FID of common/uniform and improves both
  corresponding single-intervention arms.

Common/weighted stops at 5k: it is concordantly worse than common/uniform by
`2.791` FID and `.00235` KID. The 10k comparison will decide whether axial ZCA
is genuinely at parity and whether the apparent interaction persists. No new
training is required.

Exact values are in [`factorial_5k_results.json`](factorial_5k_results.json).

## 10k result and final verdict

| representation / objective | FID-10k | KID-10k | FID delta vs ZCA common/uniform |
|---|---:|---:|---:|
| native v27, common/uniform | **24.534** | **.01765** | -0.510 |
| axial ZCA, common/uniform | 25.044 | .01902 | — |
| axial ZCA, ordered/uniform | 26.848 | .02181 | +1.804 |
| axial ZCA, ordered/weighted | 26.533 | .02082 | +1.489 |

The larger sample confirms every important 5k direction. Axial gamma-1 ZCA is
close to native v27, but loses by `.510` FID and `.00137` KID. Soft token timing
loses to its ZCA common-time control by `1.804` FID and `.00280` KID. Adding the
tempered weights recovers only `.315` FID and `.00100` KID from ordered/uniform;
the combined intervention still loses to ZCA common/uniform by `1.489` FID and
`.00180` KID.

This isolates the cause of the earlier catastrophic result. Leaving the latent
in PCA coordinates destroyed the prior's literal token geometry; symmetric
rotate-back whitening does not. Nevertheless, removing the axial magnitude
spectrum provides no generative advantage, and manually reintroducing an
image-inspired hierarchy at native token indices is harmful in this form.

Retain native v27 with common time and uniform loss. Close post-hoc whitening
and explicit native-token SNR/loss ordering as primary design directions. If
hierarchy is revisited, it should be co-learned with the representation or
attached as an auxiliary subspace/readout rather than imposed on arbitrary
native token indices.

Exact final values are in
[`factorial_final_results.json`](factorial_final_results.json).
