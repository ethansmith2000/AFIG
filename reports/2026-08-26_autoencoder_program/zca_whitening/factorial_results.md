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
