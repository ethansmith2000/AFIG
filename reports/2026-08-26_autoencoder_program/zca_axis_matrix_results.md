# Complete ZCA axis × hierarchy matrix

## 5k gate

All twelve new cells completed normally. No whitening axis or hierarchy
objective beats the native v27 external control.

| transform | common / uniform | ordered / uniform | common / weighted | ordered / weighted |
|---|---:|---:|---:|---:|
| channel ZCA | 28.504 / .01959 | 28.819 / .01978 | 28.699 / .01939 | **28.395 / .01928** |
| sequence ZCA | **28.909 / .02119** | 29.304 / .02093 | 29.409 / .02109 | 29.669 / **.02068** |
| axial ZCA | **27.246 / .01917** | 29.099 / .02144 | 30.037 / .02152 | 28.740 / .02085 |
| flattened ZCA | 34.771 / **.02552** | 35.071 / .02638 | 35.242 / .02609 | **34.289** / .02589 |

Each entry is `FID-5k / KID-5k`. Native v27 is `26.743 / .01754`.

The common/uniform axis ordering is native, axial, channel, sequence, then
flattened. Axial conditioning is closer to native than either marginal axis in
isolation, while complete flattened whitening is decisively destructive even
after rotation back. This indicates that preserving the literal output axes is
necessary but not sufficient: the locality and separability of the linear map
also matter.

The explicit hierarchy has no consistent positive main effect. Channel ZCA's
joint ordered/weighted cell improves its common/uniform control by only `.109`
FID and `.00031` KID. For sequence ZCA, every objective variant worsens FID,
although KID moves slightly in the other direction. For flattened ZCA, the
joint cell recovers `.482` FID but worsens KID. These are small within-cache
effects and none approach native v27.

## Frozen continuation

All four channel and all four sequence cells qualify for 10k because their
within-transform gaps are below two FID or FID/KID disagree. Flattened
common/uniform stops because it is concordantly `8.028` FID behind native and
`7.524` behind axial. Its three objective variants qualify only to resolve
their small within-transform effects. This produces eleven 10k evaluations;
no additional model training is required.

Exact values are in [`zca_axis_matrix_5k_results.json`](zca_axis_matrix_5k_results.json).
