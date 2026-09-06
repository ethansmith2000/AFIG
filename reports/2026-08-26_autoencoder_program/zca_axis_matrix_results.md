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

## 10k outcome

| transform | common / uniform | ordered / uniform | common / weighted | ordered / weighted |
|---|---:|---:|---:|---:|
| channel ZCA | 26.441 / .01970 | **26.379** / .01986 | 26.738 / **.01966** | 26.395 / .01970 |
| sequence ZCA | **26.448** / .02080 | 26.877 / .02054 | 27.170 / .02102 | 27.389 / **.02053** |
| axial ZCA | **25.044 / .01902** | 26.848 / .02181 | stopped at 5k | 26.533 / .02082 |
| flattened ZCA | stopped at 5k | 32.797 / .02643 | 33.261 / .02626 | **31.849 / .02607** |

Each entry is `FID-10k / KID-10k`; stopped cells retain their complete 5k
measurement above. Native v27 is `24.534 / .01765`.

Channel and sequence common/uniform converge to almost identical FID, but both
lose approximately `1.91` to native and `1.40` to axial. Channel timing changes
FID by only `-.062` while worsening KID; its joint intervention changes FID by
only `-.046` with essentially identical KID. For sequence ZCA, timing worsens
FID by `.428` while slightly improving KID, and the joint intervention worsens
FID by `.940`. These are metric tradeoffs and sampling-scale fluctuations, not
useful hierarchy gains.

The three continuing flattened arms remain at FID `31.85–33.26` and KID near
`.026`, confirming that unrestricted token-feature mixing is harmful even
after rotating back. The 5k exact-cache factorial remains the appropriate
within-flattened objective comparison because its rejected common/uniform cell
correctly stopped under the frozen rule.

## Final decision

The complete experiment separates two conclusions:

1. Rotate-back matters: ZCA avoids the catastrophic failure caused by exposing
   PCA modes as literal tokens.
2. Native geometry still wins: no whitening axis and no token SNR/loss
   intervention improves both decoded FID and KID.

Retain native v27 with common time and uniform loss. Do not replicate post-hoc
ZCA or the manually imposed native-token hierarchy. A future hierarchy should
be learned jointly with the tokenizer/prior or expressed through a dedicated
readout whose semantics are trained, rather than assigned to native token
indices after fitting covariance.

Exact final values are in
[`zca_axis_matrix_final_results.json`](zca_axis_matrix_final_results.json).
