# Progressive tokenizer design

## Representation

The canonical bottleneck contains 32 ordered whole-image vectors of width 64:

```text
z = E(x),       z.shape = [batch, 32, 64]
x_hat = D(z)
```

This is 2,048 continuous coordinates for a 3,072-scalar CIFAR image. The initial
autoencoder is deterministic in both directions. The encoder ends in an
affine-free LayerNorm followed by an unconstrained learned linear projection;
there is no normalization after that projection.

## Architecture

The encoder patchifies the image into an 8-by-8 grid, applies eight width-512
bidirectional Transformer blocks, and lets 32 unique learned Perceiver queries
pool the resulting features. Two pooling blocks alternate query-to-image cross
attention, query self-attention, and feed-forward computation.

The autoencoder decoder begins with 64 content-free spatial queries, represented
by a shared learned output token plus a unique 2-D output position. Eight
width-512 decoder blocks alternate spatial self-attention, cross-attention to the
continuous latent sequence, and feed-forward computation. A final normalization
and patch projection reconstruct the image.

This decoder is not the future autoregressive model or its stochastic token
head. It is only the deterministic decoding half of the autoencoder.

## Gate A: complete-sequence reconstruction

The first objective is ordinary pixel MSE in the `[-1, 1]` training space:

```text
L_full = MSE(x, D(E(x))).
```

Its purpose is to establish the reconstruction ceiling and inspect the emergent
latent distribution without imposing an ordering regularizer prematurely.

The initial optimizer uses a 1,000-step linear warmup to `2e-4`. The first run
showed a clear late-step instability after its best 5k checkpoint, so the
canonical continuation restores that checkpoint and fine-tunes at `5e-5`. The
held-out curve reached a second plateau around 9--10k, after which the run steps
down again to `2e-5` for its reconstruction-ceiling phase.

## Gate B: progressive prefixes

For an ordered prefix of length `K`, the same decoder masks every latent after
`z_K`:

```text
x_hat_K = D(z_1, ..., z_K)
L = L_full + E_K[MSE(x, x_hat_K)],   K uniform in {1, ..., 31}.
```

Every training example always receives the complete reconstruction loss and one
random shorter-prefix loss. Token order is fixed; no token permutation is used.
The expectation is a compute-efficient estimate of training every possible
prefix.

Gate B trains the same architecture jointly from scratch. A 100-step CIFAR
pilot initialized from Gate A showed severe interference: while prefix quality
began improving, complete-sequence PSNR fell from 41.18 to 29.69 dB even though
the warmup had reached only `5e-6`. The ordered objective must therefore shape
the encoder and decoder together instead of attempting to rewrite a converged,
distributed code.

The completed 30k Gate-B checkpoint reaches full-test prefix PSNR
`20.54 / 22.61 / 25.14 / 28.84 / 31.95 / 35.73` dB at
`K = 1 / 2 / 4 / 8 / 16 / 32`. This is a smooth successive-refinement code,
with a 5.45 dB complete-reconstruction cost relative to the unordered Gate-A
ceiling.

## Gate A result

The selected 25k checkpoint reconstructs the complete CIFAR-10 test set at
41.18 dB PSNR (pixel MSE `7.63e-5`). It outperforms the old 53-by-64 ring codec's
38.27 dB while using 2,048 instead of 3,392 latent scalars. A chronological 30k
checkpoint reached 40.75 dB and is retained separately.

The unconstrained clean latent has global mean `0.071`, standard deviation
`0.935`, covariance effective rank `21.7 / 64`, and coordinate standard
deviations spanning `0.503--1.330`. Slot RMS lies in `0.884--1.056`. Thus the
classic final-normalization-plus-projection interface naturally learned a
well-scaled code without a hard spherical constraint or injected noise.

As expected, the complete-sequence objective did not spontaneously order the
registers: prefix PSNR is 10.62, 10.68, 11.60, 12.79, 17.13, and 41.18 dB at
lengths 1, 2, 4, 8, 16, and 32. Gate B directly tests whether asymmetric prefix
reconstruction can turn the same high-quality code into successive refinement.

## Deferred constraints

Latent AWGN, explicit power constraints, semantic teacher losses, and soft
continuous prefix gates are ablations rather than v1 defaults. They should be
introduced only in response to measured failure modes such as fragile latent
encoding, one-token information concentration, tail starvation, or poor prior
modelability.

## Measurements

Final evaluation reports reconstruction at prefix lengths 1, 2, 4, 8, 16, and
32, together with:

- normalized and physical-pixel MSE;
- PSNR;
- coordinate standard-deviation range;
- latent covariance effective rank;
- mean per-token peak-to-RMS ratio;
- RMS by ordered latent slot.

If the representation passes, fixed-total-coordinate comparisons of `16 x 128`,
`32 x 64`, and `64 x 32` separate token-sequence depth from per-token width.

## Gate C: joint generative positive control

Generative modeling begins with all 32 tokens noised and denoised jointly. This
removes autoregressive exposure bias and the capacity of a small per-token head
as confounds. The frozen 12.5k Gate-B encoder produces a fixed CIFAR latent
cache. Its train split has tensor-wide mean `0.0645`, standard deviation
`0.7649`, and range `[-4.19, 4.47]`; slot standard deviations span only
`0.672--0.833`. The baseline therefore uses one population mean and scale for
the complete tensor, not coordinate or slot whitening.

The model is a 12-layer, width-512 bidirectional DiT-style rectified flow over
the `32 x 64` tensor. It combines learned absolute slot embeddings with fp32
1-D RoPE in attention, uses QKNorm and canonical AdaLN-Zero, and predicts the
velocity of the straight path

```text
z_t = (1 - t) epsilon + t z,
v_target = z - epsilon.
```

Training samples `t` uniformly and applies ordinary unweighted MSE. Sampling
uses 50-step Heun integration. EMA, CFG, per-coordinate whitening, timestep
weighting, and class conditioning are absent from the first control.

The joint control improved monotonically but slowly: 5k/10k/15k/20k FID was
`127.64 / 94.27 / 80.53 / 75.30`, with corresponding KID
`0.1288 / 0.0882 / 0.0719 / 0.0654`. At 20k the decoded samples contain
recognizable animals, vehicles, and outdoor layouts, establishing that the
continuous latent is generatively viable. Its quality remains substantially
behind the earlier pixel and patch-local controls, so modelability is not yet a
solved property of the tokenizer.

## Gate D: ordered autoregressive prior

The causal baseline factorizes the learned sequence in its intended order. The
trunk receives the exact shifted sequence

```text
input:  [BOS, z_1, ..., z_31]
target: [z_1, z_2, ..., z_32].
```

Learned target-position embeddings identify which register each residual state
must predict, while 1-D RoPE and QKNorm shape causal attention. No separate
source-token metadata is added: the fixed one-step shift and target identity are
sufficient. Unit tests perturb each target and verify that it cannot influence
its own condition.

The output distribution is a shared six-block conditional rectified-flow MLP.
It concatenates the width-512 trunk state with the timestep embedding, passes
them through a two-layer fusion MLP, and supplies the resulting condition to
canonical AdaLN-Zero blocks. Training noises all 32 teacher-forced targets in
parallel; inference generates one complete 64-D register at a time with 50-step
Heun integration. The normalization, optimizer, flow path, lack of EMA, and
flat loss weighting match Gate C.

The initial comparison used the tokenizer frozen at 12.5k so modeling could
begin before the codec run completed. AR 5k/10k/15k/20k FID was
`139.44 / 123.08 / 113.21 / 105.90`, with KID
`0.1390 / 0.1205 / 0.1092 / 0.1013`. At every matched checkpoint this is worse
than joint flow, even though AR teacher-forced validation MSE is much lower.
For example, at 10k AR validation MSE is `0.289` versus joint flow's `0.424`,
while FID is `123.08` versus `94.27`. This is direct evidence of exposure bias.

The AR cache contains one fixed orientation for each of 50k training images.
By 20k, training MSE fell to `0.215` while held-out teacher-forced MSE worsened
from `0.287` at 15k to `0.319`. Thus the late AR run also overfits. A canonical
follow-up should encode both original and horizontally flipped images from the
completed tokenizer before testing an exposure-aware objective. The present
priors remain valid controls, but should not be mistaken for models of the
final 30k latent representation.
