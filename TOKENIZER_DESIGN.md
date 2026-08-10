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
