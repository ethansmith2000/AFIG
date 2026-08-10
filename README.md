# Progressive Continuous Image Tokens

This branch studies deterministic whole-image tokenization followed by
probabilistic continuous-token generation. The direct Fourier-generation era is
preserved in Git history through commit `336eea7`; explicit Fourier coefficients
are no longer the primary generative representation.

The first gate is deliberately conventional:

```text
CIFAR image
  -> 64 spatial patch features
  -> 32 learned Perceiver registers
  -> 32 x 64 continuous latent sequence
  -> spatial-query Transformer decoder
  -> reconstructed image
```

Encoding and decoding are deterministic. There is initially no KL term,
quantization, posterior sampling, latent noise, or hard latent normalization.

## Canonical configuration

- CIFAR-10 at `32 x 32`.
- `4 x 4` patches: 64 encoder patch tokens.
- Encoder and decoder width 512, 8 layers, 8 heads.
- QKNorm and float32 2-D RoPE tables for spatial self-attention.
- Two Perceiver pooling blocks and 32 unique learned pooling queries.
- Final affine-free LayerNorm plus projection to 64 dimensions per register.
- 64 spatial decoder queries cross-attend to the latent sequence.
- AdamW `lr=2e-4`, betas `(0.9, 0.995)`, matrix-only weight decay `0.05`.

See [`TOKENIZER_DESIGN.md`](TOKENIZER_DESIGN.md) for the experiment sequence and
objective definitions.

## Training

Use the shared lifetime GPU claim launcher on this machine:

```bash
cd /workspace/AFIG
gpu-claim run --owner AFIG --job tokenizer-n32-d64-full-s1 --wait -- \
  python -u train_progressive_tokenizer.py \
    --output_dir tokenizer_runs/n32-d64-full-s1 \
    --objective full
```

The trainer overwrites one resumable `checkpoint_latest.pt` and removes it after
writing the final model-only checkpoint. It records reconstruction panels,
streaming metrics, final prefix curves, and latent covariance diagnostics.

CPU-sized correctness smoke:

```bash
CUDA_VISIBLE_DEVICES='' python train_progressive_tokenizer.py \
  --smoke --output_dir /tmp/progressive-tokenizer-smoke
python -m pytest -q tests/test_progressive_tokenizer.py
```

## Planned modeling gate

Once the autoencoder passes reconstruction:

1. Freeze it and extract clean `32 x 64` latent sequences.
2. Train a joint noncausal continuous model over the complete tensor.
3. Require semantic decoded samples from that positive control.
4. Train an autoregressive prior
   `p(z_1, ..., z_32) = product_i p(z_i | z_<i)`.
5. Keep the token distribution head modular between rectified flow and a
   conditional normalizing flow.

The joint positive control uses the frozen 12.5k progressive-tokenizer
checkpoint. CIFAR is encoded once into a fixed latent cache, then standardized
with one tensor-wide population mean and standard deviation. The prior is a
12-block, width-512 bidirectional Transformer rectified flow with learned
absolute slot identity, fp32 1-D RoPE, QKNorm, and canonical AdaLN-Zero. It uses
uniform flow times, an unweighted velocity loss, Heun sampling, and no EMA.

```bash
python cache_progressive_latents.py \
  --tokenizer_checkpoint tokenizer_runs/n32-d64-prefix-s1/checkpoint_012500.pt \
  --output tokenizer_runs/n32-d64-prefix-s1/latents_012500.pt

scripts/run_progressive_joint_flow.sh
```
