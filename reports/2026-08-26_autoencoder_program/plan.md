# Autoencoder representation program (2026-08-26)

## Current conclusion

The best learned full-generation point is the unordered `64x16` tokenizer:
moderate dimensional compression removes nuisance variation while retaining an
acceptable reconstruction floor. Increasing width improves reconstruction and
flat flow MSE but not decoded FID; exact reshaping proves that the tokenizer's
native register boundary also matters.

The next program should optimize a three-way objective rather than reconstruction
alone:

1. **Distortion:** clean reconstruction FID/PSNR.
2. **Modelability:** matched-prior decoded FID at a fixed compute budget.
3. **Robustness:** decoder sensitivity to plausible off-manifold latent error.

Effective rank and slot utilization are diagnostics, not selection metrics.

## Figure contract — prefix decoding

- **Question:** how does decoding evolve as the first `k` latent tokens become
  available, and what behavior was purchased by nested-prefix training?
- **Evidence:** the first six fixed preview examples from the v5 progressive and
  v8 unordered checkpoints; columns are reference and `k=1,2,4,8,16,32,64`.
- **Takeaway:** progressive training provides coherent coarse-to-fine decoding;
  the unordered model purchases better full-generation FID at the cost of useful
  partial decoding. This is a product tradeoff, not evidence that prefix ordering
  improves full-length generative modelability.
- **Artifact:** `prefix_decode_comparison.png`, reproducibly built by
  `scripts/build_prefix_decode_comparison.py`.

## E5 — fixed-cache PCA rate control

Start from the completed unordered `64x48` cache and fit PCA on a fixed training
subset. Inverse-project retained coefficients before the unchanged decoder.

1. Oracle-only ranks: `128, 256, 512, 768, 1024, 1536, 2048, 3072`.
2. Record reconstruction FID/PSNR and retained variance for every rank.
3. Select at most two ranks that bracket the `64x16` clean rFID of 6.08.
4. Only those ranks receive matched 60k priors.

This varies effective rate inside one trained representation. It therefore
separates the causal effect of spectral truncation from independent tokenizer
training and tests whether concentrated high-rate codes recover modelability.

Status: completed at `2026-08-26T01:31:45Z` through the shared GPU queue. The
full-rank validation reproduced the existing clean result (rFID 3.040 versus
3.040 previously; PSNR 45.25 versus 45.30), so the truncation curve is valid.

- Launcher: `scripts/run_e5_pca_oracle.sh`.
- Evaluator: `scripts/evaluate_pca_truncation_oracle.py`.
- Output: `pca_oracle_v9_n64d48/metrics.json` and `reconstructions.png`.
- Basis: `tokenizer_runs/v9-unordered-vae-n64d48-s1/pca_basis_25k.pt`.

| retained rank | variance | PSNR | clean rFID |
|---:|---:|---:|---:|
| 128 | 53.42% | 20.53 | 121.93 |
| 256 | 65.90% | 22.52 | 82.09 |
| 512 | 80.68% | 25.36 | 35.91 |
| 768 | 88.98% | 27.60 | 18.78 |
| 1,024 | 93.85% | 29.77 | 11.56 |
| 1,536 | 98.71% | 35.89 | 4.65 |
| 2,048 | 99.99% | 45.24 | 3.04 |
| 3,072 | 100.00% | 45.25 | 3.04 |

The `64x16` nonlinear tokenizer reaches clean rFID 6.08 with 1,024 scalars,
substantially better rate-distortion than top-1,024 PCA on the high-rate code.
The 1,536-PC point is the selected generative control: it preserves a better
oracle floor than `64x16` while removing half the high-rate coordinates. Train
it as 64 tokens by 24 coefficients so token count remains native; inverse PCA
before the unchanged `64x48` decoder.

## Autoencoder exploration sequence

Do not launch a broad architecture/objective grid. Preserve the `64x16` unordered
baseline and change one causal axis per stage.

### Stage A — encoder allocation and latent formation

- Compare the current one-layer cross-attention pool with deeper latent pooling
  and a convolutional/local stem feeding the same 64 latent queries.
- Hold decoder, latent shape, parameter budget, data, steps, and full-only
  objective fixed where possible.
- Measure whether each encoder actually reduces clean rFID without inflating
  effective rank, dead slots, or decoder sensitivity.

### Stage B — posterior/noise parameterization

- Replace the historically collapsed hard-clamped pseudo-VAE with explicit
  deterministic latents plus controlled decoder-input jitter as the clean
  baseline.
- Separately test a soft-floor variational posterior whose variance remains
  trainable. Log the full log-variance distribution and reject arms pinned to a
  boundary.
- Sweep only a small number of noise levels chosen around the observed prior
  error scale; do not infer robustness from clean reconstruction.

### Stage C — representation regularization

- Test controlled spectral concentration or PCA-aligned penalties, but target a
  rate-distortion region rather than blindly minimizing effective rank.
- Treat balanced slot usage and dead-slot penalties separately from spectral
  rank; the `64x32/48` runs show that they are not the same quantity.
- Consider slot dropout only if variable-rate behavior is desired, and report it
  as a product objective because prefix training already showed a full-FID tax.

### Stage D — decoder and perceptual objective

- Change decoder capacity only after encoder-side controls identify a promising
  representation; otherwise encoder and decoder effects are inseparable.
- Compare pixel MSE with a restrained perceptual or frequency-aware term while
  guarding against visually plausible but information-losing reconstructions.
- Always retain clean rFID and latent-noise sensitivity; PSNR alone is not a
  promotion criterion.

## Promotion gates

Every tokenizer gets the same 15k budget and must report clean rFID/PSNR,
decoder sensitivity, effective rank, coordinate spread, slot RMS, and posterior
statistics. Only arms that improve the distortion/robustness frontier receive a
prior screen. Final selection uses decoded FID/KID under the matched prior recipe,
followed by a larger-sample evaluation and another training seed for claimed
improvements.
