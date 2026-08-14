# AFIG independent method audit request — 2026-08-11

## Requested review posture

Please begin with an independent audit of the method and its motivations. Do not
assume that our favored explanation is correct, and do not begin by proposing a
large architecture sweep. Reconstruct the causal argument in your own words,
identify where evidence supports it and where we have made an inference, and
rank the most plausible failure modes by consequence and confidence.

Please keep three questions separate:

1. Does the deterministic tokenizer preserve the image distribution in a form
   that should be generatively modelable?
2. Can a joint diffusion/flow model learn the complete latent distribution?
3. Given a workable joint model, does the ordered autoregressive factorization
   work, or does exposure/first-token difficulty dominate?

We especially want criticism of the conceptual setup, normalization, objective,
optimizer, evaluation logic, and train/inference alignment. A code bug audit is
welcome after the method audit, but should not replace it. Explicitly distinguish
verified facts, plausible interpretations, and speculation.

## Motivation and project transition

AFIG began as autoregressive generation of Fourier coefficients ordered from
low to high frequency. Direct compact-FFT generation repeatedly lagged pixel and
local-transform controls. We explored representation geometry, normalization,
phase/amplitude factorization, positional identity, and frequency-group latent
autoencoders. The current phase deliberately steps away from literal FFT
coordinates to test a more general premise:

> Can an image be encoded as a short ordered sequence of continuous whole-image
> registers, where every longer prefix increases reconstruction fidelity, and
> can those registers then be generated jointly or autoregressively?

The motivation is analogous to successive refinement: early registers should
carry the largest loss-reducing/global contribution, while later registers add
residual fidelity. This preserves a meaningful causal order without forcing one
token to correspond to one Fourier coefficient. It is also meant to reduce the
514-step sequence and fragile per-coefficient diffusion heads used previously.

This is not yet an ImageNet result. All current results below use unconditional
CIFAR-10 at 32 by 32 pixels.

## Current tokenizer

The selected tokenizer is deterministic and has no KL, quantization, latent
noise, adversarial loss, or perceptual loss.

- Input: RGB CIFAR image mapped to `[-1, 1]`; random horizontal flip in training.
- Patchification: learned 4 by 4 stride-4 convolution, producing 64 width-512
  spatial states.
- Encoder: eight transformer blocks, width 512, eight heads, MLP ratio 4.
- Spatial identity: learned absolute patch embeddings plus float32 2-D RoPE.
- Normalization: non-affine LayerNorm in residual blocks; learned affine RMSNorm
  on Q and K head dimensions; standard scaled-dot-product attention.
- Pool: 16 learned queries perform one cross-attention operation over the 64
  encoded patch states. The selected `cross_only` variant exports the attended
  value result directly; it has no learned-query residual, latent self-attention,
  or pooling FFN.
- Export: final non-affine LayerNorm and a linear projection produce 16 tokens of
  64 continuous coordinates (1,024 scalars total versus 3,072 pixel scalars).
- Decoder: latent input projection plus unique latent-position embeddings; 64
  learned spatial output queries pass through eight transformer decoder blocks
  with spatial self-attention and cross-attention to the available latent prefix;
  a linear patch head reconstructs pixels.
- Size: about 60.0 million parameters for the selected cross-only tokenizer.

The progressive training objective is

```text
L = MSE(D(z_1, ..., z_16), x)
  + MSE(D(z_1, ..., z_K), x),  K uniform in {1, ..., 15} per example.
```

Thus every example has a full-code reconstruction and one randomly sampled
shorter-prefix reconstruction. There are no permutations and no explicit
orthogonality, covariance, power, information, or rate constraints.

The selected cross-only tokenizer's test PSNR at K=1/2/4/8/16 is
`20.37 / 22.39 / 24.84 / 27.95 / 32.62` dB. Its latent global mean/std is
approximately `-0.048 / 0.680`; coordinate standard deviations span
`0.419--1.378`; coordinate covariance effective rank is about `28/64`.

## Latent cache and normalization

The frozen tokenizer encodes each of 50k CIFAR training images in its original
orientation and as a deterministic horizontal flip, giving 100k latent tensors.
The test cache contains the original 10k test images only. Priors use exactly one
population scalar mean and standard deviation over the complete training cache:

```text
y = (z - global_mean) / global_std
```

There is no per-slot, per-coordinate, covariance, or nonlinear whitening. The
motivation was to avoid silently equalizing semantic contributions and to keep
the interface simple. Unlike physical Fourier coordinates, however, these
learned latent coordinates have substantial gauge freedom; the reviewer should
challenge whether preserving their raw relative scales has any principled value.

## Joint rectified-flow positive control

The joint model denoises all 16 by 64 standardized latent coordinates together.

- Architecture: bidirectional 12-block DiT-style transformer, width 512, eight
  heads, MLP ratio 4, about 70.3M parameters.
- Identity: learned absolute token embeddings plus float32 1-D RoPE.
- Blocks: non-affine LayerNorm, affine RMSNorm Q/K, SwiGLU FFN, canonical
  AdaLN-Zero modulation/gates.
- Forward path: `y_t = (1-t) eps + t y`, with `eps ~ N(0,I)` and `t ~ U[0,1]`.
- Target: velocity `v = y - eps`.
- Loss: ordinary unweighted coordinate-mean MSE; no Min-SNR/logit-normal
  weighting, class conditioning, EMA, or CFG.
- Sampling: 50-step Heun from t=0 to t=1.

Results for matched 16 by 64 tokenizers:

| tokenizer | 5k FID / KID | 20k FID / KID | validation flow MSE at 20k |
|---|---:|---:|---:|
| cross-only | 119.94 / 0.11706 | **74.97 / 0.06691** | 0.588 |
| residual pool | 121.19 / 0.11727 | 85.49 / 0.07694 | **0.552** |

The residual tokenizer has stronger intermediate reconstruction prefixes and is
easier under latent flow MSE, yet its decoded FID is 10.5 points worse. This is
why we selected cross-only and why we do not trust latent MSE as the final model
selection metric. The earlier 32 by 64 joint prior reached FID 75.30 at 20k, so
16 by 64 preserves essentially the same generative quality with half the latent
coordinates and half the token count. It does not improve absolute quality.

Legacy controls, implemented before the current tokenizer rewrite, put pixel and
patch-local-DCT generation near FID 31--32. Treat this comparison as directional,
not perfectly matched; one requested control is a current-code pixel baseline.

## Ordered autoregressive rectified flow

The AR model uses exact shifted teacher forcing:

```text
trunk input: [BOS, z_1, ..., z_15]
targets:     [z_1, z_2, ..., z_16]
```

- Trunk: 12 causal transformer blocks, width 512, eight heads, MLP ratio 4,
  learned target-slot embeddings and float32 1-D RoPE.
- Token head: shared six-block width-512 conditional rectified-flow MLP.
- Conditioning: concatenate the trunk state and timestep embedding, process with
  a two-layer fusion MLP, then use the result for AdaLN-Zero-style modulation.
- Training: all teacher-forced targets are noised in parallel, with an
  independent uniform timestep per token. The path, velocity target, flat MSE,
  global normalization, and optimizer match the joint model.
- Inference: generate one complete 64-D token at a time; each token uses 50-step
  Heun, and the causal trunk is recomputed using completed generated history.
- Alignment/leakage tests exist and pass.

At 20k, AR FID/KID is `100.08 / 0.09510`, versus joint
`74.97 / 0.06691`. Teacher-forced validation loss is best around step 14k
(`0.415`) and worsens to `0.430` by 20k while training loss continues falling.
Token 1 remains by far the hardest: its test MSE is about `1.15`; most later
tokens lie around `0.20--0.64`. The current 20k result may therefore be past the
best validation checkpoint; FID has not yet been measured at every earlier AR
checkpoint.

## Optimizer and numerical setup

Tokenizer and both priors currently use:

```text
optimizer:       AdamW
learning rate:   1e-4
warmup:          linear for 1,000 steps, then constant LR (no decay)
betas:           (0.9, 0.995)
epsilon:         1e-8
weight decay:    0.05 on ordinary matrix/kernel weights only
gradient clip:   global norm 1.0
precision:       BF16 autocast; TF32 enabled
compile:         torch.compile(mode="default", fullgraph=True)
```

Biases, learned positions/queries/tokens, normalization parameters, and all
parameters with fewer than two dimensions receive zero weight decay. Ordinary
linear/conv weights—including AdaLN modulation matrices—receive weight decay.
There is no EMA. Tokenizer batch size is 512 for 15k updates; prior batch size is
256 for 20k updates. The prior schedule presents 5.12M cached samples, or about
51 passes over the 100k original-plus-flip cache.

Please audit whether constant LR, the beta choices, clipping, weight-decay
grouping, lack of EMA, zero-initialized AdaLN/output paths, or only 20k prior
updates plausibly limit results. Note that the AR validation curve already
overfits after roughly 14k, whereas the joint FID improved strongly from 5k to
20k and may still be undertrained. Do not apply one training-length conclusion
to both models.

## Candidate explanations we want challenged

These are hypotheses, not conclusions:

1. **Tokenizer distribution ceiling.** Pixel MSE and a deterministic decoder may
   discard or average perceptual detail. We have PSNR but have not yet reported
   FID of real test images reconstructed through the frozen tokenizer.
2. **Off-manifold decoder sensitivity.** The decoder is trained only on exact
   encoder outputs. Small prior errors may land off the learned latent manifold
   and decode disproportionately poorly; no latent noise or decoder robustness
   training is used.
3. **Unconstrained latent geometry.** A high-PSNR deterministic code need not have
   a simple density. One scalar normalization leaves coordinate/slot anisotropy
   and correlations intact. Effective rank over the 64 channel coordinates does
   not characterize the complete 1,024-D distribution.
4. **Joint prior undertraining or objective mismatch.** Joint FID improves by
   roughly 45 points between 5k and 20k. A longer joint run, timestep/solver
   audit, or different parameterization could matter.
5. **First-token and exposure difficulty.** The first 64-D token must model an
   unconditional multimodal global code from BOS. Later tokens are easy under
   clean teacher forcing; generated errors then alter every later condition.
6. **Shared token head.** A single conditional flow MLP models all slot
   distributions. Target identity arrives through the trunk condition, but a
   dedicated first-token model or small joint initial block might be materially
   better.
7. **Progressive objective versus modelability.** Nested reconstruction makes
   prefixes useful but may create nonlinear, heteroskedastic residual tokens
   rather than statistically clean innovations.
8. **Small-data generalization.** Horizontal flips double cached views but not
   semantic diversity. ImageNet-32 is planned, but a larger dataset could hide
   rather than diagnose tokenizer/objective defects if basic ceilings are not
   measured first.

## High-information controls to assess

Please rank these and propose better discriminators if needed:

1. Compute FID/KID between real CIFAR test images and their full tokenizer
   reconstructions. Also compare generated images against the reconstructed-real
   distribution. This separates decoder distortion from prior mismatch.
2. Continue the selected joint 16 by 64 model substantially beyond 20k, evaluating
   fixed checkpoints, before declaring its architecture or flow objective weak.
3. Train a matched current-code joint pixel-space diffusion/flow baseline with
   comparable parameter count, optimizer, augmentation, samples, solver, and FID
   implementation. The older pixel result is not a perfect control.
4. Measure full flattened latent covariance/intrinsic structure and test an
   exactly invertible normalization ablation: per-slot/channel standardization,
   channel covariance whitening, or carefully regularized full whitening. The
   decoder can invert this transform, so reconstruction is unchanged.
5. Measure decoder sensitivity to structured and isotropic perturbations of real
   latents; compare errors with the prior's actual residuals.
6. Evaluate AR checkpoints around 12.5k--15k, where held-out teacher-forced loss
   is best, rather than using only the overfit 20k checkpoint.
7. Compare solver step counts and Euler/Heun convergence to rule out integration
   error before changing the learned distribution.
8. Only after these diagnostics, test exposure-aware generation, joint generation
   of the first few tokens, or ImageNet-32 scaling.

## Requested response format

Please return:

1. A short restatement of what the method is trying to achieve.
2. The strongest and weakest assumptions in the causal argument.
3. A ranked list of likely bottlenecks, each with confidence, supporting evidence,
   counterevidence, and the cheapest decisive test.
4. Any mathematical/objective or optimizer concerns.
5. Any concrete implementation risks, with file and symbol references.
6. The three highest-information next experiments, including what each possible
   outcome would imply.
7. Anything important we appear not to have considered.

## Canonical files and artifacts

Start with:

- `TOKENIZER_DESIGN.md` — chronological design rationale and results.
- `INDEPENDENT_AUDIT_BRIEF.md` — this self-contained current-state brief.
- `progressive_tokenizer/model.py` — tokenizer architecture and prefix masking.
- `progressive_tokenizer/joint_flow.py` — joint rectified flow.
- `progressive_tokenizer/autoregressive_flow.py` — shifted causal trunk and token
  flow head.
- `progressive_tokenizer/training.py` — optimizer parameter grouping.
- `train_progressive_tokenizer.py`
- `train_progressive_joint_flow.py`
- `train_progressive_ar_flow.py`
- `tests/test_progressive_tokenizer.py`

Selected configs and metrics:

- `tokenizer_runs/v2-cross-n16-d64-s1/config.json`
- `tokenizer_runs/v2-cross-n16-d64-s1/metrics_final.json`
- `prior_runs/v2-joint-cross-n16-d64-s1/config.json`
- `prior_runs/v2-joint-cross-n16-d64-s1/history.jsonl`
- `prior_evals/v2-joint-cross-n16-d64-020000/metrics.json`
- `prior_runs/v2-ar-cross-n16-d64-s1/config.json`
- `prior_runs/v2-ar-cross-n16-d64-s1/history.jsonl`
- `prior_evals/v2-ar-cross-n16-d64-020000/metrics.json`

The working tree is intentionally not yet a clean archival commit. Treat the
files above as the current implementation, and report any mismatch between this
brief, configs, checkpoints, and code.
