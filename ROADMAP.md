# AFIG roadmap — 2026-08-16

Working plan following the box loss and the diffusability-theory thread.
Companion docs: `TOKENIZER_DESIGN.md` (Gates A–H),
`reports/2026-08-16_wandb_recovery/runs.md` (lost-campaign inventory and
reimplementation specs), `reports/2026-08-16_diffusability_theory/notes.md`
(theory + instruments).

Standing rules (unchanged): decoded FID decides model selection; teacher-forced
/ cross-representation latent MSE is diagnostic only; ±0.3 FID protocol noise;
two seeds for any selection decision; eager/compiled results never pooled.

New operational rules (the box loss must not repeat):

- Every trainer/eval run ends by syncing `prior_evals/` + `metrics_final.json`
  into git and uploading the checkpoint + cache to W&B artifacts (or HF Hub).
- Code lands in git **before** its first multi-hour run, not after.

## Current state (recovered)

| item | status |
|---|---|
| v2 `64x16` cross-only tokenizer | spec + final evals recovered; weights lost (34.06 dB full, rFID ~7.2) |
| Joint prior @60k | FID 39.37 (reference ceiling); weights lost |
| One-at-a-time AR @60k | FID 81.68; weights lost |
| Block-concat AR | negative (109.66), closed |
| Rolling o8 / o64 / o8-b768 | trained on lost box; **no decoded FID anywhere**; code lost, spec recovered |
| history-robust (+reliability), headpos, tokenwise-centered | same — undecided, spec recovered |
| v3 VAE kl sweep / cross-causal / 128x8 (+ 60k AR each) | tokenizer evals recovered; AR verdicts undecided; code lost, spec recovered |
| Audit E1 (recipe ceiling) / E2 (matched pixel baseline) | never run |

## Phase 0 — rebuild and harden (unblocks everything)

- [ ] Reimplement the lost features from the recovered specs, with tests,
      committed as they land: `variational` (+`kl_weight`) and `cross_causal`
      pooling in the tokenizer; rolling prior trainer (headless per-token-time
      model, `local_data_time = clamp(frontier - idx/overlap, 0, 1)`, loss on
      active registers); history-noise + per-token reliability conditioning;
      `head_position_conditioning`; tokenwise boundary projections.
- [ ] Retrain v2 `64x16` cross-only tokenizer (15k x 512) and rebuild the
      100k/10k caches. **Validation gate:** final prefix PSNRs match the
      recovered W&B numbers within noise; then upload weights + caches.
- [ ] Retrain the two reference priors on that cache: joint 60k and
      one-at-a-time AR 60k. **Validation gate:** FID within ~1–2 of 39.37 /
      81.68. These are the comparison anchors for everything below.
- [ ] Wire the artifact-sync step into the trainers (or a post-run script).

## Phase 1 — decide the undecided (highest information per GPU-hour)

Ordered by expected information value; all evaluated with decoded FID at
matched steps, plus the standard extras (oracle-prefix sweep, slot-wise
temperature probe, per-token conditional MSE) where applicable.

1. [ ] **Rolling o8 vs o64 vs AR vs joint** at matched 60k. The headline
       question: does schedule-causal generation with full attention close the
       joint–AR gap? Sweep frontier steepness at inference on the o64 model
       (one model gives the whole joint↔AR curve).
2. [ ] **history-robust (+reliability) vs plain AR** — the exposure fix,
       cheap and already half-validated (unlabeled variant was killed early).
3. [ ] **v3 tokenizer shaping under a matched prior**: VAE-kl1e4 first (it
       matched the deterministic recon ceiling), then cross-causal if VAE is
       positive. Run each cache under the *joint* prior for the cleanest
       modelability verdict, plus the better of {AR, rolling} from step 1.
4. [ ] headpos / tokenwise-centered as small controls if step 1–2 results
       leave the trunk+head AR track alive.

## Phase 2 — ceiling anchors (can run in parallel with Phase 1 on this box)

- [ ] **E1 recipe run**: joint prior on the selected cache with EMA (~0.9995),
      LR decay, logit-normal t, 150–200k steps. Establishes the real ceiling;
      every "gap" statement is provisional until this exists.
- [ ] **E2 matched pixel baseline**: same 512-wide DiT, optimizer, budget,
      solver, eval protocol, on pixels. Pins the representation tax.
- [ ] CNF-NLL diagnostic: exact per-token NLL through the existing RF head
      (ODE + Hutchinson divergence) added to the eval battery — restores a
      principled cross-arm metric at zero retraining cost.

## Phase 3 — latent shaping (theory → practice)

- [ ] **Axis-A/B scorecard tooling** run on every cache as a standard
      tokenizer eval: eigenband per-sample ordering consistency
      (margin-normalized), per-direction activity CV, energy-covariance minus
      Gaussian prediction, conditioning-gain (nonlinear prefix predictability
      at matched noise levels). Prediction on record: current latents are
      audio-like on axis A.
- [ ] **Energy-consistency regularizer arm**: per-coordinate kurtosis→3 and/or
      per-token energy-CV penalty — the Gaussianization component of the KL,
      unbundled from the rate penalty — compared head-to-head with VAE-kl1e4
      on the scorecard + joint FID.
- [ ] **Frontier-noise AE training**: finite-σ generalization of prefix
      masking (sample a frontier position, noise tokens per the rolling
      schedule, reconstruct). Co-designs tokenizer with the rolling sampler;
      targets conditioning order at the sampler's actual state distribution.
- [ ] **Crescendo gauge experiment** (cheap, anytime after Phase 0): per-token
      descending population rescale of the existing cache (profile matched to
      Gate-H functional order), same joint prior, inverse at decode. Tests
      schedule design in isolation; the flat gauge (whitening) already lost.

## Phase 4 — generation-track upgrades (gated on Phases 1–2)

Two tracks, kept deliberately alive in parallel — different perks:

**One-at-a-time AR** (streaming, KV-cacheable, anytime prefixes, modular
heads, exact NLL):
- [ ] Cross-attention conditioning for the head (the bandwidth fix — the
      leading suspect from the audit) at matched steps vs single-vector AdaLN.
- [ ] Full-covariance GMM head as a cheap density-family floor; NF (spline
      coupling) head if one-pass sampling / exact NLL become priorities.
- [ ] Slot-wise temperature schedule (eval-only, free).

**Rolling / diffusion-forcing** (joint-grade conditioning, schedule as an
inference knob):
- [ ] Frontier schedule shaped by the measured spectral half-recovery curve
      (registers 3/6/15/27) instead of linear.
- [ ] KV-caching for frozen (fully denoised) history to cut sampling cost.
- [ ] If Phase 3's frontier-noise tokenizer lands: retrain rolling on it —
      the co-designed pair is the thesis experiment of the project.

## Exit criteria / decision gates

- Rolling ≥ joint − ~2 FID at matched budget → fixed-order sequential
  generation is vindicated; prioritize rolling + co-design; AR track becomes
  the efficiency/streaming variant.
- Rolling stuck near AR → conditioning bandwidth was not the story; elevate
  latent shaping (Phase 3) and the E1/E2 anchors to primary.
- E2 pixel ≈ E1 joint → representation tax is small; scale question
  (ImageNet-32) opens. E2 ≪ E1 → tax is real; Phase 3 is the project.
- Defer ImageNet-32 until the CIFAR tax is measured (unchanged from Gate F).
