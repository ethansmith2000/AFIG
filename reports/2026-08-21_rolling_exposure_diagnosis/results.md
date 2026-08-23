# Why the rolling engine loses to joint (2026-08-21)

> **Provenance note (2026-08-23).** The rolling engine was removed from the
> tree; the per-register MSE table below is not regenerable from HEAD. The code
> that produced it is recoverable at commit **f906d4a** (`git show
> f906d4a:progressive_tokenizer/rolling_flow.py`). Regenerate from that hash
> before publishing these numbers. All runs are single-seed.

All runs on the v5 vae-kl1e4 cache, 60k steps, 5k-sample decoded FID.

## The ladder

| config | FID | delta |
|---|---:|---|
| joint control | 35.85 | — |
| roll8 baseline (masked loss, bidirectional) | 70.85 | — |
| + supervise_all_tokens | 73.93 | +3.1 |
| + causal | 83.79 | +9.9 |
| + time jitter 0.02 / prefix noise 0.1 | 72.76 | **-11.0** |

Only the exposure-bias fix helps. It is also the same intervention that moved
the AR engine (79.38 -> 77.07).

## The diagnosis: training loss and FID are anti-correlated

Per-register MSE at end of training, and the same numbers weighted by each
register's share of latent energy (measured on the vae cache):

| model | 0-8 | 8-16 | 16-32 | 32-48 | 48-64 | energy-weighted | FID |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 0.563 | 0.640 | 0.677 | 0.632 | 0.574 | 0.621 (worst) | **35.85** |
| roll64 | 0.970 | 0.801 | 0.554 | 0.317 | 0.190 | 0.522 | 51.69 |
| roll8 | 0.978 | 0.571 | 0.390 | 0.266 | 0.166 | **0.434 (best)** | 70.85 |

energy share by band: 16.5 / 13.0 / 26.1 / 22.8 / 21.5 %.

Joint's per-register MSE is flat. Rolling's is steeply sloped: much worse than
joint on early registers, much better on late ones — and better *overall* once
weighted by energy. Yet its samples are far worse.

The two facts reconcile only one way. Rolling's late-register accuracy is
**conditional on ground-truth clean context**: at training, register 50 attends
to the true registers 0-49. At inference it attends to the model's own, which
carry MSE ~0.98 against a target variance of 2.0 — roughly half the variance
unexplained. The late-register accuracy is borrowed against a prefix that does
not exist at sampling time.

Everything orders on one axis, *how much the model leans on ground-truth clean
context*: joint (none) has the worst loss and the best FID; roll8 (a long clean
prefix) has the best loss and the worst FID; roll64 sits between. The ordering
is monotone and reversed between the two metrics.

### Consequences

1. **Rolling and joint training losses are not comparable.** Do not read them on
   a shared scale; only decoded FID ranks these engines.
2. **supervise_all_tokens is a bad idea, not a fix.** At overlap 8 about 87% of
   registers sit at schedule endpoints, where the optimal prediction is trivial
   (`-input` at t=0, `input` at t=1) with an irreducible floor. Unmasking spends
   ~87% of the gradient there, diluting the real task ~8x. The original mask was
   excluding junk targets, not starving the model. The earlier "supervision
   density confound" reading of roll8 vs joint was wrong.
3. **Causal hurts and the damage is not a receptive-field effect.** Per-band
   degradation is smallest on registers 0-8 (+2%) and largest mid-sequence
   (+5-6%), consistent with degraded early representations propagating forward
   rather than early registers being starved of context.
4. **Prefix noise 0.1 was an order of magnitude too gentle.** It was sized to
   sampler discretisation drift, which was never the real mismatch. The measured
   prefix error corresponds to context near t ~ 0.5-0.7, not t ~ 0.95 — and 0.1
   still bought 11 FID.

## Follow-up launched

Masked loss + bidirectional (the best baseline), with context corruption raised
to the measured scale:

- roll8-pn30: jitter 0.05, prefix noise 0.3
- roll8-pn50: jitter 0.10, prefix noise 0.5
- roll8-mix: jitter 0.05, prefix noise 0.3, independent_time_probability 0.25
- roll64-pn30: jitter 0.05, prefix noise 0.3

`independent_time_probability` draws independent per-register times for a
fraction of samples. Under the strict frontier, early registers *only* ever see
all-noise context and late registers *only* ever see perfect context; neither
sees the intermediate quality that actually occurs at inference. The mixture
covers that middle ground, and decouples the trained model from any single
inference schedule.
