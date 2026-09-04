# Post-campaign cleanup audit

Date: 2026-09-04 UTC

Status: read-only audit complete; no artifacts or source files deleted.

## Conclusion

The repository checkout occupies about 74 GiB, but tracked source and result
files occupy only 16 MiB and `.git` occupies 60 MiB. Nearly all storage is in
ignored experiment state: `tokenizer_runs` is 33 GiB and `prior_runs` is 41
GiB. Source-history rewriting is neither necessary nor useful.

The safest high-yield cleanup is to remove redundant optimizer-bearing resume
checkpoints while retaining every model-only final checkpoint and every durable
evaluation. A second reversible tier removes regenerable latent caches from
sunsetted tokenizer arms while retaining the selected v27 seed-2 cache and the
final v34 cache.

## Proposed disk cleanup

| Tier | Exact class | Count | Reclaim | Why it is recoverable |
|---|---|---:|---:|---|
| A | `tokenizer_runs/**/checkpoint_latest.pt` and `prior_runs/**/checkpoint_latest.pt` | 67 | 49.335 GiB | Every directory has a corresponding `checkpoint_final.pt`; every one also has a `wandb_backup_attempted` marker. Training is complete, so optimizer/scheduler state is no longer needed for the selected evaluation. |
| B | tokenizer `latents*.pt` caches except v27 seed 2 and v34 | 28 | 5.898 GiB | These are deterministic/regenerable from retained tokenizer finals. The selected cache and the final-screen cache remain local. |

Together these tiers reclaim approximately **55.23 GiB**, reducing the checkout
from about 74 GiB to about 19 GiB. They retain all 67 model-only final
checkpoints (16.445 GiB), v27 seed-2 and v34 caches, tracked FID/KID outputs,
W&B run identifiers, reports, and launch/config metadata.

Do not delete final checkpoints, the two retained caches, `prior_evals`, or
report files in this pass. More aggressive removal of sunsetted final
checkpoints can be considered later, after independently confirming the remote
W&B artifacts rather than relying only on local backup-attempt markers.

## Proposed source cleanup

The full test suite currently collects 65 tests but stops on six import errors.
This is a coherent legacy-family failure, not a progressive-path regression:
commit `d1756ae6` removed `frequency.py` and `diffusion_decoder.py` under “wipe
clean,” while their callers and tests remained.

The definitely broken legacy family is:

- `autoencoder_models.py`
- `model_continuous.py`
- `model_joint_latent_diffusion.py`
- `train_autoencoder.py`
- `train_continuous.py`
- `train_joint_latent_diffusion.py`
- `sample_joint_latent_diffusion.py`
- `sample_ring_latent_continuous.py`
- `control_pixel_diffusion.py`
- `build_control_blind_sheet.py`
- `tests/test_autoencoder.py`
- `tests/test_control_pixel_diffusion.py`
- `tests/test_diffusion_decoder.py`
- `tests/test_joint_phase_oracle.py`
- `tests/test_latent_continuous.py`
- `tests/test_model_continuous.py`

This set is only about 520 KiB including `tests/test_smoke.py`; removing it is
about code coherence, not disk savings. `tests/test_smoke.py` should be split:
its legacy continuous-frequency tests should go with the family, while its
finite FID/KID check should move to a small `live_evaluation` test because that
module remains part of every progressive evaluator.

Before source deletion, tag commit `167c18e` as the complete pre-cleanup
research state. Then remove the family in one reviewable commit, repair the
smoke-test split, run the complete suite, and verify no current progressive
entrypoint imports a deleted file. Historical reports should remain unchanged;
git history and the tag retain the old implementation.

## Keep list

- The selected v27 residual-Perceiver implementation, final checkpoints, and
  seed-2 cache.
- V34 final tokenizer/prior checkpoints and cache as the closing architecture
  control.
- `progressive_tokenizer/`, current progressive train/evaluate entrypoints,
  `live_evaluation.py`, and their tests.
- Joint, autoregressive, PCA, spectral, hierarchy, sensitivity, and
  conditioning analysis code that still imports cleanly. These are small and
  remain useful research controls even when their experimental directions are
  not selected.
- All machine-readable protocols, metrics, samples, and journal/roadmap
  conclusions.

## Runtime-service cleanup

There are 69 completed `afig*.conf` supervisor definitions outside the git
repository. They consume negligible disk but clutter status output and would be
reconsidered on a supervisor/container restart. After artifact cleanup, archive
their names in the journal and remove the exited definitions from
`/etc/supervisor/conf.d`, then run `supervisorctl reread` and `update`. This is a
separate runtime-hygiene step and should not be conflated with research-data
retention.
