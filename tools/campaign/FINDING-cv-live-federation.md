# Campaign finding: live-federation cross-validation fails closed at 0.4.1

**Date:** 2026-08-30 · **Status:** OPEN (documented for the next release) ·
**Severity:** capability unavailable in live federation; fail-closed, no
disclosure impact.

## Summary

`ds.flower.cross_validate()` (equivalently `ds.flower.fit(cross_validation =
k)`) deterministically fails over a real SuperLink/SuperNode federation at
v0.4.1 with:

```
Error: Federated cross-validation failed or produced a forbidden fold
artifact; no result was accepted.
```

Reproduced 4/4 attempts (heart cohort, `pytorch_logreg`, 3 sites, folds = 3,
rounds 3 and 5, epsilon 4), on an otherwise idle host where the same
harness runs `ds.flower.fit()` cells green, including immediately before and
after. Repro: `tools/campaign/run_cv_demo.R` (this directory) over the
DSLite-PSOCK federation of `campaign_lib.R`.

## Diagnosis so far

- The ServerApp reaches `_run_cross_validation`, builds the fold-1 strategy
  and configures every round with the full roster (`configure_train: Sampled
  3 nodes (out of 3)` for all rounds), then aborts with
  `RuntimeError: cross-validation requires every round of every fold`
  (`server_app.py:745`): `strategy.available_rounds` never fills.
- A round becomes unavailable only when some node reply carries
  `public-preflight-unavailable = 1` or `execution-unavailable = 1`
  (`_RequireCompleteTrain.aggregate_train`). The fit path tolerates such
  rounds (availability-only history); the CV path strictly requires all
  rounds, by design.
- Instrumentation of the node-side exception fallback
  (`_safe_fallback_reply`) captured **no** traceback: the unavailable
  replies do not come from the swallowed-exception path. The remaining
  emitters are the deliberate unavailable replies and the stateless replay
  path (`_replay_reply`, which echoes cached flags and falls back with
  `execution_unavailable = TRUE` on an inexact cache match — a plausible
  suspect if Flower redelivers or re-keys messages across CV folds, since
  the cache key includes `request_id`/`fold`/`release-index`).
- No shipped harness executes CV over a live federation at this tag: the
  package's CV tests are contract/KAT-level (budget split, provenance,
  job-pin rejection), and the scheduled CI integration federation trains a
  plain DP logreg. The committed suites are green; this finding is about
  the one path they do not reach.

## What this does and does not affect

- Every committed campaign evidence cell (17 fit cells across two
  contracts) is unaffected; `ds.flower.fit` is green throughout.
- The failure is fail-closed on both sides (server refuses a partial CV;
  nodes disclose nothing). No privacy impact.
- The CV protocol-property tests (fold assignment invariance, budget split,
  all-or-nothing release) remain valid statements about the design.

## Suggested next steps (release-owner work)

1. Reproduce with server-side logging of the per-reply metric flags in
   `aggregate_train` (which node, which flag, which round) — one run
   suffices given determinism. Note the runner is byte-verified in
   lock-step across both packages, so any instrumentation or fix implies a
   coordinated dual-package release.
2. Prime suspects, in order: the stateless-replay cache-key match for
   `cv-train` claims (operation/fold/release-index vs what CV rounds
   actually carry); per-fold pin validation in the neural claim handler
   (`pins["num_rounds"]` vs per-fold `claim["num_rounds"]`).
3. Once fixed, `run_cv_demo.R` is the acceptance harness: 3 replicates,
   released pooled OOF metrics vs the pooled out-of-fold `glm` twin.
