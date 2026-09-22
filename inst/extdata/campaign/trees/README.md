# Native-tree evaluation: blocked before scoring

Both unchanged 0.5.0 native-tree contracts were attempted with their required
**one-round** schedule on breast, epsilon 1, delta 1e-6, three sites and
registry defaults. Each stopped during the first planned replicate (split
seed 20260820), after the Flower training exchange and before local prediction:

```text
Saved native-tree ensemble violates its canonical container contract.
```

This is a different failure from the superseded five-round preflight issue.
The error is raised in dsFlowerClient's
`.validate_native_tree_ensemble_artifact()` (`R/validate.R:217`), called from
`.native_tree_release_metadata()` in `R/run.R`. The guard checks the saved
ensemble's container fields and canonical bytes. The exact failing subcheck
was not retained, so no narrower root cause is asserted. No package code,
canonical runner, registry, privacy defaults or mechanism was modified or
bypassed. The required fallback failed at the same guard; evaluation stopped.

| Contract | Dataset | n | epsilon | Attempt | Wall time |
|---|---|---:|---:|---|---:|
| random_forest | breast | 683 | 1 | First replicate failed before scoring | 49.94 s |
| extra_trees | breast | 683 | 1 | Authorized fallback, same failure | 51.19 s |
| random_forest | breast | 683 | 4, 8 | Not run after blocker | — |
| random_forest | cdc9k | 9,000 planned | 1, 4, 8 | Not run after blocker | — |

No central, federated-DP or trivial test metric, gap, mean or SD exists.
No completed three-replicate cell exists. The test prediction call was never
reached and the central comparator runs after the federated scoring call.
The released-model checkpoint and replicate-metric files are absent.
Training exchanges occurred, so this is not a claim of zero privacy
expenditure. Diagnostics in `summary.json` are **null/unassessed**, never PASS.

The two `failure_*.json` files are the original run-emitted failure records.
The `attempts.json` companion records environment and requested design,
distinguishing requested values from retained node reports. Sanitized
SuperLink/SuperNode traces are in `diagnostics_*.log`.
`registration_*.json` preserves the pre-run tooling hashes and options.

The design and reproduction commands are in
[`tools/campaign/trees/README.md`](../../../../tools/campaign/trees/README.md).
The tools-only commit was pushed as `99e6928` before the fallback completed.
The protocol was frozen in the local tooling commit before either attempt;
the commit was rebased without content changes onto the shared branch.

Sources and citations:

- [UCI 15, Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original):
  Wolberg, W. (1990). *Breast Cancer Wisconsin (Original)* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z.
  Raw 699 records, 683 complete cases, nine features, class 4 positive.
- [UCI 891, CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators):
  *CDC Diabetes Health Indicators* (2017) [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C53919. Public CDC BRFSS-derived
  table linked by UCI to Alex Teboul's preparation. The planned cohort
  uses the existing stratified 9,000-row selection at seed 20260819.

Dataset download URLs and verified raw SHA-256 digests are in `attempts.json`
and the tooling README. No new measurements are inferred from provenance.

Pod: `pod-flower-tabular`, ID `2sy2g4pb3xwgqt`, hostname `b85270415cca`.
Both package versions: 0.5.0. Matching client/node runner SHA-256:
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
The existing installation was reused: original provisioning 97 seconds,
new provisioning 0 seconds. No Flower processes remained after cleanup;
the pod is left running. Local copies are under
`dsflower-cells/evidence/trees/`, and the current stop note is
`dsflower-cells/BLOCKED_TREES.md`.
