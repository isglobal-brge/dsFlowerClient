# Executed representative native-tree cells

dsFlowerClient **0.5.1** (patch `598d21c4627496a65d44034440f2d221ce6e04a2`), dsFlower **0.5.0**.
Both runner SHA-256 hashes: `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.

Seven cells, 21 scored replicates; each used random_forest, three sites, one Flower round,
8 trees per node, depth 4, registry-default max_features, delta 1e-6 and unit clipping.
The central comparator uses 8 trees/depth 4 on the same pooled training split.
Splits/central seeds are 20260820–20260822; privacy randomness remains node-owned.
All models trained once and were scored once; no scored settings were changed.

| Contract | Dataset | n | Epsilon | Central AUC | Fed-DP AUC | Fed accuracy | Majority accuracy | AUC gap mean ± SD | Diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| random_forest | breast | 683 | 1 | 0.996567 | 0.690933 | 0.712895 | 0.649635 | -0.305634 ± 0.150421 | PASS |
| random_forest | breast | 683 | 4 | 0.996567 | 0.986501 | 0.846715 | 0.649635 | -0.010066 ± 0.006466 | PASS |
| random_forest | breast | 683 | 8 | 0.996567 | 0.993914 | 0.948905 | 0.649635 | -0.002653 ± 0.003427 | PASS |
| random_forest | cdc9k | 9,000 | 1 | 0.807081 | 0.695477 | 0.860741 | 0.860556 | -0.111604 ± 0.019161 | PASS |
| random_forest | cdc9k | 9,000 | 4 | 0.807081 | 0.744952 | 0.860556 | 0.860556 | -0.062129 ± 0.014745 | FAIL |
| random_forest | cdc9k | 9,000 | 8 | 0.807081 | 0.757392 | 0.860556 | 0.860556 | -0.049689 ± 0.010058 | FAIL |
| random_forest | cdc45k | 45,000 | 8 | 0.805514 | 0.796944 | 0.860667 | 0.860667 | -0.008571 ± 0.002521 | FAIL |

Trivial AUC is 0.5 throughout. Gaps are paired federated-minus-central AUC;
SD is the sample SD across three seeds. Each cell JSON and summary.json retain AUC,
accuracy, Brier and log-loss for all three comparators, including replicate values and SDs.

The registered epsilon-8 diagnostic requires mean AUC > 0.5 **and** accuracy strictly
above majority accuracy. Breast passes. CDC9k fails because accuracy equals the
majority baseline. The single pre-declared CDC45k/epsilon-8 alternative also fails
that accuracy condition, despite narrowing the AUC gap. No additional alternative ran.
The overall primary epsilon-8 diagnostic remains failed; the alternative does not replace it.
The older permissive floor is retained only as a separate descriptive field.

The gap combines algorithm, federation and DP effects: the native release averages
24 trees across three node forests while the central comparator uses the per-node
contract default of 8 trees. It is not a causal estimate of privacy cost alone.

## Validation and provenance

- R native-tree, validation/cross-validation and import suites: 478 checks, no failures, warnings or skips.
- Python canonicalization, native prediction, import and artifact validation suites: 14 tests passed.
- The exact saved failing 0.5.0 artifact now validates unchanged; see diagnosis-0.5.0/.
- Full saved-release audit confirmed three clients in every training RDS and three
  members in every hash-bound ensemble, with matching size, tree count/depth and node privacy responses.
- The old harness read an absent fit$n_clients field and serialized []. Only that reporting
  field was repaired; release-audit.json records original/corrected JSON hashes and count sources.
  The initial matrix summary command stopped on this assertion after all 21 runs had finished.
  Summary was regenerated after the audit, with no retraining or rescoring.
- Exact executed tooling: `162a880ce83488855c1cc5fa63fc2598707f9c21`.
  The current reproduction tooling includes the client-count reporting correction.
- Original failure evidence is preserved under failed-0.5.0/.
- Public releases and site-secret files remain on the pod under /workspace/cells/runs/trees-0.5.1/.
  Secrets are not included in this evidence.
- Training processes drained and the pod remains running for the reviewer.

Reproduction: [tools/campaign/trees/README.md](../../../../tools/campaign/trees/README.md).
Diagnosis: [TREES_DIAGNOSIS.md](diagnosis-0.5.0/TREES_DIAGNOSIS.md).
