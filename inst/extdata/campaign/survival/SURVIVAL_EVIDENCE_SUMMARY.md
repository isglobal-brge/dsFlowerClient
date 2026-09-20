# Survival campaign evidence summary

Current rows combine frozen v1 AFT evidence with the selected hazard-v2 confirmation. The 90 v1 cohort cells, three synthetic successes and seven failed attempts remain unchanged; `v1/summary.json` preserves their original report. The current report uses 60 AFT cells and 30 hazard-v2 cells, with three matched seeds (1101, 1102, 1103) per row.

Hazard selection: h06, 10 rounds × 4 local epochs, batch 64, SGD learning rate 0.05, K=10 equal-width bins over 1825 days. Selection used the highest mean inner-validation C-index over twelve cells per candidate; all six candidates and 72 development scores are retained in `hazard-v2/hazard_v2_selection.json`. Confirmation reuses the v1 outer holdouts and is not independent validation. The per-run privacy budgets do not account for the whole development sweep.

C-index and held-out NLL are mean ± sample SD over three public split replicates. Student-t 95% intervals are in `summary.json`; overlapping splits and only three replicates limit their interpretation. Compare NLL only within the matching likelihood/grid and against its matching null. Ranking utility does not establish probability calibration.

## C-index — mean ± SD

| Cohort / subset | Variant | Protocol | ε | Federated-DP | Pooled-DP | Pooled-nonprivate | Null |
|---|---|---|---:|---:|---:|---:|---:|
| lung1 / full | hazard | v2 / h06 | 1 | 0.496 ± 0.023 | 0.459 ± 0.005 | 0.477 ± 0.031 | 0.500 ± 0.000 |
| lung1 / full | hazard | v2 / h06 | 4 | 0.460 ± 0.055 | 0.472 ± 0.029 | 0.477 ± 0.031 | 0.500 ± 0.000 |
| lung1 / full | hazard | v2 / h06 | 8 | 0.467 ± 0.034 | 0.480 ± 0.041 | 0.477 ± 0.031 | 0.500 ± 0.000 |
| lung1 / full | lognormal | v1 | 1 | 0.513 ± 0.053 | 0.507 ± 0.032 | 0.527 ± 0.013 | 0.500 ± 0.000 |
| lung1 / full | lognormal | v1 | 4 | 0.517 ± 0.061 | 0.511 ± 0.062 | 0.527 ± 0.013 | 0.500 ± 0.000 |
| lung1 / full | lognormal | v1 | 8 | 0.517 ± 0.058 | 0.519 ± 0.058 | 0.527 ± 0.013 | 0.500 ± 0.000 |
| lung1 / full | weibull | v1 | 1 | 0.512 ± 0.062 | 0.515 ± 0.047 | 0.499 ± 0.006 | 0.500 ± 0.000 |
| lung1 / full | weibull | v1 | 4 | 0.519 ± 0.057 | 0.512 ± 0.061 | 0.499 ± 0.006 | 0.500 ± 0.000 |
| lung1 / full | weibull | v1 | 8 | 0.518 ± 0.060 | 0.511 ± 0.060 | 0.499 ± 0.006 | 0.500 ± 0.000 |
| support2 / full | hazard | v2 / h06 | 1 | 0.587 ± 0.014 | 0.612 ± 0.014 | 0.622 ± 0.003 | 0.500 ± 0.000 |
| support2 / full | hazard | v2 / h06 | 4 | 0.589 ± 0.009 | 0.623 ± 0.003 | 0.622 ± 0.003 | 0.500 ± 0.000 |
| support2 / full | hazard | v2 / h06 | 8 | 0.595 ± 0.004 | 0.621 ± 0.004 | 0.622 ± 0.003 | 0.500 ± 0.000 |
| support2 / full | lognormal | v1 | 1 | 0.634 ± 0.005 | 0.640 ± 0.003 | 0.649 ± 0.004 | 0.500 ± 0.000 |
| support2 / full | lognormal | v1 | 4 | 0.636 ± 0.003 | 0.639 ± 0.002 | 0.649 ± 0.004 | 0.500 ± 0.000 |
| support2 / full | lognormal | v1 | 8 | 0.634 ± 0.007 | 0.639 ± 0.004 | 0.649 ± 0.004 | 0.500 ± 0.000 |
| support2 / full | weibull | v1 | 1 | 0.623 ± 0.004 | 0.626 ± 0.001 | 0.639 ± 0.005 | 0.500 ± 0.000 |
| support2 / full | weibull | v1 | 4 | 0.622 ± 0.007 | 0.627 ± 0.003 | 0.639 ± 0.005 | 0.500 ± 0.000 |
| support2 / full | weibull | v1 | 8 | 0.622 ± 0.005 | 0.627 ± 0.003 | 0.639 ± 0.005 | 0.500 ± 0.000 |
| support2 / heterogeneous | hazard | v2 / h06 | 8 | 0.593 ± 0.006 | 0.622 ± 0.005 | 0.622 ± 0.003 | 0.500 ± 0.000 |
| support2 / heterogeneous | lognormal | v1 | 8 | 0.633 ± 0.004 | 0.639 ± 0.003 | 0.649 ± 0.005 | 0.500 ± 0.000 |
| support2 / heterogeneous | weibull | v1 | 8 | 0.614 ± 0.006 | 0.627 ± 0.003 | 0.639 ± 0.005 | 0.500 ± 0.000 |
| support2 / small600 | hazard | v2 / h06 | 1 | 0.477 ± 0.034 | 0.514 ± 0.028 | 0.525 ± 0.009 | 0.500 ± 0.000 |
| support2 / small600 | hazard | v2 / h06 | 4 | 0.467 ± 0.028 | 0.511 ± 0.014 | 0.525 ± 0.009 | 0.500 ± 0.000 |
| support2 / small600 | hazard | v2 / h06 | 8 | 0.473 ± 0.010 | 0.526 ± 0.008 | 0.525 ± 0.009 | 0.500 ± 0.000 |
| support2 / small600 | lognormal | v1 | 1 | 0.557 ± 0.012 | 0.594 ± 0.011 | 0.638 ± 0.007 | 0.500 ± 0.000 |
| support2 / small600 | lognormal | v1 | 4 | 0.570 ± 0.009 | 0.592 ± 0.006 | 0.638 ± 0.007 | 0.500 ± 0.000 |
| support2 / small600 | lognormal | v1 | 8 | 0.569 ± 0.006 | 0.593 ± 0.007 | 0.638 ± 0.007 | 0.500 ± 0.000 |
| support2 / small600 | weibull | v1 | 1 | 0.561 ± 0.004 | 0.578 ± 0.013 | 0.627 ± 0.009 | 0.500 ± 0.000 |
| support2 / small600 | weibull | v1 | 4 | 0.568 ± 0.006 | 0.590 ± 0.009 | 0.627 ± 0.009 | 0.500 ± 0.000 |
| support2 / small600 | weibull | v1 | 8 | 0.566 ± 0.010 | 0.588 ± 0.010 | 0.627 ± 0.009 | 0.500 ± 0.000 |

## Held-out NLL — mean ± SD

| Cohort / subset | Variant | Protocol | ε | Federated-DP | Pooled-DP | Pooled-nonprivate | Null |
|---|---|---|---:|---:|---:|---:|---:|
| lung1 / full | hazard | v2 / h06 | 1 | 0.314 ± 0.039 | 0.277 ± 0.021 | 0.256 ± 0.013 | 0.285 ± 0.010 |
| lung1 / full | hazard | v2 / h06 | 4 | 0.297 ± 0.017 | 0.258 ± 0.014 | 0.256 ± 0.013 | 0.285 ± 0.010 |
| lung1 / full | hazard | v2 / h06 | 8 | 0.294 ± 0.015 | 0.257 ± 0.012 | 0.256 ± 0.013 | 0.285 ± 0.010 |
| lung1 / full | lognormal | v1 | 1 | 6.445 ± 0.158 | 6.415 ± 0.151 | 6.391 ± 0.161 | 6.378 ± 0.166 |
| lung1 / full | lognormal | v1 | 4 | 6.424 ± 0.161 | 6.404 ± 0.173 | 6.391 ± 0.161 | 6.378 ± 0.166 |
| lung1 / full | lognormal | v1 | 8 | 6.424 ± 0.165 | 6.401 ± 0.174 | 6.391 ± 0.161 | 6.378 ± 0.166 |
| lung1 / full | weibull | v1 | 1 | 6.729 ± 0.223 | 6.546 ± 0.124 | 6.409 ± 0.165 | 6.402 ± 0.172 |
| lung1 / full | weibull | v1 | 4 | 6.693 ± 0.115 | 6.553 ± 0.105 | 6.409 ± 0.165 | 6.402 ± 0.172 |
| lung1 / full | weibull | v1 | 8 | 6.693 ± 0.114 | 6.551 ± 0.120 | 6.409 ± 0.165 | 6.402 ± 0.172 |
| support2 / full | hazard | v2 / h06 | 1 | 0.135 ± 0.003 | 0.131 ± 0.002 | 0.128 ± 0.002 | 0.147 ± 0.002 |
| support2 / full | hazard | v2 / h06 | 4 | 0.133 ± 0.002 | 0.127 ± 0.002 | 0.128 ± 0.002 | 0.147 ± 0.002 |
| support2 / full | hazard | v2 / h06 | 8 | 0.133 ± 0.002 | 0.128 ± 0.002 | 0.128 ± 0.002 | 0.147 ± 0.002 |
| support2 / full | lognormal | v1 | 1 | 5.670 ± 0.070 | 5.651 ± 0.057 | 5.479 ± 0.016 | 5.816 ± 0.044 |
| support2 / full | lognormal | v1 | 4 | 5.647 ± 0.046 | 5.670 ± 0.064 | 5.479 ± 0.016 | 5.816 ± 0.044 |
| support2 / full | lognormal | v1 | 8 | 5.665 ± 0.064 | 5.664 ± 0.058 | 5.479 ± 0.016 | 5.816 ± 0.044 |
| support2 / full | weibull | v1 | 1 | 5.417 ± 0.031 | 5.786 ± 0.018 | 4.985 ± 0.028 | 5.134 ± 0.035 |
| support2 / full | weibull | v1 | 4 | 5.444 ± 0.044 | 5.763 ± 0.067 | 4.985 ± 0.028 | 5.134 ± 0.035 |
| support2 / full | weibull | v1 | 8 | 5.441 ± 0.024 | 5.779 ± 0.076 | 4.985 ± 0.028 | 5.134 ± 0.035 |
| support2 / heterogeneous | hazard | v2 / h06 | 8 | 0.133 ± 0.002 | 0.128 ± 0.002 | 0.128 ± 0.002 | 0.147 ± 0.002 |
| support2 / heterogeneous | lognormal | v1 | 8 | 5.675 ± 0.068 | 5.652 ± 0.064 | 5.483 ± 0.019 | 5.816 ± 0.044 |
| support2 / heterogeneous | weibull | v1 | 8 | 5.418 ± 0.030 | 5.829 ± 0.125 | 4.979 ± 0.033 | 5.134 ± 0.035 |
| support2 / small600 | hazard | v2 / h06 | 1 | 0.196 ± 0.007 | 0.174 ± 0.011 | 0.153 ± 0.004 | 0.187 ± 0.002 |
| support2 / small600 | hazard | v2 / h06 | 4 | 0.178 ± 0.006 | 0.155 ± 0.005 | 0.153 ± 0.004 | 0.187 ± 0.002 |
| support2 / small600 | hazard | v2 / h06 | 8 | 0.177 ± 0.004 | 0.153 ± 0.004 | 0.153 ± 0.004 | 0.187 ± 0.002 |
| support2 / small600 | lognormal | v1 | 1 | 6.060 ± 0.155 | 5.811 ± 0.157 | 5.523 ± 0.018 | 5.819 ± 0.043 |
| support2 / small600 | lognormal | v1 | 4 | 6.051 ± 0.138 | 5.818 ± 0.145 | 5.523 ± 0.018 | 5.819 ± 0.043 |
| support2 / small600 | lognormal | v1 | 8 | 6.071 ± 0.147 | 5.785 ± 0.112 | 5.523 ± 0.018 | 5.819 ± 0.043 |
| support2 / small600 | weibull | v1 | 1 | 5.293 ± 0.114 | 5.450 ± 0.106 | 5.015 ± 0.028 | 5.135 ± 0.036 |
| support2 / small600 | weibull | v1 | 4 | 5.237 ± 0.074 | 5.534 ± 0.253 | 5.015 ± 0.028 | 5.135 ± 0.036 |
| support2 / small600 | weibull | v1 | 8 | 5.244 ± 0.090 | 5.504 ± 0.202 | 5.015 ± 0.028 | 5.135 ± 0.036 |

## Preregistered utility diagnostics

These are empirical diagnostics, not privacy proofs. Shortfall is G = pooled-nonprivate − federated-DP, the reverse of the historic delta sign. A FLAG is retained for review.

| Cohort / subset | Variant | Adjacent-ε envelope | ε=8 designated floor | Small-N trend | Near-central flags (historic / minimum-site) |
|---|---|---|---|---|---|
| lung1 / full | hazard | 1→4: FLAG, 4→8: PASS | N/A | N/A | 2 / 3 |
| lung1 / full | lognormal | 1→4: PASS, 4→8: PASS | N/A | N/A | 0 / 0 |
| lung1 / full | weibull | 1→4: PASS, 4→8: PASS | N/A | N/A | 0 / 0 |
| support2 / full | hazard | 1→4: PASS, 4→8: PASS | FAIL | N/A | 0 / 0 |
| support2 / full | lognormal | 1→4: PASS, 4→8: PASS | PASS | N/A | 0 / 0 |
| support2 / full | weibull | 1→4: FLAG, 4→8: PASS | PASS | N/A | 0 / 0 |
| support2 / heterogeneous | hazard | N/A | N/A | N/A | 0 / 0 |
| support2 / heterogeneous | lognormal | N/A | N/A | N/A | 0 / 0 |
| support2 / heterogeneous | weibull | N/A | N/A | N/A | 0 / 0 |
| support2 / small600 | hazard | 1→4: PASS, 4→8: PASS | N/A | PASS | 0 / 0 |
| support2 / small600 | lognormal | 1→4: PASS, 4→8: PASS | N/A | PASS | 0 / 0 |
| support2 / small600 | weibull | 1→4: PASS, 4→8: PASS | N/A | FLAG | 0 / 0 |

V1 AFT investigation notes are preserved; new hazard flags remain pending reviewer investigation. Designated floors require both C-index ≥ 0.60 and C-index ≥ null + 0.05; failures remain failures.

## Retained failed attempts

| Evidence record | Recorded cause |
|---|---|
| `failure-support2-full-weibull-eps1-seed1101-fuse-startup.json` | Could not verify the dsFlower runner compatibility on the remote nodes. |
| `failure-synthetic-hazard-concurrent-host.json` | Federated training failed (status 1); no model was accepted or saved. |
| `failure-synthetic-lognormal-concurrent-host.json` | Federated training failed (status 1); no model was accepted or saved. |
| `failure-synthetic-weibull-bound-array.json` | The nodes completed without publishing a private model. |
| `failure-synthetic-weibull-install-order.json` | Bundled declarative model validator not found. |
| `failure-synthetic-weibull-postprocess-timeout.json` | ! ms is not a length 1 integer |
| `failure-synthetic-weibull-symbol-census.json` | Could not verify that target DataSHIELD symbols are unused. |

Raw h06 confirmation cells and preregistration/selection/completion records are under `hazard-v2/`. `hazard-v2/source-digests.json` records their pod2 source paths and SHA-256 digests; `v1/source-digests.json` records all original v1 JSON digests. No cells were rerun or scores changed.
