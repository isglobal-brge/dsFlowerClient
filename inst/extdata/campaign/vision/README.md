# BUS-BRA vision campaign

R5 was declared before training and is now executed: SGD lr 3, momentum 0.9, two local epochs, full-site batches (284/site; 852 pooled), five rounds. Selected under the real DP contract on training patients only; real inner-validation AUC 0.705486.

See the [complete declaration and comparator definitions](../../../../tools/campaign/vision/README.md) and [frozen protocol](../../../../tools/campaign/vision/r5_impl/protocol.json). Fixed 852/212 patient split, three sites of 284; three seeds; epsilon order 8, 4, 1; delta 1e-6; patient unit; clipping norm 1. All fits precede one held-out scoring pass.

R3: schedule-limited at registry defaults. R4: selection lesson—non-private pruning does not transfer under DP; final scoring stopped before test access. Original records remain unchanged.

BUS-BRA provenance: thesis key `gomezflores_busbra_2024`, paper DOI `10.1002/mp.16812`, dataset DOI `10.5281/zenodo.8231412`.

# BUS-BRA R5 corrected cell

Declared SGD lr 3, momentum 0.9, two local epochs, full-site batch (284/site; 852 pooled), five rounds. Selected on training patients under the real DP contract. Fixed 852/212 patient split; three training seeds. Primary per-image metrics; mean ± sample SD.

| ε | Central AUC | Nonprivate federated AUC | Federated-DP AUC | Pooled-DP AUC | Trivial AUC | DP accuracy | Majority accuracy | AUC gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.757 ± 0.000 | 0.500 ± 0.000 | 0.660 ± 0.008 | 0.686 ± 0.010 | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.674 ± 0.000 | -0.098 ± 0.008 |
| 4 | 0.757 ± 0.000 | 0.500 ± 0.000 | 0.617 ± 0.083 | 0.646 ± 0.032 | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.674 ± 0.000 | -0.140 ± 0.083 |
| 1 | 0.757 ± 0.000 | 0.500 ± 0.000 | 0.534 ± 0.037 | 0.563 ± 0.046 | 0.500 ± 0.000 | 0.643 ± 0.059 | 0.674 ± 0.000 | -0.223 ± 0.037 |

Primary image metrics:

| ε | Arm | AUC | Accuracy | Brier | Log-loss |
|---:|---|---:|---:|---:|---:|
| 8 | central | 0.757 ± 0.000 | 0.743 ± 0.000 | 0.185 ± 0.000 | 0.579 ± 0.000 |
| 8 | nonprivate_federated | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.326 ± 0.000 | 11.267 ± 0.000 |
| 8 | federated_dp | 0.660 ± 0.008 | 0.674 ± 0.000 | 0.326 ± 0.000 | 10.855 ± 0.412 |
| 8 | pooled_dp | 0.686 ± 0.010 | 0.674 ± 0.000 | 0.326 ± 0.000 | 5.541 ± 2.757 |
| 8 | trivial | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.220 ± 0.000 | 0.631 ± 0.000 |
| 4 | central | 0.757 ± 0.000 | 0.743 ± 0.000 | 0.185 ± 0.000 | 0.579 ± 0.000 |
| 4 | nonprivate_federated | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.326 ± 0.000 | 11.267 ± 0.000 |
| 4 | federated_dp | 0.617 ± 0.083 | 0.674 ± 0.000 | 0.326 ± 0.000 | 8.856 ± 4.143 |
| 4 | pooled_dp | 0.646 ± 0.032 | 0.674 ± 0.000 | 0.326 ± 0.000 | 8.180 ± 3.752 |
| 4 | trivial | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.220 ± 0.000 | 0.631 ± 0.000 |
| 1 | central | 0.757 ± 0.000 | 0.743 ± 0.000 | 0.185 ± 0.000 | 0.579 ± 0.000 |
| 1 | nonprivate_federated | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.326 ± 0.000 | 11.267 ± 0.000 |
| 1 | federated_dp | 0.534 ± 0.037 | 0.643 ± 0.059 | 0.339 ± 0.036 | 5.465 ± 4.355 |
| 1 | pooled_dp | 0.563 ± 0.046 | 0.652 ± 0.039 | 0.328 ± 0.010 | 3.390 ± 1.603 |
| 1 | trivial | 0.500 ± 0.000 | 0.674 ± 0.000 | 0.220 ± 0.000 | 0.631 ± 0.000 |

Secondary patient metrics:

| ε | Arm | AUC | Accuracy | Brier | Log-loss |
|---:|---|---:|---:|---:|---:|
| 8 | central | 0.811 ± 0.000 | 0.759 ± 0.000 | 0.160 ± 0.000 | 0.498 ± 0.000 |
| 8 | nonprivate_federated | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.330 ± 0.000 | 11.404 ± 0.000 |
| 8 | federated_dp | 0.694 ± 0.021 | 0.670 ± 0.000 | 0.330 ± 0.000 | 11.023 ± 0.431 |
| 8 | pooled_dp | 0.719 ± 0.012 | 0.670 ± 0.000 | 0.330 ± 0.000 | 5.607 ± 2.816 |
| 8 | trivial | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.221 ± 0.000 | 0.635 ± 0.000 |
| 4 | central | 0.811 ± 0.000 | 0.759 ± 0.000 | 0.160 ± 0.000 | 0.498 ± 0.000 |
| 4 | nonprivate_federated | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.330 ± 0.000 | 11.404 ± 0.000 |
| 4 | federated_dp | 0.656 ± 0.100 | 0.670 ± 0.000 | 0.330 ± 0.000 | 8.972 ± 4.197 |
| 4 | pooled_dp | 0.677 ± 0.040 | 0.670 ± 0.000 | 0.330 ± 0.000 | 8.327 ± 3.850 |
| 4 | trivial | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.221 ± 0.000 | 0.635 ± 0.000 |
| 1 | central | 0.811 ± 0.000 | 0.759 ± 0.000 | 0.160 ± 0.000 | 0.498 ± 0.000 |
| 1 | nonprivate_federated | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.330 ± 0.000 | 11.404 ± 0.000 |
| 1 | federated_dp | 0.546 ± 0.037 | 0.635 ± 0.052 | 0.345 ± 0.032 | 5.452 ± 4.569 |
| 1 | pooled_dp | 0.578 ± 0.059 | 0.653 ± 0.030 | 0.333 ± 0.005 | 3.393 ± 1.634 |
| 1 | trivial | 0.500 ± 0.000 | 0.670 ± 0.000 | 0.221 ± 0.000 | 0.635 ± 0.000 |

Interpretations:

- DP-aware selected schedule: federated-DP AUC 0.659782; central gap -0.097537; accuracy 0.673797 versus majority 0.673797. Diagnostic does not pass; annotation only. Brier 0.326203, log-loss 10.854644; discrimination and calibration must be interpreted separately.
- DP-aware selected schedule: federated-DP AUC 0.616977; central gap -0.140342; accuracy 0.673797 versus majority 0.673797. Diagnostic does not pass; annotation only. Brier 0.326193, log-loss 8.856419; discrimination and calibration must be interpreted separately.
- DP-aware selected schedule: federated-DP AUC 0.534397; central gap -0.222922; accuracy 0.642602 versus majority 0.673797. Diagnostic does not pass; annotation only. Brier 0.338951, log-loss 5.465402; discrimination and calibration must be interpreted separately.

Limits:

- Primary evaluation is per image; partitioning, head training and privacy are per patient. Patient-mean-feature evaluation is secondary.
- All three R5 seeds use the same 852/212 patient split (split seed 20260919). SD describes three training realizations, not split or population uncertainty.
- The converged central fit is deterministic and reused across all seeds and epsilons; its zero SD is not an uncertainty estimate.
- The nonprivate federated twin removes both clipping and noise, retaining the finite schedule, Poisson sampling, expected-batch divisor and FedAvg structure.
- Nonprivate-twin Poisson draws use independent benchmark randomness; initialization and sampling law are matched, not individual sampling draws.
- Canonical federated image prediction retains the released numerical defaults; direct twins and secondary patient inference disable TF32 and use deterministic controls. Training tensor parity is verified; test-feature bitwise parity across these inference routes is not asserted.
- The central-to-private AUC gap combines finite optimization, federation, clipping and noise; it is not a pure noise effect.
- The derived private-training gap (federated-DP minus nonprivate federated) includes clipping, noise and their resulting training trajectories; it is not a pure noise effect.
- Selection used one 681/171 patient inner split, three emulated noise seeds over 736 candidates, then three actual epsilon-8 confirmations; selection uncertainty is not estimated.
- The original cell had already scored this cohort. R5 did not access held-out records before the locked final scoring pass; no claim of a previously unused external test cohort is made.
- Epsilon is the per-training five-round guarantee. No campaign-wide privacy composition claim is made across diagnosis, cells, twins or repeated public-cohort fits.
- Node-owned secret randomness is not published; the public seed alone does not reproduce DP sampling or noise.

R3 was schedule-limited at registry defaults. R4 demonstrated that non-private pruning does not transfer under DP; its final scoring was stopped before test access. All original records remain unchanged.

BUS-BRA: Gómez-Flores, Gregorio-Calas and Pereira (2024), Medical Physics 51:3110–3123. Paper DOI 10.1002/mp.16812; dataset DOI 10.5281/zenodo.8231412; thesis key `gomezflores_busbra_2024`.
