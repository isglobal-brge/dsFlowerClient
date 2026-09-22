# BUS-BRA vision R5: declared corrected cell

Declared before corrected training and held-out scoring. The immutable protocol is [r5_impl/protocol.json](r5_impl/protocol.json).

Selected **SGD, learning rate 3, momentum 0.9, two local epochs, full-site batch, weight decay 0, L1 0, scheduler none, five rounds**. Full site means 284 on each outer training site and 852 for pooled DP. Both use q=1, expected-batch divisor equal to their population, and ten accounted updates. Each mechanism is recalibrated independently.

Selection was made under the real released privacy contract on training patients only: 736 schedules ranked by faithful DP emulation on the fixed 681/171 inner split; the top three each confirmed once in actual federations at epsilon 8. The selected real inner-validation AUC was **0.7054858934**, accuracy **0.6783625731**, Brier **0.3216374269**, log-loss **11.0760075525**. These are selection estimates, with selection optimism and poor calibration. See [diagnosis](../../../inst/extdata/campaign/vision/r5/VISION_DIAGNOSIS_R5.md).

## Fixed protocol and comparators

Retain the original 20260919 outer split: **852 training / 212 held-out patients**, three original sites of **284 training patients**. Seeds **20260919, 20260920, 20260921** vary initialization/training randomness only. Split SHA256: `99af89943e5083db5f88a3723b147c129786acd763f99d85c298da5dc579463d`.

Run epsilon **8, 4, 1 in that order**, three seeds each, delta **1e-6**, patient privacy unit, clipping norm **1**, five rounds, equal site weights. Frozen ImageNet ResNet18, released image preprocessing, patient mean features/modal labels, two-logit affine head and architectural FiniteClamp remain unchanged. Extract training features afresh; diagnosis caches are not final training inputs.

- **Central:** converged pooled C=1 L2 logistic head on 852 patient feature vectors, unpenalized intercept; same affine model class. One deterministic fit reused across seeds and budgets.
- **Non-private federated finite-schedule twin:** three sites, selected schedule and identical seed initialization, optimizer reset each round, Poisson geometry and expected-batch division, equal weights; remove gradient clipping and Gaussian noise, retain architectural finite clamp, numerical totalization and selected regularization. One fit per seed reused across budgets.
- **Pooled DP:** unchanged released private fitter, all 852 patients, batch 852, same selected schedule and five optimizer-reset rounds, separately calibrated accountant, same initialization and privacy operations.
- **Trivial:** training-image majority/prevalence for primary image metrics; training-patient majority/prevalence for secondary patient metrics.

Primary metrics remain malignant-positive **per-image AUC, accuracy at 0.5, Brier, log-loss**. Patient-mean-feature metrics are secondary. Gap = federated-DP AUC minus converged central AUC. Report mean and sample SD across the three seeds; shared test patients mean SD measures training variation, not population uncertainty. Node-owned secure randomness is not determined by the public seed. Epsilon is per fit, not campaign-wide composition.

Complete all nine federations and all comparators, verify exact training tensors, node optimizer pins and full-horizon accounting, then open held-out data in **one exclusive scoring pass**. Never rerun a scored cell; no further alternative. Diagnostics are annotations only. The non-private twin removes clipping as well as noise; AUC gaps are not pure noise effects.

## Execution and provenance

Use `r5_impl/run_matrix.py`, then `r5_impl/score.py --verify-only`, then `r5_impl/score.py`, each with `--root /workspace/cells-vision` after sourcing `environment.sh`, using the existing GPU Python. Drivers refuse overwriting runs or repeating scoring. Reuse only pod-flower-vision (A40), leave it running. Packages remain dsFlower 0.5.1 / dsFlowerClient 0.5.0, runner SHA256 `2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`. R3/R4 and R5 diagnosis records remain unchanged.

R3 is **schedule-limited at registry defaults** (finite-schedule central AUC 0.596 ± 0.042). R4 is a **selection lesson: non-private pruning does not transfer under DP**; its two actual inner DP AUCs were 0.501 and 0.479, and its final matrix was stopped before test access. R5 selects directly under DP and has no post-test alternative.

BUS-BRA v1.0: Gómez-Flores, Gregorio-Calas and Pereira (2024), *BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems*, Medical Physics 51:3110–3123. Paper DOI `10.1002/mp.16812`; dataset DOI `10.5281/zenodo.8231412`; thesis citation key **`gomezflores_busbra_2024`**. CC BY 4.0; archive/checkpoint hashes are retained in the protocol. Patient records, features, predictions and secrets stay on the pod.
