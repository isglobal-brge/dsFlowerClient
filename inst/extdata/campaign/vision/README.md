# BUS-BRA vision classification — dsFlower 0.5.1

The server import-guard patch unblocked `pytorch_resnet18`. All nine frozen
configurations completed five rounds on three sites, with central, pooled-DP
and trivial twins. Each model was scored once after all training completed.
The epsilon-8 utility comparisons are annotations and did not trigger tuning.

Execution token: `FLOWER_CELLS_VISION_2026-09-22`.

## Measured results

Values are arithmetic mean ± sample SD over seeds 20260919, 20260920 and
20260921. AUC gap is federated-DP minus central, calculated within seed before
aggregation. Metrics are per image; malignant is positive, and the accuracy
threshold is 0.5. Full precision, all replicate metrics, node manifests,
privacy configuration and accountant history are in the three cell JSONs.

| ε | Central AUC | Federated-DP AUC | Pooled-DP AUC | Trivial AUC | AUC gap |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5962 ± 0.0415 | 0.5321 ± 0.0398 | 0.5526 ± 0.0522 | 0.5000 ± 0.0000 | -0.0641 ± 0.0201 |
| 4 | 0.5962 ± 0.0415 | 0.5329 ± 0.0391 | 0.5552 ± 0.0578 | 0.5000 ± 0.0000 | -0.0633 ± 0.0195 |
| 8 | 0.5962 ± 0.0415 | 0.5328 ± 0.0401 | 0.5522 ± 0.0540 | 0.5000 ± 0.0000 | -0.0634 ± 0.0206 |

At ε = 8, mean AUC is 0.5328 (above 0.5); mean accuracy is 0.5200, versus majority rate 0.6704. Both diagnostic conditions hold for 0/3 replicates and do not hold for the means. This is an annotation only; all cells have status executed.

Outside the assessed ε=8 cell, the raw ε=4/20260921 diagnostic has a floating-point tie edge: accuracy equals the majority baseline, but its strict comparison differs by 1e-16. The raw score is retained and annotated; it does not affect the ε=8 diagnostic.

| ε | Branch | AUC | Accuracy | Brier | Log-loss |
|---:|---|---:|---:|---:|---:|
| 1 | central | 0.5962 ± 0.0415 | 0.6704 ± 0.0195 | 0.2158 ± 0.0067 | 0.6224 ± 0.0148 |
| 1 | federated_dp | 0.5321 ± 0.0398 | 0.5245 ± 0.1165 | 0.2488 ± 0.0216 | 0.6908 ± 0.0439 |
| 1 | pooled_dp | 0.5526 ± 0.0522 | 0.6606 ± 0.0123 | 0.2281 ± 0.0146 | 0.6519 ± 0.0364 |
| 1 | trivial | 0.5000 ± 0.0000 | 0.6704 ± 0.0195 | 0.2212 ± 0.0070 | 0.6343 ± 0.0148 |
| 4 | central | 0.5962 ± 0.0415 | 0.6704 ± 0.0195 | 0.2158 ± 0.0067 | 0.6224 ± 0.0148 |
| 4 | federated_dp | 0.5329 ± 0.0391 | 0.5227 ± 0.1138 | 0.2486 ± 0.0209 | 0.6903 ± 0.0425 |
| 4 | pooled_dp | 0.5552 ± 0.0578 | 0.6651 ± 0.0136 | 0.2263 ± 0.0129 | 0.6469 ± 0.0312 |
| 4 | trivial | 0.5000 ± 0.0000 | 0.6704 ± 0.0195 | 0.2212 ± 0.0070 | 0.6343 ± 0.0148 |
| 8 | central | 0.5962 ± 0.0415 | 0.6704 ± 0.0195 | 0.2158 ± 0.0067 | 0.6224 ± 0.0148 |
| 8 | federated_dp | 0.5328 ± 0.0401 | 0.5200 ± 0.1190 | 0.2495 ± 0.0229 | 0.6922 ± 0.0467 |
| 8 | pooled_dp | 0.5522 ± 0.0540 | 0.6678 ± 0.0161 | 0.2263 ± 0.0134 | 0.6475 ± 0.0328 |
| 8 | trivial | 0.5000 ± 0.0000 | 0.6704 ± 0.0195 | 0.2212 ± 0.0070 | 0.6343 ± 0.0148 |

## Data, model and privacy

BUS-BRA v1.0 contains 1,875 ultrasound images from 1,064 patients. Labels come
from the collection's `Pathology` field (benign/malignant); the released `Case`
identifier defines the privacy unit. Each seed reuses the byte-identical
segmentation split: 852 training patients, 212 test patients and three sites
of 284 training patients. Split SHA-256 values are recorded in every cell.

| Seed | Training images | Test images | Training patients | Test patients |
|---:|---:|---:|---:|---:|
| 20260919 | 1501 | 374 | 852 | 212 |
| 20260920 | 1500 | 375 | 852 | 212 |
| 20260921 | 1507 | 368 | 852 | 212 |

The actual dsImaging image-asset route admits three DSLite custodians with
patient identifiers and pathology vocabulary. The ImageNet ResNet-18 backbone
is frozen; transformed images have shape 3 × 224 × 224, feature vectors have
512 entries and the fixed feature domain is [-1,000,000, 1,000,000]. The
512-to-2 linear head has 1,026 trainable parameters and cross-entropy loss.
The released runner pools image features within patient and uses the modal
patient label before clipping. Source labels are consistent within patient.

The unchanged contract defaults are SGD 0.001, batch 32, one local epoch per
round, no momentum, weight decay, L1 penalty or scheduler. Privacy is patient
replace-one adjacency, clipping norm 1, delta 1e-6 and epsilon 1, 4 or 8 per
five-round fit. Each site accounts for 284 patients, Poisson rate 1/9 and
45 steps. The node-reported noise multipliers are 6.796875, 2.20703125 and
1.4599609375 respectively. All 135 node-round captures independently match
the declared mechanism and observed steps. Independent PRV composition checks
meet each requested epsilon/delta pair. Full node privacy settings, including
egress controls, are retained rather than reconstructed from defaults.

The central twin uses the same initial head, backbone, patient pooling and
five-epoch schedule with pooled Poisson sampling and no clipping or noise.
It is the matched finite-schedule reference, not an optimized upper bound.
The pooled-DP twin uses the unchanged released training function on 852
patients, Poisson rate 1/27 and 135 steps. The central fit is reused across
epsilons for each seed. The trivial baseline uses training-majority class
predictions and training prevalence for probability metrics.

Actual training features, targets and initial tensors match between node and
twin paths. The pushed inference routes retain their numerical defaults:
canonical prediction permits cuDNN TF32, while direct twin feature extraction
disables it and enables deterministic algorithms. Test feature bitwise parity
is not asserted; see [the data-free runtime record](verification-0.5.1/prediction-numerics.json).
No campaign-wide privacy composition is claimed across repeated public-cohort
fits. Public seeds control splits and initial heads; node-owned cryptographic
randomness also determines DP sampling/noise.

## Release verification and execution

Installed dsFlower **0.5.1** and dsFlowerClient **0.5.0**. The server patch is
`c4eaaf153db4a7204878bf1d8995c16faa615110` on
`fix/import-guard-torch-generated-modules`. The patch branch was read-only;
only its server package was installed. Both installed canonical runner hashes
and the CI source sync check match:

```text
2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724
```

The guarded, data-free vision construction passed with the active default-deny
finder, canonical runner pin and benchmark observer disabled. Nine guard tests
passed with no skips, three synthetic metric tests passed, and dsImaging
admission passed. [Verification records](verification-0.5.1/README.md) retain
the install, versions, guard hashes, dependency versions and input checks.

Execution used only `pod-flower-vision`, id `0nk8si7zupfczc`, NVIDIA A40,
root `/workspace/cells-vision`. The pod was left running. Every federation
cleaned up its workers successfully. No other pod was used.

The foreground training matrix took 179.7 minutes; federations ranged from 17.4 to 19.2 minutes. Group drivers including feature verification and twins ranged from 18.6 to 20.6 minutes. The single scoring process took 7.2 minutes. Verification start through completed scoring took 3.20 hours. Local epochs were already at the minimum of one; this setting was retained. Exact timestamps and per-replicate timings are recorded in the JSON evidence.

Recovered issues are retained: an early preflight ran during staged install;
the installer provisioned unused default Python environments; and the scorer
originally assumed the parent artifact directory. The corrected artifact path
was verified on a synthetic image before held-out access. The prepared Python
runtime and training settings were unchanged. Three independent canonical
epsilon predictions run concurrently within the one foreground scoring job
for each seed. The exclusive [scoring lock](verification-0.5.1/scoring-lock.json)
pins all artifacts and scoring drivers before test access; no scored
configuration was changed or rerun. Original 0.5.0 blocked records are preserved
byte-for-byte under [blocked-0.5.0](blocked-0.5.0/README.md).

## Reproduction and provenance

Follow the [vision driver reproduction instructions](https://github.com/isglobal-brge/dsFlowerClient/tree/evidence/representative-cells/tools/campaign/vision).
They install the server patch with existing Python environments, verify
versions and runner sync, perform the guarded construction and input checks,
run the foreground matrix, score once and assemble JSON evidence. Use a fresh
unscored run root for a reproduction. Do not rerun scoring on this pod.

[BUS-BRA v1.0](https://zenodo.org/records/8231412), dataset DOI
[10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412).
Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024), *BUS-BRA: A Breast
Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems*.
Medical Physics 51:3110–3123, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Zenodo declares CC BY 4.0; the archive attribution licence also requires the
paper citation. Publisher metadata and licence texts are retained in
[verification-0.5.1/provenance](verification-0.5.1/provenance/).

Archive SHA-256:
`ba3e6ed19cc37c682d8d39e25435bbf8a555a12cb7e641b5f2117685c95580ff`.
Checkpoint `resnet18-f37072fd.pth` SHA-256:
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Patient data, predictions, model arrays and node secrets remain on the pod.
Committed evidence contains aggregate measurements, public tensor hashes and
provenance only. [summary.json](summary.json) indexes the executed cell records.
