## R4 once-scored results

Records: [ε=1](har_r4_window_pytorch_lstm_eps1.json), [ε=4](har_r4_window_pytorch_lstm_eps4.json), [ε=8](har_r4_window_pytorch_lstm_eps8.json).

The selected R4 schedule did **not** improve held-out ε=8 AUC over R3:
0.787685 versus 0.801364.
At ε=8, the paired ordered AUC contrasts are
-0.060526 ± 0.007338
for finite schedule plus federation,
-0.003144 ± 0.009037 for clipping, and
-0.118309 ± 0.014971
for added noise plus independent sampling variation. These are annotations,
not an exact causal allocation. No training or alternative followed scoring.

The selected schedule and all new training were frozen before the exclusive
TEST scoring marker. All nine federations passed their 135 node-round checks;
six matched noiseless twins were scored once. The R3 central model identities
and scores are reused on the byte-identical split. No scored model was retrained.

Values below are mean ± sample SD over three seeds. The AUC gap is paired
federated-DP minus central. All scores use the held-out subjects’ windows.

### Macro one-vs-rest AUC

| ε | Central (R3) | Non-private federated | Clipped noiseless | Federated-DP | Trivial | DP − central |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.969665 ± 0.014018 | 0.909138 ± 0.012948 | 0.905994 ± 0.013402 | 0.743989 ± 0.007536 | 0.500000 ± 0.000000 | -0.225675 ± 0.019423 |
| 4 | 0.969665 ± 0.014018 | 0.909138 ± 0.012948 | 0.905994 ± 0.013402 | 0.779427 ± 0.013191 | 0.500000 ± 0.000000 | -0.190238 ± 0.017454 |
| 8 | 0.969665 ± 0.014018 | 0.909138 ± 0.012948 | 0.905994 ± 0.013402 | 0.787685 ± 0.011003 | 0.500000 ± 0.000000 | -0.181979 ± 0.023962 |

### Accuracy

| ε | Central (R3) | Non-private federated | Clipped noiseless | Federated-DP | Trivial |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.791879 ± 0.089138 | 0.612035 ± 0.059457 | 0.637032 ± 0.017555 | 0.357199 ± 0.005412 | 0.182219 ± 0.000000 |
| 4 | 0.791879 ± 0.089138 | 0.612035 ± 0.059457 | 0.637032 ± 0.017555 | 0.385364 ± 0.016521 | 0.182219 ± 0.000000 |
| 8 | 0.791879 ± 0.089138 | 0.612035 ± 0.059457 | 0.637032 ± 0.017555 | 0.410926 ± 0.050873 | 0.182219 ± 0.000000 |

### Log-loss

| ε | Central (R3) | Non-private federated | Clipped noiseless | Federated-DP | Trivial |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.566915 ± 0.210415 | 0.864885 ± 0.094352 | 1.123201 ± 0.139061 | 1.480044 ± 0.056640 | 1.789941 ± 0.000000 |
| 4 | 0.566915 ± 0.210415 | 0.864885 ± 0.094352 | 1.123201 ± 0.139061 | 1.339908 ± 0.013600 | 1.789941 ± 0.000000 |
| 8 | 0.566915 ± 0.210415 | 0.864885 ± 0.094352 | 1.123201 ± 0.139061 | 1.302039 ± 0.073154 | 1.789941 ± 0.000000 |

Pooled-DP was not run. Accuracy-versus-majority and chance-AUC checks are
annotations only. No alternative schedule or rerun followed these outcomes.

The R3 diagnosis measures clipping loss −0.076769 and a further marginal-noise
estimate −0.086579 on the inner split. Its ordered schedule/federation terms
depend on sampling and optimizer resets; they do not reallocate the historical
held-out gap. See the [complete diagnosis](../../../../tools/campaign/sequence/r4/SEQUENCE_DIAGNOSIS_R4.md).

Public evidence includes node-reported privacy settings, independent full-horizon
accounting, all per-seed arm metrics, model/runner hashes and wall-clock timings.
Declaration commit: `8c3e4451bc3338d169c59bbc7f435e05da69ad4a`.
The sequence pod remains running. No package code was changed.

---

# Corrected R4 window-level cell: pre-run declaration

Declared 2026-09-22T07:13:30.459226+00:00 before full-data R4 training or TEST scoring.
Selected **Adam 0.01, batch 512, 4 local epochs × five rounds**,
hidden32, no scheduler or penalties. Selection uses TRAIN subjects only,
with the diagnosis’s unchanged subject-disjoint inner split: fit subjects
3, 5, 6, 7, 11, 14, 15, 16, 19, 21, 22, 23, 26, 27, 28, 29; validation subjects 1, 8, 17, 25, 30.
Three subject-disjoint sites retain their original membership after removing
inner-validation subjects. The two highest noiseless-clipped diagnosis
candidates were compared using the REAL window-DP contract at ε=8 across
seeds 20260922–20260924; final-round mean inner-validation macro-AUC selects
the winner. No early checkpoint selection or held-out access was used.

| Adam LR / local epochs / batch | Inner AUC mean ± SD | Accuracy mean ± SD | Log-loss mean ± SD |
|---|---:|---:|---:|
| 0.01 / 4 / 512 | 0.771937 ± 0.030525 | 0.386652 ± 0.027010 | 1.288269 ± 0.069811 |
| 0.003 / 8 / 256 | 0.754319 ± 0.005975 | 0.356078 ± 0.000323 | 1.365349 ± 0.015789 |

The corrected cell uses ε∈{1,4,8}, δ=1e-6, window unit, unit clipping, the
same 21 TRAIN and nine held-out subjects, original three sites (2,553 /
2,397 / 2,402 windows), split and three seeds as `har_window_*`.
The selected schedule has [100, 100, 100] optimizer steps per site.
Raw 128×9 token-major windows receive the public channel-bound transform
`clip(x,-b,b)/b` once, with `b=(1,1,1,1,1,1,2,2,2)`.
**This is a window-level mechanism measurement on subject-disjoint sites
and provides no subject-level protection.**

Comparators on the identical split: the same network centrally trained
without privacy under the R3 nominal schedule (Adam .01, batch256, 20 epochs,
Adam reset every four epochs, 580 steps); selected-schedule non-private
federated twin without clipping or noise; selected-schedule clipped noiseless
federated twin; and TRAIN-frequency/majority trivial predictor. Reuse the
frozen R3 central model identities and metrics after exact split, bounds and
initialization verification. The two noiseless twins use matched public
Poisson streams and equal site weights; real DP retains node-owned randomness.
The optional pooled-DP twin is not scheduled.

Report macro one-vs-rest AUC, accuracy and log-loss, mean ± sample SD over
three seeds, and paired federated-DP minus central AUC gaps. Diagnostics
are annotations. Complete all training before the exclusive scoring marker;
read held-out windows once, score each new model once, then make no further
alternative or rerun. Selection supplies no end-to-end private guarantee.
No campaign-wide composition guarantee is claimed.

HAR provenance: Anguita et al. (2013), *A Public Domain Dataset for Human
Activity Recognition Using Smartphones*, ESANN; UCI dataset 240, DOI
10.24432/C54S4K. Thesis citation keys: `anguita_har_2013` and `uci_har`.
The existing official archive SHA-256 and all R3 evidence are preserved.
No package code changes; reuse `pod-flower-sequence-2` and leave it running.

---

# UCI HAR sequence cells: original and corrected R3

Token: `FLOWER_CELLS_SEQUENCE_2026-09-22`. The nine pre-R4 scored cells are indexed in
[summary.json](summary.json), with individual replicates and diagnostics in the
linked JSON files. The original pipeline did not learn: its noiseless central
twin gave **0.435143 / 0.169551 / 1.799129** (macro OVR AUC / accuracy /
log-loss), near chance utility. These original results are a pipeline/specification
failure, not a privacy utility finding. Their records remain unchanged.

## Diagnosis and declaration

TRAIN-only checks ruled out **(a) layout** and **(d) label misalignment**: the
contract receives `[N,128,9]` from C-order token-major flat rows, and prepared
features, targets and subjects match the archive exactly. **(b) unstandardised
inputs was not the cause:** the old contract already applied clipped affine
scaling using TRAIN-derived bounds. **(c) the schedule was inadequate:** only
five updates at SGD 0.001. In addition, the patient path collapsed all activities
within each subject to one average sequence and its modal label, leaving only
21 central training examples. No package plumbing defect was found.

The same contract-built GPU LSTM learned on 5,564 windows from 16 TRAIN subjects.
On 1,788 inner-validation windows from TRAIN subjects 1, 8, 17, 25 and 30,
the selected central-only run achieved **0.988999812 /
0.937360179 / 0.240914434** after
20 epochs. The schedule was selected by final inner-validation macro AUC between
Adam learning rates 0.003 and 0.01. TEST was not accessed for diagnosis or selection.
See [diagnostic audit](r3/diagnosis/audit.json),
[recorded experiments](r3/diagnosis/experiments.json), and
[the declaration committed before corrected training](../../../../tools/campaign/sequence/README.md)
at `c6136371e644bd0c9a9e945c53f8d5d544475115`; the binding
[protocol](../../../../tools/campaign/sequence/r3/protocol.json) is also embedded
in every corrected cell.

The corrected declaration is **Adam 0.01, batch
256, 4 local epochs × five rounds**,
hidden size 32, six classes, no scheduler or penalties. Inputs use the same
128 × 9 time-major layout. Fixed public design bounds are ±1 g for body
acceleration, ±1 rad/s for gyro, and ±2 g for total acceleration; the unchanged
contract applies `clip(x,-b,b)/b` once. These constants are declared design
choices, not empirical extrema or guaranteed archive limits.

Three subject-disjoint sites contain seven subjects each (2,553 / 2,397 / 2,402
windows). Both corrected units use seeds 20260922–20260924, epsilon 1 / 4 / 8,
delta 1e-6 and clipping norm 1. **Subject:** the released patient contract clips
each pooled subject's gradient, seven units/site and 20 steps/site. It does not
aggregate losses or gradients over the original windows within each subject;
that requested behavior is unsupported without a mechanism change. **Window:**
the released row contract clips each window, with 200 steps/site. This is a
window-level mechanism measurement and provides no subject-level protection.

The corrected central comparator is trained and scored once per seed on all
7,352 original TRAIN windows, without DP or federation, with identical
architecture, initialization, bounds and nominal 20-epoch schedule. Adam resets
every four epochs; ordinary shuffled minibatches give 580 updates. Its three
models are shared across all six corrected cells. The subject comparison
therefore includes the pooling/task mismatch. Trivial probabilities are TRAIN
class frequencies; classification is the TRAIN majority class.

## All scored cells

Metric triples are mean **macro one-vs-rest AUC / accuracy / log-loss** over three
training seeds. The gap is federated-DP minus paired central macro AUC, with
sample SD. Diagnostic thresholds are annotations only. The original central
comparator used 21 subject averages; corrected rows use the shared full-window
central comparator described above.

| Run | Contract | Dataset | Privacy unit | Epsilon | Central | Federated-DP | Trivial | AUC gap mean ± SD | Diagnostics |
|---|---|---|---|---:|---|---|---|---|---|
| [original](pilot_uci_har_pytorch_lstm_eps1.json) | pytorch_lstm | UCI HAR | subject pooled | 1 | 0.435143 / 0.169551 / 1.799129 | 0.434854 / 0.169551 / 1.799015 | 0.500000 / 0.182219 / 1.789941 | -0.000289 ± 0.006243 | AUC > .5: no; accuracy > majority: no |
| [original](pilot_uci_har_pytorch_lstm_eps4.json) | pytorch_lstm | UCI HAR | subject pooled | 4 | 0.435143 / 0.169551 / 1.799129 | 0.434235 / 0.169551 / 1.799279 | 0.500000 / 0.182219 / 1.789941 | -0.000908 ± 0.000795 | AUC > .5: no; accuracy > majority: no |
| [original](pilot_uci_har_pytorch_lstm_eps8.json) | pytorch_lstm | UCI HAR | subject pooled | 8 | 0.435143 / 0.169551 / 1.799129 | 0.435451 / 0.169551 / 1.799082 | 0.500000 / 0.182219 / 1.789941 | +0.000308 ± 0.000544 | AUC > .5: no; accuracy > majority: no |
| [R3](har_subject_pytorch_lstm_eps1.json) | pytorch_lstm | UCI HAR | subject pooled | 1 | 0.969665 / 0.791879 / 0.566915 | 0.460432 / 0.180975 / 1.799538 | 0.500000 / 0.182219 / 1.789941 | -0.509233 ± 0.074314 | AUC > .5: no; accuracy > majority: no |
| [R3](har_subject_pytorch_lstm_eps4.json) | pytorch_lstm | UCI HAR | subject pooled | 4 | 0.969665 / 0.791879 / 0.566915 | 0.471355 / 0.164687 / 1.797785 | 0.500000 / 0.182219 / 1.789941 | -0.498310 ± 0.076037 | AUC > .5: no; accuracy > majority: no |
| [R3](har_subject_pytorch_lstm_eps8.json) | pytorch_lstm | UCI HAR | subject pooled | 8 | 0.969665 / 0.791879 / 0.566915 | 0.534196 / 0.157561 / 1.794589 | 0.500000 / 0.182219 / 1.789941 | -0.435469 ± 0.030243 | AUC > .5: yes; accuracy > majority: no |
| [R3](har_window_pytorch_lstm_eps1.json) | pytorch_lstm | UCI HAR | window (row) | 1 | 0.969665 / 0.791879 / 0.566915 | 0.742579 / 0.357086 / 1.372999 | 0.500000 / 0.182219 / 1.789941 | -0.227086 ± 0.019217 | AUC > .5: yes; accuracy > majority: yes |
| [R3](har_window_pytorch_lstm_eps4.json) | pytorch_lstm | UCI HAR | window (row) | 4 | 0.969665 / 0.791879 / 0.566915 | 0.788904 / 0.414659 / 1.363176 | 0.500000 / 0.182219 / 1.789941 | -0.180760 ± 0.018169 | AUC > .5: yes; accuracy > majority: yes |
| [R3](har_window_pytorch_lstm_eps8.json) | pytorch_lstm | UCI HAR | window (row) | 8 | 0.969665 / 0.791879 / 0.566915 | 0.801364 / 0.484221 / 1.348498 | 0.500000 / 0.182219 / 1.789941 | -0.168301 ± 0.029491 | AUC > .5: yes; accuracy > majority: yes |

Full per-seed predicted-class counts, mean probabilities, probability spans,
metrics, timing, initialization hashes, release policies and accountant captures
are in the corrected JSONs. Original diagnostics remain in their pilot JSONs.

## Interpretation and limits

- Original: no useful central learning, so near-zero DP-versus-central gaps do
  not establish privacy utility.
- Subject: a valid subject-unit measurement of the released pooled surrogate,
  in the 21-unit regime. Its gap combines pooling/task change, federation,
  clipping and noise; it cannot isolate the cost of privacy.
- Window: window privacy only, despite subject-disjoint sites. The gap combines
  federation, sampling, clipping and noise; no noiseless federated control or
  pooled-DP twin was scheduled.
- Metrics use held-out windows; the split is fixed. SD measures training
  variation across three seeds, not split or population uncertainty. Central
  results repeated across corrected table rows are the same three models.
- Each epsilon/seed/unit is a separate per-training mechanism contract. No
  composed campaign guarantee is claimed. Non-DP inner selection on public
  TRAIN data does not provide an end-to-end private model-selection guarantee.
- Public seeds determine initialization. Node-owned cryptographic sampling and
  noise remain unchanged; no deterministic noise replacement is used.

## Provenance and preservation

dsFlower **0.5.1**, server patch
`c4eaaf153db4a7204878bf1d8995c16faa615110` on
`fix/import-guard-torch-generated-modules`; dsFlowerClient
**0.5.0**, release
`50dda000a32ffcbdd039c2b74c909df451392bfb`. Declaration commit:
`c6136371e644bd0c9a9e945c53f8d5d544475115`. Runtime tooling commit:
`c613637`. Both installed canonical runner hashes:
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`. No R3 package code, guard, privacy
mechanism or accountant changes. Exact environment and tooling hashes are in
[r3/runtime.json](r3/runtime.json).

All 18 corrected federations and three central models were verified before the
exclusive [R3 scoring marker](r3/test-scoring-started.json). The one final pass
loaded TEST once and predicted each of those 21 models once. Every scored cell
is retained; none was rerun or retuned after scoring. The public observer's
270 node-round captures and independent accounting are included in the cells.
The pod `pod-flower-sequence` remains running at `/workspace/cells-sequence`.

All 47 original files match
[the frozen hash manifest](r3/original-evidence-hashes.json). The original report
and index are retained byte-for-byte as [README-r1.md](README-r1.md) and
[summary-r1.json](summary-r1.json); pilot JSONs and both old scoring and failure
records are unchanged. The pre-training dsFlower 0.5.0 import failure remains
under [blocked-0.5.0](blocked-0.5.0/README.md) and is indexed separately from the
nine scored cells. Report assembly reads recorded JSONs only and does not score
or open any dataset split.

## Dataset citation

[UCI HAR, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[official archive](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip),
DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0.
Archive SHA-256: `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
