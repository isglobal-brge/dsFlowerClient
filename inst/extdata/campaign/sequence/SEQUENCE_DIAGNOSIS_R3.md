# Sequence diagnosis R3

Token: `FLOWER_CELLS_SEQUENCE_2026-09-22`. 2026-09-22.
No package code changes. Pod: `pod-flower-sequence`, NVIDIA A40.

The pipeline can learn with the unchanged contract-built LSTM: TRAIN-subject
inner validation reaches **0.988999812 macro one-vs-rest AUC, 0.937360179 accuracy,
0.240914434 log-loss**. The previous central twin did not fit the window task:
it trained 21 subject-averaged, modal-labelled examples for five SGD updates.
Its old held-out result (0.435142921 AUC / 0.169550956 accuracy / 1.799129 log-loss)
is preserved as a pipeline/specification failure, not a privacy utility result.
A constant predicted class is not necessarily exactly uniform probabilities;
AUC can differ from 0.5 because small probability differences still rank windows.

## TRAIN-only causal checks

The official training archive was reopened and checked byte-for-value against
prepared feature values, labels and subject IDs. No TEST features, labels or
subjects were read for diagnosis. Fit subjects were 3,5,6,7,11,14,15,16,19,21,22,
23,26,27,28,29 (5,564 windows); held-out TRAIN subjects were 1,8,17,25,30 (1,788
windows). All comparisons use seed 20260922 and the exact contract-built
DPLSTM(9,32), final-timestep state, six-logit head on CUDA. No DP or federation.

| Central diagnostic condition | Optimizer steps | Validation macro-AUC | Accuracy | Log-loss |
|---|---:|---:|---:|---:|
| Public bounds; windows; Adam .003, 20 epochs | 440 | 0.965103476 | 0.786912752 | 0.550719321 |
| Public bounds; windows; Adam .01, 20 epochs (selected) | 440 | 0.988999812 | 0.937360179 | 0.240914434 |
| Old bounds; windows; SGD .001, 5 epochs | 870 | 0.560018968 | 0.140380313 | 1.796451449 |
| Old bounds; windows; Adam .01, 20 epochs | 440 | 0.957820474 | 0.775167785 | 0.625357330 |
| Old bounds; pooled subjects; SGD .001, 5 updates | 5 | 0.523767612 | 0.152125280 | 1.805508137 |
| Public bounds; pooled subjects; Adam .01, 20 updates | 20 | 0.530474937 | 0.177852349 | 3.447253942 |

These are diagnostics on held-out TRAIN subjects, not new scores of the official
TEST split. Pooled diagnostic fits use the 16 inner-fit subjects; the original
scored central used all 21 training subjects. Therefore the old pooled diagnostic
is a matched inner-split reproduction of the failure mode, not a rerun of the
old scored cell.

## Which suspected cause broke the previous run?

- **(a) No layout defect.** `prepare_public_data.py` stacks nine channels last,
  then C-flattens. The captured model spec declares reshape `[128,9]`.
  A forward pre-hook verified exact recurrent input `[2,128,9]` and values.
  `R/model_registry.R:814` builds reshape/LSTM/head;
  `model_spec.py:389` uses batch-first recurrence and line 400 takes the final
  timestep. The 1,152 columns were not treated as one token and were not
  feature-major. No additional 0.5.1 plumbing patch is indicated.
- **(b) Not unstandardised.** `client_app.py:658` already clips and applies
  `(x-center)/half_range`; the former bounds were TRAIN extrema widened 10%.
  With those old bounds, the corrected window schedule reaches 0.957820474 AUC
  / 0.775167785 accuracy. The fixed public bounds improve this comparison,
  but missing scaling did not cause the old failure.
- **(c) Yes: the schedule was ineffective.** SGD .001, batch32, one local epoch
  and five rounds gave only five updates on seven pooled subjects/site (also
  five central updates on 21 pooled subjects). Even without pooling, five
  epochs at this LR give 0.140380313 validation accuracy and 1.796451449
  log-loss. The corrected fit uses an allowed Adam optimizer and sufficient
  steps; no mechanism is changed.
- **(d) No accidental label misalignment.** Fresh official TRAIN reads exactly
  equal prepared X/y/subject arrays; labels are original activity codes minus
  one. Site filters preserve alignment. However, the released patient path
  intentionally replaces a subject's many activity labels with one modal label.

**Additional decisive specification failure: patient pooling destroys the task.**
`client_app.py:583` averages all features in a subject and uses modal categorical
labels; `_train_neural` invokes it at lines 750–752. The 21 resulting targets
have counts `[6,0,0,1,5,9]`: no upstairs or downstairs examples. Correcting the
schedule while retaining pooling still gives only 0.177852349 accuracy and
3.447253942 validation log-loss. Such a gap against a full-window central model
cannot isolate the effect of privacy noise. Per-subject clipping of an aggregate
of original-window losses is unsupported by this contract; adding it would be
a mechanism change, outside this task.

## Frozen corrected cells

The declaration is in `tools/campaign/sequence/README.md` and
`tools/campaign/sequence/r3/protocol.json`, before corrected training/scoring.
Choose the maximum final-epoch inner-validation macro-AUC of two predeclared
candidates: Adam .003 or .01, batch256, 20 epochs, optimizer reset every four
epochs. Select .01. Keep hidden32, 128 tokens, nine features, six classes,
no scheduler, no penalties. Three sites, five rounds, seeds 20260922–20260924,
epsilon 1/4/8, delta1e-6, clipping norm1.

Use fixed public channel clip/scale constants `(1,1,1,1,1,1,2,2,2)` in g for
acceleration and rad/s for gyro; the runner transforms once. These are chosen
public bounds, not guaranteed raw archive extrema. TRAIN clipping fractions
per channel are recorded in `r3/diagnosis/audit.json` (gyro x/y/z approximately
3.95%/3.43%/0.74%). No fitted moments or TEST-derived extrema are used.

The **subject** cell uses the unchanged released patient surrogate, per-subject
clipping after feature-mean/modal-label pooling, seven units/site and 20 updates.
The requested full-window subject-loss mechanism remains unsupported and is
not claimed. The **window** cell retains every row, uses 2,553/2,397/2,402 units,
200 updates/site and Poisson q=.1, and provides window-level protection only.

The central reference is shared once per seed, trained on all 7,352 windows with
the selected 20-epoch schedule (580 ordinary minibatch updates), no DP or
federation. Trivial is TRAIN class-frequency probabilities and the majority
class. Final metrics are macro OVR AUC, accuracy, log-loss; gap is DP minus
paired central macro-AUC, mean and sample SD across seeds. All diagnostics are
annotations. One exclusive R3 test-scoring marker covers all six cells; no
scored cell is rerun and no alternative is scheduled.

Public TRAIN-only non-DP schedule selection is not an end-to-end private model
selection claim. Each epsilon/seed/cell is a separate training budget; no
campaign-wide composition guarantee is asserted.

## Provenance and retained tooling interruption

Installed dsFlower 0.5.1, server fix commit
`c4eaaf153db4a7204878bf1d8995c16faa615110`; dsFlowerClient 0.5.0 release
`50dda000a32ffcbdd039c2b74c909df451392bfb`. Canonical runners and guard remain
unchanged. Original evidence is from client commit `e77615c`; old pilot JSON
hashes are retained in `r3/original-evidence-hashes.json`, with original summary
and README copied byte-for-byte to `summary-r1.json` and `README-r1.md`.

The TRAIN-only helper needed float32 conversion for pooled diagnostic arrays,
matching the released totalizer. Two completed candidate results were retained;
a partial default-schedule window ablation was stopped and restarted unscored.
The interruption, executed initial source, exact histories and selection are
retained under `r3/diagnosis/`. No federation or test-scored model was repeated.

Official archive SHA-256:
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
[UCI dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[official signal documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names),
[DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0.
Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
