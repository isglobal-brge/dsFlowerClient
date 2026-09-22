# BUS-BRA vision R4: declared corrected cell

Token: `FLOWER_CELLS_VISION_R4_2026-09-22`. Declared at 2026-09-22T05:36:23.774294+00:00, before corrected training and held-out scoring. The immutable declaration is [r4/protocol.json](r4/protocol.json).

## Training-only diagnosis and selection

Exact released ResNet-18 features (224 pixels), patient mean pooling and modal labels yield a converged fixed-C=1 logistic **five-fold patient inner-CV AUC 0.779405 ± 0.025903**. Out-of-fold AUC is 0.777700. L-BFGS-B converged in every fold, with no fitted feature scaling or preprocessing change.

A 25-candidate nonprivate search used the 681/171 patient inner split and the update count/Poisson rate of a 227-patient site, resetting optimizer state each round. Its top two candidates were confirmed through actual three-site dsImaging/DSLite federations at epsilon 8, delta 1e-6 and patient clipping norm 1.

| Candidate | Nonprivate inner AUC | Actual DP inner AUC | DP accuracy | DP Brier | DP log-loss |
|---|---:|---:|---:|---:|---:|
| Adam .003, 20 epochs, batch 32 | 0.776332 | 0.500940 | 0.678363 | 0.290158 | 1.502840 |
| Adam .01, 20 epochs, batch 32 | 0.772727 | 0.478683 | 0.660819 | 0.313500 | 2.215141 |

**Selected: Adam learning rate 0.003, 20 local epochs per round, batch 32, five rounds, weight decay 0, L1 0, scheduler none.** Highest actual inner-validation DP AUC at epsilon 8 selects .003. Both private confirmations are weak. This is retained as a finding; there is no diagnostic veto or post-test alternative. The parameter surface permits learning rates up to 10 and local epochs up to 1000; no parameter cap prevented the nonprivate head from learning. This search does not exhaust that surface.

All [sweep results, convergence records and hashed identifiers](r4/diagnosis/) are retained. The diagnosis feature cache remains on the pod and is not used by final training.

## Fixed split and final protocol

The original seeds had different outer splits. To avoid training-selection leakage into another seed's test cohort, R4 fixes the existing **20260919 split: 852 training patients, 212 held-out patients, and the original three sites of 284**. Seeds 20260919, 20260920 and 20260921 now vary initialization/training randomness only. The original full split JSON, held-out labels, images and previous predictions are not read in R4 before final scoring; only prepared training-site collections are used. The archived split hash is `99af89943e5083db5f88a3723b147c129786acd763f99d85c298da5dc579463d` and is checked when scoring opens the split.

Run the selected schedule once for epsilon 1, 4 and 8, delta 1e-6, patient privacy unit and clipping norm 1. Each site takes 180 updates per round, 900 over five rounds, with Poisson rate 1/9. Pooled-DP takes 2700 updates, Poisson rate 1/27. Each epsilon/seed is a separate public-cohort training budget; no end-to-end private model-selection or campaign-wide composition guarantee is claimed.

## Declared comparators and scoring

- Central: converged fixed-C=1 L2 logistic regression on all 852 patient feature vectors, unpenalized intercept; same affine model class represented as two logits. One deterministic fit is reused across seeds and epsilons.
- Nonprivate federated finite-schedule twin: three-site FedAvg, identical seed initialization and selected schedule, optimizer reset each round, Poisson sampling and expected-batch divisor, without clipping or noise. One fit per seed, reused across epsilons.
- Pooled-DP: unchanged released private fitter on all 852 patient features with the selected schedule and its own calibrated pooled mechanism.
- Trivial: training-image prevalence for probability metrics and training-majority class for accuracy.

Primary metrics remain per image: malignant-positive AUC, accuracy at 0.5, Brier and log-loss. Gap is federated-DP AUC minus converged-central AUC, paired within seed then summarized by arithmetic mean and sample SD. Patient-mean-feature metrics are secondary annotations from the same final scoring pass. The three seeds share one test cohort; SD is training variation, not population/split uncertainty. Central zero SD reflects reuse. The private-minus-nonprivate federated difference includes clipping and noise, not a pure noise effect.

All nine fits and twins must pass tensor, schedule and independent accounting checks before an exclusive scoring lock opens any held-out records. Score each fixed model once. Diagnostics remain annotations. Never rerun a scored configuration or introduce another alternative.

## Execution and reproduction

Use only `pod-flower-vision`, root `/workspace/cells-vision`, NVIDIA A40; leave the pod running. Installed dsFlower 0.5.1 and dsFlowerClient 0.5.0 and canonical runner hashes are unchanged. Raw training images may be copied byte-for-byte to POSIX `/tmp` for repeated access; metadata path rewrites and image hashes are audited. Final twin extraction is fresh and shared only in process memory.

```sh
source /workspace/cells-vision/src/dsFlowerClient/tools/campaign/vision/environment.sh
# In a fresh R4 workspace only; existing scored workspaces must not be rerun.
PY=/workspace/cells-vision/venvs/pytorch-gpu/bin/python
TOOLS=/workspace/cells-vision/src/dsFlowerClient/tools/campaign/vision/r4
"$PY" "$TOOLS/diagnose.py" --root /workspace/cells-vision
"$PY" "$TOOLS/confirm.py" --root /workspace/cells-vision
# Freeze the selected schedule and this declaration before continuing.
"$PY" "$TOOLS/run_matrix.py" --root /workspace/cells-vision
"$PY" "$TOOLS/score.py" --root /workspace/cells-vision --verify-only
"$PY" "$TOOLS/score.py" --root /workspace/cells-vision
"$PY" "$TOOLS/report.py" --root /workspace/cells-vision --out /workspace/cells-vision/evidence-r4
```

The first inner .003 launch stopped before any head initialization/update because SuperLink exceeded its 15-second readiness timeout. It is retained, with clean teardown and no scoring. The driver now permits the same process an additional 90 seconds through the existing readiness helper only for that exact startup error; package functions and model settings are not changed. A separate untrained retry completed. Both candidate fits completed and were validated exactly once.

## Historical cell and provenance

The original registry-default cell is **schedule-limited**: SGD .001, batch32, one local epoch, five rounds; its central comparator was a finite-schedule twin with AUC **0.596183 ± 0.041527**, not a converged reference. Original cell JSONs and verification records remain byte-identical. The original evidence README and summary are archived as `README-r1.md` and `summary-r1.json`; the original driver README is [README-before-r4.md](r4/README-before-r4.md).

[BUS-BRA v1.0](https://zenodo.org/records/8231412), 1875 images from 1064 patients. Gómez-Flores W, Gregorio-Calas MJ, Pereira WCA (2024), *BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems*, Medical Physics 51:3110–3123, [doi:10.1002/mp.16812](https://doi.org/10.1002/mp.16812). Dataset DOI [10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412). Thesis citation key: `gomezflores_busbra_2024`. CC BY 4.0; retained archive licence requires attribution. Archive and checkpoint hashes are in the protocol and original provenance records.

Results and interpretations: [campaign evidence](../../../inst/extdata/campaign/vision/README.md). Patient records, feature caches, predictions, model arrays and node secrets remain on the pod.
