# Segmentation evidence for dsFlowerClient 0.5.0

The current `batch16/` and `batch64/` evidence is **protocol v5: public BUSI
decoder pretraining followed by federated DP fine-tuning**. `protocol.md` and
`protocol-v5.md` are identical, with SHA256
`28a713f8ffe6f7bb7b74e7d59d13d1be265d5738b43503c60e6581a67e0c5820`.
The four cohort evidence files are byte-identical to the actual v5 files
verified by the September 20 reconciliation. No training or scoring was rerun
for this release packaging.

## Versions and interpretation

| Location | Protocol | Interpretation |
|---|---|---|
| `batch16/`, `batch64/` | v5, `28a713f8…` | Public initialization; batch16 is the selected primary arm, batch64 the prespecified sensitivity arm. |
| `retained/v4/batch16/`, `retained/v4/batch64/` | v4, `ea5915e5…` | Random initialization after public decoder/schedule development; retained negative utility boundary. |
| `retained/v3/batch16/`, `retained/v3/batch64/` | v3, `5a157dd8…` | Earlier random initialization with the original decoder; retained negative utility boundary. |

Each retained directory has its original `protocol.md`. The original v3 text is
also available as `protocol-v3.md`; `protocol-v4.md` includes its inherited v3
definitions. The v1/v2 protocols, invalidation record and diagnostic files
remain historical evidence, not current v5 results.

The v4 batch16 BUS-BRA foreground-positive federated-DP Dice means are 0.000,
0.005 and 0.212 at epsilon 1, 4 and 8 (0.00, 0.01 and 0.21 to two decimals).
The corresponding v5 means are 0.619, 0.638 and 0.646. V3's BUS-BRA epsilon8
mean is 0.000. These are separate experiments, never pooled across versions.

The current `campaign-status.json` combines only execution status from the two
v5 arm files and binds their file digests; it does not pool scores. The original
`summary-v5.json` retains development selection, confirmation floors and prior
version evidence. `selection.json` records the chosen narrow 9,521-parameter
decoder, batch16, 20 rounds and 60 public pretraining epochs.

## Public pretraining and scope

BUSI attribution: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast
ultrasound images*. Data in Brief. 2020;28:104863.
DOI: https://doi.org/10.1016/j.dib.2019.104863.
The source is Arya Shah's Kaggle mirror, version 1 (2021-03-14):
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset.
The mirror's CC0 declaration is preserved as provenance, not an assertion
about original rights.

`provenance/busi-v1/provenance.json` pins the dataset ZIP SHA256
`7fffc86a517934da55021f66641d81383058d26a30adb0ffab3780ca7e52d57d`.
`public-pretraining-manifest.json` pins the complete public-stage outputs;
`public-pretraining/audit.json` has SHA256
`d59334151e074de56afaaec81e92930ddaad8371f7f2665577096ce4ff7d9d57` and
records all 780 images and source/normalized mask digests. The epoch20/epoch60
seed JSONs bind dataset, audit, encoder, checkpoint and individual tensor
digests. Dataset and checkpoint tensors are not redistributed here.

BUSI pretraining is public and nonprivate, outside DP accounting. Its patient
mapping is unavailable; no BUSI held-out utility or patient privacy claim is
made. Within-BUSI duplicates were retained, and cross-dataset near-duplicate
screening was not claimed. Confirmation reuses previously inspected public
outer splits; it is not an unseen clinical evaluation.

The constructor still defaults to the original 41,537-parameter decoder with
random initialization. `decoder="narrow"` selects the smaller architecture,
but does not load a BUSI checkpoint. V5 applied its hash-bound public checkpoint
through campaign-only initialization tooling outside the trusted DP runner,
updating both the live decoder and transmitted Flower arrays. The standard
constructor exposes no public-checkpoint loading argument. These v5 results
therefore describe that disclosed benchmark workflow. Registration remains
`vetted=FALSE`; passing a utility floor does not itself promote the contract.

## Reconciliation and reading the records

[RECONCILE_V5.md](reconciliation-v5-20260920/RECONCILE_V5.md) and its
[audit-results.json](reconciliation-v5-20260920/pod-results/audit-results.json)
retain the stale-path finding: the previous packaged “current” files were
exact v3 copies. All six audited batch16 full-cohort epsilon8 models reproduced
their saved scores, and their captured initialization equalled the pretrained
tensors. Checkpoint NPZ hashes and concatenated tensor hashes use different
hash domains. Batch64 was not independently forward-scored in that audit.

In each cohort file, `replicates` holds the full-cohort cells and
`additional_replicates` holds the extensions. Filter by `variant` and `epsilon`,
then take the arithmetic mean of
`federated_dp.foreground_positive.dice` over the three distinct seeds
20260919/20260920/20260921. `summaries` and `additional_summaries` retain sample
SD and 95% Student-t intervals (df=2). Privacy mechanisms, exact twins,
artifact hashes, trivial baselines and envelope diagnostics remain in the
original records. The formal v5 utility floor is for BUS-BRA; applying its
threshold to BrEaST is descriptive.
