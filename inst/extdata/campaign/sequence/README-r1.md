# UCI HAR sequence cell: executed and scored once

Token: `FLOWER_CELLS_SEQUENCE_2026-09-22`. Nine `pytorch_lstm` replicates
completed on the NVIDIA A40 in `pod-flower-sequence`: epsilon 1, 8, 4 in that
order, seeds 20260922–20260924, five rounds, three sites, delta 1e-6 and unit
clipping. Local epochs stayed at the default one; federation times were
107.85–132.17 seconds. GRU was tested under the guard but no HAR fallback was
needed. All 135 node-round captures passed the tensor, population and accountant
checks, and all federations cleaned up. The pod is left running.

The official subject-disjoint split has 7,352 training windows / 21 subjects,
three sites of seven subjects (2,553 / 2,397 / 2,402 windows), and 2,947 test
windows / nine subjects. Each input is a 128 × 9 inertial window flattened in
time-major order, with six activity classes. Bounds use TRAIN only.

The released patient path averages windows within each subject and chooses its
modal class before per-subject clipping. Thus each site trains on seven pooled
sequences; the noiseless central twin uses the same 21 pooled sequences. This
is not a conventional classifier trained on each individual window. Hidden
size 32, SGD 0.001, batch size 32, one local epoch and no scheduler stayed fixed.

Table entries are mean **macro one-vs-rest AUC / accuracy / log-loss** across
three seeded training replicates. Gap is federated-DP minus central macro AUC,
with sample SD. All individual metrics and SDs are in the cell JSON files.

| Epsilon | Central | Federated-DP | Trivial | AUC gap mean ± SD |
|---:|---|---|---|---|
| 1 | 0.435143 / 0.169551 / 1.799129 | 0.434854 / 0.169551 / 1.799015 | 0.500000 / 0.182219 / 1.789941 | -0.000289 ± 0.006243 |
| 4 | 0.435143 / 0.169551 / 1.799129 | 0.434235 / 0.169551 / 1.799279 | 0.500000 / 0.182219 / 1.789941 | -0.000908 ± 0.000795 |
| 8 | 0.435143 / 0.169551 / 1.799129 | 0.435451 / 0.169551 / 1.799082 | 0.500000 / 0.182219 / 1.789941 | +0.000308 ± 0.000544 |

Mean macro AUC is below 0.5 at every epsilon. At epsilon 8, mean accuracy is
0.169551 versus the majority baseline's 0.182219. These are utility annotations,
not execution pass/fail criteria. No scored cell was rerun or retuned. Seeds
fix public initialization; DP noise retains the node's cryptographic randomness.
The split is fixed, so SD describes training variation, not split uncertainty.
Each epsilon is a separate training contract, not a composed campaign guarantee.

## Patch, versions and provenance

Installed dsFlower **0.5.1**, dsFlowerClient **0.5.0**. Server patch branch:
`fix/import-guard-torch-generated-modules`, commit
`c4eaaf153db4a7204878bf1d8995c16faa615110`, directly based on 0.5.0 main
`408f08c539329e2711260050ab40a6567aa4d89e`. No native-tree changes, tag or merge.
The guard verifies this one generated module's installed torch generators,
generation directory and exact template source, then executes the checked
snapshot. Other foreign imports remain default-deny. Privacy mechanisms,
accounting, clipping, registry defaults and both canonical runners are unchanged.

Runner SHA-256 in both installed packages:
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`.
Installed guard SHA-256:
`3ae7c9ce6750c81c00e8c70e618d0bc52978486d7736587f39c716ff401fe98d`.
R 4.6.1; Python 3.11.10; torch 2.6.0+cu124; Opacus 1.6.0; Flower 1.31.0.
All Python dependency versions match the blocked run. Executed final tooling:
`9557545f6649f440c43566b6dcf4553db38bef65`; `runtime.json` records exact file
hashes. A subsequent reporting-only change disambiguates preparation/test access
in the JSON metadata (`reporting-correction.json`); no metrics were recomputed.

The 256 Python checks and server R Python-environment/runtime suites passed
within the 20-minute cap. See `verification/`, `diagnosis-0.5.1/` and
[the full diagnosis](SEQUENCE_DIAGNOSIS.md). Original 0.5.0 records and failed
LSTM/GRU import stacks remain byte-for-byte under `blocked-0.5.0/`.

Two unscored central-tooling interruptions were retained: a float32/int64 label
hash mismatch, then missing shared aliases in its checkpoint. Corrected target
hashes match every node capture; replayed central parameter tensors were
bit-identical. The completed federation was retained, not repeated. See
`tooling-recovery.json`, `tooling-interruption/` and
`central-replay-verification.json`. The first resumed execution duration excludes
interrupted wrapper overhead, as explicitly recorded in the recovery note.

## Reproduction

Use [the campaign commands](../../../../tools/campaign/sequence/README.md) and
`release-source.json` in that tooling directory. Install the server patch into
the pod library with `DSFLOWER_SKIP_PYTHON_SETUP=1 R CMD INSTALL`, preserving
the frozen Python environments. Run preflight, runtime capture, the foreground
matrix and then `score_and_assemble.py` once. `--resume-unscored` retains only
hash-verified completed phases and refuses any training after the scoring marker.
The completed pod must not be rescored or retrained. Use a fresh root for an
independent reproduction. All commands address only `pod-flower-sequence`.

`execution-audit.json` verifies nine completed replicates, 135 node rounds,
model hashes, unchanged protocol and identical central checkpoints across
epsilon for each seed before test access. `test-scoring-started.json` is the
exclusive scoring marker, created only after all models and checks completed.
One final scorer invocation read TEST and produced all three epsilon JSONs.
`summary.json` indexes their SHA-256 hashes; `audit.json` is the original
TRAIN-only preparation audit. Its `test_accessed=false` refers to preparation;
scored cells explicitly distinguish preparation from final test access.

## Dataset source

[UCI HAR, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[official archive](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip),
SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Licence: CC BY 4.0. Dataset DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
