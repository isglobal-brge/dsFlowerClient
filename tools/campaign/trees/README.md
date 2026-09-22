# Representative native-tree utility cells

Three-site DSLite/PSOCK federation, with SuperNodes and SuperLink on loopback,
using dsFlower 0.5.0 and the canonical-container fix in dsFlowerClient 0.5.1.
Client patch branch: `fix/native-tree-canonical-container`, commit
`598d21c4627496a65d44034440f2d221ce6e04a2`.
Only client container serialization validation changed. Runner SHA-256 remains
`2135902bc710825b77b2f6a397c0040e051fe042fe1707b148b7e88ae71d2724`
on both sides. Registry defaults and all privacy mechanisms are unchanged.
The original failures and protocol are preserved under `failed-0.5.0/`.

## Frozen design

The primary matrix is `random_forest` × `breast, cdc9k` × epsilon `1, 4, 8`.
Each cell has three splits, seeds 20260820–20260822, using the existing
stratified 80/20 split and round-robin stratified allocation to three sites.
CDC cohort selection retains seed 20260819 and the existing loader exactly.
Native trees train their complete schedule in **one Flower round**; this is
their contract schedule. Registry defaults: 8 trees per node, depth 4,
`max_features=auto` (ceil(sqrt(p))). Delta is 1e-6, row privacy,
replace-one adjacency, and node clipping norm 1. No schedule search.

The node owns sticky secret randomness. Split and central forest seeds are
recorded; they do not seed or replace the native privacy RNG. Each replicate
uses fresh persistent site-secret files, retained only on the pod. These
public benchmark fits have per-training guarantees, not a claim that the
entire multi-cell campaign spends one epsilon.

Bounds and cuts are declared before loading or scoring any holdout:

- Breast: all nine ordinal features [1,10], cuts 1.5,2.5,…,9.5.
- CDC: binary indicators [0,1], cut 0.5; BMI [0,100], cuts 5,10,…,95;
  GenHlth [1,5], Age category [1,13], Education [1,6], Income [1,8],
  with half-integer cuts between all categories; MentHlth and PhysHlth
  [0,30], cuts 4.5,9.5,…,29.5.

These are public domain declarations, not empirical test extrema. BMI uses
an intentionally broad declared cap. The ordinal coding follows the public
dataset definitions. Both central and native models use these bounds and
the same public bins; sklearn receives bin indices. No test-based tuning,
threshold selection, model selection or rescore with altered settings.

The pooled central comparator is sklearn RandomForestClassifier with 8
trees, depth 4 and ceil(sqrt(p)) features, all other sklearn defaults, fit
on the identical pooled training set. This matches model family, per-node
size/depth and public feature representation, not the whole algorithm:
sklearn bootstraps pooled records; native RF assigns each record to one
tree by a secret PRF and averages three node forests (24 total trees).
The gap therefore combines algorithm, federation and DP costs and is not
a causal estimate of privacy cost alone. The trivial predictor uses the
training prevalence for probabilities and majority class at threshold 0.5.
Report AUC, accuracy, Brier and log-loss, plus paired fed-minus-central gaps.

## Diagnostics frozen before scoring

The primary diagnostic at epsilon 8 is **mean federated-DP AUC > 0.5 AND
mean accuracy strictly above mean majority-class accuracy**, across the three
registered seeds. Per-replicate flags are also recorded. This supersedes the
original permissive utility floor before any held-out scoring in this campaign.
The inherited floor and epsilon-monotonicity/near-zero-gap checks remain
secondary descriptive diagnostics in the JSON; they do not control selection.

If either primary dataset fails the epsilon-8 diagnostic, run exactly one
pre-declared alternative: `random_forest`, `cdc45k`, epsilon 8, the same three
seeds, sites, bounds, registry defaults and one-round schedule. The 45,000-row
cohort uses the unchanged fixed-seed loader used by the logreg n-scaling arm.
This alternative is a separate cell, never a replacement for a failed result.
Use `extra_trees` only if random_forest still fails to execute after the fix;
it is not a utility-search option. Never rerun a scored cell with changed settings.

## Data provenance

- [UCI 15: Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).
  Wolberg, W. (1990). *Breast Cancer Wisconsin (Original)* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z.
  Raw 699 rows; remove incomplete records to retain 683; omit sample ID;
  malignant class 4 is positive. Raw SHA-256:
  `402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64`.
- [UCI 891: CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).
  *CDC Diabetes Health Indicators* (2017) [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C53919. Source: CDC BRFSS public survey
  data; UCI links the Alex Teboul diabetes health indicators preparation.
  UCI's 253,680-row binary table, 21 features, ID removed;
  fixed-seed stratified 9,000-row cohort. `Diabetes_binary=1` includes
  prediabetes/diabetes. Raw SHA-256:
  `9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`.

Raw downloads and prepared cohort SHA-256 digests are recorded in each cell.

## Reproduction

Use the work-root layout documented in `../README.md`. The existing pod
`pod-flower-tabular` (2sy2g4pb3xwgqt) was provisioned in 97 seconds
(2026-09-22T00:06:28Z to 00:08:05Z). Verify and reuse its installations.
`provision.sh` records that initial setup; it is not rerun for these cells.

```sh
export R_LIBS=/workspace/cells/Rlib
# Set to the pushed fix/native-tree-canonical-container commit used to install:
export DSFLOWER_CLIENT_SOURCE_COMMIT=598d21c4627496a65d44034440f2d221ce6e04a2
export DSFLOWER_VENV_ROOT=/workspace/cells/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells/client
export DSFLOWER_NODE_SECRET_FILE=/workspace/cells/smoke/parent-node-secret
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/cells/dsFlowerClient
Rscript tools/campaign/trees/run_cell.R --contract random_forest \
  --dataset breast --epsilon 1 --replicates 3 --rounds 1 --sites 3 \
  --root /workspace/cells --out inst/extdata/campaign/trees
```

Run `python3 tools/campaign/trees/run_matrix.py` in the foreground for breast
epsilon 1,8,4, then cdc9k epsilon 1,8,4 and the conditional cdc45k endpoint.
The matrix stops on any execution failure; an authorized extra-trees fallback
must preserve that failure and use the same frozen data/splits/bounds.
The runner refuses existing cell/run paths to prevent accidental rescoring.
Runs are stored under `/workspace/cells/runs/trees-0.5.1/`.
Use a separate work root for independent reproduction. Native releases and
site secrets remain in the pod work root; only public evidence is committed.
Run `python3 tools/campaign/trees/summarize.py inst/extdata/campaign/trees`
to validate the evidence and rebuild `summary.json` without training.

## Patch and regression verification

In the dsFlowerClient patch checkout, install with
`R CMD INSTALL --no-configure -l /workspace/cells/Rlib .` to reuse cached Python
environments. The node package remains 0.5.0. Run the R native-tree, validation
and import tests and Python canonical JSON, prediction, import and artifact
validation tests. `tests/testthat/fixtures/generate-native-tree-training.py`
regenerates the small actual-training fixture using fixed public synthetic
records; its test-only fixed secret never enters campaign training.

`diagnose.R` is the one-training, unscored 0.5.0 reproduction. It captures the
public ensemble and validation inputs before failure cleanup and does not
call held-out prediction. Its preserved comparison and release bytes are in
`inst/extdata/campaign/trees/diagnosis-0.5.0/`; see `TREES_DIAGNOSIS.md` there.
The patched validator is also exercised directly on those exact saved bytes.

## Evidence count correction

The executed scoring tooling was frozen in commit
`162a880ce83488855c1cc5fa63fc2598707f9c21`; each cell pins its exact tool hashes.
That harness queried `fit$n_clients`, a field absent from the fit return value,
so it serialized an empty array. Its equality assertion accepted an empty vector.
`audit_releases.py` verifies the saved training RDS count, ensemble SHA-256 and
size, three ensemble members, and three node privacy responses, then repairs
only that reporting field. It records original/corrected JSON digests and the
original field in `release-audit.json` and each replicate. No scored model,
metric, split, parameter or prediction is altered, retrained or rescored.
The current harness reads the count from the validated ensemble and requires
one integer value. This reporting correction was applied after the scored matrix.

To audit the original records once on their original pod, before summarizing:

```sh
R_LIBS=/workspace/cells/Rlib python3 tools/campaign/trees/audit_releases.py \
  inst/extdata/campaign/trees /workspace/cells/runs/trees-0.5.1
python3 tools/campaign/trees/summarize.py inst/extdata/campaign/trees
```

The fixed harness already records the count for new independent reproductions.
