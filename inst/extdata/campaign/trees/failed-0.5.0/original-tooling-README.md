# Representative native-tree utility cells

Three-site DSLite/PSOCK federation, with SuperNodes and SuperLink on loopback,
using unchanged dsFlower and dsFlowerClient 0.5.0. This directory contains a
track-local copy of the existing campaign library; package code, runner,
registry and privacy defaults are unchanged.

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

Reuse the binary campaign checks in `vignettes/utility-campaign.Rmd`:

1. Utility floor: mean accuracy >= mean trivial accuracy - 0.02 **or**
   mean AUC > 0.6, reported for every cell and explicitly at epsilon 8.
2. Epsilon monotonicity: a paired AUC gap may worsen by at most the larger
   of the adjacent cells' gap SDs.
3. Near-zero-cost flag: cohort n × epsilon < 2000, central mean AUC < 0.95,
   and absolute mean AUC gap < 0.005. Also report a separate minimum-site-n
   companion flag without substituting it for the original rule.
4. The original seed-dispersion check targets heart, absent here: report
   it as not applicable, and show endpoint SDs descriptively.

No new stronger claim is implied by passing the inherited permissive floor.
If an epsilon-8 floor fails, run at most one alternative: default
`extra_trees` at epsilon 8 on the first failing dataset in breast, cdc9k
order. It has 32 trees per node, depth 3, with a matched-size/depth sklearn
ExtraTreesClassifier comparator. Also use this named fallback if RF fails
internally; preserve the error. Never alter or rerun a scored cell. Other
envelope flags are reported, not used to search for better results.

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
export DSFLOWER_VENV_ROOT=/workspace/cells/venvs
export DSFLOWER_CLIENT_VENV_ROOT=/workspace/cells/client
export DSFLOWER_NODE_SECRET_FILE=/workspace/cells/smoke/parent-node-secret
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /workspace/cells/dsFlowerClient
Rscript tools/campaign/trees/run_cell.R --contract random_forest \
  --dataset breast --epsilon 1 --replicates 3 --rounds 1 --sites 3 \
  --root /workspace/cells --out inst/extdata/campaign/trees
```

Run breast epsilon 1,8,4, then cdc9k epsilon 1,8,4, in the foreground.
The runner refuses existing cell/run paths to prevent accidental rescoring.
Use a separate work root for independent reproduction. Native releases and
site secrets remain in the pod work root; only public evidence is committed.
Run `python3 tools/campaign/trees/summarize.py inst/extdata/campaign/trees`
to validate the evidence and rebuild `summary.json` without training.
