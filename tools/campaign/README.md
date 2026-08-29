# dsFlower utility campaign runner

Real federated DP training over DSLite with real public UCI data, producing
committed evidence JSONs in `inst/extdata/campaign/`. The federation pattern
(one DSLite server per PSOCK R worker, SuperNodes + SuperLink over loopback)
is copied from `tools/integration/dslite-multinode-smoke.R` and generalised to
N sites in `campaign_lib.R`.

## Prerequisites

A provisioned work root (see `p0_smoke.sh` in the thesis repo) containing:

- `Rlib/` — dsFlower + dsFlowerClient installed (`R CMD INSTALL`)
- `venvs/pytorch/` — server venv (torch, flwr, opacus, ...)
- `client/venv/` — client venv
- `data_cache/` — downloaded raw datasets with `CHECKSUMS.sha256`:
  - `breast-cancer-wisconsin.data` (UCI Breast Cancer Wisconsin Original)
  - `processed.cleveland.data` (UCI Heart Disease, processed Cleveland)
  - `cdc_diabetes_health_indicators.csv` (UCI 891, CDC Diabetes Health
    Indicators, full 253,680-row table; the runner takes a fixed-seed
    stratified 9,000-row cohort)

## Run one cell

```sh
ROOT=<work-root>
export R_LIBS="$ROOT/Rlib:$(Rscript -e 'cat(paste(.libPaths(), collapse=":"))')"
export DSFLOWER_VENV_ROOT="$ROOT/venvs"
export DSFLOWER_CLIENT_VENV_ROOT="$ROOT/client"
export DSFLOWER_NODE_SECRET_FILE="$ROOT/smoke/parent-node-secret"

cd dsFlowerClient
Rscript tools/campaign/run_cell.R \
  --contract pytorch_logreg --dataset breast --epsilon 8 \
  --replicates 3 --rounds 5 --sites 3 \
  --root "$ROOT" --out inst/extdata/campaign
```

Datasets: `breast`, `heart`, `cdc9k`. One JSON per cell is written as
`pilot_<dataset>_<contract>_eps<E>.json` (schema `dsflower-campaign-v1`).

## Design decisions (disclosed in the artifacts)

- Per replicate `r`: seed `20260819 + r` drives a stratified 80/20
  train/test split; the train side is dealt round-robin per class into 3
  even, stratified site shards. The SAME split feeds the central and
  federated branches.
- Central baseline: noiseless pooled `glm(target ~ ., binomial)`.
- Trivial baseline: majority class / prevalence.
- Federated-DP evaluation is channel B: the released `model.pt` artifact is
  consumed locally via `ds.flower.predict(fit, test, type = "prob")` on the
  same held-out test set.
- Node privacy contract is set inside each worker via
  `options(dsflower.dp_per_training_epsilon = E,
  dsflower.dp_per_training_delta = 1e-6)`; clipping and DP unit stay at
  dsFlower defaults.
- Feature bounds are public-schema declarations computed from the pooled
  TRAIN side of each replicate: observed min/max widened by 10% of the range
  (+/-0.5 for constant columns). Recorded per replicate in the artifact.
- Model params were tuned ONCE on breast at epsilon = 8 (two sanity runs:
  `local_epochs` 1 vs 2 at lr 0.1, batch 32; 2 was strictly better) and then
  frozen for every cell: `learning_rate = 0.1, batch_size = 32,
  local_epochs = 2`.
