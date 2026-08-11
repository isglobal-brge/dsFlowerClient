# Cross-validate a tabular neural or native-tree model across federated data

Runs `folds` complete, clean-initialized federated trainings. Each node
assigns whole privacy units to folds with its custodial HMAC, trains
every fold only on the complement, and keeps held-out sufficient
statistics in namespaced Flower runtime memory. Only one pooled
differentially-private OOF metric vector is released after all folds
finish; no fold model, prediction, site metric, or fold metric is saved.
Native-tree cross-validation supports binary classification and bounded
regression for the five registered engines. It reuses each engine's
canonical ensemble contract; XGBoost additionally requires the verified
node-owned native bundle. Native engines run exactly one Flower round
per fold, while neural models use the requested rounds per fold.

## Usage

``` r
ds.flower.cross_validate(
  conns,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  target,
  features = NULL,
  model = "pytorch_logreg",
  model_params = list(),
  torch_backend = "auto",
  strategy = "fedavg",
  strategy_params = list(),
  rounds = 5L,
  task = NULL,
  folds = 3L,
  output_dir = NULL,
  output_name = NULL,
  silent = FALSE,
  verbose = FALSE,
  feature_bounds = NULL,
  target_levels = NULL,
  target_bounds = NULL,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character())
)
```

## Arguments

- conns:

  DSI connections object.

- data:

  Optional character data source resolved by
  [`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md).

- resource:

  Optional Opal resource name.

- symbol:

  Optional assigned DataSHIELD symbol. Defaults to `"D"` when `data`,
  `resource`, and `symbol` are all NULL.

- target:

  One target-column name, or exactly `num_labels` distinct target-column
  names for `pytorch_multilabel`.

- features:

  Character vector of feature column names, or NULL for model-specific
  auto handling.

- model:

  Character model name or `dsflower_model` object.

- model_params:

  Named list passed to
  [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md)
  when `model` is a character value.

- torch_backend:

  Character; requested node-side torch backend (`"auto"`, `"cpu"`, or a
  GPU selector).

- strategy:

  Character strategy name or `dsflower_strategy` object.

- strategy_params:

  Named list passed to
  [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md)
  when `strategy` is a character value.

- rounds:

  Integer number of federated rounds.

- task:

  Optional character task name or `dsflower_task` object.

- folds:

  Integer in `[2, 10]`; number of actual federated folds.

- output_dir:

  Optional character; parent directory the trained model is saved under
  (created if missing). Defaults to `./dsflower_output`.

- output_name:

  Optional character; folder/file name for this model inside
  `output_dir` (extension is added automatically). Defaults to an
  auto-generated model id.

- silent:

  Logical; when `TRUE`, suppress the training progress feedback
  (connection, per-round and completion messages).

- verbose:

  Logical; when `TRUE`, also print the raw flwr run log (debugging). The
  tidy per-round progress is shown regardless unless `silent = TRUE`.

- feature_bounds:

  Optional public feature bounds as `list(lower=..., upper=...)` in
  feature order.

- target_levels:

  Optional ordered public classification label vocabulary. Non-numeric
  labels require it; missing or unknown values map to public code zero.
  Multilabel applies the same public two-level vocabulary independently
  to every target column. Binary native-tree models require exactly two
  ordered levels.

- target_bounds:

  Required public `list(lower=..., upper=...)` for regression/count
  models.

- allow_insecure_http:

  Character vector of exact connection names allowed to use plaintext
  HTTP. Empty by default. This exception does not provide transport
  security; use it only behind an independently trusted network.

## Value

A `dsflower_cv` object containing only pooled OOF metrics and public job
metadata.

## Details

The node-owned per-job privacy contract reserves 80 percent for training
and divides it evenly across folds, with the remaining 20 percent used
once for the OOF release. Larger `folds` therefore gives each training
less privacy budget and can reduce model utility; three is the practical
default.
