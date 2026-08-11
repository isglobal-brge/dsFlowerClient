# Fit a federated model in one call

High-level convenience API for the common workflow: connect to assigned
DataSHIELD data, build a recipe, run Flower, clean up server-side
handles, and return the trained run object. Advanced users can call
[`ds.flower.submit()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.submit.md)
directly for finer control.

## Usage

``` r
ds.flower.fit(
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
  output_dir = NULL,
  output_name = NULL,
  silent = FALSE,
  verbose = FALSE,
  feature_bounds = NULL,
  feature_cuts = NULL,
  target_levels = NULL,
  target_bounds = NULL,
  allow_insecure_http = getOption("dsflower.dsi_allow_insecure_http", character()),
  data_kind = NULL,
  holdout = NULL,
  cross_validation = NULL
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

- feature_cuts:

  Required for native-tight tree models: a list containing one strictly
  increasing vector of public cut points per feature, each strictly
  inside `feature_bounds`. Seven data-independent cuts per feature are a
  practical benchmark-backed starting point; they are never inferred
  from private node data.

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

- data_kind:

  Optional input kind, `"tabular"` or `"image"`. It is inferred when the
  registered model supports exactly one kind; models registered for both
  require an explicit choice.

- holdout:

  Optional numeric test fraction strictly between zero and one, with at
  most six decimal places. The node assigns complete privacy units
  before training, trains only on the complement, and returns the final
  model together with one pooled differentially-private test metric
  release. This supports tabular neural/native-tree models and native
  dsFlower vision.

- cross_validation:

  Optional integer in `[2, 10]`. This runs a dedicated metrics-only
  tabular CV job for neural or binary/regression native-tree models. It
  releases one pooled DP OOF result and saves no fold model or
  prediction. When `rounds` is omitted, native-tree CV uses its required
  single round per fold; an explicit value is never overwritten. Prefer
  [`ds.flower.cross_validate()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.cross_validate.md)
  for this workflow.

## Value

A `dsflower_run` object, or a `dsflower_cv` when `cross_validation` is
set.
