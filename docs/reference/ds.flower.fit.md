# Fit a federated model in one call

High-level convenience API for the common workflow: connect to assigned
DataSHIELD data, build a recipe, run Flower, clean up server-side
handles, and return the trained run object. Advanced users can keep
using
[`ds.flower.connect()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.connect.md),
[`ds.flower.recipe()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.recipe.md),
and
[`ds.flower.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.md)
directly.

## Usage

``` r
ds.flower.fit(
  conns,
  data = NULL,
  resource = NULL,
  symbol = NULL,
  target,
  features = NULL,
  model = "sklearn_logreg",
  model_params = list(),
  strategy = "fedavg",
  strategy_params = list(),
  privacy = "auto",
  privacy_params = list(),
  rounds = 5L,
  task = NULL,
  label_set = NULL,
  masks = NULL,
  evaluation_only = FALSE,
  detached = FALSE,
  verbose = TRUE,
  disconnect = TRUE,
  run_args = list()
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

  Character target column name, or length-two vector for survival tasks.

- features:

  Character vector of feature column names, or NULL for
  template-specific auto handling.

- model:

  Character model name or `dsflower_model` object.

- model_params:

  Named list passed to
  [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md)
  when `model` is a character value.

- strategy:

  Character strategy name or `dsflower_strategy` object.

- strategy_params:

  Named list passed to
  [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md)
  when `strategy` is a character value.

- privacy:

  Character privacy name or `dsflower_privacy` object. The default
  `"auto"` uses the strongest practical profile detected from the
  connected servers.

- privacy_params:

  Named list passed to
  [`ds.flower.privacy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.md)
  when `privacy` is a character value.

- rounds:

  Integer number of federated rounds.

- task:

  Optional character task name or `dsflower_task` object.

- label_set:

  Optional imaging label-set name.

- masks:

  Optional mask asset alias for segmentation.

- evaluation_only:

  Logical; if TRUE, blocks model release.

- detached:

  Logical; passed to
  [`ds.flower.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.md).

- verbose:

  Logical; print training output.

- disconnect:

  Logical; remove server-side Flower handles on exit.

- run_args:

  Named list of additional arguments passed to
  [`ds.flower.run()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.run.md).

## Value

A `dsflower_run` object.
