# Create a Flower federated learning recipe

A recipe combines the analyst-controlled specification objects needed
for a federated learning experiment. Privacy policy is not part of the
recipe; it is selected and enforced by each data node. Task can be
inferred from the model when not specified.

## Usage

``` r
ds.flower.recipe(
  model,
  strategy = ds.flower.strategy.fedavg(),
  task = NULL,
  num_rounds = 5L,
  target = NULL,
  features = NULL
)
```

## Arguments

- model:

  A `dsflower_model` object or character model name accepted by
  [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md).

- strategy:

  A `dsflower_strategy` object or character strategy name accepted by
  [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md).

- task:

  A `dsflower_task` object, character task name, or NULL to infer from
  model.

- num_rounds:

  Integer; number of federated training rounds.

- target:

  Character; target column name(s). Multiple targets are supported only
  by the multilabel enforced-DP model.

- features:

  Character vector; feature column names, or NULL for auto.

## Value

A `dsflower_recipe` S3 object.
