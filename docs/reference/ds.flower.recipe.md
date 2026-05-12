# Create a Flower federated learning recipe

A recipe combines all specification objects needed for a federated
learning experiment. Template is always inferred from the model. Task
can be inferred from the model if not specified.

## Usage

``` r
ds.flower.recipe(
  model,
  strategy = ds.flower.strategy.fedavg(),
  privacy = ds.flower.privacy.clinical_default(),
  task = NULL,
  num_rounds = 5L,
  target = NULL,
  target_column = NULL,
  label_set = NULL,
  features = NULL,
  feature_columns = NULL,
  masks = NULL,
  evaluation_only = FALSE
)
```

## Arguments

- model:

  A `dsflower_model` object or character model name accepted by
  [`ds.flower.model()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.model.md).

- strategy:

  A `dsflower_strategy` object or character strategy name accepted by
  [`ds.flower.strategy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.strategy.md).

- privacy:

  A `dsflower_privacy` object or character privacy name accepted by
  [`ds.flower.privacy()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.md).

- task:

  A `dsflower_task` object, character task name, or NULL to infer from
  model.

- num_rounds:

  Integer; number of federated training rounds.

- target:

  Character; target column name(s). For survival: c("time", "event").

- target_column:

  Alias for `target` (backward compat).

- label_set:

  Character; name of the label set to use (imaging datasets).

- features:

  Character vector; feature column names, or NULL for auto.

- feature_columns:

  Alias for `features` (backward compat).

- masks:

  Character; mask asset alias for segmentation, or NULL.

- evaluation_only:

  Logical; if TRUE, blocks model release.

## Value

A `dsflower_recipe` S3 object.
