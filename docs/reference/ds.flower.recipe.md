# Create a Flower federated learning recipe

A recipe combines all specification objects needed for a federated
learning experiment: task type, model architecture, aggregation
strategy, privacy settings, and data configuration.

## Usage

``` r
ds.flower.recipe(
  task,
  model,
  strategy,
  privacy = ds.flower.privacy.research(),
  num_rounds = 5L,
  target_column = "target",
  feature_columns = NULL
)
```

## Arguments

- task:

  A `dsflower_task` object.

- model:

  A `dsflower_model` object.

- strategy:

  A `dsflower_strategy` object.

- privacy:

  A `dsflower_privacy` object (default: research mode).

- num_rounds:

  Integer; number of federated training rounds.

- target_column:

  Character; name of the target column.

- feature_columns:

  Character vector; names of feature columns, or NULL.

## Value

A `dsflower_recipe` S3 object.
