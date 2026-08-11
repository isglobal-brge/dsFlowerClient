# Create a dsFlower CatBoost-style private boosting request

This is dsFlower's reviewed numeric oblivious-tree engine. It does not
load CatBoost CBM files or the upstream CatBoost runtime. Categoricals,
ordered target statistics, analyst seeds and executable callbacks are
deliberately outside this safe numeric profile.

## Usage

``` r
ds.flower.model.catboost(
  task = c("binary", "regression"),
  n_estimators = 8L,
  depth = 2L,
  learning_rate = 0.25,
  l2_leaf_reg = 1,
  max_delta_step = 1
)
```

## Arguments

- task:

  Either `"binary"` or `"regression"`.

- n_estimators:

  Positive boosting iteration count.

- depth:

  Positive oblivious-tree depth, at most 16.

- learning_rate:

  Numeric in `(0,1]`.

- l2_leaf_reg:

  Positive L2 leaf regularization.

- max_delta_step:

  Positive public leaf-step bound.

## Value

A `dsflower_model` request object.
