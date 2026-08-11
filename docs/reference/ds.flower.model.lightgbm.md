# Create a dsFlower LightGBM-style private boosting request

This is dsFlower's reviewed asymmetric public-bin numeric boosting
engine. It does not load or claim compatibility with the upstream
LightGBM binary/model runtime. Splits follow a fixed public transcript
and private histograms are released only through the node-owned sticky
Gaussian mechanism.

## Usage

``` r
ds.flower.model.lightgbm(
  task = c("binary", "regression"),
  n_estimators = 8L,
  max_depth = 2L,
  num_leaves = 4L,
  learning_rate = 0.25,
  min_data_in_leaf = 1L,
  min_gain_to_split = 0,
  reg_alpha = 0,
  reg_lambda = 1,
  max_delta_step = 1
)
```

## Arguments

- task:

  Either `"binary"` or `"regression"`.

- n_estimators:

  Positive boosting iteration count.

- max_depth:

  Positive maximum depth, at most 32.

- num_leaves:

  Positive leaf ceiling between 2 and 256 and compatible with
  `max_depth`.

- learning_rate:

  Numeric in `(0,1]`.

- min_data_in_leaf:

  Positive public minimum noisy count used for splits.

- min_gain_to_split:

  Non-negative public split threshold.

- reg_alpha:

  Non-negative L1 leaf regularization.

- reg_lambda:

  Positive L2 leaf regularization.

- max_delta_step:

  Positive public leaf-step bound.

## Value

A `dsflower_model` request object.
