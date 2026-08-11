# Create a native-tight XGBoost request spec

This constructor exposes only the typed, data-only parameter profile
used by the trusted native XGBoost adapter. Feature bounds and complete
public cuts are supplied with the training request. Privacy epsilon,
delta, clipping, randomness, objectives, callbacks and I/O are
node-owned and cannot be set here. The constructor is implemented
client-side; operational availability is probed afresh on every selected
node at submission time. Released ensembles support private external or
resubstitution validation. Benchmark-driven defaults use 8 trees, depth
2 and learning rate 0.25 for binary classification, and 5 trees, depth 2
and learning rate 0.30 for regression. Explicit values always take
precedence. Seven data-independent public cuts per feature are a
practical starting point; callers still supply every cut explicitly, and
dsFlower never derives them from private data.

## Usage

``` r
ds.flower.model.xgboost(
  task = c("binary", "regression"),
  n_estimators = 8L,
  max_depth = 2L,
  learning_rate = 0.25,
  min_child_weight = 1,
  min_split_loss = 0,
  reg_alpha = 0,
  reg_lambda = 1,
  max_delta_step = 1
)
```

## Arguments

- task:

  Either `"binary"` or `"regression"`.

- n_estimators:

  Positive integer number of boosting rounds/trees. Defaults to 8 for
  binary classification and 5 for regression.

- max_depth:

  Positive integer tree depth, at most 30. Defaults to 2.

- learning_rate:

  Numeric in `(0,1]`. Defaults to 0.25 for binary classification and
  0.30 for regression.

- min_child_weight:

  Non-negative minimum child Hessian weight.

- min_split_loss:

  Non-negative minimum split loss reduction.

- reg_alpha:

  Non-negative L1 leaf regularization.

- reg_lambda:

  Positive L2 leaf regularization, which keeps the leaf denominator
  strictly positive.

- max_delta_step:

  Positive maximum leaf-weight step. Together with the learning rate it
  fixes the public leaf-value bound.

## Value

A `dsflower_model` request object. It does not claim that the native
backend is installed or enabled.
