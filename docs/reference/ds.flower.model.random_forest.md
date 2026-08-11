# Create an adaptive private Random Forest request spec

The trusted node grows adaptive trees with a fixed public schedule and
custodial sticky randomness. Split histograms and terminal sufficient
statistics are released through the node-owned DP mechanism; analyst
seeds, callbacks, arbitrary objectives and executable model code are not
accepted. This is a disjoint-partition DP forest: each effective privacy
unit is assigned to one tree. It is not upstream bootstrap/bagging
Random Forest, and small cohorts can therefore have fewer effective
units available per tree. Defaults are 8 trees of depth 4 for binary
classification and 4 trees of depth 4 for regression. At request
construction, `max_features = NULL` or `"auto"` resolves from the public
feature count to `ceil(sqrt(p))` for binary classification or
`ceil(p/3)` for regression. The wire contract always contains the
resulting exact integer.

## Usage

``` r
ds.flower.model.random_forest(
  task = c("binary", "regression"),
  n_estimators = 8L,
  max_depth = 4L,
  max_features = NULL
)
```

## Arguments

- task:

  Either `"binary"` or `"regression"`.

- n_estimators:

  Positive number of trees, at most 512.

- max_depth:

  Positive tree depth, at most 12.

- max_features:

  `NULL`, `"auto"`, or a positive integer no greater than the public
  feature count.

## Value

A `dsflower_model` request object. Operational availability is probed on
every selected node at submission.
