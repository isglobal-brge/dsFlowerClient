# Create a private ExtraTrees request spec

The trusted node creates a complete data-independent random topology
from its custodial PRF and releases all leaf sufficient statistics once
through the joint Gaussian mechanism. No analyst seed, callback,
objective or private split search is accepted.

## Usage

``` r
ds.flower.model.extra_trees(
  task = c("binary", "regression"),
  n_estimators = 32L,
  max_depth = 3L
)
```

## Arguments

- task:

  Either `"binary"` or `"regression"`.

- n_estimators:

  Positive number of trees, at most 512.

- max_depth:

  Positive complete-tree depth, at most 12.

## Value

A `dsflower_model` request object. Operational availability is probed on
every selected node at submission.
