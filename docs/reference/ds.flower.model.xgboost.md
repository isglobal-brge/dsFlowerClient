# Create an XGBoost model spec

Federated XGBoost using a histogram aggregation protocol. In trusted
demo profiles the histogram protocol can run without Secure Aggregation
so that the method path is testable end-to-end. In consortium and
clinical profiles, Secure Aggregation is still enforced by the selected
server trust profile.

## Usage

``` r
ds.flower.model.xgboost(
  n_trees = 10L,
  max_depth = 3L,
  eta = 0.3,
  reg_lambda = 1,
  n_bins = 64L,
  objective = "binary:logistic"
)
```

## Arguments

- n_trees:

  Integer; number of boosting rounds.

- max_depth:

  Integer; maximum tree depth.

- eta:

  Numeric; learning rate (shrinkage).

- reg_lambda:

  Numeric; L2 regularization term.

- n_bins:

  Integer; number of histogram bins.

- objective:

  Character; XGBoost objective function.

## Value

A `dsflower_model` S3 object.
