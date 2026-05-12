# Create an Elastic Net model spec

Convenience constructor for Elastic Net regularization (L1 + L2
penalty). Internally uses the `sklearn_sgd` template with
`penalty = "elasticnet"`. Useful for variable selection in
high-dimensional data (genomics, radiomics).

## Usage

``` r
ds.flower.model.sklearn_elastic_net(
  l1_ratio = 0.15,
  alpha = 1e-04,
  loss = "log_loss",
  lr_schedule = "optimal",
  eta0 = 0.01,
  max_iter = 1000L
)
```

## Arguments

- l1_ratio:

  Numeric; mixing parameter (0 = L2 only, 1 = L1 only). Default 0.15.

- alpha:

  Numeric; regularization constant. Default 0.0001.

- loss:

  Character; loss function. Default "log_loss" (logistic).

- lr_schedule:

  Character; learning rate schedule.

- eta0:

  Numeric; initial learning rate for constant/invscaling/adaptive
  schedules.

- max_iter:

  Integer; maximum local sklearn iterations.

## Value

A `dsflower_model` S3 object using the sklearn_sgd template.
