# Create a Log-Normal AFT survival model spec

Log-normal Accelerated Failure Time parametric survival model.
Alternative to Cox PH when proportional hazards assumption fails.
Predicts log(T) = X\*beta + sigma\*epsilon, epsilon ~ N(0,1).

## Usage

``` r
ds.flower.model.pytorch_lognormal_aft(
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs.

## Value

A `dsflower_model` S3 object.

## Details

NOTE: This is specifically log-normal AFT, not the full AFT family
(Weibull, log-logistic, etc.).
