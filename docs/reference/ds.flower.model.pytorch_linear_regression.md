# Create a PyTorch Linear Regression model spec

Continuous outcome prediction (MSE loss).

## Usage

``` r
ds.flower.model.pytorch_linear_regression(
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- learning_rate:

  Numeric in `(0, 10]`; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
