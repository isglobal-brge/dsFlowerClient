# Create a PyTorch Logistic Regression model spec

DP-SGD capable linear classifier for binary classification.

## Usage

``` r
ds.flower.model.pytorch_logreg(
  learning_rate = 0.1,
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
