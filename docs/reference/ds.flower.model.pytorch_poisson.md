# Create a Poisson Regression model spec

Count data modeling (hospital events, readmissions, adverse events).
Uses Poisson NLL loss with log link.

## Usage

``` r
ds.flower.model.pytorch_poisson(
  hidden_layers = "",
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- hidden_layers:

  Character; comma-separated hidden layer sizes (empty = linear).

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs.

## Value

A `dsflower_model` S3 object.
