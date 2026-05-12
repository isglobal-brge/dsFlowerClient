# Create a PyTorch MLP model spec

Create a PyTorch MLP model spec

## Usage

``` r
ds.flower.model.pytorch_mlp(
  hidden_layers = c(64L, 32L),
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- hidden_layers:

  Integer vector; hidden layer sizes.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
