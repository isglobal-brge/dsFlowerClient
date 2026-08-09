# Create a PyTorch Multi-Class Classifier model spec

Configurable MLP or linear classifier with CrossEntropyLoss.

## Usage

``` r
ds.flower.model.pytorch_multiclass(
  hidden_layers = integer(0),
  n_classes = 3L,
  learning_rate = 0.1,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- hidden_layers:

  Integer vector; hidden layer sizes (empty for linear).

- n_classes:

  Integer; number of output classes.

- learning_rate:

  Numeric in `(0, 10]`; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
