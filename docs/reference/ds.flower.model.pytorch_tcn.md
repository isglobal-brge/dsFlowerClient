# Create a PyTorch TCN model spec

Temporal Convolutional Network for time series classification.

## Usage

``` r
ds.flower.model.pytorch_tcn(
  input_shape,
  channels = 8L,
  levels = 3L,
  n_classes = 2L,
  learning_rate = 0.001,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- input_shape:

  Integer vector `c(channels, sequence_length)` whose product must equal
  the staged feature count.

- channels:

  Integer; number of hidden convolution channels.

- levels:

  Integer; number of dilated TCN blocks.

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
