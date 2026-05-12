# Create a PyTorch TCN model spec

Temporal Convolutional Network for time series classification.

## Usage

``` r
ds.flower.model.pytorch_tcn(
  n_channels = 1L,
  kernel_size = 3L,
  n_layers = 4L,
  learning_rate = 0.001,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- n_channels:

  Integer; number of input channels.

- kernel_size:

  Integer; convolution kernel size.

- n_layers:

  Integer; number of TCN blocks.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
