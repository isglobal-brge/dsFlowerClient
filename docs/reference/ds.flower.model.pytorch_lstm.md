# Create a PyTorch LSTM model spec

LSTM for longitudinal EHR and sequential clinical data.

## Usage

``` r
ds.flower.model.pytorch_lstm(
  hidden_size = 64L,
  num_layers = 2L,
  learning_rate = 0.001,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- hidden_size:

  Integer; LSTM hidden state size.

- num_layers:

  Integer; number of LSTM layers.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
