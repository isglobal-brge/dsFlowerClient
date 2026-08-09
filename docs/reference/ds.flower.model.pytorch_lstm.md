# Create a PyTorch LSTM model spec

LSTM for longitudinal EHR and sequential clinical data.

## Usage

``` r
ds.flower.model.pytorch_lstm(
  n_tokens,
  n_features,
  hidden = 32L,
  n_classes = 2L,
  learning_rate = 0.001,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- n_tokens:

  Integer; number of time points per sample.

- n_features:

  Integer; number of features per time point. The product
  `n_tokens * n_features` must equal the staged feature count.

- hidden:

  Integer; LSTM hidden state size.

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
