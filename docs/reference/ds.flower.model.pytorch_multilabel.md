# Create a Multi-Label Classification model spec

Multiple binary outcomes per sample (phenotyping, multi-endpoint). Uses
BCEWithLogitsLoss per label. Training requires exactly `n_labels`
distinct target columns; one public two-level vocabulary is applied to
each.

## Usage

``` r
ds.flower.model.pytorch_multilabel(
  n_labels = 2L,
  hidden_layers = c(64L, 32L),
  learning_rate = 0.1,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- n_labels:

  Integer; number of label columns.

- hidden_layers:

  Integer vector; hidden layer sizes.

- learning_rate:

  Numeric in `(0, 10]`; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs.

## Value

A `dsflower_model` S3 object.
