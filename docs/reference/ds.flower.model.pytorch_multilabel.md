# Create a Multi-Label Classification model spec

Multiple binary outcomes per sample (phenotyping, multi-endpoint). Uses
BCEWithLogitsLoss per label.

## Usage

``` r
ds.flower.model.pytorch_multilabel(
  n_labels = 2L,
  hidden_layers = "64,32",
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- n_labels:

  Integer; number of label columns.

- hidden_layers:

  Character; comma-separated hidden layer sizes.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs.

## Value

A `dsflower_model` S3 object.
