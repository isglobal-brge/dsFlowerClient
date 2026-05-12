# Create a PyTorch ResNet-18 model spec

Standard image classification backbone.

## Usage

``` r
ds.flower.model.pytorch_resnet18(
  n_classes = 2L,
  learning_rate = 0.001,
  batch_size = 32L,
  local_epochs = 1L,
  image_size = 224L
)
```

## Arguments

- n_classes:

  Integer; number of output classes.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

- image_size:

  Integer; square resize dimension before training.

## Value

A `dsflower_model` S3 object.
