# Create a PyTorch U-Net 2D model spec

Medical image segmentation (organs, tumors, lesions).

## Usage

``` r
ds.flower.model.pytorch_unet2d(
  n_classes = 1L,
  learning_rate = 0.001,
  batch_size = 8L,
  local_epochs = 1L,
  image_size = 224L,
  base_channels = 64L
)
```

## Arguments

- n_classes:

  Integer; number of segmentation classes.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

- image_size:

  Integer; square resize dimension before training.

- base_channels:

  Integer; number of channels in the first U-Net block.

## Value

A `dsflower_model` S3 object.
