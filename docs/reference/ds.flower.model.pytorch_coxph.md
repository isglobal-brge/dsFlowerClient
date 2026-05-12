# Create a PyTorch Cox Proportional Hazards model spec

Survival/time-to-event analysis with partial likelihood loss.

## Usage

``` r
ds.flower.model.pytorch_coxph(
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs per round.

## Value

A `dsflower_model` S3 object.
