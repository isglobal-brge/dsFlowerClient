# Create a Cause-Specific Cox model spec

Multi-cause survival analysis via cause-specific hazard modeling.
Separate Cox partial likelihood per event cause.

## Usage

``` r
ds.flower.model.pytorch_cause_specific_cox(
  n_causes = 2L,
  learning_rate = 0.01,
  batch_size = 32L,
  local_epochs = 1L
)
```

## Arguments

- n_causes:

  Integer; number of competing event types.

- learning_rate:

  Numeric; learning rate.

- batch_size:

  Integer; batch size.

- local_epochs:

  Integer; local training epochs.

## Value

A `dsflower_model` S3 object.

## Details

NOTE: This is cause-specific hazard modeling, NOT Fine-Gray
sub-distribution hazards. Each cause has its own Cox PH model.
