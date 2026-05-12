# Create a FedAdam strategy spec

FedAdam uses adaptive learning rates on the server side via Adam
optimizer for more stable convergence in heterogeneous settings.

## Usage

``` r
ds.flower.strategy.fedadam(
  server_learning_rate = 0.01,
  tau = 0.001,
  fraction_fit = 1,
  fraction_evaluate = 1
)
```

## Arguments

- server_learning_rate:

  Numeric; server-side learning rate (eta).

- tau:

  Numeric; controls adaptivity (higher = more stable).

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

- fraction_evaluate:

  Numeric; fraction of clients used for evaluation (0-1).

## Value

A `dsflower_strategy` S3 object.
