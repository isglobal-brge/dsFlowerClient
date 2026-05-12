# Create a FedAdagrad strategy spec

FedAdagrad uses adaptive learning rates on the server side via Adagrad
optimizer. Suited for sparse gradients and non-IID distributions.

## Usage

``` r
ds.flower.strategy.fedadagrad(
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
