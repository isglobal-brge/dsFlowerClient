# Create a FedAvg strategy spec

Federated Averaging: each node trains, sends updated weights, and the
server computes a weighted average. The number of participating clients
is determined automatically from the number of connected servers.

## Usage

``` r
ds.flower.strategy.fedavg(fraction_fit = 1, fraction_evaluate = 1)
```

## Arguments

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

- fraction_evaluate:

  Numeric; fraction of clients used for evaluation (0-1).

## Value

A `dsflower_strategy` S3 object.
