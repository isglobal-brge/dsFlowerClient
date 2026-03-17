# Create a FedAvg strategy spec

Create a FedAvg strategy spec

## Usage

``` r
ds.flower.strategy.fedavg(
  fraction_fit = 1,
  fraction_evaluate = 1,
  min_fit_clients = 2L,
  min_available_clients = 2L
)
```

## Arguments

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

- fraction_evaluate:

  Numeric; fraction of clients used for evaluation (0-1).

- min_fit_clients:

  Integer; minimum number of clients for training.

- min_available_clients:

  Integer; minimum number of available clients.

## Value

A `dsflower_strategy` S3 object.
