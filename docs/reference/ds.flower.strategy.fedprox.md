# Create a FedProx strategy spec

Create a FedProx strategy spec

## Usage

``` r
ds.flower.strategy.fedprox(
  proximal_mu = 0.1,
  fraction_fit = 1,
  min_fit_clients = 2L,
  min_available_clients = 2L
)
```

## Arguments

- proximal_mu:

  Numeric; proximal term weight.

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

- min_fit_clients:

  Integer; minimum number of clients for training.

- min_available_clients:

  Integer; minimum number of available clients.

## Value

A `dsflower_strategy` S3 object.
