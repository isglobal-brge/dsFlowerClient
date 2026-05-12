# Create a FedProx strategy spec

FedProx adds a proximal term to keep local models closer to the global
model. Helps with heterogeneous (non-IID) data.

## Usage

``` r
ds.flower.strategy.fedprox(proximal_mu = 0.1, fraction_fit = 1)
```

## Arguments

- proximal_mu:

  Numeric; proximal term weight.

- fraction_fit:

  Numeric; fraction of clients used for training (0-1).

## Value

A `dsflower_strategy` S3 object.
