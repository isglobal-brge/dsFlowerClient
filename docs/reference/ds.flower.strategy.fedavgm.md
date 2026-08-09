# Create a FedAvgM strategy spec

Create a FedAvgM strategy spec

## Usage

``` r
ds.flower.strategy.fedavgm(server_learning_rate = 1, server_momentum = 0)
```

## Arguments

- server_learning_rate:

  Positive server learning rate.

- server_momentum:

  Server momentum in `[0,1)`.

## Value

A `dsflower_strategy` S3 object.
