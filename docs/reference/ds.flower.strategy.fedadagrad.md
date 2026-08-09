# Create a FedAdagrad strategy spec

Create a FedAdagrad strategy spec

## Usage

``` r
ds.flower.strategy.fedadagrad(server_learning_rate = 0.1, tau = 0.001)
```

## Arguments

- server_learning_rate:

  Positive server learning rate (Flower `eta`).

- tau:

  Positive adaptivity regularizer.

## Value

A `dsflower_strategy` S3 object.
