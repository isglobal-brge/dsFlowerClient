# Create a FedYogi strategy spec

Create a FedYogi strategy spec

## Usage

``` r
ds.flower.strategy.fedyogi(
  server_learning_rate = 0.01,
  beta_1 = 0.9,
  beta_2 = 0.99,
  tau = 0.001
)
```

## Arguments

- server_learning_rate:

  Positive server learning rate (Flower `eta`).

- beta_1:

  First-moment coefficient in `[0,1)`.

- beta_2:

  Second-moment coefficient in `[0,1)`.

- tau:

  Positive adaptivity regularizer.

## Value

A `dsflower_strategy` S3 object.
