# Create a differential privacy spec

Create a differential privacy spec

## Usage

``` r
ds.flower.privacy.dp(epsilon = 1, delta = 1e-05, clipping_norm = 1)
```

## Arguments

- epsilon:

  Numeric; privacy budget.

- delta:

  Numeric; probability of privacy leakage.

- clipping_norm:

  Numeric; gradient clipping norm.

## Value

A `dsflower_privacy` S3 object with mode = "dp".
