# Create a high_sensitivity_dp privacy spec

The most restrictive profile. Requires patient-level DP-SGD, SecAgg, and
3+ clients.

## Usage

``` r
ds.flower.privacy.high_sensitivity_dp(
  epsilon = 1,
  delta = 1e-05,
  clipping_norm = 1
)
```

## Arguments

- epsilon:

  Numeric; privacy budget (default 1.0).

- delta:

  Numeric; probability of privacy leakage (default 1e-5).

- clipping_norm:

  Numeric; gradient clipping norm (default 1.0).

## Value

A `dsflower_privacy` S3 object with mode = "high_sensitivity_dp".
