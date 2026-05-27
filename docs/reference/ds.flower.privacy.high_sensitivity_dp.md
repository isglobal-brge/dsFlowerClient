# Create a high_sensitivity_dp privacy spec

High-sensitivity profile for patient-level DP-SGD. The server allows
this profile only for templates validated for Opacus per-example
gradients and still requires Secure Aggregation and the profile's
minimum client policy.

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
