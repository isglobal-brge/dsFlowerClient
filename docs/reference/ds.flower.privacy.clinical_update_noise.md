# Create a clinical_update_noise privacy spec

Update-level differential privacy hardening: clips weight updates and
adds calibrated Gaussian noise before aggregation. SecAgg enforced.

## Usage

``` r
ds.flower.privacy.clinical_update_noise(
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

  Numeric; update clipping norm (default 1.0).

## Value

A `dsflower_privacy` S3 object with mode = "clinical_update_noise".

## Details

NOTE: This is NOT patient-level DP-SGD. It protects against an
honest-but- curious aggregator seeing individual updates, but does not
provide formal per-example privacy guarantees. For formal DP, use
[`ds.flower.privacy.high_sensitivity_dp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.high_sensitivity_dp.md)
which uses Opacus DP-SGD.
