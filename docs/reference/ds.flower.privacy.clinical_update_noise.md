# Create a clinical_update_noise privacy spec

Update-level hardening profile. Where the selected template supports it,
model updates or histogram contributions are clipped and perturbed with
calibrated Gaussian noise before aggregation. SecAgg is enforced by the
server profile.

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

This is not patient-level DP-SGD. It is useful when a study wants a
stricter update-sharing posture, but it should not be reported as formal
per-example Differential Privacy. For that setting use
[`ds.flower.privacy.high_sensitivity_dp()`](https://isglobal-brge.github.io/dsFlowerClient/reference/ds.flower.privacy.high_sensitivity_dp.md)
with a compatible PyTorch template.
