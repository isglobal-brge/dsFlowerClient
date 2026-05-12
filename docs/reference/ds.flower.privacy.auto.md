# Create an automatic privacy spec

The automatic privacy policy is resolved at run time after server
capabilities are inspected. It chooses `prefer` when all connected
servers report Secure Aggregation support; otherwise it chooses
`fallback`. Explicit privacy specs remain available for production
deployments that require a fixed policy.

## Usage

``` r
ds.flower.privacy.auto(
  prefer = "clinical_default",
  fallback = "trusted_internal"
)
```

## Arguments

- prefer:

  Character privacy profile to use when SecAgg is available.

- fallback:

  Character privacy profile to use otherwise.

## Value

A `dsflower_privacy` object with mode = "auto".
