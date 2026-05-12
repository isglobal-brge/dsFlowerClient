# Create a clinical_default privacy spec (recommended)

The recommended default for clinical federated learning. Requires
SecAgg, suppresses per-node metrics, enforces fixed client sampling.

## Usage

``` r
ds.flower.privacy.clinical_default()
```

## Value

A `dsflower_privacy` S3 object with mode = "clinical_default".
